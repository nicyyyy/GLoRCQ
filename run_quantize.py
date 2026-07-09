#!/usr/bin/env python3
"""
run_quantize.py

TileQ 1D + GLoRCQ cross-layer group sharing.

Key idea: TileQ uses activation-scaled SVD of W_orig, then VQ-quantizes the residual.
  Per-expert lora = diag(Sa) @ U @ diag(Si) @ V, residual R = W - lora.
  TileQ 1D shares U within a layer (60 experts/layer).
  We extend this to CROSS-LAYER sharing: G=128 experts from different layers share U.

Stacking convention (row=1, shared U):
  big_tensor = [W_scale_T_1 | W_scale_T_2 | ... | W_scale_T_G]  shape (in_d, G*out_d)
  SVD → U_shared (in_d, rank) [SAME for all G], Vh_all (rank, G*out_d)
  Per-expert k: V_k = Vh_all[:, k*out_d:(k+1)*out_d]

Pipeline:
  Phase 0: Load model, capture initial layer inputs.
  Phase 1: For each layer, collect activation scales for all MoE experts.
           Advance inps through each layer (FP16 forward pass).
  Phase 2: Cross-layer group G experts of same type → shared SVD → fill WR.
  Phase 2.5 (optional): Plain scalar GPTQ for attention layers (--attn_bits).
  Phase 3: TileQ's gptvq_fwrd_lora unchanged — VQ quantizes residuals.
"""

import sys, os

_HERE  = os.path.dirname(os.path.abspath(__file__))  # GLoRCQ root
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import gc
import math
import functools
import tqdm
import argparse
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM, MixtralDecoderLayer
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
# Qwen3-MoE (Qwen3-30B-A3B): 128 experts × top-8, moe_intermediate=768,
# container `mlp`, no shared_expert. Added in transformers 4.51.
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    _HAS_QWEN3_MOE = True
except ImportError:
    Qwen3MoeForCausalLM = None
    _HAS_QWEN3_MOE = False

from get_scale_quant import get_normal_lora
from sketch.r1_sketch import get_best_sketch_fp16_ret
import quantizer as quant_module
from utils.moe_utils import get_moe_qlayers_name, find_layers, is_shared_expert
from utils.get_calib_data import get_wikitext2_
import importlib

import quantizer.vector_quant_utils_lora as _vqmod

# Patch VQQuantizer: hardcoded columns_per_group=256 doesn't divide Qwen's
# expert dim (1408). Find the largest valid value, falling back to row grouping.
_OrigVQQuantizer = _vqmod.VQQuantizer
class _PatchedVQQuantizer(_OrigVQQuantizer):
    def get_groupsize(self, X, groupsize):
        import numpy as np
        if self.columns_per_group is not None:
            cpg = self.columns_per_group
            if groupsize < cpg:
                cpg = groupsize if cpg % groupsize == 0 else 1
            # Try cpg, cpg//2, cpg//4, ... down to 1
            candidate = cpg
            while candidate >= 1:
                if (groupsize % candidate == 0 and
                        X.shape[1] % candidate == 0):
                    rpg = groupsize // candidate
                    if X.shape[0] % rpg == 0:
                        self.columns_per_group = candidate
                        self.rows_per_group    = rpg
                        self.groups_per_column = X.shape[0] // rpg
                        return candidate
                candidate //= 2
            # No valid columns_per_group — fall through to row grouping
            self.columns_per_group = None

        # Fallback: each row is its own group (groups_per_column = out_d)
        self.rows_per_group    = 1
        self.groups_per_column = X.shape[0]
        return X.shape[1]   # groupsize = full row

_vqmod.VQQuantizer = _PatchedVQQuantizer

# Patch GPTVQ_lora.fasterquant: auto-adjust ha_bsize/id_bsize to the largest
# power-of-2 that divides W.shape[1] (TileQ's Walsh-Hadamard requires m % n == 0
# and n must be a power of 2; Qwen's in_d=1408 needs 128, not 256).
_OrigGPTVQ = _vqmod.GPTVQ_lora
_orig_fasterquant = _OrigGPTVQ.fasterquant

def _patched_fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1,
                         actorder=False, static_groups=False,
                         include_m_step=False, use_vq=False, svd_rank=None,
                         hessian_weighted_lookups=False, only_init_kmeans=False,
                         ha_bsize=256, id_bsize=256):
    if self.lora.get("U") is None:
        return _orig_fasterquant(self, blocksize=blocksize, percdamp=percdamp,
            groupsize=groupsize, actorder=actorder, static_groups=static_groups,
            include_m_step=include_m_step, use_vq=use_vq, svd_rank=svd_rank,
            hessian_weighted_lookups=hessian_weighted_lookups,
            only_init_kmeans=only_init_kmeans,
            ha_bsize=ha_bsize, id_bsize=id_bsize)
    W = self.layer.weight.data
    in_d = W.shape[1]
    # Find largest power-of-2 that both divides in_d AND divides in_d//p (Hadamard needs m%n==0)
    # Simplest: largest power-of-2 factor of in_d
    p = 1
    while p * 2 <= ha_bsize and in_d % (p * 2) == 0:
        p *= 2
    ha_bsize = p
    id_bsize = min(id_bsize, ha_bsize)
    return _orig_fasterquant(self, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, actorder=actorder, static_groups=static_groups,
        include_m_step=include_m_step, use_vq=use_vq, svd_rank=svd_rank,
        hessian_weighted_lookups=hessian_weighted_lookups,
        only_init_kmeans=only_init_kmeans,
        ha_bsize=ha_bsize, id_bsize=id_bsize)

_OrigGPTVQ.fasterquant = _patched_fasterquant

from quantizer.vector_quant_utils_lora import gptvq_fwrd_lora

from joint_optim import GPTQJoint

DEV            = torch.device('cuda:0')
qtype          = torch.float16
SCALE_CLAMP_MIN = 1e-4
SCALE_TIME      = 2.4


def _fake_quant_int8_col(x: torch.Tensor) -> torch.Tensor:
    """Per-column symmetric int8 fake-quant (round-trip fp16→int8→fp16)."""
    scale = x.abs().max(dim=0, keepdim=True).values.clamp(min=1e-8) / 127.0
    return (x / scale).round().clamp(-128, 127).mul(scale).to(x.dtype)


def _fake_quant_int8_row(x: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric int8 fake-quant (round-trip fp16→int8→fp16)."""
    scale = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
    return (x / scale).round().clamp(-128, 127).mul(scale).to(x.dtype)


def _reoptimize_U_given_Vq(big_tensor: torch.Tensor, Vh_q: torch.Tensor,
                            Si: torch.Tensor) -> torch.Tensor:
    """Given int8-quantized Vh_all and singular values, solve for U that minimizes
    ||big_tensor - U @ diag(Si) @ Vh_q||_F (closed-form least-squares).
    big_tensor: (in_d, n_g*out_d) float64
    Vh_q: (srank, n_g*out_d) float16
    Si: (srank,) float16
    Returns: U_new (in_d, srank) float16
    """
    B = torch.diag(Si.double()) @ Vh_q.double()  # (srank, n_g*out_d)
    BBT = B @ B.T                                  # (srank, srank)
    BBT_inv = torch.linalg.pinv(BBT)
    U_new = (big_tensor @ B.T @ BBT_inv).to(torch.float16)  # (in_d, srank)
    return U_new


# ---------------------------------------------------------------------------
# Real-quant packing helpers (int8 absmax)
# ---------------------------------------------------------------------------
def _quant_intN_absmax(tensor: torch.Tensor, nbits: int):
    """Per-column absmax symmetric quant. Returns (q_int8, scale_fp16)."""
    maxval = 2 ** (nbits - 1) - 1
    scale  = tensor.float().abs().max(dim=0).values.clamp(min=1e-8)
    q      = (tensor.float() / scale * maxval).round().clamp(-maxval, maxval).to(torch.int8)
    return q, scale.half()


def _quant_intN_absmax_rowwise(tensor: torch.Tensor, nbits: int):
    """Per-row absmax symmetric quant. Returns (q_int8, scale_fp16 (rows,1))."""
    maxval = 2 ** (nbits - 1) - 1
    scale  = tensor.float().abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    q      = (tensor.float() / scale * maxval).round().clamp(-maxval, maxval).to(torch.int8)
    return q, scale.half()


def _dequant_intN(q: torch.Tensor, scale: torch.Tensor, nbits: int) -> torch.Tensor:
    """Dequantize intN (stored as int8) → float32. Per-column scale."""
    maxval = 2 ** (nbits - 1) - 1
    return q.float() / maxval * scale.float()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_blocks(model):
    if isinstance(model, (LlamaForCausalLM,)) or \
            model.__class__.__name__ == "LlamaForCausalLM":
        return model.model.layers
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers
    elif isinstance(model, (MixtralForCausalLM, Qwen2MoeForCausalLM)):
        return model.model.layers
    elif _HAS_QWEN3_MOE and isinstance(model, Qwen3MoeForCausalLM):
        return model.model.layers
    else:
        raise NotImplementedError(type(model))


def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM,)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb   = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, (MixtralForCausalLM, Qwen2MoeForCausalLM)) or \
         (_HAS_QWEN3_MOE and isinstance(model, Qwen3MoeForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        # Qwen2Moe/Qwen3Moe have a top-level rotary_emb; Mixtral does NOT
        # (rotary is per-attention-layer). Guard the attribute.
        if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
            model.model.rotary_emb = model.model.rotary_emb.to(device)
    else:
        raise NotImplementedError(type(model))


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_module(root, path):
    cur = root
    for part in path.split('.'):
        cur = getattr(cur, part)
    return cur


def compute_act_scales(input_feat, device, scale_time=SCALE_TIME):
    """TileQ activation scale: mean(|feat|)^scale_time, auto-fallback if ratio too large."""
    max_attempts, attempt = 10, 0
    st = scale_time
    while attempt < max_attempts:
        mean_feat = input_feat.abs().view(-1, input_feat.shape[-1]).mean(0)
        mean_feat = mean_feat.pow(st).to(device)
        mean_feat[torch.isinf(mean_feat)] = torch.finfo(mean_feat.dtype).max / 2
        if mean_feat.dtype == torch.float16:
            mean_feat = mean_feat.clamp(min=SCALE_CLAMP_MIN)
        elif mean_feat.dtype == torch.bfloat16:
            mean_feat = mean_feat.clamp(min=1e-14)
        else:
            mean_feat = mean_feat.clamp(min=1e-15)
        scales = mean_feat / (
            (mean_feat.max().float() * mean_feat.min().float()).sqrt().to(mean_feat.dtype)
        )
        if not torch.any(torch.isnan(scales)) and scales.max() / scales.min() < 1e5:
            return scales
        st = max(0.5, st - 0.2)
        attempt += 1

    # fallback: all-ones
    return torch.ones(input_feat.shape[-1], device=device, dtype=input_feat.dtype)


# ---------------------------------------------------------------------------
# Phase 1: collect activation scales per MoE expert
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_phase1(model, layers, dataloader, nsamples, fix_rank, qbit, group_size,
                   lora_bit, lora_iter, quant_infos, wtypes):
    """
    For each transformer layer (TileQ-style: one sample at a time to avoid OOM):
      - Capture input features via forward hooks.
      - For attention/normal layers: compute lora via get_normal_lora (TileQ style).
      - For MoE routing experts: compute activation scales only (defer SVD to Phase 2).
      - Advance `inps` through the layer to feed the next.

    Returns:
      WR              : list[dict] — lora dicts per layer (attention filled, MoE empty)
      all_expert_recs : list[dict] — {layer, name, scales, Sa, W_orig, shape}
    """
    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    seqlen = model.seqlen

    inps         = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=DEV)
    layer_kwargs = {}
    cache_i      = [0]

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            inps[cache_i[0]] = inp
            cache_i[0] += 1
            raise ValueError
        def __getattr__(self, name):
            if name == "module":
                return self._modules["module"]
            try:
                return getattr(self._modules["module"], name)
            except KeyError:
                raise AttributeError(name)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # Keep embed_tokens and rotary_emb on GPU — needed again in Phase 3 (gptvq_fwrd_lora)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    WR              = [{} for _ in layers]
    all_expert_recs = []

    for i in tqdm.tqdm(range(len(layers)), desc="Phase 1: collect scales"):
        layer = layers[i].cuda()

        named_linears = get_named_linears(layer)

        def cache_hook(m, x, y, name, feat_dict):
            feat_dict[name].append(x[0].detach().cpu())

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_hook, name=name, feat_dict=input_feat)
                )
            )
        # Process one sample at a time to avoid OOM on attention layers
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()
        inps, outs = outs, inps
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Unified name-based dispatch. utils/moe_utils._has_experts_segment
        # already handles both Qwen (`.mlp.experts.`) and Mixtral
        # (`.block_sparse_moe.experts.`) name patterns.
        normal_qlist, shared_qlist, regular_qlist = get_moe_qlayers_name(named_linears)
        _wtypes = wtypes

        # Attention / shared expert: get_normal_lora (TileQ unchanged).
        # We also categorize params for the corrected avg-bits report at the end
        # (shared_expert stays fp16 in E11, attention is later re-quantized at attn_bits).
        for name in normal_qlist + shared_qlist:
            if name not in input_feat:
                continue
            module = get_module(layer, name)
            n_params = module.weight.shape[0] * module.weight.shape[1]
            _lora_size_before = quant_infos.get('lora_size', 0.0)
            WR[i][name] = get_normal_lora(
                module, input_feat[name], qbit, group_size,
                fix_rank, 0.1, quant_infos,
                lora_bit=lora_bit, lora_iter=lora_iter,
            )
            _lora_added = quant_infos.get('lora_size', 0.0) - _lora_size_before
            if is_shared_expert(name):
                # shared_expert stays fp16 in E11 → don't count in avg-bits denominator
                quant_infos['shared_params'] = quant_infos.get('shared_params', 0.0) + n_params
            else:
                # Attention layer — track separately so we can apply the real attn_bits
                # at report time (get_normal_lora added quant_size at qbit, not attn_bits).
                quant_infos['attn_params']    = quant_infos.get('attn_params', 0.0) + n_params
                quant_infos['attn_lora_bits'] = quant_infos.get('attn_lora_bits', 0.0) + _lora_added

        # MoE routing experts: collect scales only
        for name in regular_qlist:
            if name not in input_feat:
                feat = torch.ones(2, get_module(layer, name).weight.shape[1],
                                  dtype=qtype)
            else:
                feat = input_feat[name]

            module = get_module(layer, name)
            W      = module.weight.data.detach().clone()   # (out_d, in_d)
            scales = compute_act_scales(feat, W.device)
            Sa     = (torch.tensor(1.0, dtype=torch.float32) / scales.float()).cpu()

            all_expert_recs.append({
                'layer':  i,
                'name':   name,
                'scales': scales.cpu().float(),
                'Sa':     Sa,
                'W_orig': W.cpu(),
                'shape':  tuple(W.shape),   # (out_d, in_d)
            })

        del input_feat, handles
        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    del inps, outs
    return WR, all_expert_recs


# ---------------------------------------------------------------------------
# Phase 2: cross-layer group sharing → shared SVD → fill WR
# ---------------------------------------------------------------------------
@torch.no_grad()
def fill_phase2(WR, all_expert_recs, fix_rank, lora_bit, lora_iter, qbit,
                G, quant_infos, wtypes, int8_lora=False, int8_lora_v=False,
                max_err_threshold=60.0, cluster_method='traversal',
                cluster_recon_weight=0.0, cluster_seed=42, cluster_rank=None):
    """
    Group G routing experts of the same type from different layers.
    Stack their activation-scaled weights as (in_d, G*out_d) → shared SVD.
    Fill WR[layer][name] = {U: shared, V: per-expert, Si: shared, Sa: per-expert}.

    Grouping metric (`cluster_method`):
      - 'traversal' (default): flat slice of collect order; behavior unchanged
        from prior HEAD. Same-group experts are consecutive in insertion order.
      - 'grassmannian': subspace-distance spectral clustering. Same-group experts
        share high overlap in the top-r singular subspace of activation-scaled W.
        Attention wtypes are ignored (this pipeline routes attn through
        `gptq_attn_4bit`, not through fill_phase2).
    """
    # Separate by weight type
    type_to_recs = defaultdict(list)
    for r in all_expert_recs:
        for wt in wtypes:
            if wt in r['name']:
                type_to_recs[wt].append(r)
                break

    # Optional Grassmannian reordering: reorder each type_to_recs[wt] so that
    # a slice of size G contains members of the same cluster.
    #
    # HEAD's Phase 2 uses G as the SVD-group size (# experts per shared U),
    # not # clusters. We therefore ask cluster_residuals for
    #     n_clusters = ceil(n_total / G)
    # so that each cluster is roughly G experts, and consecutive slices of
    # size G after argsort(labels) land in the same cluster (up to spectral
    # imbalance, which we accept as an approximation).
    if cluster_method == 'grassmannian':
        try:
            from cross_layer_share import cluster_residuals
        except ImportError as e:
            raise RuntimeError(
                "cluster_method='grassmannian' requires cross_layer_share.py "
                "at repo root; got ImportError: " + str(e))

        print(f"\n[Phase 2 pre] cluster_method=grassmannian "
              f"(recon_weight={cluster_recon_weight}, seed={cluster_seed})", flush=True)

        # Number of clusters per wtype = ceil(N_wtype / G). All MoE wtypes share
        # the same N here (one entry per expert per layer per wtype), so a single
        # value is fine.
        wt_order = list(wtypes)
        n_per_wtype = max((len(type_to_recs[wt]) for wt in wt_order), default=0)
        n_clusters = max(1, (n_per_wtype + G - 1) // G)
        print(f"[Phase 2 pre] target n_clusters per wtype = "
              f"ceil({n_per_wtype}/{G}) = {n_clusters}", flush=True)

        # Auto-fallback: Grassmannian needs enough clusters to be discriminative.
        # Empirically Mixtral-8x7B (256 experts, 4 clusters at G=64) degrades PPL
        # by +1.3 vs traversal because each cluster absorbs too many heterogeneous
        # experts. Fall back to traversal when n_clusters is too small.
        _GRASS_MIN_CLUSTERS = 8
        _grass_ok = (n_clusters >= _GRASS_MIN_CLUSTERS)
        if not _grass_ok:
            print(f"[Phase 2 pre] AUTO-FALLBACK to traversal: n_clusters={n_clusters} "
                  f"< {_GRASS_MIN_CLUSTERS} (Grassmannian too coarse for this "
                  f"expert count; small MoE like Mixtral).", flush=True)
            cluster_method = 'traversal'

        if _grass_ok:
            all_residuals = []
            for wt in wt_order:
                for r in type_to_recs[wt]:
                    all_residuals.append({
                        'type': wt,
                        'weight_orig': r['W_orig'],
                        '_act_scale_cpu': r['scales'],
                        'hessian_diag': None,
                    })

            _cluster_rank_eff = cluster_rank if cluster_rank is not None else fix_rank
            print(f"[Phase 2 pre] clustering rank = {_cluster_rank_eff} "
                  f"(fix_rank={fix_rank})", flush=True)
            assignments, _wtype_indices = cluster_residuals(
                all_residuals, rank=fix_rank,
                G_moe=n_clusters, G_attn=n_clusters,
                seed=cluster_seed, share_attn=False,
                hessian_svd=False, recon_weight=cluster_recon_weight,
                rank_cluster=_cluster_rank_eff,
                cluster_on_original=True,
            )

            # Reorder each type_to_recs[wt] by group label (stable) so consecutive
            # slices of size G land in the same cluster.
            import numpy as _np
            for wt in wt_order:
                if wt not in assignments:
                    continue
                labels = assignments[wt]
                order  = _np.argsort(labels, kind='stable')
                type_to_recs[wt] = [type_to_recs[wt][int(i)] for i in order]
            print("[Phase 2 pre] cluster_residuals done; type_to_recs reordered.\n", flush=True)

    for wtype, recs in type_to_recs.items():
        if not recs:
            continue
        in_d  = recs[0]['shape'][1]
        out_d = recs[0]['shape'][0]
        n_total = len(recs)
        print(f"\n[Phase 2] {wtype}: {n_total} experts → groups of {G}, "
              f"in_d={in_d}, out_d={out_d}, rank={fix_rank}", flush=True)

        for g_start in tqdm.tqdm(range(0, n_total, G),
                                  desc=f"  SVD {wtype}", leave=False):
            group = recs[g_start:g_start + G]
            n_g   = len(group)

            # --- Stack activation-scaled weights: (in_d, n_g * out_d) ---
            W_cols = []
            for r in group:
                scales_k = r['scales'].cuda().float()
                W_k      = r['W_orig'].cuda().float()
                # W_scale_T_k = diag(scales) @ W.T, shape (in_d, out_d)
                W_st = torch.diag(scales_k) @ W_k.T
                W_cols.append(W_st)
            big_tensor = torch.cat(W_cols, dim=1).double()  # (in_d, n_g*out_d) float64
            del W_cols
            gc.collect()
            torch.cuda.empty_cache()

            # --- Rank-1 sketch (TileQ style) ---
            _, U_list, Vh_list, S_list, _, _, srank = get_best_sketch_fp16_ret(
                big_tensor, qbit, fix_rank=fix_rank, max_sketch_iter=lora_iter
            )
            if not int8_lora_v:
                del big_tensor
                torch.cuda.empty_cache()

            if srank == 0:
                if int8_lora_v:
                    del big_tensor
                for r in group:
                    WR[r['layer']][r['name']] = {'U': None}
                continue

            # U_shared: (in_d, srank) — SHARED across all n_g experts
            U_shared = torch.vstack(
                [t.to(torch.float16) for t in U_list[:srank]]
            ).T.to(DEV)    # (in_d, srank)

            # Vh_all: (srank, n_g*out_d) — split per expert
            Vh_all = torch.vstack(
                [t.to(torch.float16) for t in Vh_list[:srank]]
            ).to(DEV)      # (srank, n_g*out_d)

            Si = torch.tensor(S_list[:srank], dtype=qtype).to(DEV)  # (srank,)

            # --- Optional int8 for V with U re-optimization ---
            if int8_lora_v:
                Vh_all = _fake_quant_int8_row(Vh_all.clone())   # int8 per row (per singular component)
                U_shared = _reoptimize_U_given_Vq(big_tensor, Vh_all, Si)
                del big_tensor
                torch.cuda.empty_cache()
            elif int8_lora:
                U_shared = _fake_quant_int8_col(U_shared)

            # --- Distribute per-expert V, fill WR ---
            for k, r in enumerate(group):
                V_k  = Vh_all[:, k * out_d:(k + 1) * out_d]  # (srank, out_d)
                Sa_k = r['Sa'].to(DEV).to(qtype)               # (in_d,)

                # Sanity check: |(W - lora)|max < max_err_threshold  (TileQ convention, default 60)
                # lora_T = diag(Sa) @ U @ diag(Si) @ V, shape (in_d, out_d)
                USiV = U_shared.float() @ torch.diag(Si.float()) @ V_k.float()  # (in_d, out_d)
                lora_T = Sa_k.float().unsqueeze(1) * USiV                        # broadcast (in_d, out_d)
                lora = lora_T.T                                                   # (out_d, in_d)
                del USiV
                W_orig_dev = r['W_orig'].to(DEV).float()
                max_err = (W_orig_dev - lora).abs().max().item()
                del lora_T, lora, W_orig_dev

                if max_err > max_err_threshold:
                    print(f"    [WARN] {r['layer']}/{r['name']}: max_err={max_err:.1f} > {max_err_threshold:.1f}, skip")
                    WR[r['layer']][r['name']] = {'U': None}
                else:
                    WR[r['layer']][r['name']] = {
                        'U':  U_shared,     # shared — same tensor ref; GPTVQ reads, never writes
                        'V':  V_k,
                        'Si': Si,
                        'Sa': Sa_k,
                    }

                # Bit accounting
                u_bit = 8 if (int8_lora or int8_lora_v) else lora_bit
                v_bit = 8 if int8_lora_v else lora_bit
                _moe_lora_delta = srank * (in_d / n_g) * u_bit + srank * out_d * v_bit
                quant_infos['lora_size']    += _moe_lora_delta
                quant_infos['total_size']   += out_d * in_d * 16
                quant_infos['quant_size']   += out_d * in_d * qbit
                quant_infos['lora_rank']    += srank
                quant_infos['layer_cnt']    += 1
                # Categorized (for corrected avg-bits report)
                quant_infos['moe_params']    = quant_infos.get('moe_params', 0.0) + out_d * in_d
                quant_infos['moe_lora_bits'] = quant_infos.get('moe_lora_bits', 0.0) + _moe_lora_delta

            del U_shared, Vh_all, Si
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n[Phase 2] Done. Filled WR for {len(all_expert_recs)} MoE experts.", flush=True)


def _wr_to_cpu(WR):
    """Move all tensors in WR to CPU so they can be torch.save'd."""
    for layer_dict in WR:
        for name, d in layer_dict.items():
            if isinstance(d, dict):
                layer_dict[name] = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                }
    return WR


def _wr_to_dev(WR, device):
    """Move all tensors in WR to device after loading from cache."""
    for layer_dict in WR:
        for name, d in layer_dict.items():
            if isinstance(d, dict):
                layer_dict[name] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                }
    return WR


# ---------------------------------------------------------------------------
# Phase 2.5: plain scalar GPTQ for attention layers
# ---------------------------------------------------------------------------
@torch.no_grad()
def gptq_attn_4bit(model, layers, dataloader, nsamples, attn_bits=4,
                   groupsize=128, percdamp=0.01):
    """Plain scalar GPTQ (no LoRA) for self_attn.{q,k,v,o}_proj layers."""
    dtype  = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    seqlen = model.seqlen
    inps   = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=DEV)
    cache_i = [0]
    layer_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            inps[cache_i[0]] = inp
            cache_i[0] += 1
            raise ValueError
        def __getattr__(self, name):
            if name == "module":
                return self._modules["module"]
            try:
                return getattr(self._modules["module"], name)
            except KeyError:
                raise AttributeError(name)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for i in tqdm.tqdm(range(len(layers)), desc=f"Attn GPTQ {attn_bits}-bit"):
        layer      = layers[i].to(DEV)
        named_lins = get_named_linears(layer)
        attn_names = [n for n in named_lins if 'self_attn' in n]

        if not attn_names:
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            inps, outs = outs, inps
            layers[i] = layer.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        gptq = {n: GPTQJoint(named_lins[n], nbits=attn_bits, sym=False, mse=True)
                for n in attn_names}

        handles = []
        for n in attn_names:
            def _hook(m, inp, out, _n=n):
                gptq[_n].add_batch(inp[0], out)
            handles.append(named_lins[n].register_forward_hook(_hook))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for n in attn_names:
            gptq[n].prepare_hessian(percdamp=percdamp, act_alpha=0.0)
            _, Q_W, packed = gptq[n].fasterquant(W_lora=None, groupsize=groupsize, real_quant=True)
            named_lins[n].weight.data = Q_W.to(dtype).to(DEV)
            named_lins[n].gptq_packed = packed

        # Advance inps with quantized attention weights
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Real-quant export: dump U/SV/Sa to cross_layer_info.pt
# ---------------------------------------------------------------------------
@torch.no_grad()
def _strip_fp16_quantized_weights(output_path, assignments, attn_gptq_packs, vq_residuals):
    """Remove fp16 weights for quantized layers from safetensors shards.

    This is the main storage win. Fake-quant save_pretrained writes full fp16
    weights for ALL Linear layers, but for real-quant inference the quantized
    experts' fp16 weights are redundant — inference rebuilds them from
    cross_layer_info.pt's VQ codes / centroids / LoRA. Same for attention
    layers if attn_bits<16 (GPTQ int4 packed in attn_gptq_packs).

    IMPORTANT: only strips layers that ACTUALLY got VQ-quantized. Experts
    skipped by the max_err filter remain as fp16 `Fp16LinearShim` at inference
    time and MUST keep their fp16 weight. Those live in `assignments` but
    have `vq_residuals[wt][local_idx] is None` — we exclude those.

    On Qwen1.5-MoE-A2.7B: this drops safetensors from ~28 GB → ~1 GB.
    Total real-quant checkpoint size: ~32 GB → ~6 GB.

    Writes a marker file `.stripped_real_quant` so the inference loader
    knows to use the meta-device + partial-state-dict path.
    """
    import glob, json
    from safetensors.torch import safe_open, save_file
    from safetensors import safe_open as _safe_open_meta

    # 1. Build set of param names to strip — ONLY layers that were actually
    # VQ-quantized. Skipped experts (kept as Fp16LinearShim) MUST retain fp16.
    to_strip = set()
    for wt, records in assignments.items():
        vq_list = vq_residuals.get(wt, [None] * len(records))
        for local_idx, r in enumerate(records):
            if local_idx >= len(vq_list) or vq_list[local_idx] is None:
                # Skipped by max_err filter — kept as Fp16LinearShim, needs fp16 weight.
                continue
            L, E = r['layer'], r['expert']
            for prefix in ('mlp.experts', 'block_sparse_moe.experts'):
                to_strip.add(f"model.layers.{L}.{prefix}.{E}.{wt}.weight")
    for (L, an) in attn_gptq_packs.keys():
        to_strip.add(f"model.layers.{L}.self_attn.{an}.weight")

    # 2. Find safetensors shards
    shards = sorted(glob.glob(os.path.join(output_path, 'model-*.safetensors')))
    if not shards:
        one = os.path.join(output_path, 'model.safetensors')
        if os.path.exists(one):
            shards = [one]

    if not shards:
        print("  [strip] no safetensors shards found — nothing to strip")
        return

    # 3. Rewrite each shard, dropping to_strip entries
    total_stripped = 0
    total_bytes_saved = 0
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        pre_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        for name in list(tensors.keys()):
            if name in to_strip:
                total_bytes_saved += tensors[name].numel() * tensors[name].element_size()
                del tensors[name]
                total_stripped += 1
        # If shard becomes empty, save an empty file anyway (index.json still points here)
        save_file(tensors, shard)

    # 4. Update model.safetensors.index.json — remove stripped names from weight_map
    index_path = os.path.join(output_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        wm = index.get('weight_map', {})
        for k in list(wm.keys()):
            if k in to_strip:
                del wm[k]
        index['weight_map'] = wm
        # Also update total_size in metadata (best-effort)
        if 'metadata' in index and 'total_size' in index['metadata']:
            index['metadata']['total_size'] -= total_bytes_saved
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    # 5. Write marker file so inference loader knows this checkpoint is stripped
    with open(os.path.join(output_path, '.stripped_real_quant'), 'w') as f:
        f.write("Real-quant checkpoint with fp16 weights stripped for quantized experts.\n")
        f.write(f"stripped_count={total_stripped}\n")
        f.write(f"bytes_saved={total_bytes_saved}\n")

    print(f"  [strip] Removed {total_stripped} fp16 weight tensors "
          f"({total_bytes_saved/1e9:.2f} GB saved)")


def _export_real_quant_pack(WR, output_path, model_path, *,
                             fix_rank, G, qbit, group_size, attn_bits,
                             int8_lora, int8_lora_v, model=None):
    """Pack U/V*Si/Sa as int8 + scale into cross_layer_info.pt for real-quant inference.

    Layout:
      shared_matrices[wtype][gid] = {"U_int8": (in_d, srank) int8, "U_scale": (srank,) fp16}
      per_expert_V[wtype]         = list of dicts (ordered by traversal index local_idx):
            {"SV_int8": (out_d, srank) int8, "SV_scale": (srank,) fp16,
             "Sa": (in_d,) fp16}
      assignments[wtype]          = list of dicts {layer, expert, group_id, local_idx}
      config                      = {...}
    """
    import os
    from collections import defaultdict

    wtypes = ['gate_proj', 'up_proj', 'down_proj']
    # Some models use w1/w2/w3 — detect via first non-empty WR record
    if WR and WR[0]:
        first_names = list(WR[0].keys())
        if any('w1' in n or 'w2' in n or 'w3' in n for n in first_names):
            wtypes = ['w1', 'w2', 'w3']

    # 1. Discover group assignments via U identity (Python id())
    # Each unique U_shared tensor → one (wtype, group_id)
    u_id_to_gid = {}    # (wtype, id(U)) -> group_id
    next_gid = defaultdict(int)
    shared_matrices = {wt: {} for wt in wtypes}
    per_expert_V    = {wt: [] for wt in wtypes}
    assignments     = {wt: [] for wt in wtypes}

    def _match_wtype(name):
        for wt in wtypes:
            if wt in name:
                return wt
        return None

    for layer_idx, layer_dict in enumerate(WR):
        for name, d in layer_dict.items():
            if not isinstance(d, dict):
                continue
            if d.get('U') is None:
                continue
            wt = _match_wtype(name)
            if wt is None:
                continue
            # Only export routing experts (cross-layer shared U). Shared experts &
            # attention use per-layer get_normal_lora and are already fused into
            # safetensors fp16 weights — no LoRA correction needed at inference.
            if '.experts.' not in name:
                continue
            U   = d['U']
            V   = d['V']
            Si  = d['Si']
            Sa  = d['Sa']

            # Group id by U identity
            key = (wt, id(U))
            if key not in u_id_to_gid:
                gid = next_gid[wt]
                next_gid[wt] += 1
                u_id_to_gid[key] = gid
                if int8_lora:
                    # Quantize U once per unique tensor
                    U_int8, U_scale = _quant_intN_absmax(U.to(torch.float16).cuda(), 8)
                    shared_matrices[wt][gid] = {
                        'U_int8':  U_int8.cpu(),
                        'U_scale': U_scale.cpu(),
                    }
                else:
                    # fp16 storage — for models where int8 LoRA loses too much
                    # precision (e.g. Mixtral, whose 4096×14336 dims amplify int8
                    # error). ~2× LoRA storage cost but correctness-preserving.
                    shared_matrices[wt][gid] = {
                        'U_fp16': U.to(torch.float16).cpu(),
                    }
            gid = u_id_to_gid[key]

            # Extract expert idx from name like "mlp.experts.7.gate_proj"
            expert_idx = -1
            for tok in name.split('.'):
                if tok.isdigit():
                    expert_idx = int(tok)
                    break

            # Per-expert SV = V_k * Si (broadcast). V_k: (srank, out_d), Si: (srank,)
            # Inference convention: SV stored as (out_d, srank)
            SV = (V.to(torch.float16).cuda() * Si.to(torch.float16).cuda().unsqueeze(1)).T  # (out_d, srank)
            entry = {'Sa': Sa.to(torch.float16).cpu()}
            if int8_lora_v:
                SV_int8, SV_scale = _quant_intN_absmax(SV, 8)
                entry['SV_int8']  = SV_int8.cpu()
                entry['SV_scale'] = SV_scale.cpu()
            else:
                entry['SV_fp16']  = SV.to(torch.float16).cpu()

            local_idx = len(per_expert_V[wt])
            per_expert_V[wt].append(entry)
            assignments[wt].append({
                'layer':     layer_idx,
                'expert':    expert_idx,
                'group_id':  gid,
                'local_idx': local_idx,
            })

    config = {
        'model_path': model_path,
        'rank':       fix_rank,
        'G_moe':      G,
        'nbits':      qbit,
        'groupsize':  group_size,
        'attn_bits':  attn_bits,
        'u_bits':     8,
        'sv_bits':    8,
        'method':     'tileq_glorcq_e11',
        'int8_lora':   int8_lora,
        'int8_lora_v': int8_lora_v,
    }

    # ---- VQ residual codes/centroids/perm/diag for routing experts ----
    # `vq_residuals[wt]` is a list aligned with `assignments[wt]` (same local_idx).
    # Each entry: {codes, centroids, perm, diag_signs, vdim, in_d, out_d}.
    vq_residuals = {wt: [None] * len(assignments[wt]) for wt in wtypes}
    # GPTQ-packed dicts for attention layers, keyed by (layer_idx, attn_name).
    attn_gptq_packs = {}

    if model is not None:
        from utils.moe_utils import is_regular_expert
        # Build a name lookup by traversing model.model.layers
        layers = model.model.layers if hasattr(model, 'model') else model.layers
        # For routing experts: find by (layer_idx, name) and match against assignments
        # assignments[wt] entries have layer / expert / local_idx — we re-derive name
        # from layer/expert/wt.

        def _find_routing_linear(layer_module, expert_idx, wt):
            """Return the routing expert linear submodule, or None if not present.

            Handles both Qwen2Moe (`layer.mlp.experts`) and Mixtral
            (`layer.block_sparse_moe.experts`) container attrs.
            """
            container = (
                getattr(layer_module, 'mlp', None)
                or getattr(layer_module, 'block_sparse_moe', None)
            )
            if container is None:
                return None
            experts = getattr(container, 'experts', None)
            if experts is None or expert_idx >= len(experts):
                return None
            return getattr(experts[expert_idx], wt, None)

        # Derive `codes` (uint8) from Q_rotated + centroids via nearest-centroid
        # assignment, so inference-time load skips the expensive derive step.
        # Runs on GPU when the linear's Q_rotated is on GPU (typical during
        # Phase 3), else on CPU.
        @torch.no_grad()
        def _derive_codes(Q, centroids, vdim, in_d):
            out_d, _in_d = Q.shape
            n_vecs = _in_d // vdim
            n_cb, K, _v = centroids.shape
            codes_per_cb = n_vecs // n_cb
            Qf = Q.float()
            cf = centroids.float()
            codes = torch.empty(out_d, n_vecs, dtype=torch.uint8, device=Q.device)
            # Chunk output rows to keep the intermediate distance tensor small.
            chunk = 1024 if codes_per_cb <= 128 else 32
            for cb_id in range(n_cb):
                v_lo = cb_id * codes_per_cb
                v_hi = v_lo + codes_per_cb
                cb = cf[cb_id]
                for r0 in range(0, out_d, chunk):
                    r1 = min(r0 + chunk, out_d)
                    Q_block = Qf[r0:r1, v_lo * vdim:v_hi * vdim].reshape(
                        r1 - r0, codes_per_cb, vdim)
                    d = (Q_block.unsqueeze(2) - cb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
                    codes[r0:r1, v_lo:v_hi] = d.argmin(-1).to(torch.uint8)
            return codes.cpu()

        for wt in wtypes:
            for entry in assignments[wt]:
                li, ei, local = entry['layer'], entry['expert'], entry['local_idx']
                lin = _find_routing_linear(layers[li], ei, wt)
                if lin is None or not hasattr(lin, 'gptvq_Q_rotated'):
                    continue
                vdim = getattr(lin, 'gptvq_vdim', 4)
                # Prefer codes computed inside the quantizer (correct rotated
                # space, matches centroids). Fall back to deriving from Q if the
                # linear didn't attach codes (old training runs).
                codes = getattr(lin, 'gptvq_codes', None)
                if codes is None:
                    codes = _derive_codes(
                        lin.gptvq_Q_rotated, lin.gptvq_centroids,
                        vdim, int(lin.weight.shape[1]),
                    )
                vq_residuals[wt][local] = {
                    # NOTE: `Q_rotated` intentionally omitted — redundant with codes+centroids
                    # (~25× larger than codes; kernel + Python fallback both work from codes).
                    # Codes NOT bit-packed: default codebook is K=256 (8-bit values), so each
                    # code already fills a full uint8 byte. Packing at 4-bit lost half the bits
                    # (verified via inspect: codes reduced to [0,15] after pack/unpack round-trip).
                    'codes':         codes.to(torch.uint8),           # (out_d, in_d/vdim) uint8, 8-bit index
                    'centroids':     lin.gptvq_centroids,             # (n_blocks, K, vdim) fp16
                    'perm':          lin.gptvq_perm,                  # (in_d,) int32
                    'diag_signs':    lin.gptvq_diag_signs,            # (in_d,) fp16
                    'vdim':          vdim,
                    'in_d':          int(lin.weight.shape[1]),
                    'out_d':         int(lin.weight.shape[0]),
                    'rotate_size':   getattr(lin, 'gptvq_rotate_size', 256),
                    'partial_size':  getattr(lin, 'gptvq_partial_size', 256),
                }

        # Attention layers: pull GPTQ packed dicts if available
        for li, layer_module in enumerate(layers):
            attn = getattr(layer_module, 'self_attn', None)
            if attn is None:
                continue
            for an in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                sub = getattr(attn, an, None)
                if sub is None or not hasattr(sub, 'gptq_packed'):
                    continue
                attn_gptq_packs[(li, an)] = sub.gptq_packed

    out_path = os.path.join(output_path, 'cross_layer_info.pt')
    torch.save({
        'config':          config,
        'assignments':     assignments,
        'shared_matrices': shared_matrices,
        'per_expert_V':    per_expert_V,
        'vq_residuals':    vq_residuals,
        'attn_gptq_packs': attn_gptq_packs,
    }, out_path)

    # Return assignments + vq_residuals + attn_gptq_packs so caller can optionally strip fp16.
    # vq_residuals is critical: shim experts (max_err skip) have None entries and must
    # NOT be stripped — the inference loader keeps them as Fp16LinearShim needing fp16.
    return assignments, vq_residuals, attn_gptq_packs

    n_groups   = sum(len(g) for g in shared_matrices.values())
    n_experts  = sum(len(lst) for lst in per_expert_V.values())
    n_vq       = sum(1 for wt in vq_residuals for v in vq_residuals[wt] if v is not None)
    n_attn     = len(attn_gptq_packs)
    print(f"  Wrote {out_path}: {n_groups} shared U groups, {n_experts} expert SV entries, "
          f"{n_vq} VQ residuals, {n_attn} attn GPTQ packs.", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tileq_glorcq(model_path, output_path, qbit=2, fix_rank=32, G=128,
                     group_size=128, lora_bit=16, lora_iter=8,
                     ha_bsize=256, id_bsize=256, use_cache=True, attn_bits=16,
                     phase1_cache_path=None, int8_lora=False, int8_lora_v=False,
                     export_real_quant=True, pool_kmeans=False,
                     strip_fp16_quantized=False, max_err_threshold=60.0,
                     cluster_method='traversal', cluster_recon_weight=0.0,
                     cluster_seed=42, cluster_rank=None):
    # ---- Load model ----
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=qtype, low_cpu_mem_usage=True,
    )
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model.seqlen = 4096

    layers = get_blocks(model)

    nsamples = 128
    dataloader, _, _ = get_wikitext2_(nsamples, 0, model.seqlen, model_path)

    # Weight types for this model
    if isinstance(model, Qwen2MoeForCausalLM):
        wtypes = ['gate_proj', 'up_proj', 'down_proj']
    elif isinstance(model, MixtralForCausalLM):
        wtypes = ['w1', 'w2', 'w3']
    else:
        wtypes = ['gate_proj', 'up_proj', 'down_proj']

    quant_infos = dict(lora_rank=0.0, lora_size=0.0, total_size=0.0,
                       quant_size=0.0, layer_cnt=0.0, origin_size=0.0)

    phase1_cache = phase1_cache_path if phase1_cache_path else (output_path + '_phase1_cache.pt')

    print("=" * 60)
    print(f"  TileQ + GLoRCQ cross-layer sharing")
    print(f"  model:      {model_path}")
    print(f"  output:     {output_path}")
    print(f"  qbit={qbit}  rank={fix_rank}  G={G}  lora_bit={lora_bit}  lora_iter={lora_iter}")
    print(f"  max_err_threshold={max_err_threshold}  (Phase 2 shim skip cutoff)")
    print(f"  wtypes:     {wtypes}")
    print(f"  cache:      {phase1_cache} ({'hit' if (use_cache and os.path.exists(phase1_cache)) else 'miss'})")
    print("=" * 60)

    # ---- Phase 1: collect features + activation scales ----
    if use_cache and os.path.exists(phase1_cache):
        print(f"\n[Phase 1] Loading from cache: {phase1_cache}", flush=True)
        cached = torch.load(phase1_cache, map_location='cpu')
        WR             = cached['WR']
        all_expert_recs = cached['all_expert_recs']
        quant_infos.update(cached.get('quant_infos', {}))
        # Move attention lora tensors to GPU so Phase 3 can use them
        _wr_to_dev(WR, DEV)
        # Also ensure model's embed/rotary are on GPU (needed by gptvq_fwrd_lora)
        move_embed(model, "cuda")
        print(f"  Loaded {len(all_expert_recs)} MoE expert records from cache.", flush=True)
    else:
        print("\n[Phase 1] Feature collection + activation scales ...", flush=True)
        WR, all_expert_recs = collect_phase1(
            model, layers, dataloader, nsamples, fix_rank, qbit, group_size,
            lora_bit, lora_iter, quant_infos, wtypes,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Collected {len(all_expert_recs)} MoE expert records.", flush=True)

        if use_cache:
            print(f"  Saving Phase 1 cache → {phase1_cache}", flush=True)
            torch.save({
                'WR':             _wr_to_cpu([dict(d) for d in WR]),
                'all_expert_recs': all_expert_recs,
                'quant_infos':    dict(quant_infos),
            }, phase1_cache)
            # Restore WR tensors to GPU after saving
            _wr_to_dev(WR, DEV)

    # ---- Phase 2: cross-layer SVD ----
    print("\n[Phase 2] Cross-layer group sharing (G={}) ...".format(G), flush=True)
    fill_phase2(WR, all_expert_recs, fix_rank, lora_bit, lora_iter, qbit,
                G, quant_infos, wtypes, int8_lora=int8_lora, int8_lora_v=int8_lora_v,
                max_err_threshold=max_err_threshold,
                cluster_method=cluster_method,
                cluster_recon_weight=cluster_recon_weight,
                cluster_seed=cluster_seed,
                cluster_rank=cluster_rank)
    del all_expert_recs
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase 2.5: scalar GPTQ for attention (if requested) ----
    if attn_bits < 16:
        print(f"\n[Phase 2.5] {attn_bits}-bit GPTQ for attention layers ...", flush=True)
        gptq_attn_4bit(model, layers, dataloader, nsamples,
                       attn_bits=attn_bits, groupsize=group_size, percdamp=0.01)

    # ---- Phase 3: VQ quantization (TileQ unchanged) ----
    print("\n[Phase 3] VQ quantization of residuals ...", flush=True)
    from argparse import Namespace
    p3_args = Namespace(
        nsamples       = nsamples,
        w_bits         = qbit,
        w_asym         = True,
        int8_down_proj = False,
        w_clip         = True,
        w_groupsize    = group_size,
        percdamp       = 0.01,
        act_order      = False,
        id_bsize       = id_bsize,
        ha_bsize       = ha_bsize,
        pool_kmeans    = pool_kmeans,
    )
    gptvq_fwrd_lora(model, WR, dataloader, DEV, p3_args)

    # ---- Bit stats ----
    q_bit    = 16.0 * (quant_infos['quant_size'] / quant_infos['total_size']) if quant_infos['total_size'] > 0 else 0
    loraq_bit = 16.0 * ((quant_infos['quant_size'] + quant_infos['lora_size']) / quant_infos['total_size']) if quant_infos['total_size'] > 0 else 0
    avg_rank = quant_infos['lora_rank'] / max(1, quant_infos['layer_cnt'])
    print(f"\n  avg_rank={avg_rank:.1f}  qbit={q_bit:.4f}  lora+qbit={loraq_bit:.4f}  [legacy: attn counted as qbit]")

    # ---- Corrected avg bits (attention at actual attn_bits, shared_expert excluded) ----
    attn_p   = quant_infos.get('attn_params', 0.0)
    moe_p    = quant_infos.get('moe_params', 0.0)
    shared_p = quant_infos.get('shared_params', 0.0)
    if attn_p + moe_p > 0:
        # Attention actually stored at attn_bits (if <16, i.e. quantized); shared_expert stays fp16.
        _attn_wbit_actual = attn_bits if attn_bits < 16 else qbit
        attn_wbits = attn_p * _attn_wbit_actual
        moe_wbits  = moe_p  * qbit
        attn_lora  = quant_infos.get('attn_lora_bits', 0.0)
        moe_lora   = quant_infos.get('moe_lora_bits',  0.0)
        total_wbits = attn_wbits + moe_wbits
        total_lora  = attn_lora + moe_lora
        total_p     = attn_p + moe_p
        avg_weight  = total_wbits / total_p
        avg_lora    = total_lora / total_p
        avg_bits    = (total_wbits + total_lora) / total_p
        print(f"\n  [Corrected avg bits over (attn+MoE routing) params, shared_expert excluded]")
        print(f"    attn:   {attn_p/1e6:>7.1f} M params  @ {_attn_wbit_actual}-bit weight  ({attn_wbits/1e6:>10.1f} M bits)")
        print(f"    MoE:    {moe_p/1e9:>7.2f} B params  @ {qbit}-bit weight  ({moe_wbits/1e6:>10.1f} M bits)")
        if shared_p > 0:
            print(f"    shared: {shared_p/1e6:>7.1f} M params  @ FP16 (excluded from avg)")
        print(f"    LoRA:   attn={attn_lora/1e6:.1f}M + MoE={moe_lora/1e6:.1f}M = {total_lora/1e6:.1f} M bits")
        print(f"    → weight={avg_weight:.4f}  lora={avg_lora:.4f}  TOTAL={avg_bits:.4f} bits/param"
              f"  (Extra above qbit={qbit}: {avg_bits - qbit:+.4f})")

    # ---- Save ----
    print(f"\n[save] {output_path}", flush=True)
    model.save_pretrained(output_path)
    enc.save_pretrained(output_path)

    if export_real_quant:
        print("\n[export] Building cross_layer_info.pt (int8 U/SV + Sa + vq_residuals) ...", flush=True)
        assignments_out, vq_residuals_out, attn_gptq_packs_out = _export_real_quant_pack(
            WR, output_path, model_path,
            fix_rank=fix_rank, G=G, qbit=qbit,
            group_size=group_size, attn_bits=attn_bits,
            int8_lora=int8_lora, int8_lora_v=int8_lora_v,
            model=model,
        )
        if strip_fp16_quantized:
            print("\n[strip] Removing fp16 weights for quantized experts from safetensors ...", flush=True)
            _strip_fp16_quantized_weights(output_path, assignments_out, attn_gptq_packs_out, vq_residuals_out)

    print("✓ Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="TileQ + GLoRCQ cross-layer sharing")
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--output_path', type=str, required=True)
    p.add_argument('--qbit',        type=int, default=2)
    p.add_argument('--fix_rank',    type=int, default=32)
    p.add_argument('--G',           type=int, default=128,  help='experts per cross-layer group')
    p.add_argument('--group_size',  type=int, default=128,  help='VQ group size (columns)')
    p.add_argument('--lora_bit',    type=int, default=16,   help='bits for U/V storage')
    p.add_argument('--lora_iter',   type=int, default=8,    help='rank-1 sketch iterations')
    p.add_argument('--ha_bsize',    type=int, default=256)
    p.add_argument('--id_bsize',    type=int, default=256)
    p.add_argument('--no_cache',    action='store_true', default=False,
                   help='force recompute Phase 1 even if cache exists')
    p.add_argument('--attn_bits',   type=int, default=16,
                   help='scalar GPTQ bits for attention layers (16=disable)')
    p.add_argument('--phase1_cache_path', type=str, default=None,
                   help='override Phase 1 cache path (default: output_path + _phase1_cache.pt)')
    p.add_argument('--int8_lora', action='store_true', default=False,
                   help='Fake-quant shared U matrix to int8 before VQ quantization (experiment)')
    p.add_argument('--int8_lora_v', action='store_true', default=False,
                   help='Int8 V_k with U re-optimization to compensate quantization error')
    p.add_argument('--export_real_quant', action='store_true', default=True,
                   help='Write cross_layer_info.pt with int8 U/SV/Sa for real-quant inference')
    p.add_argument('--no_export_real_quant', dest='export_real_quant', action='store_false',
                   help='Disable real-quant export')
    p.add_argument('--pool_kmeans', action='store_true', default=False,
                   help='Use group-shared VQ codebook in Phase 3 (pool k-means across rows)')
    p.add_argument('--strip_fp16_quantized', action='store_true', default=False,
                   help='Strip fp16 weights of quantized experts from safetensors after export '
                        '(saves ~85% checkpoint size; inference loader reconstructs from VQ codes). '
                        'Requires updated model_builder that handles missing tensors.')
    p.add_argument('--max_err_threshold', type=float, default=60.0,
                   help='Phase 2 shim skip threshold: if |W - lora|_max > threshold, expert '
                        'is skipped and kept as Fp16LinearShim at inference. Pass "inf" to '
                        'force all experts to VQ4 (no shim). Default 60.0 (TileQ convention).')
    p.add_argument('--cluster_method', type=str, default='traversal',
                   choices=['traversal', 'grassmannian'],
                   help='Phase 2 expert grouping metric. "traversal" (default) is a flat '
                        'slice of collect order (byte-identical to prior HEAD). "grassmannian" '
                        'runs spectral clustering on the top-r singular subspaces of '
                        'activation-scaled expert weights before Phase 2 SVD.')
    p.add_argument('--cluster_recon_weight', type=float, default=0.0,
                   help='Weight alpha in [0,1] for cross-reconstruction distance in the '
                        'clustering metric: D = (1-alpha)*D_grass + alpha*D_recon. '
                        'Only used when --cluster_method grassmannian. Default 0 (pure Grassmannian).')
    p.add_argument('--cluster_seed', type=int, default=42,
                   help='Random seed for spectral clustering (only used with --cluster_method grassmannian).')
    p.add_argument('--cluster_rank', type=int, default=0,
                   help='Rank used for computing SVD-basis subspace in Grassmannian distance. '
                        '0 = fall back to --fix_rank (default). Old SOTA used 32.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tileq_glorcq(
        model_path  = args.model_path,
        output_path = args.output_path,
        qbit        = args.qbit,
        fix_rank    = args.fix_rank,
        G           = args.G,
        group_size  = args.group_size,
        lora_bit    = args.lora_bit,
        lora_iter   = args.lora_iter,
        ha_bsize    = args.ha_bsize,
        id_bsize    = args.id_bsize,
        use_cache   = not args.no_cache,
        attn_bits   = args.attn_bits,
        phase1_cache_path = args.phase1_cache_path,
        int8_lora   = args.int8_lora,
        int8_lora_v = args.int8_lora_v,
        export_real_quant = args.export_real_quant,
        pool_kmeans = args.pool_kmeans,
        strip_fp16_quantized = args.strip_fp16_quantized,
        max_err_threshold = args.max_err_threshold,
        cluster_method = args.cluster_method,
        cluster_recon_weight = args.cluster_recon_weight,
        cluster_seed = args.cluster_seed,
        cluster_rank = args.cluster_rank if args.cluster_rank > 0 else None,
    )
