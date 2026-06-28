#!/usr/bin/env python3
"""
tileq1d/run_tileq_glorcq.py

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
  Phase 3: TileQ's gptvq_fwrd_lora unchanged — VQ quantizes residuals.
"""

import sys, os

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)                             # GLoRCQ root
_TILEQ  = os.path.normpath(os.path.join(_PARENT, '..', 'tileq'))  # /home/qyyang/repo/tileq

# IMPORTANT: _PARENT must come before _TILEQ so that GLoRCQ transformers imports
# (run_tileq_glorcq itself) resolve correctly, BUT the tileq sub-modules
# (vector_quant_utils_lora etc.) do their wildcard imports AFTER this file's
# top-level setup, so we patch hadamard_utils explicitly below.
for _p in [_PARENT, _TILEQ]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

from get_scale_quant import get_normal_lora
from sketch.r1_sketch import get_best_sketch_fp16_ret
import quantizer as quant_module
from utils.moe_utils import get_moe_qlayers_name, find_layers
from utils.get_calib_data import get_wikitext2_

# TileQ's vector_quant_utils_lora does `from utils.hadamard_utils import *` at module
# load time, but Python resolves `utils` to GLoRCQ's utils/ (it's earlier in sys.path)
# where block_diagonal_walsh_matrix is commented out.  Inject it from TileQ's copy
# into the module's global namespace before importing gptvq_fwrd_lora.
import importlib, types
_tileq_hadamard = importlib.util.spec_from_file_location(
    "tileq_hadamard_utils",
    os.path.join(_TILEQ, "utils", "hadamard_utils.py"),
)
_tileq_hadamard_mod = importlib.util.module_from_spec(_tileq_hadamard)
_tileq_hadamard.loader.exec_module(_tileq_hadamard_mod)

import quantizer.vector_quant_utils_lora as _vqmod
for _name in dir(_tileq_hadamard_mod):
    if not _name.startswith('_'):
        setattr(_vqmod, _name, getattr(_tileq_hadamard_mod, _name))

# TileQ hardcodes columns_per_group=256, but Qwen's MoE expert out_d=1408 only
# divides 64 and 128.  Patch VQQuantizer to pick the largest valid columns_per_group.
_OrigVQQuantizer = _vqmod.VQQuantizer
class _PatchedVQQuantizer(_OrigVQQuantizer):
    def get_groupsize(self, X, groupsize):
        import numpy as np
        # TileQ hardcodes columns_per_group=256, gp=65536.
        # Qwen expert in_d=1408 or out_d=1408 (not divisible by 256).
        # Strategy: try columns_per_group path with decreasing cpg;
        # if none works, fall back to whole-row grouping (one group per row).
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

# joint_optim (GLoRCQ) imports get_moe_config from utils.moe_utils, but sys.path
# has _TILEQ first so sys.modules['utils.moe_utils'] is tileq's version which lacks
# get_moe_config.  Override it with GLoRCQ's version before importing joint_optim.
_glorcq_moe_spec = importlib.util.spec_from_file_location(
    "utils.moe_utils",
    os.path.join(_PARENT, "utils", "moe_utils.py"),
)
_glorcq_moe_mod = importlib.util.module_from_spec(_glorcq_moe_spec)
_glorcq_moe_spec.loader.exec_module(_glorcq_moe_mod)
sys.modules['utils.moe_utils'] = _glorcq_moe_mod

from joint_optim import GPTQJoint

DEV            = torch.device('cuda:0')
qtype          = torch.float16
SCALE_CLAMP_MIN = 1e-4
SCALE_TIME      = 2.4


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
    else:
        raise NotImplementedError(type(model))


def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM,)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb   = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, (MixtralForCausalLM, Qwen2MoeForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb   = model.model.rotary_emb.to(device)
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

        if isinstance(layer, MixtralDecoderLayer):
            normal_qlist  = []  # skip attention for Mixtral (handled elsewhere)
            shared_qlist  = []
            regular_qlist = [n for n in named_linears if any(w in n for w in ['w1','w2','w3'])]
            _wtypes       = ['w1', 'w2', 'w3']
        else:
            normal_qlist, shared_qlist, regular_qlist = get_moe_qlayers_name(named_linears)
            _wtypes = wtypes

        # Attention / shared expert: get_normal_lora (TileQ unchanged)
        for name in normal_qlist + shared_qlist:
            if name not in input_feat:
                continue
            module = get_module(layer, name)
            WR[i][name] = get_normal_lora(
                module, input_feat[name], qbit, group_size,
                fix_rank, 0.1, quant_infos,
                lora_bit=lora_bit, lora_iter=lora_iter,
            )

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
                G, quant_infos, wtypes):
    """
    Group G routing experts of the same type from different layers.
    Stack their activation-scaled weights as (in_d, G*out_d) → shared SVD.
    Fill WR[layer][name] = {U: shared, V: per-expert, Si: shared, Sa: per-expert}.
    """
    # Separate by weight type
    type_to_recs = defaultdict(list)
    for r in all_expert_recs:
        for wt in wtypes:
            if wt in r['name']:
                type_to_recs[wt].append(r)
                break

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
            del big_tensor
            torch.cuda.empty_cache()

            if srank == 0:
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

            # --- Distribute per-expert V, fill WR ---
            for k, r in enumerate(group):
                V_k  = Vh_all[:, k * out_d:(k + 1) * out_d]  # (srank, out_d)
                Sa_k = r['Sa'].to(DEV).to(qtype)               # (in_d,)

                # Sanity check: |(W - lora)|max < 60  (TileQ convention)
                # lora_T = diag(Sa) @ U @ diag(Si) @ V, shape (in_d, out_d)
                USiV = U_shared.float() @ torch.diag(Si.float()) @ V_k.float()  # (in_d, out_d)
                lora_T = Sa_k.float().unsqueeze(1) * USiV                        # broadcast (in_d, out_d)
                lora = lora_T.T                                                   # (out_d, in_d)
                del USiV
                W_orig_dev = r['W_orig'].to(DEV).float()
                max_err = (W_orig_dev - lora).abs().max().item()
                del lora_T, lora, W_orig_dev

                if max_err > 60:
                    print(f"    [WARN] {r['layer']}/{r['name']}: max_err={max_err:.1f} > 60, skip")
                    WR[r['layer']][r['name']] = {'U': None}
                else:
                    WR[r['layer']][r['name']] = {
                        'U':  U_shared,     # shared — same tensor ref; GPTVQ reads, never writes
                        'V':  V_k,
                        'Si': Si,
                        'Sa': Sa_k,
                    }

                # Bit accounting
                quant_infos['lora_size'] += srank * (in_d / n_g + out_d) * lora_bit
                quant_infos['total_size'] += out_d * in_d * 16
                quant_infos['quant_size'] += out_d * in_d * qbit
                quant_infos['lora_rank']  += srank
                quant_infos['layer_cnt']  += 1

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
            _, Q_W = gptq[n].fasterquant(W_lora=None, groupsize=groupsize)
            named_lins[n].weight.data = Q_W.to(dtype).to(DEV)

        # Advance inps with quantized attention weights
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tileq_glorcq(model_path, output_path, qbit=2, fix_rank=32, G=128,
                     group_size=128, lora_bit=16, lora_iter=8,
                     ha_bsize=256, id_bsize=256, use_cache=True, attn_bits=16,
                     phase1_cache_path=None):
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
                G, quant_infos, wtypes)
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
    )
    gptvq_fwrd_lora(model, WR, dataloader, DEV, p3_args)

    # ---- Bit stats ----
    q_bit    = 16.0 * (quant_infos['quant_size'] / quant_infos['total_size']) if quant_infos['total_size'] > 0 else 0
    loraq_bit = 16.0 * ((quant_infos['quant_size'] + quant_infos['lora_size']) / quant_infos['total_size']) if quant_infos['total_size'] > 0 else 0
    avg_rank = quant_infos['lora_rank'] / max(1, quant_infos['layer_cnt'])
    print(f"\n  avg_rank={avg_rank:.1f}  qbit={q_bit:.4f}  lora+qbit={loraq_bit:.4f}")

    # ---- Save ----
    print(f"\n[save] {output_path}", flush=True)
    model.save_pretrained(output_path)
    enc.save_pretrained(output_path)
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
    )
