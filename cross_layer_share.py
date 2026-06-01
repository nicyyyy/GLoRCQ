"""
cross_layer_share.py

Cross-layer expert clustering and U-matrix sharing for GLoRCQ pipeline.

Pipeline:
  Stage 1 : Quantize all MoE experts + attention layers,
            collect residuals  E = W - Q(W)
  Stage 2 : Grassmannian clustering per weight type
            (gate/up/down/q/k/v/o_proj independently)
  Stage 3 : Compute shared U per group (int8 storage, fp16 compute)
            + per-expert V (int8 storage)
  Stage 4 : Fake-quant reconstruction:
            W_approx = Q(W) + U_fp16 @ diag(S) @ V_fp16.T
  Stage 5 : Compute & print average bits/param for the whole model
"""

import math
import gc
import tqdm
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.cluster import SpectralClustering

import quantizer as quant_module
from sketch.r1_sketch import get_best_sketch_fp16_ret
from utils.moe_utils import (
    find_layers,
    is_regular_expert,
    is_shared_expert,
    extract_expert_info,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEV = torch.device("cuda")

_ATTN_WTYPES = {"q_proj", "k_proj", "v_proj", "o_proj"}
_MOE_WTYPES  = {"gate_proj", "up_proj", "down_proj"}
_ALL_WTYPES  = _ATTN_WTYPES | _MOE_WTYPES

# Mixtral uses w1/w2/w3 instead of gate_proj/up_proj/down_proj.
# Map them to canonical names so all downstream code works unchanged.
_MIXTRAL_WTYPE_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


# ---------------------------------------------------------------------------
# Helpers: layer-name parsing
# ---------------------------------------------------------------------------
def _wtype_from_name(name: str):
    """Return the canonical weight type suffix, else None.

    Handles both Qwen (gate_proj/up_proj/down_proj) and Mixtral (w1/w2/w3).
    """
    for wt in _ALL_WTYPES:
        if name == wt or name.endswith("." + wt):
            return wt
    # Mixtral aliases
    suffix = name.rsplit(".", 1)[-1] if "." in name else name
    if suffix in _MIXTRAL_WTYPE_MAP:
        return _MIXTRAL_WTYPE_MAP[suffix]
    return None


def _expert_idx_from_name(name: str) -> int:
    """Return expert index (>=0), -1 for attention, -2 for shared expert."""
    if is_regular_expert(name):
        idx, _ = extract_expert_info(name)
        return idx if idx is not None else -3
    if is_shared_expert(name):
        return -2
    return -1


# ---------------------------------------------------------------------------
# Helpers: embedding movement (generic)
# ---------------------------------------------------------------------------
def _move_model_embeds(model, device):
    """Move embedding / norm layers that must be on the same device as layer 0."""
    m = getattr(model, "model", model)
    for attr in ("embed_tokens", "norm", "rotary_emb"):
        if hasattr(m, attr):
            setattr(m, attr, getattr(m, attr).to(device))
    # OPT-style
    dec = getattr(m, "decoder", None)
    if dec is not None:
        for attr in ("embed_tokens", "embed_positions"):
            if hasattr(dec, attr):
                setattr(dec, attr, getattr(dec, attr).to(device))


# ---------------------------------------------------------------------------
# Stage 1 — GPTQ 1-bit quantizer (no LoRA)
# ---------------------------------------------------------------------------
class GPTQ1bit:
    """Minimal GPTQ for direct n-bit quantization (no LoRA subtraction)."""

    def __init__(self, layer: nn.Linear, nbits: int = 4,
                 sym: bool = False, mse: bool = True):
        self.layer   = layer
        self.dev     = layer.weight.device
        W            = layer.weight.data
        self.rows    = W.shape[0]
        self.columns = W.shape[1]
        self.H       = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        if nbits == 1:
            self.quantizer = quant_module.BiWeightQuantizer()
            self.quantizer.configure(1, perchannel=True, sym=True, mse=False)
        else:
            self.quantizer = quant_module.WeightQuantizer()
            self.quantizer.configure(nbits, perchannel=True, sym=sym, mse=mse)

    def add_batch(self, inp, out):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()        # (columns, batch*seq)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()

    @torch.no_grad()
    def fasterquant(self, blocksize: int = 128, percdamp: float = 0.01,
                    groupsize: int = 128):
        """
        Quantize W in-place using GPTQ with BiWeightQuantizer.
        Returns (W_orig_cpu_fp16, Q_W_cpu_fp16).
        """
        W = self.layer.weight.data.clone().float()   # (out, in)
        W_orig = W.clone().half().cpu()

        H = self.H
        if torch.all(H == 0):
            # No calibration data — skip GPTQ, return original weight
            return W_orig, W_orig.clone()

        # Initial global find_params (fallback for groupsize=-1)
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        try:
            H = torch.linalg.cholesky(H)
        except torch._C._LinAlgError:
            H += torch.eye(self.columns, device=self.dev) * 1e-6
            H = (H + H.T) / 2
            H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2    = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1    = W[:, i1:i2].clone()
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Per-group scale update
                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i]  = q
                err1      = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            Q[:, i1:i2]  = Q1
            W[:, i2:]   -= Err1 @ Hinv[i1:i2, i2:]

        Q = Q.reshape(self.layer.weight.shape).half()
        self.layer.weight.data = Q
        return W_orig, Q.cpu()

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 1 — main: quantize all layers, collect residuals
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_and_collect_residuals(model, layers, dataloader, args):
    """
    Layer-by-layer GPTQ n-bit quantization.

    Returns all_residuals: list of dicts
      {layer, expert, type, module_name, shape,
       weight_orig (CPU fp16), weight_quant (CPU fp16)}
    Only regular experts (expert >= 0) and attention (expert == -1) are tracked;
    shared experts (expert == -2) are quantized but not tracked for sharing.
    """
    nbits = getattr(args, 'qbit', 4)
    sym   = getattr(args, 'sym',  False)
    mse   = getattr(args, 'w_clip', True)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # --- Capture input to layer 0 ---
    dtype = next(iter(model.parameters())).dtype
    inps  = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=DEV,
    )
    cache        = {"i": 0}
    layer_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError
        def __getattr__(self, name):
            if name == "module":
                return self._modules["module"]
            try:
                return getattr(self._modules["module"], name)
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' has no attr '{name}'")

    layers[0] = layers[0].to(DEV)
    _move_model_embeds(model, DEV)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module

    _move_model_embeds(model, "cpu")
    torch.cuda.empty_cache()

    # --- Layer-by-layer GPTQ loop ---
    outs         = torch.zeros_like(inps)
    all_residuals = []

    for layer_idx in tqdm.tqdm(range(len(layers)),
                               desc="Stage 1: GPTQ quantization"):
        layer = layers[layer_idx].to(DEV)
        full  = quant_module.find_qlayers(layer, [nn.Linear])

        # Build GPTQ instances for trackable layers
        gptq = {}
        for name in full:
            if _wtype_from_name(name) is None:
                continue
            if "lm_head" in name:
                continue
            gptq[name] = GPTQ1bit(full[name], nbits=nbits, sym=sym, mse=mse)

        # Register hooks to accumulate Hessian
        def _add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(full[name].register_forward_hook(_add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # Quantize and collect residuals
        for name, g in gptq.items():
            wtype      = _wtype_from_name(name)
            expert_idx = _expert_idx_from_name(name)

            W_orig, Q_W = g.fasterquant(
                percdamp=args.percdamp,
                groupsize=args.groupsize,
            )

            # Track regular experts (>=0) and attention (-1); skip shared (-2)
            if expert_idx >= 0 or expert_idx == -1:
                all_residuals.append({
                    "layer":        layer_idx,
                    "expert":       expert_idx,
                    "type":         wtype,
                    "module_name":  name,
                    "shape":        tuple(W_orig.shape),   # (out, in)
                    "weight_orig":  W_orig,                # CPU fp16
                    "weight_quant": Q_W,                   # CPU fp16
                })
            g.free()
        del gptq

        # Propagate quantized outputs to next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[layer_idx] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print(f"\n[Stage 1] Collected {len(all_residuals)} residual records.")
    return all_residuals


# ---------------------------------------------------------------------------
# Stage 2 — Grassmannian distance + SpectralClustering
# ---------------------------------------------------------------------------
def _compute_svd_u(E_T: torch.Tensor, rank: int) -> np.ndarray:
    """Randomized truncated SVD; returns U (in, rank) as float32 numpy."""
    r = min(rank, E_T.shape[0], E_T.shape[1])
    try:
        U, _, _ = torch.svd_lowrank(E_T.float().to(DEV), q=r, niter=4)
        return U.cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros((E_T.shape[0], r), dtype=np.float32)


def _grassmannian_dist_matrix(E_T_list, rank: int) -> np.ndarray:
    """
    Pairwise Grassmannian geodesic distance on the residual U subspaces.

    d(i,j) = || arccos( σ(U_i^T U_j) ) ||_2  / (√r · π/2)  ∈ [0,1]

    Computation is done on GPU with batched SVD for speed.
    """
    N = len(E_T_list)
    print(f"  [grassmannian] SVD for {N} modules, rank={rank} ...", flush=True)
    U_np = np.stack([_compute_svd_u(E_T, rank) for E_T in E_T_list], axis=0)  # (N, in, r)
    _, _, r = U_np.shape
    max_d = np.sqrt(r) * (np.pi / 2)

    # Move to GPU for batched distance computation
    U_all = torch.from_numpy(U_np).to(DEV)  # (N, in_d, r)
    D = torch.zeros(N, N, device=DEV)

    print(f"  [grassmannian] {N*(N-1)//2} pairs (GPU-batched) ...", flush=True)
    # Process in chunks to limit GPU memory for svdvals
    _chunk = 2048
    for i in range(N - 1):
        U_i_T = U_all[i].T.unsqueeze(0)          # (1, r, in_d)
        rest = U_all[i + 1:]                       # (M, in_d, r)
        M = rest.shape[0]
        dists_list = []
        for c0 in range(0, M, _chunk):
            c1 = min(c0 + _chunk, M)
            M_batch = torch.bmm(
                U_i_T.expand(c1 - c0, -1, -1),    # (chunk, r, in_d)
                rest[c0:c1],                        # (chunk, in_d, r)
            )                                       # (chunk, r, r)
            sigma = torch.linalg.svdvals(M_batch)   # (chunk, r)
            sigma = sigma.clamp(0.0, 1.0)
            angles = torch.arccos(sigma)
            d = angles.pow(2).sum(dim=1).sqrt() / (max_d + 1e-12)
            dists_list.append(d)
        dists = torch.cat(dists_list)
        D[i, i + 1:] = dists
        D[i + 1:, i] = dists
        if (i + 1) % 200 == 0 or i == N - 2:
            print(f"    row {i+1}/{N-1}", flush=True)
    return D.cpu().numpy()


def cluster_residuals(all_residuals, rank: int, G_moe: int, G_attn: int,
                      seed: int = 42, share_attn: bool = False):
    """
    Run Grassmannian SpectralClustering independently for each wtype.

    When share_attn=False (default), attention layers (q/k/v/o_proj) skip
    Grassmannian clustering and each record is assigned its own independent
    group (no cross-layer sharing for attention).

    Returns:
      assignments   : {wtype: ndarray(N,)} — group_id per record
      wtype_indices : {wtype: list[int]}   — indices into all_residuals
    """
    assignments   = {}
    wtype_indices = {}

    for wtype in sorted(_ALL_WTYPES):
        idxs = [i for i, r in enumerate(all_residuals) if r["type"] == wtype]
        if not idxs:
            continue

        N = len(idxs)

        # ------------------------------------------------------------------
        # No-share attention: each record gets its own independent group
        # ------------------------------------------------------------------
        if wtype in _ATTN_WTYPES and not share_attn:
            labels = np.arange(N, dtype=np.int64)
            print(f"\n[Stage 2] {wtype}: N={N}  "
                  f"[no-share-attn] each of {N} records is its own group",
                  flush=True)
            assignments[wtype]   = labels
            wtype_indices[wtype] = idxs
            continue

        G = G_attn if wtype in _ATTN_WTYPES else G_moe
        G = min(G, N)

        print(f"\n[Stage 2] {wtype}: N={N}, G={G}", flush=True)
        subset  = [all_residuals[i] for i in idxs]
        E_T_list = [(r["weight_orig"] - r["weight_quant"]).float().T
                    for r in subset]   # list of (in, out)

        D = _grassmannian_dist_matrix(E_T_list, rank)

        if G <= 1 or N == 1:
            labels = np.zeros(N, dtype=np.int64)
        else:
            pos_vals = D[D > 0]
            sigma2   = float(pos_vals.mean() ** 2) if len(pos_vals) else 1.0
            K        = np.exp(-D ** 2 / (2 * max(sigma2, 1e-6)))
            np.fill_diagonal(K, 1.0)
            sc = SpectralClustering(
                n_clusters=G, affinity="precomputed",
                random_state=seed, assign_labels="kmeans", n_init=5,
            )
            labels = sc.fit_predict(K).astype(np.int64)

        assignments[wtype]   = labels
        wtype_indices[wtype] = idxs
        counts = np.bincount(labels)
        print(f"  → group sizes: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}", flush=True)

    return assignments, wtype_indices


# ---------------------------------------------------------------------------
# Stage 3+4 — Shared U / V computation + fake-quant reconstruction
# ---------------------------------------------------------------------------
def _quant_int8_absmax(T: torch.Tensor):
    """
    Per-column absmax int8 quantization (simulates storage in int8).
    Returns (T_int8: int8, scale: fp16) where scale has shape (n_cols,).
    """
    scale  = T.float().abs().max(dim=0).values.clamp(min=1e-5)  # (cols,)
    T_int8 = (T.float() / scale).mul(127).round().clamp(-127, 127).to(torch.int8)
    return T_int8, scale.half()


def _dequant_int8(T_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 → float32 using per-column scale."""
    return T_int8.float() / 127.0 * scale.float()


def _quant_intN_absmax(tensor: torch.Tensor, nbits: int):
    """
    Per-column absmax symmetric quantization to nbits integers.
    Storage is always int8 tensor (no bit-packing); quantization precision
    is nbits (so INT4 uses 4-bit range [-7, 7] but stored in int8).

    Returns (q_int8: int8, scale: fp16).
    """
    maxval = 2 ** (nbits - 1) - 1          # INT8→127, INT4→7, INT2→1
    scale  = tensor.float().abs().max(dim=0).values.clamp(min=1e-8)
    q      = (tensor.float() / scale * maxval).round().clamp(-maxval, maxval).to(torch.int8)
    return q, scale.half()


def _dequant_intN(q: torch.Tensor, scale: torch.Tensor, nbits: int) -> torch.Tensor:
    """Dequantize intN (stored as int8) → float32 using per-column scale."""
    maxval = 2 ** (nbits - 1) - 1
    return q.float() / maxval * scale.float()


def compute_shared_and_reconstruct(all_residuals, assignments, wtype_indices,
                                    layers, rank: int, analyze: bool = False,
                                    uv_bits: int = 4, hessian_svd: bool = True,
                                    u_bits: int = None, sv_bits: int = None):
    """
    For each (wtype, group_id):
      1. Stack residuals  →  E_cat = [E_1 | ... | E_K]  (in, out*K)
      2. Truncated SVD    →  U (in, rank), S (rank,), V_all (out*K, rank)
      3. Int8-quantize U, V
      4. Fake-quant reconstruct:
             W_approx_i = Q(W_i) + U_fp16 @ diag(S) @ V_i_fp16.T
      5. Store W_approx in each record

    Returns:
      shared_matrices : {wtype: {group_id: {U_int8, U_scale}}}
      per_expert_V    : {wtype: [{SV_int8, SV_scale}]}
                        (same ordering as wtype_indices[wtype])
    """
    u_bits  = u_bits  if u_bits  is not None else uv_bits
    sv_bits = sv_bits if sv_bits is not None else uv_bits
    shared_matrices = {}
    per_expert_V    = {}

    for wtype in sorted(assignments.keys()):
        labels = assignments[wtype]
        idxs   = wtype_indices[wtype]
        subset = [all_residuals[i] for i in idxs]
        G      = int(labels.max()) + 1

        print(f"\n[Stage 3+4] {wtype}: N={len(subset)}, G={G}", flush=True)
        shared_matrices[wtype] = {}
        per_expert_V[wtype]    = [None] * len(subset)

        for group_id in tqdm.tqdm(range(G), desc=f"  {wtype} groups",
                                  leave=False):
            member_lis = [li for li, lbl in enumerate(labels) if lbl == group_id]
            if not member_lis:
                continue
            members = [subset[li] for li in member_lis]

            # Build per-member offsets in the concatenated E_cat
            offsets = []
            start   = 0
            E_list  = []
            for r in members:
                out_d, in_d = r["shape"]
                E_T = (r["weight_orig"] - r["weight_quant"]).float().T.to(DEV)
                E_list.append(E_T)
                offsets.append((start, start + out_d))
                start += out_d

            E_cat     = torch.cat(E_list, dim=1)   # (in_d, Σout_d)
            eff_rank  = min(rank, E_cat.shape[0], E_cat.shape[1])
            E_norm    = E_cat.norm().item()

            # ── Hessian-weighted SVD (Stage 3) ──────────────────────────
            h_diags = [r.get("hessian_diag") for r in members]
            use_hessian = hessian_svd and all(h is not None for h in h_diags)

            if use_hessian:
                h_avg = torch.stack([h.to(DEV) for h in h_diags]).mean(0)  # (in_d,)
                h_sqrt = h_avg.sqrt().clamp(min=1e-6)
                E_cat_w = E_cat * h_sqrt.unsqueeze(1)
            else:
                E_cat_w = E_cat
                h_sqrt = None

            U_w, S, V_all = torch.svd_lowrank(E_cat_w, q=eff_rank, niter=4)

            if h_sqrt is not None:
                U = U_w / h_sqrt.unsqueeze(1)
                U_col_norm = U.norm(dim=0).clamp(min=1e-6)
                U = U / U_col_norm
                S = S * U_col_norm
            else:
                U = U_w

            # U: (in_d, eff_rank), S: (eff_rank,), V_all: (Σout_d, eff_rank)
            print(f"    group {group_id:3d}: ‖E_cat‖={E_norm:.4f}  "
                  f"S.max={S.max().item():.4f}  S.sum={S.sum().item():.4f}  "
                  f"S[0]/S[-1]={S[0].item()/S[-1].item():.1f}"
                  f"{'  [H-weighted]' if use_hessian else ''}",
                  flush=True)

            del E_cat, E_cat_w, E_list
            torch.cuda.empty_cache()

            # Int{u_bits}-quantize shared U
            U_int8, U_scale = _quant_intN_absmax(U, u_bits)    # CPU int8 / fp16
            U_fp16          = _dequant_intN(U_int8.to(DEV), U_scale.to(DEV), u_bits)

            shared_matrices[wtype][group_id] = {
                "U_int8":  U_int8.cpu(),    # (in_d, eff_rank) int8
                "U_scale": U_scale.cpu(),   # (eff_rank,) fp16
                # S is pre-fused into per-expert SV; no longer stored here
            }

            # Per-expert V: Hessian-weighted least-squares or plain slice
            for k, li in enumerate(member_lis):
                r = subset[li]
                h_k = r.get("hessian_diag")

                if use_hessian and h_k is not None:
                    # Hessian-weighted optimal V: min ||(E_k - U @ M_k) @ diag(sqrt(h_k))||_F
                    h_k_dev = h_k.to(DEV)                          # (in_d,)
                    E_T_k = (r["weight_orig"] - r["weight_quant"]).float().T.to(DEV)  # (in_d, out_d)
                    UH = U * h_k_dev.unsqueeze(1)                  # (in_d, rank)
                    G_mat = UH.T @ U                               # (rank, rank)
                    V_k = torch.linalg.solve(G_mat, UH.T @ E_T_k) # (rank, out_d)
                    V_k = V_k.T                                    # (out_d, rank)
                    # Extract S_k (V_k column norms), normalize
                    S_k = V_k.norm(dim=0).clamp(min=1e-6)
                    V_k_normed = V_k / S_k
                else:
                    s_off, e_off = offsets[k]
                    V_k_normed = V_all[s_off:e_off, :]             # (out_d, rank)
                    S_k = S                                        # use shared S

                # Pre-fuse: SV_k = V_k_normed * S_k  (the matrix used at inference)
                SV_k = V_k_normed * S_k.unsqueeze(0)              # (out_d, rank)

                # Int{sv_bits}-quantize SV_k (lower precision, per-expert private)
                SV_int8, SV_scale = _quant_intN_absmax(SV_k, sv_bits)
                SV_fp16           = _dequant_intN(SV_int8.to(DEV), SV_scale.to(DEV), sv_bits)

                per_expert_V[wtype][li] = {
                    "SV_int8":  SV_int8.cpu(),   # (out_d, eff_rank) int8
                    "SV_scale": SV_scale.cpu(),  # (eff_rank,) fp16
                }

                # W_approx = Q(W) + U @ SV_k.T  (SV_k already has S baked in)
                Q_W_T       = r["weight_quant"].float().T.to(DEV)  # (in_d, out_d)
                W_approx_T  = Q_W_T + U_fp16 @ SV_fp16.T
                r["weight_approx"] = W_approx_T.T.half().cpu()     # (out_d, in_d)

                if analyze:
                    E_T_i        = (r["weight_orig"] - r["weight_quant"]).float().T.to(DEV)
                    # Ideal approximation (original U, no intN quantization)
                    SV_k_fp      = V_k_normed.float() * S_k.float().unsqueeze(0)
                    E_lora_fp    = U.float() @ SV_k_fp.T
                    # Actual intN approximation
                    E_lora_int8_ = U_fp16.float() @ SV_fp16.float().T

                    E_norm_i    = E_T_i.norm().item()
                    err_trunc   = (E_T_i - E_lora_fp).norm().item()
                    err_total   = (E_T_i - E_lora_int8_).norm().item()

                    W_orig_T    = r["weight_orig"].float().T.to(DEV)
                    Q_W_T_      = r["weight_quant"].float().T.to(DEV)
                    err_before  = (W_orig_T - Q_W_T_).norm().item()
                    err_after   = (W_orig_T - (Q_W_T_ + E_lora_int8_)).norm().item()

                    print(f"      [{r['layer']:2d}/{r['module_name']}] "
                          f"‖E‖={E_norm_i:.3f}  "
                          f"SVD_cap={E_lora_fp.norm().item()/E_norm_i*100:.1f}%  "
                          f"int{sv_bits}_cap={E_lora_int8_.norm().item()/E_norm_i*100:.1f}%  "
                          f"err_trunc={err_trunc/E_norm_i:.3f}  "
                          f"err_total={err_total/E_norm_i:.3f}  "
                          f"‖W-Q(W)‖={err_before:.3f}→‖W-W_approx‖={err_after:.3f}  "
                          f"{'[WORSE!]' if err_after > err_before else '[OK]'}",
                          flush=True)
                    del E_T_i, E_lora_fp, E_lora_int8_, SV_k_fp, W_orig_T, Q_W_T_
                    torch.cuda.empty_cache()

            del U, S, V_all, U_fp16
            torch.cuda.empty_cache()

    # Write reconstructed weights back to model
    print("\n[Stage 4] Writing reconstructed weights to model ...", flush=True)
    _write_back_weights(all_residuals, wtype_indices, layers)

    return shared_matrices, per_expert_V


def _write_back_weights(all_residuals, wtype_indices, layers):
    """Update model layer weights in-place from the reconstructed tensors."""
    # layer_idx → {module_name: weight_approx}
    layer_updates = defaultdict(dict)
    for wtype, idxs in wtype_indices.items():
        for li, global_i in enumerate(idxs):
            r = all_residuals[global_i]
            if "weight_approx" in r:
                layer_updates[r["layer"]][r["module_name"]] = r["weight_approx"]

    n_actual = 0
    n_expected = sum(len(v) for v in layer_updates.values())
    for layer_idx, updates in tqdm.tqdm(
            sorted(layer_updates.items()), desc="  Writing weights"):
        layer = layers[layer_idx].to(DEV)
        full  = quant_module.find_qlayers(layer, [nn.Linear])
        for name, W_approx in updates.items():
            if name in full:
                W_old = full[name].weight.data.clone()
                full[name].weight.data = W_approx.to(DEV)
                diff = (full[name].weight.data.float() - W_old.float()).norm().item()
                W_norm = W_old.float().norm().item()
                print(f"    layer {layer_idx:3d}  {name}: "
                      f"‖ΔW‖={diff:.4f}  ‖W‖={W_norm:.4f}  "
                      f"rel={diff/W_norm:.4f}", flush=True)
                n_actual += 1
            else:
                print(f"    [WARN] layer {layer_idx}  {name}: NOT FOUND in find_qlayers",
                      flush=True)
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    print(f"  [write-back] Actual writes: {n_actual} / expected: {n_expected} "
          f"(W_approx = Q(W) + U@S@V.T)")


# ---------------------------------------------------------------------------
# Stage 5 — Average bits/param for the whole model
# ---------------------------------------------------------------------------
def compute_avg_bits(all_residuals, assignments, wtype_indices,
                     shared_matrices, rank: int, groupsize: int, nbits: int = 4,
                     uv_bits: int = 4, use_turboquant: bool = False,
                     u_bits: int = None, sv_bits: int = None):
    """
    Compute and print average bits/param for the full quantized model.

    Components (per weight element):
      1. nbits-bit quantized weights
      2. Quantizer scale overhead:
         - GPTQ: per-group fp16 scale (one per 'groupsize' cols per row)
         - TurboQuant: per-row fp16 norm (one per row)
      3. Shared U (u_bits, amortized over K experts in group) + U scale (fp16)
      4. Per-expert SV (sv_bits, pre-fused V*S)              + SV scale (fp16)
    """
    _u_bits  = u_bits  if u_bits  is not None else uv_bits
    _sv_bits = sv_bits if sv_bits is not None else uv_bits
    total_params      = 0
    bits_quant_weight = 0
    bits_gptq_scale   = 0
    bits_U            = 0
    bits_SV           = 0

    for wtype, idxs in wtype_indices.items():
        labels     = assignments[wtype]
        group_K    = defaultdict(int)           # group_id → member count
        for lbl in labels:
            group_K[int(lbl)] += 1

        for li, global_i in enumerate(idxs):
            r        = all_residuals[global_i]
            out_d, in_d = r["shape"]
            n        = out_d * in_d
            g        = int(labels[li])
            K        = group_K[g]
            eff_rank = min(rank, in_d, out_d)  # actual rank used

            total_params      += n
            bits_quant_weight += n * nbits
            rec_method = r.get("quant_method", "turboquant" if use_turboquant else "gptq")
            if rec_method == "turboquant":
                # TurboQuant: per-row fp16 norm (one norm per row, no per-group scale)
                bits_gptq_scale   += out_d * 16
            else:
                # GPTQ: one fp16 scale per row per group-of-columns
                bits_gptq_scale   += out_d * math.ceil(in_d / groupsize) * 16
            # Shared U (intN) + U_scale (fp16, per column) — amortized
            bits_U            += (in_d * eff_rank * _u_bits + eff_rank * 16) / K
            # Per-expert SV (intN, pre-fused) + SV_scale (fp16, per column)
            bits_SV           += out_d * eff_rank * _sv_bits + eff_rank * 16

    if total_params == 0:
        print("[bit-width] No quantized parameters found.")
        return

    def bpp(b):
        return b / total_params

    total_bits = (bits_quant_weight + bits_gptq_scale + bits_U + bits_SV)

    print("\n" + "=" * 57)
    print("  Model Bit-Width Summary (cross-layer sharing)")
    print("=" * 57)
    print(f"  Total quantized params:               {total_params:>14,}")
    print(f"  {nbits}-bit weights:                        {bpp(bits_quant_weight):>9.4f} bits/param")
    scale_label = "Hybrid quant scale" if use_turboquant else "GPTQ scale factors"
    print(f"  {scale_label:25s} (fp16):           {bpp(bits_gptq_scale):>9.4f} bits/param")
    print(f"  Shared U            (int{_u_bits}, amortized):{bpp(bits_U):>9.4f} bits/param")
    print(f"  Per-expert SV       (int{_sv_bits}):            {bpp(bits_SV):>9.4f} bits/param")
    print("  " + "-" * 53)
    print(f"  Total average:                        {bpp(total_bits):>9.4f} bits/param")
    print(f"  Original model (fp16):                {16.0:>9.4f} bits/param")
    print(f"  Compression ratio:                    {16.0 / bpp(total_bits):>9.2f}x")
    print("=" * 57)


# ---------------------------------------------------------------------------
# Save cross_layer_info.pt
# ---------------------------------------------------------------------------
def save_cross_layer_info(output_path, config, assignments, wtype_indices,
                           shared_matrices, per_expert_V, all_residuals):
    """
    Save cross_layer_info.pt alongside the HuggingFace model.

    Structure:
      {
        "config":          dict,
        "assignments":     {wtype: [{layer, expert, group_id}]},
        "shared_matrices": {wtype: {group_id: {U_int8, U_scale, S}}},
        "per_expert_V":    {wtype: [{V_int8, V_scale}]},
      }
    The i-th entry in assignments[wtype] corresponds to the i-th entry in
    per_expert_V[wtype] and to wtype_indices[wtype][i] in all_residuals.
    """
    import os
    os.makedirs(output_path, exist_ok=True)

    assignments_out = {}
    for wtype, idxs in wtype_indices.items():
        labels  = assignments[wtype]
        records = []
        for li, global_i in enumerate(idxs):
            r = all_residuals[global_i]
            records.append({
                "layer":    r["layer"],
                "expert":   r["expert"],
                "group_id": int(labels[li]),
            })
        assignments_out[wtype] = records

    save_dict = {
        "config":          config,
        "assignments":     assignments_out,
        "shared_matrices": shared_matrices,
        "per_expert_V":    per_expert_V,
    }

    path = os.path.join(output_path, "cross_layer_info.pt")
    torch.save(save_dict, path)
    print(f"\n[save] Cross-layer info → {path}")


# ---------------------------------------------------------------------------
# Save real-quantized model (packed weights + LoRA parameters)
# ---------------------------------------------------------------------------
def save_real_quant(output_path, model, all_records, shared_matrices,
                    per_expert_V, assignments, wtype_indices, args):
    """
    Save real-quantized model with packed weights and LoRA parameters.

    Output files:
      output_path/
      ├── config.json           (HF model config)
      ├── tokenizer*/           (tokenizer files, saved separately)
      ├── glorcq_model.pt       (packed quant weights per layer per module)
      └── cross_layer_info.pt   (shared U, per-expert V, assignments)

    glorcq_model.pt structure:
      {
        "model_config": {qbit, rank, groupsize, ...},
        "layers": {
          layer_idx: {
            module_name: {
              # GPTQ: scales, zeros, qweight_int, bits, groupsize, sym
              # TurboQuant: packed_indices, norms, bits, dim
            }
          }
        }
      }
    """
    import os
    os.makedirs(output_path, exist_ok=True)

    model_config = {
        "model_path":     args.model_path,
        "qbit":           args.qbit,
        "rank":           args.rank,
        "groupsize":      args.groupsize,
        "n_iter":         args.n_iter,
        "use_turboquant": args.use_turboquant,
        "act_alpha":      args.act_alpha,
        "uv_bits":        args.uv_bits,
        "rotation_type":  getattr(args, "rotation_type", "qr"),
    }

    # Collect per-layer per-module packed weights
    layers_data = {}
    for r in all_records:
        layer_idx = r["layer"]
        module_name = r["module_name"]

        if layer_idx not in layers_data:
            layers_data[layer_idx] = {}

        if r["quant_method"] == "gptq" and "gptq_packed" in r:
            layers_data[layer_idx][module_name] = r["gptq_packed"]
        elif r["quant_method"] == "turboquant" and "turbo_compressed" in r:
            layers_data[layer_idx][module_name] = r["turbo_compressed"]
        else:
            # Fallback: store dequantized weight (should not happen in real_quant mode)
            layers_data[layer_idx][module_name] = {
                "weight_quant": r["weight_quant"],
                "quant_method": r["quant_method"],
            }

    save_dict = {
        "model_config": model_config,
        "layers": layers_data,
    }

    model_path = os.path.join(output_path, "glorcq_model.pt")
    torch.save(save_dict, model_path)
    print(f"\n[save] Real-quant model → {model_path}")

    # Also save HF config for model reconstruction
    model.config.save_pretrained(output_path)
    print(f"[save] HF config → {output_path}/config.json")
