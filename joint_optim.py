"""
joint_optim.py

Core implementation of Quantization + LoRA optimization for GLoRCQ.

Two modes:
  - GPTQ (use_turboquant=False): Joint alternating optimization
      1. Capture Hessian H once
      2. prepare_hessian(): cache Hinv + L_lower + act_scale
      3. Alternate n_iter times: Q = GPTQ(W - W_lora) → SVD → update W_lora
      4. Final GPTQ(W - W_lora_final) + symmetry fix
  - Hybrid (use_turboquant=True): Per-module dispatch
      Attention: GPTQ alternating n_iter rounds + Hessian-weighted SVD + symmetry fix
      MoE: TurboQuant(W) sequential → plain SVD → one-shot LoRA

Stages 2-5 (Grassmannian clustering + shared U) are in cross_layer_share.py.
"""

import math
import gc
import sys
import os

# Ensure glorcq/ is on sys.path when this module is imported directly
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tqdm
import torch
import torch.nn as nn

import quantizer as quant_module
from cross_layer_share import (
    DEV,
    _wtype_from_name,
    _expert_idx_from_name,
    _move_model_embeds,
)
from utils.moe_utils import is_regular_expert, is_shared_expert, get_moe_config
from turbo_weight_quantizer import TurboWeightQuantizer


# ---------------------------------------------------------------------------
# GPTQJoint: re-entrant GPTQ with optional LoRA subtraction
# ---------------------------------------------------------------------------
class GPTQJoint:
    """
    GPTQ quantizer that supports:
      - Multiple calls to fasterquant (re-entrant via W_orig_saved)
      - Hessian-weighted low-rank approximation of the residual
      - Activation equalization via per-channel scaling (AWQ-style)
      - Cholesky caching (prepare_hessian) to avoid redundant computation
    """

    def __init__(self, layer: nn.Linear, nbits: int = 4,
                 sym: bool = False, mse: bool = True):
        self.layer   = layer
        self.dev     = layer.weight.device
        W            = layer.weight.data
        self.rows    = W.shape[0]    # out_d
        self.columns = W.shape[1]    # in_d
        self.H       = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        # Save original weight on CPU (fp16) so fasterquant can be called repeatedly
        self.W_orig_saved = W.clone().cpu().half()

        # Activation statistics for equalization
        self.act_sum   = None   # (in_d,) float32, accumulated abs activations
        self.act_count = 0

        # Cached Hessian decompositions (filled by prepare_hessian, cleared by free)
        self.Hinv      = None   # upper triangular Cholesky of H_eq^{-1}, for GPTQ
        self.L_lower   = None   # lower triangular Cholesky of H, for hessian_weighted_svd
        self.dead      = None   # bool tensor (in_d,)
        self.act_scale = None   # per-channel scale (in_d,) float32, None = no equalization

        if nbits == 1:
            self.quantizer = quant_module.BiWeightQuantizer()
            self.quantizer.configure(1, perchannel=True, sym=True, mse=False)
        else:
            self.quantizer = quant_module.WeightQuantizer()
            self.quantizer.configure(nbits, perchannel=True, sym=sym, mse=mse)

    def add_batch(self, inp, out):
        """Accumulate Hessian estimate and activation statistics from one batch."""
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        # Save for activation equalization before any transformation
        inp_for_act = inp.reshape(-1, inp.shape[-1])   # (B*T, in_d)

        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()       # (in_d, batch*seq)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()

        # Activation mean tracking for equalization
        x = inp_for_act.float()   # (B*T, in_d)
        # BUG 5 fix: filter inf/nan to prevent act_scale from being corrupted
        x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        if self.act_sum is None:
            self.act_sum   = x.abs().sum(0)
            self.act_count = x.shape[0]
        else:
            self.act_sum   += x.abs().sum(0)
            self.act_count += x.shape[0]

    @torch.no_grad()
    def prepare_hessian(self, percdamp: float = 0.01, act_alpha: float = 0.5,
                        use_turboquant: bool = False,
                        act_scale_override: torch.Tensor = None):
        """
        Compute and cache Cholesky decompositions for GPTQ and SVD.

        Must be called once after all add_batch calls, before any fasterquant
        or hessian_weighted_svd calls.

        Also computes act_scale for activation equalization based on
        accumulated activation statistics.

        Args:
            percdamp:  Hessian damping factor (default 0.01)
            act_alpha: Activation equalization exponent (0=none, 0.5=AWQ default)
            use_turboquant: If True, skip Hinv Cholesky (not needed for TurboQuant)
        """
        H = self.H.clone()
        self.dead = torch.diag(H) == 0
        H[self.dead, self.dead] = 1

        # ── Activation Equalization ───────────────────────────────────────────
        # Skip act_scale for TurboQuant: its per-row L2 norm normalization
        # is incompatible with column scaling (act_scale distorts row geometry)
        if act_scale_override is not None and not use_turboquant:
            self.act_scale = act_scale_override.to(self.dev)
            H_eq = H * self.act_scale.unsqueeze(1) * self.act_scale.unsqueeze(0)
        elif (not use_turboquant
                and self.act_sum is not None and self.act_count > 0 and act_alpha > 0):
            s = (self.act_sum / self.act_count).pow(act_alpha).clamp(min=1e-4)
            self.act_scale = s                           # (in_d,) float32
            H_eq = H * s.unsqueeze(1) * s.unsqueeze(0)  # H_eq = diag(s) H diag(s)
        else:
            self.act_scale = None
            H_eq = H

        # ── Cache Hinv for GPTQ (based on H_eq) ──────────────────────────────
        # Skip when using TurboQuant (no GPTQ column-by-column compensation)
        if not use_turboquant:
            damp   = percdamp * torch.mean(torch.diag(H_eq))
            H_eq_d = H_eq.clone()
            H_eq_d.diagonal().add_(damp)
            try:
                H_chol = torch.linalg.cholesky(H_eq_d)
            except torch._C._LinAlgError:
                # Robust fallback: clamp eigenvalues to ensure positive-definite
                H_eq_d = (H_eq_d + H_eq_d.T) / 2
                eigvals, eigvecs = torch.linalg.eigh(H_eq_d)
                eigvals = eigvals.clamp(min=max(damp.item(), 1e-6))
                H_eq_d = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                H_chol = torch.linalg.cholesky(H_eq_d)
            Hinv_raw = torch.cholesky_inverse(H_chol)
            Hinv_raw = (Hinv_raw + Hinv_raw.T) / 2
            Hinv_raw.diagonal().clamp_(min=1e-10)
            try:
                self.Hinv = torch.linalg.cholesky(Hinv_raw, upper=True)
            except torch._C._LinAlgError:
                eigvals, eigvecs = torch.linalg.eigh(Hinv_raw)
                eigvals = eigvals.clamp(min=1e-10)
                Hinv_raw = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                self.Hinv = torch.linalg.cholesky(Hinv_raw, upper=True)

        # ── Cache L_lower for hessian_weighted_svd (based on H_eq) ────────────
        # Skip when using TurboQuant (no Hessian-weighted SVD needed)
        if not use_turboquant:
            damp_eq_svd = percdamp * torch.mean(torch.diag(H_eq))
            H_eq_svd = H_eq.clone()
            H_eq_svd.diagonal().add_(damp_eq_svd)
            try:
                self.L_lower = torch.linalg.cholesky(H_eq_svd)
            except torch._C._LinAlgError:
                print(f"[WARN] prepare_hessian: L_lower Cholesky failed for "
                      f"{self.layer}, falling back to unweighted SVD.", flush=True)
                self.L_lower = None   # fallback to plain SVD

        # Cache H_eq diagonal for Stage 3 Hessian-weighted SVD
        self.H_eq_diag = torch.diag(H_eq).cpu().float()  # (in_d,) float32

        # Cache W_orig on GPU to avoid repeated CPU→GPU transfers in fasterquant
        self.W_orig_gpu = self.W_orig_saved.float().to(self.dev)

    @torch.no_grad()
    def fasterquant(self, W_lora=None, blocksize: int = 128,
                    percdamp: float = 0.01, groupsize: int = 128,
                    reset_quant: bool = True, real_quant: bool = False):
        """
        Quantize W_r = W_orig - W_lora using GPTQ.

        Uses cached Hinv and dead mask from prepare_hessian().
        Applies activation equalization (act_scale) if available.

        Re-entrant: always starts from self.W_orig_saved, never modifies self.H.

        Args:
            W_lora: optional (out_d, in_d) float32 GPU tensor — the current
                    LoRA approximation to subtract before quantizing.
            blocksize, percdamp, groupsize: standard GPTQ hyperparameters.
                    percdamp is kept for API compatibility; damping is now
                    computed once in prepare_hessian().
            real_quant: if True, also collect per-group scales/zeros and
                        integer quantized values for packed storage.

        Returns:
            (W_orig_cpu_fp16, Q_W_cpu_fp16) when real_quant=False
            (W_orig_cpu_fp16, Q_W_cpu_fp16, gptq_packed) when real_quant=True
              gptq_packed: dict with 'scales', 'zeros', 'qweight_int'
        """
        W      = self.W_orig_gpu if hasattr(self, 'W_orig_gpu') and self.W_orig_gpu is not None \
                 else self.W_orig_saved.float().to(self.dev)   # (out_d, in_d)
        W_orig = self.W_orig_saved                         # CPU fp16 (reference only)

        if W_lora is not None:
            W_r = W - W_lora.to(self.dev).float()
        else:
            W_r = W.clone()

        if torch.all(self.H == 0):
            # No calibration data — return unquantized W_r
            Q = W_r.half()
            self.layer.weight.data = Q
            self.Q_gpu = Q.float()
            if real_quant:
                return W_orig, Q.cpu(), None
            return W_orig, Q.cpu()

        # Activation equalization: scale columns of W_r down before GPTQ
        if self.act_scale is not None:
            W_r = W_r * (1.0 / self.act_scale)    # (out_d, in_d) / (in_d,)

        W_r[:, self.dead] = 0                      # zero out dead columns
        Hinv = self.Hinv                           # use cached Hinv (not cloned)

        # Reset quantizer so find_params is triggered fresh (needed for re-entry)
        if reset_quant:
            self.quantizer.scale = torch.zeros(1, device=self.dev)
            self.quantizer.zero  = torch.zeros(1, device=self.dev)

        # Initial global find_params (fallback for groupsize == -1)
        if not self.quantizer.ready():
            self.quantizer.find_params(W_r)

        # For real_quant: collect per-group scale/zero and integer codes
        if real_quant:
            n_groups = (self.columns + groupsize - 1) // groupsize if groupsize > 0 else 1
            all_scales = torch.zeros(self.rows, n_groups, device=self.dev)
            all_zeros  = torch.zeros(self.rows, n_groups, device=self.dev)
            Q_int      = torch.zeros(self.rows, self.columns, dtype=torch.int32,
                                     device=self.dev)

        Q = torch.zeros_like(W_r)
        for i1 in range(0, self.columns, blocksize):
            i2    = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1    = W_r[:, i1:i2].clone()
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W_r[:, (i1 + i):(i1 + i + groupsize)])
                    if real_quant:
                        gi = (i1 + i) // groupsize
                        all_scales[:, gi] = self.quantizer.scale.squeeze()
                        all_zeros[:, gi]  = self.quantizer.zero.squeeze()

                q          = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i]   = q

                if real_quant:
                    # Store the integer code (before dequant) for packing
                    if self.quantizer.sym:
                        q_int = torch.round(w / self.quantizer.scale.squeeze()).clamp(
                            -(self.quantizer.maxq + 1), self.quantizer.maxq)
                    else:
                        q_int = (torch.round(w / self.quantizer.scale.squeeze())
                                 + self.quantizer.zero.squeeze()).clamp(
                            0, self.quantizer.maxq)
                    Q_int[:, i1 + i] = q_int.int()

                err1       = (w - q) / d
                W1[:, i:]  -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_r[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # Restore original scale after GPTQ
        if self.act_scale is not None:
            Q = Q * self.act_scale                 # (out_d, in_d)
            if real_quant:
                # Scales also need to incorporate act_scale for correct dequant
                for gi in range(all_scales.shape[1]):
                    col0 = gi * groupsize
                    col1 = min(col0 + groupsize, self.columns)
                    # avg act_scale for the group (approximate)
                    all_scales[:, gi] = all_scales[:, gi] * self.act_scale[col0:col1].mean()

        Q = Q.reshape(self.layer.weight.shape).half()
        self.layer.weight.data = Q
        self.Q_gpu = Q.float()  # cache on GPU for callers to avoid CPU↔GPU roundtrip

        if real_quant:
            gptq_packed = {
                "scales": all_scales.cpu().half(),
                "zeros": all_zeros.cpu().half(),
                "qweight_int": Q_int.cpu(),
                "bits": self.quantizer.bits,
                "groupsize": groupsize,
                "sym": getattr(self.quantizer, 'sym', False),
            }
            return W_orig, Q.cpu(), gptq_packed
        return W_orig, Q.cpu()

    @torch.no_grad()
    def turbo_quantize(self, turbo_quantizer, W_lora=None, name: str = "",
                       real_quant: bool = False):
        """
        TurboQuant-based quantization (replaces fasterquant).

        Uses random rotation + Lloyd-Max optimal codebook instead of GPTQ's
        per-group scalar quantization with Hessian-based error compensation.

        No act_scale is applied: TurboQuant's per-row L2 norm normalization
        is incompatible with column scaling (act_scale distorts row geometry).

        Args:
            turbo_quantizer: TurboWeightQuantizer instance
            W_lora: optional (out_d, in_d) float32 GPU tensor
            name: module name for diagnostic printing
            real_quant: if True, also return compressed dict for packed storage

        Returns:
            (W_orig_cpu_fp16, Q_W_cpu_fp16) when real_quant=False
            (W_orig_cpu_fp16, Q_W_cpu_fp16, compressed_dict) when real_quant=True
        """
        W = self.W_orig_gpu if hasattr(self, 'W_orig_gpu') and self.W_orig_gpu is not None \
            else self.W_orig_saved.float().to(self.dev)
        W_orig = self.W_orig_saved  # CPU fp16

        if W_lora is not None:
            W_r = W - W_lora.to(self.dev).float()
        else:
            W_r = W.clone()

        W_r[:, self.dead] = 0                      # zero out dead columns

        if real_quant:
            Q, compressed = turbo_quantizer.quantize_compressed(W_r)
        else:
            Q = turbo_quantizer.quantize_dequantize(W_r)
            compressed = None

        Q = Q.half()
        self.layer.weight.data = Q.clone()
        self.Q_gpu = Q.float()  # cache on GPU for callers to avoid CPU↔GPU roundtrip
        if real_quant:
            return W_orig, Q.cpu(), compressed
        return W_orig, Q.cpu()

    @torch.no_grad()
    def hessian_weighted_svd(self, E: torch.Tensor, rank: int):
        """
        Output-space low-rank approximation of residual E.

        Minimizes ||(E - M) X||_F  subject to rank(M) <= rank,
        equivalently ||(E - M) L||_F  where H = X X^T / N = L L^T.

        Uses cached L_lower from prepare_hessian() if available.
        Falls back to plain (unweighted) SVD if L_lower is None.

        Args:
            E:    (out_d, in_d) float32 on DEV
            rank: target rank

        Returns:
            U: (out_d, rank) fp16
            S: (rank,)       fp16
            V: (in_d, rank)  fp16
        """
        eff_rank = min(rank, E.shape[0], E.shape[1])
        if self.L_lower is not None:
            L  = self.L_lower
            EL = E @ L                                    # (out_d, in_d)
            try:
                U, S, Vp = torch.svd_lowrank(EL, q=eff_rank, niter=4)
                # Vp: (in_d, eff_rank); back-transform: solve L^T V = Vp
                V = torch.linalg.solve_triangular(L.T, Vp, upper=True)  # (in_d, eff_rank)
                # Absorb per-column scale of V into S to prevent fp16 overflow.
                # W_lora = U @ diag(S) @ V^T is mathematically unchanged.
                V_col_norm = V.norm(dim=0).clamp(min=1e-6)   # (eff_rank,)
                V = V / V_col_norm                            # unit columns
                S = S * V_col_norm                            # scale absorbed into S
                return U.half(), S.half(), V.half()
            except (torch._C._LinAlgError, RuntimeError) as e:
                print(f"  [hessian_weighted_svd] SVD on EL failed ({e}), "
                      f"falling back to plain SVD on E.", flush=True)
                # fall through to plain SVD below
        # Fallback: plain SVD on E (no Hessian weighting)
        try:
            U, S, Vp = torch.svd_lowrank(E, q=eff_rank, niter=4)
            return U.half(), S.half(), Vp.half()
        except (torch._C._LinAlgError, RuntimeError) as e:
            print(f"  [hessian_weighted_svd] plain SVD also failed ({e}), "
                  f"returning zero LoRA.", flush=True)
            U  = torch.zeros(E.shape[0], eff_rank, dtype=torch.float16, device=E.device)
            S  = torch.zeros(eff_rank,              dtype=torch.float16, device=E.device)
            Vp = torch.zeros(E.shape[1], eff_rank,  dtype=torch.float16, device=E.device)
            return U, S, Vp

    def free(self):
        del self.H
        self.H        = None
        self.Hinv     = None
        self.L_lower  = None
        self.dead     = None
        self.act_scale = None
        self.H_eq_diag = None
        if hasattr(self, 'W_orig_gpu') and self.W_orig_gpu is not None:
            del self.W_orig_gpu
            self.W_orig_gpu = None
        if hasattr(self, 'Q_gpu') and self.Q_gpu is not None:
            del self.Q_gpu
            self.Q_gpu = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# AWQ-style per-group alpha search
# ---------------------------------------------------------------------------
@torch.no_grad()
def search_act_scale_pergroup(g, alpha_grid=None, groupsize=128):
    """
    AWQ-style per-group alpha search using diagonal Hessian proxy.

    For each group of `groupsize` input channels, independently selects
    the best activation equalization exponent alpha that minimizes
    Hessian-diagonal-weighted quantization error in the original domain:
        sum_{j in group} H_{jj} * ||W[:,j] - s_j * Q(W[:,j]/s_j)||^2

    Returns:
        act_scale: (in_d,) float32 tensor on GPU, or None if no equalization.
        alpha_map: list of per-group best alphas (for logging)
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if g.act_sum is None or g.act_count == 0:
        return None, []

    W = g.W_orig_saved.float().to(g.dev)            # (out_d, in_d)
    mean_act = (g.act_sum / g.act_count).to(g.dev)  # (in_d,)
    H_diag = torch.diag(g.H).clone()                # (in_d,)

    dead = H_diag == 0
    H_diag[dead] = 0
    W_work = W.clone()
    W_work[:, dead] = 0

    proxy_q = quant_module.WeightQuantizer()
    proxy_q.configure(g.quantizer.bits, perchannel=True,
                      sym=getattr(g.quantizer, 'sym', False),
                      mse=True)

    mean_act_safe = mean_act.clamp(min=1e-8)
    n_groups = (g.columns + groupsize - 1) // groupsize
    best_alpha_per_group = [0.0] * n_groups
    best_err_per_group = [float('inf')] * n_groups

    for alpha in alpha_grid:
        if alpha > 0:
            s_full = mean_act_safe.pow(alpha).clamp(min=1e-4)
        else:
            s_full = None

        for gi in range(n_groups):
            col0 = gi * groupsize
            col1 = min(col0 + groupsize, g.columns)

            W_group = W_work[:, col0:col1]           # (out_d, gs)
            h_group = H_diag[col0:col1]              # (gs,)

            if s_full is not None:
                s_group = s_full[col0:col1]
                W_scaled = W_group / s_group
            else:
                W_scaled = W_group
                s_group = None

            proxy_q.find_params(W_scaled)
            Q_scaled = proxy_q.quantize(W_scaled)

            # Error in original domain
            if s_group is not None:
                E_group = W_group - Q_scaled * s_group
            else:
                E_group = W_group - Q_scaled

            col_err = E_group.pow(2).sum(dim=0)      # (gs,)
            err = (col_err * h_group).sum().item()

            if err < best_err_per_group[gi]:
                best_err_per_group[gi] = err
                best_alpha_per_group[gi] = alpha

    # Build composite act_scale
    act_scale = torch.ones(g.columns, device=g.dev, dtype=torch.float32)
    any_nonzero = False
    for gi in range(n_groups):
        alpha = best_alpha_per_group[gi]
        if alpha > 0:
            any_nonzero = True
            col0 = gi * groupsize
            col1 = min(col0 + groupsize, g.columns)
            act_scale[col0:col1] = mean_act_safe[col0:col1].pow(alpha).clamp(min=1e-4)

    if not any_nonzero:
        return None, best_alpha_per_group

    return act_scale, best_alpha_per_group


# ---------------------------------------------------------------------------
# Stage 1 — Alternating Joint Quantization + Hessian-weighted SVD
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_joint(model, layers, dataloader, args, use_turboquant: bool = False,
                   model_type: str = ""):
    """
    Layer-by-layer alternating quantization + Hessian-weighted LoRA optimization.

    When use_turboquant=True, replaces GPTQ with TurboQuant row-wise vector
    quantization (random rotation + Lloyd-Max optimal codebook).

    Returns all_records: list of dicts in the same format as
    cross_layer_share.quantize_and_collect_residuals(), i.e.:
      {layer, expert, type, module_name, shape,
       weight_orig (CPU fp16), weight_quant (CPU fp16)}
    where E_final = weight_orig - weight_quant is the residual fed to
    Stage 2 (Grassmannian clustering) and Stage 3+4 (cross-layer shared U).
    """
    nbits = getattr(args, 'qbit',   4)
    sym   = getattr(args, 'sym',    False)
    mse   = getattr(args, 'w_clip', True)
    real_quant = getattr(args, 'real_quant', False)
    moe_cfg = get_moe_config(model_type)

    # Create TurboQuant quantizer if requested
    turbo_quantizer = None
    if use_turboquant:
        turbo_quantizer = TurboWeightQuantizer(nbits=nbits, device=DEV)
        print(f"  [TurboQuant] Enabled: {nbits}-bit row-wise vector quantization "
              f"(random rotation + Lloyd-Max codebook)", flush=True)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # Capture inputs to the first transformer layer
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Layer-by-layer alternating optimization
    # -----------------------------------------------------------------------
    outs        = torch.zeros_like(inps)
    all_records = []

    _stage1_desc = ("Stage 1: Hybrid GPTQ(attn)+TurboQuant(MoE)"
                     if use_turboquant
                     else "Stage 1: Joint GPTQ+LoRA")
    for layer_idx in tqdm.tqdm(range(len(layers)), desc=_stage1_desc):
        layer = layers[layer_idx].to(DEV)
        full  = quant_module.find_qlayers(layer, [nn.Linear])

        # Build GPTQJoint instances for trackable weight types only
        gptq = {}
        for name in full:
            if _wtype_from_name(name) is None:
                continue
            if "lm_head" in name:
                continue
            gptq[name] = GPTQJoint(full[name], nbits=nbits, sym=sym, mse=mse)

        # ------------------------------------------------------------------
        # Single forward pass to accumulate Hessian H for all modules
        # ------------------------------------------------------------------
        def _add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(full[name].register_forward_hook(_add_batch(name)))
        _batch_size = getattr(args, 'calib_batch_size', 8)
        for j in range(0, args.nsamples, _batch_size):
            _end = min(j + _batch_size, args.nsamples)
            _actual = _end - j
            _batch = torch.cat([inps[k].unsqueeze(0) for k in range(j, _end)])
            _out_batch = layer(_batch, **layer_kwargs)[0]
            for k in range(_actual):
                outs[j + k] = _out_batch[k]
        for h in handles:
            h.remove()

        # Compute and cache Cholesky decompositions + activation scales once
        default_alpha = getattr(args, 'act_alpha', 0.6)
        do_search = getattr(args, 'search_act_alpha', False)

        for name, g in gptq.items():
            eidx = _expert_idx_from_name(name)
            if use_turboquant and eidx >= 0:
                # Regular MoE expert: TurboQuant, no act_scale
                # (shared experts eidx==-2 fall through to attention path below)
                g.prepare_hessian(percdamp=args.percdamp, act_alpha=default_alpha,
                                  use_turboquant=True)
                continue

            # Attention module (or all modules when use_turboquant=False)
            if do_search:
                act_scale, alpha_map = search_act_scale_pergroup(
                    g, groupsize=args.groupsize)
                # Log per-group alpha distribution
                from collections import Counter
                dist = Counter(alpha_map)
                dist_str = " ".join(f"{a}:{c}" for a, c in sorted(dist.items()))
                print(f"    {name}: per-group AWQ [{dist_str}]", flush=True)
                g.prepare_hessian(percdamp=args.percdamp, act_alpha=0.0,
                                  use_turboquant=False,
                                  act_scale_override=act_scale)
            else:
                g.prepare_hessian(percdamp=args.percdamp, act_alpha=default_alpha,
                                  use_turboquant=False)

        if use_turboquant:
            # ==============================================================
            # HYBRID: GPTQ(attn) + TurboQuant(MoE)
            # ==============================================================
            attn_names = [n for n in gptq if _expert_idx_from_name(n) == -1]
            moe_names  = [n for n in gptq if _expert_idx_from_name(n) >= 0]

            # Phase A: GPTQ alternating for attention
            lora_state_attn = {name: (None, None, None) for name in attn_names}
            early_stop_tol = getattr(args, 'early_stop_tol', 0)
            prev_frob  = {name: float('inf') for name in attn_names}
            converged  = {name: False for name in attn_names}

            for it in range(args.n_iter):
                for name in attn_names:
                    if converged[name]:
                        continue
                    g = gptq[name]
                    U, S, V = lora_state_attn[name]
                    W_lora = U.float() @ (S.unsqueeze(1) * V.T.float()) if U is not None else None

                    W_orig, Q_W = g.fasterquant(
                        W_lora=W_lora, percdamp=args.percdamp,
                        groupsize=args.groupsize, reset_quant=(it == 0),
                    )
                    E = g.W_orig_gpu - g.Q_gpu
                    U_new, S_new, V_new = g.hessian_weighted_svd(E, args.rank)
                    lora_state_attn[name] = (U_new, S_new, V_new)
                    W_lora_new = U_new.float() @ (S_new.unsqueeze(1) * V_new.T.float())
                    full[name].weight.data = (g.Q_gpu + W_lora_new).half()

                    if early_stop_tol > 0:
                        E_new = g.W_orig_gpu - g.Q_gpu - W_lora_new
                        curr_frob = E_new.norm().item()
                        rel_improve = (prev_frob[name] - curr_frob) / (prev_frob[name] + 1e-12)
                        if rel_improve < early_stop_tol:
                            converged[name] = True
                        prev_frob[name] = curr_frob

            # Final pass + symmetry fix for attention
            for name in attn_names:
                g = gptq[name]
                expert_idx = _expert_idx_from_name(name)
                wtype      = _wtype_from_name(name)
                U, S, V    = lora_state_attn[name]
                W_orig     = g.W_orig_saved
                W_lora_final = U.float() @ (S.unsqueeze(1) * V.T.float()) if U is not None else None

                _, Q_W_final = g.fasterquant(
                    W_lora=W_lora_final, percdamp=args.percdamp, groupsize=args.groupsize,
                )
                E_sym = g.W_orig_gpu - g.Q_gpu
                U_sym, S_sym, V_sym = g.hessian_weighted_svd(E_sym, args.rank)
                W_lora_sym = U_sym.float() @ (S_sym.unsqueeze(1) * V_sym.T.float())
                full[name].weight.data = (g.Q_gpu + W_lora_sym).half()

                if expert_idx >= 0 or expert_idx == -1:
                    record = {
                        "layer":        layer_idx,
                        "expert":       expert_idx,
                        "type":         wtype,
                        "module_name":  name,
                        "shape":        tuple(W_orig.shape),
                        "weight_orig":  W_orig,
                        "weight_quant": Q_W_final,
                        "hessian_diag": g.H_eq_diag,
                        "quant_method": "gptq",
                    }
                    # For real_quant: re-run final pass to capture packed data
                    if real_quant:
                        _, _, gptq_packed = g.fasterquant(
                            W_lora=W_lora_sym.half().cpu().float() if W_lora_sym is not None else None,
                            percdamp=args.percdamp, groupsize=args.groupsize,
                            real_quant=True,
                        )
                        record["gptq_packed"] = gptq_packed
                        # Restore the correct weight (fasterquant overwrites it)
                        full[name].weight.data = (g.Q_gpu + W_lora_sym).half()
                    all_records.append(record)
                g.free()

            # Phase B: Batched TurboQuant for MoE
            # Group moe_names by in_d (same-dim experts can be batched)
            from collections import defaultdict as _defaultdict
            _dim_groups = _defaultdict(list)
            for name in moe_names:
                _dim_groups[gptq[name].columns].append(name)

            turbo_bs = getattr(args, 'turbo_batch_size', 0)

            for _in_d, _group_names in _dim_groups.items():
                bs = turbo_bs if turbo_bs > 0 else len(_group_names)
                for _c0 in range(0, len(_group_names), bs):
                    _chunk = _group_names[_c0:_c0 + bs]

                    if real_quant:
                        # real_quant needs per-expert compressed data → sequential
                        for name in _chunk:
                            g = gptq[name]
                            W_orig, Q_W, compressed = g.turbo_quantize(
                                turbo_quantizer, W_lora=None,
                                name=f"L{layer_idx}.{name}", real_quant=True)
                            E = g.W_orig_gpu - g.Q_gpu
                            U, S, V = g.hessian_weighted_svd(E, args.rank)
                            W_lora = U.float() @ (S.unsqueeze(1) * V.T.float())
                            full[name].weight.data = (g.Q_gpu + W_lora).half()
                            expert_idx = _expert_idx_from_name(name)
                            wtype = _wtype_from_name(name)
                            if expert_idx >= 0 or expert_idx == -1:
                                record = {
                                    "layer": layer_idx, "expert": expert_idx,
                                    "type": wtype, "module_name": name,
                                    "shape": tuple(W_orig.shape),
                                    "weight_orig": W_orig, "weight_quant": Q_W,
                                    "hessian_diag": g.H_eq_diag,
                                    "quant_method": "turboquant",
                                }
                                if compressed is not None:
                                    record["turbo_compressed"] = compressed
                                all_records.append(record)
                            g.free()
                        continue

                    # Batch quantize: stack weights → one TurboQuant call → split
                    W_list = [gptq[n].W_orig_gpu for n in _chunk]
                    out_dims = [w.shape[0] for w in W_list]
                    W_stacked = torch.cat(W_list, dim=0)  # (Σout_d, in_d)

                    Q_stacked = turbo_quantizer.quantize_dequantize(W_stacked)

                    # Split back and do per-expert SVD + record
                    _off = 0
                    for name, _od in zip(_chunk, out_dims):
                        g = gptq[name]
                        Q_exp = Q_stacked[_off:_off + _od]
                        _off += _od

                        # Cache quantized result (same as turbo_quantize would)
                        g.Q_gpu = Q_exp.float()
                        g.layer.weight.data = Q_exp.half().clone()
                        Q_W = Q_exp.cpu().half()

                        E = g.W_orig_gpu - g.Q_gpu
                        U, S, V = g.hessian_weighted_svd(E, args.rank)
                        W_lora = U.float() @ (S.unsqueeze(1) * V.T.float())
                        full[name].weight.data = (g.Q_gpu + W_lora).half()

                        expert_idx = _expert_idx_from_name(name)
                        wtype = _wtype_from_name(name)
                        if expert_idx >= 0 or expert_idx == -1:
                            record = {
                                "layer": layer_idx, "expert": expert_idx,
                                "type": wtype, "module_name": name,
                                "shape": tuple(g.W_orig_saved.shape),
                                "weight_orig": g.W_orig_saved,
                                "weight_quant": Q_W,
                                "hessian_diag": g.H_eq_diag,
                                "quant_method": "turboquant",
                            }
                            all_records.append(record)
                        g.free()
        else:
            # ==============================================================
            # GPTQ: Alternating optimization (n_iter rounds) + symmetry fix
            # ==============================================================
            lora_state = {name: (None, None, None) for name in gptq}

            early_stop_tol = getattr(args, 'early_stop_tol', 0)
            prev_frob  = {name: float('inf') for name in gptq}
            converged  = {name: False for name in gptq}

            for it in range(args.n_iter):
                for name, g in gptq.items():
                    if converged[name]:
                        continue

                    U, S, V = lora_state[name]

                    # Current LoRA: W_lora = U @ diag(S) @ V^T, shape (out_d, in_d)
                    if U is not None:
                        W_lora = U.float() @ (S.unsqueeze(1) * V.T.float())
                    else:
                        W_lora = None

                    W_orig, Q_W = g.fasterquant(
                        W_lora=W_lora,
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        reset_quant=(it == 0),
                    )

                    # Residual: E = W_orig - Q_W  (Hessian-weighted SVD target)
                    E = g.W_orig_gpu - g.Q_gpu
                    U_new, S_new, V_new = g.hessian_weighted_svd(E, args.rank)
                    lora_state[name] = (U_new, S_new, V_new)

                    # Update layer weight to W_approx = Q + LoRA
                    W_lora_new = U_new.float() @ (S_new.unsqueeze(1) * V_new.T.float())
                    full[name].weight.data = (g.Q_gpu + W_lora_new).half()

                    # Early stopping: check relative Frobenius improvement
                    if early_stop_tol > 0:
                        E_new = g.W_orig_gpu - g.Q_gpu - W_lora_new
                        curr_frob   = E_new.norm().item()
                        rel_improve = ((prev_frob[name] - curr_frob)
                                       / (prev_frob[name] + 1e-12))
                        if rel_improve < early_stop_tol:
                            converged[name] = True
                            print(f"  [early stop] {name} at iter {it}, "
                                  f"frob={curr_frob:.4f}", flush=True)
                        prev_frob[name] = curr_frob

            # --------------------------------------------------------------
            # Final quantization pass + symmetry fix (GPTQ only)
            # --------------------------------------------------------------
            for name, g in gptq.items():
                expert_idx = _expert_idx_from_name(name)
                wtype      = _wtype_from_name(name)

                U, S, V = lora_state[name]
                W_orig  = g.W_orig_saved    # CPU fp16

                if U is not None:
                    W_lora_final = U.float() @ (S.unsqueeze(1) * V.T.float())
                else:
                    W_lora_final = None

                _, Q_W_final = g.fasterquant(
                    W_lora=W_lora_final,
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                )

                # Symmetry fix: recompute LoRA aligned with Q_final
                E_sym = g.W_orig_gpu - g.Q_gpu
                U_sym, S_sym, V_sym = g.hessian_weighted_svd(E_sym, args.rank)
                W_lora_sym = U_sym.float() @ (S_sym.unsqueeze(1) * V_sym.T.float())
                full[name].weight.data = (g.Q_gpu + W_lora_sym).half()

                if expert_idx >= 0 or expert_idx == -1:
                    record = {
                        "layer":        layer_idx,
                        "expert":       expert_idx,
                        "type":         wtype,
                        "module_name":  name,
                        "shape":        tuple(W_orig.shape),
                        "weight_orig":  W_orig,
                        "weight_quant": Q_W_final,
                        "hessian_diag": g.H_eq_diag,
                        "quant_method": "gptq",
                    }
                    if real_quant:
                        _, _, gptq_packed = g.fasterquant(
                            W_lora=W_lora_sym.half().cpu().float() if W_lora_sym is not None else None,
                            percdamp=args.percdamp, groupsize=args.groupsize,
                            real_quant=True,
                        )
                        record["gptq_packed"] = gptq_packed
                        full[name].weight.data = (g.Q_gpu + W_lora_sym).half()
                    all_records.append(record)
                g.free()

        del gptq

        # Propagate final W_approx outputs to provide inputs for the next layer
        _batch_size = getattr(args, 'calib_batch_size', 8)
        for j in range(0, args.nsamples, _batch_size):
            _end = min(j + _batch_size, args.nsamples)
            _actual = _end - j
            _batch = torch.cat([inps[k].unsqueeze(0) for k in range(j, _end)])
            _out_batch = layer(_batch, **layer_kwargs)[0]
            for k in range(_actual):
                outs[j + k] = _out_batch[k]

        layers[layer_idx] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if use_turboquant:
        print(f"\n[Stage 1] Collected {len(all_records)} records "
              f"(hybrid: GPTQ(attn, n_iter={args.n_iter}) + TurboQuant(MoE, sequential)).")
    else:
        print(f"\n[Stage 1] Collected {len(all_records)} records "
              f"(joint GPTQ+LoRA, n_iter={args.n_iter}).")
    return all_records
