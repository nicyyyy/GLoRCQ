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

DEV = torch.device('cuda:0')


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
        # Only for GPTQ (attention) path. TurboQuant (MoE) skips L_lower to
        # avoid storing 64 × in_d² matrices simultaneously (OOM). Instead,
        # hessian_weighted_svd will use diagonal H_eq weighting as a fallback.
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
        # Fallback: diagonal Hessian-weighted SVD using cached H_eq_diag.
        # When L_lower is unavailable (TurboQuant MoE path), scale columns of E
        # by sqrt(H_eq_diag) before SVD — approximates Hessian weighting with
        # O(in_d) memory instead of O(in_d²), then un-scales V afterwards.
        try:
            if self.H_eq_diag is not None:
                w = self.H_eq_diag.to(E.device).float().sqrt().clamp(min=1e-6)
                E_scaled = E * w.unsqueeze(0)                    # (out_d, in_d)
                U, S_sc, Vw = torch.svd_lowrank(E_scaled, q=eff_rank, niter=4)
                V_unscaled = Vw / w.unsqueeze(1)                 # (in_d, eff_rank)
                col_norms = V_unscaled.norm(dim=0).clamp(min=1e-10)
                V = V_unscaled / col_norms.unsqueeze(0)          # unit columns
                S = S_sc * col_norms
            else:
                U, S, V = torch.svd_lowrank(E, q=eff_rank, niter=4)
            return U.half(), S.half(), V.half()
        except (torch._C._LinAlgError, RuntimeError) as e:
            print(f"  [hessian_weighted_svd] diagonal-H SVD failed ({e}), "
                  f"returning zero LoRA.", flush=True)
            U  = torch.zeros(E.shape[0], eff_rank, dtype=torch.float16, device=E.device)
            S  = torch.zeros(eff_rank,              dtype=torch.float16, device=E.device)
            V  = torch.zeros(E.shape[1], eff_rank,  dtype=torch.float16, device=E.device)
            return U, S, V

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
# Batched randomized SVD helper
# ---------------------------------------------------------------------------
def _batched_rsvd(X: torch.Tensor, rank: int, niter: int = 4):
    """
    Batched randomized SVD: X (K, m, n) → U (K, m, rank), S (K, rank), Vp (K, n, rank).

    Replaces K sequential torch.svd_lowrank calls with batched bmm operations,
    giving much better GPU utilization when K is large (e.g., K=120 MoE experts).

    Equivalent quality to svd_lowrank(X[k], q=rank, niter=niter) for each k.
    """
    K, m, n = X.shape
    r = min(rank + 10, m, n)               # rank + oversampling, clamped to matrix dims
    Omega = torch.randn(K, n, r, device=X.device, dtype=X.dtype)
    Y = torch.bmm(X, Omega)                # (K, m, r)
    for _ in range(niter):
        Q, _ = torch.linalg.qr(Y)         # (K, m, r)
        Z = torch.bmm(X.transpose(1, 2), Q)   # (K, n, r)
        Q2, _ = torch.linalg.qr(Z)        # (K, n, r)
        Y = torch.bmm(X, Q2)              # (K, m, r)
    Q, _ = torch.linalg.qr(Y)             # (K, m, r)
    B = torch.bmm(Q.transpose(1, 2), X)   # (K, r, n)
    U_B, S, Vh = torch.linalg.svd(B, full_matrices=False)  # (K,r,r), (K,r), (K,r,n)
    U = torch.bmm(Q, U_B[:, :, :rank])    # (K, m, rank)
    return U, S[:, :rank], Vh[:, :rank, :].transpose(1, 2)  # (K,m,rank),(K,rank),(K,n,rank)


