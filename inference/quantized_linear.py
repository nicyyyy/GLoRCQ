"""
GLoRCQ quantized linear layer for inference.

Replaces nn.Linear with a quantized version that supports:
  - GPTQ backend: per-group scalar dequant + matmul
  - TurboQuant backend: rotation + codebook dequant + matmul
  - LoRA compensation: x @ U * S @ V^T added to quantized output
"""

import os

import torch


# Compiled LoRA: fuse (a * S) @ V.T into one Triton kernel via Inductor
@torch.compile(mode="default", fullgraph=True)
def _compiled_lora_sv(a, S, V_T):
    """Compute (a * S) @ V_T with operator fusion."""
    return (a * S) @ V_T
import torch.nn as nn

# Try to import fused CUDA kernel for TurboQuant
try:
    from inference.kernels import turbo_dequant_matmul_fused, is_cuda_available
    _HAS_FUSED_KERNEL = is_cuda_available()
except ImportError:
    _HAS_FUSED_KERNEL = False

# Try to import fused CUDA kernel for GPTQ
try:
    from inference.kernels import gptq_dequant_matmul_fused, is_gptq_cuda_available
    _HAS_GPTQ_FUSED_KERNEL = is_gptq_cuda_available()
except ImportError:
    _HAS_GPTQ_FUSED_KERNEL = False

# Try to import fused CUDA kernel for VQ4
try:
    from inference.kernels import vq4_dequant_matmul, is_vq4_cuda_available
    _HAS_VQ4_FUSED_KERNEL = is_vq4_cuda_available()
except ImportError:
    _HAS_VQ4_FUSED_KERNEL = False


# ---------------------------------------------------------------------------
# Pure-PyTorch dequant+matmul implementations (initial version)
# ---------------------------------------------------------------------------
def gptq_dequant_matmul(x, qweight_int, scales, zeros, bits, groupsize, sym=False):
    """
    Dequantize GPTQ weights and compute matmul.

    Args:
        x: (batch, in_d) fp16 input
        qweight_int: (out_d, in_d) int32 quantized weights
        scales: (out_d, n_groups) fp16 per-group scale
        zeros: (out_d, n_groups) fp16 per-group zero point
        bits: quantization bits
        groupsize: columns per group
        sym: symmetric quantization flag

    Returns:
        y: (batch, out_d) fp16
    """
    out_d, in_d = qweight_int.shape
    device = x.device

    # Dequantize: W = scale * (q - zero) for each group
    W = torch.zeros(out_d, in_d, dtype=x.dtype, device=device)
    n_groups = scales.shape[1]
    for gi in range(n_groups):
        col0 = gi * groupsize
        col1 = min(col0 + groupsize, in_d)
        q_slice = qweight_int[:, col0:col1].to(x.dtype)
        if sym:
            W[:, col0:col1] = q_slice * scales[:, gi:gi+1]
        else:
            W[:, col0:col1] = (q_slice - zeros[:, gi:gi+1]) * scales[:, gi:gi+1]

    return x @ W.T


def turbo_dequant_matmul(x, packed_indices, norms, Pi, centroids, bits, dim):
    """
    Runtime dequant + matmul for TurboQuant weights.

    Instead of materializing the full (out_d, in_d) weight matrix, we:
      1. Rotate the input: x_rot = x @ Pi^T   (one matmul, reused for all rows)
      2. Process output rows in blocks: unpack indices → lookup centroids →
         multiply by norms → matmul with rotated input
    This keeps weights in compressed form on GPU, saving ~7.5x memory.

    Args:
        x: (batch, in_d) fp16 input
        packed_indices: (out_d, packed_len) uint8 bit-packed codebook indices
        norms: (out_d,) fp32 per-row L2 norms
        Pi: (in_d, in_d) fp16 rotation matrix
        centroids: (2^bits,) fp32 scalar codebook
        bits: quantization bits (typically 2)
        dim: input dimension (in_d, for unpacking)

    Returns:
        y: (batch, out_d) fp16
    """
    out_d = packed_indices.shape[0]
    device = x.device

    # 1. Rotate input once: x_rot = x @ Pi^T
    x_rot = x.float() @ Pi.float().T  # (batch, in_d)

    # 2. Block-wise dequant + matmul to avoid full (out_d, in_d) materialization
    y = torch.zeros(x.shape[0], out_d, device=device, dtype=torch.float32)
    BLOCK = 1024  # rows per block — balances memory vs overhead
    for o0 in range(0, out_d, BLOCK):
        o1 = min(o0 + BLOCK, out_d)
        # Unpack indices for this block: (block_size, packed_len) → (block_size, in_d)
        idx = _unpack_indices(packed_indices[o0:o1], bits, dim)
        # Lookup centroids and scale by per-row norms
        W_block = centroids[idx] * norms[o0:o1, None]  # (block_size, in_d) fp32
        # Matmul: (batch, in_d) @ (in_d, block_size) → (batch, block_size)
        y[:, o0:o1] = x_rot @ W_block.T

    return y.half()


def _derive_vq4_codes(Q_rotated, centroids, vdim, in_d):
    """Nearest-centroid assignment from Q_rotated + centroids → uint8 codes.

    Runs on the CPU side (Q_rotated is fp16 on CPU when unpacked from
    cross_layer_info.pt) so that only the resulting small uint8 codes
    (out_d, in_d/vdim) get uploaded to GPU, not the large Q_rotated.

    Args:
        Q_rotated: (out_d, in_d) fp16 tensor (CPU)
        centroids: (n_cb, K=256, vdim) fp16 tensor (CPU)
        vdim: 4
        in_d: input feature dim

    Returns:
        codes: (out_d, in_d/vdim) uint8 tensor (CPU)
    """
    out_d, _in_d = Q_rotated.shape
    assert _in_d == in_d, f"in_d mismatch: {_in_d} vs {in_d}"
    assert _in_d % vdim == 0, f"in_d {_in_d} not divisible by vdim {vdim}"
    n_vecs = _in_d // vdim
    n_cb, K, _v = centroids.shape
    assert _v == vdim, f"centroid vdim {_v} != {vdim}"
    codes_per_cb = n_vecs // n_cb

    # Run the nearest-centroid assignment on GPU when available: on CPU this is
    # the dominant load-time cost (~15-20 min for Mixtral's 768 large experts).
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Qf = Q_rotated.float().to(dev)
    cf = centroids.float().to(dev)  # (n_cb, K, vdim)
    codes = torch.empty(out_d, n_vecs, dtype=torch.uint8)

    # For each codebook block, assign vectors to nearest centroid.
    for cb_id in range(n_cb):
        v_lo = cb_id * codes_per_cb
        v_hi = v_lo + codes_per_cb
        # Q slice for this codebook: (out_d, codes_per_cb, vdim)
        Q_block = Qf[:, v_lo * vdim:v_hi * vdim].reshape(out_d, codes_per_cb, vdim)
        cb = cf[cb_id]  # (K, vdim)
        # ||Q - c||^2 = ||Q||^2 + ||c||^2 - 2 Q·c
        # (out_d, codes_per_cb, K)
        d = (Q_block.unsqueeze(2) - cb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
        codes[:, v_lo:v_hi] = d.argmin(-1).to(torch.uint8).cpu()
        del Q_block, d

    del Qf, cf
    return codes


def _unpack_indices(packed, bits, d):
    """
    Unpack bit-packed uint8 indices to long tensor.

    Inlined from turboquant.quantizer._unpack_indices to avoid import dependency
    at runtime (the function is simple bit manipulation).

    Args:
        packed: (..., packed_len) uint8
        bits: bits per index
        d: original dimension (to trim padding)

    Returns:
        (..., d) long tensor of codebook indices
    """
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = (packed.unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


# ---------------------------------------------------------------------------
# Fp16LinearShim: fallback for experts whose quantization was skipped
# ---------------------------------------------------------------------------
class Fp16LinearShim(nn.Module):
    """Passthrough for MoE-expert nn.Linear modules whose quant was skipped
    at quant-time (max_err > threshold — typically Qwen3 down_proj with rank=16).

    moe_block fast paths call ``expert.gate_proj(x, precomputed_xU=..., precomputed_x_rot=...)``
    which plain nn.Linear rejects. This shim keeps the original fp16 weight and
    accepts (ignores) the kwargs so mixed VQ4 + fp16 expert lists work uniformly.
    Reports ``quant_type='fp16_passthrough'`` so callers can detect and dispatch.
    """
    quant_type = "fp16_passthrough"

    def __init__(self, linear):
        super().__init__()
        w = linear.weight.data
        self.weight = nn.Parameter(w, requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x, **kwargs):
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GLoRCQLinear: quantized inference layer
# ---------------------------------------------------------------------------
class GLoRCQLinear(nn.Module):
    """
    Quantized linear layer for GLoRCQ inference.

    Replaces nn.Linear with a quantized version that supports GPTQ and
    TurboQuant backends, plus optional LoRA compensation.

    Forward: y = dequant(W_q) @ x + (x @ U) * S @ V^T + bias
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight storage (set by load_gptq or load_turbo)
        self.quant_type = None       # "gptq" or "turbo"

        # GPTQ parameters
        self.qweight_int = None      # (out_d, in_d) int32
        self.scales = None           # (out_d, n_groups) fp16
        self.zeros = None            # (out_d, n_groups) fp16
        self.gptq_bits = None
        self.gptq_groupsize = None
        self.gptq_sym = False

        # TurboQuant parameters
        self.packed_indices = None   # (out_d, packed_len) uint8 packed codebook indices
        self.norms = None            # (out_d,) per-row L2 norms
        self.turbo_bits = None
        self.turbo_dim = None
        self.turbo_seed = 42
        self._rotation_cache = None  # RotationCache ref for runtime dequant
        self._turbo_dequant_W = None  # legacy: cached dequantized weight (pre-decompressed)

        # LoRA compensation
        self.U = None                # (in_d, rank) fp16 — shared across cluster
        self.SV = None               # (out_d, rank) fp16 — S pre-fused into V: V * S[None,:]
        self.Sa = None               # (in_d,) fp16 — per-input-channel activation scale (optional)
        self.cluster_id = None       # set by model_builder for cluster-parallel LoRA

        # VQ4 (vdim=4 group-shared codebook) backend
        self.vq_codes = None         # (out_d, in_d/vdim) uint8
        self.vq_centroids = None     # (n_blocks, K, vdim) fp16
        self._vq4_py_offs = None     # cached cb-offset index for python dequant
        self.vq_perm = None          # (in_d,) long
        self.vq_diag_signs = None    # (in_d,) fp16
        self.vq_vdim = None

        # Bias
        self.bias_param = None       # (out_d,) fp16

    def load_gptq(self, packed_data, device="cuda"):
        """Load GPTQ quantized weights from packed data dict."""
        self.quant_type = "gptq"
        # Store qweight as int8 (not int32). 4-bit GPTQ codes live in a small
        # signed-safe range (e.g. [0,15] unsigned nibbles), which int8 holds
        # losslessly. BOTH consumers already reduce to int8/float on entry —
        # the fused CUDA kernel takes qweight_i8 and gptq_dequant_matmul_fused
        # (kernels/__init__.py) casts int32->int8 EVERY decode token, while the
        # python fallback _dequant_gptq()/_gptq_pytorch_fallback() cast to
        # half/float. So storing int8 is bit-identical to the previous behaviour
        # while (a) cutting the attn qweight footprint 4x and (b) eliminating
        # that per-token full-matrix cast. Guard: only narrow when values fit
        # signed int8, so any higher-bit GPTQ is left untouched.
        _qw = packed_data["qweight_int"]
        if _qw.dtype != torch.int8:
            _lo, _hi = int(_qw.min()), int(_qw.max())
            if -128 <= _lo and _hi <= 127:
                _qw = _qw.to(torch.int8)
        self.qweight_int = _qw.to(device)
        self.scales = packed_data["scales"].to(device)
        self.zeros = packed_data["zeros"].to(device)
        self.gptq_bits = packed_data["bits"]
        self.gptq_groupsize = packed_data["groupsize"]
        self.gptq_sym = packed_data.get("sym", False)

    def load_turbo(self, compressed_data, rotation_cache, device="cuda"):
        """
        Load TurboQuant compressed weights for runtime dequant.

        Weights stay in packed form on GPU. Each forward call performs:
          x_rot = x @ Pi^T, then block-wise unpack → centroid lookup → matmul.

        This saves ~7.5x GPU memory compared to pre-decompressing to fp16.

        Args:
            compressed_data: dict with packed_indices, norms, bits, dim, seed
            rotation_cache: RotationCache instance managing Pi and centroids
            device: target device
        """
        self.quant_type = "turbo"
        self.turbo_bits = compressed_data["bits"]
        self.turbo_dim = compressed_data["dim"]
        self.turbo_seed = compressed_data.get("seed", 42)
        self.packed_indices = compressed_data["packed_indices"].to(device)
        self.norms = compressed_data["norms"].float().to(device)
        self._rotation_cache = rotation_cache
        # Ensure Pi and centroids for this (dim, bits, seed) are loaded
        rotation_cache.register(self.turbo_dim, self.turbo_bits, self.turbo_seed)

    def load_turbo_predequant(self, compressed_data, turbo_quantizer, device="cuda"):
        """
        Legacy: pre-dequantize TurboQuant weights and cache as fp16.

        Uses more GPU memory but avoids per-forward dequant overhead.
        Kept for comparison / fallback.
        """
        self.quant_type = "turbo"
        self.turbo_bits = compressed_data["bits"]
        self.turbo_dim = compressed_data["dim"]

        from collections import namedtuple
        MSEQuantized = namedtuple("MSEQuantized", ["indices", "norms", "bits"])
        q = MSEQuantized(
            indices=compressed_data["packed_indices"].to(device),
            norms=compressed_data["norms"].to(device),
            bits=compressed_data["bits"],
        )
        self._turbo_dequant_W = turbo_quantizer.dequantize(q).half().to(device)

    def load_lora(self, U, S, V, Sa=None, device="cuda"):
        """Load LoRA compensation matrices, pre-fusing S into V.

        Precomputes SV = V * S[None, :] so that inference uses
        ``a @ SV.T`` instead of ``(a * S) @ V.T``, eliminating the
        elementwise multiply kernel at each forward pass.

        If ``Sa`` (per-input-channel activation scale) is provided, it is stored
        and applied to ``x`` before the ``x @ U`` matmul in forward.
        """
        if U is not None:
            self.U = U.half().to(device)
            # Pre-fuse: SV[i, j] = V[i, j] * S[j]  (out_d, rank)
            self.SV = (V * S.unsqueeze(0)).half().to(device)
            if Sa is not None:
                self.Sa = Sa.half().to(device)

    def load_sv(self, U, SV, Sa=None, device="cuda"):
        """Load pre-fused SV matrix directly (new format).

        Used when SV = V_normed * S is pre-computed at quantization time;
        no fusion step needed here. ``Sa`` is the per-input-channel activation
        scale used to construct the original LoRA; applied to x in forward.
        """
        if U is not None:
            self.U  = U.half().to(device)
            self.SV = SV.half().to(device)  # (out_d, rank) fp16, already fused
            if Sa is not None:
                self.Sa = Sa.half().to(device)  # (in_d,) fp16

    # Class-level cache: (in_d, rotate_size, partial_size, device) → diagI_rot fp16
    _DIAGI_CACHE = {}

    @classmethod
    def _get_diagI(cls, in_d, rotate_size, partial_size, device):
        key = (in_d, rotate_size, partial_size, str(device))
        if key in cls._DIAGI_CACHE:
            return cls._DIAGI_CACHE[key]
        from utils.hadamard_utils import (
            block_diagonal_walsh_matrix,
            create_diagI_matrix_upper,
        )
        rs = int(rotate_size)
        while rs > 1 and in_d % rs != 0:
            rs //= 2
        rot = block_diagonal_walsh_matrix(in_d, rs, device).to(torch.float16)
        diagI = create_diagI_matrix_upper(rot, rot.shape[0] - partial_size).to(device).to(torch.float16)
        cls._DIAGI_CACHE[key] = diagI
        return diagI

    def load_vq4(self, *, Q_rotated=None, centroids=None, perm, diag_signs, vdim,
                 in_d, out_d, rotate_size=256, partial_size=256,
                 codes=None, codes_packed=None, codes_n_vecs=None,
                 device="cuda"):
        """Load vdim=4 group-shared VQ residual weights.

        Two storage modes:
          (a) Q_rotated: (out_d, in_d) fp16 — bit-exact, large
          (b) codes + centroids: (out_d, in_d/vdim) uint8 + (n_cb, K, vdim) fp16 — packed

        The Python fallback rebuilds Q from codes+centroids when available; else uses
        Q_rotated directly. The CUDA kernel (when wired) operates on codes+centroids.

        PD_rot is decomposed as Ppermute (gather) + diagI_rot (block-Walsh matmul).
        We store only the (in_d,) permutation sigma per expert; diagI_rot is shared
        across all experts with the same (in_d, rotate_size, partial_size).
        """
        self.quant_type = "vq4"
        # Unpack 4-bit codes if the checkpoint shipped packed form (2 codes/byte).
        # codes_packed shape: (out_d, ceil(n_vecs/2)) uint8. Reconstruct by
        # splitting low/high nibble; drop trailing pad if n_vecs was odd.
        if codes is None and codes_packed is not None:
            cp = codes_packed
            lo = (cp & 0xF).to(torch.uint8)
            hi = ((cp >> 4) & 0xF).to(torch.uint8)
            # Interleave lo (even positions) and hi (odd positions).
            out_d_c, n_half = cp.shape
            codes = torch.empty(out_d_c, n_half * 2, dtype=torch.uint8, device=cp.device)
            codes[:, 0::2] = lo
            codes[:, 1::2] = hi
            if codes_n_vecs is not None and codes_n_vecs < codes.shape[1]:
                codes = codes[:, :codes_n_vecs].contiguous()

        # DERIVE codes from Q_rotated + centroids IF codes not shipped in the
        # checkpoint. Older cross_layer_info.pt files only store Q_rotated (fp16
        # (out_d, in_d), ~117 MB per Mixtral expert × 768 = ~90 GB HBM if uploaded).
        # Deriving codes on CPU keeps GPU memory usage low: only the 8-bit codes
        # (~30 MB per Mixtral expert) get uploaded.
        if codes is None and Q_rotated is not None and centroids is not None:
            codes = _derive_vq4_codes(Q_rotated, centroids, vdim, in_d)

        # If codes+centroids are provided (or derived), we DON'T need Q_rotated
        # on GPU: the CUDA kernel operates on codes+centroids directly; the
        # Python fallback also rebuilds Q from them.
        if codes is not None and centroids is not None:
            self.vq_Q_rotated = None
        else:
            self.vq_Q_rotated = (Q_rotated.half().to(device)
                                 if Q_rotated is not None else None)
        self.vq_centroids = centroids.half().to(device) if centroids is not None else None
        if codes is not None:
            self.vq_codes = codes.to(device)
        self.vq_vdim = vdim
        self.in_features = in_d
        self.out_features = out_d

        # Build the sigma index used by Ppermute (per-expert).
        # Ppermute[col_idx, j] = 1 where sigma[j] = col_idx (see
        # utils.hadamard_utils.construct_partial_permutation_matrix_upper).
        # x @ Ppermute is equivalent to x[..., sigma] — a fast gather.
        import numpy as np
        S = np.array(perm.cpu())
        all_indices = set(range(in_d))
        S_set = set(int(x) for x in S)
        front = list(int(x) for x in S)
        back = sorted(all_indices - S_set)
        sigma = torch.tensor(front + back, dtype=torch.long, device=device)
        self.vq_perm_sigma = sigma                          # (in_d,) long

        # Shared diagI_rot for this (in_d, rotate_size, partial_size)
        self.vq_diagI = GLoRCQLinear._get_diagI(
            in_d, rotate_size, partial_size, device)         # (in_d, in_d) fp16 — SHARED
        # Kept for backwards-compat / debug: leave vq_PD_rot as None (unused).
        self.vq_PD_rot = None

    def _dequant_gptq(self):
        """Dequantize GPTQ weights to fp16."""
        out_d, in_d = self.qweight_int.shape
        W = torch.zeros(out_d, in_d, dtype=torch.float16, device=self.qweight_int.device)
        n_groups = self.scales.shape[1]
        for gi in range(n_groups):
            col0 = gi * self.gptq_groupsize
            col1 = min(col0 + self.gptq_groupsize, in_d)
            q_slice = self.qweight_int[:, col0:col1].half()
            if self.gptq_sym:
                W[:, col0:col1] = q_slice * self.scales[:, gi:gi+1]
            else:
                W[:, col0:col1] = (q_slice - self.zeros[:, gi:gi+1]) * self.scales[:, gi:gi+1]
        return W

    def _vq4_matmul_python(self, x):
        """Python fallback for vq4: y = (x_permuted @ diagI) @ Q.T

        If codes+centroids are loaded, Q is rebuilt by codebook lookup. Otherwise
        fall back to the saved fp16 Q_rotated.
        """
        # Gather + block-Walsh rotation (replaces dense (in_d, in_d) matmul).
        # Accumulate the reduction in fp32: this fallback fires only for large-in_d
        # projections (e.g. Mixtral w2, in_d=14336) where an fp16-accumulated matmul
        # overflows fp16 range → inf → NaN. fp16 in/out but fp32 accumulation.
        x_rot = (x[..., self.vq_perm_sigma].float() @ self.vq_diagI.float())
        if self.vq_codes is not None and self.vq_centroids is not None:
            # Rebuild Q from codes + centroids with ONE fused gather (the per-cb
            # python loop was ~n_cb kernel launches per call — a large share of
            # prefill time). global idx = cb_id*K + code; identical fp16 Q values
            # and the same matmul afterwards => byte-identical to the old loop.
            out_d, codes_per_row = self.vq_codes.shape
            n_cb, K, vdim = self.vq_centroids.shape
            codes_per_cb = codes_per_row // n_cb
            offs = self._vq4_py_offs
            if offs is None or offs.shape[0] != codes_per_row or offs.device != x_rot.device:
                offs = (torch.arange(codes_per_row, device=x_rot.device)
                        // codes_per_cb) * K                     # (codes_per_row,) long
                self._vq4_py_offs = offs
            idx = self.vq_codes.long() + offs.unsqueeze(0)       # (out_d, codes_per_row)
            Q = self.vq_centroids.reshape(n_cb * K, vdim)[idx]   # (out_d, cpr, vdim) fp16
            Q = Q.reshape(out_d, codes_per_row * vdim)
            if os.environ.get("GLORCQ_PREFILL_DEQUANT", "0") == "1":
                # fp16 tensor-core GEMM. cuBLAS accumulates half matmuls in fp32
                # internally (no fp16-accumulation overflow), and fp16 x_rot is
                # exactly what the fused-kernel path already feeds. Skips the
                # 2x fp32 Q conversion + runs ~10x faster than fp32 sgemm.
                return (x_rot.half() @ Q.T).float()
            return x_rot @ Q.float().T          # fp32 (caller casts at the very end)
        return x_rot @ self.vq_Q_rotated.float().T   # fp32


    def forward(self, x, precomputed_xU=None, precomputed_x_rot=None):
        """
        Forward pass: dequantize weights, matmul, add LoRA compensation.

        Args:
            x: (..., in_features) input tensor
            precomputed_xU: optional (..., rank) tensor — pre-computed ``x @ U``
                from cluster-parallel MoE. Skips redundant ``x @ U`` when
                multiple experts in the same cluster share the same U matrix.
            precomputed_x_rot: optional (..., in_features) float32 tensor —
                pre-rotated input ``x @ Pi.T`` for TurboQuant. Skips redundant
                rotation when multiple experts share the same Pi matrix.
        Returns:
            y: (..., out_features) output tensor
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # 1. Quantized weight matmul (S is pre-fused into SV, so no kernel-level
        # LoRA fusion needed — LoRA is always computed in Python below).
        if self.quant_type == "gptq":
            if _HAS_GPTQ_FUSED_KERNEL:
                y = gptq_dequant_matmul_fused(
                    x, self.qweight_int, self.scales, self.zeros,
                    self.gptq_groupsize, self.gptq_sym, lora_USV=None)
            else:
                W = self._dequant_gptq()
                y = x @ W.T
        elif self.quant_type == "turbo":
            if self._rotation_cache is not None:
                Pi = self._rotation_cache.get_pi(
                    self.turbo_dim, self.turbo_bits, self.turbo_seed)
                centroids = self._rotation_cache.get_centroids(
                    self.turbo_dim, self.turbo_bits, self.turbo_seed)
                if _HAS_FUSED_KERNEL and self.turbo_bits == 2:
                    y = turbo_dequant_matmul_fused(
                        x, self.packed_indices, self.norms,
                        Pi, centroids, self.turbo_bits, self.turbo_dim,
                        lora_USV=None,
                        precomputed_x_rot=precomputed_x_rot)
                else:
                    y = turbo_dequant_matmul(
                        x, self.packed_indices, self.norms,
                        Pi, centroids, self.turbo_bits, self.turbo_dim)
            elif self._turbo_dequant_W is not None:
                y = x @ self._turbo_dequant_W.T
            else:
                raise RuntimeError("TurboQuant weights not loaded")
        elif self.quant_type == "vq4":
            n_cb, K_cb, vdim = (self.vq_centroids.shape
                                if self.vq_centroids is not None else (0, 0, 0))
            # Kernel shmem budget: (K + n_cb * K_CB * VDIM) * 2 bytes must fit
            # in ~100 KB (A100 max opt-in). For Mixtral down_proj (in_d=14336,
            # n_cb=56) this is 143 KB → exceeds → silent kernel-launch fail →
            # "invalid argument" on next op. Fall back to Python for those.
            K_in = self.in_features
            shmem_bytes = (K_in + n_cb * K_cb * vdim) * 2
            _MAX_SHMEM = 96 * 1024  # 96 KB safe margin below 100 KB A100 limit
            # Prefill fast path (opt-in): for multi-token inputs the naive fused
            # GEMM kernel is ~200x off cuBLAS; dequant-to-fp16 + cuBLAS (the same
            # _vq4_matmul_python used by large-shmem projections) is far faster.
            # Env-gated, default OFF => byte-identical everywhere unless set.
            _prefill_dequant = (
                x.shape[0] > 4
                and os.environ.get("GLORCQ_PREFILL_DEQUANT", "0") == "1")
            can_use_kernel = (
                _HAS_VQ4_FUSED_KERNEL
                and self.vq_codes is not None
                and shmem_bytes <= _MAX_SHMEM
                and not _prefill_dequant
            )
            if can_use_kernel:
                # Rotate x (or reuse precomputed_x_rot), then call fused kernel
                if precomputed_x_rot is not None:
                    x_rot = precomputed_x_rot
                else:
                    x_permuted = x.half()[..., self.vq_perm_sigma]
                    x_rot = x_permuted @ self.vq_diagI
                codes_per_row = self.vq_codes.shape[1]
                codes_per_cb = codes_per_row // n_cb
                y = vq4_dequant_matmul(
                    x_rot, self.vq_codes, self.vq_centroids,
                    n_cb, codes_per_cb,
                )
            else:
                y = self._vq4_matmul_python(x)
        else:
            raise RuntimeError(f"Unknown quant_type: {self.quant_type}")

        # 2. LoRA compensation: y += a @ SV.T  where SV = V * S[None,:]
        # SV is pre-fused at load time — no elementwise multiply needed here.
        # If self.Sa is set, apply x_scaled = x * Sa before the x @ U matmul to
        # match the training-time formula lora^T = diag(Sa) @ U @ diag(Si) @ V.
        # NOTE: When Sa is set, we MUST recompute x @ U because Sa is per-expert
        # while precomputed_xU shares x @ U across a cluster of experts.
        a = None
        if self.SV is not None:
            if precomputed_xU is not None and self.Sa is None:
                a = precomputed_xU      # reuse cluster-parallel x @ U (no Sa)
            else:
                # fp32 throughout: Mixtral's large activation scale Sa makes
                # x*Sa overflow fp16 (>65504) → inf → NaN. Do the Sa multiply AND
                # the in_d reduction in fp32.
                if self.Sa is not None:
                    x_for_lora = x.float() * self.Sa.float()
                else:
                    x_for_lora = x.float()
                a = x_for_lora @ self.U.float()          # (..., rank)
            y = y.float() + a.float() @ self.SV.float().T        # fp32 accumulate

        # 3. Bias (keep fp32 accumulation)
        if self.bias_param is not None:
            y = y.float() + self.bias_param.float()

        # Cast back to the input dtype only at the very end — avoids intermediate
        # fp16 overflow (inf→NaN) in the large-in_d VQ4 matmul + LoRA reductions.
        y = y.to(x.dtype)

        # Restore original shape
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)

        return y

    def extra_repr(self):
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", quant_type={self.quant_type}"
        if self.U is not None:
            s += f", lora_rank={self.U.shape[1]}"
        return s


# ---------------------------------------------------------------------------
# GLoRCQExperts: quantized MoE experts (drop-in replacement for HF XxxExperts)
# ---------------------------------------------------------------------------
class GLoRCQExperts(nn.Module):
    """
    Quantized MoE experts — drop-in replacement for HF's XxxExperts classes.

    Uses 3 separate per-expert GLoRCQLinear lists (gate, up, down) to match
    the per-expert weight format from quantization Stage 1.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_experts = nn.ModuleList([
            GLoRCQLinear(hidden_dim, intermediate_dim)
            for _ in range(num_experts)
        ])
        self.up_experts = nn.ModuleList([
            GLoRCQLinear(hidden_dim, intermediate_dim)
            for _ in range(num_experts)
        ])
        self.down_experts = nn.ModuleList([
            GLoRCQLinear(intermediate_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """
        Same dispatch logic as HF XxxExperts.forward():
        1. Build one_hot expert_mask from top_k_index
        2. For each active expert: gather tokens -> gate+up -> act*mul -> down -> scatter
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_out = self.gate_experts[expert_idx](current_state)
            up_out = self.up_experts[expert_idx](current_state)
            current_hidden_states = self.act_fn(gate_out) * up_out

            current_hidden_states = self.down_experts[expert_idx](current_hidden_states)
            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx,
                current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states
