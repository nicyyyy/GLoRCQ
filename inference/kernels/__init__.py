"""
Python wrappers for fused dequant+matmul CUDA kernels.

Tries to import the compiled CUDA extensions; falls back to pure-PyTorch
implementations if not available.  If a CUDA kernel fails at runtime (e.g.
unsupported dimensions), the wrapper catches the error, logs a diagnostic
warning, and transparently retries with the pure-PyTorch path.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Cache for empty placeholder tensors (keyed by (device, dtype)).
# Avoids ~240 aten::empty_strided CPU calls/step when lora_USV is always None.
_empty_tensor_cache: dict = {}


def _empty_lora(device):
    """Return a cached empty float32 tensor for the no-LoRA case."""
    key = (str(device), torch.float32)
    if key not in _empty_tensor_cache:
        _empty_tensor_cache[key] = torch.empty(0, dtype=torch.float32, device=device)
    return _empty_tensor_cache[key]


# ---------------------------------------------------------------------------
# TurboQuant CUDA extension
# ---------------------------------------------------------------------------
_cuda_ext = None
try:
    from . import _turbo_matmul_cuda
    _cuda_ext = _turbo_matmul_cuda
except ImportError:
    pass


def is_cuda_available():
    """Check whether the compiled TurboQuant CUDA kernel is available."""
    return _cuda_ext is not None


# ---------------------------------------------------------------------------
# GPTQ CUDA extension
# ---------------------------------------------------------------------------
_gptq_cuda_ext = None
try:
    from . import _gptq_matmul_cuda
    _gptq_cuda_ext = _gptq_matmul_cuda
except ImportError:
    pass


def is_gptq_cuda_available():
    """Check whether the compiled GPTQ CUDA kernel is available."""
    return _gptq_cuda_ext is not None


# ---------------------------------------------------------------------------
# VQ4 CUDA extension
# ---------------------------------------------------------------------------
_vq4_cuda_ext = None
try:
    from . import _vq4_matmul_cuda
    _vq4_cuda_ext = _vq4_matmul_cuda
except ImportError:
    pass


def is_vq4_cuda_available():
    """Check whether the compiled VQ4 CUDA kernel is available."""
    return _vq4_cuda_ext is not None


_HAS_VQ4_GROUPED_GEMV = (
    _vq4_cuda_ext is not None
    and hasattr(_vq4_cuda_ext, "vq4_dequant_grouped_gemv")
)

_HAS_LORA_GROUPED_GEMV = (
    _vq4_cuda_ext is not None
    and hasattr(_vq4_cuda_ext, "lora_grouped_gemv")
)


def lora_grouped_gemv(A, B):
    """Batched GEMV replacing torch.bmm((E,1,K),(E,K,M)) → (E,M).
    Good for small-K, large-M (rank@out_d, SV pattern).

    Args:
        A: (E, K) or (E, 1, K) fp16
        B: (E, K, M) fp16
    Returns:
        Y: (E, M) fp16
    """
    if not _HAS_LORA_GROUPED_GEMV:
        if A.dim() == 2:
            A = A.unsqueeze(1)
        return torch.bmm(A, B).squeeze(1)
    return _vq4_cuda_ext.lora_grouped_gemv(A, B)


_HAS_LORA_U_GROUPED_GEMV = (
    _vq4_cuda_ext is not None
    and hasattr(_vq4_cuda_ext, "lora_u_grouped_gemv")
)

_HAS_SILU_AND_MUL = (
    _vq4_cuda_ext is not None
    and hasattr(_vq4_cuda_ext, "silu_and_mul")
)


def silu_and_mul(gate, up):
    """Fused SiLU(gate) * up in one CUDA kernel.

    Args:
        gate: fp16 tensor
        up:   fp16 tensor (same shape)
    Returns:
        fp16 tensor of same shape
    """
    if not _HAS_SILU_AND_MUL:
        return torch.nn.functional.silu(gate) * up
    return _vq4_cuda_ext.silu_and_mul(gate, up)


def lora_u_grouped_gemv(A, BT):
    """Batched K-parallel GEMV for large-K, small-M pattern.

    Equivalent to torch.bmm((E,1,K), (E,K,M)) → (E,M) but BT is pre-transposed
    to (E, M, K) so K is contiguous inner dim for coalesced K-parallel reduction.

    Args:
        A: (E, K) or (E, 1, K) fp16
        BT: (E, M, K) fp16 — pre-transposed weight
    Returns:
        Y: (E, M) fp16
    """
    if not _HAS_LORA_U_GROUPED_GEMV:
        # Fallback: transpose BT back to (E, K, M) and use bmm
        if A.dim() == 2:
            A = A.unsqueeze(1)
        return torch.bmm(A, BT.transpose(-1, -2)).squeeze(1)
    return _vq4_cuda_ext.lora_u_grouped_gemv(A, BT)


def vq4_dequant_matmul(x_rot, codes, centroids, n_cb, codes_per_cb):
    """Fused VQ4 dequant + matmul.
    Args:
        x_rot:         (B, K) fp16 — pre-rotated input (x @ PD_rot)
        codes:         (N, K/vdim) uint8
        centroids:     (n_cb, K_CB=256, vdim=4) fp16
        n_cb:          int — number of codebooks per row
        codes_per_cb:  int — code positions per codebook
    Returns:
        y:             (B, N) fp16
    """
    if _vq4_cuda_ext is None:
        raise RuntimeError("VQ4 CUDA kernel not available")
    return _vq4_cuda_ext.vq4_dequant_matmul(
        x_rot, codes, centroids, int(n_cb), int(codes_per_cb),
    )


def vq4_dequant_grouped_gemv(x_grouped, codes_cat, centroids_cat,
                              E, N, n_cb, codes_per_cb):
    """Grouped VQ4 GEMV for MoE (E experts with different inputs).
    Args:
        x_grouped:     (E, K) fp16
        codes_cat:     (E*N, K/vdim) uint8
        centroids_cat: (E*n_cb, K_CB=256, vdim=4) fp16
    Returns:
        y_cat:         (E*N,) fp16
    """
    if not _HAS_VQ4_GROUPED_GEMV:
        raise RuntimeError("VQ4 grouped GEMV not available")
    return _vq4_cuda_ext.vq4_dequant_grouped_gemv(
        x_grouped, codes_cat, centroids_cat,
        int(E), int(N), int(n_cb), int(codes_per_cb),
    )


_HAS_VQ4_GROUPED_GEMV_INDEXED = (
    _vq4_cuda_ext is not None
    and hasattr(_vq4_cuda_ext, "vq4_dequant_grouped_gemv_indexed")
)


def vq4_dequant_grouped_gemv_indexed(x_grouped, codes_all, centroids_all, sel,
                                     G, N, n_cb, codes_per_cb):
    """Expert-INDEXED grouped VQ4 GEMV: gathers the top-k expert weights inside
    the kernel via `sel` (int32, (G,)), reading codes/centroids from the FULL
    stacked (E_full-expert) tensors — no explicit index_select copy.
    Args:
        x_grouped:     (G, K) fp16 — per-slot rotated input
        codes_all:     (E_full*N, K/vdim) uint8 — ALL experts
        centroids_all: (E_full*n_cb, K_CB, vdim) fp16 — ALL experts
        sel:           (G,) int32 — expert index per slot
    Returns:
        y_cat:         (G*N,) fp16
    """
    if not _HAS_VQ4_GROUPED_GEMV_INDEXED:
        raise RuntimeError("VQ4 indexed grouped GEMV not available")
    return _vq4_cuda_ext.vq4_dequant_grouped_gemv_indexed(
        x_grouped, codes_all, centroids_all, sel,
        int(G), int(N), int(n_cb), int(codes_per_cb),
    )


# ---------------------------------------------------------------------------
# torch.compile custom op wrappers (allows Dynamo to trace through CUDA calls)
# ---------------------------------------------------------------------------
_HAS_GROUPED_GEMV = (
    _cuda_ext is not None
    and hasattr(_cuda_ext, "turbo_dequant_grouped_gemv")
)

if _cuda_ext is not None:
    @torch.library.custom_op("glorcq::turbo_dequant_matmul", mutates_args=())
    def _turbo_dequant_matmul_op(
        x_rot: torch.Tensor,
        packed_indices: torch.Tensor,
        norms: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        return _cuda_ext.turbo_dequant_matmul(
            x_rot, packed_indices, norms, centroids,
        )

    @_turbo_dequant_matmul_op.register_fake
    def _turbo_fake(x_rot, packed_indices, norms, centroids):
        return torch.empty(
            x_rot.shape[0], packed_indices.shape[0],
            dtype=torch.float16, device=x_rot.device,
        )

if _gptq_cuda_ext is not None:
    @torch.library.custom_op("glorcq::gptq_dequant_matmul", mutates_args=())
    def _gptq_dequant_matmul_op(
        x: torch.Tensor,
        qweight_i8: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        groupsize: int,
        sym: bool,
        lora_out: torch.Tensor,
    ) -> torch.Tensor:
        return _gptq_cuda_ext.gptq_dequant_matmul(
            x, qweight_i8, scales, zeros, groupsize, sym, lora_out,
        )

    @_gptq_dequant_matmul_op.register_fake
    def _gptq_fake(x, qweight_i8, scales, zeros, groupsize, sym, lora_out):
        return torch.empty(
            x.shape[0], qweight_i8.shape[0],
            dtype=torch.float32, device=x.device,
        )


# ---------------------------------------------------------------------------
# Pure-PyTorch fallbacks (reference implementations)
# ---------------------------------------------------------------------------
def _unpack_indices_2bit(packed, K):
    """Unpack (N, K/4) uint8 → (N, K) long indices for 2-bit quantization."""
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    unpacked = (packed.unsqueeze(-1) >> shifts) & 0x3   # (..., K/4, 4)
    unpacked = unpacked.reshape(packed.shape[0], -1)     # (N, K)
    return unpacked[:, :K].long()


def _pytorch_fallback(x_rot, packed_indices, norms, centroids, K):
    """
    Pure PyTorch reference: dequant + matmul for 2-bit TurboQuant.

    Args:
        x_rot:           (B, K) fp32, already rotated
        packed_indices:  (N, K/4) uint8
        norms:           (N,) fp32
        centroids:       (4,) fp32
        K:               input dimension

    Returns:
        y: (B, N) fp32
    """
    out_d = packed_indices.shape[0]
    device = x_rot.device

    y = torch.zeros(x_rot.shape[0], out_d, device=device, dtype=torch.float32)
    BLOCK = 1024
    for o0 in range(0, out_d, BLOCK):
        o1 = min(o0 + BLOCK, out_d)
        idx = _unpack_indices_2bit(packed_indices[o0:o1], K)  # (block, K) long
        W_block = centroids[idx] * norms[o0:o1, None]         # (block, K) fp32
        y[:, o0:o1] = x_rot @ W_block.T

    return y


def _gptq_pytorch_fallback(x, qweight_i8, scales, zeros, groupsize, sym):
    """
    Pure PyTorch reference: dequant + matmul for GPTQ (int8 qweights).

    Args:
        x:           (B, K) fp32
        qweight_i8:  (N, K) int8
        scales:      (N, n_groups) fp16
        zeros:       (N, n_groups) fp16
        groupsize:   int
        sym:         bool

    Returns:
        y: (B, N) fp32
    """
    N, K = qweight_i8.shape
    device = x.device
    n_groups = scales.shape[1]

    W = torch.zeros(N, K, dtype=torch.float32, device=device)
    for gi in range(n_groups):
        col0 = gi * groupsize
        col1 = min(col0 + groupsize, K)
        q_slice = qweight_i8[:, col0:col1].float()
        if sym:
            W[:, col0:col1] = q_slice * scales[:, gi:gi+1].float()
        else:
            W[:, col0:col1] = (q_slice - zeros[:, gi:gi+1].float()) * scales[:, gi:gi+1].float()

    return x @ W.T


# ---------------------------------------------------------------------------
# Public API: TurboQuant
# ---------------------------------------------------------------------------
def turbo_dequant_matmul_fused(x, packed_indices, norms, Pi, centroids,
                                bits, dim, lora_USV=None,
                                precomputed_x_rot=None):
    """
    Fused TurboQuant dequant + matmul with optional LoRA fusion.

    1. Rotate input:  x_rot = x.float() @ Pi.float().T  (or reuse precomputed)
    2. Pre-compute LoRA output (if provided via lora_USV)
    3. Call CUDA kernel (or PyTorch fallback)

    Args:
        x:              (B, K) fp16/fp32 input
        packed_indices: (N, K/4) uint8 packed 2-bit indices
        norms:          (N,) fp32 per-row norms
        Pi:             (K, K) rotation matrix (or None if precomputed_x_rot given)
        centroids:      (4,) fp32 codebook
        bits:           quantization bits (must be 2 for CUDA kernel)
        dim:            input dimension K
        lora_USV:       optional (U, S, V) tuple for LoRA fusion
        precomputed_x_rot: optional (B, K) float32 — pre-rotated input

    Returns:
        y: (B, N) fp16
    """
    # 1. Rotate input — or reuse precomputed rotation
    if precomputed_x_rot is not None:
        x_rot = precomputed_x_rot
    else:
        x_rot = (x.float() @ Pi.float().T).half()  # (B, K) fp16

    # 2. CUDA kernel or fallback
    K = x_rot.shape[1]
    if _cuda_ext is not None and bits == 2:
        try:
            y = _turbo_dequant_matmul_op(
                x_rot.contiguous(),
                packed_indices.contiguous(),
                norms.contiguous(),
                centroids.contiguous(),
            )
        except RuntimeError as e:
            B = x_rot.shape[0]
            N = packed_indices.shape[0]
            logger.warning(
                "TurboQuant CUDA kernel failed (B=%d, K=%d, N=%d): %s  "
                "— falling back to PyTorch", B, K, N, e)
            y = _pytorch_fallback(x_rot, packed_indices, norms, centroids, K)
    else:
        y = _pytorch_fallback(x_rot, packed_indices, norms, centroids, K)

    # Python-level LoRA (prefill path via quantized_linear)
    if lora_USV is not None:
        U, S, V = lora_USV
        y = y + ((x @ U) * S @ V.T)

    return y


def turbo_dequant_grouped_gemv_fused(x_grouped, packed_cat, norms_cat, centroids, N_out):
    """
    Grouped GEMV: E experts with different inputs in a single kernel call.

    Each expert k uses input x_grouped[k, :] and weight rows
    packed_cat[k*N_out:(k+1)*N_out, :].  Replaces a Python loop of E separate
    turbo_dequant_matmul_fused calls — reduces kernel launches from E to 1.

    Args:
        x_grouped:  (E, K) fp16 — one pre-rotated input row per expert
        packed_cat: (E*N_out, K/4) uint8 — all experts' weights concatenated
        norms_cat:  (E*N_out,) fp32 — all experts' per-row norms
        centroids:  (4,) fp32 — shared 2-bit codebook
        N_out:      int — output features per expert

    Returns:
        y: (E*N_out,) fp16
    """
    if _HAS_GROUPED_GEMV:
        return _cuda_ext.turbo_dequant_grouped_gemv(
            x_grouped.contiguous(),
            packed_cat.contiguous(),
            norms_cat.contiguous(),
            centroids.contiguous(),
            N_out,
        )
    # Fallback: loop over experts
    E = x_grouped.shape[0]
    K = x_grouped.shape[1]
    outs = []
    for k in range(E):
        x_k = x_grouped[k:k+1]  # (1, K) fp16
        p_k = packed_cat[k * N_out: (k + 1) * N_out]
        n_k = norms_cat[k * N_out: (k + 1) * N_out]
        y_k = _pytorch_fallback(x_k.float(), p_k, n_k, centroids, K)  # (1, N_out)
        outs.append(y_k.half().squeeze(0))  # (N_out,)
    return torch.cat(outs, dim=0)  # (E*N_out,)


# ---------------------------------------------------------------------------
# Public API: GPTQ
# ---------------------------------------------------------------------------
def gptq_dequant_matmul_fused(x, qweight_int, scales, zeros,
                                groupsize, sym=False, lora_USV=None):
    """
    Fused GPTQ dequant + matmul with optional LoRA fusion.

    1. Convert int32 → int8 if needed
    2. Pre-compute LoRA output (if provided)
    3. Call CUDA kernel (or PyTorch fallback)

    Args:
        x:           (B, K) fp16/fp32 input
        qweight_int: (N, K) int32 or int8 quantized weights
        scales:      (N, n_groups) fp16 per-group scale
        zeros:       (N, n_groups) fp16 per-group zero point
        groupsize:   columns per group
        sym:         symmetric quantization flag
        lora_USV:    optional (U, S, V) tuple for LoRA fusion

    Returns:
        y: (B, N) fp16
    """
    # 1. Convert int32 → int8 (if needed)
    if qweight_int.dtype != torch.int8:
        qweight_i8 = qweight_int.to(torch.int8)
    else:
        qweight_i8 = qweight_int

    # 2. Pre-compute LoRA
    lora_out = _empty_lora(x.device)
    if lora_USV is not None:
        U, S, V = lora_USV
        lora_out = ((x @ U) * S @ V.T).float()  # (B, N)

    # 3. CUDA kernel or fallback
    if _gptq_cuda_ext is not None:
        try:
            y = _gptq_dequant_matmul_op(
                x.float().contiguous(),
                qweight_i8.contiguous(),
                scales.contiguous(),
                zeros.contiguous(),
                groupsize,
                sym,
                lora_out.contiguous(),
            )
        except RuntimeError as e:
            B = x.shape[0]
            K = x.shape[1]
            N = qweight_i8.shape[0]
            logger.warning(
                "GPTQ CUDA kernel failed (B=%d, K=%d, N=%d, gs=%d): %s  "
                "— falling back to PyTorch", B, K, N, groupsize, e)
            y = _gptq_pytorch_fallback(
                x.float(), qweight_i8, scales, zeros, groupsize, sym)
            if lora_out.numel() > 0:
                y += lora_out
    else:
        y = _gptq_pytorch_fallback(
            x.float(), qweight_i8, scales, zeros, groupsize, sym)
        if lora_out.numel() > 0:
            y += lora_out

    return y.half()
