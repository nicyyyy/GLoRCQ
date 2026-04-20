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
# torch.compile custom op wrappers (allows Dynamo to trace through CUDA calls)
# ---------------------------------------------------------------------------
if _cuda_ext is not None:
    @torch.library.custom_op("glorcq::turbo_dequant_matmul", mutates_args=())
    def _turbo_dequant_matmul_op(
        x_rot: torch.Tensor,
        packed_indices: torch.Tensor,
        norms: torch.Tensor,
        centroids: torch.Tensor,
        lora_out: torch.Tensor,
    ) -> torch.Tensor:
        return _cuda_ext.turbo_dequant_matmul(
            x_rot, packed_indices, norms, centroids, lora_out,
        )

    @_turbo_dequant_matmul_op.register_fake
    def _turbo_fake(x_rot, packed_indices, norms, centroids, lora_out):
        return torch.empty(
            x_rot.shape[0], packed_indices.shape[0],
            dtype=torch.float32, device=x_rot.device,
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
                                bits, dim, lora_USV=None):
    """
    Fused TurboQuant dequant + matmul with optional LoRA fusion.

    1. Rotate input:  x_rot = x.float() @ Pi.float().T
    2. Pre-compute LoRA output (if provided)
    3. Call CUDA kernel (or PyTorch fallback)

    Args:
        x:              (B, K) fp16/fp32 input
        packed_indices: (N, K/4) uint8 packed 2-bit indices
        norms:          (N,) fp32 per-row norms
        Pi:             (K, K) rotation matrix
        centroids:      (4,) fp32 codebook
        bits:           quantization bits (must be 2 for CUDA kernel)
        dim:            input dimension K
        lora_USV:       optional (U, S, V) tuple for LoRA fusion

    Returns:
        y: (B, N) fp16
    """
    # 1. Rotate input (cuBLAS)
    x_rot = x.float() @ Pi.float().T  # (B, K)

    # 2. Pre-compute LoRA
    lora_out = torch.empty(0, device=x.device, dtype=torch.float32)
    if lora_USV is not None:
        U, S, V = lora_USV
        lora_out = ((x @ U) * S @ V.T).float()  # (B, N)

    # 3. CUDA kernel or fallback
    K = x_rot.shape[1]
    if _cuda_ext is not None and bits == 2:
        try:
            y = _turbo_dequant_matmul_op(
                x_rot.contiguous(),
                packed_indices.contiguous(),
                norms.contiguous(),
                centroids.contiguous(),
                lora_out.contiguous(),
            )
        except RuntimeError as e:
            B = x_rot.shape[0]
            N = packed_indices.shape[0]
            logger.warning(
                "TurboQuant CUDA kernel failed (B=%d, K=%d, N=%d): %s  "
                "— falling back to PyTorch", B, K, N, e)
            y = _pytorch_fallback(x_rot, packed_indices, norms, centroids, K)
            if lora_out.numel() > 0:
                y += lora_out
    else:
        y = _pytorch_fallback(x_rot, packed_indices, norms, centroids, K)
        if lora_out.numel() > 0:
            y += lora_out

    return y.half()


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
    lora_out = torch.empty(0, device=x.device, dtype=torch.float32)
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
