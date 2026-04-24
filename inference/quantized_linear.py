"""
GLoRCQ quantized linear layer for inference.

Replaces nn.Linear with a quantized version that supports:
  - GPTQ backend: per-group scalar dequant + matmul
  - TurboQuant backend: rotation + codebook dequant + matmul
  - LoRA compensation: x @ U * S @ V^T added to quantized output
"""

import torch
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
        self.U = None                # (in_d, rank) fp16
        self.S = None                # (rank,) fp16
        self.V = None                # (out_d, rank) fp16
        self.cluster_id = None       # set by model_builder for cluster-parallel LoRA

        # Bias
        self.bias_param = None       # (out_d,) fp16

    def load_gptq(self, packed_data, device="cuda"):
        """Load GPTQ quantized weights from packed data dict."""
        self.quant_type = "gptq"
        self.qweight_int = packed_data["qweight_int"].to(device)
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

    def load_lora(self, U, S, V, device="cuda"):
        """Load LoRA compensation matrices."""
        if U is not None:
            self.U = U.half().to(device)
            self.S = S.half().to(device)
            self.V = V.half().to(device)

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

    def forward(self, x, precomputed_xU=None):
        """
        Forward pass: dequantize weights, matmul, add LoRA compensation.

        Args:
            x: (..., in_features) input tensor
            precomputed_xU: optional (..., rank) tensor — pre-computed ``x @ U``
                from cluster-parallel MoE. Skips redundant ``x @ U`` when
                multiple experts in the same cluster share the same U matrix.
        Returns:
            y: (..., out_features) output tensor
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # 1. Quantized weight matmul
        # When precomputed_xU is provided, skip kernel-level LoRA fusion
        # (the Python LoRA path below will use precomputed_xU instead).
        _skip_lora_fusion = (precomputed_xU is not None)
        lora_fused = False
        if self.quant_type == "gptq":
            if _HAS_GPTQ_FUSED_KERNEL:
                lora_USV = None
                if self.U is not None and not _skip_lora_fusion:
                    lora_USV = (self.U, self.S, self.V)
                y = gptq_dequant_matmul_fused(
                    x, self.qweight_int, self.scales, self.zeros,
                    self.gptq_groupsize, self.gptq_sym, lora_USV=lora_USV)
                lora_fused = (lora_USV is not None)
            else:
                W = self._dequant_gptq()
                y = x @ W.T
        elif self.quant_type == "turbo":
            if self._rotation_cache is not None:
                Pi = self._rotation_cache.get_pi(
                    self.turbo_dim, self.turbo_bits, self.turbo_seed)
                centroids = self._rotation_cache.get_centroids(
                    self.turbo_dim, self.turbo_bits, self.turbo_seed)
                # Fused CUDA kernel path: dequant + matmul + optional LoRA in one kernel
                if _HAS_FUSED_KERNEL and self.turbo_bits == 2:
                    lora_USV = None
                    if self.U is not None and not _skip_lora_fusion:
                        lora_USV = (self.U, self.S, self.V)
                    y = turbo_dequant_matmul_fused(
                        x, self.packed_indices, self.norms,
                        Pi, centroids, self.turbo_bits, self.turbo_dim,
                        lora_USV=lora_USV)
                    lora_fused = (self.U is not None)
                else:
                    y = turbo_dequant_matmul(
                        x, self.packed_indices, self.norms,
                        Pi, centroids, self.turbo_bits, self.turbo_dim)
            elif self._turbo_dequant_W is not None:
                # Legacy pre-decompressed path
                y = x @ self._turbo_dequant_W.T
            else:
                raise RuntimeError("TurboQuant weights not loaded")
        else:
            raise RuntimeError(f"Unknown quant_type: {self.quant_type}")

        # 2. LoRA compensation: y += (x @ U) * S @ V^T (skip if already fused)
        if self.U is not None and not lora_fused:
            if precomputed_xU is not None:
                a = precomputed_xU      # reuse cluster-parallel result
            else:
                a = x @ self.U          # (..., rank)
            b = a * self.S              # (..., rank)
            y = y + b @ self.V.T        # (..., out_d)

        # 3. Bias
        if self.bias_param is not None:
            y = y + self.bias_param

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
