"""
TurboQuant-based weight quantizer for row-wise vector quantization.

Replaces GPTQ's per-group scalar quantization with:
  1. Random rotation Pi (shared per in_d)
  2. Lloyd-Max optimal codebook (per bit-width and dimension)
  3. Per-row L2 norm storage

Supports two rotation modes:
  - "qr": Full random orthogonal matrix via QR (original, 32 MB for d=2048)
  - "hadamard": Randomized Hadamard Transform (zero storage, O(d log d))

Usage:
  tq = TurboWeightQuantizer(nbits=2, rotation_type="hadamard")
  Q_W = tq.quantize_dequantize(W)  # W: (out_d, in_d) float32 GPU
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thirdpart", "turboquant"))

import torch

try:
    from turboquant.quantizer import TurboQuantMSE
except ImportError:
    TurboQuantMSE = None

# Cache TurboQuantMSE instances by (dim, nbits, seed, rotation_type)
_TURBO_CACHE = {}


class TurboWeightQuantizer:
    """Row-wise TurboQuant MSE quantizer for weight matrices."""

    def __init__(self, nbits: int = 2, device=None, seed: int = 42,
                 rotation_type: str = "qr"):
        if TurboQuantMSE is None:
            raise ImportError(
                "TurboQuant is not installed. Install it with:\n"
                "  uv pip install -e thirdpart/turboquant\n"
                "or remove --use_turboquant from your command."
            )
        self.nbits = nbits
        self.device = device or torch.device("cuda:0")
        self.seed = seed
        self.rotation_type = rotation_type

        if rotation_type == "hadamard":
            from hadamard_rotation import is_hadamard_available
            if not is_hadamard_available():
                raise ImportError(
                    "fast-hadamard-transform not installed. Install with:\n"
                    "  pip install thirdpart/fast-hadamard-transform\n"
                    "or use --rotation_type qr")

        # RHT signs cache: {dim: signs_tensor}
        self._rht_signs = {}

    def _get_rht_signs(self, dim: int):
        """Get or create RHT sign vector for given dimension."""
        if dim not in self._rht_signs:
            from hadamard_rotation import generate_rht_signs
            self._rht_signs[dim] = generate_rht_signs(dim, self.seed, self.device)
        return self._rht_signs[dim]

    def _get_quantizer(self, dim: int) -> TurboQuantMSE:
        """Get TurboQuantMSE instance (for codebook + boundaries)."""
        key = (dim, self.nbits, self.seed, self.rotation_type)
        if key not in _TURBO_CACHE:
            _TURBO_CACHE[key] = TurboQuantMSE(
                dim=dim, bits=self.nbits,
                device=self.device, dtype=torch.float32,
                seed=self.seed,
            )
        return _TURBO_CACHE[key]

    @torch.no_grad()
    def quantize_dequantize(self, W: torch.Tensor) -> torch.Tensor:
        """
        Quantize and immediately dequantize a weight matrix.

        Args:
            W: (out_d, in_d) weight matrix on GPU, float32
        Returns:
            Q_W: (out_d, in_d) dequantized weight, same device/dtype as input
        """
        if self.rotation_type == "hadamard":
            return self._quantize_dequantize_rht(W)
        # Original QR path
        out_d, in_d = W.shape
        orig_dtype = W.dtype
        W_f32 = W.float()
        tq = self._get_quantizer(in_d)
        q = tq.quantize(W_f32)       # MSEQuantized (packed indices + norms)
        Q_W = tq.dequantize(q)       # (out_d, in_d) float32
        return Q_W.to(dtype=orig_dtype)

    @torch.no_grad()
    def _quantize_dequantize_rht(self, W: torch.Tensor) -> torch.Tensor:
        """RHT path: external Hadamard rotation + TurboQuant scalar quantization."""
        from hadamard_rotation import rht_forward, rht_backward

        orig_dtype = W.dtype
        dim = W.shape[-1]
        signs = self._get_rht_signs(dim)
        tq = self._get_quantizer(dim)  # for centroids + decision_boundaries

        W_f32 = W.float()

        # Per-row L2 norm (same as TurboQuantMSE.quantize)
        norms = W_f32.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        W_unit = W_f32 / norms

        # RHT forward rotation
        W_rot = rht_forward(W_unit, signs)

        # Scalar quantize per coordinate (searchsorted into Lloyd-Max buckets)
        indices = torch.searchsorted(tq.decision_boundaries, W_rot.contiguous())

        # Dequantize: centroid lookup
        W_rot_hat = tq.centroids[indices]

        # RHT backward + rescale by norms
        W_hat = rht_backward(W_rot_hat, signs) * norms
        return W_hat.to(dtype=orig_dtype)

    @torch.no_grad()
    def quantize_compressed(self, W: torch.Tensor):
        """
        Quantize a weight matrix and return both dequantized weights and
        compressed representation for real-quant saving.

        Args:
            W: (out_d, in_d) weight matrix on GPU, float32
        Returns:
            Q_W: (out_d, in_d) dequantized weight, same device/dtype as input
            compressed: dict with packed indices and norms for reconstruction
        """
        if self.rotation_type == "hadamard":
            return self._quantize_compressed_rht(W)
        # Original QR path
        out_d, in_d = W.shape
        orig_dtype = W.dtype
        W_f32 = W.float()
        tq = self._get_quantizer(in_d)
        q = tq.quantize(W_f32)
        Q_W = tq.dequantize(q)
        compressed = {
            "packed_indices": q.indices.cpu(),
            "norms": q.norms.cpu(),
            "bits": q.bits,
            "dim": in_d,
            "seed": self.seed,
            "rotation_type": "qr",
        }
        return Q_W.to(dtype=orig_dtype), compressed

    @torch.no_grad()
    def _quantize_compressed_rht(self, W: torch.Tensor):
        """RHT path for quantize_compressed."""
        from hadamard_rotation import rht_forward, rht_backward
        from turboquant.quantizer import _pack_indices

        orig_dtype = W.dtype
        dim = W.shape[-1]
        signs = self._get_rht_signs(dim)
        tq = self._get_quantizer(dim)

        W_f32 = W.float()
        norms = W_f32.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        W_unit = W_f32 / norms

        W_rot = rht_forward(W_unit, signs)
        indices = torch.searchsorted(tq.decision_boundaries, W_rot.contiguous())

        # Dequantize for returning Q_W
        W_rot_hat = tq.centroids[indices]
        W_hat = rht_backward(W_rot_hat, signs) * norms

        # Pack indices for storage
        packed = _pack_indices(indices, self.nbits)
        compressed = {
            "packed_indices": packed.cpu(),
            "norms": norms.squeeze(-1).cpu(),
            "bits": self.nbits,
            "dim": dim,
            "seed": self.seed,
            "rotation_type": "hadamard",
        }
        return W_hat.to(dtype=orig_dtype), compressed
