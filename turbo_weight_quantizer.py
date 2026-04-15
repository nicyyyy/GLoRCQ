"""
TurboQuant-based weight quantizer for row-wise vector quantization.

Replaces GPTQ's per-group scalar quantization with:
  1. Random rotation Pi (shared per in_d)
  2. Lloyd-Max optimal codebook (per bit-width and dimension)
  3. Per-row L2 norm storage

Usage:
  tq = TurboWeightQuantizer(nbits=2)
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

# Cache TurboQuantMSE instances by (dim, nbits, seed) to avoid regeneration
_TURBO_CACHE = {}


class TurboWeightQuantizer:
    """Row-wise TurboQuant MSE quantizer for weight matrices."""

    def __init__(self, nbits: int = 2, device=None, seed: int = 42):
        if TurboQuantMSE is None:
            raise ImportError(
                "TurboQuant is not installed. Install it with:\n"
                "  uv pip install -e thirdpart/turboquant\n"
                "or remove --use_turboquant from your command."
            )
        self.nbits = nbits
        self.device = device or torch.device("cuda:0")
        self.seed = seed

    def _get_quantizer(self, dim: int) -> TurboQuantMSE:
        key = (dim, self.nbits, self.seed)
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
        out_d, in_d = W.shape
        orig_dtype = W.dtype
        W_f32 = W.float()
        tq = self._get_quantizer(in_d)
        q = tq.quantize(W_f32)       # MSEQuantized (packed indices + norms)
        Q_W = tq.dequantize(q)       # (out_d, in_d) float32
        return Q_W.to(dtype=orig_dtype)

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
        }
        return Q_W.to(dtype=orig_dtype), compressed
