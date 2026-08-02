"""
Randomized Hadamard Transform (RHT) utilities for TurboQuant rotation replacement.

RHT provides an alternative to the full QR random orthogonal matrix used in TurboQuant:
  - No stored rotation matrix (only d sign bits vs d×d dense matrix)
  - O(d log d) computation (vs O(d²) for dense matmul)
  - Eliminates 32 MB HBM read per rotation for d=2048

Mathematical equivalence:
  Full QR:  y = Pi · x       where Pi ∈ R^{d×d} is random orthogonal (from QR of Gaussian)
  RHT:      y = (1/√d) H · D · x   where D = diag(±1), H = Hadamard matrix

Both produce coordinates that are approximately Gaussian(0, 1/d) in high dimensions,
which matches the TurboQuant codebook assumption.

Requires: pip install fast-hadamard-transform (Dao-AILab)
"""

import torch

try:
    from fast_hadamard_transform import hadamard_transform
    _HAS_FAST_HADAMARD = True
except ImportError:
    _HAS_FAST_HADAMARD = False


def is_hadamard_available():
    """Check if fast-hadamard-transform is installed."""
    return _HAS_FAST_HADAMARD


def generate_rht_signs(d, seed=42, device="cuda"):
    """Generate random ±1 sign vector for Randomized Hadamard Transform.

    Args:
        d: dimension
        seed: random seed (deterministic, matches quantization-time seed)
        device: target device

    Returns:
        signs: (d,) float32 tensor of ±1 values
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=rng) * 2 - 1
    return signs.float().to(device)


def rht_forward(x, signs):
    """Apply Randomized Hadamard Transform: y = (1/√d) · H · diag(signs) · x

    Args:
        x: (..., d) input tensor (float32)
        signs: (d,) ±1 sign vector

    Returns:
        y: (..., d) rotated tensor, preserving L2 norm
    """
    if not _HAS_FAST_HADAMARD:
        raise RuntimeError("fast-hadamard-transform not installed. "
                          "Install with: pip install fast-hadamard-transform")
    x_signed = x * signs
    # scale = 1/√d makes the transform norm-preserving: ||Hx|| = ||x||
    return hadamard_transform(x_signed, scale=1.0 / (x.shape[-1] ** 0.5))


def rht_backward(y, signs):
    """Apply inverse RHT: x = diag(signs) · (1/√d) · H · y

    Since H is symmetric and (1/√d)H is orthogonal, the inverse is:
      x = D · (1/√d) H · y

    Args:
        y: (..., d) rotated tensor (float32)
        signs: (d,) ±1 sign vector

    Returns:
        x: (..., d) original-space tensor
    """
    if not _HAS_FAST_HADAMARD:
        raise RuntimeError("fast-hadamard-transform not installed.")
    h_y = hadamard_transform(y, scale=1.0 / (y.shape[-1] ** 0.5))
    return h_y * signs
