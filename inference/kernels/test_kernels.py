"""
Correctness tests for TurboQuant fused dequant+matmul CUDA kernel.

Generates random TurboQuant-format data and compares CUDA kernel output
against a pure-PyTorch reference implementation.
"""

import sys
import torch
import time

# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, no dependency on the extension)
# ---------------------------------------------------------------------------

def unpack_indices_2bit(packed, K):
    """Unpack (N, K/4) uint8 → (N, K) long."""
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    unpacked = (packed.unsqueeze(-1) >> shifts) & 0x3
    unpacked = unpacked.reshape(packed.shape[0], -1)
    return unpacked[:, :K].long()


def reference_turbo_matmul(x_rot, packed_indices, norms, centroids, K,
                           lora_out=None):
    """Pure PyTorch reference: y[b,n] = norms[n] * sum_k centroid[idx] * x_rot[b,k]."""
    N = packed_indices.shape[0]
    B = x_rot.shape[0]
    y = torch.zeros(B, N, device=x_rot.device, dtype=torch.float32)

    BLOCK = 1024
    for o0 in range(0, N, BLOCK):
        o1 = min(o0 + BLOCK, N)
        idx = unpack_indices_2bit(packed_indices[o0:o1], K)
        W_block = centroids[idx] * norms[o0:o1, None]
        y[:, o0:o1] = x_rot @ W_block.T

    if lora_out is not None and lora_out.numel() > 0:
        y += lora_out

    return y


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_test_data(B, N, K, device="cuda", with_lora=False, lora_rank=64):
    """Generate random TurboQuant format test data."""
    assert K % 4 == 0, "K must be divisible by 4"

    # Random rotation matrix (orthogonal)
    Pi = torch.randn(K, K, device=device, dtype=torch.float32)
    Pi, _ = torch.linalg.qr(Pi)

    # Random input
    x = torch.randn(B, K, device=device, dtype=torch.float16)

    # Packed indices: (N, K/4) uint8, each byte = 4 x 2-bit indices
    packed_indices = torch.randint(0, 256, (N, K // 4), device=device,
                                   dtype=torch.uint8)

    # Per-row norms
    norms = torch.randn(N, device=device, dtype=torch.float32).abs() + 0.1

    # Centroids: 4 random fp32 values
    centroids = torch.randn(4, device=device, dtype=torch.float32)

    # Pre-compute rotated input
    x_rot = x.float() @ Pi.float().T

    # LoRA
    lora_out = torch.empty(0, device=device, dtype=torch.float32)
    if with_lora:
        U = torch.randn(K, lora_rank, device=device, dtype=torch.float16)
        S = torch.randn(lora_rank, device=device, dtype=torch.float16)
        V = torch.randn(N, lora_rank, device=device, dtype=torch.float16)
        lora_out = ((x @ U) * S @ V.T).float()

    return x_rot, packed_indices, norms, centroids, lora_out, K


def run_test(name, B, N, K, with_lora=False, tol=0.01):
    """Run a single correctness test."""
    print(f"  [{name}] B={B}, N={N}, K={K}, lora={with_lora} ... ", end="", flush=True)

    x_rot, packed, norms, centroids, lora_out, K_dim = make_test_data(
        B, N, K, with_lora=with_lora)

    # Reference
    ref = reference_turbo_matmul(x_rot, packed, norms, centroids, K_dim,
                                  lora_out if lora_out.numel() > 0 else None)

    # CUDA kernel
    from _turbo_matmul_cuda import turbo_dequant_matmul
    out = turbo_dequant_matmul(x_rot, packed, norms, centroids, lora_out)
    torch.cuda.synchronize()

    # Compare
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    ref_scale = ref.abs().mean().item()
    rel_err = mean_err / (ref_scale + 1e-8)

    passed = max_err < tol
    status = "PASS" if passed else "FAIL"
    print(f"{status}  max_err={max_err:.6f}  mean_err={mean_err:.6f}  "
          f"rel_err={rel_err:.6f}  ref_scale={ref_scale:.4f}")

    if not passed:
        # Print more diagnostics
        diff = (out - ref).abs()
        print(f"    Top-5 errors: {torch.topk(diff.flatten(), 5).values.tolist()}")
        print(f"    Output sample (kernel): {out[0, :5].tolist()}")
        print(f"    Output sample (ref):    {ref[0, :5].tolist()}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        sys.exit(0)

    try:
        from _turbo_matmul_cuda import turbo_dequant_matmul
    except ImportError as e:
        print(f"Failed to import CUDA extension: {e}")
        print("Please build first: cd glorcq/inference/kernels && pip install -e .")
        sys.exit(1)

    print("=" * 70)
    print("TurboQuant Fused Dequant+Matmul Kernel — Correctness Tests")
    print("=" * 70)

    results = []

    # Test 1: GEMV batch=1, gate_proj dims
    results.append(run_test("GEMV gate_proj B=1", B=1, N=14336, K=4096))

    # Test 2: GEMV batch=1, down_proj dims
    results.append(run_test("GEMV down_proj B=1", B=1, N=4096, K=14336))

    # Test 3: Batched GEMV batch=4
    results.append(run_test("GEMV gate_proj B=4", B=4, N=14336, K=4096))

    # Test 4: GEMM batch=16
    results.append(run_test("GEMM gate_proj B=16", B=16, N=14336, K=4096))

    # Test 5: GEMM with LoRA fusion
    results.append(run_test("GEMM+LoRA B=16", B=16, N=14336, K=4096, with_lora=True))

    # Test 6: GEMV without LoRA (empty tensor)
    results.append(run_test("GEMV no-LoRA B=1", B=1, N=14336, K=4096, with_lora=False))

    # Test 7: Small dimensions (edge case)
    results.append(run_test("Small B=1", B=1, N=64, K=64))

    # Test 8: GEMM batch=32 down_proj
    results.append(run_test("GEMM down_proj B=32", B=32, N=4096, K=14336))

    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("All tests PASSED!")

        # Quick speed comparison
        print("\n--- Speed comparison (GEMV B=1, N=14336, K=4096) ---")
        x_rot, packed, norms, centroids, lora_out, K_dim = make_test_data(
            1, 14336, 4096)

        # Warmup
        for _ in range(10):
            turbo_dequant_matmul(x_rot, packed, norms, centroids, lora_out)
        torch.cuda.synchronize()

        # CUDA kernel timing
        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            turbo_dequant_matmul(x_rot, packed, norms, centroids, lora_out)
        torch.cuda.synchronize()
        cuda_us = (time.perf_counter() - t0) / iters * 1e6

        # PyTorch reference timing
        for _ in range(10):
            reference_turbo_matmul(x_rot, packed, norms, centroids, K_dim)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            reference_turbo_matmul(x_rot, packed, norms, centroids, K_dim)
        torch.cuda.synchronize()
        pytorch_us = (time.perf_counter() - t0) / iters * 1e6

        print(f"  CUDA kernel: {cuda_us:.1f} us")
        print(f"  PyTorch ref: {pytorch_us:.1f} us")
        print(f"  Speedup: {pytorch_us / cuda_us:.2f}x")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
