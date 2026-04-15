"""
Correctness tests for GPTQ fused dequant+matmul CUDA kernel.

Generates random GPTQ-format data and compares CUDA kernel output
against a pure-PyTorch reference implementation.
"""

import sys
import torch
import time


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------

def reference_gptq_dequant_matmul(x, qweight_i8, scales, zeros, groupsize,
                                   sym, lora_out=None):
    """
    Pure PyTorch reference: dequant GPTQ weights + matmul.

    Args:
        x:           (B, K) fp32
        qweight_i8:  (N, K) int8
        scales:      (N, n_groups) fp16
        zeros:       (N, n_groups) fp16
        groupsize:   int
        sym:         bool
        lora_out:    (B, N) fp32 or None

    Returns:
        y: (B, N) fp32
    """
    N, K = qweight_i8.shape
    n_groups = scales.shape[1]
    device = x.device

    W = torch.zeros(N, K, dtype=torch.float32, device=device)
    for gi in range(n_groups):
        col0 = gi * groupsize
        col1 = min(col0 + groupsize, K)
        q_slice = qweight_i8[:, col0:col1].float()
        if sym:
            W[:, col0:col1] = q_slice * scales[:, gi:gi+1].float()
        else:
            W[:, col0:col1] = (q_slice - zeros[:, gi:gi+1].float()) * scales[:, gi:gi+1].float()

    y = x.float() @ W.T

    if lora_out is not None and lora_out.numel() > 0:
        y += lora_out

    return y


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_gptq_test_data(B, N, K, groupsize=128, sym=False, bits=2,
                         device="cuda", with_lora=False, lora_rank=64):
    """Generate random GPTQ format test data."""
    # Quantized weights: int8 range based on bits and sym
    if sym:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1

    qweight_i8 = torch.randint(qmin, qmax + 1, (N, K), device=device,
                                dtype=torch.int8)

    # Per-group scales and zeros
    n_groups = (K + groupsize - 1) // groupsize
    scales = (torch.randn(N, n_groups, device=device, dtype=torch.float16).abs() * 0.1
              + 0.01)
    if sym:
        zeros = torch.zeros(N, n_groups, device=device, dtype=torch.float16)
    else:
        zeros = torch.randn(N, n_groups, device=device, dtype=torch.float16) * 0.5

    # Input
    x = torch.randn(B, K, device=device, dtype=torch.float32)

    # LoRA
    lora_out = torch.empty(0, device=device, dtype=torch.float32)
    if with_lora:
        U = torch.randn(K, lora_rank, device=device, dtype=torch.float16)
        S = torch.randn(lora_rank, device=device, dtype=torch.float16)
        V = torch.randn(N, lora_rank, device=device, dtype=torch.float16)
        lora_out = ((x.half() @ U) * S @ V.T).float()

    return x, qweight_i8, scales, zeros, groupsize, sym, lora_out


def run_test(name, B, N, K, groupsize=128, sym=False, bits=2,
             with_lora=False, tol=0.01):
    """Run a single correctness test."""
    print(f"  [{name}] B={B}, N={N}, K={K}, gs={groupsize}, "
          f"sym={sym}, bits={bits}, lora={with_lora} ... ",
          end="", flush=True)

    x, qweight_i8, scales, zeros, gs, is_sym, lora_out = make_gptq_test_data(
        B, N, K, groupsize=groupsize, sym=sym, bits=bits,
        with_lora=with_lora)

    # Reference
    ref = reference_gptq_dequant_matmul(
        x, qweight_i8, scales, zeros, gs, is_sym,
        lora_out if lora_out.numel() > 0 else None)

    # CUDA kernel
    from glorcq.inference.kernels._gptq_matmul_cuda import gptq_dequant_matmul
    out = gptq_dequant_matmul(
        x.contiguous(), qweight_i8.contiguous(),
        scales.contiguous(), zeros.contiguous(),
        gs, is_sym, lora_out.contiguous())
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
        from glorcq.inference.kernels._gptq_matmul_cuda import gptq_dequant_matmul
    except ImportError as e:
        print(f"Failed to import CUDA extension: {e}")
        print("Please build first: cd glorcq/inference/kernels && pip install -e .")
        sys.exit(1)

    print("=" * 70)
    print("GPTQ Fused Dequant+Matmul Kernel — Correctness Tests")
    print("=" * 70)

    results = []

    # Test 1: GEMV B=1, (4096,4096) asym
    results.append(run_test(
        "GEMV q_proj B=1 asym", B=1, N=4096, K=4096, sym=False))

    # Test 2: GEMV B=1, (14336,4096) sym
    results.append(run_test(
        "GEMV gate_proj B=1 sym", B=1, N=14336, K=4096, sym=True))

    # Test 3: GEMV B=4, (4096,14336)
    results.append(run_test(
        "GEMV down_proj B=4", B=4, N=4096, K=14336))

    # Test 4: GEMM B=16, (14336,4096) asym
    results.append(run_test(
        "GEMM gate_proj B=16 asym", B=16, N=14336, K=4096, sym=False))

    # Test 5: GEMM B=32, (4096,14336) sym
    results.append(run_test(
        "GEMM down_proj B=32 sym", B=32, N=4096, K=14336, sym=True))

    # Test 6: With LoRA fusion
    results.append(run_test(
        "GEMM+LoRA B=16", B=16, N=14336, K=4096, with_lora=True))

    # Test 7: Without LoRA
    results.append(run_test(
        "GEMV no-LoRA B=1", B=1, N=14336, K=4096, with_lora=False))

    # Test 8: Small dims (edge case)
    results.append(run_test(
        "Small B=1 64x64", B=1, N=64, K=64, groupsize=64))

    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("All tests PASSED!")

        # Quick speed comparison
        print("\n--- Speed comparison (GEMV B=1, N=14336, K=4096) ---")
        x, qw, sc, zr, gs, sym, lora = make_gptq_test_data(
            1, 14336, 4096, sym=False)

        from glorcq.inference.kernels._gptq_matmul_cuda import gptq_dequant_matmul

        # Warmup
        for _ in range(10):
            gptq_dequant_matmul(x, qw, sc, zr, gs, sym, lora)
        torch.cuda.synchronize()

        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            gptq_dequant_matmul(x, qw, sc, zr, gs, sym, lora)
        torch.cuda.synchronize()
        cuda_us = (time.perf_counter() - t0) / iters * 1e6

        # PyTorch reference timing
        for _ in range(10):
            reference_gptq_dequant_matmul(x, qw, sc, zr, gs, sym)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            reference_gptq_dequant_matmul(x, qw, sc, zr, gs, sym)
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
