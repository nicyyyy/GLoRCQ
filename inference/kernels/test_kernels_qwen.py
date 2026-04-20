"""
Correctness tests for CUDA kernels at actual Qwen1.5-MoE-A2.7B dimensions.

The original test_kernels.py and test_gptq_kernel.py only tested Mixtral-scale
dimensions (K=4096, N=14336). Qwen1.5-MoE uses much smaller expert dimensions:

  gate_proj / up_proj:   (K=2048, N=1408)
  down_proj:             (K=1408, N=2048)
  shared_expert gate/up: (K=2048, N=5632)
  shared_expert down:    (K=5632, N=2048)
  attn q_proj:           (K=2048, N=2048)

Run with CUDA_LAUNCH_BLOCKING=1 to get precise error locations:
  CUDA_LAUNCH_BLOCKING=1 python test_kernels_qwen.py
"""

import sys
import os
import torch
import traceback

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def unpack_indices_2bit(packed, K):
    """Unpack (N, K/4) uint8 -> (N, K) long."""
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    unpacked = (packed.unsqueeze(-1) >> shifts) & 0x3
    unpacked = unpacked.reshape(packed.shape[0], -1)
    return unpacked[:, :K].long()


def reference_turbo_matmul(x_rot, packed_indices, norms, centroids, K,
                           lora_out=None):
    """Pure PyTorch reference for TurboQuant."""
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


def reference_gptq_dequant_matmul(x, qweight_i8, scales, zeros, groupsize,
                                   sym, lora_out=None):
    """Pure PyTorch reference for GPTQ."""
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
# Data generators
# ---------------------------------------------------------------------------

def make_turbo_data(B, N, K, device="cuda", with_lora=False, lora_rank=64):
    """Generate random TurboQuant data."""
    assert K % 4 == 0, "K must be divisible by 4"
    packed_indices = torch.randint(0, 256, (N, K // 4), device=device,
                                   dtype=torch.uint8)
    norms = torch.randn(N, device=device, dtype=torch.float32).abs() + 0.1
    centroids = torch.randn(4, device=device, dtype=torch.float32)
    if B > 0:
        x_rot = torch.randn(B, K, device=device, dtype=torch.float32)
    else:
        x_rot = torch.empty(0, K, device=device, dtype=torch.float32)

    lora_out = torch.empty(0, device=device, dtype=torch.float32)
    if with_lora and B > 0:
        lora_out = torch.randn(B, N, device=device, dtype=torch.float32) * 0.1

    return x_rot, packed_indices, norms, centroids, lora_out


def make_gptq_data(B, N, K, groupsize=128, sym=False, device="cuda",
                    with_lora=False):
    """Generate random GPTQ data."""
    qweight_i8 = torch.randint(-2, 3, (N, K), device=device, dtype=torch.int8)
    n_groups = (K + groupsize - 1) // groupsize
    scales = (torch.randn(N, n_groups, device=device, dtype=torch.float16).abs()
              * 0.1 + 0.01)
    if sym:
        zeros = torch.zeros(N, n_groups, device=device, dtype=torch.float16)
    else:
        zeros = torch.randn(N, n_groups, device=device, dtype=torch.float16) * 0.5

    if B > 0:
        x = torch.randn(B, K, device=device, dtype=torch.float32)
    else:
        x = torch.empty(0, K, device=device, dtype=torch.float32)
    lora_out = torch.empty(0, device=device, dtype=torch.float32)
    if with_lora and B > 0:
        lora_out = torch.randn(B, N, device=device, dtype=torch.float32) * 0.1

    return x, qweight_i8, scales, zeros, groupsize, sym, lora_out


# ---------------------------------------------------------------------------
# Single-test runners
# ---------------------------------------------------------------------------

def run_turbo_test(name, B, N, K, with_lora=False, tol=0.01):
    """Run one TurboQuant kernel test. Returns (passed, error_msg)."""
    label = (f"  [TurboQuant {name}] B={B}, N={N}, K={K}, "
             f"lora={with_lora}")
    print(f"{label} ... ", end="", flush=True)

    try:
        x_rot, packed, norms, centroids, lora_out = make_turbo_data(
            B, N, K, with_lora=with_lora)

        ref = reference_turbo_matmul(
            x_rot, packed, norms, centroids, K,
            lora_out if lora_out.numel() > 0 else None)

        from _turbo_matmul_cuda import turbo_dequant_matmul
        out = turbo_dequant_matmul(
            x_rot.contiguous(), packed.contiguous(),
            norms.contiguous(), centroids.contiguous(),
            lora_out.contiguous())
        torch.cuda.synchronize()

        # Verify CUDA context is still healthy (catches async errors from B=0)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        if B == 0:
            passed = out.shape == (0, N)
            print(f"{'PASS' if passed else 'FAIL'}  shape={out.shape}")
            return passed

        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        ref_scale = ref.abs().mean().item()
        rel_err = mean_err / (ref_scale + 1e-8)

        passed = max_err < tol
        status = "PASS" if passed else "FAIL"
        print(f"{status}  max_err={max_err:.6f}  rel_err={rel_err:.6f}")

        if not passed:
            print(f"    ref_scale={ref_scale:.4f}  mean_err={mean_err:.6f}")
            diff = (out - ref).abs()
            print(f"    Top-5 errors: "
                  f"{torch.topk(diff.flatten(), min(5, diff.numel())).values.tolist()}")

        return passed

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False


def run_gptq_test(name, B, N, K, groupsize=128, sym=False,
                  with_lora=False, tol=0.01):
    """Run one GPTQ kernel test. Returns passed."""
    label = (f"  [GPTQ {name}] B={B}, N={N}, K={K}, gs={groupsize}, "
             f"sym={sym}, lora={with_lora}")
    print(f"{label} ... ", end="", flush=True)

    try:
        x, qw, sc, zr, gs, is_sym, lora_out = make_gptq_data(
            B, N, K, groupsize=groupsize, sym=sym, with_lora=with_lora)

        ref = reference_gptq_dequant_matmul(
            x, qw, sc, zr, gs, is_sym,
            lora_out if lora_out.numel() > 0 else None)

        from _gptq_matmul_cuda import gptq_dequant_matmul
        out = gptq_dequant_matmul(
            x.contiguous(), qw.contiguous(),
            sc.contiguous(), zr.contiguous(),
            gs, is_sym, lora_out.contiguous())
        torch.cuda.synchronize()

        # Verify CUDA context is still healthy (catches async errors from B=0)
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        if B == 0:
            passed = out.shape == (0, N)
            print(f"{'PASS' if passed else 'FAIL'}  shape={out.shape}")
            return passed

        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        ref_scale = ref.abs().mean().item()
        rel_err = mean_err / (ref_scale + 1e-8)

        passed = max_err < tol
        status = "PASS" if passed else "FAIL"
        print(f"{status}  max_err={max_err:.6f}  rel_err={rel_err:.6f}")

        if not passed:
            print(f"    ref_scale={ref_scale:.4f}  mean_err={mean_err:.6f}")
            diff = (out - ref).abs()
            print(f"    Top-5 errors: "
                  f"{torch.topk(diff.flatten(), min(5, diff.numel())).values.tolist()}")

        return passed

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test matrix: Qwen1.5-MoE-A2.7B actual dimensions
# ---------------------------------------------------------------------------

# (name, K, N) — all actual Qwen1.5-MoE layer shapes
QWEN_DIMS = [
    ("expert gate_proj",       2048, 1408),
    ("expert up_proj",         2048, 1408),
    ("expert down_proj",       1408, 2048),
    ("shared_expert gate_proj", 2048, 5632),
    ("shared_expert up_proj",  2048, 5632),
    ("shared_expert down_proj", 5632, 2048),
    ("attn q_proj",            2048, 2048),
    ("attn k_proj",            2048, 256),
    ("attn v_proj",            2048, 256),
    ("attn o_proj",            2048, 2048),
]

# Batch sizes: B=0 (MoE empty expert), B<=4 -> GEMV path, B>4 -> GEMM path
BATCH_SIZES = [0, 1, 2, 4, 5, 8, 16, 32]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        sys.exit(0)

    # Check which extensions are available
    turbo_ok = False
    gptq_ok = False
    try:
        from _turbo_matmul_cuda import turbo_dequant_matmul
        turbo_ok = True
    except ImportError as e:
        print(f"TurboQuant CUDA extension not available: {e}")

    try:
        from _gptq_matmul_cuda import gptq_dequant_matmul
        gptq_ok = True
    except ImportError as e:
        print(f"GPTQ CUDA extension not available: {e}")

    if not turbo_ok and not gptq_ok:
        print("No CUDA extensions available. Build first:")
        print("  cd glorcq/inference/kernels && pip install -e . --force-reinstall")
        sys.exit(1)

    blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0") == "1"
    if not blocking:
        print("TIP: Run with CUDA_LAUNCH_BLOCKING=1 for precise error locations.\n")

    all_results = []
    fail_list = []

    # -----------------------------------------------------------------------
    # TurboQuant tests
    # -----------------------------------------------------------------------
    if turbo_ok:
        print("=" * 72)
        print("TurboQuant Kernel — Qwen1.5-MoE Dimension Tests")
        print("=" * 72)

        for dim_name, K, N in QWEN_DIMS:
            if K % 4 != 0:
                print(f"  Skipping {dim_name} K={K} (not divisible by 4)")
                continue
            for B in BATCH_SIZES:
                path = "GEMV" if B <= 4 else "GEMM"
                tag = f"{dim_name} {path}"
                ok = run_turbo_test(tag, B, N, K)
                all_results.append(ok)
                if not ok:
                    fail_list.append(f"TurboQuant {tag} B={B} N={N} K={K}")

            # Also test with LoRA for B=1 and B=8
            for B in [1, 8]:
                path = "GEMV" if B <= 4 else "GEMM"
                tag = f"{dim_name} {path}+LoRA"
                ok = run_turbo_test(tag, B, N, K, with_lora=True)
                all_results.append(ok)
                if not ok:
                    fail_list.append(f"TurboQuant {tag} B={B} N={N} K={K}")

    # -----------------------------------------------------------------------
    # GPTQ tests
    # -----------------------------------------------------------------------
    if gptq_ok:
        print()
        print("=" * 72)
        print("GPTQ Kernel — Qwen1.5-MoE Dimension Tests")
        print("=" * 72)

        for dim_name, K, N in QWEN_DIMS:
            for B in BATCH_SIZES:
                path = "GEMV" if B <= 4 else "GEMM"
                tag = f"{dim_name} {path}"
                ok = run_gptq_test(tag, B, N, K, groupsize=128)
                all_results.append(ok)
                if not ok:
                    fail_list.append(f"GPTQ {tag} B={B} N={N} K={K}")

            # Test with LoRA for B=1 and B=8
            for B in [1, 8]:
                path = "GEMV" if B <= 4 else "GEMM"
                tag = f"{dim_name} {path}+LoRA"
                ok = run_gptq_test(tag, B, N, K, groupsize=128, with_lora=True)
                all_results.append(ok)
                if not ok:
                    fail_list.append(f"GPTQ {tag} B={B} N={N} K={K}")

            # Test sym=True for B=1
            ok = run_gptq_test(f"{dim_name} GEMV sym", 1, N, K,
                               groupsize=128, sym=True)
            all_results.append(ok)
            if not ok:
                fail_list.append(f"GPTQ {dim_name} sym B=1 N={N} K={K}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    passed = sum(all_results)
    total = len(all_results)
    print(f"Results: {passed}/{total} passed")

    if fail_list:
        print(f"\nFailed tests ({len(fail_list)}):")
        for f in fail_list:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests PASSED!")


if __name__ == "__main__":
    main()
