"""Unit test: VQ4 CUDA kernel vs Python reference reconstruction.

Loads a real expert's codes+centroids from e11_vq4_gs_hwl, computes forward with:
  (a) CUDA kernel: vq4_dequant_matmul(x_rot, codes, centroids, n_cb, codes_per_cb)
  (b) Python:      rebuild Q from codes+centroids, then x_rot @ Q.T
Compares outputs. Also tests grouped GEMV.
"""

import sys, torch
sys.path.insert(0, '/home/qyyang/repo/GLoRCQ')

from inference.kernels import (
    vq4_dequant_matmul, vq4_dequant_grouped_gemv, is_vq4_cuda_available,
)


def python_ref(x_rot, codes, centroids):
    """Rebuild Q, then x_rot @ Q.T"""
    out_d, codes_per_row = codes.shape
    n_cb, K_cb, vdim = centroids.shape
    codes_per_cb = codes_per_row // n_cb
    in_d = codes_per_row * vdim
    Q = torch.empty(out_d, in_d, dtype=torch.float16, device=codes.device)
    for cb_idx in range(n_cb):
        c0 = cb_idx * codes_per_cb
        c1 = c0 + codes_per_cb
        cb = centroids[cb_idx]                               # (K_cb, vdim)
        codes_blk = codes[:, c0:c1].long()                    # (out_d, codes_per_cb)
        looked = cb[codes_blk]                                # (out_d, codes_per_cb, vdim)
        Q[:, c0*vdim:c1*vdim] = looked.reshape(out_d, codes_per_cb * vdim)
    return x_rot @ Q.T


def test_single(desc, in_d, out_d, n_cb, batch=1):
    device = 'cuda:0'
    vdim = 4
    codes_per_row = in_d // vdim
    codes_per_cb = codes_per_row // n_cb
    torch.manual_seed(42)

    x_rot = torch.randn(batch, in_d, device=device, dtype=torch.float16) * 0.1
    codes = torch.randint(0, 256, (out_d, codes_per_row), device=device, dtype=torch.uint8)
    centroids = torch.randn(n_cb, 256, vdim, device=device, dtype=torch.float16) * 0.03

    y_py = python_ref(x_rot, codes, centroids)
    y_cu = vq4_dequant_matmul(x_rot, codes, centroids, n_cb, codes_per_cb)

    err = (y_py.float() - y_cu.float()).abs()
    print(f'{desc}: batch={batch} in_d={in_d} out_d={out_d} n_cb={n_cb}')
    print(f'  max_err={err.max():.5f}, mean_err={err.mean():.6f}, y_py.std={y_py.std():.4f}')
    passed = err.max() < 1e-2   # fp16 tolerance
    print(f'  {"PASS" if passed else "FAIL"}')
    return passed


def test_grouped(E, N, K, n_cb):
    device = 'cuda:0'
    vdim = 4
    codes_per_row = K // vdim
    codes_per_cb = codes_per_row // n_cb
    torch.manual_seed(123)

    x_grouped = torch.randn(E, K, device=device, dtype=torch.float16) * 0.1
    codes_cat = torch.randint(0, 256, (E*N, codes_per_row), device=device, dtype=torch.uint8)
    centroids_cat = torch.randn(E*n_cb, 256, vdim, device=device, dtype=torch.float16) * 0.03

    # Python: per-expert loop
    y_py_list = []
    for e in range(E):
        x_e = x_grouped[e:e+1]
        codes_e = codes_cat[e*N:(e+1)*N]
        cents_e = centroids_cat[e*n_cb:(e+1)*n_cb]
        y_e = python_ref(x_e, codes_e, cents_e)
        y_py_list.append(y_e[0])
    y_py = torch.cat(y_py_list, dim=0)                          # (E*N,)

    y_cu = vq4_dequant_grouped_gemv(x_grouped, codes_cat, centroids_cat, E, N, n_cb, codes_per_cb)

    err = (y_py.float() - y_cu.float()).abs()
    print(f'grouped_gemv: E={E} N={N} K={K} n_cb={n_cb}')
    print(f'  max_err={err.max():.5f}, mean_err={err.mean():.6f}, y_py.std={y_py.std():.4f}')
    passed = err.max() < 1e-2
    print(f'  {"PASS" if passed else "FAIL"}')
    return passed


if __name__ == '__main__':
    assert is_vq4_cuda_available(), "VQ4 CUDA kernel not available"
    print(f'CUDA kernel available: {is_vq4_cuda_available()}\n')

    print('=== Single-batch tests (GEMV path) ===')
    test_single('gate_proj shape', 2048, 1408, 1, batch=1)
    test_single('gate_proj shape B=4', 2048, 1408, 1, batch=4)
    test_single('down_proj shape', 1408, 2048, 11, batch=1)

    print('\n=== Multi-batch tests (GEMM path) ===')
    test_single('gate_proj prefill', 2048, 1408, 1, batch=32)
    test_single('down_proj prefill', 1408, 2048, 11, batch=32)

    print('\n=== Grouped GEMV (MoE) tests ===')
    test_grouped(E=4, N=1408, K=2048, n_cb=1)
    test_grouped(E=4, N=2048, K=1408, n_cb=11)
