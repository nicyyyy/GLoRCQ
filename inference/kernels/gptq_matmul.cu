/*
 * gptq_matmul.cu — Fused dequant + matmul CUDA kernels for GPTQ
 *                  + pybind11 wrapper (single-file build).
 *
 * Two kernels:
 *   1. GEMV  (B <= 4):  warp-per-row, memory-bound decode path
 *   2. GEMM  (B > 4):   tiled matmul with on-the-fly dequant for prefill
 *
 * Weight layout (plain int8, NOT bit-packed):
 *   qweight_i8: (N, K) int8  — one quantized value per element
 *   scales:     (N, n_groups) fp16 — per-group per-row scale
 *   zeros:      (N, n_groups) fp16 — per-group per-row zero point
 *
 * Dequant formula:
 *   sym:  W[n, k] = qweight_i8[n, k] * scales[n, k / groupsize]
 *   asym: W[n, k] = (qweight_i8[n, k] - zeros[n, k / groupsize]) * scales[n, k / groupsize]
 *
 * If lora_out is provided:  y[b, n] += lora_out[b, n]
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// -------------------------------------------------------------------------
// Device helpers
// -------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// -------------------------------------------------------------------------
// Kernel 1: GEMV — one warp per output row, for B <= 4
// -------------------------------------------------------------------------
static constexpr int WARPS_PER_BLOCK = 4;

__global__ void gptq_dequant_gemv_kernel(
    const float* __restrict__ x,              // (B, K)
    const int8_t* __restrict__ qweight_i8,    // (N, K) int8
    const half* __restrict__ scales,           // (N, n_groups) fp16
    const half* __restrict__ zeros,            // (N, n_groups) fp16
    const float* __restrict__ lora_out,        // (B, N) or nullptr
    float* __restrict__ y,                     // (B, N)
    int N, int K,
    int groupsize,
    int n_groups,
    int sym,
    int lora_valid)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (n >= N) return;

    const int b = blockIdx.y;

    const int8_t* qrow = qweight_i8 + (long long)n * K;
    const float* x_row = x + (long long)b * K;
    const half* scale_row = scales + (long long)n * n_groups;
    const half* zero_row = zeros + (long long)n * n_groups;

    float acc = 0.0f;

    // Each lane processes elements lane, lane+32, lane+64, ...
    // Process in groups: preload scale/zero per group boundary
    int prev_gi = -1;
    float s_val = 0.0f;
    float z_val = 0.0f;

    for (int k = lane; k < K; k += 32) {
        int gi = k / groupsize;
        if (gi != prev_gi) {
            s_val = __half2float(scale_row[gi]);
            z_val = sym ? 0.0f : __half2float(zero_row[gi]);
            prev_gi = gi;
        }

        float q = static_cast<float>(qrow[k]);
        float w = (q - z_val) * s_val;
        acc += w * x_row[k];
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        if (lora_valid) {
            acc += lora_out[(long long)b * N + n];
        }
        y[(long long)b * N + n] = acc;
    }
}


// -------------------------------------------------------------------------
// Kernel 2: GEMM — tiled matmul with on-the-fly dequant, for B > 4
// -------------------------------------------------------------------------
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 64;
static constexpr int GEMM_THREADS = 256;
static constexpr int TM = 4;
static constexpr int TN = 4;

__global__ void gptq_dequant_gemm_kernel(
    const float* __restrict__ x,              // (B, K)
    const int8_t* __restrict__ qweight_i8,    // (N, K) int8
    const half* __restrict__ scales,           // (N, n_groups) fp16
    const half* __restrict__ zeros,            // (N, n_groups) fp16
    const float* __restrict__ lora_out,        // (B, N) or nullptr
    float* __restrict__ y,                     // (B, N)
    int B, int N, int K,
    int groupsize,
    int n_groups,
    int sym,
    int lora_valid)
{
    const int bm_start = blockIdx.x * BM;
    const int bn_start = blockIdx.y * BN;

    const int thread_row = threadIdx.x / (BN / TN);  // 0..15
    const int thread_col = threadIdx.x % (BN / TN);  // 0..15

    float C_reg[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            C_reg[i][j] = 0.0f;

    __shared__ float A_smem[BM][BK];
    __shared__ float B_smem[BN][BK];

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BK;

        // Load A tile: x[bm_start..bm_start+BM, k_start..k_start+BK]
        for (int idx = threadIdx.x; idx < BM * BK; idx += GEMM_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            int b_idx = bm_start + row;
            int k_idx = k_start + col;
            if (b_idx < B && k_idx < K)
                A_smem[row][col] = x[(long long)b_idx * K + k_idx];
            else
                A_smem[row][col] = 0.0f;
        }

        // Load B tile: dequant qweight[bn_start..bn_start+BN, k_start..k_start+BK]
        for (int idx = threadIdx.x; idx < BN * BK; idx += GEMM_THREADS) {
            int n_local = idx / BK;
            int k_local = idx % BK;
            int n_idx = bn_start + n_local;
            int k_idx = k_start + k_local;
            if (n_idx < N && k_idx < K) {
                int gi = k_idx / groupsize;
                float s_val = __half2float(scales[(long long)n_idx * n_groups + gi]);
                float z_val = sym ? 0.0f : __half2float(zeros[(long long)n_idx * n_groups + gi]);
                float q = static_cast<float>(qweight_i8[(long long)n_idx * K + k_idx]);
                B_smem[n_local][k_local] = (q - z_val) * s_val;
            } else {
                B_smem[n_local][k_local] = 0.0f;
            }
        }

        __syncthreads();

        // Compute C_reg += A @ B^T
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a_vals[TM];
            float b_vals[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_vals[i] = A_smem[thread_row * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_vals[j] = B_smem[thread_col * TN + j][kk];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    C_reg[i][j] += a_vals[i] * b_vals[j];
        }

        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int b_idx = bm_start + thread_row * TM + i;
        if (b_idx >= B) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int n_idx = bn_start + thread_col * TN + j;
            if (n_idx >= N) continue;
            float val = C_reg[i][j];
            if (lora_valid) {
                val += lora_out[(long long)b_idx * N + n_idx];
            }
            y[(long long)b_idx * N + n_idx] = val;
        }
    }
}


// -------------------------------------------------------------------------
// Host-side: pybind11 wrapper + dispatch
// -------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor gptq_dequant_matmul_cuda(
    torch::Tensor x,              // (B, K) fp32
    torch::Tensor qweight_i8,     // (N, K) int8
    torch::Tensor scales,         // (N, n_groups) fp16
    torch::Tensor zeros,          // (N, n_groups) fp16
    int groupsize,
    bool sym,
    torch::Tensor lora_out)       // (B, N) fp32 or empty
{
    CHECK_CUDA(x);
    CHECK_CUDA(qweight_i8);
    CHECK_CUDA(scales);
    CHECK_CUDA(zeros);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(qweight_i8);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zeros);

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(qweight_i8.dtype() == torch::kInt8, "qweight_i8 must be int8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be fp16");

    const int B = x.size(0);
    const int K = x.size(1);
    const int N = qweight_i8.size(0);
    const int n_groups = scales.size(1);

    TORCH_CHECK(qweight_i8.size(1) == K,
                "qweight_i8 shape mismatch: expected (N, K)");
    TORCH_CHECK(scales.size(0) == N, "scales row count mismatch");
    TORCH_CHECK(zeros.size(0) == N, "zeros row count mismatch");
    TORCH_CHECK(zeros.size(1) == n_groups, "zeros col count mismatch");

    int lora_valid = 0;
    const float* lora_ptr = nullptr;
    if (lora_out.numel() > 0) {
        CHECK_CUDA(lora_out);
        CHECK_CONTIGUOUS(lora_out);
        TORCH_CHECK(lora_out.dtype() == torch::kFloat32, "lora_out must be fp32");
        TORCH_CHECK(lora_out.size(0) == B && lora_out.size(1) == N,
                     "lora_out shape mismatch");
        lora_valid = 1;
        lora_ptr = lora_out.data_ptr<float>();
    }

    auto y = torch::zeros({B, N}, x.options());  // (B, N) fp32

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int sym_int = sym ? 1 : 0;

    if (B <= 4) {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
        gptq_dequant_gemv_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            qweight_i8.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
            lora_ptr,
            y.data_ptr<float>(),
            N, K,
            groupsize,
            n_groups,
            sym_int,
            lora_valid);
    } else {
        dim3 block(GEMM_THREADS);
        dim3 grid((B + BM - 1) / BM, (N + BN - 1) / BN);
        gptq_dequant_gemm_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            qweight_i8.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
            lora_ptr,
            y.data_ptr<float>(),
            B, N, K,
            groupsize,
            n_groups,
            sym_int,
            lora_valid);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gptq_dequant_matmul", &gptq_dequant_matmul_cuda,
          "GPTQ fused dequant + matmul (CUDA)",
          py::arg("x"),
          py::arg("qweight_i8"),
          py::arg("scales"),
          py::arg("zeros"),
          py::arg("groupsize"),
          py::arg("sym"),
          py::arg("lora_out"));
}
