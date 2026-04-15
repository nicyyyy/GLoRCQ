/*
 * turbo_matmul.cu — Fused dequant + matmul CUDA kernels for TurboQuant 2-bit
 *                   + pybind11 wrapper (single-file build).
 *
 * Two kernels:
 *   1. GEMV  (B <= 4):  warp-per-row, memory-bound decode path
 *   2. GEMM  (B > 4):   tiled matmul with on-the-fly dequant for prefill
 *
 * Weight layout:
 *   packed_indices: (N, K/4) uint8  — 4 x 2-bit indices per byte
 *   norms:          (N,)    fp32    — per-row L2 norms
 *   centroids:      (4,)    fp32    — codebook (shared across all rows)
 *
 * Computation:
 *   y[b, n] = norms[n] * Σ_k centroid[unpack(packed[n,k/4], k%4)] * x_rot[b, k]
 *   If lora_out is provided: y[b, n] += lora_out[b, n]
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// -------------------------------------------------------------------------
// Device helpers
// -------------------------------------------------------------------------

__device__ __forceinline__ float centroid_lookup(
    float c0, float c1, float c2, float c3, int idx)
{
    return (idx & 2) ? ((idx & 1) ? c3 : c2) : ((idx & 1) ? c1 : c0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// -------------------------------------------------------------------------
// Kernel 1: GEMV — one warp per output row, for B <= 4
// -------------------------------------------------------------------------
static constexpr int WARPS_PER_BLOCK = 4;

__global__ void turbo_dequant_gemv_2bit_kernel(
    const float* __restrict__ x_rot,          // (B, K)
    const uint8_t* __restrict__ packed,        // (N, K/4) uint8
    const float* __restrict__ norms,           // (N,)
    float c0, float c1, float c2, float c3,
    const float* __restrict__ lora_out,        // (B, N) or nullptr
    float* __restrict__ y,                     // (B, N)
    int N, int K,
    int lora_valid)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (n >= N) return;

    const int b = blockIdx.y;

    const int packed_cols = K / 4;
    const uint8_t* packed_row = packed + (long long)n * packed_cols;
    const float* x_row = x_rot + (long long)b * K;

    float acc = 0.0f;

    // Process in chunks of 32*4 = 128 bytes = 512 indices at a time.
    // Each lane reads 4 consecutive bytes (uint32) per iteration — coalesced.
    const int total_uint32s = packed_cols / 4;  // (K/4) / 4 = K/16
    const int iters = total_uint32s / 32;

    for (int it = 0; it < iters; it++) {
        int byte_offset = (it * 32 + lane) * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(packed_row + byte_offset);

        int elem_base = byte_offset * 4;  // each byte = 4 elements

        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            uint8_t byte_val = (packed4 >> (bi * 8)) & 0xFF;
            #pragma unroll
            for (int pi = 0; pi < 4; pi++) {
                int idx = (byte_val >> (pi * 2)) & 3;
                int k = elem_base + bi * 4 + pi;
                acc += centroid_lookup(c0, c1, c2, c3, idx) * x_row[k];
            }
        }
    }

    // Handle remaining elements (K not divisible by 512)
    int processed = iters * 512;
    // Distribute remainder across lanes
    for (int k = processed + lane; k < K; k += 32) {
        int byte_idx = k / 4;
        int pos = k % 4;
        int idx = (packed_row[byte_idx] >> (pos * 2)) & 3;
        acc += centroid_lookup(c0, c1, c2, c3, idx) * x_row[k];
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        acc *= norms[n];
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

__global__ void turbo_dequant_gemm_2bit_kernel(
    const float* __restrict__ x_rot,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ norms,
    float c0, float c1, float c2, float c3,
    const float* __restrict__ lora_out,
    float* __restrict__ y,
    int B, int N, int K,
    int lora_valid)
{
    const int bm_start = blockIdx.x * BM;
    const int bn_start = blockIdx.y * BN;

    const int thread_row = threadIdx.x / (BN / TN);  // 0..15
    const int thread_col = threadIdx.x % (BN / TN);  // 0..15

    const int packed_cols = K / 4;

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

        // Load A tile
        for (int idx = threadIdx.x; idx < BM * BK; idx += GEMM_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            int b_idx = bm_start + row;
            int k_idx = k_start + col;
            if (b_idx < B && k_idx < K)
                A_smem[row][col] = x_rot[(long long)b_idx * K + k_idx];
            else
                A_smem[row][col] = 0.0f;
        }

        // Load B tile (dequant on the fly)
        for (int idx = threadIdx.x; idx < BN * BK; idx += GEMM_THREADS) {
            int n_local = idx / BK;
            int k_local = idx % BK;
            int n_idx = bn_start + n_local;
            int k_idx = k_start + k_local;
            if (n_idx < N && k_idx < K) {
                int byte_idx = k_idx / 4;
                int pos = k_idx % 4;
                uint8_t byte_val = packed[(long long)n_idx * packed_cols + byte_idx];
                int cidx = (byte_val >> (pos * 2)) & 3;
                B_smem[n_local][k_local] = centroid_lookup(c0, c1, c2, c3, cidx) * norms[n_idx];
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

torch::Tensor turbo_dequant_matmul_cuda(
    torch::Tensor x_rot,            // (B, K) fp32
    torch::Tensor packed_indices,    // (N, K/4) uint8
    torch::Tensor norms,             // (N,) fp32
    torch::Tensor centroids,         // (4,) fp32
    torch::Tensor lora_out)          // (B, N) fp32 or empty
{
    CHECK_CUDA(x_rot);
    CHECK_CUDA(packed_indices);
    CHECK_CUDA(norms);
    CHECK_CUDA(centroids);
    CHECK_CONTIGUOUS(x_rot);
    CHECK_CONTIGUOUS(packed_indices);
    CHECK_CONTIGUOUS(norms);
    CHECK_CONTIGUOUS(centroids);

    TORCH_CHECK(x_rot.dtype() == torch::kFloat32, "x_rot must be fp32");
    TORCH_CHECK(packed_indices.dtype() == torch::kUInt8, "packed_indices must be uint8");
    TORCH_CHECK(norms.dtype() == torch::kFloat32, "norms must be fp32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be fp32");
    TORCH_CHECK(centroids.numel() == 4, "centroids must have 4 elements");

    const int B = x_rot.size(0);
    const int K = x_rot.size(1);
    const int N = packed_indices.size(0);

    TORCH_CHECK(packed_indices.size(1) == K / 4,
                "packed_indices shape mismatch: expected (N, K/4)");
    TORCH_CHECK(norms.size(0) == N, "norms shape mismatch");
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for 2-bit packing");

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

    auto y = torch::zeros({B, N}, x_rot.options());

    // Copy centroids to host (only 4 floats, negligible overhead)
    float c0, c1, c2, c3;
    cudaMemcpy(&c0, centroids.data_ptr<float>() + 0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c1, centroids.data_ptr<float>() + 1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c2, centroids.data_ptr<float>() + 2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c3, centroids.data_ptr<float>() + 3, sizeof(float), cudaMemcpyDeviceToHost);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (B <= 4) {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
        turbo_dequant_gemv_2bit_kernel<<<grid, block, 0, stream>>>(
            x_rot.data_ptr<float>(),
            packed_indices.data_ptr<uint8_t>(),
            norms.data_ptr<float>(),
            c0, c1, c2, c3,
            lora_ptr,
            y.data_ptr<float>(),
            N, K, lora_valid);
    } else {
        dim3 block(GEMM_THREADS);
        dim3 grid((B + BM - 1) / BM, (N + BN - 1) / BN);
        turbo_dequant_gemm_2bit_kernel<<<grid, block, 0, stream>>>(
            x_rot.data_ptr<float>(),
            packed_indices.data_ptr<uint8_t>(),
            norms.data_ptr<float>(),
            c0, c1, c2, c3,
            lora_ptr,
            y.data_ptr<float>(),
            B, N, K, lora_valid);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turbo_dequant_matmul", &turbo_dequant_matmul_cuda,
          "TurboQuant 2-bit fused dequant + matmul (CUDA)",
          py::arg("x_rot"),
          py::arg("packed_indices"),
          py::arg("norms"),
          py::arg("centroids"),
          py::arg("lora_out"));
}
