/*
 * vq4_matmul.cu — Fused dequant + matmul CUDA kernels for GLoRCQ VQ4 (vdim=4, K=256)
 *                 + pybind11 wrapper.
 *
 * Two kernels:
 *   1. GEMV       (B <= 4):    warp-per-row, memory-bound decode path
 *   2. GEMM       (B > 4):     tiled matmul with on-the-fly dequant for prefill
 *   3. Grouped-GEMV (MoE):     E experts with different inputs, batched
 *
 * Weight layout:
 *   codes:     (N, K/vdim) uint8    — 1 x 8-bit centroid index per vdim-vector
 *   centroids: (n_cb, K_cb, vdim) fp16 — per-codebook lookup tables (K_cb=256, vdim=4)
 *              codes[n, k] indexes centroids[k / codes_per_cb, :, :]
 *
 * Computation:
 *   y[b, n] = Σ_k Σ_d centroids[cb_id(k), codes[n, k], d] * x_rot[b, k*vdim + d]
 *
 * codes_per_cb = (K/vdim) / n_cb   (how many code positions share one codebook)
 * For gate/up (in_d=2048, n_cb=1): codes_per_cb = 512.
 * For down (in_d=1408, n_cb=11):    codes_per_cb = 32.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static constexpr int VDIM = 4;
static constexpr int K_CB = 256;    // codebook size (2^(vdim*bits) = 2^8)
static constexpr int WARPS_PER_BLOCK = 4;    // slim-block: matches turbo, 16 blocks/SM on A100

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
//
// Shared memory layout:
//   x_smem[K]              — cooperatively loaded x_rot[b, :] as fp32
//   cb_smem[n_cb, K_CB*VDIM] — all codebooks loaded once per block
// -------------------------------------------------------------------------

__global__ void vq4_dequant_gemv_kernel(
    const __half* __restrict__ x_rot,          // (B, K) fp16
    const uint8_t* __restrict__ codes,         // (N, K/VDIM) uint8
    const __half* __restrict__ centroids,      // (n_cb, K_CB, VDIM) fp16
    __half* __restrict__ y,                    // (B, N) fp16
    int N, int K, int n_cb, int codes_per_cb)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (n >= N) return;
    const int b = blockIdx.y;

    const int codes_per_row = K / VDIM;    // = n_cb * codes_per_cb
    const uint8_t* codes_row = codes + (long long)n * codes_per_row;

    // Shared memory: x_smem (K fp32) + cb_smem (n_cb * K_CB * VDIM fp32)
    extern __shared__ float smem[];
    float* x_smem = smem;
    float* cb_smem = smem + K;

    // Cooperative load: x_rot[b, :]
    const __half* x_row = x_rot + (long long)b * K;
    for (int i = threadIdx.x; i < K; i += WARPS_PER_BLOCK * 32) {
        x_smem[i] = __half2float(x_row[i]);
    }
    // Cooperative load: centroids (n_cb * K_CB * VDIM)
    const int cb_total = n_cb * K_CB * VDIM;
    for (int i = threadIdx.x; i < cb_total; i += WARPS_PER_BLOCK * 32) {
        cb_smem[i] = __half2float(centroids[i]);
    }
    __syncthreads();

    float acc = 0.0f;
    // Each lane strides over the codes_per_row positions
    for (int k = lane; k < codes_per_row; k += 32) {
        int cb_id = k / codes_per_cb;
        int code  = (int)codes_row[k];
        const float* cb_ptr = cb_smem + (cb_id * K_CB + code) * VDIM;
        #pragma unroll
        for (int d = 0; d < VDIM; d++) {
            acc += cb_ptr[d] * x_smem[k * VDIM + d];
        }
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        y[(long long)b * N + n] = __float2half(acc);
    }
}

// -------------------------------------------------------------------------
// Kernel 2: GEMM — tiled matmul with on-the-fly dequant for B > 4
//
// Layout mirrors turbo_dequant_gemm_2bit_kernel: tile-based, dequantize into
// shared memory then accumulate. Simplified: one thread per output element,
// since VQ4 dequant is heavier than TurboQuant.
// -------------------------------------------------------------------------

static constexpr int GEMM_BM = 8;
static constexpr int GEMM_BN = 32;
static constexpr int GEMM_BK = 64;
static constexpr int GEMM_THREADS = 256;

__global__ void vq4_dequant_gemm_kernel(
    const __half* __restrict__ x_rot,          // (B, K) fp16
    const uint8_t* __restrict__ codes,         // (N, K/VDIM) uint8
    const __half* __restrict__ centroids,      // (n_cb, K_CB, VDIM) fp16
    __half* __restrict__ y,                    // (B, N) fp16
    int B, int N, int K, int n_cb, int codes_per_cb)
{
    const int by = blockIdx.x * GEMM_BM;
    const int bx = blockIdx.y * GEMM_BN;
    const int tid = threadIdx.x;

    // Shared memory: (n_cb, K_CB, VDIM) codebook cache
    extern __shared__ float cb_smem[];
    const int cb_total = n_cb * K_CB * VDIM;
    for (int i = tid; i < cb_total; i += GEMM_THREADS) {
        cb_smem[i] = __half2float(centroids[i]);
    }
    __syncthreads();

    const int codes_per_row = K / VDIM;

    // Each thread iterates over tile cells (BM*BN elements distributed round-robin
    // across GEMM_THREADS threads).
    const int tile_size = GEMM_BM * GEMM_BN;
    for (int cell = tid; cell < tile_size; cell += GEMM_THREADS) {
        int local_b = cell / GEMM_BN;
        int local_n = cell % GEMM_BN;
        int b = by + local_b;
        int n = bx + local_n;
        if (b >= B || n >= N) continue;

        const uint8_t* codes_row = codes + (long long)n * codes_per_row;
        const __half* x_row = x_rot + (long long)b * K;

        float acc = 0.0f;
        for (int k = 0; k < codes_per_row; k++) {
            int cb_id = k / codes_per_cb;
            int code  = (int)codes_row[k];
            const float* cb_ptr = cb_smem + (cb_id * K_CB + code) * VDIM;
            #pragma unroll
            for (int d = 0; d < VDIM; d++) {
                acc += cb_ptr[d] * __half2float(x_row[k * VDIM + d]);
            }
        }
        y[(long long)b * N + n] = __float2half(acc);
    }
}

// -------------------------------------------------------------------------
// Kernel 3: Grouped-GEMV — E experts with different inputs
//
// x_grouped[e, :]                       — one row per expert
// codes_cat[e*N + n, :]                 — concatenated codes for E experts
// centroids_cat[e*n_cb + cb, :, :]      — concatenated codebooks
// y_cat[e*N + n]                        — output
// -------------------------------------------------------------------------

// Streamed-codebook variant for down_proj (n_cb>1): loads only ONE codebook
// at a time into shmem (~2KB vs ~22KB for n_cb=11), boosting occupancy.
// Each lane processes codes_per_cb sequential codes (all sharing same cb_id).
__global__ __launch_bounds__(128, 8)
void vq4_dequant_grouped_gemv_kernel_streamed(
    const __half* __restrict__ x_grouped,      // (E, K) fp16
    const uint8_t* __restrict__ codes_cat,     // (E*N, K/VDIM) uint8
    const __half* __restrict__ centroids_cat,  // (E*n_cb, K_CB, VDIM) fp16
    __half* __restrict__ y_cat,                // (E*N,) fp16
    int E, int N, int K, int n_cb, int codes_per_cb,
    int blocks_per_expert)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int expert_idx  = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_in_expert = local_block * WARPS_PER_BLOCK + warp_id_in_block;
    if (expert_idx >= E) return;

    const int codes_per_row = K / VDIM;

    // Shmem: x_smem[K] + cb_smem[K_CB * VDIM] (one codebook only)
    extern __shared__ __half smem_h[];
    __half* x_smem  = smem_h;
    __half* cb_smem = smem_h + K;

    // Load x_grouped[expert_idx, :] as half2
    const __half2* x_src = reinterpret_cast<const __half2*>(x_grouped) + (long long)expert_idx * (K/2);
    __half2* x_dst = reinterpret_cast<__half2*>(x_smem);
    for (int i = threadIdx.x; i < K/2; i += 128) {
        x_dst[i] = x_src[i];
    }

    if (row_in_expert >= N) {
        // Still need to help load x_smem then early exit (must sync first)
        __syncthreads();
        return;
    }

    const int global_row = expert_idx * N + row_in_expert;
    const uint8_t* codes_row = codes_cat + (long long)global_row * codes_per_row;

    __half2 acc2 = __float2half2_rn(0.f);
    const __half2* x_h2 = reinterpret_cast<const __half2*>(x_smem);
    const __half2* cb_h2 = reinterpret_cast<const __half2*>(cb_smem);
    const int cb_size_h2 = K_CB * VDIM / 2;   // 512 half2's per codebook

    // Load initial codebook + sync x_smem
    __syncthreads();

    // For each codebook, stream-load and process its code range
    for (int cb_id = 0; cb_id < n_cb; ++cb_id) {
        // Load codebook cb_id for this expert
        const __half2* cb_src = reinterpret_cast<const __half2*>(centroids_cat)
                              + (long long)(expert_idx * n_cb + cb_id) * cb_size_h2;
        __half2* cb_dst = reinterpret_cast<__half2*>(cb_smem);
        for (int i = threadIdx.x; i < cb_size_h2; i += 128) {
            cb_dst[i] = cb_src[i];
        }
        __syncthreads();

        // Process codes for this codebook: k in [cb_id*codes_per_cb, (cb_id+1)*codes_per_cb)
        const int k_start = cb_id * codes_per_cb;
        const int k_end   = k_start + codes_per_cb;
        for (int k = k_start + lane; k < k_end; k += 32) {
            int code = (int)codes_row[k];
            int base = code * 2;
            __half2 c01 = cb_h2[base + 0];
            __half2 c23 = cb_h2[base + 1];
            __half2 x01 = x_h2[k * 2 + 0];
            __half2 x23 = x_h2[k * 2 + 1];
            acc2 = __hfma2(c01, x01, acc2);
            acc2 = __hfma2(c23, x23, acc2);
        }
        __syncthreads();   // barrier before overwriting cb_smem
    }

    // Reduce
    float acc = __half2float(__hadd(__low2half(acc2), __high2half(acc2)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) {
        y_cat[global_row] = __float2half(acc);
    }
}

// Slim-block VQ4 grouped GEMV with fp16 shared memory + vectorized code loads.
// - WARPS_PER_BLOCK=4 (128 threads) → 16 blocks/SM on A100 (matches turbo).
// - fp16 x_smem + fp16 cb_smem (halved shmem footprint).
// - Packed uint32 code loads (4 codes/load) via __ldg.
// - Half-precision FMA via __hfma2 along VDIM=4 (=2 half2 pairs).
// - fp16 accumulate with fp32 warp-shuffle reduce.
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 8)
void vq4_dequant_grouped_gemv_kernel(
    const __half* __restrict__ x_grouped,      // (E, K) fp16
    const uint8_t* __restrict__ codes_cat,     // (E*N, K/VDIM) uint8
    const __half* __restrict__ centroids_cat,  // (E*n_cb, K_CB, VDIM) fp16
    __half* __restrict__ y_cat,                // (E*N,) fp16
    int E, int N, int K, int n_cb, int codes_per_cb,
    int blocks_per_expert)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_idx  = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_in_expert = local_block * WARPS_PER_BLOCK + warp_id_in_block;
    if (expert_idx >= E) return;

    const int codes_per_row = K / VDIM;   // 512 for gate/up in_d=2048; 352 for down in_d=1408

    // -- fp16 shared memory: x_smem[K] + cb_smem[n_cb*K_CB*VDIM] --
    // gate/up: K=2048 (4KB) + cb 2KB = 6KB.  down: K=1408 (2.8KB) + cb 22KB = 25KB.
    extern __shared__ __half smem_h[];
    __half* x_smem  = smem_h;
    __half* cb_smem = smem_h + K;

    // Vectorized fp16 loads via half2
    const int cb_total = n_cb * K_CB * VDIM;

    // Load x_grouped[expert_idx, :] as half2 (K/2 half2 elements)
    const __half2* x_src = reinterpret_cast<const __half2*>(x_grouped) + (long long)expert_idx * (K/2);
    __half2* x_dst = reinterpret_cast<__half2*>(x_smem);
    for (int i = threadIdx.x; i < K/2; i += WARPS_PER_BLOCK * 32) {
        x_dst[i] = x_src[i];
    }
    // Load codebooks (fp16) as half2
    const __half2* cb_src = reinterpret_cast<const __half2*>(centroids_cat)
                          + (long long)expert_idx * (cb_total / 2);
    __half2* cb_dst = reinterpret_cast<__half2*>(cb_smem);
    for (int i = threadIdx.x; i < cb_total/2; i += WARPS_PER_BLOCK * 32) {
        cb_dst[i] = cb_src[i];
    }
    __syncthreads();

    if (row_in_expert >= N) return;

    const int global_row = expert_idx * N + row_in_expert;
    const uint8_t* codes_row = codes_cat + (long long)global_row * codes_per_row;

    // Half-precision half2 accumulator
    __half2 acc2 = __float2half2_rn(0.f);

    const __half2* cb_h2 = reinterpret_cast<const __half2*>(cb_smem);
    const __half2* x_h2  = reinterpret_cast<const __half2*>(x_smem);

    if (n_cb == 1) {
        // Fast path (gate/up): single codebook, packed uint32 code loads.
        const uint32_t* codes_row_u32 = reinterpret_cast<const uint32_t*>(codes_row);
        const int n_u32 = codes_per_row >> 2;   // 128 for codes_per_row=512

        for (int k4 = lane; k4 < n_u32; k4 += 32) {
            uint32_t cw = __ldg(codes_row_u32 + k4);   // 4 codes packed as uint32
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int code = (cw >> (j * 8)) & 0xFF;
                int k = (k4 << 2) + j;
                // Each centroid entry = VDIM=4 halves = 2 half2's, contiguous
                __half2 c01 = cb_h2[code * 2 + 0];
                __half2 c23 = cb_h2[code * 2 + 1];
                __half2 x01 = x_h2[k * 2 + 0];
                __half2 x23 = x_h2[k * 2 + 1];
                acc2 = __hfma2(c01, x01, acc2);
                acc2 = __hfma2(c23, x23, acc2);
            }
        }
    } else {
        // General path (down-proj, n_cb=11): scalar code loads with cb_id lookup.
        for (int k = lane; k < codes_per_row; k += 32) {
            int cb_id = k / codes_per_cb;
            int code  = (int)codes_row[k];
            int base  = (cb_id * K_CB + code) * 2;   // 2 half2's per entry
            __half2 c01 = cb_h2[base + 0];
            __half2 c23 = cb_h2[base + 1];
            __half2 x01 = x_h2[k * 2 + 0];
            __half2 x23 = x_h2[k * 2 + 1];
            acc2 = __hfma2(c01, x01, acc2);
            acc2 = __hfma2(c23, x23, acc2);
        }
    }

    // Horizontal add of half2 → float, then warp reduce
    float acc = __half2float(__hadd(__low2half(acc2), __high2half(acc2)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) {
        y_cat[global_row] = __float2half(acc);
    }
}

// -------------------------------------------------------------------------
// Kernel 4: Fused gather + VQ dequant + GEMV
//
// Combines two ops into a single kernel:
//   1. Gather: x_gathered[j] = x[b, sigma[j]]
//      (For inputs where only a partial-Walsh is needed after gather, the
//       Walsh rotation of the last `ws` cols must be done separately BEFORE
//       calling this kernel.)
//   2. VQ dequant + matmul: y[b,n] = Σ_k centroids[cb(k), codes[n,k], d] * x_gathered[k*vdim+d]
//
// This avoids materializing x_gathered in HBM — it lives in shared memory only.
//
// Shared memory layout:
//   x_smem[K]                        — float, gathered input
//   cb_smem[n_cb * K_CB * VDIM]      — float, codebooks
// -------------------------------------------------------------------------

__global__ void vq4_fused_gather_gemv_kernel(
    const __half* __restrict__ x_walshed,     // (B, K) fp16 — already Walsh-rotated
    const int32_t* __restrict__ sigma_inv,    // (K,) int32 — inverse permutation
    const uint8_t* __restrict__ codes,        // (N, K/VDIM) uint8
    const __half* __restrict__ centroids,     // (n_cb, K_CB, VDIM) fp16
    __half* __restrict__ y,                   // (B, N) fp16
    int N, int K, int n_cb, int codes_per_cb)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    const int b = blockIdx.y;

    // Shared memory
    extern __shared__ float smem[];
    float* x_smem = smem;
    float* cb_smem = smem + K;

    // Gather via sigma_inv: x_smem[j] = x_walshed[b, sigma_inv[j]]
    // Note: caller passes sigma_inv so this becomes a simple index read.
    // Actually simpler: x_smem[sigma[j]] = x_walshed[b, j] (scatter form)
    // But scatter requires atomics; gather via sigma_inv is safer.
    const __half* x_row = x_walshed + (long long)b * K;
    for (int j = threadIdx.x; j < K; j += WARPS_PER_BLOCK * 32) {
        int src = sigma_inv[j];
        x_smem[j] = __half2float(x_row[src]);
    }
    // Load centroids
    const int cb_total = n_cb * K_CB * VDIM;
    for (int i = threadIdx.x; i < cb_total; i += WARPS_PER_BLOCK * 32) {
        cb_smem[i] = __half2float(centroids[i]);
    }
    __syncthreads();

    if (n >= N) return;

    const int codes_per_row = K / VDIM;
    const uint8_t* codes_row = codes + (long long)n * codes_per_row;

    float acc = 0.0f;
    for (int k = lane; k < codes_per_row; k += 32) {
        int cb_id = k / codes_per_cb;
        int code  = (int)codes_row[k];
        const float* cb_ptr = cb_smem + (cb_id * K_CB + code) * VDIM;
        #pragma unroll
        for (int d = 0; d < VDIM; d++) {
            acc += cb_ptr[d] * x_smem[k * VDIM + d];
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[(long long)b * N + n] = __float2half(acc);
    }
}

// -------------------------------------------------------------------------
// Host-side dispatchers
// -------------------------------------------------------------------------

at::Tensor vq4_fused_gather_gemv_cuda(
    at::Tensor x_walshed,     // (B, K) fp16 — Walsh-rotated input (pre-computed)
    at::Tensor sigma_inv,     // (K,) int32
    at::Tensor codes,         // (N, K/VDIM) uint8
    at::Tensor centroids,     // (n_cb, K_CB, VDIM) fp16
    int64_t n_cb,
    int64_t codes_per_cb)
{
    TORCH_CHECK(x_walshed.is_cuda(), "x_walshed must be CUDA");
    TORCH_CHECK(x_walshed.dtype() == torch::kHalf, "x_walshed must be fp16");
    TORCH_CHECK(sigma_inv.dtype() == torch::kInt32, "sigma_inv must be int32");
    TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(centroids.dtype() == torch::kHalf, "centroids must be fp16");

    x_walshed = x_walshed.contiguous();
    sigma_inv = sigma_inv.contiguous();
    codes = codes.contiguous();
    centroids = centroids.contiguous();

    const int B = x_walshed.size(0);
    const int K = x_walshed.size(1);
    const int N = codes.size(0);

    auto y = torch::empty({B, N}, x_walshed.options());

    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t smem = (K + n_cb * K_CB * VDIM) * sizeof(float);
    cudaFuncSetAttribute(
        vq4_fused_gather_gemv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vq4_fused_gather_gemv_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(x_walshed.data_ptr<at::Half>()),
        sigma_inv.data_ptr<int32_t>(),
        codes.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, (int)n_cb, (int)codes_per_cb);
    return y;
}

at::Tensor vq4_dequant_matmul_cuda(
    at::Tensor x_rot,          // (B, K) fp16
    at::Tensor codes,          // (N, K/VDIM) uint8
    at::Tensor centroids,      // (n_cb, K_CB, VDIM) fp16
    int64_t n_cb,
    int64_t codes_per_cb)
{
    TORCH_CHECK(x_rot.is_cuda(), "x_rot must be CUDA");
    TORCH_CHECK(x_rot.dtype() == torch::kHalf, "x_rot must be fp16");
    TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(centroids.dtype() == torch::kHalf, "centroids must be fp16");

    x_rot = x_rot.contiguous();
    codes = codes.contiguous();
    centroids = centroids.contiguous();

    const int B = x_rot.size(0);
    const int K = x_rot.size(1);
    const int N = codes.size(0);

    auto y = torch::empty({B, N}, x_rot.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (B <= 4) {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
        size_t smem = (K + n_cb * K_CB * VDIM) * sizeof(float);
        // Opt in to dynamic shared memory > 48 KB (needed when n_cb ≥ 6 with K=1408).
        cudaFuncSetAttribute(
            vq4_dequant_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        vq4_dequant_gemv_kernel<<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_rot.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K, (int)n_cb, (int)codes_per_cb);
    } else {
        dim3 block(GEMM_THREADS);
        dim3 grid((B + GEMM_BM - 1) / GEMM_BM, (N + GEMM_BN - 1) / GEMM_BN);
        size_t smem = n_cb * K_CB * VDIM * sizeof(float);
        cudaFuncSetAttribute(
            vq4_dequant_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        vq4_dequant_gemm_kernel<<<grid, block, smem, stream>>>(
            reinterpret_cast<const __half*>(x_rot.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            B, N, K, (int)n_cb, (int)codes_per_cb);
    }
    return y;
}

at::Tensor vq4_dequant_grouped_gemv_cuda(
    at::Tensor x_grouped,     // (E, K) fp16
    at::Tensor codes_cat,     // (E*N, K/VDIM) uint8
    at::Tensor centroids_cat, // (E*n_cb, K_CB, VDIM) fp16
    int64_t E,
    int64_t N,
    int64_t n_cb,
    int64_t codes_per_cb)
{
    TORCH_CHECK(x_grouped.is_cuda(), "x_grouped must be CUDA");
    TORCH_CHECK(x_grouped.dtype() == torch::kHalf, "x_grouped must be fp16");
    TORCH_CHECK(codes_cat.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(centroids_cat.dtype() == torch::kHalf, "centroids must be fp16");

    x_grouped = x_grouped.contiguous();
    codes_cat = codes_cat.contiguous();
    centroids_cat = centroids_cat.contiguous();

    const int K = x_grouped.size(1);

    auto y_cat = torch::empty({E * N}, x_grouped.options());

    int blocks_per_expert = ((int)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((int)E * blocks_per_expert);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t smem = (K + n_cb * K_CB * VDIM) * sizeof(__half);
    cudaFuncSetAttribute(
        vq4_dequant_grouped_gemv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    vq4_dequant_grouped_gemv_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(x_grouped.data_ptr<at::Half>()),
        codes_cat.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(centroids_cat.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y_cat.data_ptr<at::Half>()),
        (int)E, (int)N, K, (int)n_cb, (int)codes_per_cb, blocks_per_expert);

    return y_cat;
}

// -------------------------------------------------------------------------
// Kernel 3b: EXPERT-INDEXED grouped VQ dequant + GEMV (DeepSeek top-k gather).
// Identical math to vq4_dequant_grouped_gemv_kernel, but codes/centroids are
// read from the FULL (E_full-expert) tensors at row sel[g] instead of g. This
// gathers the top-k expert weights INSIDE the GEMV (no explicit index_select
// copy of codes/centroids). x_grouped and y are indexed by the local slot g;
// only codes/centroids use sel.
// -------------------------------------------------------------------------
__global__ void vq4_dequant_grouped_gemv_indexed_kernel(
    const __half* __restrict__ x_grouped,      // (G, K) fp16 — per-slot input
    const uint8_t* __restrict__ codes_all,     // (E_full*N, K/VDIM) uint8
    const __half* __restrict__ centroids_all,  // (E_full*n_cb, K_CB, VDIM) fp16
    const int32_t* __restrict__ sel,           // (G,) int32 — expert index per slot
    __half* __restrict__ y_cat,                // (G*N,) fp16
    int G, int N, int K, int n_cb, int codes_per_cb,
    int blocks_per_expert, int reorder_ok)
{
    const int warp_id_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int slot_idx    = blockIdx.x / blocks_per_expert;   // 0..G-1 (local)
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row_in_expert = local_block * WARPS_PER_BLOCK + warp_id_in_block;
    if (slot_idx >= G) return;
    const int src_expert = sel[slot_idx];                     // real expert row

    const int codes_per_row = K / VDIM;

    extern __shared__ __half smem_h[];
    __half* x_smem  = smem_h;
    __half* cb_smem = smem_h + K;
    const int cb_total = n_cb * K_CB * VDIM;

    const __half2* x_src = reinterpret_cast<const __half2*>(x_grouped) + (long long)slot_idx * (K/2);
    __half2* x_dst = reinterpret_cast<__half2*>(x_smem);
    for (int i = threadIdx.x; i < K/2; i += WARPS_PER_BLOCK * 32) x_dst[i] = x_src[i];
    const __half2* cb_src = reinterpret_cast<const __half2*>(centroids_all)
                          + (long long)src_expert * (cb_total / 2);
    __half2* cb_dst = reinterpret_cast<__half2*>(cb_smem);
    for (int i = threadIdx.x; i < cb_total/2; i += WARPS_PER_BLOCK * 32) cb_dst[i] = cb_src[i];
    __syncthreads();

    if (row_in_expert >= N) return;

    const int src_row = src_expert * N + row_in_expert;
    const int out_row = slot_idx  * N + row_in_expert;
    const uint8_t* codes_row = codes_all + (long long)src_row * codes_per_row;

    __half2 acc2 = __float2half2_rn(0.f);
    const __half2* cb_h2 = reinterpret_cast<const __half2*>(cb_smem);
    const __half2* x_h2  = reinterpret_cast<const __half2*>(x_smem);

    if (n_cb == 1) {
        const uint32_t* codes_row_u32 = reinterpret_cast<const uint32_t*>(codes_row);
        const int n_u32 = codes_per_row >> 2;
        for (int k4 = lane; k4 < n_u32; k4 += 32) {
            uint32_t cw = __ldg(codes_row_u32 + k4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int code = (cw >> (j * 8)) & 0xFF;
                int k = (k4 << 2) + j;
                __half2 c01 = cb_h2[code * 2 + 0];
                __half2 c23 = cb_h2[code * 2 + 1];
                __half2 x01 = x_h2[k * 2 + 0];
                __half2 x23 = x_h2[k * 2 + 1];
                acc2 = __hfma2(c01, x01, acc2);
                acc2 = __hfma2(c23, x23, acc2);
            }
        }
    } else if (reorder_ok && ((codes_per_row | codes_per_cb) & 3) == 0) {
        // ILP variant (opt-in via reorder_ok): uint32 code loads (4/load), ONE
        // cb_id divide per aligned-4 group, 4 independent accumulators to break
        // the fma dependency chain and hide the data-dependent codebook-gather
        // latency (the profiled bound; +5.5% e2e on Mixtral decode). REORDERS
        // fp16 accumulation (per-lane partition + acc tree) -> NOT byte-identical
        // with the scalar branch; callers wanting bit-exactness (DeepSeek) pass
        // reorder_ok=0 and take the unchanged scalar loop below.
        const uint32_t* codes_u32 = reinterpret_cast<const uint32_t*>(codes_row);
        const int n_u32 = codes_per_row >> 2;
        __half2 acc2a = __float2half2_rn(0.f), acc2b = __float2half2_rn(0.f);
        __half2 acc2c = __float2half2_rn(0.f), acc2d = __float2half2_rn(0.f);
        for (int k4 = lane; k4 < n_u32; k4 += 32) {
            uint32_t cw   = __ldg(codes_u32 + k4);
            const int k0  = k4 << 2;
            const int cboff = (k0 / codes_per_cb) * K_CB;
            int b0 = (cboff + ( cw        & 0xFF)) * 2;
            int b1 = (cboff + ((cw >> 8)  & 0xFF)) * 2;
            int b2 = (cboff + ((cw >> 16) & 0xFF)) * 2;
            int b3 = (cboff + ((cw >> 24) & 0xFF)) * 2;
            __half2 g0=cb_h2[b0], g1=cb_h2[b0+1], g2=cb_h2[b1], g3=cb_h2[b1+1];
            __half2 g4=cb_h2[b2], g5=cb_h2[b2+1], g6=cb_h2[b3], g7=cb_h2[b3+1];
            const int xo = k0 << 1;
            __half2 x0=x_h2[xo],   x1=x_h2[xo+1], x2=x_h2[xo+2], x3=x_h2[xo+3];
            __half2 x4=x_h2[xo+4], x5=x_h2[xo+5], x6=x_h2[xo+6], x7=x_h2[xo+7];
            acc2a=__hfma2(g0,x0,acc2a); acc2a=__hfma2(g1,x1,acc2a);
            acc2b=__hfma2(g2,x2,acc2b); acc2b=__hfma2(g3,x3,acc2b);
            acc2c=__hfma2(g4,x4,acc2c); acc2c=__hfma2(g5,x5,acc2c);
            acc2d=__hfma2(g6,x6,acc2d); acc2d=__hfma2(g7,x7,acc2d);
        }
        acc2 = __hadd2(__hadd2(acc2a, acc2b), __hadd2(acc2c, acc2d));
    } else {
        for (int k = lane; k < codes_per_row; k += 32) {
            int cb_id = k / codes_per_cb;
            int code  = (int)codes_row[k];
            int base  = (cb_id * K_CB + code) * 2;
            __half2 c01 = cb_h2[base + 0];
            __half2 c23 = cb_h2[base + 1];
            __half2 x01 = x_h2[k * 2 + 0];
            __half2 x23 = x_h2[k * 2 + 1];
            acc2 = __hfma2(c01, x01, acc2);
            acc2 = __hfma2(c23, x23, acc2);
        }
    }
    float acc = __half2float(__hadd(__low2half(acc2), __high2half(acc2)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y_cat[out_row] = __float2half(acc);
}

at::Tensor vq4_dequant_grouped_gemv_indexed_cuda(
    at::Tensor x_grouped,      // (G, K) fp16
    at::Tensor codes_all,      // (E_full*N, K/VDIM) uint8
    at::Tensor centroids_all,  // (E_full*n_cb, K_CB, VDIM) fp16
    at::Tensor sel,            // (G,) int32
    int64_t G,
    int64_t N,
    int64_t n_cb,
    int64_t codes_per_cb,
    int64_t reorder_ok)
{
    TORCH_CHECK(x_grouped.is_cuda(), "x_grouped must be CUDA");
    TORCH_CHECK(x_grouped.dtype() == torch::kHalf, "x_grouped must be fp16");
    TORCH_CHECK(codes_all.dtype() == torch::kUInt8, "codes must be uint8");
    TORCH_CHECK(centroids_all.dtype() == torch::kHalf, "centroids must be fp16");
    TORCH_CHECK(sel.dtype() == torch::kInt32, "sel must be int32");

    x_grouped = x_grouped.contiguous();
    codes_all = codes_all.contiguous();
    centroids_all = centroids_all.contiguous();
    sel = sel.contiguous();

    const int K = x_grouped.size(1);
    auto y_cat = torch::empty({G * N}, x_grouped.options());
    int blocks_per_expert = ((int)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((int)G * blocks_per_expert);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t smem = (K + n_cb * K_CB * VDIM) * sizeof(__half);
    cudaFuncSetAttribute(
        vq4_dequant_grouped_gemv_indexed_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    vq4_dequant_grouped_gemv_indexed_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(x_grouped.data_ptr<at::Half>()),
        codes_all.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(centroids_all.data_ptr<at::Half>()),
        sel.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(y_cat.data_ptr<at::Half>()),
        (int)G, (int)N, K, (int)n_cb, (int)codes_per_cb, blocks_per_expert,
        (int)reorder_ok);
    return y_cat;
}

// -------------------------------------------------------------------------
// Kernel 4: LoRA Grouped-GEMV
//
// Replaces torch.bmm((E, 1, K), (E, K, M)) → (E, 1, M) with a single kernel.
// Eliminates cuBLAS per-batch launch overhead (60 launches → 1 launch).
//
// Layout:
//   A:  (E, K)        fp16 — per-expert input vector
//   B:  (E, K, M)     fp16 — per-expert weight matrix (K,M contiguous, M-minor)
//   Y:  (E, M)        fp16 — per-expert output
//
// Grid: (E, ceil(M / OUT_TILE))
// Block: 1 warp (32 threads)
// OUT_TILE = 32  → each thread handles 1 output m
//
// Coalescing: 32 threads read 32 consecutive m of B[e, k, :] per k — 1 sector.
// A[e, :] cached in shmem (fp16), K up to 2048 → 4 KB shmem.
// -------------------------------------------------------------------------
static constexpr int LORA_OUT_TILE = 32;

__global__ __launch_bounds__(32, 16)
void lora_grouped_gemv_kernel(
    const __half* __restrict__ A,   // (E, K)
    const __half* __restrict__ B,   // (E, K, M)
    __half* __restrict__ Y,         // (E, M)
    int K, int M)
{
    const int e = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int lane = threadIdx.x;                    // 0..31
    const int m = m_tile * LORA_OUT_TILE + lane;

    // Load A[e, :] into shmem (dynamic size K)
    extern __shared__ __half a_smem[];
    const __half* a_row = A + e * K;
    for (int k = lane; k < K; k += 32) {
        a_smem[k] = a_row[k];
    }
    __syncwarp();

    if (m >= M) return;

    const __half* b_col = B + (size_t)e * K * M + m;    // walks with stride M
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += __half2float(a_smem[k]) * __half2float(b_col[(size_t)k * M]);
    }
    Y[(size_t)e * M + m] = __float2half(acc);
}

// Host launcher
at::Tensor lora_grouped_gemv_cuda(
    at::Tensor A,   // (E, K) fp16 — can also accept (E, 1, K) → auto-reshape
    at::Tensor B)   // (E, K, M) fp16
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kHalf && B.dtype() == torch::kHalf, "fp16 only");
    TORCH_CHECK(B.dim() == 3, "B must be (E, K, M)");

    // Accept A as (E, K) or (E, 1, K)
    at::Tensor A2 = A;
    if (A2.dim() == 3) {
        TORCH_CHECK(A2.size(1) == 1, "A middle dim must be 1");
        A2 = A2.view({A2.size(0), A2.size(2)});
    }
    TORCH_CHECK(A2.dim() == 2, "A must be (E, K) or (E, 1, K)");

    A2 = A2.contiguous();
    B  = B.contiguous();

    const int E = (int)A2.size(0);
    const int K = (int)A2.size(1);
    const int M = (int)B.size(2);
    TORCH_CHECK((int)B.size(0) == E, "batch mismatch");
    TORCH_CHECK((int)B.size(1) == K, "K mismatch");

    auto Y = torch::empty({E, M}, A2.options());

    dim3 grid(E, (M + LORA_OUT_TILE - 1) / LORA_OUT_TILE);
    dim3 block(32);
    size_t smem = K * sizeof(__half);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(
        lora_grouped_gemv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    lora_grouped_gemv_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(A2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        K, M);

    return Y;
}

// NOTE: an EXPERT-INDEXED LoRA grouped-GEMV (gather U/SV inside the kernel via
// sel) was implemented and benchmarked, but the small-rank (M=rank=16, K=in_d)
// GEMV is far slower than cuBLAS bmm — it regressed decode 25.31 -> 18.26 tok/s.
// So the LoRA gather stays as index_select + bmm; only codes/centroids are
// gathered in-kernel (vq4_dequant_grouped_gemv_indexed). Not added here.

// -------------------------------------------------------------------------
// Kernel 5: LoRA-U Grouped-GEMV (K-parallel reduce, rank-M output)
//
// Replaces torch.bmm((E, 1, K), (E, K, M)) where K is LARGE (1408..2048)
// and M is SMALL (rank=32). K-parallel: warp threads reduce along K.
//
// Layout:
//   A:  (E, K)          fp16 — per-expert input vector
//   BT: (E, M, K)       fp16 — per-expert weight (K-minor for coalesced reads)
//                         Pre-transposed from (E, K, M).
//   Y:  (E, M)          fp16 — per-expert output
//
// Grid: (E,)                — one block per expert
// Block: 128 threads (4 warps)
// Each warp handles M/4 rank rows. Each lane reduces K/32 elements.
// -------------------------------------------------------------------------
__global__ __launch_bounds__(128, 8)
void lora_u_grouped_gemv_kernel(
    const __half* __restrict__ A,    // (E, K)
    const __half* __restrict__ BT,   // (E, M, K)  ← already transposed
    __half* __restrict__ Y,          // (E, M)
    int K, int M)
{
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;    // 0..3
    const int warps = 4;
    const int M_per_warp = (M + warps - 1) / warps;
    const int r_start = warp * M_per_warp;
    const int r_end = min(r_start + M_per_warp, M);

    // Load A[e, :] into shmem
    extern __shared__ __half a_smem[];
    const __half* a_row = A + e * K;
    for (int k = tid; k < K; k += blockDim.x) {
        a_smem[k] = a_row[k];
    }
    __syncthreads();

    // Each warp handles M_per_warp rank rows
    for (int r = r_start; r < r_end; ++r) {
        const __half* b_row = BT + (size_t)e * M * K + (size_t)r * K;
        float acc = 0.0f;
        // Half2-vectorized: 32 lanes stride through K/2 half2 elements
        const __half2* a_row2 = reinterpret_cast<const __half2*>(a_smem);
        const __half2* b_row2 = reinterpret_cast<const __half2*>(b_row);
        const int K2 = K >> 1;
        #pragma unroll 2
        for (int k = lane; k < K2; k += 32) {
            __half2 a2 = a_row2[k];
            __half2 b2 = b_row2[k];
            float2 af = __half22float2(a2);
            float2 bf = __half22float2(b2);
            acc += af.x * bf.x + af.y * bf.y;
        }
        // Warp reduce
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            Y[e * M + r] = __float2half(acc);
        }
    }
}

at::Tensor lora_u_grouped_gemv_cuda(
    at::Tensor A,       // (E, K) or (E, 1, K) fp16
    at::Tensor BT)      // (E, M, K) fp16 — pre-transposed
{
    TORCH_CHECK(A.is_cuda() && BT.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kHalf && BT.dtype() == torch::kHalf, "fp16 only");
    TORCH_CHECK(BT.dim() == 3, "BT must be (E, M, K)");

    at::Tensor A2 = A;
    if (A2.dim() == 3) {
        TORCH_CHECK(A2.size(1) == 1, "A middle dim must be 1");
        A2 = A2.view({A2.size(0), A2.size(2)});
    }
    A2 = A2.contiguous();
    BT = BT.contiguous();

    const int E = (int)A2.size(0);
    const int K = (int)A2.size(1);
    const int M = (int)BT.size(1);
    TORCH_CHECK((int)BT.size(0) == E, "batch mismatch");
    TORCH_CHECK((int)BT.size(2) == K, "K mismatch");
    TORCH_CHECK(K % 2 == 0, "K must be even (half2 vectorization)");

    auto Y = torch::empty({E, M}, A2.options());

    dim3 grid(E);
    dim3 block(128);
    size_t smem = K * sizeof(__half);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(
        lora_u_grouped_gemv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    lora_u_grouped_gemv_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(A2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(BT.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        K, M);

    return Y;
}

// -------------------------------------------------------------------------
// Kernel 6: Fused SiLU(gate) * up
//
// Replaces `F.silu(gate) * up` (two element-wise kernels + intermediate)
// with a single memory pass. Applies half2 vectorization.
// -------------------------------------------------------------------------
__global__ void silu_and_mul_kernel(
    const __half2* __restrict__ gate,   // (n_h2,)
    const __half2* __restrict__ up,     // (n_h2,)
    __half2* __restrict__ out,          // (n_h2,)
    int n_h2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_h2) return;
    __half2 g = gate[idx];
    __half2 u = up[idx];
    // SiLU: g * sigmoid(g) = g / (1 + exp(-g))
    float2 gf = __half22float2(g);
    float2 uf = __half22float2(u);
    gf.x = gf.x / (1.0f + __expf(-gf.x));
    gf.y = gf.y / (1.0f + __expf(-gf.y));
    __half2 result = __floats2half2_rn(gf.x * uf.x, gf.y * uf.y);
    out[idx] = result;
}

at::Tensor silu_and_mul_cuda(at::Tensor gate, at::Tensor up)
{
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(gate.dtype() == torch::kHalf, "fp16 only");
    TORCH_CHECK(gate.sizes() == up.sizes(), "shape mismatch");

    at::Tensor gate_c = gate.contiguous();
    at::Tensor up_c   = up.contiguous();
    at::Tensor out    = torch::empty_like(gate_c);

    const int64_t n = gate_c.numel();
    TORCH_CHECK(n % 2 == 0, "numel must be even for half2 vectorization");
    const int64_t n_h2 = n / 2;

    const int threads = 256;
    const int blocks = (int)((n_h2 + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    silu_and_mul_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half2*>(gate_c.data_ptr<at::Half>()),
        reinterpret_cast<const __half2*>(up_c.data_ptr<at::Half>()),
        reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
        (int)n_h2);

    return out;
}

// -------------------------------------------------------------------------
// pybind
// -------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq4_dequant_matmul", &vq4_dequant_matmul_cuda,
          "VQ4 fused dequant + matmul (auto GEMV/GEMM)",
          pybind11::arg("x_rot"),
          pybind11::arg("codes"),
          pybind11::arg("centroids"),
          pybind11::arg("n_cb"),
          pybind11::arg("codes_per_cb"));
    m.def("vq4_dequant_grouped_gemv", &vq4_dequant_grouped_gemv_cuda,
          "VQ4 grouped GEMV for MoE (E experts)",
          pybind11::arg("x_grouped"),
          pybind11::arg("codes_cat"),
          pybind11::arg("centroids_cat"),
          pybind11::arg("E"),
          pybind11::arg("N"),
          pybind11::arg("n_cb"),
          pybind11::arg("codes_per_cb"));
    m.def("vq4_dequant_grouped_gemv_indexed", &vq4_dequant_grouped_gemv_indexed_cuda,
          "VQ4 grouped dequant+GEMV with expert indexing (top-k gather in kernel)",
          pybind11::arg("x_grouped"),
          pybind11::arg("codes_all"),
          pybind11::arg("centroids_all"),
          pybind11::arg("sel"),
          pybind11::arg("G"),
          pybind11::arg("N"),
          pybind11::arg("n_cb"),
          pybind11::arg("codes_per_cb"),
          pybind11::arg("reorder_ok") = 0);
    m.def("vq4_fused_gather_gemv", &vq4_fused_gather_gemv_cuda,
          "VQ4 fused gather + dequant + matmul (Walsh pre-done outside)",
          pybind11::arg("x_walshed"),
          pybind11::arg("sigma_inv"),
          pybind11::arg("codes"),
          pybind11::arg("centroids"),
          pybind11::arg("n_cb"),
          pybind11::arg("codes_per_cb"));
    m.def("lora_grouped_gemv", &lora_grouped_gemv_cuda,
          "Batched GEMV replacing bmm((E,1,K), (E,K,M)) → (E,M)",
          pybind11::arg("A"),
          pybind11::arg("B"));
    m.def("lora_u_grouped_gemv", &lora_u_grouped_gemv_cuda,
          "K-parallel batched GEMV for large-K small-M pattern (E,1,K)@(E,M,K)^T",
          pybind11::arg("A"),
          pybind11::arg("BT"));
    m.def("silu_and_mul", &silu_and_mul_cuda,
          "Fused SiLU(gate) * up in one memory pass",
          pybind11::arg("gate"),
          pybind11::arg("up"));
}
