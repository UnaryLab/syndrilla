#include "gf2.cuh"
#include "osd0.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <climits>

#define THREADS 256

__global__ void k_osd0_fused(
    const uint64_t* __restrict__ H_packed,   // [M, W]   shared across samples
    const uint8_t*  __restrict__ synd,       // [B, M]   syndrome bits
    const int32_t*  __restrict__ order,      // [B, N]   column order
    uint8_t*        __restrict__ e_out,      // [B, N]   output
    int M, int W, int N, int A_rank
) {
    extern __shared__ char _smem[];
    uint64_t* aug      = reinterpret_cast<uint64_t*>(_smem);     // [M*W]
    int32_t*  row_pcol = reinterpret_cast<int32_t*>(aug + (size_t)M * W);  // [M]
    __shared__ int s_pivot;   // lowest pivot-row index found this column
    __shared__ int s_found;   // pivots found so far (== rank when complete)

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    const int      wN    = N >> 6;
    const uint64_t maskN = 1ULL << (N & 63);

    // First stage all of H. Word wN holds both H bits and the syndrome bit, so
    // the syndrome OR below must wait until every H word (including wN) is
    // written — otherwise a staging write from another thread can clobber the
    // syndrome bit. This barrier is essential once the block spans >1 warp.
    for (int i = tid; i < M * W; i += bsz) aug[i] = H_packed[i];
    if (tid == 0) s_found = 0;
    __syncthreads();
    for (int r = tid; r < M; r += bsz) {
        row_pcol[r] = -1;
        if (synd[b * M + r]) aug[(size_t)r * W + wN] |= maskN;   // syndrome column
    }
    __syncthreads();

    for (int ci = 0; ci < N; ci++) {
        if (s_found >= A_rank) break;            // full rank reached (uniform)

        const int      c  = order[b * N + ci];
        const int      wc = c >> 6;
        const uint64_t mc = 1ULL << (c & 63);

        if (tid == 0) s_pivot = INT_MAX;
        __syncthreads();

        // Lowest unused row with bit c set is the pivot.
        for (int r = tid; r < M; r += bsz) {
            if (row_pcol[r] == -1 && (aug[(size_t)r * W + wc] & mc))
                atomicMin(&s_pivot, r);
        }
        __syncthreads();

        const int piv = s_pivot;                  // uniform across the block
        if (piv != INT_MAX) {
            if (tid == 0) { row_pcol[piv] = c; s_found++; }
            const uint64_t* prow = aug + (size_t)piv * W;
            // Eliminate column c from every other row that has it set.
            for (int r = tid; r < M; r += bsz) {
                if (r != piv && (aug[(size_t)r * W + wc] & mc))
                    gf2_xor_row(aug + (size_t)r * W, prow, W);
            }
        }
        __syncthreads();                          // single barrier, block-uniform
    }

    for (int c = tid; c < N; c += bsz) e_out[b * N + c] = 0;
    __syncthreads();
    for (int r = tid; r < M; r += bsz) {
        const int c = row_pcol[r];
        if (c != -1)
            e_out[b * N + c] = (aug[(size_t)r * W + wN] & maskN) ? 1 : 0;
    }
}

void osd0_fused_cuda(
    torch::Tensor H_packed, torch::Tensor synd, torch::Tensor order,
    torch::Tensor e_out, int64_t N, int64_t A_rank, int64_t block_size
) {
    OSD_CHECK(H_packed); OSD_CHECK(synd); OSD_CHECK(order); OSD_CHECK(e_out);
    const int M = (int)H_packed.size(0);
    const int W = (int)H_packed.size(1);
    const int B = (int)synd.size(0);
    if (B == 0) return;

    const size_t smem = (size_t)M * W * sizeof(uint64_t) + (size_t)M * sizeof(int32_t);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(k_osd0_fused,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd0_fused<<<B, (int)block_size, smem, stream>>>(
        reinterpret_cast<const uint64_t*>(H_packed.data_ptr<int64_t>()),
        synd.data_ptr<uint8_t>(),
        order.data_ptr<int32_t>(),
        e_out.data_ptr<uint8_t>(),
        M, W, (int)N, (int)A_rank);
}

int64_t fused_smem_bytes(int64_t M, int64_t W) {
    return M * W * (int64_t)sizeof(uint64_t) + M * (int64_t)sizeof(int32_t);
}

int64_t fused_smem_limit() {
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 48 * 1024;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)v;
}

// For the current column step, each sample picks the lowest-index row that has
// no pivot yet and has the step's column bit set. One thread per sample (the
// per-step path is the large-code / debugging fallback, so simplicity beats
// occupancy here). Writes pivot_row[b] (-1 if none) and marks row_pcol.

__global__ void k_osd_pivot(
    const uint64_t* __restrict__ aug,        // [B, M, W]
    const int32_t*  __restrict__ order,      // [B, N]
    int32_t*        __restrict__ row_pcol,   // [B, M]  in/out
    int32_t*        __restrict__ pivot_row,  // [B]     out
    int B, int M, int W, int N, int step
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int      c  = order[b * N + step];
    const int      wc = c >> 6;
    const uint64_t mc = 1ULL << (c & 63);
    const uint64_t* A = aug + (size_t)b * M * W;

    int piv = -1;
    for (int r = 0; r < M; r++) {
        if (row_pcol[b * M + r] == -1 && (A[(size_t)r * W + wc] & mc)) { piv = r; break; }
    }
    pivot_row[b] = piv;
    if (piv >= 0) row_pcol[b * M + piv] = c;
}

void osd_pivot_cuda(
    torch::Tensor aug, torch::Tensor order,
    torch::Tensor row_pcol, torch::Tensor pivot_row, int64_t step, int64_t N
) {
    OSD_CHECK(aug); OSD_CHECK(order); OSD_CHECK(row_pcol); OSD_CHECK(pivot_row);
    const int B = (int)aug.size(0);
    const int M = (int)aug.size(1);
    const int W = (int)aug.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_pivot<<<(B + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(aug.data_ptr<int64_t>()),
        order.data_ptr<int32_t>(),
        row_pcol.data_ptr<int32_t>(),
        pivot_row.data_ptr<int32_t>(),
        B, M, W, (int)N, (int)step);
}

// For the current column step, XOR each sample's pivot row into every other row
// that has the step's column bit set. One thread per (sample, row) pair so the
// heaviest step still parallelises across the batch and the rows.

__global__ void k_osd_eliminate(
    uint64_t*       __restrict__ aug,        // [B, M, W]  in/out
    const int32_t*  __restrict__ order,      // [B, N]
    const int32_t*  __restrict__ pivot_row,  // [B]
    int B, int M, int W, int N, int step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;

    const int r = idx % M;
    const int b = idx / M;

    const int piv = pivot_row[b];
    if (piv < 0 || r == piv) return;

    const int      c  = order[b * N + step];
    const int      wc = c >> 6;
    const uint64_t mc = 1ULL << (c & 63);
    uint64_t* A = aug + (size_t)b * M * W;

    if (A[(size_t)r * W + wc] & mc)
        gf2_xor_row(A + (size_t)r * W, A + (size_t)piv * W, W);
}

void osd_eliminate_cuda(
    torch::Tensor aug, torch::Tensor order,
    torch::Tensor pivot_row, int64_t step, int64_t N
) {
    OSD_CHECK(aug); OSD_CHECK(order); OSD_CHECK(pivot_row);
    const int B = (int)aug.size(0);
    const int M = (int)aug.size(1);
    const int W = (int)aug.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_eliminate<<<(B * M + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<uint64_t*>(aug.data_ptr<int64_t>()),
        order.data_ptr<int32_t>(),
        pivot_row.data_ptr<int32_t>(),
        B, M, W, (int)N, (int)step);
}

// After elimination, every pivot row's syndrome bit is the solution bit for its
// pivot column. Free (non-pivot) columns stay 0. e_out must be pre-zeroed by the
// caller. One thread per (sample, row) pair.

__global__ void k_osd_solve(
    const uint64_t* __restrict__ aug,       // [B, M, W]
    const int32_t*  __restrict__ row_pcol,  // [B, M]
    uint8_t*        __restrict__ e_out,     // [B, N]  pre-zeroed
    int B, int M, int W, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;

    const int r = idx % M;
    const int b = idx / M;

    const int c = row_pcol[b * M + r];
    if (c == -1) return;

    const int      wN    = N >> 6;
    const uint64_t maskN = 1ULL << (N & 63);
    const uint64_t* A = aug + (size_t)b * M * W;
    e_out[b * N + c] = (A[(size_t)r * W + wN] & maskN) ? 1 : 0;
}

void osd_solve_cuda(
    torch::Tensor aug, torch::Tensor row_pcol, torch::Tensor e_out, int64_t N
) {
    OSD_CHECK(aug); OSD_CHECK(row_pcol); OSD_CHECK(e_out);
    const int B = (int)aug.size(0);
    const int M = (int)aug.size(1);
    const int W = (int)aug.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_solve<<<(B * M + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(aug.data_ptr<int64_t>()),
        row_pcol.data_ptr<int32_t>(),
        e_out.data_ptr<uint8_t>(),
        B, M, W, (int)N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("osd0_fused", &osd0_fused_cuda,
          "Full OSD-0 GF(2) solve per sample in one launch (CUDA)");
    m.def("fused_smem_bytes", &fused_smem_bytes,
          "Dynamic shared memory (bytes) the fused kernel needs for [M, W]");
    m.def("fused_smem_limit", &fused_smem_limit,
          "Maximum opt-in dynamic shared memory per block on the current device");

    m.def("osd_pivot", &osd_pivot_cuda,
          "Per-step: select pivot row for the current column (CUDA)");
    m.def("osd_eliminate", &osd_eliminate_cuda,
          "Per-step: XOR pivot row into rows holding the column bit (CUDA)");
    m.def("osd_solve", &osd_solve_cuda,
          "Per-step: read out OSD-0 estimate from pivot rows (CUDA)");
}
