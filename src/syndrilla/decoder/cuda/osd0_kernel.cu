#include "osd0.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <climits>

#define THREADS 256
#define FULL_MASK 0xffffffffu

// OSD-0 in two phases per sample.
// 1. k_osd_scan finds the pivot columns: the greedy independent set of H's
//    columns in reliability order. It runs forward elimination on the rows that
//    have no pivot yet, holding only T (row u of the eliminated matrix is
//    T_u . H), so a column's bit on row u is the parity T_u . h_c. The syndrome
//    rides along as T . s; the scan stops early once it is zero on every row
//    without a pivot (see k_osd_scan).
// 2. The pivot columns found are gathered into a workspace ws [B, M, Wk]
//    (column j = H column order[b, j], bit K = syndrome) and eliminated with
//    Gauss-Jordan (k_osd0_fused or the per-step kernels). Columns without a
//    pivot cause no row operations, so this matches Gauss-Jordan over all N
//    columns. Bits of already-processed columns are never read again, so row
//    XORs start at the word of the current column.

// Fill ws with the first K columns of each sample's order, from a CSC copy of H.
// One thread per (sample, workspace column). ws must be zeroed by the caller.
__global__ void k_osd_gather(
    uint64_t*       __restrict__ ws,       // [B, M, Wk]
    const int64_t*  __restrict__ colptr,   // [N + 2]; column N is empty (padding)
    const int32_t*  __restrict__ rowidx,   // [nnz]
    const int32_t*  __restrict__ order,    // [B, N]
    int B, int M, int Wk, int N, int K
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * K) return;
    const int b = (int)(idx / K);
    const int j = (int)(idx % K);
    const int c = order[(int64_t)b * N + j];
    const unsigned long long bit = 1ULL << (j & 63);
    uint64_t* A = ws + (size_t)b * M * Wk + (j >> 6);
    for (int64_t p = colptr[c]; p < colptr[c + 1]; p++)
        atomicOr(reinterpret_cast<unsigned long long*>(A + (size_t)rowidx[p] * Wk), bit);
}

void osd_gather_cuda(
    torch::Tensor ws, torch::Tensor colptr, torch::Tensor rowidx,
    torch::Tensor order, int64_t K
) {
    OSD_CHECK(ws); OSD_CHECK(colptr); OSD_CHECK(rowidx); OSD_CHECK(order);
    const int B = (int)ws.size(0);
    const int M = (int)ws.size(1);
    const int W = (int)ws.size(2);
    const int N = (int)order.size(1);
    if (B == 0 || K == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_gather<<<(int)(((int64_t)B * K + THREADS - 1) / THREADS), THREADS, 0, stream>>>(
        reinterpret_cast<uint64_t*>(ws.data_ptr<int64_t>()),
        colptr.data_ptr<int64_t>(), rowidx.data_ptr<int32_t>(),
        order.data_ptr<int32_t>(), B, M, W, N, (int)K);
}

// Whole elimination of columns 0..K-1 for one sample per block, in shared
// memory. Writes the eliminated rows back to ws, plus row_pcol and found.
__global__ void k_osd0_fused(
    uint64_t*       __restrict__ ws,         // [B, M, Wk]  in/out
    int32_t*        __restrict__ row_pcol,   // [B, M]      out (-1 = no pivot)
    int32_t*        __restrict__ found_out,  // [B]         out
    int M, int Wk, int K, int A_rank
) {
    extern __shared__ char _smem[];
    uint64_t* aug = reinterpret_cast<uint64_t*>(_smem);                  // [M*Wk]
    int32_t*  rp  = reinterpret_cast<int32_t*>(aug + (size_t)M * Wk);    // [M]
    __shared__ int s_pivot;   // lowest pivot-row index found this column
    __shared__ int s_found;   // pivots found so far (== rank when complete)

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    uint64_t* g = ws + (size_t)b * M * Wk;
    for (int i = tid; i < M * Wk; i += bsz) aug[i] = g[i];
    for (int r = tid; r < M; r += bsz) rp[r] = -1;
    if (tid == 0) s_found = 0;
    __syncthreads();

    for (int j = 0; j < K; j++) {
        if (s_found >= A_rank) break;            // full rank reached (uniform)

        const int      wc = j >> 6;
        const uint64_t mc = 1ULL << (j & 63);

        if (tid == 0) s_pivot = INT_MAX;
        __syncthreads();

        // Lowest unused row with bit j set is the pivot.
        for (int r = tid; r < M; r += bsz) {
            if (rp[r] == -1 && (aug[(size_t)r * Wk + wc] & mc))
                atomicMin(&s_pivot, r);
        }
        __syncthreads();

        const int piv = s_pivot;                  // uniform across the block
        if (piv != INT_MAX) {
            if (tid == 0) { rp[piv] = j; s_found++; }
            const uint64_t* prow = aug + (size_t)piv * Wk;
            for (int r = tid; r < M; r += bsz) {
                uint64_t* row = aug + (size_t)r * Wk;
                if (r != piv && (row[wc] & mc))
                    for (int w = wc; w < Wk; w++) row[w] ^= prow[w];
            }
        }
        __syncthreads();                          // single barrier, block-uniform
    }

    for (int i = tid; i < M * Wk; i += bsz) g[i] = aug[i];
    for (int r = tid; r < M; r += bsz) row_pcol[(int64_t)b * M + r] = rp[r];
    if (tid == 0) found_out[b] = s_found;
}

void osd0_fused_ws_cuda(
    torch::Tensor ws, torch::Tensor row_pcol, torch::Tensor found,
    int64_t K, int64_t A_rank, int64_t block_size
) {
    OSD_CHECK(ws); OSD_CHECK(row_pcol); OSD_CHECK(found);
    const int B = (int)ws.size(0);
    const int M = (int)ws.size(1);
    const int W = (int)ws.size(2);
    if (B == 0) return;

    const size_t smem = (size_t)M * W * sizeof(uint64_t) + (size_t)M * sizeof(int32_t);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(k_osd0_fused,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd0_fused<<<B, (int)block_size, smem, stream>>>(
        reinterpret_cast<uint64_t*>(ws.data_ptr<int64_t>()),
        row_pcol.data_ptr<int32_t>(), found.data_ptr<int32_t>(),
        M, W, (int)K, (int)A_rank);
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

// Per-step path. colw [B, M] caches each row's word of the current column, so
// the per-step bit tests read one coalesced array instead of strided rows.
// pivbuf [3, B] rotates: slot j%3 holds step j's pivot (INT_MAX = none), slot
// (j+1)%3 collects step j+1's pivot, slot (j+2)%3 is reset for step j+2.

// Load colw for column j and find step j's pivot into pivbuf slot j%3.
__global__ void k_osd_load_col(
    const uint64_t* __restrict__ ws, uint64_t* __restrict__ colw,
    const int32_t* __restrict__ row_pcol, int32_t* __restrict__ pivbuf,
    int B, int M, int Wk, int j
) {
    const int b = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    const uint64_t cw = ws[((size_t)b * M + r) * Wk + (j >> 6)];
    colw[(int64_t)b * M + r] = cw;
    if (row_pcol[(int64_t)b * M + r] == -1 && ((cw >> (j & 63)) & 1ULL))
        atomicMin(&pivbuf[(j % 3) * B + b], r);
}

void osd_load_col_cuda(
    torch::Tensor ws, torch::Tensor colw, torch::Tensor row_pcol,
    torch::Tensor pivbuf, int64_t j
) {
    OSD_CHECK(ws); OSD_CHECK(colw); OSD_CHECK(row_pcol); OSD_CHECK(pivbuf);
    const int B = (int)ws.size(0);
    const int M = (int)ws.size(1);
    const int W = (int)ws.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((M + THREADS - 1) / THREADS, B);
    k_osd_load_col<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ws.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(colw.data_ptr<int64_t>()),
        row_pcol.data_ptr<int32_t>(), pivbuf.data_ptr<int32_t>(), B, M, W, (int)j);
}

// One Gauss-Jordan step for column j: XOR step j's pivot row into every other
// row with bit j set (the warp XORs its hit rows one at a time with coalesced
// words from word j/64 onward), then find step j+1's pivot: the lowest unused
// row with bit j+1 set. Grid (row chunk, sample).
__global__ void k_osd_step(
    uint64_t* __restrict__ ws,        // [B, M, Wk]  in/out
    uint64_t* __restrict__ colw,      // [B, M]      in/out
    int32_t*  __restrict__ row_pcol,  // [B, M]      in/out
    int32_t*  __restrict__ pivbuf,    // [3, B]      in/out
    int32_t*  __restrict__ found,     // [B]         in/out
    int B, int M, int Wk, int j, int K
) {
    const int b    = blockIdx.y;
    const int r    = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int piv  = pivbuf[(j % 3) * B + b];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        pivbuf[((j + 2) % 3) * B + b] = INT_MAX;
        if (piv != INT_MAX) { row_pcol[(int64_t)b * M + piv] = j; found[b]++; }
    }
    const bool in = r < M;
    const int  wc = j >> 6;
    uint64_t* A  = ws + (size_t)b * M * Wk;
    uint64_t  cw = in ? colw[(int64_t)b * M + r] : 0;

    if (piv != INT_MAX) {                     // block-uniform
        const bool hit = in && r != piv && ((cw >> (j & 63)) & 1ULL);
        unsigned hits = __ballot_sync(FULL_MASK, hit);
        const uint64_t* prow = A + (size_t)piv * Wk;
        const int r0 = r - lane;
        while (hits) {
            const int l = __ffs(hits) - 1;
            hits &= hits - 1;
            uint64_t* row = A + (size_t)(r0 + l) * Wk;
            for (int w = wc + lane; w < Wk; w += 32) row[w] ^= prow[w];
        }
        if (hit) cw ^= prow[wc];
        __syncwarp();
    }

    const int jn = j + 1;
    if (!in || jn >= K) return;
    if ((jn >> 6) != wc) cw = A[(size_t)r * Wk + (jn >> 6)];
    colw[(int64_t)b * M + r] = cw;
    if (r != piv && row_pcol[(int64_t)b * M + r] == -1 && ((cw >> (jn & 63)) & 1ULL))
        atomicMin(&pivbuf[(jn % 3) * B + b], r);
}

void osd_step_cuda(
    torch::Tensor ws, torch::Tensor colw, torch::Tensor row_pcol,
    torch::Tensor pivbuf, torch::Tensor found, int64_t j, int64_t K
) {
    OSD_CHECK(ws); OSD_CHECK(colw); OSD_CHECK(row_pcol); OSD_CHECK(pivbuf); OSD_CHECK(found);
    const int B = (int)ws.size(0);
    const int M = (int)ws.size(1);
    const int W = (int)ws.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((M + THREADS - 1) / THREADS, B);
    k_osd_step<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<uint64_t*>(ws.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(colw.data_ptr<int64_t>()),
        row_pcol.data_ptr<int32_t>(), pivbuf.data_ptr<int32_t>(),
        found.data_ptr<int32_t>(), B, M, W, (int)j, (int)K);
}

#define SCAN_COLS 256
#define SCAN_ROWS 4096
#define SCAN_THREADS 512

// Pivot-column scan, one block per sample, all N columns in one launch.
// T is the row-operation matrix, kept compact: Tr row u is T_u (bits over
// original rows) and TuT row k is T transposed (bit u = T_u[k]). A row is
// stored only once it is first written, in a per-sample pool slot (Pr for Tr,
// Pc for TuT) that rslot / cslot map it to; a row without a slot is the unit
// vector. For column c the bits on the pivot-free rows are
// v = alive & XOR_{k in h_c} TuT[k]. The pivot is the lowest row p in v; then
// every other row u in v gets T_u ^= T_p (Tr rows u, and TuT[k] ^= v for every
// k in T_p). Writes the order positions of the pivots to piv_pos. A sample
// that needs more slots than its pool holds stops and sets overflow; the
// caller reruns it with pools of M rows, which never overflow.
//
// Early stop (when synd is given): sres = T . s is updated by the same row
// operations. Rows without a pivot are zero on every pivot column found so far,
// and the pivot rows restricted to those columns are triangular with full rank,
// so sres being zero on all rows without a pivot means s lies in the span of
// the pivot columns found so far. The full pivot set S has full column rank, so
// H_S e = s has a unique solution; it is then zero on the later pivots, and
// Gauss-Jordan on the prefix gives the same e_v as on all of S. The scan stops
// there. An inconsistent syndrome never lies in that span, so it never stops.
__global__ void k_osd_scan(
    uint64_t*       __restrict__ Pr,       // [B, Rmax, Uw]  zero on entry
    uint64_t*       __restrict__ Pc,       // [B, Cmax, Uw]  zero on entry
    int32_t*        __restrict__ rslot,    // [B, M]  Tr row -> Pr slot, -1 on entry
    int32_t*        __restrict__ cslot,    // [B, M]  TuT row -> Pc slot, -1 on entry
    const int64_t*  __restrict__ colptr,   // [N + 2]
    const int32_t*  __restrict__ rowidx,   // [nnz]
    const int32_t*  __restrict__ order,    // [B, N]
    int32_t*        __restrict__ piv_pos,  // [B, rank]
    int32_t*        __restrict__ found_out,// [B]
    const uint64_t* __restrict__ synd,     // [B, Uw] packed syndrome, or nullptr (no stop)
    uint8_t*        __restrict__ stopped,  // [B]  1 if the scan stopped early
    int32_t*        __restrict__ scan_end, // [B]  columns scanned
    uint8_t*        __restrict__ overflow, // [B]  1 if a pool ran out of slots
    int M, int Uw, int N, int rank, int Rmax, int Cmax
) {
    extern __shared__ uint64_t sh[];
    uint64_t* v     = sh;           // [Uw]
    uint64_t* alive = sh + Uw;      // [Uw]
    uint64_t* sres  = sh + 2 * Uw;  // [Uw]  T . s
    __shared__ int s_min[3];
    __shared__ int64_t s_p0[SCAN_COLS];        // colptr of each staged column
    __shared__ int s_beg[SCAN_COLS + 1];       // staged column i: s_rows[s_beg[i] .. s_beg[i+1])
    __shared__ int s_rows[SCAN_ROWS];
    __shared__ int s_cnt;
    __shared__ int s_nr, s_nc, s_ovf;          // Pr / Pc slots taken; pool overflow
    const int b = blockIdx.x, tid = threadIdx.x, bsz = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nwarp = bsz >> 5;
    uint64_t* PR = Pr + (size_t)b * Rmax * Uw;
    uint64_t* PC = Pc + (size_t)b * Cmax * Uw;
    int32_t* rs = rslot + (size_t)b * M;
    int32_t* cs = cslot + (size_t)b * M;
    const int32_t* ord = order + (int64_t)b * N;

    const bool use_stop = synd != nullptr;
    bool nz = false;
    for (int w = tid; w < Uw; w += bsz) {
        const int left = M - 64 * w;
        alive[w] = left >= 64 ? ~0ULL : ((1ULL << left) - 1);
        sres[w] = use_stop ? synd[(size_t)b * Uw + w] : 0;
        nz |= (sres[w] & alive[w]) != 0;
    }
    if (tid == 0) { s_min[0] = INT_MAX; s_nr = 0; s_nc = 0; s_ovf = 0; }
    int found = 0;
    int j0 = 0, jcnt = 0;     // columns j0 .. j0+jcnt-1 are staged
    bool direct = false;      // one column too long to stage: read rowidx directly
    const bool any_left = __syncthreads_or(nz);   // also the setup barrier
    bool done = use_stop && !any_left;
    bool ovf = false;

    int j = 0;
    for (; j < N && found < rank && !done; j++) {
        if (j == j0 + jcnt) {
            // Stage the row indices of up to SCAN_COLS next columns in shared
            // memory. Every reader of the previous stage is behind the last
            // barrier, so the writes below cannot race with them.
            if (tid < SCAN_COLS) {
                const int jj = j + tid;
                int64_t p0 = 0; int d = 0;
                if (jj < N) { const int c = ord[jj]; p0 = colptr[c]; d = (int)(colptr[c + 1] - p0); }
                s_p0[tid] = p0;
                s_beg[tid + 1] = d;
            }
            __syncthreads();
            if (tid == 0) {
                int acc = 0, n = 0;
                for (; n < SCAN_COLS && j + n < N; n++) {
                    if (acc + s_beg[n + 1] > SCAN_ROWS) break;
                    acc += s_beg[n + 1];
                    s_beg[n + 1] = acc;
                }
                s_beg[0] = 0;
                s_cnt = n;
            }
            __syncthreads();
            j0 = j;
            jcnt = s_cnt;
            direct = jcnt == 0;
            if (direct) jcnt = 1;
            else
                for (int i = warp; i < jcnt; i += nwarp)
                    for (int q = lane; q < s_beg[i + 1] - s_beg[i]; q += 32)
                        s_rows[s_beg[i] + q] = rowidx[s_p0[i] + q];
            __syncthreads();
        }
        // Three slots: this step's slot, the next step's (reset here; its last
        // reader, step j-2, is behind the barrier of step j-1), and the
        // previous step's (a thread may still read it after a no-pivot step).
        int* sm = &s_min[j % 3];
        if (tid == 0) s_min[(j + 1) % 3] = INT_MAX;
        const int q0 = direct ? 0 : s_beg[j - j0], q1 = direct ? 0 : s_beg[j - j0 + 1];
        const int64_t p0 = direct ? s_p0[0] : 0, p1 = direct ? p0 + s_beg[1] : 0;
        for (int w = tid; w < Uw; w += bsz) {
            uint64_t x = 0;
            for (int q = q0; q < q1; q++) {
                const int k = s_rows[q], sl = cs[k];
                if (sl >= 0) x ^= PC[(size_t)sl * Uw + w];
                else if ((k >> 6) == w) x ^= 1ULL << (k & 63);
            }
            for (int64_t p = p0; p < p1; p++) {
                const int k = rowidx[p], sl = cs[k];
                if (sl >= 0) x ^= PC[(size_t)sl * Uw + w];
                else if ((k >> 6) == w) x ^= 1ULL << (k & 63);
            }
            x &= alive[w];
            v[w] = x;
            if (x) atomicMin(sm, 64 * w + __ffsll((long long)x) - 1);
        }
        __syncthreads();
        const int piv = *sm;                           // block-uniform
        if (piv == INT_MAX) continue;

        if (tid == 0) piv_pos[(int64_t)b * rank + found] = j;
        found++;
        const int      pw = piv >> 6;
        const uint64_t pm = 1ULL << (piv & 63);
        const int      rp = rs[piv];
        const uint64_t* Tp = rp >= 0 ? PR + (size_t)rp * Uw : nullptr;   // nullptr: T_p = e_p
        const bool sp = use_stop && ((sres[pw] >> (piv & 63)) & 1ULL);

        // Take slots (initialized to the unit vector) for the rows about to be
        // written: Tr rows u in v without the pivot, and, when that set is not
        // empty, TuT rows k in T_p.
        bool vo = false;
        for (int w = tid; w < Uw; w += bsz) {
            uint64_t bits = (w == pw) ? (v[w] & ~pm) : v[w];
            vo |= bits != 0;
            while (bits) {
                const int u = 64 * w + __ffsll((long long)bits) - 1;
                bits &= bits - 1;
                if (rs[u] < 0) {
                    const int s = atomicAdd(&s_nr, 1);
                    if (s < Rmax) { rs[u] = s; PR[(size_t)s * Uw + (u >> 6)] = 1ULL << (u & 63); }
                    else s_ovf = 1;
                }
            }
        }
        const bool vany = __syncthreads_or(vo);
        if (vany)
            for (int w = tid; w < Uw; w += bsz) {
                uint64_t bits = Tp ? Tp[w] : (w == pw ? pm : 0);
                while (bits) {
                    const int k = 64 * w + __ffsll((long long)bits) - 1;
                    bits &= bits - 1;
                    if (cs[k] < 0) {
                        const int s = atomicAdd(&s_nc, 1);
                        if (s < Cmax) { cs[k] = s; PC[(size_t)s * Uw + (k >> 6)] = 1ULL << (k & 63); }
                        else s_ovf = 1;
                    }
                }
            }
        __syncthreads();
        if (s_ovf) { ovf = true; break; }              // block-uniform

        if (vany) {
            // TuT[k] ^= v (without the pivot row) for every k with T_p[k] set.
            for (int wq = warp; wq < Uw; wq += nwarp) {
                uint64_t bits = Tp ? Tp[wq] : (wq == pw ? pm : 0);
                while (bits) {
                    const int k = 64 * wq + __ffsll((long long)bits) - 1;
                    bits &= bits - 1;
                    uint64_t* row = PC + (size_t)cs[k] * Uw;
                    for (int w = lane; w < Uw; w += 32) row[w] ^= (w == pw) ? (v[w] & ~pm) : v[w];
                }
            }
            // Tr[u] ^= T_p for every other pivot-free row u with the bit.
            for (int wq = warp; wq < Uw; wq += nwarp) {
                uint64_t bits = v[wq] & ((wq == pw) ? ~pm : ~0ULL);
                while (bits) {
                    const int u = 64 * wq + __ffsll((long long)bits) - 1;
                    bits &= bits - 1;
                    uint64_t* row = PR + (size_t)rs[u] * Uw;
                    if (Tp) { for (int w = lane; w < Uw; w += 32) row[w] ^= Tp[w]; }
                    else if (lane == 0) row[pw] ^= pm;
                }
            }
        }
        __syncthreads();
        if (sp)
            for (int w = tid; w < Uw; w += bsz) sres[w] ^= (w == pw) ? (v[w] & ~pm) : v[w];
        if (tid == 0) alive[pw] &= ~pm;
        __syncthreads();
        if (use_stop) {
            nz = false;
            for (int w = tid; w < Uw; w += bsz) nz |= (sres[w] & alive[w]) != 0;
            done = !__syncthreads_or(nz);
        }
    }
    if (tid == 0) {
        found_out[b] = found;
        stopped[b] = done;
        scan_end[b] = j;
        overflow[b] = ovf;
    }
}

void osd_scan_cuda(
    torch::Tensor Pr, torch::Tensor Pc, torch::Tensor rslot, torch::Tensor cslot,
    torch::Tensor colptr, torch::Tensor rowidx, torch::Tensor order, torch::Tensor piv_pos,
    torch::Tensor found, torch::Tensor synd, torch::Tensor stopped, torch::Tensor scan_end,
    torch::Tensor overflow
) {
    OSD_CHECK(Pr); OSD_CHECK(Pc); OSD_CHECK(rslot); OSD_CHECK(cslot);
    OSD_CHECK(colptr); OSD_CHECK(rowidx); OSD_CHECK(order); OSD_CHECK(piv_pos);
    OSD_CHECK(found); OSD_CHECK(synd); OSD_CHECK(stopped); OSD_CHECK(scan_end);
    OSD_CHECK(overflow);
    const int B = (int)Pr.size(0);
    const int Uw = (int)Pr.size(2);
    if (B == 0) return;
    const size_t smem = 3 * (size_t)Uw * sizeof(uint64_t);
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(k_osd_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_scan<<<B, SCAN_THREADS, smem, stream>>>(
        reinterpret_cast<uint64_t*>(Pr.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(Pc.data_ptr<int64_t>()),
        rslot.data_ptr<int32_t>(), cslot.data_ptr<int32_t>(),
        colptr.data_ptr<int64_t>(), rowidx.data_ptr<int32_t>(), order.data_ptr<int32_t>(),
        piv_pos.data_ptr<int32_t>(), found.data_ptr<int32_t>(),
        synd.numel() ? reinterpret_cast<const uint64_t*>(synd.data_ptr<int64_t>()) : nullptr,
        stopped.data_ptr<uint8_t>(), scan_end.data_ptr<int32_t>(), overflow.data_ptr<uint8_t>(),
        (int)rslot.size(1), Uw, (int)order.size(1), (int)piv_pos.size(1),
        (int)Pr.size(1), (int)Pc.size(1));
}

// Read out: each pivot row's syndrome bit (bit K) is the solution bit of its
// column, H column order[b, row_pos]. Free columns stay 0 (e_out pre-zeroed).
// One thread per (sample, row).
__global__ void k_osd_solve(
    const uint64_t* __restrict__ ws,        // [B, M, Wk]
    const int32_t*  __restrict__ row_pos,   // [B, M]
    const int32_t*  __restrict__ order,     // [B, N]
    uint8_t*        __restrict__ e_out,     // [B, Ne]
    int B, int M, int Wk, int No, int Ne, int K
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * M) return;
    const int r = (int)(idx % M);
    const int b = (int)(idx / M);
    const int j = row_pos[(int64_t)b * M + r];
    if (j < 0) return;
    const uint64_t* A = ws + (size_t)b * M * Wk;
    e_out[(int64_t)b * Ne + order[(int64_t)b * No + j]] =
        (A[(size_t)r * Wk + (K >> 6)] & (1ULL << (K & 63))) ? 1 : 0;
}

void osd_solve_ws_cuda(
    torch::Tensor ws, torch::Tensor row_pos, torch::Tensor order,
    torch::Tensor e_out, int64_t K
) {
    OSD_CHECK(ws); OSD_CHECK(row_pos); OSD_CHECK(order); OSD_CHECK(e_out);
    const int B = (int)ws.size(0);
    const int M = (int)ws.size(1);
    const int W = (int)ws.size(2);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_osd_solve<<<(int)(((int64_t)B * M + THREADS - 1) / THREADS), THREADS, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ws.data_ptr<int64_t>()),
        row_pos.data_ptr<int32_t>(), order.data_ptr<int32_t>(),
        e_out.data_ptr<uint8_t>(), B, M, W, (int)order.size(1), (int)e_out.size(1), (int)K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("osd_gather", &osd_gather_cuda,
          "Pack the first K ordered columns of H into the per-sample workspace (CUDA)");
    m.def("osd0_fused", &osd0_fused_ws_cuda,
          "Eliminate workspace columns 0..K-1, one block per sample in shared memory (CUDA)");
    m.def("fused_smem_bytes", &fused_smem_bytes,
          "Dynamic shared memory (bytes) the fused kernel needs for [M, W]");
    m.def("fused_smem_limit", &fused_smem_limit,
          "Maximum opt-in dynamic shared memory per block on the current device");
    m.def("osd_load_col", &osd_load_col_cuda,
          "Per-step: cache column j's words and find step j's pivot (CUDA)");
    m.def("osd_step", &osd_step_cuda,
          "Per-step: eliminate column j and find step j+1's pivot (CUDA)");
    m.def("osd_scan", &osd_scan_cuda,
          "Find each sample's pivot columns in reliability order (CUDA)");
    m.def("osd_solve", &osd_solve_ws_cuda,
          "Read out the OSD-0 estimate from the pivot rows (CUDA)");
}
