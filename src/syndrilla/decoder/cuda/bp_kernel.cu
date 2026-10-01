#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/Atomic.cuh>       // gpuAtomicAdd
#include <math.h>
#include <climits>
#include <type_traits>

#define THREADS 256

static inline int grid1d(int64_t n) {
    const int64_t blocks = (n + THREADS - 1) / THREADS;
    TORCH_CHECK(blocks <= INT_MAX, "launch needs ", blocks, " blocks of ", THREADS,
                " threads for ", n, " elements, above the gridDim.x limit of ", INT_MAX,
                "; reduce the batch size");
    return (int)blocks;
}

// Every kernel indexes raw data pointers assuming row-major contiguous layout.
// A transposed-stride tensor (e.g. a syndrome built via (H @ e.T).T) would be
// silently read as other samples' data: fail loudly instead.
#define CHECK_INPUT(x)                                                  \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");            \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// Warp shuffle for scalar_t. c10::Half / c10::BFloat16 have no shuffle overload,
// so they travel as float, which holds every half and bfloat16 value exactly.
template <typename scalar_t>
__device__ __forceinline__ scalar_t shfl_xor(scalar_t v, int mask) {
    return (scalar_t)__shfl_xor_sync(0xffffffffu, (float)v, mask);
}
template <>
__device__ __forceinline__ double shfl_xor<double>(double v, int mask) {
    return __shfl_xor_sync(0xffffffffu, v, mask);
}

// Comparison helpers for the min-sum row update. With F64INT (the default) and
// double they work on the bit pattern with integer ops, because FP64 compares
// run at 1/64 rate on consumer GPUs; results match the floating-point compares
// bit for bit. F64INT = false uses the floating-point compares for double too.
// mag_key(x) orders non-NaN magnitudes like x itself (-0.0 and +0.0 share a
// key); a NaN magnitude never becomes a minimum, as with `<` on doubles.
template <typename scalar_t>
__device__ __forceinline__ scalar_t mag_key(scalar_t v) { return v; }
__device__ __forceinline__ unsigned long long mag_key(double v) {
    return (unsigned long long)__double_as_longlong(v) & 0x7fffffffffffffffULL;
}
// is_pos(v) == (v > 0); abs_val(v) == (v < 0 ? -v : v), so abs_val(-0.0) is -0.0.
template <typename scalar_t>
__device__ __forceinline__ bool is_pos(scalar_t v) { return v > (scalar_t)0; }
__device__ __forceinline__ bool is_pos(double v) {
    long long x = __double_as_longlong(v);
    return x > 0 && x <= 0x7ff0000000000000LL;
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(scalar_t v) { return (v < (scalar_t)0) ? -v : v; }
__device__ __forceinline__ double abs_val(double v) {
    unsigned long long x = (unsigned long long)__double_as_longlong(v);
    unsigned long long m = x & 0x7fffffffffffffffULL;
    bool is_neg = (x >> 63) && m != 0 && m <= 0x7ff0000000000000ULL;
    return __longlong_as_double((long long)(is_neg ? m : x));
}
// The helpers above for F64INT, the generic templates (floating-point compares
// for double too) otherwise.
template <bool F64INT, typename T>
__device__ __forceinline__ auto key_of(T v) {
    if constexpr (F64INT) return mag_key(v); else return mag_key<T>(v);
}
template <bool F64INT, typename T>
__device__ __forceinline__ bool pos_of(T v) {
    if constexpr (F64INT) return is_pos(v); else return is_pos<T>(v);
}
template <bool F64INT, typename T>
__device__ __forceinline__ T abs_of(T v) {
    if constexpr (F64INT) return abs_val(v); else return abs_val<T>(v);
}

// (v, k) strictly before (w, kw): smaller magnitude, ties go to the lower k.
// A serial k-order scan with `absval < min` keeps exactly this order, since a
// later entry never displaces an equal one; the reduction below reproduces it.
template <bool F64INT = true, typename scalar_t>
__device__ __forceinline__ bool lex_less(scalar_t v, int k, scalar_t w, int kw) {
    auto kv = key_of<F64INT>(v), kwk = key_of<F64INT>(w);
    return kv < kwk || (kv == kwk && k < kw);
}

// CSR edge layout of the check rows: row_ptr [M + 1] and col [nnz] with real
// edges only (col[e] < N). Row c holds edges e = begin(c) .. end(c) - 1 in (c, k)
// order, and an edge buffer [B, nnz + 1] holds edge e of sample b at
// b * stride + e (the last slot of every sample is zero; see llr_csr).
struct CsrRows {
    const int32_t* row_ptr; const int32_t* col;
    __device__ __forceinline__ int begin(int c) const { return __ldg(row_ptr + c); }
    __device__ __forceinline__ int end(int c) const { return __ldg(row_ptr + c + 1); }
    __device__ __forceinline__ int var(int e) const { return __ldg(col + e); }
    static constexpr bool PADDED = false;
};

// Padded edge layout of the check rows (edge_layout: padded): col [M, D] holds
// every check's D slots, dummy edges with col[e] == N. Row c holds edges
// e = c * D .. c * D + D - 1, and an edge buffer [B, M * D + 1] holds edge e of
// sample b at b * stride + e (the last slot of every sample is zero).
struct PaddedRows {
    const int32_t* col; int D, N;
    __device__ __forceinline__ int begin(int c) const { return c * D; }
    __device__ __forceinline__ int end(int c) const { return c * D + D; }
    __device__ __forceinline__ int var(int e) const { return __ldg(col + e); }
    static constexpr bool PADDED = true;   // dummy edges have col[e] >= N
};

// Fixed-point rounding of the quantized decoders, the arithmetic of
// syndrilla.utils.fp2fxp with floor rounding: floor(v * scale) clamped to
// [lo, hi], divided by scale. scale = 2^frac_width, lo = -2^(int_width +
// frac_width), hi = 2^(int_width + frac_width) - 1. The division is a multiply by
// inv = 2^-frac_width. Multiplying by a power of two is exact, so the result
// matches fp2fxp bit for bit; NaN stays NaN.
struct Fxp { double scale, inv, lo, hi; };

static inline Fxp make_fxp(int64_t int_width, int64_t frac_width) {
    return Fxp{ldexp(1.0, (int)frac_width), ldexp(1.0, -(int)frac_width),
               -ldexp(1.0, (int)(int_width + frac_width)),
               ldexp(1.0, (int)(int_width + frac_width)) - 1.0};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fxp(scalar_t v, const Fxp& q) {
    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value,
                                            double, float>::type;
    acc_t x = floor((acc_t)v * (acc_t)q.scale);
    x = x < (acc_t)q.lo ? (acc_t)q.lo : (x > (acc_t)q.hi ? (acc_t)q.hi : x);
    return (scalar_t)(x * (acc_t)q.inv);
}

// One warp per (b, c) check row; lane j handles k = j, j+32, ... (k is the
// position within the row) so the warp reads the row contiguously. Launch with
// B*M*32 threads and THREADS a multiple of 32, so every warp is either fully in
// range or fully out.
// FROM_LV = true: the VN→CN message a = l_v[b, n] − b_c2v[e] is computed on the
// fly and b_c2v is overwritten in place. Each lane reads and writes only its own
// k entries, so no lane reads an entry another lane writes.
// FROM_LV = false: a is read from the a_v2c edge buffer that l_row points at
// (filled by k_vn_update_csr).
// cn_row runs one check row c with the whole warp (every lane present): l_row
// and b_c2v point at the sample's LLR row (or a_v2c row) and edge buffer, s_neg
// at the row's ±1 syndrome sign.
// WARP = false runs the row on one thread (lane 0): a serial scan in k order
// with no merge, the same minima and tie order as the warp merge.
// Rows with dummy edges (PaddedRows) skip them in both passes.
// QUANT = true rounds with q (fxp) as bp_norm_min_sum_quant does: the VN→CN
// message a in both passes, beta, the selected minimum, and the outgoing message.
template <typename scalar_t, bool QUANT = false, typename Rows = CsrRows, bool WARP = true,
          bool F64INT = true, bool FROM_LV = true>
__device__ __forceinline__ void cn_row(
    const scalar_t* l_row, const scalar_t* s_neg_p, Rows rows,
    scalar_t* b_c2v, double beta, int c, int lane, Fxp q = Fxp{1.0, 1.0, 0.0, 0.0}
) {
    constexpr int step = WARP ? 32 : 1;
    const int lo = rows.begin(c), deg = rows.end(c) - lo;
    const int64_t base = lo;

    // Minima are tracked in scalar_t: a minimum is one of the row's own |a|, so a
    // wider type adds no precision, and the second pass matches |a| against min0
    // by key in the same type, as the PyTorch reference takes min in its dtype.
    // Each lane keeps its two smallest |a| with their k (INT_MAX = empty); the
    // sign product is kept as the parity of non-positive entries.
    int      neg  = 0;
    scalar_t min0 = (scalar_t)INFINITY;   // smallest |a|
    scalar_t min1 = (scalar_t)INFINITY;   // second smallest |a|
    int      k0   = INT_MAX, k1 = INT_MAX;

    for (int k = lane; k < deg; k += step) {
        int n = rows.var(lo + k);
        if constexpr (Rows::PADDED) { if (n >= rows.N) continue; }
        scalar_t val = FROM_LV ? (scalar_t)(l_row[n] - b_c2v[base + k]) : l_row[base + k];
        if constexpr (QUANT) val = fxp(val, q);
        // sign(0) maps to −1, matching torch.where(sgn == 0, −1, sgn).
        neg ^= pos_of<F64INT>(val) ? 0 : 1;

        scalar_t absval = abs_of<F64INT>(val);
        auto     key    = key_of<F64INT>(absval);
        if (key < key_of<F64INT>(min0)) { min1 = min0; k1 = k0; min0 = absval; k0 = k; }
        else if (key < key_of<F64INT>(min1)) { min1 = absval; k1 = k; }
    }

    // Butterfly merge: each step keeps the two lex-smallest of both lanes' pairs,
    // so every lane ends with the row's (min0, min1) and sign parity.
    if constexpr (WARP)
    for (int off = 16; off > 0; off >>= 1) {
        scalar_t o0 = shfl_xor(min0, off), o1 = shfl_xor(min1, off);
        int      j0 = __shfl_xor_sync(0xffffffffu, k0, off);
        int      j1 = __shfl_xor_sync(0xffffffffu, k1, off);
        neg ^= __shfl_xor_sync(0xffffffffu, neg, off);
        if (lex_less<F64INT>(o0, j0, min0, k0)) {
            if (lex_less<F64INT>(o1, j1, min0, k0)) { min1 = o1; k1 = j1; }
            else                                    { min1 = min0; k1 = k0; }
            min0 = o0; k0 = j0;
        } else if (lex_less<F64INT>(o0, j0, min1, k1)) {
            min1 = o0; k1 = j0;
        }
    }
    scalar_t sign_prod = neg ? (scalar_t)-1 : (scalar_t)1;

    scalar_t s_neg       = *s_neg_p;
    scalar_t scaled_beta = (scalar_t)beta;
    if constexpr (QUANT) scaled_beta = fxp(scaled_beta, q);
    // Outgoing sign excludes k's own contribution: s_neg * sign_prod * sign_k,
    // since sign_k² = 1. sign_prod and sign_k are ±1 and IEEE rounding is
    // sign-symmetric, so beta * (s_neg * sign_prod * sign_k) equals
    // ±(beta * (s_neg * sign_prod)) exactly; double uses that form to save two
    // FP64 multiplies per edge, other types keep the direct product (faster there).
    scalar_t out_pos = scaled_beta * (s_neg * sign_prod);   // sign_k = +1
    scalar_t out_neg = -out_pos;                             // sign_k = -1
    auto     key0    = key_of<F64INT>(min0);

    for (int k = lane; k < deg; k += step) {
        int n = rows.var(lo + k);
        if constexpr (Rows::PADDED) { if (n >= rows.N) continue; }
        scalar_t val = FROM_LV ? (scalar_t)(l_row[n] - b_c2v[base + k]) : l_row[base + k];
        if constexpr (QUANT) val = fxp(val, q);
        scalar_t min_result = (key_of<F64INT>(abs_of<F64INT>(val)) == key0) ? min1 : min0;
        if constexpr (QUANT) min_result = fxp(min_result, q);

        scalar_t out;
        if constexpr (std::is_same<scalar_t, double>::value) {
            out = (pos_of<F64INT>(val) ? out_pos : out_neg) * min_result;
        } else {
            scalar_t sign_k = pos_of<F64INT>(val) ? (scalar_t)1 : (scalar_t)-1;
            out = scaled_beta * (s_neg * sign_prod * sign_k) * min_result;
        }
        if constexpr (QUANT) out = fxp(out, q);
        b_c2v[base + k] = out;
    }
}

// num_iters (nullable): rows of samples with num_iters[b] != -1 return at once,
// leaving their b_c2v untouched. The whole warp shares b, so it returns as a
// unit and the shuffles in cn_row always run with every lane present.
// WARP = false: one thread per (b, c) check row, launched with B*M threads.
template <typename scalar_t, bool QUANT = false, typename Rows = CsrRows, bool WARP = true,
          bool F64INT = true, bool FROM_LV = true>
__global__ void k_cn_update(
    const scalar_t* l_v,                            // [B, N_ext], or a_v2c if !FROM_LV
    const scalar_t* __restrict__ syndrome_neg_bc,  // [B, M]  ±1 per check
    Rows            rows,
    scalar_t*       b_c2v,                          // edge buffer, output
    const int64_t*  __restrict__ num_iters,        // [B]  or nullptr
    double beta,
    int B, int M, int64_t stride, int N,
    Fxp q = Fxp{1.0, 1.0, 0.0, 0.0}
) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> (WARP ? 5 : 0);
    if (row >= (int64_t)B * M) return;
    int lane = WARP ? threadIdx.x & 31 : 0;

    int c    = (int)(row % M);
    int b    = (int)(row / M);
    if (num_iters != nullptr && num_iters[b] != -1) return;
    const int64_t off = (int64_t)b * stride;
    cn_row<scalar_t, QUANT, Rows, WARP, F64INT, FROM_LV>(
        l_v + (FROM_LV ? (int64_t)b * (N + 1) : off), syndrome_neg_bc + (int64_t)b * M + c,
        rows, b_c2v + off, beta, c, lane, q);
}

// VN→CN messages into their own edge buffer (fuse_vn: false): a_v2c[b, e] =
// l_v[b, col[e]] − b_c2v[b, e] for the n_edges edges, the same expression cn_row
// computes on the fly. Samples with num_iters[b] != -1 are skipped.
template <typename scalar_t>
__global__ void k_vn_update_csr(
    const scalar_t* __restrict__ l_v,       // [B, N_ext]
    const scalar_t* __restrict__ b_c2v,     // [B, stride]
    const int32_t*  __restrict__ col,       // [n_edges]
    scalar_t*       __restrict__ a_v2c,     // [B, stride], output
    const int64_t*  __restrict__ num_iters, // [B]
    int B, int64_t n_edges, int64_t stride, int N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * n_edges) return;
    int64_t e = idx % n_edges;
    int     b = (int)(idx / n_edges);
    if (num_iters[b] != -1) return;
    const int64_t i = (int64_t)b * stride + e;
    a_v2c[i] = (scalar_t)(l_v[(int64_t)b * (N + 1) + __ldg(col + e)] - b_c2v[i]);
}

// LLR of variable n on the CSR edge buffer [B, nnz + 1]: the sum of its edge
// messages, then the channel LLR u last. Adding u last matches the association
// of the PyTorch decoder, l_v = (sum of edge messages) + u_init; floating-point
// addition is not associative, and seeding the sum with u drifts by about 1e-15
// at float64, enough to flip the hard decision l_v <= 0 on bits at the boundary.
// VN_eid[n, vd] is the flat edge id of variable n's vd-th edge in (c, k) order,
// padded with nnz, whose slot is always zero. acc never holds -0.0 (it starts at
// +0.0), so the padding adds leave it unchanged. b_row is the sample's edge buffer.
template <typename scalar_t>
__device__ __forceinline__ scalar_t llr_csr(
    const scalar_t* b_row, const int32_t* __restrict__ VN_eid, int n, int VD, scalar_t u
) {
    scalar_t acc = (scalar_t)0;                     // edges first, u_init last
    for (int vd = 0; vd < VD; vd++) acc += b_row[VN_eid[n * VD + vd]];
    return acc + u;
}

template <typename scalar_t>
__global__ void k_llr_hard_update_csr(
    const scalar_t* __restrict__ u_init,    // [B, N_ext]
    const scalar_t* __restrict__ b_c2v,     // [B, nnz + 1]
    const int32_t*  __restrict__ VN_eid,    // [N_ext, VD]
    scalar_t*       __restrict__ l_v,       // [B, N_ext], output
    uint8_t*        __restrict__ e_v,       // [B, N_ext], output
    const int64_t*  __restrict__ num_iters, // [B]  skip samples != -1
    int B, int64_t stride, int N_ext, int VD
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * N_ext) return;

    int n = (int)(idx % N_ext);
    int b = (int)(idx / N_ext);
    if (num_iters[b] != -1) return;

    scalar_t acc = llr_csr(b_c2v + (int64_t)b * stride, VN_eid, n, VD, u_init[idx]);
    l_v[idx] = acc;
    e_v[idx] = (acc <= (scalar_t)0) ? 1 : 0;
}

// Ablation baseline (vn_gather: false): one atomicAdd per edge into the variable sum.
// sum[b, col[e]] += b_c2v[b, e] for the n_edges edges in no fixed order; sum
// must be zero on entry. Samples with num_iters[b] != -1 are skipped.
template <typename scalar_t>
__global__ void k_vn_sum_atomic(
    const scalar_t* __restrict__ b_c2v,     // [B, stride]
    const int32_t*  __restrict__ col,       // [n_edges]
    scalar_t*       sum,                    // [B, N_ext], accumulated
    const int64_t*  __restrict__ num_iters, // [B]
    int B, int64_t n_edges, int64_t stride, int N_ext
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * n_edges) return;
    int64_t e = idx % n_edges;
    int     b = (int)(idx / n_edges);
    if (num_iters[b] != -1) return;
    gpuAtomicAdd(sum + (int64_t)b * N_ext + __ldg(col + e), b_c2v[(int64_t)b * stride + e]);
}

// l_v = sum + u_init (the channel LLR last) and the uint8 hard decision, for
// samples with num_iters[b] == -1.
template <typename scalar_t>
__global__ void k_llr_hard_from_sum(
    const scalar_t* __restrict__ u_init,    // [B, N_ext]
    const scalar_t* __restrict__ sum,       // [B, N_ext]
    scalar_t*       __restrict__ l_v,       // [B, N_ext], output
    uint8_t*        __restrict__ e_v,       // [B, N_ext], output
    const int64_t*  __restrict__ num_iters, // [B]
    int B, int N_ext
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * N_ext) return;
    if (num_iters[idx / N_ext] != -1) return;
    scalar_t acc = sum[idx] + u_init[idx];
    l_v[idx] = acc;
    e_v[idx] = (acc <= (scalar_t)0) ? 1 : 0;
}

// Parity of the uint8 hard decisions e_row over check row c, reduced across the
// warp (every lane present); every lane gets the result.
template <typename Rows>
__device__ __forceinline__ int row_parity(const uint8_t* e_row, Rows rows, int c, int N, int lane) {
    int parity = 0;
    for (int e = rows.begin(c) + lane, hi = rows.end(c); e < hi; e += 32) {
        int n = rows.var(e);
        if (n < N && e_row[n]) parity ^= 1;
    }
    for (int off = 16; off > 0; off >>= 1)
        parity ^= __shfl_xor_sync(0xffffffffu, parity, off);
    return parity;
}

// One warp per (b, c) check row, same layout as k_cn_update. Compares the
// parity of the uint8 hard decision against the target syndrome and sets
// mismatch[b] = 1 on any difference; racing writes all store 1. Samples with
// num_iters[b] != -1 are skipped as a whole warp.
template <typename scalar_t, typename Rows = CsrRows>
__global__ void k_syndrome_check(
    const uint8_t*  __restrict__ e_v,       // [B, N_ext]
    Rows            rows,
    const scalar_t* __restrict__ syndrome,  // [B, M]
    const int64_t*  __restrict__ num_iters, // [B]
    int32_t*        __restrict__ mismatch,  // [B]  set to 1 on mismatch
    int B, int M, int N
) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= (int64_t)B * M) return;
    int lane = threadIdx.x & 31;

    int c     = (int)(row % M);
    int b     = (int)(row / M);
    if (num_iters[b] != -1) return;
    int parity = row_parity(e_v + (int64_t)b * (N + 1), rows, c, N, lane);
    if (lane == 0 && (scalar_t)parity != syndrome[(int64_t)b * M + c])
        mismatch[b] = 1;
}

// One thread per sample: a still-running sample with no mismatch converges at
// this iteration. Resets mismatch[b] to 0 for the next iteration's check.
__global__ void k_convergence_flag(
    int32_t* __restrict__ mismatch,   // [B]
    int64_t* __restrict__ num_iters,  // [B]  output
    int64_t* __restrict__ converges,  // [B]  output
    int64_t iter, int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || num_iters[b] != -1) return;
    if (mismatch[b]) { mismatch[b] = 0; return; }
    num_iters[b] = iter;
    converges[b] = 1;
}

// Whole BP-NMS decode for one sample per block, all iterations in one launch.
// Each iteration runs the per-step kernels' phases with the same device code
// (cn_row, llr_csr, row_parity) on the CSR layout, separated by __syncthreads:
// warp per check row for the CN update (l_v − b_c2v on the fly, u_init at
// iteration 1), threads over variables for the LLR sum and uint8 hard decision,
// warp per check row for the syndrome check into a block flag. b_c2v
// [B, nnz + 1] (zero on entry), l_v and e_v live in global memory. A block
// returns once its sample converges.
// cnt (nullable) turns on the rebatch cap with the per-step host check's
// semantics: after iteration i every still-running sample stops iff the number
// of samples converged at iterations <= i reaches stop_count. cnt holds two
// zeroed int32 counters per iteration, converged at i and finished i; a running
// block adds itself to finished i and spins until every live sample has
// finished i, so the count it reads is final. The spin needs all B blocks
// co-resident, which the host checks before launch.
// QUANT = true runs cn_row with the fixed-point rounding q, compiled for blocks
// of up to 1024 threads (the f64 instance would otherwise need more registers
// than a 1024-thread block has); QUANT = false sets no bound (0).
// Rows = PaddedRows runs the padded edge layout; F64INT as in cn_row.
template <typename scalar_t, bool QUANT = false, typename Rows = CsrRows, bool F64INT = true>
__global__ void __launch_bounds__(QUANT ? 1024 : 0) k_bp_nms_persistent(
    const scalar_t* __restrict__ u_init,    // [B, N_ext]
    const scalar_t* __restrict__ sneg,      // [B, M]  ±1 syndrome signs
    const scalar_t* __restrict__ syndrome,  // [B, M]
    Rows rows,
    const int32_t*  __restrict__ VN_eid,    // [N_ext, VD]
    scalar_t* b_c2v,                        // [B, nnz + 1], zero on entry
    scalar_t* l_v,                          // [B, N_ext], output
    uint8_t*  e_v,                          // [B, N_ext], output
    int64_t*  __restrict__ num_iters,       // [B], output
    int64_t*  __restrict__ converges,       // [B], output
    int B, int M, int64_t stride, int N, int VD, int max_iter,
    int* cnt,                               // [2 * (max_iter + 1)] zeroed, or nullptr
    int stop_count,
    Fxp q = Fxp{1.0, 1.0, 0.0, 0.0}
) {
    const int b = blockIdx.x, tid = threadIdx.x, bsz = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nw = bsz >> 5;
    const int N_ext = N + 1;
    const scalar_t* u  = u_init + (int64_t)b * N_ext;
    const scalar_t* sn = sneg + (int64_t)b * M;
    const scalar_t* sy = syndrome + (int64_t)b * M;
    scalar_t*       bc = b_c2v + (int64_t)b * stride;
    scalar_t*       lv = l_v + (int64_t)b * N_ext;
    uint8_t*        ev = e_v + (int64_t)b * N_ext;
    int* conv_at = cnt;                                  // samples converged at iteration i
    int* fin     = cnt ? cnt + (max_iter + 1) : nullptr; // samples that finished iteration i

    __shared__ int flag_s, stop_s;
    int conv_iter = 0, stop_iter = max_iter;
    int c_prev = 0;   // samples converged through the previous iteration (thread 0, cap only)

    for (int iter = 1; iter <= max_iter; iter++) {
        const double beta = 1.0 - ldexp(1.0, -iter);
        const scalar_t* l_row = iter == 1 ? u : lv;
        for (int c = warp; c < M; c += nw)
            cn_row<scalar_t, QUANT, Rows, true, F64INT>(l_row, sn + c, rows, bc, beta, c, lane, q);
        __syncthreads();

        for (int n = tid; n < N_ext; n += bsz) {
            scalar_t acc = llr_csr(bc, VN_eid, n, VD, u[n]);
            lv[n] = acc;
            ev[n] = (acc <= (scalar_t)0) ? 1 : 0;
        }
        // flag_s was last read after the previous iteration's final barrier.
        if (tid == 0) flag_s = 0;
        __syncthreads();

        for (int c = warp; c < M; c += nw) {
            int parity = row_parity(ev, rows, c, N, lane);
            if (lane == 0 && (scalar_t)parity != sy[c]) flag_s = 1;
        }
        __syncthreads();
        const int mism = flag_s;

        if (cnt != nullptr) {
            if (tid == 0) {
                int stop = 0;
                if (!mism) atomicAdd(&conv_at[iter], 1);
                __threadfence();
                atomicAdd(&fin[iter], 1);
                if (mism) {
                    const int alive = B - c_prev;
                    while (atomicAdd(&fin[iter], 0) < alive) { }
                    __threadfence();
                    c_prev += atomicAdd(&conv_at[iter], 0);
                    stop = c_prev >= stop_count;
                }
                stop_s = stop;
            }
            __syncthreads();
        }
        if (!mism) { conv_iter = iter; break; }
        if (cnt != nullptr && stop_s) { stop_iter = iter; break; }
    }
    if (tid == 0) {
        num_iters[b] = conv_iter ? (int64_t)conv_iter : (int64_t)stop_iter;
        converges[b] = conv_iter ? 1 : 0;
    }
}

// Calls f(std::true_type{}) if v, else f(std::false_type{}), so a runtime flag
// picks a template instance.
template <typename F>
static void with_bool(bool v, F&& f) {
    if (v) f(std::true_type{}); else f(std::false_type{});
}

// Per-step BP-NMS. col [nnz] selects the CSR edge layout (CsrRows), col [M, D]
// the padded one (PaddedRows); b_c2v is [B, col.numel() + 1]. warp, f64_int and
// from_lv pick the k_cn_update instance (WARP, F64INT, FROM_LV); they must keep
// their defaults when the fixed-point rounding is on.
void vn_cn_update_csr_cuda(
    torch::Tensor l_v,             // [B, N_ext], or a_v2c [B, col.numel() + 1] if !from_lv
    torch::Tensor syndrome_neg_bc, // [B, M]
    torch::Tensor row_ptr,         // [M + 1]  int32
    torch::Tensor col,             // [nnz] or [M, D]  int32
    torch::Tensor b_c2v,           // [B, col.numel() + 1]  previous CN→VN, overwritten in place
    torch::Tensor num_iters,       // [B]  int64, samples != -1 are skipped
    double beta,
    int64_t N,
    int64_t int_width = -1,        // >= 0: fixed-point rounding Q(int_width).(frac_width)
    int64_t frac_width = 0,
    bool warp = true,
    bool f64_int = true,
    bool from_lv = true
) {
    CHECK_INPUT(l_v); CHECK_INPUT(syndrome_neg_bc); CHECK_INPUT(row_ptr); CHECK_INPUT(col);
    CHECK_INPUT(b_c2v); CHECK_INPUT(num_iters);
    TORCH_CHECK(row_ptr.scalar_type() == at::kInt && col.scalar_type() == at::kInt,
                "row_ptr and col must be int32");
    TORCH_CHECK(b_c2v.size(1) == col.numel() + 1, "b_c2v must be [B, col.numel() + 1]");
    const int64_t width = from_lv ? N + 1 : b_c2v.size(1);
    TORCH_CHECK(l_v.size(1) == width, "l_v must be [B, ", width, "], got width ", l_v.size(1));
    const bool padded = col.dim() == 2, quant = int_width >= 0;
    TORCH_CHECK(!quant || (warp && f64_int && from_lv && !padded),
                "the fixed-point rounding runs with the default kernel knobs only");
    int B = (int)b_c2v.size(0), M = (int)row_ptr.size(0) - 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        b_c2v.scalar_type(), "vn_cn_update_csr_cuda",
    [&]() {
        auto launch = [&](auto kernel, auto rows, int64_t threads, Fxp q) {
            kernel<<<grid1d(threads), THREADS, 0, stream>>>(
                l_v.data_ptr<scalar_t>(),
                syndrome_neg_bc.data_ptr<scalar_t>(),
                rows,
                b_c2v.data_ptr<scalar_t>(),
                num_iters.data_ptr<int64_t>(),
                beta,
                B, M, b_c2v.size(1), (int)N, q
            );
        };
        const CsrRows csr{row_ptr.data_ptr<int32_t>(), col.data_ptr<int32_t>()};
        const Fxp none{1.0, 1.0, 0.0, 0.0};
        if (quant) {
            launch(k_cn_update<scalar_t, true>, csr, (int64_t)B * M * 32,
                   make_fxp(int_width, frac_width));
            return;
        }
        with_bool(warp, [&](auto W) { with_bool(f64_int, [&](auto I) { with_bool(from_lv, [&](auto L) {
            constexpr bool w = decltype(W)::value, l = decltype(L)::value;
            // F64INT changes only double; other types take the F64INT = true instance.
            constexpr bool fi = decltype(I)::value || !std::is_same<scalar_t, double>::value;
            const int64_t threads = (int64_t)B * M * (w ? 32 : 1);
            if (padded)
                launch(k_cn_update<scalar_t, false, PaddedRows, w, fi, l>,
                       PaddedRows{col.data_ptr<int32_t>(), (int)col.size(1), (int)N}, threads, none);
            else
                launch(k_cn_update<scalar_t, false, CsrRows, w, fi, l>, csr, threads, none);
        }); }); });
    });
}

// a_v2c[b, e] = l_v[b, col[e]] − b_c2v[b, e] for every edge e of col (either
// layout), skipping samples with num_iters != -1 (fuse_vn: false).
void vn_update_csr_cuda(
    torch::Tensor l_v,       // [B, N_ext]
    torch::Tensor b_c2v,     // [B, col.numel() + 1]
    torch::Tensor col,       // [nnz] or [M, D]  int32
    torch::Tensor a_v2c,     // [B, col.numel() + 1]  output
    torch::Tensor num_iters, // [B]  int64
    int64_t N
) {
    CHECK_INPUT(l_v); CHECK_INPUT(b_c2v); CHECK_INPUT(col); CHECK_INPUT(a_v2c);
    CHECK_INPUT(num_iters);
    TORCH_CHECK(col.scalar_type() == at::kInt, "col must be int32");
    TORCH_CHECK(l_v.size(1) == N + 1, "l_v must be [B, N + 1]");
    TORCH_CHECK(b_c2v.size(1) == col.numel() + 1 && a_v2c.sizes() == b_c2v.sizes(),
                "b_c2v and a_v2c must be [B, col.numel() + 1]");
    int B = (int)b_c2v.size(0);
    const int64_t n_edges = col.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        b_c2v.scalar_type(), "vn_update_csr_cuda",
    [&]() {
        k_vn_update_csr<scalar_t><<<grid1d((int64_t)B * n_edges), THREADS, 0, stream>>>(
            l_v.data_ptr<scalar_t>(), b_c2v.data_ptr<scalar_t>(), col.data_ptr<int32_t>(),
            a_v2c.data_ptr<scalar_t>(), num_iters.data_ptr<int64_t>(),
            B, n_edges, b_c2v.size(1), (int)N);
    });
}

void llr_hard_update_csr_cuda(
    torch::Tensor u_init,    // [B, N_ext]
    torch::Tensor b_c2v,     // [B, nnz + 1]
    torch::Tensor VN_eid,    // [N_ext, VD]  int32 flat edge ids, padded with nnz
    torch::Tensor l_v,       // [B, N_ext]  modified in-place
    torch::Tensor e_v,       // [B, N_ext]  uint8, modified in-place
    torch::Tensor num_iters  // [B]  int64, samples != -1 are skipped
) {
    CHECK_INPUT(u_init); CHECK_INPUT(b_c2v); CHECK_INPUT(VN_eid);
    CHECK_INPUT(l_v); CHECK_INPUT(e_v); CHECK_INPUT(num_iters);
    TORCH_CHECK(e_v.scalar_type() == at::kByte, "e_v must be uint8");
    TORCH_CHECK(VN_eid.scalar_type() == at::kInt, "VN_eid must be int32");
    int B     = (int)l_v.size(0);
    int N_ext = (int)l_v.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "llr_hard_update_csr_cuda",
    [&]() {
        k_llr_hard_update_csr<scalar_t><<<grid1d((int64_t)B * N_ext), THREADS, 0, stream>>>(
            u_init.data_ptr<scalar_t>(),
            b_c2v.data_ptr<scalar_t>(),
            VN_eid.data_ptr<int32_t>(),
            l_v.data_ptr<scalar_t>(),
            e_v.data_ptr<uint8_t>(),
            num_iters.data_ptr<int64_t>(),
            B, b_c2v.size(1), N_ext, (int)VN_eid.size(1)
        );
    });
}

// vn_gather: false, the ablation baseline for llr_hard_update_csr: sum (zeroed
// here) gets one atomicAdd per edge of col (either layout), then l_v = sum +
// u_init and the uint8 hard decision, skipping samples with num_iters != -1.
void llr_hard_update_atomic_cuda(
    torch::Tensor u_init,    // [B, N_ext]
    torch::Tensor b_c2v,     // [B, col.numel() + 1]
    torch::Tensor col,       // [nnz] or [M, D]  int32
    torch::Tensor sum,       // [B, N_ext]  scratch
    torch::Tensor l_v,       // [B, N_ext]  modified in-place
    torch::Tensor e_v,       // [B, N_ext]  uint8, modified in-place
    torch::Tensor num_iters  // [B]  int64, samples != -1 are skipped
) {
    CHECK_INPUT(u_init); CHECK_INPUT(b_c2v); CHECK_INPUT(col); CHECK_INPUT(sum);
    CHECK_INPUT(l_v); CHECK_INPUT(e_v); CHECK_INPUT(num_iters);
    TORCH_CHECK(e_v.scalar_type() == at::kByte, "e_v must be uint8");
    TORCH_CHECK(col.scalar_type() == at::kInt, "col must be int32");
    TORCH_CHECK(b_c2v.size(1) == col.numel() + 1, "b_c2v must be [B, col.numel() + 1]");
    TORCH_CHECK(sum.sizes() == l_v.sizes(), "sum must be [B, N_ext]");
    int B = (int)l_v.size(0), N_ext = (int)l_v.size(1);
    const int64_t n_edges = col.numel();
    sum.zero_();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "llr_hard_update_atomic_cuda",
    [&]() {
        k_vn_sum_atomic<scalar_t><<<grid1d((int64_t)B * n_edges), THREADS, 0, stream>>>(
            b_c2v.data_ptr<scalar_t>(), col.data_ptr<int32_t>(), sum.data_ptr<scalar_t>(),
            num_iters.data_ptr<int64_t>(), B, n_edges, b_c2v.size(1), N_ext);
        k_llr_hard_from_sum<scalar_t><<<grid1d((int64_t)B * N_ext), THREADS, 0, stream>>>(
            u_init.data_ptr<scalar_t>(), sum.data_ptr<scalar_t>(), l_v.data_ptr<scalar_t>(),
            e_v.data_ptr<uint8_t>(), num_iters.data_ptr<int64_t>(), B, N_ext);
    });
}

// col [nnz] selects the CSR edge layout, col [M, D] the padded one.
void syndrome_check_csr_cuda(
    torch::Tensor e_v,       // [B, N_ext]  uint8
    torch::Tensor row_ptr,   // [M + 1]  int32
    torch::Tensor col,       // [nnz] or [M, D]  int32
    torch::Tensor syndrome,  // [B, M]
    torch::Tensor num_iters, // [B]  int64
    torch::Tensor mismatch,  // [B]  int32, set to 1 on mismatch
    int64_t N
) {
    CHECK_INPUT(e_v); CHECK_INPUT(row_ptr); CHECK_INPUT(col); CHECK_INPUT(syndrome);
    CHECK_INPUT(num_iters); CHECK_INPUT(mismatch);
    TORCH_CHECK(e_v.scalar_type() == at::kByte, "e_v must be uint8");
    TORCH_CHECK(mismatch.scalar_type() == at::kInt, "mismatch must be int32");
    int B = (int)syndrome.size(0), M = (int)syndrome.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        syndrome.scalar_type(), "syndrome_check_csr_cuda",
    [&]() {
        auto launch = [&](auto kernel, auto rows) {
            kernel<<<grid1d((int64_t)B * M * 32), THREADS, 0, stream>>>(
                e_v.data_ptr<uint8_t>(),
                rows,
                syndrome.data_ptr<scalar_t>(),
                num_iters.data_ptr<int64_t>(),
                mismatch.data_ptr<int32_t>(),
                B, M, (int)N
            );
        };
        if (col.dim() == 2)
            launch(k_syndrome_check<scalar_t, PaddedRows>,
                   PaddedRows{col.data_ptr<int32_t>(), (int)col.size(1), (int)N});
        else
            launch(k_syndrome_check<scalar_t, CsrRows>,
                   CsrRows{row_ptr.data_ptr<int32_t>(), col.data_ptr<int32_t>()});
    });
}

void convergence_flag_update_cuda(
    torch::Tensor mismatch,  // [B]  int32, reset to 0 in-place
    torch::Tensor num_iters, // [B]  int64, modified in-place
    torch::Tensor converges, // [B]  int64, modified in-place
    int64_t iter
) {
    CHECK_INPUT(mismatch); CHECK_INPUT(num_iters); CHECK_INPUT(converges);
    TORCH_CHECK(mismatch.scalar_type() == at::kInt, "mismatch must be int32");
    int B = (int)num_iters.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    k_convergence_flag<<<grid1d(B), THREADS, 0, stream>>>(
        mismatch.data_ptr<int32_t>(),
        num_iters.data_ptr<int64_t>(),
        converges.data_ptr<int64_t>(),
        iter, B
    );
}

// Blocks of the persistent kernel instance the device holds at once at block_size threads.
template <typename K>
static int64_t persistent_coresident(K kernel, int block_size) {
    int dev = 0, sms = 0, per_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block_size, 0);
    return (int64_t)per_sm * sms;
}

// Calls f(kernel, padded tag) with the k_bp_nms_persistent<scalar_t, QUANT,
// Rows, F64INT> instance for the flags; quant needs padded false and f64_int true.
template <typename scalar_t, typename F>
static void with_persistent(bool quant, bool padded, bool f64_int, F&& f) {
    TORCH_CHECK(!quant || (!padded && f64_int),
                "the fixed-point rounding runs with the default kernel knobs only");
    if (quant) { f(k_bp_nms_persistent<scalar_t, true>, std::false_type{}); return; }
    with_bool(padded, [&](auto P) { with_bool(f64_int, [&](auto I) {
        using Rows = typename std::conditional<decltype(P)::value, PaddedRows, CsrRows>::type;
        // F64INT changes only double; other types take the F64INT = true instance.
        constexpr bool fi = decltype(I)::value || !std::is_same<scalar_t, double>::value;
        f(k_bp_nms_persistent<scalar_t, false, Rows, fi>, P);
    }); });
}

// Co-resident block count of the persistent kernel for like's dtype, with the
// fixed-point rounding on if quant, the padded edge layout if padded and the
// floating-point double compares if not f64_int.
int64_t persistent_max_blocks(torch::Tensor like, int64_t block_size, bool quant = false,
                              bool padded = false, bool f64_int = true) {
    int64_t n = 0;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        like.scalar_type(), "persistent_max_blocks",
    [&]() {
        with_persistent<scalar_t>(quant, padded, f64_int, [&](auto kernel, auto) {
            n = persistent_coresident(kernel, (int)block_size);
        });
    });
    return n;
}

// Full BP-NMS decode in one launch of k_bp_nms_persistent, one block per sample.
// Pass cnt (int32 [2 * (max_iter + 1)], zeroed) and stop_count to turn on the
// rebatch cap; the capped launch is refused unless all B blocks are co-resident.
// col [nnz] selects the CSR edge layout, col [M, D] the padded one (VN_eid and
// b_c2v then index the [B, M * D + 1] padded edge buffer).
void bp_nms_persistent_cuda(
    torch::Tensor u_init,          // [B, N_ext]
    torch::Tensor syndrome_neg_bc, // [B, M]
    torch::Tensor syndrome,        // [B, M]
    torch::Tensor row_ptr,         // [M + 1]  int32
    torch::Tensor col,             // [nnz] or [M, D]  int32
    torch::Tensor VN_eid,          // [N_ext, VD]  int32
    torch::Tensor b_c2v,           // [B, col.numel() + 1]  zeroed, used as scratch
    torch::Tensor l_v,             // [B, N_ext]  output
    torch::Tensor e_v,             // [B, N_ext]  uint8, output
    torch::Tensor num_iters,       // [B]  int64, output
    torch::Tensor converges,       // [B]  int64, output
    int64_t N, int64_t max_iter, int64_t block_size,
    c10::optional<torch::Tensor> cnt = c10::nullopt,
    int64_t stop_count = 0,
    int64_t int_width = -1,        // >= 0: fixed-point rounding Q(int_width).(frac_width)
    int64_t frac_width = 0,
    bool f64_int = true
) {
    CHECK_INPUT(u_init); CHECK_INPUT(syndrome_neg_bc); CHECK_INPUT(syndrome);
    CHECK_INPUT(row_ptr); CHECK_INPUT(col); CHECK_INPUT(VN_eid); CHECK_INPUT(b_c2v);
    CHECK_INPUT(l_v); CHECK_INPUT(e_v); CHECK_INPUT(num_iters); CHECK_INPUT(converges);
    TORCH_CHECK(row_ptr.scalar_type() == at::kInt && col.scalar_type() == at::kInt &&
                VN_eid.scalar_type() == at::kInt, "row_ptr, col and VN_eid must be int32");
    TORCH_CHECK(e_v.scalar_type() == at::kByte, "e_v must be uint8");
    TORCH_CHECK(b_c2v.size(1) == col.numel() + 1, "b_c2v must be [B, col.numel() + 1]");
    TORCH_CHECK(block_size % 32 == 0 && block_size > 0 && block_size <= 1024,
                "block_size must be a multiple of 32 in [32, 1024]");
    int* cnt_ptr = nullptr;
    if (cnt.has_value()) {
        CHECK_INPUT(cnt.value());
        TORCH_CHECK(cnt.value().scalar_type() == at::kInt &&
                    cnt.value().numel() >= 2 * (max_iter + 1),
                    "cnt must be int32 with at least 2 * (max_iter + 1) entries");
        cnt_ptr = cnt.value().data_ptr<int>();
    }
    int B = (int)u_init.size(0), M = (int)row_ptr.size(0) - 1;
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "bp_nms_persistent_cuda",
    [&]() {
        const bool quant = int_width >= 0;
        with_persistent<scalar_t>(quant, col.dim() == 2, f64_int, [&](auto kernel, auto P) {
            if (cnt_ptr) {
                const int64_t fit = persistent_coresident(kernel, (int)block_size);
                TORCH_CHECK(B <= fit, "capped persistent launch needs all ", B,
                            " blocks co-resident, the device holds ", fit, " at ",
                            block_size, " threads");
            }
            auto launch = [&](auto rows, Fxp q) {
                kernel<<<B, (int)block_size, 0, stream>>>(
                    u_init.data_ptr<scalar_t>(),
                    syndrome_neg_bc.data_ptr<scalar_t>(),
                    syndrome.data_ptr<scalar_t>(),
                    rows,
                    VN_eid.data_ptr<int32_t>(),
                    b_c2v.data_ptr<scalar_t>(),
                    l_v.data_ptr<scalar_t>(),
                    e_v.data_ptr<uint8_t>(),
                    num_iters.data_ptr<int64_t>(),
                    converges.data_ptr<int64_t>(),
                    B, M, b_c2v.size(1), (int)N, (int)VN_eid.size(1), (int)max_iter,
                    cnt_ptr, (int)stop_count, q
                );
            };
            const Fxp q = quant ? make_fxp(int_width, frac_width) : Fxp{1.0, 1.0, 0.0, 0.0};
            if constexpr (decltype(P)::value)
                launch(PaddedRows{col.data_ptr<int32_t>(), (int)col.size(1), (int)N}, q);
            else
                launch(CsrRows{row_ptr.data_ptr<int32_t>(), col.data_ptr<int32_t>()}, q);
        });
    });
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bp_nms_persistent launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vn_cn_update_csr",
          &vn_cn_update_csr_cuda,
          "VN + CN update in one pass on the CSR edge layout: b_c2v ← β · sign_out · "
          "2-min(l_v[col] − b_c2v), in place, skipping samples with num_iters != -1; "
          "int_width >= 0 turns on the fixed-point rounding of bp_norm_min_sum_quant (CUDA)",
          py::arg("l_v"), py::arg("syndrome_neg_bc"), py::arg("row_ptr"), py::arg("col"),
          py::arg("b_c2v"), py::arg("num_iters"), py::arg("beta"), py::arg("N"),
          py::arg("int_width") = -1, py::arg("frac_width") = 0, py::arg("warp") = true,
          py::arg("f64_int") = true, py::arg("from_lv") = true);

    m.def("vn_update_csr",
          &vn_update_csr_cuda,
          "VN→CN messages into their own edge buffer: a_v2c ← l_v[col] − b_c2v, "
          "skipping samples with num_iters != -1 (CUDA)",
          py::arg("l_v"), py::arg("b_c2v"), py::arg("col"), py::arg("a_v2c"),
          py::arg("num_iters"), py::arg("N"));

    m.def("llr_hard_update_atomic",
          &llr_hard_update_atomic_cuda,
          "LLR accumulation with one atomicAdd per edge, then the uint8 hard "
          "decision, skipping samples with num_iters != -1 (CUDA)",
          py::arg("u_init"), py::arg("b_c2v"), py::arg("col"), py::arg("sum"),
          py::arg("l_v"), py::arg("e_v"), py::arg("num_iters"));

    m.def("llr_hard_update_csr",
          &llr_hard_update_csr_cuda,
          "LLR accumulation and uint8 hard decision in one pass on the CSR edge layout, "
          "skipping samples with num_iters != -1 (CUDA)");

    m.def("syndrome_check_csr",
          &syndrome_check_csr_cuda,
          "Per-sample mismatch flag on the CSR edge layout: mismatch[b] ← 1 if "
          "parity(e_v[col]) ≠ syndrome, skipping samples with num_iters != -1 (CUDA)");

    m.def("convergence_flag_update",
          &convergence_flag_update_cuda,
          "Record convergence of samples with mismatch = 0, reset mismatch (CUDA)");

    m.def("bp_nms_persistent",
          &bp_nms_persistent_cuda,
          "Full BP-NMS loop in one launch, one block per sample, on the CSR edge layout; "
          "pass cnt + stop_count to turn on the rebatch cap, int_width >= 0 to turn on "
          "the fixed-point rounding of bp_norm_min_sum_quant (CUDA)",
          py::arg("u_init"), py::arg("syndrome_neg_bc"), py::arg("syndrome"),
          py::arg("row_ptr"), py::arg("col"), py::arg("VN_eid"), py::arg("b_c2v"),
          py::arg("l_v"), py::arg("e_v"), py::arg("num_iters"), py::arg("converges"),
          py::arg("N"), py::arg("max_iter"), py::arg("block_size"),
          py::arg("cnt") = py::none(), py::arg("stop_count") = 0,
          py::arg("int_width") = -1, py::arg("frac_width") = 0, py::arg("f64_int") = true);

    m.def("persistent_max_blocks",
          &persistent_max_blocks,
          "Blocks of the persistent kernel the device holds at once, for like's dtype "
          "and block_size threads, with the fixed-point rounding on if quant",
          py::arg("like"), py::arg("block_size"), py::arg("quant") = false,
          py::arg("padded") = false, py::arg("f64_int") = true);
}
