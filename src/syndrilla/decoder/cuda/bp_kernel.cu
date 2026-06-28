/*
 * bp_kernel.cu — shared CUDA kernels for the BP decoders.
 *
 * Lives in decoder/ (not under any one decoder) because every CUDA decoder reuses
 * it: bp_norm_min_sum_cuda plus the lottery / quant / relay / branch *_cuda ports.
 *
 * Two execution paths share this file:
 *
 *  A. FUSED PATH (default, fastest) — k_bp_nms_fused runs the entire decoding
 *     loop for one batch sample inside a single thread block, with all working
 *     state in shared memory and per-block early termination. One kernel launch
 *     decodes the whole batch.
 *
 *  B. PER-STEP PATH (debug / fallback) — seven modular kernels, one per logical
 *     step of the decoding pipeline, driven by a Python loop:
 *
 *      1. k_init_messages      — first-iteration gather: a_v2c ← u_init[V_c_col]
 *      2. k_vn_update          — variable-node update:   a_v2c ← l_v[V_c_col] − b_c2v
 *      3. k_cn_update          — check-node update:      b_c2v ← β · sign_out · 2-min
 *      4. k_llr_update         — LLR accumulation:       l_v   ← u_init + Σ b_c2v
 *      5. k_hard_decision      — threshold:              e_v   ← (l_v ≤ 0)
 *      6. k_syndrome_est       — parity check:           s_est ← (Σ e_v[V_c_col]) mod 2
 *      7. k_convergence_update — per-sample convergence snapshot
 *
 * Tensor layout conventions
 * ─────────────────────────
 *   B   = batch size
 *   M   = number of check nodes (rows of H)
 *   N   = number of variable nodes (cols of H, excluding dummy)
 *   N_ext = N + 1  (the "+1" is the dummy column used to pad irregular H rows)
 *   D   = maximum check-node degree (columns in the padded V_c_col / V_c_row tables)
 *   VD  = maximum variable-node degree (columns in the precomputed VN_adj tables)
 *
 *   V_c_col[c, k]  — column (variable) index of the k-th edge of check node c.
 *                    Padding entries store the value N (the dummy variable index).
 *   VN_adj_c[n,vd] — check-node index of the vd-th edge of variable n. -1 = pad.
 *   VN_adj_k[n,vd] — degree slot k of that same edge, for indexing into b_c2v.
 *
 * Precision note
 * ──────────────
 *   All arithmetic is performed in scalar_t (the tensor dtype). Earlier
 *   revisions tracked the two minima in `double`, which forced fp64
 *   instructions into the float32 path — on consumer GPUs (fp64 at 1/32–1/64
 *   throughput) that single detail made the float32 kernels as slow as the
 *   float64 ones. Keeping everything in scalar_t also matches the PyTorch
 *   reference bit-for-bit, which compares/sorts in the tensor dtype.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

// ─── helpers ──────────────────────────────────────────────────────────────────

static inline int grid1d(int n) {
    return (n + THREADS - 1) / THREADS;
}

// Every kernel indexes raw data pointers assuming row-major contiguous layout.
// A transposed-stride tensor (e.g. a syndrome built via (H @ e.T).T) would be
// silently read as other samples' data — fail loudly instead.
#define CHECK_INPUT(x)                                                  \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");            \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 1 — init_messages
//
// On the first iteration only. Gathers the initial channel LLR (u_init) into
// the edge message buffer.
//
//   a_v2c[b, c, k] = u_init[b, V_c_col[c, k]]   if V_c_col[c,k] < N  (real edge)
//                  = 0                            if V_c_col[c,k] == N (dummy edge)
//
// Thread assignment: one thread per (b, c, k) edge — flat index over B·M·D.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_init_messages(
    const scalar_t* __restrict__ u_init,   // [B, N_ext]
    const int64_t*  __restrict__ V_c_col,  // [M, D]
    scalar_t*       __restrict__ a_v2c,    // [B, M, D]  — output
    int B, int M, int D, int N             // N = H_shape[1], not N_ext
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M * D) return;

    int k = idx % D;
    int c = (idx / D) % M;
    int b = idx / (D * M);

    int n     = (int)V_c_col[c * D + k];
    int N_ext = N + 1;
    // Dummy edges (n == N) are zeroed so they don't perturb the CN update.
    a_v2c[b * M * D + c * D + k] =
        (n < N) ? u_init[b * N_ext + n] : (scalar_t)0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 2 — vn_update
//
// Iterations 2, 3, …  Computes the extrinsic VN→CN message by subtracting
// the previous incoming CN→VN message from the updated LLR.
//
//   a_v2c[b, c, k] = l_v[b, V_c_col[c,k]] − b_c2v[b, c, k]   (real edge)
//                  = 0                                          (dummy edge)
//
// Thread assignment: one thread per (b, c, k).
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_vn_update(
    const scalar_t* __restrict__ l_v,      // [B, N_ext]  — updated LLR
    const scalar_t* __restrict__ b_c2v,    // [B, M, D]   — previous CN→VN
    const int64_t*  __restrict__ V_c_col,  // [M, D]
    scalar_t*       __restrict__ a_v2c,    // [B, M, D]   — output
    int B, int M, int D, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M * D) return;

    int k = idx % D;
    int c = (idx / D) % M;
    int b = idx / (D * M);

    int n = (int)V_c_col[c * D + k];
    if (n >= N) {
        a_v2c[b * M * D + c * D + k] = (scalar_t)0;
        return;
    }
    int N_ext = N + 1;
    a_v2c[b * M * D + c * D + k] =
        l_v[b * N_ext + n] - b_c2v[b * M * D + c * D + k];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 3 — cn_update
//
// Normalized Min-Sum check-node update. One thread per (b, c) pair.
// Two sequential passes over the D edges of check node c:
//
//   Pass 1 — accumulate sign product and track the two smallest |a_v2c| values.
//   Pass 2 — write outgoing message for each edge:
//               b_c2v[b,c,k] = β · (s_neg[b,c] · sign_prod · sign(a[k])) · min_result[k]
//
//   where:
//     sign_prod   = product of sign(a[b,c,j]) for all real j
//     s_neg[b,c]  = +1 if syndrome[b,c]==0, else −1
//     sign(a)·sign_prod·sign(a) = sign_prod/sign(a) = product excluding k's own sign
//     min_result  = second-smallest |a| if |a[k]|==smallest, else smallest
//
//   β = 1 − 2^{−iter}  (schedule-based normalization, passed from Python)
//
// Thread assignment: one thread per (b, c) — flat index over B·M.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_cn_update(
    const scalar_t* __restrict__ a_v2c,           // [B, M, D]
    const scalar_t* __restrict__ syndrome_neg_bc,  // [B, M]  — ±1 per check
    const int64_t*  __restrict__ V_c_col,          // [M, D]
    scalar_t*       __restrict__ b_c2v,             // [B, M, D]  — output
    double beta,
    int B, int M, int D, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;

    int c    = idx % M;
    int b    = idx / M;
    int base = b * M * D + c * D;

    // ── Pass 1: sign product + 2-minimum over real edges ─────────────────────
    // Minima are tracked in scalar_t, NOT double — see precision note above.
    scalar_t sign_prod = (scalar_t)1;
    scalar_t min0      = (scalar_t)INFINITY;   // smallest |a|
    scalar_t min1      = (scalar_t)INFINITY;   // second smallest |a|

    for (int k = 0; k < D; k++) {
        int n = (int)V_c_col[c * D + k];
        if (n >= N) continue;                        // skip dummy edges

        scalar_t val = a_v2c[base + k];
        // sign(0) maps to −1, matching torch.where(sgn == 0, −1, sgn).
        sign_prod = (val > (scalar_t)0) ? sign_prod : -sign_prod;

        scalar_t absval = (val < (scalar_t)0) ? -val : val;
        if (absval < min0) { min1 = min0; min0 = absval; }
        else if (absval < min1) { min1 = absval; }
    }

    scalar_t s_neg       = syndrome_neg_bc[b * M + c];
    scalar_t scaled_beta = (scalar_t)beta;

    // ── Pass 2: write normalized min-sum messages ─────────────────────────────
    for (int k = 0; k < D; k++) {
        int n = (int)V_c_col[c * D + k];
        if (n >= N) { b_c2v[base + k] = (scalar_t)0; continue; }

        scalar_t val    = a_v2c[base + k];
        scalar_t absval = (val < (scalar_t)0) ? -val : val;
        scalar_t sign_k = (val > (scalar_t)0) ? (scalar_t)1 : (scalar_t)-1;

        // Outgoing sign excludes k's own contribution:
        //   s_neg * sign_prod_all * sign_k  ≡  s_neg * (sign_prod / sign_k)
        //   Because sign_k² = 1, division == multiplication.
        scalar_t out_sign   = s_neg * sign_prod * sign_k;
        scalar_t min_result = (absval == min0) ? min1 : min0;

        b_c2v[base + k] = scaled_beta * out_sign * min_result;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 4 — llr_update
//
// Accumulates CN→VN messages into the per-variable LLR estimate.
//
//   l_v[b, n] = u_init[b, n]  +  Σ_{vd} b_c2v[b, VN_adj_c[n,vd], VN_adj_k[n,vd]]
//
// VN_adj_c and VN_adj_k are the precomputed transpose of V_c_col: for each
// variable node n, they list the (check index c, degree slot k) pairs of all
// edges incident on n. Entries with VN_adj_c[n,vd] == -1 are padding (no more
// edges for that variable).
//
// Thread assignment: one thread per (b, n) — flat index over B·N_ext.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_llr_update(
    const scalar_t* __restrict__ u_init,    // [B, N_ext]
    const scalar_t* __restrict__ b_c2v,     // [B, M, D]
    const int64_t*  __restrict__ VN_adj_c,  // [N_ext, VD]  check index, -1=pad
    const int64_t*  __restrict__ VN_adj_k,  // [N_ext, VD]  degree slot k, -1=pad
    scalar_t*       __restrict__ l_v,       // [B, N_ext]  — output
    int B, int M, int D, int N_ext, int VD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N_ext) return;

    int n = idx % N_ext;
    int b = idx / N_ext;

    // Sum the edge messages FIRST, then add the channel LLR last — this matches
    // the PyTorch reference's association l_v = u_init + scatter_add(b_c2v).
    // Floating-point addition is non-associative: seeding acc with u_init
    // regroups the sum and drifts ~1e-15 at float64, enough to flip the hard
    // decision on bits whose posterior LLR sits on the l_v<=0 boundary.
    scalar_t acc = (scalar_t)0;
    for (int vd = 0; vd < VD; vd++) {
        int c = (int)VN_adj_c[n * VD + vd];
        if (c < 0) break;                           // hit padding sentinel
        int k = (int)VN_adj_k[n * VD + vd];
        acc += b_c2v[b * M * D + c * D + k];
    }
    acc += u_init[b * N_ext + n];                   // channel LLR added last
    l_v[b * N_ext + n] = acc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 5 — hard_decision
//
//   e_v[b, n] = 1  if l_v[b, n] <= 0
//             = 0  otherwise
//
// Thread assignment: one thread per element — flat index over B·N_ext.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_hard_decision(
    const scalar_t* __restrict__ l_v,  // [B, N_ext]
    scalar_t*       __restrict__ e_v,  // [B, N_ext]  — output
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    e_v[idx] = (l_v[idx] <= (scalar_t)0) ? (scalar_t)1 : (scalar_t)0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 6 — syndrome_est
//
// Computes the estimated syndrome by XOR-ing hard decisions over check edges.
//
//   s_est[b, c] = ( Σ_k e_v[b, V_c_col[c,k]] ) mod 2
//
// Thread assignment: one thread per (b, c) — flat index over B·M.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_syndrome_est(
    const scalar_t* __restrict__ e_v,      // [B, N_ext]
    const int64_t*  __restrict__ V_c_col,  // [M, D]
    scalar_t*       __restrict__ s_est,    // [B, M]  — output
    int B, int M, int D, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;

    int c     = idx % M;
    int b     = idx / M;
    int N_ext = N + 1;
    int parity = 0;

    for (int k = 0; k < D; k++) {
        int n = (int)V_c_col[c * D + k];
        // Dummy variable (n == N) always has e_v = 0 (l_v[N] = inf > 0).
        if (n < N && e_v[b * N_ext + n] > (scalar_t)0.5f) parity ^= 1;
    }
    s_est[b * M + c] = (scalar_t)parity;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 7 — convergence_update
//
// For each batch sample b that has not yet converged (num_iters[b] == -1):
//   — Check whether s_est[b, c] == syndrome[b, c] for ALL check nodes c.
//   — If so: record num_iters[b] = iter, mark converges[b] = 1, and snapshot
//     e_v[b] → e_out[b], l_v[b] → l_out[b].
//
// Because the copy of N_ext scalars is done inside the thread, this kernel is
// efficient for small N_ext (typical QEC codes) and avoids a separate scatter
// kernel plus host-side bookkeeping.
//
// Thread assignment: one thread per b — flat index over B.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename scalar_t>
__global__ void k_convergence_update(
    const scalar_t* __restrict__ s_est,     // [B, M]
    const scalar_t* __restrict__ syndrome,  // [B, M]
    const scalar_t* __restrict__ e_v,       // [B, N_ext]
    const scalar_t* __restrict__ l_v,       // [B, N_ext]
    scalar_t*       __restrict__ e_out,     // [B, N_ext]  — output
    scalar_t*       __restrict__ l_out,     // [B, N_ext]  — output
    int64_t*        __restrict__ num_iters, // [B]         — output
    int64_t*        __restrict__ converges, // [B]         — output
    int64_t iter,
    int B, int M, int N_ext
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (num_iters[b] != -1) return;  // already converged in a prior iteration

    // Check all syndrome bits for this sample.
    for (int c = 0; c < M; c++) {
        if (s_est[b * M + c] != syndrome[b * M + c]) return;
    }

    // All checks satisfied — record convergence and snapshot buffers.
    num_iters[b] = iter;
    converges[b] = 1;
    int base = b * N_ext;
    for (int n = 0; n < N_ext; n++) {
        e_out[base + n] = e_v[base + n];
        l_out[base + n] = l_v[base + n];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 8 — k_bp_nms_fused
//
// Runs the ENTIRE BP Normalized Min-Sum decoding loop for one batch sample
// inside one thread block. Eliminates the three overheads that dominate the
// per-step path:
//   1. Python-level iteration loop      (O(max_iter) Python frames)
//   2. Per-iteration GPU→CPU syncs      (early-termination .item() checks)
//   3. Per-step kernel-launch latency   (7 launches × max_iter)
//
// Design
// ──────
//  • One block per sample (grid.x = B); stride-loops cover M·D edges,
//    N_ext variables, and M checks regardless of block size.
//  • ALL state lives in shared memory: the int64 adjacency tables are copied
//    once per block as int32 (halving on-chip footprint), the channel LLR
//    u_init is staged once, and the message/LLR buffers never touch global
//    memory inside the loop.
//  • The CN update is split into two phases so it parallelises over edges,
//    not just checks: phase A computes per-check sign products and the two
//    minima (M threads), phase B writes all M·D outgoing messages.
//  • Early termination: after each iteration the block computes the syndrome
//    of the current hard decision in shared memory and AND-reduces a match
//    flag. All threads see the same flag after __syncthreads(), so the break
//    is uniform and race-free. Converged samples exit immediately — the block
//    retires and frees the SM for the remaining samples.
//  • num_iters / converges are written by thread 0 at the end, giving the
//    exact same per-sample semantics as the PyTorch reference.
//
// Shared-memory layout (int32 region first, padded to 16 B, then scalar_t):
//   int32 : V_c_col[M·D] | VN_adj_c[N_ext·VD] | VN_adj_k[N_ext·VD] | synd[M] | flag[1]
//   scalar: a_v2c[M·D] | b_c2v[M·D] | l_v[N_ext] | e_v[N_ext] | u_init[N_ext]
//           | s_neg[M] | sign_prod[M] | min0[M] | min1[M]
// ═══════════════════════════════════════════════════════════════════════════════

// Shared-memory size of the int32 region, padded so the scalar_t region that
// follows is 16-byte aligned. Must match the host-side computation exactly.
__host__ __device__ static inline size_t fused_int_bytes(int E, int A, int M) {
    return (((size_t)(E + 2 * A + M + 1) * sizeof(int32_t)) + 15) & ~(size_t)15;
}

template <typename scalar_t>
__global__ void k_bp_nms_fused(
    const scalar_t* __restrict__ u_init_g,   // [B, N_ext]  channel LLR (with dummy ∞)
    const scalar_t* __restrict__ s_neg_g,    // [B, M]      ±1 syndrome signs
    const int64_t*  __restrict__ V_c_col,    // [M, D]      check→var adjacency
    const int64_t*  __restrict__ VN_adj_c,   // [N_ext, VD] var→check index (-1 = pad)
    const int64_t*  __restrict__ VN_adj_k,   // [N_ext, VD] var→check degree slot
    scalar_t* __restrict__ e_out_g,          // [B, N_ext]  output hard decisions
    scalar_t* __restrict__ l_out_g,          // [B, N_ext]  output LLRs
    int64_t*  __restrict__ num_iters_g,      // [B]         convergence iteration
    int64_t*  __restrict__ converges_g,      // [B]         1 = converged
    int M, int D, int N, int N_ext, int VD, int max_iter,
    int* __restrict__ g_conv_count,          // grid-wide converged counter (nullptr ⇒ uncapped)
    int stop_count                           // iter_speedup cap: stop once g_conv_count ≥ this
) {
    const int b   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int bsz = (int)blockDim.x;
    const int E   = M * D;        // padded edge count
    const int A   = N_ext * VD;   // padded var-adjacency count

    // ── Shared-memory carve-up (must mirror fused_int_bytes) ─────────────────
    extern __shared__ char _smem[];
    int32_t* vcol_s = (int32_t*)_smem;          // [E]
    int32_t* adjc_s = vcol_s + E;               // [A]
    int32_t* adjk_s = adjc_s + A;               // [A]
    int32_t* synd_s = adjk_s + A;               // [M]   0/1 target syndrome
    int32_t* flag_s = synd_s + M;               // [1]   mismatch flag

    scalar_t* a_v2c  = (scalar_t*)(_smem + fused_int_bytes(E, A, M));
    scalar_t* b_c2v  = a_v2c  + E;
    scalar_t* l_v    = b_c2v  + E;
    scalar_t* e_v    = l_v    + N_ext;
    scalar_t* u_s    = e_v    + N_ext;          // staged channel LLR
    scalar_t* sneg_s = u_s    + N_ext;          // ±1 syndrome signs
    scalar_t* sp_s   = sneg_s + M;              // per-check sign product
    scalar_t* mn0_s  = sp_s   + M;              // per-check smallest |a|
    scalar_t* mn1_s  = mn0_s  + M;              // per-check 2nd smallest |a|

    // ── Stage adjacency (as int32), syndrome and channel LLR into shared ─────
    const int bM  = b * M;
    const int bNe = b * N_ext;
    for (int i = tid; i < E; i += bsz) vcol_s[i] = (int32_t)V_c_col[i];
    for (int i = tid; i < A; i += bsz) {
        adjc_s[i] = (int32_t)VN_adj_c[i];
        adjk_s[i] = (int32_t)VN_adj_k[i];
    }
    for (int c = tid; c < M; c += bsz) {
        scalar_t sn = s_neg_g[bM + c];
        sneg_s[c] = sn;
        synd_s[c] = (sn < (scalar_t)0) ? 1 : 0;   // s_neg = −1 ⇔ syndrome bit 1
    }
    for (int n = tid; n < N_ext; n += bsz) u_s[n] = u_init_g[bNe + n];
    __syncthreads();

    // ── Decoding loop with per-block early termination ───────────────────────
    int conv_iter = 0;   // 0 = not converged; else the convergence iteration
    int stop_iter = max_iter;  // iterations this block actually ran (cap may stop it early)
    for (int iter = 1; iter <= max_iter; iter++) {
        const scalar_t beta = (scalar_t)(1.0 - ldexp(1.0, -iter));  // 1 − 2^{−iter}

        // Step 1 — VN update → a_v2c   (parallel over edges)
        for (int e = tid; e < E; e += bsz) {
            int n = vcol_s[e];
            if (n >= N) { a_v2c[e] = (scalar_t)0; continue; }
            a_v2c[e] = (iter == 1) ? u_s[n] : l_v[n] - b_c2v[e];
        }
        __syncthreads();

        // Step 2a — per-check sign product + two minima  (parallel over checks)
        for (int c = tid; c < M; c += bsz) {
            int base = c * D;
            scalar_t sp = (scalar_t)1;
            scalar_t m0 = (scalar_t)INFINITY, m1 = (scalar_t)INFINITY;
            for (int k = 0; k < D; k++) {
                if (vcol_s[base + k] >= N) continue;       // skip dummy edges
                scalar_t v = a_v2c[base + k];
                sp = (v > (scalar_t)0) ? sp : -sp;          // sign(0) → −1
                scalar_t av = (v < (scalar_t)0) ? -v : v;
                if (av < m0) { m1 = m0; m0 = av; }
                else if (av < m1) { m1 = av; }
            }
            sp_s[c] = sp; mn0_s[c] = m0; mn1_s[c] = m1;
        }
        __syncthreads();

        // Step 2b — outgoing CN→VN messages  (parallel over edges)
        for (int e = tid; e < E; e += bsz) {
            int n = vcol_s[e];
            if (n >= N) { b_c2v[e] = (scalar_t)0; continue; }
            int c = e / D;
            scalar_t v  = a_v2c[e];
            scalar_t av = (v < (scalar_t)0) ? -v : v;
            scalar_t sk = (v > (scalar_t)0) ? (scalar_t)1 : (scalar_t)-1;
            scalar_t mr = (av == mn0_s[c]) ? mn1_s[c] : mn0_s[c];
            b_c2v[e] = beta * sneg_s[c] * sp_s[c] * sk * mr;
        }
        __syncthreads();

        // Steps 3+4 — LLR update and hard decision, fused (parallel over vars).
        // Sum the edge messages FIRST, then add the channel LLR last, matching
        // the PyTorch reference (l_v = u_init + scatter_add(b_c2v)). Floating
        // addition is non-associative, so seeding acc with u_s[n] would regroup
        // the sum and drift ~1e-15 at float64 — enough to flip the hard decision
        // on bits whose posterior LLR sits on the l_v<=0 boundary.
        for (int n = tid; n < N_ext; n += bsz) {
            scalar_t acc = (scalar_t)0;
            int arow = n * VD;
            for (int vd = 0; vd < VD; vd++) {
                int c = adjc_s[arow + vd];
                if (c < 0) break;                          // padding sentinel
                acc += b_c2v[c * D + adjk_s[arow + vd]];
            }
            acc += u_s[n];                                 // channel LLR added last
            l_v[n] = acc;                                  // dummy var stays +inf
            e_v[n] = (acc <= (scalar_t)0) ? (scalar_t)1 : (scalar_t)0;
        }
        if (tid == 0) *flag_s = 0;
        __syncthreads();

        // Step 5 — syndrome check (parallel over checks). Any mismatching check
        // sets the shared flag; the racing writes all store the same value 1.
        for (int c = tid; c < M; c += bsz) {
            int base = c * D;
            int parity = 0;
            for (int k = 0; k < D; k++) {
                int n = vcol_s[base + k];
                if (n < N && e_v[n] > (scalar_t)0.5) parity ^= 1;
            }
            if (parity != synd_s[c]) *flag_s = 1;
        }
        __syncthreads();

        // Uniform decision: every thread reads the same flag after the barrier.
        if (*flag_s == 0) {
            conv_iter = iter;
            // Count this block toward the grid-wide converged total (capped runs).
            if (g_conv_count != nullptr && tid == 0) atomicAdd(g_conv_count, 1);
            break;
        }
        // (The flag is only reset between the next iteration's two barriers,
        //  so this read can never race with the reset.)

        // ── iter_speedup cap: stop this (still-unconverged) block early once the
        // batch as a whole has hit the learned converged fraction. The decision
        // MUST be uniform across the block, so thread 0 reads the grid-wide
        // counter once and broadcasts it via flag_s (free here until the next
        // iteration resets it at the top), and every thread decides after the
        // barrier. conv_iter stays 0 ⇒ this block is left converge==0 (the
        // deferred tail that main.py re-decodes uncapped). atomicAdd(_,0) forces
        // an L2-coherent read of writes from other blocks.
        if (g_conv_count != nullptr) {
            if (tid == 0) *flag_s = atomicAdd(g_conv_count, 0);
            __syncthreads();
            if (*flag_s >= stop_count) { stop_iter = iter; break; }  // ran `iter`, not max_iter
            __syncthreads();   // all threads done reading flag_s before it is reused
        }
    }

    // ── Write results to global memory ────────────────────────────────────────
    for (int n = tid; n < N_ext; n += bsz) {
        e_out_g[bNe + n] = e_v[n];
        l_out_g[bNe + n] = l_v[n];
    }
    if (tid == 0) {
        num_iters_g[b] = (conv_iter > 0) ? (int64_t)conv_iter : (int64_t)stop_iter;
        converges_g[b] = (conv_iter > 0) ? 1 : 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host dispatch functions — called from Python via pybind11
// Each function is a thin wrapper that performs type dispatch and launches the
// appropriate kernel template instance.
// ═══════════════════════════════════════════════════════════════════════════════

void init_messages_cuda(
    torch::Tensor u_init,   // [B, N_ext]
    torch::Tensor V_c_col,  // [M, D]  int64
    torch::Tensor a_v2c     // [B, M, D]  — modified in-place
) {
    CHECK_INPUT(u_init); CHECK_INPUT(V_c_col); CHECK_INPUT(a_v2c);
    int B     = a_v2c.size(0);
    int M     = a_v2c.size(1);
    int D     = a_v2c.size(2);
    int N_ext = (int)u_init.size(1);
    int N     = N_ext - 1;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "init_messages_cuda",
    [&]() {
        k_init_messages<scalar_t><<<grid1d(B * M * D), THREADS, 0, stream>>>(
            u_init.data_ptr<scalar_t>(),
            V_c_col.data_ptr<int64_t>(),
            a_v2c.data_ptr<scalar_t>(),
            B, M, D, N
        );
    });
}

void vn_update_cuda(
    torch::Tensor l_v,      // [B, N_ext]
    torch::Tensor b_c2v,    // [B, M, D]
    torch::Tensor V_c_col,  // [M, D]  int64
    torch::Tensor a_v2c,    // [B, M, D]  — modified in-place
    int64_t N               // H_shape[1]
) {
    CHECK_INPUT(l_v); CHECK_INPUT(b_c2v); CHECK_INPUT(V_c_col); CHECK_INPUT(a_v2c);
    int B = a_v2c.size(0), M = a_v2c.size(1), D = a_v2c.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        l_v.scalar_type(), "vn_update_cuda",
    [&]() {
        k_vn_update<scalar_t><<<grid1d(B * M * D), THREADS, 0, stream>>>(
            l_v.data_ptr<scalar_t>(),
            b_c2v.data_ptr<scalar_t>(),
            V_c_col.data_ptr<int64_t>(),
            a_v2c.data_ptr<scalar_t>(),
            B, M, D, (int)N
        );
    });
}

void cn_update_cuda(
    torch::Tensor a_v2c,           // [B, M, D]
    torch::Tensor syndrome_neg_bc, // [B, M]
    torch::Tensor V_c_col,         // [M, D]  int64
    torch::Tensor b_c2v,           // [B, M, D]  — modified in-place
    double beta,
    int64_t N
) {
    CHECK_INPUT(a_v2c); CHECK_INPUT(syndrome_neg_bc); CHECK_INPUT(V_c_col); CHECK_INPUT(b_c2v);
    int B = a_v2c.size(0), M = a_v2c.size(1), D = a_v2c.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a_v2c.scalar_type(), "cn_update_cuda",
    [&]() {
        k_cn_update<scalar_t><<<grid1d(B * M), THREADS, 0, stream>>>(
            a_v2c.data_ptr<scalar_t>(),
            syndrome_neg_bc.data_ptr<scalar_t>(),
            V_c_col.data_ptr<int64_t>(),
            b_c2v.data_ptr<scalar_t>(),
            beta,
            B, M, D, (int)N
        );
    });
}

void llr_update_cuda(
    torch::Tensor u_init,    // [B, N_ext]
    torch::Tensor b_c2v,     // [B, M, D]
    torch::Tensor VN_adj_c,  // [N_ext, VD]  int64
    torch::Tensor VN_adj_k,  // [N_ext, VD]  int64
    torch::Tensor l_v,       // [B, N_ext]  — modified in-place
    int64_t VD
) {
    CHECK_INPUT(u_init); CHECK_INPUT(b_c2v); CHECK_INPUT(VN_adj_c);
    CHECK_INPUT(VN_adj_k); CHECK_INPUT(l_v);
    int B     = (int)l_v.size(0);
    int N_ext = (int)l_v.size(1);
    int M     = (int)b_c2v.size(1);
    int D     = (int)b_c2v.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "llr_update_cuda",
    [&]() {
        k_llr_update<scalar_t><<<grid1d(B * N_ext), THREADS, 0, stream>>>(
            u_init.data_ptr<scalar_t>(),
            b_c2v.data_ptr<scalar_t>(),
            VN_adj_c.data_ptr<int64_t>(),
            VN_adj_k.data_ptr<int64_t>(),
            l_v.data_ptr<scalar_t>(),
            B, M, D, N_ext, (int)VD
        );
    });
}

void hard_decision_cuda(
    torch::Tensor l_v,  // [B, N_ext]
    torch::Tensor e_v   // [B, N_ext]  — modified in-place
) {
    CHECK_INPUT(l_v); CHECK_INPUT(e_v);
    int total = (int)l_v.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        l_v.scalar_type(), "hard_decision_cuda",
    [&]() {
        k_hard_decision<scalar_t><<<grid1d(total), THREADS, 0, stream>>>(
            l_v.data_ptr<scalar_t>(),
            e_v.data_ptr<scalar_t>(),
            total
        );
    });
}

void syndrome_est_cuda(
    torch::Tensor e_v,      // [B, N_ext]
    torch::Tensor V_c_col,  // [M, D]  int64
    torch::Tensor s_est,    // [B, M]  — modified in-place
    int64_t N
) {
    CHECK_INPUT(e_v); CHECK_INPUT(V_c_col); CHECK_INPUT(s_est);
    int B = (int)s_est.size(0), M = (int)s_est.size(1);
    int D = (int)V_c_col.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        e_v.scalar_type(), "syndrome_est_cuda",
    [&]() {
        k_syndrome_est<scalar_t><<<grid1d(B * M), THREADS, 0, stream>>>(
            e_v.data_ptr<scalar_t>(),
            V_c_col.data_ptr<int64_t>(),
            s_est.data_ptr<scalar_t>(),
            B, M, D, (int)N
        );
    });
}

void convergence_update_cuda(
    torch::Tensor s_est,     // [B, M]
    torch::Tensor syndrome,  // [B, M]
    torch::Tensor e_v,       // [B, N_ext]
    torch::Tensor l_v,       // [B, N_ext]
    torch::Tensor e_out,     // [B, N_ext]  — modified in-place
    torch::Tensor l_out,     // [B, N_ext]  — modified in-place
    torch::Tensor num_iters, // [B]  int64  — modified in-place
    torch::Tensor converges, // [B]  int64  — modified in-place
    int64_t iter
) {
    CHECK_INPUT(s_est); CHECK_INPUT(syndrome); CHECK_INPUT(e_v); CHECK_INPUT(l_v);
    CHECK_INPUT(e_out); CHECK_INPUT(l_out); CHECK_INPUT(num_iters); CHECK_INPUT(converges);
    int B     = (int)s_est.size(0);
    int M     = (int)s_est.size(1);
    int N_ext = (int)l_v.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        s_est.scalar_type(), "convergence_update_cuda",
    [&]() {
        k_convergence_update<scalar_t><<<grid1d(B), THREADS, 0, stream>>>(
            s_est.data_ptr<scalar_t>(),
            syndrome.data_ptr<scalar_t>(),
            e_v.data_ptr<scalar_t>(),
            l_v.data_ptr<scalar_t>(),
            e_out.data_ptr<scalar_t>(),
            l_out.data_ptr<scalar_t>(),
            num_iters.data_ptr<int64_t>(),
            converges.data_ptr<int64_t>(),
            iter, B, M, N_ext
        );
    });
}

// ── Fused-kernel dispatch ─────────────────────────────────────────────────────

// Full BP-NMS decode in one fused kernel. The capped and uncapped paths are the SAME
// kernel (k_bp_nms_fused), so this is ONE entry point: pass conv_count (a pre-zeroed
// int32 [1] device tensor) + stop_count to activate the iter_speedup cap — each block
// bumps the grid-wide counter on convergence and any still-running block stops once it
// reaches stop_count, deferring its (unconverged) sample to main.py's extra queue. Omit
// conv_count (None) to run uncapped: conv_count_ptr is then nullptr and the kernel skips
// every cap branch (`if (g_conv_count != nullptr)`). Entirely on-device, no host syncs.
void bp_nms_fused_cuda(
    torch::Tensor u_init,       // [B, N_ext]
    torch::Tensor syndrome_neg, // [B, M]   — ±1 syndrome signs
    torch::Tensor V_c_col,      // [M, D]  int64
    torch::Tensor VN_adj_c,     // [N_ext, VD]  int64
    torch::Tensor VN_adj_k,     // [N_ext, VD]  int64
    torch::Tensor e_out,        // [B, N_ext]  — written in-place
    torch::Tensor l_out,        // [B, N_ext]  — written in-place
    torch::Tensor num_iters,    // [B]  int64  — written in-place
    torch::Tensor converges,    // [B]  int64  — written in-place
    int64_t N, int64_t VD, int64_t max_iter, int64_t block_size,
    c10::optional<torch::Tensor> conv_count = c10::nullopt,  // [1] int32, or None ⇒ uncapped
    int64_t stop_count = 0                                    // cap threshold (ignored when uncapped)
) {
    CHECK_INPUT(u_init); CHECK_INPUT(syndrome_neg); CHECK_INPUT(V_c_col);
    CHECK_INPUT(VN_adj_c); CHECK_INPUT(VN_adj_k); CHECK_INPUT(e_out);
    CHECK_INPUT(l_out); CHECK_INPUT(num_iters); CHECK_INPUT(converges);

    // No counter ⇒ uncapped (kernel skips every cap branch); else activate the cap.
    int* conv_count_ptr = nullptr;
    if (conv_count.has_value()) {
        CHECK_INPUT(conv_count.value());
        conv_count_ptr = conv_count.value().data_ptr<int>();
    }

    int B     = (int)u_init.size(0);
    int N_ext = (int)u_init.size(1);
    int M     = (int)syndrome_neg.size(1);
    int D     = (int)V_c_col.size(1);
    int E     = M * D;
    int A     = N_ext * (int)VD;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_init.scalar_type(), "bp_nms_fused_cuda",
    [&]() {
        size_t smem_bytes = fused_int_bytes(E, A, M)
                          + (size_t)(2 * E + 3 * N_ext + 4 * M) * sizeof(scalar_t);

        // Blocks needing more than the default 48 KB static limit must opt in.
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(
                k_bp_nms_fused<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem_bytes);
        }

        k_bp_nms_fused<scalar_t><<<B, (int)block_size, smem_bytes, stream>>>(
            u_init.data_ptr<scalar_t>(),
            syndrome_neg.data_ptr<scalar_t>(),
            V_c_col.data_ptr<int64_t>(),
            VN_adj_c.data_ptr<int64_t>(),
            VN_adj_k.data_ptr<int64_t>(),
            e_out.data_ptr<scalar_t>(),
            l_out.data_ptr<scalar_t>(),
            num_iters.data_ptr<int64_t>(),
            converges.data_ptr<int64_t>(),
            M, D, (int)N, N_ext, (int)VD, (int)max_iter,
            conv_count_ptr, (int)stop_count
        );
    });
}

// Shared memory the fused kernel needs for a given problem size + dtype size.
// Lets Python decide up-front whether the fused path fits on this GPU.
int64_t fused_smem_bytes(int64_t M, int64_t D, int64_t N_ext, int64_t VD,
                         int64_t scalar_size) {
    int E = (int)(M * D);
    int A = (int)(N_ext * VD);
    return (int64_t)fused_int_bytes(E, A, (int)M)
         + (2 * M * D + 3 * N_ext + 4 * M) * scalar_size;
}

// Maximum opt-in dynamic shared memory per block on the current device.
int64_t fused_smem_limit() {
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 48 * 1024;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)v;
}

// ═══════════════════════════════════════════════════════════════════════════════
// pybind11 module registration
// ═══════════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_messages",
          &init_messages_cuda,
          "First-iteration gather: a_v2c ← u_init[V_c_col] (CUDA)");

    m.def("vn_update",
          &vn_update_cuda,
          "Variable-node update: a_v2c ← l_v[V_c_col] − b_c2v (CUDA)");

    m.def("cn_update",
          &cn_update_cuda,
          "Check-node update: b_c2v ← β · sign_out · 2-min(a_v2c) (CUDA)");

    m.def("llr_update",
          &llr_update_cuda,
          "LLR accumulation: l_v ← u_init + Σ b_c2v via variable adjacency (CUDA)");

    m.def("hard_decision",
          &hard_decision_cuda,
          "Hard decision: e_v ← (l_v ≤ 0) (CUDA)");

    m.def("syndrome_est",
          &syndrome_est_cuda,
          "Syndrome estimation: s_est ← parity(e_v[V_c_col]) (CUDA)");

    m.def("convergence_update",
          &convergence_update_cuda,
          "Per-sample convergence tracking and buffer snapshot (CUDA)");

    m.def("bp_nms_fused",
          &bp_nms_fused_cuda,
          "Full BP-NMS loop in one kernel with per-block early termination; pass "
          "conv_count + stop_count to activate the iter_speedup cap, omit for uncapped (CUDA)",
          py::arg("u_init"), py::arg("syndrome_neg"), py::arg("V_c_col"),
          py::arg("VN_adj_c"), py::arg("VN_adj_k"), py::arg("e_out"),
          py::arg("l_out"), py::arg("num_iters"), py::arg("converges"),
          py::arg("N"), py::arg("VD"), py::arg("max_iter"), py::arg("block_size"),
          py::arg("conv_count") = py::none(), py::arg("stop_count") = 0);

    m.def("fused_smem_bytes",
          &fused_smem_bytes,
          "Dynamic shared memory (bytes) the fused kernel needs for a problem size");

    m.def("fused_smem_limit",
          &fused_smem_limit,
          "Maximum opt-in dynamic shared memory per block on the current device");
}
