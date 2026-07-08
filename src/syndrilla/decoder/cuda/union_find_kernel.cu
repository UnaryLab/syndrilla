/*
 * union_find_kernel.cu -- standalone CUDA Union-Find decoder for the
 * union_find_cuda extension.
 *
 * One file per decoder kernel, matching bp_kernel.cu / osd0_kernel.cu. This is a
 * from-scratch GPU port of the batched-tensor Union-Find decoder in
 * decoder/union_find/union_find.py -- it reproduces that decoder's forward() output
 * bit-for-bit (e-diff == 0), NOT the C++ chaeyeunpark reference. It borrows nothing
 * from mwpm.py: no robin-hood hash-map order, no per-thread arena, no blossom.
 *
 * The decode is the classic Delfosse-Nickerson pipeline over the graphlike detector
 * graph (V = M+1 vertices with a boundary vertex at id 0, N edges = qubit columns):
 *
 *     grow / fuse  ->  spanning forest  ->  leaf-rake peel
 *
 * Every tie-break is min-vertex-id / min-edge-index and every reduction is
 * min / xor / parity (associative + commutative), so a block-parallel port is
 * order-independent and therefore bit-exact regardless of thread scheduling. See
 * union_find.py for the reference math (functions _grow_fuse / _spanning_forest /
 * _peel), whose control flow this file mirrors 1:1.
 *
 * Two paths share the same block-cooperative device functions (d_grow_fuse /
 * d_spanning_forest / d_peel) and are wired into one Python module at the bottom:
 *
 *   -- Fused path (default) --
 *     Whole decode per shot in a single launch: one thread block decodes one shot,
 *     staging every working array into shared memory.
 *
 *   -- Per-step path (fallback / debug) --
 *     The three phase kernels driven by a Python loop over global-memory scratch,
 *     for codes whose working set exceeds the fused kernel's shared-memory budget.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define UF_CHECK(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

// ════════════════════════════════════════════════════════════════════════════
// Block-cooperative device phases (shared by the fused and per-step paths)
//
// Each function is called by every thread of a block and cooperates over the
// shot's V vertices / N edges with grid-stride loops and __syncthreads(). The
// working arrays are plain int* -- shared memory on the fused path, per-block
// global scratch on the per-step path -- so the exact same code drives both,
// which guarantees the two paths are bit-identical.
// ════════════════════════════════════════════════════════════════════════════

// Min-vertex-id connected components over the currently grown edges (support == 2).
// One-hop label relaxation to a fixpoint: label[v] converges to the smallest vertex
// id reachable from v via grown edges (== union_find.py _connected_components, whose
// FastSV variant reaches the same unique min-label fixpoint). label ends flat
// (label[v] IS the component min), so no separate flattening pass is needed.
__device__ void d_components(
    int* __restrict__ label, const int* __restrict__ support,
    const int* __restrict__ eu, const int* __restrict__ ev, int V, int N
) {
    const int tid = threadIdx.x, bsz = blockDim.x;
    for (int v = tid; v < V; v += bsz) label[v] = v;
    __syncthreads();

    __shared__ int s_changed;
    for (int it = 0; it < V; it++) {
        if (tid == 0) s_changed = 0;
        __syncthreads();
        for (int q = tid; q < N; q += bsz) {
            if (support[q] == 2) {
                const int u = eu[q], w = ev[q];
                const int lu = label[u], lw = label[w];
                const int m = lu < lw ? lu : lw;
                if (lu > m && atomicMin(&label[u], m) > m) s_changed = 1;
                if (lw > m && atomicMin(&label[w], m) > m) s_changed = 1;
            }
        }
        __syncthreads();
        if (s_changed == 0) break;
        __syncthreads();
    }
}

// Grow / fuse fixpoint (union_find.py _grow_fuse). Repeatedly: measure each cluster's
// syndrome parity, grow every edge touching an odd cluster (delta = odd[eu]+odd[ev],
// support saturating at 2), and recompute the min-id components. A cluster containing
// the boundary vertex (id 0) is treated even -- the open boundary absorbs its unpaired
// defect -- so it stops growing (no-op for toric codes, where vertex 0 stays isolated).
// On return label[v] == component min and support[q] == 2 marks the grown edges.
__device__ void d_grow_fuse(
    const int* __restrict__ synd, int* __restrict__ label, int* __restrict__ support,
    int* __restrict__ parity, const int* __restrict__ eu, const int* __restrict__ ev,
    int V, int N
) {
    const int tid = threadIdx.x, bsz = blockDim.x;
    for (int q = tid; q < N; q += bsz) support[q] = 0;
    for (int v = tid; v < V; v += bsz) label[v] = v;
    __syncthreads();

    __shared__ int s_anyodd;
    for (int r = 0; r < V; r++) {
        // cluster parity: xor each vertex's syndrome bit into its root bucket
        for (int v = tid; v < V; v += bsz) parity[v] = 0;
        if (tid == 0) s_anyodd = 0;
        __syncthreads();
        for (int v = tid; v < V; v += bsz)
            if (synd[v]) atomicXor(&parity[label[v]], 1);
        __syncthreads();

        // an odd, non-boundary cluster still wants to grow
        for (int v = tid; v < V; v += bsz) {
            const int lv = label[v];
            if ((parity[lv] & 1) && lv != 0) s_anyodd = 1;
        }
        __syncthreads();
        if (s_anyodd == 0) break;

        // grow: +1 support per odd endpoint, saturating at 2 (fully grown)
        for (int q = tid; q < N; q += bsz) {
            const int lu = label[eu[q]], lw = label[ev[q]];
            const int oddu = (parity[lu] & 1) && lu != 0;
            const int oddw = (parity[lw] & 1) && lw != 0;
            const int delta = oddu + oddw;
            if (delta) { const int s = support[q] + delta; support[q] = s > 2 ? 2 : s; }
        }
        __syncthreads();

        d_components(label, support, eu, ev, V, N);
        __syncthreads();
    }
}

// Spanning forest over the grown edges (union_find.py _spanning_forest). BFS distance
// from each component's representative (its min-id vertex, dist 0); a vertex's parent
// edge is the smallest-index grown edge to a distance-1 neighbour. Writes parent[v]
// (== v when a representative / isolated) and tree[v] (parent edge index, or -1).
__device__ void d_spanning_forest(
    const int* __restrict__ label, const int* __restrict__ support,
    int* __restrict__ dist, int* __restrict__ best, int* __restrict__ parent,
    int* __restrict__ tree, const int* __restrict__ eu, const int* __restrict__ ev,
    int V, int N
) {
    const int tid = threadIdx.x, bsz = blockDim.x;
    const int INF = V + 1;
    for (int v = tid; v < V; v += bsz) dist[v] = (label[v] == v) ? 0 : INF;
    __syncthreads();

    __shared__ int s_changed;
    for (int it = 0; it < V; it++) {
        if (tid == 0) s_changed = 0;
        __syncthreads();
        for (int q = tid; q < N; q += bsz) {
            if (support[q] == 2) {
                const int u = eu[q], w = ev[q];
                const int du = dist[u], dw = dist[w];
                if (du + 1 < dw && atomicMin(&dist[w], du + 1) > du + 1) s_changed = 1;
                if (dw + 1 < du && atomicMin(&dist[u], dw + 1) > dw + 1) s_changed = 1;
            }
        }
        __syncthreads();
        if (s_changed == 0) break;
        __syncthreads();
    }

    // parent edge = smallest-index grown edge to a distance-1-closer neighbour
    for (int v = tid; v < V; v += bsz) best[v] = N;
    __syncthreads();
    for (int q = tid; q < N; q += bsz) {
        if (support[q] == 2) {
            const int u = eu[q], w = ev[q];
            const int du = dist[u], dw = dist[w];
            if (du == dw - 1) atomicMin(&best[w], q);   // u is w's parent
            if (dw == du - 1) atomicMin(&best[u], q);   // w is u's parent
        }
    }
    __syncthreads();
    for (int v = tid; v < V; v += bsz) {
        const int be = best[v];
        if (be < N) { const int a = eu[be], b = ev[be]; parent[v] = (a == v) ? b : a; tree[v] = be; }
        else        { parent[v] = v; tree[v] = -1; }
    }
    __syncthreads();
}

// Leaf-rake peeling of the spanning forest (union_find.py _peel). Repeatedly rake the
// active leaves: a leaf carrying a defect adds its tree edge to the correction and
// pushes its parity up to its parent. Representatives (tree == -1) are never active.
// correction[q] ends 1 iff qubit q is in the estimate.
__device__ void d_peel(
    const int* __restrict__ synd, const int* __restrict__ parent,
    const int* __restrict__ tree, int* __restrict__ child, int* __restrict__ syndcur,
    int* __restrict__ par, int* __restrict__ active, int* __restrict__ correction,
    int V, int N
) {
    const int tid = threadIdx.x, bsz = blockDim.x;
    for (int q = tid; q < N; q += bsz) correction[q] = 0;
    for (int v = tid; v < V; v += bsz) { syndcur[v] = synd[v]; active[v] = tree[v] >= 0; }
    __syncthreads();

    __shared__ int s_anyleaf;
    for (int it = 0; it < V; it++) {
        for (int v = tid; v < V; v += bsz) child[v] = 0;
        if (tid == 0) s_anyleaf = 0;
        __syncthreads();
        for (int v = tid; v < V; v += bsz)
            if (active[v]) atomicAdd(&child[parent[v]], 1);
        __syncthreads();
        for (int v = tid; v < V; v += bsz)
            if (active[v] && child[v] == 0) s_anyleaf = 1;
        __syncthreads();
        if (s_anyleaf == 0) break;

        for (int v = tid; v < V; v += bsz) par[v] = 0;
        __syncthreads();
        for (int v = tid; v < V; v += bsz) {
            if (active[v] && child[v] == 0 && syndcur[v] == 1) {   // defect-carrying leaf
                correction[tree[v]] = 1;                           // distinct edge per leaf
                atomicXor(&par[parent[v]], 1);
            }
        }
        __syncthreads();
        for (int v = tid; v < V; v += bsz) syndcur[v] ^= (par[v] & 1);
        __syncthreads();
        for (int v = tid; v < V; v += bsz)
            if (active[v] && child[v] == 0) { syndcur[v] = 0; active[v] = 0; }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Fused path -- whole decode per shot in one launch (default)
// ════════════════════════════════════════════════════════════════════════════
//
// One thread block decodes one shot. Every working array lives in dynamic shared
// memory (layout below, 10 int arrays of length V + 2 of length N); the constant
// graph (eu / ev) is read from global memory. The three phases run back-to-back
// with a barrier between them.

__global__ void k_uf_fused(
    const int* __restrict__ synd_all,    // [B, V]  internal syndrome (boundary col 0)
    int*       __restrict__ corr_all,    // [B, N]  output correction
    const int* __restrict__ eu,          // [N]     edge endpoint min
    const int* __restrict__ ev,          // [N]     edge endpoint max
    int V, int N, int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x, bsz = blockDim.x;

    extern __shared__ int smem[];
    int* s_label   = smem;               // [V]
    int* s_dist    = s_label   + V;      // [V]
    int* s_parent  = s_dist    + V;      // [V]
    int* s_tree    = s_parent  + V;      // [V]
    int* s_child   = s_tree    + V;      // [V]
    int* s_syndcur = s_child   + V;      // [V]
    int* s_par     = s_syndcur + V;      // [V]  (parity in grow, par_flip in peel)
    int* s_active  = s_par     + V;      // [V]
    int* s_best    = s_active  + V;      // [V]
    int* s_synd    = s_best    + V;      // [V]
    int* s_support = s_synd    + V;      // [N]
    int* s_corr    = s_support + N;      // [N]

    const int* synd = synd_all + (size_t)b * V;
    for (int v = tid; v < V; v += bsz) s_synd[v] = synd[v];
    __syncthreads();

    d_grow_fuse(s_synd, s_label, s_support, s_par, eu, ev, V, N);
    __syncthreads();
    d_spanning_forest(s_label, s_support, s_dist, s_best, s_parent, s_tree, eu, ev, V, N);
    __syncthreads();
    d_peel(s_synd, s_parent, s_tree, s_child, s_syndcur, s_par, s_active, s_corr, V, N);
    __syncthreads();

    int* corr = corr_all + (size_t)b * N;
    for (int q = tid; q < N; q += bsz) corr[q] = s_corr[q];
}

int64_t fused_smem_bytes(int64_t V, int64_t N) {
    return (10 * V + 2 * N) * (int64_t)sizeof(int);
}

int64_t fused_smem_limit() {
    int dev = 0;
    cudaGetDevice(&dev);
    int v = 48 * 1024;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)v;
}

torch::Tensor uf_fused_cuda(
    torch::Tensor synd, torch::Tensor eu, torch::Tensor ev,
    int64_t V, int64_t N, int64_t block_size
) {
    UF_CHECK(synd); UF_CHECK(eu); UF_CHECK(ev);
    const int B = (int)synd.size(0);
    auto corr = torch::empty({B, (long)N},
                             torch::dtype(torch::kInt32).device(synd.device()));
    if (B == 0) return corr;

    const size_t smem = (size_t)fused_smem_bytes(V, N);
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(k_uf_fused,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_fused<<<B, (int)block_size, smem, stream>>>(
        synd.data_ptr<int32_t>(),
        corr.data_ptr<int32_t>(),
        eu.data_ptr<int32_t>(),
        ev.data_ptr<int32_t>(),
        (int)V, (int)N, B);
    return corr;
}

// ════════════════════════════════════════════════════════════════════════════
// Per-step path -- the three phase kernels, driven by a Python loop (fallback)
// ════════════════════════════════════════════════════════════════════════════
//
// One thread block per shot, exactly as the fused path, but every working array is
// per-block global-memory scratch (allocated by the caller) instead of shared
// memory -- for codes whose fused working set exceeds the shared-memory budget, and
// for debugging one phase at a time. Because the same d_* device functions run, the
// output is bit-identical to the fused path.

__global__ void k_uf_grow_fuse(
    const int* __restrict__ synd, int* __restrict__ label, int* __restrict__ support,
    int* __restrict__ par, const int* __restrict__ eu, const int* __restrict__ ev,
    int V, int N, int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    d_grow_fuse(synd + (size_t)b * V, label + (size_t)b * V, support + (size_t)b * N,
                par + (size_t)b * V, eu, ev, V, N);
}

__global__ void k_uf_spanning_forest(
    const int* __restrict__ label, const int* __restrict__ support,
    int* __restrict__ dist, int* __restrict__ best, int* __restrict__ parent,
    int* __restrict__ tree, const int* __restrict__ eu, const int* __restrict__ ev,
    int V, int N, int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    d_spanning_forest(label + (size_t)b * V, support + (size_t)b * N,
                      dist + (size_t)b * V, best + (size_t)b * V,
                      parent + (size_t)b * V, tree + (size_t)b * V, eu, ev, V, N);
}

__global__ void k_uf_peel(
    const int* __restrict__ synd, const int* __restrict__ parent,
    const int* __restrict__ tree, int* __restrict__ child, int* __restrict__ syndcur,
    int* __restrict__ par, int* __restrict__ active, int* __restrict__ corr,
    int V, int N, int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    d_peel(synd + (size_t)b * V, parent + (size_t)b * V, tree + (size_t)b * V,
           child + (size_t)b * V, syndcur + (size_t)b * V, par + (size_t)b * V,
           active + (size_t)b * V, corr + (size_t)b * N, V, N);
}

void uf_grow_fuse_cuda(
    torch::Tensor synd, torch::Tensor label, torch::Tensor support, torch::Tensor par,
    torch::Tensor eu, torch::Tensor ev, int64_t V, int64_t N, int64_t block_size
) {
    UF_CHECK(synd); UF_CHECK(label); UF_CHECK(support); UF_CHECK(par);
    const int B = (int)synd.size(0);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_grow_fuse<<<B, (int)block_size, 0, stream>>>(
        synd.data_ptr<int32_t>(), label.data_ptr<int32_t>(),
        support.data_ptr<int32_t>(), par.data_ptr<int32_t>(),
        eu.data_ptr<int32_t>(), ev.data_ptr<int32_t>(), (int)V, (int)N, B);
}

void uf_spanning_forest_cuda(
    torch::Tensor label, torch::Tensor support, torch::Tensor dist, torch::Tensor best,
    torch::Tensor parent, torch::Tensor tree, torch::Tensor eu, torch::Tensor ev,
    int64_t V, int64_t N, int64_t block_size
) {
    UF_CHECK(label); UF_CHECK(support); UF_CHECK(dist); UF_CHECK(best);
    UF_CHECK(parent); UF_CHECK(tree);
    const int B = (int)label.size(0);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_spanning_forest<<<B, (int)block_size, 0, stream>>>(
        label.data_ptr<int32_t>(), support.data_ptr<int32_t>(),
        dist.data_ptr<int32_t>(), best.data_ptr<int32_t>(),
        parent.data_ptr<int32_t>(), tree.data_ptr<int32_t>(),
        eu.data_ptr<int32_t>(), ev.data_ptr<int32_t>(), (int)V, (int)N, B);
}

void uf_peel_cuda(
    torch::Tensor synd, torch::Tensor parent, torch::Tensor tree, torch::Tensor child,
    torch::Tensor syndcur, torch::Tensor par, torch::Tensor active,
    torch::Tensor corr, int64_t V, int64_t N, int64_t block_size
) {
    UF_CHECK(synd); UF_CHECK(parent); UF_CHECK(tree); UF_CHECK(child);
    UF_CHECK(syndcur); UF_CHECK(par); UF_CHECK(active); UF_CHECK(corr);
    const int B = (int)synd.size(0);
    if (B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_peel<<<B, (int)block_size, 0, stream>>>(
        synd.data_ptr<int32_t>(), parent.data_ptr<int32_t>(),
        tree.data_ptr<int32_t>(), child.data_ptr<int32_t>(),
        syndcur.data_ptr<int32_t>(), par.data_ptr<int32_t>(),
        active.data_ptr<int32_t>(), corr.data_ptr<int32_t>(), (int)V, (int)N, B);
}

// ════════════════════════════════════════════════════════════════════════════
// Python bindings
// ════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uf_fused", &uf_fused_cuda,
          "Full Union-Find decode per shot in one launch (CUDA)");
    m.def("fused_smem_bytes", &fused_smem_bytes,
          "Dynamic shared memory (bytes) the fused kernel needs for [V, N]");
    m.def("fused_smem_limit", &fused_smem_limit,
          "Maximum opt-in dynamic shared memory per block on the current device");

    m.def("uf_grow_fuse", &uf_grow_fuse_cuda,
          "Per-step: grow/fuse fixpoint, writes min-id labels + grown support (CUDA)");
    m.def("uf_spanning_forest", &uf_spanning_forest_cuda,
          "Per-step: BFS spanning forest, writes parent + tree edge (CUDA)");
    m.def("uf_peel", &uf_peel_cuda,
          "Per-step: leaf-rake peel of the forest into the correction (CUDA)");
}
