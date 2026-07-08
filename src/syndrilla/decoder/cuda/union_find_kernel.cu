// union_find_kernel.cu -- bit-exact CUDA port of the Union-Find decoder in union_find.py.
//
// GOAL: produce, for every syndrome shot, the SAME correction bits as
// decoder/union_find/union_find.py (a bit-exact reproduction of chaeyeunpark/UnionFind,
// Delfosse-Nickerson arXiv:1709.06218). This is NOT an independent UF -- it faithfully
// reproduces union_find.py's deterministic tsl::robin_map/robin_set iteration order
// (identity vertex hash, pow2 growth from 0, load 0.5, robin-hood insert, backward-shift
// erase, range-insert reserve), so e-diff == 0. Sorted order alone mismatches ~20% of shots.
//
// PARALLELISM: one CUDA thread decodes one shot. UF is inherently sequential per shot (the
// grown clusters and the spanning-forest peel order depend on the robin-table visitation
// order); the batch is the parallel axis.
//
// STRUCTURE: mirrors union_find.py 1:1. Python objects become fixed-capacity per-thread
// int32 arenas. The robin tables (odd_roots + one border-vertex table per cluster root) live
// in a per-thread bump-allocated bucket arena; every other Python dict/list is a dense [M]/[N]
// array or a ring buffer. The reference is toric-only, so the edge lattice is built once on the
// host and passed as read-only CSR (neighbour + qubit index per vertex, in tsl bucket order).
//
// STATUS: authored without a local GPU; validated bit-for-bit against union_find.py by an
// arena-faithful numpy emulation (scratch uf_emulate.py) across toric d=3..13, and on hardware
// by tests/test_union_find_cuda.py (assert e-diff == 0). Capacities are bounded by M/N; on
// arena overflow a thread sets err and the host redoes that shot on the exact CPU decoder.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

#define THREADS 128
static inline int grid1d(int64_t n) { return (int)((n + THREADS - 1) / THREADS); }

#define RT_EMPTY (-1)      // bucket dist sentinel (== None in _RobinTable)
#define ERR_OK        0
#define ERR_OVERFLOW  1    // per-thread bucket arena exceeded

// ============================================================================
// Per-thread state. `work` is one int32 slice with dense arrays at fixed offsets
// (computed from M, N -- identical on host and device). `arena`/`top` is the
// robin-hood bucket bump-allocator; `fail` trips on arena overflow.
// ============================================================================
struct St {
    // read-only lattice (shared across threads)
    const int64_t* conn_off;   // [M+1]
    const int*     conn_nbr;   // [E2]
    const int*     conn_qub;   // [E2]
    const int*     vcc;        // [M] vertex_connection_count (degree)
    int M, N;

    // per-thread dense arrays (into `work`)
    int* is_root;    // [M] 0/1
    int* r_size;     // [M]
    int* r_parity;   // [M]
    int* support;    // [N] 0..2
    int* conn_cnt;   // [M]
    int* rov;        // [M] root_of_vertex
    int* bt_mask;    // [M] border table: mask
    int* bt_size;    // [M] border table: size
    int* bt_off;     // [M] border table: bucket-region offset (-1 => absent/deleted)
    int* vcount;     // [M] peeling degree
    int* synd;       // [M] mutable syndrome
    int* fu; int* fv; int* fq;   // [N+1] fuse_list FIFO (edge u,v,qubit)
    int* pu; int* pv; int* pq;   // [M+1] peeling_edges deque
    int* keybuf;     // [M] keys() snapshot for grow / merge_boundary
    int* oddkeybuf;  // [M] keys() snapshot for the odd_roots decode loop
    int* odd_mask; int* odd_size; int* odd_off;  // odd_roots scalars

    // robin-hood bucket arena
    int* arena; int top; int cap; int fail;
};

// number of int32 slots the dense `work` slice needs for one thread
static inline int64_t work_stride(int M, int N) {
    return (int64_t)15 * M + (int64_t)4 * N + 9;  // see the layout in st_bind()
}

__host__ __device__ static inline void st_bind(St& s, int* w, int M, int N) {
    int o = 0;
    s.is_root  = w + o; o += M;
    s.r_size   = w + o; o += M;
    s.r_parity = w + o; o += M;
    s.support  = w + o; o += N;
    s.conn_cnt = w + o; o += M;
    s.rov      = w + o; o += M;
    s.bt_mask  = w + o; o += M;
    s.bt_size  = w + o; o += M;
    s.bt_off   = w + o; o += M;
    s.vcount   = w + o; o += M;
    s.synd     = w + o; o += M;
    s.fu = w + o; o += N + 1;
    s.fv = w + o; o += N + 1;
    s.fq = w + o; o += N + 1;
    s.pu = w + o; o += M + 1;
    s.pv = w + o; o += M + 1;
    s.pq = w + o; o += M + 1;
    s.keybuf    = w + o; o += M;
    s.oddkeybuf = w + o; o += M;
    s.odd_mask = w + o; o += 1;
    s.odd_size = w + o; o += 1;
    s.odd_off  = w + o; o += 1;
}

// ---------------------------------------------------------------- robin table
// Identity hash on vertex ids (std::hash<uint32_t>). All ops mirror _RobinTable
// in union_find.py. A table is (mask,size,off) scalars; buckets live in `arena`
// at `off`: bucket ib -> [dist @ off+2*ib, key @ off+2*ib+1], dist=-1 => empty.

__device__ static inline int rt_pow2(int v) {
    if (v == 0) return 1;
    if ((v & (v - 1)) == 0) return v;
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// allocate a fresh region of `nbuckets` empty buckets; returns off (or -1 on overflow)
__device__ static inline int rt_alloc(St& s, int nbuckets) {
    int off = s.top;
    int need = 2 * nbuckets;
    if (off + need > s.cap) { s.fail = 1; return -1; }
    s.top += need;
    for (int i = 0; i < nbuckets; i++) s.arena[off + 2 * i] = RT_EMPTY;
    return off;
}

__device__ static inline void rt_place_on_rehash(St& s, int off, int mask, int key) {
    int ib = key & mask, dist = 0;
    for (;;) {
        int* b = s.arena + off + 2 * ib;
        if (b[0] == RT_EMPTY) { b[0] = dist; b[1] = key; return; }
        if (dist > b[0]) { int d2 = b[0], k2 = b[1]; b[0] = dist; b[1] = key; dist = d2; key = k2; }
        dist++; ib = (ib + 1) & mask;
    }
}

__device__ static inline void rt_rehash_impl(St& s, int* pmask, int* psize, int* poff, int count) {
    int old_off = *poff, old_mask = *pmask;
    if (count == 0) { *poff = -1; *pmask = 0; return; }
    int c = rt_pow2(count);
    int new_off = rt_alloc(s, c);
    if (new_off < 0) return;               // overflow: leave table as-is, s.fail set
    int new_mask = c - 1;
    if (old_off != -1) {
        int nb = old_mask + 1;
        for (int ib = 0; ib < nb; ib++) {
            int* b = s.arena + old_off + 2 * ib;
            if (b[0] != RT_EMPTY) rt_place_on_rehash(s, new_off, new_mask, b[1]);
        }
    }
    *poff = new_off; *pmask = new_mask;
}

__device__ static inline void rt_rehash(St& s, int* pmask, int* psize, int* poff, int count) {
    int c = count > 2 * (*psize) ? count : 2 * (*psize);
    rt_rehash_impl(s, pmask, psize, poff, c);
}

__device__ static inline void rt_reserve(St& s, int* pmask, int* psize, int* poff, int count) {
    rt_rehash(s, pmask, psize, poff, 2 * count);
}

__device__ static inline int rt_bucket_count(int* pmask, int* poff) {
    return (*poff != -1) ? (*pmask + 1) : 0;
}

__device__ static inline void rt_insert_value_impl(St& s, int off, int mask, int ib, int dist, int key) {
    int* b = s.arena + off + 2 * ib;
    int d2 = b[0], k2 = b[1]; b[0] = dist; b[1] = key; dist = d2; key = k2;
    ib = (ib + 1) & mask; dist++;
    while (s.arena[off + 2 * ib] != RT_EMPTY) {
        int* c = s.arena + off + 2 * ib;
        if (dist > c[0]) { int dd = c[0], kk = c[1]; c[0] = dist; c[1] = key; dist = dd; key = kk; }
        ib = (ib + 1) & mask; dist++;
    }
    s.arena[off + 2 * ib] = dist; s.arena[off + 2 * ib + 1] = key;
}

__device__ static inline bool rt_insert(St& s, int* pmask, int* psize, int* poff, int key) {
    if (*poff != -1) {
        int off = *poff, mask = *pmask, ib = key & mask, dist = 0;
        for (;;) {
            int* b = s.arena + off + 2 * ib;
            if (b[0] == RT_EMPTY || dist > b[0]) break;
            if (b[1] == key) return false;
            ib = (ib + 1) & mask; dist++;
        }
    }
    int bc = rt_bucket_count(pmask, poff);
    int load_threshold = bc ? bc / 2 : 0;
    if (*psize >= load_threshold) {
        // grow by (mask+1)*2 -- for an empty table (off=-1, mask=0) this is 2, NOT
        // bucket_count*2 which is 0. Mirrors _RobinTable.insert (union_find.py).
        rt_rehash_impl(s, pmask, psize, poff, (*pmask + 1) * 2);
        if (s.fail) return false;
    }
    int off = *poff, mask = *pmask, ib = key & mask, dist = 0;
    while (s.arena[off + 2 * ib] != RT_EMPTY && dist <= s.arena[off + 2 * ib]) {
        ib = (ib + 1) & mask; dist++;
    }
    if (s.arena[off + 2 * ib] == RT_EMPTY) {
        s.arena[off + 2 * ib] = dist; s.arena[off + 2 * ib + 1] = key;
    } else {
        rt_insert_value_impl(s, off, mask, ib, dist, key);
    }
    (*psize)++;
    return true;
}

__device__ static inline void rt_range_insert(St& s, int* pmask, int* psize, int* poff,
                                              const int* keys, int nkeys) {
    int bc = rt_bucket_count(pmask, poff);
    int load_threshold = bc ? bc / 2 : 0;
    int nb_free = load_threshold - *psize;
    if (nkeys > 0 && nb_free < nkeys) {
        rt_reserve(s, pmask, psize, poff, *psize + nkeys);
        if (s.fail) return;
    }
    for (int i = 0; i < nkeys; i++) {
        rt_insert(s, pmask, psize, poff, keys[i]);
        if (s.fail) return;
    }
}

__device__ static inline void rt_erase(St& s, int* pmask, int* psize, int* poff, int key) {
    if (*poff == -1) return;
    int off = *poff, mask = *pmask, ib = key & mask, dist = 0;
    for (;;) {
        int* b = s.arena + off + 2 * ib;
        if (b[0] == RT_EMPTY || dist > b[0]) return;
        if (b[1] == key) {
            b[0] = RT_EMPTY;
            (*psize)--;
            int prev = ib, cur = (ib + 1) & mask;
            while (s.arena[off + 2 * cur] != RT_EMPTY && s.arena[off + 2 * cur] > 0) {
                s.arena[off + 2 * prev] = s.arena[off + 2 * cur] - 1;
                s.arena[off + 2 * prev + 1] = s.arena[off + 2 * cur + 1];
                s.arena[off + 2 * cur] = RT_EMPTY;
                prev = cur; cur = (cur + 1) & mask;
            }
            return;
        }
        ib = (ib + 1) & mask; dist++;
    }
}

// snapshot keys() in bucket order into out[]; returns count
__device__ static inline int rt_keys(St& s, int* pmask, int* poff, int* out) {
    if (*poff == -1) return 0;
    int off = *poff, nb = *pmask + 1, n = 0;
    for (int ib = 0; ib < nb; ib++)
        if (s.arena[off + 2 * ib] != RT_EMPTY) out[n++] = s.arena[off + 2 * ib + 1];
    return n;
}

// ------------------------------------------------------------ decoder helpers
__device__ static inline int find_root(St& s, int vx) {
    int* rov = s.rov;
    int tmp = rov[vx];
    if (tmp == vx) return vx;
    // path compression: mirror union_find.py _find_root exactly. The path is
    // [rov[vx], ..., root] (the query vertex itself is NOT compressed); flatten it.
    int root = tmp;
    for (;;) { root = tmp; tmp = rov[root]; if (tmp == root) break; }
    int x = rov[vx];
    while (x != root) { int nx = rov[x]; rov[x] = root; x = nx; }
    return root;
}

__device__ static inline int get_size(St& s, int r) {
    return s.is_root[r] ? s.r_size[r] : 0;
}

// ================================================================ main kernel
__global__ void k_uf_decode(
    const int64_t* conn_off, const int* conn_nbr, const int* conn_qub, const int* vcc,
    int M, int N, const uint8_t* synd_in, int B,
    int* work, int64_t work_str, int* arena, int arena_cap,
    uint8_t* e_out, int* err_out)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    St s;
    s.conn_off = conn_off; s.conn_nbr = conn_nbr; s.conn_qub = conn_qub; s.vcc = vcc;
    s.M = M; s.N = N;
    st_bind(s, work + (int64_t)b * work_str, M, N);
    s.arena = arena + (int64_t)b * arena_cap; s.top = 0; s.cap = arena_cap; s.fail = 0;

    // ---- reset dense state ----
    for (int i = 0; i < M; i++) {
        s.is_root[i] = 0; s.r_size[i] = 0; s.r_parity[i] = 0;
        s.conn_cnt[i] = 0; s.rov[i] = i; s.bt_off[i] = -1; s.bt_mask[i] = 0; s.bt_size[i] = 0;
        s.vcount[i] = 0;
    }
    for (int i = 0; i < N; i++) { s.support[i] = 0; e_out[(int64_t)b * N + i] = 0; }
    for (int i = 0; i < M; i++) s.synd[i] = synd_in[(int64_t)b * M + i] ? 1 : 0;
    *s.odd_off = -1; *s.odd_mask = 0; *s.odd_size = 0;

    // ---- init_cluster ----
    // roots = syndrome vertices in ascending id order
    int nroots = 0;
    for (int n = 0; n < M; n++) if (s.synd[n] & 1) nroots++;
    rt_reserve(s, s.odd_mask, s.odd_size, s.odd_off, 2 * nroots);
    if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
    for (int n = 0; n < M; n++) {
        if (s.synd[n] & 1) {
            s.is_root[n] = 1;
            rt_insert(s, s.odd_mask, s.odd_size, s.odd_off, n);
            s.r_size[n] = 1; s.r_parity[n] = 1;
            if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
        }
    }
    for (int n = 0; n < M; n++) {
        if (s.synd[n] & 1) {
            rt_insert(s, &s.bt_mask[n], &s.bt_size[n], &s.bt_off[n], n);  // creates region
            if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
        }
    }

    int f_head = 0, f_tail = 0, FCAP = N + 1;
    int p_head = 0, p_tail = 0, p_cnt = 0, PCAP = M + 1;

    // ---- decode loop ----
    while (*s.odd_size != 0) {
        int nok = rt_keys(s, s.odd_mask, s.odd_off, s.oddkeybuf);
        for (int oi = 0; oi < nok; oi++) {
            int root = s.oddkeybuf[oi];
            // grow(root)
            int nbv = rt_keys(s, &s.bt_mask[root], &s.bt_off[root], s.keybuf);
            for (int bi = 0; bi < nbv; bi++) {
                int bv = s.keybuf[bi];
                int64_t k0 = s.conn_off[bv], k1 = s.conn_off[bv + 1];
                for (int64_t k = k0; k < k1; k++) {
                    int v = s.conn_nbr[k], eidx = s.conn_qub[k];
                    int elt = s.support[eidx];
                    if (elt == 2) continue;
                    elt++; s.support[eidx] = elt;
                    if (elt == 2) {
                        int e0 = bv < v ? bv : v, e1 = bv < v ? v : bv;
                        s.conn_cnt[e0]++; s.conn_cnt[e1]++;
                        s.fu[f_tail] = e0; s.fv[f_tail] = e1; s.fq[f_tail] = eidx;
                        f_tail = (f_tail + 1) % FCAP;
                    }
                }
            }
        }
        // fusion()
        while (f_head != f_tail) {
            int u = s.fu[f_head], v = s.fv[f_head], q = s.fq[f_head];
            f_head = (f_head + 1) % FCAP;
            int r1 = find_root(s, u), r2 = find_root(s, v);
            if (r1 == r2) continue;
            s.pu[p_tail] = u; s.pv[p_tail] = v; s.pq[p_tail] = q;
            p_tail = (p_tail + 1) % PCAP; p_cnt++;
            if (get_size(s, r1) < get_size(s, r2)) { int t = r1; r1 = r2; r2 = t; }
            s.rov[r2] = r1;
            if (!s.is_root[r2]) {
                s.r_size[r1] += 1;
                rt_insert(s, &s.bt_mask[r1], &s.bt_size[r1], &s.bt_off[r1], r2);
                if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
            } else {
                // merge(r1, r2)
                int new_parity = s.r_parity[r1] + s.r_parity[r2];
                if (new_parity & 1) rt_insert(s, s.odd_mask, s.odd_size, s.odd_off, r1);
                else                rt_erase(s, s.odd_mask, s.odd_size, s.odd_off, r1);
                if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
                s.r_size[r1] += s.r_size[r2];
                s.r_parity[r1] = new_parity;
                rt_erase(s, s.odd_mask, s.odd_size, s.odd_off, r2);
                s.r_size[r2] = 0; s.r_parity[r2] = 0; s.is_root[r2] = 0;
                // merge_boundary(r1, r2)
                int nb2 = rt_keys(s, &s.bt_mask[r2], &s.bt_off[r2], s.keybuf);
                rt_range_insert(s, &s.bt_mask[r1], &s.bt_size[r1], &s.bt_off[r1], s.keybuf, nb2);
                if (s.fail) { err_out[b] = ERR_OVERFLOW; return; }
                for (int i = 0; i < nb2; i++) {
                    int vx = s.keybuf[i];
                    if (s.conn_cnt[vx] == s.vcc[vx])
                        rt_erase(s, &s.bt_mask[r1], &s.bt_size[r1], &s.bt_off[r1], vx);
                }
                s.bt_off[r2] = -1; s.bt_size[r2] = 0; s.bt_mask[r2] = 0;  // del
            }
        }
    }

    // ---- peeling ----
    for (int i = 0; i < M; i++) s.vcount[i] = 0;
    int idx = p_head;
    for (int i = 0; i < p_cnt; i++) {
        s.vcount[s.pu[idx]]++; s.vcount[s.pv[idx]]++;
        idx = (idx + 1) % PCAP;
    }
    while (p_cnt > 0) {
        int back = (p_tail - 1 + PCAP) % PCAP;
        int eu = s.pu[back], ev = s.pv[back], eq = s.pq[back];
        p_tail = back; p_cnt--;
        int u, v;
        if (s.vcount[eu] == 1) { u = eu; v = ev; }
        else if (s.vcount[ev] == 1) { u = ev; v = eu; }
        else {
            p_head = (p_head - 1 + PCAP) % PCAP;
            s.pu[p_head] = eu; s.pv[p_head] = ev; s.pq[p_head] = eq;
            p_cnt++;
            continue;
        }
        s.vcount[u]--; s.vcount[v]--;
        if (s.synd[u] == 1) {
            e_out[(int64_t)b * N + eq] = 1;
            s.synd[u] = 0;
            s.synd[v] ^= 1;
        }
    }
    err_out[b] = ERR_OK;
}

// ==================================================================== host API
#define UF_CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be CUDA & contiguous")

std::vector<torch::Tensor> uf_decode(
    torch::Tensor conn_off, torch::Tensor conn_nbr, torch::Tensor conn_qub, torch::Tensor vcc,
    int64_t M, int64_t N, torch::Tensor synd, int64_t arena_cap)
{
    UF_CHECK(conn_off); UF_CHECK(conn_nbr); UF_CHECK(conn_qub); UF_CHECK(vcc); UF_CHECK(synd);
    const int B = (int)synd.size(0);
    auto dev = synd.device();
    auto e_out = torch::zeros({B, N}, torch::dtype(torch::kUInt8).device(dev));
    auto err_out = torch::zeros({B}, torch::dtype(torch::kInt32).device(dev));
    if (B == 0) return {e_out, err_out};

    const int64_t work_str = work_stride((int)M, (int)N);
    auto work = torch::empty({(int64_t)B * work_str}, torch::dtype(torch::kInt32).device(dev));
    auto arena = torch::empty({(int64_t)B * arena_cap}, torch::dtype(torch::kInt32).device(dev));

    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_decode<<<grid1d(B), THREADS, 0, stream>>>(
        conn_off.data_ptr<int64_t>(), conn_nbr.data_ptr<int32_t>(),
        conn_qub.data_ptr<int32_t>(), vcc.data_ptr<int32_t>(),
        (int)M, (int)N, synd.data_ptr<uint8_t>(), B,
        work.data_ptr<int32_t>(), work_str, arena.data_ptr<int32_t>(), (int)arena_cap,
        e_out.data_ptr<uint8_t>(), err_out.data_ptr<int32_t>());
    return {e_out, err_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uf_decode", &uf_decode,
          "Bit-exact Union-Find decode, one thread per shot (CUDA). "
          "Returns (e_out[B,N] uint8, err[B] int32).");
}
