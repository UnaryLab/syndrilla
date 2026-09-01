#ifndef UNION_FIND_SERIAL_CUH
#define UNION_FIND_SERIAL_CUH

#if defined(__CUDACC__)
#define UF_HD __host__ __device__
#else
#define UF_HD
#endif

// round_up_pow2, matching union_find.py._round_up_pow2 (0 -> 1).
UF_HD inline int uf_round_up_pow2(int value) {
    if (value == 0) return 1;
    if ((value & (value - 1)) == 0) return value;
    int p = 1;
    while (p < value) p <<= 1;
    return p;
}

// Per-shot scratch capacity for every robin table. The initial roots/odd_roots
// reserve(2n) drives a table up to round_up_pow2(4n) <= round_up_pow2(4V); borders
// stay under that. +4 is slack against off-by-one growth.
UF_HD inline int uf_maxcap(int V) { return uf_round_up_pow2(4 * V + 4); }

//
// buckets stored as two parallel arrays: rd[i] = distance-from-ideal (-1 == empty),
// rk[i] = key. `cap` is a power of two (0 == empty table), `mask == cap-1`, `size` the
// element count. Every method mirrors the corresponding _RobinTable method so bucket
// order is identical. `tmp` is caller scratch (>= uf_maxcap) used only during rehash.
struct RSet {
    int* rd;   // [maxcap] distance, -1 empty
    int* rk;   // [maxcap] key
    int cap;   // bucket count (power of two, 0 = empty)
    int mask;  // cap - 1
    int size;  // element count
};

UF_HD inline void rs_init(RSet* s, int* rd, int* rk) {
    s->rd = rd; s->rk = rk; s->cap = 0; s->mask = 0; s->size = 0;
}

// _place_on_rehash
UF_HD inline void rs_place(RSet* s, int key) {
    int ib = key & s->mask, dist = 0;
    for (;;) {
        if (s->rd[ib] < 0) { s->rd[ib] = dist; s->rk[ib] = key; return; }
        if (dist > s->rd[ib]) {
            int td = s->rd[ib], tk = s->rk[ib];
            s->rd[ib] = dist; s->rk[ib] = key; dist = td; key = tk;
        }
        dist++; ib = (ib + 1) & s->mask;
    }
}

// _rehash_impl(count): rebuild in place. Gather current keys in bucket order into tmp,
// clear to the new capacity, reinsert in that same order (matches Python's `for entry in
// old`).
UF_HD inline void rs_rehash_impl(RSet* s, int count, int* tmp) {
    int n = 0;
    for (int i = 0; i < s->cap; i++)
        if (s->rd[i] >= 0) tmp[n++] = s->rk[i];
    int newcap = (count == 0) ? 0 : uf_round_up_pow2(count);
    s->cap = newcap; s->mask = newcap - 1;
    for (int i = 0; i < newcap; i++) s->rd[i] = -1;
    for (int i = 0; i < n; i++) rs_place(s, tmp[i]);
}

// _rehash(count): count = max(count, 2*size)
UF_HD inline void rs_rehash(RSet* s, int count, int* tmp) {
    int c = count > 2 * s->size ? count : 2 * s->size;
    rs_rehash_impl(s, c, tmp);
}

// reserve(count): _rehash(2*count)
UF_HD inline void rs_reserve(RSet* s, int count, int* tmp) {
    rs_rehash(s, 2 * count, tmp);
}

UF_HD inline bool rs_contains(const RSet* s, int key) {
    if (s->cap == 0) return false;
    int ib = key & s->mask, dist = 0;
    for (;;) {
        if (s->rd[ib] < 0 || dist > s->rd[ib]) return false;
        if (s->rk[ib] == key) return true;
        ib = (ib + 1) & s->mask; dist++;
    }
}

// _insert_value_impl (unconditional steal at entry bucket)
UF_HD inline void rs_insert_value(RSet* s, int ib, int dist, int key) {
    int ed = s->rd[ib], ek = s->rk[ib];
    s->rd[ib] = dist; s->rk[ib] = key; dist = ed; key = ek;
    ib = (ib + 1) & s->mask; dist++;
    while (s->rd[ib] >= 0) {
        if (dist > s->rd[ib]) {
            int td = s->rd[ib], tk = s->rk[ib];
            s->rd[ib] = dist; s->rk[ib] = key; dist = td; key = tk;
        }
        ib = (ib + 1) & s->mask; dist++;
    }
    s->rd[ib] = dist; s->rk[ib] = key;
}

// insert(key) -> true if newly inserted (mirrors _RobinTable.insert)
UF_HD inline bool rs_insert(RSet* s, int key, int* tmp) {
    if (s->cap != 0) {
        int ib = key & s->mask, dist = 0;
        for (;;) {
            if (s->rd[ib] < 0 || dist > s->rd[ib]) break;
            if (s->rk[ib] == key) return false;
            ib = (ib + 1) & s->mask; dist++;
        }
    }
    int load_threshold = (s->cap != 0) ? (s->mask + 1) / 2 : 0;
    if (s->size >= load_threshold) rs_rehash_impl(s, (s->mask + 1) * 2, tmp);
    int ib = key & s->mask, dist = 0;
    while (s->rd[ib] >= 0 && dist <= s->rd[ib]) { ib = (ib + 1) & s->mask; dist++; }
    if (s->rd[ib] < 0) { s->rd[ib] = dist; s->rk[ib] = key; }
    else rs_insert_value(s, ib, dist, key);
    s->size++;
    return true;
}

// erase(key) with backward-shift deletion
UF_HD inline bool rs_erase(RSet* s, int key) {
    if (s->cap == 0) return false;
    int ib = key & s->mask, dist = 0;
    for (;;) {
        if (s->rd[ib] < 0 || dist > s->rd[ib]) return false;
        if (s->rk[ib] == key) {
            s->rd[ib] = -1; s->size--;
            int prev = ib, cur = (ib + 1) & s->mask;
            while (s->rd[cur] > 0) {
                s->rd[prev] = s->rd[cur] - 1; s->rk[prev] = s->rk[cur];
                s->rd[cur] = -1; prev = cur; cur = (cur + 1) & s->mask;
            }
            return true;
        }
        ib = (ib + 1) & s->mask; dist++;
    }
}

// range_insert(keys[0..nb)) -- transliteration of _RobinTable.range_insert
UF_HD inline void rs_range_insert(RSet* s, const int* keys, int nb, int* tmp) {
    int load_threshold = (s->cap != 0) ? (s->mask + 1) / 2 : 0;
    int nb_free = load_threshold - s->size;
    if (nb > 0 && nb_free < nb) rs_reserve(s, s->size + nb, tmp);
    for (int i = 0; i < nb; i++) rs_insert(s, keys[i], tmp);
}

// keys() in bucket order -> out[], returns count
UF_HD inline int rs_keys(const RSet* s, int* out) {
    int n = 0;
    for (int i = 0; i < s->cap; i++) if (s->rd[i] >= 0) out[n++] = s->rk[i];
    return n;
}

UF_HD inline int uf_find_root(int* root_of, int v, int* path) {
    int tmp = root_of[v];
    if (tmp == v) return v;
    int np = 0, root = tmp;
    for (;;) {
        root = tmp; path[np++] = root; tmp = root_of[root];
        if (tmp == root) break;
    }
    for (int i = 0; i < np; i++) root_of[path[i]] = root;
    return root;
}

//
// One struct of raw pointers the caller carves out of a single buffer (host malloc or
// device global scratch). uf_scratch_ints(V,N) returns the total int count needed.
struct UFScratch {
    int* root_of;   // [V]
    int* sizeA;     // [V]   (0 default == Python dict .get(r,0))
    int* parityA;   // [V]
    int* synd_vec;  // [V]
    int* conncnt;   // [V]   connection_counts
    int* vcount;    // [V]   peel vertex degree
    int* touch_bd;  // [V]   bool
    int* active;    // [V]   gathered odd roots this round
    int* kbuf;      // [V]   scratch key list (border/b2 keys)
    int* bsize;     // [V]  border set size
    int* bcap;      // [V]  border set cap
    int* bmask;     // [V]  border set mask
    int* pe_u;      // [V]  peeling edges (endpoint u)
    int* pe_v;      // [V]  peeling edges (endpoint v)
    int* pe_q;      // [V]  peeling edges (qubit index)
    int* path;      // [V]  find_root path buffer
    int* dq;        // [V+1] peel deque (edge indices)
    int* support;   // [N]
    int* corr;      // [N]   output correction
    int* fu;        // [N]   fuse-list staging (endpoint u)
    int* fv;        // [N]   fuse-list staging (endpoint v)
    int* fq;        // [N]   fuse-list staging (qubit index)
    int* tmp;       // [maxcap] rehash gather
    int* roots_d;   // [maxcap]
    int* roots_k;   // [maxcap]
    int* odd_d;     // [maxcap]
    int* odd_k;     // [maxcap]
    int* bord_d;    // [V*maxcap] border-set distances
    int* bord_k;    // [V*maxcap] border-set keys
};

UF_HD inline long uf_scratch_ints(int V, int N) {
    long mc = uf_maxcap(V);
    return 16L * V + (V + 1) + 5L * N + 5L * mc + 2L * V * mc;
}

// Carve a UFScratch out of a flat int buffer `buf` (>= uf_scratch_ints(V,N)).
UF_HD inline void uf_carve(UFScratch* s, int* buf, int V, int N) {
    int mc = uf_maxcap(V);
    int* p = buf;
    s->root_of = p; p += V;
    s->sizeA = p; p += V;
    s->parityA = p; p += V;
    s->synd_vec = p; p += V;
    s->conncnt = p; p += V;
    s->vcount = p; p += V;
    s->touch_bd = p; p += V;
    s->active = p; p += V;
    s->kbuf = p; p += V;
    s->bsize = p; p += V;
    s->bcap = p; p += V;
    s->bmask = p; p += V;
    s->pe_u = p; p += V;
    s->pe_v = p; p += V;
    s->pe_q = p; p += V;
    s->path = p; p += V;
    s->dq = p; p += (V + 1);
    s->support = p; p += N;
    s->corr = p; p += N;
    s->fu = p; p += N;
    s->fv = p; p += N;
    s->fq = p; p += N;
    s->tmp = p; p += mc;
    s->roots_d = p; p += mc;
    s->roots_k = p; p += mc;
    s->odd_d = p; p += mc;
    s->odd_k = p; p += mc;
    s->bord_d = p; p += (long)V * mc;
    s->bord_k = p; p += (long)V * mc;
}

// Load a border RSet view for root `r` (buffers offset r*maxcap).
UF_HD inline void uf_border_load(RSet* b, UFScratch* s, int r, int mc) {
    b->rd = s->bord_d + (long)r * mc;
    b->rk = s->bord_k + (long)r * mc;
    b->cap = s->bcap[r]; b->mask = s->bmask[r]; b->size = s->bsize[r];
}
UF_HD inline void uf_border_store(const RSet* b, UFScratch* s, int r) {
    s->bcap[r] = b->cap; s->bmask[r] = b->mask; s->bsize[r] = b->size;
}

//
// Lattice inputs (from uf_build_lattice): V, N, M (real checks), B (boundary vertex or
// -1), CSR adjacency conn_off[V+1] / conn_nbr[E2] / conn_q[E2] (neighbour + its qubit
// index, in tsl order), vcc[V] (degree). `synd` is length-M 0/1. Writes corr[N].
UF_HD inline void uf_decode_shot(
    int V, int N, int M, int B,
    const int* conn_off, const int* conn_nbr, const int* conn_q, const int* vcc,
    const int* synd, UFScratch* s
) {
    const int mc = uf_maxcap(V);
    for (int i = 0; i < N; i++) { s->support[i] = 0; s->corr[i] = 0; }
    for (int v = 0; v < V; v++) {
        s->root_of[v] = v; s->sizeA[v] = 0; s->parityA[v] = 0;
        s->synd_vec[v] = 0; s->conncnt[v] = 0; s->touch_bd[v] = 0;
        s->bsize[v] = 0; s->bcap[v] = 0; s->bmask[v] = 0;
    }

    RSet roots, odd;
    rs_init(&roots, s->roots_d, s->roots_k);
    rs_init(&odd, s->odd_d, s->odd_k);

    // count syndrome vertices among real checks 0..M-1
    int n = 0;
    for (int v = 0; v < M; v++) if (synd[v] & 1) n++;
    if (n) { rs_reserve(&roots, 2 * n, s->tmp); rs_reserve(&odd, 2 * n, s->tmp); }
    for (int r = 0; r < M; r++) {
        if (!(synd[r] & 1)) continue;
        rs_insert(&roots, r, s->tmp);
        rs_insert(&odd, r, s->tmp);
        s->sizeA[r] = 1; s->parityA[r] = 1; s->synd_vec[r] = 1;
        RSet b; uf_border_load(&b, s, r, mc); rs_init(&b, b.rd, b.rk);
        rs_insert(&b, r, s->tmp); uf_border_store(&b, s, r);
    }

    int pe_n = 0;  // peeling_edges count
    const int max_rounds = N + 2;
    for (int round = 0; round < max_rounds; round++) {
        // active = [r for r in odd.keys() if not touch_bd[r]]  (bucket order)
        int na = 0;
        int no = rs_keys(&odd, s->kbuf);
        for (int i = 0; i < no; i++) {
            int r = s->kbuf[i];
            if (!s->touch_bd[r]) s->active[na++] = r;
        }
        if (na == 0) break;

        // ---- grow: stage every edge that reaches support 2, in iteration order ----
        int fuse_n = 0;
        for (int ai = 0; ai < na; ai++) {
            int root = s->active[ai];
            RSet b; uf_border_load(&b, s, root, mc);
            int nbk = rs_keys(&b, s->kbuf);  // border[root].keys() in bucket order
            for (int bi = 0; bi < nbk; bi++) {
                int bv = s->kbuf[bi];
                int off = conn_off[bv], deg = conn_off[bv + 1] - off;
                for (int j = 0; j < deg; j++) {
                    int w = conn_nbr[off + j], q = conn_q[off + j];
                    int e0 = bv < w ? bv : w, e1 = bv < w ? w : bv;
                    int sup = s->support[q];
                    if (sup == 2) continue;
                    s->support[q] = sup + 1;
                    if (sup + 1 == 2) {
                        s->conncnt[e0]++; s->conncnt[e1]++;
                        s->fu[fuse_n] = e0; s->fv[fuse_n] = e1; s->fq[fuse_n] = q;
                        fuse_n++;
                    }
                }
            }
        }

        // ---- fuse: union clusters in fuse order, record fusion-forest edges ----
        for (int fi = 0; fi < fuse_n; fi++) {
            int e0 = s->fu[fi], e1 = s->fv[fi], q = s->fq[fi];
            int r1 = uf_find_root(s->root_of, e0, s->path);
            int r2 = uf_find_root(s->root_of, e1, s->path);
            if (r1 == r2) continue;
            s->pe_u[pe_n] = e0; s->pe_v[pe_n] = e1; s->pe_q[pe_n] = q; pe_n++;
            if (s->sizeA[r1] < s->sizeA[r2]) { int t = r1; r1 = r2; r2 = t; }
            s->root_of[r2] = r1;
            if (!rs_contains(&roots, r2)) {
                // r2 is a lone (non-syndrome) vertex absorbed into r1
                s->sizeA[r1] += 1;
                RSet b1; uf_border_load(&b1, s, r1, mc);
                rs_insert(&b1, r2, s->tmp); uf_border_store(&b1, s, r1);
                if (r2 == B) s->touch_bd[r1] = 1;
            } else {
                int new_parity = s->parityA[r1] + s->parityA[r2];
                if (new_parity & 1) rs_insert(&odd, r1, s->tmp);
                else rs_erase(&odd, r1);
                s->sizeA[r1] += s->sizeA[r2];
                s->parityA[r1] = new_parity;
                rs_erase(&odd, r2);
                s->sizeA[r2] = 0; s->parityA[r2] = 0;
                rs_erase(&roots, r2);
                // border merge: b1.range_insert(b2.keys()); then drop fully-connected
                RSet b1; uf_border_load(&b1, s, r1, mc);
                RSet b2; uf_border_load(&b2, s, r2, mc);
                int nk2 = rs_keys(&b2, s->kbuf);  // snapshot of b2.keys()
                rs_range_insert(&b1, s->kbuf, nk2, s->tmp);
                for (int i = 0; i < nk2; i++) {
                    int vtx = s->kbuf[i];
                    if (s->conncnt[vtx] == vcc[vtx]) rs_erase(&b1, vtx);
                }
                uf_border_store(&b1, s, r1);
                s->bsize[r2] = 0; s->bcap[r2] = 0; s->bmask[r2] = 0;  // del border[r2]
                if (s->touch_bd[r2] || s->touch_bd[r1]) s->touch_bd[r1] = 1;
            }
        }
    }

    // ---- peeling: LIFO leaf-rake over the fusion forest (decode_shot peel) ----
    for (int v = 0; v < V; v++) s->vcount[v] = 0;
    for (int i = 0; i < pe_n; i++) { s->vcount[s->pe_u[i]]++; s->vcount[s->pe_v[i]]++; }
    if (pe_n > 0) {
        int ringcap = pe_n + 1;
        int head = 0, dqsize = pe_n;
        for (int i = 0; i < pe_n; i++) s->dq[i] = i;   // deque = peeling_edges in order
        long guard = 0, maxguard = (long)pe_n * pe_n + pe_n + 16;
        while (dqsize > 0 && guard++ < maxguard) {
            int bi = (head + dqsize - 1) % ringcap;    // back
            int idx = s->dq[bi]; dqsize--;             // pop back
            int a = s->pe_u[idx], bb = s->pe_v[idx];
            int u, vv;
            if (a != B && s->vcount[a] == 1) { u = a; vv = bb; }
            else if (bb != B && s->vcount[bb] == 1) { u = bb; vv = a; }
            else {
                head = (head - 1 + ringcap) % ringcap; // appendleft (not yet a leaf)
                s->dq[head] = idx; dqsize++;
                continue;
            }
            s->vcount[u]--; s->vcount[vv]--;
            if (s->synd_vec[u] == 1) {
                s->corr[s->pe_q[idx]] = 1;
                s->synd_vec[u] = 0; s->synd_vec[vv] ^= 1;
            }
        }
    }
}

//
// Runs once per parity matrix on the CPU (inside the extension); never called from a
// device kernel, so it may use std::vector. It is implicitly __host__ (no UF_HD), so
// nvcc never emits device code for it -- do NOT guard with __CUDA_ARCH__: that macro is
// defined during the device compilation pass, which still PARSES every host function
// (including uf_build_lattice_ext), so #if'ing the definition out makes that call fail
// to resolve. Leaving it unguarded is the correct "host-only" pattern.
#include <vector>
#include <utility>

// Edge robin table replicating _RobinTable with std::hash<Edge> == u ^ (v<<1). Stores
// the first qubit index that maps to each edge (construct_edge_idx: first q wins).
struct UFEdgeRobin {
    std::vector<int> ed, eu, ev, eq;  // ed[i] < 0 == empty; parallel arrays
    int cap = 0, mask = 0, size = 0;
    static int hash(int u, int v) { return u ^ (v << 1); }

    void place(int u, int v, int q) {  // reinsert during rehash, dist recomputed from 0
        int ib = hash(u, v) & mask, dist = 0;
        for (;;) {
            if (ed[ib] < 0) { ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q; return; }
            if (dist > ed[ib]) {
                int td = ed[ib], tu = eu[ib], tv = ev[ib], tq = eq[ib];
                ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q;
                dist = td; u = tu; v = tv; q = tq;
            }
            dist++; ib = (ib + 1) & mask;
        }
    }
    void rehash_impl(int count) {
        std::vector<int> od = ed, ou = eu, ov = ev, oq = eq;
        int ocap = cap;
        int newcap = (count == 0) ? 0 : uf_round_up_pow2(count);
        cap = newcap; mask = newcap - 1;
        ed.assign(newcap, -1); eu.assign(newcap, 0);
        ev.assign(newcap, 0); eq.assign(newcap, 0);
        for (int i = 0; i < ocap; i++) if (od[i] >= 0) place(ou[i], ov[i], oq[i]);
    }
    void insert_value(int ib, int dist, int u, int v, int q) {
        int ed0 = ed[ib], eu0 = eu[ib], ev0 = ev[ib], eq0 = eq[ib];
        ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q;
        dist = ed0; u = eu0; v = ev0; q = eq0;
        ib = (ib + 1) & mask; dist++;
        while (ed[ib] >= 0) {
            if (dist > ed[ib]) {
                int td = ed[ib], tu = eu[ib], tv = ev[ib], tq = eq[ib];
                ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q;
                dist = td; u = tu; v = tv; q = tq;
            }
            ib = (ib + 1) & mask; dist++;
        }
        ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q;
    }
    // returns true if newly inserted (first appearance of this edge)
    bool insert(int u, int v, int q) {
        if (cap != 0) {
            int ib = hash(u, v) & mask, dist = 0;
            for (;;) {
                if (ed[ib] < 0 || dist > ed[ib]) break;
                if (eu[ib] == u && ev[ib] == v) return false;
                ib = (ib + 1) & mask; dist++;
            }
        }
        int lt = (cap != 0) ? (mask + 1) / 2 : 0;
        if (size >= lt) rehash_impl((mask + 1) * 2);
        int ib = hash(u, v) & mask, dist = 0;
        while (ed[ib] >= 0 && dist <= ed[ib]) { ib = (ib + 1) & mask; dist++; }
        if (ed[ib] < 0) { ed[ib] = dist; eu[ib] = u; ev[ib] = v; eq[ib] = q; }
        else insert_value(ib, dist, u, v, q);
        size++;
        return true;
    }
};

// Build the detector graph. qp_w[q] in {0,1,2}, qp_r0/qp_r1 the (increasing) detector
// rows of column q (qp_r1 used only when qp_w==2). Outputs V, boundary B (-1 toric),
// CSR adjacency conn_off/conn_nbr/conn_q (neighbour + first-qubit index, tsl order) and
// vcc (degree). This is host-only and produces the exact arrays uf_decode_shot expects.
inline void uf_build_lattice(
    int M, int N, const int* qp_w, const int* qp_r0, const int* qp_r1,
    int* outV, int* outB,
    std::vector<int>& conn_off, std::vector<int>& conn_nbr,
    std::vector<int>& conn_q, std::vector<int>& vcc
) {
    bool has_boundary = false;
    for (int q = 0; q < N; q++) if (qp_w[q] == 1) has_boundary = true;
    int B = has_boundary ? M : -1;
    int V = M + (has_boundary ? 1 : 0);

    UFEdgeRobin et;
    for (int q = 0; q < N; q++) {
        int w = qp_w[q], a, b;
        if (w >= 2) { a = qp_r0[q]; b = qp_r1[q]; }
        else if (w == 1) { a = qp_r0[q]; b = B; }  // boundary vertex id == M
        else continue;                              // weight-0: never grown
        int lo = a < b ? a : b, hi = a < b ? b : a;
        et.insert(lo, hi, q);                       // first q for this edge wins
    }

    std::vector<std::vector<std::pair<int, int>>> conns(V);  // (neighbour, qubit idx)
    for (int i = 0; i < et.cap; i++) {
        if (et.ed[i] < 0) continue;
        int u = et.eu[i], v = et.ev[i], q = et.eq[i];
        conns[u].push_back(std::make_pair(v, q));
        conns[v].push_back(std::make_pair(u, q));
    }

    conn_off.assign(V + 1, 0);
    for (int v = 0; v < V; v++) conn_off[v + 1] = conn_off[v] + (int)conns[v].size();
    conn_nbr.assign(conn_off[V], 0);
    conn_q.assign(conn_off[V], 0);
    vcc.assign(V, 0);
    for (int v = 0; v < V; v++) {
        vcc[v] = (int)conns[v].size();
        int off = conn_off[v];
        for (int j = 0; j < (int)conns[v].size(); j++) {
            conn_nbr[off + j] = conns[v][j].first;
            conn_q[off + j] = conns[v][j].second;
        }
    }
    *outV = V; *outB = B;
}

#endif  // UNION_FIND_SERIAL_CUH
