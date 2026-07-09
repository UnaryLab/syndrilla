// mwpm_kernel.cu -- bit-exact CUDA port of the native sparse-blossom MWPM decoder in mwpm.py.
//
// GOAL: produce, for every syndrome shot, the SAME correction bits as
// decoder/mwpm/mwpm.py (which is itself a bit-exact transcription of PyMatching v2).
// This is NOT an independent optimal blossom -- it faithfully reproduces mwpm.py's
// deterministic event ordering (radix-heap LIFO-within-bucket, QueuedEventTracker,
// boundary-first-then-column-sorted neighbor order, tie-breaks), so e-diff == 0.
//
// PARALLELISM: one CUDA thread decodes one shot. The
// blossom is inherently sequential per shot; the batch is the parallel axis.
//
// STRUCTURE: mirrors mwpm.py 1:1. Python objects/pointers become fixed-capacity
// per-thread SoA arenas indexed by int32 handles (-1 == None/SIZE_MAX). Python lists
// become arrays+counts or intrusive linked lists. Python recursion becomes explicit
// stacks. Cross-references to mwpm.py line ranges are given at each block.
//
// SCOPE: the flood + matching runs on the GPU for all N <= 64*OBSW (OBSW=4 -> N<=256,
// covering surface d<=11); above that the host falls back to the CPU decoder and this kernel
// sets err=2. CORRECTION EXTRACTION mirrors mwpm.py's own branch on N (mwpm.py:1642):
//   * N <= 64  -- the flood-accumulated obs_mask IS the correction (out_mask). Fast, exact.
//   * N  > 64  -- the obs_mask picks a DIFFERENT (equal-weight) degenerate representative than
//                 PyMatching, so mwpm.py instead reconstructs each matched pair's explicit
//                 shortest qubit path with the SearchFlooder. To stay bit-exact there, this
//                 kernel emits the same MATCH EDGES (out_mef/out_met: loc_from/loc_to detector
//                 node ids, -1 = boundary) at the identical base cases, and the host runs
//                 mwpm.py's SearchFlooder over them (mwpm_cuda._corr_from_match_edges). Both
//                 outputs are always produced; the host selects by N. The observable payload
//                 is opaque (XOR-only), so OBSW just widens the word count -- no algorithm
//                 change to the matching.
//
// STATUS: authored without a local GPU; validate on hardware with tests/test_mwpm_cuda*
// (assert e-diff == 0 vs mwpm.py). Capacity constants below are generous but bounded by
// M; on overflow a thread sets err (see ERR_*) instead of corrupting memory.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

#define THREADS 128
static inline int grid1d(int64_t n) { return (int)((n + THREADS - 1) / THREADS); }

// ------------------------------------------------------------------ constants
#define SIZE_MAX32 (-1)
#define INF64 ((int64_t)1 << 62)
#define WEIGHT 2            // uniform edge weight W (mwpm.py: W = 2)

// FloodCheckEvent kinds (mwpm.py:376-378)
#define LOOK_AT_NODE            0
#define LOOK_AT_SHRINKING_REGION 1
#define NO_FLOOD_CHECK_EVENT    2

// MwpmEvent types (mwpm.py:391-394)
#define NO_EVENT            0
#define REGION_HIT_REGION   1
#define REGION_HIT_BOUNDARY 2
#define BLOSSOM_SHATTER     3

// error codes reported per shot
#define ERR_OK        0
#define ERR_OVERFLOW  1   // an arena capacity was exceeded
#define ERR_UNSUPPORTED 2 // N > 64*OBSW (obs_mask too narrow; host uses the CPU fallback)
#define ERR_ITER_CAP  3   // flood loop exceeded a safety bound -- guards against a per-shot
                          // hang from a latent structural bug; the host redoes the shot on
                          // the exact CPU blossom (bit-exact), so results stay correct.

// ---------------------------------------------------------- observable mask
// The observable/correction payload is OPAQUE to the matcher: everywhere below it is only
// zero-initialized, XOR-combined (^ / ^=), copied, or written to the output. It NEVER feeds
// a branch, comparison, weight, ordering, or tie-break. So widening it from a single uint64
// to OBSW 64-bit words lifts the old N<=64 cap WITHOUT changing any algorithm logic -- every
// expression body in this file is byte-for-byte unchanged; only the *type* changes.
// OBSW words -> supports N (qubits == H columns) up to 64*OBSW. 4 -> 256 (surface d<=11).
// MUST match _OBSW in decoder/mwpm/mwpm_cuda.py (the CSR obs packer + output expander).
#ifndef OBSW
#define OBSW 4
#endif
struct Obs {
    uint64_t w[OBSW];
    __device__ __forceinline__ Obs() {}               // uninitialized, like the old uint64 arrays
    __device__ __forceinline__ Obs(uint64_t lo) {     // enables `Obs x = 0;` (all kernel literals are 0)
        w[0] = lo;
        #pragma unroll
        for (int i = 1; i < OBSW; i++) w[i] = 0;
    }
};
__device__ __forceinline__ Obs operator^(const Obs& a, const Obs& b) {
    Obs o;
    #pragma unroll
    for (int i = 0; i < OBSW; i++) o.w[i] = a.w[i] ^ b.w[i];
    return o;
}
__device__ __forceinline__ Obs& operator^=(Obs& a, const Obs& b) {
    #pragma unroll
    for (int i = 0; i < OBSW; i++) a.w[i] ^= b.w[i];
    return a;
}

// ============================================================================
// Per-thread arena. All pointers below are ALREADY offset to thread b's slice,
// so field access is e.g. S.nd_region[i]. Capacities are passed in.
// ============================================================================
struct St {
    // --- static graph (shared, NOT per-thread) ---
    const int* g_off;     // [M+1] CSR offsets into g_nbr/g_obs
    const int* g_nbr;     // [E] neighbor node id, or -1 for the boundary
    const Obs* g_obs;     // [E] observable mask of the edge (bit=qubit, OBSW words)
    int M, N;

    int REGCAP, ALTCAP, BCCAP, QCAP, CHILDCAP, STKCAP, MECAP;

    // --- detector nodes [M] (mwpm.py DetectorNode) ---
    int*      nd_region;      // region_that_arrived
    int*      nd_region_top;  // region_that_arrived_top
    int64_t*  nd_wrapped;     // wrapped_radius_cached
    int*      nd_reached;     // reached_from_source (node id or -1)
    Obs*      nd_obs;         // observables_crossed_from_source
    int64_t*  nd_arrival;     // radius_of_arrival
    int*      nd_shell_next;  // intrusive shell-stack link (node id or -1)
    // node_event_tracker (QueuedEventTracker)
    int64_t*  nd_td;  int64_t* nd_tq;  uint8_t* nd_thd;  uint8_t* nd_thq;

    // --- region pool [REGCAP] (mwpm.py GraphFillRegion) ---
    int*      rg_bparent;      // blossom_parent
    int*      rg_bparent_top;  // blossom_parent_top
    int*      rg_alt;          // alt_tree_node
    int*      rg_slope;        // radius.slope
    int64_t*  rg_yint;         // radius.yint
    int64_t*  rg_std; int64_t* rg_stq; uint8_t* rg_sthd; uint8_t* rg_sthq; // shrink tracker
    int*      rg_match_region; // match.region
    int*      rg_mf;  int* rg_mt;  Obs* rg_mo;           // match.edge (from,to,obs)
    int*      rg_bc_start; int* rg_bc_count;             // blossom_children block
    int*      rg_shell_head; int* rg_shell_size;         // intrusive shell stack

    // --- blossom_children pool [BCCAP] of RegionEdge ---
    int*      bc_region; int* bc_ef; int* bc_et; Obs* bc_eo;

    // --- alt-tree pool [ALTCAP] (mwpm.py AltTreeNode) ---
    int*      at_inner; int* at_outer;
    int*      at_if; int* at_it; Obs* at_io;         // inner_to_outer_edge
    int*      at_par;                                // parent.alt_tree_node
    int*      at_pf; int* at_pt; Obs* at_po;         // parent.edge
    int*      at_nchild; uint8_t* at_visited;

    // --- children pool [ALTCAP*CHILDCAP] of AltTreeEdge ---
    int*      ch_alt; int* ch_ef; int* ch_et; Obs* ch_eo;

    // --- radix-heap event pool [QCAP] (doubly linked buckets) ---
    int64_t*  ev_time; int* ev_kind; int* ev_target;
    int*      ev_next; int* ev_prev;
    int*      bkt_head; int* bkt_tail;   // [65]

    // --- scratch stacks for de-recursed traversals ---
    int*      stk;   // [STKCAP]
    int*      stk2;  // [STKCAP]
    int*      stk3;  // [STKCAP] -- op_total_area ONLY. It is called from inside
                     // shatter_descendants' stk/stk2 post-order walk, so it must not
                     // alias either of those (see op_total_area / shatter_descendants).

    // per-thread scalars (each a length-1 slice)
    int64_t*  q_curtime;
    int*      q_num; int* q_free;
    int*      reg_count; int* alt_count; int* bc_count;
    int*      err;

    // per-thread hang guard: counts queue dequeues over the whole decode; a correct decode
    // finishes far below guard_max. On trip we set ERR_ITER_CAP so the host redoes this shot
    // on the exact CPU blossom. Plain (non-pointer) members: St is a per-thread stack object.
    int64_t   guard;
    int64_t   guard_max;
};

// convenience scalar refs
#define CURT   (*S.q_curtime)
#define QNUM   (*S.q_num)
#define QFREE  (*S.q_free)
#define REGN   (*S.reg_count)
#define ALTN   (*S.alt_count)
#define BCN    (*S.bc_count)
#define ERRF   (*S.err)

__device__ __forceinline__ void set_err(St& S, int code) {
    if (ERRF == ERR_OK) ERRF = code;
}

// ============================================================================
// std::bit_width (mwpm.py RadixHeapQueue.cur_bit_bucket_for, line 71-72)
// bit_width(x) = number of bits, bit_width(0)=0. x = time ^ cur_time >= 0.
// ============================================================================
__device__ __forceinline__ int bit_width(uint64_t x) {
    return x == 0 ? 0 : (64 - __clzll((unsigned long long)x));
}

// ============================================================================
// Radix-heap queue (mwpm.py RadixHeapQueue, lines 59-106).
// Buckets are doubly linked lists preserving Python list semantics:
//   enqueue = append at tail; dequeue pop = remove tail (back()); redistribution
//   iterates head->tail and appends to targets. cur_bit_bucket_for uses cur_time.
// ============================================================================
__device__ __forceinline__ int q_alloc_slot(St& S) {
    int s = QFREE;
    if (s < 0) { set_err(S, ERR_OVERFLOW); return -1; }
    QFREE = S.ev_next[s];
    return s;
}
__device__ __forceinline__ void q_free_slot(St& S, int s) {
    S.ev_next[s] = QFREE; QFREE = s;
}
__device__ __forceinline__ void bkt_append(St& S, int b, int s) {
    // append event slot s at tail of bucket b
    S.ev_next[s] = -1;
    S.ev_prev[s] = S.bkt_tail[b];
    if (S.bkt_tail[b] >= 0) S.ev_next[S.bkt_tail[b]] = s;
    else S.bkt_head[b] = s;
    S.bkt_tail[b] = s;
}
__device__ __forceinline__ void q_enqueue(St& S, int64_t time, int kind, int target) {
    int b = bit_width((uint64_t)(time ^ CURT));
    int s = q_alloc_slot(S);
    if (s < 0) return;
    S.ev_time[s] = time; S.ev_kind[s] = kind; S.ev_target[s] = target;
    bkt_append(S, b, s);
    QNUM += 1;
}
// dequeue: returns event slot (unlinked from its bucket) or -1 if empty. Caller
// reads fields then q_free_slot(). Mirrors mwpm.py dequeue() lines 79-100.
__device__ __forceinline__ int q_dequeue(St& S) {
    // hang guard: every unit of forward progress dequeues one event, so bounding dequeues
    // bounds the whole decode. Tripping it aborts the shot to the CPU fallback (see kernel).
    if (++S.guard > S.guard_max) { set_err(S, ERR_ITER_CAP); return -1; }
    if (QNUM == 0) return -1;
    if (S.bkt_head[0] < 0) {
        int b = 1;
        while (S.bkt_head[b] < 0) b++;
        if (b == 1) {
            // swap bucket 0 and 1; cur_time += 1
            int h = S.bkt_head[0], t = S.bkt_tail[0];
            S.bkt_head[0] = S.bkt_head[1]; S.bkt_tail[0] = S.bkt_tail[1];
            S.bkt_head[1] = h; S.bkt_tail[1] = t;
            CURT += 1;
        } else {
            // min_time over bucket b, then redistribute
            int64_t min_time = INF64;
            for (int e = S.bkt_head[b]; e >= 0; e = S.ev_next[e])
                if (S.ev_time[e] < min_time) min_time = S.ev_time[e];
            CURT = min_time;
            int e = S.bkt_head[b];
            S.bkt_head[b] = -1; S.bkt_tail[b] = -1;      // detach whole bucket
            while (e >= 0) {
                int nxt = S.ev_next[e];
                int nb = bit_width((uint64_t)(S.ev_time[e] ^ CURT));
                bkt_append(S, nb, e);
                e = nxt;
            }
        }
    }
    // pop back() of bucket 0 -> LIFO
    int s = S.bkt_tail[0];
    S.bkt_tail[0] = S.ev_prev[s];
    if (S.ev_prev[s] >= 0) S.ev_next[S.ev_prev[s]] = -1;
    else S.bkt_head[0] = -1;
    QNUM -= 1;
    return s;
}

// ============================================================================
// QueuedEventTracker (mwpm.py lines 112-149). Generic over the 4 tracker fields.
// ============================================================================
__device__ __forceinline__ void trk_set_desired(
        St& S, int64_t* td, int64_t* tq, uint8_t* thd, uint8_t* thq,
        int64_t time, int kind, int target) {
    *thd = 1; *td = time;
    if (!*thq || *tq > time) {
        *tq = time; *thq = 1;
        q_enqueue(S, time, kind, target);
    }
}
__device__ __forceinline__ void trk_set_no_desired(uint8_t* thd) { *thd = 0; }

// dequeue_decision (lines 136-149). ev_time is the dequeued event time; kind/target
// are needed to re-enqueue if the desired time moved.
__device__ __forceinline__ bool trk_dequeue_decision(
        St& S, int64_t* td, int64_t* tq, uint8_t* thd, uint8_t* thq,
        int64_t ev_time, int kind, int target) {
    if (!*thq || ev_time != *tq) return false;
    *thq = 0;
    if (!*thd) return false;
    if (ev_time != *td) {
        *thq = 1; *tq = *td;
        q_enqueue(S, *td, kind, target);
        return false;
    }
    *thd = 0;
    return true;
}

// tracker accessors for a detector node / region
#define NODE_TRK(i) &S.nd_td[i], &S.nd_tq[i], &S.nd_thd[i], &S.nd_thq[i]
#define RG_STRK(r)  &S.rg_std[r], &S.rg_stq[r], &S.rg_sthd[r], &S.rg_sthq[r]

// ============================================================================
// Varying: distance(t) = slope*t + yint, slope in {-1,0,+1} (mwpm.py 443-489).
// Stored per region as (rg_slope, rg_yint). Helpers are pure functions.
// ============================================================================
__device__ __forceinline__ int64_t vary_at(int slope, int64_t yint, int64_t t) {
    return (int64_t)slope * t + yint;
}
// then_growing_at_time / then_frozen / then_shrinking return new (slope,yint).
__device__ __forceinline__ void vary_then_growing(int* slope, int64_t* yint, int64_t t) {
    int64_t d = vary_at(*slope, *yint, t); *slope = 1;  *yint = d - t;
}
__device__ __forceinline__ void vary_then_frozen(int* slope, int64_t* yint, int64_t t) {
    int64_t d = vary_at(*slope, *yint, t); *slope = 0;  *yint = d;
}
__device__ __forceinline__ void vary_then_shrinking(int* slope, int64_t* yint, int64_t t) {
    int64_t d = vary_at(*slope, *yint, t); *slope = -1; *yint = d + t;
}

// local_radius of a detector node (mwpm.py DetectorNode.local_radius, 567-572):
// top = region_that_arrived_top; if None -> ZERO_FROZEN (slope 0, yint 0);
// else region.radius.plus_const(wrapped_radius_cached).
__device__ __forceinline__ void node_local_radius(
        St& S, int i, int* slope, int64_t* yint) {
    int top = S.nd_region_top[i];
    if (top < 0) { *slope = 0; *yint = 0; }
    else { *slope = S.rg_slope[top]; *yint = S.rg_yint[top] + S.nd_wrapped[i]; }
}

// ============================================================================
// Region pool / alt pool allocation (bump; no reuse -- see file header).
// ============================================================================
__device__ __forceinline__ int new_region(St& S) {
    int r = REGN++;
    if (r >= S.REGCAP) { set_err(S, ERR_OVERFLOW); return 0; }
    S.rg_bparent[r] = -1; S.rg_bparent_top[r] = r; S.rg_alt[r] = -1;
    S.rg_slope[r] = 1; S.rg_yint[r] = 0;                 // Varying(1,0) growing
    S.rg_std[r] = 0; S.rg_stq[r] = 0; S.rg_sthd[r] = 0; S.rg_sthq[r] = 0;
    S.rg_match_region[r] = -1; S.rg_mf[r] = -1; S.rg_mt[r] = -1; S.rg_mo[r] = 0;
    S.rg_bc_start[r] = -1; S.rg_bc_count[r] = 0;
    S.rg_shell_head[r] = -1; S.rg_shell_size[r] = 0;
    return r;
}
__device__ __forceinline__ int new_alt(St& S, int inner, int outer,
        int if_, int it_, Obs io_) {
    int a = ALTN++;
    if (a >= S.ALTCAP) { set_err(S, ERR_OVERFLOW); return 0; }
    S.at_inner[a] = inner; S.at_outer[a] = outer;
    S.at_if[a] = if_; S.at_it[a] = it_; S.at_io[a] = io_;
    S.at_par[a] = -1; S.at_pf[a] = -1; S.at_pt[a] = -1; S.at_po[a] = 0;
    S.at_nchild[a] = 0; S.at_visited[a] = 0;
    if (inner >= 0) S.rg_alt[inner] = a;
    if (outer >= 0) S.rg_alt[outer] = a;
    return a;
}

// add_child (mwpm.py AltTreeNode.add_child, 726-728): append AltTreeEdge(child,edge)
// to parent.children; set child.parent = AltTreeEdge(parent, edge.reversed()).
__device__ __forceinline__ void alt_add_child(
        St& S, int parent, int child, int ef, int et, Obs eo) {
    int k = S.at_nchild[parent];
    if (k >= S.CHILDCAP) { set_err(S, ERR_OVERFLOW); return; }
    int base = parent * S.CHILDCAP + k;
    S.ch_alt[base] = child; S.ch_ef[base] = ef; S.ch_et[base] = et; S.ch_eo[base] = eo;
    S.at_nchild[parent] = k + 1;
    // child.parent = (parent, reversed(edge)) : reversed swaps from/to, keeps obs
    S.at_par[child] = parent; S.at_pf[child] = et; S.at_pt[child] = ef; S.at_po[child] = eo;
}
// unstable_erase one child of `parent` whose alt_tree_node == `child` (mwpm.py 526-533).
__device__ __forceinline__ void alt_erase_child(St& S, int parent, int child) {
    int n = S.at_nchild[parent];
    int base = parent * S.CHILDCAP;
    int i = 0;
    while (i < n) {
        if (S.ch_alt[base + i] == child) {
            int last = n - 1;
            S.ch_alt[base + i] = S.ch_alt[base + last];
            S.ch_ef[base + i]  = S.ch_ef[base + last];
            S.ch_et[base + i]  = S.ch_et[base + last];
            S.ch_eo[base + i]  = S.ch_eo[base + last];
            n = last;
        } else i++;
    }
    S.at_nchild[parent] = n;
}

// ============================================================================
// Intrusive shell-area stack (mwpm.py GraphFillRegion.shell_area list).
// A node belongs to exactly one region's shell at a time. append pushes to head
// (most-recent); pop removes head; iterate head->... == reversed(shell_area), which
// is exactly what do_op_for_each_node_in_total_area wants (sa[len-1-i]).
// ============================================================================
__device__ __forceinline__ void shell_push(St& S, int r, int node) {
    S.nd_shell_next[node] = S.rg_shell_head[r];
    S.rg_shell_head[r] = node;
    S.rg_shell_size[r] += 1;
}
__device__ __forceinline__ int shell_pop(St& S, int r) {
    int node = S.rg_shell_head[r];               // last-appended (Python list back())
    S.rg_shell_head[r] = S.nd_shell_next[node];
    S.rg_shell_size[r] -= 1;
    S.nd_shell_next[node] = -1;
    return node;
}

// ============================================================================
// node reset (mwpm.py DetectorNode.reset, 601-608)
// ============================================================================
__device__ __forceinline__ void node_reset(St& S, int i) {
    S.nd_obs[i] = 0; S.nd_reached[i] = -1; S.nd_arrival[i] = 0;
    S.nd_region[i] = -1; S.nd_region_top[i] = -1; S.nd_wrapped[i] = 0;
    S.nd_shell_next[i] = -1;
    S.nd_td[i] = 0; S.nd_tq[i] = 0; S.nd_thd[i] = 0; S.nd_thq[i] = 0;
}

// compute_wrapped_radius (mwpm.py 583-591)
__device__ __forceinline__ int64_t compute_wrapped_radius(St& S, int i) {
    if (S.nd_reached[i] < 0) return 0;
    int64_t total = 0;
    int r = S.nd_region[i];
    int top = S.nd_region_top[i];
    while (r != top) { total += S.rg_yint[r]; r = S.rg_bparent[r]; }
    return total - S.nd_arrival[i];
}
// heir_region_on_shatter (mwpm.py 593-599)
__device__ __forceinline__ int heir_region_on_shatter(St& S, int i) {
    int r = S.nd_region[i];
    int top = S.nd_region_top[i];
    while (true) {
        int p = S.rg_bparent[r];
        if (p == top) return r;
        r = p;
    }
}

// index_of_neighbor for a detector node (mwpm.py 577-581): first k with g_nbr==target.
__device__ __forceinline__ int index_of_neighbor(St& S, int node, int target) {
    int s = S.g_off[node], e = S.g_off[node + 1];
    for (int k = s; k < e; k++) if (S.g_nbr[k] == target) return k - s;
    return SIZE_MAX32;   // mwpm.py raises; caller never expects miss
}

// ============================================================================
// GraphFlooder region-growth primitives (mwpm.py GraphFlooder, 823-1080).
// Forward declarations for the de-recursed "op over total area" helpers.
// ============================================================================
__device__ void reschedule_events_at_detector_node(St& S, int node);

// do_region_created_at_empty_detector_node (mwpm.py 830-837)
__device__ void do_region_created_at_empty(St& S, int region, int node) {
    S.nd_reached[node] = node;
    S.nd_arrival[node] = 0;
    S.nd_region[node] = region;
    S.nd_region_top[node] = region;
    S.nd_wrapped[node] = 0;
    shell_push(S, region, node);
    reschedule_events_at_detector_node(S, node);
}

// _find_next_at_growing (mwpm.py 839-864). Returns (best_neighbor local idx, best_time).
__device__ void find_next_growing(St& S, int node, int64_t rad1_yint,
        int* best_nb, int64_t* best_time) {
    *best_time = INF64; *best_nb = SIZE_MAX32;
    int s = S.g_off[node], e = S.g_off[node + 1];
    int start = 0;
    if (e > s && S.g_nbr[s] < 0) {   // boundary neighbor at index 0
        int64_t w = WEIGHT;
        int64_t ct = w - rad1_yint;
        if (ct < *best_time) { *best_time = ct; *best_nb = 0; }
        start = 1;
    }
    for (int k = s + start; k < e; k++) {
        int nb = S.g_nbr[k];
        // has_same_owner_as: region_that_arrived_top equal (both may be -1)
        if (S.nd_region_top[node] == S.nd_region_top[nb]) continue;
        int slope2; int64_t yint2; node_local_radius(S, nb, &slope2, &yint2);
        if (slope2 == -1) continue;                 // rad2 shrinking
        int64_t ct = (int64_t)WEIGHT - rad1_yint - yint2;
        if (slope2 == 1) ct = ct >> 1;              // both growing -> halve (arith shift)
        if (ct < *best_time) { *best_time = ct; *best_nb = k - s; }
    }
}

// _find_next_at_not_growing (mwpm.py 866-881)
__device__ void find_next_not_growing(St& S, int node, int64_t rad1_yint,
        int* best_nb, int64_t* best_time) {
    (void)rad1_yint;
    *best_time = INF64; *best_nb = SIZE_MAX32;
    int s = S.g_off[node], e = S.g_off[node + 1];
    int start = 0;
    if (e > s && S.g_nbr[s] < 0) start = 1;
    // rad1_yint is node.local_radius().y_intercept()
    int slope1; int64_t yint1; node_local_radius(S, node, &slope1, &yint1);
    for (int k = s + start; k < e; k++) {
        int nb = S.g_nbr[k];
        int slope2; int64_t yint2; node_local_radius(S, nb, &slope2, &yint2);
        if (slope2 == 1) {
            int64_t ct = (int64_t)WEIGHT - yint1 - yint2;
            if (ct < *best_time) { *best_time = ct; *best_nb = k - s; }
        }
    }
}

// find_next_event_at_node (mwpm.py 883-888)
__device__ void find_next_event_at_node(St& S, int node, int* best_nb, int64_t* best_time) {
    int slope1; int64_t yint1; node_local_radius(S, node, &slope1, &yint1);
    if (slope1 == 1) find_next_growing(S, node, yint1, best_nb, best_time);
    else find_next_not_growing(S, node, yint1, best_nb, best_time);
}

// reschedule_events_at_detector_node (mwpm.py 890-897)
__device__ void reschedule_events_at_detector_node(St& S, int node) {
    int bi; int64_t t; find_next_event_at_node(S, node, &bi, &t);
    if (bi == SIZE_MAX32) trk_set_no_desired(&S.nd_thd[node]);
    else trk_set_desired(S, NODE_TRK(node), t, LOOK_AT_NODE, node);
}

// schedule_tentative_shrink_event (mwpm.py 899-906)
__device__ void schedule_tentative_shrink_event(St& S, int region) {
    int64_t t;
    if (S.rg_shell_size[region] == 0) {
        t = S.rg_yint[region];                       // time_of_x_intercept_for_shrinking
    } else {
        int last = S.rg_shell_head[region];          // shell_area[-1] == last appended
        int slope; int64_t yint; node_local_radius(S, last, &slope, &yint);
        t = yint;                                    // -t + yint = 0 -> t = yint
    }
    trk_set_desired(S, RG_STRK(region), t, LOOK_AT_SHRINKING_REGION, region);
}

// do_region_arriving_at_empty_detector_node (mwpm.py 908-923)
__device__ void do_region_arriving_at_empty(St& S, int region, int empty_node,
        int from_node, int from_to_empty_idx) {
    int fs = S.g_off[from_node];
    S.nd_obs[empty_node] = S.nd_obs[from_node] ^ S.g_obs[fs + from_to_empty_idx];
    S.nd_reached[empty_node] = S.nd_reached[from_node];
    S.nd_arrival[empty_node] = vary_at(S.rg_slope[region], S.rg_yint[region], CURT);
    S.nd_region[empty_node] = region;
    S.nd_region_top[empty_node] = S.rg_bparent_top[region];
    S.nd_wrapped[empty_node] = compute_wrapped_radius(S, empty_node);
    shell_push(S, region, empty_node);
    reschedule_events_at_detector_node(S, empty_node);
}

// ---- de-recursed do_op_for_each_node_in_total_area (mwpm.py 644-649) ----
// op codes: 0 = reschedule_events_at_detector_node, 1 = node tracker set_no_desired.
// Order: for each region (starting `root`, DFS preorder over blossom_children), emit
// its shell nodes head->... (== reversed(shell_area)); then descend into children.
__device__ void op_total_area(St& S, int root, int opcode) {
    // Uses S.stk3 (NOT S.stk): op_total_area is invoked from within shatter_descendants'
    // live S.stk/S.stk2 post-order traversal (via set_region_frozen), so sharing S.stk here
    // would clobber that traversal's pending nodes and corrupt the matching.
    int sp = 0;
    S.stk3[sp++] = root;
    while (sp > 0) {
        int r = S.stk3[--sp];
        for (int node = S.rg_shell_head[r]; node >= 0; node = S.nd_shell_next[node]) {
            if (opcode == 0) reschedule_events_at_detector_node(S, node);
            else trk_set_no_desired(&S.nd_thd[node]);
        }
        // push children in reverse so they pop in forward (list) order
        int cs = S.rg_bc_start[r], cc = S.rg_bc_count[r];
        for (int i = cc - 1; i >= 0; i--) {
            if (sp >= S.STKCAP) { set_err(S, ERR_OVERFLOW); return; }
            S.stk3[sp++] = S.bc_region[cs + i];
        }
    }
}

// do_region_shrinking (mwpm.py 925-940). Returns an MwpmEvent via out params
// (etype + fields). We encode MwpmEvent inline in the caller.
// Forward-declared helpers for the two special cases:
__device__ int do_blossom_shattering(St& S, int region, int* out);      // fills out MwpmEvent
__device__ int do_degenerate_implosion(St& S, int region, int* out);

// MwpmEvent encoding: out[0]=etype; then depending on type:
//   REGION_HIT_REGION:  out[1]=region1, out[2]=region2, out[3]=edge_from, out[4]=edge_to, out5=obs(hi/lo)
//   REGION_HIT_BOUNDARY:out[1]=region, out[2]=edge_from, out[3]=edge_to, obs
//   BLOSSOM_SHATTER:    out[1]=blossom_region, out[2]=in_parent_region, out[3]=in_child_region
// Because obs is 64-bit we carry it separately (out_obs). We pack ints in out[] and
// obs in a parallel out_obs[1]. See MwpmEvt struct in caller.
struct MwpmEvt {
    int etype;
    int r1, r2, region, blossom, in_parent, in_child;
    int ef, et; Obs eo;        // the CompressedEdge carried by the event
};

__device__ void do_region_shrinking(St& S, int region, MwpmEvt* ev);

// ============================================================================
// The full flooder / matcher live below; because of heavy mutual recursion we
// forward declare the Mwpm-level handlers and implement everything after St setup.
// ============================================================================

// set_region_growing / frozen / shrinking (mwpm.py 1026-1047)
__device__ void set_region_growing(St& S, int region) {
    vary_then_growing(&S.rg_slope[region], &S.rg_yint[region], CURT);
    trk_set_no_desired(&S.rg_sthd[region]);
    op_total_area(S, region, 0);
}
__device__ void set_region_frozen(St& S, int region) {
    bool was_shrinking = (S.rg_slope[region] == -1);
    vary_then_frozen(&S.rg_slope[region], &S.rg_yint[region], CURT);
    trk_set_no_desired(&S.rg_sthd[region]);
    if (was_shrinking) op_total_area(S, region, 0);
}
__device__ void set_region_shrinking(St& S, int region) {
    vary_then_shrinking(&S.rg_slope[region], &S.rg_yint[region], CURT);
    schedule_tentative_shrink_event(S, region);
    op_total_area(S, region, 1);
}

// add_match (mwpm.py GraphFillRegion.add_match, 636-638)
__device__ void region_add_match(St& S, int region, int other, int ef, int et, Obs eo) {
    S.rg_match_region[region] = other; S.rg_mf[region] = ef; S.rg_mt[region] = et; S.rg_mo[region] = eo;
    // other.match = (region, reversed(edge))
    S.rg_match_region[other] = region; S.rg_mf[other] = et; S.rg_mt[other] = ef; S.rg_mo[other] = eo;
}

// do_region_shrinking (mwpm.py 925-940)
__device__ void do_region_shrinking(St& S, int region, MwpmEvt* ev) {
    if (S.rg_shell_size[region] == 0) { do_blossom_shattering(S, region, (int*)ev); return; }
    if (S.rg_shell_size[region] == 1 && S.rg_bc_count[region] == 0) {
        do_degenerate_implosion(S, region, (int*)ev); return;
    }
    int leaving = shell_pop(S, region);
    S.nd_region[leaving] = -1; S.nd_region_top[leaving] = -1; S.nd_wrapped[leaving] = 0;
    S.nd_reached[leaving] = -1; S.nd_arrival[leaving] = 0; S.nd_obs[leaving] = 0;
    reschedule_events_at_detector_node(S, leaving);
    schedule_tentative_shrink_event(S, region);
    ev->etype = NO_EVENT;
}

// do_degenerate_implosion (mwpm.py 976-986)
__device__ int do_degenerate_implosion(St& S, int region, int* out) {
    MwpmEvt* ev = (MwpmEvt*)out;
    int atn = S.rg_alt[region];
    int patn = S.at_par[atn];
    ev->etype = REGION_HIT_REGION;
    ev->r1 = S.at_outer[patn];
    ev->r2 = S.at_outer[atn];
    // edge = (parent.edge.loc_to, inner_to_outer_edge.loc_to,
    //         inner_to_outer_edge.obs ^ parent.edge.obs)
    ev->ef = S.at_pt[atn];               // parent.edge.loc_to
    ev->et = S.at_it[atn];               // inner_to_outer_edge.loc_to
    ev->eo = S.at_io[atn] ^ S.at_po[atn];
    return REGION_HIT_REGION;
}

// do_blossom_shattering (mwpm.py 988-994)
__device__ int do_blossom_shattering(St& S, int region, int* out) {
    MwpmEvt* ev = (MwpmEvt*)out;
    int atn = S.rg_alt[region];
    int pf = S.at_pf[atn];               // parent.edge.loc_from (a node id)
    int iof = S.at_if[atn];              // inner_to_outer_edge.loc_from (a node id)
    ev->etype = BLOSSOM_SHATTER;
    ev->blossom = region;
    ev->in_parent = heir_region_on_shatter(S, pf);
    ev->in_child = heir_region_on_shatter(S, iof);
    return BLOSSOM_SHATTER;
}

// do_neighbor_interaction (mwpm.py 942-964)
__device__ void do_neighbor_interaction(St& S, int src, int src_to_dst_idx, int dst, MwpmEvt* ev) {
    if (S.nd_region[src] >= 0 && S.nd_region[dst] < 0) {
        do_region_arriving_at_empty(S, S.nd_region_top[src], dst, src, src_to_dst_idx);
        ev->etype = NO_EVENT; return;
    }
    if (S.nd_region[dst] >= 0 && S.nd_region[src] < 0) {
        int idx = index_of_neighbor(S, dst, src);
        do_region_arriving_at_empty(S, S.nd_region_top[dst], src, dst, idx);
        ev->etype = NO_EVENT; return;
    }
    int ss = S.g_off[src];
    ev->etype = REGION_HIT_REGION;
    ev->r1 = S.nd_region_top[src];
    ev->r2 = S.nd_region_top[dst];
    ev->ef = S.nd_reached[src];
    ev->et = S.nd_reached[dst];
    ev->eo = S.nd_obs[src] ^ S.nd_obs[dst] ^ S.g_obs[ss + src_to_dst_idx];
}

// do_region_hit_boundary_interaction (mwpm.py 966-974)
__device__ void do_region_hit_boundary_interaction(St& S, int node, MwpmEvt* ev) {
    int ns = S.g_off[node];
    ev->etype = REGION_HIT_BOUNDARY;
    ev->region = S.nd_region_top[node];
    ev->ef = S.nd_reached[node];
    ev->et = -1;
    ev->eo = S.nd_obs[node] ^ S.g_obs[ns + 0];   // neighbor_observables[0] (boundary edge)
}

// do_look_at_node_event (mwpm.py 1049-1063)
__device__ void do_look_at_node_event(St& S, int node, MwpmEvt* ev) {
    int bi; int64_t t; find_next_event_at_node(S, node, &bi, &t);
    if (t == CURT) {
        trk_set_desired(S, NODE_TRK(node), CURT, LOOK_AT_NODE, node);
        int nb = S.g_nbr[S.g_off[node] + bi];
        if (nb < 0) { do_region_hit_boundary_interaction(S, node, ev); return; }
        do_neighbor_interaction(S, node, bi, nb, ev); return;
    } else if (bi != SIZE_MAX32) {
        trk_set_desired(S, NODE_TRK(node), t, LOOK_AT_NODE, node);
    }
    ev->etype = NO_EVENT;
}

// create_blossom (mwpm.py 996-1007). contained_regions given as a contiguous block
// in the bc pool [start, start+count) already populated by the caller.
__device__ int create_blossom(St& S, int bc_start, int bc_count) {
    int blossom = new_region(S);
    S.rg_slope[blossom] = 1; S.rg_yint[blossom] = -CURT;   // growing_zero_at(cur_time)
    S.rg_bc_start[blossom] = bc_start; S.rg_bc_count[blossom] = bc_count;
    for (int i = 0; i < bc_count; i++) {
        int child = S.bc_region[bc_start + i];
        vary_then_frozen(&S.rg_slope[child], &S.rg_yint[child], CURT);
        // wrap_into_blossom(blossom): set blossom_parent, and for each descendant+self
        // set blossom_parent_top and each shell node's region_that_arrived_top & wrapped.
        S.rg_bparent[child] = blossom;
        // de-recursed do_op_for_each_descendant_and_self over `child`
        int sp = 0; S.stk2[sp++] = child;
        while (sp > 0) {
            int d = S.stk2[--sp];
            S.rg_bparent_top[d] = blossom;
            for (int node = S.rg_shell_head[d]; node >= 0; node = S.nd_shell_next[node]) {
                S.nd_region_top[node] = blossom;
                S.nd_wrapped[node] = compute_wrapped_radius(S, node);
            }
            int cs = S.rg_bc_start[d], cc = S.rg_bc_count[d];
            for (int j = cc - 1; j >= 0; j--) {
                if (sp >= S.STKCAP) { set_err(S, ERR_OVERFLOW); break; }
                S.stk2[sp++] = S.bc_region[cs + j];
            }
        }
        trk_set_no_desired(&S.rg_sthd[child]);
    }
    op_total_area(S, blossom, 0);
    return blossom;
}

// ============================================================================
// GraphFlooder main loop (mwpm.py 1009-1080)
// ============================================================================
__device__ void process_tentative_event(St& S, int kind, int target, MwpmEvt* ev) {
    if (kind == LOOK_AT_NODE) { do_look_at_node_event(S, target, ev); return; }
    if (kind == LOOK_AT_SHRINKING_REGION) { do_region_shrinking(S, target, ev); return; }
    ev->etype = NO_EVENT;
}

// dequeue_valid + run_until_next_mwpm_notification (mwpm.py 1018-1079)
__device__ void run_until_next_notification(St& S, MwpmEvt* ev) {
    while (true) {
        // dequeue_valid
        int kind = NO_FLOOD_CHECK_EVENT, target = -1; int64_t etime = 0;
        while (true) {
            int slot = q_dequeue(S);
            if (slot < 0) { kind = NO_FLOOD_CHECK_EVENT; break; }
            etime = S.ev_time[slot]; kind = S.ev_kind[slot]; target = S.ev_target[slot];
            bool ok;
            if (kind == LOOK_AT_NODE)
                ok = trk_dequeue_decision(S, NODE_TRK(target), etime, LOOK_AT_NODE, target);
            else // LOOK_AT_SHRINKING_REGION
                ok = trk_dequeue_decision(S, RG_STRK(target), etime, LOOK_AT_SHRINKING_REGION, target);
            q_free_slot(S, slot);
            if (ok) break;
        }
        if (kind == NO_FLOOD_CHECK_EVENT) { ev->etype = NO_EVENT; return; }
        process_tentative_event(S, kind, target, ev);
        if (ev->etype != NO_EVENT) return;
    }
}

// ============================================================================
// AltTreeNode operations (become_root, MRCA, prune) -- mwpm.py 730-817
// ============================================================================

// most_recent_common_ancestor (mwpm.py 745-777). Returns alt id or -1.
__device__ int alt_mrca(St& S, int a, int b) {
    int this_cur = a; S.at_visited[this_cur] = 1;
    int other_cur = b; S.at_visited[other_cur] = 1;
    int common = -1;
    while (true) {
        int tp = S.at_par[this_cur];
        int op = S.at_par[other_cur];
        if (tp >= 0 || op >= 0) {
            if (tp >= 0) {
                this_cur = tp;
                if (S.at_visited[this_cur]) { common = this_cur; break; }
                S.at_visited[this_cur] = 1;
            }
            if (op >= 0) {
                other_cur = op;
                if (S.at_visited[other_cur]) { common = other_cur; break; }
                S.at_visited[other_cur] = 1;
            }
        } else {
            // clear visited chains for a and b, return -1
            for (int n = a; n >= 0 && S.at_visited[n]; ) { S.at_visited[n] = 0; n = S.at_par[n]; }
            for (int n = b; n >= 0 && S.at_visited[n]; ) { S.at_visited[n] = 0; n = S.at_par[n]; }
            return -1;
        }
    }
    S.at_visited[common] = 0;
    int p = S.at_par[common];
    while (p >= 0 && S.at_visited[p]) { S.at_visited[p] = 0; p = S.at_par[p]; }
    return common;
}

// become_root (mwpm.py 730-743). Recursive in Python (old_parent.become_root()).
// Linearized: walk to the current root collecting the path, then apply the pointer
// reversals from the root down toward `self`, exactly reproducing the recursion's
// post-order effect. See mwpm.py for the per-step transform.
__device__ void alt_become_root(St& S, int self) {
    // collect path self -> root : path[0]=self, path[len-1]=root
    int sp = 0;
    int cur = self;
    while (cur >= 0) { S.stk[sp++] = cur; if (sp >= S.STKCAP) { set_err(S, ERR_OVERFLOW); return; } cur = S.at_par[cur]; }
    // Python: become_root() recurses parent first, THEN rewires. Equivalent to
    // processing nodes from root's child down to self. For node X with old parent P
    // (=path[i+1]): P.become_root already done; then:
    //   P.inner_region = X.inner_region; P.inner_to_outer_edge = X.parent.edge
    //   X.inner_region = None
    //   erase X from P.children
    //   X.parent = empty
    //   X.add_child(AltTreeEdge(P, X.inner_to_outer_edge.reversed()))
    //   X.inner_to_outer_edge = empty
    // We must iterate X from the deepest recursion return upward == path[len-2]..path[0]?
    // Python recursion: self.become_root() calls parent.become_root() first, so the
    // parent is fully rooted before self's body runs; self's body runs LAST. The body
    // uses self and old_parent=self.parent (pre-mutation). Since each node's body only
    // touches itself and its (original) parent, and parents are processed before
    // children, we run bodies in order root-child ... self  == path[len-2] down to 0.
    for (int i = sp - 2; i >= 0; i--) {
        int X = S.stk[i];
        int P = S.stk[i + 1];             // X's original parent (== path[i+1])
        // capture X.parent.edge (X->P edge stored on X as at_pf/pt/po)
        int xpf = S.at_pf[X], xpt = S.at_pt[X]; Obs xpo = S.at_po[X];
        // P.inner_region = X.inner_region ; P.inner_to_outer_edge = X.parent.edge
        S.at_inner[P] = S.at_inner[X];
        if (S.at_inner[P] >= 0) S.rg_alt[S.at_inner[P]] = P;
        S.at_if[P] = xpf; S.at_it[P] = xpt; S.at_io[P] = xpo;
        // X.inner_region = None
        S.at_inner[X] = -1;
        // erase X from P.children
        alt_erase_child(S, P, X);
        // X.parent = empty
        S.at_par[X] = -1; S.at_pf[X] = -1; S.at_pt[X] = -1; S.at_po[X] = 0;
        // X.add_child(AltTreeEdge(old_parent P, X.inner_to_outer_edge.reversed()))
        //   inner_to_outer_edge of X was (at_if/it/io) BEFORE we overwrite it below.
        int xif = S.at_if[X], xit = S.at_it[X]; Obs xio = S.at_io[X];
        // add_child(parent=X, child=P, edge = reversed(X.inner_to_outer_edge))
        alt_add_child(S, X, P, xit, xif, xio);
        // X.inner_to_outer_edge = empty
        S.at_if[X] = -1; S.at_it[X] = -1; S.at_io[X] = 0;
    }
}

// prune_upward_path_stopping_before (mwpm.py 785-817). Collects orphan child edges and
// `pruned` RegionEdge pairs into caller-provided arrays. Returns counts via out params.
// orphan edges are AltTreeEdges (child alt + edge). pruned entries are (region, edge).
struct RegionEdgeBuf { int region; int ef, et; Obs eo; };
__device__ void alt_prune_upward(St& S, int self, int prune_parent, bool back,
        int* orphan_alt, int* orphan_ef, int* orphan_et, Obs* orphan_eo, int* n_orphan,
        RegionEdgeBuf* pruned, int* n_pruned, int cap) {
    int no = 0, np = 0;
    int current = self;
    while (current != prune_parent) {
        // move current.children into orphan list
        int cc = S.at_nchild[current], cbase = current * S.CHILDCAP;
        for (int i = 0; i < cc; i++) {
            if (no >= cap) { set_err(S, ERR_OVERFLOW); break; }
            orphan_alt[no] = S.ch_alt[cbase + i];
            orphan_ef[no] = S.ch_ef[cbase + i]; orphan_et[no] = S.ch_et[cbase + i];
            orphan_eo[no] = S.ch_eo[cbase + i]; no++;
        }
        S.at_nchild[current] = 0;
        int par = S.at_par[current];
        if (back) {
            // pruned += RegionEdge(inner_region, inner_to_outer_edge)
            if (np < cap) { pruned[np].region = S.at_inner[current];
                pruned[np].ef = S.at_if[current]; pruned[np].et = S.at_it[current];
                pruned[np].eo = S.at_io[current]; np++; }
            // pruned += RegionEdge(parent.outer_region, parent.edge.reversed())
            if (np < cap) { pruned[np].region = S.at_outer[par];
                pruned[np].ef = S.at_pt[current]; pruned[np].et = S.at_pf[current];
                pruned[np].eo = S.at_po[current]; np++; }
        } else {
            // pruned += RegionEdge(outer_region, inner_to_outer_edge.reversed())
            if (np < cap) { pruned[np].region = S.at_outer[current];
                pruned[np].ef = S.at_it[current]; pruned[np].et = S.at_if[current];
                pruned[np].eo = S.at_io[current]; np++; }
            // pruned += RegionEdge(inner_region, parent.edge)
            if (np < cap) { pruned[np].region = S.at_inner[current];
                pruned[np].ef = S.at_pf[current]; pruned[np].et = S.at_pt[current];
                pruned[np].eo = S.at_po[current]; np++; }
        }
        alt_erase_child(S, par, current);
        S.rg_alt[S.at_outer[current]] = -1;
        S.rg_alt[S.at_inner[current]] = -1;
        current = par;
    }
    *n_orphan = no; *n_pruned = np;
}

// ============================================================================
// Mwpm matcher (mwpm.py 1101-1315)
// ============================================================================

// create_detection_event (mwpm.py 1107-1112)
__device__ void create_detection_event(St& S, int node) {
    int region = new_region(S);
    int alt = new_alt(S, -1, region, -1, -1, 0);   // AltTreeNode.root(region)
    S.rg_alt[region] = alt;
    do_region_created_at_empty(S, region, node);
}

// shatter_descendants_into_matches_and_freeze (mwpm.py 1117-1131). Recursion over
// children (post-order). Linearized with an explicit stack; we process a node AFTER
// its children. Uses stk2 as a work stack of (alt, state) pairs -- state 0 = first
// visit (push children), state 1 = process self.
__device__ void shatter_descendants(St& S, int root) {
    // iterative post-order: push root; use a visited marker via at_visited-like temp?
    // We reuse stk (node) and stk2 (phase). phase 0 = expand, 1 = process.
    int sp = 0;
    S.stk[sp] = root; S.stk2[sp] = 0; sp++;
    while (sp > 0) {
        int node = S.stk[sp - 1];
        int phase = S.stk2[sp - 1];
        if (phase == 0) {
            S.stk2[sp - 1] = 1;
            int cc = S.at_nchild[node], cbase = node * S.CHILDCAP;
            for (int i = cc - 1; i >= 0; i--) {   // push children (order doesn't affect matches)
                if (sp >= S.STKCAP) { set_err(S, ERR_OVERFLOW); return; }
                S.stk[sp] = S.ch_alt[cbase + i]; S.stk2[sp] = 0; sp++;
            }
        } else {
            sp--;
            // body (mwpm.py 1120-1131)
            if (S.at_inner[node] >= 0) {
                S.at_par[node] = -1; S.at_pf[node] = -1; S.at_pt[node] = -1; S.at_po[node] = 0;
                region_add_match(S, S.at_inner[node], S.at_outer[node],
                                 S.at_if[node], S.at_it[node], S.at_io[node]);
                set_region_frozen(S, S.at_inner[node]);
                set_region_frozen(S, S.at_outer[node]);
                S.rg_alt[S.at_inner[node]] = -1;
                S.rg_alt[S.at_outer[node]] = -1;
            }
            if (S.at_outer[node] >= 0) S.rg_alt[S.at_outer[node]] = -1;
        }
    }
}

// make_child (mwpm.py 1160-1173)
__device__ int make_child(St& S, int parent, int cir, int cor, int cif, int cit,
        Obs cio, int cef, int cet, Obs ceo) {
    int child = new_alt(S, cir, cor, cif, cit, cio);
    alt_add_child(S, parent, child, cef, cet, ceo);
    return child;
}

// handle_tree_hitting_boundary (mwpm.py 1133-1138)
__device__ void handle_tree_hitting_boundary(St& S, MwpmEvt* ev) {
    int node = S.rg_alt[ev->region];
    alt_become_root(S, node);
    shatter_descendants(S, node);
    // event.region.match = Match(None, event.edge)
    S.rg_match_region[ev->region] = -1;
    S.rg_mf[ev->region] = ev->ef; S.rg_mt[ev->region] = ev->et; S.rg_mo[ev->region] = ev->eo;
    set_region_frozen(S, ev->region);
}

// handle_tree_hitting_boundary_match (mwpm.py 1140-1147)
__device__ void handle_tree_hitting_boundary_match(St& S, int unmatched, int matched,
        int ef, int et, Obs eo) {
    int alt = S.rg_alt[unmatched];
    region_add_match(S, unmatched, matched, ef, et, eo);
    set_region_frozen(S, unmatched);
    alt_become_root(S, alt);
    shatter_descendants(S, alt);
}

// handle_tree_hitting_other_tree (mwpm.py 1149-1158)
__device__ void handle_tree_hitting_other_tree(St& S, MwpmEvt* ev) {
    int alt1 = S.rg_alt[ev->r1];
    int alt2 = S.rg_alt[ev->r2];
    alt_become_root(S, alt1);
    alt_become_root(S, alt2);
    shatter_descendants(S, alt1);
    shatter_descendants(S, alt2);
    region_add_match(S, ev->r1, ev->r2, ev->ef, ev->et, ev->eo);
    set_region_frozen(S, ev->r1);
    set_region_frozen(S, ev->r2);
}

// handle_tree_hitting_match (mwpm.py 1175-1188)
__device__ void handle_tree_hitting_match(St& S, int unmatched, int matched,
        int ef, int et, Obs eo) {
    int alt = S.rg_alt[unmatched];
    int other = S.rg_match_region[matched];
    int mmf = S.rg_mf[matched], mmt = S.rg_mt[matched]; Obs mmo = S.rg_mo[matched];
    make_child(S, alt, matched, other, mmf, mmt, mmo, ef, et, eo);
    // other.match.clear(); matched.match.clear()
    S.rg_match_region[other] = -1;
    S.rg_match_region[matched] = -1;
    set_region_shrinking(S, matched);
    set_region_growing(S, other);
}

// handle_tree_hitting_self (mwpm.py 1190-1209)
__device__ void handle_tree_hitting_self(St& S, MwpmEvt* ev, int common) {
    int alt1 = S.rg_alt[ev->r1];
    int alt2 = S.rg_alt[ev->r2];
    // Prune both upward paths. Buffers are per-thread local; PRUNE_CAP bounds the path
    // length (an alternating-tree branch) -- overflow sets ERR_OVERFLOW (grow if it fires).
    const int PRUNE_CAP = 128;
    RegionEdgeBuf pruned1[PRUNE_CAP]; RegionEdgeBuf pruned2[PRUNE_CAP];
    int orphan1_alt[PRUNE_CAP], orphan2_alt[PRUNE_CAP];
    int orphan1_ef[PRUNE_CAP], orphan1_et[PRUNE_CAP]; Obs orphan1_eo[PRUNE_CAP];
    int orphan2_ef[PRUNE_CAP], orphan2_et[PRUNE_CAP]; Obs orphan2_eo[PRUNE_CAP];
    int no1=0, np1=0, no2=0, np2=0;
    alt_prune_upward(S, alt1, common, true,  orphan1_alt, orphan1_ef, orphan1_et, orphan1_eo, &no1, pruned1, &np1, PRUNE_CAP);
    alt_prune_upward(S, alt2, common, false, orphan2_alt, orphan2_ef, orphan2_et, orphan2_eo, &no2, pruned2, &np2, PRUNE_CAP);

    // blossom_cycle = pruned2 + reversed(pruned1) + [RegionEdge(region1, edge)]
    int bc_start = BCN;
    int total = np2 + np1 + 1;
    if (BCN + total > S.BCCAP) { set_err(S, ERR_OVERFLOW); return; }
    int w = bc_start;
    for (int i = 0; i < np2; i++) { S.bc_region[w]=pruned2[i].region; S.bc_ef[w]=pruned2[i].ef; S.bc_et[w]=pruned2[i].et; S.bc_eo[w]=pruned2[i].eo; w++; }
    for (int i = np1 - 1; i >= 0; i--) { S.bc_region[w]=pruned1[i].region; S.bc_ef[w]=pruned1[i].ef; S.bc_et[w]=pruned1[i].et; S.bc_eo[w]=pruned1[i].eo; w++; }
    S.bc_region[w]=ev->r1; S.bc_ef[w]=ev->ef; S.bc_et[w]=ev->et; S.bc_eo[w]=ev->eo; w++;
    BCN += total;

    S.rg_alt[S.at_outer[common]] = -1;
    int blossom = create_blossom(S, bc_start, total);
    S.at_outer[common] = blossom;
    S.rg_alt[blossom] = common;
    for (int i = 0; i < no1; i++) alt_add_child(S, common, orphan1_alt[i], orphan1_ef[i], orphan1_et[i], orphan1_eo[i]);
    for (int i = 0; i < no2; i++) alt_add_child(S, common, orphan2_alt[i], orphan2_ef[i], orphan2_et[i], orphan2_eo[i]);
}

// handle_blossom_shattering (mwpm.py 1211-1277)
__device__ void handle_blossom_shattering(St& S, MwpmEvt* ev) {
    int blossom = ev->blossom;
    int bc_start = S.rg_bc_start[blossom];
    int bsize = S.rg_bc_count[blossom];
    // clear_blossom_parent for each child (mwpm.py 1212-1213)
    for (int i = 0; i < bsize; i++) {
        int child = S.bc_region[bc_start + i];
        // clear_blossom_parent (656-665): blossom_parent=None; op sets blossom_parent_top
        // and node.region_that_arrived_top and recomputes wrapped_radius.
        S.rg_bparent[child] = -1;
        int sp = 0; S.stk2[sp++] = child;
        while (sp > 0) {
            int d = S.stk2[--sp];
            S.rg_bparent_top[d] = child;
            for (int node = S.rg_shell_head[d]; node >= 0; node = S.nd_shell_next[node]) {
                S.nd_region_top[node] = child;
                S.nd_wrapped[node] = compute_wrapped_radius(S, node);
            }
            int cs = S.rg_bc_start[d], cc = S.rg_bc_count[d];
            for (int j = cc - 1; j >= 0; j--) { if (sp >= S.STKCAP) { set_err(S,ERR_OVERFLOW); break; } S.stk2[sp++] = S.bc_region[cs + j]; }
        }
    }
    int blossom_alt = S.rg_alt[blossom];
    int parent_idx = 0, child_idx = 0;
    for (int i = 0; i < bsize; i++) {
        if (S.bc_region[bc_start + i] == ev->in_parent) parent_idx = i;
        if (S.bc_region[bc_start + i] == ev->in_child) child_idx = i;
    }
    int gap = (child_idx + bsize - parent_idx) % bsize;
    int current_alt = S.at_par[blossom_alt];
    alt_erase_child(S, current_alt, blossom_alt);
    // child_edge = blossom_alt.parent.edge.reversed()
    int ce_f = S.at_pt[blossom_alt], ce_t = S.at_pf[blossom_alt]; Obs ce_o = S.at_po[blossom_alt];
    int evens_start, evens_end;
    if (gap % 2 == 0) {
        evens_start = child_idx + 1;
        evens_end = child_idx + bsize - gap;
        for (int i = parent_idx; i < parent_idx + gap; i += 2) {
            int k0 = i % bsize, k1 = (i + 1) % bsize;
            int cir = S.bc_region[bc_start + k0];
            int cor = S.bc_region[bc_start + k1];
            int cif = S.bc_ef[bc_start + k0], cit = S.bc_et[bc_start + k0]; Obs cio = S.bc_eo[bc_start + k0];
            current_alt = make_child(S, current_alt, cir, cor, cif, cit, cio, ce_f, ce_t, ce_o);
            ce_f = S.bc_ef[bc_start + k1]; ce_t = S.bc_et[bc_start + k1]; ce_o = S.bc_eo[bc_start + k1];
            set_region_shrinking(S, S.at_inner[current_alt]);
            set_region_growing(S, S.at_outer[current_alt]);
        }
    } else {
        evens_start = parent_idx + 1;
        evens_end = parent_idx + gap;
        for (int i = 0; i < bsize - gap; i += 2) {
            int k1 = (parent_idx + bsize - i) % bsize;
            int k2 = (parent_idx + bsize - i - 1) % bsize;
            int k3 = (parent_idx + bsize - i - 2) % bsize;
            int cir = S.bc_region[bc_start + k1];
            int cor = S.bc_region[bc_start + k2];
            // edge = blossom_cycle[k2].edge.reversed()
            int cif = S.bc_et[bc_start + k2], cit = S.bc_ef[bc_start + k2]; Obs cio = S.bc_eo[bc_start + k2];
            current_alt = make_child(S, current_alt, cir, cor, cif, cit, cio, ce_f, ce_t, ce_o);
            // child_edge = blossom_cycle[k3].edge.reversed()
            ce_f = S.bc_et[bc_start + k3]; ce_t = S.bc_ef[bc_start + k3]; ce_o = S.bc_eo[bc_start + k3];
            set_region_shrinking(S, S.at_inner[current_alt]);
            set_region_growing(S, S.at_outer[current_alt]);
        }
    }
    // evens loop (mwpm.py 1261-1272)
    for (int j = evens_start; j < evens_end; j += 2) {
        int k1 = j % bsize, k2 = (j + 1) % bsize;
        int r1 = S.bc_region[bc_start + k1], r2 = S.bc_region[bc_start + k2];
        region_add_match(S, r1, r2, S.bc_ef[bc_start + k1], S.bc_et[bc_start + k1], S.bc_eo[bc_start + k1]);
        op_total_area(S, r1, 0);
        op_total_area(S, r2, 0);
    }
    // finalize (mwpm.py 1273-1277)
    S.at_inner[blossom_alt] = S.bc_region[bc_start + child_idx];
    set_region_shrinking(S, S.at_inner[blossom_alt]);
    S.rg_alt[S.bc_region[bc_start + child_idx]] = blossom_alt;
    alt_add_child(S, current_alt, blossom_alt, ce_f, ce_t, ce_o);
}

// handle_region_hit_region (mwpm.py 1279-1303)
__device__ void handle_region_hit_region(St& S, MwpmEvt* ev) {
    int alt1 = S.rg_alt[ev->r1];
    int alt2 = S.rg_alt[ev->r2];
    if (alt1 >= 0 && alt2 >= 0) {
        int common = alt_mrca(S, alt1, alt2);
        if (common < 0) handle_tree_hitting_other_tree(S, ev);
        else handle_tree_hitting_self(S, ev, common);
    } else if (alt1 >= 0) {
        if (S.rg_match_region[ev->r2] >= 0)
            handle_tree_hitting_match(S, ev->r1, ev->r2, ev->ef, ev->et, ev->eo);
        else
            handle_tree_hitting_boundary_match(S, ev->r1, ev->r2, ev->ef, ev->et, ev->eo);
    } else {
        // edge reversed
        if (S.rg_match_region[ev->r1] >= 0)
            handle_tree_hitting_match(S, ev->r2, ev->r1, ev->et, ev->ef, ev->eo);
        else
            handle_tree_hitting_boundary_match(S, ev->r2, ev->r1, ev->et, ev->ef, ev->eo);
    }
}

// process_event (mwpm.py 1305-1315)
__device__ void process_event(St& S, MwpmEvt* ev) {
    if (ev->etype == REGION_HIT_REGION) handle_region_hit_region(S, ev);
    else if (ev->etype == REGION_HIT_BOUNDARY) handle_tree_hitting_boundary(S, ev);
    else if (ev->etype == BLOSSOM_SHATTER) handle_blossom_shattering(S, ev);
}

// ============================================================================
// Extraction, obs_mask path (mwpm.py 1318-1360). Recursive; linearized with an
// explicit work stack that accumulates obs_mask (XOR) and weight (sum).
// ============================================================================

// cleanup_shell_area (mwpm.py 640-642): reset each node in region's shell.
__device__ void cleanup_shell_area(St& S, int region) {
    int node = S.rg_shell_head[region];
    while (node >= 0) { int nxt = S.nd_shell_next[node]; node_reset(S, node); node = nxt; }
    S.rg_shell_head[region] = -1; S.rg_shell_size[region] = 0;
}

// clear_blossom_parent_ignoring_wrapped_radius (mwpm.py 667-675)
__device__ void clear_bparent_ignore_wrapped(St& S, int region) {
    S.rg_bparent[region] = -1;
    int sp = 0; S.stk2[sp++] = region;
    while (sp > 0) {
        int d = S.stk2[--sp];
        S.rg_bparent_top[d] = region;
        for (int node = S.rg_shell_head[d]; node >= 0; node = S.nd_shell_next[node])
            S.nd_region_top[node] = region;
        int cs = S.rg_bc_start[d], cc = S.rg_bc_count[d];
        for (int j = cc - 1; j >= 0; j--) { if (sp >= S.STKCAP) { set_err(S,ERR_OVERFLOW); break; } S.stk2[sp++] = S.bc_region[cs + j]; }
    }
}

// pair_and_shatter_extract_matches (mwpm.py 1318-1337). Modifies region matches, adds
// the odd-cycle pair matches, recursively extracting each; returns the `subblossom`
// region to continue with. Accumulates into *mask,*weight.
// Because shatter_blossom_and_extract_matches is recursive and calls this, we implement
// the whole extraction as one iterative routine using an explicit region work stack.
//
// We follow mwpm.py's control flow precisely (1339-1360):
//   shatter(region):
//     region.cleanup_shell_area()
//     if region.match.region is not None:
//         region.match.region.cleanup_shell_area()
//         if not region.blossom_children and not match.region.blossom_children:
//             return MatchingResult(match.edge.obs, region.rad.yint + match.region.rad.yint)
//     elif not region.blossom_children:
//         return MatchingResult(match.edge.obs, region.rad.yint)
//     res = 0
//     if region.blossom_children: region = pair_and_shatter(region, res)   # may recurse
//     if region.match.region and region.match.region.blossom_children:
//         pair_and_shatter(region.match.region, res)                        # may recurse
//     res += shatter(region)
//     return res
//
// We implement recursion with an explicit stack of "frames". Each frame tracks the
// region and which sub-calls remain. Sub-calls generated by pair_and_shatter (the
// odd-cycle pair matches) are pushed as independent shatter() tasks whose results XOR
// into the shared accumulator.
struct Frame { int region; int stage; int cont_region; int idx; int num; int index0; };

// Record a match edge (loc_from node id, loc_to node id or -1 for boundary) at each base
// case, PARALLEL to the obs_mask accumulation. This mirrors mwpm.py's
// shatter_blossom_and_extract_match_edges (mwpm.py:1476-1490): the N>64 correction path
// reconstructs each matched pair's shortest qubit path via SearchFlooder on the host, so the
// GPU only needs to emit the same match edges the obs_mask base cases already visit. The
// obs_mask output is left intact (used verbatim for N<=64); the host picks per N.
__device__ __forceinline__ void me_push(St& S, int region, int* me_from, int* me_to, int* me_n) {
    int k = *me_n;
    if (k >= S.MECAP) { set_err(S, ERR_OVERFLOW); return; }
    me_from[k] = S.rg_mf[region];
    me_to[k]   = S.rg_mt[region];
    *me_n = k + 1;
}

__device__ void shatter_extract(St& S, int start_region, Obs* mask, int64_t* weight,
                                int* me_from, int* me_to, int* me_n) {
    // explicit recursion stack over shatter() invocations
    // We push tasks; each task is a region to shatter whose result XORs into mask/weight.
    // pair_and_shatter enumerates pairs and pushes their regions as tasks too.
    // Use S.stk as a task stack of region ids (results all fold into the same mask/weight,
    // which is valid because MatchingResult composition is XOR/sum and order-independent
    // for the accumulator -- but the STRUCTURAL mutations must follow program order).
    //
    // Order matters for the mutations inside pair_and_shatter (add_match etc.). We must
    // execute exactly as mwpm.py: process region fully (its pairing + its own tail
    // shatter) before returning. We therefore do a manual recursion.
    int sp = 0;
    Frame* fr = (Frame*)S.stk;   // reuse stk memory as Frame array (sizeof(Frame)=24 -> need 6 ints)
    // Guard capacity: each Frame uses 6 ints.
    int frame_cap = S.STKCAP / 6;
    fr[sp].region = start_region; fr[sp].stage = 0; sp++;

    while (sp > 0) {
        Frame* f = &fr[sp - 1];
        int region = f->region;
        if (f->stage == 0) {
            // base-case checks
            cleanup_shell_area(S, region);
            int mreg = S.rg_match_region[region];
            if (mreg >= 0) {
                cleanup_shell_area(S, mreg);
                if (S.rg_bc_count[region] == 0 && S.rg_bc_count[mreg] == 0) {
                    *mask ^= S.rg_mo[region];
                    *weight += S.rg_yint[region] + S.rg_yint[mreg];
                    me_push(S, region, me_from, me_to, me_n);   // match_edges.append(match.edge)
                    sp--; continue;
                }
            } else if (S.rg_bc_count[region] == 0) {
                *mask ^= S.rg_mo[region];
                *weight += S.rg_yint[region];
                me_push(S, region, me_from, me_to, me_n);       // match_edges.append(match.edge)
                sp--; continue;
            }
            // res starts 0 (folded into shared mask/weight)
            f->stage = 1;
            if (S.rg_bc_count[region] != 0) {
                // region = pair_and_shatter_extract_matches(region, res)
                // -- inline the non-recursive part, push pair tasks, update f->region
                int newregion = -1;
                // clear_blossom_parent_ignoring_wrapped_radius for each child
                int bs = S.rg_bc_start[region], bn = S.rg_bc_count[region];
                for (int i = 0; i < bn; i++) clear_bparent_ignore_wrapped(S, S.bc_region[bs + i]);
                // subblossom = region.match.edge.loc_from.region_that_arrived_top
                int lf = S.rg_mf[region];                 // node id (loc_from)
                int subblossom = S.nd_region_top[lf];
                // subblossom.match = region.match
                S.rg_match_region[subblossom] = S.rg_match_region[region];
                S.rg_mf[subblossom] = S.rg_mf[region]; S.rg_mt[subblossom] = S.rg_mt[region]; S.rg_mo[subblossom] = S.rg_mo[region];
                if (S.rg_match_region[subblossom] >= 0)
                    S.rg_match_region[S.rg_match_region[subblossom]] = subblossom;
                *weight += S.rg_yint[region];
                int index0 = 0;
                for (int i = 0; i < bn; i++) if (S.bc_region[bs + i] == subblossom) { index0 = i; break; }
                for (int i = 0; i + 1 < bn; i += 2) {
                    int reA = S.bc_region[bs + (index0 + i + 1) % bn];
                    int reB = S.bc_region[bs + (index0 + i + 2) % bn];
                    int eef = S.bc_ef[bs + (index0 + i + 1) % bn];
                    int eet = S.bc_et[bs + (index0 + i + 1) % bn];
                    Obs eeo = S.bc_eo[bs + (index0 + i + 1) % bn];
                    region_add_match(S, reA, reB, eef, eet, eeo);
                    // res += shatter(reA)  -> push as task (folds into mask/weight)
                    if (sp >= frame_cap) { set_err(S, ERR_OVERFLOW); return; }
                    fr[sp].region = reA; fr[sp].stage = 0; sp++;
                    f = &fr[sp - 1 - 0];  // NOTE: f invalidated after push; recompute below
                    f = &fr[ (sp-1) ]; (void)f;
                }
                newregion = subblossom;
                // Re-fetch frame pointer (array may have grown); update the ORIGINAL frame.
                // The original frame is at some index < sp; find it: it's the one with stage==1
                // we were operating on. We stored pushes above it. Its index is (sp-1-#pushes).
                // Simpler: we tracked it as the frame before pushes. Recompute by storing region.
                // To keep correctness we locate it: it is the deepest stage==1 frame with region==region.
                // Given single-threaded per-lane execution, easier approach: store cont in a side slot.
                // We set f->region=newregion and f->stage=2 on the ORIGINAL frame:
                // find original frame index:
                // (Pushes were added at top; original is right below them.)
                // Count pushes:
                // -- handled by scanning: the original frame is the first stage==1 below top pushes.
                // For robustness we mark via cont_region.
                // Locate original: it's at index (sp-1) minus number_of_pushed. We know pushes = pairs.
                // Recompute pairs count:
                // (index arithmetic) -- see below fixups.
                // We'll instead set a sentinel by searching downward for stage==1.
                int oi = sp - 1;
                while (oi >= 0 && !(fr[oi].stage == 1 && fr[oi].region == region)) oi--;
                if (oi >= 0) { fr[oi].region = newregion; fr[oi].stage = 2; }
                continue;
            } else {
                f->stage = 2;   // no blossom children on region; fallthrough
                continue;
            }
        }
        if (f->stage == 2) {
            int region2 = f->region;   // possibly updated to subblossom
            int mreg = S.rg_match_region[region2];
            if (mreg >= 0 && S.rg_bc_count[mreg] != 0) {
                // pair_and_shatter_extract_matches(region.match.region, res)
                int bs = S.rg_bc_start[mreg], bn = S.rg_bc_count[mreg];
                for (int i = 0; i < bn; i++) clear_bparent_ignore_wrapped(S, S.bc_region[bs + i]);
                int lf = S.rg_mf[mreg];
                int subblossom = S.nd_region_top[lf];
                S.rg_match_region[subblossom] = S.rg_match_region[mreg];
                S.rg_mf[subblossom] = S.rg_mf[mreg]; S.rg_mt[subblossom] = S.rg_mt[mreg]; S.rg_mo[subblossom] = S.rg_mo[mreg];
                if (S.rg_match_region[subblossom] >= 0)
                    S.rg_match_region[S.rg_match_region[subblossom]] = subblossom;
                *weight += S.rg_yint[mreg];
                int index0 = 0;
                for (int i = 0; i < bn; i++) if (S.bc_region[bs + i] == subblossom) { index0 = i; break; }
                for (int i = 0; i + 1 < bn; i += 2) {
                    int reA = S.bc_region[bs + (index0 + i + 1) % bn];
                    int reB = S.bc_region[bs + (index0 + i + 2) % bn];
                    region_add_match(S, reA, reB, S.bc_ef[bs + (index0 + i + 1) % bn],
                                     S.bc_et[bs + (index0 + i + 1) % bn], S.bc_eo[bs + (index0 + i + 1) % bn]);
                    if (sp >= frame_cap) { set_err(S, ERR_OVERFLOW); return; }
                    fr[sp].region = reA; fr[sp].stage = 0; sp++;
                }
                // Note: pair_and_shatter returns subblossom but mwpm.py discards it here
                // (region stays region2 for the final tail shatter). Actually mwpm.py:
                //   self.pair_and_shatter_extract_matches(region.match.region, res)  (no reassign)
                // then res += shatter(region)  -- region unchanged.
                // Re-find original frame and set stage 3.
                int oi = sp - 1;
                while (oi >= 0 && !(fr[oi].stage == 2 && fr[oi].region == region2)) oi--;
                if (oi >= 0) fr[oi].stage = 3;
                continue;
            }
            f->stage = 3;
            continue;
        }
        if (f->stage == 3) {
            // res += shatter(region2): convert this frame into a fresh shatter of region2
            int region2 = f->region;
            f->stage = 0;   // restart as base shatter on (possibly updated) region2
            f->region = region2;
            continue;
        }
    }
}

// ============================================================================
// Kernel: one thread per shot.
// ============================================================================
__global__ void k_mwpm_decode(
        // static graph
        const int* g_off, const int* g_nbr, const Obs* g_obs, int M, int N,
        // capacities
        int REGCAP, int ALTCAP, int BCCAP, int QCAP, int CHILDCAP, int STKCAP,
        // syndrome + outputs
        const uint8_t* synd, Obs* out_mask, int* out_err, int B,
        // match-edge outputs (N>64 path): per shot up to MECAP (from,to) node-id pairs
        int* out_mef, int* out_met, int* out_menum, int MECAP,
        // per-thread arenas (each [B*cap])
        int* nd_region, int* nd_region_top, int64_t* nd_wrapped, int* nd_reached,
        Obs* nd_obs, int64_t* nd_arrival, int* nd_shell_next,
        int64_t* nd_td, int64_t* nd_tq, uint8_t* nd_thd, uint8_t* nd_thq,
        int* rg_bparent, int* rg_bparent_top, int* rg_alt, int* rg_slope, int64_t* rg_yint,
        int64_t* rg_std, int64_t* rg_stq, uint8_t* rg_sthd, uint8_t* rg_sthq,
        int* rg_match_region, int* rg_mf, int* rg_mt, Obs* rg_mo,
        int* rg_bc_start, int* rg_bc_count, int* rg_shell_head, int* rg_shell_size,
        int* bc_region, int* bc_ef, int* bc_et, Obs* bc_eo,
        int* at_inner, int* at_outer, int* at_if, int* at_it, Obs* at_io,
        int* at_par, int* at_pf, int* at_pt, Obs* at_po, int* at_nchild, uint8_t* at_visited,
        int* ch_alt, int* ch_ef, int* ch_et, Obs* ch_eo,
        int64_t* ev_time, int* ev_kind, int* ev_target, int* ev_next, int* ev_prev,
        int* bkt_head, int* bkt_tail,
        int* stk, int* stk2, int* stk3,
        int64_t* q_curtime, int* q_num, int* q_free,
        int* reg_count, int* alt_count, int* bc_count, int* errbuf) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    St S;
    S.g_off = g_off; S.g_nbr = g_nbr; S.g_obs = g_obs; S.M = M; S.N = N;
    S.REGCAP = REGCAP; S.ALTCAP = ALTCAP; S.BCCAP = BCCAP; S.QCAP = QCAP;
    S.CHILDCAP = CHILDCAP; S.STKCAP = STKCAP; S.MECAP = MECAP;
    // offset every pointer to thread b's slice
    S.nd_region = nd_region + (int64_t)b*M; S.nd_region_top = nd_region_top + (int64_t)b*M;
    S.nd_wrapped = nd_wrapped + (int64_t)b*M; S.nd_reached = nd_reached + (int64_t)b*M;
    S.nd_obs = nd_obs + (int64_t)b*M; S.nd_arrival = nd_arrival + (int64_t)b*M;
    S.nd_shell_next = nd_shell_next + (int64_t)b*M;
    S.nd_td = nd_td + (int64_t)b*M; S.nd_tq = nd_tq + (int64_t)b*M;
    S.nd_thd = nd_thd + (int64_t)b*M; S.nd_thq = nd_thq + (int64_t)b*M;
    S.rg_bparent = rg_bparent + (int64_t)b*REGCAP; S.rg_bparent_top = rg_bparent_top + (int64_t)b*REGCAP;
    S.rg_alt = rg_alt + (int64_t)b*REGCAP; S.rg_slope = rg_slope + (int64_t)b*REGCAP; S.rg_yint = rg_yint + (int64_t)b*REGCAP;
    S.rg_std = rg_std + (int64_t)b*REGCAP; S.rg_stq = rg_stq + (int64_t)b*REGCAP;
    S.rg_sthd = rg_sthd + (int64_t)b*REGCAP; S.rg_sthq = rg_sthq + (int64_t)b*REGCAP;
    S.rg_match_region = rg_match_region + (int64_t)b*REGCAP;
    S.rg_mf = rg_mf + (int64_t)b*REGCAP; S.rg_mt = rg_mt + (int64_t)b*REGCAP; S.rg_mo = rg_mo + (int64_t)b*REGCAP;
    S.rg_bc_start = rg_bc_start + (int64_t)b*REGCAP; S.rg_bc_count = rg_bc_count + (int64_t)b*REGCAP;
    S.rg_shell_head = rg_shell_head + (int64_t)b*REGCAP; S.rg_shell_size = rg_shell_size + (int64_t)b*REGCAP;
    S.bc_region = bc_region + (int64_t)b*BCCAP; S.bc_ef = bc_ef + (int64_t)b*BCCAP;
    S.bc_et = bc_et + (int64_t)b*BCCAP; S.bc_eo = bc_eo + (int64_t)b*BCCAP;
    S.at_inner = at_inner + (int64_t)b*ALTCAP; S.at_outer = at_outer + (int64_t)b*ALTCAP;
    S.at_if = at_if + (int64_t)b*ALTCAP; S.at_it = at_it + (int64_t)b*ALTCAP; S.at_io = at_io + (int64_t)b*ALTCAP;
    S.at_par = at_par + (int64_t)b*ALTCAP; S.at_pf = at_pf + (int64_t)b*ALTCAP;
    S.at_pt = at_pt + (int64_t)b*ALTCAP; S.at_po = at_po + (int64_t)b*ALTCAP;
    S.at_nchild = at_nchild + (int64_t)b*ALTCAP; S.at_visited = at_visited + (int64_t)b*ALTCAP;
    S.ch_alt = ch_alt + (int64_t)b*ALTCAP*CHILDCAP; S.ch_ef = ch_ef + (int64_t)b*ALTCAP*CHILDCAP;
    S.ch_et = ch_et + (int64_t)b*ALTCAP*CHILDCAP; S.ch_eo = ch_eo + (int64_t)b*ALTCAP*CHILDCAP;
    S.ev_time = ev_time + (int64_t)b*QCAP; S.ev_kind = ev_kind + (int64_t)b*QCAP;
    S.ev_target = ev_target + (int64_t)b*QCAP; S.ev_next = ev_next + (int64_t)b*QCAP; S.ev_prev = ev_prev + (int64_t)b*QCAP;
    S.bkt_head = bkt_head + (int64_t)b*65; S.bkt_tail = bkt_tail + (int64_t)b*65;
    S.stk = stk + (int64_t)b*STKCAP; S.stk2 = stk2 + (int64_t)b*STKCAP;
    S.stk3 = stk3 + (int64_t)b*STKCAP;
    S.q_curtime = q_curtime + b; S.q_num = q_num + b; S.q_free = q_free + b;
    S.reg_count = reg_count + b; S.alt_count = alt_count + b; S.bc_count = bc_count + b;
    S.err = errbuf + b;

    // ---- per-thread init ----
    ERRF = ERR_OK; REGN = 0; ALTN = 0; BCN = 0; QNUM = 0; CURT = 0;
    // Hang guard bound: generous vs. any real decode (event counts are ~O(M)); a false trip
    // only costs a CPU redo of that one shot, never correctness, so we err large.
    S.guard = 0; S.guard_max = (int64_t)2000000 + (int64_t)200000 * (int64_t)M;
    for (int i = 0; i <= 64; i++) { S.bkt_head[i] = -1; S.bkt_tail[i] = -1; }
    for (int i = 0; i < QCAP; i++) S.ev_next[i] = i + 1; S.ev_next[QCAP - 1] = -1; QFREE = 0;
    for (int i = 0; i < M; i++) node_reset(S, i);

    // Per-thread match-edge output slice (N>64 path); count starts at 0 so any early
    // return leaves out_menum[b]==0 (the host reads no stale pairs).
    int* mef = out_mef + (int64_t)b * MECAP;
    int* met = out_met + (int64_t)b * MECAP;
    int   me_n = 0;
    out_menum[b] = 0;

    if (N > 64 * OBSW) { out_mask[b] = 0; out_err[b] = ERR_UNSUPPORTED; return; }

    // ---- detection events (nonzero syndrome bits) ----
    const uint8_t* srow = synd + (int64_t)b*M;
    bool any = false;
    for (int i = 0; i < M; i++) if (srow[i]) { create_detection_event(S, i); any = true; }
    if (!any) { out_mask[b] = 0; out_err[b] = ERRF; return; }

    // ---- flood until no more notifications ----
    MwpmEvt ev;
    while (true) {
        run_until_next_notification(S, &ev);
        // The hang guard trips inside q_dequeue (sets ERRF); catch it before acting on ev.
        if (ERRF != ERR_OK) { out_mask[b] = 0; out_err[b] = ERRF; return; }
        if (ev.etype == NO_EVENT) break;
        process_event(S, &ev);
        if (ERRF != ERR_OK) { out_mask[b] = 0; out_err[b] = ERRF; return; }
    }

    // ---- extract obs_mask (N<=64) AND match edges (N>64), one traversal (mwpm.py 1549-1563
    //      / 1657-1672). The obs_mask is the correction for N<=64; the emitted (from,to)
    //      match-edge node pairs drive the host SearchFlooder path reconstruction for N>64. ----
    Obs mask = 0; int64_t weight = 0;
    for (int i = 0; i < M; i++) {
        if (srow[i] && S.nd_region[i] >= 0) {
            shatter_extract(S, S.nd_region_top[i], &mask, &weight, mef, met, &me_n);
            if (ERRF != ERR_OK) { out_mask[b] = 0; out_err[b] = ERRF; return; }
        }
    }
    out_mask[b] = mask;
    out_menum[b] = me_n;
    out_err[b] = ERRF;
}

// ============================================================================
// Host launcher + pybind
// ============================================================================
std::vector<torch::Tensor> mwpm_decode_cuda(
        torch::Tensor g_off, torch::Tensor g_nbr, torch::Tensor g_obs,
        int64_t M, int64_t N, torch::Tensor synd) {
    TORCH_CHECK(synd.is_cuda(), "synd must be CUDA");
    int64_t B = synd.size(0);
    auto dev = synd.device();
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);
    auto opt_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opt_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(dev);

    // capacities derived from M (generous; overflow reported via err)
    int REGCAP = (int)(4 * M + 16);
    int ALTCAP = (int)(4 * M + 16);
    int BCCAP  = (int)(32 * M + 64);
    int QCAP   = (int)(8 * M + 64);
    int CHILDCAP = 32;
    int STKCAP = (int)(8 * M + 64);
    // Match edges per shot <= number of detection events <= M (each matched pair -- two
    // detectors, or one detector + boundary -- emits exactly one edge). M is a safe upper
    // bound; overflow is reported via err (host falls back to the CPU blossom for that shot).
    int MECAP = (int)M;

    g_off = g_off.to(torch::kInt32).contiguous();
    g_nbr = g_nbr.to(torch::kInt32).contiguous();
    // g_obs carries per-edge observable masks as OBSW int64 words (bit=qubit) laid out
    // [E, OBSW] row-major (avoids a Python-side torch.uint64 dependency); reinterpreted to
    // the OBSW-word `Obs` struct below. Packed by _build_csr in mwpm_cuda.py (OBSW must match).
    g_obs = g_obs.to(torch::kInt64).contiguous();
    auto synd_u8 = synd.to(torch::kUInt8).contiguous();

    auto out_mask = torch::zeros({B, OBSW}, opt_i64);   // [B, OBSW] int64 bit-pattern of the Obs mask
    auto out_err  = torch::zeros({B}, opt_i32);
    // Match-edge outputs (N>64 path): per shot up to MECAP (from,to) detector node-id pairs
    // (to == -1 means the boundary). The host reconstructs the qubit path per pair.
    auto out_mef   = torch::zeros({B, MECAP}, opt_i32);
    auto out_met   = torch::zeros({B, MECAP}, opt_i32);
    auto out_menum = torch::zeros({B}, opt_i32);

    auto NI32 = [&](int64_t n){ return torch::empty({B, n}, opt_i32); };
    auto NI64 = [&](int64_t n){ return torch::empty({B, n}, opt_i64); };
    auto NU8  = [&](int64_t n){ return torch::empty({B, n}, opt_u8); };
    // Observable arenas: n Obs slots per thread, each OBSW int64 words. Backed by an int64
    // tensor of n*OBSW columns and reinterpret_cast<Obs*> at launch.
    auto NOBS = [&](int64_t n){ return torch::empty({B, n * OBSW}, opt_i64); };
    (void)opt_u64;

    auto nd_region=NI32(M), nd_region_top=NI32(M), nd_reached=NI32(M), nd_shell_next=NI32(M);
    auto nd_wrapped=NI64(M), nd_arrival=NI64(M), nd_td=NI64(M), nd_tq=NI64(M);
    auto nd_obs=NOBS(M); auto nd_thd=NU8(M), nd_thq=NU8(M);

    auto rg_bparent=NI32(REGCAP), rg_bparent_top=NI32(REGCAP), rg_alt=NI32(REGCAP), rg_slope=NI32(REGCAP);
    auto rg_yint=NI64(REGCAP), rg_std=NI64(REGCAP), rg_stq=NI64(REGCAP);
    auto rg_sthd=NU8(REGCAP), rg_sthq=NU8(REGCAP);
    auto rg_match_region=NI32(REGCAP), rg_mf=NI32(REGCAP), rg_mt=NI32(REGCAP); auto rg_mo=NOBS(REGCAP);
    auto rg_bc_start=NI32(REGCAP), rg_bc_count=NI32(REGCAP), rg_shell_head=NI32(REGCAP), rg_shell_size=NI32(REGCAP);

    auto bc_region=NI32(BCCAP), bc_ef=NI32(BCCAP), bc_et=NI32(BCCAP); auto bc_eo=NOBS(BCCAP);

    auto at_inner=NI32(ALTCAP), at_outer=NI32(ALTCAP), at_if=NI32(ALTCAP), at_it=NI32(ALTCAP);
    auto at_io=NOBS(ALTCAP);
    auto at_par=NI32(ALTCAP), at_pf=NI32(ALTCAP), at_pt=NI32(ALTCAP); auto at_po=NOBS(ALTCAP);
    auto at_nchild=NI32(ALTCAP); auto at_visited=NU8(ALTCAP);

    auto ch_alt=NI32((int64_t)ALTCAP*CHILDCAP), ch_ef=NI32((int64_t)ALTCAP*CHILDCAP), ch_et=NI32((int64_t)ALTCAP*CHILDCAP);
    auto ch_eo=NOBS((int64_t)ALTCAP*CHILDCAP);

    auto ev_time=NI64(QCAP); auto ev_kind=NI32(QCAP), ev_target=NI32(QCAP), ev_next=NI32(QCAP), ev_prev=NI32(QCAP);
    auto bkt_head=NI32(65), bkt_tail=NI32(65);
    auto stk=NI32(STKCAP), stk2=NI32(STKCAP), stk3=NI32(STKCAP);

    auto q_curtime=torch::empty({B}, opt_i64);
    auto q_num=torch::empty({B}, opt_i32), q_free=torch::empty({B}, opt_i32);
    auto reg_count=torch::empty({B}, opt_i32), alt_count=torch::empty({B}, opt_i32), bc_count=torch::empty({B}, opt_i32);

    auto stream = at::cuda::getCurrentCUDAStream();
    k_mwpm_decode<<<grid1d(B), THREADS, 0, stream>>>(
        g_off.data_ptr<int>(), g_nbr.data_ptr<int>(),
        reinterpret_cast<const Obs*>(g_obs.data_ptr<int64_t>()), (int)M, (int)N,
        REGCAP, ALTCAP, BCCAP, QCAP, CHILDCAP, STKCAP,
        synd_u8.data_ptr<uint8_t>(),
        reinterpret_cast<Obs*>(out_mask.data_ptr<int64_t>()), out_err.data_ptr<int>(), (int)B,
        out_mef.data_ptr<int>(), out_met.data_ptr<int>(), out_menum.data_ptr<int>(), MECAP,
        nd_region.data_ptr<int>(), nd_region_top.data_ptr<int>(), nd_wrapped.data_ptr<int64_t>(),
        nd_reached.data_ptr<int>(), reinterpret_cast<Obs*>(nd_obs.data_ptr<int64_t>()), nd_arrival.data_ptr<int64_t>(),
        nd_shell_next.data_ptr<int>(), nd_td.data_ptr<int64_t>(), nd_tq.data_ptr<int64_t>(),
        nd_thd.data_ptr<uint8_t>(), nd_thq.data_ptr<uint8_t>(),
        rg_bparent.data_ptr<int>(), rg_bparent_top.data_ptr<int>(), rg_alt.data_ptr<int>(),
        rg_slope.data_ptr<int>(), rg_yint.data_ptr<int64_t>(), rg_std.data_ptr<int64_t>(),
        rg_stq.data_ptr<int64_t>(), rg_sthd.data_ptr<uint8_t>(), rg_sthq.data_ptr<uint8_t>(),
        rg_match_region.data_ptr<int>(), rg_mf.data_ptr<int>(), rg_mt.data_ptr<int>(), reinterpret_cast<Obs*>(rg_mo.data_ptr<int64_t>()),
        rg_bc_start.data_ptr<int>(), rg_bc_count.data_ptr<int>(), rg_shell_head.data_ptr<int>(), rg_shell_size.data_ptr<int>(),
        bc_region.data_ptr<int>(), bc_ef.data_ptr<int>(), bc_et.data_ptr<int>(), reinterpret_cast<Obs*>(bc_eo.data_ptr<int64_t>()),
        at_inner.data_ptr<int>(), at_outer.data_ptr<int>(), at_if.data_ptr<int>(), at_it.data_ptr<int>(), reinterpret_cast<Obs*>(at_io.data_ptr<int64_t>()),
        at_par.data_ptr<int>(), at_pf.data_ptr<int>(), at_pt.data_ptr<int>(), reinterpret_cast<Obs*>(at_po.data_ptr<int64_t>()),
        at_nchild.data_ptr<int>(), at_visited.data_ptr<uint8_t>(),
        ch_alt.data_ptr<int>(), ch_ef.data_ptr<int>(), ch_et.data_ptr<int>(), reinterpret_cast<Obs*>(ch_eo.data_ptr<int64_t>()),
        ev_time.data_ptr<int64_t>(), ev_kind.data_ptr<int>(), ev_target.data_ptr<int>(),
        ev_next.data_ptr<int>(), ev_prev.data_ptr<int>(), bkt_head.data_ptr<int>(), bkt_tail.data_ptr<int>(),
        stk.data_ptr<int>(), stk2.data_ptr<int>(), stk3.data_ptr<int>(),
        q_curtime.data_ptr<int64_t>(), q_num.data_ptr<int>(), q_free.data_ptr<int>(),
        reg_count.data_ptr<int>(), alt_count.data_ptr<int>(), bc_count.data_ptr<int>(), out_err.data_ptr<int>());

    return {out_mask, out_err, out_mef, out_met, out_menum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mwpm_decode", &mwpm_decode_cuda,
          "Bit-exact batched MWPM decode (one CUDA thread per shot; obs_mask for N<=64, "
          "match-edges for N>64; N<=64*OBSW)",
          py::arg("g_off"), py::arg("g_nbr"), py::arg("g_obs"),
          py::arg("M"), py::arg("N"), py::arg("synd"));
}
