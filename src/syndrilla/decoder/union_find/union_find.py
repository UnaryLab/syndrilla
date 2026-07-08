"""union_find.py -- batched PyTorch Union-Find decoder for toric/graphlike codes.

Union-Find (Delfosse-Nickerson, arXiv:1709.06218) is the unweighted syndrome-validation +
peeling decoder. This module provides the Syndrilla decoder that runs it fully batched:

  * ``create(decoder_cfg, bundle=...)`` -- the Syndrilla ``nn.Module`` decoder consuming
    ``io_dict`` (algorithm key ``union_find``). Its runtime decode is the fully-batched,
    loop-free tensor algorithm (all B shots run together as tensor ops -- no per-shot
    Python loop): it emits valid corrections (``H @ c == synd``), but is NOT bit-for-bit
    identical to the reference C++ decoder (chaeyeunpark/UnionFind) -- it matches its
    *logical error rate*, not the exact degenerate correction it picks.

The toric lattice is built by ``build_lattice_from_parity`` (below), a clean-room port of
the reference's ``LatticeFromParity.hpp`` that reproduces the ``tsl::robin_map`` iteration
order exactly (``_RobinTable``, edge hash ``u ^ (v<<1)``, power-of-two growth from 0,
max-load-factor 0.5, robin-hood insertion, backward-shift erase) so edge indexing matches
the reference.

The decoder handles any **graphlike** CSS code: every qubit column of H must touch *at
most two* parity rows. A boundary vertex (index 0) is prepended and weight-1 columns
(open boundaries, e.g. planar surface codes) become boundary edges, mirroring the repo's
MWPM decoder; toric codes (all weight-2 columns) leave the boundary vertex isolated and
decode exactly as before. Columns of weight > 2 (non-graphlike) are rejected. The C++
reference ``UnionFindPy.Decoder`` is used only in the tests as the logical-error-rate
oracle (its domain is toric codes).

YAML algorithm key: union_find
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

Edge = Tuple[int, int]  # always (min(u,v), max(u,v)), matching C++ Edge{u,v}


def _make_edge(u: int, v: int) -> Edge:
    """Mirror C++ ``Edge(ul, vl)``: u = min, v = max."""
    return (u, v) if u < v else (v, u)


# =========================================================================== #
# _RobinTable: faithful replica of tsl::robin_set / robin_map *iteration order*.
#
# We only need the order in which keys are visited (not the mapped values, tracked
# separately), so a single open-addressing table suffices. Mirrors robin-map commit
# 622443f: power_of_two_growth_policy<2>, DEFAULT_MAX_LOAD_FACTOR 0.5, initial buckets
# 0, robin-hood insertion (steal from the rich), backward-shift erase, in-bucket-order
# rehash reinsertion. std::hash is identity on integers (libc++/libstdc++), so vertex
# keys hash to themselves and Edge keys to ``u ^ (v<<1)`` (see std::hash<Edge>).
# Each bucket is ``None`` (empty) or ``[dist_from_ideal, key]``.
# =========================================================================== #
def _round_up_pow2(value: int) -> int:
    if value == 0:
        return 1
    if (value & (value - 1)) == 0:
        return value
    return 1 << (value - 1).bit_length()


class _RobinTable:
    __slots__ = ("_hash", "mask", "size", "buckets")

    def __init__(self, hashfn: Callable[[object], int]) -> None:
        self._hash = hashfn
        self.mask = 0
        self.size = 0
        self.buckets: List[Optional[list]] = []  # length mask+1 (0 => empty table)

    # ---- growth / rehash ---------------------------------------------------
    def _rehash_impl(self, count: int) -> None:
        if count == 0:
            new_mask, new_buckets = 0, []
        else:
            c = _round_up_pow2(count)
            new_mask, new_buckets = c - 1, [None] * c
        old = self.buckets
        self.buckets = new_buckets
        self.mask = new_mask
        for entry in old:
            if entry is not None:
                self._place_on_rehash(entry[1])

    def _rehash(self, count: int) -> None:
        count = max(count, 2 * self.size)  # ceil(size / 0.5)
        self._rehash_impl(count)

    def reserve(self, count: int) -> None:
        self._rehash(2 * count)  # ceil(count / 0.5)

    def _place_on_rehash(self, key) -> None:
        ib = self._hash(key) & self.mask
        dist = 0
        while True:
            b = self.buckets[ib]
            if b is None:
                self.buckets[ib] = [dist, key]
                return
            if dist > b[0]:
                self.buckets[ib] = [dist, key]
                dist, key = b[0], b[1]
            dist += 1
            ib = (ib + 1) & self.mask

    def _insert_value_impl(self, ib: int, dist: int, key) -> None:
        # Unconditional steal at entry bucket (caller guarantees dist > existing dist).
        e = self.buckets[ib]
        self.buckets[ib] = [dist, key]
        dist, key = e[0], e[1]
        ib = (ib + 1) & self.mask
        dist += 1
        while self.buckets[ib] is not None:
            if dist > self.buckets[ib][0]:
                e = self.buckets[ib]
                self.buckets[ib] = [dist, key]
                dist, key = e[0], e[1]
            ib = (ib + 1) & self.mask
            dist += 1
        self.buckets[ib] = [dist, key]

    # ---- public ops --------------------------------------------------------
    def contains(self, key) -> bool:
        if not self.buckets:
            return False
        ib = self._hash(key) & self.mask
        dist = 0
        while True:
            b = self.buckets[ib]
            if b is None or dist > b[0]:
                return False
            if b[1] == key:
                return True
            ib = (ib + 1) & self.mask
            dist += 1

    def insert(self, key) -> bool:
        # Existence probe (skip while dist <= stored dist), mirroring insert_impl.
        if self.buckets:
            ib = self._hash(key) & self.mask
            dist = 0
            while True:
                b = self.buckets[ib]
                if b is None or dist > b[0]:
                    break
                if b[1] == key:
                    return False
                ib = (ib + 1) & self.mask
                dist += 1
        # rehash_on_extreme_load: grow when size >= floor(bucket_count * 0.5).
        # (An empty table has bucket_count 0 => threshold 0 => always grows first.)
        load_threshold = (self.mask + 1) // 2 if self.buckets else 0
        if self.size >= load_threshold:
            self._rehash_impl((self.mask + 1) * 2)
        # Re-probe for the insertion point in the (possibly grown) table.
        ib = self._hash(key) & self.mask
        dist = 0
        while self.buckets[ib] is not None and dist <= self.buckets[ib][0]:
            ib = (ib + 1) & self.mask
            dist += 1
        if self.buckets[ib] is None:
            self.buckets[ib] = [dist, key]
        else:
            self._insert_value_impl(ib, dist, key)
        self.size += 1
        return True

    def range_insert(self, keys: List) -> None:
        """Transformation of tsl ``insert(first, last)``: reserve for the whole range up-front
        (when free buckets are insufficient) before inserting one-by-one. The up-front
        reserve changes the final bucket count -- and thus iteration order -- versus
        inserting the same keys individually, so it must be replicated faithfully."""
        nb = len(keys)
        load_threshold = (self.mask + 1) // 2 if self.buckets else 0
        nb_free = load_threshold - self.size
        if nb > 0 and nb_free < nb:
            self.reserve(self.size + nb)
        for k in keys:
            self.insert(k)

    def erase(self, key) -> bool:
        if not self.buckets:
            return False
        ib = self._hash(key) & self.mask
        dist = 0
        while True:
            b = self.buckets[ib]
            if b is None or dist > b[0]:
                return False
            if b[1] == key:
                self.buckets[ib] = None
                self.size -= 1
                prev = ib
                cur = (ib + 1) & self.mask
                while self.buckets[cur] is not None and self.buckets[cur][0] > 0:
                    self.buckets[prev] = [
                        self.buckets[cur][0] - 1,
                        self.buckets[cur][1],
                    ]
                    self.buckets[cur] = None
                    prev = cur
                    cur = (cur + 1) & self.mask
                return True
            ib = (ib + 1) & self.mask
            dist += 1

    def keys(self) -> List:
        """Iterate keys in bucket order (== tsl iteration order)."""
        return [b[1] for b in self.buckets if b is not None]

    def empty(self) -> bool:
        return self.size == 0


def _edge_hash(e: Edge) -> int:
    return e[0] ^ (e[1] << 1)  # std::hash<Edge>: h1 ^ (h2 << 1), h identity


# =========================================================================== #
# Lattice (transformation of LatticeFromParity.hpp)
# =========================================================================== #
@dataclass
class Lattice:
    """Detector lattice built from a graphlike (toric) parity matrix H ([M, N])."""

    num_vertices: int  # M (parity rows)
    num_edges: int  # N (qubit columns)
    vertex_connections: List[List[int]]  # [M] neighbor lists (tsl iteration order)
    edge_to_qubit: Dict[Edge, int]  # Edge(u,v) -> qubit index
    vertex_connection_count: List[int]  # degree of each vertex

    def edge_idx(self, edge: Edge) -> int:
        return self.edge_to_qubit[edge]


def build_lattice_from_parity(H) -> Lattice:
    """Transformation of ``LatticeFromParity`` (single-shot constructor).

    Every column must have weight >= 2 (toric). ``vertex_connections`` is built by
    iterating the ``edge_idx`` table in tsl bucket order -- not sorted -- so the grow
    step's fuse order matches the reference exactly.
    """
    H = np.asarray(H.todense() if hasattr(H, "todense") else H).astype(np.uint8)
    M, N = H.shape

    # construct_qubit_associated_parities: rows per qubit, increasing row order.
    qubit_parities: List[List[int]] = [[] for _ in range(N)]
    for p in range(M):
        for q in np.nonzero(H[p])[0]:
            qubit_parities[int(q)].append(p)

    # construct_edge_idx: iterate q = 0..N-1, first qubit to map to an edge wins.
    edge_table = _RobinTable(_edge_hash)
    edge_to_qubit: Dict[Edge, int] = {}
    for q in range(N):
        qp = qubit_parities[q]
        if len(qp) < 2:
            raise ValueError(
                f"Column {q} has weight {len(qp)} < 2; the chaeyeunpark Union-Find "
                f"reference is toric-only (every qubit must touch two parities)."
            )
        edge = _make_edge(qp[0], qp[1])
        if edge_table.insert(edge):  # first appearance
            edge_to_qubit[edge] = q

    # construct_vertex_connections_from_edges: iterate edge table in bucket order.
    conns: List[List[int]] = [[] for _ in range(M)]
    for edge in edge_table.keys():
        u, v = edge
        conns[u].append(v)
        conns[v].append(u)

    return Lattice(
        num_vertices=M,
        num_edges=N,
        vertex_connections=conns,
        edge_to_qubit=edge_to_qubit,
        vertex_connection_count=[len(c) for c in conns],
    )


# =========================================================================== #
# Batched decode graph build (toric edge endpoints from the parity matrix).
# =========================================================================== #
def _edge_endpoints_from_parity(H_np):
    """Vectorized graphlike graph build: the detector vertices each qubit column links.

    A boundary vertex is prepended at index 0 and the M detector rows are shifted to
    1..M (mirroring the repo's MWPM decoder's shared open-boundary node). Each qubit
    column becomes one graph edge, by column weight:

      * weight 2  -> ``(r0+1, r1+1)``   interior edge between two detectors
      * weight 1  -> ``(0, r0+1)``      boundary edge (open boundary, e.g. surface code)
      * weight 0  -> ``(0, 0)``         boundary self-loop (qubit in no check; inert)
      * weight >2 -> rejected           not graphlike (as in MWPM); needs a hypergraph decoder

    Returns ``(eu, ev, V, is_pure_toric)``: int64 arrays [N] of (min, max) endpoints, the
    vertex count ``V = M + 1``, and whether every column has weight exactly two (a closed
    toric code with no boundary edge -- the domain of the bit-exact CUDA kernel). Placing
    the boundary at id 0 makes it every boundary-touching cluster's representative, so the
    peel sinks that cluster's unpaired defect into the boundary (where it is absorbed).
    """
    H_np = np.asarray(H_np).astype(np.uint8)
    M, N = H_np.shape
    # np.nonzero on H^T visits columns 0..N-1 in order, rows increasing within a column.
    cols, rows = np.nonzero(H_np.T)
    counts = np.bincount(cols, minlength=N)
    if np.any(counts > 2):
        bad = int(np.argmax(counts > 2))
        raise ValueError(
            f"Column {bad} has weight {int(counts[bad])} > 2; H is not graphlike, so the "
            f"union_find decoder does not apply (use a hypergraph/correlated decoder)."
        )
    eu = np.zeros(N, dtype=np.int64)  # weight-0 default: boundary self-loop (0, 0)
    ev = np.zeros(N, dtype=np.int64)
    per_col = np.split(
        rows + 1, np.cumsum(counts)[:-1]
    )  # detectors per column, +1 shift
    for q in range(N):
        r = per_col[q]
        if r.size == 2:
            eu[q], ev[q] = int(r[0]), int(r[1])  # rows increasing => already (min, max)
        elif r.size == 1:
            ev[q] = int(r[0])  # boundary edge (0, detector); eu stays 0
    is_pure_toric = bool(np.all(counts == 2))
    return eu, ev, M + 1, is_pure_toric


# =========================================================================== #
# Syndrilla decoder module (io_dict contract).
# =========================================================================== #
class create(torch.nn.Module):
    """Union-Find decoder consuming one of the bundle's Hx/Hz (toric/graphlike) matrices."""

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super().__init__()
        logger.info("Creating union-find decoder.")

        # ---- device -----------------------------------------------------------
        device_cfg = decoder_cfg.get("device", {})
        self.device = device_cfg.get(
            "device_type", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device not in {
            "cuda",
            "cpu",
            torch.device("cuda"),
            torch.device("cpu"),
        }:
            logger.warning(
                f"Invalid input device <{self.device}>, default to available device."
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            device_idx = device_cfg.get("device_idx", 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(
                    f"Invalid device index <{device_idx}>, default to cuda:0."
                )
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_idx}")

        # ---- dtype ------------------------------------------------------------
        self.dtype = decoder_cfg.get("dtype", "float64")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(f"Invalid dtype <{self.dtype}>, default to float64.")
            self.dtype = "float64"
        self.dtype = torch.__dict__[self.dtype]

        # ---- check type & matrix bundle ---------------------------------------
        self.check_type = decoder_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(f"Invalid check type <{self.check_type}>, default to hx.")
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "union_find requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(
            self.check_type
        )

        H_np = np.asarray(self.H_matrix.detach().cpu().numpy()).astype(np.uint8)
        self.M, self.N = H_np.shape  # M detector rows, N qubit columns

        # Graphlike graph tensors for the batched decode -- the (up to two) detector
        # vertices each qubit column links, constant across every decode, on-device as
        # buffers. A boundary vertex is prepended at index 0, so the internal vertex
        # dimension is V = M + 1 (self.M stays the syndrome width). ``is_pure_toric``
        # tells the CUDA subclass whether the bit-exact (toric-only) kernel applies.
        eu, ev, self.V, self.is_pure_toric = _edge_endpoints_from_parity(H_np)
        self.register_buffer("eu", torch.from_numpy(eu).to(self.device))
        self.register_buffer("ev", torch.from_numpy(ev).to(self.device))

        self.algo = "union_find"
        self.batch_size = 1
        self.num_max_iter = self.N + 1
        self.cap = None
        self.cap_bypass = False
        self.cap_active_last = False
        logger.info("Complete.")

    # ----------------------------------------------------------------------- #
    # Batched decode primitives. Every tensor is [B, M] (vertices) or [B, N]
    # (edges); nothing loops over the batch or over individual vertices/edges.
    # ----------------------------------------------------------------------- #
    # Convergence is checked only every _CHECK_EVERY iterations rather than every one:
    # each check is a GPU->CPU sync that stalls the async launch pipeline, and once a
    # loop reaches its fixpoint the extra iterations are idempotent (gather/scatter
    # reproduce the same tensors), so batching the checks changes no result -- it only
    # trades a few redundant device-side iterations for far fewer host syncs.
    _CHECK_EVERY = 4

    def _connected_components(self, grown, init_root=None):
        """Component label (= min vertex id in the component) per vertex over the grown
        edges, as a fully-flattened forest (``root[v]`` == the component's min vertex id).

        Uses FastSV-style label propagation (Zhang-Azad-Hu): each round hooks a vertex's
        *parent* and *grandparent* onto its neighbour's grandparent and then shortcuts to
        the grandparent, so labels travel a doubling distance per round -- O(log D) rounds
        instead of the O(D) of plain one-hop ``amin`` propagation (~4x fewer iterations on
        surface-code clusters, measured). It converges to the exact same unique min-label
        components, so the whole downstream decode is bit-identical to the naive version.

        grown: [B, N] bool. ``init_root`` [B, M] optionally warm-starts from a previous
        (coarser-grown) round -- growth only merges clusters, so a prior round's flattened
        labels are a valid, faster starting point. Returns root [B, M] int64.
        """
        B = grown.shape[0]
        M, N = self.V, self.N  # M = internal vertex count (detectors + boundary)
        eu = self.eu.view(1, N).expand(B, N)
        ev = self.ev.view(1, N).expand(B, N)
        if init_root is None:
            f = torch.arange(M, device=self.device).view(1, M).expand(B, M).contiguous()
        else:
            f = init_root  # read-only here (each round rebuilds via clone/minimum)
        big = torch.full((B, N), M, dtype=torch.long, device=self.device)
        for it in range(M):  # bounded; O(log D) rounds in practice
            gp = torch.gather(f, 1, f)  # grandparent gp[v] = f[f[v]]
            fu, fv = torch.gather(f, 1, eu), torch.gather(f, 1, ev)  # parent at ends
            gpu, gpv = torch.gather(gp, 1, eu), torch.gather(gp, 1, ev)  # gp at ends
            val_v = torch.where(grown, gpv, big)  # non-grown edges never win the amin
            val_u = torch.where(grown, gpu, big)
            new_f = f.clone()
            # hook the parent AND grandparent of each endpoint onto the other's grandparent
            new_f.scatter_reduce_(1, fu, val_v, reduce="amin")
            new_f.scatter_reduce_(1, fv, val_u, reduce="amin")
            new_f.scatter_reduce_(1, gpu, val_v, reduce="amin")
            new_f.scatter_reduce_(1, gpv, val_u, reduce="amin")
            new_f = torch.minimum(new_f, gp)  # shortcut toward the grandparent
            converged = (it + 1) % self._CHECK_EVERY == 0 and torch.equal(new_f, f)
            f = new_f
            if converged:
                break
        # Flatten any residual parent chains to the component representative so root[v] is
        # the component's min id (required: it indexes the per-cluster parity scatter).
        # Pointer-jumping halves chain length per step -> M.bit_length() steps suffice for
        # any chain <= M, run unconditionally (no per-step host sync).
        for _ in range(max(1, M.bit_length())):
            f = torch.gather(f, 1, f)
        return f

    def _grow_fuse(self, synd):
        """Run the grow/fuse fixed point. Returns (root [B, M], grown [B, N])."""
        B = synd.shape[0]
        M, N = self.V, self.N  # M = internal vertex count (detectors + boundary)
        eu, ev = self.eu, self.ev
        support = torch.zeros((B, N), dtype=torch.long, device=self.device)
        root = torch.arange(M, device=self.device).view(1, M).expand(B, M).contiguous()
        for _ in range(M):  # bounded outer loop (BP-`max_iter` style)
            parity_root = torch.zeros((B, M), dtype=torch.long, device=self.device)
            parity_root.scatter_add_(1, root, synd)
            cluster_parity = torch.gather(parity_root, 1, root).remainder(2)
            odd = cluster_parity == 1  # [B, M]
            # A cluster containing the boundary vertex (id 0, always its component's min ->
            # its root) is even: the open boundary absorbs the unpaired defect, so it stops
            # growing. No-op for toric codes, where the boundary vertex stays isolated.
            odd = odd & (root != 0)
            if not odd.any():
                break
            delta = odd[:, eu].long() + odd[:, ev].long()  # [B, N]
            support = torch.clamp(support + delta, max=2)
            grown = support == 2
            # Warm-start from the previous round's labels: growth only merges clusters,
            # so last round's root is a valid start that converges to the same fixpoint.
            root = self._connected_components(grown, init_root=root)
        return root, support == 2

    def _spanning_forest(self, root, grown):
        """BFS forest over grown edges rooted at each component's representative
        (the min-id vertex). Returns (parent [B, M], tree_edge [B, M], has_parent [B, M]).
        ``tree_edge[b, v]`` is the qubit index of the edge linking v to its parent.
        """
        B = root.shape[0]
        M, N = self.V, self.N  # M = internal vertex count (detectors + boundary)
        eu = self.eu.view(1, N).expand(B, N)
        ev = self.ev.view(1, N).expand(B, N)
        vids = torch.arange(M, device=self.device).view(1, M).expand(B, M)
        INF = M + 1

        # distance from the nearest representative (root[v] == v) via grown edges
        is_rep = root == vids
        dist = torch.where(is_rep, torch.zeros_like(root), torch.full_like(root, INF))
        big = torch.full((B, N), INF, dtype=torch.long, device=self.device)
        for it in range(M):  # bounded BFS relaxation
            du = torch.gather(dist, 1, eu)
            dv = torch.gather(dist, 1, ev)
            cand_v = torch.where(grown, du + 1, big)  # reach ev from eu
            cand_u = torch.where(grown, dv + 1, big)  # reach eu from ev
            new_dist = dist.clone()
            new_dist.scatter_reduce_(1, ev, cand_v, reduce="amin")
            new_dist.scatter_reduce_(1, eu, cand_u, reduce="amin")
            converged = (it + 1) % self._CHECK_EVERY == 0 and torch.equal(
                new_dist, dist
            )
            dist = new_dist
            if converged:
                break

        # parent edge of v = smallest-index grown edge to a dist-1 neighbour
        du = torch.gather(dist, 1, eu)
        dv = torch.gather(dist, 1, ev)
        edge_id = torch.arange(N, device=self.device).view(1, N).expand(B, N)
        INF_E = N
        big_e = torch.full((B, N), INF_E, dtype=torch.long, device=self.device)
        qual_v = grown & (du == dv - 1)  # edge is v's (=ev) parent edge
        qual_u = grown & (dv == du - 1)  # edge is u's (=eu) parent edge
        cand_edge_v = torch.where(qual_v, edge_id, big_e)
        cand_edge_u = torch.where(qual_u, edge_id, big_e)
        best_edge = torch.full((B, M), INF_E, dtype=torch.long, device=self.device)
        best_edge.scatter_reduce_(1, ev, cand_edge_v, reduce="amin")
        best_edge.scatter_reduce_(1, eu, cand_edge_u, reduce="amin")

        has_parent = best_edge < INF_E
        e = best_edge.clamp(max=N - 1)  # safe gather index; masked by has_parent
        pe_u = self.eu[e]  # [B, M]
        pe_v = self.ev[e]
        vids_full = torch.arange(M, device=self.device).view(1, M).expand(B, M)
        parent = torch.where(pe_u == vids_full, pe_v, pe_u)
        parent = torch.where(has_parent, parent, vids_full)
        tree_edge = torch.where(has_parent, best_edge, torch.full_like(best_edge, -1))
        return parent, tree_edge, has_parent

    def _peel(self, synd, parent, tree_edge, has_parent):
        """Leaf-rake peeling of the spanning forest. Returns correction [B, N] int64."""
        B = synd.shape[0]
        M, N = self.V, self.N  # M = internal vertex count (detectors + boundary)
        correction = torch.zeros((B, N), dtype=torch.long, device=self.device)
        synd_cur = synd.clone()
        active = has_parent.clone()  # representatives are never peeled
        safe_edge = tree_edge.clamp(min=0)  # -1 (reps) -> 0; masked by corr_val
        for it in range(M):  # bounded; one layer of leaves per round
            child_count = torch.zeros((B, M), dtype=torch.long, device=self.device)
            child_count.scatter_add_(1, parent, active.long())
            is_leaf = active & (child_count == 0)
            # A round with no leaves leaves every tensor below unchanged (flip is empty),
            # so only probe for termination every _CHECK_EVERY rounds to cut host syncs.
            if (it + 1) % self._CHECK_EVERY == 0 and not is_leaf.any():
                break
            flip = is_leaf & (synd_cur == 1)
            correction.scatter_add_(1, safe_edge, flip.long())  # distinct edges/leaf
            par_flip = torch.zeros((B, M), dtype=torch.long, device=self.device)
            par_flip.scatter_add_(1, parent, flip.long())
            synd_cur = synd_cur ^ par_flip.remainder(2)  # XOR flips into parents
            synd_cur = torch.where(is_leaf, torch.zeros_like(synd_cur), synd_cur)
            active = active & ~is_leaf
        return correction

    def forward(self, io_dict):
        logger.info("Initializing union-find decoding.")
        synd_in = io_dict["synd"]
        B, M = synd_in.shape
        self.batch_size = B
        assert M == self.M, f"syndrome width {M} != H rows {self.M}"

        synd_det = (synd_in != 0).to(device=self.device, dtype=torch.long)  # [B, M]
        # Prepend the boundary vertex (index 0, never a defect): internal syndrome is [B, V].
        synd = torch.zeros((B, self.V), dtype=torch.long, device=self.device)
        synd[:, 1:] = synd_det
        root, grown = self._grow_fuse(synd)
        parent, tree_edge, has_parent = self._spanning_forest(root, grown)
        correction = self._peel(synd, parent, tree_edge, has_parent)  # [B, N] int64

        e_v = correction.to(device=self.device, dtype=self.dtype)
        # sign-encoded soft output (+1 bit 0, -1 bit 1); UF carries no real LLR
        llr = (1.0 - 2.0 * e_v).to(device=self.device, dtype=self.dtype)
        converge = torch.ones(B, dtype=torch.int64, device=self.device)
        # iter mirrors the syndrome weight per shot (clamped >= 1)
        iters = synd.sum(dim=1).clamp(min=1).to(torch.int64)

        logger.info("Complete.")
        io_dict.update({"e_v": e_v, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
