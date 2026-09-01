from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

Edge = Tuple[int, int]  # always (min(u,v), max(u,v)), matching C++ Edge{u,v}


def _make_edge(u: int, v: int) -> Edge:
    """Mirror C++ ``Edge(ul, vl)``: u = min, v = max."""
    return (u, v) if u < v else (v, u)


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


def _edge_hash(e: Edge) -> int:
    return e[0] ^ (e[1] << 1)  # std::hash<Edge>: h1 ^ (h2 << 1), h identity


# Lattice (transformation of LatticeFromParity.hpp)
@dataclass
class Lattice:
    """Detector lattice built from a graphlike parity matrix H ([M, N])."""

    num_vertices: int  # M for toric, M+1 for surface (extra boundary vertex)
    num_edges: int  # N (qubit columns)
    vertex_connections: List[List[int]]  # neighbor lists (tsl iteration order)
    edge_to_qubit: Dict[Edge, int]  # Edge(u,v) -> qubit index
    vertex_connection_count: List[int]  # degree of each vertex
    boundary_vertex: Optional[int] = None  # id M for surface, None for toric


def build_lattice_from_parity(H) -> Lattice:
    """Transformation of ``LatticeFromParity`` (single-shot constructor).

    The graph is loaded differently for the two code families, selected purely by
    the column weights of H:

      * TORIC (every column weight 2): built exactly as the reference -- M vertices,
        no boundary, ``vertex_connections`` in tsl bucket order -- so the grow/fuse
        order (and thus the correction) matches ``chaeyeunpark/UnionFind`` bit-for-bit.

      * SURFACE / open boundary (some column weight 1): one boundary vertex is added
        at index M and each weight-1 (boundary) qubit becomes an edge ``(r0, M)``.
        Interior (weight-2) edges keep the identical indexing, so this is a strict
        superset of the toric build -- toric matrices, having no weight-1 column,
        never trigger the boundary branch and stay byte-identical.

    Columns of weight > 2 are non-graphlike and rejected.
    """
    H = np.asarray(H.todense() if hasattr(H, "todense") else H).astype(np.uint8)
    M, N = H.shape

    # construct_qubit_associated_parities: rows per qubit, increasing row order.
    qubit_parities: List[List[int]] = [[] for _ in range(N)]
    for p in range(M):
        for q in np.nonzero(H[p])[0]:
            qubit_parities[int(q)].append(p)

    weights = [len(qp) for qp in qubit_parities]
    if any(w > 2 for w in weights):
        bad = next(q for q, w in enumerate(weights) if w > 2)
        raise ValueError(
            f"Column {bad} has weight {weights[bad]} > 2; H is not graphlike, so the "
            f"union_find decoder does not apply (use a hypergraph/correlated decoder)."
        )

    # Surface codes have weight-1 boundary columns; toric codes do not. Only add the
    # boundary vertex when one is actually needed, so toric indexing is unchanged.
    has_boundary = any(w == 1 for w in weights)
    boundary_vertex = M if has_boundary else None
    num_vertices = M + (1 if has_boundary else 0)

    # construct_edge_idx: iterate q = 0..N-1, first qubit to map to an edge wins.
    edge_table = _RobinTable(_edge_hash)
    edge_to_qubit: Dict[Edge, int] = {}
    for q in range(N):
        qp = qubit_parities[q]
        if len(qp) >= 2:
            edge = _make_edge(qp[0], qp[1])  # interior edge
        elif len(qp) == 1:
            edge = _make_edge(qp[0], boundary_vertex)  # boundary edge
        else:
            continue  # weight-0: qubit in no check, never grown
        if edge_table.insert(edge):  # first appearance
            edge_to_qubit[edge] = q

    # construct_vertex_connections_from_edges: iterate edge table in bucket order.
    conns: List[List[int]] = [[] for _ in range(num_vertices)]
    for edge in edge_table.keys():
        u, v = edge
        conns[u].append(v)
        conns[v].append(u)

    return Lattice(
        num_vertices=num_vertices,
        num_edges=N,
        vertex_connections=conns,
        edge_to_qubit=edge_to_qubit,
        vertex_connection_count=[len(c) for c in conns],
        boundary_vertex=boundary_vertex,
    )


# Reference-faithful serial decode (grow -> fuse -> peel over the fusion forest).
def _find_root(root_of, v):
    """Union-find find with path compression (reference ``find_root``)."""
    tmp = root_of[v]
    if tmp == v:
        return v
    path = []
    root = tmp
    while True:
        root = tmp
        path.append(root)
        tmp = root_of[root]
        if tmp == root:
            break
    for x in path:
        root_of[x] = root
    return root


def decode_shot(lattice: Lattice, syndrome) -> np.ndarray:
    """Reference-faithful single-shot Union-Find decode over ``lattice``.

    Reproduces chaeyeunpark/UnionFind exactly: growth iterates ``odd_roots`` x
    ``border_vertices`` x ``vertex_connections`` in tsl order (via ``_RobinTable``),
    recording each cluster-merging edge into the fusion forest (``peeling_edges``);
    the peel operates on that forest (its result is a per-subtree parity, independent
    of peel order), so a toric matrix yields the reference's exact correction. On a surface lattice the boundary vertex neutralises an odd
    cluster that reaches it (``touch_bd``) and the peel roots at the boundary (that
    vertex is never the peeled leaf) so an unpaired defect is absorbed there rather
    than stranded on a real check -- corrections stay valid (``H @ c == synd``).

    ``syndrome``: length-M (real checks) 0/1 iterable. Returns an [N] uint8 correction.
    """
    V = lattice.num_vertices
    N = lattice.num_edges
    B = lattice.boundary_vertex  # None for toric
    M = V if B is None else B  # number of real checks (syndrome width)
    edge_idx = lattice.edge_to_qubit
    conns = lattice.vertex_connections
    vcc = lattice.vertex_connection_count

    synd = [int(x) & 1 for x in syndrome]
    syndrome_vertices = [n for n in range(M) if synd[n] != 0]
    n = len(syndrome_vertices)

    root_of = list(range(V))
    roots = _RobinTable(lambda k: k)
    odd_roots = _RobinTable(lambda k: k)
    size: Dict[int, int] = {}
    parity: Dict[int, int] = {}
    border: Dict[int, _RobinTable] = {}
    touch_bd = set()  # roots whose cluster contains the boundary vertex
    if n:
        roots.reserve(2 * n)
        odd_roots.reserve(2 * n)
    for r in syndrome_vertices:
        roots.insert(r)
        odd_roots.insert(r)
        size[r] = 1
        parity[r] = 1
        bs = _RobinTable(lambda k: k)
        bs.insert(r)
        border[r] = bs

    support = [0] * N
    connection_counts = [0] * V
    peeling_edges: List[Edge] = []
    synd_vec = [0] * V
    for v in syndrome_vertices:
        synd_vec[v] = 1

    max_rounds = N + 2  # growth is monotone; converges in <= N rounds
    for _round in range(max_rounds):
        active = [r for r in odd_roots.keys() if r not in touch_bd]
        if not active:
            break
        fuse_list: List[Edge] = []
        for root in active:
            for bv in border[root].keys():
                for w in conns[bv]:
                    e = (bv, w) if bv < w else (w, bv)
                    q = edge_idx[e]
                    s = support[q]
                    if s == 2:
                        continue
                    support[q] = s + 1
                    if s + 1 == 2:
                        connection_counts[e[0]] += 1
                        connection_counts[e[1]] += 1
                        fuse_list.append(e)
        for fe in fuse_list:
            r1 = _find_root(root_of, fe[0])
            r2 = _find_root(root_of, fe[1])
            if r1 == r2:
                continue
            peeling_edges.append(fe)
            if size.get(r1, 0) < size.get(r2, 0):
                r1, r2 = r2, r1
            root_of[r2] = r1
            if not roots.contains(r2):
                # r2 is a lone (non-syndrome) vertex being absorbed
                size[r1] = size.get(r1, 0) + 1
                border[r1].insert(r2)
                if r2 == B:
                    touch_bd.add(r1)
            else:
                new_parity = parity.get(r1, 0) + parity.get(r2, 0)
                if new_parity % 2 == 1:
                    odd_roots.insert(r1)
                else:
                    odd_roots.erase(r1)
                size[r1] = size.get(r1, 0) + size.get(r2, 0)
                parity[r1] = new_parity
                odd_roots.erase(r2)
                size.pop(r2, None)
                parity.pop(r2, None)
                roots.erase(r2)
                b1, b2 = border[r1], border[r2]
                b1.range_insert(b2.keys())
                for vtx in b2.keys():
                    if connection_counts[vtx] == vcc[vtx]:
                        b1.erase(vtx)
                del border[r2]
                if (r2 in touch_bd) or (r1 in touch_bd):
                    touch_bd.add(r1)

    vertex_count: Dict[int, int] = {}
    for e in peeling_edges:
        vertex_count[e[0]] = vertex_count.get(e[0], 0) + 1
        vertex_count[e[1]] = vertex_count.get(e[1], 0) + 1
    correction = np.zeros(N, dtype=np.uint8)
    dq = deque(peeling_edges)
    while dq:
        leaf = dq.pop()  # back
        a, b = leaf
        if a != B and vertex_count.get(a, 0) == 1:
            u, v = a, b
        elif b != B and vertex_count.get(b, 0) == 1:
            u, v = b, a
        else:
            dq.appendleft(leaf)  # not a peelable leaf yet
            continue
        vertex_count[u] -= 1
        vertex_count[v] -= 1
        if synd_vec[u] == 1:
            correction[edge_idx[leaf]] = 1
            synd_vec[u] = 0
            synd_vec[v] ^= 1
    return correction


class create(torch.nn.Module):
    """Union-Find decoder consuming one of the bundle's Hx/Hz (toric/graphlike) matrices."""

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super().__init__()
        logger.info("Creating union-find decoder.")

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

        self.dtype = decoder_cfg.get("dtype", "float64")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(f"Invalid dtype <{self.dtype}>, default to float64.")
            self.dtype = "float64"
        self.dtype = torch.__dict__[self.dtype]

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

        self.lattice = build_lattice_from_parity(H_np)
        self.V = self.lattice.num_vertices
        self.boundary = self.lattice.boundary_vertex
        self.has_boundary = self.boundary is not None

        self.algo = "union_find"
        self.batch_size = 1
        self.num_max_iter = self.N + 1
        logger.info(
            f"Complete. (V={self.V}, N={self.N}, "
            f"boundary={'yes' if self.has_boundary else 'no'})"
        )

    def forward(self, io_dict):
        logger.info("Initializing union-find decoding.")
        synd_in = io_dict["synd"]
        B, M = synd_in.shape
        self.batch_size = B
        assert M == self.M, f"syndrome width {M} != H rows {self.M}"

        synd_np = (synd_in != 0).to(dtype=torch.uint8).detach().cpu().numpy()
        correction = np.zeros((B, self.N), dtype=np.int64)
        for b in range(B):
            correction[b] = decode_shot(self.lattice, synd_np[b])

        e_v = torch.from_numpy(correction).to(device=self.device, dtype=self.dtype)
        # sign-encoded soft output (+1 bit 0, -1 bit 1); UF carries no real LLR
        llr = (1.0 - 2.0 * e_v).to(device=self.device, dtype=self.dtype)
        converge = torch.ones(B, dtype=torch.int64, device=self.device)
        # iter mirrors the syndrome weight per shot (clamped >= 1)
        iters = (
            torch.from_numpy(synd_np.sum(axis=1).astype(np.int64))
            .clamp(min=1)
            .to(self.device)
        )

        logger.info("Complete.")
        io_dict.update({"e_v": e_v, "iter": iters, "llr": llr, "converge": converge})
        return io_dict
