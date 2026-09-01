import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger


@dataclass
class MatchingGraph:
    """Detector graph built from a graphlike parity-check matrix H of shape [M, N].

    Nodes 0..M-1 are checks; node M is the single shared boundary. Each qubit column
    j becomes at most one edge whose id IS j, so a correction over edges maps back to
    a correction over the N qubits directly.
    """

    num_nodes: int  # M + 1 (checks + boundary)
    boundary: int  # the boundary node index, == M
    num_qubits: int  # N (columns of H)
    edge_u: np.ndarray  # [E] first endpoint of each edge (int64)
    edge_v: np.ndarray  # [E] second endpoint (== boundary for weight-1 columns)
    edge_qubit: np.ndarray  # [E] qubit/column index that the edge corresponds to
    # CSR adjacency over nodes -> incident (neighbor, edge) for fast growth/peeling.
    adj_offsets: np.ndarray  # [num_nodes + 1]
    adj_neighbor: np.ndarray  # [2E]
    adj_edge: np.ndarray  # [2E]


def build_matching_graph(H) -> MatchingGraph:
    """Build the detector graph from a graphlike GF(2) check matrix H ([M, N]).

    Raises ValueError if any column has weight > 2 (H is not graphlike, so plain
    matching does not apply -- a hypergraph/correlated decoder would be needed).
    """
    H = np.asarray(H).astype(np.uint8)
    M, N = H.shape
    boundary = M
    num_nodes = M + 1

    edge_u, edge_v, edge_qubit = [], [], []
    for j in range(N):
        checks = np.nonzero(H[:, j])[0]
        w = checks.size
        if w == 0:
            continue  # qubit in no check: never an edge here
        if w == 1:
            edge_u.append(int(checks[0]))
            edge_v.append(boundary)  # boundary edge
        elif w == 2:
            edge_u.append(int(checks[0]))
            edge_v.append(int(checks[1]))
        else:
            raise ValueError(
                f"Column {j} has weight {w} > 2; H is not graphlike, so the "
                f"MWPM decoder does not apply."
            )
        edge_qubit.append(j)

    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    edge_qubit = np.asarray(edge_qubit, dtype=np.int64)

    # Build CSR adjacency: each edge contributes (u->v) and (v->u).
    E = edge_u.size
    deg = np.zeros(num_nodes, dtype=np.int64)
    for e in range(E):
        deg[edge_u[e]] += 1
        deg[edge_v[e]] += 1
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(deg)
    neighbor = np.empty(2 * E, dtype=np.int64)
    edge_of = np.empty(2 * E, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for e in range(E):
        u, v = int(edge_u[e]), int(edge_v[e])
        neighbor[cursor[u]] = v
        edge_of[cursor[u]] = e
        cursor[u] += 1
        neighbor[cursor[v]] = u
        edge_of[cursor[v]] = e
        cursor[v] += 1

    return MatchingGraph(
        num_nodes=num_nodes,
        boundary=boundary,
        num_qubits=N,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_qubit=edge_qubit,
        adj_offsets=offsets,
        adj_neighbor=neighbor,
        adj_edge=edge_of,
    )


SIZE_MAX = -1  # sentinel for "no index"
INF = 1 << 62


class RadixHeapQueue:
    def __init__(self):
        self.bit_buckets = [[] for _ in range(64 + 1)]
        self.cur_time = 0
        self._num = 0

    def size(self):
        return self._num

    def empty(self):
        return self._num == 0

    def cur_bit_bucket_for(self, time):
        return (time ^ self.cur_time).bit_length()  # std::bit_width

    def enqueue(self, event):
        b = self.cur_bit_bucket_for(event.time)
        self.bit_buckets[b].append(event)
        self._num += 1

    def dequeue(self):
        if self._num == 0:
            return None
        if not self.bit_buckets[0]:
            b = 1
            while not self.bit_buckets[b]:
                b += 1
            if b == 1:
                self.bit_buckets[0], self.bit_buckets[1] = (
                    self.bit_buckets[1],
                    self.bit_buckets[0],
                )
                self.cur_time += 1
            else:
                src = self.bit_buckets[b]
                min_time = min(e.time for e in src)
                self.cur_time = min_time
                for e in src:
                    self.bit_buckets[self.cur_bit_bucket_for(e.time)].append(e)
                src.clear()
        self._num -= 1
        return self.bit_buckets[0].pop()  # back() + pop_back()  -> LIFO

    def reset(self):
        for bk in self.bit_buckets:
            bk.clear()
        self.cur_time = 0
        self._num = 0


class QueuedEventTracker:
    def __init__(self):
        self.desired_time = 0
        self.queued_time = 0
        self.has_desired_time = False
        self.has_queued_time = False

    def clear(self):
        self.desired_time = 0
        self.queued_time = 0
        self.has_desired_time = False
        self.has_queued_time = False

    def set_desired_event(self, ev, queue):
        self.has_desired_time = True
        self.desired_time = ev.time
        if (not self.has_queued_time) or self.queued_time > ev.time:
            self.queued_time = ev.time
            self.has_queued_time = True
            queue.enqueue(ev)

    def set_no_desired_event(self):
        self.has_desired_time = False

    def dequeue_decision(self, ev, queue):
        if (not self.has_queued_time) or ev.time != self.queued_time:
            return False
        self.has_queued_time = False
        if not self.has_desired_time:
            return False
        if ev.time != self.desired_time:
            self.has_queued_time = True
            self.queued_time = self.desired_time
            ev.time = self.desired_time
            queue.enqueue(ev)
            return False
        self.has_desired_time = False
        return True


class Event:
    __slots__ = ("time", "node")

    def __init__(self, time, node):
        self.time = time
        self.node = node


class SearchNode:
    __slots__ = (
        "neighbors",
        "neighbor_weights",
        "neighbor_obs",
        "reached_from_source",
        "index_of_predecessor",
        "distance_from_source",
        "tracker",
        "idx",
    )

    def __init__(self, idx):
        self.idx = idx
        self.neighbors = []  # SearchNode or None (boundary)
        self.neighbor_weights = []
        self.neighbor_obs = []  # list of list-of-fault-ids
        self.reached_from_source = None
        self.index_of_predecessor = SIZE_MAX
        self.distance_from_source = 0
        self.tracker = QueuedEventTracker()

    def index_of_neighbor(self, target):
        for i, n in enumerate(self.neighbors):
            if n is target:
                return i
        return SIZE_MAX

    def reset(self):
        self.reached_from_source = None
        self.index_of_predecessor = SIZE_MAX
        self.distance_from_source = 0
        self.tracker.clear()


def build_search_graph(H, W=2):
    """Build search nodes from H, matching pymatching neighbor ordering:
    boundary edge (weight-1 column) at index 0, then non-boundary neighbors in
    increasing column (fault-id) order. Uniform integer weight W per edge."""
    H = np.asarray(H).astype(np.uint8)
    M, N = H.shape
    nodes = [SearchNode(i) for i in range(M)]
    # collect edges per column
    boundary_edges = [[] for _ in range(M)]  # (fault) for weight-1 cols
    normal_edges = [[] for _ in range(M)]  # (other_node, fault)
    for j in range(N):
        rows = np.nonzero(H[:, j])[0]
        if len(rows) == 1:
            r = int(rows[0])
            boundary_edges[r].append(j)
        elif len(rows) == 2:
            a, b = int(rows[0]), int(rows[1])
            normal_edges[a].append((b, j))
            normal_edges[b].append((a, j))
        # len 0 or >2 ignored
    for r in range(M):
        # boundary first
        for j in boundary_edges[r]:
            nodes[r].neighbors.append(None)
            nodes[r].neighbor_weights.append(W)
            nodes[r].neighbor_obs.append([j])
        # then normal in column order
        for other, j in sorted(normal_edges[r], key=lambda x: x[1]):
            nodes[r].neighbors.append(nodes[other])
            nodes[r].neighbor_weights.append(W)
            nodes[r].neighbor_obs.append([j])
    return nodes


class SearchFlooder:
    BOUNDARY = 1
    DETECTOR_NODE = 0
    NO_TARGET = 2

    def __init__(self, nodes):
        self.nodes = nodes
        self.queue = RadixHeapQueue()
        self.reached_nodes = []
        self.target_type = self.NO_TARGET

    def find_next_event(self, node):
        best_time = INF
        best_neighbor = SIZE_MAX
        cur = self.queue.cur_time
        start = 0
        if node.neighbors and node.neighbors[0] is None:
            if self.target_type == self.BOUNDARY:
                w = node.neighbor_weights[0]
                collision_time = cur + w - (cur - node.distance_from_source)
                if collision_time < best_time:
                    best_time = collision_time
                    best_neighbor = 0
            start = 1
        for i in range(start, len(node.neighbors)):
            w = node.neighbor_weights[i]
            nb = node.neighbors[i]
            if nb.reached_from_source is node.reached_from_source:
                continue
            elif nb.reached_from_source is None:
                collision_time = cur + w - (cur - node.distance_from_source)
            else:
                covered_this = cur - node.distance_from_source
                covered_nb = cur - nb.distance_from_source
                collision_time = cur + (w - covered_this - covered_nb) // 2
            if collision_time < best_time:
                best_time = collision_time
                best_neighbor = i
        return best_neighbor, best_time

    def reschedule(self, node):
        bi, t = self.find_next_event(node)
        if bi == SIZE_MAX:
            node.tracker.set_no_desired_event()
        else:
            node.tracker.set_desired_event(Event(t, node), self.queue)

    def do_search_starting(self, src):
        src.reached_from_source = src
        src.index_of_predecessor = SIZE_MAX
        src.distance_from_source = 0
        self.reached_nodes.append(src)
        self.reschedule(src)

    def do_search_exploring_empty(self, empty_node, empty_to_from_index):
        from_node = empty_node.neighbors[empty_to_from_index]
        empty_node.reached_from_source = from_node.reached_from_source
        empty_node.index_of_predecessor = empty_to_from_index
        empty_node.distance_from_source = (
            empty_node.neighbor_weights[empty_to_from_index]
            + from_node.distance_from_source
        )
        self.reached_nodes.append(empty_node)
        self.reschedule(empty_node)

    def do_look_at_node_event(self, node):
        bi, t = self.find_next_event(node)
        if t == self.queue.cur_time:
            dst = node.neighbors[bi]
            if dst is None:
                return (node, bi)
            elif (
                node.reached_from_source is not None and dst.reached_from_source is None
            ):
                self.do_search_exploring_empty(dst, dst.index_of_neighbor(node))
                node.tracker.set_desired_event(
                    Event(self.queue.cur_time, node), self.queue
                )
                return (None, SIZE_MAX)
            else:
                return (node, bi)
        elif bi != SIZE_MAX:
            node.tracker.set_desired_event(Event(t, node), self.queue)
        return (None, SIZE_MAX)

    def run_until_collision(self, src, dst):
        if dst is None:
            self.target_type = self.BOUNDARY
        else:
            self.target_type = self.DETECTOR_NODE
            self.do_search_starting(dst)
        self.do_search_starting(src)
        while not self.queue.empty():
            ev = self.queue.dequeue()
            if ev.node.tracker.dequeue_decision(ev, self.queue):
                ce = self.do_look_at_node_event(ev.node)
                if ce[0] is not None:
                    return ce
        return (None, SIZE_MAX)

    def _trace_back_from_node(self, node, out):
        cur = node
        while cur.index_of_predecessor != SIZE_MAX:
            pred_idx = cur.index_of_predecessor
            out.append((cur, pred_idx))
            cur = cur.neighbors[pred_idx]

    def iter_edges_from_collision(self, collision_edge, out):
        node, nbr_idx = collision_edge
        self._trace_back_from_node(node, out)
        other = node.neighbors[nbr_idx]
        if other is not None:
            self._trace_back_from_node(other, out)
        out.append((node, nbr_idx))

    def reset(self):
        for n in self.reached_nodes:
            n.reset()
        self.reached_nodes.clear()
        self.queue.reset()

    def path_obs(self, src_idx, dst_idx):
        """Return the set of fault-ids on the shortest path from src to dst
        (dst_idx = SIZE_MAX/None for boundary)."""
        src = self.nodes[src_idx]
        dst = None if (dst_idx is None or dst_idx == SIZE_MAX) else self.nodes[dst_idx]
        ce = self.run_until_collision(src, dst)
        out = []
        if ce[0] is not None:
            self.iter_edges_from_collision(ce, out)
        faults = []
        for node, nbr_idx in out:
            faults.extend(node.neighbor_obs[nbr_idx])
        self.reset()
        return faults


W = 2
INF = 1 << 62

# Event kinds
LOOK_AT_NODE = 0
LOOK_AT_SHRINKING_REGION = 1
NO_FLOOD_CHECK_EVENT = 2


class FloodCheckEvent:
    __slots__ = ("time", "kind", "target")

    def __init__(self, time, kind, target):
        self.time = time
        self.kind = kind
        self.target = target


# MwpmEvent types
NO_EVENT = 0
REGION_HIT_REGION = 1
REGION_HIT_BOUNDARY = 2
BLOSSOM_SHATTER = 3


class MwpmEvent:
    __slots__ = (
        "etype",
        "region1",
        "region2",
        "edge",
        "region",
        "blossom_region",
        "in_parent_region",
        "in_child_region",
    )

    def __init__(self, etype):
        self.etype = etype

    @staticmethod
    def no_event():
        return MwpmEvent(NO_EVENT)

    @staticmethod
    def region_hit_region(r1, r2, edge):
        e = MwpmEvent(REGION_HIT_REGION)
        e.region1 = r1
        e.region2 = r2
        e.edge = edge
        return e

    @staticmethod
    def region_hit_boundary(region, edge):
        e = MwpmEvent(REGION_HIT_BOUNDARY)
        e.region = region
        e.edge = edge
        return e

    @staticmethod
    def blossom_shatter(blossom_region, in_parent, in_child):
        e = MwpmEvent(BLOSSOM_SHATTER)
        e.blossom_region = blossom_region
        e.in_parent_region = in_parent
        e.in_child_region = in_child
        return e


class Varying:
    __slots__ = ("slope", "yint")

    def __init__(self, slope, yint):
        self.slope = slope
        self.yint = yint

    def y_intercept(self):
        return self.yint

    def is_growing(self):
        return self.slope == 1

    def is_shrinking(self):
        return self.slope == -1

    def is_frozen(self):
        return self.slope == 0

    def get_distance_at_time(self, t):
        return self.slope * t + self.yint

    def time_of_x_intercept_for_shrinking(self):
        # slope=-1: -t + yint = 0 -> t = yint
        return self.yint

    def then_growing_at_time(self, t):
        d = self.get_distance_at_time(t)
        return Varying(1, d - t)

    def then_frozen_at_time(self, t):
        d = self.get_distance_at_time(t)
        return Varying(0, d)

    def then_shrinking_at_time(self, t):
        d = self.get_distance_at_time(t)
        return Varying(-1, d + t)

    def plus_const(self, c):
        return Varying(self.slope, self.yint + c)

    @staticmethod
    def growing_zero_at(t):
        return Varying(1, -t)


ZERO_FROZEN = Varying(0, 0)


class CompressedEdge:
    __slots__ = ("loc_from", "loc_to", "obs_mask")

    def __init__(self, loc_from, loc_to, obs_mask):
        self.loc_from = loc_from
        self.loc_to = loc_to
        self.obs_mask = obs_mask

    def reversed(self):
        return CompressedEdge(self.loc_to, self.loc_from, self.obs_mask)


class Match:
    __slots__ = ("region", "edge")

    def __init__(self, region=None, edge=None):
        self.region = region
        self.edge = edge

    def clear(self):
        self.region = None


class RegionEdge:
    __slots__ = ("region", "edge")

    def __init__(self, region, edge):
        self.region = region
        self.edge = edge


def unstable_erase(vec, pred):
    i = 0
    while i < len(vec):
        if pred(vec[i]):
            vec[i] = vec[-1]
            vec.pop()
        else:
            i += 1


class DetectorNode:
    __slots__ = (
        "idx",
        "neighbors",
        "neighbor_weights",
        "neighbor_observables",
        "region_that_arrived",
        "region_that_arrived_top",
        "wrapped_radius_cached",
        "reached_from_source",
        "observables_crossed_from_source",
        "radius_of_arrival",
        "node_event_tracker",
    )

    def __init__(self, idx):
        self.idx = idx
        self.neighbors = []
        self.neighbor_weights = []
        self.neighbor_observables = []
        self.region_that_arrived = None
        self.region_that_arrived_top = None
        self.wrapped_radius_cached = 0
        self.reached_from_source = None
        self.observables_crossed_from_source = 0
        self.radius_of_arrival = 0
        self.node_event_tracker = QueuedEventTracker()

    def local_radius(self):
        if self.region_that_arrived_top is None:
            return ZERO_FROZEN
        return self.region_that_arrived_top.radius.plus_const(
            self.wrapped_radius_cached
        )

    def has_same_owner_as(self, other):
        return self.region_that_arrived_top is other.region_that_arrived_top

    def index_of_neighbor(self, target):
        for k, n in enumerate(self.neighbors):
            if n is target:
                return k
        raise ValueError("Failed to find neighbor.")

    def compute_wrapped_radius(self):
        if self.reached_from_source is None:
            return 0
        total = 0
        r = self.region_that_arrived
        while r is not self.region_that_arrived_top:
            total += r.radius.y_intercept()
            r = r.blossom_parent
        return total - self.radius_of_arrival

    def heir_region_on_shatter(self):
        r = self.region_that_arrived
        while True:
            p = r.blossom_parent
            if p is self.region_that_arrived_top:
                return r
            r = p

    def reset(self):
        self.observables_crossed_from_source = 0
        self.reached_from_source = None
        self.radius_of_arrival = 0
        self.region_that_arrived = None
        self.region_that_arrived_top = None
        self.wrapped_radius_cached = 0
        self.node_event_tracker.clear()


class GraphFillRegion:
    __slots__ = (
        "blossom_parent",
        "blossom_parent_top",
        "alt_tree_node",
        "radius",
        "shrink_event_tracker",
        "match",
        "blossom_children",
        "shell_area",
    )

    def __init__(self):
        self.blossom_parent = None
        self.blossom_parent_top = self
        self.alt_tree_node = None
        self.radius = Varying(1, 0)  # (0<<2)+1 -> growing, yint 0
        self.shrink_event_tracker = QueuedEventTracker()
        self.match = Match(None, None)
        self.blossom_children = []
        self.shell_area = []

    def add_match(self, region, edge):
        self.match = Match(region, edge)
        region.match = Match(self, edge.reversed())

    def cleanup_shell_area(self):
        for n in self.shell_area:
            n.reset()

    def do_op_for_each_node_in_total_area(self, func):
        sa = self.shell_area
        for i in range(len(sa)):
            func(sa[len(sa) - i - 1])
        for child in self.blossom_children:
            child.region.do_op_for_each_node_in_total_area(func)

    def do_op_for_each_descendant_and_self(self, func):
        func(self)
        for child in self.blossom_children:
            child.region.do_op_for_each_descendant_and_self(func)

    def clear_blossom_parent(self):
        self.blossom_parent = None

        def op(descendant):
            descendant.blossom_parent_top = self
            for n in descendant.shell_area:
                n.region_that_arrived_top = self
                n.wrapped_radius_cached = n.compute_wrapped_radius()

        self.do_op_for_each_descendant_and_self(op)

    def clear_blossom_parent_ignoring_wrapped_radius(self):
        self.blossom_parent = None

        def op(descendant):
            descendant.blossom_parent_top = self
            for n in descendant.shell_area:
                n.region_that_arrived_top = self

        self.do_op_for_each_descendant_and_self(op)

    def wrap_into_blossom(self, new_top):
        self.blossom_parent = new_top

        def op(descendant):
            descendant.blossom_parent_top = new_top
            for n in descendant.shell_area:
                n.region_that_arrived_top = new_top
                n.wrapped_radius_cached = n.compute_wrapped_radius()

        self.do_op_for_each_descendant_and_self(op)


class AltTreeEdge:
    __slots__ = ("alt_tree_node", "edge")

    def __init__(self, alt_tree_node=None, edge=None):
        self.alt_tree_node = alt_tree_node
        self.edge = edge if edge is not None else CompressedEdge(None, None, 0)


class AltTreeNode:
    __slots__ = (
        "inner_region",
        "outer_region",
        "inner_to_outer_edge",
        "parent",
        "children",
        "visited",
    )

    def __init__(self, inner_region, outer_region, inner_to_outer_edge):
        self.inner_region = inner_region
        self.outer_region = outer_region
        self.inner_to_outer_edge = inner_to_outer_edge
        self.parent = AltTreeEdge()
        self.children = []
        self.visited = False
        if inner_region is not None:
            inner_region.alt_tree_node = self
        if outer_region is not None:
            outer_region.alt_tree_node = self

    @staticmethod
    def root(outer_region):
        return AltTreeNode(None, outer_region, CompressedEdge(None, None, 0))

    def add_child(self, child):
        self.children.append(child)
        child.alt_tree_node.parent = AltTreeEdge(self, child.edge.reversed())

    def become_root(self):
        if self.parent.alt_tree_node is None:
            return
        old_parent = self.parent.alt_tree_node
        old_parent.become_root()
        self.parent.alt_tree_node.inner_region = self.inner_region
        self.parent.alt_tree_node.inner_to_outer_edge = self.parent.edge
        self.inner_region = None
        unstable_erase(
            self.parent.alt_tree_node.children, lambda x: x.alt_tree_node is self
        )
        self.parent = AltTreeEdge()
        self.add_child(AltTreeEdge(old_parent, self.inner_to_outer_edge.reversed()))
        self.inner_to_outer_edge = CompressedEdge(None, None, 0)

    def most_recent_common_ancestor(self, other):
        this_current = self
        this_current.visited = True
        other_current = other
        other_current.visited = True
        common = None
        while True:
            this_parent = this_current.parent.alt_tree_node
            other_parent = other_current.parent.alt_tree_node
            if this_parent is not None or other_parent is not None:
                if this_parent is not None:
                    this_current = this_parent
                    if this_current.visited:
                        common = this_current
                        break
                    this_current.visited = True
                if other_parent is not None:
                    other_current = other_parent
                    if other_current.visited:
                        common = other_current
                        break
                    other_current.visited = True
            else:
                # clean up visited flags before returning None
                self._clear_visited_chain()
                other._clear_visited_chain()
                return None
        common.visited = False
        p = common.parent.alt_tree_node
        while p is not None and p.visited:
            p.visited = False
            p = p.parent.alt_tree_node
        return common

    def _clear_visited_chain(self):
        n = self
        while n is not None and n.visited:
            n.visited = False
            n = n.parent.alt_tree_node

    def prune_upward_path_stopping_before(self, prune_parent, back):
        orphan_edges = []
        pruned = []
        current = self
        while current is not prune_parent:
            for c in current.children:
                orphan_edges.append(c)
            current.children = []
            if back:
                pruned.append(
                    RegionEdge(current.inner_region, current.inner_to_outer_edge)
                )
                pruned.append(
                    RegionEdge(
                        current.parent.alt_tree_node.outer_region,
                        current.parent.edge.reversed(),
                    )
                )
            else:
                pruned.append(
                    RegionEdge(
                        current.outer_region, current.inner_to_outer_edge.reversed()
                    )
                )
                pruned.append(RegionEdge(current.inner_region, current.parent.edge))
            unstable_erase(
                current.parent.alt_tree_node.children,
                lambda ce, cur=current: ce.alt_tree_node is cur,
            )
            current.outer_region.alt_tree_node = None
            current.inner_region.alt_tree_node = None
            current = current.parent.alt_tree_node
        return orphan_edges, pruned


class GraphFlooder:
    def __init__(self, nodes):
        self.graph_nodes = nodes
        self.queue = RadixHeapQueue()
        self.match_edges = []

    # ---- region growth state ----
    def do_region_created_at_empty_detector_node(self, region, node):
        node.reached_from_source = node
        node.radius_of_arrival = 0
        node.region_that_arrived = region
        node.region_that_arrived_top = region
        node.wrapped_radius_cached = 0
        region.shell_area.append(node)
        self.reschedule_events_at_detector_node(node)

    def _find_next_at_growing(self, node, rad1):
        best_time = INF
        best_neighbor = SIZE_MAX
        start = 0
        if node.neighbors and node.neighbors[0] is None:
            weight = node.neighbor_weights[0]
            collision_time = weight - rad1.y_intercept()
            if collision_time < best_time:
                best_time = collision_time
                best_neighbor = 0
            start = 1
        for i in range(start, len(node.neighbors)):
            weight = node.neighbor_weights[i]
            neighbor = node.neighbors[i]
            if node.has_same_owner_as(neighbor):
                continue
            rad2 = neighbor.local_radius()
            if rad2.is_shrinking():
                continue
            collision_time = weight - rad1.y_intercept() - rad2.y_intercept()
            if rad2.is_growing():
                collision_time >>= 1
            if collision_time < best_time:
                best_time = collision_time
                best_neighbor = i
        return best_neighbor, best_time

    def _find_next_at_not_growing(self, node, rad1):
        best_time = INF
        best_neighbor = SIZE_MAX
        start = 0
        if node.neighbors and node.neighbors[0] is None:
            start = 1
        for i in range(start, len(node.neighbors)):
            weight = node.neighbor_weights[i]
            neighbor = node.neighbors[i]
            rad2 = neighbor.local_radius()
            if rad2.is_growing():
                collision_time = weight - rad1.y_intercept() - rad2.y_intercept()
                if collision_time < best_time:
                    best_time = collision_time
                    best_neighbor = i
        return best_neighbor, best_time

    def find_next_event_at_node(self, node):
        rad1 = node.local_radius()
        if rad1.is_growing():
            return self._find_next_at_growing(node, rad1)
        else:
            return self._find_next_at_not_growing(node, rad1)

    def reschedule_events_at_detector_node(self, node):
        bi, t = self.find_next_event_at_node(node)
        if bi == SIZE_MAX:
            node.node_event_tracker.set_no_desired_event()
        else:
            node.node_event_tracker.set_desired_event(
                FloodCheckEvent(t, LOOK_AT_NODE, node), self.queue
            )

    def schedule_tentative_shrink_event(self, region):
        if not region.shell_area:
            t = region.radius.time_of_x_intercept_for_shrinking()
        else:
            t = region.shell_area[-1].local_radius().time_of_x_intercept_for_shrinking()
        region.shrink_event_tracker.set_desired_event(
            FloodCheckEvent(t, LOOK_AT_SHRINKING_REGION, region), self.queue
        )

    def do_region_arriving_at_empty_detector_node(
        self, region, empty_node, from_node, from_to_empty_index
    ):
        empty_node.observables_crossed_from_source = (
            from_node.observables_crossed_from_source
            ^ from_node.neighbor_observables[from_to_empty_index]
        )
        empty_node.reached_from_source = from_node.reached_from_source
        empty_node.radius_of_arrival = region.radius.get_distance_at_time(
            self.queue.cur_time
        )
        empty_node.region_that_arrived = region
        empty_node.region_that_arrived_top = region.blossom_parent_top
        empty_node.wrapped_radius_cached = empty_node.compute_wrapped_radius()
        region.shell_area.append(empty_node)
        self.reschedule_events_at_detector_node(empty_node)

    def do_region_shrinking(self, region):
        if not region.shell_area:
            return self.do_blossom_shattering(region)
        elif len(region.shell_area) == 1 and not region.blossom_children:
            return self.do_degenerate_implosion(region)
        else:
            leaving = region.shell_area.pop()
            leaving.region_that_arrived = None
            leaving.region_that_arrived_top = None
            leaving.wrapped_radius_cached = 0
            leaving.reached_from_source = None
            leaving.radius_of_arrival = 0
            leaving.observables_crossed_from_source = 0
            self.reschedule_events_at_detector_node(leaving)
            self.schedule_tentative_shrink_event(region)
            return MwpmEvent.no_event()

    def do_neighbor_interaction(self, src, src_to_dst_index, dst):
        if src.region_that_arrived is not None and dst.region_that_arrived is None:
            self.do_region_arriving_at_empty_detector_node(
                src.region_that_arrived_top, dst, src, src_to_dst_index
            )
            return MwpmEvent.no_event()
        elif dst.region_that_arrived is not None and src.region_that_arrived is None:
            self.do_region_arriving_at_empty_detector_node(
                dst.region_that_arrived_top, src, dst, dst.index_of_neighbor(src)
            )
            return MwpmEvent.no_event()
        else:
            return MwpmEvent.region_hit_region(
                src.region_that_arrived_top,
                dst.region_that_arrived_top,
                CompressedEdge(
                    src.reached_from_source,
                    dst.reached_from_source,
                    src.observables_crossed_from_source
                    ^ dst.observables_crossed_from_source
                    ^ src.neighbor_observables[src_to_dst_index],
                ),
            )

    def do_region_hit_boundary_interaction(self, node):
        return MwpmEvent.region_hit_boundary(
            node.region_that_arrived_top,
            CompressedEdge(
                node.reached_from_source,
                None,
                node.observables_crossed_from_source ^ node.neighbor_observables[0],
            ),
        )

    def do_degenerate_implosion(self, region):
        atn = region.alt_tree_node
        return MwpmEvent.region_hit_region(
            atn.parent.alt_tree_node.outer_region,
            atn.outer_region,
            CompressedEdge(
                atn.parent.edge.loc_to,
                atn.inner_to_outer_edge.loc_to,
                atn.inner_to_outer_edge.obs_mask ^ atn.parent.edge.obs_mask,
            ),
        )

    def do_blossom_shattering(self, region):
        atn = region.alt_tree_node
        return MwpmEvent.blossom_shatter(
            region,
            atn.parent.edge.loc_from.heir_region_on_shatter(),
            atn.inner_to_outer_edge.loc_from.heir_region_on_shatter(),
        )

    def create_blossom(self, contained_regions):
        blossom = GraphFillRegion()
        blossom.radius = Varying.growing_zero_at(self.queue.cur_time)
        blossom.blossom_children = contained_regions
        for re in blossom.blossom_children:
            re.region.radius = re.region.radius.then_frozen_at_time(self.queue.cur_time)
            re.region.wrap_into_blossom(blossom)
            re.region.shrink_event_tracker.set_no_desired_event()
        blossom.do_op_for_each_node_in_total_area(
            lambda n: self.reschedule_events_at_detector_node(n)
        )
        return blossom

    def dequeue_decision(self, ev):
        if ev.kind == LOOK_AT_NODE:
            return ev.target.node_event_tracker.dequeue_decision(ev, self.queue)
        elif ev.kind == LOOK_AT_SHRINKING_REGION:
            return ev.target.shrink_event_tracker.dequeue_decision(ev, self.queue)
        elif ev.kind == NO_FLOOD_CHECK_EVENT:
            return True
        raise ValueError("Unrecognized event type")

    def dequeue_valid(self):
        while True:
            ev = self.queue.dequeue()
            if ev is None:
                return FloodCheckEvent(0, NO_FLOOD_CHECK_EVENT, None)
            if self.dequeue_decision(ev):
                return ev

    def set_region_growing(self, region):
        region.radius = region.radius.then_growing_at_time(self.queue.cur_time)
        region.shrink_event_tracker.set_no_desired_event()
        region.do_op_for_each_node_in_total_area(
            lambda n: self.reschedule_events_at_detector_node(n)
        )

    def set_region_frozen(self, region):
        was_shrinking = region.radius.is_shrinking()
        region.radius = region.radius.then_frozen_at_time(self.queue.cur_time)
        region.shrink_event_tracker.set_no_desired_event()
        if was_shrinking:
            region.do_op_for_each_node_in_total_area(
                lambda n: self.reschedule_events_at_detector_node(n)
            )

    def set_region_shrinking(self, region):
        region.radius = region.radius.then_shrinking_at_time(self.queue.cur_time)
        self.schedule_tentative_shrink_event(region)
        region.do_op_for_each_node_in_total_area(
            lambda n: n.node_event_tracker.set_no_desired_event()
        )

    def do_look_at_node_event(self, node):
        bi, t = self.find_next_event_at_node(node)
        if t == self.queue.cur_time:
            node.node_event_tracker.set_desired_event(
                FloodCheckEvent(self.queue.cur_time, LOOK_AT_NODE, node), self.queue
            )
            if node.neighbors[bi] is None:
                return self.do_region_hit_boundary_interaction(node)
            neighbor = node.neighbors[bi]
            return self.do_neighbor_interaction(node, bi, neighbor)
        elif bi != SIZE_MAX:
            node.node_event_tracker.set_desired_event(
                FloodCheckEvent(t, LOOK_AT_NODE, node), self.queue
            )
        return MwpmEvent.no_event()

    def process_tentative_event(self, ev):
        if ev.kind == LOOK_AT_NODE:
            return self.do_look_at_node_event(ev.target)
        elif ev.kind == LOOK_AT_SHRINKING_REGION:
            return self.do_region_shrinking(ev.target)
        raise ValueError("Unknown tentative event type")

    def run_until_next_mwpm_notification(self):
        while True:
            ev = self.dequeue_valid()
            if ev.kind == NO_FLOOD_CHECK_EVENT:
                return MwpmEvent.no_event()
            notification = self.process_tentative_event(ev)
            if notification.etype != NO_EVENT:
                return notification


class MatchingResult:
    __slots__ = ("obs_mask", "weight")

    def __init__(self, obs_mask=0, weight=0):
        self.obs_mask = obs_mask
        self.weight = weight

    def iadd(self, rhs):
        self.obs_mask ^= rhs.obs_mask
        self.weight += rhs.weight
        return self


class Mwpm:
    def __init__(self, flooder, search_flooder=None):
        self.flooder = flooder
        self.search_flooder = search_flooder
        self.live_alt_nodes = 0

    def create_detection_event(self, node):
        region = GraphFillRegion()
        alt = AltTreeNode.root(region)
        self.live_alt_nodes += 1
        region.alt_tree_node = alt
        self.flooder.do_region_created_at_empty_detector_node(region, node)

    def _del_alt(self, node):
        self.live_alt_nodes -= 1

    def shatter_descendants_into_matches_and_freeze(self, alt_tree_node):
        for child_edge in alt_tree_node.children:
            self.shatter_descendants_into_matches_and_freeze(child_edge.alt_tree_node)
        if alt_tree_node.inner_region is not None:
            alt_tree_node.parent = AltTreeEdge()
            alt_tree_node.inner_region.add_match(
                alt_tree_node.outer_region, alt_tree_node.inner_to_outer_edge
            )
            self.flooder.set_region_frozen(alt_tree_node.inner_region)
            self.flooder.set_region_frozen(alt_tree_node.outer_region)
            alt_tree_node.inner_region.alt_tree_node = None
            alt_tree_node.outer_region.alt_tree_node = None
        if alt_tree_node.outer_region is not None:
            alt_tree_node.outer_region.alt_tree_node = None
        self._del_alt(alt_tree_node)

    def handle_tree_hitting_boundary(self, event):
        node = event.region.alt_tree_node
        node.become_root()
        self.shatter_descendants_into_matches_and_freeze(node)
        event.region.match = Match(None, event.edge)
        self.flooder.set_region_frozen(event.region)

    def handle_tree_hitting_boundary_match(
        self, unmatched_region, matched_region, edge
    ):
        alt = unmatched_region.alt_tree_node
        unmatched_region.add_match(matched_region, edge)
        self.flooder.set_region_frozen(unmatched_region)
        alt.become_root()
        self.shatter_descendants_into_matches_and_freeze(alt)

    def handle_tree_hitting_other_tree(self, event):
        alt1 = event.region1.alt_tree_node
        alt2 = event.region2.alt_tree_node
        event.region1.alt_tree_node.become_root()
        event.region2.alt_tree_node.become_root()
        self.shatter_descendants_into_matches_and_freeze(alt1)
        self.shatter_descendants_into_matches_and_freeze(alt2)
        event.region1.add_match(event.region2, event.edge)
        self.flooder.set_region_frozen(event.region1)
        self.flooder.set_region_frozen(event.region2)

    def make_child(
        self,
        parent,
        child_inner_region,
        child_outer_region,
        child_inner_to_outer_edge,
        child_compressed_edge,
    ):
        child = AltTreeNode(
            child_inner_region, child_outer_region, child_inner_to_outer_edge
        )
        self.live_alt_nodes += 1
        parent.add_child(AltTreeEdge(child, child_compressed_edge))
        return child

    def handle_tree_hitting_match(self, unmatched_region, matched_region, edge):
        alt = unmatched_region.alt_tree_node
        self.make_child(
            alt,
            matched_region,
            matched_region.match.region,
            matched_region.match.edge,
            edge,
        )
        other_match = matched_region.match.region
        other_match.match.clear()
        matched_region.match.clear()
        self.flooder.set_region_shrinking(matched_region)
        self.flooder.set_region_growing(other_match)

    def handle_tree_hitting_self(self, event, common_ancestor):
        alt1 = event.region1.alt_tree_node
        alt2 = event.region2.alt_tree_node
        orphan1, pruned1 = alt1.prune_upward_path_stopping_before(common_ancestor, True)
        orphan2, pruned2 = alt2.prune_upward_path_stopping_before(
            common_ancestor, False
        )
        blossom_cycle = list(pruned2)
        p1s = len(pruned1)
        for i in range(p1s):
            blossom_cycle.append(pruned1[p1s - i - 1])
        blossom_cycle.append(RegionEdge(event.region1, event.edge))
        common_ancestor.outer_region.alt_tree_node = None
        blossom_region = self.flooder.create_blossom(blossom_cycle)
        common_ancestor.outer_region = blossom_region
        blossom_region.alt_tree_node = common_ancestor
        for c in orphan1:
            common_ancestor.add_child(c)
        for c in orphan2:
            common_ancestor.add_child(c)

    def handle_blossom_shattering(self, event):
        for child in event.blossom_region.blossom_children:
            child.region.clear_blossom_parent()
        blossom_cycle = event.blossom_region.blossom_children
        blossom_alt_node = event.blossom_region.alt_tree_node
        bsize = len(blossom_cycle)
        parent_idx = 0
        child_idx = 0
        for i in range(bsize):
            if blossom_cycle[i].region is event.in_parent_region:
                parent_idx = i
            if blossom_cycle[i].region is event.in_child_region:
                child_idx = i
        gap = (child_idx + bsize - parent_idx) % bsize
        current_alt_node = event.blossom_region.alt_tree_node.parent.alt_tree_node
        unstable_erase(
            current_alt_node.children, lambda x: x.alt_tree_node is blossom_alt_node
        )
        child_edge = blossom_alt_node.parent.edge.reversed()
        if gap % 2 == 0:
            evens_start = child_idx + 1
            evens_end = child_idx + bsize - gap
            for i in range(parent_idx, parent_idx + gap, 2):
                current_alt_node = self.make_child(
                    current_alt_node,
                    blossom_cycle[i % bsize].region,
                    blossom_cycle[(i + 1) % bsize].region,
                    blossom_cycle[i % bsize].edge,
                    child_edge,
                )
                child_edge = blossom_cycle[(i + 1) % bsize].edge
                self.flooder.set_region_shrinking(current_alt_node.inner_region)
                self.flooder.set_region_growing(current_alt_node.outer_region)
        else:
            evens_start = parent_idx + 1
            evens_end = parent_idx + gap
            for i in range(0, bsize - gap, 2):
                k1 = (parent_idx + bsize - i) % bsize
                k2 = (parent_idx + bsize - i - 1) % bsize
                k3 = (parent_idx + bsize - i - 2) % bsize
                current_alt_node = self.make_child(
                    current_alt_node,
                    blossom_cycle[k1].region,
                    blossom_cycle[k2].region,
                    blossom_cycle[k2].edge.reversed(),
                    child_edge,
                )
                child_edge = blossom_cycle[k3].edge.reversed()
                self.flooder.set_region_shrinking(current_alt_node.inner_region)
                self.flooder.set_region_growing(current_alt_node.outer_region)
        for j in range(evens_start, evens_end, 2):
            k1 = j % bsize
            k2 = (j + 1) % bsize
            blossom_cycle[k1].region.add_match(
                blossom_cycle[k2].region, blossom_cycle[k1].edge
            )
            blossom_cycle[k1].region.do_op_for_each_node_in_total_area(
                lambda n: self.flooder.reschedule_events_at_detector_node(n)
            )
            blossom_cycle[k2].region.do_op_for_each_node_in_total_area(
                lambda n: self.flooder.reschedule_events_at_detector_node(n)
            )
        blossom_alt_node.inner_region = blossom_cycle[child_idx].region
        self.flooder.set_region_shrinking(blossom_alt_node.inner_region)
        blossom_cycle[child_idx].region.alt_tree_node = blossom_alt_node
        current_alt_node.add_child(AltTreeEdge(blossom_alt_node, child_edge))
        self._del_alt(event.blossom_region)  # blossom region freed

    def handle_region_hit_region(self, event):
        alt1 = event.region1.alt_tree_node
        alt2 = event.region2.alt_tree_node
        if alt1 is not None and alt2 is not None:
            common = alt1.most_recent_common_ancestor(alt2)
            if common is None:
                self.handle_tree_hitting_other_tree(event)
            else:
                self.handle_tree_hitting_self(event, common)
        elif alt1 is not None:
            if event.region2.match.region is not None:
                self.handle_tree_hitting_match(event.region1, event.region2, event.edge)
            else:
                self.handle_tree_hitting_boundary_match(
                    event.region1, event.region2, event.edge
                )
        else:
            if event.region1.match.region is not None:
                self.handle_tree_hitting_match(
                    event.region2, event.region1, event.edge.reversed()
                )
            else:
                self.handle_tree_hitting_boundary_match(
                    event.region2, event.region1, event.edge.reversed()
                )

    def process_event(self, event):
        if event.etype == REGION_HIT_REGION:
            self.handle_region_hit_region(event)
        elif event.etype == REGION_HIT_BOUNDARY:
            self.handle_tree_hitting_boundary(event)
        elif event.etype == BLOSSOM_SHATTER:
            self.handle_blossom_shattering(event)
        elif event.etype == NO_EVENT:
            pass
        else:
            raise ValueError("Unrecognized event type")

    # ---- extraction (obs_mask path, num_observables <= 64) ----
    def pair_and_shatter_extract_matches(self, region, res):
        for r in region.blossom_children:
            r.region.clear_blossom_parent_ignoring_wrapped_radius()
        subblossom = region.match.edge.loc_from.region_that_arrived_top
        subblossom.match = region.match
        if subblossom.match.region is not None:
            subblossom.match.region.match.region = subblossom
        res.weight += region.radius.y_intercept()
        index = 0
        for i, e in enumerate(region.blossom_children):
            if e.region is subblossom:
                index = i
                break
        num_children = len(region.blossom_children)
        for i in range(0, num_children - 1, 2):
            re1 = region.blossom_children[(index + i + 1) % num_children]
            re2 = region.blossom_children[(index + i + 2) % num_children]
            re1.region.add_match(re2.region, re1.edge)
            res.iadd(self.shatter_blossom_and_extract_matches(re1.region))
        return subblossom

    def shatter_blossom_and_extract_matches(self, region):
        region.cleanup_shell_area()
        if region.match.region is not None:
            region.match.region.cleanup_shell_area()
            if not region.blossom_children and not region.match.region.blossom_children:
                res = MatchingResult(
                    region.match.edge.obs_mask,
                    region.radius.y_intercept()
                    + region.match.region.radius.y_intercept(),
                )
                return res
        elif not region.blossom_children:
            return MatchingResult(
                region.match.edge.obs_mask, region.radius.y_intercept()
            )
        res = MatchingResult(0, 0)
        if region.blossom_children:
            region = self.pair_and_shatter_extract_matches(region, res)
        if region.match.region is not None and region.match.region.blossom_children:
            self.pair_and_shatter_extract_matches(region.match.region, res)
        res.iadd(self.shatter_blossom_and_extract_matches(region))
        return res

    # ---- extraction (match-edges path, num_observables > 64) ----
    def pair_and_shatter_extract_match_edges(self, region, match_edges):
        for r in region.blossom_children:
            r.region.clear_blossom_parent_ignoring_wrapped_radius()
        subblossom = region.match.edge.loc_from.region_that_arrived_top
        subblossom.match = region.match
        if subblossom.match.region is not None:
            subblossom.match.region.match.region = subblossom
        index = 0
        for i, e in enumerate(region.blossom_children):
            if e.region is subblossom:
                index = i
                break
        num_children = len(region.blossom_children)
        for i in range(0, num_children - 1, 2):
            re1 = region.blossom_children[(index + i + 1) % num_children]
            re2 = region.blossom_children[(index + i + 2) % num_children]
            re1.region.add_match(re2.region, re1.edge)
            self.shatter_blossom_and_extract_match_edges(re1.region, match_edges)
        return subblossom

    def shatter_blossom_and_extract_match_edges(self, region, match_edges):
        region.cleanup_shell_area()
        if region.match.region is not None:
            region.match.region.cleanup_shell_area()
            if not region.blossom_children and not region.match.region.blossom_children:
                match_edges.append(region.match.edge)
                return
        elif not region.blossom_children:
            match_edges.append(region.match.edge)
            return
        if region.blossom_children:
            region = self.pair_and_shatter_extract_match_edges(region, match_edges)
        if region.match.region is not None and region.match.region.blossom_children:
            self.pair_and_shatter_extract_match_edges(region.match.region, match_edges)
        self.shatter_blossom_and_extract_match_edges(region, match_edges)


def build_main_graph(H, W=2):
    H = np.asarray(H).astype(np.uint8)
    M, N = H.shape
    nodes = [DetectorNode(i) for i in range(M)]
    boundary_edges = [[] for _ in range(M)]
    normal_edges = [[] for _ in range(M)]
    for j in range(N):
        rows = np.nonzero(H[:, j])[0]
        if len(rows) == 1:
            boundary_edges[int(rows[0])].append(j)
        elif len(rows) == 2:
            a, b = int(rows[0]), int(rows[1])
            normal_edges[a].append((b, j))
            normal_edges[b].append((a, j))
    for r in range(M):
        for j in boundary_edges[r]:
            nodes[r].neighbors.append(None)
            nodes[r].neighbor_weights.append(W)
            nodes[r].neighbor_observables.append(1 << j)
        for other, j in sorted(normal_edges[r], key=lambda x: x[1]):
            nodes[r].neighbors.append(nodes[other])
            nodes[r].neighbor_weights.append(W)
            nodes[r].neighbor_observables.append(1 << j)
    return nodes


# Public entry point: NativeMatcher (bit-exact PyMatching from_check_matrix(H) decode)
def _parse_H(H):
    """(M, N, boundary_edges, normal_edges) from a dense parity-check matrix."""
    H = np.asarray(H).astype(np.uint8)
    M, N = H.shape
    boundary_edges = [[] for _ in range(M)]
    normal_edges = [[] for _ in range(M)]
    for j in range(N):
        rows = np.nonzero(H[:, j])[0]
        if len(rows) == 1:
            boundary_edges[int(rows[0])].append(j)
        elif len(rows) == 2:
            a, b = int(rows[0]), int(rows[1])
            normal_edges[a].append((b, j))
            normal_edges[b].append((a, j))
    return M, N, boundary_edges, normal_edges


def _parse_graph(g):
    """(M, N, boundary_edges, normal_edges) from a MatchingGraph (boundary node
    == g.boundary). Preserves the edge/column order used by build_matching_graph."""
    M = int(g.boundary)
    N = int(g.num_qubits)
    boundary = int(g.boundary)
    boundary_edges = [[] for _ in range(M)]
    normal_edges = [[] for _ in range(M)]
    eu, ev, eq = g.edge_u, g.edge_v, g.edge_qubit
    for e in range(int(eu.size)):
        u, v, q = int(eu[e]), int(ev[e]), int(eq[e])
        if v == boundary:
            boundary_edges[u].append(q)
        elif u == boundary:
            boundary_edges[v].append(q)
        else:
            normal_edges[u].append((v, q))
            normal_edges[v].append((u, q))
    return M, N, boundary_edges, normal_edges


def _build_nodes(node_cls, M, boundary_edges, normal_edges, W_):
    nodes = [node_cls(i) for i in range(M)]
    for r in range(M):
        for j in boundary_edges[r]:
            nodes[r].neighbors.append(None)
            nodes[r].neighbor_weights.append(W_)
            if node_cls is DetectorNode:
                nodes[r].neighbor_observables.append(1 << j)
            else:
                nodes[r].neighbor_obs.append([j])
        for other, j in sorted(normal_edges[r], key=lambda x: x[1]):
            nodes[r].neighbors.append(nodes[other])
            nodes[r].neighbor_weights.append(W_)
            if node_cls is DetectorNode:
                nodes[r].neighbor_observables.append(1 << j)
            else:
                nodes[r].neighbor_obs.append([j])
    return nodes


class NativeMatcher:
    """Bit-exact native reimplementation of PyMatching's ``Matching.from_check_matrix(H)``.

    Build once from a check matrix (``from_check_matrix``) or a MatchingGraph
    (``from_graph``); call ``decode`` per syndrome. The search graph is built once and reused;
    the (stateful) flood graph is rebuilt per shot.
    """

    def __init__(self, M, N, boundary_edges, normal_edges):
        self.M = M
        self.N = N
        self.num_observables = N
        self._boundary_edges = boundary_edges
        self._normal_edges = normal_edges
        self._search_nodes = _build_nodes(
            SearchNode, M, boundary_edges, normal_edges, W
        )
        self._search_flooder = SearchFlooder(self._search_nodes)

    @classmethod
    def from_check_matrix(cls, H):
        return cls(*_parse_H(H))

    @classmethod
    def from_graph(cls, g):
        return cls(*_parse_graph(g))

    def seed_regions(self, detection_events):
        """Phase 1 -- build a fresh flood graph and grow a region at each defect."""
        nodes = _build_nodes(
            DetectorNode, self.M, self._boundary_edges, self._normal_edges, W
        )
        flooder = GraphFlooder(nodes)
        mwpm = Mwpm(flooder)
        flooder.queue.cur_time = 0
        for det in detection_events:
            mwpm.create_detection_event(nodes[det])
        return nodes, mwpm

    def run_flood(self, nodes, mwpm):
        """Phase 2 -- pump collision events (blossoms form/shatter) until none remain."""
        flooder = mwpm.flooder
        while True:
            event = flooder.run_until_next_mwpm_notification()
            if event.etype == NO_EVENT:
                break
            mwpm.process_event(event)

    def extract_correction(self, nodes, mwpm, detection_events):
        """Phase 3 -- read matched edges out into a correction over the N qubits.

        Returns ``(correction [N] uint8, weight float)``. Handles both readout
        branches: the obs_mask fast path (num_observables <= 64) and the
        per-edge path (> 64)."""
        corr = np.zeros(self.N, dtype=np.uint8)
        if self.num_observables <= 64:
            res = MatchingResult(0, 0)
            for det in detection_events:
                node = nodes[det]
                if node.region_that_arrived is not None:
                    res.iadd(
                        mwpm.shatter_blossom_and_extract_matches(
                            node.region_that_arrived_top
                        )
                    )
            mask = res.obs_mask
            # Vectorized bit-expansion of the N-bit obs_mask into corr (bit j is
            # qubit j; mask fits one uint64 on this branch).
            bits = np.arange(self.N, dtype=np.uint64)
            corr[:] = ((np.uint64(mask) >> bits) & np.uint64(1)).astype(np.uint8)
            weight = res.weight // W
        else:
            match_edges = []
            for det in detection_events:
                node = nodes[det]
                if node.region_that_arrived is not None:
                    mwpm.shatter_blossom_and_extract_match_edges(
                        node.region_that_arrived_top, match_edges
                    )
            weight = 0
            for me in match_edges:
                src = me.loc_from.idx
                dst = None if me.loc_to is None else me.loc_to.idx
                faults = self._search_flooder.path_obs(src, dst)
                weight += len(faults)
                for f in faults:
                    corr[f] ^= 1
        return corr, float(weight)

    def decode(self, syndrome):
        """Return the correction bit-vector (uint8 [N]), identical to PyMatching's decode."""
        corr, _w = self._decode_impl(syndrome)
        return corr

    def decode_with_weight(self, syndrome):
        """Return (correction [N] uint8, weight float) matching pymatching decode(return_weight=True)."""
        return self._decode_impl(syndrome)

    def _decode_impl(self, syndrome):
        s = np.asarray(syndrome).astype(np.uint8)
        detection_events = np.nonzero(s)[0].tolist()
        if not detection_events:
            return np.zeros(self.N, dtype=np.uint8), 0.0
        nodes, mwpm = self.seed_regions(detection_events)
        self.run_flood(nodes, mwpm)
        return self.extract_correction(nodes, mwpm, detection_events)


# Framework-facing decode helpers (bit-exact to PyMatching).
def decode_single_full(g, syndrome):
    """Decode one syndrome (``[M]`` uint8) natively, bit-identically to PyMatching.

    A ``NativeMatcher`` is built once per graph and cached on it, so repeated calls with
    the same ``g`` reuse the search graph.

    Returns ``(correction, weight, rounds)``:
      * correction -- ``[N]`` uint8, ``H @ correction % 2 == syndrome`` (== PyMatching's);
      * weight     -- float, the minimum matching weight (== PyMatching's optimal);
      * rounds     -- informational fired-check (defect) count.
    """
    syndrome = np.asarray(syndrome).astype(np.uint8)
    matcher = getattr(g, "_native_matcher", None)
    if matcher is None:
        matcher = NativeMatcher.from_graph(g)
        g._native_matcher = matcher
    correction, weight = matcher.decode_with_weight(syndrome)
    rounds = int(syndrome[: g.boundary].sum())
    return correction, weight, rounds


def decode_single(g, syndrome, return_rounds=False):
    """Framework-facing single-syndrome decode (correction over N qubits)."""
    correction, _weight, rounds = decode_single_full(g, syndrome)
    if return_rounds:
        return correction, rounds
    return correction


_MP_MATCHER = None  # per-worker NativeMatcher, set by _mp_worker_init


def _mp_worker_init(H_np):
    global _MP_MATCHER
    _MP_MATCHER = NativeMatcher.from_check_matrix(np.ascontiguousarray(H_np))


def _mp_worker_decode(syndrome_row):
    return _MP_MATCHER.decode(syndrome_row)


class create(torch.nn.Module):
    """Native exact-weight MWPM decoder consuming one of the bundle's Hx/Hz matrices."""

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super().__init__()
        logger.info("Creating MWPM decoder.")

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
                "mwpm requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(
            self.check_type
        )

        # Build the detector graph once from the dense H (graphlike: col weight <= 2).
        H_np = np.asarray(self.H_matrix.detach().cpu().numpy()).astype(np.uint8)
        self.graph = build_matching_graph(H_np)
        # The bit-exact native sparse-blossom matcher (built once, reused per shot).
        self.matcher = NativeMatcher.from_check_matrix(H_np)

        self._H_np = H_np
        self._num_workers = int(decoder_cfg.get("num_workers", os.cpu_count() or 1))
        self._mp_min_batch = int(decoder_cfg.get("mp_min_batch", 64))
        self._pool = None

        self.algo = "mwpm"
        self.batch_size = 1
        # MWPM is non-iterative; report a nominal cap of the edge count.
        self.num_max_iter = self.graph.edge_u.size + 1

        # MWPM is not iterative in the BP sense; the adaptive cap does not apply.
        self.cap = None
        self.cap_bypass = False
        self.cap_active_last = False
        logger.info("Complete.")

    def _get_pool(self, B):
        """Lazily build (and reuse) the shot-parallel process pool, or ``None``
        for the sequential path (single worker, or a batch too small to amortise
        the pool overhead)."""
        if self._num_workers <= 1 or B < self._mp_min_batch:
            return None
        if self._pool is None:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            self._pool = ProcessPoolExecutor(
                max_workers=self._num_workers,
                mp_context=ctx,
                initializer=_mp_worker_init,
                initargs=(self._H_np,),
            )
        return self._pool

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def forward(self, io_dict):
        logger.info("Initializing MWPM decoding.")
        synd = io_dict["synd"]
        dev, dt = self.device, self.dtype
        B, M = synd.shape
        self.batch_size = B
        N = self.H_shape[1]

        synd_np = (synd.detach().cpu().numpy() != 0).astype(np.uint8)  # [B, M]

        pool = self._get_pool(B)
        if pool is not None:
            chunksize = max(1, B // (self._num_workers * 4))
            corrections = list(
                pool.map(_mp_worker_decode, synd_np, chunksize=chunksize)
            )
        else:
            matcher = self.matcher
            corrections = []
            for s in synd_np:
                detection_events = np.nonzero(s)[0].tolist()
                nodes, mwpm = matcher.seed_regions(detection_events)
                matcher.run_flood(nodes, mwpm)
                corr, _weight = matcher.extract_correction(
                    nodes, mwpm, detection_events
                )
                corrections.append(corr)
        e_v = np.asarray(corrections, dtype=np.uint8).reshape(B, N)
        # Per-shot defect count is a pure row-sum -- vectorized out of the shot loop.
        iters = synd_np.sum(1).astype(np.int64)

        e_v_t = torch.from_numpy(e_v).to(device=dev, dtype=dt)
        # Sign-encoded soft output: +1 where bit=0, -1 where bit=1 (a downstream
        # hard-decision `<=0 -> 1` stays consistent; MWPM carries no real LLR).
        llr = (1.0 - 2.0 * e_v_t).to(device=dev, dtype=dt)
        converge = torch.ones(B, dtype=torch.int64, device=dev)
        # Report at least one iteration: MetricState.report_metric histograms (iter - 1),
        # which must stay non-negative (a trivial all-zero syndrome matches 0 defects).
        iters_t = torch.from_numpy(iters).clamp(min=1).to(device=dev)

        logger.info("Complete.")
        io_dict.update(
            {"e_v": e_v_t, "iter": iters_t, "llr": llr, "converge": converge}
        )
        return io_dict
