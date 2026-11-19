"""Two-layer qubit coloring for the BB split of SymBreak, arXiv:2412.02885 Sec. VI.

bb_layers(hx_col, hz_col, n) returns a bool tensor of shape (n,). True is the
"yellow" (upper) layer and False is the "blue" (lower) layer. The inputs are
padded column-index tensors, one row per check, with pad value n.

Rules:
(a) Each X check and each Z check has exactly half of its qubits in each layer.
(b) In each layer, the Tanner graph has no 4-cycle. The graph has the X checks,
    the Z checks and the qubits of that layer. Two checks of one type must not
    share two qubits of one color. An X check and a Z check must not share two
    qubits of one color.
(c) For each X check and Z check that share qubits, each layer holds an odd
    number of the shared qubits. Then each child of a color split anticommutes
    with that opposite check, so the split breaks the degeneracy (Sec. IV-B).

Construction. The function does not need the BB polynomials. It derives the
layers from the overlap structure. For an X check and a Z check that share two
qubits, (b) and (c) both require that the two qubits have different colors.
Union-find with parity applies these "differ" constraints. Two checks of one
type that share two qubits add the same constraint from (b). A contradiction
means that no coloring exists. The solver then picks one flip per connected
component so that (a) holds, with exact backtracking. The yellow layer of each
component is the side that holds its smallest qubit index. The result is
deterministic and uses no random numbers.

The function raises ValueError when no coloring satisfies (a), (b) and (c).
Some cases fail at once: a check with an odd degree; two same-type checks
that share three or more qubits; an X/Z pair that shares an odd number of
qubits, or four or more qubits. With four or more, an odd split puts at least
three shared qubits in one layer, so (b) fails.

Which rules hold, and how they relate:
* (a), (b) and (c) hold on each returned coloring.
* (c) does not follow from (a) and (b). In BB [[144,12,12]], other 3/3
  colorings exist that satisfy same-type (b) and break (c). One example puts
  the even-index A qubits and the odd-index B qubits in one layer. So the
  function imposes (c).
* In BB [[144,12,12]], two checks of one type share at most one qubit. Thus
  (b) for same-type check pairs holds for every coloring. The X/Z part of (b)
  is the same as (c), because each X/Z pair shares exactly two qubits.
* In BB [[144,12,12]], the differ constraints connect all qubits. Thus the
  coloring is unique up to a color swap. It is the column block split of
  Hx = [A|B]: the A block (qubits 0 to 71) is yellow and the B block is blue.
  Each X check then has its three A qubits in one layer and its three B qubits
  in the other. Each layer is a 3-regular bipartite graph that has 6-cycles and
  no 4-cycles.
* This block split is the yellow/blue grouping of the paper's Fig. 7. Sec. VI-A
  says that its grouping differs from the "layers, A and B" of the original BB
  paper (Fig. 3). Those layers are edge classes (A edges and B edges of each
  check), not a qubit coloring, so they are not this split.

The function is not a BB detector. A code that satisfies the rules returns a
coloring. The rotated surface code is one example: a checkerboard coloring
satisfies (a), (b) and (c).
"""

import torch


def _rows(col, n):
    return [sorted(set(row[row != n].tolist())) for row in col.detach().cpu()]


def bb_layers(hx_col, hz_col, n):
    """Return the yellow-layer mask (n,) for the BB color split; see module docstring."""
    n = int(n)
    rows = {"X": _rows(hx_col, n), "Z": _rows(hz_col, n)}
    for kind, checks in rows.items():
        for c, qubits in enumerate(checks):
            if len(qubits) % 2:
                raise ValueError(
                    f"no BB layering: {kind} check {c} has odd degree "
                    f"{len(qubits)}, so rule (a) fails"
                )

    parent, parity = list(range(n)), [0] * n

    def find(q):
        path = []
        while parent[q] != q:
            path.append(q)
            q = parent[q]
        flip = 0
        for p in reversed(path):  # path compression, parities relative to root q
            flip ^= parity[p]
            parent[p], parity[p] = q, flip
        return q

    def differ(u, v, why):
        ru, rv = find(u), find(v)
        pu, pv = parity[u] if u != ru else 0, parity[v] if v != rv else 0
        if ru == rv:
            if pu == pv:
                raise ValueError(
                    f"no BB layering: qubits {u} and {v} must differ ({why}), "
                    "but the other constraints give them one color"
                )
            return
        parent[ru], parity[ru] = rv, pu ^ pv ^ 1

    adjacency = {kind: [[] for _ in range(n)] for kind in rows}
    for kind, checks in rows.items():
        for c, qubits in enumerate(checks):
            for q in qubits:
                adjacency[kind][q].append(c)

    def shared_pairs(a, b):
        """Yield (i, j, shared qubits) for check pairs that share qubits."""
        for i, qubits in enumerate(rows[a]):
            seen = {}
            for q in qubits:
                for j in adjacency[b][q]:
                    if a != b or j > i:
                        seen.setdefault(j, []).append(q)
            for j, shared in seen.items():
                yield i, j, shared

    for kind in rows:
        for i, j, shared in shared_pairs(kind, kind):
            if len(shared) >= 3:
                raise ValueError(
                    f"no BB layering: {kind} checks {i} and {j} share "
                    f"{len(shared)} qubits, so rule (b) fails"
                )
            if len(shared) == 2:
                differ(*shared, f"rule (b), {kind} checks {i} and {j}")
    for i, j, shared in shared_pairs("X", "Z"):
        if len(shared) != 2:
            raise ValueError(
                f"no BB layering: X check {i} and Z check {j} share "
                f"{len(shared)} qubits; rules (b) and (c) need exactly 2"
            )
        differ(*shared, f"rule (c), X check {i} and Z check {j}")

    roots = [find(q) for q in range(n)]
    order = list(dict.fromkeys(roots))  # components by smallest qubit index
    index = {root: k for k, root in enumerate(order)}
    first = {}
    for q in range(n):
        first.setdefault(roots[q], q)
    # Parity relative to the smallest qubit of the component; 0 is yellow at flip 0.
    relative = [
        (parity[q] if q != roots[q] else 0)
        ^ (parity[first[roots[q]]] if first[roots[q]] != roots[q] else 0)
        for q in range(n)
    ]
    # Per check: half size, and per component the counts of parity-0 and parity-1 qubits.
    terms = []
    for checks in rows.values():
        for qubits in checks:
            counts = {}
            for q in qubits:
                counts.setdefault(index[roots[q]], [0, 0])[relative[q]] += 1
            terms.append((len(qubits) // 2, counts))
    by_component = [[] for _ in order]
    for t, (_, counts) in enumerate(terms):
        for k in counts:
            by_component[k].append(t)
    yellow = [0] * len(terms)
    unset = [sum(map(sum, counts.values())) for _, counts in terms]

    def apply(k, flip, sign):
        for t in by_component[k]:
            p0, p1 = terms[t][1][k]
            yellow[t] += sign * (p1 if flip else p0)
            unset[t] -= sign * (p0 + p1)

    def feasible(k):
        return all(
            terms[t][0] - unset[t] <= yellow[t] <= terms[t][0] for t in by_component[k]
        )

    # Exact backtracking over component flips, flip 0 first.
    choice, k = [-1] * len(order), 0
    while 0 <= k < len(order):
        if choice[k] >= 0:
            apply(k, choice[k], -1)
        choice[k] += 1
        if choice[k] > 1:
            choice[k] = -1
            k -= 1
            continue
        apply(k, choice[k], 1)
        if feasible(k):
            k += 1
    if k < 0:
        raise ValueError(
            "no BB layering: no choice of layers gives each check half of its "
            "qubits in each layer, so rule (a) fails"
        )
    mask = [(relative[q] ^ choice[index[roots[q]]]) == 0 for q in range(n)]
    return torch.tensor(mask, dtype=torch.bool)
