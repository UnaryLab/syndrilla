from pathlib import Path

import pytest
import torch

from syndrilla.decoder.bp_symbreak.bb_layers import bb_layers
from syndrilla.matrix import load_matrices
from syndrilla.utils import read_yaml

ROOT = Path(__file__).resolve().parents[1]


def matrices(name):
    bundle = load_matrices(
        read_yaml(str(ROOT / f"examples/alist/{name}.matrix.yaml"))["matrix"],
        torch.device("cpu"),
        torch.float64,
    )
    n = bundle.select("hx")[0][1]
    cols = [bundle.select(t)[2] for t in ("hx", "hz")]
    dense = [bundle.select(t)[3].to_dense().long() for t in ("hx", "hz")]
    return cols, dense, n


def check_rules(hx, hz, yellow):
    y = yellow.long()
    for h in (hx, hz):  # (a): half of each check in each layer
        assert torch.equal(2 * (h @ y), h.sum(1))
    for color in (y, 1 - y):  # (b): no two checks share two same-color qubits
        layer = torch.cat((hx, hz)) * color
        shared = layer @ layer.T
        shared.fill_diagonal_(0)
        assert shared.max() <= 1
    overlap = hx @ hz.T  # (c): each layer holds an odd part of each X/Z overlap
    part = (hx * y) @ hz.T
    touching = overlap > 0
    assert (part[touching] % 2 == 1).all() and (
        (overlap - part)[touching] % 2 == 1
    ).all()


def test_bb_144_layers_satisfy_rules_and_split_blocks():
    (hx_col, hz_col), (hx, hz), n = matrices("bb_144_12_12")
    yellow = bb_layers(hx_col, hz_col, n)
    assert yellow.dtype == torch.bool and yellow.shape == (n,)
    check_rules(hx, hz, yellow)
    assert (hx.long() @ yellow.long() == 3).all() and (
        hz.long() @ yellow.long() == 3
    ).all()
    # The coloring is unique up to a swap: the A|B column blocks.
    assert yellow[: n // 2].all() and not yellow[n // 2 :].any()
    assert torch.equal(bb_layers(hx_col, hz_col, n), yellow)


def test_rule_c_does_not_follow_from_rule_a():
    _, (hx, hz), n = matrices("bb_144_12_12")
    even = torch.arange(n // 2) % 2 == 0
    y = torch.cat((even, ~even)).long()  # 3/3 per check, not the block split
    assert (hx @ y == 3).all() and (hz @ y == 3).all()
    part = (hx * y) @ hz.T
    assert ((hx @ hz.T > 0) & (part % 2 == 0)).any()


def test_odd_degree_surface_code_raises():
    (hx_col, hz_col), _, n = matrices("surface_10")
    with pytest.raises(ValueError, match="odd degree"):
        bb_layers(hx_col, hz_col, n)


def test_rotated_surface_has_valid_coloring():
    (hx_col, hz_col), (hx, hz), n = matrices("rotated_surface_11")
    check_rules(hx, hz, bb_layers(hx_col, hz_col, n))


def test_three_shared_qubits_raise():
    n = 6
    col = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])
    with pytest.raises(ValueError, match="share 3 qubits"):
        bb_layers(col, torch.full((1, 4), n), n)
