"""SymBreak graph splits, stopping rules, factory and fixed CPU smoke samples."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as Nms
from syndrilla.decoder.bp_symbreak.bp_symbreak import create as SymBreak
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cfg(**extra):
    return dict(device={'device_type': 'cpu'}, dtype='float64', check_type='hx',
                max_iter=50, compile=False, rebatch_opt=False, **extra)


def tiny(h=None, g=None, **extra):
    h = torch.as_tensor([[1, 1, 1, 1, 1, 1]] if h is None else h)
    g = torch.as_tensor([[1, 1, 1, 1, 0, 0]] if g is None else g)
    indices = {key: dense_to_index_format(value.numpy(), torch.device('cpu'))
               for key, value in [('hx', h), ('hz', g)]}
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None,
                             lz_matrix=None, select=lambda key: indices[key])
    settings = cfg(**extra)
    return SymBreak(settings, bundle=bundle), bundle, settings


def graph(dec):
    n, m = dec.H_shape[1], dec.H_shape[0]
    col = torch.full((m + min(dec.max_splits, m), dec.V_c_col.shape[1]), n, dtype=torch.long)
    col[:m] = dec.V_c_col
    synd = torch.zeros(len(col), dtype=torch.float64)
    synd[0] = 1
    incoming = torch.arange(col.numel(), dtype=torch.float64).view_as(col).clone()
    incoming[m:] = 0
    splittable = torch.arange(len(col)) < m
    return (col, synd, incoming, torch.zeros_like(col, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long), splittable)


def test_fig6_bp_split_and_moved_edge_reset():
    dec, _, _ = tiny(g=[[1, 0, 1, 1, 0, 1]])
    col, synd, incoming, reset, splits, splittable = graph(dec)
    posterior = torch.tensor([10., -60., -15., 5., 50., -20., float('inf')])
    old = incoming.clone()
    assert dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert col[0].tolist() == [6, 1, 6, 6, 4, 5]
    assert col[1].tolist() == [0, 6, 2, 3, 6, 6]
    assert synd[:2].tolist() == [0, 1]
    assert reset[1].tolist() == [True, False, True, True, False, False]
    assert torch.equal(incoming[0, [1, 4, 5]], old[0, [1, 4, 5]])
    assert not incoming[0, [0, 2, 3]].any()
    assert splits.tolist() == [1]
    assert not splittable.any()
    # Both children have odd overlap and their XOR restores the parent check.
    for row in col[:2]:
        assert int(torch.isin(row[row < 6], torch.tensor([0, 2, 3, 5])).sum()) % 2
    assert sorted(col[:2][col[:2] < 6].tolist()) == list(range(6))


def test_syndrome_guided_excludes_parent_and_uses_neighbour_checks():
    dec, _, _ = tiny(assign='syndrome')
    col, synd, incoming, reset, splits, splittable = graph(dec)
    posterior = torch.tensor([10., -60., -15., 5., 50., 20., float('inf')])
    assert dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert col[0][col[0] < 6].tolist() == [0, 1, 2]
    assert synd[:2].tolist() == [0, 1]  # The empty set of neighbour checks is valid.


@pytest.mark.parametrize('neighbour,expected', [
    ([0, 1, 0, 0, 0, 0], [0, 2, 3]),
    ([0, 1, 0, 0, 1, 1], [0, 2, 3]),
    ([1, 1, 1, 1, 1, 1], None),
])
def test_syndrome_partition_search_and_skip_without_mutation(neighbour, expected):
    dec, _, _ = tiny(h=[[1] * 6, neighbour], g=[[1, 0, 1, 1, 0, 1]], assign='syndrome')
    col, synd, incoming, reset, splits, splittable = graph(dec)
    synd[1] = 1
    posterior = torch.tensor([10., -60., -15., 5., 50., 20., float('inf')])
    before = [value.clone() for value in (col, synd, incoming, reset, splits, splittable)]
    changed = dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert changed == (expected is not None)
    if expected is None:
        for actual, old in zip((col, synd, incoming, reset, splits, splittable), before):
            assert torch.equal(actual, old)
        return
    first = col[0][col[0] < 6].tolist()
    assert first == expected and first != [1, 4, 5]
    assert col[2][col[2] < 6].tolist() == [1, 4, 5]
    assert synd[[0, 2]].tolist() == [0, 1]
    assert sorted(col[[0, 2]][col[[0, 2]] < 6].tolist()) == list(range(6))
    assert not splittable[[0, 2]].any()


@pytest.mark.parametrize('degree,overlap,expected', [
    (5, 4, [0, 4]), (8, 6, [0, 1, 2, 6]),
])
def test_syndrome_partition_balances_degree_and_overlap(degree, overlap, expected):
    dec, _, _ = tiny(h=[[1] * degree], g=[[1] * overlap + [0] * (degree - overlap)],
                     assign='syndrome')
    col, synd, incoming, reset, splits, splittable = graph(dec)
    posterior = torch.arange(1., degree + 2)
    assert dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert col[0][col[0] < degree].tolist() == expected
    assert (col[0] < degree).sum() == degree // 2
    assert (col[1] < degree).sum() == degree - degree // 2


def test_both_children_cannot_split_again_with_another_opposite_check():
    dec, _, _ = tiny(h=[[1] * 8], g=[[1, 1, 1, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 1, 1, 1, 0, 0]])
    col, synd, incoming, reset, splits, splittable = graph(dec)
    posterior = torch.tensor([1., 2., 3., 4., 10., 20., 30., 40., float('inf')])
    assert dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert not splittable.any() and splits.tolist() == [1]
    assert (col != 8).any(1).sum() == 2 * dec.H_shape[0]
    posterior = torch.tensor([1., 100., 100., 1., 1., 1., 100., 100., float('inf')])
    # Both children now have positive even overlap with the least reliable check.
    assert torch.isin(col, torch.tensor([0, 3, 4, 5])).sum(1).tolist() == [2, 2]
    before = [value.clone() for value in (col, synd, incoming, reset, splits, splittable)]
    assert not dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    for actual, old in zip((col, synd, incoming, reset, splits, splittable), before):
        assert torch.equal(actual, old)


def test_impossible_syndrome_partition_continues_k_monitor():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], assign='syndrome',
                     bp_iters=2, k_max=2)
    out = dec(dict(llr0=torch.full((1, 2), 3.), synd=torch.ones(1, 2)))
    assert out['iter'].tolist() == [6]
    assert out['splits'].tolist() == [0] and not out['converge'].any()


@pytest.mark.parametrize('key,value', [('bp_iters', 0), ('bp_iters', True),
    ('bp_iters', 1.5), ('k_max', 0), ('k_max', None), ('max_splits', -1),
    ('max_splits', False), ('assign', 'bad'), ('max_iter', True), ('max_iter', 0),
    ('dtype', 'float16'), ('check_type', 'bad'), ('early_termination', 0)])
def test_knob_validation(key, value):
    settings = {key: value}
    # cfg contains only the base defaults; replace rather than duplicate them.
    if key in ('max_iter', 'dtype', 'check_type'):
        _, bundle, base = tiny()
        base[key] = value
        with pytest.raises(ValueError, match=key):
            SymBreak(base, bundle=bundle)
    else:
        with pytest.raises(ValueError, match=key):
            tiny(**settings)


def test_d_monitor_stops_stagnating_shot_after_reference_block():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=2,
                     k_max=2, max_splits=0)
    out = dec(dict(llr0=torch.full((1, 2), 3.), synd=torch.tensor([[0, 1]])))
    assert out['iter'].tolist() == [6]
    assert not out['converge'].any() and not out['defer'].any()
    assert out['splits'].tolist() == [0]


def test_full_budget_disables_d_stop_and_handles_empty_batch():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=2,
                     k_max=1, max_splits=0, early_termination=False)
    out = dec(dict(llr0=torch.full((1, 2), 3.), synd=torch.tensor([[0, 1]])))
    assert out['iter'].tolist() == [dec.max_iter] and not out['converge'].any()
    out = dec(dict(llr0=torch.empty(0, 2), synd=torch.empty(0, 2)))
    assert out['llr'].shape == out['e_v'].shape == (0, 2)
    assert out['iter'].shape == out['splits'].shape == (0,)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_no_split_bit_exact_nms_every_iteration(dtype):
    dec, bundle, settings = tiny(bp_iters=1, max_splits=0, k_max=100)
    settings['dtype'] = dtype
    settings['max_iter'] = 25
    dec = SymBreak(settings, bundle=bundle)
    plain = Nms({key: value for key, value in settings.items()
                 if key not in ('bp_iters', 'max_splits', 'k_max')},
                bundle=bundle)
    generator = torch.Generator().manual_seed(17)
    llr = torch.rand((32, 6), generator=generator, dtype=dec.dtype) * 8 - 4
    synd = torch.arange(32).remainder(2).reshape(-1, 1)
    actual = dec(dict(llr0=llr, synd=synd))
    with torch.no_grad():
        expected = plain(dict(llr0=llr, synd=synd))
    assert expected['converge'].any() and not actual['splits'].any()
    for key in ('llr', 'e_v', 'iter', 'converge', 'defer'):
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_no_split_blocks_keep_hard_results_and_round_iterations(dtype):
    _, bundle, settings = tiny(bp_iters=10, max_splits=0, k_max=100)
    settings.update(dtype=dtype, max_iter=25)
    dec = SymBreak(settings, bundle=bundle)
    plain = Nms({key: value for key, value in settings.items()
                 if key not in ('bp_iters', 'max_splits', 'k_max')}, bundle=bundle)
    io = dict(llr0=torch.full((2, 6), 3., dtype=dec.dtype), synd=torch.tensor([[0], [1]]))
    actual, expected = dec(dict(io)), plain(dict(io))
    for key in ('converge', 'e_v'):
        assert torch.equal(actual[key], expected[key]), key
    rounded = ((expected['iter'] + 9) // 10 * 10).clamp_max(25)
    assert torch.equal(actual['iter'], rounded)
    assert not actual['splits'].any()
    assert not torch.equal(actual['llr'][0], expected['llr'][0])


@pytest.mark.parametrize('final_match', [False, True])
def test_transient_convergence_waits_for_block_boundary(final_match):
    dec, _, _ = tiny(h=[[1, 1]], g=[[0, 0]], bp_iters=2, max_splits=0)
    dec.max_iter = 4
    visited = []
    def scripted(outgoing, syndrome, out):
        visited.append(dec.i)
        out.zero_()
        if dec.i in (1, 3) or (dec.i == 4 and final_match):
            out[:, 0, 0] = -2
        return out
    dec.cn_update = scripted
    out = dec(dict(llr0=torch.ones(1, 2), synd=torch.ones(1, 1)))
    assert visited == [1, 2, 3, 4]
    assert out['iter'].tolist() == [4]
    assert out['converge'].tolist() == [int(final_match)]


def test_partial_final_block_checks_final_state():
    dec, bundle, settings = tiny(bp_iters=10, max_splits=0)
    settings.update(max_iter=3)
    dec = SymBreak(settings, bundle=bundle)
    full = SymBreak(dict(settings, early_termination=False), bundle=bundle)
    io = dict(llr0=torch.full((1, 6), 3.), synd=torch.zeros(1, 1))
    actual, expected = dec(dict(io)), full(dict(io))
    assert actual['iter'].tolist() == [3] and actual['converge'].tolist() == [1]
    assert torch.equal(actual['llr'], expected['llr'])


def test_unary_splits_are_finite_and_shot_local():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=1, k_max=3)
    llr = torch.full((2, 2), 3.)
    synd = torch.tensor([[0, 0], [0, 1]])
    original = {'llr0': llr.clone(), 'synd': synd.clone()}
    a = dec(dict(llr0=llr, synd=synd))
    b = dec(dict(llr0=llr, synd=synd))
    assert a['splits'][0] == 0 and a['splits'][1] > 0
    assert int(a['splits'].max()) <= dec.H_shape[0]
    assert a['iter'][0] == 1 and a['converge'][0] == 1
    assert torch.isfinite(a['llr']).all()
    for key in ('e_v', 'llr', 'iter', 'converge', 'splits'):
        assert torch.equal(a[key], b[key])
    assert torch.equal(llr, original['llr0']) and torch.equal(synd, original['synd'])


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
def test_factory_cpu(check_type):
    settings = dict(algorithm='bp_symbreak', check_type=check_type, dtype='float64',
                    device={'device_type': 'cpu'}, config=dict(bp_iters=2, max_iter=7))
    bundle = load_matrices(read_yaml(str(ROOT / 'examples/alist/surface_5.matrix.yaml'))['matrix'],
                           *parse_device_dtype(settings))
    dec = create_decoder(cfg=settings, bundle=bundle)[0]
    h = bundle.select(check_type)[3].to_dense().double()
    out = dec(dict(llr0=torch.full((3, h.shape[1]), 3.), synd=torch.zeros(3, h.shape[0])))
    assert dec.algo == 'bp_symbreak' and out['converge'].tolist() == [1, 1, 1]
    assert out['splits'].tolist() == [0, 0, 0]


@pytest.mark.parametrize('name', ['surface_10', 'bb_144_12_12'])
def test_cpu_fixed_smoke(name):
    settings = cfg()
    settings['max_iter'] = 100
    bundle = load_matrices(read_yaml(str(ROOT / f'examples/alist/{name}.matrix.yaml'))['matrix'],
                           *parse_device_dtype(settings))
    h = bundle.select('hx')[3].to_dense().double()
    p = .005
    errors = (torch.rand((512, h.shape[1]), generator=torch.Generator().manual_seed(0)) < p).double()
    synd = (errors @ h.T).remainder(2)
    llr = torch.full_like(errors, math.log((1 - p) / p))
    with torch.no_grad():
        plain = Nms(settings, bundle=bundle)(dict(llr0=llr, synd=synd))
        split = SymBreak(settings, bundle=bundle)(dict(llr0=llr, synd=synd))
    counts = {key: int((value['converge'] == 0).sum())
              for key, value in [('nms_never', plain), ('symbreak_never', split)]}
    print(f'{name} p={p} seed=0 shots=512 max_iter=100 {counts} splits={int(split["splits"].sum())}')
    assert torch.isfinite(split['llr']).all()
    early = plain['converge'].bool() & (plain['iter'] <= 10)
    assert early.any() and not split['splits'][early].any()
    for key in ('e_v', 'converge'):
        assert torch.equal(split[key][early], plain[key][early]), key
    assert (split['iter'][early] == 10).all()
    good = split['converge'].bool()
    assert torch.equal((split['e_v'][good] @ h.T).remainder(2), synd[good])
    assert ((split['iter'] >= 1) & (split['iter'] <= 100)).all()
    # Report the measured comparison. Do not assert or tune for an improvement.


def test_d_monitor_decrements_when_mismatch_decreases():
    dec, _, _ = tiny(h=torch.eye(3, dtype=torch.long), g=[[0, 0, 0]],
                     bp_iters=1, k_max=2)
    counts = [3, 2, 2, 1, 1, 1]
    def scripted(outgoing, syndrome, out):
        out.zero_()
        out[:, :counts[dec.i - 1], :] = -2
        return out
    dec.cn_update = scripted
    out = dec(dict(llr0=torch.ones(1, 3), synd=torch.zeros(1, 3)))
    assert out['iter'].tolist() == [6] and not out['converge'].any()


def test_moved_edges_use_channel_on_next_bp_iteration():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=1, k_max=3)
    original = dec.cn_update
    reset_edges = []
    native_split = dec._split
    def record_split(row, col, syndrome, posterior, hard, incoming, reset, splits, splittable):
        changed = native_split(row, col, syndrome, posterior, hard, incoming, reset, splits, splittable)
        if changed:
            reset_edges.append(reset.clone())
        return changed
    def check(outgoing, syndrome, out):
        if reset_edges:
            moved = reset_edges.pop(0)
            assert torch.equal(outgoing[0][moved], torch.full_like(outgoing[0][moved], 3.))
        return original(outgoing, syndrome, out)
    dec._split, dec.cn_update = record_split, check
    out = dec(dict(llr0=torch.full((1, 2), 3.), synd=torch.tensor([[0, 1]])))
    assert out['splits'].sum() > 0


def test_stable_equal_reliability_choice_and_split_budget():
    dec, _, _ = tiny(bp_iters=1, max_splits=1)
    col, synd, incoming, reset, splits, splittable = graph(dec)
    posterior = torch.tensor([2., 2., 2., 2., 2., 2., float('inf')])
    dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert sorted(col[0][col[0] < 6].tolist()) == [0, 4, 5]
    out = dec(dict(llr0=torch.full((1, 6), 3.), synd=torch.ones(1, 1)))
    assert int(out['splits'].max()) <= 1


def test_adaptive_cap_and_bypass():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=2)
    dec.cap = SimpleNamespace(done=True, frac=.5)
    io = dict(llr0=torch.full((2, 2), 3.), synd=torch.tensor([[0, 0], [0, 1]]))
    out = dec(dict(io))
    assert dec.cap_active_last and out['iter'].tolist() == [2, 2]
    assert out['defer'].tolist() == [False, True]
    dec.cap_bypass = True
    out = dec(dict(io))
    assert not dec.cap_active_last and not out['defer'].any()
    assert out['iter'][1] > 2


def test_factory_full_budget_registration():
    _, bundle, _ = tiny()
    dec = create_decoder(cfg=dict(algorithm='bp_symbreak', device={'device_type': 'cpu'},
        config=dict(max_iter=5, early_termination=False)), bundle=bundle)[0]
    out = dec(dict(llr0=torch.full((1, 6), 3.), synd=torch.zeros(1, 1)))
    assert out['iter'].tolist() == [5]


def test_factory_rounds_split_output_shape():
    _, bundle, _ = tiny()
    dec = create_decoder(cfg=dict(algorithm='bp_symbreak', device={'device_type': 'cpu'},
        config=dict(max_iter=5)), bundle=bundle)[0]
    out = dec(dict(llr0=torch.full((2, 6), 3.), synd=torch.zeros(2, 3, 1)))
    assert out['e_v'].shape == (2, 3, 6) and out['iter'].shape == (2, 3)
    assert out['splits'].shape == (2, 3)


def test_equal_check_reliability_selects_lowest_row():
    dec, _, _ = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]])
    col, synd, incoming, reset, splits, splittable = graph(dec)
    old = col[1].clone()
    posterior = torch.tensor([2., 2., float('inf')])
    dec._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    assert col[0].tolist() == [0, 2] and torch.equal(col[1], old)
    assert col[2].tolist() == [2, 1]
