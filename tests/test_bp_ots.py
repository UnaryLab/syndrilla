"""OTS pin rules and the shared sum-product loop, on CPU and CUDA."""
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_ots.bp_ots import create as Ots
from syndrilla.decoder.bp_sum_prod.bp_sum_prod import create as SumProduct
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.utils import read_yaml, parse_device_dtype


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(backend='cpu', **extra):
    if backend != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cfg = dict(device={'device_type': 'cpu' if backend == 'cpu' else 'cuda'},
        dtype='float64', check_type='hx', max_iter=12, compile=False, rebatch_opt=False,
        interval=2, strength=4.)
    cfg.update(extra)
    h = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]], dtype=torch.float64)
    index = dense_to_index_format(h, *parse_device_dtype(cfg)[:1])
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None, lz_matrix=None,
                             select=lambda _: index)
    base = torch.rand((9, 6), generator=torch.Generator().manual_seed(73), dtype=torch.float64) * 6 - 3
    base[0] = 2
    synd = torch.zeros(9, 4, dtype=torch.uint8)
    synd[1:, 0] = 1
    module = importlib.import_module('syndrilla.decoder.bp_ots.bp_ots' + ('_cuda' if backend == 'cuda' else ''))
    return module.create(cfg, bundle=bundle), bundle, dict(llr0=base, synd=synd), h


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_no_intervention_is_shared_sum_product(backend):
    ots, bundle, io, _ = fixture(backend, interval=100)
    cfg = dict(device={'device_type': ots.device.type}, dtype='float64', max_iter=12,
               compile=False, rebatch_opt=False)
    plain = SumProduct(cfg, bundle=bundle)
    with torch.no_grad():
        actual, expected = ots(dict(io)), plain(dict(io))
    for key in ('e_v', 'llr', 'iter', 'converge', 'defer'):
        if backend == 'cuda' and key == 'llr':
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        else:
            assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_hand_rule_ties_reset_only_oscillator_and_restore_prior(backend):
    if backend == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    device = torch.device(backend)
    channel = torch.tensor([[2., 3., 4., 5.], [2., 3., 4., 5.], [2., 3., 4., 5.]])
    posterior = torch.tensor([[3., -1., 1., .1, float('inf')],
                              [.1, 1., 2., 3., float('inf')],
                              [1., 2., 3., 4., float('inf')]])
    current = torch.full_like(channel, -4)
    counts = torch.tensor([[4, 5, 5, 1], [0, 0, 0, 0], [1, 2, 3, 4]])
    channel, posterior, current, counts = (value.to(device) for value in (channel, posterior, current, counts))
    old = posterior.clone()
    incoming = torch.cat((posterior[:, :-1] - current, torch.zeros(3, 1, device=device)), 1)[:, None, :]
    dec = SimpleNamespace(_ots_channel=channel, _ots_current=current.clone(), _ots_counts=counts,
        _ots_prior=torch.cat((current.clone(), torch.full((3, 1), float('inf'), device=device)), 1),
        _ots_incoming=incoming, strength=10.)
    implementation = Ots
    if backend == 'cuda':
        from syndrilla.decoder.bp_ots.bp_ots_cuda import create as implementation, _load_ext
        dec._ots_ext = _load_ext()
        dec._ots_edges = incoming[:, 0, :].contiguous()
        dec._ots_adjacency = torch.arange(5, device=device)[:, None]
    implementation._switch_priors(dec, torch.arange(3, device=device), posterior,
        torch.tensor([True, True, False], device=device))
    assert dec.last_ots_nodes.tolist() == [[1, 3], [-1, 0], [-1, -1]]
    expected = torch.tensor([[2., -10., 4., -10.], [-10., 3., 4., 5.], [-4., -4., -4., -4.]], device=device)
    assert torch.equal(dec._ots_current, expected)
    assert counts.tolist() == [[4, 0, 5, 1], [0, 0, 0, 0], [1, 2, 3, 4]]
    torch.testing.assert_close(posterior[:, :-1], old[:, :-1] + expected - current)
    assert torch.isposinf(posterior[:, -1]).all()


@pytest.mark.parametrize('compact', [0., .5, .75])
def test_active_rows_interval_final_iteration_and_repeated_batch(compact):
    dec, _, io, h = fixture(compact_frac=compact)
    calls = []
    native = dec._switch_priors
    def record(rows, values, active):
        assert dec.i < dec.max_iter and dec.i % dec.interval == 0
        before = values.clone()
        native(rows, values, active)
        assert torch.equal(values[~active], before[~active])
        calls.append(dec.i)
    dec._switch_priors = record
    original = {k: v.clone() for k, v in io.items()}
    with torch.no_grad():
        a = dec(dict(io)); b = dec(dict(io))
    assert calls and all(i in (2, 4, 6, 8, 10) for i in calls)
    for key in ('e_v', 'llr', 'iter', 'converge'):
        assert torch.equal(a[key], b[key])
    assert a['converge'][0] == 1 and a['iter'][0] == 1
    assert not a['converge'][1:].any()
    assert torch.equal(a['llr'] <= 0, a['e_v'].bool())
    assert torch.equal((a['e_v'][a['converge'].bool()] @ h.T).remainder(2), io['synd'][a['converge'].bool()].double())
    assert all(torch.equal(io[key], original[key]) for key in io)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('early', [True, False])
def test_cpu_pytorch_cuda_agree(dtype, early):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    outputs = []
    for backend in ('cpu', 'pytorch', 'cuda'):
        dec, _, io, _ = fixture(backend, dtype=dtype, early_termination=early)
        with torch.no_grad():
            outputs.append(dec(dict(io)))
    for out in outputs[1:]:
        torch.testing.assert_close(out['llr'].cpu(), outputs[0]['llr'], rtol=2e-5 if dtype == 'float32' else 1e-9,
                                   atol=2e-5 if dtype == 'float32' else 1e-9)
        for key in ('e_v', 'iter', 'converge'):
            assert torch.equal(out[key].cpu(), outputs[0][key]), key


@pytest.mark.parametrize('key,value', [('interval', 0), ('interval', -1), ('interval', True),
    ('interval', 1.5), ('interval', None), ('strength', 0), ('strength', float('nan')),
    ('strength', float('inf')), ('strength', True), ('max_iter', 0), ('max_iter', True), ('dtype', 'float16'), ('check_type', 'bad')])
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        fixture(**{key: value})


def test_empty_batch():
    dec, _, _, _ = fixture()
    out = dec(dict(llr0=torch.empty(0, 6), synd=torch.empty(0, 4)))
    assert out['llr'].shape == out['e_v'].shape == (0, 6)
    assert out['iter'].shape == out['converge'].shape == out['defer'].shape == (0,)


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
@pytest.mark.parametrize('early', [True, False])
def test_factory_cpu(check_type, early):
    cfg = dict(algorithm='bp_ots', check_type=check_type, device={'device_type': 'cpu'}, dtype='float64',
               config=dict(max_iter=8, compile=False, interval=2, strength=4., early_termination=early))
    bundle = load_matrices(read_yaml('examples/alist/surface_5.matrix.yaml')['matrix'], *parse_device_dtype(cfg))
    dec = create_decoder(cfg=cfg, bundle=bundle)[0]
    h = bundle.select(check_type)[3].to_dense().double()
    errors = (torch.rand((12, h.shape[1]), generator=torch.Generator().manual_seed(3)) < .1).double()
    io = dict(llr0=torch.full_like(errors, math.log(9)), synd=(errors @ h.T).remainder(2))
    with torch.no_grad():
        out = dec(io)
    assert out['e_v'].shape == errors.shape and out['iter'].shape == (12,)
    assert dec.algo == 'bp_ots' and not out['defer'].any()


def test_factory_rounds():
    _, bundle, io, _ = fixture()
    cfg = dict(algorithm='bp_ots', device={'device_type': 'cpu'}, dtype='float64',
        config=dict(max_iter=8, compile=False, interval=2))
    dec = create_decoder(cfg=cfg, bundle=bundle)[0]
    out = dec(dict(llr0=io['llr0'], synd=io['synd'][:, None, :].expand(-1, 3, -1)))
    assert out['e_v'].shape == (9, 3, 6) and out['converge'].shape == (9, 3)


def old_rule_cpu_oracle(h, base, syndrome, maximum, interval=2, strength=4.):
    """Dense tanh/atanh BP oracle for OTS hard decisions, convergence and iterations."""
    edges = h.nonzero().tolist()
    checks = [[e for e, (c, _) in enumerate(edges) if c == row] for row in range(len(h))]
    variables = [[e for e, (_, v) in enumerate(edges) if v == col] for col in range(h.shape[1])]
    records = []
    for channel, target in zip(base, syndrome):
        prior = channel.clone()
        outgoing = torch.stack([prior[v] for _, v in edges])
        incoming = torch.zeros_like(outgoing)
        previous = prior <= 0
        counts = torch.zeros(len(prior), dtype=torch.long)
        for iteration in range(1, maximum + 1):
            for c, ids in enumerate(checks):
                for edge in ids:
                    product = torch.tanh(outgoing[[other for other in ids if other != edge]] / 2).prod()
                    product *= 1 - 2 * int(target[c])
                    eps = torch.finfo(base.dtype).eps
                    incoming[edge] = 2 * product.clamp(-1 + eps, 1 - eps).atanh()
            posterior = torch.stack([prior[v] + incoming[ids].sum() for v, ids in enumerate(variables)])
            hard = posterior <= 0
            counts += (hard != previous).long()
            previous = hard.clone()
            converged = torch.equal((h @ hard.double()).remainder(2), target.double())
            if converged or iteration == maximum:
                break
            if iteration % interval == 0:
                prior = channel.clone()
                least = min(range(len(prior)), key=lambda v: (abs(float(posterior[v])), v))
                if counts.max() > 0:
                    oscillator = min(range(len(prior)), key=lambda v: (-int(counts[v]), abs(float(posterior[v])), v))
                    prior[oscillator] = -strength
                    counts[oscillator] = 0
                prior[least] = -strength
                posterior = torch.stack([prior[v] + incoming[ids].sum() for v, ids in enumerate(variables)])
            outgoing = torch.stack([posterior[v] - incoming[e] for e, (_, v) in enumerate(edges)])
        records.append((posterior, hard, iteration, converged))
    return dict(llr=torch.stack([r[0] for r in records]), e_v=torch.stack([r[1] for r in records]),
                iter=torch.tensor([r[2] for r in records]), converge=torch.tensor([r[3] for r in records]))


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('maximum', [1, 3, 12])
def test_old_rule_cpu_oracle_small_batch(dtype, maximum):
    dec, _, io, h = fixture(dtype=dtype, max_iter=maximum)
    expected = old_rule_cpu_oracle(h, io['llr0'].to(dec.dtype), io['synd'], maximum)
    with torch.no_grad():
        out = dec(dict(io))
    # The shared probability-domain base changes soft rounding and saturation.
    for key in ('e_v', 'iter', 'converge'):
        assert torch.equal(out[key], expected[key]), key
    assert out['llr'].isfinite().all()


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_all_optimization_groups_off(backend):
    default, _, io, _ = fixture(backend, rebatch_opt=True)
    groups = dict(pruning_opt=False, fusion_opt=False, mapping_opt=False,
                  gather_opt=False, memory_opt=False, rebatch_opt=False)
    off, _, _, _ = fixture(backend, **groups)
    assert default.cap is not None and off.cap is None
    if backend == 'cuda':
        assert not off._host_check_every and not off._skip_converged
        assert not off._warp_per_check and not off._fuse_vn and not off._vn_gather
        assert off._padded
        assert off._kernel_knobs() == (False, False, False, True, False)
    else:
        assert off.compact_frac == 0 and not off.reuse_buffers and not off.compile
    with torch.no_grad():
        expected, actual = default(dict(io)), off(dict(io))
    for key in ('e_v', 'converge'):
        assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_explicit_member_overrides_optimization_group(backend):
    dec, _, _, _ = fixture(backend, memory_opt=False, reuse_buffers=True,
        pruning_opt=False, compact_frac=.5, host_check_every=2, skip_converged=True,
        fusion_opt=False, fuse_vn=True, mapping_opt=False, warp_per_check=True,
        edge_layout='csr', gather_opt=False, vn_gather=True)
    if backend == 'cuda':
        assert dec._host_check_every == 2 and dec._skip_converged
        assert dec._fuse_vn and dec._warp_per_check and dec._vn_gather and not dec._padded
        assert dec._kernel_knobs() == (True, True, False, False, True)
    else:
        assert dec.reuse_buffers and dec.compact_frac == .5


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_rebatch_cap_stops_and_bypass_runs_full_decode(backend):
    dec, _, io, _ = fixture(backend, rebatch_opt=True)
    dec.cap.frac = 1 / len(io['synd'])
    with torch.no_grad():
        stopped = dec(dict(io))
    assert dec.cap_active_last and stopped['iter'].max() == 1
    assert stopped['defer'].sum() == len(io['synd']) - 1
    empty = dec(dict(llr0=io['llr0'][:0], synd=io['synd'][:0]))
    assert not dec.cap_active_last and not empty['defer'].any()
    dec.cap_bypass = True
    with torch.no_grad():
        full = dec(dict(io))
    assert not dec.cap_active_last and not full['defer'].any()
    assert full['iter'].max() == dec.max_iter


def test_explicit_compile_overrides_fusion_group_and_keeps_interval_state():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    compiled, _, io, _ = fixture('pytorch', compile=True, fusion_opt=False)
    if not compiled.compile:
        pytest.skip('torch.compile backend unavailable')
    eager, _, _, _ = fixture('pytorch', compile=False)
    io['llr0'][:7] = 2
    io['synd'][:7] = 0
    with torch.no_grad():
        expected = eager(dict(io))
        first = compiled(dict(io))
        repeated = compiled(dict(io))
    for actual in (first, repeated):
        for key in ('e_v', 'iter', 'converge'):
            assert torch.equal(actual[key], expected[key]), key
        torch.testing.assert_close(actual['llr'], expected['llr'], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('reuse', [True, False])
def test_compaction_keeps_original_shot_prior_state(reuse):
    plain, _, io, _ = fixture(compact_frac=0, reuse_buffers=reuse)
    compact, _, _, _ = fixture(compact_frac=.75, reuse_buffers=reuse)
    io['llr0'][:7] = 2
    io['synd'][:7] = 0
    seen = []
    switch = compact._switch_priors
    def record(rows, values, active):
        seen.append(rows.tolist())
        switch(rows, values, active)
    compact._switch_priors = record
    with torch.no_grad():
        expected, actual = plain(dict(io)), compact(dict(io))
    assert seen and all(rows == [7, 8] for rows in seen)
    for key in ('e_v', 'llr', 'iter', 'converge'):
        assert torch.equal(expected[key], actual[key]), key
