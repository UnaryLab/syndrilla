import importlib
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from syndrilla.decoder.bp_lottery_pre.bp_lottery_pre import create as Pre
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml


def surface(backend="cpu", algorithm="bp_lottery_pre", without_pre=False, **extra):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = dict(device={"device_type": "cpu" if backend == "cpu" else "cuda"},
               dtype="float64", check_type="hx", max_iter=12, compile=False,
               rebatch_opt=False, random_machine="sobol")
    if algorithm == "bp_lottery_pre" and not without_pre:
        cfg["pre_parallel_max_lambda"] = 15
    cfg.update(extra)
    bundle = load_matrices(read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"],
                           *parse_device_dtype(cfg))
    suffix = "_cuda" if backend == "cuda" else ""
    cls = importlib.import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}").create
    return cls(cfg, bundle=bundle), bundle.select("hx")[3].to_dense().cpu().double()


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)

def inputs(H, seed=0, shots=32):
    generator = torch.Generator().manual_seed(seed)
    error = (torch.rand((shots, H.shape[1]), generator=generator) < .1).double()
    return dict(synd=(error @ H.T).remainder(2), llr0=torch.full_like(error, math.log(9)))

def test_single_bit_fix_uses_first_candidate_without_decode():
    dec, H = surface(pre_parallel_max_lambda=181)
    syndrome = H[:, 7][None].clone()
    hard = torch.zeros((1, H.shape[1]), dtype=torch.float64)
    counts, conv = torch.tensor([12]), torch.tensor([0])
    candidates, valid, _ = dec._rank_pre(syndrome, hard)
    matching = torch.where((H[:, candidates[0]].T == syndrome).all(1))[0]
    expected = int(candidates[0, matching[0]])
    dec._bp_core = lambda *args: pytest.fail("single_bit_fix must not run a trial")
    serial, parallel = dec._pre_postprocess(syndrome, torch.ones_like(hard), hard,
                                          torch.ones_like(hard), conv, counts)
    assert hard[0, expected] == 1 and hard.sum() == 1 and conv.tolist() == [1]
    assert serial.tolist() == parallel.tolist() == [12]

def test_parallel_selection_serial_counts_and_recombination():
    calls = []

    def retry(syndrome, llr):
        calls.append(syndrome.clone())
        assert syndrome.shape == (9, 5)
        assert torch.equal(syndrome[:3], (1 - torch.eye(5)[[0, 1, 2]]).double())
        return (torch.zeros(9, 5, dtype=torch.float64), torch.arange(9).double()[:, None].expand(-1, 5),
                torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1]), torch.tensor([8, 3, 3, 10, 10, 10, 10, 5, 2]), None)

    dec = SimpleNamespace(H_shape=(5, 5), device=torch.device("cpu"), phase_knobs=[dict(flip_mode="lottery")], flip_phase_iters=[], parallel_enabled=True, max_lambda=3,
                          max_iter=12, pre_it=10,
                          V_v_row=torch.arange(5)[:, None], _bp_core=retry)
    dec._rank_pre = lambda syndrome, estimate: (torch.tensor([[0, 1, 2]]).expand(3, -1),
                                               torch.ones(3, 3, dtype=torch.bool), syndrome)
    hard = torch.zeros(4, 5, dtype=torch.float64)
    llr = torch.full_like(hard, -1)
    conv, counts = torch.tensor([1, 0, 0, 0]), torch.tensor([2, 12, 12, 12])
    serial, parallel = Pre._pre_postprocess(dec, torch.ones_like(hard), torch.ones_like(hard), hard, llr, conv, counts)
    assert len(calls) == 1 and dec.max_iter == 12
    assert serial.tolist() == [2, 20, 42, 27]
    assert parallel.tolist() == [2, 15, 22, 14]
    assert conv.tolist() == [1, 1, 0, 1]
    assert hard[1, 1] == hard[3, 2] == 1 and hard.sum() == 2
    assert llr[:, 0].tolist() == [-1, 1, -1, 8]
    assert counts.tolist() == [2, 12, 12, 12]

def test_skip_estimated_limits_trials_and_handles_empty_pool():
    calls = []

    def retry(syndrome, llr):
        calls.append(syndrome)
        assert syndrome.shape == (2, 5)
        return torch.zeros_like(llr), llr, torch.zeros(2, dtype=torch.long), torch.full((2,), 4), None

    dec = SimpleNamespace(H_shape=(5, 5), device=torch.device("cpu"), phase_knobs=[dict(flip_mode="lottery")], flip_phase_iters=[], parallel_enabled=True, max_lambda=15,
                          max_iter=8, pre_it=4, pre_degree=torch.ones(5, dtype=torch.long),
                          V_c_col=torch.arange(5)[:, None], V_v_row=torch.arange(5)[:, None],
                          _bp_core=retry)
    dec._rank_pre = lambda syndrome, estimate: Pre._rank_pre(dec, syndrome, estimate)
    hard = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float64)
    initial = hard.clone()
    counts = torch.tensor([8, 8])
    serial, parallel = Pre._pre_postprocess(dec, torch.zeros_like(hard), torch.ones_like(hard),
        hard, torch.ones_like(hard), torch.zeros(2, dtype=torch.long), counts)
    assert len(calls) == 1
    assert serial.tolist() == [16, 8] and parallel.tolist() == [12, 8]
    assert torch.equal(initial, hard)

def test_cpu_pytorch_cuda_agree():
    outputs = []
    for backend in ("cpu", "pytorch", "cuda"):
        dec, H = surface(backend)
        with torch.no_grad():
            outputs.append(dec(inputs(H)))
    for out in outputs[1:]:
        for key in ("e_v", "converge", "iter", "iter_serial", "iter_parallel"):
            assert torch.equal(outputs[0][key].cpu(), out[key].cpu()), key

@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_pre_preserves_and_improves_main_convergence(backend):
    baseline, H = surface(backend, without_pre=True)
    pre, _ = surface(backend)
    with torch.no_grad():
        old, new = baseline(inputs(H)), pre(inputs(H))
    assert not bool(old["converge"].all())
    assert bool((new["converge"] >= old["converge"]).all())
    assert new["converge"].sum() > old["converge"].sum()

def test_example_cli(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    result = subprocess.run(["syndrilla", f"-r={tmp_path}",
        "-d=examples/alist/bp_lottery_pre_hx.decoding.yaml", "-e=examples/alist/bsc.error.yaml",
        "-c=examples/alist/lx.check.yaml", "-s=examples/alist/perfect.syndrome.yaml",
        "-m=examples/alist/surface_10.matrix.yaml", "-bs=32", "-tb=1", "--seed=0"],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(Path(tmp_path).glob("result_phy_err_*.yaml"))

@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
@pytest.mark.parametrize('machine', ['sobol', 'system'])
def test_sibling_stop_preserves_winner_and_retires_siblings(backend, machine):
    outputs, trial_runs = [], []
    for stop in (False, True):
        dec, H = surface(backend, random_machine=machine)
        dec._stop_siblings = stop
        core = dec._bp_core
        calls = []
        def record(*args):
            result = core(*args)
            calls.append(tuple(v.clone() for v in (result[2], result[3])))
            return result
        dec._bp_core = record
        torch.manual_seed(713)
        with torch.no_grad():
            outputs.append(dec(inputs(H)))
        assert len(calls) == 1
        trial_runs.append(calls[0])
        if stop:
            trace = dec.last_pre_trials
            for owner in trace['owner'].unique():
                group = trace['owner'] == owner
                winner = group & trace['converged'].bool()
                if winner.any():
                    assert winner.sum() == 1
                    assert (trace['iterations'][group & trace['stopped']] <= trace['iterations'][winner]).all()
                    assert not trace['converged'][group & trace['stopped']].any()
    for key in ('e_v', 'llr', 'converge', 'iter', 'iter_parallel'):
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    assert (outputs[1]['iter_serial'] >= outputs[0]['iter_serial']).all()
    assert trial_runs[1][1].sum() < trial_runs[0][1].sum()

def test_sibling_serial_count_uses_observed_winner_and_one_decode():
    owner = torch.zeros(3, dtype=torch.long)
    state = dict(owner=owner, winner=torch.full((1,), 3, dtype=torch.long))
    dec = SimpleNamespace(device=torch.device('cpu'), _trial_state=state)
    finished = Pre._record_matches(dec, torch.tensor([False, True, True]), torch.arange(3))
    assert finished.all() and state['winner'].tolist() == [1]
    assert Pre._stopped_siblings(dec).tolist() == [True, False, True]
    calls = []
    def core(synd, llr):
        calls.append(synd)
        return torch.zeros_like(llr), llr, torch.tensor([0, 1, 0]), torch.full((3,), 3), None
    dec.phase_knobs, dec.flip_phase_iters = [dict(flip_mode="lottery")], []
    dec.parallel_enabled = True
    dec.H_shape, dec.max_lambda, dec.max_iter, dec.pre_it = (3, 3), 3, 12, 10
    dec.V_v_row, dec._bp_core = torch.arange(3)[:, None], core
    dec._rank_pre = lambda synd, est: (torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.bool), synd)
    estimate, posterior = torch.zeros(1, 3), torch.ones(1, 3)
    serial, parallel = Pre._pre_postprocess(dec, torch.ones(1, 3), torch.ones(1, 3),
        estimate, posterior, torch.zeros(1, dtype=torch.long), torch.tensor([12]))
    assert serial.item() == 12 + 10 + 3
    assert parallel.item() == 12 + 3
    assert len(calls) == 1

@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_bp_core_without_trial_groups_is_plain_lottery(backend):
    dec, H = surface(backend)
    plain, _ = surface(backend, 'bp_lottery')
    io = inputs(H, shots=8)
    with torch.no_grad():
        expected = plain(dict(io))
        actual = dec._bp_core(io['synd'], io['llr0'])
    for value, key in zip(actual[:4], ('e_v', 'llr', 'converge', 'iter')):
        assert torch.equal(value, expected[key]), key

@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_trial_rescues_have_consistent_posteriors_on_every_row(backend):
    plain, H = surface(backend, without_pre=True)
    dec, _ = surface(backend)
    io = inputs(H, seed=3, shots=256)
    with torch.no_grad():
        baseline = plain(dict(io))
        out = dec(dict(io))
    trial_rescued = (~baseline['converge'].bool() & out['converge'].bool()
                     & (out['iter_parallel'] > out['iter']))
    assert trial_rescued.any()
    assert torch.equal(out['llr'] <= 0, out['e_v'].bool())

def test_trial_zero_posterior_crosses_to_the_correct_hard_bit():
    dec = SimpleNamespace(H_shape=(1, 1), device=torch.device('cpu'), phase_knobs=[dict(flip_mode="lottery")], flip_phase_iters=[], parallel_enabled=True, max_lambda=1,
                          max_iter=12, pre_it=10, V_v_row=torch.tensor([[0]]))
    dec._rank_pre = lambda synd, est: (torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool), synd)
    dec._bp_core = lambda *args: (torch.ones(1, 1), torch.zeros(1, 1),
                                  torch.ones(1, dtype=torch.long), torch.tensor([2]), None)
    estimate, posterior = torch.ones(1, 1), -torch.ones(1, 1)
    converge = torch.zeros(1, dtype=torch.long)
    Pre._pre_postprocess(dec, torch.zeros(1, 1), torch.ones(1, 1), estimate,
                         posterior, converge, torch.tensor([12]))
    assert converge.item() == 1 and estimate.item() == 0
    assert posterior.item() == torch.finfo(posterior.dtype).tiny
    assert torch.equal(posterior <= 0, estimate.bool())


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_rank_residual_score_index_and_skips_estimate(backend):
    dec, H = surface(backend)
    io = inputs(H)
    estimate = inputs(H, seed=17)['llr0'].clone().bernoulli_(.1).to(dec.device)
    candidates, valid, residual = dec._rank_pre(io['synd'].to(dec.device), estimate)
    matrix, bits, syndrome = H.numpy().astype(int), estimate.cpu().numpy().astype(int), io['synd'].numpy().astype(int)
    expected = (syndrome + bits @ matrix.T) % 2
    score = matrix.sum(0) - 2 * (expected @ matrix)
    score[bits.astype(bool)] = np.iinfo(np.int64).max
    assert np.array_equal(candidates.cpu(), np.argsort(score, axis=1, kind='stable')[:, :15])
    assert np.array_equal(residual.cpu(), expected)
    assert valid.all()


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('compact', [0., .75])
def test_serial_pre_rescues_keeps_budget_and_skips_tried(backend, parallel, compact):
    options = dict(flip_phase_iters=[2], flip_mode=['lottery', 'pre'], pre_max_lambda=6, pre_max_iter=1)
    if parallel:
        options.update(pre_parallel_max_lambda=6, pre_parallel_max_iter=4, pre_parallel_start_iter=12)
    dec, H = surface(backend, without_pre=True, compact_frac=compact, **options)
    plain, _ = surface(backend, without_pre=True, compact_frac=compact)
    io = inputs(H, seed=3, shots=64)
    with torch.no_grad():
        old, new = plain(dict(io)), dec(dict(io))
    assert (new['converge'].bool() & ~old['converge'].bool()).any()
    assert (new['iter'] <= 12).all()
    for row in dec.last_pre_serial['tried']:
        selected = row[row >= 0]
        assert len(selected) == len(selected.unique())
    hits = dec.last_pre_serial['single_bit_fix_window'] >= 0
    assert hits.any() and new['converge'][hits].all()
    assert torch.equal(new['llr'] <= 0, new['e_v'].bool())
    match = new['converge'].cpu().bool()
    assert torch.equal((new['e_v'].cpu() @ H.T).remainder(2)[match], io['synd'][match])


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
@pytest.mark.parametrize('algorithm', ['bp_lottery', 'bp_lottery_quant', 'bp_lottery_layered'])
def test_pre_phase_available_in_lottery_variants(backend, algorithm):
    dec, H = surface(backend, algorithm, flip_phase_iters=[2, 8],
        flip_mode=['lottery', 'pre', 'lottery'], pre_max_lambda=3, pre_max_iter=2)
    with torch.no_grad():
        out = dec(inputs(H, seed=3, shots=64))
    state = dec.last_pre_serial
    assert (state['tried'] >= 0).any()
    assert (out['iter'] <= 12).all()
    valid = out['converge'].cpu().bool()
    assert torch.equal((out['e_v'].cpu() @ H.T).remainder(2)[valid], inputs(H, seed=3, shots=64)['synd'][valid])


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_pre_single_bit_fix_at_phase_boundary_before_first_window(backend):
    from syndrilla.matrix.matrix import dense_to_index_format
    if backend != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    index = dense_to_index_format(torch.ones(1, 2, dtype=torch.float64), torch.device('cpu' if backend == 'cpu' else 'cuda'))
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None, lz_matrix=None, select=lambda _: index)
    suffix = '_cuda' if backend == 'cuda' else ''
    cls = importlib.import_module('syndrilla.decoder.bp_lottery.bp_lottery' + suffix).create
    dec = cls(dict(device={'device_type': 'cpu' if backend == 'cpu' else 'cuda'}, dtype='float64',
        max_iter=3, compile=False, rebatch_opt=False, flip_phase_iters=[1], flip_mode=['off', 'pre'],
        pre_max_lambda=1, pre_max_iter=5), bundle=bundle)
    with torch.no_grad():
        out = dec(dict(synd=torch.ones(1, 1), llr0=torch.ones(1, 2)))
    assert dec.serial_windows == 0 and out['iter'].item() == 1
    assert out['converge'].item() and dec.last_pre_serial['single_bit_fix'].item()
    assert torch.equal(out['llr'] <= 0, out['e_v'].bool())


@pytest.mark.parametrize('key', ['pre_max_lambda', 'pre_max_iter'])
@pytest.mark.parametrize('value', [0, -1, True, None, 1.5])
def test_invalid_pre_phase_knobs(key, value):
    with pytest.raises(ValueError, match=key):
        surface(flip_mode='pre', **{key: value})


@pytest.mark.parametrize('key', ['pre_max_lambda', 'pre_max_iter'])
def test_pre_knobs_require_pre_phase(key):
    with pytest.raises(ValueError, match='requires a pre'):
        surface(flip_mode='lottery', **{key: 1})


@pytest.mark.parametrize('key', ['pre_parallel_max_lambda', 'pre_parallel_max_iter', 'pre_parallel_start_iter'])
@pytest.mark.parametrize('value', [0, -1, True, None, 1.5])
def test_invalid_parallel_knobs(key, value):
    with pytest.raises(ValueError, match=key):
        surface(**{key: value})


@pytest.mark.parametrize('key', ['pre_parallel_max_lambda', 'pre_parallel_max_iter', 'pre_parallel_start_iter'])
def test_parallel_knobs_require_pre_decoder(key):
    with pytest.raises(ValueError, match=f'unsupported key {key} for bp_lottery'):
        surface(algorithm='bp_lottery', **{key: 1})


def test_flat_defaults_and_inferred_parallel_start():
    dec, _ = surface(max_iter=121, without_pre=True)
    assert dec.flip_phase_iters == [40]
    assert [phase['flip_mode'] for phase in dec.phase_knobs] == ['lottery', 'pre']
    assert dec.pre_max_lambda == 16 and dec.pre_max_iter == 5
    assert dec.parallel_start == 120 and not dec.parallel_enabled
    dec, _ = surface(max_iter=121, pre_parallel_max_lambda=32)
    assert dec.parallel_start == 120 and dec.pre_it == 121 and dec.parallel_enabled
    dec, _ = surface(max_iter=121, without_pre=True, pre_parallel_max_iter=121, flip_phase_iters=[20], flip_mode=['lottery', 'pre'])
    assert dec.pre_max_lambda == 16 and dec.pre_max_iter == 5
    assert dec.max_lambda == 32 and dec.pre_it == 121 and dec.parallel_start == 100
    dec, _ = surface(max_iter=121, without_pre=True, pre_parallel_start_iter=121)
    assert dec.max_lambda == 32 and dec.pre_it == 121 and dec.parallel_start == 121
    with pytest.raises(ValueError, match='budget end'):
        surface(max_iter=121, flip_phase_iters=[20], flip_mode=['lottery', 'pre'], pre_parallel_start_iter=99)
    with pytest.raises(ValueError, match='not exceed'):
        surface(pre_parallel_start_iter=13)


@pytest.mark.parametrize('algorithm', ['bp_lottery', 'bp_lottery_quant', 'bp_lottery_layered', 'bp_lottery_pre'])
def test_inherited_default_matches_explicit_pre_schedule(algorithm):
    dec, H = surface(algorithm=algorithm, without_pre=True, max_iter=121)
    explicit, _ = surface(algorithm=algorithm, without_pre=True, max_iter=121,
        flip_phase_iters=[40], flip_mode=['lottery', 'pre'], pre_max_lambda=16, pre_max_iter=5)
    assert dec.flip_phase_iters == [40] and dec._pre_enabled
    assert dec.cap is None
    data = inputs(H, shots=8)
    with torch.no_grad():
        got, expected = dec(dict(data)), explicit(dict(data))
    for key in ('e_v', 'llr', 'converge', 'iter'):
        assert torch.equal(got[key], expected[key]), key


@pytest.mark.parametrize('extra', [{'flip_mode': 'lottery'}, {'flip_mode': ['lottery']},
    {'flip_mode': 'off'}, {'flip_mode': ['off']}, {'flip_mode': 'pre'},
    {'flip_mode': ['pre']}, {'flip_phase_iters': []}, {'flip_phase_iters': [20]}])
def test_explicit_phase_or_mode_suppresses_implicit_pre(extra):
    dec, _ = surface(without_pre=True, max_iter=121, **extra)
    assert dec.flip_phase_iters == extra.get('flip_phase_iters', [])
    mode = extra.get('flip_mode', 'lottery')
    assert [p['flip_mode'] for p in dec.phase_knobs] == (
        mode if isinstance(mode, list) else [mode] * len(dec.phase_knobs))


@pytest.mark.parametrize('algorithm', ['bp_lottery', 'bp_lottery_quant', 'bp_lottery_layered', 'bp_lottery_pre'])
@pytest.mark.parametrize('budget', [1, 39, 40])
def test_empty_default_pre_preserves_lottery_outputs_rng_and_cap(algorithm, budget):
    defaults, H = surface(algorithm=algorithm, without_pre=True, max_iter=budget,
        rebatch_opt=True, random_machine='system')
    explicit, _ = surface(algorithm=algorithm, without_pre=True, max_iter=budget,
        rebatch_opt=True, random_machine='system', flip_mode='lottery')
    assert defaults._pre_windows() == {} and not defaults._pre_enabled
    assert defaults.cap is not None and explicit.cap is not None
    defaults._init_pre_rank = lambda: pytest.fail('empty PRE must not rank')
    defaults._rank_pre = lambda *args: pytest.fail('empty PRE must not check')
    if algorithm == 'bp_lottery_pre':
        assert defaults.parallel_start == budget and not defaults.parallel_enabled
    data = inputs(H, shots=8)
    data['synd'][:4].zero_()
    results, states = [], []
    for dec in (defaults, explicit):
        dec.cap.frac = .25
        torch.manual_seed(17)
        with torch.no_grad():
            results.append(dec(dict(data)))
        states.append(torch.get_rng_state())
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        if isinstance(results[0][key], torch.Tensor):
            assert torch.equal(results[0][key], results[1][key]), key
    assert torch.equal(*states)
    assert defaults.cap.frac == explicit.cap.frac
    assert defaults.cap_active_last == explicit.cap_active_last == (budget > 1)
    assert defaults._serial_state is None and defaults.serial is None


def test_bounded_pre_phase_sets_parallel_start_from_complete_windows():
    options = dict(max_iter=121, without_pre=True, flip_phase_iters=[20, 50],
                   flip_mode=['lottery', 'pre', 'lottery'], pre_max_lambda=16,
                   pre_max_iter=5, pre_parallel_max_lambda=16)
    dec, _ = surface(**options)
    assert dec.parallel_start == 50
    dec, _ = surface(**options, pre_parallel_start_iter=60)
    assert dec.parallel_start == 60
    with pytest.raises(ValueError, match='budget end'):
        surface(**options, pre_parallel_start_iter=49)
    dec, _ = surface(**dict(options, flip_phase_iters=[20, 53]))
    assert dec.parallel_start == 50


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
@pytest.mark.parametrize('machine', ['sobol', 'system'])
def test_no_parallel_keys_equals_plain_lottery(backend, machine):
    outputs, states = [], []
    for algorithm in ['bp_lottery', 'bp_lottery_pre']:
        dec, H = surface(backend, algorithm, without_pre=True, random_machine=machine)
        torch.manual_seed(131)
        with torch.no_grad():
            outputs.append(dec(inputs(H)))
        states.append(torch.get_rng_state() if backend == 'cpu' else torch.cuda.get_rng_state())
    for key in ['e_v', 'llr', 'converge', 'iter', 'defer']:
        assert torch.equal(outputs[0][key], outputs[1][key])
    assert torch.equal(*states)


@pytest.mark.parametrize('backend', ['cpu', 'pytorch', 'cuda'])
def test_parallel_start40_hands_off_open_shots(backend):
    dec, H = surface(backend, max_iter=60, flip_mode='lottery',
                     pre_parallel_start_iter=40, pre_parallel_max_iter=5)
    seen = {}
    original = dec._pre_postprocess
    def record(synd, prior, hard, llr, converged, iterations):
        seen['open'] = ~converged.bool().clone()
        seen['iterations'] = iterations.clone()
        return original(synd, prior, hard, llr, converged, iterations)
    dec._pre_postprocess = record
    with torch.no_grad():
        dec(inputs(H, seed=3, shots=64))
    assert seen['open'].any() and (seen['iterations'][seen['open']] == 40).all()


@pytest.mark.parametrize('early', [False, True])
def test_factory_accepts_flat_pre_with_early_termination(early):
    from syndrilla.decoder.decoder import create_decoder
    template, H = surface()
    cfg = dict(algorithm='bp_lottery_pre', device={'device_type': 'cpu'}, dtype='float64', check_type='hx',
        config=dict(max_iter=6, compile=False, rebatch_opt=False, early_termination=early,
                    pre_parallel_max_lambda=3, pre_parallel_max_iter=4))
    bundle = load_matrices(read_yaml('examples/alist/surface_10.matrix.yaml')['matrix'], template.device, template.dtype)
    dec = create_decoder(cfg=cfg, bundle=bundle)[0]
    with torch.no_grad():
        out = dec(inputs(H, shots=12))
    assert (out['iter'] <= 6).all()
    if not early:
        assert (out['iter'] == 6).all()


@pytest.mark.parametrize('algorithm', ['bp_lottery', 'bp_lottery_quant', 'bp_lottery_layered', 'bp_lottery_pre'])
@pytest.mark.parametrize('machine', ['sobol', 'system'])
def test_serial_pre_cpu_pytorch_cuda_agree(algorithm, machine):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    outputs = []
    backends = ['cpu', 'pytorch', 'cuda'] if machine == 'sobol' else ['pytorch', 'cuda']
    for backend in backends:
        dec, H = surface(backend, algorithm, without_pre=True, random_machine=machine,
            flip_phase_iters=[2], flip_mode=['lottery', 'pre'], pre_max_lambda=6, pre_max_iter=1)
        torch.manual_seed(713)
        with torch.no_grad():
            outputs.append(dec(inputs(H, seed=3, shots=64)))
    for output in outputs[1:]:
        for key in ['e_v', 'converge', 'iter']:
            assert torch.equal(outputs[0][key].cpu(), output[key].cpu())


def test_empty_pre_phase_does_not_run_initial_check():
    dec, H = surface(without_pre=True, flip_phase_iters=[0], flip_mode=['pre', 'lottery'])
    with torch.no_grad():
        dec(inputs(H))
    assert dec.last_pre_serial is None and not dec._pre_enabled
    assert dec._serial_state is None and dec.serial is None


@pytest.mark.parametrize('key', ['pre_max_lambda', 'pre_parallel_max_lambda'])
def test_policy_rejects_pre_controls(key):
    message = ('not supported by bp_lottery_policy' if key == 'pre_max_lambda'
               else f'unsupported key {key} for bp_lottery_policy')
    with pytest.raises(ValueError, match=message):
        surface(algorithm='bp_lottery_policy', **{key: 1})


def test_parallel_trials_use_channel_prior_and_shifted_syndrome():
    dec, H = surface(pre_parallel_max_lambda=4, pre_parallel_max_iter=5)
    io = inputs(H, seed=3, shots=64)
    seen = []
    core = dec._bp_core
    def record(syndrome, prior):
        assert torch.equal(prior, io['llr0'][0].expand_as(prior))
        seen.append(syndrome.clone())
        return core(syndrome, prior)
    dec._bp_core = record
    with torch.no_grad():
        out = dec(dict(io))
    assert len(seen) == 1
    assert torch.equal(io['llr0'], inputs(H, seed=3, shots=64)['llr0'])
    mask = out['converge'].bool()
    assert torch.equal((out['e_v'] @ H.T).remainder(2)[mask], io['synd'][mask])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('serial', [False, True])
def test_compiled_pre_matches_eager(serial):
    options = dict(pre_parallel_max_lambda=6, pre_parallel_max_iter=4, pre_parallel_start_iter=12)
    if serial:
        options.update(flip_phase_iters=[2], flip_mode=['lottery', 'pre'], pre_max_lambda=6, pre_max_iter=1)
    outputs = []
    for compiled in [False, True]:
        dec, H = surface('pytorch', compile=compiled, **options)
        if compiled and not dec.compile:
            pytest.skip('torch.compile not available')
        with torch.no_grad():
            outputs.append(dec(inputs(H, seed=3, shots=16)))
    for key in ['e_v', 'llr', 'converge', 'iter', 'iter_serial', 'iter_parallel']:
        assert torch.equal(outputs[0][key], outputs[1][key])
