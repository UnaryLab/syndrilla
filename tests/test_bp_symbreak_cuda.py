"""CUDA SymBreak parity against both torch devices and deterministic CPU sums."""
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_symbreak.bp_symbreak import create as TorchSymBreak
from syndrilla.decoder.bp_symbreak.bp_symbreak_cuda import create as CudaSymBreak
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml
from test_bp_symbreak import tiny

ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run(dec, io):
    return dec({key: value.clone() for key, value in io.items()})


def exact(actual, expected):
    for key in ('e_v', 'converge', 'iter', 'splits', 'defer', 'llr'):
        assert torch.equal(actual[key].cpu(), expected[key].cpu()), key


def split_state(native, col, synd, incoming, reset, splits, splittable, posterior):
    d = col.shape[1]
    long0 = lambda: torch.zeros(1, dtype=torch.long, device='cuda')
    return [col[None].cuda(), synd[None].cuda(), posterior[None].cuda(), posterior[None].cuda(),
            incoming[None].cuda(), reset[None].cuda(), native._incident[None].clone(),
            (posterior[None] <= 0).byte().cuda(), torch.ones(1, dtype=torch.bool, device='cuda'),
            splits.cuda().clone(), long0(), long0(), torch.ones(1, dtype=torch.long, device='cuda'),
            long0(), torch.full((1,), -1, dtype=torch.long, device='cuda'),
            synd[None, :native.H_shape[0]].cuda().clone(),
            torch.empty((1, d), dtype=torch.bool, device='cuda'), splittable[None].cuda()]


@GPU
@pytest.mark.parametrize('bp_iters', [1, 10])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_no_split_block_boundaries_cpu_torch_cuda_parity(bp_iters, dtype):
    _, bundle, cfg = tiny(bp_iters=bp_iters, max_splits=0, k_max=100)
    cfg.update(dtype=dtype, max_iter=25)
    io = dict(llr0=torch.full((2, 6), 3.), synd=torch.tensor([[0], [1]]))
    expected = run(TorchSymBreak(cfg, bundle=bundle), io)
    for cls in (TorchSymBreak, CudaSymBreak):
        actual = run(cls(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle), io)
        exact(actual, expected)
        assert actual['iter'].tolist() == [bp_iters, 25]
        assert actual['converge'].tolist() == [1, 0]


@GPU
@pytest.mark.parametrize('max_iter', [3, 7])
def test_partial_final_block_cpu_torch_cuda_parity(max_iter):
    _, bundle, cfg = tiny(bp_iters=10, max_splits=0)
    cfg['max_iter'] = max_iter
    io = dict(llr0=torch.full((2, 6), 3.), synd=torch.zeros(2, 1))
    expected = run(TorchSymBreak(cfg, bundle=bundle), io)
    for cls in (TorchSymBreak, CudaSymBreak):
        actual = run(cls(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle), io)
        exact(actual, expected)
        assert actual['iter'].tolist() == [max_iter, max_iter]
        assert actual['converge'].tolist() == [1, 1]


@GPU
@pytest.mark.parametrize('name', ['surface_10', 'bb_144_12_12'])
@pytest.mark.parametrize('p', [.005, .01])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('assign', ['bp', 'syndrome'])
def test_real_code_cpu_torch_cuda_parity(name, p, dtype, assign):
    cfg = dict(device={'device_type': 'cpu'}, dtype=dtype, check_type='hx',
        max_iter=100, bp_iters=10, k_max=3, max_splits=10, assign=assign,
        rebatch_opt=False, compile=False)
    bundle = load_matrices(read_yaml(str(ROOT / f'examples/alist/{name}.matrix.yaml'))['matrix'],
                           *parse_device_dtype(cfg))
    h = bundle.select('hx')[3].to_dense().double()
    errors = (torch.rand((512, h.shape[1]), generator=torch.Generator().manual_seed(0),
                         dtype=torch.float64) < p).double()
    io = dict(llr0=torch.full_like(errors, math.log((1 - p) / p)), synd=(errors @ h.T).remainder(2))
    cpu = TorchSymBreak(cfg, bundle=bundle)
    gpu_cfg = dict(cfg, device={'device_type': 'cuda'})
    torch_gpu = TorchSymBreak(gpu_cfg, bundle=bundle)
    native = CudaSymBreak(gpu_cfg, bundle=bundle)
    torch.cuda.synchronize()
    start = time.perf_counter(); warm = run(native, io); torch.cuda.synchronize()
    warm_wall = time.perf_counter() - start
    start = time.perf_counter(); actual = run(native, io); torch.cuda.synchronize()
    gpu_wall = time.perf_counter() - start
    start = time.perf_counter(); expected = run(cpu, io)
    cpu_wall = time.perf_counter() - start
    start = time.perf_counter(); eager = run(torch_gpu, io); torch.cuda.synchronize()
    eager_wall = time.perf_counter() - start
    counts = {key: dict(never=int((value['converge'] == 0).sum()),
                       splits=int(value['splits'].sum()), iterations=int(value['iter'].sum()))
              for key, value in [('cpu', expected), ('torch_cuda', eager), ('kernel', actual)]}
    print(f'SYMBREAK {name} p={p} dtype={dtype} assign={assign} shots=512 seed=0 counts={counts} '
          f'wall_cpu={cpu_wall:.6f} wall_torch_cuda={eager_wall:.6f} '
          f'wall_kernel_warmup={warm_wall:.6f} wall_kernel={gpu_wall:.6f}')
    exact(actual, warm)
    exact(actual, expected)
    exact(actual, eager)
    assert torch.equal(actual['llr'].cpu(), expected['llr']), 'deterministic CPU LLR parity'
    assert torch.equal(actual['llr'], eager['llr']), 'deterministic torch CUDA LLR parity'
    assert torch.isfinite(actual['llr']).all()
    good = actual['converge'].cpu().bool()
    assert torch.equal((actual['e_v'].cpu().double()[good] @ h.T).remainder(2), io['synd'][good])


@GPU
@pytest.mark.parametrize('assign', ['bp', 'syndrome'])
@pytest.mark.parametrize('early', [True, False])
@pytest.mark.parametrize('cn_sign_parity', [True, False])
def test_tiny_split_full_budget_and_repeated_batch(assign, early, cn_sign_parity):
    cpu, bundle, cfg = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=1,
        k_max=3, assign=assign, early_termination=early, cn_sign_parity=cn_sign_parity)
    cfg['max_iter'] = 15
    cpu = TorchSymBreak(cfg, bundle=bundle)
    native = CudaSymBreak(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle)
    io = dict(llr0=torch.full((2, 2), 3.), synd=torch.tensor([[0, 0], [0, 1]]))
    actual, expected = run(native, io), run(cpu, io)
    exact(actual, expected)
    exact(run(native, io), actual)
    assert torch.equal(actual['llr'].cpu(), expected['llr'])
    assert actual['splits'][0] == 0
    assert bool(actual['splits'][1] > 0) == (assign == 'bp')
    assert int(actual['splits'].max()) <= cpu.H_shape[0]
    assert torch.isfinite(actual['llr']).all()
    if not early:
        assert actual['iter'].tolist() == [15, 15]


@GPU
@pytest.mark.parametrize('check_type', ['hx', 'hz'])
def test_factory_selects_kernel_and_force_pytorch(check_type):
    cfg = dict(algorithm='bp_symbreak', check_type=check_type, dtype='float64',
        device={'device_type': 'cuda'}, config=dict(max_iter=7, bp_iters=2, rebatch_opt=False))
    bundle = load_matrices(read_yaml(str(ROOT / 'examples/alist/surface_5.matrix.yaml'))['matrix'],
                           *parse_device_dtype(cfg))
    native = create_decoder(cfg=cfg, bundle=bundle)[0]
    eager = create_decoder(cfg=dict(cfg, force_pytorch=True), bundle=bundle)[0]
    assert native.decoder.__class__.__module__.endswith('bp_symbreak_cuda')
    assert eager.decoder.__class__.__module__.endswith('bp_symbreak')
    h = bundle.select(check_type)[3]
    io = dict(llr0=torch.full((2, h.shape[1]), 3.), synd=torch.zeros(2, 3, h.shape[0]))
    actual, expected = run(native, io), run(eager, io)
    exact(actual, expected)
    assert actual['splits'].shape == (2, 3)


@GPU
def test_cap_bypass_empty_batch_and_input_validation():
    _, bundle, cfg = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=2)
    native = CudaSymBreak(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle)
    native.cap = SimpleNamespace(done=True, frac=.5)
    io = dict(llr0=torch.full((2, 2), 3.), synd=torch.tensor([[0, 0], [0, 1]]))
    out = run(native, io)
    assert out['iter'].tolist() == [2, 2] and out['defer'].tolist() == [False, True]
    native.cap_bypass = True
    out = run(native, io)
    assert not native.cap_active_last and not out['defer'].any() and out['iter'][1] > 2
    out = native(dict(llr0=torch.empty(0, 2), synd=torch.empty(0, 2)))
    assert out['llr'].shape == (0, 2) and out['splits'].shape == (0,)
    with pytest.raises(ValueError, match='llr0'):
        native(dict(llr0=torch.full((1, 2), float('nan')), synd=torch.zeros(1, 2)))
    with pytest.raises(ValueError, match='match'):
        native(dict(llr0=torch.full((1, 2), 3.), synd=torch.zeros(1, 3)))


@pytest.mark.parametrize('key,value', [('bp_iters', 0), ('k_max', True),
    ('max_splits', -1), ('assign', 'bad'), ('dtype', 'float16'),
    ('check_type', 'bad'), ('early_termination', 1), ('max_iter', 0)])
def test_shared_validation(key, value):
    _, bundle, cfg = tiny()
    cfg[key] = value
    for decoder in (TorchSymBreak, CudaSymBreak):
        with pytest.raises(ValueError, match=key):
            decoder(cfg, bundle=bundle)


@GPU
@pytest.mark.parametrize('assign', ['bp', 'syndrome'])
def test_fig6_kernel_split_graph_messages_and_incident_ids(assign):
    from test_bp_symbreak import graph
    cpu, bundle, cfg = tiny(g=[[1, 0, 1, 1, 0, 1]], assign=assign)
    native = CudaSymBreak(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle)
    col, synd, incoming, reset, splits, splittable = graph(cpu)
    posterior = torch.tensor([10., -60., -15., 5., 50., -20., float('inf')], dtype=torch.float64)
    c, d = col.shape
    state = split_state(native, col, synd, incoming, reset, splits, splittable, posterior)
    cpu._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    native._ext.split(state, native.opposite_col, native.k_max, native.max_splits,
                      assign == 'syndrome', True)
    for index, expected in [(0, col), (1, synd), (4, incoming), (5, reset), (17, splittable)]:
        assert torch.equal(state[index][0].cpu(), expected), index
    assert state[9].tolist() == splits.tolist()
    expected_incident = cpu._graph_incident(col[None])
    expected_incident[expected_incident == c * d] = -1
    assert torch.equal(state[6].cpu(), expected_incident)


@GPU
@pytest.mark.parametrize('neighbour,expected', [
    ([0, 1, 0, 0, 0, 0], [0, 2, 3]),
    ([0, 1, 0, 0, 1, 1], [0, 2, 3]),
    ([1, 1, 1, 1, 1, 1], None),
])
def test_syndrome_kernel_partition_search_and_skip(neighbour, expected):
    from test_bp_symbreak import graph
    cpu, bundle, cfg = tiny(h=[[1] * 6, neighbour], g=[[1, 0, 1, 1, 0, 1]], assign='syndrome')
    native = CudaSymBreak(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle)
    col, synd, incoming, reset, splits, splittable = graph(cpu)
    synd[1] = 1
    posterior = torch.tensor([10., -60., -15., 5., 50., 20., float('inf')], dtype=torch.float64)
    state = split_state(native, col, synd, incoming, reset, splits, splittable, posterior)
    before = [value.clone() for value in state]
    changed = cpu._split(0, col, synd, posterior, posterior <= 0, incoming, reset, splits, splittable)
    native._ext.split(state, native.opposite_col, native.k_max, native.max_splits, True, True)
    assert changed == (expected is not None)
    for index, value in [(0, col), (1, synd), (4, incoming), (5, reset), (17, splittable)]:
        assert torch.equal(state[index][0].cpu(), value), index
    assert state[9].tolist() == splits.tolist()
    if expected is None:
        for index in (0, 1, 4, 5, 6, 9, 17):
            assert torch.equal(state[index], before[index]), index
    else:
        assert state[0][0, 0][state[0][0, 0] < 6].tolist() == expected
        assert state[1][0, [0, 2]].tolist() == [0, 1]


@GPU
def test_kernel_both_children_cannot_split_again():
    from test_bp_symbreak import graph
    cpu, bundle, cfg = tiny(h=[[1] * 8], g=[[1, 1, 1, 1, 0, 0, 0, 0],
                                          [1, 0, 0, 1, 1, 1, 0, 0]])
    native = CudaSymBreak(dict(cfg, device={'device_type': 'cuda'}), bundle=bundle)
    col, synd, incoming, reset, splits, splittable = graph(cpu)
    posterior = torch.tensor([1., 2., 3., 4., 10., 20., 30., 40., float('inf')], dtype=torch.float64)
    state = split_state(native, col, synd, incoming, reset, splits, splittable, posterior)
    native._ext.split(state, native.opposite_col, native.k_max, native.max_splits, False, True)
    assert state[9].tolist() == [1] and not state[17].any()
    assert (state[0] != 8).any(2).sum() == 2 * native.H_shape[0]
    state[3][0] = torch.tensor([1., 100., 100., 1., 1., 1., 100., 100., float('inf')],
                               dtype=torch.float64, device='cuda')
    assert torch.isin(state[0][0], torch.tensor([0, 3, 4, 5], device='cuda')).sum(1).tolist() == [2, 2]
    before = [value.clone() for value in state]
    native._ext.split(state, native.opposite_col, native.k_max, native.max_splits, False, True)
    for index in (0, 1, 4, 5, 6, 9, 17):
        assert torch.equal(state[index], before[index]), index
