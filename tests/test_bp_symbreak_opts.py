"""SymBreak ablation groups select real graph-loop implementations."""
import pytest
import torch

from syndrilla.decoder.bp_symbreak.bp_symbreak import create as TorchSymBreak
from syndrilla.decoder.bp_symbreak.bp_symbreak_cuda import create as CudaSymBreak
from test_bp_symbreak import tiny

GROUPS = dict(pruning_opt=False, fusion_opt=False, mapping_opt=False,
              gather_opt=False, memory_opt=False, rebatch_opt=False)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bp_iters', [1, 3])
def test_all_groups_off_preserve_outputs_and_member_overrides(device, bp_iters):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    _, bundle, cfg = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=bp_iters, k_max=3)
    cfg['device'] = {'device_type': device}
    cls = CudaSymBreak if device == 'cuda' else TorchSymBreak
    default = cls(cfg, bundle=bundle)
    off = cls(dict(cfg, **{key: value for key, value in GROUPS.items() if key != 'rebatch_opt'}), bundle=bundle)
    overridden = cls(dict(cfg, **{key: value for key, value in GROUPS.items() if key != 'rebatch_opt'},
        compact_frac=.5, reuse_buffers=True, c2v_gather=True, vn_gather=True, cn_sign_parity=True), bundle=bundle)
    assert off.compact_frac == 0 and not off.reuse_buffers and not off.c2v_gather and not off.cn_sign_parity
    assert overridden.compact_frac == .5 and overridden.reuse_buffers and overridden.c2v_gather and overridden.cn_sign_parity
    if device == 'cuda':
        assert not off.vn_gather and overridden.vn_gather
    assert off.cap is None
    io = dict(llr0=torch.full((12, 2), 3.), synd=torch.tensor([[0, 0]] * 10 + [[0, 1]] * 2))
    outputs = [dec({key: value.clone() for key, value in io.items()}) for dec in [default, off, overridden]]
    assert outputs[0]['iter'][:10].tolist() == [bp_iters] * 10
    for out in outputs[1:]:
        for key in ('e_v', 'converge', 'iter', 'splits', 'llr'):
            assert torch.equal(out[key].cpu(), outputs[0][key].cpu()), key


def test_torch_compaction_and_buffer_allocation_paths():
    _, bundle, cfg = tiny(h=[[1, 1], [1, 1]], g=[[1, 1]], bp_iters=2, k_max=3)
    io = dict(llr0=torch.full((12, 2), 3.), synd=torch.tensor([[0, 0]] * 10 + [[0, 1]] * 2))
    for reuse in [True, False]:
        dec = TorchSymBreak(dict(cfg, reuse_buffers=reuse), bundle=bundle)
        allocations, batch_sizes = [], []
        allocate, update = dec._graph_buffers, dec.cn_update
        def buffers(*args):
            allocations.append(args[0]); return allocate(*args)
        def cn(a, syndrome, out):
            batch_sizes.append(len(a)); return update(a, syndrome, out)
        dec._graph_buffers, dec.cn_update = buffers, cn
        dec(dict(io))
        assert min(batch_sizes) == 2 and max(batch_sizes) == 12
        assert len(allocations) == (1 if reuse else len(batch_sizes))


def test_explicit_rebatch_member_wins_group_default():
    _, bundle, cfg = tiny()
    cfg.pop('rebatch_opt')
    assert TorchSymBreak(dict(cfg, rebatch_opt=False), bundle=bundle).cap is None
    assert TorchSymBreak(dict(cfg, rebatch_opt=True), bundle=bundle).cap is not None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_all_groups_off_real_surface(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import math
    from pathlib import Path
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml
    root = Path(__file__).resolve().parents[1]
    cfg = dict(device={'device_type': device}, dtype='float64', check_type='hx',
               max_iter=100, rebatch_opt=False, compile=False)
    bundle = load_matrices(read_yaml(str(root / 'examples/alist/surface_10.matrix.yaml'))['matrix'],
                           *parse_device_dtype(cfg))
    h = bundle.select('hx')[3].to_dense().cpu().double()
    errors = (torch.rand((512, h.shape[1]), generator=torch.Generator().manual_seed(0),
                         dtype=torch.float64) < .01).double()
    io = dict(llr0=torch.full_like(errors, math.log(99)), synd=(errors @ h.T).remainder(2))
    cls = CudaSymBreak if device == 'cuda' else TorchSymBreak
    baseline = cls(cfg, bundle=bundle)({key: value.clone() for key, value in io.items()})
    off = cls(dict(cfg, **{key: value for key, value in GROUPS.items() if key != 'rebatch_opt'}), bundle=bundle)(
        {key: value.clone() for key, value in io.items()})
    for key in ('e_v', 'converge', 'iter', 'splits', 'llr'):
        assert torch.equal(off[key].cpu(), baseline[key].cpu()), key
