"""BB colored partitions, syndrome rules and the unchanged generic decoder."""
import hashlib

import pytest
import torch

from syndrilla.decoder.bp_symbreak.bp_symbreak import create as SymBreak
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml
from test_bp_symbreak import ROOT, cfg, graph, tiny


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bb(check_type='hx', **extra):
    settings = cfg()
    settings.update(check_type=check_type, **extra)
    bundle = load_matrices(read_yaml(str(ROOT / 'examples/alist/bb_144_12_12.matrix.yaml'))['matrix'],
                           *parse_device_dtype(settings))
    return SymBreak(settings, bundle=bundle), bundle, settings


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
@pytest.mark.parametrize('assign', ['bp', 'syndrome'])
def test_bb_split_uses_fixed_colors_and_breaks_overlap(check_type, assign):
    dec, _, _ = bb(check_type, split='bb', assign=assign)
    col, synd, incoming, reset, splits, splittable = graph(dec)
    n, m = dec.H_shape[1], dec.H_shape[0]
    posterior = torch.arange(1, n + 2, dtype=torch.float64)
    posterior[::2] *= -1
    posterior[n] = float('inf')
    means = posterior.abs()[dec.opposite_col].mean(1)
    g = dec.opposite_col[means.argmin()]
    overlap = torch.isin(dec.V_c_col, g).sum(1)
    means = posterior.abs()[dec.V_c_col].mean(1)
    means[(overlap == 0) | (overlap % 2 != 0)] = float('inf')
    c = int(means.argmin())
    parent = col[c].clone()
    synd.zero_()
    synd[c] = 1
    assert dec._split(0, col, synd, posterior, posterior <= 0,
                      incoming, reset, splits, splittable)
    first, second = col[c][col[c] < n], col[m][col[m] < n]
    assert len(first) == len(second) == 3
    assert dec.bb_yellow[first].all() and not dec.bb_yellow[second].any()
    assert torch.equal(torch.cat((first, second)).sort().values, parent.sort().values)
    assert int(torch.isin(first, g).sum()) % 2 == 1
    assert int(torch.isin(second, g).sum()) % 2 == 1
    assert int(synd[c]) ^ int(synd[m]) == 1
    expected = int((posterior[first] <= 0).sum()) % 2 if assign == 'bp' else 0
    assert int(synd[c]) == expected


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
@pytest.mark.parametrize('blocked', [(), ('yellow',), ('blue',), ('yellow', 'blue')])
def test_bb_zero_neighbours_uses_both_sides_of_current_graph(check_type, blocked):
    dec, _, _ = bb(check_type, split='bb', assign='syndrome')
    col, synd, incoming, reset, splits, splittable = graph(dec)
    n, m = dec.H_shape[1], dec.H_shape[0]
    # Equal reliabilities choose opposite row 0, then the first eligible own row.
    g = dec.opposite_col[0]
    overlap = torch.isin(dec.V_c_col, g).sum(1)
    c = int(((overlap > 0) & (overlap % 2 == 0)).nonzero()[0])
    parent = col[c].clone()
    yellow = parent[dec.bb_yellow[parent]]
    blue = parent[~dec.bb_yellow[parent]]
    synd.zero_()
    synd[c] = 1
    # Previously split rows block just one colored side each.
    for offset, color in enumerate(blocked):
        col[m + offset, 0] = (yellow if color == 'yellow' else blue)[0]
        synd[m + offset] = 1
    splits[0] = len(blocked)
    new = m + len(blocked)
    posterior = torch.ones(n + 1, dtype=torch.float64)
    posterior[n] = float('inf')
    before = [value.clone() for value in (col, synd, incoming, reset, splits, splittable)]
    changed = dec._split(0, col, synd, posterior, posterior <= 0,
                         incoming, reset, splits, splittable)
    assert changed == (len(blocked) < 2)
    if not changed:
        for actual, old in zip((col, synd, incoming, reset, splits, splittable), before):
            assert torch.equal(actual, old)
        return
    assert torch.equal(col[c][col[c] < n], yellow)
    assert torch.equal(col[new][col[new] < n], blue)
    assert int(synd[c]) == int('yellow' in blocked)
    assert int(synd[c]) ^ int(synd[new]) == 1


# Recorded before adding split: bb, with the seed and inputs below.
GENERIC_DIGEST = {
    ('hx', 'bp'): '19ce3d27b6e06cdc86112253673a8b07961d86d0eba80754af7f6930a3de68cb',
    ('hx', 'syndrome'): '7ecb4325dc4e8f025552c91006817cc51a0419866c372bf4f3d2a150dd249d6c',
    ('hz', 'bp'): 'fc07847a41b44405637fdd64789adf3d9c8bc1a4389b51593b25a766099a9bfa',
    ('hz', 'syndrome'): 'a8c6317c256d23e159ad0a8bfe87e639932d9ae942fd1078a570fd8f0f62dc89',
}


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
@pytest.mark.parametrize('assign', ['bp', 'syndrome'])
def test_generic_bit_identical_to_before_bb_split(check_type, assign):
    default, bundle, settings = bb(check_type, assign=assign, max_iter=12,
        bp_iters=2, k_max=10, max_splits=3, early_termination=False)
    explicit = SymBreak(dict(settings, split='generic'), bundle=bundle)
    n, m = default.H_shape[1], default.H_shape[0]
    generator = torch.Generator().manual_seed(17)
    llr = torch.rand((4, n), generator=generator, dtype=torch.float64) * 8 - 4
    synd = torch.randint(2, (4, m), generator=generator).double()
    expected = default(dict(llr0=llr, synd=synd))
    actual = explicit(dict(llr0=llr, synd=synd))
    digest = hashlib.sha256()
    for key in ('llr', 'e_v', 'iter', 'converge', 'defer', 'splits'):
        assert torch.equal(actual[key], expected[key]), key
        digest.update(actual[key].numpy().tobytes())
    assert digest.hexdigest() == GENERIC_DIGEST[check_type, assign]


def test_bb_rejects_surface_code_and_invalid_split():
    settings = cfg(split='bb')
    bundle = load_matrices(read_yaml(str(ROOT / 'examples/alist/surface_5.matrix.yaml'))['matrix'],
                           *parse_device_dtype(settings))
    with pytest.raises(ValueError, match='split: bb'):
        SymBreak(settings, bundle=bundle)
    with pytest.raises(ValueError, match='split'):
        tiny(split='unknown')


@pytest.mark.parametrize('check_type', ['hx', 'hz'])
def test_bb_factory_config(check_type):
    _, bundle, _ = bb(check_type, split='bb')
    dec = create_decoder(cfg=dict(algorithm='bp_symbreak', check_type=check_type,
        dtype='float64', device={'device_type': 'cpu'},
        config=dict(split='bb', max_iter=2, rebatch_opt=False)), bundle=bundle)[0]
    assert dec.assign == 'syndrome'
    n, m = dec.H_shape[1], dec.H_shape[0]
    out = dec(dict(llr0=torch.ones(1, n), synd=torch.zeros(1, m)))
    assert out['converge'].tolist() == [1]
    assert out['splits'].tolist() == [0]
