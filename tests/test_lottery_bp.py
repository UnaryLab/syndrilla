import os
import subprocess
import sys


sys.path.append(os.getcwd())


def _surface10_system_outputs(mod, extra=None,
                               batches=((0, 16, 0.03), (1, 16, 0.03))):
    """Per batch: e_v, iter, converge, the CPU and CUDA global RNG states after
    decoding, and cap.frac. Each (seed, B, p) in batches is a BSC batch on
    surface_10 hx, decoded in turn by one decoder with random_machine system
    after torch.manual_seed(7); extra updates the decoder config."""
    import math
    import torch
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = dict(device={'device_type': 'cuda', 'device_idx': 0}, dtype='float64',
               check_type='hx', max_iter=100, compile=False, random_machine='system')
    cfg.update(extra or {})
    mcfg = read_yaml('examples/alist/surface_10.matrix.yaml')['matrix']
    bundle = load_matrices(mcfg, *parse_device_dtype(cfg))
    H = bundle.select('hx')[3].to_dense().cpu().double()
    N = H.shape[1]
    torch.manual_seed(7)
    dec = mod.create(dict(cfg), bundle=bundle)
    outs = []
    for seed, B, p in batches:
        g = torch.Generator().manual_seed(seed)
        err = (torch.rand(B, N, generator=g) < p).double()
        synd = ((err @ H.T) % 2).to(torch.uint8)
        llr0 = torch.full((B, N), math.log((1 - p) / p), dtype=torch.float64)
        with torch.no_grad():
            out = dec({'synd': synd, 'llr0': llr0})
        o = {k: out[k].cpu() for k in ('e_v', 'iter', 'converge')}
        o['cpu_rng'] = torch.get_rng_state()
        o['cuda_rng'] = torch.cuda.get_rng_state()
        o['cap_frac'] = None if dec.cap is None else dec.cap.frac
        outs.append(o)
    return outs


def _assert_same(ref, got):
    import torch
    for b, (r, o) in enumerate(zip(ref, got)):
        for k in r:
            same = r[k] == o[k] if k == 'cap_frac' else torch.equal(r[k], o[k])
            assert same, (b, k)


def test_system_pytorch_matches_cuda():
    """With random_machine system and the same seed, the PyTorch and CUDA paths
    give equal outputs and global RNG states on two batches in a row."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    _assert_same(_surface10_system_outputs(bp_lottery),
                 _surface10_system_outputs(bp_lottery_cuda))


def test_system_pytorch_matches_cuda_cap_chosen_on_converged_batch():
    """Same as above past the rebatch cap warm-up: the cap is chosen while
    decoding a fully converged batch and applies to the batches after it."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    extra = dict(max_iter=60, rebatch_opt=True, rebatch_opt_params=dict(
        kl_eps=1e9, kl_window=1, kl_min=2, min_speedup=0.0, min_pct=50))
    batches = ((0, 96, 0.03), (5, 16, 0.01), (3, 48, 0.05),
               (2, 32, 0.2), (5, 16, 0.01), (4, 80, 0.04))
    ref = _surface10_system_outputs(bp_lottery, extra, batches)
    got = _surface10_system_outputs(bp_lottery_cuda, extra, batches)
    assert ref[0]['cap_frac'] is None and ref[1]['cap_frac'] is not None
    assert bool(ref[1]['converge'].all())
    assert int(ref[1]['iter'].max()) > 4
    _assert_same(ref, got)


def test_quant_system_pytorch_matches_cuda():
    """bp_lottery_quant with random_machine system: equal outputs and global
    RNG states on three batches: the first fully converged by iteration 4 (no
    flip, so neither path draws), the second fully converged after iteration 4
    (the last draw), the third not fully converged."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from syndrilla.decoder.bp_lottery_quant import (
        bp_lottery_quant,
        bp_lottery_quant_cuda,
    )

    extra = dict(max_iter=60, rebatch_opt=False, int_width=3, frac_width=4)
    batches = ((11, 8, 0.002), (0, 8, 0.01), (0, 64, 0.03))
    ref = _surface10_system_outputs(bp_lottery_quant, extra, batches)
    got = _surface10_system_outputs(bp_lottery_quant_cuda, extra, batches)
    assert bool(ref[0]['converge'].all()) and int(ref[0]['iter'].max()) <= 4
    assert bool(ref[1]['converge'].all()) and int(ref[1]['iter'].max()) > 4
    assert not bool(ref[2]['converge'].all())
    _assert_same(ref, got)


def test_batch_alist_hz(batch_size=1000, target_error=1000):
    decoding_yaml = 'examples/alist/lottery_bp_hz.decoding.yaml'
    logical_check_yaml = 'examples/alist/lz.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoding_yaml}',
        '-e=examples/alist/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


def test_batch_alist_hz_quant(batch_size=1000, target_error=1000):
    decoding_yaml = 'examples/alist/lottery_bp_quant_hz.decoding.yaml'
    logical_check_yaml = 'examples/alist/lz.check.yaml'
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoding_yaml}',
        '-e=examples/alist/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)


if __name__ == '__main__':
    batch_size = 100000
    target_error = 1000
    test_batch_alist_hz(batch_size, target_error)
    test_batch_alist_hz_quant(batch_size, target_error)
