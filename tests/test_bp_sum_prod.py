import torch
import math
import sys, os
import subprocess

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.utils import read_yaml, get_path, parse_device_dtype
from syndrilla.matrix import load_matrices


# The probability-domain sum-product reference decoder (bp_sum_prod) on the surface_5 code:
# end-to-end smoke tests plus exact recovery of every weight-1 error. The helpers below are
# shared with tests/test_bp_sum_prod_sc.py.


def _run(decoder_yaml, logical_check_yaml, batch_size, target_error):
    cmd = [
        'syndrilla',
        '-r=tests/test_outputs',
        f'-d={decoder_yaml}',
        '-m=examples/alist/surface_5.matrix.yaml',
        '-e=examples/alist/bsc.error.yaml',
        f'-c={logical_check_yaml}',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-bs={batch_size}',
        f'-te={target_error}',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print('STDOUT:\n', result.stdout)
    print('STDERR:\n', result.stderr)

    assert result.returncode == 0, f'syndrilla exited with code {result.returncode}'


# Correctness check (isolated from the Monte-Carlo pipeline): every weight-1 error on the
# surface_5 code must be recovered exactly. A correct BP decoder solves all weight-1 errors;
# this catches real regressions that the end-to-end smoke tests cannot.
def _weight1_exact_match(decoder_yaml, p=0.1, max_iter=None, seed=0):
    torch.manual_seed(seed)
    cfg = read_yaml(get_path(decoder_yaml))['decoding']
    if max_iter is not None:
        cfg.setdefault('config', {})['max_iter'] = max_iter
    matrix_cfg = read_yaml(get_path('examples/alist/surface_5.matrix.yaml'))['matrix']
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(cfg))
    decoder = create_decoder(cfg=cfg, bundle=bundle)[0]
    H = bundle.select(cfg.get('check_type', 'hx'))[3].to(torch.float64)
    N = H.shape[1]

    e_true = torch.eye(N, dtype=torch.float64)                 # one weight-1 error per qubit
    synd = torch.remainder(e_true @ H.t(), 2.0)
    llr0 = torch.full((N, N), math.log((1 - p) / p), dtype=torch.float64)
    out = decoder({'synd': synd.clone(), 'llr0': llr0.clone()})
    e_v = out['e_v'].to(torch.float64)
    return torch.all(e_v == e_true, dim=1).to(torch.float64).mean().item()


def test_bp_sum_prod_alist_hx(batch_size=200, target_error=10):
    _run('examples/alist/bp_sum_prod_hx.decoding.yaml', 'examples/alist/lx.check.yaml', batch_size, target_error)


def test_bp_sum_prod_alist_hz(batch_size=200, target_error=10):
    _run('examples/alist/bp_sum_prod_hz.decoding.yaml', 'examples/alist/lz.check.yaml', batch_size, target_error)


def test_bp_sum_prod_recovers_weight1():
    # deterministic sum-product reference: must recover every weight-1 error
    assert _weight1_exact_match('examples/alist/bp_sum_prod_hx.decoding.yaml') == 1.0


if __name__ == '__main__':
    batch_size = 200
    target_error = 10
    test_bp_sum_prod_alist_hx(batch_size, target_error)
    test_bp_sum_prod_alist_hz(batch_size, target_error)
    test_bp_sum_prod_recovers_weight1()
