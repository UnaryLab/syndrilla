"""Tests for the adaptive iteration speedup (`iter_speedup`: KL-paced k + offloaded hard tail).

Two kinds of checks:

  * correctness (fast, deterministic) - prove the two guarantees directly on a
    single BP decode: the cap executes strictly fewer iterations (speedup) and
    never changes a decoded result (kept samples match the baseline, and
    re-decoding the deferred tail reconstructs the baseline bit-for-bit). Plus
    the KL warm-up / cap-selection logic on synthetic histograms.

  * timing (end-to-end, opt-in print) - run the SAME problem through main() with
    the cap ON vs OFF and print each wall-clock time. Run with `pytest -s` to see
    the printout, or run the file directly to sweep several setups:
        python tests/test_iter_speedup.py            # uses SETUPS below
        python tests/test_iter_speedup.py 5 0.008    # single (distance, error_rate)
        python tests/test_iter_speedup.py 5:0.008 7:0.008

Output YAMLs are written under the directory named by `SYNDRILLA_TEST_OUTPUT`
(default `tests/test_outputs/iter_speedup`) so they survive after the test.
"""
import os
import re
import sys

import numpy as np
import torch
import pytest

sys.path.append(os.getcwd())

from syndrilla.main import main
from syndrilla.matrix import load_matrices
from syndrilla.decoder import create_decoder
from syndrilla.decoder.decoder import IterSpeedup, _projected_speedup
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import read_yaml, get_path


# ============================================================ correctness tests
# Deterministic, ~1s, no main() loop. These are the actual guarantees.

DEVICE = torch.device('cpu')
DTYPE = torch.float64
MATRIX_YAML = 'examples/alist/surface_10.matrix.yaml'
UNIT_RATE = 0.01     # low enough that BP converges the bulk in ~2 iterations
UNIT_BATCH = 1000
UNIT_MAX_ITER = 50   # high enough that the slow tail keeps the baseline busy
UNIT_FRAC = 0.9      # cap: stop each batch once 90% of samples have converged
UNIT_SEED = 0


def _unit_cfg(iter_speedup=None):
    cfg = {
        'algorithm': 'bp_norm_min_sum', 'check_type': 'hx',
        'max_iter': UNIT_MAX_ITER, 'dtype': 'float64',
        'device': {'device_type': 'cpu', 'device_idx': 0},
    }
    if iter_speedup is not None:
        cfg['iter_speedup'] = iter_speedup
    return cfg


def _build(bundle, iter_speedup=None):
    return create_decoder(cfg=_unit_cfg(iter_speedup), bundle=bundle)[0]


def _decode(dec, synd, llr, H, bypass=False):
    """Run one decode; report the result plus how many BP iterations it took."""
    inner = dec.decoder                      # unwrap the RoundFlattenWrapper
    inner.cap_bypass = bypass
    out = dec({'synd': synd.clone(), 'llr0': llr.clone(), 'H_matrix': H})
    return {
        'e_v': out['e_v'].clone(),
        'converge': out['converge'].clone(),
        'iter': out['iter'].clone(),
        'niter': int(inner.i),               # BP iterations actually executed
        'active': bool(inner.cap_active_last),
    }


def _ready_cap_decoder(bundle, frac=UNIT_FRAC):
    """A capped decoder with warm-up forced complete at a known stop fraction."""
    dec = _build(bundle, iter_speedup={'kl_eps': 1e-2})
    dec.decoder.cap.frac = frac              # `done` is true once frac is set
    dec.decoder.cap.pct = int(frac * 100)
    return dec


@pytest.fixture(scope='module')
def bundle():
    return load_matrices(read_yaml(get_path(MATRIX_YAML))['matrix'], DEVICE, DTYPE)


@pytest.fixture(scope='module')
def fixed_batch(bundle):
    """Deterministic (seeded) error batch + its perfect syndrome and channel LLR."""
    n = bundle.select('hx')[0][1]
    error_model = create_error_model(cfg={
        'model': 'bsc', 'number_channel': 1, 'rate': UNIT_RATE,
        'device': {'device_type': 'cpu', 'device_idx': 0}})
    syndrome_gen = create_syndrome(cfg={'measure': 'perfect'})

    torch.manual_seed(UNIT_SEED)
    _, loader = error_model.inject_error(torch.zeros([UNIT_BATCH, n], dtype=DTYPE), UNIT_BATCH)
    err, llr, _ = next(iter(loader))
    synd = syndrome_gen.measure_syndrome(err, _build(bundle))
    return {'llr': llr, 'synd': synd, 'H': bundle.select('hx')[3]}


def test_no_block_means_no_cap(bundle):
    """Without an `iter_speedup` block the feature is fully off (cap is None)."""
    assert _build(bundle).decoder.cap is None


def test_block_enables_cap(bundle):
    cap = _build(bundle, iter_speedup={'kl_eps': 1e-2, 'kl_window': 2, 'kl_min': 2}).decoder.cap
    assert isinstance(cap, IterSpeedup)
    assert cap.done is False and cap.frac is None


def test_kl_warmup_picks_a_cap_below_100pct():
    """A stationary iteration distribution settles the KL test and a sub-100% cap is chosen."""
    cap = IterSpeedup(kl_eps=1e-2, kl_window=2, kl_min=2)

    def hist_batch(n=UNIT_BATCH):
        t = torch.full((n,), 3, dtype=torch.long)   # 90% converge fast (iter 3)
        t[int(0.9 * n):] = 25                        # 10% form a slow tail (iter 25)
        return t

    for _ in range(6):
        cap.observe(hist_batch(), UNIT_MAX_ITER, UNIT_BATCH)
        if cap.done:
            break

    assert cap.done and cap.pct is not None
    assert 0.0 < cap.frac < 1.0


def test_projected_speedup_prefers_capping_the_tail():
    """Stopping before a heavy tail beats running to 100% (the 1x reference)."""
    hist = np.zeros(UNIT_MAX_ITER + 1)
    hist[3] = 900.0      # bulk converges at iter 3
    hist[25] = 100.0     # heavy tail at iter 25

    assert _projected_speedup([hist], 100, UNIT_BATCH) == pytest.approx(1.0, abs=1e-6)
    assert _projected_speedup([hist], 90, UNIT_BATCH) > 1.5


def test_cap_speeds_up_without_changing_results(bundle, fixed_batch):
    synd, llr, H = fixed_batch['synd'], fixed_batch['llr'], fixed_batch['H']

    base = _decode(_build(bundle), synd, llr, H)
    capped = _decode(_ready_cap_decoder(bundle), synd, llr, H)

    # the cap engaged and left a hard tail for re-decoding
    assert capped['active'] is True
    deferred = capped['converge'] == 0
    assert int(deferred.sum()) > 0

    # SPEEDUP: strictly fewer BP iterations than the uncapped baseline
    assert capped['niter'] < base['niter']

    # NO DIFFERENCE (kept part): samples the cap kept match the baseline exactly
    kept = capped['converge'] > 0
    assert torch.equal(capped['e_v'][kept], base['e_v'][kept])

    # NO DIFFERENCE (final): re-decode the deferred tail uncapped, reassemble,
    # and the full output is identical to the baseline bit-for-bit
    tail = _decode(_build(bundle), synd[deferred], llr[deferred], H)
    final = capped['e_v'].clone()
    final[deferred] = tail['e_v']
    assert torch.equal(final, base['e_v'])


def test_bypass_is_identical_to_baseline(bundle, fixed_batch):
    """A ready cap that main asks to bypass behaves exactly like no cap at all."""
    synd, llr, H = fixed_batch['synd'], fixed_batch['llr'], fixed_batch['H']

    base = _decode(_build(bundle), synd, llr, H)
    byp = _decode(_ready_cap_decoder(bundle), synd, llr, H, bypass=True)

    assert byp['active'] is False
    assert byp['niter'] == base['niter']
    assert torch.equal(byp['e_v'], base['e_v'])
    assert torch.equal(byp['converge'], base['converge'])
    assert torch.equal(byp['iter'], base['iter'])


# ================================================================ timing test
# End-to-end wall-clock via main(), cap ON vs OFF. Edit the SETUP CONSTANTS, or
# override on the command line, e.g.:
#     SYND_DISTANCE=7 SYND_ERROR_RATE=0.008 pytest -s tests/test_iter_speedup.py
#
# The defaults are chosen so the run TERMINATES quickly while still doing enough
# batches for warm-up to complete and the cap to pay off. Two failure modes to
# avoid when changing them: a high error rate hits TARGET_ERRORS in 1-2 batches
# (warm-up never finishes, no speedup), and a very low error rate never reaches
# TARGET_ERRORS at all (main() never returns -> nothing prints).
DISTANCE = int(os.environ.get('SYND_DISTANCE', 10))            # needs surface_{D}.matrix.yaml (have: 3,5,7,9,10,11)
ERROR_RATE = float(os.environ.get('SYND_ERROR_RATE', 0.005))  # BSC physical error rate
DECODER_ALGORITHM = os.environ.get('SYND_DECODER', 'bp_norm_min_sum')
DTYPE_STR = 'float32'
DEVICE_TYPE = 'cuda'
DEVICE_IDX = 0
BATCH_SIZE = 10000
TARGET_ERRORS = 1000

KL_EPS = 1e-3
KL_WINDOW = 2
KL_MIN = 3

MAX_ITER = DISTANCE * (DISTANCE - 1) * 2     # BP max iterations for distance d

OUTPUT_DIR = os.environ.get(
    'SYNDRILLA_TEST_OUTPUT', os.path.join('tests', 'test_outputs', 'iter_speedup'))


def _run(run_dir, decoder_yaml, error_yaml):
    """Run the simulator once via main(); return (decode_seconds, logical_error_rate, n_batches).

    decode_seconds is the simulator's reported decoding time (the `decoder_full`
    'total time (s)' metric) -- decoding ONLY, excluding error generation,
    syndrome measurement, logical check and YAML I/O. That decoding time is the
    only thing the cap affects, so it is the fair quantity to compare cap ON vs
    OFF (wall clock around main() would be dominated by the un-capped overheads).
    It already includes the re-decoding of the deferred hard tail.
    """
    os.makedirs(run_dir, exist_ok=True)
    sys.argv = [
        'syndrilla', f'-r={run_dir}', '-l=SUCCESS',
        f'-d={decoder_yaml}', f'-e={error_yaml}',
        '-c=examples/alist/lx.check.yaml',
        '-s=examples/alist/perfect.syndrome.yaml',
        f'-m=examples/alist/surface_{DISTANCE}.matrix.yaml',
        f'-bs={BATCH_SIZE}', f'-te={TARGET_ERRORS}',
    ]
    main()
    txt = open(os.path.join(run_dir, f'result_phy_err_{ERROR_RATE}.yaml')).read()
    full = txt.split('decoder_full:')[-1]            # the final / overall metrics block
    decode_s = float(re.search(r"total time \(s\):\s*'?([0-9.eE+-]+)'?", full).group(1))
    ler = float(re.search(r'logical error rate:\s*([0-9.eE+-]+)', full).group(1))
    n_batches = int(re.search(r'batch count:\s*(\d+)', full).group(1))
    return decode_s, ler, n_batches


def test_cap_vs_nocap_timing():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    error_yaml = os.path.join(OUTPUT_DIR, 'bsc.error.yaml')
    with open(error_yaml, 'w') as f:
        f.write('error:\n  model: bsc\n  number_channel: 1\n'
                f'  device:\n    device_type: {DEVICE_TYPE}\n    device_idx: {DEVICE_IDX}\n'
                f'  rate: {ERROR_RATE}\n')

    # two decoder configs: identical except one adds the iter_speedup block
    base = (f'decoder:\n  algorithm: {DECODER_ALGORITHM}\n  check_type: hx\n'
            f'  max_iter: {MAX_ITER}\n  dtype: {DTYPE_STR}\n'
            f'  device:\n    device_type: {DEVICE_TYPE}\n    device_idx: {DEVICE_IDX}\n')
    nocap_yaml = os.path.join(OUTPUT_DIR, 'nocap.decoder.yaml')
    with open(nocap_yaml, 'w') as f:
        f.write(base)
    cap_yaml = os.path.join(OUTPUT_DIR, 'cap.decoder.yaml')
    with open(cap_yaml, 'w') as f:
        f.write(base + f'  iter_speedup:\n    kl_eps: {KL_EPS}\n'
                f'    kl_window: {KL_WINDOW}\n    kl_min: {KL_MIN}\n')

    t_off, ler_off, nb_off = _run(os.path.join(OUTPUT_DIR, 'nocap'), nocap_yaml, error_yaml)
    t_on, ler_on, nb_on = _run(os.path.join(OUTPUT_DIR, 'cap'), cap_yaml, error_yaml)

    print(f"\n[iter_speedup]  decoder={DECODER_ALGORITHM}   code distance d={DISTANCE}"
          f"   physical error rate={ERROR_RATE}")
    print(f"  output dir   : {OUTPUT_DIR}")
    print(f"  decode OFF   : {t_off:7.3f}s   batches={nb_off:4d}   LER={ler_off:.4f}")
    print(f"  decode ON    : {t_on:7.3f}s   batches={nb_on:4d}   LER={ler_on:.4f}")
    print(f"  decode speedup: {t_off / t_on:.2f}x   ({t_off - t_on:+.3f}s)")

    # both decode every sample exactly once -> same logical error rate (within noise)
    assert abs(ler_on - ler_off) < 0.03, (ler_off, ler_on)
    # the cap reduces decoding work, so reported decoding time must not be slower
    assert t_on < t_off * 1.10, (t_off, t_on)


if __name__ == '__main__':
    # Sweep one or more (distance, error_rate) setups and print the timing table.
    SETUPS = [(9, 0.001)]
    if len(sys.argv) == 3 and ':' not in sys.argv[1]:
        SETUPS = [(int(sys.argv[1]), float(sys.argv[2]))]
    elif len(sys.argv) > 1:
        SETUPS = [(int(a.split(':')[0]), float(a.split(':')[1])) for a in sys.argv[1:]]

    base_out = OUTPUT_DIR
    for dist, rate in SETUPS:
        DISTANCE = dist
        ERROR_RATE = rate
        MAX_ITER = DISTANCE * (DISTANCE - 1) * 2
        OUTPUT_DIR = os.path.join(base_out, f'd{dist}_p{rate}')
        test_cap_vs_nocap_timing()
