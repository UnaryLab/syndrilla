"""Statistical tests for the bp_sum_prod_sc random bit sources (`_bernoulli`), independent of decoding.

Bits are drawn on the real channel site shape [B, N+1] (N+1 = 42 sites for surface_5 hx), since the
latch/memristor static per-site tensors have that shape. Tolerances are 3 sigma for the sample size;
for time-correlated sources the variance is inflated by the integrated autocorrelation time tau.
Sobol is low-discrepancy, not independent, so the lag and independence checks exclude it.

False-failure budget: every test fixes torch seed 0. There are 79 three-sigma asserts; 8 are on
sobol, whose fixed sequence makes them deterministic, which leaves 71 random ones (two-sided
p = 0.0027 each), plus 2 x 168 Bonferroni 4.5-sigma comparisons (p = 6.8e-6 each). A change in draw
order (new code path, torch version) redraws them all, so the chance of at least one false failure
is 1 - 0.9973^71 * (1 - 6.8e-6)^336 = 0.18. If one fails: rerun that test with a few other seeds and
note the seed; a failure that persists across seeds is a real bug. Widen a tolerance only with a
written derivation of the variance it covers.
"""
import math
import sys, os

import pytest
import torch
from loguru import logger

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.utils import read_yaml, get_path, parse_device_dtype
from syndrilla.matrix import load_matrices


YAML = 'examples/alist/bp_sum_prod_sc_hx.decoding.yaml'
MACHINES = ['system', 'sobol', 'smtj', 'ro', 'latch', 'memristor']
N_CYC = 4000


def _dec(dtype=None, **overrides):
    """Inner bp_sum_prod_sc module (not the round wrapper) built from the hx yaml."""
    cfg = read_yaml(get_path(YAML))['decoding']
    if dtype is not None:
        cfg['dtype'] = dtype
    if overrides.get('random_machine') == 'sobol':
        # sobol reads one sequence value per cycle; room for the longest _draw (8000 cycles)
        cfg['config']['max_iter'] = 8192
    cfg['config'].update(overrides)
    bundle = load_matrices(read_yaml(get_path('examples/alist/surface_5.matrix.yaml'))['matrix'], *parse_device_dtype(cfg))
    dec = create_decoder(cfg=cfg, bundle=bundle)[0].decoder
    _reset(dec)
    return dec


def _reset(dec):
    # the same reset forward() does at its start
    dec._rng_state = {}


def _draw(dec, ps, n=N_CYC, site='channel'):
    """Bits for batch rows at constant p = ps[row], as float64 of shape [n, len(ps), streams]:
    streams = N+1 on the channel site, M*degree (flattened) on the edge site."""
    shape = [dec.H_shape[1] + 1] if site == 'channel' else [dec.H_shape[0], dec.V_c_col.shape[1]]
    p = torch.tensor(ps, dtype=torch.float64).view(-1, *[1] * len(shape)).expand(-1, *shape).to(dec.dtype)
    bits = []
    for t in range(n):
        dec.i = t + 1                                         # the 1-indexed cycle counter forward() sets
        bits.append(dec._bernoulli(p, site))
    return torch.stack(bits).to(torch.float64).flatten(2)


def _acorr(b, k):
    """Lag-k autocorrelation along dim 0, pooled over the other dims (per-series mean removed)."""
    x = b - b.mean(0)
    return ((x[k:] * x[:-k]).sum() / (x * x).sum()).item()


def _tau(b, K=50):
    """Integrated autocorrelation time 1 + 2 sum_{k<=K} r_k, floored at 1 (conservative)."""
    return max(1.0, 1.0 + 2.0 * sum(_acorr(b, k) for k in range(1, K + 1)))


def _mean_tol(b, p):
    # 3 sigma of a mean of b.numel() bits with variance p(1-p) and autocorrelation time tau
    return 3.0 * math.sqrt(max(p * (1 - p), 1e-12) * _tau(b) / b.numel())


def _expected(dec, p):
    """Mean the source should produce: ro quantizes p to floor(p 2^k)/2^k, the others give p."""
    if dec.random_machine == 'ro':
        return math.floor(p * 2 ** dec.ro_bits) / 2 ** dec.ro_bits
    return p


# 1. mean tracks p (also in 9: sobol streams share one value per cycle, so its bits are not
# independent, but its mean error over n points is O(log n / n), far inside the independent-bit
# 3-sigma tolerance)
@pytest.mark.parametrize('machine', MACHINES)
def test_mean_tracks_p(machine):
    torch.manual_seed(0)
    dec = _dec(random_machine=machine, latch_offset_sigma=0.0, memristor_tau_sigma=0.0)
    for p in [0.05, 0.3, 0.5, 0.9]:
        b = _draw(dec, [p])
        target = _expected(dec, p)
        assert abs(b.mean().item() - target) < _mean_tol(b, target), (machine, p)


# Each site's own stream must be Bernoulli(p): per-site means spread only by sampling noise.
# 4 rows x 42 sites = 168 elements per draw, a count with a factor of 8, as real batch sizes have.
# sobol is covered by the sobol tests below: every stream shares one value per cycle by design, so
# pooling over rows and sites does not add independent samples.
@pytest.mark.parametrize('B', [4, 200])
def test_per_site_marginal(B):
    torch.manual_seed(0)
    p = 0.3
    n = 4000 if B == 4 else 200
    b = _draw(_dec(random_machine='system'), [p] * B, n=n)
    site_means = b.mean(dim=(0, 1))                          # per hardware site, pooled over rows and cycles
    noise = math.sqrt(p * (1 - p) / (n * B))
    # the std of 42 per-site means around the binomial noise; 1.5x covers its sampling error
    assert site_means.std().item() < 1.5 * noise, (site_means.std().item(), noise)
    assert abs(b.mean().item() - p) < 3 * noise / math.sqrt(site_means.numel())


def test_sobol_sequence_mapping():
    # sobol_dim selects the 1-indexed dimension of an unscrambled Sobol sequence of 2**ceil(log2(max_iter)) values
    for d in (1, 3, 7):
        dec = _dec(random_machine='sobol', sobol_dim=d, max_iter=2000)
        ref = torch.quasirandom.SobolEngine(dimension=d, scramble=False).draw(2048, dtype=torch.float64)[:, d - 1]
        assert torch.equal(dec.sobol_seq, ref), d
    assert _dec(random_machine='sobol', max_iter=1).sobol_seq.numel() == 1


def test_sobol_shared_across_sites():
    # at one cycle every stream of both sites reads the same value, so equal p gives identical bits
    dec = _dec(random_machine='sobol', sobol_dim=3)
    p_ch = torch.full([2, dec.H_shape[1] + 1], 0.3, dtype=torch.float64)
    p_edge = torch.full([2, dec.H_shape[0], dec.V_c_col.shape[1]], 0.3, dtype=torch.float64)
    seen = set()
    for i in range(1, 17):
        dec.i = i
        bits = torch.cat([dec._bernoulli(p_ch, 'channel').flatten(), dec._bernoulli(p_edge, 'edge').flatten()])
        assert (bits == bits[0]).all(), i
        seen.add(bits[0].item())
    assert seen == {True, False}


def test_sobol_per_site_mean():
    # per-site mean over 2000 cycles within the 3-sigma binomial bound
    p, n = 0.3, 2000
    b = _draw(_dec(random_machine='sobol', max_iter=2000), [p], n=n)[:, 0]
    assert (b.mean(0) - p).abs().max().item() < 3 * math.sqrt(p * (1 - p) / n)


def test_sobol_deterministic_per_decode():
    # two decodes of the 41 weight-1 syndromes give identical output with no reseeding in between;
    # max_iter 2000 lets samples converge at different cycles, so iter and e_v carry the bit stream
    dec = _dec(random_machine='sobol', max_iter=2000)
    H = dec.H_matrix.to(torch.float64)
    e = torch.eye(H.shape[1], dtype=torch.float64)
    io = lambda: {'synd': torch.remainder(e @ H.t(), 2.0), 'llr0': torch.full(e.shape, math.log(9), dtype=torch.float64)}
    torch.manual_seed(0)
    out1 = dec(io())
    torch.rand(100)                                           # move the global RNG
    out2 = dec(io())
    assert all(torch.equal(out1[k], out2[k]) for k in ('e_v', 'iter', 'llr'))


# 2. endpoints: raw p = 0 / 1 give exactly all-0 / all-1 for every machine (latch: ndtri(0) = -inf,
# ndtri(1) = +inf). With p clamped to [eps, 1 - eps] (eps = 1e-12, as forward() clamps the channel
# prior) the expected number of wrong bits in 1000 x 42 draws is below 1e-2, and none are asserted.
# sobol is excluded from the p = eps check: its cycle-1 value is 0.0, so any p > 0 gives bit 1 there.
@pytest.mark.parametrize('machine', MACHINES)
def test_endpoints(machine):
    torch.manual_seed(0)
    dec = _dec(random_machine=machine, latch_offset_sigma=0.5, memristor_tau_sigma=0.5)
    b = _draw(dec, [0.0, 1.0, 1e-12, 1.0 - 1e-12], n=1000)
    assert b[:, 0].max() == 0.0 and b[:, 1].min() == 1.0
    assert b[:, 3].min() == 1.0
    if machine != 'sobol':
        assert b[:, 2].max() == 0.0


# 3. smtj memory: lag-k autocorrelation rho^k, mean p for every rho
@pytest.mark.parametrize('rho', [0.0, 0.5, 0.9])
def test_smtj_memory(rho):
    torch.manual_seed(0)
    dec = _dec(random_machine='smtj', smtj_rho=rho)
    p = 0.3
    b = _draw(dec, [p])[:, 0]
    N = b.numel()
    # Bartlett sigmas for autocorrelation rho^k: var(r1) = (1 - rho^2)/N, var(r2) = (1 + 2rho^2 - 3rho^4)/N.
    # For this binary chain (400 replicas of this exact setup, rho in {0, 0.5, 0.9}) the measured sd is up
    # to 1.17x Bartlett and the estimator has a finite-sample bias of up to 0.73 Bartlett sigma (rho 0.9,
    # lag 2), so a true 3 sigma needs (3 * 1.17 + 0.73) / 3 = 1.41x the Bartlett 3 sigma; 1.45x is used.
    widen = 1.45
    assert abs(_acorr(b, 1) - rho) < widen * 3 * math.sqrt((1 - rho ** 2) / N)
    assert abs(_acorr(b, 2) - rho ** 2) < widen * 3 * math.sqrt((1 + 2 * rho ** 2 - 3 * rho ** 4) / N)
    # mean variance inflated by (1 + rho)/(1 - rho)
    assert abs(b.mean().item() - p) < 3 * math.sqrt(p * (1 - p) * (1 + rho) / (1 - rho) / N)


# 4. ro fair bits (read from the stored phases, which the code thresholds at 1/2)
@pytest.mark.parametrize('nu', [0.0, 0.5])
@pytest.mark.parametrize('q', [0.012, 0.5])
def test_ro_fair_bits(q, nu):
    torch.manual_seed(0)
    dec = _dec(random_machine='ro', ro_q=q, ro_nu=nu)
    p = torch.full([1, dec.H_shape[1] + 1], 0.3, dtype=torch.float64)
    fb = []
    for _ in range(N_CYC):
        dec._bernoulli(p, 'channel')
        fb.append((dec._rng_state['channel'] >= 0.5).to(torch.float64))
    fb = torch.stack(fb)                                    # [n, k, 1, S]
    assert abs(fb.mean().item() - 0.5) < _mean_tol(fb, 0.5)
    r1 = _acorr(fb, 1)
    if q == 0.012:
        # the sign of the lag-1 correlation follows frac(nu): + at nu = 0, - at nu = 1/2
        assert abs(r1) > 0.3 and (r1 > 0) == (nu == 0.0), r1
    else:
        assert abs(r1) < 3 / math.sqrt(fb.numel()), r1


def test_ro_one_bit_resolution():
    torch.manual_seed(0)
    dec = _dec(random_machine='ro', ro_bits=1, ro_q=0.5)
    ps = [0.3, 0.49, 0.5, 0.7, 1.0]
    b = _draw(dec, ps)
    for i, target in enumerate([0.0, 0.0, 0.5, 0.5, 1.0]):
        assert abs(b[:, i].mean().item() - target) < _mean_tol(b[:, i], target) + 1e-12, ps[i]


# 5. latch offset law P(1) = Phi(Phi^-1(p) + d)
def test_latch_offset_law():
    torch.manual_seed(0)
    dec = _dec(random_machine='latch', latch_offset_sigma=0.0)
    S = dec.H_shape[1] + 1
    d = torch.full([1, S], -0.5)
    d[0, 1::2] = 0.5
    dec._static['channel'] = d
    p = 0.3
    b = _draw(dec, [p])[:, 0]
    for off, sl in ((-0.5, slice(0, None, 2)), (0.5, slice(1, None, 2))):
        target = torch.special.ndtr(torch.special.ndtri(torch.tensor(p)) + off).item()
        bs = b[:, sl]                                       # sites sharing offset off, pooled
        assert abs(bs.mean().item() - target) < _mean_tol(bs, target), off


def test_latch_offset_spread():
    torch.manual_seed(0)
    p, sigma = 0.3, 0.1
    b = _draw(_dec(random_machine='latch', latch_offset_sigma=sigma), [p], n=8000)[:, 0]
    x = torch.special.ndtri(torch.tensor(p)).item()
    expected = sigma * math.exp(-x * x / 2) / math.sqrt(2 * math.pi)   # sigma phi(Phi^-1(p))
    spread = b.mean(0).std().item()                         # std of 42 per-site means
    assert expected / 2 < spread < expected * 2, (spread, expected)
    # pooled mean over the 42 sites: E[Phi(x + d)] = Phi(x / sqrt(1 + sigma^2)) for d ~ N(0, sigma), and its
    # sd is sqrt(offset term + binomial term), offset term = (sigma phi(x))^2 / 42 (the site offsets are
    # one draw), binomial term = p(1-p) / (8000 * 42); at p 0.3, sigma 0.1: sd = 0.0054, 3 sigma = 0.016
    center = torch.special.ndtr(torch.tensor(x / math.sqrt(1 + sigma ** 2))).item()
    sd = math.sqrt(expected ** 2 / b.shape[1] + p * (1 - p) / b.numel())
    assert abs(b.mean().item() - center) < 3 * sd, (b.mean().item(), center, 3 * sd)


# 6. memristor spread law
def test_memristor_sigma0_matches_system():
    torch.manual_seed(0)
    p = 0.3
    bm = _draw(_dec(random_machine='memristor', memristor_tau_sigma=0.0), [p])[:, 0]
    bs = _draw(_dec(random_machine='system'), [p])[:, 0]
    N = bm.numel()
    assert abs(bm.mean().item() - bs.mean().item()) < 3 * math.sqrt(2 * p * (1 - p) / N)
    assert abs(_acorr(bm, 1) - _acorr(bs, 1)) < 3 * math.sqrt(2 / N)


def test_memristor_tau_ratio_law():
    torch.manual_seed(0)
    dec = _dec(random_machine='memristor')
    S = dec.H_shape[1] + 1
    r = torch.full([1, S], 0.5)
    r[0, 1::2] = 2.0
    dec._static['channel'] = r
    p = 0.3
    b = _draw(dec, [p])[:, 0]
    for ratio, sl in ((0.5, slice(0, None, 2)), (2.0, slice(1, None, 2))):
        target = 1 - (1 - p) ** ratio
        bs = b[:, sl]
        assert abs(bs.mean().item() - target) < _mean_tol(bs, target), ratio


# 7. static vs dynamic state
@pytest.mark.parametrize('machine', ['latch', 'memristor'])
def test_static_shared_across_batch(machine):
    torch.manual_seed(0)
    dec = _dec(random_machine=machine, latch_offset_sigma=0.5, memristor_tau_sigma=0.5)
    b = _draw(dec, [0.3] * 4)                               # [n, 4, S]
    m = b.mean(0)                                           # per-row, per-site means
    mbar = m.mean(0, keepdim=True)
    # 4.5 sigma: Bonferroni over 4 x 42 comparisons (3 sigma each would expect ~0.45 false failures)
    sig = torch.sqrt(mbar * (1 - mbar) / b.shape[0]).clamp_min(1e-6)
    assert ((m - mbar).abs() < 4.5 * sig).all()
    # the site offset dominates the sampling noise, so rows agree site by site
    assert torch.corrcoef(m)[0, 1:].min() > 0.8


def _forward_once(dec):
    H = dec.H_matrix
    dec({'synd': torch.zeros([2, H.shape[0]], dtype=torch.float64),
         'llr0': torch.full([2, H.shape[1]], 2.0, dtype=torch.float64)})


@pytest.mark.parametrize('machine', ['latch', 'memristor'])
def test_static_unchanged_across_forward(machine):
    torch.manual_seed(0)
    dec = _dec(random_machine=machine, latch_offset_sigma=0.5, memristor_tau_sigma=0.5)
    before = {k: v.clone() for k, v in dec._static.items()}
    _forward_once(dec)
    _forward_once(dec)
    assert all(torch.equal(before[k], dec._static[k]) for k in before)
    assert list(dec._static['channel'].shape) == [1, dec.H_shape[1] + 1]
    assert list(dec._static['edge'].shape) == [1, dec.H_shape[0], dec.V_c_col.shape[1]]


@pytest.mark.parametrize('machine', ['smtj', 'ro'])
def test_dynamic_reset_by_forward(machine):
    torch.manual_seed(0)
    dec = _dec(random_machine=machine)
    _draw(dec, [0.3], n=5)
    old = dec._rng_state['channel']
    seen = []
    orig = dec._bernoulli
    def spy(p, site):
        seen.append(site in dec._rng_state)
        return orig(p, site)
    dec._bernoulli = spy
    _forward_once(dec)
    assert seen and seen[0] is False                        # state was empty at forward's first draw
    assert dec._rng_state['channel'] is not old


# 8. independence across sites (sobol excluded: all sites share one value per cycle by design)
@pytest.mark.parametrize('machine', [m for m in MACHINES if m != 'sobol'])
@pytest.mark.parametrize('site', ['channel', 'edge'])
def test_sites_independent(machine, site):
    torch.manual_seed(0)
    b = _draw(_dec(random_machine=machine), [0.3], n=8000, site=site)[:, 0]   # streams 0 and 1 of the site
    a, c = b[:, 0] - b[:, 0].mean(), b[:, 1] - b[:, 1].mean()
    corr = ((a * c).sum() / torch.sqrt((a * a).sum() * (c * c).sum())).item()
    # Bartlett: var(cross-corr of independent series) = (1 + 2 sum_k ra(k) rc(k)) / n
    tau = max(1.0, 1 + 2 * sum(_acorr(b[:, 0], k) * _acorr(b[:, 1], k) for k in range(1, 51)))
    assert abs(corr) < 3 * math.sqrt(tau / b.shape[0]), corr


# 9. low-precision dtypes
@pytest.mark.parametrize('machine', MACHINES)
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_dtype_mean(dtype, machine):
    torch.manual_seed(0)
    dec = _dec(dtype=dtype, random_machine=machine, latch_offset_sigma=0.0, memristor_tau_sigma=0.0)
    b = _draw(dec, [0.3])[:, 0]
    p = torch.tensor(0.3, dtype=dec.dtype).item()           # p as the decoder dtype stores it
    target = _expected(dec, p)
    # the source must return bool bits: a float return could carry a NaN from the uniform/normal/threshold
    # math into the decoder, and this assert fails on it. A NaN threshold inside a bool comparison gives
    # bit 0, which the mean assert below catches.
    raw = dec._bernoulli(torch.full([1, dec.H_shape[1] + 1], 0.3, dtype=dec.dtype), 'channel')
    assert raw.dtype == torch.bool and list(raw.shape) == [1, dec.H_shape[1] + 1]
    assert abs(b.mean().item() - target) < _mean_tol(b, target), (dtype, machine)


@pytest.mark.parametrize('machine', MACHINES)
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
def test_dtype_endpoint_one(machine, dtype):
    torch.manual_seed(0)
    b = _draw(_dec(dtype=dtype, random_machine=machine), [0.0, 1.0])
    assert b[:, 0].max() == 0.0 and b[:, 1].min() == 1.0


# 10. config validation (loguru warnings captured with a temporary sink)
def _build_capturing(**overrides):
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level='WARNING')
    try:
        dec = _dec(**overrides)
    finally:
        logger.remove(sink)
    return dec, msgs


DEFAULTS = {'max_iter': 50, 'counter_width': 8, 'smtj_rho': 0.5, 'ro_q': 0.012, 'ro_nu': 0.0,
            'ro_bits': 8, 'latch_offset_sigma': 0.1, 'memristor_tau_sigma': 0.0, 'sobol_dim': 1}
OUT_OF_RANGE = {'max_iter': 0, 'counter_width': -1, 'smtj_rho': 1.0, 'ro_q': -0.1,
                'ro_bits': 25, 'latch_offset_sigma': -0.1, 'memristor_tau_sigma': -0.1,
                'sobol_dim': 0}
BAD = [(k, v) for k in DEFAULTS for v in (True, 'abc')] + list(OUT_OF_RANGE.items()) + [('counter_width', 2.5)]


@pytest.mark.parametrize('key,value', BAD)
def test_cfg_fallback(key, value):
    dec, msgs = _build_capturing(**{key: value})
    assert getattr(dec, key) == DEFAULTS[key]
    assert any(key in m for m in msgs), msgs


@pytest.mark.parametrize('value', ['foo', True, 5])
def test_random_machine_fallback(value):
    dec, msgs = _build_capturing(random_machine=value)
    assert dec.random_machine == 'system'
    assert any('machine type' in m for m in msgs), msgs
