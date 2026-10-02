"""Rebatch cap: cap_active_last and io_dict["defer"] on the PyTorch and CUDA
bp_norm_min_sum, OSD-0 skipping deferred rows, capped plus re-decoded output
equal to uncapped output, and the RebatchSpeedup percentile chooser."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum as bp_py
from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu
from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.decoder.osd_0 import osd_0 as osd_py
from syndrilla.decoder.osd_0 import osd_0_cuda as osd_gpu
from syndrilla.interface.stim.stim import create as stim_iface

from test_bp_nms_stim_equiv import DEV, KEYS, MAX_ITER, NOISE, _assert_equal, _py_cfg

B = 64
IMPLS = ("pytorch", "cuda-persistent", "cuda-per_step")


def _stim(d, p):
    """(bundle, synd, llr0, obs_flips, L) for B stim rotated_memory_x circuit-noise
    shots at distance d, d rounds, rate p, seed 0."""
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": d},
        error_cfg={k: p for k in NOISE},
        syndrome_cfg={"rounds": d},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": 1},
            "device": DEV,
            "force_pytorch": True,
        },
    )
    torch.manual_seed(0)
    z = torch.zeros(B, it.error_model.num_errors, dtype=torch.float64, device="cuda")
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    sg = it.syndrome_generator
    synd = sg.measure_syndrome(e, None)
    return it.matrix_bundle, synd, llr0, sg.observable_flips.cpu(), sg._L.double()


@pytest.fixture(scope="module")
def d5():
    """About half of the 64 shots converge within MAX_ITER."""
    return _stim(5, 1e-2)


@pytest.fixture(scope="module")
def d9():
    return _stim(9, 1e-2)


def _bp(impl, bundle, frac=None, declined=None):
    """BP decoder; with `frac` its cap is fixed at that stop fraction, with
    `declined` its cap is done with no fraction."""
    if impl == "pytorch":
        dec = bp_py.create(_py_cfg("cuda-eager"), bundle=bundle)
    else:
        cfg = dict(device=DEV, check_type="hx", max_iter=MAX_ITER)
        cfg["force_per_step"] = impl == "cuda-per_step"
        dec = bp_gpu.create(cfg, bundle=bundle)
    dec.cap = None
    if frac is not None:
        dec.cap = RebatchSpeedup()
        dec.cap.frac = frac
        if impl != "pytorch":
            assert dec._use_persistent(B, True) is (impl == "cuda-persistent")
    if declined is not None:
        dec.cap = RebatchSpeedup()
        dec.cap.declined = declined
        assert dec.cap.done and dec.cap.frac is None
        if impl != "pytorch":
            assert dec._use_persistent(B, False) is (impl == "cuda-persistent")
    return dec


def _decode(dec, synd, llr0):
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].cpu() for k in KEYS + ("defer",)}


@pytest.mark.parametrize("impl", IMPLS)
def test_unreachable_cap_runs_to_max_iter(request, impl):
    """A cap at 0.99 on d=9 p=1e-2 shots, where far fewer than 99% converge, runs
    the batch to max_iter: cap_active_last False, defer all False. _use_persistent
    picks the per-step path for 64 d=9 shots, so the persistent case uses d=5."""
    d = "d5" if impl == "cuda-persistent" else "d9"
    bundle, synd, llr0, _, _ = request.getfixturevalue(d)
    dec = _bp(impl, bundle, frac=0.99)
    out = _decode(dec, synd, llr0)
    unconv = out["converge"] == 0
    assert int(unconv.sum()) > 0.01 * B
    assert bool((out["iter"][unconv] == MAX_ITER).all()), out["iter"]
    assert dec.cap_active_last is False
    assert out["defer"].dtype == torch.bool and not bool(out["defer"].any())


@pytest.mark.parametrize("impl", IMPLS)
def test_declined_cap_matches_uncapped(d5, impl):
    """A declined cap (done, frac None) decodes like no cap, bit for bit:
    cap_active_last False, defer all False."""
    bundle, synd, llr0, _, _ = d5
    dec = _bp(impl, bundle, declined="test reason")
    out = _decode(dec, synd, llr0)
    assert dec.cap_active_last is False and not bool(out["defer"].any())
    _assert_equal(out, _decode(_bp(impl, bundle), synd, llr0), "declined vs none")


@pytest.mark.parametrize("impl", IMPLS)
def test_cap_stops_early_defers_unconverged(d5, impl):
    """A cap at 0.25 stops the batch before max_iter: cap_active_last True and
    defer equals converge == 0."""
    bundle, synd, llr0, _, _ = d5
    dec = _bp(impl, bundle, frac=0.25)
    out = _decode(dec, synd, llr0)
    unconv = out["converge"] == 0
    assert bool(unconv.any()) and bool((out["iter"][unconv] < MAX_ITER).all())
    assert dec.cap_active_last is True
    assert torch.equal(out["defer"], unconv)


def _ler(e_v, obs, L):
    return float(((e_v.double() @ L.T) % 2 != obs).any(1).double().mean())


@pytest.mark.parametrize("impl", IMPLS)
def test_cap_on_matches_off(d5, impl):
    """As main does: drop the deferred rows of a capped batch and re-decode them
    uncapped (cap_bypass). The kept rows equal the uncapped decode bit for bit,
    so do the re-decoded rows, and the LER is the same."""
    bundle, synd, llr0, obs, L = d5
    off = _decode(_bp(impl, bundle), synd, llr0)
    dec = _bp(impl, bundle, frac=0.25)
    on = _decode(dec, synd, llr0)
    defer = on["defer"]
    keep = ~defer
    assert bool(defer.any()) and bool(keep.any())
    _assert_equal(
        {k: on[k][keep] for k in KEYS}, {k: off[k][keep] for k in KEYS}, "kept"
    )
    dec.cap_bypass = True
    dev = synd.device
    redo = _decode(dec, synd[defer.to(dev)], llr0[defer.to(dev)])
    assert dec.cap_active_last is False and not bool(redo["defer"].any())
    _assert_equal(redo, {k: off[k][defer] for k in KEYS}, "re-decoded")
    merged = on["e_v"].clone()
    merged[defer] = redo["e_v"]
    assert _ler(merged, obs, L) == _ler(off["e_v"], obs, L)


@pytest.fixture(scope="module")
def bp_out(d5):
    """Uncapped BP outputs on the d5 shots and a defer mask on every other
    unconverged row."""
    bundle, synd, llr0, _, _ = d5
    with torch.no_grad():
        out = _bp("pytorch", bundle)({"synd": synd.clone(), "llr0": llr0.clone()})
    conv = out["converge"]
    defer = (conv == 0) & (torch.arange(B, device=conv.device) % 2 == 0)
    assert bool(defer.any()) and bool(((conv == 0) & ~defer).any())
    return (
        bundle,
        {k: out[k] for k in ("synd", "llr", "e_v", "iter", "converge")},
        defer,
    )


@pytest.mark.parametrize("skip", [True, False], ids=["skip_conv", "all_rows"])
@pytest.mark.parametrize("osd", [osd_py, osd_gpu], ids=["osd_0", "osd_0_cuda"])
def test_osd_skips_deferred_rows(bp_out, osd, skip):
    """OSD with a defer mask keeps the input e_v, iter and converge on deferred
    rows, passes defer through, and matches the no-mask run on the other rows."""
    bundle, io, defer = bp_out
    dec = osd.create(
        dict(device=DEV, dtype="float64", check_type="hx", osd_skip_converged=skip),
        bundle=bundle,
    )

    def run(**extra):
        with torch.no_grad():
            out = dec({**{k: v.clone() for k, v in io.items()}, **extra})
        return {k: out[k].cpu() for k in ("e_v", "iter", "converge", *extra)}

    ref, got = run(), run(defer=defer.clone())
    d, keep = defer.cpu(), ~defer.cpu()
    assert not torch.equal(ref["e_v"][d], io["e_v"].cpu()[d])  # OSD changes them
    assert torch.equal(got["defer"], d)
    for k in ("e_v", "iter", "converge"):
        assert torch.equal(got[k][d], io[k].cpu().to(got[k].dtype)[d]), k
        assert torch.equal(got[k][keep], ref[k][keep]), k


def _feed(cap, iters, max_iter=50):
    """Observe the same iteration batch until the chooser is done."""
    t = torch.tensor(iters)
    for _ in range(20):
        cap.observe(t, max_iter, len(iters))
        if cap.done:
            return cap
    raise AssertionError("chooser never finished warm-up")


# 40 shots at iter 1, 30 spread over 2..31, 30 at max_iter (never converged)
SKEWED = [1] * 40 + list(range(2, 32)) + [50] * 30


def test_chooser_bounds():
    """The chosen percentile is at least min_pct (default 50) and below the
    converged share; a candidate outside that range is never picked."""
    cap = _feed(RebatchSpeedup(), SKEWED)
    assert cap.frac is not None and 50 <= cap.pct <= 69 and cap.frac == cap.pct / 100
    low = _feed(RebatchSpeedup(candidates=(49,)), SKEWED)
    assert low.frac is None and "no candidate" in low.declined
    assert _feed(RebatchSpeedup(candidates=(49,), min_pct=0), SKEWED).pct == 49
    high = _feed(RebatchSpeedup(candidates=(70,)), SKEWED)  # 70% converged
    assert high.frac is None and "no candidate" in high.declined
    top = _feed(RebatchSpeedup(candidates=(69,)), SKEWED)
    assert top.pct == 69 or "no candidate" not in top.declined


def test_chooser_declines_without_gain():
    """When every shot stops at the same iteration no candidate projects 1.1x,
    so the chooser declines: done, frac and pct None, reason names min_speedup."""
    cap = _feed(RebatchSpeedup(), [10] * 100)
    assert cap.done and cap.frac is None and cap.pct is None
    assert "min_speedup" in cap.declined
