"""bp_norm_min_sum_cuda gives the same e_v, llr, iter and converge, bit for bit,
as the PyTorch bp_norm_min_sum on the same CUDA device, on the per-step and the
persistent path, at float64 and float32. The kernels add the check messages first
and the channel LLR last, the same association as the reference, and every
surface_10 variable has at most 2 checks, so the sums are exact matches."""
import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum as bp_ref
from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = {"device_type": "cuda", "device_idx": 0}
KEYS = ("e_v", "llr", "iter", "converge")


def _cfg(dtype, max_iter, **kw):
    return dict(device=DEV, dtype=dtype, check_type="hx", max_iter=max_iter, **kw)


@pytest.fixture(scope="module")
def surface10():
    """(bundle, synd, llr0) for 32 BSC samples at p=0.03 on surface_10 hx: BP
    converges on about half of them (17 of 32 with seed 0)."""
    mcfg = read_yaml(os.path.join(ROOT, "examples", "alist", "surface_10.matrix.yaml"))[
        "matrix"
    ]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    bundle = load_matrices(mcfg, *parse_device_dtype(_cfg("float64", 1)))
    H = bundle.select("hx")[3].to_dense().cpu().double()
    M, N = H.shape
    g = torch.Generator().manual_seed(0)
    p, B = 0.03, 32
    err = (torch.rand(B, N, generator=g) < p).double()
    synd = ((err @ H.T) % 2).to(torch.uint8)
    llr0 = torch.full((B, N), math.log((1 - p) / p), dtype=torch.float64)
    return bundle, synd, llr0


def _run(dec, synd, llr0):
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].cpu() for k in KEYS}


def _assert_equal(got, ref, label):
    for k in KEYS:
        assert got[k].dtype == ref[k].dtype, (label, k, got[k].dtype, ref[k].dtype)
        diff = (got[k] != ref[k]).reshape(len(ref[k]), -1).any(1)
        assert torch.equal(
            got[k], ref[k]
        ), f"{label}: {k} differs in samples {diff.nonzero().flatten().tolist()}"


def _mixed(res):
    c = res["converge"]
    return bool((c == 1).any()) and bool((c == 0).any())


@pytest.mark.parametrize("per_step", [True, False], ids=["per_step", "persistent"])
def test_float64_matches_reference(surface10, per_step):
    bundle, synd, llr0 = surface10
    cfg = _cfg("float64", 50, force_per_step=per_step)
    dec = bp_gpu.create(cfg, bundle=bundle)
    assert dec._use_persistent(len(synd), False) is (not per_step)
    ref = _run(bp_ref.create(cfg, bundle=bundle), synd, llr0)
    _assert_equal(_run(dec, synd, llr0), ref, f"per_step={per_step}")


def test_per_step_vs_persistent_mixed_convergence(surface10):
    """max_iter=10 leaves part of the batch unconverged, so the per-step path
    (convergence_flag_update marks a sample done, and the kernels skip it from
    then on) and the persistent kernel both keep each converged sample's
    iterate mid-run, and the rest take the last iterate.
    iter is the convergence iteration or max_iter on both paths."""
    bundle, synd, llr0 = surface10
    pers_dec = bp_gpu.create(_cfg("float64", 10), bundle=bundle)
    assert pers_dec._use_persistent(len(synd), False)
    pers = _run(pers_dec, synd, llr0)
    step = _run(
        bp_gpu.create(_cfg("float64", 10, force_per_step=True), bundle=bundle),
        synd,
        llr0,
    )
    assert _mixed(pers), pers["converge"]
    assert bool((pers["iter"][pers["converge"] == 1] < 10).any())  # mid-run snapshot
    _assert_equal(step, pers, "per_step vs persistent")
    ref = _run(bp_ref.create(_cfg("float64", 10), bundle=bundle), synd, llr0)
    _assert_equal(step, ref, "per_step vs reference")


def test_float32_per_step_matches_reference(surface10):
    bundle, synd, llr0 = surface10
    cfg = _cfg("float32", 50, force_per_step=True)
    got = _run(bp_gpu.create(cfg, bundle=bundle), synd, llr0)
    assert got["llr"].dtype == torch.float32
    _assert_equal(got, _run(bp_ref.create(cfg, bundle=bundle), synd, llr0), "float32")


def _stim(dtype, B, max_iter, d=5):
    """(bundle, synd, llr0) for B stim rotated_memory_x circuit-noise shots at
    p=3e-3, distance d, d rounds."""
    from syndrilla.interface.stim.stim import create as stim_iface

    noise = (
        "after_clifford_depolarization",
        "after_reset_flip_probability",
        "before_measure_flip_probability",
        "before_round_data_depolarization",
    )
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": d},
        error_cfg={k: 3e-3 for k in noise},
        syndrome_cfg={"rounds": d},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": max_iter},
            "dtype": dtype,
            "device": DEV,
            "force_pytorch": True,
        },
    )
    torch.manual_seed(0)
    z = torch.zeros(
        B, it.error_model.num_errors, dtype=getattr(torch, dtype), device="cuda"
    )
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    synd = it.syndrome_generator.measure_syndrome(e, None)
    return it.matrix_bundle, synd, llr0


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_capped_persistent_matches_per_step(dtype):
    """With the rebatch cap at frac 0.5, the persistent path stops the batch at the
    same iteration as the per-step host check and gives the same outputs, on 64
    stim rotated_memory_x d=5 circuit-noise shots at p=3e-3."""
    from syndrilla.decoder.decoder import RebatchSpeedup

    B, max_iter = 64, 50
    bundle, synd, llr0 = _stim(dtype, B, max_iter)
    outs = []
    for per_step in (False, True):
        dec = bp_gpu.create(
            _cfg(dtype, max_iter, force_per_step=per_step), bundle=bundle
        )
        dec.cap = RebatchSpeedup()
        dec.cap.frac = 0.5
        assert dec._use_persistent(B, True) is (not per_step)
        outs.append(_run(dec, synd, llr0))
        assert dec.cap_active_last
    pers, step = outs
    unconv = pers["iter"][pers["converge"] == 0]
    assert len(unconv) and bool((unconv < max_iter).all()), pers["iter"]
    _assert_equal(pers, step, "capped persistent vs per_step")


def test_no_coresident_block_falls_back_to_per_step(monkeypatch):
    """When persistent_max_blocks reports 0 (the persistent kernel does not fit a
    block of PERSISTENT_THREADS threads), the decoder takes the per-step path,
    capped or not, and gives the same outputs as force_per_step."""
    B, max_iter = 64, 50
    bundle, synd, llr0 = _stim("float64", B, max_iter)
    ref = _run(
        bp_gpu.create(_cfg("float64", max_iter, force_per_step=True), bundle=bundle),
        synd,
        llr0,
    )
    monkeypatch.setattr(bp_gpu._load_ext(), "persistent_max_blocks", lambda *a: 0)
    dec = bp_gpu.create(_cfg("float64", max_iter), bundle=bundle)
    assert dec._max_coresident == 0
    assert dec._use_persistent(B, False) is False
    assert dec._use_persistent(B, True) is False
    _assert_equal(_run(dec, synd, llr0), ref, "no co-resident block vs per_step")


def test_bp_sf_retries_fall_back_to_per_step(monkeypatch):
    """bp_sf_cuda runs its SF retries on bp_nms_persistent, and on the per-step
    loop when persistent_max_blocks reports 0 or with force_per_step. All three
    give the same outputs on 64 stim rotated_memory_x d=5 shots at p=3e-3, where
    the retries run."""
    from syndrilla.decoder.bp_sf import bp_sf_cuda

    B, max_iter = 64, 181
    bundle, synd, llr0 = _stim("float64", B, max_iter)
    sf = {"topk": 20, "w_min": 0, "w_max": 2, "n_sample": 200}
    ext = bp_gpu._load_ext()
    launch = ext.bp_nms_persistent
    calls = []

    def counted(*a):
        calls.append(1)
        return launch(*a)

    monkeypatch.setattr(ext, "bp_nms_persistent", counted)

    def run(**kw):
        torch.manual_seed(0)
        dec = bp_sf_cuda.create(_cfg("float64", max_iter, sf=sf, **kw), bundle=bundle)
        calls.clear()
        return _run(dec, synd, llr0), len(calls)

    ref, n_ref = run()
    assert n_ref > 0, "no SF retry ran"
    step, n_step = run(force_per_step=True)
    assert n_step == 0
    _assert_equal(step, ref, "bp_sf force_per_step vs persistent retries")
    monkeypatch.setattr(ext, "persistent_max_blocks", lambda *a: 0)
    got, n_got = run()
    assert n_got == 0
    _assert_equal(got, ref, "bp_sf no co-resident block vs persistent retries")
    _assert_equal(got, step, "bp_sf no co-resident block vs force_per_step")


def _assert_close(got, ref, label):
    """e_v, iter and converge equal; llr allclose at rtol 1e-6, atol 1e-12 on the
    converged rows and at rtol 1e-4, atol 1e-4 on the rest, where the order of
    the atomic sums compounds over the iterations (stim d=9 at 50 iterations:
    up to 2e-5 relative and 1.3e-5 absolute)."""
    for k in ("e_v", "iter", "converge"):
        assert torch.equal(got[k], ref[k]), (label, k)
    c = ref["converge"] == 1
    assert torch.allclose(got["llr"][c], ref["llr"][c], rtol=1e-6, atol=1e-12), label
    assert torch.allclose(got["llr"][~c], ref["llr"][~c], rtol=1e-4, atol=1e-4), label


@pytest.mark.parametrize("max_iter", [10, 50])
@pytest.mark.parametrize(
    "group", ["pruning_opt", "fusion_opt", "mapping_opt", "gather_opt"]
)
def test_group_off_matches_default(surface10, group, max_iter):
    """Setting a group key with CUDA members false (pruning_opt:
    host_check_every 0, skip_converged false; fusion_opt: persistent_kernel
    false, fuse_vn false; mapping_opt: warp_per_check false, f64_int_compare
    false, edge_layout padded; gather_opt: vn_gather false) gives the default
    outputs bit for bit, with part of the batch unconverged at max_iter 10.
    gather_opt sums in no fixed order, so its llr is allclose."""
    bundle, synd, llr0 = surface10
    on = bp_gpu.create(_cfg("float64", max_iter), bundle=bundle)
    off = bp_gpu.create(_cfg("float64", max_iter, **{group: False}), bundle=bundle)
    assert on._use_persistent(len(synd), False)
    assert off._use_persistent(len(synd), False) is False
    ref = _run(on, synd, llr0)
    if max_iter == 10:
        assert _mixed(ref), ref["converge"]
    check = _assert_close if group == "gather_opt" else _assert_equal
    check(_run(off, synd, llr0), ref, f"{group} off vs on")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "kw",
    [
        {"warp_per_check": False},
        {"fuse_vn": False},
        {"f64_int_compare": False},
        {"f64_int_compare": False, "persistent_kernel": False},
        {"edge_layout": "padded"},
        {"edge_layout": "padded", "persistent_kernel": False},
    ],
    ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()),
)
def test_kernel_knob_matches_default(surface10, kw, dtype):
    """Each kernel knob off gives the default outputs bit for bit, with part of
    the batch unconverged at max_iter 10. warp_per_check and fuse_vn off run per-step;
    f64_int_compare false and edge_layout padded run on either path."""
    bundle, synd, llr0 = surface10
    B = len(synd)
    ref = _run(bp_gpu.create(_cfg(dtype, 10), bundle=bundle), synd, llr0)
    assert _mixed(ref), ref["converge"]
    dec = bp_gpu.create(_cfg(dtype, 10, **kw), bundle=bundle)
    per_step = "warp_per_check" in kw or "fuse_vn" in kw or "persistent_kernel" in kw
    assert dec._use_persistent(B, False) is not per_step
    _assert_equal(_run(dec, synd, llr0), ref, str(kw))


@pytest.mark.parametrize("case", ["surface10", "stim_d9"])
def test_vn_gather_off_close_to_default(surface10, case):
    """vn_gather false (one atomicAdd per edge, per-step) gives the default e_v,
    iter and converge, and llr allclose (_assert_close), on surface_10 at
    max_iter 10 and on 64 stim d=9 shots at max_iter 50, both with part of the
    batch unconverged."""
    if case == "surface10":
        bundle, synd, llr0 = surface10
        max_iter = 10
    else:
        max_iter = 50
        bundle, synd, llr0 = _stim("float64", 64, max_iter, d=9)
    ref = _run(bp_gpu.create(_cfg("float64", max_iter), bundle=bundle), synd, llr0)
    assert _mixed(ref), ref["converge"]
    dec = bp_gpu.create(_cfg("float64", max_iter, vn_gather=False), bundle=bundle)
    assert dec._use_persistent(len(synd), False) is False
    _assert_close(_run(dec, synd, llr0), ref, case)


@pytest.mark.parametrize(
    "kw",
    [
        {"host_check_every": 1},
        {"host_check_every": 0},
        {"skip_converged": False},
        {"skip_converged": False, "host_check_every": 1},
    ],
    ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()),
)
def test_pruning_knobs_match_per_step(surface10, kw):
    """Each pruning knob alone on the per-step path gives the default per-step
    outputs bit for bit, with part of the batch unconverged at max_iter 10."""
    bundle, synd, llr0 = surface10
    ref = _run(
        bp_gpu.create(_cfg("float64", 10, force_per_step=True), bundle=bundle),
        synd,
        llr0,
    )
    dec = bp_gpu.create(_cfg("float64", 10, force_per_step=True, **kw), bundle=bundle)
    _assert_equal(_run(dec, synd, llr0), ref, str(kw))


def test_persistent_kernel_knob_and_couplings(surface10):
    """persistent_kernel true takes the persistent kernel where auto would not,
    false and its alias force_per_step never take it, an explicit
    persistent_kernel key wins over force_per_step, skip_converged false,
    warp_per_check false, fuse_vn false and vn_gather false force the per-step
    path, persistent_kernel accepts only true, false
    and auto (as bools or case-insensitive strings), and an explicit member key
    wins over a false group key."""
    bundle = surface10[0]

    def dec(**kw):
        d = bp_gpu.create(_cfg("float64", 10, **kw), bundle=bundle)
        d.H_shape = (bp_gpu.PERSISTENT_MAX_M + 1, d.N)
        return d

    assert dec()._use_persistent(1, False) is False
    assert dec(persistent_kernel=True)._use_persistent(1, False) is True
    assert dec(persistent_kernel=True, force_per_step=True)._use_persistent(1, False)
    big = 4 * torch.cuda.get_device_properties(0).multi_processor_count
    for kw in (
        {"persistent_kernel": False},
        {"force_per_step": True},
        {"persistent_kernel": True, "skip_converged": False},
        {"persistent_kernel": True, "warp_per_check": False},
        {"persistent_kernel": True, "fuse_vn": False},
        {"persistent_kernel": True, "vn_gather": False},
        {"persistent_kernel": "false"},
        {"fusion_opt": False},
    ):
        d = dec(**kw)
        d.H_shape = (90, d.N)
        assert d._use_persistent(big, False) is False, kw
    d = dec(fusion_opt=False, persistent_kernel="auto", fuse_vn=True)
    d.H_shape = (90, d.N)
    assert d._use_persistent(big, False) is True
    assert dec(persistent_kernel="True")._use_persistent(1, False) is True
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError):
            dec(persistent_kernel=bad)
    with pytest.raises(ValueError):
        dec(edge_layout="dense")
    d = dec(pruning_opt=False, skip_converged=True, host_check_every=8)
    assert d._skip_converged is True and d._host_check_every == 8
    d = dec(pruning_opt=False)
    assert d._skip_converged is False and d._host_check_every == 0
