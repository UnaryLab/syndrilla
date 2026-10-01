"""On a stim rotated surface code circuit (d=5, 5 rounds, undecomposed DEM, so
H has columns with more than 2 checks), the pure-PyTorch bp_norm_min_sum on
CUDA gives the same e_v, llr, iter and converge, bit for bit, as
bp_norm_min_sum_cuda on the per-step path, at float64 and float32. Two runs of
the PyTorch decoder on the same inputs also give identical outputs. Both hold
with the iteration body compiled (`compile: True`)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum_cuda as bp_gpu
from syndrilla.interface.stim.stim import create as stim_iface

DEV = {"device_type": "cuda", "device_idx": 0}
KEYS = ("e_v", "llr", "iter", "converge")
P, B, SEED, MAX_ITER = 0.01, 64, 0, 50
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)


@pytest.fixture(
    scope="module",
    params=[(d, c) for c in (False, True) for d in ("float64", "float32")],
    ids=lambda p: f"{p[0]}-{'compile' if p[1] else 'eager'}",
)
def case(request):
    """(PyTorch decoder, per-step CUDA decoder, synd, llr0) for B circuit-noise
    shots with seed SEED. The PyTorch decoder comes from the stim interface
    with force_pytorch, the CUDA one is built on the same matrix bundle."""
    dtype, compile = request.param
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": MAX_ITER, "compile": compile},
            "dtype": dtype,
            "device": DEV,
            "force_pytorch": True,
        },
    )
    ref = it.decoders[0]
    ref = getattr(ref, "decoder", ref)
    assert ref.compile is compile
    gpu = bp_gpu.create(
        dict(
            device=DEV,
            dtype=dtype,
            check_type="hx",
            max_iter=MAX_ITER,
            force_per_step=True,
        ),
        bundle=it.matrix_bundle,
    )
    assert int(it.matrix_bundle.select("hx")[3].to_dense().sum(0).max()) > 2

    torch.manual_seed(SEED)
    z = torch.zeros(
        B, it.error_model.num_errors, dtype=getattr(torch, dtype), device="cuda"
    )
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    synd = it.syndrome_generator.measure_syndrome(e, None)
    return ref, gpu, synd, llr0


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


def test_cuda_per_step_matches_pytorch(case):
    ref, gpu, synd, llr0 = case
    assert not type(ref).__module__.endswith("_cuda"), type(ref).__module__
    assert not hasattr(ref, "_use_persistent")
    assert type(gpu).__module__.endswith("bp_norm_min_sum_cuda")
    assert gpu._use_persistent(B, False) is False
    assert ref.max_iter == gpu.max_iter == MAX_ITER
    want = _run(ref, synd, llr0)
    converged = int(want["converge"].sum())
    assert 0 < converged < B, converged
    _assert_equal(_run(gpu, synd, llr0), want, "per_step vs pytorch")


def test_pytorch_run_to_run(case):
    ref, _, synd, llr0 = case
    _assert_equal(_run(ref, synd, llr0), _run(ref, synd, llr0), "run 2 vs run 1")


def test_compile_ignored_on_cpu():
    """On CPU `compile: True` runs the eager path: same outputs, no graph compiled."""
    from torch._dynamo.utils import counters

    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum as bp_ref

    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": MAX_ITER, "compile": True},
            "device": {"device_type": "cpu"},
            "force_pytorch": True,
        },
    )
    on = it.decoders[0]
    on = getattr(on, "decoder", on)
    assert on.compile is False
    off = bp_ref.create(
        dict(
            device={"device_type": "cpu"},
            check_type="hx",
            max_iter=MAX_ITER,
            compile=False,
        ),
        bundle=it.matrix_bundle,
    )
    torch.manual_seed(SEED)
    z = torch.zeros(B, it.error_model.num_errors, dtype=torch.float64)
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    synd = it.syndrome_generator.measure_syndrome(e, None)
    graphs = counters["stats"]["unique_graphs"]
    got = _run(on, synd, llr0)
    assert counters["stats"]["unique_graphs"] == graphs
    assert 0 < int(got["converge"].sum()) < B
    _assert_equal(got, _run(off, synd, llr0), "cpu compile flag vs off")


def test_compile_default_on_cuda():
    """Without the `compile` key the PyTorch decoder on CUDA compiles."""
    from torch.utils._triton import has_triton

    if not (torch._dynamo.is_dynamo_supported() and has_triton()):
        pytest.skip("torch.compile or Triton not available")
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": MAX_ITER},
            "device": DEV,
            "force_pytorch": True,
        },
    )
    dec = it.decoders[0]
    assert getattr(dec, "decoder", dec).compile is True


from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum as bp_py

from syndrilla.decoder.knobs import GROUP_OFF

GROUPS = tuple(GROUP_OFF)
MODES = ("cpu", "cuda-eager", "cuda-compile")


@pytest.fixture(scope="module")
def knob_case():
    """(bundle, synd, llr0, {mode: default decoder}) for B float64 circuit-noise
    shots with seed SEED."""
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": MAX_ITER},
            "device": DEV,
            "force_pytorch": True,
        },
    )
    torch.manual_seed(SEED)
    z = torch.zeros(B, it.error_model.num_errors, dtype=torch.float64, device="cuda")
    _, dl = it.error_model.inject_error(z, B)
    e, llr0, _ = next(iter(dl))
    synd = it.syndrome_generator.measure_syndrome(e, None)
    return it.matrix_bundle, synd, llr0, {}


def _py_cfg(mode, **kw):
    dev = {"device_type": "cpu"} if mode == "cpu" else DEV
    if mode == "cuda-eager":
        kw.setdefault("compile", False)
    return dict(device=dev, check_type="hx", max_iter=MAX_ITER, **kw)


def _py_run(case, mode, **kw):
    bundle, synd, llr0, cache = case
    if not kw and mode in cache:
        return cache[mode]
    dec = bp_py.create(_py_cfg(mode, **kw), bundle=bundle)
    if mode == "cpu":
        synd, llr0 = synd.cpu(), llr0.cpu()
    out = _run(dec, synd, llr0), dec
    if not kw:
        cache[mode] = out
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("group", GROUPS)
def test_pytorch_group_off_matches_default(knob_case, group, mode):
    """Setting a group key false gives the default outputs bit for bit on the
    PyTorch decoder, on CPU and on CUDA eager and compiled. gather_opt
    off on CUDA sums with atomic index_add_ in no fixed order: the llr gap grows
    by about 1e-15 per iteration and reaches about 1e-7 relative on rows left
    unconverged at MAX_ITER, so llr there is allclose at rtol 1e-6 and the rest
    equal."""
    ref, on = _py_run(knob_case, mode)
    got, off = _py_run(knob_case, mode, **{group: False})
    assert 0 < int(ref["converge"].sum()) < B
    if mode == "cuda-compile" and group in ("pruning_opt", "memory_opt", "rebatch_opt"):
        assert on.compile is True and off.compile is True
    elif mode == "cuda-compile":
        assert on.compile is True and off.compile is False
    if group == "gather_opt" and mode != "cpu":
        for k in ("e_v", "iter", "converge"):
            assert torch.equal(got[k], ref[k]), k
        assert torch.allclose(got["llr"], ref["llr"], rtol=1e-6, atol=1e-12)
    else:
        _assert_equal(got, ref, f"{group} off vs on ({mode})")


def test_pytorch_knob_couplings():
    """c2v_gather false and cn_sign_parity false turn compile off; a
    subclass that overrides c2v and cn_update (bp_sum_prod) ignores both and
    keeps compile; an explicit member key wins over a false group key;
    compact_frac outside [0, 0.75] raises."""
    from torch.utils._triton import has_triton

    from syndrilla.decoder.bp_sum_prod import bp_sum_prod

    if not (torch._dynamo.is_dynamo_supported() and has_triton()):
        pytest.skip("torch.compile or Triton not available")
    it = stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": 5},
        error_cfg={k: P for k in NOISE},
        syndrome_cfg={"rounds": 5},
        decoding_cfg={
            "algorithm": "bp_norm_min_sum",
            "check_type": "hx",
            "config": {"max_iter": MAX_ITER, "compile": False},
            "device": DEV,
            "force_pytorch": True,
        },
    )
    bundle = it.matrix_bundle

    def dec(cls=bp_py.create, **kw):
        cfg = dict(device=DEV, check_type="hx", max_iter=MAX_ITER, **kw)
        return cls(cfg, bundle=bundle)

    assert dec().compile is True
    d = dec(c2v_gather=False)
    assert d.c2v_gather is False and d.compile is False
    d = dec(cn_sign_parity=False)
    assert d.cn_sign_parity is False and d.compile is False
    d = dec(bp_sum_prod.create, c2v_gather=False, cn_sign_parity=False)
    assert d.c2v_gather is True and d.cn_sign_parity is True and d.compile
    d = dec(fusion_opt=False, compile=True)
    assert d.compile is True
    d = dec(pruning_opt=False, compact_frac=0.5)
    assert d.compact_frac == 0.5
    for bad in (-0.1, 0.8):
        with pytest.raises(ValueError):
            dec(compact_frac=bad)
    d = dec(pruning_opt=False, memory_opt=False, mapping_opt=False)
    assert d.compact_frac == 0 and d.reuse_buffers is False
    assert d.cn_sign_parity is False and d.compile is False


@pytest.mark.parametrize(
    "groups",
    [("mapping_opt",), ("fusion_opt",), ("mapping_opt", "fusion_opt")],
    ids="-".join,
)
def test_cuda_group_off_matches_default(knob_case, groups):
    """mapping_opt false (warp_per_check, f64_int_compare, edge_layout padded)
    and fusion_opt false (persistent_kernel, fuse_vn), alone and together, give
    the default bp_norm_min_sum_cuda outputs bit for bit on the stim circuit,
    where H has columns with more than 2 checks and rows of unequal degree."""
    bundle, synd, llr0, _ = knob_case

    def dec(**kw):
        cfg = dict(device=DEV, check_type="hx", max_iter=MAX_ITER, **kw)
        return bp_gpu.create(cfg, bundle=bundle)

    ref = _run(dec(), synd, llr0)
    assert 0 < int(ref["converge"].sum()) < B
    off = dec(**{g: False for g in groups})
    assert off._use_persistent(B, False) is False
    _assert_equal(_run(off, synd, llr0), ref, f"{groups} off vs on (cuda)")


DEFAULT_BLOCK = {"kl_eps": 1e-4, "kl_window": 3, "kl_min": 3}
# same batch each call: KL falls below kl_eps near call 20, the cap is chosen
# a few calls later and the remaining calls run capped
CALLS = 26


def _calls(dec, synd, llr0):
    outs = []
    for _ in range(CALLS):
        outs.append(_run(dec, synd, llr0))
        outs[-1]["capped"] = dec.cap_active_last
    return outs


def test_rebatch_opt(knob_case):
    """rebatch_opt true without a rebatch_opt_params block equals the default block:
    same warm-up, same chosen cap, same outputs on the capped calls. rebatch_opt
    false builds no cap, with or without a block, and its outputs equal a
    decoder whose cap is None; the warm-up calls of the capped decoder match
    them too."""
    from syndrilla.decoder.decoder import RebatchSpeedup

    bundle, synd, llr0, _ = knob_case

    def dec(**kw):
        return bp_py.create(_py_cfg("cuda-eager", **kw), bundle=bundle)

    on, blk = dec(), dec(rebatch_opt_params=dict(DEFAULT_BLOCK))
    off, off_blk = dec(rebatch_opt=False), dec(
        rebatch_opt=False, rebatch_opt_params=dict(DEFAULT_BLOCK)
    )
    bare = dec(rebatch_opt=False)
    bare.cap = None
    assert isinstance(on.cap, RebatchSpeedup) and isinstance(blk.cap, RebatchSpeedup)
    assert off.cap is None and off_blk.cap is None
    cfg = {"rebatch_opt": False, "rebatch_opt_params": DEFAULT_BLOCK}
    assert RebatchSpeedup.from_cfg(cfg) is None

    got_on, got_blk = _calls(on, synd, llr0), _calls(blk, synd, llr0)
    got_off = _calls(off, synd, llr0)
    want = _calls(bare, synd, llr0)
    assert on.cap.done and on.cap.pct == blk.cap.pct
    assert len(on.cap.hists) == len(blk.cap.hists) < CALLS
    warm = len(on.cap.hists)
    assert [o["capped"] for o in got_on] == [False] * warm + [True] * (CALLS - warm)
    for i in range(CALLS):
        assert got_blk[i]["capped"] == got_on[i]["capped"]
        assert not got_off[i]["capped"]
        _assert_equal(got_blk[i], got_on[i], f"default block vs none, call {i}")
        _assert_equal(got_off[i], want[i], f"rebatch_opt off vs cap None, call {i}")
        if i < warm:
            _assert_equal(got_on[i], want[i], f"warm-up vs cap None, call {i}")


def _main_result(
    tmp_path,
    monkeypatch,
    rebatch_opt,
    rounds,
    measure="phenomenological",
    algorithm="bp_norm_min_sum",
    tb=30,
    bs=100,
    log_out=None,
):
    """Run main() on CPU `algorithm`, surface_5, `measure` noise with `rounds`
    rounds, -bs `bs` -tb `tb`, and return its result yaml and the number of extra
    batches it ran. `log_out`, a list, receives the run log text."""
    import sys

    import yaml

    from syndrilla.main import main

    run = tmp_path / f"run_{rebatch_opt}_{rounds}_{measure}_{algorithm}_{bs}"
    run.mkdir()
    dec = run / "dec.decoding.yaml"
    dec.write_text(
        yaml.safe_dump(
            {
                "decoding": {
                    "algorithm": algorithm,
                    "check_type": "hx",
                    "dtype": "float64",
                    "device": {"device_type": "cpu"},
                    "config": {"max_iter": 181},
                    "rebatch_opt": rebatch_opt,
                }
            }
        )
    )
    syn = run / "phen.syndrome.yaml"
    syn.write_text(
        yaml.safe_dump(
            {
                "syndrome": {
                    "measure": measure,
                    "rounds": rounds,
                    "measurement_error_rate": 0.05,
                }
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "syndrilla",
            f"-r={run}",
            f"-d={dec}",
            "-m=examples/alist/surface_5.matrix.yaml",
            "-e=examples/alist/bsc.error.yaml",
            "-c=examples/alist/lx.check.yaml",
            f"-s={syn}",
            f"-bs={bs}",
            f"-tb={tb}",
        ],
    )
    torch.manual_seed(SEED)
    main()
    (out,) = run.glob("result_phy_err_*.yaml")
    (log,) = run.glob("main-*.log")
    n_extra = log.read_text().count("deferred samples from the extra queue")
    if log_out is not None:
        log_out.append(log.read_text())
    return yaml.safe_load(out.read_text()), n_extra


def test_rebatch_opt_multi_round_matches_off(tmp_path, monkeypatch):
    """With rounds=3 rebatch_opt true gives the same batch count, iterations and
    logical error rate as rebatch_opt false."""
    on, _ = _main_result(tmp_path, monkeypatch, True, 3)
    off, _ = _main_result(tmp_path, monkeypatch, False, 3)
    assert on["decoder_full"]["batch count"] == off["decoder_full"]["batch count"] == 30
    a, b = on["decoder_0"], off["decoder_0"]
    assert a["sample count"] == b["sample count"]
    assert a["average iteration"] == b["average iteration"]
    assert a["hx"]["logical error rate"] == b["hx"]["logical error rate"]


@pytest.mark.parametrize("measure", ["phenomenological", "perfect"])
def test_rebatch_opt_deferred_matches_off(tmp_path, monkeypatch, measure):
    """With rounds=1 the cap defers samples into extra batches, and rebatch_opt
    true gives the same sample count, iterations and logical error rate as false."""
    on, n_on = _main_result(tmp_path, monkeypatch, True, 1, measure, tb=60)
    off, n_off = _main_result(tmp_path, monkeypatch, False, 1, measure, tb=60)
    assert n_on > 0 and n_off == 0
    a, b = on["decoder_0"], off["decoder_0"]
    assert a["sample count"] == b["sample count"] == 6000
    assert a["average iteration"] == b["average iteration"]
    assert a["hx"]["logical error rate"] == b["hx"]["logical error rate"]


def test_rebatch_opt_checkpoint_drains_queue(tmp_path, monkeypatch):
    """Across the periodic save at primary batch 100, the deferred queue is decoded
    before the save, and rebatch_opt true gives the same sample count, iterations
    and logical error rate as false."""
    logs = []
    on, n_on = _main_result(tmp_path, monkeypatch, True, 1, tb=120, bs=10, log_out=logs)
    off, n_off = _main_result(tmp_path, monkeypatch, False, 1, tb=120, bs=10)
    assert n_on > 0 and n_off == 0
    log = logs[0]
    at_100 = log.index("batch count 100/120")
    assert "deferred samples from the extra queue" in log[at_100 : log.index("Save batch log")]
    a, b = on["decoder_0"], off["decoder_0"]
    assert a["sample count"] == b["sample count"] == 1200
    assert a["average iteration"] == b["average iteration"]
    assert a["hx"]["logical error rate"] == b["hx"]["logical error rate"]


def test_rebatch_opt_off_in_training(tmp_path, monkeypatch):
    """A -t run on a bp_norm_min_sum -> saq chain with rebatch_opt true runs the
    first decoder uncapped."""
    import json
    import sys

    import yaml

    import syndrilla.main as main_mod
    from syndrilla.utils import read_yaml

    built = []

    def create_decoder(*a, **kw):
        built.extend(main_mod.create_decoder.__wrapped__(*a, **kw))
        return built

    create_decoder.__wrapped__ = main_mod.create_decoder
    monkeypatch.setattr(main_mod, "create_decoder", create_decoder)
    def plain(path, key):
        return json.loads(json.dumps(read_yaml(path)[key]))

    saq_cfg = plain("examples/alist/train_saq_hx.decoding.yaml", "decoding")["config"]
    dec = tmp_path / "chain.decoding.yaml"
    dec.write_text(
        yaml.safe_dump(
            {
                "decoding": {
                    "algorithm": ["bp_norm_min_sum", "saq"],
                    "check_type": "hx",
                    "dtype": "float32",
                    "device": {"device_type": "cpu"},
                    "config": [{"max_iter": 20}, saq_cfg],
                    "rebatch_opt": True,
                }
            }
        )
    )
    cfg = plain("examples/alist/train_saq_hx.training.yaml", "training")
    cfg["budget"].update(epochs=1, test_batches=2, validation_batches=1)
    tr = tmp_path / "small.training.yaml"
    tr.write_text(yaml.safe_dump({"training": cfg}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "syndrilla",
            "-t",
            f"-r={tmp_path}",
            f"-d={dec}",
            "-m=examples/alist/surface_5.matrix.yaml",
            "-e=examples/alist/bsc_train.error.yaml",
            "-s=examples/alist/perfect.syndrome.yaml",
            f"-tr={tr}",
            "-bs=8",
        ],
    )
    main_mod.main()
    assert built[0].decoder.cap is None
    (log,) = tmp_path.glob("main-*.log")
    assert "rebatch_opt is off during training" in log.read_text()


def test_rebatch_speedup_key_rejected():
    """The old rebatch_speedup key raises and names its replacements."""
    from syndrilla.decoder.decoder import RebatchSpeedup

    with pytest.raises(ValueError, match="rebatch_opt.*rebatch_opt_params"):
        RebatchSpeedup.from_cfg({"rebatch_speedup": True})
