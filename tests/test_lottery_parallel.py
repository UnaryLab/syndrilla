"""bp_lottery_parallel: the same seeds give the same outputs, the selection picks
per shot the converged replica with the fewest iterations (select min_iter) or
the smallest llr0 sum over its flipped bits (select min_flip_prior), or the smallest
posterior llr sum over its flipped bits (select min_flip_posterior), or, if none converged,
the replica with the fewest unsatisfied checks, with ties to the lowest index;
the CUDA port matches the PyTorch module, and defer is [B] and true only where
no replica converged. bp_lottery without flip_start_iter matches
flip_start_iter 4. Each accuracy knob (report_agreement,
flip_anneal, flip_tiebreak, stuck_check_weight, flip_temperature, flip_undo,
consensus_every, copy_on_stall, syndrome_flip_last_n, flip_interval,
ensemble_llr) set to
its off value matches it absent, and on it changes the outputs, with the CUDA
port matching the PyTorch module."""
import math
import os

import pytest
import torch
from loguru import logger

from syndrilla.decoder.bp_lottery.bp_lottery import create as lottery_py
from syndrilla.decoder.bp_lottery_parallel import bp_lottery_parallel as par_py
from syndrilla.decoder.bp_lottery_parallel import bp_lottery_parallel_cuda as par_gpu
from syndrilla.decoder.decoder import RebatchSpeedup, create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the llr [8, N] per machine of the outputs with every knob absent, from the
# reference implementation, for test_knobs_absent_frozen
FROZEN_LLR = os.path.join(ROOT, "tests", "data", "lottery_parallel_frozen_llr.pt")
DEV = {"device_type": "cuda", "device_idx": 0}
KEYS = ("e_v", "llr", "iter", "converge", "defer")
SEEDS = [11, 22, 33, 44]
MACHINES = ["sobol", "system"]
SELECTS = ["min_iter", "min_flip_prior", "min_flip_posterior"]


def _cfg(random_machine, seeds=SEEDS, **kw):
    return dict(
        device=DEV,
        dtype="float64",
        check_type="hx",
        max_iter=50,
        compile=False,
        random_machine=random_machine,
        num_parallel=len(seeds),
        seeds=seeds,
        **kw,
    )


@pytest.fixture(scope="module")
def surface10():
    """(bundle, synd, llr0) for 32 BSC samples at p=0.03 on surface_10 hx.
    Every test using it needs CUDA and skips without it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    bundle = _bundle(_cfg("sobol"))
    return (bundle, *_data(bundle, 0.03, 32, 0))


def _bundle(cfg):
    """The surface_10 matrices on cfg's device and dtype."""
    mcfg = read_yaml(os.path.join(ROOT, "examples", "alist", "surface_10.matrix.yaml"))[
        "matrix"
    ]
    for v in mcfg.values():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(ROOT, v["path"])
    return load_matrices(mcfg, *parse_device_dtype(cfg))


def _data(bundle, p, B, seed):
    """(synd, llr0) for B BSC samples at rate p, drawn with torch seed seed."""
    H = bundle.select("hx")[3].to_dense().cpu().double()
    N = H.shape[1]
    g = torch.Generator().manual_seed(seed)
    err = (torch.rand(B, N, generator=g) < p).double()
    synd = ((err @ H.T) % 2).to(torch.uint8)
    llr0 = torch.full((B, N), math.log((1 - p) / p), dtype=torch.float64)
    return synd, llr0


def _run(dec, synd, llr0):
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].cpu() for k in KEYS}


def _assert_equal(got, ref, label):
    for k in KEYS:
        assert torch.equal(got[k], ref[k]), (label, k)


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
@pytest.mark.parametrize("rm", MACHINES)
def test_same_seeds_same_output(surface10, mod, rm):
    bundle, synd, llr0 = surface10
    a = _run(mod.create(_cfg(rm), bundle=bundle), synd, llr0)
    b = _run(mod.create(_cfg(rm), bundle=bundle), synd, llr0)
    _assert_equal(a, b, f"{rm} rerun")


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("rm", MACHINES)
def test_final_iteration_matches_plain_bp(backend, rm):
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum, bp_norm_min_sum_cuda

    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    plain = bp_norm_min_sum_cuda if backend == "cuda" else bp_norm_min_sum
    cfg = {
        **_cfg(rm),
        "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
        "max_iter": 3,
        "flip_start_iter": 2,
    }
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.1, 16, 0)
    out = _run(mod.create(cfg, bundle=bundle), synd, llr0)
    ref = _run(plain.create(cfg, bundle=bundle), synd, llr0)
    assert not bool(out["converge"].all())
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    _assert_equal(out, ref, "no final flip")


def _dense_h(bundle):
    return bundle.select("hx")[3].to_dense().cpu().double()


# all knobs absent, float64 on cpu, 8 shots at p=0.06 (data seed 7): the
# outputs with every knob absent, from the reference implementation, per machine as
# (e_v nonzero indices per shot, converge, iter)
FROZEN = {
    "sobol": (
        [
            [48, 94, 101, 132, 135, 151, 152, 156, 163],
            [12, 18, 66, 73, 82, 87, 106, 130, 142, 152, 173],
            [4, 40, 54, 121, 132, 136, 146, 179],
            [7, 9, 23, 43, 75, 77, 87, 110, 116, 123],
            [4, 28, 29, 43, 59, 78, 80, 91, 115, 116, 134, 144, 151, 169],
            [10, 11, 16, 105, 107, 111, 134, 159],
            [8, 11, 43, 95, 146, 168, 179, 180],
            [12, 32, 35, 37, 50, 54, 69, 70, 77, 106, 107, 119, 120, 134, 135],
        ],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [6, 10, 4, 4, 16, 7, 50, 50],
    ),
    "system": (
        [
            [49, 94, 101, 132, 144, 151, 152, 156, 163],
            [12, 18, 66, 73, 82, 87, 106, 130, 142, 152, 173],
            [4, 40, 54, 121, 132, 136, 146, 179],
            [7, 9, 23, 43, 75, 77, 87, 110, 116, 123],
            [4, 27, 29, 37, 38, 43, 59, 78, 80, 91, 115, 144, 151, 169],
            [10, 11, 16, 105, 107, 111, 134, 159],
            [8, 11, 43, 146, 168, 177, 178, 180],
            [12, 32, 35, 37, 50, 54, 69, 70, 77, 106, 107, 119, 120, 134, 135],
        ],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [6, 11, 6, 4, 17, 5, 7, 50],
    ),
}


@pytest.mark.parametrize("rm", MACHINES)
def test_knobs_absent_frozen(rm):
    """With every knob absent, e_v, converge, iter and LLR magnitudes equal the
    frozen reference outputs; LLR signs agree with e_v. Runs on the CPU, so
    it needs no CUDA."""
    cfg = {**_cfg(rm), "device": {"device_type": "cpu"}}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.06, 8, 7)
    torch.manual_seed(5)
    out = _run(par_py.create(cfg, bundle=bundle), synd, llr0)
    ones, conv, it = FROZEN[rm]
    assert [r.nonzero().flatten().tolist() for r in out["e_v"]] == ones
    assert out["converge"].tolist() == conv
    assert out["iter"].tolist() == it
    # The frozen LLR includes a final sign flip after its hard decision.
    assert torch.equal(out["llr"].abs(), torch.load(FROZEN_LLR)[rm].abs())
    assert torch.equal(out["llr"] <= 0, out["e_v"])


def _warnings(build):
    """The loguru WARNING-and-above text logged while build() runs."""
    lines = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        build()
    finally:
        logger.remove(sink)
    return "".join(lines)


@pytest.mark.parametrize("name", ["lottery_parallel_bp", "lottery_parallel_bposd"])
def test_example_config_no_warning(surface10, name):
    """The example configs load through create_decoder with no unknown-key
    warning."""
    path = os.path.join(ROOT, "examples", "alist", f"{name}_hx.decoding.yaml")
    text = _warnings(lambda: create_decoder(path, bundle=surface10[0]))
    assert "unknown config keys" not in text


def test_unknown_key_warning():
    """A misspelled knob is named in a warning; the top-level osd_0 knobs of
    a [bp_lottery_parallel, osd_0] chain are not."""
    cfg = {**_cfg("sobol"), "device": {"device_type": "cpu"}, "flip_undoo": True}
    text = _warnings(lambda: par_py.create(cfg, bundle=_bundle(cfg)))
    assert "unknown config keys ['flip_undoo']" in text

    chain = {
        "algorithm": ["bp_lottery_parallel", "osd_0"],
        "check_type": "hx",
        "dtype": "float64",
        "device": {"device_type": "cpu"},
        "osd_early_stop": False,
        "workspace_bytes": 1 << 30,
        "memory_opt": True,
        "sparse_h": True,
        "config": {"max_iter": 50, "num_parallel": len(SEEDS), "seeds": SEEDS},
    }
    text = _warnings(lambda: create_decoder(cfg=chain, bundle=_bundle(cfg)))
    assert "unknown config keys" not in text


@pytest.mark.parametrize("sel", SELECTS)
@pytest.mark.parametrize("rm", MACHINES)
@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_selection_rule(backend, rm, sel):
    """The returned row of shot b is replica best[b] of the B * K decode."""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg(rm, select=sel),
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.03, 32, 0)
    dec = mod.create(cfg, bundle=bundle)
    raw = {}

    def base(d):
        raw.update(super(type(dec), dec).forward(d))
        return raw

    with torch.no_grad():
        got = dec._parallel_forward({"synd": synd.clone(), "llr0": llr0.clone()}, base)
    K, B = len(SEEDS), len(synd)
    conv, it = raw["converge"].view(K, B).cpu(), raw["iter"].view(K, B).cpu()
    e_v = raw["e_v"].view(K, B, -1).cpu()
    llr = raw["llr"].view(K, B, -1).cpu().double()
    H = _dense_h(bundle)
    for b in range(B):
        ks = [k for k in range(K) if conv[k, b] == 1]
        if not ks:
            unsat = [int(((H @ e_v[k, b] + synd[b]) % 2).sum()) for k in range(K)]
            best = min(range(K), key=lambda k: (unsat[k], k))
        elif sel == "min_flip_prior":
            best = min(ks, key=lambda k: (float(llr0[b] @ e_v[k, b]), k))
        elif sel == "min_flip_posterior":
            best = min(ks, key=lambda k: (float(llr[k, b] @ e_v[k, b].double()), k))
        else:
            best = min(ks, key=lambda k: (it[k, b], k))
        for key in ("e_v", "llr", "iter", "converge"):
            assert torch.equal(got[key][b], raw[key][best * B + b]), (b, key)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_min_flip_posterior_more_negative_sum_wins(backend, dtype):
    """The flipped-bit posterior sum selects the more negative row, even
    when its iteration count is higher and both prior scores tie."""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol", seeds=[1, 2]), "dtype": dtype,
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "select": "min_flip_posterior"}
    bundle = _bundle(cfg)
    dec = mod.create(cfg, bundle=bundle)
    e_v = torch.zeros(2, dec.H_shape[1], device=dec.device, dtype=dec.dtype)
    e_v[:, 0] = 1
    llr = torch.ones_like(e_v)
    llr[:, 0] = torch.tensor([-2, -8], device=dec.device, dtype=dec.dtype)
    assert torch.equal(llr <= 0, e_v.bool())
    raw = {"e_v": e_v, "llr": llr,
           "iter": torch.tensor([1, 7], device=dec.device),
           "converge": torch.ones(2, device=dec.device),
           "defer": torch.zeros(2, device=dec.device, dtype=torch.bool)}
    io = {"synd": _dense_h(bundle)[:, 0].to(torch.uint8).unsqueeze(0),
          "llr0": torch.ones(1, dec.H_shape[1], dtype=dec.dtype)}
    for select, winner in (("min_iter", 0), ("min_flip_prior", 0), ("min_flip_posterior", 1)):
        dec.select = select
        got = dec._parallel_forward(dict(io), lambda _: raw)
        for key in KEYS:
            assert torch.equal(got[key], raw[key][winner:winner + 1]), (select, key)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ensemble_llr_probability_mean(dtype):
    cfg = {**_cfg("sobol", seeds=[1, 2, 3]), "device": {"device_type": "cpu"},
           "dtype": dtype, "report_agreement": True}
    bundle = _bundle(cfg)
    dec = par_py.create(cfg, bundle=bundle)
    h = _dense_h(bundle).to(getattr(torch, dtype))
    K, B, N = 3, 2, h.shape[1]
    probabilities = torch.tensor([[0.1, 0.6, 0.0, 1.0], [0.3, 0.8, 0.0, 1.0],
                                  [0.4, 0.9, 0.0, 1.0]], dtype=getattr(torch, dtype))
    probabilities = probabilities[:, None].repeat(1, B, (N + 3) // 4)[:, :, :N]
    probabilities[:, 1] = 1 - probabilities[:, 1]
    raw = {
        "e_v": (probabilities >= 0.5).to(h.dtype).reshape(K * B, N),
        "llr": (torch.log(1 - probabilities) - torch.log(probabilities)).reshape(K * B, N),
        "iter": torch.tensor([3, 1, 2]).repeat_interleave(B),
        "converge": torch.ones(K * B, dtype=torch.long),
        "defer": torch.zeros(K * B, dtype=torch.bool),
    }
    io = {"synd": raw["e_v"][:B] @ h.T % 2, "llr0": torch.ones(B, N, dtype=h.dtype)}
    ref = dec._parallel_forward(dict(io), lambda _: raw)
    dec.ensemble_llr = True
    got = dec._parallel_forward(dict(io), lambda _: raw)
    eps = torch.finfo(h.dtype).eps
    mean = probabilities.mean(0).clamp(eps, 1 - eps)
    expected = torch.log((1 - mean) / mean)
    torch.testing.assert_close(got["llr"], expected)
    assert bool(torch.isfinite(got["llr"]).all())
    assert torch.equal(got["llr"] <= 0, got["e_v"])
    for key in (*KEYS, "agreement"):
        if key != "llr":
            assert torch.equal(got[key], ref[key]), key


@pytest.mark.parametrize("sel", SELECTS)
@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_selection_synthetic(backend, sel):
    """Crafted B * K base outputs, K=3, B=6, llr0[b, n] = n + 1, one entry per
    shot: (converge, iter, set bits of e_v) per replica, and the pick under
    min_iter and min_flip_prior; min_flip_posterior picks [2, 2, 2, 1, 2, 1].
    shot 0: iter tie of replicas 1 and 2, all llr0 sums tie
    shot 1: replica 2 has more iterations than replica 1 but a smaller llr0 sum
            (3 against 10), and unconverged replica 0 has sum 0
    shot 2: llr0 sum tie of replicas 1 and 2, replica 2 has fewer iterations
    shot 3: none converged, replicas 1 and 2 satisfy the syndrome (tie)
    shot 4: none converged, only replica 2 satisfies the syndrome
    shot 5: only replica 1 converged"""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol", seeds=[1, 2, 3], select=sel),
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    bundle = _bundle(cfg)
    dec = mod.create(cfg, bundle=bundle)
    H = _dense_h(bundle)
    M, N = H.shape
    K, B, X = 3, 6, [0, 1]
    shots = [
        ([1, 1, 1], [5, 3, 3], [[0, 1], [0, 1], [0, 1]], 1, 0),
        ([0, 1, 1], [1, 4, 6], [[], [9], [0, 1]], 1, 2),
        ([0, 1, 1], [9, 8, 5], [[0], [2, 3], [1, 4]], 2, 1),
        ([0, 0, 0], [50, 50, 50], [[], X, X], 1, 1),
        ([0, 0, 0], [50, 50, 50], [[], [], X], 2, 2),
        ([0, 1, 0], [3, 10, 2], [[], [5], []], 1, 1),
    ]
    e_v = torch.zeros(K * B, N, dtype=torch.float64)
    for b, (_, _, bits, _, _) in enumerate(shots):
        for k in range(K):
            e_v[k * B + b, bits[k]] = 1.0
    x = torch.zeros(N, dtype=torch.float64)
    x[X] = 1.0
    hx = (H @ x) % 2
    assert hx.any()
    synd = torch.zeros(B, M, dtype=torch.uint8)
    synd[3] = synd[4] = hx.to(torch.uint8)
    conv = torch.tensor([s[0] for s in shots], device=dec.device).T.flatten()
    raw = {
        "e_v": e_v.to(dec.device),
        "llr": torch.arange(K * B, device=dec.device).double().unsqueeze(1).repeat(1, N) + 0.5,
        "iter": torch.tensor([s[1] for s in shots], device=dec.device).T.flatten(),
        "converge": conv,
        "defer": conv == 0,
    }
    raw["llr"] *= 1 - 2 * raw["e_v"]
    assert torch.equal(raw["llr"] <= 0, raw["e_v"].bool())
    llr0 = torch.arange(1, N + 1, dtype=torch.float64).repeat(B, 1)
    got = dec._parallel_forward({"synd": synd, "llr0": llr0}, lambda d: dict(raw))
    pick = [s[3] if sel == "min_iter" else s[4] for s in shots]
    if sel == "min_flip_posterior":
        pick = [2, 2, 2, 1, 2, 1]
    idx = torch.tensor([k * B + b for b, k in enumerate(pick)], device=dec.device)
    for key in ("e_v", "llr", "iter", "converge"):
        assert torch.equal(got[key], raw[key][idx]), key
    assert got["defer"].tolist() == [False, False, False, True, True, False]


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("select", ["fewest", "min_llr0", "min_prior", "min_posterior"])
def test_select_bad_value(backend, select):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol", select=select),
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    with pytest.raises(ValueError, match="^select must"):
        mod.create(cfg, bundle=_bundle(cfg))


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_defaults(surface10, mod):
    """num_parallel 8, select min_iter and flip_start_iter 0; a yaml
    flip_start_iter overrides it."""
    bundle, _, _ = surface10
    cfg = {k: v for k, v in _cfg("sobol").items() if k not in ("num_parallel", "seeds")}
    dec = mod.create(cfg, bundle=bundle)
    assert (dec.num_parallel, len(dec.seeds), dec.select) == (8, 8, "min_iter")
    assert dec.flip_start_iter == 0
    assert mod.create({**cfg, "flip_start_iter": 3}, bundle=bundle).flip_start_iter == 3


@pytest.mark.parametrize("rm", MACHINES)
def test_lottery_flip_start_default(surface10, rm):
    """bp_lottery without flip_start_iter gives the outputs of flip_start_iter 4."""
    bundle, synd, llr0 = surface10
    cfg = {k: v for k, v in _cfg(rm).items() if k not in ("num_parallel", "seeds")}
    torch.manual_seed(7)
    a = _run(lottery_py(cfg, bundle=bundle), synd, llr0)
    torch.manual_seed(7)
    b = _run(lottery_py({**cfg, "flip_start_iter": 4}, bundle=bundle), synd, llr0)
    _assert_equal(a, b, rm)


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_k1_shape_contract(surface10, mod):
    """K=1 draws from a scrambled Sobol sequence, not bp_lottery's, so only the
    output shapes and dtypes are checked."""
    bundle, synd, llr0 = surface10
    B, N = llr0.shape
    out = _run(mod.create(_cfg("sobol", seeds=[5]), bundle=bundle), synd, llr0)
    assert out["e_v"].shape == out["llr"].shape == (B, N)
    assert out["iter"].shape == out["converge"].shape == out["defer"].shape == (B,)
    assert out["defer"].dtype == torch.bool


def test_bad_seeds_length(surface10):
    bundle, _, _ = surface10
    with pytest.raises(ValueError):
        par_py.create({**_cfg("sobol"), "num_parallel": 3}, bundle=bundle)


@pytest.mark.parametrize("num_parallel", [2.0, True])
def test_num_parallel_bad_value(surface10, num_parallel):
    bundle, _, _ = surface10
    with pytest.raises(ValueError, match="^num_parallel must"):
        par_py.create({**_cfg("sobol"), "num_parallel": num_parallel}, bundle=bundle)


@pytest.mark.parametrize("sel", SELECTS)
@pytest.mark.parametrize("rm", MACHINES)
@pytest.mark.parametrize("agree_stop", [0, 2])
def test_cuda_matches_pytorch(surface10, rm, sel, agree_stop):
    bundle, synd, llr0 = surface10
    cfg = _cfg(rm, select=sel, agree_stop=agree_stop)
    py = par_py.create(cfg, bundle=bundle)
    gpu = par_gpu.create(cfg, bundle=bundle)
    ref = _run(py, synd, llr0)
    got = _run(gpu, synd, llr0)
    if agree_stop:
        assert bool(py._ag_done.any()) and bool(gpu._ag_done.any())
    for k in ("e_v", "iter", "converge", "defer"):
        assert torch.equal(got[k], ref[k]), (rm, k)
    assert torch.allclose(got["llr"], ref["llr"], rtol=1e-6), rm


@pytest.mark.parametrize("sel", SELECTS)
def test_cuda_matches_pytorch_two_batches(surface10, sel):
    """With random_machine system, the replica generators advance the same on
    both paths, so a second batch (other B and p) after a first batch whose rows
    all converge past flip_start_iter still matches, and both paths leave the
    global CPU and CUDA RNG states equal after each batch."""
    bundle, _, _ = surface10
    cfg = {**_cfg("system", seeds=[11, 22], select=sel), "max_iter": 60}
    batches = [
        _data(bundle, p, B, seed) for p, B, seed in ((0.03, 4, 1), (0.06, 256, 999))
    ]

    def decode_all(mod):
        torch.manual_seed(5)
        dec = mod.create(cfg, bundle=bundle)
        res = []
        for synd, llr0 in batches:
            out = _run(dec, synd, llr0)
            res.append((out, torch.get_rng_state(), torch.cuda.get_rng_state()))
        return res

    py_res, gpu_res = decode_all(par_py), decode_all(par_gpu)
    # the first batch fully converges, the case where a global draw could differ
    assert bool(py_res[0][0]["converge"].all())
    for b, ((ref, ref_cpu, ref_cuda), (got, got_cpu, got_cuda)) in enumerate(
        zip(py_res, gpu_res)
    ):
        for k in ("e_v", "iter", "converge", "defer"):
            assert torch.equal(got[k], ref[k]), (b, k)
        assert torch.allclose(got["llr"], ref["llr"], rtol=1e-6), b
        assert torch.equal(got_cpu, ref_cpu), (b, "cpu rng")
        assert torch.equal(got_cuda, ref_cuda), (b, "cuda rng")


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_defer_with_cap(surface10, mod):
    """With the rebatch cap at frac 0.8 of the B * K rows, defer is [B] and true
    exactly where no replica converged; a shot with some but not all replicas
    converged is not deferred."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.05, 32, 0)
    dec = mod.create(_cfg("sobol"), bundle=bundle)
    dec.cap = RebatchSpeedup()
    dec.cap.frac = 0.8
    raw = {}

    def base(d):
        raw.update(super(type(dec), dec).forward(d))
        return raw

    with torch.no_grad():
        out = dec._parallel_forward({"synd": synd.clone(), "llr0": llr0.clone()}, base)
    assert dec.cap_active_last
    assert out["defer"].shape == (len(synd),) and out["defer"].dtype == torch.bool
    assert torch.equal(out["defer"], out["converge"] == 0)
    conv = raw["converge"].view(len(SEEDS), -1) == 1
    mixed = conv.any(0) & ~conv.all(0)
    assert bool(mixed.any())
    assert not bool(out["defer"][mixed].any())


# accuracy knobs: each off by default; absent equals the off value
KNOB_OFF = {
    "report_agreement": False,
    "agree_stop": 0,
    "ensemble_llr": False,
    "flip_anneal": [1, 1],
    "flip_tiebreak": "llr",
    "stuck_check_weight": False,
    "flip_temperature": None,
    "flip_undo": False,
    "consensus_every": 0,
    "copy_on_stall": 0,
    "syndrome_flip_last_n": 0,
    "flip_interval": 1,
}
KNOB_ON = {
    "ensemble_llr": {"ensemble_llr": True},
    "flip_start_iter": {"flip_start_iter": [3, 0, 10, 0]},
    # set in the test: replica 0 on the machine other than the shared one
    "random_machine": {},
    "flip_anneal": {"flip_anneal": [3, 1]},
    "flip_anneal_per_replica": {"flip_anneal": [[0, 2], [1, 1], [3, 0], [2, 2]]},
    "flip_tiebreak": {"flip_tiebreak": "osc"},
    "stuck_check_weight": {"stuck_check_weight": True},
    "flip_temperature": {"flip_temperature": 1.0},
    "flip_undo": {"flip_undo": True},
    "consensus_every": {"consensus_every": 5, "consensus_llr": 10.0},
    "consensus_llr": {"consensus_every": 5, "consensus_llr": 2.0},
    "consensus_syndrome_flip": {"consensus_every": 5, "syndrome_flip_last_n": 2},
    "copy_on_stall": {"copy_on_stall": 3},
    "syndrome_flip_last_n": {"syndrome_flip_last_n": 2},
    "flip_interval": {"flip_interval": 2},
    "flip_interval_per_replica": {"flip_interval": [1, 2, 3, 4]},
}


def _run_rng(mod, cfg, bundle, synd, llr0, keys=KEYS):
    """Outputs, then the global CPU and CUDA RNG states and the replica
    generator states, after one decode from torch seed 5."""
    torch.manual_seed(5)
    dec = mod.create(cfg, bundle=bundle)
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    out = {k: out[k].cpu() for k in keys}
    gens = torch.stack([g.get_state() for g in dec._gens])
    return out, torch.get_rng_state(), torch.cuda.get_rng_state(), gens


def _same_run(a, b):
    return all(torch.equal(a[0][k], b[0][k]) for k in a[0]) and all(
        torch.equal(x, y) for x, y in zip(a[1:], b[1:])
    )


@pytest.mark.parametrize("key", sorted(KNOB_OFF))
@pytest.mark.parametrize("rm", MACHINES)
@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_knob_off_identical(surface10, mod, rm, key):
    """A knob set to its off value gives the outputs and RNG states of the
    knob absent."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    a = _run_rng(mod, _cfg(rm), bundle, synd, llr0)
    b = _run_rng(mod, {**_cfg(rm), key: KNOB_OFF[key]}, bundle, synd, llr0)
    assert _same_run(a, b), key


@pytest.mark.parametrize("name", sorted(KNOB_ON))
@pytest.mark.parametrize("rm", MACHINES)
def test_knob_changes_output(surface10, rm, name):
    """Each knob on changes the outputs, and the CUDA port matches the PyTorch
    module with it on (outputs and RNG states)."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    on = {**_cfg(rm), **KNOB_ON[name]}
    if name == "random_machine":
        other = MACHINES[1 - MACHINES.index(rm)]
        on["random_machine"] = [other, rm, rm, rm]
    off = _run_rng(par_py, _cfg(rm), bundle, synd, llr0)
    ref = _run_rng(par_py, on, bundle, synd, llr0)
    assert not all(torch.equal(off[0][k], ref[0][k]) for k in KEYS), name
    assert _same_run(_run_rng(par_gpu, on, bundle, synd, llr0), ref), name


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_flip_interval_2_flip_iterations(surface10, mod):
    """With flip_interval 2 (flip_start_iter 0 here), each path flips at
    iterations 1, 3, ..., 17 of max_iter 19 on a batch that does not fully
    converge, and on no other iteration."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.1, 16, 0)
    seen = []

    class Rec(mod.create):
        def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
            seen.append(self.i)
            return super().sign_flip_cn_rand_new(syndrome, s_est, l_v)

    cfg = {**_cfg("sobol"), "max_iter": 19, "flip_interval": 2}
    out = _run(Rec(cfg, bundle=bundle), synd, llr0)
    assert not bool(out["converge"].all())
    assert seen == list(range(1, 19, 2)), seen


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_flip_interval_2_gates_resample_and_undo(surface10, mod):
    """With flip_interval 2, flip_undo, copy_on_stall 3 and flip_anneal [3, 1],
    the copy_on_stall resample and the flip_undo step run only at the flip
    iterations 1, 3, ..., 17, and each copies or undoes at least once."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.1, 16, 0)
    resample, undo = [], []

    class Rec(mod.create):
        _picking = False

        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self._hook_i = i
            return super()._iter_hook(i, l_v, e_v, active, syndrome)

        def _resample(self, rows, active, l_v, e_v, syndrome):
            before = l_v.clone()
            super()._resample(rows, active, l_v, e_v, syndrome)
            resample.append((self._hook_i, not torch.equal(before, l_v)))

        def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
            self._picking = True
            try:
                return super().sign_flip_cn_rand_new(syndrome, s_est, l_v)
            finally:
                self._picking = False

        def _apply_flip(self, l_v, var, mask):
            if not self._picking:  # the flip_undo step
                undo.append((self._hook_i, bool(mask.any())))
            return super()._apply_flip(l_v, var, mask)

    cfg = {
        **_cfg("sobol"),
        "max_iter": 19,
        "flip_interval": 2,
        "flip_undo": True,
        "copy_on_stall": 3,
        "flip_anneal": [3, 1],
    }
    _run(Rec(cfg, bundle=bundle), synd, llr0)
    flip_iters = set(range(1, 19, 2))
    assert {i for i, _ in resample} <= flip_iters, resample
    assert {i for i, _ in undo} <= flip_iters, undo
    assert any(c for _, c in resample), resample
    assert any(u for _, u in undo), undo


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_report_agreement(surface10, mod):
    """report_agreement adds agreement [B], the converged replicas whose e_v
    equals the returned one (at least 1 on a converged shot, 0 otherwise), and
    leaves the other outputs unchanged."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    keys = KEYS + ("agreement",)
    a = _run_rng(mod, _cfg("sobol"), bundle, synd, llr0)
    b = _run_rng(
        mod, {**_cfg("sobol"), "report_agreement": True}, bundle, synd, llr0, keys
    )
    assert all(torch.equal(a[0][k], b[0][k]) for k in KEYS)
    agree, conv = b[0]["agreement"], b[0]["converge"] == 1
    assert agree.shape == (len(synd),)
    assert bool((agree[conv] >= 1).all()) and bool((agree[~conv] == 0).all())
    assert int(agree.max()) <= len(SEEDS) and bool((agree[conv] < len(SEEDS)).any())


@pytest.mark.parametrize(
    "bad",
    [
        {"report_agreement": "false"},
        {"ensemble_llr": "false"},
        {"ensemble_llr": 1},
        {"ensemble_llr": None},
        {"flip_anneal": [1, 2, 3]},
        {"flip_anneal": 3},
        {"flip_anneal": [[1, 2, 3], [1, 2], [1, 2], [1, 2]]},
        {"flip_anneal": [1, "x"]},
        {"flip_tiebreak": "dice"},
        {"stuck_check_weight": 1},
        {"flip_temperature": 0},
        {"flip_temperature": "1"},
        {"flip_undo": "yes"},
        {"consensus_every": -1},
        {"consensus_every": 2.0},
        {"consensus_every": 5, "consensus_llr": -1.0},
        {"consensus_every": 5, "consensus_llr": "x"},
        {"consensus_llr": 5.0},
        {"copy_on_stall": True},
        {"copy_on_stall": 1.5},
        {"syndrome_flip_last_n": len(SEEDS) + 1},
    ],
)
def test_knob_bad_value(surface10, bad):
    bundle, _, _ = surface10
    with pytest.raises(ValueError):
        par_py.create({**_cfg("sobol"), **bad}, bundle=bundle)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("flip_start_iter", value)
        for value in (-3, 2.0, True, "3", None, {}, (0, 1, 2, 3), [], [0] * 3,
                  [0] * 5, [-1, 0, 0, 0], [False, 0, 0, 0], [2.0, 0, 0, 0],
                  ["3", 0, 0, 0], [None, 0, 0, 0], [[0], 0, 0, 0])
    ] + [
        ("flip_interval", value)
        for value in (0, -1, 2.0, True, False, "2", None, {}, (1, 2, 3, 4), [],
                  [1] * 3, [1] * 5, [0, 1, 1, 1], [-1, 1, 1, 1],
                  [True, 1, 1, 1], [False, 1, 1, 1], [2.0, 1, 1, 1],
                  ["2", 1, 1, 1], [None, 1, 1, 1], [[1], 1, 1, 1])
    ] + [
        ("agree_stop", value)
        for value in (-1, len(SEEDS) + 1, 2.0, True, False, None)
    ] + [
        ("random_machine", value)
        for value in ("dice", 1, 1.0, True, None, {}, ("sobol",) * 4, [],
                  ["sobol"] * 3, ["sobol"] * 5, ["dice", "sobol", "sobol", "sobol"],
                  [False, "sobol", "sobol", "sobol"], [1.0, "sobol", "sobol", "sobol"],
                  [None, "sobol", "sobol", "sobol"], [["sobol"], "sobol", "sobol", "sobol"])
    ],
)
def test_replica_config_bad_value(backend, key, value):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol"), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           key: value}
    with pytest.raises(ValueError, match=f"^{key} must"):
        mod.create(cfg, bundle=_bundle(cfg))


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("value", [None, [], [{}] * len(SEEDS)])
def test_obsolete_key_rejected(backend, value):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol"), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "replica_knobs": value}
    with pytest.raises(ValueError, match="^replica_knobs"):
        mod.create(cfg, bundle=_bundle(cfg))


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
@pytest.mark.parametrize("value", [False, 0, None])
def test_obsolete_syndrome_flip_key_rejected(mod, value):
    cfg = {**_cfg("sobol"), "syndrome_flip_replicas": value,
           "syndrome_flip_last_n": 2}
    with pytest.raises(
        ValueError,
        match=r"^syndrome_flip_replicas is no longer supported; use syndrome_flip_last_n\.$",
    ):
        mod.create(cfg)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("rm", MACHINES)
@pytest.mark.parametrize("start", [0, 3])
@pytest.mark.parametrize("interval", [1, 2, 3])
def test_scalar_matches_replica_copies(backend, rm, start, interval):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg(rm), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "flip_start_iter": start, "flip_interval": interval}
    bundle = _bundle(cfg)
    scalar = mod.create(cfg, bundle=bundle)
    copies = {**cfg, "flip_start_iter": [start] * len(SEEDS),
              "flip_interval": [interval] * len(SEEDS),
              "random_machine": [rm] * len(SEEDS)}
    dec = mod.create(copies, bundle=bundle)
    interval_copies = mod.create({**cfg, "flip_interval": [interval] * len(SEEDS)},
                                 bundle=bundle)
    assert copies["flip_start_iter"] == [start] * len(SEEDS)
    assert copies["random_machine"] == [rm] * len(SEEDS)
    assert copies["flip_interval"] == [interval] * len(SEEDS)
    assert dec.flip_start_iter == start and dec.random_machine == rm
    assert dec._fi.tolist() == [interval] * len(SEEDS)
    assert dec.flip_interval == interval
    for error_rate, B, seed in ((0.03, 4, 1), (0.06, 32, 999)):
        synd, llr0 = _data(bundle, error_rate, B, seed)
        expected = _run(scalar, synd, llr0)
        _assert_equal(_run(dec, synd, llr0), expected, "scalar copies")
        _assert_equal(_run(interval_copies, synd, llr0), expected, "interval copies")
        assert all(torch.equal(copy_generator.get_state(), scalar_generator.get_state())
                   for copy_generator, scalar_generator in zip(dec._gens, scalar._gens))
        assert all(torch.equal(copy_generator.get_state(), scalar_generator.get_state())
                   for copy_generator, scalar_generator in zip(interval_copies._gens, scalar._gens))


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("machine", [*MACHINES, ["sobol", "system", "system", "sobol"]])
@pytest.mark.parametrize("starts,intervals", [
    (0, [1, 2, 3, 4]),
    ([1, 4, 2, 9], 3),
    ([5, 2, 10, 4], [2, 3, 4, 5]),
    ([12, 11, 2, 0], [1, 3, 15, 30]),
])
@pytest.mark.parametrize("compacted", [False, True])
def test_replica_flip_grid_and_draws(backend, machine, starts, intervals, compacted):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = par_gpu if backend == "cuda" else par_py

    class Recorder(module.create):
        def _draw_r(self, count):
            drawn = super()._draw_r(count)
            self.draws.append((self.i, self._active.clone(), drawn.clone()))
            return drawn

        def _apply_flip(self, llrs, variables, mask):
            self.flips.append(mask.clone())
            return super()._apply_flip(llrs, variables, mask)

    cfg = {**_cfg(machine), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "max_iter": 13, "flip_start_iter": starts, "flip_interval": intervals}
    decoder = Recorder(cfg, bundle=_bundle(cfg))
    decoder._shots = 2
    decoder._reset_state()
    row_ids = torch.tensor([1, 2, 5, 6] if compacted else list(range(8)), device=decoder.device)
    decoder._hook_rows = row_ids
    llrs = torch.ones(len(row_ids), decoder.H_shape[1] + 1, device=decoder.device,
                      dtype=decoder.dtype)
    decisions = torch.zeros_like(llrs)
    syndrome = torch.zeros(len(row_ids), decoder.H_shape[0], device=decoder.device,
                           dtype=decoder.dtype)
    syndrome[:, 0] = 1
    start_values = starts if isinstance(starts, list) else [starts] * len(SEEDS)
    interval_values = intervals if isinstance(intervals, list) else [intervals] * len(SEEDS)
    generators = [torch.Generator(device=decoder.device).manual_seed(seed) for seed in SEEDS]
    decoder.draws, decoder.flips = [], []
    for iteration in range(1, decoder.max_iter + 1):
        expected = torch.tensor([
            iteration < decoder.max_iter and iteration > start_values[row // 2]
            and (iteration - start_values[row // 2] - 1) % interval_values[row // 2] == 0
            for row in row_ids.tolist()
        ], device=decoder.device)
        previous_draws = len(decoder.draws)
        decoder._iter_hook(iteration, llrs, decisions, torch.ones_like(expected), syndrome)
        if not expected.any():
            assert len(decoder.draws) == previous_draws
            continue
        assert len(decoder.draws) == previous_draws + 1
        actual_iteration, actual_mask, actual_draws = decoder.draws[-1]
        assert actual_iteration == iteration
        assert torch.equal(actual_mask, expected)
        assert torch.equal(decoder.flips[-1], expected)
        expected_draws = torch.cat([
            torch.rand(2, generator=generator, device=decoder.device, dtype=decoder.dtype)
            if sampler == "system" else decoder._rk[replica, iteration - 1].expand(2)
            for replica, (sampler, generator) in enumerate(zip(decoder._machines, generators))
        ])[row_ids]
        assert torch.equal(actual_draws, expected_draws)
    assert all(torch.equal(actual.get_state(), expected.get_state())
               for actual, expected in zip(decoder._gens, generators))


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("starts", [0, [0, 2, 1, 4]])
@pytest.mark.parametrize("compacted", [False, True])
def test_replica_grid_preserves_undo_and_counts_stalls(backend, starts, compacted):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = par_gpu if backend == "cuda" else par_py

    class Recorder(module.create):
        def _resample(self, rows, active, llrs, decisions, syndrome):
            self.stall_mask = active.clone()
            return super()._resample(rows, active, llrs, decisions, syndrome)

        def sign_flip_cn_rand_new(self, syndrome, estimate, llrs):
            self.picking = True
            try:
                return super().sign_flip_cn_rand_new(syndrome, estimate, llrs)
            finally:
                self.picking = False

        def _apply_flip(self, llrs, variables, mask):
            if not self.picking:
                self.undo_mask = mask.clone()
            return super()._apply_flip(llrs, variables, mask)

    intervals = [2, 3, 4, 5]
    cfg = {**_cfg("system"), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "max_iter": 13, "flip_start_iter": starts, "flip_interval": intervals,
           "flip_undo": True, "copy_on_stall": 2}
    decoder = Recorder(cfg, bundle=_bundle(cfg))
    decoder._shots = 2
    decoder._reset_state()
    row_ids = torch.tensor([1, 2, 5, 6] if compacted else list(range(8)), device=decoder.device)
    decoder._hook_rows = row_ids
    llrs = torch.ones(len(row_ids), decoder.H_shape[1] + 1, device=decoder.device,
                      dtype=decoder.dtype)
    decisions = torch.zeros_like(llrs)
    decoder._c2v = torch.zeros(len(row_ids), 1, device=decoder.device)
    start_values = starts if isinstance(starts, list) else [starts] * len(SEEDS)
    expected_best = torch.full_like(decoder._st_best, decoder.H_shape[0] + 1)
    expected_age = torch.zeros_like(decoder._st_age)
    expected_pending = torch.zeros_like(decoder._fa_var, dtype=torch.bool)
    expected_unsat = torch.zeros_like(decoder._fa_u)
    decoder.picking = False
    undo_count = 0
    for iteration in range(1, decoder.max_iter + 1):
        unsatisfied = 1 if iteration == 1 else 2
        syndrome = torch.zeros(len(row_ids), decoder.H_shape[0], device=decoder.device,
                               dtype=decoder.dtype)
        syndrome[:, :unsatisfied] = 1
        expected_grid = torch.tensor([
            iteration < decoder.max_iter and iteration > start_values[row // 2]
            and (iteration - start_values[row // 2] - 1) % intervals[row // 2] == 0
            for row in row_ids.tolist()
        ], device=decoder.device)
        expected_undo = expected_grid & expected_pending[row_ids] & (unsatisfied > expected_unsat[row_ids])
        for position, row in enumerate(row_ids.tolist()):
            if expected_grid[position]:
                expected_age[row] = 0 if unsatisfied < expected_best[row] else (expected_age[row] + 1) % 2
                expected_best[row] = min(unsatisfied, int(expected_best[row]))
                expected_pending[row] = not bool(expected_undo[position])
                if expected_pending[row]:
                    expected_unsat[row] = unsatisfied
        decoder.stall_mask = decoder.undo_mask = None
        decoder._iter_hook(iteration, llrs, decisions, torch.ones_like(expected_grid), syndrome)
        if expected_grid.any():
            assert torch.equal(decoder.stall_mask, expected_grid)
            assert torch.equal(decoder.undo_mask, expected_undo)
            undo_count += int(expected_undo.sum())
        else:
            assert decoder.stall_mask is None and decoder.undo_mask is None
        assert torch.equal(decoder._st_best, expected_best)
        assert torch.equal(decoder._st_age, expected_age)
        assert torch.equal(decoder._fa_var >= 0, expected_pending)
        assert torch.equal(decoder._fa_u, expected_unsat)
    assert undo_count > 0


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("machine", [*MACHINES, ["sobol", "system", "system", "sobol"]])
@pytest.mark.parametrize("starts,intervals,extra", [
    (0, [1, 2, 3, 4], {}),
    ([1, 4, 2, 9], 3, {}),
    ([5, 2, 10, 4], [2, 3, 4, 5], {}),
    ([1, 4, 2, 9], [2, 3, 4, 5], {"flip_undo": True, "copy_on_stall": 3,
                                  "flip_anneal": [3, 1], "syndrome_flip_last_n": 2}),
])
def test_replica_intervals_outputs_and_backend_parity(backend, machine, starts, intervals, extra):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg(machine), "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "flip_start_iter": starts, "flip_interval": intervals, **extra}
    bundle = _bundle(cfg)
    decoder = module.create(cfg, bundle=bundle)
    reference = par_py.create(cfg, bundle=bundle)
    baseline = module.create({**cfg, "flip_interval": 1}, bundle=bundle)
    changed = False
    for error_rate, batch_size, seed in ((0.03, 4, 1), (0.06, 32, 999)):
        syndrome, llr0 = _data(bundle, error_rate, batch_size, seed)
        actual = _run(decoder, syndrome, llr0)
        _assert_equal(actual, _run(reference, syndrome, llr0), "replica intervals parity")
        assert all(torch.equal(actual_generator.get_state(), expected_generator.get_state())
                   for actual_generator, expected_generator in zip(decoder._gens, reference._gens))
        without_intervals = _run(baseline, syndrome, llr0)
        changed |= any(not torch.equal(actual[key], without_intervals[key]) for key in KEYS)
    assert changed


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("seeds,starts,machines", [
    ([11], [0], ["sobol"]),
    (SEEDS, [5, 2, 10, 4], ["sobol", "system", "system", "sobol"]),
])
def test_replica_config_lists(backend, seeds, starts, machines):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg(machines, seeds=seeds),
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
           "flip_start_iter": starts}
    dec = mod.create(cfg, bundle=_bundle(cfg))
    assert dec._fsi.tolist() == starts
    assert dec.flip_start_iter == min(starts)
    assert dec._machines == machines
    assert dec.random_machine == ("system" if "system" in machines else "sobol")
    assert cfg["flip_start_iter"] == starts and cfg["random_machine"] == machines


@pytest.mark.parametrize("mod", [par_py, par_gpu], ids=["pytorch", "cuda"])
def test_syndrome_flip_estimate(surface10, mod):
    """With syndrome_flip_last_n, converge is 1 exactly where the returned
    e_v satisfies the true syndrome."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    cfg = {**_cfg("sobol"), "syndrome_flip_last_n": 2}
    out = _run(mod.create(cfg, bundle=bundle), synd, llr0)
    ok = ((out["e_v"] @ _dense_h(bundle).T) % 2 == synd.double()).all(1)
    assert torch.equal(ok, out["converge"] == 1)


@pytest.mark.parametrize("rm", MACHINES)
def test_syndrome_flip_converge_every_row(surface10, rm):
    """On every syndrome-flip row of the B * K decode (not only the returned
    one), converge is 1 exactly where e_v satisfies the true syndrome, also
    on rows still unconverged at max_iter."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.05, 256, 0)
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    knobs = {
        "syndrome_flip_last_n": 4,
        "flip_anneal": [3, 1],
        "flip_temperature": 1.0,
        "flip_undo": True,
    }
    cfg = {**_cfg(rm, seeds=seeds), "dtype": "float32", **knobs}

    class Rec(par_py.create):
        def _exit_hook(self, l_v, e_v, num_iters, converges):
            super()._exit_hook(l_v, e_v, num_iters, converges)
            self.raw = (e_v.clone(), converges.clone())

    dec = Rec(cfg, bundle=bundle)
    _run(dec, synd, llr0)
    e_v, conv = dec.raw
    H = _dense_h(bundle)
    e = (e_v[:, : H.shape[1]] != 0).double().cpu()
    ok = ((e @ H.T) % 2 == synd.double().repeat(len(seeds), 1)).all(1)
    sf = torch.arange(len(ok)) >= 4 * len(synd)
    assert torch.equal(ok[sf], conv.view(-1).cpu()[sf] == 1)


@pytest.mark.parametrize(
    "extra",
    [
        {"syndrome_flip_last_n": 2},
        {"syndrome_flip_last_n": 2, "copy_on_stall": 3},
        {"copy_on_stall": 3},
        {"syndrome_flip_last_n": 2, "flip_undo": True},
    ],
    ids=["sf", "sf_stall", "stall", "sf_undo"],
)
def test_anneal_no_variable_twice(surface10, extra):
    """Under flip_anneal [3, 3], the flips of one hook call (the bp_lottery flip
    and its repeats, not the flip_undo step) never pick the same variable
    twice on a row, on LLR and syndrome-flip rows alike."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 256, 999)

    class Rec(par_py.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self._log = []
            super()._iter_hook(i, l_v, e_v, active, syndrome)
            seen = torch.zeros_like(l_v, dtype=torch.long)
            for var, mask in self._log:
                seen.scatter_add_(1, var.unsqueeze(1), mask.long().unsqueeze(1))
            self.flips += sum(int(m.sum()) for _, m in self._log[1:])
            self.twice += int((seen > 1).sum())

        def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
            self._rec = True
            try:
                return super().sign_flip_cn_rand_new(syndrome, s_est, l_v)
            finally:
                self._rec = False

        def _apply_flip(self, l_v, var, mask):
            if getattr(self, "_rec", False):
                self._log.append((var.clone(), mask.clone()))
            super()._apply_flip(l_v, var, mask)

    dec = Rec({**_cfg("sobol"), "flip_anneal": [3, 3], **extra}, bundle=bundle)
    dec.flips = dec.twice = 0
    _run(dec, synd, llr0)
    assert dec.flips > 0
    assert dec.twice == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"copy_on_stall": 3},
        {"syndrome_flip_last_n": 2, "flip_anneal": [3, 3]},
    ],
    ids=["stall", "sf_anneal"],
)
def test_stuck_check_unsatisfied(surface10, monkeypatch, extra):
    """With stuck_check_weight, each flipped row's drawn check is unsatisfied,
    also on a row whose unsatisfied checks all have run 0."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 256, 999)
    picked = []
    real = par_py.cn_row_mask
    monkeypatch.setattr(
        par_py, "cn_row_mask", lambda V, idx, N: picked.append(idx) or real(V, idx, N)
    )

    class Rec(par_py.create):
        def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
            unsat = ((syndrome + s_est) % 2.0).bool()
            picked.clear()
            out = super().sign_flip_cn_rand_new(syndrome, s_est, l_v)
            flip = self._active & unsat.any(1)
            hit = unsat.gather(1, picked[0].unsqueeze(1)).squeeze(1)
            self.picks += int(flip.sum())
            self.bad += int((flip & ~hit).sum())
            return out

    cfg = {**_cfg("sobol"), "stuck_check_weight": True, **extra}
    dec = Rec(cfg, bundle=bundle)
    dec.picks = dec.bad = 0
    _run(dec, synd, llr0)
    assert dec.picks > 0
    assert dec.bad == 0


def test_resample_copies_knob_state(surface10):
    """copy_on_stall: a copied row takes, with the sibling's l_v and e_v, the
    sibling's per-row knob state (stuck runs, oscillation counts, previous
    hard decision, pending flip_undo step)."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 256, 999)

    class Rec(par_py.create):
        def _state(self, rows, l_v, e_v):
            xs = (self._stuck, self._osc, self._prev, self._fa_var, self._fa_u)
            return [l_v.clone(), e_v.clone()] + [x[rows].clone() for x in xs]

        def _resample(self, rows, active, l_v, e_v, syndrome):
            pre = self._state(rows, l_v, e_v)
            super()._resample(rows, active, l_v, e_v, syndrome)
            post = self._state(rows, l_v, e_v)
            R = len(l_v)

            def same(a, b, r):
                return (a == b[r]).reshape(R, -1).all(1)

            for r in (post[0] != pre[0]).any(1).nonzero().flatten().tolist():
                src = same(pre[0], post[0], r) & same(pre[1], post[1], r)
                assert bool(src.any()), r
                for a, b in zip(pre[2:], post[2:]):
                    src &= same(a, b, r)
                assert bool(src.any()), r
                self.copied += 1

    cfg = {
        **_cfg("sobol"),
        "copy_on_stall": 3,
        "flip_tiebreak": "osc",
        "stuck_check_weight": True,
        "flip_undo": True,
    }
    dec = Rec(cfg, bundle=bundle)
    dec.copied = 0
    _run(dec, synd, llr0)
    assert dec.copied > 0


def test_cross_kind_copy_drops_undo(surface10):
    """copy_on_stall with syndrome_flip_last_n and flip_undo: a row that
    copies a replica of the other kind (LLR or syndrome flip) has no pending
    undo."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 256, 999)

    class Rec(par_py.create):
        cross = 0

        def _resample(self, rows, active, l_v, e_v, syndrome):
            pre = l_v.clone()
            super()._resample(rows, active, l_v, e_v, syndrome)
            kind = self._per_row(self._sf_rep, len(l_v))
            for r in (pre != l_v).any(1).nonzero().flatten().tolist():
                src = (pre == l_v[r]).all(1)
                if bool(src.any()) and not bool((src & (kind == kind[r])).any()):
                    self.cross += 1
                    assert int(self._fa_var[r]) == -1, r

    cfg = {
        **_cfg("sobol"),
        "syndrome_flip_last_n": 2,
        "flip_undo": True,
        "copy_on_stall": 3,
    }
    dec = Rec(cfg, bundle=bundle)
    _run(dec, synd, llr0)
    assert dec.cross > 0


def test_consensus_skips_rows_with_syndrome_flips(surface10):
    """consensus_every with copy_on_stall and syndrome_flip_last_n: an LLR
    row that copied a syndrome-flip row (a variable in _sf_var) gets no
    frozen variables and no flip candidates."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 256, 999)

    class Rec(par_py.create):
        seen = 0

        def _consensus(self, i, rows, active, l_v):
            super()._consensus(i, rows, active, l_v)
            if i % self._cons_m == 0:
                x = ~self._no_sf(slice(None))
                x[2 * self._shots :] = False  # LLR rows only
                self.seen += int(x.sum())
                assert not bool(self._cz_frozen[x].any())
                assert not bool(self._cz_dis[x].any())

    cfg = {
        **_cfg("sobol"),
        "consensus_every": 2,
        "syndrome_flip_last_n": 2,
        "copy_on_stall": 3,
    }
    dec = Rec(cfg, bundle=bundle)
    _run(dec, synd, llr0)
    assert dec.seen > 0


def test_consensus_skips_syndrome_flip_rows(surface10):
    """consensus_every with syndrome_flip_last_n: the syndrome-flip rows get
    no frozen variables and no flip candidates."""
    bundle, synd, llr0 = surface10
    cfg = {**_cfg("sobol"), "consensus_every": 2, "syndrome_flip_last_n": 2}
    dec = par_py.create(cfg, bundle=bundle)
    _run(dec, synd, llr0)
    B = len(synd)
    assert dec._cz_frozen is not None
    assert not bool(dec._cz_frozen[2 * B :].any())
    assert not bool(dec._cz_dis[2 * B :].any())
    assert bool(dec._cz_dis[: 2 * B].any())


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_final_consensus_matches_plain_bp(backend):
    from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum, bp_norm_min_sum_cuda

    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    plain = bp_norm_min_sum_cuda if backend == "cuda" else bp_norm_min_sum
    cfg = {
        **_cfg("sobol"),
        "device": {"device_type": "cpu" if backend == "cpu" else "cuda"},
        "max_iter": 3,
        "flip_start_iter": 3,
        "consensus_every": 2,
        "consensus_llr": 0.0,
    }
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.1, 16, 0)
    # Identical replicas freeze iteration 2 unchanged; iteration 3 must keep
    # its own posterior instead of restoring iteration 2's frozen values.
    out = _run(mod.create(cfg, bundle=bundle), synd, llr0)
    ref = _run(plain.create(cfg, bundle=bundle), synd, llr0)
    assert not bool(out["converge"].all())
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    _assert_equal(out, ref, "no terminal consensus")


def test_consensus_llr_value(surface10):
    """consensus_llr 2.0 gives other outputs than the default 10.0."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    a = _run(
        par_py.create({**_cfg("sobol"), "consensus_every": 5}, bundle=bundle),
        synd,
        llr0,
    )
    cfg = {**_cfg("sobol"), "consensus_every": 5, "consensus_llr": 2.0}
    b = _run(par_py.create(cfg, bundle=bundle), synd, llr0)
    assert not all(torch.equal(a[k], b[k]) for k in KEYS)


@pytest.mark.parametrize("rm", MACHINES)
def test_knob_float32_compiled(surface10, rm):
    """At float32 with compile true, flip_tiebreak osc changes the outputs and
    the CUDA port matches the compiled PyTorch module."""
    bundle, _, _ = surface10
    synd, llr0 = _data(bundle, 0.06, 128, 999)
    base = {**_cfg(rm), "dtype": "float32", "compile": True}
    on = {**base, "flip_tiebreak": "osc"}
    off = _run_rng(par_py, base, bundle, synd, llr0)
    ref = _run_rng(par_py, on, bundle, synd, llr0)
    assert not all(torch.equal(off[0][k], ref[0][k]) for k in KEYS)
    assert _same_run(_run_rng(par_gpu, on, bundle, synd, llr0), ref)


@pytest.mark.parametrize("rm", MACHINES)
def test_agree_stop_off_cpu(rm):
    cfg = {**_cfg(rm), "device": {"device_type": "cpu"}}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.06, 32, 999)
    runs = []
    for config in (cfg, {**cfg, "agree_stop": 0}):
        torch.manual_seed(5)
        dec = par_py.create(config, bundle=bundle)
        runs.append((_run(dec, synd, llr0), torch.get_rng_state(),
                     torch.stack([g.get_state() for g in dec._gens])))
    _assert_equal(runs[0][0], runs[1][0], "agree_stop off")
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1:], runs[1][1:]))


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("syndrome_flip", [False, True])
def test_agree_stop_converged_groups(backend, compacted, syndrome_flip):
    """Only cached first convergences vote; disagreement, active matches and
    stopped rows do not. Canonical decisions survive compaction and SF."""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("system"), "max_iter": 5, "agree_stop": 2,
           "flip_start_iter": 99, "syndrome_flip_last_n": 2 if syndrome_flip else 0,
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    dec = mod.create(cfg, bundle=_bundle(cfg))
    K, B, N = len(SEEDS), 3, dec.H_shape[1]
    dec._shots = B
    dec._reset_state()
    canonical = torch.zeros(K * B, N + 1, dtype=torch.bool, device=dec.device)
    canonical[B + 1, 0] = canonical[3 * B + 1, 0] = True
    canonical[2::B, 1] = True
    first = torch.tensor([1, 1, 2, 3, 2, 99, 99, 99, 1, 99, 99, 99], device=dec.device)
    rows = torch.arange(K * B, device=dec.device)
    for iteration in range(1, dec.max_iter + 1):
        dec._hook_rows = rows
        active = (first[rows] > iteration) & ~dec._ag_stopped[rows]
        decisions = canonical[rows].clone()
        # CUDA skip_converged=False can rewrite a previously converged row.
        decisions[first[rows] < iteration, 2] = True
        if syndrome_flip:
            dec._sf_var[:, 0] = True
            decisions ^= dec._sf_var[rows]
        llrs = torch.ones(len(rows), N + 1, device=dec.device, dtype=dec.dtype)
        syndrome = torch.zeros(len(rows), dec.H_shape[0], device=dec.device, dtype=dec.dtype)
        dec._iter_hook(iteration, llrs, decisions.to(dec.dtype), active, syndrome)
        if compacted and iteration < dec.max_iter:
            stop = torch.zeros_like(active) if dec._hook_stop is None else dec._hook_stop
            rows = rows[active & ~stop]
    assert dec._ag_done.tolist() == [3, 0, 2]
    assert dec._ag_count.tolist() == [2, 0, 2]
    assert dec._ag_winner.tolist() == [0, -1, 8]
    known = dec._ag_iter > 0
    expected_hash = (canonical[:, :N].long() * dec._ag_weights).sum(1)
    assert torch.equal(dec._ag_hash[known], expected_hash[known])
    assert bool((dec._ag_iter[dec._ag_stopped] == 0).all())
    assert int(dec._ag_stopped.sum()) == 4
    assert dec._hook_stop is None  # the final iteration cannot reuse a stop mask


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("threshold", [1, len(SEEDS)])
@pytest.mark.parametrize("iteration", [2, 3])
def test_agree_stop_boundary(backend, threshold, iteration):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol"), "agree_stop": threshold, "max_iter": 3,
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    dec = mod.create(cfg, bundle=_bundle(cfg))
    dec._shots = 1
    dec._reset_state()
    dec._hook_rows = torch.arange(len(SEEDS), device=dec.device)
    decisions = torch.zeros(len(SEEDS), dec.H_shape[1] + 1, device=dec.device)
    llrs = torch.ones_like(decisions)
    active = torch.zeros(len(SEEDS), device=dec.device, dtype=torch.bool)
    syndrome = torch.zeros(len(SEEDS), dec.H_shape[0], device=dec.device)
    dec._iter_hook(iteration, llrs, decisions, active, syndrome)
    assert dec._ag_done.item() == (iteration if iteration < dec.max_iter else 0)
    if iteration < dec.max_iter:
        assert dec._ag_winner.item() == 0  # equal convergence iterations tie by replica
        assert dec._ag_count.item() == len(SEEDS)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("threshold", [1, 2, len(SEEDS)])
@pytest.mark.parametrize("case,extra", [
    ("plain", {}),
    ("buffers_off", {"compact_frac": 0, "reuse_buffers": False}),
    ("skip_off", {"skip_converged": False, "host_check_every": 1}),
    ("sf", {"syndrome_flip_last_n": 2, "copy_on_stall": 2,
            "flip_undo": True, "flip_anneal": [3, 1]}),
    ("ensemble", {"ensemble_llr": True}),
    ("compiled", {"compile": True, "dtype": "float32"}),
    ("cap", {}),
])
def test_agree_stop_real_rows(backend, threshold, case, extra):
    """Agreed shots select their earliest member but report detection time;
    stopped rows keep their own posterior and cannot vote or converge later."""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("system"), "max_iter": 30, "agree_stop": threshold,
           "report_agreement": True, "select": "min_flip_posterior", **extra,
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.06, 32, 999)

    class Recorder(mod.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            super()._iter_hook(i, l_v, e_v, active, syndrome)
            if self._hook_stop is not None and self._hook_stop.any():
                stop = self._hook_stop
                rows = self._hook_rows[stop]
                hard = e_v[stop, :self.H_shape[1]] != 0
                llrs = l_v[stop, :self.H_shape[1]].clone()
                if self._sf_var is not None:
                    flipped = self._sf_var[rows, :self.H_shape[1]]
                    hard ^= flipped
                    llrs = torch.where(flipped, -llrs, llrs)
                self.stops.append((rows.clone(), i, hard.clone(), llrs))

        def _exit_hook(self, l_v, e_v, num_iters, converges):
            super()._exit_hook(l_v, e_v, num_iters, converges)
            self.raw = {"e_v": e_v[:, :-1].clone(), "llr": l_v[:, :-1].clone(),
                        "iter": num_iters.clone(), "converge": converges.clone()}

    dec = Recorder(cfg, bundle=bundle)
    dec.stops = []
    if case == "cap":
        dec.cap = RebatchSpeedup()
        dec.cap.frac = 0.25
    with torch.no_grad():
        out = dec({"synd": synd.clone(), "llr0": llr0.clone()})
    raw, B, K = dec.raw, len(synd), len(SEEDS)
    assert out["iter"].dtype == raw["iter"].dtype
    agreed = dec._ag_done > 0
    assert agreed.any()
    if threshold < K:
        assert dec.stops
    if case == "compiled" and backend == "pytorch":
        assert dec.compile
    for rows, iteration, hard, llrs in dec.stops:
        assert bool((raw["converge"][rows] == 0).all())
        assert bool((raw["iter"][rows] == iteration).all())
        assert torch.equal(raw["e_v"][rows].bool(), hard)
        assert torch.equal(raw["llr"][rows], llrs)
        assert torch.equal(raw["llr"][rows] <= 0, hard)
    for b in range(B):
        if not bool(agreed[b]):
            continue
        group = [k * B + b for k in range(K)
                 if raw["converge"][k * B + b] == 1
                 and torch.equal(raw["e_v"][k * B + b], out["e_v"][b])]
        assert len(group) >= threshold
        first = min(group, key=lambda r: (int(raw["iter"][r]), r))
        detection = sorted(int(raw["iter"][r]) for r in group)[threshold - 1]
        assert torch.equal(out["llr"][b], raw["llr"][first])
        assert int(out["iter"][b]) == detection < dec.max_iter
        assert int(out["converge"][b]) == 1
        assert int(out["agreement"][b]) == len(group)
    H = _dense_h(bundle)
    assert bool((((out["e_v"][agreed].cpu().double() @ H.T) % 2)
                 == synd[agreed.cpu()]).all())
    # No convergence may disappear just because the loop broke on a host/cap check.
    for b in range(B):
        for k in range(K):
            r = k * B + b
            matches = [q * B + b for q in range(K)
                       if raw["converge"][q * B + b] == 1
                       and raw["iter"][q * B + b] < dec.max_iter
                       and torch.equal(raw["e_v"][q * B + b], raw["e_v"][r])]
            if len(matches) >= threshold:
                assert bool(agreed[b])
    if case != "cap":
        plain = mod.create({**cfg, "agree_stop": 0}, bundle=bundle)
        with torch.no_grad():
            baseline = plain({"synd": synd.clone(), "llr0": llr0.clone()})
        for key in (*KEYS, "agreement"):
            assert torch.equal(out[key][~agreed], baseline[key][~agreed]), (case, key)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize("before_stop", [False, True])
def test_hook_stop_generic_cap_and_reset(backend, before_stop):
    """A generic stop mask does not require agreement's early hook, never
    counts toward the cap, and snapshots the stopped row on each forward."""
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = par_gpu if backend == "cuda" else par_py
    cfg = {**_cfg("sobol", seeds=[11]), "max_iter": 5, "skip_converged": False,
           "host_check_every": 0, "compact_frac": 0,
           "device": {"device_type": "cpu" if backend == "cpu" else "cuda"}}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.2, 2, 0)

    class Stopper(mod.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            assert i <= 3
            self._hook_stop = active & ((self._hook_rows == 0) if i == 1 else (i == 3))
            if i == 1:
                assert bool(self._hook_stop.any())
                self.first = (l_v[0, :-1].clone(), e_v[0, :-1].clone())

    dec = Stopper(cfg, bundle=bundle)
    dec._hook_before_stop = before_stop
    dec.cap = RebatchSpeedup()
    dec.cap.frac = 0.5
    for _ in range(2):
        out = _run(dec, synd, llr0)
        assert out["iter"].tolist() == [1, 3]
        assert out["converge"].tolist() == [0, 0]
        assert not bool(out["defer"].any())
        assert torch.equal(out["llr"][0], dec.first[0].cpu())
        assert torch.equal(out["e_v"][0], dec.first[1].cpu().to(out["e_v"]))


def test_hook_stop_compiled_singleton():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = {**_cfg("sobol", seeds=[11]), "max_iter": 5, "compile": True,
           "dtype": "float32", "compact_frac": 0}
    bundle = _bundle(cfg)
    synd, llr0 = _data(bundle, 0.2, 2, 0)

    class Stopper(par_py.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self.shapes.append(len(active))
            self._hook_stop = active & ((self._hook_rows == 0) if i == 1 else (i == 3))
            if i == 1:
                self.first = (l_v[0, :-1].clone(), e_v[0, :-1].clone())

    dec = Stopper(cfg, bundle=bundle)
    assert dec.compile
    dec._hook_before_stop = True
    dec.shapes = []
    out = _run(dec, synd, llr0)
    assert dec.shapes == [2, 1, 1]
    assert out["iter"].tolist() == [1, 3]
    assert out["converge"].tolist() == [0, 0]
    assert torch.equal(out["llr"][0], dec.first[0].cpu())
    assert torch.equal(out["e_v"][0], dec.first[1].cpu())
