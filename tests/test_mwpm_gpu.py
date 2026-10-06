import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures.process import BrokenProcessPool
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from syndrilla.decoder.mwpm_gpu import mwpm_gpu, mwpm_gpu_cuda
from syndrilla.matrix.matrix import dense_to_index_format


sys.path.append(os.getcwd())


# A triangle, a fourth node, and two boundary edges exercise blossom matching.
H = np.array([
    [1, 0, 1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 0],
], dtype=np.uint8)
CANDIDATES = np.array(list(product((0, 1), repeat=H.shape[1])), dtype=np.uint8)
SYNDROMES = np.array(list(product((0, 1), repeat=H.shape[0])), dtype=np.uint8)


def _bundle(h=H, device="cpu"):
    indices = dense_to_index_format(h, torch.device(device))
    return SimpleNamespace(Hx_matrix=SimpleNamespace(get_index=lambda: indices))


def _cfg(**extra):
    return dict(device={"device_type": "cpu"}, dtype="float64", num_workers=1, **extra)


@pytest.mark.parametrize("n", [8, 65])
def test_nonuniform_matches_brute_force(n):
    h = np.pad(H, ((0, 0), (0, n - H.shape[1])))
    rng = np.random.default_rng(7)
    for _ in range(5):
        weights = 2 * rng.integers(1, 12, n)
        matcher = mwpm_gpu.NativeMatcher.from_check_matrix(*h.nonzero(), *h.shape, weights=weights)
        costs = CANDIDATES @ weights[:H.shape[1]]
        for syndrome in SYNDROMES:
            correction, cost = matcher.decode_with_weight(syndrome)
            valid = ((CANDIDATES @ H.T) % 2 == syndrome).all(1)
            assert np.array_equal(h @ correction % 2, syndrome)
            assert correction @ weights == costs[valid].min()
            assert cost == correction @ weights


def test_weighted_parallel_edges_and_shortest_path():
    h = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
    for weights, expected in [([8, 2, 8, 2], [0, 1, 0, 0]), ([2, 2, 8, 2], [1, 0, 0, 0])]:
        matcher = mwpm_gpu.NativeMatcher.from_check_matrix(*h.nonzero(), *h.shape, weights=weights)
        assert matcher.decode([1, 1]).tolist() == expected
        assert matcher.decode([1, 0]).tolist() == [0, 0, 0, 1]
    h = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    matcher = mwpm_gpu.NativeMatcher.from_check_matrix(*h.nonzero(), *h.shape, weights=[10, 2, 2])
    assert matcher.decode([1, 1, 0]).tolist() == [0, 1, 1]


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_float_parallel_winner_selected_before_rounding(backend, monkeypatch):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    h = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    module = mwpm_gpu_cuda if backend == "cuda" else mwpm_gpu
    decoder = module.create({**_cfg(weights="prior"), "device": {"device_type": backend}},
                            bundle=_bundle(h, backend))
    if backend == "cuda":
        assert decoder._use_kernel
        monkeypatch.setattr(decoder, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
        weights = torch.tensor([[1.00000001, 1.0, 2.5]], dtype=torch.float64, device=backend)
        assert int(decoder._weighted_csr(weights)[2].max()) == 2 * (2**24 - 1)
    out = decoder({"synd": torch.tensor([[1, 0]], device=backend),
                   "llr0": torch.tensor([[1.00000001, 1.0, 2.5]], dtype=torch.float64, device=backend)})
    assert out["e_v"].tolist() == [[0, 1, 0]]


@pytest.mark.parametrize("weight,cost", [(0., 0), (0.5 - 2**-53, 0), (0.5, 2), (0.5 + 2**-53, 2)])
def test_float_weights_round_half_away_and_preserve_zero(weight, cost):
    h = np.eye(2, dtype=np.uint8)
    matcher = mwpm_gpu.NativeMatcher.from_check_matrix(
        *h.nonzero(), *h.shape, weights=np.array([weight, 2**24 - 1]))
    correction, actual_cost = matcher.decode_with_weight([1, 0])
    assert correction.tolist() == [1, 0]
    assert matcher._weights[0] == cost
    assert actual_cost == cost / 2


@pytest.mark.parametrize("mode", ["posterior", "prior"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_constant_llr_matches_uniform(mode, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    bundle = _bundle(device=backend)
    module = mwpm_gpu_cuda if backend == "cuda" else mwpm_gpu
    cfg = {**_cfg(weights="uniform"), "device": {"device_type": backend}}
    syndrome = torch.from_numpy(SYNDROMES)
    llr = torch.full((len(syndrome), H.shape[1]), 3.0, dtype=torch.float64)
    uniform_decoder = module.create(cfg, bundle=bundle)
    assert uniform_decoder.skip_converged is False
    uniform = uniform_decoder({"synd": syndrome})
    weighted = module.create({**cfg, "weights": mode}, bundle=bundle)(
        {"synd": syndrome, "llr0" if mode == "prior" else "llr": llr})
    for key in ("e_v", "llr", "iter", "converge"):
        assert torch.equal(uniform[key], weighted[key]), key
    matcher = mwpm_gpu.NativeMatcher.from_check_matrix(*H.nonzero(), *H.shape)
    correction, cost = matcher.decode_with_weight(SYNDROMES[-1])
    assert cost == correction.sum()


@pytest.mark.parametrize("mode", ["posterior", "prior"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_zero_llr_uses_bp_hard_decision(mode, backend, monkeypatch):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    module = mwpm_gpu_cuda if backend == "cuda" else mwpm_gpu
    dec = module.create({**_cfg(weights=mode), "device": {"device_type": backend}},
                        bundle=_bundle(np.array([[1, 1]]), backend))
    if backend == "cuda":
        assert dec._use_kernel
        monkeypatch.setattr(dec, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
    out = dec({"synd": torch.zeros(1, 1), "llr0" if mode == "prior" else "llr": torch.zeros(1, 2)})
    assert out["e_v"].tolist() == [[1.0, 1.0]]
    assert torch.equal(out["llr"] <= 0, out["e_v"])


@pytest.mark.parametrize("backend", ["serial", "pool", "cuda"])
@pytest.mark.parametrize("mode", ["posterior", "prior"])
def test_signed_llr_matches_brute_force(backend, mode, monkeypatch):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda" if backend == "cuda" else "cpu"
    cfg = _cfg(weights=mode)
    cfg["device"] = {"device_type": device}
    if backend == "pool":
        cfg.update(num_workers=2, mp_min_batch=1)
    mod = mwpm_gpu_cuda if backend == "cuda" else mwpm_gpu
    dec = mod.create(cfg, bundle=_bundle(device=device))
    if backend == "cuda":
        assert dec._use_kernel
        monkeypatch.setattr(dec, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
    llr = np.array([0.0, -0.2, 0.6, -1.0, 2.0, -9.0, 0.9, 1.4])
    posterior = np.stack([np.roll(llr, i % len(llr)) for i in range(len(SYNDROMES))])
    out = dec({
        "synd": torch.from_numpy(SYNDROMES),
        "llr0" if mode == "prior" else "llr": torch.from_numpy(posterior),
        "llr" if mode == "prior" else "llr0": torch.full(posterior.shape, 20.0),
    })
    correction = out["e_v"].cpu().numpy().astype(np.uint8)
    assert np.array_equal(correction @ H.T % 2, SYNDROMES)
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    assert bool(out["converge"].all())
    assert out["e_v"].device.type == device
    for i, syndrome in enumerate(SYNDROMES):
        raw = np.abs(posterior[i])
        normalization = 1.0 if np.equal(raw, np.floor(raw)).all() else (2**24 - 1) / raw.max()
        scaled = raw * normalization
        lower = np.floor(scaled)
        weights = 2 * (lower + (scaled - lower >= 0.5)).astype(np.int64)
        flipped = (posterior[i] <= 0).astype(np.uint8)
        valid = ((CANDIDATES @ H.T) % 2 == syndrome).all(1)
        costs = (CANDIDATES ^ flipped) @ weights
        assert (correction[i] ^ flipped) @ weights == costs[valid].min()
    if backend == "pool":
        assert dec._pool is not None
        dec._pool.shutdown(wait=True)
        dec._pool = None


@pytest.mark.parametrize("backend", ["serial", "pool", "cuda_fallback"])
def test_prior_matcher_reuse_matches_posterior(backend, monkeypatch):
    if backend == "cuda_fallback" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = _cfg(weights="prior")
    if backend == "cuda_fallback":
        interface = _stim_surface5()
        bundle = interface.matrix_bundle
        torch.manual_seed(1234)
        errors, _ = interface.error_model.inject_error(
            torch.zeros(64, interface.error_model.num_errors, dtype=torch.float64))
        syndrome = interface.syndrome_generator.measure_syndrome(errors, None)
        prior = interface.error_model.get_llr(errors)
        assert prior.shape[1] == 502
        assert bool(syndrome.any())
    elif backend == "pool":
        bundle = _bundle(np.array([[1, 1, 1, 1], [1, 1, 0, 0]]))
        syndrome = torch.tensor([[1, 1]] * 16)
        patterns = torch.tensor([[1., 20., 3., 30.], [20., 1., 30., 3.]], dtype=torch.float64)
        order = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
        prior = patterns[order]
    else:
        bundle = _bundle()
        syndrome = torch.tensor(SYNDROMES)
        base = torch.tensor([0., 0.1, 0.6, 1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float64)
        other = base.clone()
        other[4] += 1
        prior = torch.stack((base, base + 0.001, -base, other)).repeat(4, 1)
    reference = mwpm_gpu.create({**cfg, "weights": "posterior"}, bundle=bundle)
    if backend == "pool":
        cfg.update(num_workers=2, mp_min_batch=1)
    if backend == "cuda_fallback":
        cfg["device"] = {"device_type": "cuda"}
    decoder = (mwpm_gpu_cuda if backend == "cuda_fallback" else mwpm_gpu).create(cfg, bundle=bundle)
    if backend == "cuda_fallback":
        assert not decoder._use_kernel
    build = mwpm_gpu.NativeMatcher.from_check_matrix
    built = []

    def record(*args, **kwargs):
        built.append(tuple(kwargs["weights"]))
        return build(*args, **kwargs)

    changed = prior.clone()
    changed[:, 2] += 1
    try:
        for llr in (prior, changed, prior):
            ref = reference({"synd": syndrome, "llr": llr})
            if backend == "pool":
                expected = torch.zeros_like(llr)
                expected[torch.arange(len(order)), order] = 1
                assert torch.equal(ref["e_v"], expected)
                assert torch.unique(ref["e_v"], dim=0).shape[0] == 2
            built.clear()
            with monkeypatch.context() as patch:
                if backend != "pool":
                    patch.setattr(mwpm_gpu.NativeMatcher, "from_check_matrix", record)
                out = decoder({"synd": syndrome, "llr0": llr})
            for key in ("e_v", "llr", "iter", "converge"):
                assert torch.equal(out[key].cpu(), ref[key]), key
            if backend != "pool":
                unique = {tuple(row) for row in llr.abs().tolist()}
                assert set(built) == unique
                assert len(built) == len(unique) < len(llr)
        if backend == "pool":
            assert decoder._pool is not None
    finally:
        if decoder._pool is not None:
            decoder._pool.shutdown(wait=True)
            decoder._pool = None


def test_prior_worker_builds_each_unique_weight_once_per_batch(monkeypatch):
    weights = [(2 * (i + 1),) + (2,) * (H.shape[1] - 1) for i in range(33)]
    shots = [(SYNDROMES[i % len(SYNDROMES)], row) for i, row in enumerate(weights * 2)]
    build = mwpm_gpu.NativeMatcher.from_check_matrix
    expected = [build(*H.nonzero(), *H.shape, weights=w).decode(s) for s, w in shots]
    built = []

    def record(*args, **kwargs):
        built.append(tuple(kwargs["weights"]))
        return build(*args, **kwargs)

    mwpm_gpu._mp_worker_init(*H.nonzero(), *H.shape, weighted=True)
    monkeypatch.setattr(mwpm_gpu.NativeMatcher, "from_check_matrix", record)
    for generation in (1, 2):
        for i, ((syndrome, weight), ref) in enumerate(zip(shots, expected)):
            got = mwpm_gpu._mp_worker_decode_prior((generation, syndrome, i % len(weights), weight))
            assert np.array_equal(got, ref)
        assert len(built) == generation * len(weights)
        assert set(built) == set(weights)
    mwpm_gpu._mp_worker_init(*H.nonzero(), *H.shape, weighted=True)
    got = mwpm_gpu._mp_worker_decode_prior((2, shots[0][0], 0, shots[0][1]))
    assert np.array_equal(got, expected[0])
    assert len(built) == 2 * len(weights) + 1


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("mode", ["uniform", "posterior"])
@pytest.mark.parametrize("cuda_initialized", [False, True])
def test_pool_context_and_initializer(fallback, mode, cuda_initialized, monkeypatch):
    cfg = {**_cfg(weights=mode), "num_workers": 2, "mp_min_batch": 2}
    dec = mwpm_gpu.create(cfg, bundle=_bundle())
    contexts, created = [], []
    context = object()
    start = "forkserver" if cuda_initialized else "fork"
    pool = SimpleNamespace(shutdown=lambda **kwargs: None)

    def get_context(method=None):
        contexts.append(method)
        if fallback and method == start:
            raise ValueError("start method unavailable")
        return context

    def executor(**kwargs):
        created.append(kwargs)
        return pool

    monkeypatch.setattr(mwpm_gpu.mp, "get_context", get_context)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialized)
    monkeypatch.setattr(mwpm_gpu, "ProcessPoolExecutor", executor)
    assert dec._get_pool(1) is None
    assert dec._get_pool(2) is dec._get_pool(3) is pool
    assert contexts == ([start, None] if fallback else [start])
    assert len(created) == 1
    assert created[0]["max_workers"] == 2
    assert created[0]["mp_context"] is context
    assert created[0]["initializer"] is mwpm_gpu._mp_worker_init
    rows, cols, m, n, weighted = created[0]["initargs"]
    assert np.array_equal(rows, dec._H_coo[0])
    assert np.array_equal(cols, dec._H_coo[1])
    assert (m, n, weighted) == (*H.shape, mode == "posterior")


@pytest.mark.parametrize("backend", ["cpu_uniform", "cpu_posterior", "cuda_posterior"])
@pytest.mark.parametrize("failure", ["map", "iteration"])
@pytest.mark.parametrize("fail_again", [False, True])
def test_broken_pool_retries_forward(backend, failure, fail_again, monkeypatch):
    cuda = backend == "cuda_posterior"
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mode = "uniform" if backend == "cpu_uniform" else "posterior"
    h = np.pad(H, ((0, 0), (0, 65 - H.shape[1])))
    cfg = _cfg(weights=mode, skip_converged=mode == "posterior")
    io = {"synd": torch.tensor(SYNDROMES[1:5]),
          "llr": torch.full((4, 65), 3.5, dtype=torch.float64),
          "e_v": torch.zeros(4, 65, dtype=torch.float64),
          "converge": torch.tensor([1, 0, 0, 0])}
    ref = mwpm_gpu.create(cfg, bundle=_bundle(h))({k: v.clone() for k, v in io.items()})
    cfg.update(num_workers=2, mp_min_batch=1, device={"device_type": "cuda" if cuda else "cpu"})
    dec = (mwpm_gpu_cuda if cuda else mwpm_gpu).create(cfg, bundle=_bundle(h, "cuda" if cuda else "cpu"))
    if cuda:
        monkeypatch.setattr(dec, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
    pools, batches = [], []

    class Pool:
        def __init__(self, **kwargs):
            self.fail = not pools or fail_again
            self.shutdown_calls = []
            pools.append(self)
            kwargs["initializer"](*kwargs["initargs"])

        def map(self, fn, shots, **kwargs):
            shots = list(shots)
            batches.append(shots)
            if self.fail and failure == "map":
                raise BrokenProcessPool("submit failed")

            def results():
                for shot in shots:
                    yield fn(shot)
                    if self.fail:
                        raise BrokenProcessPool("worker died after first result")

            return results()

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(mwpm_gpu, "ProcessPoolExecutor", Pool)
    incoming = {k: v.clone() for k, v in io.items()}
    if fail_again:
        with pytest.raises(BrokenProcessPool):
            dec(io)
        assert dec._pool is None
        for key in incoming:
            assert torch.equal(io[key], incoming[key]), key
    else:
        out = dec(io)
        assert dec._pool is pools[1]
        for key in ("e_v", "llr", "iter", "converge"):
            assert torch.equal(out[key].cpu(), ref[key]), key
    assert len(pools) == len(batches) == 2
    assert pools[0].shutdown_calls == [{"wait": False}]
    assert pools[1].shutdown_calls == ([{"wait": False}] if fail_again else [])
    assert len(batches[0]) == len(batches[1]) == (4 if mode == "uniform" else 3)
    for first, second in zip(*batches):
        if mode == "uniform":
            assert np.array_equal(first, second)
        else:
            assert all(np.array_equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("weights", [
    [2], [True] * 8, [-2] * 8, [3] * 8, [[2] * 8],
    [float("nan")] * 8, [float("inf")] * 8,
    [1 << 62] * 8, [1 << 60] * 8,
])
def test_invalid_native_weights(weights):
    with pytest.raises(ValueError):
        mwpm_gpu.NativeMatcher.from_check_matrix(*H.nonzero(), *H.shape, weights=weights)


def test_invalid_weight_mode():
    cfg = _cfg(weights="other")
    with pytest.raises(ValueError):
        mwpm_gpu.create(cfg, bundle=_bundle())


@pytest.mark.parametrize("llr", [
    None, torch.ones(1, 8), torch.ones(16, 7),
    torch.full((16, 8), float("nan")), torch.full((16, 8), float("inf")),
])
@pytest.mark.parametrize("mode", ["posterior", "prior"])
def test_invalid_posterior(llr, mode):
    dec = mwpm_gpu.create(_cfg(weights=mode), bundle=_bundle())
    io = {"synd": torch.from_numpy(SYNDROMES)}
    if llr is not None:
        io["llr0" if mode == "prior" else "llr"] = llr
    with pytest.raises(ValueError):
        dec(io)


@pytest.mark.parametrize("key,value", [
    ("weights", "posterior"),
    ("skip_converged", True),
])
def test_weight_keys_belong_in_config(key, value):
    from syndrilla.decoder.decoder import resolve_configs

    with pytest.raises(ValueError, match="decoding.config"):
        resolve_configs({"algorithm": "mwpm_gpu", key: value})
    assert resolve_configs({"algorithm": "mwpm_gpu", "config": {key: value}})[0][key] == value


def test_bp_mwpm_gpu_chain_example():
    from syndrilla.decoder.decoder import create_decoder
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = read_yaml("examples/alist/bp_mwpm_hx.decoding.yaml")["decoding"]
    cfg["algorithm"][-1] = "mwpm_gpu"
    assert cfg["algorithm"] == ["bp_norm_min_sum", "mwpm_gpu"]
    cfg["config"][0].update(max_iter=2, compile=False)
    cfg["config"][1]["num_workers"] = 1
    matrix = read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"]
    bundle = load_matrices(matrix, *parse_device_dtype(cfg))
    h = bundle.select("hx")[3].to_dense().cpu().double()
    g = torch.Generator().manual_seed(9)
    err = (torch.rand(16, h.shape[1], generator=g) < 0.1).double()
    syndrome = (err @ h.T) % 2
    bp, matching = create_decoder(cfg=cfg, bundle=bundle)
    out = bp({"synd": syndrome.clone(), "llr0": torch.full_like(err, 2.0)})
    assert bool((out["llr"] < 0).any())
    assert not bool(out["converge"].all())
    out = matching(out)
    assert torch.equal(out["e_v"] @ h.T % 2, syndrome)
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    assert bool(out["converge"].all())


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_skip_converged_requires_bool(value):
    with pytest.raises(ValueError, match="skip_converged"):
        mwpm_gpu.create(_cfg(skip_converged=value), bundle=_bundle())


@pytest.mark.parametrize("module", [mwpm_gpu, mwpm_gpu_cuda], ids=["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["uniform", "prior"])
def test_skip_converged_requires_posterior_weights(module, mode):
    with pytest.raises(ValueError, match="requires weights='posterior'"):
        module.create(_cfg(skip_converged=True, weights=mode), bundle=_bundle())


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("all_converged", [False, True])
def test_skip_converged_preserves_rows(backend, all_converged, monkeypatch):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = _cfg(weights="posterior")
    cfg["device"] = {"device_type": backend}
    mod = mwpm_gpu_cuda if backend == "cuda" else mwpm_gpu
    if all_converged:
        monkeypatch.setattr(mwpm_gpu_cuda, "_load_kernel", lambda: pytest.fail("all rows already converged"))
    dec = mod.create(cfg, bundle=_bundle(device=backend))
    indices = [np.flatnonzero(((CANDIDATES @ H.T) % 2 == s).all(1))[0] for s in SYNDROMES]
    incoming = torch.tensor(CANDIDATES[indices], dtype=torch.float64, device=backend)
    posterior = (1 - 2 * incoming) * 3.5
    keep = torch.ones(len(SYNDROMES), dtype=torch.bool, device=backend)
    if not all_converged:
        keep[1::2] = False
        incoming[~keep] = 0
    calls = []
    decode = dec._decode

    def record(io):
        calls.append(len(io["synd"]))
        return decode(io)

    monkeypatch.setattr(dec, "_decode", record)
    out = dec({"synd": torch.tensor(SYNDROMES, device=backend), "llr": posterior.clone(),
               "e_v": incoming.clone(), "converge": keep.long()})
    assert calls == ([] if all_converged else [int((~keep).sum())])
    assert torch.equal(out["e_v"][keep], incoming[keep])
    assert torch.equal(out["llr"][keep], posterior[keep])
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    assert np.array_equal(out["e_v"].cpu().numpy() @ H.T % 2, SYNDROMES)
    assert bool(out["converge"].all())


@pytest.mark.parametrize("mode", ["uniform", "posterior"])
@pytest.mark.parametrize("workers", [1, 2], ids=["sequential", "pool"])
def test_cuda_matches_cpu_surface10(mode, workers, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum_cuda import create as BP
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = {**_cfg(weights=mode), "device": {"device_type": "cuda"}}
    bundle = load_matrices(read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"], *parse_device_dtype(cfg))
    h = bundle.select("hx")[3].to_dense().cpu().double()
    g = torch.Generator().manual_seed(20261005)
    errors = (torch.rand(32, h.shape[1], generator=g) < 0.1).double()
    bp = BP({**cfg, "max_iter": 3, "compile": False}, bundle=bundle)
    posterior = bp({"synd": errors @ h.T % 2, "llr0": torch.full_like(errors, 2.0)})
    assert not bool(posterior["converge"].all())
    cpu = mwpm_gpu.create({**cfg, "device": {"device_type": "cpu"}}, bundle=bundle)
    gpu = mwpm_gpu_cuda.create({**cfg, "num_workers": workers, "mp_min_batch": 1}, bundle=bundle)
    calls = []
    kernel = gpu._kernel or mwpm_gpu_cuda._load_kernel()

    def record(*args):
        calls.append(args[5].shape[0])
        return kernel.mwpm_decode(*args)

    monkeypatch.setattr(gpu, "_kernel", SimpleNamespace(mwpm_decode=record))
    monkeypatch.setattr(gpu, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
    ref = cpu(dict(posterior))
    got = gpu(dict(posterior))
    assert calls == [len(errors)]
    assert (gpu._pool is not None) == (workers > 1)
    for key in ("e_v", "llr", "iter", "converge"):
        assert torch.equal(got[key].cpu(), ref[key].cpu()), key


def _stim_surface5():
    from syndrilla.interface import create_interface

    return create_interface(cfg={
        "backend": "stim", "decompose_errors": True,
        "circuit": {"code": "surface_code:rotated_memory_x", "distance": 5, "rounds": 5,
                    "after_clifford_depolarization": 0.001,
                    "after_reset_flip_probability": 0.001,
                    "before_measure_flip_probability": 0.001,
                    "before_round_data_depolarization": 0.001},
    })


def test_prior_not_worse_than_uniform_decomposed_surface5():
    interface = _stim_surface5()
    bundle = interface.matrix_bundle
    h = torch.tensor(bundle.Hx_matrix.get_dense(), dtype=torch.float64)
    logical = torch.tensor(np.asarray(bundle.lx_matrix), dtype=torch.float64)
    cfg = {**_cfg(), "num_workers": 32, "mp_min_batch": 1}
    decoders = {mode: mwpm_gpu.create({**cfg, "weights": mode}, bundle=bundle)
                for mode in ("uniform", "prior")}
    counts = dict.fromkeys(decoders, 0)
    shots, batch_size = 100000, 10000
    started = time.perf_counter()
    torch.manual_seed(1234)
    try:
        for _ in range(shots // batch_size):
            errors, _ = interface.error_model.inject_error(torch.zeros(batch_size, h.shape[1], dtype=torch.float64))
            syndrome = interface.syndrome_generator.measure_syndrome(errors, None)
            truth = interface.syndrome_generator.observable_flips
            llr0 = interface.error_model.get_llr(errors)
            for mode, decoder in decoders.items():
                out = decoder({"synd": syndrome, "llr0": llr0})
                assert torch.equal(out["e_v"] @ h.T % 2, syndrome)
                predicted = out["e_v"] @ logical.T % 2
                counts[mode] += int((predicted != truth).any(1).sum())
    finally:
        for decoder in decoders.values():
            if decoder._pool is not None:
                decoder._pool.shutdown(wait=True)
                decoder._pool = None
    print(f"Stim d5r5 p=0.001 seed=1234 shots={shots}: "
          f"uniform={counts['uniform']} ({counts['uniform'] / shots:.6f}), "
          f"prior={counts['prior']} ({counts['prior'] / shots:.6f}), "
          f"elapsed={time.perf_counter() - started:.2f}s", flush=True)
    assert counts["prior"] <= counts["uniform"]


def test_cuda_zero_defects_skip_reconstruction(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    h = np.pad(H, ((0, 0), (0, 65 - H.shape[1])))
    cfg = {**_cfg(weights="posterior"), "device": {"device_type": "cuda"},
           "num_workers": 2, "mp_min_batch": 1}
    gpu = mwpm_gpu_cuda.create(cfg, bundle=_bundle(h, "cuda"))
    monkeypatch.setattr(mwpm_gpu.NativeMatcher, "from_check_matrix",
                        lambda *args, **kwargs: pytest.fail("zero defects need no matcher"))
    out = gpu({"synd": torch.zeros(3, len(h)), "llr": torch.ones(3, 65)})
    assert not bool(out["e_v"].any())
    assert gpu._pool is None


def test_cuda_parallel_winner_preserves_original_slots(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    h = np.array([[1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 0, 1]])
    patterns = torch.tensor([[1., 2., 5., 4., 6.], [5., 2., 1., 4., 6.], [2., 1., 2., 4., 6.]])
    syndrome = torch.tensor(list(product((0, 1), repeat=3))).repeat_interleave(3, dim=0)
    io = {"synd": syndrome, "llr": patterns.repeat(8, 1)}
    cpu = mwpm_gpu.create(_cfg(weights="posterior"), bundle=_bundle(h))
    cfg = {**_cfg(weights="posterior"), "device": {"device_type": "cuda"}}
    gpu = mwpm_gpu_cuda.create(cfg, bundle=_bundle(h, "cuda"))
    neighbors, observables, _ = gpu._weighted_csr(patterns.cuda().double())
    assert neighbors[:, :2].tolist() == [[1, 2], [1, 2], [1, 2]]
    assert observables[:, 0, 0].tolist() == [1, 4, 1]
    monkeypatch.setattr(gpu, "_cpu_decode", lambda *args: pytest.fail("unexpected CPU fallback"))
    ref, got = cpu(dict(io)), gpu(dict(io))
    assert torch.equal(got["e_v"].cpu(), ref["e_v"])
    from syndrilla.decoder.mwpm import mwpm

    expected = mwpm.create(_cfg(weights="posterior"), bundle=_bundle(h))(dict(io))
    assert torch.equal(ref["e_v"], expected["e_v"])


@pytest.mark.parametrize("n", [4, 65, 257])
def test_cuda_weighted_parallel_edges_and_fallback(n, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    h = np.pad(np.array([[1, 1, 1, 1], [1, 1, 0, 0]]), ((0, 0), (0, n - 4)))
    llr = torch.ones(3, n, dtype=torch.float64)
    llr[:, :4] = torch.tensor([[4., 1., 4., 1.], [1., 4., -1., 4.], [1., 1., 0., 2.]])
    io = {"synd": torch.tensor([[1, 1], [1, 0], [0, 1]]), "llr": llr}
    cpu = mwpm_gpu.create(_cfg(weights="posterior"), bundle=_bundle(h))
    cfg = {**_cfg(weights="posterior"), "device": {"device_type": "cuda"}}
    gpu = mwpm_gpu_cuda.create(cfg, bundle=_bundle(h, "cuda"))
    ref = cpu(dict(io))
    got = gpu(dict(io))
    assert torch.equal(got["e_v"].cpu(), ref["e_v"])
    if gpu._use_kernel:
        kernel = gpu._kernel

        def fail(*args):
            mask, err, mef, met, count = kernel.mwpm_decode(*args)
            return mask, torch.ones_like(err), mef, met, count

        monkeypatch.setattr(gpu, "_kernel", SimpleNamespace(mwpm_decode=fail))
        got = gpu(dict(io))
        assert torch.equal(got["e_v"].cpu(), ref["e_v"])
        assert torch.equal(got["llr"].cpu(), ref["llr"])


@pytest.mark.parametrize("example", ["mwpm_hx", "bp_mwpm_hx"])
def test_batch_alist_hx(example, tmp_path):
    decoding = yaml.safe_load(Path(f"examples/alist/{example}.decoding.yaml").read_text())
    cfg = decoding["decoding"]
    cfg["algorithm"] = ["mwpm_gpu" if stage == "mwpm" else stage for stage in cfg["algorithm"]]
    cfg["device"] = {"device_type": "cpu"}
    cfg["force_pytorch"] = True
    stages = cfg.setdefault("config", [{}])
    stages[-1].update(num_workers=2, mp_min_batch=1)
    if len(stages) > 1:
        stages[0]["compile"] = False
    decoding_yaml = tmp_path / "decoder.yaml"
    decoding_yaml.write_text(yaml.safe_dump(decoding))
    cmd = [
        "syndrilla",
        f"-r={tmp_path}",
        f"-d={decoding_yaml}",
        "-e=examples/alist/bsc.error.yaml",
        "-c=examples/alist/lx.check.yaml",
        "-s=examples/alist/perfect.syndrome.yaml",
        "-m=examples/alist/surface_10.matrix.yaml",
        "-bs=1000",
        "-te=10",
        "--seed=25",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(f"{example} CLI_RETURN_CODE: {result.returncode}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "result_phy_err_0.1.yaml").exists()


@pytest.mark.parametrize("cuda_initialized", [False, True])
def test_cpu_factory_pool_subprocess(cuda_initialized, tmp_path):
    if cuda_initialized and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    script = textwrap.dedent("""\
        import torch
        from syndrilla.decoder.decoder import create_decoder
        from syndrilla.decoder.mwpm_gpu import mwpm_gpu
        from syndrilla.matrix import load_matrices
        from syndrilla.utils import read_yaml, parse_device_dtype

        def run():
            INITIALIZE_CUDA
            cfg = dict(algorithm='mwpm_gpu', dtype='float64', device=dict(device_type='cpu'),
                       config=dict(weights='posterior', num_workers=2, mp_min_batch=1))
            bundle = load_matrices(read_yaml('examples/alist/surface_10.matrix.yaml')['matrix'],
                                   *parse_device_dtype(cfg))
            h = bundle.select('hx')[3].to_dense().double()
            dec = create_decoder(cfg=cfg, bundle=bundle)[0]
            assert type(dec).__module__ != mwpm_gpu.__name__
            errors = torch.eye(h.shape[1], dtype=torch.float64)[:4]
            synd = errors @ h.T % 2
            try:
                out = dec(dict(synd=synd, llr=torch.full_like(errors, 3.0)))
                assert torch.equal(out['e_v'] @ h.T % 2, synd)
                assert dec._pool._initializer is mwpm_gpu._mp_worker_init
                assert dec._pool._mp_context.get_start_method() == EXPECTED_START
                print('FACTORY_POOL_OK', dec._pool._mp_context.get_start_method())
            finally:
                if dec._pool is not None:
                    dec._pool.shutdown(wait=True)
                    dec._pool = None
        """)
    script = script.replace("INITIALIZE_CUDA", "torch.cuda.init()" if cuda_initialized else "assert not torch.cuda.is_initialized()")
    script = script.replace("EXPECTED_START", repr("forkserver" if cuda_initialized else "fork"))
    script += "\nif __name__ == '__main__':\n    run()\n" if cuda_initialized else "\nrun()\n"
    path = tmp_path / "factory_pool.py"
    path.write_text(script)
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FACTORY_POOL_OK" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
