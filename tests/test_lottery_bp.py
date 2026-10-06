import os
import subprocess
import sys

import pytest


sys.path.append(os.getcwd())


def _surface10_system_outputs(mod, extra=None, batches=((0, 16, 0.03), (1, 16, 0.03))):
    """Per batch: e_v, llr, iter, converge, the CPU and CUDA global RNG states after
    decoding, and cap.frac. Each (seed, B, p) in batches is a BSC batch on
    surface_10 hx, decoded in turn by one decoder with random_machine system
    after torch.manual_seed(7); extra updates the decoder config."""
    import math
    import torch
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = dict(
        device={"device_type": "cuda", "device_idx": 0},
        dtype="float64",
        check_type="hx",
        max_iter=100,
        compile=False,
        random_machine="system",
    )
    cfg.update(extra or {})
    mcfg = read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"]
    bundle = load_matrices(mcfg, *parse_device_dtype(cfg))
    H = bundle.select("hx")[3].to_dense().cpu().double()
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
            out = dec({"synd": synd, "llr0": llr0})
        o = {k: out[k].cpu() for k in ("e_v", "llr", "iter", "converge")}
        o["cpu_rng"] = torch.get_rng_state()
        o["cuda_rng"] = (
            torch.cuda.get_rng_state()
            if torch.cuda.is_available() else torch.empty(0, dtype=torch.uint8)
        )
        o["cap_frac"] = None if dec.cap is None else dec.cap.frac
        outs.append(o)
    return outs


def _assert_same(ref, got):
    import torch

    for b, (r, o) in enumerate(zip(ref, got)):
        for k in r:
            same = r[k] == o[k] if k == "cap_frac" else torch.equal(r[k], o[k])
            assert same, (b, k)


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
@pytest.mark.parametrize(
    "variant", ["", "_quant", "_policy"], ids=["lottery", "quant", "policy"]
)
@pytest.mark.parametrize("rm", ["sobol", "system"])
def test_final_iteration_matches_plain_bp(backend, variant, rm):
    from importlib import import_module
    import torch

    if backend != "cpu":
        _cuda_or_skip()
    suffix = "_cuda" if backend == "cuda" else ""
    name = "bp_lottery" + variant
    lottery = import_module(f"syndrilla.decoder.{name}.{name}{suffix}")
    name = "bp_norm_min_sum" + ("_quant" if variant == "_quant" else "")
    plain = import_module(f"syndrilla.decoder.{name}.{name}{suffix}")
    cfg = dict(
        device={"device_type": "cpu" if backend == "cpu" else "cuda"},
        max_iter=1 if variant == "_policy" else 3,
        flip_start_iter=2,
        random_machine=rm,
    )
    if variant == "_quant":
        cfg.update(int_width=3, frac_width=4)
    batches = ((0, 16, 0.1),)
    out = _surface10_system_outputs(lottery, cfg, batches)[0]
    ref = _surface10_system_outputs(plain, cfg, batches)[0]
    assert not bool(out["converge"].all())
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    _assert_same([ref], [out])


def test_system_pytorch_matches_cuda():
    """With random_machine system and the same seed, the PyTorch and CUDA paths
    give equal outputs and global RNG states on two batches in a row."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    _assert_same(
        _surface10_system_outputs(bp_lottery),
        _surface10_system_outputs(bp_lottery_cuda),
    )


def test_system_pytorch_matches_cuda_cap_chosen_on_converged_batch():
    """Same as above past the rebatch cap warm-up: the cap is chosen while
    decoding a fully converged batch and applies to the batches after it."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    extra = dict(
        max_iter=60,
        rebatch_opt=True,
        rebatch_opt_params=dict(
            kl_eps=1e9, kl_window=1, kl_min=2, min_speedup=0.0, min_pct=50
        ),
    )
    batches = (
        (0, 96, 0.03),
        (5, 16, 0.01),
        (3, 48, 0.05),
        (2, 32, 0.2),
        (5, 16, 0.01),
        (4, 80, 0.04),
    )
    ref = _surface10_system_outputs(bp_lottery, extra, batches)
    got = _surface10_system_outputs(bp_lottery_cuda, extra, batches)
    assert ref[0]["cap_frac"] is None and ref[1]["cap_frac"] is not None
    assert bool(ref[1]["converge"].all())
    assert int(ref[1]["iter"].max()) > 4
    _assert_same(ref, got)


def test_quant_system_pytorch_matches_cuda():
    """bp_lottery_quant with random_machine system: equal outputs and global
    RNG states on three batches: the first fully converged by iteration 4 (no
    flip, so neither path draws), the second fully converged after iteration 4
    (the last draw), the third not fully converged."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from syndrilla.decoder.bp_lottery_quant import (
        bp_lottery_quant,
        bp_lottery_quant_cuda,
    )

    extra = dict(max_iter=60, rebatch_opt=False, int_width=3, frac_width=4)
    batches = ((11, 8, 0.002), (0, 8, 0.01), (0, 64, 0.03))
    ref = _surface10_system_outputs(bp_lottery_quant, extra, batches)
    got = _surface10_system_outputs(bp_lottery_quant_cuda, extra, batches)
    assert bool(ref[0]["converge"].all()) and int(ref[0]["iter"].max()) <= 4
    assert bool(ref[1]["converge"].all()) and int(ref[1]["iter"].max()) > 4
    assert not bool(ref[2]["converge"].all())
    _assert_same(ref, got)


_INTERVAL_BATCHES = (
    (11, 8, 0.002),
    (0, 8, 0.01),
    (34, 2, 0.03),
    (3, 4, 0.02),
    (25, 4, 0.02),
    (0, 64, 0.03),
)


def _assert_interval_cases(out, k):
    """out holds a batch fully converged at a flip iteration past 4 (where the
    CUDA path makes the extra system draw), one fully converged at another
    iteration past 4, and one not fully converged."""
    import types
    from syndrilla.decoder.bp_lottery.bp_lottery import is_flip_iter

    d = types.SimpleNamespace(flip_start_iter=4, flip_interval=k, max_iter=60)
    ends = [int(o["iter"].max()) for o in out if bool(o["converge"].all())]
    assert any(t > 4 and is_flip_iter(d, t) for t in ends), ends
    assert any(t > 4 and not is_flip_iter(d, t) for t in ends), ends
    assert len(ends) < len(out)


def _cuda_or_skip():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_flip_interval_1_matches_default():
    """flip_interval 1 gives the outputs and RNG states of a config without the
    key, on both paths."""
    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    for mod in (bp_lottery, bp_lottery_cuda):
        for rm in ("sobol", "system"):
            _assert_same(
                _surface10_system_outputs(mod, dict(random_machine=rm)),
                _surface10_system_outputs(
                    mod, dict(random_machine=rm, flip_interval=1)
                ),
            )


def test_flip_interval_2_flip_iterations():
    """With flip_start_iter 4 and flip_interval 2, each path flips at
    iterations 5, 7, ..., 17 of max_iter 19 on a batch that does not fully
    converge, and on no other iteration."""
    import types

    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    for mod in (bp_lottery, bp_lottery_cuda):
        seen = []

        class Rec(mod.create):
            def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
                seen.append(self.i)
                return super().sign_flip_cn_rand_new(syndrome, s_est, l_v)

        out = _surface10_system_outputs(
            types.SimpleNamespace(create=Rec),
            dict(max_iter=19, flip_start_iter=4, flip_interval=2),
            ((0, 16, 0.1),),
        )
        assert not bool(out[0]["converge"].all())
        assert seen == list(range(5, 19, 2)), (mod.__name__, seen)


def test_flip_interval_pytorch_matches_cuda():
    """flip_interval 2 and 3, sobol and system: equal outputs and global RNG
    states on both paths, over batches that fully converge at a flip
    iteration, at another iteration, and not at all."""
    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    for k in (2, 3):
        for rm in ("sobol", "system"):
            extra = dict(
                max_iter=60, rebatch_opt=False, random_machine=rm, flip_interval=k
            )
            ref = _surface10_system_outputs(bp_lottery, extra, _INTERVAL_BATCHES)
            got = _surface10_system_outputs(bp_lottery_cuda, extra, _INTERVAL_BATCHES)
            _assert_interval_cases(ref, k)
            _assert_same(ref, got)


def test_quant_flip_interval_2_flip_iterations():
    """bp_lottery_quant with flip_start_iter 4 and flip_interval 2: each path
    flips at iterations 5, 7, ..., 17 of max_iter 19 on a batch that does not
    fully converge, and on no other iteration."""
    import types

    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery_quant import (
        bp_lottery_quant,
        bp_lottery_quant_cuda,
    )

    for mod in (bp_lottery_quant, bp_lottery_quant_cuda):
        seen = []

        class Rec(mod.create):
            def sign_flip(self, syndrome, s_est, l_v, active):
                seen.append(self.i)
                return super().sign_flip(syndrome, s_est, l_v, active)

        out = _surface10_system_outputs(
            types.SimpleNamespace(create=Rec),
            dict(
                max_iter=19,
                flip_start_iter=4,
                flip_interval=2,
                int_width=3,
                frac_width=4,
            ),
            ((0, 16, 0.1),),
        )
        assert not bool(out[0]["converge"].all())
        assert seen == list(range(5, 19, 2)), (mod.__name__, seen)


def test_quant_flip_interval_pytorch_matches_cuda():
    """bp_lottery_quant with flip_interval 2, sobol and system: equal outputs
    and global RNG states on both paths."""
    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery_quant import (
        bp_lottery_quant,
        bp_lottery_quant_cuda,
    )

    for rm in ("sobol", "system"):
        extra = dict(
            max_iter=60,
            rebatch_opt=False,
            int_width=3,
            frac_width=4,
            random_machine=rm,
            flip_interval=2,
        )
        ref = _surface10_system_outputs(bp_lottery_quant, extra, _INTERVAL_BATCHES)
        got = _surface10_system_outputs(bp_lottery_quant_cuda, extra, _INTERVAL_BATCHES)
        _assert_interval_cases(ref, 2)
        _assert_same(ref, got)


def test_flip_interval_bad_values_raise():
    """A flip_interval that is not an int >= 1 raises ValueError on both
    paths; the message shows a string value with its quotes."""
    import pytest

    _cuda_or_skip()
    from syndrilla.decoder.bp_lottery import bp_lottery, bp_lottery_cuda

    for mod in (bp_lottery, bp_lottery_cuda):
        for v in (0, -1, 1.5, True, "2"):
            with pytest.raises(ValueError, match="^flip_interval must"):
                _surface10_system_outputs(mod, dict(flip_interval=v), ())
        with pytest.raises(ValueError, match="got <'2'>"):
            _surface10_system_outputs(mod, dict(flip_interval="2"), ())


def test_batch_alist_hz(batch_size=1000, target_error=1000):
    decoding_yaml = "examples/alist/lottery_bp_hz.decoding.yaml"
    logical_check_yaml = "examples/alist/lz.check.yaml"
    cmd = [
        "syndrilla",
        "-r=tests/test_outputs",
        f"-d={decoding_yaml}",
        "-e=examples/alist/bsc.error.yaml",
        f"-c={logical_check_yaml}",
        "-s=examples/alist/perfect.syndrome.yaml",
        f"-bs={batch_size}",
        f"-te={target_error}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)


def test_batch_alist_hz_quant(batch_size=1000, target_error=1000):
    decoding_yaml = "examples/alist/lottery_bp_quant_hz.decoding.yaml"
    logical_check_yaml = "examples/alist/lz.check.yaml"
    cmd = [
        "syndrilla",
        "-r=tests/test_outputs",
        f"-d={decoding_yaml}",
        "-e=examples/alist/bsc.error.yaml",
        f"-c={logical_check_yaml}",
        "-s=examples/alist/perfect.syndrome.yaml",
        f"-bs={batch_size}",
        f"-te={target_error}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)


if __name__ == "__main__":
    batch_size = 100000
    target_error = 1000
    test_batch_alist_hz(batch_size, target_error)
    test_batch_alist_hz_quant(batch_size, target_error)
