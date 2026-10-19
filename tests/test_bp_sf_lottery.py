import math
import random
import subprocess
import tempfile
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml


def _surface10(backend, algorithm="bp_sf_lottery", **extra):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    suffix = "_cuda" if backend == "cuda" else ""
    mod = import_module(f"syndrilla.decoder.{algorithm}.{algorithm}{suffix}")
    cfg = dict(
        device={"device_type": "cpu" if backend == "cpu" else "cuda"},
        dtype="float64",
        check_type="hx",
        max_iter=12,
        compile=False,
        random_machine="sobol",
        flip_phase_iters=[2], flip_mode=["off", "lottery"],
    )
    cfg.update(extra)
    if algorithm == "bp_sf":
        for key in ("random_machine", "flip_phase_iters", "flip_mode"):
            cfg.pop(key, None)
    matrix_cfg = read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"]
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(cfg))
    H = bundle.select("hx")[3].to_dense().cpu().double()
    g = torch.Generator().manual_seed(0)
    err = (torch.rand(32, H.shape[1], generator=g) < 0.1).double()
    synd = ((err @ H.T) % 2).to(torch.uint8)
    llr0 = torch.full_like(err, math.log(9))
    return mod.create(cfg, bundle=bundle), H, {"synd": synd, "llr0": llr0}, err


def _surface10_outputs(backend):
    outputs = []
    for w_max in (0, 1):
        sf = dict(w_min=1, w_max=w_max, topk=4, n_sample=4)
        decoder, H, io, _ = _surface10(backend, sf=sf)
        assert decoder.algo == "bp_sf_lottery"
        assert decoder.cap is None
        random.seed(0)
        with torch.no_grad():
            out = decoder(io)
        assert decoder.cap is None
        converged = out["converge"].cpu().bool()
        decoded_synd = (out["e_v"].cpu().double() @ H.T) % 2
        assert torch.equal(decoded_synd[converged], io["synd"][converged])
        keys = ("e_v", "converge", "iter", "iter_serial", "iter_parallel")
        outputs.append({k: out[k].cpu() for k in keys})
    return outputs


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_sf_preserves_convergence_and_syndrome(backend):
    baseline, sf = _surface10_outputs(backend)
    assert bool(baseline["converge"].any()) and not bool(baseline["converge"].all())
    assert int(sf["converge"].sum()) > int(baseline["converge"].sum())


def test_pytorch_matches_cuda():
    ref = _surface10_outputs("pytorch")
    got = _surface10_outputs("cuda")
    for baseline, native in zip(ref, got):
        for key in baseline:
            assert torch.equal(baseline[key], native[key]), key


def test_bp_sf_pytorch_matches_cuda_main_iterations():
    outputs = []
    for backend in ("pytorch", "cuda"):
        decoder, _, io, _ = _surface10(
            backend, "bp_sf", sf=dict(w_min=1, w_max=2, topk=4, n_sample=3)
        )
        random.seed(0)
        with torch.no_grad():
            outputs.append(decoder(io)["iter"].cpu())
    assert torch.equal(*outputs)


@pytest.mark.parametrize("algorithm", ["bp_sf", "bp_sf_lottery"])
@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_iteration_counts_recount_each_trial(algorithm, backend):
    from syndrilla.decoder.bp_sf.bp_sf import sample_n_choose_k

    decoder, H, io, _ = _surface10(
        backend, algorithm, sf=dict(w_min=1, w_max=2, topk=4, n_sample=3)
    )
    own_osc = torch.zeros(32, H.shape[1] + 1, dtype=torch.long, device=decoder.device)
    previous = torch.zeros_like(own_osc, dtype=torch.bool)
    main = {}
    hook, postprocess = decoder._iter_hook, decoder._sf_postprocess

    def record_hook(i, llr, hard, active, syndrome):
        if not main:
            rows = (torch.arange(32, device=decoder.device)
                    if backend == "cuda" else decoder._hook_rows)
            own_osc[rows] += hard.bool() != previous[rows]
            previous[rows] = hard.bool()
        hook(i, llr, hard, active, syndrome)

    def record_main(syndrome, llr, e_v, final_llr, converge, counts, osc):
        assert torch.equal(osc, own_osc[:, :-1])
        main.update(iter=counts.clone(), converge=converge.clone(), e_v=e_v.clone(), llr=final_llr.clone())
        return postprocess(syndrome, llr, e_v, final_llr, converge, counts, osc)

    decoder._iter_hook, decoder._sf_postprocess = record_hook, record_main
    random.seed(1)
    with torch.no_grad():
        out = decoder(dict(io))
    random.seed(1)
    candidates = torch.argsort(own_osc[:, :-1], dim=1, stable=True)[:, -4:].cpu()
    serial, parallel = main["iter"].cpu().clone(), main["iter"].cpu().clone()
    expected_e = main["e_v"].cpu().clone()
    expected_llr = main["llr"].cpu().clone()
    expected_conv = main["converge"].cpu().clone()
    winners, draws = [], []
    for shot in torch.where(main["converge"].cpu() == 0)[0].tolist():
        combos = [slots for weight in (1, 2)
                  for slots in sample_n_choose_k(range(4), weight, 3)]
        draws.append(tuple(combos))
        elapsed, winner = int(serial[shot]), None
        trial_counts, successes = [], []
        for k, slots in enumerate(combos):
            bits = candidates[shot, list(slots)]
            trial_synd = (io["synd"][shot].double() + H[:, bits].sum(1)) % 2
            with torch.no_grad():
                estimate, llr, conv, count, _ = decoder._bp_core(
                    trial_synd[None], io["llr0"][shot:shot + 1]
                )
            trial_counts.append(int(count[0]))
            if bool(conv[0]):
                successes.append((int(count[0]), k, estimate[0].cpu().clone(), llr[0].cpu().clone(), bits))
            if winner is None:
                elapsed += int(count[0])
                if bool(conv[0]):
                    winner = k
                    serial[shot] = elapsed
        if winner is None:
            serial[shot] = elapsed
            parallel[shot] = int(main["iter"][shot]) + max(trial_counts)
        else:
            count, _, estimate, llr, bits = min(successes, key=lambda trial: trial[:2])
            parallel[shot] = int(main["iter"][shot]) + count
            expected_e[shot], expected_llr[shot] = estimate, llr
            expected_e[shot, bits] = 1 - expected_e[shot, bits]
            expected_conv[shot] = 1
        winners.append(winner)
    assert bool(main["converge"].any())
    assert any(k is not None and k > 0 for k in winners) and None in winners
    assert draws[0] != draws[1]
    assert torch.equal(out["iter"], main["iter"])
    assert torch.equal(out["e_v"].cpu(), expected_e)
    assert torch.equal(out["llr"].cpu(), expected_llr)
    assert torch.equal(out["converge"].cpu(), expected_conv)
    for key, expected in (("iter_serial", serial), ("iter_parallel", parallel)):
        assert out[key].dtype == torch.long and out[key].shape == (32,)
        assert out[key].device == out["iter"].device
        assert torch.equal(out[key].cpu(), expected), key


def test_parallel_winner_is_fastest_success_and_ties_use_first():
    from syndrilla.decoder.bp_sf.bp_sf import create

    def retry(syndrome, llr):
        expected = torch.eye(5, dtype=torch.float64)[[4, 1, 3]].repeat(4, 1)
        assert torch.equal(syndrome, expected)
        assert torch.equal(llr, torch.ones_like(llr))
        return (torch.zeros(12, 5, dtype=torch.float64),
                torch.arange(1, 13, dtype=torch.float64)[:, None].expand(-1, 5),
                torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]),
                torch.tensor([8, 3, 3, 10, 10, 10, 10, 10, 10, 10, 5, 2]), None)

    decoder = SimpleNamespace(
        H_shape=(5, 5), topk=3, w_min=1, w_max=1, n_sample=3, max_iter=10,
        device=torch.device("cpu"), dtype=torch.float64,
        V_v_row=torch.arange(5)[:, None], _bp_core=retry,
    )
    e_v, llr = torch.zeros(5, 5, dtype=torch.float64), torch.full((5, 5), -1.0, dtype=torch.float64)
    converged, counts = torch.tensor([1, 0, 0, 0, 0]), torch.tensor([3, 10, 10, 10, 10])
    osc = torch.tensor([[1, 8, 2, 9, 4]]).expand(5, -1)
    serial, parallel = create._sf_postprocess(
        decoder, torch.zeros(5, 5, dtype=torch.float64), torch.ones(5, 5, dtype=torch.float64),
        e_v, llr, converged, counts, osc,
    )
    assert serial.tolist() == [3, 18, 30, 40, 25]
    assert parallel.tolist() == [3, 13, 20, 20, 12]
    assert counts.tolist() == [3, 10, 10, 10, 10]
    assert converged.tolist() == [1, 1, 1, 0, 1]
    assert llr[:, 0].tolist() == [-1, 2, 5, -1, 12]
    expected_e = torch.zeros_like(e_v)
    expected_e[1, 1] = expected_e[2, 1] = expected_e[4, 3] = 1
    assert torch.equal(e_v, expected_e)


def test_candidate_boundary_ties_match_cpu_and_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from syndrilla.decoder.bp_sf.bp_sf import create

    selected = []

    def retry(syndrome, llr):
        selected.append(syndrome.nonzero(as_tuple=True)[1].cpu().tolist())
        return torch.zeros_like(llr), llr, syndrome.new_zeros(2, dtype=torch.long), syndrome.new_full((2,), 6, dtype=torch.long), None

    for device in ("cpu", "cuda"):
        decoder = SimpleNamespace(
            H_shape=(6, 6), topk=2, w_min=1, w_max=1, n_sample=2, max_iter=6,
            device=torch.device(device), dtype=torch.float64,
            V_v_row=torch.arange(6, device=device)[:, None], _bp_core=retry,
        )
        zeros = torch.zeros(1, 6, dtype=torch.float64, device=device)
        create._sf_postprocess(
            decoder, zeros, zeros, zeros.clone(), zeros.clone(),
            torch.zeros(1, dtype=torch.long, device=device), torch.full((1,), 6, device=device),
            torch.tensor([[2, 5, 5, 4, 5, 1]], device=device),
        )
    assert selected == [[2, 4], [2, 4]]


def test_unconverged_shots_draw_their_own_combinations():
    from syndrilla.decoder.bp_sf.bp_sf import create, sample_n_choose_k

    seen = []

    def retry(syndrome, llr):
        seen.append(syndrome.clone())
        assert torch.equal(llr, torch.ones_like(llr))
        return torch.zeros(4, 4), torch.ones(4, 4), torch.zeros(4, dtype=torch.long), torch.full((4,), 6), None

    decoder = SimpleNamespace(
        H_shape=(4, 4), topk=4, w_min=2, w_max=2, n_sample=2, max_iter=6,
        device=torch.device("cpu"), dtype=torch.float32,
        V_v_row=torch.arange(4)[:, None], _bp_core=retry,
    )
    random.seed(0)
    draws = [sample_n_choose_k(range(4), 2, 2) for _ in range(2)]
    assert set(draws[0]) != set(draws[1])
    ascending_candidates = torch.tensor([0, 2, 3, 1])
    expected = []
    for combos in draws:
        for slots in combos:
            shift = torch.zeros(4)
            shift[ascending_candidates[list(slots)]] = 1
            expected.append(shift)
    random.seed(0)
    create._sf_postprocess(
        decoder, torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 4), torch.ones(2, 4),
        torch.zeros(2, dtype=torch.long), torch.full((2,), 6), torch.tensor([[1, 4, 2, 3]]).expand(2, -1),
    )
    assert len(seen) == 1 and torch.equal(seen[0], torch.stack(expected))


@pytest.mark.parametrize("case", ["disabled", "converged", "no_combinations"])
def test_sf_early_returns_independent_iteration_clones(case):
    from syndrilla.decoder.bp_sf.bp_sf import create

    decoder = SimpleNamespace(w_max=0 if case == "disabled" else 1,
                              H_shape=(1, 3), topk=3, w_min=1, n_sample=0,
                              device=torch.device("cpu"))
    counts = torch.tensor([3, 4, 5, 6])
    converge = torch.ones(4, dtype=torch.long) if case == "converged" else torch.zeros(4, dtype=torch.long)
    serial, parallel = create._sf_postprocess(
        decoder, None, None, None, None, converge, counts, torch.zeros(4, 3),
    )
    assert torch.equal(serial, counts) and torch.equal(parallel, counts)
    assert len({counts.data_ptr(), serial.data_ptr(), parallel.data_ptr()}) == 3


@pytest.mark.parametrize("algorithm", ["bp_sf", "bp_sf_lottery"])
@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_disabled_sf_iteration_clones(algorithm, backend):
    decoder, _, io, _ = _surface10(backend, algorithm, sf=dict(w_max=0))
    random.seed(0)
    with torch.no_grad():
        out = decoder(io)
    keys = ("iter", "iter_serial", "iter_parallel")
    assert len({out[key].data_ptr() for key in keys}) == 3
    for key in keys[1:]:
        assert torch.equal(out[key], out["iter"])
    out["iter_serial"][0] += 1
    assert torch.equal(out["iter_parallel"], out["iter"])
    out["iter_parallel"][1] += 1
    assert out["iter_serial"][1] == out["iter"][1]


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_plain_lottery_has_no_sf_iteration_keys(backend):
    decoder, _, io, _ = _surface10(backend, "bp_lottery")
    random.seed(0)
    with torch.no_grad():
        out = decoder(io)
    assert "iter_serial" not in out and "iter_parallel" not in out


def _cli_result(decoding_yaml, run_dir, check="lx", batch_size=32):
    result = subprocess.run(
        [
            "syndrilla",
            f"-r={run_dir}",
            f"-d={decoding_yaml}",
            "-e=examples/alist/bsc.error.yaml",
            f"-c=examples/alist/{check}.check.yaml",
            "-s=examples/alist/perfect.syndrome.yaml",
            "-m=examples/alist/surface_10.matrix.yaml",
            f"-bs={batch_size}",
            "-tb=1",
            "--seed=0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(Path(run_dir).glob("result_phy_err_*.yaml"))
    assert len(paths) == 1
    return yaml.safe_load(paths[0].read_text())


@pytest.mark.parametrize("check,logical", [("hx", "lx"), ("hz", "lz")])
def test_example_cli(check, logical):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with tempfile.TemporaryDirectory() as run_dir:
        _cli_result(f"examples/alist/bp_sf_lottery_{check}.decoding.yaml", run_dir, logical)


def _bounded_cli(root, algorithm, w_max, batch_size=32):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        algorithm=algorithm, device={"device_type": "cpu"}, dtype="float64", check_type="hx",
        config=dict(max_iter=12, compile=False, random_machine="sobol", flip_phase_iters=[2], flip_mode=["off", "lottery"],
                    sf=dict(w_min=1, w_max=w_max, topk=4, n_sample=3)),
    )
    if algorithm == "bp_lottery":
        cfg["config"].pop("sf")
    if algorithm == "bp_sf":
        for key in ("random_machine", "flip_phase_iters", "flip_mode"):
            cfg["config"].pop(key)
    path = root / "decoding.yaml"
    path.write_text(yaml.safe_dump({"decoding": cfg}))
    return _cli_result(path, root, batch_size=batch_size)


def _assert_histograms(entry, shots):
    assert entry["sample count"] == shots
    for suffix in ("serial", "parallel"):
        hist = entry[f"iteration count {suffix}"]
        assert sum(hist) == shots and hist[-1] > 0
        mean = sum((i + 1) * count for i, count in enumerate(hist)) / shots
        assert entry[f"average iteration {suffix}"] == pytest.approx(mean)
        assert f"iteration distribution {suffix}" in entry


@pytest.mark.parametrize("algorithm", ["bp_sf", "bp_sf_lottery", "bp_lottery"])
def test_cli_saves_optional_iteration_metrics(algorithm):
    with tempfile.TemporaryDirectory() as root:
        entry = _bounded_cli(root, algorithm, 2)["decoder_0"]
    keys = {f"{name} {suffix}" for suffix in ("serial", "parallel")
            for name in ("average iteration", "iteration distribution", "iteration count")}
    if algorithm == "bp_lottery":
        assert not keys.intersection(entry)
    else:
        assert keys.issubset(entry)
        _assert_histograms(entry, 32)


def test_pool_actual_sf_yamls_pads_histograms():
    from itertools import zip_longest
    from syndrilla.parallel import pool_results

    with tempfile.TemporaryDirectory() as root:
        results = [_bounded_cli(Path(root) / "enabled", "bp_sf", 2),
                   _bounded_cli(Path(root) / "disabled", "bp_sf", 0, batch_size=16)]
    entries = [result["decoder_0"] for result in results]
    assert len(entries[0]["iteration count serial"]) != len(entries[1]["iteration count serial"])
    pooled = pool_results(results)["decoder_0"]
    _assert_histograms(pooled, 48)
    for suffix in ("serial", "parallel"):
        key = f"iteration count {suffix}"
        expected = [sum(column) for column in zip_longest(*(e[key] for e in entries), fillvalue=0)]
        assert pooled[key] == expected
        mean = sum(e[f"average iteration {suffix}"] * sum(e[key]) for e in entries) / sum(expected)
        assert pooled[f"average iteration {suffix}"] == pytest.approx(mean)
