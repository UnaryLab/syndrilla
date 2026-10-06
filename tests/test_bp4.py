import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from syndrilla.decoder.bp4 import bp4, bp4_cuda
from syndrilla.decoder.decoder import RebatchSpeedup
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype, read_yaml

sys.path.append(os.getcwd())


def _bundle(cfg):
    matrix = read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"]
    return load_matrices(matrix, *parse_device_dtype(cfg))


@pytest.mark.parametrize("backend", ["cpu", "pytorch", "cuda"])
def test_bp4_returns_joint_hard_and_sector_marginals(backend, monkeypatch):
    if backend != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = dict(device={"device_type": "cpu" if backend == "cpu" else "cuda"},
               dtype="float64", max_iter=1, compile=False, rebatch_opt=False)
    bundle = _bundle(cfg)
    hx, hz = [bundle.select(k)[3].to_dense().cpu().double() for k in ("hx", "hz")]
    B, N = 16, hx.shape[1]
    prior = torch.tensor([0.88, 0.04, 0.03, 0.05], dtype=torch.float64)
    pauli = torch.multinomial(prior, B * N, replacement=True,
                              generator=torch.Generator().manual_seed(25)).reshape(B, N)
    pauli[0] = 0
    x, z = ((pauli == 1) | (pauli == 2)).double(), ((pauli == 2) | (pauli == 3)).double()
    io = {"synd": torch.stack((z @ hx.T % 2, x @ hz.T % 2), 1),
          "llr0": prior[None, :, None].expand(B, 4, N).clone()}
    snapshots = []
    step = bp4._step

    def record(*args):
        result = step(*args)
        snapshots.append((result[1][:, :, :N].clone(), result[3][:, :, :N].clone()))
        return result

    monkeypatch.setattr(bp4, "_step", record)
    mod = bp4_cuda if backend == "cuda" else bp4
    out = mod.create(cfg, bundle=bundle)(dict(io))
    posterior, hard = snapshots[0]
    I, X, Y, Z = posterior.unbind(1)
    expected = torch.stack((torch.log(I + X) - torch.log(Z + Y),
                            torch.log(I + Z) - torch.log(X + Y)), 1)
    assert out["llr"].shape == (B, 2, N)
    assert bool(torch.isfinite(out["llr"]).all())
    torch.testing.assert_close(out["llr"], expected)
    assert torch.equal(out["e_v"], hard)
    assert bool(out["e_v"].any())
    good = out["converge"].cpu() == 1
    assert bool(good.any()) and not bool(good.all())
    e = out["e_v"].cpu()
    assert torch.equal(e[good, 0] @ hx.T % 2, io["synd"][good, 0])
    assert torch.equal(e[good, 1] @ hz.T % 2, io["synd"][good, 1])
    if backend != "cpu":
        cpu_cfg = {**cfg, "device": {"device_type": "cpu"}}
        ref = bp4.create(cpu_cfg, bundle=_bundle(cpu_cfg))(dict(io))
        for key in ("e_v", "iter", "converge"):
            assert torch.equal(out[key].cpu(), ref[key]), key
        torch.testing.assert_close(out["llr"].cpu(), ref["llr"])
    if backend == "cuda":
        ref = bp4.create(cfg, bundle=bundle)(dict(io))
        for key in ("e_v", "llr", "iter", "converge"):
            assert torch.equal(out[key], ref[key]), key


def test_bp4_joint_decision_can_disagree_with_marginal_sign(monkeypatch):
    cfg = dict(device={"device_type": "cpu"}, dtype="float64", max_iter=1,
               compile=False, rebatch_opt=False)
    dec = bp4.create(cfg, bundle=_bundle(cfg))
    M, N = dec.H_shape
    prior = torch.tensor([0.4, 0.05, 0.3, 0.25], dtype=torch.float64)

    def step(message, old, chan, *args):
        posterior = prior[None, :, None].expand_as(chan)
        choice = posterior.argmax(1)
        hard = torch.stack(((choice == 2) | (choice == 3),
                            (choice == 1) | (choice == 2)), 1)
        return message, posterior, torch.ones(1, dtype=torch.bool), hard

    monkeypatch.setattr(bp4, "_step", step)
    out = dec({"synd": torch.zeros(1, 2, M),
               "llr0": prior[None, :, None].expand(1, 4, N)})
    assert bool(out["converge"].all())
    assert not bool(out["e_v"].any())
    expected = torch.tensor([0.45 / 0.55, 0.65 / 0.35], dtype=torch.float64).log()
    torch.testing.assert_close(out["llr"], expected[None, :, None].expand(1, 2, N))
    assert bool((out["llr"][:, 0] < 0).all())


@pytest.mark.parametrize("capped", [False, True])
def test_bp4_snapshots_at_each_rows_stop(capped, monkeypatch):
    cfg = dict(device={"device_type": "cpu"}, dtype="float64", max_iter=3,
               compile=False, rebatch_opt=False)
    bundle = _bundle(cfg)
    dec = bp4.create(cfg, bundle=bundle)
    if capped:
        dec.cap = RebatchSpeedup()
        dec.cap.frac = 0.25
    B, (M, N) = 4, dec.H_shape
    prior = torch.empty(B, 4, N, dtype=torch.float64)
    for row in range(B):
        prior[row] = (1 - (row + 1) / 8) / 3
        prior[row, 0] = (row + 1) / 8
    calls, expected = [], {}

    def step(message, old, chan, *args):
        iteration = len(calls) + 1
        ids = (chan[:, 0, 0] * 8).long() - 1
        calls.append(ids.tolist())
        choice = (ids + iteration) % 4
        peak = 0.55 + 0.1 * iteration
        posterior = torch.full_like(chan, (1 - peak) / 3)
        posterior.scatter_(1, choice[:, None, None].expand(-1, 1, N + 1), peak)
        hard = torch.stack(((choice == 2) | (choice == 3),
                            (choice == 1) | (choice == 2)), 1)[:, :, None].expand(-1, -1, N + 1)
        converged = (ids == 0) if iteration == 1 else ((ids == 1) | (ids == 2))
        for row, ident in enumerate(ids.tolist()):
            if ident not in expected:
                if capped or bool(converged[row]) or iteration == 3:
                    expected[ident] = (posterior[row, :, :N].clone(), hard[row, :, :N].clone())
        return message, posterior, converged, hard

    monkeypatch.setattr(bp4, "_step", step)
    out = dec({"synd": torch.zeros(B, 2, M), "llr0": prior})
    assert calls == ([[0, 1, 2, 3]] if capped else [[0, 1, 2, 3], [0, 1, 2, 3], [3]])
    assert out["iter"].tolist() == ([1, 1, 1, 1] if capped else [1, 2, 2, 3])
    for row, (posterior, hard) in expected.items():
        I, X, Y, Z = posterior
        llr = torch.stack((torch.log(I + X) - torch.log(Z + Y),
                           torch.log(I + Z) - torch.log(X + Y)))
        assert torch.equal(out["e_v"][row], hard)
        torch.testing.assert_close(out["llr"][row], llr)


def test_bp4(batch_size=10000):
    with tempfile.TemporaryDirectory(prefix="syndrilla-bp4-") as tmp:
        run = Path(tmp)
        decoding = yaml.safe_load(Path("examples/alist/bp4.decoding.yaml").read_text())
        cfg = decoding["decoding"]
        cfg["device"] = {"device_type": "cuda" if torch.cuda.is_available() else "cpu"}
        cfg["force_pytorch"] = True
        cfg["config"].update(max_iter=181, damping_factor=0.1, compile=False, rebatch_opt=False)
        error = yaml.safe_load(Path("examples/alist/depol.error.yaml").read_text())
        error["error"]["rate"] = 0.01
        (run / "decoder.yaml").write_text(yaml.safe_dump(decoding))
        (run / "error.yaml").write_text(yaml.safe_dump(error))
        cmd = [
            'syndrilla',
            f'-r={run}',
            f'-d={run / "decoder.yaml"}',
            f'-e={run / "error.yaml"}',
            '-c=examples/alist/lx.check.yaml',
            '-s=examples/alist/perfect.syndrome.yaml',
            '-m=examples/alist/surface_5.matrix.yaml',
            f'-bs={batch_size}',
            '-tb=1',
            '--seed=25',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print('CLI_RETURN_CODE:', result.returncode)
        assert result.returncode == 0, result.stderr
        metrics = read_yaml(str(run / "result_phy_err_0.01.yaml"))
        rates = {sector: metrics["decoder_full"][sector]["logical error rate"]
                 for sector in ("hx", "hz")}
        convergence = 1 - metrics["decoder_0"]["hx"]["syndrome frame error rate"]
        print('surface5 p=0.01 seed=25:', rates, 'convergence:', convergence)
        # Current seed-25 LER is 0.1066 in both sectors; convergence is 0.8934.
        assert all(rate <= (1 - convergence) + 0.005 for rate in rates.values()), rates
        assert convergence > 0.85, convergence


if __name__ == '__main__':
    test_bp4()
