import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_norm_min_sum_layered import bp_norm_min_sum_layered as layered
from syndrilla.decoder.bp_norm_min_sum_layered import bp_norm_min_sum_layered_cuda as layered_cuda
from syndrilla.decoder.bp_norm_min_sum import bp_norm_min_sum as flooding
from syndrilla.matrix import MatrixBundle, load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = Path(__file__).resolve().parents[1]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
KEYS = ("e_v", "llr", "converge", "iter")


def cfg(device="cuda", dtype="float64", **kwargs):
    return dict(device={"device_type": device, "device_idx": 0}, dtype=dtype,
                check_type="hx", max_iter=181, rebatch_opt=False, compile=False, **kwargs)


@pytest.fixture(scope="module")
def surface():
    matrix = read_yaml(str(ROOT / "examples/alist/surface_10.matrix.yaml"))["matrix"]
    for value in matrix.values():
        if isinstance(value, dict) and "path" in value:
            value["path"] = str(ROOT / value["path"])
    bundle = load_matrices(matrix, *parse_device_dtype(cfg("cpu")))
    H = bundle.select("hx")[3].to_dense().double()
    p = 0.03
    err = (torch.rand((1000, H.shape[1]), generator=torch.Generator().manual_seed(0)) < p).double()
    syndrome = ((err @ H.T) % 2).to(torch.uint8)
    llr0 = torch.full(err.shape, math.log((1 - p) / p), dtype=torch.float64)
    return bundle, syndrome, llr0


def run(dec, surface):
    with torch.no_grad():
        return dec({"synd": surface[1].clone(), "llr0": surface[2].clone()})


@CUDA
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("early", [True, False])
def test_pytorch_cuda_equivalence(surface, dtype, early):
    settings = cfg(dtype=dtype, early_termination=early)
    ref = run(layered.create(settings, bundle=surface[0]), surface)
    got = run(layered_cuda.create(settings, bundle=surface[0]), surface)
    torch.testing.assert_close(got["llr"], ref["llr"], rtol=1e-5, atol=1e-5)
    for key in ("e_v", "converge", "iter"):
        assert torch.equal(got[key], ref[key]), key


@CUDA
def test_convergence_at_least_flooding(surface):
    serial = run(layered_cuda.create(cfg(), bundle=surface[0]), surface)
    parallel = run(flooding.create(cfg(), bundle=surface[0]), surface)
    assert serial["converge"].sum() >= parallel["converge"].sum()


@CUDA
def test_late_oscillation_below_flooding(surface):
    class History:
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            if i >= self.max_iter - 20:
                self.history.append(e_v[:, :-1].clone())

    class Serial(History, layered_cuda.create):
        pass

    class Parallel(History, flooding.create):
        pass

    means = []
    for cls in (Serial, Parallel):
        dec = cls(cfg(early_termination=False), bundle=surface[0])
        dec.history = []
        run(dec, surface)
        history = torch.stack(dec.history)
        assert len(history) == 21
        means.append((history[1:] != history[:-1]).float().mean())
    assert means[0] < means[1], means


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_io_contract_and_layers(surface, device):
    dec = layered.create(cfg(device), bundle=surface[0])
    out = run(dec, surface)
    assert set(KEYS).issubset(out)
    assert out["e_v"].shape == out["llr"].shape == surface[2].shape
    assert out["converge"].shape == out["iter"].shape == (1000,)
    assert out["e_v"].dtype == out["llr"].dtype == torch.float64
    assert out["converge"].dtype == out["iter"].dtype == torch.int64
    assert torch.all((out["iter"] >= 1) & (out["iter"] <= dec.max_iter))
    assert not dec.compile
    assert torch.equal(torch.cat(dec.layers).sort().values,
                       torch.arange(dec.H_shape[0], device=dec.device))
    for cols, real in zip(dec.layer_cols, dec.layer_real):
        assert cols[real].unique().numel() == real.sum()
    assert all(torch.equal(a, b) for a, b in zip(dec.layers, layered.check_layers(dec.V_c_col, dec.H_shape[1])))


@CUDA
@pytest.mark.parametrize("reuse_buffers", [True, False])
def test_compaction_keeps_hooks_and_outputs(surface, reuse_buffers):
    class Hooked(layered.create):
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            assert len(self._hook_rows) == len(l_v) == len(active)
            assert torch.equal(syndrome.cpu().to(torch.uint8), surface[1][self._hook_rows.cpu()])
            self.smallest_batch = min(self.smallest_batch, len(l_v))

        def _exit_hook(self, l_v, e_v, num_iters, converges):
            assert l_v.shape == e_v.shape == (1000, self.H_shape[1] + 1)
            assert torch.isinf(l_v[:, -1]).all()
            assert (num_iters >= 1).all()
            self.exit_called = True

    reference = run(layered.create(cfg(compact_frac=0), bundle=surface[0]), surface)
    dec = Hooked(cfg(reuse_buffers=reuse_buffers), bundle=surface[0])
    dec.smallest_batch, dec.exit_called = 1000, False
    got = run(dec, surface)
    assert dec.smallest_batch < 1000
    assert dec.exit_called
    for key in KEYS:
        torch.testing.assert_close(got[key], reference[key], rtol=0, atol=0)


def test_all_converged_hook_timing_matches_nms(surface):
    class Hooks:
        def _iter_hook(self, i, l_v, e_v, active, syndrome):
            self.calls.append((i, bool(active.any())))

    class Serial(Hooks, layered.create):
        pass

    class Parallel(Hooks, flooding.create):
        pass

    zero = (surface[0], torch.zeros_like(surface[1][:8]), surface[2][:8])
    decoders = [cls(cfg("cpu"), bundle=surface[0]) for cls in (Serial, Parallel)]
    for dec in decoders:
        dec.calls = []
        out = run(dec, zero)
        assert out["converge"].all()
        assert (out["iter"] == 1).all()
        assert dec.i == 2
    assert decoders[0].calls == decoders[1].calls == [(1, False)]


@pytest.mark.parametrize("iterations", [2, 3])
def test_cpu_matches_independent_per_check_reference(iterations):
    H = torch.tensor([[1, 1, 1, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1, 0]])
    index = dense_to_index_format(H.numpy(), torch.device("cpu"))
    matrix = SimpleNamespace(get_index=lambda: index)
    logical = torch.zeros((0, H.shape[1]))
    bundle = MatrixBundle(matrix, matrix, logical, logical)
    settings = cfg("cpu", early_termination=False)
    settings["max_iter"] = iterations
    dec = layered.create(settings, bundle=bundle)
    llr0 = torch.tensor([[2.25, -1.5, 0.625, 3.75, -2.5, 0.0],
                         [-0.75, 4.0, 1.125, -2.25, 0.5, 2.75]], dtype=torch.float64)
    syndrome = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    posterior = llr0.clone()
    previous = torch.zeros((len(llr0), *H.shape), dtype=torch.float64)
    order = torch.cat(dec.layers).tolist()
    assert order == [0, 2, 1, 3]
    for iteration in range(1, iterations + 1):
        beta = 1.0 - 2.0 ** -iteration
        for check in order:
            variables = H[check].nonzero().flatten()
            for batch in range(len(llr0)):
                old = previous[batch, check, variables]
                extrinsic = posterior[batch, variables] - old
                messages = []
                for edge in range(len(variables)):
                    others = torch.cat((extrinsic[:edge], extrinsic[edge + 1:]))
                    parity = (int((others <= 0).sum()) + int(syndrome[batch, check])) % 2
                    messages.append((-1 if parity else 1) * beta * others.abs().min())
                new = torch.stack(messages)
                posterior[batch, variables] += new - old
                previous[batch, check, variables] = new
    got = run(dec, (bundle, syndrome, llr0))
    torch.testing.assert_close(got["llr"], posterior, rtol=0, atol=0)
    assert torch.equal(got["e_v"], (posterior <= 0).to(torch.float64))
