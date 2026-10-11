import math
from pathlib import Path

import pytest
import torch

from syndrilla.decoder.bp_lottery_layered import bp_lottery_layered as lottery
from syndrilla.decoder.bp_lottery_layered import bp_lottery_layered_cuda as lottery_cuda
from syndrilla.decoder.bp_norm_min_sum_layered import bp_norm_min_sum_layered as layered
from syndrilla.decoder.bp_norm_min_sum_layered import bp_norm_min_sum_layered_cuda as layered_cuda
from syndrilla.decoder.bp_lottery import bp_lottery as flooding
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
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
@pytest.mark.parametrize("random_machine", ["sobol", "system"])
def test_pytorch_cuda_equivalence(surface, dtype, random_machine):
    settings = cfg(dtype=dtype, random_machine=random_machine)
    outputs, rngs = [], []
    for cls in (lottery.create, lottery_cuda.create):
        torch.manual_seed(7)
        outputs.append(run(cls(settings, bundle=surface[0]), surface))
        rngs.append(torch.cuda.get_rng_state())
    torch.testing.assert_close(outputs[1]["llr"], outputs[0]["llr"], rtol=1e-5, atol=1e-5)
    for key in ("e_v", "converge", "iter"):
        assert torch.equal(outputs[1][key], outputs[0][key]), key
    assert torch.equal(rngs[1], rngs[0])


@CUDA
def test_convergence_at_least_both_baselines(surface):
    counts = {}
    for name, cls in (("layered_lottery", lottery_cuda.create),
                      ("layered", layered_cuda.create), ("lottery", flooding.create)):
        counts[name] = int(run(cls(cfg(), bundle=surface[0]), surface)["converge"].sum())
    print(f"Converged out of 1000: {counts}")
    assert counts["layered_lottery"] >= counts["layered"], counts
    assert counts["layered_lottery"] >= counts["lottery"], counts


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_io_contract_and_plugin_loading(surface, device):
    settings = read_yaml(str(ROOT / "examples/alist/bp_lottery_layered_hx.decoding.yaml"))["decoding"]
    settings["device"]["device_type"] = device
    settings["rebatch_opt"] = False
    settings["config"]["max_iter"] = 20
    dec = create_decoder(cfg=settings, bundle=surface[0])[0]
    batch = (surface[0], surface[1][:16], surface[2][:16])
    out = run(dec, batch)
    assert set(KEYS).issubset(out)
    assert out["e_v"].shape == out["llr"].shape == batch[2].shape
    assert out["converge"].shape == out["iter"].shape == (16,)
    assert out["e_v"].dtype == out["llr"].dtype == torch.float64
    assert out["converge"].dtype == out["iter"].dtype == torch.int64
    assert torch.all((out["iter"] >= 1) & (out["iter"] <= dec.max_iter))
    assert dec.algo == "bp_lottery_layered"
    assert isinstance(dec.decoder, flooding.create)
    assert isinstance(dec.decoder, layered.create if device == "cpu" else layered_cuda.create)


def test_inherits_lottery_without_new_lottery_code():
    for cls in (lottery.create, lottery_cuda.create):
        for method in ("_draw_r", "sign_flip_cn_rand_new"):
            assert getattr(cls, method) is getattr(flooding.create, method)
    assert lottery_cuda.create.__mro__[:4] == (
        lottery_cuda.create, lottery.create, flooding.create, layered_cuda.create)


def test_lottery_defaults_and_no_flips_before_start(surface):
    settings = cfg("cpu")
    settings["max_iter"] = 4
    dec = lottery.create(settings, bundle=surface[0])
    assert (dec.random_machine, dec.flip_start_iter, dec.flip_interval) == ("sobol", 4, 1)
    batch = (surface[0], surface[1][:16], surface[2][:16])
    got = run(dec, batch)
    expected = run(layered.create(settings, bundle=surface[0]), batch)
    for key in KEYS:
        torch.testing.assert_close(got[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("reuse_buffers", [True, False])
@pytest.mark.parametrize("compact_frac", [0, 0.75])
def test_transient_flips_preserve_prior_on_every_sweep(surface, reuse_buffers, compact_frac):
    class Checked(lottery.create):
        def _sweep(self, l_v, previous, syndrome):
            if self._pending_flip is not None:
                self.undo_count += int((self._pending_flip[2] != 0).sum())
                if len(l_v) != len(self._pending_flip[0]):
                    self.remapped_count += 1
            super()._sweep(l_v, previous, syndrome)
            expected = torch.cat((surface[2][:1].expand(len(l_v), -1),
                                  l_v.new_full((len(l_v), 1), float("inf"))), 1)
            expected.index_add_(1, self.V_c_col.flatten(), previous.flatten(1))
            error = (l_v[:, :-1] - expected[:, :-1]).abs().max().item()
            self.max_error = max(self.max_error, error)
            torch.testing.assert_close(l_v, expected, rtol=1e-12, atol=1e-12)
            self.smallest_batch = min(self.smallest_batch, len(l_v))

    settings = cfg("cpu", reuse_buffers=reuse_buffers, compact_frac=compact_frac)
    settings["max_iter"] = 30
    dec = Checked(settings, bundle=surface[0])
    dec.undo_count = dec.remapped_count = 0
    dec.max_error, dec.smallest_batch = 0, len(surface[1])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        got = run(dec, surface)
        plain = run(layered.create(settings, bundle=surface[0]), surface)
    finally:
        torch.set_num_threads(threads)
    assert dec.undo_count > 0
    assert not torch.equal(got["llr"], plain["llr"])
    if compact_frac:
        assert dec.smallest_batch < len(surface[1])
        assert dec.remapped_count > 0
    else:
        assert dec.smallest_batch == len(surface[1])
    print(f"Transient prior: reuse_buffers={reuse_buffers}, compact_frac={compact_frac}, "
          f"max_error={dec.max_error}, undos={dec.undo_count}, remaps={dec.remapped_count}")
