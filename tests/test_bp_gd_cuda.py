"""Guided-decimation parity on CPU torch, CUDA torch and native CUDA BP."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from syndrilla.decoder.bp_gd.bp_gd import create as TorchGd
from syndrilla.decoder.bp_gd.bp_gd_cuda import create as CudaGd
from syndrilla.decoder.decoder import create_decoder
from syndrilla.matrix import load_matrices
from syndrilla.matrix.matrix import dense_to_index_format
from syndrilla.utils import parse_device_dtype, read_yaml

ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def problem(name="surface_10", dtype="float64", p=.05, shots=512, **extra):
    cfg = dict(device={"device_type": "cpu"}, dtype=dtype, check_type="hx",
        max_iter=61, round_iters=5, max_rounds=12, llr_max=25., order="reliability",
        random_machine="sobol", rebatch_opt=False, compile=False)
    cfg.update(extra)
    bundle = load_matrices(read_yaml(str(ROOT / f"examples/alist/{name}.matrix.yaml"))["matrix"],
                           *parse_device_dtype(cfg))
    h = bundle.select("hx")[3].to_dense().double()
    errors = (torch.rand((shots, h.shape[1]), generator=torch.Generator().manual_seed(270),
                         dtype=torch.float64) < p).double()
    io = dict(llr0=torch.full_like(errors, math.log((1-p)/p)), synd=(errors @ h.T).remainder(2))
    return cfg, bundle, io


def decode(cls, cfg, bundle, io):
    dec = cls(cfg, bundle=bundle)
    with torch.no_grad():
        result = dec({key: value.clone() for key, value in io.items()})
    return dec, result


def exact(actual, expected, converged_only=False):
    for key in ("converge", "iter", "rounds"):
        assert torch.equal(actual[key].cpu(), expected[key].cpu()), key
    mask = expected["converge"].cpu().bool() if converged_only else slice(None)
    assert torch.equal(actual["e_v"].cpu()[mask], expected["e_v"].cpu()[mask]), "e_v"


@GPU
@pytest.mark.parametrize("name", ["surface_10", "bb_144_12_12"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", ["reliability", "random"])
@pytest.mark.parametrize("early", [True, False])
def test_cpu_torch_cuda_kernel_parity(name, dtype, order, early):
    cfg, bundle, io = problem(name, dtype, order=order, early_termination=early)
    _, expected = decode(TorchGd, cfg, bundle, io)
    gpu_cfg = dict(cfg, device={"device_type": "cuda"})
    for cls in (TorchGd, CudaGd):
        dec, actual = decode(cls, gpu_cfg, bundle, io)
        exact(actual, expected, converged_only=dtype == "float32")
        assert int(actual["rounds"].max()) <= 11
        assert int(actual["iter"].max()) <= 60
        assert actual["rounds"].any(), (name, dtype, order, early)
        repeated = dec({key: value.clone() for key, value in io.items()})
        exact(repeated, actual)


@GPU
@pytest.mark.parametrize("fuse", [True, False])
def test_cuda_saturation_hand_value(fuse):
    index = dense_to_index_format(torch.tensor([[1., 1.]], dtype=torch.float64), "cpu")
    bundle = SimpleNamespace(Hx_matrix=None, Hz_matrix=None, lx_matrix=None, lz_matrix=None,
                             select=lambda _: index)
    cfg = dict(device={"device_type": "cuda"}, dtype="float64", max_iter=1,
               round_iters=1, saturation=2., fuse_vn=fuse, compile=False, rebatch_opt=False)
    io = dict(llr0=torch.tensor([[12., -9.]], dtype=torch.float64),
              synd=torch.zeros(1, 1, dtype=torch.float64))
    for cls in (TorchGd, CudaGd):
        _, out = decode(cls, cfg, bundle, io)
        torch.testing.assert_close(out["llr"].cpu(), torch.tensor([[10., -7.]], dtype=torch.float64),
                                   rtol=0, atol=1e-14)


@GPU
@pytest.mark.parametrize("force", [False, True])
def test_factory_selects_cuda_backend(force):
    cfg, bundle, io = problem(shots=16)
    cfg.pop("device")
    cfg.pop("dtype")
    cfg.pop("check_type")
    dec = create_decoder(cfg=dict(algorithm="bp_gd", device={"device_type": "cuda"},
        dtype="float64", force_pytorch=force, config=cfg), bundle=bundle)[0]
    assert dec.decoder.__class__.__module__.endswith("bp_gd" if force else "bp_gd_cuda")
    out = dec(io)
    assert out["rounds"].shape == (16,)


@GPU
@pytest.mark.parametrize("extra", [
    {"edge_layout": "padded"},
    {"pruning_opt": False, "fusion_opt": False, "mapping_opt": False,
     "gather_opt": False, "memory_opt": False, "rebatch_opt": False},
])
def test_kernel_options_keep_hard_results(extra):
    cfg, bundle, io = problem(shots=64)
    cfg["device"] = {"device_type": "cuda"}
    _, expected = decode(CudaGd, cfg, bundle, io)
    _, actual = decode(CudaGd, dict(cfg, **extra), bundle, io)
    exact(actual, expected)


@pytest.mark.parametrize("key,value", [("round_iters", 0), ("llr_max", 0),
    ("max_rounds", 0), ("saturation", 0), ("saturation", float("inf")),
    ("order", "bad"), ("random_machine", "bad"), ("typo", True)])
def test_invalid_cuda_config_rejected_before_device_setup(key, value):
    with pytest.raises(ValueError, match=key):
        CudaGd(dict(device={"device_type": "cuda"}, max_iter=20, **{key: value}))
