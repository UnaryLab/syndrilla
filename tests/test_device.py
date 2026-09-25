import math

import pytest
import torch
from loguru import logger

from syndrilla.decoder import create_decoder
from syndrilla.decoder.relay_bp.relay_bp import create as create_relay_bp
from syndrilla.matrix import load_matrices
from syndrilla.metric.metric import MetricState, _same_setting
from syndrilla.utils import get_path, parse_device_dtype, read_yaml


def _warnings(fn):
    """Run fn() and return its result plus the warnings it logged."""
    msgs = []
    sink = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    try:
        return fn(), msgs
    finally:
        logger.remove(sink)


def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _fake_cuda(monkeypatch, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def _resolve(cfg):
    """parse_device_dtype(cfg) plus the warnings it logged."""
    (device, dtype), msgs = _warnings(lambda: parse_device_dtype(cfg))
    return device, dtype, msgs


def test_mps_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype, msgs = _resolve({"dtype": "float32", "device": {"device_type": "mps"}})
    assert device == torch.device("cpu") and dtype == torch.float32
    assert any("<mps>" in m for m in msgs)


def test_mps_float64_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    _no_cuda(monkeypatch)
    device, dtype, msgs = _resolve({"dtype": "float64", "device": {"device_type": "mps"}})
    assert device == torch.device("cpu") and dtype == torch.float64
    assert any("float64" in m for m in msgs)


def test_default_is_never_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, _, msgs = _resolve({"dtype": "float32"})
    assert device == torch.device("cpu") and not msgs


def test_default_with_cuda_is_cuda0(monkeypatch):
    _fake_cuda(monkeypatch)
    device, _, msgs = _resolve({"dtype": "float32"})
    assert device == torch.device("cuda:0") and not msgs


def test_cuda_index_out_of_range_falls_back_to_cuda0(monkeypatch):
    _fake_cuda(monkeypatch, count=1)
    device, _, msgs = _resolve(
        {"dtype": "float32", "device": {"device_type": "cuda", "device_idx": 3}}
    )
    assert device == torch.device("cuda:0")
    assert any("<3>" in m for m in msgs)


def test_unknown_dtype_warns_and_gives_float64(monkeypatch):
    _no_cuda(monkeypatch)
    device, dtype, msgs = _resolve({"dtype": "float99", "device": {"device_type": "cpu"}})
    assert device == torch.device("cpu") and dtype == torch.float64
    assert any("float99" in m for m in msgs)


@pytest.mark.parametrize("name", ["int32", "half", "float"])
def test_unaccepted_dtype_on_mps_gives_cpu_float64(monkeypatch, name):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    _no_cuda(monkeypatch)
    device, dtype, msgs = _resolve({"dtype": name, "device": {"device_type": "mps"}})
    assert device == torch.device("cpu") and dtype == torch.float64
    assert any(f"<{name}>" in m for m in msgs)


@pytest.mark.parametrize("device", [None, "gpu"])
def test_relay_bp_bad_device_falls_back(monkeypatch, device):
    _no_cuda(monkeypatch)
    cfg = read_yaml(get_path("examples/alist/relay_bp_hx.decoding.yaml"))["decoding"]
    cfg = {**cfg, **cfg.pop("config"), "device": device}
    matrix_cfg = read_yaml(get_path("examples/alist/surface_5.matrix.yaml"))["matrix"]
    bundle = load_matrices(matrix_cfg, torch.device("cpu"), torch.float64)
    decoder, msgs = _warnings(lambda: create_relay_bp(cfg, bundle=bundle))
    assert decoder.device == torch.device("cpu")
    if device == "gpu":
        assert any("<gpu>" in m for m in msgs)
    else:
        assert not msgs


@pytest.mark.parametrize("saved, now", [("cuda", "cuda:0"), ("cuda:0", "cuda")])
def test_checkpoint_device_cuda_matches_cuda0(saved, now):
    assert _same_setting("device", saved, now)
    assert not _same_setting("device", "cuda:1", now)
    assert not _same_setting("dtype", "cuda", "cuda:0")


@pytest.mark.parametrize("saved, now", [("cuda", "cuda:0"), ("cuda:0", "cuda")])
def test_train_validate_checkpoint_cuda_matches_cuda0(saved, now):
    state = MetricState(0, 0, None)
    state._fingerprint = {"device": now, "dtype": "torch.float32"}
    state.train_validate_checkpoint({"device": saved, "dtype": "torch.float32"}, "run.pt")
    with pytest.raises(ValueError, match="device"):
        state.train_validate_checkpoint({"device": "cuda:1", "dtype": "torch.float32"}, "run.pt")


def _decode(device_type, synd, llr0):
    cfg = read_yaml(get_path("examples/alist/bp_hx.decoding.yaml"))["decoding"]
    cfg["dtype"] = "float32"
    cfg["device"] = {"device_type": device_type, "device_idx": 0}
    matrix_cfg = read_yaml(get_path("examples/alist/surface_5.matrix.yaml"))["matrix"]
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(cfg))
    decoder = create_decoder(cfg=cfg, bundle=bundle)[0]
    assert decoder.device.type == device_type
    out = decoder({"synd": synd.clone(), "llr0": llr0.clone()})
    return {k: out[k].cpu() for k in ("e_v", "converge", "iter")}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_bp_norm_min_sum_mps_matches_cpu():
    p, B = 0.05, 256
    matrix_cfg = read_yaml(get_path("examples/alist/surface_5.matrix.yaml"))["matrix"]
    H = load_matrices(matrix_cfg, torch.device("cpu"), torch.float32).select("hx")[3]
    H = H.to(torch.float32)
    g = torch.Generator().manual_seed(0)
    err = (torch.rand(B, H.shape[1], generator=g) < p).to(torch.float32)
    synd = torch.remainder(err @ H.t(), 2.0)
    llr0 = torch.full_like(err, math.log((1 - p) / p))

    cpu = _decode("cpu", synd, llr0)
    mps = _decode("mps", synd, llr0)
    for k in cpu:
        assert torch.equal(cpu[k], mps[k]), f"<{k}> differs between cpu and mps"
