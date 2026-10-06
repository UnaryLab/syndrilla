from types import SimpleNamespace

import numpy as np
import pymatching
import pytest
import torch
from loguru import logger

from syndrilla.decoder.decoder import create_decoder
from syndrilla.decoder.mwpm import mwpm
from syndrilla.decoder.mwpm_gpu import mwpm_gpu, mwpm_gpu_cuda
from syndrilla.matrix.matrix import dense_to_index_format


def _bundle(h, device="cpu"):
    indices = dense_to_index_format(h, torch.device(device))
    matrix = SimpleNamespace(get_index=lambda: indices)
    return SimpleNamespace(Hx_matrix=matrix, Hz_matrix=matrix)


def _cfg(**extra):
    return {"device": {"device_type": "cpu"}, "dtype": "float64", **extra}


def _graph(seed=40):
    rng = np.random.default_rng(seed)
    h = np.zeros((6, 22), dtype=np.uint8)
    h[:, :6] = np.eye(6, dtype=np.uint8)
    for col in range(6, 20):
        h[rng.choice(6, size=int(rng.integers(1, 3)), replace=False), col] = 1
    return h


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_configuration_logs_host_decoding_once():
    expected = "mwpm decodes on the host CPU with PyMatching and returns tensors on the configured device."
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="INFO",
                      filter=lambda record: record["message"] == expected)
    try:
        mwpm.create(_cfg(weights="uniform"), bundle=_bundle(np.eye(2)))
        assert records == []
        decoder = mwpm.create(_cfg(weights="uniform", device={"device_type": "cuda"}),
                              bundle=_bundle(np.eye(2), "cuda"))
        assert len(records) == 1
        assert records[0]["level"].name == "INFO"
        for _ in range(2):
            decoder({"synd": torch.tensor([[1, 0]], device="cuda")})
        assert len(records) == 1
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("mode", ["uniform", "posterior", "prior"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_factory_random_graph_syndrome_and_output_device(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng, h = np.random.default_rng(401), _graph()
    errors = rng.integers(0, 2, (32, h.shape[1]), dtype=np.uint8)
    syndrome = torch.tensor(errors @ h.T % 2, device=device)
    prior = torch.tensor(rng.normal(size=errors.shape), dtype=torch.float64, device=device)
    cfg = _cfg(algorithm="mwpm", device={"device_type": device},
               config=dict(weights=mode, num_workers=32))
    decoder = create_decoder(cfg=cfg, bundle=_bundle(h, device))[0]
    assert decoder.skip_converged is (mode == "posterior")
    out = decoder({"synd": syndrome, "llr": prior, "llr0": prior})
    assert decoder.algo == "mwpm" and decoder.H_shape == h.shape
    assert np.array_equal(out["e_v"].cpu().numpy() @ h.T % 2, syndrome.cpu().numpy())
    assert torch.equal(out["llr"] <= 0, out["e_v"])
    for key in ("e_v", "llr", "iter", "converge"):
        assert out[key].device.type == device
    assert out["e_v"].dtype == out["llr"].dtype == torch.float64
    assert out["iter"].shape == out["converge"].shape == (32,)
    assert bool((out["iter"] >= 1).all()) and bool(out["converge"].all())


@pytest.mark.parametrize("mode", ["uniform", "posterior", "prior"])
def test_bfloat16_syndrome_and_output(mode):
    decoder = mwpm.create(_cfg(weights=mode, dtype="bfloat16"), bundle=_bundle(np.eye(3)))
    syndrome = torch.tensor([[1, 0, 1]], dtype=torch.bfloat16)
    prior = torch.tensor([[0, -1, 3]], dtype=torch.bfloat16)
    out = decoder({"synd": syndrome, "llr": prior, "llr0": prior})
    assert torch.equal(out["e_v"], syndrome)
    assert out["e_v"].dtype == out["llr"].dtype == torch.bfloat16
    assert out["e_v"].shape == out["llr"].shape == (1, 3)


@pytest.mark.parametrize("mode", ["uniform", "posterior", "prior"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_200_shot_native_matches_pymatching(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng, h = np.random.default_rng(402), _graph(41)
    syndrome = rng.integers(0, 2, (200, h.shape[0]), dtype=np.uint8)
    llr = rng.normal(0, 2, (200, h.shape[1]))
    llr[0, 0] = 0
    cfg = _cfg(weights=mode, num_workers=1, device={"device_type": device})
    bundle = _bundle(h, device)
    io = {"synd": torch.tensor(syndrome, device=device),
          "llr0": torch.tensor(llr, device=device), "llr": torch.tensor(llr, device=device)}
    got = mwpm.create(cfg, bundle=bundle)(dict(io))
    native = mwpm_gpu_cuda if device == "cuda" else mwpm_gpu
    ref = native.create(cfg, bundle=bundle)(dict(io))
    got_bits, ref_bits = (out["e_v"].cpu().numpy().astype(np.uint8) for out in (got, ref))
    assert np.array_equal(got_bits @ h.T % 2, syndrome)
    assert np.array_equal(ref_bits @ h.T % 2, syndrome)
    np.testing.assert_array_equal(got_bits, ref_bits)
    print(f"{device}/{mode}: all 200 PyMatching/native e_v match exactly")


@pytest.mark.parametrize("mode", ["posterior", "prior"])
def test_signed_zero_source_and_parallel_edge_selection(mode):
    h = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    values = torch.tensor([[1., 8., 3., 4.], [8., 1., 3., 4.],
                           [0., 0., -3., -4.]], dtype=torch.float64)
    syndrome = torch.tensor([[1, 1], [1, 1], [0, 0]])
    selected, other = ("llr0", "llr") if mode == "prior" else ("llr", "llr0")
    out = mwpm.create(_cfg(weights=mode), bundle=_bundle(h))({
        "synd": syndrome, selected: values, other: torch.full_like(values, 20),
    })
    assert out["e_v"].tolist() == [[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 1., 1.]]
    assert torch.equal(out["llr"] <= 0, out["e_v"])


@pytest.mark.parametrize("mode", ["uniform", "posterior", "prior"])
@pytest.mark.parametrize("h", [np.zeros((2, 0)), np.zeros((2, 5)), np.array([[1, 0, 0]])],
                         ids=["no-columns", "no-edges", "trailing-zero-columns"])
def test_fault_ids_preserve_all_columns(h, mode):
    syndrome = torch.zeros(3, h.shape[0])
    if h.any():
        syndrome[:, 0] = 1
    prior = torch.full((3, h.shape[1]), -2.0)
    out = mwpm.create(_cfg(weights=mode), bundle=_bundle(h))({"synd": syndrome, "llr": prior, "llr0": prior})
    assert out["e_v"].shape == (3, h.shape[1])
    assert np.array_equal(out["e_v"].numpy() @ h.T % 2, syndrome.numpy())
    expected = np.ones((3, h.shape[1])) if mode != "uniform" else np.tile(h.any(0), (3, 1))
    assert np.array_equal(out["e_v"].numpy(), expected)


@pytest.mark.parametrize("extra", [
    {"weights": "bad"}, {"weights": "llr"},
    {"skip_converged": 1}, {"skip_converged": None},
    {"skip_converged": True, "weights": "uniform"},
    {"skip_converged": True, "weights": "prior"},
])
def test_config_validation_matches_native(extra):
    for module in (mwpm, mwpm_gpu):
        with pytest.raises(ValueError, match="weights must be.*posterior" if extra.get("weights") == "llr" else None):
            module.create(_cfg(**extra), bundle=_bundle(_graph()))


@pytest.mark.parametrize("module", [mwpm, mwpm_gpu], ids=["pymatching", "native"])
def test_default_prior_requires_llr0(module):
    decoder = module.create(_cfg(), bundle=_bundle(np.eye(2)))
    assert decoder.weights == "prior"
    assert decoder.skip_converged is False
    with pytest.raises(ValueError, match="llr0"):
        decoder({"synd": torch.zeros(1, 2), "llr": torch.ones(1, 2)})
    out = decoder({"synd": torch.tensor([[1, 0]]), "llr0": torch.ones(1, 2)})
    assert out["e_v"].tolist() == [[1, 0]]


@pytest.mark.parametrize("mode", ["posterior", "prior"])
def test_float_weights_preserve_zero_and_fractional_costs(mode, monkeypatch):
    decoder = mwpm.create(_cfg(weights=mode), bundle=_bundle(np.eye(3)))
    builds = []
    matching = decoder._matching

    def record(weights=None):
        builds.append(weights.copy())
        return matching(weights)

    monkeypatch.setattr(decoder, "_matching", record)
    values = torch.tensor([[0., -0.25, 1.5]], dtype=torch.float64)
    key = "llr0" if mode == "prior" else "llr"
    out = decoder({"synd": torch.tensor([[1, 0, 1]]), key: values})
    np.testing.assert_array_equal(builds[0], [0., 0.25, 1.5])
    assert out["e_v"].tolist() == [[1, 0, 1]]


@pytest.mark.parametrize("mode", ["posterior", "prior"])
@pytest.mark.parametrize("bad", [None, torch.zeros(2, 3), torch.full((3, 3), float("nan")),
                                  torch.full((3, 3), float("inf")), torch.ones(3, 3, dtype=torch.complex64)])
def test_invalid_weight_input(mode, bad):
    io = {"synd": torch.zeros(3, 3)}
    if bad is not None:
        io["llr0" if mode == "prior" else "llr"] = bad
    with pytest.raises(ValueError):
        mwpm.create(_cfg(weights=mode), bundle=_bundle(np.eye(3)))(io)


def test_hyperedge_is_rejected():
    with pytest.raises(ValueError, match="Column 0"):
        mwpm.create(_cfg(), bundle=_bundle(np.ones((3, 1))))


def test_backend_weight_limit_checks_actual_weights():
    decoder = mwpm.create(_cfg(weights="prior"), bundle=_bundle(np.eye(2)))
    syndrome = torch.tensor([[1, 0]])
    out = decoder({"synd": syndrome, "llr0": torch.ones(1, 2)})
    assert torch.equal(out["e_v"], syndrome)
    with pytest.raises(ValueError, match="PyMatching"):
        decoder({"synd": syndrome, "llr0": torch.full((1, 2), 2**24)})


@pytest.mark.parametrize("all_converged", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_skip_converged_preserves_rows_and_batches_only_pending(all_converged, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    h = np.eye(3)
    cfg = _cfg(weights="posterior", device={"device_type": device})
    decoder = mwpm.create(cfg, bundle=_bundle(h, device))
    syndrome = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]], device=device)
    incoming = syndrome.double()
    llr = (1 - 2 * incoming) * 3.25
    keep = torch.ones(4, dtype=torch.bool, device=device)
    if not all_converged:
        keep[1::2] = False
        incoming[~keep] = 0
    calls = []
    decode_batch = pymatching.Matching.decode_batch

    def record(self, shots, **kwargs):
        calls.append(len(shots))
        return decode_batch(self, shots, **kwargs)

    monkeypatch.setattr(pymatching.Matching, "decode_batch", record)
    out = decoder({"synd": syndrome, "llr": llr.clone(), "e_v": incoming.clone(),
                   "converge": keep.long(), "iter": torch.full((4,), 7, device=device)})
    assert sum(calls) == (0 if all_converged else 2)
    assert torch.equal(out["e_v"][keep], incoming[keep])
    assert torch.equal(out["llr"][keep], llr[keep])
    assert bool((out["iter"][keep] == 7).all())
    assert torch.equal(out["e_v"], syndrome)


@pytest.mark.parametrize("module,device", [
    (mwpm, "cpu"), (mwpm, "cuda"), (mwpm_gpu, "cpu"), (mwpm_gpu_cuda, "cuda"),
], ids=["pymatching-cpu", "pymatching-cuda", "native-cpu", "native-cuda"])
def test_explicit_false_posterior_rematches_all_rows(module, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    decoder = module.create(_cfg(weights="posterior", skip_converged=False,
                                 device={"device_type": device}), bundle=_bundle(np.eye(3), device))
    syndrome = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], device=device)
    calls, decode = [], decoder._decode

    def record(io):
        calls.append(len(io["synd"]))
        return decode(io)

    monkeypatch.setattr(decoder, "_decode", record)
    out = decoder({"synd": syndrome, "llr": torch.full((4, 3), 3.25, device=device),
                   "e_v": torch.zeros(4, 3, device=device), "converge": torch.ones(4, device=device)})
    assert calls == [4]
    assert torch.equal(out["e_v"], syndrome)
    assert torch.equal(out["llr"], 1 - 2 * out["e_v"])


def test_prior_groups_unique_weights_for_decode_batch(monkeypatch):
    h = np.array([[1, 1], [1, 0]])
    decoder = mwpm.create(_cfg(weights="prior"), bundle=_bundle(h))
    llr = torch.tensor([[1., 4.], [-1., -4.], [1.125, 4.125],
                        [5., 2.], [-5., -2.], [5.125, 2.125]], dtype=torch.float64)
    builds, batches = [], []
    matching, decode_batch = decoder._matching, pymatching.Matching.decode_batch

    def build(weights=None):
        builds.append(tuple(weights))
        return matching(weights)

    def batch(self, shots, **kwargs):
        batches.append(len(shots))
        return decode_batch(self, shots, **kwargs)

    monkeypatch.setattr(decoder, "_matching", build)
    monkeypatch.setattr(pymatching.Matching, "decode_batch", batch)
    syndrome = torch.tensor([[1, 1], [1, 0], [0, 1]]).repeat(2, 1)
    out = decoder({"synd": syndrome, "llr0": llr})
    assert set(builds) == {(1, 4), (1.125, 4.125), (5, 2), (5.125, 2.125)} and len(builds) == 4
    assert sorted(batches) == [1, 1, 2, 2]
    assert np.array_equal(out["e_v"].numpy() @ h.T % 2, syndrome.numpy())


@pytest.mark.parametrize("distance,rate,shots", [(5, 0.001, 10000), (9, 0.01, 20000)])
def test_stim_example_matches_direct_dem_observable_predictions(distance, rate, shots):
    from syndrilla.error_model.stim_circuit.stim_circuit import NOISE_KEYS
    from syndrilla.interface import create_interface
    from syndrilla.utils import read_yaml

    cfg = read_yaml("examples/stim/stim_mwpm.interface.yaml")["interface"]
    assert cfg["distance"] == 5 and cfg["decompose_errors"] is True
    cfg["distance"] = distance
    interface = create_interface(cfg=cfg, error_cfg={key: rate for key in NOISE_KEYS},
                                 syndrome_cfg={"rounds": distance})
    dem = interface.circuit.detector_error_model(decompose_errors=True)
    syndrome, truth, _ = dem.compile_sampler(seed=1234).sample(shots)
    direct = pymatching.Matching.from_detector_error_model(dem).decode_batch(syndrome)
    decoder = mwpm.create(_cfg(), bundle=interface.matrix_bundle)
    priors = interface.error_model.priors
    llr0 = torch.log((1 - priors) / priors).expand(len(syndrome), -1)
    out = decoder({"synd": torch.from_numpy(syndrome), "llr0": llr0})
    logical = np.asarray(interface.matrix_bundle.lx_matrix)
    predicted = (out["e_v"].numpy() @ logical.T % 2).astype(np.uint8)
    mismatches = int(np.count_nonzero(predicted != direct))
    errors = int(np.count_nonzero(np.any(predicted != truth, axis=1)))
    print(f"Stim d{distance}r{distance} p={rate} seed=1234 shots={len(syndrome)} "
          f"observables={dem.num_observables}: mismatches={mismatches}, logical_errors={errors}")
    assert predicted.shape == direct.shape == (shots, dem.num_observables)
    np.testing.assert_array_equal(predicted, direct)


def test_existing_bp_mwpm_chain_uses_pymatching():
    from syndrilla.matrix import load_matrices
    from syndrilla.utils import parse_device_dtype, read_yaml

    cfg = read_yaml("examples/alist/bp_mwpm_hx.decoding.yaml")["decoding"]
    assert cfg["algorithm"] == ["bp_norm_min_sum", "mwpm"]
    cfg["config"][0].update(max_iter=2, compile=False)
    bundle = load_matrices(read_yaml("examples/alist/surface_10.matrix.yaml")["matrix"], *parse_device_dtype(cfg))
    h = bundle.select("hx")[3].to_dense().cpu().double()
    errors = (torch.rand(32, h.shape[1], generator=torch.Generator().manual_seed(40)) < 0.1).double()
    syndrome = errors @ h.T % 2
    bp, matching = create_decoder(cfg=cfg, bundle=bundle)
    posterior = bp({"synd": syndrome, "llr0": torch.full_like(errors, 2.0)})
    assert not bool(posterior["converge"].all())
    assert isinstance(matching._matching(np.ones(h.shape[1])), pymatching.Matching)
    out = matching(posterior)
    assert torch.equal(out["e_v"] @ h.T % 2, syndrome)
    assert torch.equal(out["llr"] <= 0, out["e_v"])
