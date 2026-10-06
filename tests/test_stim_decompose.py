from itertools import product

import numpy as np
import pytest
import stim
import torch

from syndrilla.error_model.stim_circuit import stim_circuit as error_stim
from syndrilla.interface import create_interface
from syndrilla.matrix.stim import stim as matrix_stim
from syndrilla.syndrome.stim import stim as syndrome_stim


def _generated(distance=3, rate=0.01):
    cfg = dict(code="surface_code:rotated_memory_x", distance=distance, rounds=distance)
    cfg.update({key: rate for key in error_stim.NOISE_KEYS})
    circuit = stim.Circuit.generated(cfg["code"], **{k: v for k, v in cfg.items() if k != "code"})
    return circuit, cfg


def _oracle(dem):
    """Independent small dense oracle: separator pieces share one Bernoulli draw."""
    components, mechanisms, probabilities = {}, [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probabilities.append(inst.args_copy()[0])
        pieces = [[]]
        for target in inst.targets_copy():
            if target.is_separator():
                pieces.append([])
            else:
                pieces[-1].append(target)
        active = set()
        for piece in pieces:
            dets, obs = set(), set()
            for target in piece:
                support = dets if target.is_relative_detector_id() else obs
                support.symmetric_difference_update([target.val])
            key = (tuple(sorted(dets)), tuple(sorted(obs)))
            if key == ((), ()):
                continue
            active.symmetric_difference_update([key])
        mechanisms.append({components.setdefault(key, len(components)) for key in sorted(active)})
    incidence = np.zeros((len(mechanisms), len(components)), dtype=np.int64)
    for row, cols in enumerate(mechanisms):
        incidence[row, list(cols)] = 1
    p = np.asarray(probabilities)
    priors = np.array([(1 - np.prod(1 - 2 * p[incidence[:, j] != 0])) / 2
                       for j in range(len(components))])
    return list(components), incidence, priors


def _replay(model, draws, monkeypatch):
    def random_values(value):
        assert tuple(value.shape) == draws.shape
        return torch.tensor(~draws, device=value.device, dtype=value.dtype)

    monkeypatch.setattr(torch, "rand_like", random_values)
    zeros = torch.zeros(len(draws), model.num_errors, dtype=torch.float64)
    return model.inject_error(zeros)[0].numpy().astype(np.int64)


@pytest.mark.parametrize("distance", [3, 5])
def test_decomposition_matches_stim_mechanism_draws(distance, monkeypatch):
    circuit, _ = _generated(distance)
    dem = circuit.detector_error_model(decompose_errors=True)
    supports, incidence, expected_priors = _oracle(dem)
    h, logical, priors = matrix_stim._build_dem_matrices(circuit, decompose=True)
    actual = [(tuple(h[:, col].nonzero()[0]), tuple(logical[:, col].nonzero()[0]))
              for col in range(h.shape[1])]
    assert len(actual) == len(set(actual)) == len(supports)
    assert set(actual) == set(supports)
    assert np.all(h.getnnz(axis=0) <= 2)
    order = [supports.index(support) for support in actual]
    np.testing.assert_allclose(priors, expected_priors[order], rtol=1e-12, atol=1e-15)
    model = error_stim.create(dict(circuit=str(circuit), decompose_errors=True, device={"device_type": "cpu"}))
    np.testing.assert_array_equal(model.priors.numpy(), priors)
    detectors, observables, draws = dem.compile_sampler(seed=30).sample(128, return_errors=True)
    errors = _replay(model, draws, monkeypatch)
    np.testing.assert_array_equal(errors, (draws.astype(np.int64) @ incidence % 2)[:, order])
    np.testing.assert_array_equal(h @ errors.T % 2, detectors.T)
    np.testing.assert_array_equal(logical @ errors.T % 2, observables.T)


def test_correlated_cancellation_and_independent_prior_merging(monkeypatch):
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0 ^ D1
        error(0.2) D0 L0
        error(0.3) D0 L0 ^ D0 L0 ^ L1
        error(0.4) D0 D0 D2 L0 L0
        error(0.05) L0
        error(0.06) D0
        error(0.07) D3 ^ D3
    """)

    class Circuit:
        def __str__(self):
            return str(dem)

        def detector_error_model(self, **kwargs):
            return dem

    circuit = Circuit()
    monkeypatch.setattr(error_stim, "get_stim_circuit", lambda **kwargs: circuit)
    h, logical, priors = matrix_stim._build_dem_matrices(circuit, decompose=True)
    supports = [(tuple(h[:, col].nonzero()[0]), tuple(logical[:, col].nonzero()[0]))
                for col in range(h.shape[1])]
    expected = {((0,), (0,)): 0.26, ((1,), ()): 0.1, ((), (1,)): 0.3,
                ((2,), ()): 0.4, ((), (0,)): 0.05, ((0,), ()): 0.06}
    assert set(supports) == set(expected)
    np.testing.assert_allclose(priors, [expected[s] for s in supports], atol=1e-15)
    draws = np.array(list(product([False, True], repeat=7)), dtype=np.bool_)
    detectors, observables, _ = dem.compile_sampler().sample(len(draws), recorded_errors_to_replay=draws)
    model = error_stim.create(dict(circuit="synthetic", decompose_errors=True, device={"device_type": "cpu"}))
    errors = _replay(model, draws, monkeypatch)
    np.testing.assert_array_equal(h @ errors.T % 2, detectors.T)
    np.testing.assert_array_equal(logical @ errors.T % 2, observables.T)
    np.testing.assert_array_equal(errors[:, supports.index(((0,), (0,)))], draws[:, 0] ^ draws[:, 1])
    np.testing.assert_array_equal(errors[:, supports.index(((1,), ()))], draws[:, 0])


def test_decomposed_rate_sweep_preserves_correlated_draws():
    circuit, gen = _generated()
    model = error_stim.create(dict(circuit=str(circuit), decompose_errors=True, device={"device_type": "cpu"},
                                  rate=[0.005, 0.015, 3], circuit_gen=gen), training=True)
    h, logical, _ = matrix_stim._build_dem_matrices(circuit, decompose=True)
    actual = [(tuple(h[:, col].nonzero()[0]), tuple(logical[:, col].nonzero()[0]))
              for col in range(h.shape[1])]
    mechanism_table, component_table = [], []
    for rate in torch.linspace(0.005, 0.015, 3, dtype=torch.float64).tolist():
        dem = _generated(rate=rate)[0].detector_error_model(decompose_errors=True)
        supports, incidence, priors = _oracle(dem)
        mechanism_table.append([i.args_copy()[0] for i in dem.flattened() if i.type == "error"])
        component_table.append(priors[[supports.index(s) for s in actual]])
    mechanism_table = torch.tensor(mechanism_table, dtype=torch.float64)
    component_table = torch.from_numpy(np.asarray(component_table))
    torch.manual_seed(302)
    indices = torch.randint(3, (16,))
    draws = torch.rand_like(mechanism_table[indices]) < mechanism_table[indices]
    expected = (draws.numpy().astype(np.int64) @ incidence % 2)[:, [supports.index(s) for s in actual]]
    torch.manual_seed(302)
    errors, _ = model.inject_error(torch.zeros(16, model.num_errors, dtype=torch.float64))
    np.testing.assert_array_equal(errors.numpy(), expected)
    expected_prior = component_table[indices].clamp(1e-12, 1 - 1e-12)
    torch.testing.assert_close(model.get_llr(errors), torch.log((1 - expected_prior) / expected_prior))


def test_interface_uses_decomposed_columns_consistently():
    circuit, _ = _generated()
    interface = create_interface(cfg=dict(backend="stim", circuit=str(circuit), decompose_errors=True))
    h, logical, _ = matrix_stim._build_dem_matrices(circuit, decompose=True)
    np.testing.assert_array_equal(interface.matrix_bundle.Hx_matrix.get_dense(), h.toarray())
    np.testing.assert_array_equal(interface.matrix_bundle.lx_matrix, logical.toarray())
    assert interface.error_model.num_errors == h.shape[1]
    errors, _ = interface.error_model.inject_error(torch.zeros(32, h.shape[1], dtype=torch.float64))
    syndrome = interface.syndrome_generator.measure_syndrome(errors, None)
    np.testing.assert_array_equal(syndrome.numpy(), (h @ errors.numpy().T % 2).T)
    np.testing.assert_array_equal(interface.syndrome_generator.observable_flips.numpy(),
                                  (logical @ errors.numpy().T % 2).T)


@pytest.mark.parametrize("sweep", [False, True])
def test_false_preserves_extraction_sampling_and_rng(sweep):
    circuit, gen = _generated()
    dem = circuit.detector_error_model(decompose_errors=False)
    instructions = [inst for inst in dem.flattened() if inst.type == "error"]
    expected_h = np.zeros((dem.num_detectors, len(instructions)), dtype=np.int64)
    expected_l = np.zeros((dem.num_observables, len(instructions)), dtype=np.int64)
    for col, inst in enumerate(instructions):
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                expected_h[target.val, col] ^= 1
            elif target.is_logical_observable_id():
                expected_l[target.val, col] ^= 1
    base_priors = torch.tensor([i.args_copy()[0] for i in instructions], dtype=torch.float64)
    cfg = dict(circuit=str(circuit), device={"device_type": "cpu"})
    if sweep:
        cfg.update(rate=[0.005, 0.015, 3], circuit_gen=gen)
        table = torch.tensor([
            [i.args_copy()[0] for i in _generated(rate=rate)[0].detector_error_model().flattened()
             if i.type == "error"]
            for rate in torch.linspace(0.005, 0.015, 3, dtype=torch.float64).tolist()
        ], dtype=torch.float64)
    codeword = torch.zeros(16, len(instructions), dtype=torch.float64)
    codeword[::2, ::3] = 1
    torch.manual_seed(301)
    expected_prior = table[torch.randint(3, (16,))] if sweep else base_priors.expand_as(codeword)
    expected_error = torch.where(torch.rand_like(codeword) < expected_prior, 1 - codeword, codeword)
    expected_state = torch.get_rng_state()
    expected_llr = torch.log((1 - expected_prior.clamp(1e-12, 1 - 1e-12)) /
                             expected_prior.clamp(1e-12, 1 - 1e-12))
    for flag in ({}, {"decompose_errors": False}):
        h, logical, priors = matrix_stim._build_dem_matrices(circuit, decompose=flag.get("decompose_errors", False))
        np.testing.assert_array_equal(h.toarray(), expected_h)
        np.testing.assert_array_equal(logical.toarray(), expected_l)
        np.testing.assert_array_equal(priors, base_priors.numpy())
        model = error_stim.create({**cfg, **flag}, training=sweep)
        if sweep:
            assert torch.equal(model._prior_table, table)
        torch.manual_seed(301)
        error, _ = model.inject_error(codeword)
        assert torch.equal(error, expected_error)
        assert torch.equal(model.get_llr(error), expected_llr)
        assert torch.equal(torch.get_rng_state(), expected_state)


@pytest.mark.parametrize("flag", [0, 1, "false", None])
def test_decompose_requires_bool(flag):
    circuit, _ = _generated()
    for build in (
        lambda: matrix_stim._build_dem_matrices(circuit, decompose=flag),
        lambda: matrix_stim.create(dict(circuit=str(circuit), decompose_errors=flag), device=torch.device("cpu")),
        lambda: error_stim.create(dict(circuit=str(circuit), decompose_errors=flag)),
        lambda: syndrome_stim.create(dict(circuit=str(circuit), decompose_errors=flag)),
        lambda: create_interface(cfg=dict(backend="stim", circuit=str(circuit), decompose_errors=flag)),
    ):
        with pytest.raises(ValueError, match="decompose_errors"):
            build()


def test_undecomposable_mechanism_is_rejected():
    circuit = stim.Circuit("""
        R 0 1 2
        CORRELATED_ERROR(0.1) X0 X1 X2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    assert matrix_stim._build_dem_matrices(circuit)[0].getnnz(axis=0).tolist() == [3]
    with pytest.raises(ValueError, match="decompose_errors"):
        matrix_stim._build_dem_matrices(circuit, decompose=True)
    with pytest.raises(ValueError, match="decompose_errors"):
        error_stim.create(dict(circuit=str(circuit), decompose_errors=True))
