import os
import sys

import stim
import torch

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.interface import create_interface
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import parse_device_dtype

BATCH_SIZE = 10
NOISE = 0.01


def _circuit_and_constants(rounds):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=NOISE,
        after_reset_flip_probability=NOISE,
        before_measure_flip_probability=NOISE,
        before_round_data_depolarization=NOISE,
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    num_errors = sum(1 for i in dem.flattened() if i.type == "error")
    return str(circuit), dem.num_detectors, num_errors, dem.num_observables


# rounds=1 circuit is used by the 1-channel decoder helpers below
_CIRCUIT_STR_1CH, NUM_DETECTORS, NUM_ERRORS, NUM_OBSERVABLES = _circuit_and_constants(1)


def make_interface(rounds=1, measurement_error_rate=0.0):
    return create_interface(
        cfg={
            "backend": "stim",
            "code": "surface_code:rotated_memory_x",
            "distance": 3,
        },
        decoding_cfg={
            "dtype": "float64",
            "device": {"device_type": "cpu", "device_idx": 0},
        },
        error_cfg={
            "model": "stim_circuit",
            "after_clifford_depolarization": NOISE,
            "after_reset_flip_probability": NOISE,
            "before_measure_flip_probability": NOISE,
            "before_round_data_depolarization": NOISE,
            "number_channel": 1,
        },
        syndrome_cfg={
            "measure": "stim",
            "rounds": rounds,
            "measurement_error_rate": measurement_error_rate,
        },
    )


def _decoder_cfg_1ch():
    return {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "check_type": "hx",
        "dtype": "float64",
        "device": {"device_type": "cpu", "device_idx": 0},
        "config": [{"max_iter": 50}],  # osd_0 configures nothing
    }


def _matrix_cfg_1ch():
    check_cfg = {"file_type": "stim", "circuit": _CIRCUIT_STR_1CH, "target": "check"}
    obs_cfg = {"file_type": "stim", "circuit": _CIRCUIT_STR_1CH, "target": "observable"}
    return {
        "parity_matrix_hx": check_cfg,
        "parity_matrix_hz": check_cfg,
        "logical_check_matrix": True,
        "logical_check_lx": obs_cfg,
        "logical_check_lz": obs_cfg,
    }


def make_decoders():
    dec_cfg = _decoder_cfg_1ch()
    bundle = load_matrices(_matrix_cfg_1ch(), *parse_device_dtype(dec_cfg))
    return create_decoder(cfg=dec_cfg, bundle=bundle)


def make_2ch_components():
    """Create 2-channel (depol + bp4) components — all from config dicts."""
    decoding_cfg = {
        "algorithm": "bp4",
        "dtype": "float64",
        "device": {"device_type": "cpu", "device_idx": 0},
        "config": {"max_iter": 50, "damping_factor": 0.1},
    }
    matrix_cfg = {
        "parity_matrix_hx": {
            "file_type": "alist",
            "path": "examples/alist/surface/surface_10_hx.alist",
        },
        "parity_matrix_hz": {
            "file_type": "alist",
            "path": "examples/alist/surface/surface_10_hz.alist",
        },
        "logical_check_matrix": True,
        "logical_check_lx": {
            "file_type": "alist",
            "path": "examples/alist/surface/surface_10_lx.alist",
        },
        "logical_check_lz": {
            "file_type": "alist",
            "path": "examples/alist/surface/surface_10_lz.alist",
        },
    }
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoding_cfg))
    decoders = create_decoder(cfg=decoding_cfg, bundle=bundle)

    error_model = create_error_model(
        cfg={
            "model": "depol",
            "device": {"device_type": "cpu", "device_idx": 0},
            "rate": 0.05,
        }
    )

    syndrome_gen = create_syndrome(cfg={"measure": "perfect"})
    logical_chk = create_check(cfg={"check_type": "lx"})

    return decoders, error_model, syndrome_gen, logical_chk, bundle


class TestSyndromeDimensions:

    def test_stim_d1(self):
        iface = make_interface(rounds=1)
        err = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        synd = iface.syndrome_generator.measure_syndrome(err, None)
        obs = iface.syndrome_generator.observable_flips
        assert synd.shape == (BATCH_SIZE, NUM_DETECTORS)
        assert obs.shape == (BATCH_SIZE, NUM_OBSERVABLES)

    def test_stim_d3(self):
        # Stim bakes all QEC rounds into the detector vector, so the output is
        # always 2-D [B, num_detectors] regardless of rounds.
        iface = make_interface(rounds=3)
        _, nd3, ne3, no3 = _circuit_and_constants(3)
        err = torch.zeros(BATCH_SIZE, ne3, dtype=torch.float64)
        synd = iface.syndrome_generator.measure_syndrome(err, None)
        obs = iface.syndrome_generator.observable_flips
        assert synd.shape == (BATCH_SIZE, nd3)
        assert obs.shape == (BATCH_SIZE, no3)

    def test_stim_d3_noisy(self):
        # measurement_error_rate is folded into the stim circuit's
        # before_measure_flip_probability; output shape is unchanged.
        iface = make_interface(rounds=3, measurement_error_rate=0.05)
        _, nd3, ne3, no3 = _circuit_and_constants(3)
        err = torch.zeros(BATCH_SIZE, ne3, dtype=torch.float64)
        synd = iface.syndrome_generator.measure_syndrome(err, None)
        obs = iface.syndrome_generator.observable_flips
        assert synd.shape == (BATCH_SIZE, nd3)
        assert obs.shape == (BATCH_SIZE, no3)

    def test_stim_d1_noisy(self):
        iface = make_interface(rounds=1, measurement_error_rate=0.05)
        err = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        synd = iface.syndrome_generator.measure_syndrome(err, None)
        assert synd.shape == (BATCH_SIZE, NUM_DETECTORS)


class TestDecoderWrapperDimensions:

    def test_2d_passthrough(self):
        decoders = make_decoders()
        d = decoders[0]
        H = d.decoder.H_matrix
        synd = torch.zeros(BATCH_SIZE, NUM_DETECTORS, dtype=torch.float64)
        llr = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        out = d({"synd": synd, "llr0": llr, "H_matrix": H})
        assert out["e_v"].shape == (BATCH_SIZE, NUM_ERRORS)
        assert out["synd"].ndim == 2

    def test_3d_flatten_unflatten(self):
        decoders = make_decoders()
        d = decoders[0]
        H = d.decoder.H_matrix
        synd = torch.zeros(BATCH_SIZE, 3, NUM_DETECTORS, dtype=torch.float64)
        llr = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        out = d({"synd": synd, "llr0": llr, "H_matrix": H})
        assert out["e_v"].shape == (BATCH_SIZE, 3, NUM_ERRORS)
        assert out["llr0"].shape == (BATCH_SIZE, NUM_ERRORS)

    def test_wrapper_preserves_attrs(self):
        decoders = make_decoders()
        d = decoders[0]
        assert d.algo == "bp_norm_min_sum"
        assert hasattr(d, "dtype")
        assert hasattr(d, "device")
        assert hasattr(d, "num_max_iter")


class TestLogicalCheckDimensions:

    def test_single_round(self):
        lc = create_check(cfg={"check_type": "stim"})
        e_v = torch.zeros(BATCH_SIZE, NUM_ERRORS)
        obs = torch.zeros(BATCH_SIZE, NUM_OBSERVABLES)
        l_mat = torch.zeros(NUM_OBSERVABLES, NUM_ERRORS).numpy()
        assert lc.check(e_v, obs, l_mat).shape == (BATCH_SIZE,)

    def test_multi_round(self):
        lc = create_check(cfg={"check_type": "stim"})
        e_v = torch.zeros(BATCH_SIZE, NUM_ERRORS)
        obs = torch.zeros(BATCH_SIZE, 3, NUM_OBSERVABLES)
        l_mat = torch.zeros(NUM_OBSERVABLES, NUM_ERRORS).numpy()
        assert lc.check(e_v, obs, l_mat).shape == (BATCH_SIZE,)

    def test_multi_round_uses_the_final_round(self):
        """A multi-round check must judge against the last round, not vote the rounds.

        Rounds 0 and 1 agree with the prediction and round 2 does not, so a majority
        over the per-round verdicts would report success while the final round reports
        a logical error. The final round is the one stim's own decoders compare
        against, so it is the verdict that counts.
        """
        lc = create_check(cfg={"check_type": "stim"})
        l_mat = torch.eye(1).numpy()  # predicted_obs == e_v
        e_v = torch.ones(BATCH_SIZE, 1)
        obs = torch.ones(BATCH_SIZE, 3, 1)
        obs[:, 2, :] = 0.0  # only the last round differs

        assert torch.equal(lc.check(e_v, obs, l_mat), torch.ones(BATCH_SIZE).long())


class TestMatrixBundleDimensions:

    def test_l_matrix_1ch_hx(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix("hx", 1).shape == (NUM_OBSERVABLES, NUM_ERRORS)

    def test_l_matrix_1ch_hz(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix("hz", 1).shape == (NUM_OBSERVABLES, NUM_ERRORS)

    def test_l_matrix_2ch(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix("hx", 2).shape == (NUM_OBSERVABLES, 2, NUM_ERRORS)

    def test_select_hx(self):
        b = make_interface().matrix_bundle
        shape, _, _, H = b.select("hx")
        assert shape == (NUM_DETECTORS, NUM_ERRORS)
        assert H.shape == (NUM_DETECTORS, NUM_ERRORS)


class TestPipelineDimensions:

    def _run_pipeline(self, rounds, measurement_error_rate=0.0):
        iface = make_interface(
            rounds=rounds, measurement_error_rate=measurement_error_rate
        )
        # decoders must share the interface's circuit so their H matrix matches
        # the syndrome size produced by `syndrome_generator.measure_syndrome`.
        decoders = create_decoder(cfg=_decoder_cfg_1ch(), bundle=iface.matrix_bundle)

        error_model = iface.error_model
        syndrome_generator = iface.syndrome_generator
        logical_check = iface.logical_check
        bundle = iface.matrix_bundle
        number_channel = error_model.number_channel
        check_type = "hx"

        shape, _, _, _ = bundle.Hx_matrix.get_index()
        H_matrix = bundle.select(check_type)[3]
        l_matrix = bundle.get_l_matrix(check_type, number_channel)

        zero_qubits = torch.zeros([BATCH_SIZE, shape[1]], dtype=torch.float64)
        _, error_dataloader = error_model.inject_error(zero_qubits, BATCH_SIZE)

        shapes = {}
        for err, llr, _ in error_dataloader:
            synd = syndrome_generator.measure_syndrome(err, decoders[0])
            shapes["synd_raw"] = list(synd.shape)

            io_dict = {"synd": synd, "llr0": llr, "H_matrix": H_matrix}

            for decoder_idx in range(len(decoders)):
                io_dict = decoders[decoder_idx](io_dict)
                shapes[f"e_v_after_decoder_{decoder_idx}"] = list(io_dict["e_v"].shape)

            obs = syndrome_generator.observable_flips
            shapes["obs_flips"] = list(obs.shape)
            check = logical_check.check(io_dict["e_v"], obs.to(torch.float64), l_matrix)
            shapes["check"] = list(check.shape)
            break

        return shapes

    def test_d1(self):
        s = self._run_pipeline(rounds=1)
        assert s["synd_raw"] == [BATCH_SIZE, NUM_DETECTORS]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, NUM_ERRORS]
        assert s["e_v_after_decoder_1"] == [BATCH_SIZE, NUM_ERRORS]
        assert s["obs_flips"] == [BATCH_SIZE, NUM_OBSERVABLES]
        assert s["check"] == [BATCH_SIZE]

    def test_d3(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3)
        assert s["synd_raw"] == [BATCH_SIZE, nd3]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, ne3]
        assert s["e_v_after_decoder_1"] == [BATCH_SIZE, ne3]
        assert s["check"] == [BATCH_SIZE]

    def test_d3_noisy(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3, measurement_error_rate=0.05)
        assert s["synd_raw"] == [BATCH_SIZE, nd3]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, ne3]
        assert s["check"] == [BATCH_SIZE]

    def test_d5(self):
        _, nd5, ne5, _ = _circuit_and_constants(5)
        s = self._run_pipeline(rounds=5)
        assert s["synd_raw"] == [BATCH_SIZE, nd5]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, ne5]
        assert s["e_v_after_decoder_1"] == [BATCH_SIZE, ne5]
        assert s["check"] == [BATCH_SIZE]


class TestTwoChannelDimensions:

    def _run_2ch_pipeline(self, rounds=1):
        decoders, error_model, syndrome_gen, logical_chk, bundle = make_2ch_components()

        number_channel = error_model.number_channel
        assert number_channel == 2

        check_type = "hx"
        shape, _, _, _ = bundle.Hx_matrix.get_index()
        H_matrix = bundle.select(check_type)[3]
        l_matrix = bundle.get_l_matrix(check_type, number_channel)
        dtype = decoders[0].dtype
        B = BATCH_SIZE
        N, M = shape[1], shape[0]

        zero_qubits = torch.zeros([B, N], dtype=dtype)
        _, error_dataloader = error_model.inject_error(zero_qubits, B)

        shapes = {}
        for err, llr, _ in error_dataloader:
            shapes["err"] = list(err.shape)

            synd = syndrome_gen.measure_syndrome(err, decoders[0])
            shapes["synd_raw"] = list(synd.shape)

            if rounds > 1:
                synd = synd.unsqueeze(1).expand(B, rounds, *synd.shape[1:]).clone()
                shapes["synd_with_rounds"] = list(synd.shape)

            io_dict = {"synd": synd, "llr0": llr, "H_matrix": H_matrix}

            for decoder_idx in range(len(decoders)):
                io_dict = decoders[decoder_idx](io_dict)
                shapes[f"e_v_after_decoder_{decoder_idx}"] = list(io_dict["e_v"].shape)

            if io_dict["e_v"].ndim == err.ndim:
                check = logical_chk.check(io_dict["e_v"], err, l_matrix)
                shapes["check"] = list(check.shape)
            break

        return shapes, N, M

    def test_2ch_d1(self):
        s, N, M = self._run_2ch_pipeline(rounds=1)
        assert s["err"] == [BATCH_SIZE, 2, N]
        assert s["synd_raw"] == [BATCH_SIZE, 2, M]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, 2, N]
        assert s["check"] == [BATCH_SIZE, 2]

    def test_2ch_d3_keeps_the_rounds_axis(self):
        """A 2-channel multi-round syndrome now reaches the decoder with rounds intact.

        Voting used to collapse the rounds axis before the decoder, so this shape was
        previously [B, C, M]. With voting gone the axis survives and
        `RoundFlattenWrapper` folds it into the batch instead, so the decoder returns
        [B, d, C, N] rather than [B, C, N].
        """
        s, N, M = self._run_2ch_pipeline(rounds=3)
        assert s["synd_with_rounds"] == [BATCH_SIZE, 3, 2, M]
        assert s["e_v_after_decoder_0"] == [BATCH_SIZE, 3, 2, N]
        assert "check" not in s  # no rounds-collapsed e_v for the check to consume
