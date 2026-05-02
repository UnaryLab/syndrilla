import sys, os
import stim
import torch
import pytest

sys.path.append(os.getcwd())

from syndrilla.interface import create_interface
from syndrilla.decoder import create_decoder
from syndrilla.syndrome import create_syndrome
from syndrilla.error_model import create_error_model
from syndrilla.logical_check import create_check
from syndrilla.vote import create_vote
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype


BATCH_SIZE = 10
NOISE = 0.01


def _circuit_and_constants(rounds):
    circuit = stim.Circuit.generated(
        'surface_code:rotated_memory_x', distance=3, rounds=rounds,
        after_clifford_depolarization=NOISE, after_reset_flip_probability=NOISE,
        before_measure_flip_probability=NOISE, before_round_data_depolarization=NOISE,
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    num_errors = sum(1 for i in dem.flattened() if i.type == 'error')
    return str(circuit), dem.num_detectors, num_errors, dem.num_observables


# rounds=1 circuit is used by the 1-channel decoder helpers below
_CIRCUIT_STR_1CH, NUM_DETECTORS, NUM_ERRORS, NUM_OBSERVABLES = _circuit_and_constants(1)


# ── helpers ──────────────────────────────────────────────────────────
def make_interface(rounds=1, measurement_error_rate=0.0):
    return create_interface(
        cfg={
            'backend': 'stim',
            'code': 'surface_code:rotated_memory_x',
            'distance': 3,
        },
        decoder_cfg={
            'dtype': 'float64',
            'device': {'device_type': 'cpu', 'device_idx': 0},
        },
        error_cfg={
            'model': 'stim_circuit',
            'after_clifford_depolarization': NOISE,
            'after_reset_flip_probability': NOISE,
            'before_measure_flip_probability': NOISE,
            'before_round_data_depolarization': NOISE,
            'number_channel': 1,
        },
        syndrome_cfg={
            'measure': 'stim',
            'rounds': rounds,
            'measurement_error_rate': measurement_error_rate,
        },
    )


def make_voter():
    return create_vote(cfg={'method': 'majority_vote'})


def _decoder_cfg_1ch():
    return {
        'algorithm': ['bp_norm_min_sum', 'osd_0'],
        'check_type': 'hx',
        'max_iter': 50,
        'dtype': 'float64',
        'device': {'device_type': 'cpu', 'device_idx': 0},
    }


def _matrix_cfg_1ch():
    check_cfg = {'file_type': 'stim', 'circuit': _CIRCUIT_STR_1CH, 'target': 'check'}
    obs_cfg = {'file_type': 'stim', 'circuit': _CIRCUIT_STR_1CH, 'target': 'observable'}
    return {
        'parity_matrix_hx': check_cfg,
        'parity_matrix_hz': check_cfg,
        'logical_check_matrix': True,
        'logical_check_lx': obs_cfg,
        'logical_check_lz': obs_cfg,
    }


def make_decoders():
    dec_cfg = _decoder_cfg_1ch()
    bundle = load_matrices(_matrix_cfg_1ch(), *parse_device_dtype(dec_cfg))
    return create_decoder(cfg=dec_cfg, bundle=bundle)


def make_2ch_components():
    """Create 2-channel (depol + bp4) components — all from config dicts."""
    decoder_cfg = {
        'algorithm': 'bp4',
        'max_iter': 50,
        'dtype': 'float64',
        'damping_factor': 0.1,
        'device': {'device_type': 'cpu', 'device_idx': 0},
    }
    matrix_cfg = {
        'parity_matrix_hx': 'examples/alist/hx.matrix.yaml',
        'parity_matrix_hz': 'examples/alist/hz.matrix.yaml',
        'logical_check_matrix': True,
        'logical_check_lx': 'examples/alist/lx.matrix.yaml',
        'logical_check_lz': 'examples/alist/lz.matrix.yaml',
    }
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
    decoders = create_decoder(cfg=decoder_cfg, bundle=bundle)

    error_model = create_error_model(cfg={
        'model': 'depol',
        'device': {'device_type': 'cpu', 'device_idx': 0},
        'rate': 0.05,
    })

    syndrome_gen = create_syndrome(cfg={'measure': 'perfect'})
    logical_chk = create_check(cfg={'check_type': 'lx'})

    return decoders, error_model, syndrome_gen, logical_chk, bundle


# ── syndrome dimension tests ────────────────────────────────────────
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


# ── voter dimension tests ───────────────────────────────────────────
class TestVoterDimensions:

    def test_noop_d1(self):
        voter = make_voter()
        t = torch.zeros(BATCH_SIZE, NUM_DETECTORS)
        out = voter.apply(t, number_channel=1, rounds=1,
                          vote_stage='syndrome', current_stage='syndrome')
        assert out.shape == (BATCH_SIZE, NUM_DETECTORS)

    def test_vote_syndrome_1ch(self):
        voter = make_voter()
        t = torch.zeros(BATCH_SIZE, 3, NUM_DETECTORS)
        out = voter.apply(t, number_channel=1, rounds=3,
                          vote_stage='syndrome', current_stage='syndrome')
        assert out.shape == (BATCH_SIZE, NUM_DETECTORS)

    def test_vote_syndrome_2ch(self):
        voter = make_voter()
        t = torch.zeros(BATCH_SIZE, 3, 2, NUM_DETECTORS)
        out = voter.apply(t, number_channel=2, rounds=3,
                          vote_stage='syndrome', current_stage='syndrome')
        assert out.shape == (BATCH_SIZE, 2, NUM_DETECTORS)

    def test_vote_stage_mismatch_noop(self):
        voter = make_voter()
        t = torch.zeros(BATCH_SIZE, 3, NUM_DETECTORS)
        out = voter.apply(t, number_channel=1, rounds=3,
                          vote_stage='decoder_0', current_stage='syndrome')
        assert out.shape == (BATCH_SIZE, 3, NUM_DETECTORS)

    def test_vote_decoder_e_v(self):
        voter = make_voter()
        t = torch.zeros(BATCH_SIZE, 5, NUM_ERRORS)
        out = voter.apply(t, number_channel=1, rounds=5,
                          vote_stage='decoder_0', current_stage='decoder_0')
        assert out.shape == (BATCH_SIZE, NUM_ERRORS)


# ── decoder wrapper dimension tests ─────────────────────────────────
class TestDecoderWrapperDimensions:

    def test_2d_passthrough(self):
        decoders = make_decoders()
        d = decoders[0]
        H = d.decoder.H_matrix
        synd = torch.zeros(BATCH_SIZE, NUM_DETECTORS, dtype=torch.float64)
        llr = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        out = d({'synd': synd, 'llr0': llr, 'H_matrix': H})
        assert out['e_v'].shape == (BATCH_SIZE, NUM_ERRORS)
        assert out['synd'].ndim == 2

    def test_3d_flatten_unflatten(self):
        decoders = make_decoders()
        d = decoders[0]
        H = d.decoder.H_matrix
        synd = torch.zeros(BATCH_SIZE, 3, NUM_DETECTORS, dtype=torch.float64)
        llr = torch.zeros(BATCH_SIZE, NUM_ERRORS, dtype=torch.float64)
        out = d({'synd': synd, 'llr0': llr, 'H_matrix': H})
        assert out['e_v'].shape == (BATCH_SIZE, 3, NUM_ERRORS)
        assert out['llr0'].shape == (BATCH_SIZE, NUM_ERRORS)

    def test_wrapper_preserves_attrs(self):
        decoders = make_decoders()
        d = decoders[0]
        assert d.algo == 'bp_norm_min_sum'
        assert hasattr(d, 'dtype')
        assert hasattr(d, 'device')
        assert hasattr(d, 'num_max_iter')


# ── logical check dimension tests ───────────────────────────────────
class TestLogicalCheckDimensions:

    def test_single_round(self):
        lc = create_check(cfg={'check_type': 'stim'})
        e_v = torch.zeros(BATCH_SIZE, NUM_ERRORS)
        obs = torch.zeros(BATCH_SIZE, NUM_OBSERVABLES)
        l_mat = torch.zeros(NUM_OBSERVABLES, NUM_ERRORS).numpy()
        assert lc.check(e_v, obs, l_mat).shape == (BATCH_SIZE,)

    def test_multi_round(self):
        lc = create_check(cfg={'check_type': 'stim'})
        e_v = torch.zeros(BATCH_SIZE, NUM_ERRORS)
        obs = torch.zeros(BATCH_SIZE, 3, NUM_OBSERVABLES)
        l_mat = torch.zeros(NUM_OBSERVABLES, NUM_ERRORS).numpy()
        assert lc.check(e_v, obs, l_mat).shape == (BATCH_SIZE,)


# ── matrix bundle dimension tests ───────────────────────────────────
class TestMatrixBundleDimensions:

    def test_l_matrix_1ch_hx(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix('hx', 1).shape == (NUM_OBSERVABLES, NUM_ERRORS)

    def test_l_matrix_1ch_hz(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix('hz', 1).shape == (NUM_OBSERVABLES, NUM_ERRORS)

    def test_l_matrix_2ch(self):
        b = make_interface().matrix_bundle
        assert b.get_l_matrix('hx', 2).shape == (NUM_OBSERVABLES, 2, NUM_ERRORS)

    def test_select_hx(self):
        b = make_interface().matrix_bundle
        shape, _, _, H = b.select('hx')
        assert shape == (NUM_DETECTORS, NUM_ERRORS)
        assert H.shape == (NUM_DETECTORS, NUM_ERRORS)


# ── end-to-end pipeline dimension tests ─────────────────────────────
class TestPipelineDimensions:

    def _run_pipeline(self, rounds, vote_stage, measurement_error_rate=0.0):
        iface = make_interface(rounds=rounds,
                               measurement_error_rate=measurement_error_rate)
        # decoders must share the interface's circuit so their H matrix matches
        # the syndrome size produced by `syndrome_generator.measure_syndrome`.
        decoders = create_decoder(cfg=_decoder_cfg_1ch(), bundle=iface.matrix_bundle)
        voter = make_voter()

        error_model = iface.error_model
        syndrome_generator = iface.syndrome_generator
        logical_check = iface.logical_check
        bundle = iface.matrix_bundle
        number_channel = error_model.number_channel
        check_type = 'hx'

        shape, _, _, _ = bundle.Hx_matrix.get_index()
        H_matrix = bundle.select(check_type)[3]
        l_matrix = bundle.get_l_matrix(check_type, number_channel)

        zero_qubits = torch.zeros([BATCH_SIZE, shape[1]], dtype=torch.float64)
        _, error_dataloader = error_model.inject_error(zero_qubits, BATCH_SIZE)

        shapes = {}
        for err, llr, _ in error_dataloader:
            dr = getattr(syndrome_generator, 'rounds', 1)

            synd = syndrome_generator.measure_syndrome(err, decoders[0])
            shapes['synd_raw'] = list(synd.shape)

            synd = voter.apply(synd, number_channel, rounds=dr,
                               vote_stage=vote_stage, current_stage='syndrome')
            shapes['synd_after_vote'] = list(synd.shape)

            io_dict = {'synd': synd, 'llr0': llr, 'H_matrix': H_matrix}

            for decoder_idx in range(len(decoders)):
                io_dict = decoders[decoder_idx](io_dict)
                decoder_stage = f'decoder_{decoder_idx}'
                vote_kwargs = dict(rounds=dr, vote_stage=vote_stage, current_stage=decoder_stage)
                io_dict['e_v'] = voter.apply(io_dict['e_v'], number_channel=1, **vote_kwargs)
                io_dict['synd'] = voter.apply(io_dict['synd'], number_channel, **vote_kwargs)
                for key in ('llr', 'converge', 'iter'):
                    if key in io_dict and io_dict[key].ndim > 1:
                        io_dict[key] = voter.select_round(io_dict[key], **vote_kwargs)
                shapes[f'e_v_after_decoder_{decoder_idx}'] = list(io_dict['e_v'].shape)

            obs = syndrome_generator.observable_flips
            shapes['obs_flips'] = list(obs.shape)
            check = logical_check.check(io_dict['e_v'], obs.to(torch.float64), l_matrix)
            shapes['check'] = list(check.shape)
            break

        return shapes

    def test_d1_vote_syndrome(self):
        s = self._run_pipeline(rounds=1, vote_stage='syndrome')
        assert s['synd_raw'] == [BATCH_SIZE, NUM_DETECTORS]
        assert s['synd_after_vote'] == [BATCH_SIZE, NUM_DETECTORS]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, NUM_ERRORS]
        assert s['e_v_after_decoder_1'] == [BATCH_SIZE, NUM_ERRORS]
        assert s['obs_flips'] == [BATCH_SIZE, NUM_OBSERVABLES]
        assert s['check'] == [BATCH_SIZE]

    def test_d3_vote_syndrome(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3, vote_stage='syndrome')
        assert s['synd_raw'] == [BATCH_SIZE, nd3]
        assert s['synd_after_vote'] == [BATCH_SIZE, nd3]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, ne3]
        assert s['e_v_after_decoder_1'] == [BATCH_SIZE, ne3]
        assert s['check'] == [BATCH_SIZE]

    def test_d3_vote_decoder_0(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3, vote_stage='decoder_0')
        assert s['synd_raw'] == [BATCH_SIZE, nd3]
        assert s['synd_after_vote'] == [BATCH_SIZE, nd3]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, ne3]
        assert s['e_v_after_decoder_1'] == [BATCH_SIZE, ne3]
        assert s['check'] == [BATCH_SIZE]

    def test_d3_vote_decoder_1(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3, vote_stage='decoder_1')
        assert s['synd_raw'] == [BATCH_SIZE, nd3]
        assert s['synd_after_vote'] == [BATCH_SIZE, nd3]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, ne3]
        assert s['e_v_after_decoder_1'] == [BATCH_SIZE, ne3]
        assert s['check'] == [BATCH_SIZE]

    def test_d3_noisy_vote_syndrome(self):
        _, nd3, ne3, _ = _circuit_and_constants(3)
        s = self._run_pipeline(rounds=3, vote_stage='syndrome',
                               measurement_error_rate=0.05)
        assert s['synd_raw'] == [BATCH_SIZE, nd3]
        assert s['synd_after_vote'] == [BATCH_SIZE, nd3]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, ne3]
        assert s['check'] == [BATCH_SIZE]

    def test_d5_vote_decoder_0(self):
        _, nd5, ne5, _ = _circuit_and_constants(5)
        s = self._run_pipeline(rounds=5, vote_stage='decoder_0')
        assert s['synd_raw'] == [BATCH_SIZE, nd5]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, ne5]
        assert s['e_v_after_decoder_1'] == [BATCH_SIZE, ne5]
        assert s['check'] == [BATCH_SIZE]


# ── 2-channel dimension tests ───────────────────────────────────────
class TestTwoChannelDimensions:

    def _run_2ch_pipeline(self, rounds=1, vote_stage='syndrome'):
        decoders, error_model, syndrome_gen, logical_chk, bundle = make_2ch_components()
        voter = make_voter()

        number_channel = error_model.number_channel
        assert number_channel == 2

        check_type = 'hx'
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
            shapes['err'] = list(err.shape)

            synd = syndrome_gen.measure_syndrome(err, decoders[0])
            shapes['synd_raw'] = list(synd.shape)

            if rounds > 1:
                synd = synd.unsqueeze(1).expand(B, rounds, *synd.shape[1:]).clone()
                shapes['synd_with_rounds'] = list(synd.shape)

            synd = voter.apply(synd, number_channel, rounds=rounds,
                               vote_stage=vote_stage, current_stage='syndrome')
            shapes['synd_after_vote'] = list(synd.shape)

            io_dict = {'synd': synd, 'llr0': llr, 'H_matrix': H_matrix}

            for decoder_idx in range(len(decoders)):
                io_dict = decoders[decoder_idx](io_dict)
                decoder_stage = f'decoder_{decoder_idx}'
                vote_kwargs = dict(rounds=rounds, vote_stage=vote_stage, current_stage=decoder_stage)
                io_dict['e_v'] = voter.apply(io_dict['e_v'], number_channel=1, **vote_kwargs)
                io_dict['synd'] = voter.apply(io_dict['synd'], number_channel, **vote_kwargs)
                for key in ('llr', 'converge', 'iter'):
                    if key in io_dict and io_dict[key].ndim > 1:
                        io_dict[key] = voter.select_round(io_dict[key], **vote_kwargs)
                shapes[f'e_v_after_decoder_{decoder_idx}'] = list(io_dict['e_v'].shape)

            check = logical_chk.check(io_dict['e_v'], err, l_matrix)
            shapes['check'] = list(check.shape)
            break

        return shapes, N, M

    def test_2ch_d1(self):
        s, N, M = self._run_2ch_pipeline(rounds=1)
        assert s['err'] == [BATCH_SIZE, 2, N]
        assert s['synd_raw'] == [BATCH_SIZE, 2, M]
        assert s['synd_after_vote'] == [BATCH_SIZE, 2, M]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, 2, N]
        assert s['check'] == [BATCH_SIZE, 2]

    def test_2ch_d3_vote_syndrome(self):
        s, N, M = self._run_2ch_pipeline(rounds=3, vote_stage='syndrome')
        assert s['synd_with_rounds'] == [BATCH_SIZE, 3, 2, M]
        assert s['synd_after_vote'] == [BATCH_SIZE, 2, M]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, 2, N]
        assert s['check'] == [BATCH_SIZE, 2]

    def test_2ch_d3_vote_decoder_0(self):
        s, N, M = self._run_2ch_pipeline(rounds=3, vote_stage='decoder_0')
        assert s['synd_with_rounds'] == [BATCH_SIZE, 3, 2, M]
        assert s['synd_after_vote'] == [BATCH_SIZE, 3, 2, M]
        assert s['e_v_after_decoder_0'] == [BATCH_SIZE, 2, N]
        assert s['check'] == [BATCH_SIZE, 2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
