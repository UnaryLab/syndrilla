import sys
import os
import torch
from loguru import logger

logger.remove()

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype


SURFACE_DIR = 'examples/alist/surface'
DISTANCE = 3
BATCH = 64


def _setup(distance=DISTANCE):
    dec_cfg = {
        'algorithm': ['bp_lottery'],
        'check_type': 'hx',
        'dtype': 'float64',
        'device': {'device_type': 'cpu', 'device_idx': 0},
        'config': [{'max_iter': 10}],
    }
    mat_cfg = {
        'parity_matrix_hx': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_hx.alist'},
        'parity_matrix_hz': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_hz.alist'},
        'logical_check_matrix': True,
        'logical_check_lx': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_lx.alist'},
        'logical_check_lz': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_lz.alist'},
    }
    bundle = load_matrices(mat_cfg, *parse_device_dtype(dec_cfg))
    decoders = create_decoder(cfg=dec_cfg, bundle=bundle)
    H_dense = bundle.Hx_matrix.get_dense()
    H = torch.as_tensor(H_dense, dtype=torch.long)
    error_model = create_error_model(cfg={
        'model': 'bsc', 'number_channel': 1,
        'device': {'device_type': 'cpu', 'device_idx': 0}, 'rate': 0.05,
    })
    return decoders[0], bundle, H, error_model


def _sample_error(error_model, bundle, rounds=1, batch=BATCH):
    """Sample a BSC error. With rounds > 1 the model emits a 3-D [B, rounds, N]
    tensor whose per-round flips are CUMULATIVE (a flipped qubit stays flipped),
    just like main.py sets error_model.rounds from the syndrome generator."""
    error_model.rounds = rounds
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    zero = torch.zeros([batch, shape[1]], dtype=torch.float64)
    _, dl = error_model.inject_error(zero, batch)
    for err, _, _ in dl:
        return err


def _true_syndrome(H, err):
    """Noiseless syndrome H @ e (mod 2). Works for 2-D [B, N] and 3-D
    [B, rounds, N] errors (matmul broadcasts over the rounds axis)."""
    return ((err.long() @ H.T) % 2)


def test_shape_rounds_1():
    decoder, bundle, H, em = _setup()
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': 1, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle, rounds=1)
    assert err.ndim == 2, f'rounds=1 BSC error should be 2-D, got {err.ndim}-D'
    out = syn_gen.measure_syndrome(err, decoder)
    assert out.ndim == 2, f'rounds=1 should return 2-D, got {out.ndim}-D shape {list(out.shape)}'
    assert out.shape == (BATCH, H.shape[0]), f'expected [{BATCH}, {H.shape[0]}], got {list(out.shape)}'


def test_shape_rounds_gt1():
    decoder, bundle, H, em = _setup()
    rounds = 3
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle, rounds=rounds)
    assert err.shape == (BATCH, rounds, H.shape[1]), \
        f'rounds={rounds} BSC error should be [{BATCH}, {rounds}, {H.shape[1]}], got {list(err.shape)}'
    out = syn_gen.measure_syndrome(err, decoder)
    assert out.ndim == 3, f'rounds={rounds} should return 3-D, got {out.ndim}-D'
    assert out.shape == (BATCH, rounds, H.shape[0]), f'expected [{BATCH}, {rounds}, {H.shape[0]}], got {list(out.shape)}'


def test_zero_noise_matches_h_at_e():
    decoder, bundle, H, em = _setup()
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': 1, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle, rounds=1)
    out = syn_gen.measure_syndrome(err, decoder).long()
    expected = _true_syndrome(H, err)
    assert torch.equal(out, expected), 'rate=0 syndrome must equal H@e (mod 2)'


def test_zero_noise_matches_per_round_h_at_e():
    """With measurement_error_rate=0 the multi-round output is the noiseless
    per-round syndrome H @ e_t of the cumulative error (rounds no longer just
    replicate, because the data error itself accumulates across rounds)."""
    decoder, bundle, H, em = _setup()
    rounds = 4
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle, rounds=rounds)
    out = syn_gen.measure_syndrome(err, decoder).long()
    expected = _true_syndrome(H, err)            # [B, rounds, M]
    assert torch.equal(out, expected), 'rate=0 multi-round syndrome must equal per-round H@e_t'


def test_noise_flips_bits_vs_noiseless():
    """measurement_error_rate>0 must perturb the measured syndrome away from the
    noiseless per-round syndrome stored in syndrome_actual."""
    decoder, bundle, _, em = _setup()
    rounds = 3
    torch.manual_seed(0)
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.5,
    })
    err = _sample_error(em, bundle, rounds=rounds)
    out = syn_gen.measure_syndrome(err, decoder).long()
    assert not torch.equal(out, syn_gen.syndrome_actual.long()), \
        'with rate=0.5 the measured syndrome must differ from the noiseless syndrome_actual'


def test_syndrome_actual_is_noiseless():
    decoder, bundle, H, em = _setup()
    rounds = 5
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.5,
    })
    err = _sample_error(em, bundle, rounds=rounds)
    _ = syn_gen.measure_syndrome(err, decoder)
    expected = _true_syndrome(H, err)            # [B, rounds, M]
    assert torch.equal(syn_gen.syndrome_actual.long(), expected), \
        'syndrome_actual should hold the noiseless per-round H@e_t regardless of measurement_error_rate'


if __name__ == '__main__':
    test_shape_rounds_1()
    test_shape_rounds_gt1()
    test_zero_noise_matches_h_at_e()
    test_zero_noise_matches_per_round_h_at_e()
    test_noise_flips_bits_vs_noiseless()
    test_syndrome_actual_is_noiseless()
    print('all phenomenological tests passed')
