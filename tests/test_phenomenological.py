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
        'max_iter': 10,
        'dtype': 'float64',
        'device': {'device_type': 'cpu', 'device_idx': 0},
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


def _sample_error(error_model, bundle, batch=BATCH):
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    zero = torch.zeros([batch, shape[1]], dtype=torch.float64)
    _, dl = error_model.inject_error(zero, batch)
    for err, _, _ in dl:
        return err


def _true_syndrome(H, err):
    return ((err.long() @ H.T) % 2)


def test_shape_rounds_1():
    decoder, bundle, H, em = _setup()
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': 1, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle)
    out = syn_gen.measure_syndrome(err, decoder)
    assert out.ndim == 2, f'rounds=1 should return 2-D, got {out.ndim}-D shape {list(out.shape)}'
    assert out.shape == (BATCH, H.shape[0]), f'expected [{BATCH}, {H.shape[0]}], got {list(out.shape)}'


def test_shape_rounds_gt1():
    decoder, bundle, H, em = _setup()
    rounds = 3
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle)
    out = syn_gen.measure_syndrome(err, decoder)
    assert out.ndim == 3, f'rounds={rounds} should return 3-D, got {out.ndim}-D'
    assert out.shape == (BATCH, rounds, H.shape[0]), f'expected [{BATCH}, {rounds}, {H.shape[0]}], got {list(out.shape)}'


def test_zero_noise_matches_h_at_e():
    decoder, bundle, H, em = _setup()
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': 1, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle)
    out = syn_gen.measure_syndrome(err, decoder).long()
    expected = _true_syndrome(H, err)
    assert torch.equal(out, expected), 'rate=0 syndrome must equal H@e (mod 2)'


def test_zero_noise_replicates_across_rounds():
    decoder, bundle, _, em = _setup()
    rounds = 4
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.0,
    })
    err = _sample_error(em, bundle)
    out = syn_gen.measure_syndrome(err, decoder).long()
    base = out[:, 0, :]
    for d in range(1, rounds):
        assert torch.equal(out[:, d, :], base), f'with rate=0 every round must match round 0; round {d} differs'


def test_noise_makes_rounds_differ():
    decoder, bundle, _, em = _setup()
    rounds = 3
    torch.manual_seed(0)
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': rounds, 'measurement_error_rate': 0.5,
    })
    err = _sample_error(em, bundle)
    out = syn_gen.measure_syndrome(err, decoder).long()
    any_diff = False
    for d in range(1, rounds):
        if not torch.equal(out[:, d, :], out[:, 0, :]):
            any_diff = True
            break
    assert any_diff, 'with rate=0.5 at least one round must differ from round 0'


def test_syndrome_actual_is_noiseless():
    decoder, bundle, H, em = _setup()
    syn_gen = create_syndrome(cfg={
        'measure': 'phenomenological', 'rounds': 5, 'measurement_error_rate': 0.5,
    })
    err = _sample_error(em, bundle)
    _ = syn_gen.measure_syndrome(err, decoder)
    expected = _true_syndrome(H, err)
    assert torch.equal(syn_gen.syndrome_actual.long(), expected), \
        'syndrome_actual should hold the noiseless H@e regardless of measurement_error_rate'


if __name__ == '__main__':
    test_shape_rounds_1()
    test_shape_rounds_gt1()
    test_zero_noise_matches_h_at_e()
    test_zero_noise_replicates_across_rounds()
    test_noise_makes_rounds_differ()
    test_syndrome_actual_is_noiseless()
    print('all phenomenological tests passed')
