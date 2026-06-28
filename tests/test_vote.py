import sys
import os
import torch

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype
from syndrilla.vote import create_vote

SURFACE_DIR = 'examples/alist/surface'
DISTANCE = 9


def make_decoder_cfg(distance):
    return {
        'algorithm': ['bp_lottery', 'osd_0'],
        'check_type': 'hx',
        'max_iter': 100,
        'dtype': 'float64',
        'device': {'device_type': 'cpu', 'device_idx': 0},
    }


def make_matrix_cfg(distance):
    return {
        'parity_matrix_hx': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_hx.alist'},
        'parity_matrix_hz': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_hz.alist'},
        'logical_check_matrix': True,
        'logical_check_lx': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_lx.alist'},
        'logical_check_lz': {'file_type': 'alist', 'path': f'{SURFACE_DIR}/surface_{distance}_lz.alist'},
    }


def run_decode(decoders, error_model, syndrome_gen, logical_chk, bundle,
               target_error, batch_size, rounds, vote_stage='syndrome'):
    voter = create_vote(cfg={'method': 'majority_vote'})
    number_channel = error_model.number_channel
    check_type = 'hx'
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    H_matrix = bundle.select(check_type)[3]
    l_matrix = bundle.get_l_matrix(check_type, number_channel)
    dtype = decoders[0].dtype

    # BSC now generates the rounds dimension itself: rounds>1 yields a 3-D
    # [B, rounds, N] (cumulative) error and a matching 3-D llr, which the
    # phenomenological measurer turns into a per-round [B, rounds, M] syndrome
    # for the voter to collapse.
    error_model.rounds = rounds

    total_logical_errors = 0
    total_shots = 0

    while total_logical_errors < target_error:
        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, batch_size)

        for err, llr, _ in error_dataloader:
            synd = syndrome_gen.measure_syndrome(err, decoders[0])

            synd = voter.apply(synd, number_channel, rounds=rounds,
                               vote_stage=vote_stage, current_stage='syndrome')

            # The per-round data prior is identical every round (constant
            # log((1-p)/p)), so collapse the 3-D llr to a single 2-D prior.
            llr0 = llr if llr.ndim == 2 else llr[:, 0]
            io_dict = {'synd': synd, 'llr0': llr0, 'H_matrix': H_matrix}

            for decoder_idx, decoder in enumerate(decoders):
                io_dict = decoder(io_dict)
                decoder_stage = f'decoder_{decoder_idx}'
                io_dict['e_v'] = voter.apply(io_dict['e_v'], number_channel, rounds=rounds, vote_stage=vote_stage, current_stage=decoder_stage)
                io_dict['synd'] = voter.apply(io_dict['synd'], number_channel, rounds=rounds, vote_stage=vote_stage, current_stage=decoder_stage)
                for key in ('llr', 'converge', 'iter'):
                    if key in io_dict and io_dict[key].ndim > 1:
                        io_dict[key] = voter.select_round(io_dict[key], rounds=rounds, vote_stage=vote_stage, current_stage=decoder_stage)

            # voting collapses the rounds dimension to one estimate, so compare
            # against a single 2-D ground truth: the final accumulated error.
            err_check = err if err.ndim == 2 else err[:, -1]
            check = logical_chk.check(io_dict['e_v'].to(dtype), err_check, l_matrix)
            total_logical_errors += int(check.sum())
            total_shots += batch_size

    return total_logical_errors, total_shots


def test_lottery_vote_stages(distance=DISTANCE, rate=0.05, rounds=3,
                             target_error=100, batch_size=1000):
    print(f'\n=== Lottery BP + OSD: vote stage comparison ===')
    print(f'  surface code hx, distance={distance}, rate={rate}')
    print(f'  rounds={rounds}, target_error={target_error}')

    dec_cfg = make_decoder_cfg(distance)
    mat_cfg = make_matrix_cfg(distance)
    bundle = load_matrices(mat_cfg, *parse_device_dtype(dec_cfg))
    decoders = create_decoder(cfg=dec_cfg, bundle=bundle)
    for d in decoders:
        d.eval()

    error_model = create_error_model(cfg={
        'model': 'bsc', 'number_channel': 1,
        'device': {'device_type': 'cpu', 'device_idx': 0}, 'rate': rate,
    })
    syndrome_gen = create_syndrome(cfg={
        'measure': 'phenomenological',
        'rounds': rounds,
        'measurement_error_rate': 0.01,
    })
    logical_chk = create_check(cfg={'check_type': 'lx'})

    results = {}
    for vote_stage in ['syndrome', 'decoder_0', 'decoder_1']:
        label = {
            'syndrome': 'vote syndrome → bp_lottery → osd_0',
            'decoder_0': 'bp_lottery → vote → osd_0',
            'decoder_1': 'bp_lottery → osd_0 → vote',
        }[vote_stage]

        errors, shots = run_decode(decoders, error_model, syndrome_gen, logical_chk, bundle,
                                   target_error, batch_size, rounds, vote_stage=vote_stage)
        ler = errors / shots
        results[vote_stage] = (errors, shots, ler)
        print(f'\n  [{vote_stage}] {label}')
        print(f'    logical errors = {errors} / {shots}  (LER = {ler:.6f})')

    print(f'\n  Summary (distance={distance}, rate={rate}, rounds={rounds}):')
    for stage, (errors, shots, ler) in results.items():
        print(f'    {stage:12s}: {errors} errors / {shots} shots  (LER = {ler:.6f})')


def test_lottery_no_vote(distance=DISTANCE, rate=0.05,
                         target_error=100, batch_size=1000):
    print(f'\n=== Lottery BP + OSD: no vote (baseline) ===')
    print(f'  surface code hx, distance={distance}, rate={rate}')
    print(f'  rounds=1, target_error={target_error}')

    dec_cfg = make_decoder_cfg(distance)
    mat_cfg = make_matrix_cfg(distance)
    bundle = load_matrices(mat_cfg, *parse_device_dtype(dec_cfg))
    decoders = create_decoder(cfg=dec_cfg, bundle=bundle)
    for d in decoders:
        d.eval()

    error_model = create_error_model(cfg={
        'model': 'bsc', 'number_channel': 1,
        'device': {'device_type': 'cpu', 'device_idx': 0}, 'rate': rate,
    })
    syndrome_gen = create_syndrome(cfg={
        'measure': 'phenomenological',
        'rounds': 1,
        'measurement_error_rate': 0.01,
    })
    logical_chk = create_check(cfg={'check_type': 'lx'})

    errors, shots = run_decode(decoders, error_model, syndrome_gen, logical_chk, bundle,
                               target_error, batch_size, rounds=1, vote_stage='syndrome')
    ler = errors / shots
    print(f'  logical errors = {errors} / {shots}  (LER = {ler:.6f})')


if __name__ == '__main__':
    test_lottery_no_vote(distance=9, rate=0.05, target_error=100, batch_size=1000)
    test_lottery_vote_stages(distance=9, rate=0.05, rounds=9,
                             target_error=100, batch_size=1000)
