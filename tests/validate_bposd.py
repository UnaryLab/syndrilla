import os
import sys
import time

import numpy as np
import torch
from ldpc import BpOsdDecoder
from loguru import logger

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import get_path, parse_device_dtype, read_yaml


def test_batch_alist_hx(batch_size=1000, target_error=1000, max_batches=10):
    decoding_cfg = read_yaml(get_path('examples/alist/bposd_hx.decoding.yaml'))['decoding']
    matrix_cfg = read_yaml(get_path('examples/alist/surface_11.matrix.yaml'))['matrix']
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoding_cfg))
    decoders = create_decoder(cfg=decoding_cfg, bundle=bundle)

    num_decoders = len(decoders)
    for decoder in decoders:
        decoder.eval()

    shape = decoders[0].H_shape
    dtype = decoders[0].dtype
    decoder_device = decoders[0].device
    H_matrix = bundle.select(decoding_cfg['check_type'])[3]
    l_matrix = bundle.get_l_matrix(decoding_cfg['check_type'], 1)

    # create bposd decoder by bposd repo
    bp_osd = BpOsdDecoder(
        H_matrix.to_dense().cpu().numpy().astype(np.uint8),
        error_rate = 0.01,
        bp_method = 'product_sum',
        max_iter = 131,
        schedule = 'serial',
        osd_method = 'osd_0'
    )

    num_err = 0
    num_batches = 0

    e_v_all = [torch.empty((0, shape[1]), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
    e_all = torch.empty((0, shape[1]), dtype=dtype, device=decoder_device)

    converge_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders+1)]
    iter_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
    time_iter_all = [[] for _ in range(num_decoders)]

    error_model = create_error_model(yaml_path='examples/alist/bsc.error.yaml')

    # create syndrome
    syndrome_generator = create_syndrome(yaml_path='examples/alist/perfect.syndrome.yaml')

    logical_check = create_check(yaml_path='examples/alist/lx.check.yaml')

    while num_err <= target_error and num_batches < max_batches:
        # create error
        e_v_all = [torch.empty((0, shape[1]), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
        e_all = torch.empty((0, shape[1]), dtype=dtype, device=decoder_device)

        converge_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders+1)]
        iter_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
        time_iter_all = [[] for _ in range(num_decoders)]

        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, batch_size)

        num_batches += 1

        avg_error_rate = torch.mean(torch.sum(error_vector, 1) / shape[1])
        logger.info(f'Specified error rate <{error_model.rate}>.')
        logger.info(f'Generated error rate <{avg_error_rate}>.')

        for err, llr, _ in error_dataloader:
            err = err.to(e_all.device)
            e_all = torch.cat((e_all, err))
            synd = syndrome_generator.measure_syndrome(err, decoders[0])

            io_dict = {
                'synd': synd,
                'llr0': llr,
                'H_matrix': H_matrix
            }
            decoder_idx = 0

            if decoder_idx == 0:
            # first decoder
                start_time = time.time()
                io_dict = decoders[decoder_idx](io_dict)

                time_iter_all[decoder_idx].append(time.time() - start_time)

                e_v_all[decoder_idx] = torch.cat((e_v_all[decoder_idx], io_dict['e_v']), dim=0)
                iter_all[decoder_idx] = torch.cat((iter_all[decoder_idx], io_dict['iter']))
                converge_all[decoder_idx] = torch.cat((converge_all[decoder_idx], torch.ones_like(io_dict['converge'])), dim=0)

                converge_all[decoder_idx+1] = torch.cat((converge_all[decoder_idx+1], io_dict['converge']), dim=0)
                decoder_idx += 1

            while decoder_idx < num_decoders:
                # second decoder
                start_time = time.time()
                io_dict = decoders[1](io_dict)

                time_iter_all[decoder_idx].append(time.time() - start_time)
                e_v_all[decoder_idx] = torch.cat((e_v_all[decoder_idx], io_dict['e_v']), dim=0)
                iter_all[decoder_idx] = torch.cat((iter_all[decoder_idx], io_dict['iter']))
                converge_all[decoder_idx+1] = torch.cat((converge_all[decoder_idx+1], io_dict['converge']), dim=0)
                decoder_idx += 1

                # comparing result with bposd repo
                cbposd = []
                for i in range(batch_size):
                    decoding = bp_osd.decode(io_dict['synd'][i].cpu().numpy())
                    are_equal = np.array_equal(decoding, e_v_all[1][i].cpu().numpy())
                    cbposd.append(are_equal)
                logger.info(f'comparing to bposd the e_v result is not the same for {sum(not val for val in cbposd)} in {batch_size} number of test cases.')

        check = logical_check.check(e_v_all[num_decoders-1], e_all, l_matrix, converge_all[num_decoders])
        num_err += int(torch.sum(check))
        logger.info(f'batch {num_batches}/{max_batches} (cap), logical errors {num_err}/{target_error}')


if __name__ == '__main__':
    target_error = 1000
    batch_size = 10000
    max_batches = 10
    test_batch_alist_hx(batch_size, target_error, max_batches)

