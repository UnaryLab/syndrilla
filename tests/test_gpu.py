import os
import sys
import time

import pytest
import torch

sys.path.append(os.getcwd())

from loguru import logger

pynvml = pytest.importorskip('pynvml')

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.metric import BatchTracker, MetricState
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import get_path, parse_device_dtype, read_yaml


def get_gpu_memory_utilization(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()

    total = mem_info.total / (1024 ** 2)
    used = mem_info.used / (1024 ** 2)
    free = mem_info.free / (1024 ** 2)
    percent_used = (used / total) * 100
    return {'total_MB': total, 'used_MB': used, 'free_MB': free, 'percent_used': percent_used}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_batch_alist_hx(batch_size=1000, target_error=1000,
                        run_dir='tests/test_outputs'):
    decoder_yaml = 'examples/alist/bposd_hx.decoder.yaml'
    matrix_yaml = 'examples/alist/surface_10.matrix.yaml'

    decoder_cfg = read_yaml(get_path(decoder_yaml))['decoder']
    matrix_cfg = read_yaml(get_path(matrix_yaml))['matrix']
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
    decoders = create_decoder(cfg=decoder_cfg, bundle=bundle)

    error_model = create_error_model(yaml_path='examples/alist/bsc.error.yaml')
    syndrome_generator = create_syndrome(yaml_path='examples/alist/perfect.syndrome.yaml')
    logical_check = create_check(yaml_path='examples/alist/lx.check.yaml')
    num_decoders = len(decoders)
    dtype = decoders[0].dtype
    decoder_device = decoders[0].device
    number_channel = error_model.number_channel
    check_type = decoder_cfg.get('check_type', 'hx')
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    H_matrix = bundle.select(check_type)[3]

    algo_name = []
    num_max_iter = []
    for decoder in decoders:
        decoder.eval()
        algo_name.append(decoder.algo)
        num_max_iter.append(getattr(decoder, 'num_max_iter', 0))

    H_file_name = bundle.get_H_file_name(check_type, number_channel)
    l_matrix = bundle.get_l_matrix(check_type, number_channel)
    check_num = bundle.get_check_num(check_type, number_channel)

    metrics = MetricState(num_decoders, number_channel, decoder_device)

    num_err = 0
    num_batches = 0
    while num_err <= target_error:
        bt = BatchTracker(num_decoders, number_channel, shape, dtype, decoder_device)

        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        _, error_dataloader = error_model.inject_error(zero_qubits, batch_size)
        num_batches += 1

        for err, llr, _ in error_dataloader:
            bt.record_error(err)

            synd = syndrome_generator.measure_syndrome(err, decoders[0])

            io_dict = {'synd': synd, 'llr0': llr, 'H_matrix': H_matrix}

            for decoder_idx in range(num_decoders):
                start_time = time.time()
                io_dict = decoders[decoder_idx](io_dict)
                elapsed = time.time() - start_time

                gpu_stats = get_gpu_memory_utilization()
                for k, v in gpu_stats.items():
                    logger.info(f'GPU decoder_{decoder_idx} {k}: {v}')

                bt.record_metric(decoder_idx, io_dict, elapsed)

            has_obs_flips = hasattr(syndrome_generator, 'observable_flips') and syndrome_generator.observable_flips is not None
            check_error = syndrome_generator.observable_flips.to(dtype) if has_obs_flips else bt.e_all

            check = [[] for _ in range(num_decoders)]
            for i in range(num_decoders):
                check[i] = logical_check.check(bt.e_v_all[i], check_error, l_matrix, bt.converge_all[i + 1])
            num_err += int(torch.sum(check[num_decoders - 1]))

            if number_channel == 1:
                bt.e_v_all = [t.unsqueeze(1).expand(-1, number_channel, -1) for t in bt.e_v_all]
                check = [t.unsqueeze(1).expand(-1, number_channel) for t in check]
            for i in range(num_decoders):
                batch_result = metrics.report_metric(num_max_iter[i], bt.e_all, bt.e_v_all[i], bt.iter_all[i],
                                                     bt.time_iter_all[i], check[i], bt.converge_all[i],
                                                     bt.converge_all[i + 1], i)
                metrics.update_metric(i, batch_result)

            if num_batches % 100 == 0:
                all_metrics = metrics.get_all_metrics(num_batches, algo_name)
                metrics.save_metric(all_metrics, run_dir + '/', batch_size, target_error, str(dtype),
                                    error_model.rate, num_batches, num_err, H_file_name, check_num)

    all_metrics = metrics.get_all_metrics(num_batches, algo_name)
    metrics.save_metric(all_metrics, run_dir + '/', batch_size, target_error, str(dtype),
                        error_model.rate, num_batches, num_err, H_file_name, check_num, 1)


if __name__ == '__main__':
    test_batch_alist_hx(batch_size=100000, target_error=1000)
