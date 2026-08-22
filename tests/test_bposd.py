import torch
import sys, os, time
from loguru import logger

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.metric import report_metric, save_metric, MetricState, BatchTracker
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.utils import read_yaml, get_path, parse_device_dtype


def test_batch_alist_hx(batch_size=1000, target_error=1000,
                        run_dir='tests/test_outputs',
                        log_level='INFO'):
    decoder_yaml = 'examples/alist/bposd_hx.decoder.yaml'
    matrix_yaml = 'examples/alist/surface_11.matrix.yaml'
    error_yaml = 'examples/alist/bsc.error.yaml'
    syndrome_yaml = 'examples/alist/perfect.syndrome.yaml'
    logical_yaml = 'examples/alist/lx.check.yaml'

    # set up output log (mirrors main.py:65-67)
    logger.remove()
    os.makedirs(run_dir, exist_ok=True)
    output_log = run_dir + '/main' + '-' + str(time.time()) + '.log'
    logger.add(output_log, level=log_level)

    logger.success(f'\n----------------------------------------------\nStep 1: Create decoder\n----------------------------------------------')
    decoder_cfg = read_yaml(get_path(decoder_yaml))['decoder']
    matrix_cfg = read_yaml(get_path(matrix_yaml))['matrix']
    if not torch.cuda.is_available():
        decoder_cfg['device'] = {'device_type': 'cpu', 'device_idx': 0}
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
    decoders = create_decoder(cfg=decoder_cfg, bundle=bundle)

    logger.success(f'\n----------------------------------------------\nStep 2: Create error model\n----------------------------------------------')
    error_model = create_error_model(error_yaml)

    logger.success(f'\n----------------------------------------------\nStep 3: Create syndrome measurer\n----------------------------------------------')
    syndrome_generator = create_syndrome(syndrome_yaml)

    logger.success(f'\n----------------------------------------------\nStep 4: Create logical error checker\n----------------------------------------------')
    logical_check = create_check(logical_yaml)

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

    num_err = 0
    num_batches = 0

    metrics = MetricState(num_decoders, number_channel, decoder_device)

    logger.success(f'\n----------------------------------------------\nStep 5: Check checkpoint file\n----------------------------------------------')
    logger.info(f'No input Checkpoint file.')

    while num_err <= target_error:
        logger.success(f'\n----------------------------------------------\nStep 6: Generate error\n----------------------------------------------')
        bt = BatchTracker(num_decoders, number_channel, shape, dtype, decoder_device)

        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, batch_size)
        num_batches += 1

        avg_error_rate = torch.mean(torch.sum(error_vector, 1) / shape[1])
        logger.info(f'Specified error rate <{error_model.rate}>.')
        logger.info(f'Generated error rate <{avg_error_rate}>.')

        for err, llr, _ in error_dataloader:
            bt.record_error(err)


            logger.success(f'\n----------------------------------------------\nStep 7: Measure syndrome\n----------------------------------------------')
            synd = syndrome_generator.measure_syndrome(err, decoders[0])

            io_dict = {'synd': synd, 'llr0': llr, 'H_matrix': H_matrix}

            logger.success(f'\n----------------------------------------------\nStep 8: Decode\n----------------------------------------------')
            for decoder_idx in range(num_decoders):
                start_time = time.time()
                io_dict = decoders[decoder_idx](io_dict)
                elapsed = time.time() - start_time

                bt.record_decoder(decoder_idx, io_dict, elapsed)

            logger.success(f'\n----------------------------------------------\nStep 9: Check logical error rate\n----------------------------------------------')

            has_obs_flips = hasattr(syndrome_generator, 'observable_flips') and syndrome_generator.observable_flips is not None
            check_error = syndrome_generator.observable_flips.to(dtype) if has_obs_flips else bt.e_all

            check = [[] for _ in range(num_decoders)]
            for i in range(num_decoders):
                check[i] = logical_check.check(bt.e_v_all[i], check_error, l_matrix, bt.converge_all[i + 1])
            num_err += int(torch.sum(check[num_decoders - 1]))
            logger.info(f'number of errors at the current batch {num_err}/{target_error}')

            logger.success(f'\n----------------------------------------------\nStep 10: Aggregate metrics\n----------------------------------------------')
            if number_channel == 1:
                bt.e_v_all = [t.unsqueeze(1).expand(-1, number_channel, -1) for t in bt.e_v_all]
                check = [t.unsqueeze(1).expand(-1, number_channel) for t in check]
            for i in range(num_decoders):
                batch_result = report_metric(num_max_iter[i], bt.e_all, bt.e_v_all[i], bt.iter_all[i],
                                             bt.time_iter_all[i], check[i], bt.converge_all[i],
                                             bt.converge_all[i + 1], i)
                metrics.accumulate(i, batch_result)

            if num_batches % 100 == 0:
                logger.success(f'\n----------------------------------------------\nStep 11: Save batch log\n----------------------------------------------')
                all_metrics = metrics.get_all_metrics(num_batches, algo_name)
                save_metric(all_metrics, run_dir + '/', batch_size, target_error, str(dtype),
                            error_model.rate, num_batches, num_err, H_file_name, check_num)
                logger.success(f'Saved log to <{output_log}>.')
                logger.success(f'Saved metric results to <{run_dir}>.')

    logger.success(f'\n----------------------------------------------\nStep 12: Save final log\n----------------------------------------------')
    all_metrics = metrics.get_all_metrics(num_batches, algo_name)
    save_metric(all_metrics, run_dir + '/', batch_size, target_error, str(dtype),
                error_model.rate, num_batches, num_err, H_file_name, check_num, 1)
    logger.success(f'Saved log to <{output_log}>.')
    logger.success(f'Saved metric results to <{run_dir}>.')


if __name__ == '__main__':
    test_batch_alist_hx(batch_size=100000, target_error=1000)
