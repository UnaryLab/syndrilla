import torch, math, yaml
import sys, os, time

import numpy as np

from loguru import logger
from ldpc import BpOsdDecoder

sys.path.append(os.getcwd())

from src.decoder import create_decoder
from src.error_model import create_error_model
from src.syndrome import create_syndrome
from src.metric import report_metric, compute_avg_metrics
from src.logical_check import create_check
from src.utils import dataset


def test_batch_alist_hx(batch_size=1000, target_error=1000, save_error_llr=False):
    decoders = create_decoder(yaml_path='examples/alist/bposd_hx.decoder.yaml')
    
    num_decoders = len(decoders)
    for decoder in decoders:
        decoder.eval()

    shape = decoders[0].H_shape
    dtype = decoders[0].dtype
    decoder_device = decoders[0].device
    H_matrix = decoders[0].H_matrix
    l_matrix = decoders[0].lx_matrix

    # create bposd decoder by bposd repo
    bp_osd = BpOsdDecoder(
        H_matrix.cpu().numpy(),
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
    
    converge_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
    iter_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
    time_iter_all = [[] for _ in range(num_decoders)]

    check = [[]for _ in range(num_decoders)]
    
    if not save_error_llr:
        total_time_all = [0.0 for _ in range(num_decoders)]
        average_time_sample_all = [0.0 for _ in range(num_decoders)]
        average_iter_all = [0.0 for _ in range(num_decoders)]
        average_time_sample_iter_all = [0.0 for _ in range(num_decoders)]
        data_qubit_acc_all = [0.0 for _ in range(num_decoders)]
        correction_acc_all = [0.0 for _ in range(num_decoders)]
        logical_error_rate_all = [0.0 for _ in range(num_decoders)]
        invoke_rate_all = [0.0 for _ in range(num_decoders)]

    error_model = create_error_model(yaml_path='examples/alist/bsc.error.yaml')

    # create syndrome
    syndrome_generator = create_syndrome(yaml_path='examples/alist/perfect.syndrome.yaml')
    
    logical_check = create_check(yaml_path = './examples/alist/lx.check.yaml')
        
    while num_err < target_error:
        # create error
        if not save_error_llr: 
            e_v_all = [torch.empty((0, shape[1]), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
            e_all = torch.empty((0, shape[1]), dtype=dtype, device=decoder_device)
            
            converge_all = [torch.empty((0), dtype=dtype, device=decoder_device) for _ in range(num_decoders)]
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
                converge_all[decoder_idx] = torch.cat((converge_all[decoder_idx], torch.zeros_like(io_dict['converge'])), dim=0)
                if decoder_idx + 1 < num_decoders:
                    converge_all[decoder_idx+1] = torch.cat((converge_all[decoder_idx+1], io_dict['converge']), dim=0)
                decoder_idx += 1

            while decoder_idx < num_decoders:
                # second decoder
                start_time = time.time()
                io_dict = decoders[1](io_dict)

                time_iter_all[decoder_idx].append(time.time() - start_time)
                e_v_all[decoder_idx] = torch.cat((e_v_all[decoder_idx], io_dict['e_v']), dim=0)
                iter_all[decoder_idx] = torch.cat((iter_all[decoder_idx], io_dict['iter']))
                if decoder_idx + 1 < num_decoders:
                    converge_all[decoder_idx+1] = torch.cat((converge_all[decoder_idx+1], io_dict['converge']), dim=0)
                decoder_idx += 1       

                # comparing result with bposd repo
                cbposd = []
                for i in range(batch_size):
                    decoding = bp_osd.decode(io_dict['synd'][i].cpu().numpy())
                    are_equal = np.array_equal(decoding, e_v_all[1][i].cpu().numpy()) 
                    cbposd.append(are_equal)
                logger.info(f'comparing to bposd the e_v result is not the same for {sum(not val for val in cbposd)} in {batch_size} number of test cases.')

            if not save_error_llr: 
                check[0] = logical_check.check(e_v_all[0], e_all, l_matrix)
                for i in range(1, num_decoders):
                    check[i] = logical_check.check_osd(e_v_all[i], e_all, l_matrix, converge_all[i])
                num_err += int(torch.sum(check[num_decoders-1]))

                # report metric
                for i in range(num_decoders):
                    batch_total_time, batch_average_time_sample, batch_average_iter, batch_average_time_sample_iter, batch_data_qubit_acc, batch_correction_acc, batch_logical_error_rate, batch_invoke_rate = report_metric(e_all, e_v_all[i], iter_all[i], time_iter_all[i], check[i], converge_all[i], i)
                    total_time_all[i] += batch_total_time
                    average_time_sample_all[i] += batch_average_time_sample
                    average_iter_all[i] += batch_average_iter
                    average_time_sample_iter_all[i] += batch_average_time_sample_iter
                    data_qubit_acc_all[i] += batch_data_qubit_acc
                    correction_acc_all[i] += batch_correction_acc
                    logical_error_rate_all[i] += batch_logical_error_rate
                    invoke_rate_all[i] += batch_invoke_rate
            else:
                if num_decoders > 1:
                    logical_check_result = logical_check.check_osd(e_v_all[num_decoders-1], e_all, l_matrix, converge_all[num_decoders-1])
                    num_err = int(torch.sum(logical_check_result))
                else:
                    logical_check_result = logical_check.check(e_v_all[0], e_all, l_matrix)
                    num_err = int(torch.sum(logical_check_result))

    all_metrics = []
    if save_error_llr:
        check[0] = logical_check.check(e_v_all[0], e_all, l_matrix)
        for i in range(1, num_decoders):
            check[i] = logical_check.check_osd(e_v_all[i], e_all, l_matrix, converge_all[i])
        
        for i in range(num_decoders):
            total_time, average_time_sample, average_iter, average_time_sample_iter, data_qubit_acc, correction_acc, logical_error_rate, invoke_rate = report_metric(e_all, e_v_all[i], iter_all[i], time_iter_all[i], check[i], converge_all[i], i)
            metrics_dict = {
                'total_time': total_time,
                'average_time_sample': average_time_sample,
                'average_iter': average_iter,
                'average_time_sample_iter': average_time_sample_iter,
                'data_qubit_acc': data_qubit_acc,
                'correction_acc': correction_acc,
                'logical_error_rate': logical_error_rate,
                'invoke_rate': invoke_rate
            }
            all_metrics.append(metrics_dict)
    else:
        for i in range(num_decoders):
            total_time, average_time_sample, average_iter, average_time_sample_iter, data_qubit_acc, \
                correction_acc, logical_error_rate, invoke_rate = compute_avg_metrics(target_error, i, num_batches, total_time_all,
                                                                                average_time_sample_all,
                                                                                average_iter_all,
                                                                                average_time_sample_iter_all,
                                                                                data_qubit_acc_all,
                                                                                correction_acc_all,
                                                                                logical_error_rate_all,
                                                                                invoke_rate_all)
            metrics_dict = {
                'total_time': total_time,
                'average_time_sample': average_time_sample,
                'average_iter': average_iter,
                'average_time_sample_iter': average_time_sample_iter,
                'data_qubit_acc': data_qubit_acc,
                'correction_acc': correction_acc,
                'logical_error_rate': logical_error_rate,
                'invoke_rate': invoke_rate
            }
            all_metrics.append(metrics_dict)


if __name__ == '__main__':
    target_error = 1000
    batch_size = 10000
    save_error_llr = False
    test_batch_alist_hx(batch_size, target_error, save_error_llr)

