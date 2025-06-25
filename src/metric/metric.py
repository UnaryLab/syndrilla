import torch, time, os
import yaml
import numpy as np

from loguru import logger


def report_metric(e_estimated, e_actual, iteration, time_iteration, check, converge, decode_idx):
    """
    This function reports the decoding iteration and accuracy.
    """
    
    logger.info(f'Reporting decoding metric for decoder {decode_idx}.')
    
    sample_size = e_estimated.shape[0]

    total_time = np.sum(time_iteration)
    logger.info(f'Total time for <{sample_size}> samples: {total_time} seconds.')

    average_iter = torch.mean(iteration).item()
    logger.info(f'Average iterations per sample: {average_iter}')

    if total_time == 0:
        average_time_sample = 0
        logger.info(f'Average time per sample: {average_time_sample} seconds.')
        
        average_time_sample_iter = 0
        logger.info(f'Average time per iteration per sample: {average_time_sample_iter}')
    else:
        average_time_sample = total_time/sample_size
        logger.info(f'Average time per sample: {average_time_sample} seconds.')
        
        average_time_sample_iter = (average_time_sample/average_iter).item()
        logger.info(f'Average time per iteration per sample: {average_time_sample_iter}')

    e_estimated = e_estimated.to(e_estimated.device)
    comparison = torch.unique(e_estimated == e_actual, return_counts=True)[1]
    if int(comparison.shape[0]) == 1:
        data_qubit_acc = 1
        logger.info(f'Data qubit accuracy: {1}')
    else:
        data_qubit_acc = float(comparison[1]) / (float(comparison[1]) + float(comparison[0]))
        logger.info(f'Data qubit accuracy: {data_qubit_acc}')

    num_error = torch.sum(e_estimated != e_actual)
   
    total_ones = torch.sum((e_estimated == 1) | (e_actual == 1))
    if float(num_error) == 0:
        correction_acc = 1
        logger.info(f'Correction accuracy: {correction_acc}')
    else:
        correction_acc = 1 - float(num_error)/float(total_ones)
        logger.info(f'Correction accuracy: {correction_acc}')
    
    if int(torch.sum(check)) == 0:
        logical_error_rate = 0
        logger.info(f'Output logical check error rate is {logical_error_rate}')
    else:
        logical_error_rate = (int(torch.sum(check))/(check.size()[0]))
        logger.info(f'Output logical check error rate is {logical_error_rate}')

    if torch.isinf(torch.sum(converge)) or torch.isnan(torch.sum(converge)):
        invoke_rate = 1
        logger.info(f'The frequency of decoder invoke is {invoke_rate}')
    else:
        if int(torch.sum(converge)) == 0:
            invoke_rate = 1
            logger.info(f'The frequency of decoder invoke is {invoke_rate}')
        else:
            invoke_rate = ((check.size()[0] - int(torch.sum(converge)))/(check.size()[0]))
            logger.info(f'The frequency of decoder invoke is {invoke_rate}')

    logger.info(f'Complete.')

    return total_time, average_time_sample, average_iter, average_time_sample_iter, data_qubit_acc, correction_acc, logical_error_rate, invoke_rate


def save_metric(out_dict, curr_dir, batch_size, target_error, physical_error_rate):
    """
    Saves decoding metrics for all decoders into a single YAML file.
    
    Parameters:
        out_dict (list of dicts): each item is a dict with metric keys
        curr_dir (str): directory path to save YAML
        physical_error_rate (float or str): label for file naming
    """

    logger.info('Saving all decoding metrics to a YAML file.')

    def float_representer(dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:float', f"{value:.17e}")

    def format_time(value):
        return f'{float(value):.17e}(s)'

    all_metrics_results = {}

    total_time_sum = 0.0
    last_logical_error_rate = 0.0

    for i, decoder_metrics in enumerate(out_dict):
        decoder_key = f'decoder_{i}'

        total_time_sum += float(decoder_metrics['total_time'])
        last_logical_error_rate = float(decoder_metrics['logical_error_rate'])

        all_metrics_results[decoder_key] = {
            'algorithm': decoder_metrics['algorithm'],
            'correction accuracy': float(decoder_metrics['correction_acc']),
            'qubit accuracy': float(decoder_metrics['data_qubit_acc']),
            'average iteration': float(decoder_metrics['average_iter']),
            'average time spent per sample per iteration': format_time(decoder_metrics['average_time_sample_iter']),
            'average time spent per sample': format_time(decoder_metrics['average_time_sample']),
            'total time spent': format_time(decoder_metrics['total_time']),
            'logical error rate': float(decoder_metrics['logical_error_rate']),
            'decoder invoke rate': float(decoder_metrics['invoke_rate'])
        }

    all_metrics_results['decoder_full'] = {
        'total time (all decoders)': format_time(total_time_sum),
        'final logical error rate': last_logical_error_rate,
        'batch size': batch_size,
        'target error': target_error,
        'physical error rate': physical_error_rate
    }

    os.makedirs(curr_dir, exist_ok=True)  # Ensure directory exists
    output_path = os.path.join(curr_dir, f'result_phy_err_{physical_error_rate}.yaml')

    yaml.add_representer(float, float_representer)
    with open(output_path, 'w') as f:
        yaml.safe_dump(all_metrics_results, f, sort_keys=False)

    logger.info(f'Saved all decoder metrics to: {output_path}')


def compute_avg_metrics(target_error, i, num_batches,
                        total_time_all,
                        average_time_sample_all,
                        average_iter_all,
                        average_time_sample_iter_all,
                        data_qubit_acc_all,
                        correction_acc_all,
                        logical_error_rate_all,
                        invoke_rate_all):
    logger.info(f'Reporting decoding metric for decoder {i}.')
    total_time = total_time_all[i] / num_batches
    average_time_sample = average_time_sample_all[i] / num_batches
    average_iter = average_iter_all[i] / num_batches
    average_time_sample_iter = average_time_sample_iter_all[i] / num_batches
    data_qubit_acc = data_qubit_acc_all[i] / num_batches
    correction_acc = correction_acc_all[i] / num_batches
    logical_error_rate = logical_error_rate_all[i] / num_batches
    invoke_rate = invoke_rate_all[i] / num_batches
    logger.info(f'Total time for having <{target_error}> errors: {total_time} seconds.')
    logger.info(f'Average time per sample: {average_time_sample} seconds.')
    logger.info(f'Average iterations per sample: {average_iter}')
    logger.info(f'Average time per iteration per sample: {average_time_sample_iter}')
    logger.info(f'Data qubit accuracy: {data_qubit_acc}')
    logger.info(f'Correction accuracy: {correction_acc}')
    logger.info(f'Output logical check error rate is {logical_error_rate}')
    logger.info(f'The frequency of decoder invoke is {invoke_rate}')

    logger.info(f'Complete.')

    return total_time, average_time_sample, average_iter, average_time_sample_iter, data_qubit_acc, correction_acc, logical_error_rate, invoke_rate