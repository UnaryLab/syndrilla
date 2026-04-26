import torch, os
import yaml
import numpy as np

from loguru import logger


class MetricResult(dict):
    """Dict that also supports tuple unpacking for backward compatibility."""
    _order = [
        'total_time', 'average_time_sample', 'average_iter', 'distribution',
        'average_time_sample_iter', 'data_qubit_acc', 'data_frame_error_rate',
        'synd_frame_error_rate', 'correction_acc', 'logical_error_rate',
        'invoke_rate', 'converge_fail', 'converge_succ',
    ]

    def __iter__(self):
        return (self[k] for k in self._order)


class BatchTracker:
    """
    Per-batch decoder output buffers (e_v, iter, converge, timing).

    Replaces the e_v_all / iter_all / converge_all / time_iter_all lists that
    were previously initialised and managed inline in main.py.
    """

    def __init__(self, num_decoders, number_channel, shape, dtype, device):
        self.num_decoders = num_decoders
        self.number_channel = number_channel
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.reset()

    def reset(self):
        """Clear all buffers for a new batch."""
        nc, s, dt, dev = self.number_channel, self.shape, self.dtype, self.device
        nd = self.num_decoders
        empty_shape = (0, nc, s[1]) if nc > 1 else (0, s[1])
        self.e_v_all = [torch.empty(empty_shape, dtype=dt, device=dev) for _ in range(nd)]
        self.e_all = torch.empty(empty_shape, dtype=dt, device=dev)
        self.converge_all = [torch.empty((0), dtype=dt, device=dev) for _ in range(nd + 1)]
        self.iter_all = [torch.empty((0), dtype=dt, device=dev) for _ in range(nd)]
        self.time_iter_all = [[] for _ in range(nd)]

    def record_error(self, err):
        """Append the ground-truth error batch."""
        err = err.to(self.e_all.device)
        self.e_all = torch.cat((self.e_all, err))

    def record_decoder(self, decoder_idx, io_dict, elapsed):
        """Record one decoder's output for the current batch."""
        i = decoder_idx
        self.time_iter_all[i].append(elapsed)

        e_v = io_dict['e_v']
        converge = io_dict['converge']
        iter_val = io_dict['iter']

        if e_v.ndim > 2:
            e_v = e_v[:, 0]
        if converge.ndim > 1:
            converge = converge[:, 0]
        if iter_val.ndim > 1:
            iter_val = iter_val[:, 0]

        self.e_v_all[i] = torch.cat((self.e_v_all[i], e_v), dim=0)
        self.iter_all[i] = torch.cat((self.iter_all[i], iter_val))
        if i == 0:
            self.converge_all[i] = torch.cat(
                (self.converge_all[i], torch.zeros_like(converge)), dim=0)
        self.converge_all[i + 1] = torch.cat(
            (self.converge_all[i + 1], converge), dim=0)


class MetricState:
    """Holds all per-decoder metric accumulators."""

    # Names of scalar (per-decoder) fields
    _scalar_fields = [
        'total_time', 'average_time_sample', 'average_iter',
        'average_time_sample_iter', 'invoke_rate',
    ]
    # Names of per-channel fields
    _channel_fields = [
        'data_qubit_acc', 'data_frame_error_rate', 'synd_frame_error_rate',
        'correction_acc', 'logical_error_rate', 'converge_fail', 'converge_succ',
    ]

    def __init__(self, num_decoders, number_channel, device):
        self.num_decoders = num_decoders
        self.number_channel = number_channel
        self.device = device

        # scalar accumulators: list[float] indexed by decoder
        for field in self._scalar_fields:
            setattr(self, field, [0.0] * num_decoders)

        # distribution is special (tensor per decoder)
        self.distribution = [0.0] * num_decoders

        # per-channel accumulators: list[list[float]] indexed by [decoder][channel]
        for field in self._channel_fields:
            setattr(self, field, [[0.0] * number_channel for _ in range(num_decoders)])

        # vote tracking: per-round full-match counts accumulated across batches
        self.vote_active = False
        self.vote_stage = None
        self.vote_d_rounds = 0
        self.vote_match_counts = None
        self.vote_sample_count = 0

    def accumulate_vote(self, match_counts, sample_count, stage, d_rounds):
        """Add per-round full-match counts from one voting application."""
        if match_counts is None or sample_count <= 0 or d_rounds <= 0:
            return
        if not self.vote_active:
            self.vote_active = True
            self.vote_stage = stage
            self.vote_d_rounds = int(d_rounds)
            self.vote_match_counts = [0] * self.vote_d_rounds
        counts = list(match_counts)
        for k in range(min(self.vote_d_rounds, len(counts))):
            self.vote_match_counts[k] += int(counts[k])
        self.vote_sample_count += int(sample_count)

    def get_vote_info(self):
        """Return aggregated vote stats, or None if no voting was applied."""
        if not self.vote_active or self.vote_sample_count == 0:
            return None
        rates = [c / self.vote_sample_count for c in self.vote_match_counts]
        return {
            'active': True,
            'stage': self.vote_stage,
            'd_rounds': self.vote_d_rounds,
            'sample_count': self.vote_sample_count,
            'match_counts': list(self.vote_match_counts),
            'per_round_match_rate': rates,
        }

    def accumulate(self, decoder_idx, batch_metrics):
        """Add one batch's report_metric result for a decoder."""
        i = decoder_idx
        for field in self._scalar_fields:
            getattr(self, field)[i] += batch_metrics[field]

        self.distribution[i] += batch_metrics['distribution']

        for field in self._channel_fields:
            acc = getattr(self, field)[i]
            batch_val = batch_metrics[field]
            for ch in range(self.number_channel):
                acc[ch] += batch_val[ch]

    def compute_avg(self, decoder_idx, num_batches):
        """Compute averaged metrics for one decoder. Returns a dict."""
        i = decoder_idx

        logger.info(f'Reporting decoding metric for decoder {i}.')

        total_time = self.total_time[i]
        average_time_batch = total_time / num_batches
        average_time_sample = self.average_time_sample[i] / num_batches
        average_iter = self.average_iter[i] / num_batches
        distribution = self.distribution[i]
        average_time_sample_iter = self.average_time_sample_iter[i] / num_batches
        invoke_rate = self.invoke_rate[i] / num_batches

        logger.info(f'Decoder invoke rate: {invoke_rate}')
        logger.info(f'Total time: {total_time} seconds.')
        logger.info(f'Total number of batches: {num_batches}.')
        logger.info(f'Average time per batch: {average_time_batch} seconds.')
        logger.info(f'Average time per sample: {average_time_sample} seconds.')
        logger.info(f'Average iterations per sample: {average_iter}')
        logger.info(f'Iteration distribution: {distribution}')
        logger.info(f'Average time per iteration: {average_time_sample_iter}')

        result = {
            'total_time': total_time,
            'average_time_sample': average_time_sample,
            'average_iter': average_iter,
            'distribution': distribution,
            'average_time_sample_iter': average_time_sample_iter,
            'invoke_rate': invoke_rate,
        }

        for field in self._channel_fields:
            avg = [x / num_batches for x in getattr(self, field)[i]]
            result[field] = avg

        for ch in range(self.number_channel):
            logger.info(f'Channel idx: {ch}')
            logger.info(f'Data qubit accuracy: {result["data_qubit_acc"][ch]}')
            logger.info(f'Data qubit correction accuracy: {result["correction_acc"][ch]}')
            logger.info(f'Data frame error rate: {result["data_frame_error_rate"][ch]}')
            logger.info(f'Syndrome frame error rate: {result["synd_frame_error_rate"][ch]}')
            logger.info(f'Output logical error rate: {result["logical_error_rate"][ch]}')
            logger.info(f'Converge failure rate: {result["converge_fail"][ch]}')
            logger.info(f'Converge success rate: {result["converge_succ"][ch]}')

        logger.info(f'Complete.')
        return result

    def get_all_metrics(self, num_batches, algo_names):
        """Compute averaged metrics for all decoders. Returns list of dicts for save_metric."""
        all_metrics = []
        for i in range(self.num_decoders):
            avg = self.compute_avg(i, num_batches)
            avg['algorithm'] = algo_names[i]
            # remap keys to match save_metric expectations
            avg['converge_fail_rate'] = avg.pop('converge_fail')
            avg['converge_succ_rate'] = avg.pop('converge_succ')
            all_metrics.append(avg)
        return all_metrics

    @classmethod
    def from_checkpoint(cls, path, number_channel, device):
        """Load state from a checkpoint YAML. Returns (MetricState, ckpt_meta)."""
        with open(path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")

        full = data['decoder_full']
        ckpt_meta = {
            'H_file_name': full.get('H matrix', 0),
            'batch_size': int(full.get('batch size', 0)),
            'batch_count': int(full.get('batch count', 0)),
            'target_error': int(full.get('target error', 0)),
            'num_err': int(full.get('target error reached', 0)),
            'dtype': full.get('data type', 0),
            'physical_error_rate': float(full.get('physical error rate', 0.0)),
        }

        decoder_keys = sorted(
            [k for k in data if k.startswith('decoder_') and k[8:].isdigit()],
            key=lambda x: int(x.split('_')[1])
        )
        num_decoders = len(decoder_keys)

        state = cls(num_decoders, number_channel, device)

        for idx, key in enumerate(decoder_keys):
            entry = data[key]
            bc = ckpt_meta['batch_count']

            has_hx = 'hx' in entry
            has_hz = 'hz' in entry
            if has_hx and has_hz:
                ch_map = [('hx', 0), ('hz', 1)]
            elif has_hx:
                ch_map = [('hx', 0)]
            elif has_hz:
                ch_map = [('hz', 0)]
            else:
                ch_map = []

            state.total_time[idx] = float(entry['total time (s)']) * bc
            state.average_time_sample[idx] = float(entry['average time per sample (s)']) * bc
            state.average_iter[idx] = float(entry['average iteration']) * bc
            state.distribution[idx] = torch.tensor(entry['iteration distribution'])
            state.average_time_sample_iter[idx] = float(entry['average time per iteration (s)']) * bc
            state.invoke_rate[idx] = float(entry['decoder invoke rate']) * bc

            for ch, ch_idx in ch_map:
                ch_entry = entry[ch]
                state.data_qubit_acc[idx][ch_idx] = float(ch_entry.get('data qubit accuracy', 0.0)) * bc
                state.data_frame_error_rate[idx][ch_idx] = float(ch_entry.get('data frame error rate', 0.0)) * bc
                state.synd_frame_error_rate[idx][ch_idx] = float(ch_entry.get('syndrome frame error rate', 0.0)) * bc
                state.correction_acc[idx][ch_idx] = float(ch_entry.get('data qubit correction accuracy', 0.0)) * bc
                state.logical_error_rate[idx][ch_idx] = float(ch_entry.get('logical error rate', 0.0)) * bc
                state.converge_fail[idx][ch_idx] = float(ch_entry.get('converge failure rate', 0.0)) * bc
                state.converge_succ[idx][ch_idx] = float(ch_entry.get('converge success rate', 0.0)) * bc

        if 'vote' in data:
            v = data['vote']
            state.vote_active = True
            state.vote_stage = v.get('stage')
            state.vote_d_rounds = int(v.get('d_rounds', 0))
            state.vote_sample_count = int(v.get('total samples', 0))
            percentages = v.get('match percentage')
            if percentages is None:
                # legacy keys, kept for older checkpoints
                counts = v.get('match counts')
                if counts is not None:
                    state.vote_match_counts = [int(x) for x in counts]
                else:
                    rates = v.get('per round match rate', [])
                    if isinstance(rates, dict):
                        rates = list(rates.values())
                    state.vote_match_counts = [
                        int(round(float(r) * state.vote_sample_count)) for r in rates
                    ]
            else:
                if isinstance(percentages, dict):
                    percentages = list(percentages.values())
                state.vote_match_counts = [
                    int(round(float(p) * state.vote_sample_count / 100.0)) for p in percentages
                ]
            if state.vote_d_rounds and len(state.vote_match_counts) < state.vote_d_rounds:
                state.vote_match_counts += [0] * (state.vote_d_rounds - len(state.vote_match_counts))

        return state, ckpt_meta

    def validate_checkpoint(self, ckpt_meta, batch_size, target_error, dtype, physical_error_rate, H_file_name):
        """Validate checkpoint metadata matches current run parameters."""
        checks = [
            ('batch_size', ckpt_meta['batch_size'], batch_size),
            ('target_error', ckpt_meta['target_error'], target_error),
            ('dtype', ckpt_meta['dtype'], str(dtype)),
            ('physical_error_rate', float(ckpt_meta['physical_error_rate']), float(physical_error_rate)),
            ('H_file_name', ckpt_meta['H_file_name'], H_file_name),
        ]
        for name, ckpt_val, input_val in checks:
            if ckpt_val != input_val:
                raise FileNotFoundError(
                    f'Checkpoint file not match on {name}: ckpt({ckpt_val}), input({input_val})')

        for i in range(self.num_decoders):
            self.distribution[i] = self.distribution[i].int().to(self.device)


def report_metric(num_max_iter, e_estimated, e_actual, iteration, time_iteration, check, converge, converge_next, decode_idx):
    """
    This function reports the decoding iteration and accuracy.
    Returns a dict of batch metrics.
    """

    logger.info(f'Reporting decoding metric for decoder {decode_idx}.')

    sample_size = e_estimated.shape[0]
    number_channel = e_actual.shape[1]
    dtype = e_estimated.dtype
    total_time = np.sum(time_iteration)
    logger.info(f'Total time for <{sample_size}> samples: {total_time} seconds.')

    if iteration.numel() == 0:
        average_iter = 0.0
        distribution = torch.zeros(num_max_iter+1).to(e_estimated.device)
    else:
        distribution = torch.bincount((iteration - 1).int(), minlength=num_max_iter+1).int().to(e_estimated.device)
        average_iter = torch.mean(iteration).item()

    logger.info(f'Average iterations per sample: {average_iter}')
    logger.info(f'Maximum iterations: {distribution}')

    if total_time == 0:
        average_time_sample = 0
        logger.info(f'Average time per sample: {average_time_sample} seconds.')

        average_time_sample_iter = 0
        logger.info(f'Average time per iteration: {average_time_sample_iter}')
    else:
        average_time_sample = total_time/sample_size
        logger.info(f'Average time per sample: {average_time_sample} seconds.')

        average_time_sample_iter = (average_time_sample/average_iter).item()
        logger.info(f'Average time per iteration: {average_time_sample_iter}')

    if torch.isinf(torch.sum(converge)) or torch.isnan(torch.sum(converge)):
        invoke_rate = 1.0
        logger.info(f'Decoder invoke rate: {invoke_rate}')
    else:
        if int(torch.sum(converge)) == 0:
            invoke_rate = 1.0
            logger.info(f'Decoder invoke rate: {invoke_rate}')
        else:
            invoke_rate = 1.0 - ((int(torch.sum(converge)))/sample_size)
            logger.info(f'Decoder invoke rate: {invoke_rate}')

    if e_actual.size(1) <= 1:
        e_estimated = e_estimated.unsqueeze(1).expand(-1, e_actual.size(1), -1)
    data_qubit_acc = torch.zeros([number_channel], dtype=dtype)
    correction_acc = torch.zeros([number_channel], dtype=dtype)
    data_frame_error_rate = torch.zeros([number_channel], dtype=dtype)
    synd_frame_error_rate = torch.zeros([number_channel], dtype=dtype)
    logical_error_rate = torch.zeros([number_channel], dtype=dtype)
    converge_succ_rate = torch.zeros([number_channel], dtype=dtype)
    converge_fail_rate = torch.zeros([number_channel], dtype=dtype)
    for i in range(e_actual.size(1)):  # iterate over check_type
        check_channel = check[:, i]
        e_actual_channel = e_actual[:, i, :]
        e_estimated_channel = e_estimated[:, i, :]
        eq_i = (e_estimated_channel == e_actual_channel)
        comp_i = torch.unique(eq_i, return_counts=True)[1]
        if int(comp_i.shape[0]) == 1:
            data_qubit_acc[i] = 1.0
        else:
            data_qubit_acc[i] = float(comp_i[1]) / (float(comp_i[1]) + float(comp_i[0]))

        num_error = torch.sum(e_estimated_channel != e_actual_channel)

        total_ones = torch.sum((e_estimated_channel == 1) | (e_actual_channel == 1))
        if float(num_error) == 0:
            correction_acc[i] = 1
        else:
            correction_acc[i] = (1 - float(num_error)/float(total_ones))

        result = (e_estimated_channel == e_actual_channel).all(dim=1).int()
        num_correct = (result == 1).sum()
        num_incorrect = (result == 0).sum()
        if int(result.shape[0]) == 1:
            data_frame_error_rate[i] = 0.0
        else:
            data_frame_error_rate[i] = float(num_incorrect) / (float(num_incorrect) + float(num_correct))

        if torch.isinf(torch.sum(converge_next)) or torch.isnan(torch.sum(converge_next)):
            synd_frame_error_rate[i] = 0
        else:
            if int(torch.sum(converge_next)) == 0:
                synd_frame_error_rate[i] = 0.0
            else:
                synd_frame_error_rate[i] = ((check_channel.size()[0] - int(torch.sum(converge_next)))/(check_channel.size()[0]))

        if int(torch.sum(check_channel)) == 0:
            logical_error_rate[i] = 0
        else:
            logical_error_rate[i] = (int(torch.sum(check_channel))/(check_channel.size()[0]))

        converge_fail = torch.where((check_channel == 1) & (converge_next == 1), torch.tensor(1), torch.tensor(0))
        if int(torch.sum(converge_fail)) == 0:
            converge_fail_rate[i] = 0
        else:
            converge_fail_rate[i] = (int(torch.sum(converge_fail))/(check.size()[0]))

        converge_succ = torch.where((check_channel == 0) & (converge_next == 1), torch.tensor(1), torch.tensor(0))
        if int(torch.sum(converge_succ)) == 0:
            converge_succ_rate[i] = 0
        else:
            converge_succ_rate[i] = (int(torch.sum(converge_succ))/(check_channel.size()[0]))

        for channel in range(number_channel):
            logger.info(f'Channel {channel} metrics:')
            logger.info(f'  Data qubit accuracy: {data_qubit_acc[channel]:.6f}')
            logger.info(f'  Correction accuracy: {correction_acc[channel]:.6f}')
            logger.info(f'  Data frame error rate: {data_frame_error_rate[channel]:.6f}')
            logger.info(f'  Syndrome frame error rate: {synd_frame_error_rate[channel]:.6f}')
            logger.info(f'  Logical error rate: {logical_error_rate[channel]:.6f}')
            logger.info(f'  Converge failure rate: {converge_fail_rate[channel]:.6f}')
            logger.info(f'  Converge success rate: {converge_succ_rate[channel]:.6f}')

    return MetricResult({
        'total_time': total_time,
        'average_time_sample': average_time_sample,
        'average_iter': average_iter,
        'distribution': distribution,
        'average_time_sample_iter': average_time_sample_iter,
        'data_qubit_acc': data_qubit_acc,
        'data_frame_error_rate': data_frame_error_rate,
        'synd_frame_error_rate': synd_frame_error_rate,
        'correction_acc': correction_acc,
        'logical_error_rate': logical_error_rate,
        'invoke_rate': invoke_rate,
        'converge_fail': converge_fail_rate,
        'converge_succ': converge_succ_rate,
    })


def save_metric(out_dict, curr_dir, batch_size, target_error, dtype, physical_error_rate, num_batches, error_reach, file_name, check_num, done=0, vote_info=None):
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

    def list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    def format_time(value):
        return f'{float(value):.17e}'

    all_metrics_results = {}
    total_time_sum = 0.0
    all_check_types = ['hx', 'hz']
    final_list = []

    for i, decoder_metrics in enumerate(out_dict):
        decoder_key = f'decoder_{i}'
        if done:
            total = decoder_metrics['distribution'].sum()
            cdf = torch.cumsum(decoder_metrics['distribution'], dim=0) / total
            qs = torch.linspace(0.0, 1.0, 101, device=decoder_metrics['distribution'].device)
            indices = torch.searchsorted(cdf, qs, right=False)
            distribution = (indices + 1).int().tolist()
        else:
            distribution = decoder_metrics['distribution'].int().cpu().numpy().tolist()

        total_time_sum += float(decoder_metrics['total_time'])
        data_acc = decoder_metrics['data_qubit_acc']
        if not isinstance(data_acc, (list, tuple)):
            data_acc = [data_acc]

        check_list = []
        for idx in range(len(data_acc)):
            check_list.append({
                'data qubit accuracy': float(decoder_metrics['data_qubit_acc'][idx] if isinstance(decoder_metrics['data_qubit_acc'], (list, tuple)) else decoder_metrics['data_qubit_acc']),
                'data qubit correction accuracy': float(decoder_metrics['correction_acc'][idx] if isinstance(decoder_metrics['correction_acc'], (list, tuple)) else decoder_metrics['correction_acc']),
                'data frame error rate': float(decoder_metrics['data_frame_error_rate'][idx] if isinstance(decoder_metrics['data_frame_error_rate'], (list, tuple)) else decoder_metrics['data_frame_error_rate']),
                'syndrome frame error rate': float(decoder_metrics['synd_frame_error_rate'][idx] if isinstance(decoder_metrics['synd_frame_error_rate'], (list, tuple)) else decoder_metrics['synd_frame_error_rate']),
                'logical error rate': float(decoder_metrics['logical_error_rate'][idx] if isinstance(decoder_metrics['logical_error_rate'], (list, tuple)) else decoder_metrics['logical_error_rate']),
                'converge failure rate': float(decoder_metrics['converge_fail_rate'][idx] if isinstance(decoder_metrics['converge_fail_rate'], (list, tuple)) else decoder_metrics['converge_fail_rate']),
                'converge success rate': float(decoder_metrics['converge_succ_rate'][idx] if isinstance(decoder_metrics['converge_succ_rate'], (list, tuple)) else decoder_metrics['converge_succ_rate']),
            })
            final_list.append({
                'logical error rate': float(decoder_metrics['logical_error_rate'][idx] if isinstance(decoder_metrics['logical_error_rate'], (list, tuple)) else decoder_metrics['logical_error_rate'])
            })

        check_types = [all_check_types[check_num]] if len(check_list) == 1 else ['hx', 'hz']
        all_metrics_results[decoder_key] = {
            'algorithm': decoder_metrics['algorithm'],
            'decoder invoke rate': float(decoder_metrics['invoke_rate']),
            'average iteration': float(decoder_metrics['average_iter']),
            'iteration distribution': distribution,
            'total time (s)': format_time(decoder_metrics['total_time']),
            'average time per batch (s)': format_time(decoder_metrics['total_time']/num_batches),
            'average time per sample (s)': format_time(decoder_metrics['average_time_sample']),
            'average time per iteration (s)': format_time(decoder_metrics['average_time_sample_iter']),
        }
        for idx, check_name in enumerate(check_types[:len(check_list)]):
            all_metrics_results[decoder_key][f'{check_name}'] = check_list[idx]

    all_metrics_results['decoder_full'] = {
        'batch size': batch_size,
        'batch count': num_batches,
        'target error': target_error,
        'target error reached': error_reach,
        'data type': dtype,
        'physical error rate': physical_error_rate,
        'total time (s)': format_time(total_time_sum),
        'H matrix': file_name,
    }
    for idx, check_name in enumerate(check_types[:len(final_list)]):
        all_metrics_results['decoder_full'][f'{check_name}'] = final_list[idx]

    if vote_info is not None and vote_info.get('active'):
        rates = vote_info.get('per_round_match_rate', [])
        match_percentage = [float(r) * 100.0 for r in rates]
        all_metrics_results['vote'] = {
            'stage': vote_info['stage'],
            'd_rounds': int(vote_info['d_rounds']),
            'total samples': int(vote_info['sample_count']),
            'match percentage': match_percentage,
        }

    os.makedirs(curr_dir, exist_ok=True)
    output_path = os.path.join(curr_dir, f'result_phy_err_{physical_error_rate}.yaml')

    # Add custom representers for proper formatting
    yaml.SafeDumper.add_representer(float, float_representer)
    yaml.SafeDumper.add_representer(list, list_representer)

    with open(output_path, 'w') as f:
        yaml.safe_dump(all_metrics_results, f, sort_keys=False, default_flow_style=False)


def compute_avg_metrics(target_error, i, num_batches,
                        total_time_all,
                        average_time_sample_all,
                        average_iter_all,
                        distribution_all,
                        average_time_sample_iter_all,
                        data_qubit_acc,
                        data_frame_error_rate_all,
                        synd_frame_error_rate_all,
                        correction_acc_all,
                        logical_error_rate_all,
                        invoke_rate_all,
                        converge_fail_all,
                        converge_succ_all):
    """Backward-compatible wrapper. Prefer MetricState.compute_avg() for new code."""
    state = MetricState.__new__(MetricState)
    state.num_decoders = len(total_time_all)
    state.number_channel = len(data_qubit_acc[0]) if isinstance(data_qubit_acc[0], (list, tuple)) else 1
    state.total_time = total_time_all
    state.average_time_sample = average_time_sample_all
    state.average_iter = average_iter_all
    state.distribution = distribution_all
    state.average_time_sample_iter = average_time_sample_iter_all
    state.invoke_rate = invoke_rate_all
    state.data_qubit_acc = data_qubit_acc
    state.data_frame_error_rate = data_frame_error_rate_all
    state.synd_frame_error_rate = synd_frame_error_rate_all
    state.correction_acc = correction_acc_all
    state.logical_error_rate = logical_error_rate_all
    state.converge_fail = converge_fail_all
    state.converge_succ = converge_succ_all

    result = state.compute_avg(i, num_batches)
    return (result['total_time'], result['average_time_sample'], result['average_iter'],
            result['distribution'], result['average_time_sample_iter'],
            result['data_qubit_acc'], result['data_frame_error_rate'],
            result['synd_frame_error_rate'], result['correction_acc'],
            result['logical_error_rate'], result['invoke_rate'],
            result['converge_fail'], result['converge_succ'])


def load_checkpoint_yaml(path, number_channel):
    """Backward-compatible wrapper. Prefer MetricState.from_checkpoint() for new code."""
    state, meta = MetricState.from_checkpoint(path, number_channel, 'cpu')
    return (state.total_time, state.average_time_sample, state.average_iter,
            state.distribution, state.average_time_sample_iter,
            state.data_qubit_acc, state.data_frame_error_rate,
            state.synd_frame_error_rate, state.correction_acc,
            state.logical_error_rate, state.invoke_rate,
            state.converge_fail, state.converge_succ,
            meta['num_err'], meta['batch_size'], meta['target_error'],
            meta['dtype'], meta['physical_error_rate'], meta['batch_count'],
            meta['H_file_name'])
