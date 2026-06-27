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

    def __init__(self, num_decoders, number_channel, shape, dtype, device, rounds=1):
        self.num_decoders = num_decoders
        self.number_channel = number_channel
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.rounds = int(rounds)
        self.reset()

    def reset(self):
        """Clear all buffers for a new batch. Buffers are allocated lazily on
        first record (per-decoder), so each can have its own shape — handy when
        voting collapses rounds for some decoders but not others."""
        nd = self.num_decoders
        self.e_v_all = [None] * nd
        self.e_all = None
        self.converge_all = [None] * (nd + 1)
        self.iter_all = [None] * nd
        self.time_iter_all = [[] for _ in range(nd)]

    def ensure_buffer(self, sample, dtype=None):
        """Allocate an empty buffer matching `sample`'s shape (with batch dim 0)."""
        empty_shape = (0,) + tuple(sample.shape[1:])
        return torch.empty(empty_shape, dtype=dtype or sample.dtype, device=self.device)

    def _flatten_rounds(self, t, base_ndim):
        """If `t` carries a rounds axis at dim=1 (i.e., t.ndim > base_ndim),
        flatten it into the batch axis: [B, R, ...] → [B*R, ...]. Otherwise
        return as-is."""
        if t.ndim > base_ndim:
            R = t.shape[1]
            return t.reshape(t.shape[0] * R, *t.shape[2:])
        return t

    def record_error(self, err):
        """Append the ground-truth error batch. If self.rounds > 1, replicate
        the error vector across rounds (errors live on data qubits and don't
        depend on measurement round)."""
        err = err.to(self.device)
        if self.rounds > 1:
            R = self.rounds
            err = err.unsqueeze(1).expand(err.shape[0], R, *err.shape[1:]) \
                       .reshape(err.shape[0] * R, *err.shape[1:])
        if self.e_all is None:
            self.e_all = self.ensure_buffer(err, dtype=err.dtype)
        self.e_all = torch.cat((self.e_all, err))

    def record_decoder(self, decoder_idx, io_dict, elapsed):
        """Record one decoder's output for the current batch.

        Each per-decoder decoder N's e_v may have a rounds axis while
        decoder N+1's doesn't (or vice versa) depending on `vote_stage`.
        """
        i = decoder_idx
        self.time_iter_all[i].append(elapsed)

        e_v = io_dict['e_v'].to(device=self.device, dtype=self.dtype)
        converge = io_dict['converge'].to(device=self.device, dtype=self.dtype)
        iter_val = io_dict['iter'].to(device=self.device, dtype=self.dtype)

        # Flatten rounds-into-batch: e_v base ndim is 2 (1ch) or 3 (2ch), if no voting.
        # iter/converge base ndim is 1 (one scalar per sample).
        e_v_base = 3 if self.number_channel > 1 else 2
        e_v = self._flatten_rounds(e_v, e_v_base)
        iter_val = self._flatten_rounds(iter_val, 1)
        converge = self._flatten_rounds(converge, 1)

        if self.e_v_all[i] is None:
            self.e_v_all[i] = self.ensure_buffer(e_v, dtype=self.dtype)
        if self.iter_all[i] is None:
            self.iter_all[i] = self.ensure_buffer(iter_val, dtype=self.dtype)
        if self.converge_all[i + 1] is None:
            self.converge_all[i + 1] = self.ensure_buffer(converge, dtype=self.dtype)
        if i == 0 and self.converge_all[0] is None:
            self.converge_all[0] = self.ensure_buffer(converge, dtype=self.dtype)

        self.e_v_all[i] = torch.cat((self.e_v_all[i], e_v), dim=0)
        self.iter_all[i] = torch.cat((self.iter_all[i], iter_val))
        if i == 0:
            self.converge_all[i] = torch.cat(
                (self.converge_all[i], torch.zeros_like(converge)), dim=0)
        self.converge_all[i + 1] = torch.cat(
            (self.converge_all[i + 1], converge), dim=0)

    def keep_samples(self, mask):
        """Restrict every per-sample buffer (ground truth + all decoders) to the
        rows selected by `mask`. Used by the adaptive cap to drop the deferred
        (unconverged) tail so this batch only meters its kept, converged samples.

        Only buffers that are full-batch-aligned with `mask` are sliced. A chained
        decoder may keep buffers sized to a subset of the batch — e.g. osd_0 only
        decodes the unconverged samples, so its `iter` buffer is shorter than the
        batch — and the full-batch cap mask doesn't apply to those; leave them
        as-is rather than mis-indexing."""
        mask = mask.to(self.device)
        n = mask.shape[0]

        def _maybe_slice(t):
            return t[mask] if (t is not None and t.shape[0] == n) else t

        self.e_all = _maybe_slice(self.e_all)
        for di in range(self.num_decoders):
            self.e_v_all[di] = _maybe_slice(self.e_v_all[di])
            self.iter_all[di] = _maybe_slice(self.iter_all[di])
        for di in range(self.num_decoders + 1):
            self.converge_all[di] = _maybe_slice(self.converge_all[di])


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

        # per-decoder total sample count: per-sample rate fields are accumulated
        # weighted by each batch's sample count and divided by this (not by the
        # batch count), so reported rates stay correct when batches differ in size
        # (e.g. the adaptive cap meters only a batch's converged samples). For
        # equal-size batches this is identical to averaging over batches.
        self.total_samples = [0.0] * num_decoders

        # distribution is special (tensor per decoder)
        self.distribution = [0.0] * num_decoders

        # per-channel accumulators: list[list[float]] indexed by [decoder][channel]
        for field in self._channel_fields:
            setattr(self, field, [[0.0] * number_channel for _ in range(num_decoders)])

        # vote tracking: per-round full-match counts accumulated across batches
        self.vote_active = False
        self.vote_stage = None
        self.vote_rounds = 0
        self.vote_match_counts = None
        self.vote_sample_count = 0

    def accumulate_vote(self, match_counts, sample_count, stage, rounds, total_bits=None):
        """Add per-prefix per-sample full-match counts from one voting application.

        ``total_bits`` is accepted but ignored (kept for callers that still
        pass it).  Each entry of ``match_counts`` is a sample count where the
        prefix-k vote matched the full vote on every bit.
        """
        if match_counts is None or sample_count <= 0 or rounds <= 0:
            return
        if not self.vote_active:
            self.vote_active = True
            self.vote_stage = stage
            self.vote_rounds = int(rounds)
            self.vote_match_counts = [0] * self.vote_rounds
        counts = list(match_counts)
        for k in range(min(self.vote_rounds, len(counts))):
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
            'rounds': self.vote_rounds,
            'sample_count': self.vote_sample_count,
            'match_counts': list(self.vote_match_counts),
            'per_round_match_rate': rates,
        }

    def accumulate(self, decoder_idx, batch_metrics):
        """Add one batch's report_metric result for a decoder.

        Per-sample rates are weighted by the batch's sample count so that batches of
        different sizes are combined correctly (see total_samples). total_time is a
        raw sum and distribution is a raw histogram, so both are added unweighted."""
        i = decoder_idx
        ss = float(batch_metrics.get('sample_size', 1) or 1)
        self.total_samples[i] += ss

        self.total_time[i] += batch_metrics['total_time']
        for field in ('average_time_sample', 'average_iter',
                      'average_time_sample_iter', 'invoke_rate'):
            getattr(self, field)[i] += batch_metrics[field] * ss

        self.distribution[i] += batch_metrics['distribution']

        for field in self._channel_fields:
            acc = getattr(self, field)[i]
            batch_val = batch_metrics[field]
            for ch in range(self.number_channel):
                acc[ch] += batch_val[ch] * ss

    def compute_avg(self, decoder_idx, num_batches):
        """Compute averaged metrics for one decoder. Returns a dict."""
        i = decoder_idx

        logger.info(f'Reporting decoding metric for decoder {i}.')

        # per-sample rates were accumulated weighted by sample count -> divide by the
        # total sample count. Fall back to num_batches for the legacy
        # compute_avg_metrics() wrapper, which accumulates unweighted.
        ts = getattr(self, 'total_samples', None)
        denom = ts[i] if (ts is not None and ts[i]) else num_batches

        total_time = self.total_time[i]
        average_time_batch = total_time / num_batches
        average_time_sample = self.average_time_sample[i] / denom
        average_iter = self.average_iter[i] / denom
        distribution = self.distribution[i]
        average_time_sample_iter = self.average_time_sample_iter[i] / denom
        invoke_rate = self.invoke_rate[i] / denom

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
            'sample_count': float(denom),
        }

        for field in self._channel_fields:
            avg = [x / denom for x in getattr(self, field)[i]]
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

    def get_all_metrics(self, num_batches, algo_names, decoders=None):
        """Compute averaged metrics for all decoders. Returns list of dicts for save_metric.

        When a decoder uses rebatch_speedup, its warm-up batch count (and the chosen cap
        percentile once warm-up has finished) is attached to that decoder's metrics from
        its RebatchSpeedup ``cap``.
        """
        all_metrics = []
        for i in range(self.num_decoders):
            avg = self.compute_avg(i, num_batches)
            avg['algorithm'] = algo_names[i]
            # remap keys to match save_metric expectations
            avg['converge_fail_rate'] = avg.pop('converge_fail')
            avg['converge_succ_rate'] = avg.pop('converge_succ')
            cap = None
            if decoders is not None:
                inner = getattr(decoders[i], 'decoder', decoders[i])
                cap = getattr(inner, 'cap', None)
            if cap is not None and cap.hists:
                avg['rebatch_speedup'] = {'warmup batches': len(cap.hists)}
                if cap.pct is not None:
                    avg['rebatch_speedup']['chosen pct'] = cap.pct
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

            # per-sample rates were saved divided by the sample count, so rebuild the
            # accumulators by multiplying back by it. Old checkpoints have no 'sample
            # count'; for those (equal-size batches) it is exactly batch_count*batch_size.
            sc = float(entry.get('sample count', bc * ckpt_meta['batch_size']))
            state.total_samples[idx] = sc

            # total_time is saved raw by save_metric (NOT divided by num_batches)。
            state.total_time[idx] = float(entry['total time (s)'])
            state.average_time_sample[idx] = float(entry['average time per sample (s)']) * sc
            state.average_iter[idx] = float(entry['average iteration']) * sc
            state.distribution[idx] = torch.tensor(entry['iteration distribution'])
            state.average_time_sample_iter[idx] = float(entry['average time per iteration (s)']) * sc
            state.invoke_rate[idx] = float(entry['decoder invoke rate']) * sc

            for ch, ch_idx in ch_map:
                ch_entry = entry[ch]
                state.data_qubit_acc[idx][ch_idx] = float(ch_entry.get('data qubit accuracy', 0.0)) * sc
                state.data_frame_error_rate[idx][ch_idx] = float(ch_entry.get('data frame error rate', 0.0)) * sc
                state.synd_frame_error_rate[idx][ch_idx] = float(ch_entry.get('syndrome frame error rate', 0.0)) * sc
                state.correction_acc[idx][ch_idx] = float(ch_entry.get('data qubit correction accuracy', 0.0)) * sc
                state.logical_error_rate[idx][ch_idx] = float(ch_entry.get('logical error rate', 0.0)) * sc
                state.converge_fail[idx][ch_idx] = float(ch_entry.get('converge failure rate', 0.0)) * sc
                state.converge_succ[idx][ch_idx] = float(ch_entry.get('converge success rate', 0.0)) * sc

        if 'vote' in data:
            v = data['vote']
            state.vote_active = True
            state.vote_stage = v.get('stage')
            state.vote_rounds = int(v.get('rounds', 0))
            state.vote_sample_count = int(v.get('total samples', 0))
            percentages = v.get('prefix vote convergence rate', v.get('match percentage'))
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
            if state.vote_rounds and len(state.vote_match_counts) < state.vote_rounds:
                state.vote_match_counts += [0] * (state.vote_rounds - len(state.vote_match_counts))

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
        distribution = torch.bincount((iteration - 1).int().flatten(), minlength=num_max_iter+1).int().to(e_estimated.device)
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
        'sample_size': sample_size,
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
        raw_dist = decoder_metrics['distribution'].int().cpu()
        iteration_count = raw_dist.numpy().tolist()

        if done:
            total = decoder_metrics['distribution'].sum()
            cdf = torch.cumsum(decoder_metrics['distribution'], dim=0) / total
            qs = torch.linspace(0.0, 1.0, 101, device=decoder_metrics['distribution'].device)
            indices = torch.searchsorted(cdf, qs, right=False)
            distribution = (indices + 1).int().tolist()
        else:
            distribution = raw_dist.numpy().tolist()

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
            'sample count': int(round(float(decoder_metrics.get('sample_count', 0)))),
            'iteration distribution': distribution,
            'iteration count': iteration_count,
        }
        # rebatch_speedup (e.g. warmup batches) is reported before the timing fields.
        if decoder_metrics.get('rebatch_speedup'):
            all_metrics_results[decoder_key]['rebatch_speedup'] = decoder_metrics['rebatch_speedup']
        all_metrics_results[decoder_key].update({
            'total time (s)': format_time(decoder_metrics['total_time']),
            'average time per batch (s)': format_time(decoder_metrics['total_time']/num_batches),
            'average time per sample (s)': format_time(decoder_metrics['average_time_sample']),
            'average time per iteration (s)': format_time(decoder_metrics['average_time_sample_iter']),
        })
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
            'rounds': int(vote_info['rounds']),
            'total samples': int(vote_info['sample_count']),
            'prefix vote convergence rate': match_percentage,
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
