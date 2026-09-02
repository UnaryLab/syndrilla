import os
import time

import numpy as np
import torch
import yaml
from loguru import logger

from syndrilla.utils import get_path


class TrainResultDumper(yaml.SafeDumper):
    """`SafeDumper` with its representers frozen as they are at import."""


TrainResultDumper.yaml_representers = dict(yaml.SafeDumper.yaml_representers)


def _flow_list(dumper, data):
    """Every list inline, wrapped at the dump width rather than one item per line."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


TrainResultDumper.add_representer(list, _flow_list)


class MetricResult(dict):
    """Dict that also supports tuple unpacking for backward compatibility."""

    _order = [
        "total_time",
        "average_time_sample",
        "average_iter",
        "distribution",
        "average_time_sample_iter",
        "data_qubit_acc",
        "data_frame_error_rate",
        "synd_frame_error_rate",
        "correction_acc",
        "logical_error_rate",
        "invoke_rate",
        "converge_fail",
        "converge_succ",
    ]

    def __iter__(self):
        return (self[k] for k in self._order)


class BatchTracker:
    """Per-batch decoder output buffers (e_v, iter, converge, timing)."""

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
        first record (per-decoder), so each can have its own shape."""
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
        """Append the ground-truth error batch, on the same axis its decodings are on."""
        err = err.to(self.device)
        err_base = 3 if self.number_channel > 1 else 2
        if err.ndim > err_base:
            err = self._flatten_rounds(err, err_base)
        elif self.rounds > 1:
            R = self.rounds
            err = (
                err.unsqueeze(1)
                .expand(err.shape[0], R, *err.shape[1:])
                .reshape(err.shape[0] * R, *err.shape[1:])
            )
        if self.e_all is None:
            self.e_all = self.ensure_buffer(err, dtype=err.dtype)
        self.e_all = torch.cat((self.e_all, err))

    def record_metric(self, decoder_idx, io_dict, elapsed):
        """Record one decoder's output for the current batch."""
        i = decoder_idx
        self.time_iter_all[i].append(elapsed)

        e_v = io_dict["e_v"].to(device=self.device, dtype=self.dtype)
        converge = io_dict["converge"].to(device=self.device, dtype=self.dtype)
        iter_val = io_dict["iter"].to(device=self.device, dtype=self.dtype)

        # Flatten rounds-into-batch: e_v base ndim is 2 (1ch) or 3 (2ch).
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
                (self.converge_all[i], torch.zeros_like(converge)), dim=0
            )
        self.converge_all[i + 1] = torch.cat(
            (self.converge_all[i + 1], converge), dim=0
        )

    def keep_samples(self, mask):
        """Restrict every per-sample buffer to the rows selected by `mask`."""
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


def _add_histogram(total, batch):
    """Add one batch's iteration histogram to the run's, zero-padding to the longest."""
    # the first batch of a decoder lands on the 0.0 the run starts each histogram at
    if not torch.is_tensor(total):
        return batch
    if total.numel() < batch.numel():
        total = torch.nn.functional.pad(total, (0, batch.numel() - total.numel()))
    elif batch.numel() < total.numel():
        batch = torch.nn.functional.pad(batch, (0, total.numel() - batch.numel()))
    return total + batch


def _opt_int(value):
    """A checkpoint field that is an int, or None when the run had no such stop condition."""
    return None if value is None else int(value)


def _yaml_rate(rate):
    """A physical error rate as a yaml scalar, or a [lower, upper, points] list."""
    if isinstance(rate, (list, tuple)):
        values = [float(value) for value in rate]
        if len(values) == 3:
            # the last value is a count of rates, not a rate
            values[2] = int(rate[2])
        return values
    try:
        return float(rate)
    except (TypeError, ValueError):
        return str(rate)


PHASE_YAML_NAMES = {
    "train": ("training", "training loss", "training error"),
    "val": ("validation", "validation loss", "validation error"),
}

# what a decoder declaring no policy of its own is selected on: the validation logical
# error every run selected on before the hook existed
DEFAULT_TRAIN_SCORE_NAME = "val_class_err"


def _yaml_losses(history, phase):
    """One phase's epoch means over the whole run, a column per term."""
    _, loss_name, error_name = PHASE_YAML_NAMES[phase]
    return {
        name: [float(entry[phase][key]) for entry in history]
        for name, key in (
            (loss_name, "total"),
            (error_name, "class_err"),
        )
    }


def _train_score_name(decoder):
    """What a decoder calls the number it selects epochs on, or the run's own default."""
    return getattr(decoder, "TRAIN_SCORE_NAME", DEFAULT_TRAIN_SCORE_NAME)


def _train_stem(decoder):
    """What every file a training run writes is named after: algo, check type, size."""
    size = (
        f"dem{decoder.m}x{decoder.n}" if decoder.from_circuit_dem else f"n{decoder.n}"
    )
    return f"{decoder.algo}_{decoder.check_type}_{size}"


def _train_fingerprint(cfg, batch_size, error_rate, H_file_name, trainer):
    """The settings a resumed training run must still agree with: model, data, schedule."""
    # the objective's and the optimizer's own settings, straight from the blocks of the
    # training yaml that configured them
    loss = trainer.loss
    loss_cfg = getattr(loss, "cfg", {"function": type(loss).__module__})
    return {
        **trainer.train_fingerprint(),
        "epochs": cfg["epochs"],
        "test_batches": cfg["test_batches"],
        "validation_batches": cfg["validation_batches"],
        "error_random_seed": cfg["error_random_seed"],
        "batch_size": batch_size,
        # a rate is a [lower, upper, points] sweep as often as a single level
        "physical_error_rate": _yaml_rate(error_rate),
        "H_file_name": H_file_name,
        **{f"loss_{key}": value for key, value in loss_cfg.items()},
        **{f"optimizer_{key}": value for key, value in trainer.optimizer_cfg.items()},
    }


_TRAIN_YAML_SETUP = {
    "algorithm": "algo",
    "device": "device",
    "data type": "dtype",
    "physical error rate": "physical_error_rate",
    "batch size": "batch_size",
    "epochs": "epochs",
    "training batches count": "test_batches",
    "validation batches count": "validation_batches",
    "error random seed": "error_random_seed",
}


def _yaml_train_setup(path):
    """Read a training run's `-ckpt` result yaml back as fingerprint fields."""
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    full = (data or {}).get("train_full")
    if not isinstance(full, dict):
        raise ValueError(
            f"<{path}> carries no `train_full` block, so it is not a training run's "
            f"result yaml. `-ckpt` takes the `<stem>_result.yaml` of the resumed run."
        )
    return {key: full[name] for name, key in _TRAIN_YAML_SETUP.items() if name in full}


class MetricState:
    """This run's metrics, in either mode syndrilla runs in."""

    # Names of scalar (per-decoder) fields
    _scalar_fields = [
        "total_time",
        "average_time_sample",
        "average_iter",
        "average_time_sample_iter",
        "invoke_rate",
    ]
    # Names of per-channel fields
    _channel_fields = [
        "data_qubit_acc",
        "data_frame_error_rate",
        "synd_frame_error_rate",
        "correction_acc",
        "logical_error_rate",
        "converge_fail",
        "converge_succ",
    ]

    def __init__(self, num_decoders, number_channel, device):
        """Build the decode half; `train_initial` builds the training one."""
        self.mode = "decode"
        self.num_decoders = num_decoders
        self.number_channel = number_channel
        self.device = device

        # scalar accumulators: list[float] indexed by decoder
        for field in self._scalar_fields:
            setattr(self, field, [0.0] * num_decoders)

        self.total_samples = [0.0] * num_decoders

        # distribution is special (tensor per decoder)
        self.distribution = [0.0] * num_decoders

        # per-channel accumulators: list[list[float]] indexed by [decoder][channel]
        for field in self._channel_fields:
            setattr(self, field, [[0.0] * number_channel for _ in range(num_decoders)])

    def report_metric(
        self,
        num_max_iter,
        e_estimated,
        e_actual,
        iteration,
        time_iteration,
        check,
        converge,
        converge_next,
        decode_idx,
    ):
        """
        This function reports the decoding iteration and accuracy.
        Returns a dict of batch metrics.
        """

        logger.info(f"Reporting decoding metric for decoder {decode_idx}.")

        sample_size = e_estimated.shape[0]
        number_channel = e_actual.shape[1]
        dtype = e_estimated.dtype
        total_time = np.sum(time_iteration)
        logger.info(f"Total time for <{sample_size}> samples: {total_time} seconds.")

        if iteration.numel() == 0:
            average_iter = 0.0
            distribution = torch.zeros(num_max_iter + 1).to(e_estimated.device)
        else:
            distribution = (
                torch.bincount(
                    (iteration - 1).int().flatten(), minlength=num_max_iter + 1
                )
                .int()
                .to(e_estimated.device)
            )
            average_iter = torch.mean(iteration).item()

        logger.info(f"Average iterations per sample: {average_iter}")
        logger.info(f"Maximum iterations: {distribution}")

        if total_time == 0:
            average_time_sample = 0
            logger.info(f"Average time per sample: {average_time_sample} seconds.")

            average_time_sample_iter = 0
            logger.info(f"Average time per iteration: {average_time_sample_iter}")
        else:
            average_time_sample = total_time / sample_size
            logger.info(f"Average time per sample: {average_time_sample} seconds.")

            average_time_sample_iter = (average_time_sample / average_iter).item()
            logger.info(f"Average time per iteration: {average_time_sample_iter}")

        if torch.isinf(torch.sum(converge)) or torch.isnan(torch.sum(converge)):
            invoke_rate = 1.0
            logger.info(f"Decoder invoke rate: {invoke_rate}")
        else:
            if int(torch.sum(converge)) == 0:
                invoke_rate = 1.0
                logger.info(f"Decoder invoke rate: {invoke_rate}")
            else:
                invoke_rate = 1.0 - ((int(torch.sum(converge))) / sample_size)
                logger.info(f"Decoder invoke rate: {invoke_rate}")

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
            eq_i = e_estimated_channel == e_actual_channel
            comp_i = torch.unique(eq_i, return_counts=True)[1]
            if int(comp_i.shape[0]) == 1:
                data_qubit_acc[i] = 1.0
            else:
                data_qubit_acc[i] = float(comp_i[1]) / (
                    float(comp_i[1]) + float(comp_i[0])
                )

            num_error = torch.sum(e_estimated_channel != e_actual_channel)

            total_ones = torch.sum((e_estimated_channel == 1) | (e_actual_channel == 1))
            if float(num_error) == 0:
                correction_acc[i] = 1
            else:
                correction_acc[i] = 1 - float(num_error) / float(total_ones)

            result = (e_estimated_channel == e_actual_channel).all(dim=1).int()
            num_correct = (result == 1).sum()
            num_incorrect = (result == 0).sum()
            if int(result.shape[0]) == 1:
                data_frame_error_rate[i] = 0.0
            else:
                data_frame_error_rate[i] = float(num_incorrect) / (
                    float(num_incorrect) + float(num_correct)
                )

            if torch.isinf(torch.sum(converge_next)) or torch.isnan(
                torch.sum(converge_next)
            ):
                synd_frame_error_rate[i] = 0
            else:
                if int(torch.sum(converge_next)) == 0:
                    synd_frame_error_rate[i] = 0.0
                else:
                    synd_frame_error_rate[i] = (
                        check_channel.size()[0] - int(torch.sum(converge_next))
                    ) / (check_channel.size()[0])

            if int(torch.sum(check_channel)) == 0:
                logical_error_rate[i] = 0
            else:
                logical_error_rate[i] = int(torch.sum(check_channel)) / (
                    check_channel.size()[0]
                )

            converge_fail = torch.where(
                (check_channel == 1) & (converge_next == 1),
                torch.tensor(1),
                torch.tensor(0),
            )
            if int(torch.sum(converge_fail)) == 0:
                converge_fail_rate[i] = 0
            else:
                converge_fail_rate[i] = int(torch.sum(converge_fail)) / (
                    check.size()[0]
                )

            converge_succ = torch.where(
                (check_channel == 0) & (converge_next == 1),
                torch.tensor(1),
                torch.tensor(0),
            )
            if int(torch.sum(converge_succ)) == 0:
                converge_succ_rate[i] = 0
            else:
                converge_succ_rate[i] = int(torch.sum(converge_succ)) / (
                    check_channel.size()[0]
                )

            for channel in range(number_channel):
                logger.info(f"Channel {channel} metrics:")
                logger.info(f"  Data qubit accuracy: {data_qubit_acc[channel]:.6f}")
                logger.info(f"  Correction accuracy: {correction_acc[channel]:.6f}")
                logger.info(
                    f"  Data frame error rate: {data_frame_error_rate[channel]:.6f}"
                )
                logger.info(
                    f"  Syndrome frame error rate: {synd_frame_error_rate[channel]:.6f}"
                )
                logger.info(f"  Logical error rate: {logical_error_rate[channel]:.6f}")
                logger.info(
                    f"  Converge failure rate: {converge_fail_rate[channel]:.6f}"
                )
                logger.info(
                    f"  Converge success rate: {converge_succ_rate[channel]:.6f}"
                )

        return MetricResult(
            {
                "total_time": total_time,
                "sample_size": sample_size,
                "average_time_sample": average_time_sample,
                "average_iter": average_iter,
                "distribution": distribution,
                "average_time_sample_iter": average_time_sample_iter,
                "data_qubit_acc": data_qubit_acc,
                "data_frame_error_rate": data_frame_error_rate,
                "synd_frame_error_rate": synd_frame_error_rate,
                "correction_acc": correction_acc,
                "logical_error_rate": logical_error_rate,
                "invoke_rate": invoke_rate,
                "converge_fail": converge_fail_rate,
                "converge_succ": converge_succ_rate,
            }
        )

    def update_metric(self, decoder_idx, batch_metrics):
        """Add one batch's report_metric result for a decoder."""
        i = decoder_idx
        ss = float(batch_metrics.get("sample_size", 1) or 1)
        self.total_samples[i] += ss

        self.total_time[i] += batch_metrics["total_time"]
        for field in (
            "average_time_sample",
            "average_iter",
            "average_time_sample_iter",
            "invoke_rate",
        ):
            getattr(self, field)[i] += batch_metrics[field] * ss

        self.distribution[i] = _add_histogram(
            self.distribution[i], batch_metrics["distribution"]
        )

        for field in self._channel_fields:
            acc = getattr(self, field)[i]
            batch_val = batch_metrics[field]
            for ch in range(self.number_channel):
                acc[ch] += batch_val[ch] * ss

    def compute_avg(self, decoder_idx, num_batches):
        """Compute averaged metrics for one decoder. Returns a dict."""
        i = decoder_idx

        logger.info(f"Reporting decoding metric for decoder {i}.")

        # per-sample rates were accumulated weighted by sample count
        denom = self.total_samples[i] or num_batches

        total_time = self.total_time[i]
        average_time_batch = total_time / num_batches
        average_time_sample = self.average_time_sample[i] / denom
        average_iter = self.average_iter[i] / denom
        distribution = self.distribution[i]
        average_time_sample_iter = self.average_time_sample_iter[i] / denom
        invoke_rate = self.invoke_rate[i] / denom

        logger.info(f"Decoder invoke rate: {invoke_rate}")
        logger.info(f"Total time: {total_time} seconds.")
        logger.info(f"Total number of batches: {num_batches}.")
        logger.info(f"Average time per batch: {average_time_batch} seconds.")
        logger.info(f"Average time per sample: {average_time_sample} seconds.")
        logger.info(f"Average iterations per sample: {average_iter}")
        logger.info(f"Iteration distribution: {distribution}")
        logger.info(f"Average time per iteration: {average_time_sample_iter}")

        result = {
            "total_time": total_time,
            "average_time_sample": average_time_sample,
            "average_iter": average_iter,
            "distribution": distribution,
            "average_time_sample_iter": average_time_sample_iter,
            "invoke_rate": invoke_rate,
            "sample_count": float(denom),
        }

        for field in self._channel_fields:
            avg = [x / denom for x in getattr(self, field)[i]]
            result[field] = avg

        for ch in range(self.number_channel):
            logger.info(f"Channel idx: {ch}")
            logger.info(f'Data qubit accuracy: {result["data_qubit_acc"][ch]}')
            logger.info(
                f'Data qubit correction accuracy: {result["correction_acc"][ch]}'
            )
            logger.info(f'Data frame error rate: {result["data_frame_error_rate"][ch]}')
            logger.info(
                f'Syndrome frame error rate: {result["synd_frame_error_rate"][ch]}'
            )
            logger.info(
                f'Output logical error rate: {result["logical_error_rate"][ch]}'
            )
            logger.info(f'Converge failure rate: {result["converge_fail"][ch]}')
            logger.info(f'Converge success rate: {result["converge_succ"][ch]}')

        logger.info("Complete.")
        return result

    def get_all_metrics(self, num_batches, algo_names, decoders=None):
        """Compute averaged metrics for all decoders. Returns list of dicts for
        save_metric.
        """
        all_metrics = []
        for i in range(self.num_decoders):
            avg = self.compute_avg(i, num_batches)
            avg["algorithm"] = algo_names[i]
            # remap keys to match save_metric expectations
            avg["converge_fail_rate"] = avg.pop("converge_fail")
            avg["converge_succ_rate"] = avg.pop("converge_succ")
            cap = None
            if decoders is not None:
                inner = getattr(decoders[i], "decoder", decoders[i])
                cap = getattr(inner, "cap", None)
            if cap is not None and cap.hists:
                avg["rebatch_speedup"] = {"warmup batches": len(cap.hists)}
                if cap.pct is not None:
                    avg["rebatch_speedup"]["chosen pct"] = cap.pct
            all_metrics.append(avg)
        return all_metrics

    def save_metric(
        self,
        out_dict,
        curr_dir,
        batch_size,
        target_error,
        dtype,
        physical_error_rate,
        num_batches,
        error_reach,
        file_name,
        check_num,
        done=0,
        target_batch=None,
    ):
        """
        Saves decoding metrics for all decoders into a single YAML file.

        Parameters:
            out_dict (list of dicts): each item is a dict with metric keys
            curr_dir (str): directory path to save YAML
            physical_error_rate (float or str): label for file naming
        """

        logger.info("Saving all decoding metrics to a YAML file.")

        def float_representer(dumper, value):
            return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.17e}")

        def list_representer(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        def format_time(value):
            return f"{float(value):.17e}"

        all_metrics_results = {}
        total_time_sum = 0.0
        all_check_types = ["hx", "hz"]
        final_list = []

        for i, decoder_metrics in enumerate(out_dict):
            decoder_key = f"decoder_{i}"
            raw_dist = decoder_metrics["distribution"].int().cpu()
            iteration_count = raw_dist.numpy().tolist()

            if done:
                total = decoder_metrics["distribution"].sum()
                cdf = torch.cumsum(decoder_metrics["distribution"], dim=0) / total
                qs = torch.linspace(
                    0.0, 1.0, 101, device=decoder_metrics["distribution"].device
                )
                indices = torch.searchsorted(cdf, qs, right=False)
                distribution = (indices + 1).int().tolist()
            else:
                distribution = raw_dist.numpy().tolist()

            total_time_sum += float(decoder_metrics["total_time"])
            data_acc = decoder_metrics["data_qubit_acc"]
            if not isinstance(data_acc, (list, tuple)):
                data_acc = [data_acc]

            check_list = []
            for idx in range(len(data_acc)):
                check_list.append(
                    {
                        "data qubit accuracy": float(
                            decoder_metrics["data_qubit_acc"][idx]
                            if isinstance(
                                decoder_metrics["data_qubit_acc"], (list, tuple)
                            )
                            else decoder_metrics["data_qubit_acc"]
                        ),
                        "data qubit correction accuracy": float(
                            decoder_metrics["correction_acc"][idx]
                            if isinstance(
                                decoder_metrics["correction_acc"], (list, tuple)
                            )
                            else decoder_metrics["correction_acc"]
                        ),
                        "data frame error rate": float(
                            decoder_metrics["data_frame_error_rate"][idx]
                            if isinstance(
                                decoder_metrics["data_frame_error_rate"], (list, tuple)
                            )
                            else decoder_metrics["data_frame_error_rate"]
                        ),
                        "syndrome frame error rate": float(
                            decoder_metrics["synd_frame_error_rate"][idx]
                            if isinstance(
                                decoder_metrics["synd_frame_error_rate"], (list, tuple)
                            )
                            else decoder_metrics["synd_frame_error_rate"]
                        ),
                        "logical error rate": float(
                            decoder_metrics["logical_error_rate"][idx]
                            if isinstance(
                                decoder_metrics["logical_error_rate"], (list, tuple)
                            )
                            else decoder_metrics["logical_error_rate"]
                        ),
                        "converge failure rate": float(
                            decoder_metrics["converge_fail_rate"][idx]
                            if isinstance(
                                decoder_metrics["converge_fail_rate"], (list, tuple)
                            )
                            else decoder_metrics["converge_fail_rate"]
                        ),
                        "converge success rate": float(
                            decoder_metrics["converge_succ_rate"][idx]
                            if isinstance(
                                decoder_metrics["converge_succ_rate"], (list, tuple)
                            )
                            else decoder_metrics["converge_succ_rate"]
                        ),
                    }
                )
                final_list.append(
                    {
                        "logical error rate": float(
                            decoder_metrics["logical_error_rate"][idx]
                            if isinstance(
                                decoder_metrics["logical_error_rate"], (list, tuple)
                            )
                            else decoder_metrics["logical_error_rate"]
                        )
                    }
                )

            check_types = (
                [all_check_types[check_num]] if len(check_list) == 1 else ["hx", "hz"]
            )
            all_metrics_results[decoder_key] = {
                "algorithm": decoder_metrics["algorithm"],
                "decoder invoke rate": float(decoder_metrics["invoke_rate"]),
                "average iteration": float(decoder_metrics["average_iter"]),
                "sample count": int(
                    round(float(decoder_metrics.get("sample_count", 0)))
                ),
                "iteration distribution": distribution,
                "iteration count": iteration_count,
            }
            # rebatch_speedup (e.g. warmup batches) is reported before the timing fields.
            if decoder_metrics.get("rebatch_speedup"):
                all_metrics_results[decoder_key]["rebatch_speedup"] = decoder_metrics[
                    "rebatch_speedup"
                ]
            all_metrics_results[decoder_key].update(
                {
                    "total time (s)": format_time(decoder_metrics["total_time"]),
                    "average time per batch (s)": format_time(
                        decoder_metrics["total_time"] / num_batches
                    ),
                    "average time per sample (s)": format_time(
                        decoder_metrics["average_time_sample"]
                    ),
                    "average time per iteration (s)": format_time(
                        decoder_metrics["average_time_sample_iter"]
                    ),
                }
            )
            for idx, check_name in enumerate(check_types[: len(check_list)]):
                all_metrics_results[decoder_key][f"{check_name}"] = check_list[idx]

        all_metrics_results["decoder_full"] = {
            "batch size": batch_size,
            "batch count": num_batches,
            "target error": target_error,
            "target batch": target_batch,
            "target error reached": error_reach,
            "data type": dtype,
            "physical error rate": physical_error_rate,
            "total time (s)": format_time(total_time_sum),
            "H matrix": file_name,
        }
        for idx, check_name in enumerate(check_types[: len(final_list)]):
            all_metrics_results["decoder_full"][f"{check_name}"] = final_list[idx]

        os.makedirs(curr_dir, exist_ok=True)
        output_path = os.path.join(
            curr_dir, f"result_phy_err_{physical_error_rate}.yaml"
        )

        # Add custom representers for proper formatting
        yaml.SafeDumper.add_representer(float, float_representer)
        yaml.SafeDumper.add_representer(list, list_representer)

        with open(output_path, "w") as f:
            yaml.safe_dump(
                all_metrics_results, f, sort_keys=False, default_flow_style=False
            )

    @classmethod
    def resume_checkpoint(
        cls,
        checkpoint,
        num_decoders,
        number_channel,
        device,
        batch_size,
        target_error,
        dtype,
        error_rate,
        H_file_name,
        target_batch=None,
    ):
        """Build this run's decode state, resuming from `checkpoint` if one was given."""
        if checkpoint is None:
            logger.info("No input Checkpoint file.")
            return cls(num_decoders, number_channel, device), 0, 0
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
        metrics, ckpt_meta = cls.load_checkpoint(checkpoint, number_channel, device)
        metrics.validate_checkpoint(
            ckpt_meta,
            batch_size,
            target_error,
            dtype,
            error_rate,
            H_file_name,
            target_batch,
        )
        return metrics, ckpt_meta["num_err"], ckpt_meta["batch_count"]

    @classmethod
    def load_checkpoint(cls, path, number_channel, device):
        """Load state from a checkpoint YAML. Returns (MetricState, ckpt_meta)."""
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")

        full = data["decoder_full"]
        ckpt_meta = {
            "H_file_name": full.get("H matrix", 0),
            "batch_size": int(full.get("batch size", 0)),
            "batch_count": int(full.get("batch count", 0)),
            "target_error": _opt_int(full.get("target error", 0)),
            "target_batch": _opt_int(full.get("target batch")),
            "num_err": int(full.get("target error reached", 0)),
            "dtype": full.get("data type", 0),
            "physical_error_rate": float(full.get("physical error rate", 0.0)),
        }

        decoder_keys = sorted(
            [k for k in data if k.startswith("decoder_") and k[8:].isdigit()],
            key=lambda x: int(x.split("_")[1]),
        )
        num_decoders = len(decoder_keys)

        state = cls(num_decoders, number_channel, device)

        for idx, key in enumerate(decoder_keys):
            entry = data[key]
            bc = ckpt_meta["batch_count"]

            has_hx = "hx" in entry
            has_hz = "hz" in entry
            if has_hx and has_hz:
                ch_map = [("hx", 0), ("hz", 1)]
            elif has_hx:
                ch_map = [("hx", 0)]
            elif has_hz:
                ch_map = [("hz", 0)]
            else:
                ch_map = []

            # per-sample rates were saved divided by the sample count, so rebuild the
            # accumulators by multiplying back by it. Old checkpoints have no 'sample
            # count'; for those (equal-size batches) it is exactly batch_count*batch_size.
            sc = float(entry.get("sample count", bc * ckpt_meta["batch_size"]))
            state.total_samples[idx] = sc

            # total_time is saved raw by save_metric (NOT divided by num_batches)。
            state.total_time[idx] = float(entry["total time (s)"])
            state.average_time_sample[idx] = (
                float(entry["average time per sample (s)"]) * sc
            )
            state.average_iter[idx] = float(entry["average iteration"]) * sc
            state.distribution[idx] = torch.tensor(entry["iteration distribution"])
            state.average_time_sample_iter[idx] = (
                float(entry["average time per iteration (s)"]) * sc
            )
            state.invoke_rate[idx] = float(entry["decoder invoke rate"]) * sc

            for ch, ch_idx in ch_map:
                ch_entry = entry[ch]
                state.data_qubit_acc[idx][ch_idx] = (
                    float(ch_entry.get("data qubit accuracy", 0.0)) * sc
                )
                state.data_frame_error_rate[idx][ch_idx] = (
                    float(ch_entry.get("data frame error rate", 0.0)) * sc
                )
                state.synd_frame_error_rate[idx][ch_idx] = (
                    float(ch_entry.get("syndrome frame error rate", 0.0)) * sc
                )
                state.correction_acc[idx][ch_idx] = (
                    float(ch_entry.get("data qubit correction accuracy", 0.0)) * sc
                )
                state.logical_error_rate[idx][ch_idx] = (
                    float(ch_entry.get("logical error rate", 0.0)) * sc
                )
                state.converge_fail[idx][ch_idx] = (
                    float(ch_entry.get("converge failure rate", 0.0)) * sc
                )
                state.converge_succ[idx][ch_idx] = (
                    float(ch_entry.get("converge success rate", 0.0)) * sc
                )

        return state, ckpt_meta

    def validate_checkpoint(
        self,
        ckpt_meta,
        batch_size,
        target_error,
        dtype,
        physical_error_rate,
        H_file_name,
        target_batch=None,
    ):
        """Validate checkpoint metadata matches current run parameters."""
        checks = [
            ("batch_size", ckpt_meta["batch_size"], batch_size),
            ("target_error", ckpt_meta["target_error"], target_error),
            ("target_batch", ckpt_meta["target_batch"], target_batch),
            ("dtype", ckpt_meta["dtype"], str(dtype)),
            (
                "physical_error_rate",
                float(ckpt_meta["physical_error_rate"]),
                float(physical_error_rate),
            ),
            ("H_file_name", ckpt_meta["H_file_name"], H_file_name),
        ]
        for name, ckpt_val, input_val in checks:
            if ckpt_val != input_val:
                raise FileNotFoundError(
                    f"Checkpoint file not match on {name}: ckpt({ckpt_val}), input({input_val})"
                )

        for i in range(self.num_decoders):
            self.distribution[i] = self.distribution[i].int().to(self.device)

    KEYS = ("total", "class_err")
    # the schedule the '-tr' yaml must supply under its 'schedule' key
    TRAIN_KEYS = ("epochs", "test_batches", "validation_batches", "error_random_seed")

    @classmethod
    def train_initial(cls, cfg, run_dir, source):
        """Validate a training schedule and build a state whose training half is live."""
        logger.info(f"Reading training schedule from <{get_path(source)}>.")
        if cfg is None:
            raise ValueError(
                f"Training yaml {source} has no 'schedule' block. Add one under "
                f"`training` with {', '.join(cls.TRAIN_KEYS)}."
            )
        missing = [key for key in cls.TRAIN_KEYS if key not in cfg]
        if missing:
            raise ValueError(
                f"Training yaml {source} is missing under 'training.schedule': "
                f"{', '.join(missing)}."
            )
        for key in cls.TRAIN_KEYS:
            value = cfg[key]
            # seed 0 is a valid seed; a zero-length schedule is not
            floor = 0 if key == "error_random_seed" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < floor:
                raise ValueError(
                    f"Training yaml {source} needs <{key}> to be an integer "
                    f">= {floor}, got <{value!r}>."
                )
        state = cls(0, 0, None)
        state.mode = "train"
        state.run_dir = run_dir
        state.cfg = cfg
        state.epochs = cfg["epochs"]
        state.period = cfg["test_batches"] + cfg["validation_batches"]
        state.total_batches = state.epochs * state.period
        state.epoch = 1
        state.history = []
        state.best = float("inf")
        state.start = time.time()
        state._epoch_start = state.start
        state.batch_size = None
        state.lr = None
        state._decoder = None
        # the run itself, bound by `train_resume_checkpoint`: this half records what the
        # run produced, and the trainer is what drives it
        state._trainer = None
        state._fingerprint = None
        state.term_names = ()
        state.keys = cls.KEYS
        state.run_meta = {}
        os.makedirs(run_dir, exist_ok=True)
        state.acc = {"train": [0.0] * len(state.keys), "val": [0.0] * len(state.keys)}
        return state

    def train_bind_loss(self, loss):
        """Key this run's metric columns by the term names the bound loss declares."""
        names = tuple(getattr(loss, "term_names", ()))
        clashes = [name for name in names if name in self.KEYS]
        if clashes or len(set(names)) != len(names):
            raise ValueError(
                f"Loss <{type(loss).__module__}> declares term names that cannot be "
                f"metered: <{', '.join(names)}>. They must be distinct and not run keys."
            )
        self.term_names = names
        self.keys = (self.KEYS[0], *names, *self.KEYS[1:])
        self.acc = {"train": [0.0] * len(self.keys), "val": [0.0] * len(self.keys)}

    def train_resume_checkpoint(
        self,
        decoder,
        batch_size,
        checkpoint,
        result_yaml,
        device,
        error_model,
        trainer,
        H_file_name,
    ):
        """Set the run up on this decoder, resume it if asked, and open the first
        batch.
        """
        # the schedule was validated here, in `train_initial`; the trainer is what runs
        # to it, so it gets the numbers rather than reading the block a second time
        trainer.configure(
            decoder,
            self.epochs,
            self.period,
            self.cfg["test_batches"],
            self.cfg["error_random_seed"],
        )
        rate = error_model.rate
        # the decoder and the fingerprint this run's checkpoints are made of
        self._decoder = decoder
        self._trainer = trainer
        self._fingerprint = _train_fingerprint(
            self.cfg, batch_size, rate, H_file_name, trainer
        )
        self.train_bind_loss(trainer.loss)
        params = int(sum(p.numel() for p in decoder.parameters()))
        self.run_meta = {
            "algorithm": decoder.algo,
            "model parameters": params,
            "device": str(decoder.device),
            "data type": str(decoder.dtype),
            "physical error rate": _yaml_rate(rate),
            "batch size": batch_size,
        }
        resume_line, start = "fresh run", 0
        if result_yaml is not None:
            if not os.path.isfile(result_yaml):
                raise FileNotFoundError(
                    f"Training result yaml not found: {result_yaml}"
                )
            self.train_validate_checkpoint(
                _yaml_train_setup(result_yaml),
                result_yaml,
                keys=_TRAIN_YAML_SETUP.values(),
            )
        if checkpoint is not None:
            resume_line = self.train_load_checkpoint(checkpoint, device)
            # batches consumed by the epochs already recorded
            start = (self.epoch - 1) * self.period

        for line in (
            f"training <{decoder.algo}>: params={params:,} "
            f"device={decoder.device} dtype={decoder.dtype}",
            f"error rate {rate}, {self.cfg['test_batches']} x {batch_size} per epoch",
            f"schedule: {dict(self.cfg)}, optimizer: {dict(trainer.optimizer_cfg)}",
            resume_line,
        ):
            logger.info(line)

        self.lr = trainer.scheduler.get_last_lr()[0]
        self.batch_size = batch_size
        self._epoch_start = time.time()
        trainer.begin_batch(start, self.epoch)
        return start

    def train_load_checkpoint(self, path, device):
        """Reload a run from its `*_last.pt` and report where it picked up."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Training checkpoint not found: {path}")
        state = torch.load(path, map_location=device, weights_only=True)
        self.train_validate_checkpoint(state.get("fingerprint"), path)
        missing = [key for key in ("epoch", "best", "history") if key not in state]
        if missing:
            raise ValueError(
                f"Training checkpoint is missing the run position "
                f"<{', '.join(missing)}>; it cannot be resumed from."
            )
        # before anything is loaded onto the decoder: `best` is a record in one metric,
        # and scoring new epochs against it in another decides nothing
        score_name = _train_score_name(self._decoder)
        saved_name = state.get("score_name", DEFAULT_TRAIN_SCORE_NAME)
        if saved_name != score_name:
            raise ValueError(
                f"Cannot resume <{path}>: its best score is a <{saved_name}> and this "
                f"run selects on <{score_name}>. The two are not comparable, so the "
                f"record cannot carry over."
            )
        self._trainer.load_train_state(state)
        self.epoch = state["epoch"]
        self.best = state["best"]
        self.history = list(state["history"])
        if self.epoch > self.epochs:
            logger.warning(
                f"<{path}> already finished all {self.epochs} epochs; there is "
                f"nothing left to run."
            )
        return (
            f"resumed <{path}> at epoch {self.epoch}, "
            f"best {score_name} {self.best:.4f}"
        )

    def train_validate_checkpoint(self, saved, path, keys=None):
        """Validate a checkpoint's fingerprint matches this run's settings."""
        if saved is None:
            raise ValueError(
                f"<{path}> carries no training fingerprint, so it cannot be resumed from. "
                f"It holds weights only; use it as the decoder yaml's `checkpoint`."
            )
        fields = self._fingerprint
        if keys is not None:
            wanted = set(keys)
            fields = {key: value for key, value in fields.items() if key in wanted}
        changed = [
            f"{key}: checkpoint <{saved[key] if key in saved else 'not recorded'}> "
            f"vs now <{value}>"
            for key, value in fields.items()
            if saved.get(key) != value
        ]
        if changed:
            raise ValueError(
                f"Cannot resume <{path}>: it was written under different settings. "
                f"{'; '.join(changed)}."
            )

    def train_update_metric(self, batch_index, terms, class_err):
        """Add one finished batch in, close the epoch if it ended one, open the next."""
        expected = len(self.keys) - 1
        if len(terms) != expected:
            raise ValueError(
                f"The loss handed back <{len(terms)}> value(s) where the run is metered on "
                f"<{expected}>: `terms()` returns one value per name in `term_names`."
            )
        values = [float(v.detach() if torch.is_tensor(v) else v) for v in terms]
        # which phase a batch counts toward is the trainer's: it is the one that opened it
        phase_name = self._trainer.phase
        phase = self.acc[phase_name]
        for i, value in enumerate(values):
            phase[i] += value
        phase[-1] += float(class_err)

        # `terms` is the total followed by the loss's own named terms
        breakdown = " ".join(
            f"{name}={value:.4f}" for name, value in zip(self.term_names, values[1:])
        )
        # `batch_index` counts the batch just finished, so the one closing an epoch
        # sits at its end rather than at the start of the next
        position = (batch_index - 1) % self.period + 1
        # a batch recorded before an optimizer is bound has no rate to report
        lr = "n/a" if self.lr is None else f"{self.lr:.2e}"
        logger.info(
            f"  batch {position:4d}/{self.period}  epoch {self.epoch:4d}/{self.epochs}  "
            f"{phase_name:5s}  lr={lr}  "
            f"loss={values[0]:.4f}{f' ({breakdown})' if breakdown else ''}  "
            f"err={float(class_err):.4f}"
        )

        # a whole number of periods closes both phases of an epoch together
        if batch_index % self.period == 0:
            self.train_compute_avg()
        self._trainer.begin_batch(batch_index, self.epoch)

    def train_compute_avg(self):
        """Close an epoch: step the schedule, average both phases, save and log."""
        # once per epoch, not once per batch
        self._trainer.scheduler.step()

        # each phase's accumulated terms over the batches that phase ran
        def mean(phase, n):
            return {k: v / n for k, v in zip(self.keys, self.acc[phase])}

        tr = mean("train", self.cfg["test_batches"])
        va = mean("val", self.cfg["validation_batches"])

        # which epoch is worth keeping is the decoder's call, not the metric half's:
        # a decoder declaring no `train_score` is selected on its validation error
        score = getattr(self._decoder, "train_score", None)
        score = float(va["class_err"] if score is None else score(tr, va))
        is_best = score < self.best
        if is_best:
            self.best = score

        now = time.time()
        elapsed = now - self._epoch_start
        self._epoch_start = now

        lr = self.lr
        # a loss that declares no terms leaves the line reporting the total alone
        breakdown = " ".join(f"{name}={tr[name]:.4f}" for name in self.term_names)
        score_name = _train_score_name(self._decoder)
        # the line already reports the validation error, so only a decoder selecting on
        # something else has a number left to spell out
        selected = (
            ""
            if score_name == DEFAULT_TRAIN_SCORE_NAME
            else f"  {score_name}={score:.4f}"
        )
        line = (
            f"epoch {self.epoch:4d}/{self.epochs}  lr={lr:.2e}  "
            f"train_loss={tr['total']:.4f}{f' ({breakdown})' if breakdown else ''}  "
            f"val_loss={va['total']:.4f}  val_err={va['class_err']:.4f}{selected}  "
            f"{elapsed:.1f}s"
            f"{'  <- best' if is_best else ''}"
        )
        self.history.append(
            {
                "epoch": self.epoch,
                "lr": lr,
                "time": float(elapsed),
                "train": tr,
                "val": va,
                "score": score,
                "best": is_best,
            }
        )
        self.epoch += 1
        self.acc = {"train": [0.0] * len(self.keys), "val": [0.0] * len(self.keys)}

        stem = _train_stem(self._decoder)
        if is_best:
            # weights only, asked of the bare decoder so the keys read back into one:
            # a wrapper prefixes every key with the attribute holding it
            torch.save(
                getattr(self._decoder, "decoder", self._decoder).state_dict(),
                os.path.join(self.run_dir, f"{stem}_best.pt"),
            )
        torch.save(
            {
                **self._trainer.train_state(),
                "epoch": self.epoch,
                "best": self.best,
                "history": self.history,
                "fingerprint": self._fingerprint,
                # `best` is a number in this metric and comparable in no other
                "score_name": score_name,
            },
            os.path.join(self.run_dir, f"{stem}_last.pt"),
        )
        self.train_save_output_yaml()

        logger.info(line)
        # after the step: `lr` above was the rate this epoch ran at, this is the next's
        self.lr = self._trainer.scheduler.get_last_lr()[0]

    def train_save_output_yaml(self):
        """Write `<stem>_result.yaml`: what the run is, then the run's curve by column."""
        history = list(self.history)
        tail = history[-1:]
        # the last epoch flagged `best` is the run's best: later flags supersede earlier
        best_epoch = next((e for e in reversed(history) if e["best"]), None)
        if tail and best_epoch is not None and best_epoch["epoch"] < tail[0]["epoch"]:
            history = [best_epoch] + tail
        else:
            history = tail
        epochs = {
            "epoch": [int(entry["epoch"]) for entry in history],
            "learning rate": [float(entry["lr"]) for entry in history],
            "time (s)": [float(entry.get("time", 0.0)) for entry in history],
            "best": [bool(entry.get("best", False)) for entry in history],
            **{
                block: _yaml_losses(history, phase)
                for phase, (block, *_) in PHASE_YAML_NAMES.items()
            },
        }
        # over the whole run, not the thinned file: the epochs it drops were still run.
        # the kept epoch is the one the selection metric picked.
        val_errors = [float(entry["val"]["class_err"]) for entry in self.history]
        times = [float(entry.get("time", 0.0)) for entry in self.history]
        per_epoch = sum(times) / len(times) if times else 0.0
        samples = self.period * (self.batch_size or 0)
        stem = _train_stem(self._decoder)
        result = {
            "train_full": {
                **self.run_meta,
                "epochs": self.epochs,
                "training batches count": self.cfg["test_batches"],
                "validation batches count": self.cfg["validation_batches"],
                "error random seed": self.cfg["error_random_seed"],
                "selection metric": _train_score_name(self._decoder),
                "best selection score": float(self.best),
                "best validation error": min(val_errors, default=float(self.best)),
                "best epoch index": (
                    int(best_epoch["epoch"]) if best_epoch is not None else None
                ),
                "total time (s)": float(time.time() - self.start),
                "total epoch time (s)": float(sum(times)),
                "average time per epoch (s)": per_epoch,
                "average time per batch (s)": (
                    per_epoch / self.period if self.period else 0.0
                ),
                "average time per sample (s)": (
                    per_epoch / samples if samples else 0.0
                ),
                "best checkpoint": os.path.join(self.run_dir, f"{stem}_best.pt"),
                "last checkpoint": os.path.join(self.run_dir, f"{stem}_last.pt"),
            },
            # the curve itself; its own `epoch` list is the index, one level down
            "training result": epochs,
        }
        with open(os.path.join(self.run_dir, f"{stem}_result.yaml"), "w") as fh:
            yaml.dump(
                result,
                fh,
                Dumper=TrainResultDumper,
                sort_keys=False,
                default_flow_style=False,
            )
        return result

    def train_save_checkpoint(self):
        """Close the run out and report where its outputs landed."""
        stem = _train_stem(self._decoder)
        self.train_save_output_yaml()
        best_path = os.path.join(self.run_dir, f"{stem}_best.pt")
        last_path = os.path.join(self.run_dir, f"{stem}_last.pt")
        result_path = os.path.join(self.run_dir, f"{stem}_result.yaml")
        for line in (
            f"done in {time.time() - self.start:.1f}s. "
            f"best {_train_score_name(self._decoder)} {self.best:.4f}",
            f"checkpoints: {best_path}, {last_path}",
            f"result: {result_path}",
            f"add to the decoder yaml to use it: checkpoint: {best_path}",
            f"to continue this run, add to the same command: "
            f"-ckpt {result_path} -tckpt {last_path}",
        ):
            logger.info(line)
