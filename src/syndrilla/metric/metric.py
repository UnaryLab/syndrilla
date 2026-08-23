import json
import os
import time

import numpy as np
import torch
import yaml
from loguru import logger

from syndrilla.utils import get_path


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
        """Append the ground-truth error batch. If self.rounds > 1, replicate
        the error vector across rounds (errors live on data qubits and don't
        depend on measurement round)."""
        err = err.to(self.device)
        if self.rounds > 1:
            R = self.rounds
            err = (
                err.unsqueeze(1)
                .expand(err.shape[0], R, *err.shape[1:])
                .reshape(err.shape[0] * R, *err.shape[1:])
            )
        if self.e_all is None:
            self.e_all = self.ensure_buffer(err, dtype=err.dtype)
        self.e_all = torch.cat((self.e_all, err))

    def record_decoder(self, decoder_idx, io_dict, elapsed):
        """Record one decoder's output for the current batch.

        Each per-decoder decoder N's e_v may have a rounds axis while
        decoder N+1's doesn't, depending on what the syndrome measurer produced.
        """
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
    """This run's metrics, in either mode syndrilla runs in.

    A decode run meters per-decoder accuracy, timing and convergence; a training run
    meters per-epoch loss and owns the schedule that decides which batch belongs to
    which phase, since that is what its own accumulators are keyed by. The two have the
    same shape -- built once before the loop, fed one batch at a time, resumed from a
    checkpoint, written out at the end -- which is why they are one class, and `main.py`
    carries one `metrics` object through either mode.

    What the two modes do not share is state. A decode run's accumulators, its
    checkpoint format and its resume are disjoint from a training run's, so the members
    are grouped into a decode half and a training half and only ever one half is live.
    `mode` says which, and the training half's members carry a qualifier wherever the
    two would otherwise collide: `accumulate` and `accumulate_loss`, `begin_run` and
    `begin_train_run`, `validate_checkpoint` and `validate_train_checkpoint`. Build with
    `MetricState(num_decoders, number_channel, device)` to decode, or with
    `for_training` to train; calling across the two halves is a bug the mode flag makes
    visible rather than one this class prevents.

    The training half also owns the filtered <stem>_train.log sink: decoders log several
    INFO lines per forward pass, which over a real run would bury the epoch lines and
    grow the log to hundreds of MB, so the sink takes only the records it tags. The
    caller's own sinks are left alone. Both that log and <stem>_history.json carry the
    checkpoint stem, so a run dir shared with other runs and with the decode side --
    `tests/test_outputs`, the default -- keeps one run's log and history clear of the
    next one's rather than overwriting them.
    """

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

    # which half is live. A class attribute as well as an instance one, so the
    # `__new__`-built state in compute_avg_metrics() still answers the question.
    mode = "decode"

    def __init__(self, num_decoders, number_channel, device):
        """Build the decode half; `for_training` builds the training one."""
        self.mode = "decode"
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

    def accumulate(self, decoder_idx, batch_metrics):
        """Add one batch's report_metric result for a decoder.

        Per-sample rates are weighted by the batch's sample count so that batches of
        different sizes are combined correctly (see total_samples). total_time is a
        raw sum and distribution is a raw histogram, so both are added unweighted."""
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

        self.distribution[i] += batch_metrics["distribution"]

        for field in self._channel_fields:
            acc = getattr(self, field)[i]
            batch_val = batch_metrics[field]
            for ch in range(self.number_channel):
                acc[ch] += batch_val[ch] * ss

    def compute_avg(self, decoder_idx, num_batches):
        """Compute averaged metrics for one decoder. Returns a dict."""
        i = decoder_idx

        logger.info(f"Reporting decoding metric for decoder {i}.")

        # per-sample rates were accumulated weighted by sample count -> divide by the
        # total sample count. Fall back to num_batches for the legacy
        # compute_avg_metrics() wrapper, which accumulates unweighted.
        ts = getattr(self, "total_samples", None)
        denom = ts[i] if (ts is not None and ts[i]) else num_batches

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
        """Compute averaged metrics for all decoders. Returns list of dicts for save_metric.

        When a decoder uses rebatch_speedup, its warm-up batch count (and the chosen cap
        percentile once warm-up has finished) is attached to that decoder's metrics from
        its RebatchSpeedup ``cap``.
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

    @classmethod
    def begin_run(
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
    ):
        """Build this run's decode state, resuming from `checkpoint` if one was given.

        The `-t` counterpart is `begin_train_run`: same contract -- prepare the
        run, put a checkpoint back if there is one, and report where the run picks up --
        over the state each mode actually resumes. A decode run resumes its error count
        and its per-decoder totals; a training run resumes weights, optimizer and epoch.
        The two take different arguments because they restore different things, so the
        shared part is the contract and the call site, not a common signature.

        Returns (metrics, num_err, num_batches).
        """
        if checkpoint is None:
            logger.info("No input Checkpoint file.")
            return cls(num_decoders, number_channel, device), 0, 0
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
        metrics, ckpt_meta = cls.from_checkpoint(checkpoint, number_channel, device)
        metrics.validate_checkpoint(
            ckpt_meta, batch_size, target_error, dtype, error_rate, H_file_name
        )
        return metrics, ckpt_meta["num_err"], ckpt_meta["batch_count"]

    @classmethod
    def from_checkpoint(cls, path, number_channel, device):
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
            "target_error": int(full.get("target error", 0)),
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
    ):
        """Validate checkpoint metadata matches current run parameters."""
        checks = [
            ("batch_size", ckpt_meta["batch_size"], batch_size),
            ("target_error", ckpt_meta["target_error"], target_error),
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

    # ------------------------------------------------------------------
    # Training half (`-t`): per-epoch loss, the phase schedule, the run's
    # checkpoints and <stem>_history.json. Accumulates each batch's loss terms and logical
    # class error under its phase ('train' or 'val'), averages them at the epoch
    # boundary, formats the epoch line, and writes the run out. Live only on a state
    # built by `for_training`; the fields below do not exist on a decode
    # state, which is what keeps the two halves from being mistaken for each other.
    # ------------------------------------------------------------------

    KEYS = ("total", "lc", "lp", "ent", "class_err")
    # the schedule the '-tr' yaml must supply under its 'train' key
    TRAIN_KEYS = ("epochs", "batches_per_epoch", "val_batches", "seed")

    @classmethod
    def validate_train_cfg(cls, cfg, source):
        """Check a training schedule read off a decoder yaml and hand it back.

        Takes the schedule block itself, not the yaml it came from: this class has no
        business reading a decoder config. `source` is only used to name the file in
        error messages. `seed` stays inside `cfg` for the caller: it is the only key
        here that is not a metric concern.

        Split from `for_training` because `main.py` wants the two at different points:
        the schedule has to be checked, and its seed applied, before the decoders that
        seed initialises are built, while the state itself is only built once there is
        a decoder for it to meter.
        """
        logger.info(f"Reading training schedule from <{get_path(source)}>.")
        if cfg is None:
            raise ValueError(
                f"Decoder yaml {source} has no 'train' block, so there is no schedule "
                f"to train on. Add one under `decoder` with "
                f"{', '.join(cls.TRAIN_KEYS)}."
            )
        missing = [key for key in cls.TRAIN_KEYS if key not in cfg]
        if missing:
            raise ValueError(
                f"Decoder yaml {source} is missing under 'decoder.train': "
                f"{', '.join(missing)}."
            )
        for key in cls.TRAIN_KEYS:
            value = cfg[key]
            # seed 0 is a valid seed; a zero-length schedule is not
            floor = 0 if key == "seed" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < floor:
                raise ValueError(
                    f"Decoder yaml {source} needs <{key}> to be an integer "
                    f">= {floor}, got <{value!r}>."
                )
        return cfg

    @classmethod
    def for_training(cls, run_dir, cfg):
        """Build a state whose training half is live, on an already-checked schedule.

        The decode half is built empty rather than skipped: a training run has no
        decoders to meter, and `num_decoders == 0` says exactly that. `__init__` is the
        decode constructor because that is the signature the decode call sites and the
        checkpoint loader already use, so the training one arrives through here.
        """
        state = cls(0, 0, None)
        state.mode = "train"
        state._init_training(run_dir, cfg)
        return state

    def _init_training(self, run_dir, cfg):
        self.run_dir = run_dir
        self.cfg = cfg
        self.epochs = cfg["epochs"]
        self.period = cfg["batches_per_epoch"] + cfg["val_batches"]
        self.total_batches = self.epochs * self.period
        self.epoch = 1
        self.history = []
        self.best = float("inf")
        self.start = time.time()
        self.log = logger.bind(train=True)
        self._sink_id = None
        # the phase the batch now open runs in, 'train' or 'val'. Set by `begin_batch`
        # before any batch is drawn; the initial value only names the phase a run opens
        # in, which is the training one.
        self.phase = "train"
        # the rate the epoch now running is training at, refreshed by `record_epoch`
        # once the schedule has stepped past it. Unset until `begin_train_run` has an
        # optimizer to read it off, as is the decoder the schedule is driving.
        self.lr = None
        self._decoder = None
        self._fingerprint = None
        os.makedirs(run_dir, exist_ok=True)
        self.reset_epoch()

    def reset_epoch(self):
        """Clear both phases' loss accumulators for the next epoch."""
        self.acc = {"train": [0.0] * len(self.KEYS), "val": [0.0] * len(self.KEYS)}

    def open_log(self, header_lines):
        """Start the <stem>_train.log sink and emit the run header to stdout and the log.

        The stem comes from the bound decoder, so this has to run after `bind_decoder`,
        as `begin_train_run` does.
        """
        self._sink_id = logger.add(
            os.path.join(self.run_dir, f"{self._decoder.checkpoint_stem()}_train.log"),
            level="INFO",
            filter=lambda record: record["extra"].get("train", False),
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        )
        for line in header_lines:
            print(line)
            self.log.info(line)
        print()

    def close_log(self):
        if self._sink_id is not None:
            logger.remove(self._sink_id)
            self._sink_id = None

    def is_training(self, batch_index):
        """True if the given 0-based batch of the run is a training batch, not a val one.

        The schedule's answer for an arbitrary batch. What phase the batch now open is
        in is `self.phase`, set by `begin_batch` when it opened it -- ask that rather
        than this, so the decision is made once and read back, not made twice.
        """
        return (batch_index % self.period) < self.cfg["batches_per_epoch"]

    def epoch_done(self, done):
        """True once `done` batches complete an epoch's train and validation phases."""
        return done % self.period == 0

    def accumulate_loss(self, terms, class_err):
        """Add one batch's (total, lc, lp, ent) and class error to the phase it ran in.

        The `-t` counterpart of `accumulate`, and named apart from it because the two
        take different things from a batch: decode meters a decoder's output, training
        meters the loss read off it. Which phase to charge it to is not asked of the
        caller: `self.phase` is the one this class opened the batch in, and charging a
        batch to a phase it did not run in is exactly what recomputing it invites.

        The terms are detached on the way in: a training batch hands them over still
        attached to the graph its backward pass just used, and reading a number off one
        of those keeps the graph alive for as long as the value is held.
        """
        phase = self.acc[self.phase]
        for i, value in enumerate(terms):
            phase[i] += float(value.detach() if torch.is_tensor(value) else value)
        phase[-1] += float(class_err)

    def record_batch(self, batch_index, terms, class_err):
        """Fold one finished batch in, close the epoch if it ended one, open the next.

        `batch_index` is how many batches the run has now done, so `batch_index - 1` is
        the one being recorded and `batch_index` is the one about to be drawn. The
        phase the finished batch ran in is still `self.phase` at this point, since it is
        the closing `begin_batch` that moves it on.

        What the loop is left with is the part that is genuinely its own: run the
        decoder, read the loss off it, step the optimizer. Everything keyed by where
        the run has got to -- which phase, which epoch, when to average, when to
        checkpoint, when to reseed -- happens here, in one call, in an order the loop
        cannot get wrong.
        """
        self.accumulate_loss(terms, class_err)
        if self.epoch_done(batch_index):
            self.record_epoch()
        self.begin_batch(batch_index)

    def _mean(self, phase, n):
        return {k: v / n for k, v in zip(self.KEYS, self.acc[phase])}

    @property
    def batches_done(self):
        """Batches consumed by the epochs already recorded.

        Derived rather than counted: `record_epoch` only ever runs on an epoch
        boundary, so the batch counter and the epoch counter cannot disagree, and a
        resumed run does not have to carry a second number that could drift from this.
        """
        return (self.epoch - 1) * self.period

    def train_state(self):
        """The run position this class owns, for a resumable checkpoint.

        The weights and the optimizer are the decoder's to save; where the run had got
        to is this class's. `main.py` merges the two into one resume checkpoint.
        """
        return {"epoch": self.epoch, "best": self.best, "history": self.history}

    def load_train_state(self, state):
        """Restore the run position saved by `train_state`."""
        missing = [key for key in ("epoch", "best", "history") if key not in state]
        if missing:
            raise ValueError(
                f"Training checkpoint is missing the run position "
                f"<{', '.join(missing)}>; it cannot be resumed from."
            )
        self.epoch = state["epoch"]
        self.best = state["best"]
        self.history = list(state["history"])

    def fingerprint(self, decoder, batch_size):
        """The settings a resumed run must still agree with.

        Resuming reinstates an optimizer, a schedule and a batch position that were
        built against these values; changing any of them silently would make the
        continued run something other than the run being continued. Split by owner:
        the decoder states what the model is, this class adds the schedule it drives
        it with. Training-side counterpart of `validate_checkpoint`.
        """
        return {
            **decoder.train_fingerprint(),
            "epochs": self.cfg["epochs"],
            "batches_per_epoch": self.cfg["batches_per_epoch"],
            "val_batches": self.cfg["val_batches"],
            "seed": self.cfg["seed"],
            "batch_size": batch_size,
        }

    def validate_train_checkpoint(self, saved, decoder, batch_size, path):
        """Refuse to resume from a checkpoint written under different settings.

        The `-t` counterpart of `validate_checkpoint`: that one checks a decode run's
        yaml against the run now being continued, this one a training run's `.pt`.
        """
        if saved is None:
            raise ValueError(
                f"<{path}> carries no training fingerprint, so it cannot be resumed "
                f"from. It holds weights only; point the decoder yaml's `checkpoint` "
                f"key at it to decode, or start a fresh run."
            )
        current = self.fingerprint(decoder, batch_size)
        changed = [
            f"{key}: checkpoint <{saved.get(key)}> vs now <{value}>"
            for key, value in current.items()
            if saved.get(key) != value
        ]
        if changed:
            raise ValueError(
                f"Cannot resume <{path}>: it was written under different settings. "
                f"{'; '.join(changed)}."
            )

    def begin_train_run(self, decoder, batch_size, checkpoint, device, error_model):
        """Set the decoder up for this run, resume it if asked, and open the first batch.

        Everything here is keyed by state this half already owns -- the schedule, the
        run directory, the fingerprint and the log sink -- or by the decoder and the
        error model it is handed, so between them the caller has nothing left to pass.
        The run header is built here for the same reason, since every value in it is
        already on one side or the other.

        Three orderings are load-bearing. `check_train_batch` runs first, so a batch
        shape this decoder cannot learn from stops the run before an optimizer, a log
        or a checkpoint has been built on it. `configure_optimizer` runs before the
        resume, because Adam's moments are keyed by parameter and need an optimizer to
        be restored into. And `lr` is read after the resume, so a continued run reports
        the rate its restored schedule is actually at rather than the fresh-run one.

        Returns the batch index the run picks up at, 0 for a fresh run, which is the
        `num_batches` half of what `begin_run` hands back on the decode side.
        """
        decoder.check_train_batch(error_model.rounds, error_model.number_channel)
        decoder.configure_optimizer(self.epochs)
        self.bind_decoder(decoder, self.fingerprint(decoder, batch_size))
        resume_line, start = "fresh run", 0
        if checkpoint is not None:
            resume_line = self.resume_from(checkpoint, batch_size, device)
            start = self.batches_done
        self.open_log(
            (
                f"training <{decoder.algo}>: "
                f"params={sum(p.numel() for p in decoder.parameters()):,} "
                f"device={decoder.device} dtype={decoder.dtype}",
                f"error rate {error_model.rate}, "
                f"{self.cfg['batches_per_epoch']} x {batch_size} per epoch",
                f"config: {dict(self.cfg)}",
                resume_line,
            )
        )
        self.lr = decoder.current_lr()
        self.begin_batch(start)
        return start

    def begin_batch(self, batch_index):
        """Open a batch: seed the phase it starts, if it starts one, and enter that phase.

        Seeding is per phase rather than per epoch because the two phases want opposite
        things. Training reseeds to the same value every epoch, so the model sees one
        fixed training set instead of a fresh draw each time round. Validation reseeds
        to an epoch-dependent value, so it keeps sampling new errors rather than
        replaying the training set's tail.

        Both are a function of where the run is, not of the draws before them, so a
        resumed run continues the same sequence a straight-through run would have.

        Switching the decoder is done here, not left to the caller, because the phase
        this class picks and the mode the decoder runs the batch in are the same
        decision: a batch metered as validation that still built a graph, or the
        reverse, is a run silently training on the wrong set. For the same reason the
        phase is kept in `self.phase` and returned, rather than left for a caller to
        work out again from the batch index -- one decision, made here, read back
        everywhere else.
        """
        position = batch_index % self.period
        if position == 0:
            self.seed_train_phase()
        elif position == self.cfg["batches_per_epoch"]:
            self.seed_val_phase()
        self.phase = "train" if self.is_training(batch_index) else "val"
        if self._decoder is not None:
            self._decoder.set_training(self.phase == "train")
        return self.phase

    def seed_train_phase(self):
        """Seed the training stream, identically for every epoch.

        No epoch term: that is what makes each epoch train on the same batches. The
        multiplier keeps neighbouring run seeds apart, so `seed` and `seed + 1` do not
        share a stream.
        """
        torch.manual_seed(self.cfg["seed"] * 1_000_003)

    def seed_val_phase(self):
        """Seed the validation stream, freshly for each epoch.

        Offset well clear of the training seed so validation never replays training
        errors, and carrying the epoch so each round of validation is new data.
        """
        torch.manual_seed(self.cfg["seed"] * 1_000_003 + 9_999_991 + self.epoch)

    def bind_decoder(self, decoder, fingerprint):
        """Remember what this run's checkpoints are made of.

        The decoder supplies its own half of the state and owns the file format; this
        class decides when a checkpoint is written, where it goes and what run position
        travels with it. Binding once here keeps every later call free of both.
        """
        self._decoder = decoder
        self._fingerprint = fingerprint

    def resume_from(self, path, batch_size, device):
        """Reload a run from its `*_last.pt` and report where it picked up.

        Both halves are put back together: the decoder's weights, optimizer, schedule
        and RNG, and this class's epoch, best and history. Call it after the decoder has
        an optimizer, since Adam's moments are keyed by parameter and need one to be
        restored into.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Training checkpoint not found: {path}")
        state = torch.load(path, map_location=device, weights_only=True)
        self.validate_train_checkpoint(
            state.get("fingerprint"), self._decoder, batch_size, path
        )
        self._decoder.load_train_state(state)
        self.load_train_state(state)
        if self.epoch > self.epochs:
            logger.warning(
                f"<{path}> already finished all {self.epochs} epochs; there is "
                f"nothing left to run."
            )
        return (
            f"resumed <{path}> at epoch {self.epoch}, "
            f"best val class error {self.best:.4f}"
        )

    def save_checkpoint(self, is_best):
        """Write this epoch's checkpoints, merging the run position into the decoder's."""
        self._decoder.save_checkpoints(
            self.run_dir,
            is_best,
            extra={**self.train_state(), "fingerprint": self._fingerprint},
        )

    def record_epoch(self):
        """Close an epoch: step the schedule, average both phases, log, and reset.

        The checkpoint write happens here rather than in the caller: this class is
        what knows the run directory, whether the epoch improved and where the run has
        got to. The decoder still owns the file format and supplies its own half of the
        state, through `save_checkpoint`. The schedule step is here for the same kind of
        reason: the cosine schedule advances once per epoch, and this is the only place
        that knows an epoch just ended, so putting it anywhere else leaves a caller able
        to step it twice or not at all.

        Three orderings are load-bearing. The callback fires after the epoch is folded
        into `history` and `epoch` has advanced, so the state it is handed describes
        the run to *resume* -- the next epoch to run, not the one just finished. The
        epoch line prints after the callback, so seeing a line means that epoch's
        checkpoint is already on disk and a run interrupted there is recoverable. And
        `lr` is refreshed at the end, after the step: the epoch being recorded ran at
        the rate carried in `self.lr` since the last call, while what the decoder
        reports from here on belongs to the epoch about to start.
        """
        self._decoder.lr_step()

        tr = self._mean("train", self.cfg["batches_per_epoch"])
        va = self._mean("val", self.cfg["val_batches"])

        is_best = va["class_err"] < self.best
        if is_best:
            self.best = va["class_err"]

        lr = self.lr
        line = (
            f"epoch {self.epoch:4d}/{self.epochs}  lr={lr:.2e}  "
            f"loss={tr['total']:.4f} (lc={tr['lc']:.4f} lp={tr['lp']:.4f} ent={tr['ent']:.4f})  "
            f"val_loss={va['total']:.4f}  val_class_err={va['class_err']:.4f}"
            f"{'  <- best' if is_best else ''}"
        )
        self.history.append({"epoch": self.epoch, "lr": lr, "train": tr, "val": va})
        self.epoch += 1
        self.reset_epoch()
        self.save_checkpoint(is_best)
        print(line)
        self.log.info(line)
        self.lr = self._decoder.current_lr()

    def save_history(self):
        """Write <stem>_history.json and report where the run's checkpoints landed.

        The name comes from the bound decoder, so the paths printed are the ones
        actually written rather than a fixed pair this class assumed.
        """
        stem = self._decoder.checkpoint_stem()
        history_path = os.path.join(self.run_dir, f"{stem}_history.json")
        with open(history_path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        best_path = os.path.join(self.run_dir, f"{stem}.pt")
        last_path = os.path.join(self.run_dir, f"{stem}_last.pt")
        print(
            f"\ndone in {time.time() - self.start:.1f}s. "
            f"best val class error {self.best:.4f}"
        )
        print(f"checkpoints: {best_path}, {last_path}")
        print(f"history: {history_path}")
        print(f"\nadd to the decoder yaml to use it:\n\n  checkpoint: {best_path}\n")
        print(
            f"to continue this run, add to the same command:\n\n  -tckpt {last_path}\n"
        )


def report_metric(
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
            torch.bincount((iteration - 1).int().flatten(), minlength=num_max_iter + 1)
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
            data_qubit_acc[i] = float(comp_i[1]) / (float(comp_i[1]) + float(comp_i[0]))

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
            converge_fail_rate[i] = int(torch.sum(converge_fail)) / (check.size()[0])

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
            logger.info(f"  Converge failure rate: {converge_fail_rate[channel]:.6f}")
            logger.info(f"  Converge success rate: {converge_succ_rate[channel]:.6f}")

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


def save_metric(
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
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

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
                        if isinstance(decoder_metrics["data_qubit_acc"], (list, tuple))
                        else decoder_metrics["data_qubit_acc"]
                    ),
                    "data qubit correction accuracy": float(
                        decoder_metrics["correction_acc"][idx]
                        if isinstance(decoder_metrics["correction_acc"], (list, tuple))
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
            "sample count": int(round(float(decoder_metrics.get("sample_count", 0)))),
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
        "target error reached": error_reach,
        "data type": dtype,
        "physical error rate": physical_error_rate,
        "total time (s)": format_time(total_time_sum),
        "H matrix": file_name,
    }
    for idx, check_name in enumerate(check_types[: len(final_list)]):
        all_metrics_results["decoder_full"][f"{check_name}"] = final_list[idx]

    os.makedirs(curr_dir, exist_ok=True)
    output_path = os.path.join(curr_dir, f"result_phy_err_{physical_error_rate}.yaml")

    # Add custom representers for proper formatting
    yaml.SafeDumper.add_representer(float, float_representer)
    yaml.SafeDumper.add_representer(list, list_representer)

    with open(output_path, "w") as f:
        yaml.safe_dump(
            all_metrics_results, f, sort_keys=False, default_flow_style=False
        )


def compute_avg_metrics(
    target_error,
    i,
    num_batches,
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
    converge_succ_all,
):
    """Backward-compatible wrapper. Prefer MetricState.compute_avg() for new code."""
    state = MetricState.__new__(MetricState)
    state.num_decoders = len(total_time_all)
    state.number_channel = (
        len(data_qubit_acc[0]) if isinstance(data_qubit_acc[0], (list, tuple)) else 1
    )
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
    return (
        result["total_time"],
        result["average_time_sample"],
        result["average_iter"],
        result["distribution"],
        result["average_time_sample_iter"],
        result["data_qubit_acc"],
        result["data_frame_error_rate"],
        result["synd_frame_error_rate"],
        result["correction_acc"],
        result["logical_error_rate"],
        result["invoke_rate"],
        result["converge_fail"],
        result["converge_succ"],
    )


def load_checkpoint_yaml(path, number_channel):
    """Backward-compatible wrapper. Prefer MetricState.from_checkpoint() for new code."""
    state, meta = MetricState.from_checkpoint(path, number_channel, "cpu")
    return (
        state.total_time,
        state.average_time_sample,
        state.average_iter,
        state.distribution,
        state.average_time_sample_iter,
        state.data_qubit_acc,
        state.data_frame_error_rate,
        state.synd_frame_error_rate,
        state.correction_acc,
        state.logical_error_rate,
        state.invoke_rate,
        state.converge_fail,
        state.converge_succ,
        meta["num_err"],
        meta["batch_size"],
        meta["target_error"],
        meta["dtype"],
        meta["physical_error_rate"],
        meta["batch_count"],
        meta["H_file_name"],
    )
