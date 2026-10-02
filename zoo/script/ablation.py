"""Add-one-in ablation of the six optimization groups on BP + OSD-0.

Sweeps two axes: code distance d (--distances) and circuit error rate p (--ps).
Runs bp_norm_min_sum followed by osd_0 on stim surface_code:rotated_memory_x
circuits (rounds = d, every circuit noise key at p) on a cumulative ladder of
configs: all_off, then the group keys turned on one at a time in the order
memory_opt, pruning_opt, fusion_opt, mapping_opt, gather_opt, rebatch_opt, so
+rebatch_opt is all on. Two GPU paths: the CUDA kernels (bp_norm_min_sum_cuda,
osd_0_cuda) and the PyTorch decoders (force_pytorch, BP compile at its
default; mapping_opt or gather_opt off forces eager BP, so turning both on is
part of those steps' effect). Group keys go into each stage's config block and
into the top-level matrix config, where memory_opt false also turns sparse_h
off (dense H). Each pass decodes all shots in batches the way main.py's batch
loop does: with the rebatch cap (rebatch_opt) on, BP warms the cap up over the
first batches, then stops each batch at the chosen percentile, and the shots BP
has not converged are deferred and decoded again uncapped in later batches.
Warmup passes repeat until the cap is chosen, at most --max-warmup-passes; a
row whose cap has not settled by then is timed with the cap off. Before the
first row of each (d, p, dtype, path) one untimed build (all on) and one batch
decode of 64 shots loads the CUDA context, extensions and workspaces, so no row
is charged for them. For each (d, p, path, dtype, config) it records the median
decode wall time of a pass over the timed passes after the warmup passes (and
the BP and OSD stage times bp_ms and osd_ms), the speedup over the previous
ladder step (speedup_step) and over all_off (speedup_total), the decode's peak GPU memory, the resident decoder memory
after warmup, the device memory of H, the logical error rate and the chosen
cap, and appends each row as soon as it finishes to one markdown table (--out)
and to a CSV with the same path and a .csv suffix. The reference rows a
requested config needs (the step just before it and all_off) are added to the
run set and run first, in ladder order. all_off and +rebatch_opt (the two ends)
run at every --dtypes value; the middle steps run at float32 only. Each pass
decodes --shots shots (default 4096). The row timeout (default 1800 s) is checked at every
batch boundary; a pass stopped there reports time = measured time x shots /
shots done (extrapolated), the LER over the shots done, and ends the row. With
--batch-size auto (the default is a fixed 64), each row probes its own batch size: the
largest power of two from 64 up to --shots // --min-batches (default 16, so
every pass has at least 16 batches and the BP cap can warm up) whose one-batch
decode with that row's config fits in GPU memory and in the row timeout.

Usage:
    conda run -n syndrilla python zoo/script/ablation.py [--distances 3 9 15 21 27] [--ps 5e-4 1e-3 5e-3 1e-2] [--shots 4096] [--batch-size 64] [--row-timeout 1800] ...
"""
import argparse
import csv
import datetime
import gc
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder import create_decoder
from syndrilla.interface.stim.stim import create as stim_iface
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype

STEPS = (  # ladder order
    "memory_opt",
    "pruning_opt",
    "fusion_opt",
    "mapping_opt",
    "gather_opt",
    "rebatch_opt",
)
LADDER = ("all_off", *(f"+{s}" for s in STEPS))
# config k turns the first k steps on; +rebatch_opt is all on
CONFIGS = {c: {s: i < k for i, s in enumerate(STEPS)} for k, c in enumerate(LADDER)}
ENDS = (LADDER[0], LADDER[-1])  # the only configs run at every --dtypes value
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)
DEV = {"device_type": "cuda", "device_idx": 0}
MAX_ITER = 181
SEED = 0
ABLATE_DTYPE = "float32"  # the only dtype for the middle steps


def make_case(d, p, dtype):
    """Stim interface (no decoders) for one distance, error rate and dtype."""
    return stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": d},
        error_cfg={k: p for k in NOISE},
        syndrome_cfg={"rounds": d},
        decoding_cfg={"check_type": "hx", "dtype": dtype, "device": DEV},
    )


def batches(it, dtype, shots, batch_size):
    """Yield (synd, llr0, sampled observable flips) for the shots in batches.
    Batch i is drawn with seed SEED + i, so every pass and every config at a
    given d, p and dtype decodes the same shots."""
    for i, start in enumerate(range(0, shots, batch_size)):
        n = min(batch_size, shots - start)
        torch.manual_seed(SEED + i)
        np.random.seed(SEED + i)
        z = torch.zeros(
            n, it.error_model.num_errors, dtype=getattr(torch, dtype), device="cuda"
        )
        _, dl = it.error_model.inject_error(z, n)
        e, llr0, _ = next(iter(dl))
        synd = it.syndrome_generator.measure_syndrome(e, None)
        yield synd, llr0, it.syndrome_generator.observable_flips.clone()


def build(it, path, dtype, config):
    """(matrix bundle, [BP, OSD] decoders, H for decode, memory_allocated after the
    bundle and H are loaded and before the decoders are built) for one path and one
    ablation config."""
    keys = CONFIGS[config]
    groups = dict(keys)
    bp = {"max_iter": MAX_ITER, **groups}
    bundle = load_matrices(
        {**it.matrix_cfg, **groups}, *parse_device_dtype({"device": DEV, "dtype": dtype})
    )
    cfg = {
        "algorithm": ["bp_norm_min_sum", "osd_0"],
        "check_type": "hx",
        "dtype": dtype,
        "device": DEV,
        "force_pytorch": path == "pytorch",
        "config": [bp, dict(groups)],
    }
    H = bundle.select("hx")[3]  # the driver's H, allocated before base
    base = torch.cuda.memory_allocated()
    decs = create_decoder(cfg=cfg, bundle=bundle)
    for dec in decs:
        dec.eval()
    return bundle, decs, H, base


def decode(decs, synd, llr0, H):
    """(io after the last stage, BP's converge flags, [BP, OSD] seconds), each
    stage's call timed between torch.cuda.synchronize calls."""
    io = {"synd": synd.clone(), "llr0": llr0.clone(), "H_matrix": H}
    stage_secs = [0.0] * len(decs)
    with torch.no_grad():
        for i, dec in enumerate(decs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            io = dec(io)
            torch.cuda.synchronize()
            stage_secs[i] += time.perf_counter() - t
            if i == 0:
                bp_converge = io["converge"].clone()
    return io, bp_converge, stage_secs


def probe_batch_size(it, decs, H, dtype, limit, deadline):
    """Largest power-of-two batch size from 64 up to limit whose decode of one
    freshly drawn batch fits in GPU memory and ends before deadline; None when
    the decode at 64 ends past deadline. Raises the out-of-memory error when 64
    does not fit. BP runs with cap_bypass on, so probe batches do not feed the
    cap's warm-up."""
    inner = getattr(decs[0], "decoder", decs[0])
    inner.cap_bypass = True
    size, fit = 64, None
    try:
        while size <= max(limit, 64):
            batch = None
            try:
                batch = next(batches(it, dtype, size, size))
                decode(decs, *batch[:2], H)
                if time.monotonic() > deadline:
                    break
                fit = size
            except torch.cuda.OutOfMemoryError:
                if size == 64:
                    raise
                break
            finally:
                del batch
                gc.collect()
                torch.cuda.empty_cache()
            size *= 2
    finally:
        inner.cap_bypass = False
    return fit


def run_pass(it, decs, H, l_matrix, shots, batch_size, dtype, deadline):
    """(decode seconds, BP seconds, OSD seconds, logical failures, peak bytes,
    shots done) of one pass
    over all shots, run as main.py's batch loop runs them. Once some shot has a
    result, the pass stops at the first batch boundary (main loop or deferred
    flush) past deadline; shots done counts the shots with a result so far
    (all shots for a full pass). When BP stopped a batch at the cap,
    the shots BP did not converge are deferred and their results dropped; the
    deferred shots are decoded again uncapped, a batch at a time once the queue
    holds batch_size shots, and the rest after the last batch. Time is the sum
    of the decode calls, each between torch.cuda.synchronize calls; BP and OSD
    seconds sum each stage's call inside them (see decode); peak is
    max_memory_allocated during a call minus the allocation just before it, max
    over the calls."""
    inner = getattr(decs[0], "decoder", decs[0])
    capped = getattr(inner, "cap", None) is not None
    secs, bp_secs, osd_secs, fails, peak, done = 0.0, 0.0, 0.0, 0, 0, 0
    queue = None  # deferred (synd, llr0, obs)

    def run(synd, llr0, obs, bypass):
        nonlocal secs, bp_secs, osd_secs, fails, peak, queue, done
        if capped:
            inner.cap_bypass = bypass
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        io, bp_converge, stage_secs = decode(decs, synd, llr0, H)
        torch.cuda.synchronize()
        secs += time.perf_counter() - t
        bp_secs += stage_secs[0]
        osd_secs += stage_secs[1]
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
        fail = it.logical_check.check(io["e_v"], obs, l_matrix, io["converge"])
        if capped and inner.cap_active_last:
            if "defer" in io:
                keep = ~io["defer"].flatten()
            else:
                keep = bp_converge.flatten() > 0
            late = (synd[~keep], llr0[~keep], obs[~keep])
            queue = late if queue is None else tuple(map(torch.cat, zip(queue, late)))
            fail = fail[keep.to(fail.device)]
        fails += int(fail.sum())
        done += len(fail)

    def flush(n):
        nonlocal queue
        head = tuple(t[:n] for t in queue)
        queue = tuple(t[n:] for t in queue)
        run(*head, bypass=True)

    def late():
        return done > 0 and time.monotonic() > deadline

    for synd, llr0, obs in batches(it, dtype, shots, batch_size):
        if late():
            return secs, bp_secs, osd_secs, fails, peak, done
        run(synd, llr0, obs, bypass=False)
        while queue is not None and len(queue[0]) >= batch_size:
            if late():
                return secs, bp_secs, osd_secs, fails, peak, done
            flush(batch_size)
    while queue is not None and len(queue[0]):
        if late():
            return secs, bp_secs, osd_secs, fails, peak, done
        flush(batch_size)
    return secs, bp_secs, osd_secs, fails, peak, done


class UtilSampler:
    """Samples torch.cuda.utilization(0) and torch.cuda.memory_usage(0) every
    0.1 s on a daemon thread between start() and stop(); stop() returns the mean
    of each, rounded to one decimal, or None when no sample landed."""

    def start(self):
        self.sm, self.mem, self.halt = [], [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.halt.wait(0.1):
            self.sm.append(torch.cuda.utilization(0))
            self.mem.append(torch.cuda.memory_usage(0))

    def stop(self):
        self.halt.set()
        self.thread.join()
        return tuple(round(float(statistics.mean(v)), 1) if v else None for v in (self.sm, self.mem))


def tensor_mib(H):
    """Device memory of H: the dense tensor, or the indices and values of a sparse one."""
    if H.is_sparse:
        return (H._indices().nbytes + H._values().nbytes) / 2**20
    return H.nbytes / 2**20


def measure(it, bundle, decs, H, dtype, batch_size, args, deadline, base):
    """(median ms per pass, median BP ms, median OSD ms (each the median over the
    same passes as time), peak MiB above the pre-call allocation, resident MiB
    after warmup above base, LERs per timed pass, status, cap note, sm_util,
    mem_util) of BP + OSD. sm_util and mem_util are the mean NVML utilization
    percents sampled every 0.1 s during the timed passes (see UtilSampler).
    Warmup passes repeat until the cap is chosen (one pass when the cap is off),
    at most args.max_warmup_passes; if the cap has not settled by then, the cap
    is removed so the timed passes run uncapped, and the cap note says so (else
    None). Stops after any warmup or timed pass that ends past deadline. A pass
    stopped at deadline inside it (warmup or timed) ends the measurement: its
    time extrapolated to all shots (seconds x shots / shots done) and its LER
    over the shots done become the row's last (for a warmup pass, only)
    result."""
    l_matrix = bundle.get_l_matrix("hx", 1)
    shape = (args.shots, batch_size, dtype, deadline)
    inner = getattr(decs[0], "decoder", decs[0])
    cap = getattr(inner, "cap", None)
    note = None
    times, bps, osds, peaks, lers = [], [], [], [], []
    partial = None  # shots done by a pass stopped at deadline

    def one_pass():
        nonlocal partial
        secs, bp_secs, osd_secs, fails, peak, done = run_pass(
            it, decs, H, l_matrix, *shape
        )
        if done < args.shots:
            partial = done
        scale = args.shots / done
        return secs * scale, bp_secs * scale, osd_secs * scale, fails / done, peak

    for n in range(1, args.max_warmup_passes + 1):  # compile, caches, cap warm-up
        secs, bp_secs, osd_secs, ler, peak = one_pass()
        if partial is not None:
            times, bps, osds, lers, peaks = [secs], [bp_secs], [osd_secs], [ler], [peak]
            break
        if cap is None or cap.done or time.monotonic() > deadline:
            break
        if n == args.max_warmup_passes:
            note = f"cap not settled after {len(cap.hists)} batches"
            inner.cap = None
    torch.cuda.synchronize()
    resident = (torch.cuda.memory_allocated() - base) / 2**20
    sampler = UtilSampler()
    sampler.start()
    try:
        for _ in range(args.repeats):
            if partial is not None or time.monotonic() > deadline:
                break
            secs, bp_secs, osd_secs, ler, peak = one_pass()
            times.append(secs)
            bps.append(bp_secs)
            osds.append(osd_secs)
            peaks.append(peak)
            lers.append(ler)
    finally:
        sm_util, mem_util = sampler.stop()
    ms = 1e3 * statistics.median(times) if times else None
    bp_ms = 1e3 * statistics.median(bps) if bps else None
    osd_ms = 1e3 * statistics.median(osds) if osds else None
    mib = max(peaks) / 2**20 if peaks else None
    if partial is not None:
        status = f"timeout, {partial} of {args.shots} shots"
    elif len(times) < args.repeats:
        status = f"timeout after {len(times)} timed passes"
    else:
        status = "ok"
    return ms, bp_ms, osd_ms, mib, resident, lers, status, note, sm_util, mem_util


def describe_cap(decs):
    """The cap BP chose: percentile and warm-up batches, "off: <reason>" when
    warm-up declined a cap, or "-" with no cap."""
    cap = getattr(getattr(decs[0], "decoder", decs[0]), "cap", None)
    if cap is None:
        return "-"
    if not cap.done:
        return f"not settled after {len(cap.hists)} batches"
    if cap.declined is not None:
        return f"off: {cap.declined}"
    return f"p{cap.pct} after {len(cap.hists)} batches"


def describe(decs, path):
    """Decoder modules, and for the PyTorch path whether BP runs compiled."""
    inner = [getattr(d, "decoder", d) for d in decs]
    names = "+".join(
        os.path.basename(sys.modules[type(x).__module__].__file__)[:-3] for x in inner
    )
    compiled = ("yes" if inner[0].compile else "no") if path == "pytorch" else "-"
    return names, compiled


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def fmt_ler(lers):
    if not lers:
        return "-"
    s = f"{statistics.mean(lers):.4f}"
    if min(lers) != max(lers):
        s += f" (varies {min(lers):.4f} to {max(lers):.4f})"
    return s


def run_row(it, path, dtype, config, batch_size, args):
    """Row fields (batch size, decoders, compiled, ms, BP ms, OSD ms, peak MiB,
    resident MiB, H MiB, sm_util, mem_util, LER, cap, status). With batch_size "auto" the row probes its own
    batch size with its own decoders."""
    deadline = time.monotonic() + args.row_timeout
    names, compiled, ms, bp_ms, osd_ms, mib, res_mib, h_mib, lers = (
        "-",
        "-",
        None,
        None,
        None,
        None,
        None,
        None,
        [],
    )
    cap = "-"
    sm_util = mem_util = None
    bundle = decs = H = None
    bs = None if batch_size == "auto" else batch_size
    try:
        bundle, decs, H, base = build(it, path, dtype, config)
        names, compiled = describe(decs, path)
        h_mib = tensor_mib(H)
        if bs is None:
            limit = args.shots // args.min_batches
            bs = probe_batch_size(it, decs, H, dtype, limit, deadline)
        if bs is None:
            status = "timeout in probe"
        else:
            ms, bp_ms, osd_ms, mib, res_mib, lers, status, note, sm_util, mem_util = measure(
                it, bundle, decs, H, dtype, bs, args, deadline, base
            )
            cap = note or describe_cap(decs)
    except (torch.cuda.OutOfMemoryError, MemoryError):
        status = "does not fit"
    except Exception as exc:
        status = f"error: {type(exc).__name__}"
        logger.exception(f"{path} {dtype} {config} failed")
    del bundle, decs, H
    gc.collect()
    torch.cuda.empty_cache()
    return bs, names, compiled, ms, bp_ms, osd_ms, mib, res_mib, h_mib, sm_util, mem_util, lers, cap, status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 9, 15, 21, 27])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument(
        "--batch-size",
        type=lambda s: s if s == "auto" else int(s),
        default=64,
        help="int (default 64), or auto: largest power of two from 64 that fits, probed per row",
    )
    ap.add_argument(
        "--min-batches",
        type=int,
        default=16,
        help="auto batch size only: B is at most --shots // this, so a pass has at least this many batches",
    )
    ap.add_argument("--ps", type=float, nargs="+", default=[5e-4, 1e-3, 5e-3, 1e-2])
    ap.add_argument(
        "--paths", nargs="+", default=["cuda", "pytorch"], choices=["cuda", "pytorch"]
    )
    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=["float64", "float32"],
        choices=["float64", "float32"],
    )
    ap.add_argument(
        "--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS)
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--row-timeout",
        type=float,
        default=1800,
        help="seconds per row; checked at each batch boundary",
    )
    ap.add_argument(
        "--max-warmup-passes",
        type=int,
        default=1,
        help="warmup passes before a row is timed with the cap off if it has not settled",
    )
    ap.add_argument(
        "--out",
        default=f"zoo/ablation/{datetime.date.today():%Y-%m-%d}-ablation-results.md",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(  # warnings, plus the cap's warm-up done line
        sys.stderr,
        filter=lambda r: r["level"].no >= 30 or "warm-up done" in r["message"],
    )

    cmd = "conda run -n syndrilla python " + " ".join(sys.argv)
    header = [
        "# Optimization group ablation (add-one-in), BP + OSD-0",
        "",
        f"- Hardware: {torch.cuda.get_device_name(0)}; torch {torch.__version__}, CUDA {torch.version.cuda}",
        f"- Circuits: stim surface_code:rotated_memory_x, rounds = d, every circuit noise key at p, p in {', '.join(f'{p:g}' for p in args.ps)}; check_type hx",
        f"- Decoders: bp_norm_min_sum (max_iter {MAX_ITER}) then osd_0. Path cuda = CUDA kernels; path pytorch = force_pytorch on the GPU, BP compile left at its default in every row; mapping_opt or gather_opt off forces eager BP (compiled = no), so compiled BP is part of the effect of the step that turns the last of the two on",
        f"- Configs (add-one-in): a cumulative ladder all_off, {', '.join(LADDER[1:])}; config k sets the first k group keys true and the rest false, in each stage's config block and in the top-level matrix config, so {LADDER[-1]} is all on. memory_opt false turns sparse_h off (dense int64 H on the device), so sparse H arrives with +memory_opt. rebatch_opt false turns the BP iteration cap off, so the cap and deferred shots arrive with +rebatch_opt. {' and '.join(ENDS)} run at every --dtypes value; the middle steps run at {ABLATE_DTYPE} only",
        f"- Shots: {args.shots} shots in batches of B = "
        + (f"the batch column: per row, with that row's config, one freshly drawn batch is decoded at 64, 128, ... up to --shots // --min-batches ({args.shots // args.min_batches}, so every pass has at least {args.min_batches} batches for the cap warm-up), and B is the largest size before the first out-of-memory error or the first decode that ends past --row-timeout (timeout in probe when that happens at 64)"
           if args.batch_size == "auto" else str(args.batch_size))
        + f"; batch i drawn with seed {SEED} + i, same shots for every config and pass at a given d, p and dtype",
        f"- Batching: each pass decodes every batch as main.py's batch loop does. With the cap on ({LADDER[-1]} only), BP observes each batch's iteration histogram until the KL test settles (warm-up), then stops each batch once the chosen percentile of its shots has converged; the shots BP did not converge in a capped batch are deferred, their results dropped, and decoded again uncapped, a batch at a time once B are queued and the rest after the last batch",
        f"- Time: BP + OSD decode only, summed over the decode calls of a pass (each call between torch.cuda.synchronize calls; shot generation not timed), warmup passes then {args.repeats} timed passes, median. Warmup passes repeat until the cap is chosen (one pass when it is off), at most {args.max_warmup_passes} passes ({args.max_warmup_passes} x ceil(shots / B) batches), so the timed passes run with the cap already chosen; a row whose cap has not settled by then is timed with the cap off. Before the first row of each d, p, dtype and path, one untimed build with all groups on and one decode of a 64-shot batch load the CUDA context, extensions and workspaces, so the first row is not charged for them. bp_ms and osd_ms split it by stage: each is the BP or OSD call alone between torch.cuda.synchronize calls, summed over a pass and median over the same timed passes as time; the syncs between stages and the input clone are inside time_ms only, so bp_ms + osd_ms is a little under time_ms. A pass that reaches --row-timeout stops at the next batch boundary and ends the row: its time, bp_ms and osd_ms are extrapolated (measured time x shots / shots done) and its LER is over the shots done. speedup_step = time of the previous ladder step / this time; speedup_total = all_off time / this time; both empty for all_off and when the reference row has no time (did not run at this dtype, did not fit, or timed out before any time). The reference rows a requested config needs (the previous step and all_off) are added when --configs leaves them out and run first, in ladder order",
        "- Peak MiB: per-call working memory above the resident state after warmup: torch.cuda.max_memory_allocated during a decode call minus the allocation just before it, max over the calls of the timed passes",
        "- Resident MiB: torch.cuda.memory_allocated after the warmup pass minus memory_allocated before the decoders are built (the bundle and the H passed to decode already exist then), so it covers decoder buffers, reuse_buffers state and warmup caches",
        "- H MiB: device memory of the H tensor the decoders get from the bundle (dense tensor bytes, or sparse indices plus values)",
        "- sm_util, mem_util: sm_util is the NVML device-wide percent of time a kernel was running (torch.cuda.utilization), mem_util the NVML device-wide percent of time the memory controller was busy (torch.cuda.memory_usage); both are sampled every 0.1 s during the timed passes, including the untimed shot generation and logical check inside each pass, and averaged (- when no sample landed). NVML averages over its own window of 1/6 s to 1 s, so rows whose timed pass is under about 1 s report a smeared value, and another process on the same GPU raises both",
        "- LER: final e_v through the DEM observable matrix compared with the sampled observable flips (stim logical check), failures over all shots of a pass (a deferred shot counts once, from its uncapped decode; over the shots done for a pass stopped at the timeout), mean over the timed passes; a range is shown when passes differ. Rows with gather_opt false (all_off through +mapping_opt) use atomic adds (vn_gather false on the cuda path, c2v_gather false on the pytorch path), so they are nondeterministic and their LER can vary run to run",
        "- Cap: the percentile BP chose (stop each batch once that percent of its shots converged) and the warm-up batches the KL test used (the log line warm-up done), read after the last pass; - when the cap is off; cap not settled after N batches when the warmup passes ran out and the row was timed with the cap off. The cap is a percentile, so the stop iteration varies per batch",
        f"- Status: ok; timeout, N of S shots (a pass stopped at --row-timeout {args.row_timeout:g} s after N of S shots, time extrapolated); timeout after N timed passes (row past the timeout between passes); timeout in probe (the 64-shot probe decode ended past the timeout); does not fit (out of memory at 64); or error",
        f"- Command: `{cmd}`",
        "",
        "| d | p | H (MxN) | path | dtype | batch | config | decoders | compiled | time ms (median) | bp_ms | osd_ms | speedup_step | speedup_total | peak MiB | resident MiB | H MiB | sm_util | mem_util | LER | cap | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(header) + "\n")
    csv_path = Path(args.out).with_suffix(".csv")
    csv_file = open(csv_path, "w", newline="")
    csv_out = csv.writer(csv_file)
    csv_out.writerow(
        "d,M,N,p,path,dtype,batch,config,decoders,compiled,time_ms,bp_ms,osd_ms,speedup_step,speedup_total,peak_mib,"
        "resident_mib,h_mib,sm_util,mem_util,ler_mean,ler_min,ler_max,cap,status".split(",")
    )
    csv_file.flush()

    run = {LADDER[0]} | set(args.configs)
    run |= {LADDER[LADDER.index(c) - 1] for c in args.configs if c != LADDER[0]}
    configs = [c for c in LADDER if c in run]  # ladder order: references first

    for d in args.distances:
        for p in args.ps:
            for dtype in args.dtypes:
                it = make_case(d, p, dtype)
                M, N = it.matrix_bundle.select("hx")[0]
                for path in args.paths:
                    _, decs, H, _ = build(it, path, dtype, LADDER[-1])
                    decode(decs, *next(batches(it, dtype, 64, 64))[:2], H)
                    del decs, H
                    gc.collect()
                    torch.cuda.empty_cache()
                    ref_ms = {}  # config -> ms at this (d, p, dtype, path)
                    for config in configs:
                        if config not in ENDS and dtype != ABLATE_DTYPE:
                            continue
                        (
                            bs,
                            names,
                            compiled,
                            ms,
                            bp_ms,
                            osd_ms,
                            mib,
                            res_mib,
                            h_mib,
                            sm_util,
                            mem_util,
                            lers,
                            cap,
                            status,
                        ) = run_row(it, path, dtype, config, args.batch_size, args)
                        ref_ms[config] = ms

                        def speedup(ref):
                            t = ref_ms.get(ref) if config != LADDER[0] else None
                            return t / ms if t and ms else None

                        step = speedup(LADDER[LADDER.index(config) - 1])
                        total = speedup(LADDER[0])
                        line = (
                            f"| {d} | {p:g} | {M}x{N} | {path} | {dtype} | {fmt(bs, 'd')} | {config} | {names} | {compiled} | "
                            f"{fmt(ms, '.2f')} | {fmt(bp_ms, '.2f')} | {fmt(osd_ms, '.2f')} | {fmt(step, '.2f')} | {fmt(total, '.2f')} | {fmt(mib, '.1f')} | "
                            f"{fmt(res_mib, '.1f')} | {fmt(h_mib, '.1f')} | {fmt(sm_util, '.1f')} | {fmt(mem_util, '.1f')} | {fmt_ler(lers)} | {cap} | {status} |"
                        )
                        with open(args.out, "a") as f:
                            f.write(line + "\n")
                        csv_out.writerow(
                            [
                                d,
                                M,
                                N,
                                p,
                                path,
                                dtype,
                                bs,
                                config,
                                names,
                                compiled,
                                ms,
                                bp_ms,
                                osd_ms,
                                step,
                                total,
                                mib,
                                res_mib,
                                h_mib,
                                sm_util,
                                mem_util,
                                statistics.mean(lers) if lers else None,
                                min(lers, default=None),
                                max(lers, default=None),
                                cap,
                                status,
                            ]
                        )
                        csv_file.flush()
                        print(line, flush=True)
                del it
                gc.collect()
                torch.cuda.empty_cache()
    csv_file.close()
    print(f"wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
