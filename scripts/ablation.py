"""Ablation of the six optimization groups on BP + OSD-0.

Runs bp_norm_min_sum followed by osd_0 on stim surface_code:rotated_memory_x
circuits (rounds = d, every circuit noise key at p) with all groups on and with
each group key set false in turn, on two GPU paths: the CUDA kernels
(bp_norm_min_sum_cuda, osd_0_cuda) and the PyTorch decoders (force_pytorch, BP
compile on by default). Group keys go into each stage's config block and into
the top-level matrix config, where memory_opt false also turns sparse_h off
(dense H). Each pass decodes all shots in batches the way main.py's batch loop
does: with the rebatch cap (rebatch_opt) on, BP warms the cap up over the first
batches, then stops each batch at the chosen percentile, and the shots BP has
not converged are deferred and decoded again uncapped in later batches. Warmup
passes repeat until the cap is chosen, at most --max-warmup-passes; a row whose
cap has not settled by then is timed with the cap off. Before the first row of
each (d, dtype, path) one untimed build and batch decode loads the CUDA context,
extensions and workspaces, so no row is charged for them. For each (d, path,
dtype, config) it records the median decode wall time of a pass over the timed
passes after the warmup passes, the decode's peak GPU memory, the
resident decoder memory after warmup, the device memory of H, the logical error
rate over all shots and the chosen cap, and appends each row to one markdown
table as soon as it finishes. The reference rows a requested config needs for
its time ratio (all_on, and on the pytorch path all_on, compile=false) are
added to the run set and run before the rows that use them.

Usage:
    conda run -n syndrilla python scripts/ablation.py [--distances 15 27] ...
"""
import argparse
import datetime
import gc
import os
import statistics
import sys
import time

import numpy as np
import torch
from loguru import logger

from syndrilla.decoder import create_decoder
from syndrilla.interface.stim.stim import create as stim_iface
from syndrilla.matrix import load_matrices
from syndrilla.utils import parse_device_dtype

GROUPS = (
    "pruning_opt",
    "fusion_opt",
    "mapping_opt",
    "gather_opt",
    "memory_opt",
    "rebatch_opt",
)
CONFIGS = {
    "all_on": {},
    "all_on, compile=false": {"compile": False},
    **{f"{g}=false": {g: False} for g in GROUPS},
    "memory_opt=false, sparse_h=true": {"memory_opt": False, "sparse_h": True},
}
PYTORCH_ONLY = ("all_on, compile=false",)
EAGER_REF = ("mapping_opt=false", "gather_opt=false")
NOISE = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)
DEV = {"device_type": "cuda", "device_idx": 0}
MAX_ITER = 181
SEED = 0


def make_case(d, p, dtype):
    """Stim interface (no decoders) for one distance and dtype."""
    return stim_iface(
        {"code": "surface_code:rotated_memory_x", "distance": d},
        error_cfg={k: p for k in NOISE},
        syndrome_cfg={"rounds": d},
        decoding_cfg={"check_type": "hx", "dtype": dtype, "device": DEV},
    )


def batches(it, dtype, shots, batch_size):
    """Yield (synd, llr0, sampled observable flips) for the shots in batches.
    Batch i is drawn with seed SEED + i, so every pass and every config at a
    given d and dtype decodes the same shots."""
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
    groups = {k: v for k, v in keys.items() if k in GROUPS}
    top = {k: v for k, v in keys.items() if k in GROUPS or k == "sparse_h"}
    bp = {"max_iter": MAX_ITER, **groups}
    if "compile" in keys:
        bp["compile"] = keys["compile"]
    bundle = load_matrices(
        {**it.matrix_cfg, **top}, *parse_device_dtype({"device": DEV, "dtype": dtype})
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
    """(io after the last stage, BP's converge flags)."""
    io = {"synd": synd.clone(), "llr0": llr0.clone(), "H_matrix": H}
    with torch.no_grad():
        for i, dec in enumerate(decs):
            io = dec(io)
            if i == 0:
                bp_converge = io["converge"].clone()
    return io, bp_converge


def run_pass(it, decs, H, l_matrix, shots, batch_size, dtype):
    """(decode seconds, logical failures, peak bytes) of one pass over all shots,
    run as main.py's batch loop runs them. When BP stopped a batch at the cap,
    the shots BP did not converge are deferred and their results dropped; the
    deferred shots are decoded again uncapped, a batch at a time once the queue
    holds batch_size shots, and the rest after the last batch. Time is the sum
    of the decode calls, each between torch.cuda.synchronize calls; peak is
    max_memory_allocated during a call minus the allocation just before it, max
    over the calls."""
    inner = getattr(decs[0], "decoder", decs[0])
    capped = getattr(inner, "cap", None) is not None
    secs, fails, peak = 0.0, 0, 0
    queue = None  # deferred (synd, llr0, obs)

    def run(synd, llr0, obs, bypass):
        nonlocal secs, fails, peak, queue
        if capped:
            inner.cap_bypass = bypass
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        io, bp_converge = decode(decs, synd, llr0, H)
        torch.cuda.synchronize()
        secs += time.perf_counter() - t
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
        fail = it.logical_check.check(io["e_v"], obs, l_matrix, io["converge"])
        if capped and inner.cap_active_last:
            keep = bp_converge.flatten() > 0
            late = (synd[~keep], llr0[~keep], obs[~keep])
            queue = late if queue is None else tuple(map(torch.cat, zip(queue, late)))
            fail = fail[keep.to(fail.device)]
        fails += int(fail.sum())

    def flush(n):
        nonlocal queue
        head = tuple(t[:n] for t in queue)
        queue = tuple(t[n:] for t in queue)
        run(*head, bypass=True)

    for synd, llr0, obs in batches(it, dtype, shots, batch_size):
        run(synd, llr0, obs, bypass=False)
        while queue is not None and len(queue[0]) >= batch_size:
            flush(batch_size)
    while queue is not None and len(queue[0]):
        flush(batch_size)
    return secs, fails, peak


def tensor_mib(H):
    """Device memory of H: the dense tensor, or the indices and values of a sparse one."""
    if H.is_sparse:
        return (H._indices().nbytes + H._values().nbytes) / 2**20
    return H.nbytes / 2**20


def measure(it, bundle, decs, H, dtype, args, deadline, base):
    """(median ms per pass, peak MiB above the pre-call allocation, resident MiB
    after warmup above base, LERs per timed pass, timed passes done, timed out,
    cap note) of BP + OSD. Warmup passes repeat until the cap is chosen (one
    pass when the cap is off), at most args.max_warmup_passes; if the cap has
    not settled by then, the cap is removed so the timed passes run uncapped,
    and the cap note says so (else None). Stops after any warmup or timed pass
    that ends past deadline."""
    l_matrix = bundle.get_l_matrix("hx", 1)
    shape = (args.shots, args.batch_size, dtype)
    inner = getattr(decs[0], "decoder", decs[0])
    cap = getattr(inner, "cap", None)
    note = None
    for n in range(1, args.max_warmup_passes + 1):  # compile, caches, cap warm-up
        run_pass(it, decs, H, l_matrix, *shape)
        if cap is None or cap.done or time.monotonic() > deadline:
            break
        if n == args.max_warmup_passes:
            note = f"cap not settled after {len(cap.hists)} batches"
            inner.cap = None
    torch.cuda.synchronize()
    resident = (torch.cuda.memory_allocated() - base) / 2**20
    times, peaks, lers = [], [], []
    for _ in range(args.repeats):
        if time.monotonic() > deadline:
            break
        secs, fails, peak = run_pass(it, decs, H, l_matrix, *shape)
        times.append(secs)
        peaks.append(peak)
        lers.append(fails / args.shots)
    ms = 1e3 * statistics.median(times) if times else None
    mib = max(peaks) / 2**20 if peaks else None
    return ms, mib, resident, lers, len(times), len(times) < args.repeats, note


def describe_cap(decs):
    """The cap BP chose: percentile and warm-up batches, or "-" with no cap."""
    cap = getattr(getattr(decs[0], "decoder", decs[0]), "cap", None)
    if cap is None:
        return "-"
    if not cap.done:
        return f"not settled after {len(cap.hists)} batches"
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


def run_row(it, path, dtype, config, args):
    """Row fields (decoders, compiled, ms, peak MiB, resident MiB, H MiB, LER, cap,
    status)."""
    deadline = time.monotonic() + args.row_timeout
    names, compiled, ms, mib, res_mib, h_mib, lers = "-", "-", None, None, None, None, []
    cap = "-"
    bundle = decs = H = None
    try:
        bundle, decs, H, base = build(it, path, dtype, config)
        names, compiled = describe(decs, path)
        h_mib = tensor_mib(H)
        ms, mib, res_mib, lers, n, timed_out, note = measure(
            it, bundle, decs, H, dtype, args, deadline, base
        )
        cap = note or describe_cap(decs)
        status = f"timeout after {n} timed passes" if timed_out else "ok"
    except (torch.cuda.OutOfMemoryError, MemoryError):
        status = "does not fit"
    except Exception as exc:
        status = f"error: {type(exc).__name__}"
        logger.exception(f"{path} {dtype} {config} failed")
    del bundle, decs, H
    gc.collect()
    torch.cuda.empty_cache()
    return names, compiled, ms, mib, res_mib, h_mib, lers, cap, status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--distances", type=int, nargs="+", default=[15, 27])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--p", type=float, default=3e-3)
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
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--row-timeout",
        type=float,
        default=1800,
        help="seconds per row; checked after each warmup and timed pass",
    )
    ap.add_argument(
        "--max-warmup-passes",
        type=int,
        default=4,
        help="warmup passes before a row is timed with the cap off if it has not settled",
    )
    ap.add_argument(
        "--out", default=f"reports/{datetime.date.today():%Y-%m-%d}-ablation-results.md"
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(  # warnings, plus the cap's warm-up done line
        sys.stderr,
        filter=lambda r: r["level"].no >= 30 or "warm-up done" in r["message"],
    )

    cmd = "conda run -n syndrilla python " + " ".join(sys.argv)
    header = [
        "# Optimization group ablation, BP + OSD-0",
        "",
        f"- Hardware: {torch.cuda.get_device_name(0)}; torch {torch.__version__}, CUDA {torch.version.cuda}",
        f"- Circuits: stim surface_code:rotated_memory_x, rounds = d, every circuit noise key at p = {args.p:g}; check_type hx",
        f"- Decoders: bp_norm_min_sum (max_iter {MAX_ITER}) then osd_0. Path cuda = CUDA kernels; path pytorch = force_pytorch on the GPU, BP compile left at its default (on), so a group that forces eager shows compiled = no",
        "- Configs: all_on, then one group key false in each stage's config block and in the top-level matrix config. memory_opt=false turns sparse_h off (dense int64 H on the device); memory_opt=false, sparse_h=true keeps the sparse H with the other memory_opt members off. rebatch_opt=false turns the BP iteration cap off. all_on, compile=false (pytorch only) sets compile false on the BP stage",
        f"- Shots: {args.shots} shots in batches of B = {args.batch_size}; batch i drawn with seed {SEED} + i, same shots for every config and pass at a given d and dtype",
        "- Batching: each pass decodes every batch as main.py's batch loop does. With the cap on (every config except rebatch_opt=false), BP observes each batch's iteration histogram until the KL test settles (warm-up), then stops each batch once the chosen percentile of its shots has converged; the shots BP did not converge in a capped batch are deferred, their results dropped, and decoded again uncapped, a batch at a time once B are queued and the rest after the last batch",
        f"- Time: BP + OSD decode only, summed over the decode calls of a pass (each call between torch.cuda.synchronize calls; shot generation not timed), warmup passes then {args.repeats} timed passes, median. Warmup passes repeat until the cap is chosen (one pass when it is off), at most {args.max_warmup_passes} passes ({args.max_warmup_passes} x {-(-args.shots // args.batch_size)} batches), so the timed passes run with the cap already chosen; a row whose cap has not settled by then is timed with the cap off. Before the first row of each d, dtype and path, one untimed build and one batch decode with all_on load the CUDA context, extensions and workspaces, so the first row is not charged for them. x vs ref = time / all_on time; the pytorch mapping_opt=false and gather_opt=false rows (eager BP) use the all_on, compile=false row instead. These reference rows run first at each d, dtype and path, and are added when --configs leaves them out",
        "- Peak MiB: per-call working memory above the resident state after warmup: torch.cuda.max_memory_allocated during a decode call minus the allocation just before it, max over the calls of the timed passes",
        "- Resident MiB: torch.cuda.memory_allocated after the warmup pass minus memory_allocated before the decoders are built (the bundle and the H passed to decode already exist then), so it covers decoder buffers, reuse_buffers state and warmup caches",
        "- H MiB: device memory of the H tensor the decoders get from the bundle (dense tensor bytes, or sparse indices plus values)",
        "- LER: final e_v through the DEM observable matrix compared with the sampled observable flips (stim logical check), failures over all shots of a pass (a deferred shot counts once, from its uncapped decode), mean over the timed passes; a range is shown when passes differ. The gather_opt=false rows use atomic adds (vn_gather false on the cuda path, c2v_gather false on the pytorch path), so they are nondeterministic and their LER can vary run to run",
        "- Cap: the percentile BP chose (stop each batch once that percent of its shots converged) and the warm-up batches the KL test used (the log line warm-up done), read after the last pass; - when the cap is off; cap not settled after N batches when the warmup passes ran out and the row was timed with the cap off. The cap is a percentile, so the stop iteration varies per batch",
        f"- Status: ok, timeout (row past --row-timeout {args.row_timeout:g} s, checked after each warmup and timed pass), does not fit (out of memory), or error",
        f"- Command: `{cmd}`",
        "",
        "| d | H (MxN) | path | dtype | config | decoders | compiled | time ms (median) | x vs ref | peak MiB | resident MiB | H MiB | LER | cap | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(header) + "\n")

    run = set(args.configs) | {"all_on"}
    if "pytorch" in args.paths and run & set(EAGER_REF):
        run.add("all_on, compile=false")
    configs = [c for c in CONFIGS if c in run]  # CONFIGS lists the references first

    for d in args.distances:
        for dtype in args.dtypes:
            it = make_case(d, args.p, dtype)
            M, N = it.matrix_bundle.select("hx")[0]
            for path in args.paths:
                _, decs, H, _ = build(it, path, dtype, "all_on")
                decode(decs, *next(batches(it, dtype, args.batch_size, args.batch_size))[:2], H)
                del decs, H
                gc.collect()
                torch.cuda.empty_cache()
                ref_ms = {}
                for config in configs:
                    if config in PYTORCH_ONLY and path != "pytorch":
                        continue
                    names, compiled, ms, mib, res_mib, h_mib, lers, cap, status = (
                        run_row(it, path, dtype, config, args)
                    )
                    ref_ms[config] = ms
                    ref_key = (
                        "all_on, compile=false"
                        if path == "pytorch" and config in EAGER_REF
                        else "all_on"
                    )
                    ref = ref_ms.get(ref_key)
                    ratio = ms / ref if ms is not None and ref else None
                    line = (
                        f"| {d} | {M}x{N} | {path} | {dtype} | {config} | {names} | {compiled} | "
                        f"{fmt(ms, '.2f')} | {fmt(ratio, '.2f')} | {fmt(mib, '.1f')} | "
                        f"{fmt(res_mib, '.1f')} | {fmt(h_mib, '.1f')} | {fmt_ler(lers)} | {cap} | {status} |"
                    )
                    with open(args.out, "a") as f:
                        f.write(line + "\n")
                    print(line, flush=True)
            del it
            gc.collect()
            torch.cuda.empty_cache()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
