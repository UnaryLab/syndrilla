"""syndrilla-parallel: run syndrilla as several shot-parallel workers, one or more per GPU, and pool the results.

Subcommands: run (one launch, described below), sweep-gen and sweep (point
folders, see syndrilla.sweep). A call without a subcommand prints them.

Each worker is a separate `python -m syndrilla.main` process whose
CUDA_VISIBLE_DEVICES holds one physical GPU id, taken from this process's
CUDA_VISIBLE_DEVICES when it is set, so the decoding yaml should use
device_type cuda with device_idx 0. Each worker writes to
<run_dir>/w<index>/ (result yaml, main-*.log and syndrilla.log). Without
--seed the workers run unseeded and draw independent shots; with --seed b,
worker k (in launch order over all workers) gets --seed=b+k. A decoding yaml
with device_type cpu runs as one worker, in <run_dir>/w0/, with no
CUDA_VISIBLE_DEVICES set; --gpus and --workers-per-gpu are then ignored.

Stop: either or both of -te and -tb may be given (-te 1000 when neither is).
With one of them, each worker decodes its share of that target: the shares
differ by at most one and add up to the target, and the target must be at
least the number of workers. A lone -tb ends exactly at it; a lone -te ends at
or above it, each worker stopping at its own share.

With both, the targets are pooled: every worker gets the full -tb, so its own
stop rule is only a safety net. This process reads every worker's result yaml
each --poll-interval seconds (a worker saves it every --save-interval
batches) and, once pooled fails reach -te or pooled batches reach -tb,
whichever comes first, sends SIGTERM to all workers; each then finishes its
current batch and saves. The pooled totals end at or above the target, since
each worker can run past it by about --save-interval batches (counts are read
from the saved yamls), plus the batches it decodes during one poll interval,
plus the batch in flight and the drain of its deferred queue at SIGTERM. A
worker that reaches its own target first exits and the others continue.

While waiting, one pooled progress line is printed per poll in which a yaml
changed. Workers run in their own session, so a Ctrl-C in the terminal reaches
only this process, which then sends SIGTERM to every worker, waits for each to
save, and exits with status 1 without merging.

Probe: before any worker dir is created, one batch (-tb=1) at the given -bs is
decoded with the same flags on the first worker's GPU in a scratch dir under
<run_dir> that is removed afterwards, so a bad yaml, a decoder not on cuda, or
an out-of-memory error of one worker at that -bs fails in seconds, and the
torch extension cache is built once before the workers start. It does not catch
memory pressure from several workers sharing a GPU (--workers-per-gpu > 1).
A failed probe removes the scratch dir and any part of <run_dir> it created, so
the run dir is left as it was; if <run_dir> existed before, the scratch dir is
kept and the path of its probe.log is printed. --no-probe skips it.

Resume: a worker dir that already holds a result yaml is resumed: that yaml
is passed to the worker as -ckpt, so a rerun of the same command continues a
stopped launch. A resume needs the same -te/-tb and ordered worker/device list;
otherwise the launcher exits before starting any worker. The targets and workers of a
launch are kept in <run_dir>/launch.yaml, written before the workers start. A failed or
interrupted launch leaves no merged or pooled yaml behind (the worker yamls
stay, and a rerun resumes and rewrites them). A run dir that holds a result
yaml but no launch.yaml and no worker dir (a single syndrilla run) is refused.

If a worker fails, the others are stopped (SIGTERM, so each saves its result
yaml). When all workers finish, the launcher writes:
  <run_dir>/merged_result.yaml: pooled counts, pooled logical error rate
    (total fails / total shots) with a 95 percent Wilson interval, throughput,
    the stop rule that fired with the pooled values it saw, and one row per
    worker;
  <run_dir>/result_phy_err_<rate>.yaml: the pooled result in the single-run
    result yaml schema (see pool_results for what is pooled).

Every flag this script does not define (-d -e -c -s -m -i -bs -l) is passed to
each worker unchanged. -ckpt, -t, -tr, -tckpt and their long forms are refused.

Stopping a worker while it builds a CUDA kernel can leave a file named lock in
that kernel's build dir under ~/.cache/torch_extensions/ (or
$TORCH_EXTENSIONS_DIR), and later runs then wait on it; delete that lock file
to recover.

Usage:
    syndrilla-parallel run --gpus 0 1 -r=runs/multi -te=1000 -d=... -i=... -e=... -s=... -bs=10000
"""

import argparse
import glob
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml

# flags of syndrilla.main that the launcher refuses, with the reason
REFUSED = [
    (("-ckpt", "--checkpoint_yaml"), "one checkpoint cannot be split over workers"),
    (("-t", "--train"), "training runs as a single process"),
    (("-tr", "--training_yaml"), "training runs as a single process"),
    (("-tckpt", "--train_checkpoint"), "training runs as a single process"),
]

POOLED_NOTE = (
    "# pooled over {n} workers by syndrilla-parallel: sample count, iteration count,\n"
    "# batch count, target error reached and total time are summed; target error and\n"
    "# target batch are the launcher's pooled targets; rates and averages are weighted\n"
    "# by sample count; iteration distribution is recomputed from the summed iteration\n"
    "# count; average time per batch is total time over batch count; seed is null and\n"
    "# rebatch_opt is left out (not pooled).\n"
)


def split_target(total, n):
    """Split total into n integer shares that differ by at most one and sum to total."""
    base, rem = divmod(total, n)
    return [base + (k < rem) for k in range(n)]


def physical_ids(indices, visible):
    """Map device indices to the ids to put in each child's CUDA_VISIBLE_DEVICES.

    visible is the parent's CUDA_VISIBLE_DEVICES value or None; when set, index k
    means its k-th entry, and the list ends at the first empty entry, as in CUDA.
    """
    if visible is None:
        return [str(k) for k in indices]
    ids = []
    for x in visible.split(","):
        if not x.strip():
            break
        ids.append(x.strip())
    for k in indices:
        if not 0 <= k < len(ids):
            sys.exit(f"--gpus index {k} is outside CUDA_VISIBLE_DEVICES={visible}")
    return [ids[k] for k in indices]


def wilson(fails, shots, z=1.959963984540054):
    """95 percent Wilson score interval for fails / shots."""
    if shots == 0:
        return [0.0, 1.0]
    p = fails / shots
    den = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / den
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def result_file(run_dir):
    """Path of the one result_phy_err_*.yaml in run_dir, or None if there is none."""
    files = glob.glob(os.path.join(run_dir, "result_phy_err_*.yaml"))
    if len(files) > 1:
        sys.exit(f"expected one result_phy_err_*.yaml in {run_dir}, found {files}")
    return files[0] if files else None


def decoder_keys(res):
    keys = [k for k in res if k.startswith("decoder_") and k != "decoder_full"]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def summary(res):
    """Shots, fails, batches, iteration sum and decode time of one result yaml."""
    last = res[decoder_keys(res)[-1]]
    full = res["decoder_full"]
    shots = int(last["sample count"])
    return {
        "shots": shots,
        "fails": int(full["target error reached"]),
        "batches": int(full["batch count"]),
        "iteration sum": float(last["average iteration"]) * shots,
        "decode time (s)": float(full["total time (s)"]),
        "physical error rate": full["physical error rate"],
    }


def percentiles(hist):
    """101 percentile iteration counts from a histogram, as save_metric writes them."""
    total = sum(hist)
    out, acc, i = [], 0.0, 0
    cdf = []
    for h in hist:
        acc += h
        cdf.append(acc / total if total else 0.0)
    for q in range(101):
        while i < len(cdf) - 1 and cdf[i] < q / 100:
            i += 1
        out.append(i + 1)
    return out


def pool_results(results):
    """Pool worker result yamls into one yaml in the single-run schema."""
    first = results[0]
    batches = sum(int(r["decoder_full"]["batch count"]) for r in results)
    pooled = {}
    for key in decoder_keys(first):
        entries = [r[key] for r in results]
        w = [float(e["sample count"]) for e in entries]
        total_w = sum(w)

        def avg(values):
            return (
                sum(float(v) * x for v, x in zip(values, w)) / total_w
                if total_w
                else 0.0
            )

        hist = [sum(col) for col in zip(*(e["iteration count"] for e in entries))]
        out = {}
        for name, value in entries[0].items():
            if name == "algorithm":
                out[name] = value
            elif name == "sample count":
                out[name] = int(total_w)
            elif name == "iteration count":
                out[name] = hist
            elif name == "iteration distribution":
                out[name] = percentiles(hist)
            elif name == "rebatch_opt":
                continue
            elif name == "total time (s)":
                out[name] = sum(float(e[name]) for e in entries)
            elif name == "average time per batch (s)":
                out[name] = sum(float(e["total time (s)"]) for e in entries) / batches
            elif isinstance(value, dict):  # hx / hz rates
                out[name] = {k: avg([e[name][k] for e in entries]) for k in value}
            else:
                out[name] = avg([e[name] for e in entries])
        pooled[key] = out

    full = dict(first["decoder_full"])
    fulls = [r["decoder_full"] for r in results]
    for name in ("batch count", "target error reached"):
        full[name] = sum(int(f[name]) for f in fulls)
    full["total time (s)"] = sum(float(f["total time (s)"]) for f in fulls)
    full["seed"] = None
    last = pooled[decoder_keys(first)[-1]]
    for check in ("hx", "hz"):
        if check in full:
            full[check] = {"logical error rate": last[check]["logical error rate"]}
    pooled["decoder_full"] = full
    return pooled


def tail(path, n=20):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def stop(procs):
    """Terminate every worker still running and wait for it to exit."""
    for w in procs:
        if w["proc"].poll() is None:
            w["proc"].terminate()
    for w in procs:
        w["proc"].wait()


# flags of syndrilla.main that sweep takes from each point folder
POINT_FLAGS = [
    ("-d", "--decoding_yaml"),
    ("-i", "--interface_yaml"),
    ("-e", "--error_yaml"),
    ("-s", "--syndrome_yaml"),
    ("-c", "--logical_yaml"),
    ("-m", "--matrix_yaml"),
]


def add_run_flags(parser):
    """The launcher flags shared by run and sweep."""
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU indices as torch numbers them, that is positions in this process's "
        "CUDA_VISIBLE_DEVICES when it is set; default every device torch.cuda.device_count() reports.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Worker processes per GPU, default 1.",
    )
    parser.add_argument(
        "-te",
        "--target_error",
        type=int,
        default=None,
        help="Pooled number of logical errors to stop at, split over workers when given alone; "
        "default 1000 unless -tb is given.",
    )
    parser.add_argument(
        "-tb",
        "--target_batch",
        type=int,
        default=None,
        help="Pooled number of batches to stop at, split over workers when given alone; "
        "with -te, the run stops at whichever is reached first.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Batches between a worker's result saves, at least 100, passed to every worker when given. "
        "Default the worker's own, 100.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between reads of the worker result yamls. Default 60.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; worker k in launch order gets --seed=<seed+k>. Default unseeded.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the one-batch probe decode run before the workers start.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands, then exit without launching: run prints the probe and each worker's "
        "environment and command; sweep prints each point's label and each worker's command with its physical device.",
    )
    for flags, _ in REFUSED:
        parser.add_argument(
            *flags, nargs="?", const=True, default=None, help=argparse.SUPPRESS
        )


def parse_args():
    """Launcher args and the flags passed through to every worker."""
    parser = argparse.ArgumentParser(
        prog="syndrilla-parallel",
        description="Run syndrilla as shot-parallel workers over GPUs and pool the results.",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run",
        help="Run one launch of workers and pool the results.",
        description="Run syndrilla as shot-parallel workers over GPUs and pool the results. "
        "Unknown flags are passed to every worker.",
        allow_abbrev=False,
    )
    run.add_argument(
        "-r",
        "--run_dir",
        type=str,
        required=True,
        help="Parent run directory; worker i writes to <run_dir>/w<i>/, "
        "and a worker dir that already holds a result yaml is resumed.",
    )
    add_run_flags(run)
    gen = sub.add_parser(
        "sweep-gen",
        help="Write one point folder per combination of a sweeping configs yaml.",
        description="Write one point folder per combination of a sweeping configs yaml "
        "directly under -r, named <code>_<check_type>_<p>_<d>_<dtype>. Run from the repo root.",
        allow_abbrev=False,
    )
    gen.add_argument(
        "-c",
        "--config",
        required=True,
        help="Sweeping configs yaml, e.g. zoo/script/sweeping_configs.yaml.",
    )
    gen.add_argument("-r", "--run_dir", required=True, help="Sweep directory.")
    gen.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the config yamls of point folders that already hold a result yaml; the results are kept.",
    )
    sw = sub.add_parser(
        "sweep",
        help="Run every point folder of a sweep dir as one launch each.",
        description="Run every point folder under -r, in sorted order, like run with -r set to "
        "that folder and -d -e -c -s -m taken from its yamls. A point starts when w "
        "(--workers-per-point) slots are free (--workers-per-gpu slots per GPU). "
        "With --seed b, point i, worker j gets --seed=<b + i*w + j>. "
        "Writes <run_dir>/sweep_results.csv. Unknown flags are passed to every worker.",
        allow_abbrev=False,
    )
    sw.add_argument("-r", "--run_dir", type=str, required=True, help="Sweep directory.")
    sw.add_argument(
        "--workers-per-point",
        type=int,
        default=1,
        help="Workers per point, sharing GPUs when capacity allows; default 1.",
    )
    sw.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep at the first failed point; by default the sweep continues.",
    )
    add_run_flags(sw)
    for flags in POINT_FLAGS:
        sw.add_argument(*flags, default=None, help=argparse.SUPPRESS)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)
    args, passthrough = parser.parse_known_args()
    if args.command == "sweep-gen":
        if passthrough:
            gen.error(f"unrecognized arguments: {' '.join(passthrough)}")
        return args, passthrough
    if args.command == "sweep":
        for flags in POINT_FLAGS:
            if getattr(args, flags[1][2:]) is not None:
                sw.error(f"{' / '.join(flags)} comes from each point folder in sweep")
    for flags, why in REFUSED:
        if getattr(args, flags[1][2:]) is not None:
            sys.exit(f"{' / '.join(flags)} is not supported by this launcher: {why}")
    if args.save_interval is not None and args.save_interval < 100:
        parser.error(f"--save-interval must be at least 100, got {args.save_interval}")
    if args.target_error is None and args.target_batch is None:
        args.target_error = 1000
    # With both targets, the stop is pooled and every worker gets the full -tb,
    # which bounds each worker's work even if this process dies. With one target,
    # each worker gets its share of it.
    args.pooled = args.target_error is not None and args.target_batch is not None
    if args.target_batch is not None:
        args.worker_target = ("-tb", "target batch", args.target_batch)
    else:
        args.worker_target = ("-te", "target error", args.target_error)
    return args, passthrough


def gpu_list(args):
    """--gpus, or every device torch reports."""
    if args.gpus is not None:
        return args.gpus
    import torch

    return list(range(torch.cuda.device_count()))


def decoding_device(passthrough):
    """device_type of the -d decoding yaml in passthrough, cuda when unset."""
    dec = argparse.ArgumentParser(add_help=False)
    dec.add_argument("-d", "--decoding_yaml")
    path = dec.parse_known_args(passthrough)[0].decoding_yaml
    if path is None:
        return "cuda"
    if not os.path.isfile(path):
        sys.exit(f"decoding yaml {path} not found")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ((cfg or {}).get("decoding", {}).get("device") or {}).get(
        "device_type", "cuda"
    )


def worker_shares(args, count):
    flag, _, target = args.worker_target
    if count < 1:
        sys.exit("no workers to run")
    if args.pooled:
        return [target] * count
    if target < count:
        sys.exit(f"{flag} {target} is smaller than the {count} workers")
    return split_target(target, count)


def preflight_workers(args, count):
    """Validate saved layout, targets and shares without assigning any devices.

    Return the recorded ordered worker/device list, or None for a fresh launch.
    """
    shares = worker_shares(args, count)
    planned = {f"w{i}" for i in range(count)}
    legacy = [d for pattern in ("gpu*_w*", "cpu_w*")
              for d in glob.glob(os.path.join(args.run_dir, pattern)) if os.path.isdir(d)]
    if legacy:
        sys.exit(f"{args.run_dir} holds an old worker layout (gpuG_wJ/cpu_w0); "
                 "flat-worker launches cannot use it. Use a new -r directory.")
    present = {os.path.basename(d) for d in glob.glob(os.path.join(args.run_dir, "w*"))
               if os.path.isdir(d) and re.fullmatch(r"w\d+", os.path.basename(d))}
    launch = os.path.join(args.run_dir, "launch.yaml")
    single = result_file(args.run_dir) if os.path.isdir(args.run_dir) else None
    if single and not present and not os.path.isfile(launch):
        sys.exit(
            f"{args.run_dir} holds {os.path.basename(single)} from a single syndrilla run; "
            f"resume it with syndrilla -r={args.run_dir} -ckpt={single} and the same flags, "
            f"or use a new -r for syndrilla-parallel."
        )
    if present and present != planned:
        sys.exit(f"{args.run_dir} holds worker dirs from a different layout. "
                 f"Present, not planned: {sorted(present - planned)}. "
                 f"Planned, not present: {sorted(planned - present)}.")
    recorded = None
    if os.path.isfile(launch):
        with open(launch) as f:
            old = yaml.safe_load(f)
        if not isinstance(old, dict) or not isinstance(old.get("workers"), list):
            sys.exit(f"{launch} has an old worker layout without recorded devices; use a new -r directory.")
        recorded = old["workers"]
        if (len(recorded) != count or any(
            not isinstance(w, dict) or set(w) != {"name", "device"}
            or w["name"] != f"w{i}" or not isinstance(w["device"], str) or not w["device"]
            for i, w in enumerate(recorded)
        )):
            sys.exit(f"{launch} records a different worker layout; a resume needs the same ordered workers.")
        targets = launch_targets(args)
        if any(old.get(key) != value for key, value in targets.items()):
            sys.exit(f"{launch} records different targets; a resume needs the same -te/-tb. "
                     "Use a new -r for different targets.")
    elif present:
        sys.exit(f"{args.run_dir} has worker directories without a launch.yaml worker/device list; use a new -r directory.")
    flag, target_key, _ = args.worker_target
    for i, share in enumerate(shares):
        checkpoint = result_file(os.path.join(args.run_dir, f"w{i}"))
        if checkpoint is not None:
            with open(checkpoint) as f:
                full = yaml.safe_load(f)["decoder_full"]
            if full.get(target_key) != share:
                sys.exit(f"{checkpoint} was run with {target_key} {full.get(target_key)}, but this "
                         f"launch gives w{i} {flag}={share}; a resume needs the same targets and worker layout.")
    return recorded


def plan_workers(args, passthrough, devices=None):
    """Plan flat workers using actual physical device IDs, or the CPU sentinel."""
    flag, _, _ = args.worker_target
    cpu = decoding_device(passthrough) == "cpu"
    if cpu:
        if args.command == "run" and (args.gpus is not None or args.workers_per_gpu != 1):
            print("device_type cpu: one worker on cpu, --gpus and --workers-per-gpu are ignored")
        devices = ["cpu"]
    elif devices is None:
        if args.workers_per_gpu < 1:
            sys.exit("--workers-per-gpu must be positive")
        ids = physical_ids(gpu_list(args), os.environ.get("CUDA_VISIBLE_DEVICES"))
        devices = [device for device in ids for _ in range(args.workers_per_gpu)]
    workers = [{"name": f"w{i}", "device": device} for i, device in enumerate(devices)]
    shares = worker_shares(args, len(workers))
    recorded = preflight_workers(args, len(workers))
    if recorded is not None and recorded != workers:
        sys.exit(f"{args.run_dir}/launch.yaml records workers {recorded}, but this launch gives "
                 f"{workers}; a resume needs the same ordered physical devices.")

    plan = []
    for i, (worker, share) in enumerate(zip(workers, shares)):
        name, device = worker["name"], worker["device"]
        wdir = os.path.join(args.run_dir, name)
        ckpt = result_file(wdir) if os.path.isdir(wdir) else None
        seed = None if args.seed is None else args.seed + i
        before = None
        if ckpt is not None:
            with open(ckpt) as f:
                ckpt_res = yaml.safe_load(f)
            before = summary(ckpt_res)
            if seed is None:
                seed = ckpt_res["decoder_full"].get("seed")
        cmd = [sys.executable, "-m", "syndrilla.main", f"-r={wdir}", f"{flag}={share}"]
        if args.save_interval is not None:
            cmd.append(f"--save-interval={args.save_interval}")
        if args.seed is not None:
            cmd.append(f"--seed={seed}")
        if ckpt is not None:
            cmd.append(f"-ckpt={ckpt}")
        env = {"PYTHONUNBUFFERED": "1"}
        if device != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = device
        plan.append({
            **worker, "target": share, "dir": wdir,
            "log": os.path.join(wdir, "syndrilla.log"), "seed": seed,
            "resumed": ckpt is not None, "before": before, "last": before,
            "cmd": cmd + passthrough, "env": env, "wall": None,
        })
    return plan


def launch_record(args, plan):
    return {**launch_targets(args),
            "workers": [{"name": w["name"], "device": w["device"]} for w in plan]}


def launch_targets(args):
    """The targets given to this launch, as kept in <run_dir>/launch.yaml."""
    return {"target error": args.target_error, "target batch": args.target_batch}


def probe_cmd(probe_dir, passthrough):
    # one batch at the -bs in passthrough, the batch size each worker uses
    return (
        [sys.executable, "-m", "syndrilla.main", f"-r={probe_dir}"]
        + passthrough
        + ["-tb=1"]
    )


def probe(args, passthrough, env):
    """Decode one batch in a scratch dir under the run dir; exit with its log tail on failure.

    On failure, the run dir and any parent dirs that this call created are removed;
    if the run dir existed before, the scratch dir and its log are kept instead.
    """
    created = []  # run dir and its missing parents, deepest first
    d = os.path.abspath(args.run_dir)
    while not os.path.exists(d):
        created.append(d)
        d = os.path.dirname(d)
    os.makedirs(args.run_dir, exist_ok=True)
    probe_dir = tempfile.mkdtemp(prefix="probe_", dir=args.run_dir)
    cmd = probe_cmd(probe_dir, passthrough)
    passed = failed = False
    print(f"probe: {' '.join(cmd)}", flush=True)
    out = os.path.join(probe_dir, "probe.log")
    kept = "" if created else f"\nprobe log kept: {out}"
    try:
        with open(out, "w") as f:
            code = subprocess.call(
                cmd, env=dict(os.environ, **env), stdout=f, stderr=subprocess.STDOUT
            )
        logs = glob.glob(os.path.join(probe_dir, "main-*.log"))
        text = "".join(open(p, errors="replace").read() for p in logs)
        if code != 0:
            # syndrilla's own log holds the error message, its output the traceback
            last = "\n".join(text.splitlines()[-5:])
            failed = True
            sys.exit(
                f"probe failed with exit code {code}; last lines of its log:\n{last}\n"
                f"last lines of its output:\n{tail(out, 12)}{kept}"
            )
        # the decoder falls back to cpu with this warning when cuda is unavailable
        bad = [line for line in text.splitlines() if "unavailable input device" in line]
        device = decoding_device(passthrough)
        if device not in ("cuda", "cpu") or (device == "cuda" and bad):
            failed = True
            sys.exit(
                f"probe decoded on device_type {device}, not cuda; set device_type cuda "
                f"with device_idx 0 in the decoding yaml{': ' + bad[0] if bad else ''}{kept}"
            )
        passed = True
    finally:
        if not (failed and not created):
            shutil.rmtree(probe_dir)
        if not passed:
            for d in created:
                os.rmdir(d)
    print("probe passed", flush=True)


def spawn_workers(plan, label=""):
    for w in plan:
        os.makedirs(w["dir"], exist_ok=True)
        # append, so a resumed worker keeps the log of its earlier launches
        w["file"] = open(w["log"], "a")
        w["proc"] = subprocess.Popen(
            w["cmd"],
            env=dict(os.environ, **w["env"]),
            stdout=w["file"],
            stderr=subprocess.STDOUT,
            start_new_session=True,  # a terminal Ctrl-C reaches only this process
        )
        print(
            f"{label}worker {w['name']}: "
            + ("cpu " if w["device"] == "cpu" else f"CUDA_VISIBLE_DEVICES={w['device']} ")
            + f"{w['cmd'][4]} seed={w['seed']} resumed={w['resumed']}",
            flush=True,
        )


def pooled_now(plan):
    """Re-read every worker's result yaml; return (pooled totals, whether any yaml changed)."""
    changed = False
    for w in plan:
        path = result_file(w["dir"])
        try:
            with open(path) as f:
                s = summary(yaml.safe_load(f))
        # ponytail: a yaml caught mid-write fails to parse or lacks a key and is
        # read again at the next poll; a cut that still parses is not detected
        except Exception:
            continue
        changed |= s != w["last"]
        w["last"] = s
    total = {
        key: sum(w["last"][key] for w in plan if w["last"])
        for key in ("shots", "fails", "batches")
    }
    return total, changed


def start_launch(args, plan, label=""):
    """Write launch.yaml, remove an earlier launch's merged and pooled yamls, start the workers.

    Returns the launch state that step and finish take.
    """
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "launch.yaml"), "w") as f:
        yaml.safe_dump(launch_record(args, plan), f, sort_keys=False)
    # removed, so a failed launch leaves no merged or pooled yaml behind
    for old in [os.path.join(args.run_dir, "merged_result.yaml")] + glob.glob(
        os.path.join(args.run_dir, "result_phy_err_*.yaml")
    ):
        if os.path.isfile(old):
            os.remove(old)
    start = time.time()
    zero = {"shots": 0, "fails": 0, "batches": 0}
    state = {
        "args": args,
        "plan": plan,
        "label": label,
        "start": start,
        "next poll": start + args.poll_interval,
        "base": {key: sum((w["before"] or zero)[key] for w in plan) for key in zero},
        "stopped": None,
        "failed": [],
    }
    spawn_workers(plan, label)
    return state


def step(state):
    """One check of a launch: worker exits, then the pooled stop at each poll.

    Returns True once every worker has exited or one has failed.
    """
    args, plan, start = state["args"], state["plan"], state["start"]
    for w in plan:
        code = w["proc"].poll()
        if w["wall"] is None and code is not None:
            w["wall"] = time.time() - start
            # a worker stopped before it set its SIGTERM handler exits on the
            # signal and keeps its last saved yaml
            if code != 0 and not (state["stopped"] and code == -signal.SIGTERM):
                state["failed"].append(w)
    if state["stopped"] is None and time.time() >= state["next poll"]:
        state["next poll"] += args.poll_interval
        total, changed = pooled_now(plan)
        elapsed = time.time() - start
        rule = None
        if args.pooled and total["fails"] >= args.target_error:
            rule = "pooled -te"
        elif args.pooled and total["batches"] >= args.target_batch:
            rule = "pooled -tb"
        if changed:
            base = state["base"]
            rate = {k: (total[k] - base[k]) / elapsed for k in total}
            eta = [
                (target - total[k]) / rate[k] if rate[k] else math.inf
                for k, target in (
                    ("fails", args.target_error),
                    ("batches", args.target_batch),
                )
                if target is not None
            ]
            lo, hi = wilson(total["fails"], total["shots"])
            ler = total["fails"] / total["shots"] if total["shots"] else 0.0
            print(
                f"{state['label']}[{elapsed:.0f}s] pooled shots={total['shots']} fails={total['fails']} "
                f"batches={total['batches']} LER={ler:.4e} 95% CI=[{lo:.4e}, {hi:.4e}] "
                f"{rate['shots']:.1f} shots/s, ~{max(0.0, min(eta)):.0f}s to target",
                flush=True,
            )
        if rule:
            state["stopped"] = {"rule": rule, "seconds after launch": round(elapsed, 3)}
            state["stopped"].update({f"pooled {k}": v for k, v in total.items()})
            print(
                f"{state['label']}{rule} reached (fails={total['fails']}, batches={total['batches']}); "
                f"sending SIGTERM to all workers",
                flush=True,
            )
            for w in plan:
                if w["proc"].poll() is None:
                    w["proc"].terminate()
    return bool(state["failed"]) or all(w["wall"] is not None for w in plan)


def finish(state):
    """Stop the other workers if one failed and print the failed logs.

    Returns (failed workers, stop record for the merged yaml, launch wall seconds).
    """
    plan, failed = state["plan"], state["failed"]
    if failed:
        stop(plan)
    for w in plan:
        w["file"].close()
    for w in failed:
        print(
            f"{state['label']}worker {w['name']} failed with exit code {w['proc'].returncode}, "
            f"last 20 lines of {w['log']}:\n{tail(w['log'])}",
            file=sys.stderr,
        )
    if failed:
        print(f"{state['label']}the other workers were stopped", file=sys.stderr)
    wall = time.time() - state["start"]
    return failed, state["stopped"] or {"rule": "worker targets"}, wall


def merge(args, plan, stopped, launch_wall):
    """Write merged_result.yaml and the pooled result yaml from the worker yamls; return the merged dict."""
    target_key = args.worker_target[1]
    results, rows, missing, names = [], [], [], []
    for w in plan:
        path = result_file(w["dir"])
        if path is None:  # stopped by the pooled target before its first save
            missing.append(w["name"])
            continue
        with open(path) as f:
            res = yaml.safe_load(f)
        results.append(res)
        r = summary(res)
        new_shots = r["shots"] - (w["before"] or {"shots": 0})["shots"]
        names.append(w["name"])
        rows.append(
            {
                "device": w["device"],
                target_key: w["target"],
                "seed": w["seed"],
                "resumed": w["resumed"],
                "exit code": w["proc"].returncode,
                "batches": r["batches"],
                "shots": r["shots"],
                "shots this launch": new_shots,
                "fails": r["fails"],
                "logical error rate 95% Wilson interval": wilson(
                    r["fails"], r["shots"]
                ),
                "average iteration (last decoder)": r["iteration sum"] / r["shots"]
                if r["shots"]
                else 0.0,
                "decode time (s)": r["decode time (s)"],
                "wall (s)": round(w["wall"], 3),
                "shots per decode second": r["shots"] / r["decode time (s)"]
                if r["decode time (s)"]
                else 0.0,
                "shots per wall second (this launch)": new_shots / w["wall"],
                "physical error rate": r["physical error rate"],
                "_iter": r["iteration sum"],
            }
        )
    if not rows:
        sys.exit("no worker saved a result yaml; nothing merged")
    rates = {row["physical error rate"] for row in rows}
    if len(rates) != 1:
        sys.exit(
            f"workers report different physical error rates {sorted(rates)}; nothing merged"
        )
    shots = sum(r["shots"] for r in rows)
    fails = sum(r["fails"] for r in rows)
    iters = sum(r.pop("_iter") for r in rows)
    new_shots = sum(r["shots this launch"] for r in rows)
    ler = fails / shots if shots else 0.0
    interval = wilson(fails, shots)
    merged = {
        "physical error rate": rows[0]["physical error rate"],
        "workers": len(rows),
        "workers with no result yaml": missing,
        "resumed workers": [w["name"] for w in plan if w["resumed"]],
        "target error": args.target_error,
        "target batch": args.target_batch,
        "stop": stopped,
        "batches": sum(r["batches"] for r in rows),
        "shots": shots,
        "fails": fails,
        "logical error rate": ler,
        "logical error rate 95% Wilson interval": interval,
        "average iteration (last decoder)": iters / shots if shots else 0.0,
        "shots per decode second (sum over workers)": sum(
            r["shots per decode second"] for r in rows
        ),
        "shots per wall second (this launch)": new_shots / launch_wall,
        "launch wall (s)": round(launch_wall, 3),
        "per worker": dict(zip(names, rows)),
    }
    out = os.path.join(args.run_dir, "merged_result.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    pooled = pool_results(results)
    pooled["decoder_full"]["target error"] = args.target_error
    pooled["decoder_full"]["target batch"] = args.target_batch
    pooled_path = os.path.join(
        args.run_dir,
        f"result_phy_err_{pooled['decoder_full']['physical error rate']}.yaml",
    )
    with open(pooled_path, "w") as f:
        f.write(POOLED_NOTE.format(n=len(results)))
        yaml.safe_dump(pooled, f, sort_keys=False)
    print(
        f"pooled: workers={len(rows)} stop={stopped['rule']} shots={shots} fails={fails} "
        f"LER={ler:.6e} 95% CI=[{interval[0]:.6e}, {interval[1]:.6e}] "
        f"decode={merged['shots per decode second (sum over workers)']:.1f} shots/s "
        f"wall={merged['shots per wall second (this launch)']:.1f} shots/s "
        f"({launch_wall:.1f}s) -> {out}, {pooled_path}",
        flush=True,
    )
    return merged


def main():
    args, passthrough = parse_args()
    if args.command == "sweep-gen":
        from syndrilla import sweep

        with open(args.config) as f:
            config = yaml.safe_load(f)
        folders = sweep.generate(config, args.run_dir, args.force)
        print(f"wrote {len(folders)} point folders under {args.run_dir}")
        return
    if args.command == "sweep":
        from syndrilla import sweep

        sweep.run_sweep(args, passthrough)
        return

    plan = plan_workers(args, passthrough)
    if args.dry_run:
        if not args.no_probe:
            probe_dir = os.path.join(args.run_dir, "probe_<scratch>")
            env = " ".join(f"{k}={v}" for k, v in plan[0]["env"].items())
            print(f"probe: {env} {' '.join(probe_cmd(probe_dir, passthrough))}")
        for w in plan:
            env = " ".join(f"{k}={v}" for k, v in w["env"].items())
            print(f"{w['name']}: {env} {' '.join(w['cmd'])}")
        return

    if not args.no_probe:
        probe(args, passthrough, plan[0]["env"])
    try:
        state = start_launch(args, plan)
        while not step(state):
            # ponytail: 0.5 s exit polling, only bounds the per-worker wall time resolution
            time.sleep(0.5)
    except BaseException as e:
        # any exit here sends SIGTERM to every started worker and waits for it to save
        stop([w for w in plan if "proc" in w])
        if isinstance(e, KeyboardInterrupt):
            sys.exit("interrupted; all workers stopped")
        raise
    failed, stopped, launch_wall = finish(state)
    if failed:
        sys.exit(1)
    merge(args, plan, stopped, launch_wall)


if __name__ == "__main__":
    main()
