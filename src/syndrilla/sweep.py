"""Sweep point folders for syndrilla-parallel: sweep-gen writes them, sweep runs them.

sweep-gen writes one point folder per combination in a sweeping configs yaml
(decoder, code, check_type, probability, distance, dtype lists) directly under
the sweep dir, named <code>_<check_type>_<probability>_<distance>_<dtype>, each
holding <decoder>_<check_type>.decoding.yaml, bsc.error.yaml, lx.check.yaml or
lz.check.yaml, perfect.syndrome.yaml and matrix.yaml. Templates and matrix
paths are relative to the repo root (examples/alist/), so it runs from there.

sweep runs every point folder (sorted by name) like syndrilla-parallel run with
-r set to that folder; see run_sweep.
"""

import csv
import glob
import itertools
import os
import sys
import time
from argparse import Namespace

from syndrilla import parallel as par
from syndrilla.utils import read_yaml, write_yaml

TEMPLATES = "examples/alist/"

# flag of syndrilla.main and the file pattern it takes in a point folder
POINT_YAMLS = [
    ("-d", "*.decoding.yaml"),
    ("-e", "*.error.yaml"),
    ("-c", "*.check.yaml"),
    ("-s", "*.syndrome.yaml"),
    ("-m", "matrix.yaml"),
]

CSV_FIELDS = [
    "folder",
    "code",
    "check_type",
    "p",
    "d",
    "dtype",
    "shots",
    "fails",
    "LER",
    "Wilson low",
    "Wilson high",
    "wall s",
    "status",
]


def write_point(path, decoder, code, check_type, probability, distance, dtype):
    """Write the five yamls of one point folder."""
    config = read_yaml(f"{TEMPLATES}{decoder}_{check_type}.decoding.yaml")
    config["decoding"]["dtype"] = dtype
    config["decoding"]["check_type"] = check_type
    stage = config["decoding"].setdefault("config", {})
    stage = stage[0] if isinstance(stage, list) else stage
    if code == "surface":
        stage["max_iter"] = distance * 2 * (distance - 1) + 1
    else:
        stage["max_iter"] = distance * 2 * distance
    write_yaml(os.path.join(path, f"{decoder}_{check_type}.decoding.yaml"), config)

    config = read_yaml(f"{TEMPLATES}perfect.syndrome.yaml")
    write_yaml(os.path.join(path, "perfect.syndrome.yaml"), config)

    config = read_yaml(f"{TEMPLATES}bsc.error.yaml")
    config["error"]["rate"] = probability
    write_yaml(os.path.join(path, "bsc.error.yaml"), config)

    alist = f"examples/alist/{code}/{code}_{distance}"
    matrix = {
        "parity_matrix_hx": {"file_type": "alist", "path": f"{alist}_hx.alist"},
        "parity_matrix_hz": {"file_type": "alist", "path": f"{alist}_hz.alist"},
        "logical_check_matrix": True,
        "logical_check_lx": {"file_type": "alist", "path": f"{alist}_lx.alist"},
        "logical_check_lz": {"file_type": "alist", "path": f"{alist}_lz.alist"},
    }
    write_yaml(os.path.join(path, "matrix.yaml"), {"matrix": matrix})

    check = "lx.check.yaml" if check_type == "hx" else "lz.check.yaml"
    write_yaml(os.path.join(path, check), read_yaml(TEMPLATES + check))


def generate(config, sweep_dir, force=False):
    """Write every point folder of config under sweep_dir; return the folder paths.

    Exits without writing anything if a point folder already holds a result yaml,
    unless force is set.
    """
    if len(config["decoder"]) > 1:
        sys.exit(
            f"decoder lists {config['decoder']}, but a point folder holds one decoding yaml "
            f"and its name has no decoder; give one decoder per sweep dir. Nothing written."
        )
    keys = ["decoder", "code", "check_type", "probability", "distance", "dtype"]
    points = []
    for dec, code, ct, p, d, dt in itertools.product(*(config[k] for k in keys)):
        path = os.path.join(sweep_dir, f"{code}_{ct}_{p}_{d}_{dt}")
        points.append((path, dec, code, ct, p, d, dt))
    held = sorted(
        {
            pt[0]
            for pt in points
            if glob.glob(
                os.path.join(pt[0], "**", "result_phy_err_*.yaml"), recursive=True
            )
        }
    )
    if held and not force:
        sys.exit(
            f"{len(held)} point folders already hold a result yaml, for example {held[0]}; "
            f"nothing written. --force rewrites their config yamls and keeps the results."
        )
    for pt in points:
        os.makedirs(pt[0], exist_ok=True)
        write_point(*pt)
    return sorted({pt[0] for pt in points})


def point_flags(folder):
    """The -d -e -c -s -m flags of a point folder; exits unless each matches one file."""
    flags = []
    for flag, pattern in POINT_YAMLS:
        found = glob.glob(os.path.join(folder, pattern))
        if len(found) != 1:
            sys.exit(
                f"point {folder} needs exactly one {pattern} for {flag}, found {found}"
            )
        flags.append(f"{flag}={found[0]}")
    return flags


def csv_row(folder, merged=None, wall=None, status="not run"):
    name = os.path.basename(folder)
    parts = name.rsplit("_", 4)
    row = dict(zip(CSV_FIELDS, [name] + (parts if len(parts) == 5 else [""] * 5)))
    if merged:
        row.update(
            shots=merged["shots"],
            fails=merged["fails"],
            LER=merged["logical error rate"],
        )
        row["Wilson low"], row["Wilson high"] = merged[
            "logical error rate 95% Wilson interval"
        ]
    if wall is not None:
        row["wall s"] = round(wall, 3)
    row["status"] = status
    return row


def run_sweep(args, passthrough):
    """Run every point folder under args.run_dir as a syndrilla-parallel launch.

    Points are a queue in sorted folder order; point i starts when k GPUs each
    have a free worker slot (--workers-per-gpu slots per GPU) and runs one
    worker on each, in worker dirs gpu0_w0 .. gpu<k-1>_w0 (the point's k GPU
    slots, not GPU indices). Point i, worker j gets seed base + i*k + j.
    A point folder that holds merged_result.yaml is done: its CSV row is built
    from that yaml and it is not launched again.
    """
    k = args.gpus_per_point
    gpus = par.gpu_list(args)
    devices = dict(
        zip(gpus, par.physical_ids(gpus, os.environ.get("CUDA_VISIBLE_DEVICES")))
    )
    if not 1 <= k <= len(gpus):
        sys.exit(f"--gpus-per-point {k} must be between 1 and the {len(gpus)} GPUs")
    folders = sorted(
        d
        for d in glob.glob(os.path.join(args.run_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.decoding.yaml"))
    )
    if not folders:
        sys.exit(
            f"{args.run_dir} holds no point folder (a folder with a *.decoding.yaml)"
        )

    # plan every point first, so a refusal stops the sweep before any point starts
    points = []
    for i, folder in enumerate(folders):
        pargs = Namespace(**vars(args))
        pargs.run_dir, pargs.gpus, pargs.workers_per_gpu = folder, list(range(k)), 1
        if args.seed is not None:
            pargs.seed = args.seed + i * k
        flags = point_flags(folder) + passthrough
        pt = {
            "args": pargs,
            "flags": flags,
            "plan": par.plan_workers(pargs, flags),
            "label": f"[{os.path.basename(folder)}] ",
            "row": csv_row(folder),
        }
        merged = os.path.join(folder, "merged_result.yaml")
        if os.path.isfile(merged):
            m = read_yaml(merged)
            pt["row"] = csv_row(folder, m, m["launch wall (s)"], "done")
            pt["done"] = True
        points.append(pt)

    for pt in points:
        if pt.get("done"):
            print(f"{pt['label']}done, merged_result.yaml kept, not launched")
    queue = [pt for pt in points if not pt.get("done")]
    if args.dry_run:
        for pt in queue:
            print(pt["label"])
            for w in pt["plan"]:
                print(f"  {w['name']} on GPU slot {w['gpu']}: {' '.join(w['cmd'])}")
        return

    used = {g: 0 for g in gpus}
    running, failed = [], []
    try:
        while queue or running:
            free = sorted(
                (g for g in gpus if used[g] < args.workers_per_gpu),
                key=lambda g: used[g],
            )
            if queue and len(free) >= k and not (failed and args.fail_fast):
                pt = queue.pop(0)
                pt["gpus"] = free[:k]
                for g in pt["gpus"]:
                    used[g] += 1
                for w, g in zip(pt["plan"], pt["gpus"]):
                    if w["gpu"] is not None:
                        w["device"] = w["env"]["CUDA_VISIBLE_DEVICES"] = devices[g]
                print(f"{pt['label']}starting on GPUs {pt['gpus']}", flush=True)
                try:
                    if not args.no_probe:
                        # ponytail: the probe blocks the poll loop of running points for its seconds
                        par.probe(pt["args"], pt["flags"], pt["plan"][0]["env"])
                    pt["state"] = par.start_launch(pt["args"], pt["plan"], pt["label"])
                    running.append(pt)
                except SystemExit as e:
                    print(f"{pt['label']}failed: {e}", file=sys.stderr, flush=True)
                    pt["row"]["status"] = "failed"
                    failed.append(pt)
                    for g in pt["gpus"]:
                        used[g] -= 1
                continue
            for pt in list(running):
                if not par.step(pt["state"]):
                    continue
                running.remove(pt)
                for g in pt["gpus"]:
                    used[g] -= 1
                bad, stopped, wall = par.finish(pt["state"])
                if bad:
                    pt["row"].update(
                        csv_row(pt["args"].run_dir, wall=wall, status="failed")
                    )
                    failed.append(pt)
                    continue
                try:
                    merged = par.merge(pt["args"], pt["plan"], stopped, wall)
                    pt["row"] = csv_row(pt["args"].run_dir, merged, wall, "done")
                except SystemExit as e:
                    print(f"{pt['label']}{e}", file=sys.stderr, flush=True)
                    pt["row"]["status"] = "failed"
                    failed.append(pt)
            if failed and args.fail_fast and running:
                print(
                    "--fail-fast: stopping the running points",
                    file=sys.stderr,
                    flush=True,
                )
                for pt in running:
                    par.stop(pt["plan"])
                    pt["row"]["status"] = "stopped"
                running = []
            if failed and args.fail_fast:
                queue = []
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("interrupted; stopping all started points", file=sys.stderr)
        failed.append(None)
    finally:
        # on any exit, every started worker of an unfinished point gets SIGTERM and
        # saves; the row takes the counts of the worker yamls it saved
        for pt in points:
            started = [w for w in pt["plan"] if "proc" in w]
            if started and pt["row"]["status"] in ("not run", "stopped"):
                par.stop(started)
                t = par.pooled_now(pt["plan"])[0]
                m = None
                if t["shots"]:
                    m = dict(t, **{"logical error rate": t["fails"] / t["shots"]})
                    m["logical error rate 95% Wilson interval"] = par.wilson(
                        t["fails"], t["shots"]
                    )
                pt["row"] = csv_row(pt["args"].run_dir, m, status="stopped")
        out = os.path.join(args.run_dir, "sweep_results.csv")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(pt["row"] for pt in points)
        print(f"sweep results -> {out}", flush=True)
    names = [os.path.basename(pt["args"].run_dir) for pt in failed if pt]
    if names:
        print(f"failed points: {names}", file=sys.stderr)
    if failed:
        sys.exit(1)
