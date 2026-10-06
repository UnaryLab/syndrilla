"""Speed benchmark: syndrilla mwpm_gpu (the native sparse-blossom port) vs PyMatching v2 on the stim
surface_code:rotated_memory_x decomposed DEM (interface key decompose_errors true),
all four stim noise knobs at p, rounds = distance.

One fixed shot set (torch seed --seed, chunks of --chunk, syndrilla's own
mechanism sampler) is drawn once and fed to every row:
  a  syndrilla mwpm_gpu, uniform weights, cpu, num_workers 32
  b  syndrilla mwpm_gpu, weights prior, cpu, num_workers 32
  c  syndrilla mwpm_gpu, uniform weights, cuda device (the path that actually ran
     is read from the decoder after create and put in the label)
  d  PyMatching Matching.from_detector_error_model on syndrilla's decomposed
     DEM (iface.error_model.dem), decode_batch, single process
  e  PyMatching Matching.from_check_matrix on syndrilla's H and observable
     matrix, no weights, so every edge weighs 1 (uniform)
  f  PyMatching from_check_matrix on syndrilla's H with the integer grid mwpm_gpu
     normalizes to, round(|llr0| * (2^24 - 1) / max |llr0|); agreement check for b
  g  PyMatching from_check_matrix on syndrilla's H with float |llr0| weights;
     agreement check for d
Each row is run once for warm-up, then --repeats timed passes over all chunks;
wall is the median, sampling excluded, CUDA synced. LER is scored on the
warm-up pass with the stim logical check (an unconverged shot counts as an
error). "differs vs d" is the number of shots whose predicted observable flip
differs from row d's; the raw corrections live on different column sets
(syndrilla components vs PyMatching edges), so the observable is the only
common comparison.

    conda run -n syndrilla --no-capture-output python zoo/script/mwpm_vs_pymatching.py \
        [--distance 9] [--p 0.01] [--shots 20000] [--chunk 4096] [--repeats 3]
"""
import argparse
import logging
import platform
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from belief_matching_stim import NOISE, make_interface  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "zoo" / "belief_matching"
log = logging.getLogger("mwpm_vs_pymatching")


def make_syndrilla(bundle, device, config):
    from syndrilla.decoder import create_decoder

    cfg = {
        "algorithm": "mwpm_gpu",
        "check_type": "hx",
        "dtype": "float32",
        "device": {"device_type": device, "device_idx": 0},
        "config": config,
    }
    (dec,) = create_decoder(cfg=cfg, bundle=bundle)
    dec.eval()
    return dec


def cuda_path(dec):
    """Which mwpm_gpu_cuda path the decoder takes; decided once at create (mwpm_gpu_cuda.py:102)."""
    inner = next((m for m in dec.modules() if hasattr(m, "_use_kernel")), None)
    if inner is None:
        return "no _use_kernel attribute found (not the mwpm_gpu_cuda decoder)"
    if inner._use_kernel:
        return f"CUDA blossom kernel (N={inner.N} <= 256)" + (
            ", host-side correction reconstruction (N > 64), over the num_workers pool when num_workers > 1 and the batch is at least mp_min_batch, else serial"
            if inner.N > 64
            else ""
        )
    return (
        f"serial single-process CPU blossom fallback, N={inner.N} > 256 (mwpm_gpu_cuda.py:102, :264-269); "
        "the kernel is compiled at create but never launched"
    )


def syndrilla_runner(dec, device, l_matrix):
    import torch

    dev = torch.device(device)
    L = torch.as_tensor(l_matrix, dtype=torch.float32)

    def run(chunk):
        io = {"synd": chunk["synd"].to(dev), "llr0": chunk["llr0"].to(dev)}
        out = dec(io)
        e_v = out["e_v"].to("cpu").float()
        pred = (e_v @ L.T) % 2
        return pred.to(torch.uint8), out["converge"].to("cpu")

    return run


def pymatching_runner(matching):
    import numpy as np
    import torch

    def run(chunk):
        synd = chunk["synd"].to("cpu").numpy().astype(np.uint8)
        pred = matching.decode_batch(synd)
        return torch.as_tensor(pred, dtype=torch.uint8), None

    return run


def time_row(name, run, chunks, lcheck, l_matrix, args):
    import torch

    cuda = torch.cuda.is_available()

    def one_pass():
        preds, convs, wall = [], [], 0.0
        for c in chunks:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred, conv = run(c)
            if cuda:
                torch.cuda.synchronize()
            wall += time.perf_counter() - t0
            preds.append(pred)
            convs.append(conv)
        return torch.cat(preds), (None if convs[0] is None else torch.cat(convs)), wall

    pred, conv, warm = one_pass()  # warm-up, also the scored pass
    times = [one_pass()[2] for _ in range(args.repeats)]
    truth = torch.cat([c["truth"] for c in chunks]).to("cpu")
    fails = int((pred != truth).any(dim=1).sum())
    if conv is not None:
        fails_conv = int(((pred != truth).any(dim=1) | (conv.to("cpu") == 0)).sum())
    else:
        fails_conv = fails
    shots = int(truth.shape[0])
    from syndrilla.parallel import wilson

    lo, hi = wilson(fails_conv, shots)
    med = statistics.median(times)
    log.info(
        "%s: warm %.3f s, timed %s, median %.3f s, %d/%d errors",
        name,
        warm,
        [f"{t:.3f}" for t in times],
        med,
        fails_conv,
        shots,
    )
    return {
        "row": name,
        "shots": shots,
        "median wall (s)": med,
        "shots/s": shots / med,
        "logical errors": fails_conv,
        "LER": fails_conv / shots,
        "lo": lo,
        "hi": hi,
        "times": times,
        "pred": pred,
        "unconverged": 0 if conv is None else int((conv == 0).sum()),
    }


def env_info():
    import torch
    import pymatching
    import stim

    cpu = next(
        (
            l.split(":", 1)[1].strip()
            for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")
        ),
        platform.processor(),
    )
    import os

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "gpu": gpu,
        "torch": torch.__version__,
        "pymatching": pymatching.__version__,
        "stim": stim.__version__,
        "python": platform.python_version(),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--distance", type=int, default=9)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--only", default="a,b,c,d,e,f,g")
    ap.add_argument(
        "--out", default=str(OUT_DIR / f"{date.today()}-mwpm-vs-pymatching.md")
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out.with_suffix(".log")),
        ],
    )
    from loguru import logger as loguru_logger

    cuda_lines = []
    loguru_logger.add(
        lambda m: cuda_lines.append(m.record["message"])
        if "mwpm" in m.record["name"]
        else None,
        level="DEBUG",
    )
    loguru_logger.add(out.with_suffix(".loguru.log"), level="DEBUG")

    import torch
    import pymatching

    env = env_info()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    log.info("start: commit %s, env %s, args %s", commit, env, vars(args))

    iface = make_interface(
        args.distance, args.p, "cuda" if torch.cuda.is_available() else "cpu"
    )
    bundle, emodel, syndrome, lcheck = (
        iface.matrix_bundle,
        iface.error_model,
        iface.syndrome_generator,
        iface.logical_check,
    )
    shape = bundle.select("hx")[0]
    l_matrix = bundle.get_l_matrix("hx", 1)
    dem = emodel.dem

    # one shot set for every row
    torch.manual_seed(args.seed)
    chunks = []
    probe = make_syndrilla(bundle, "cpu", {"weights": "uniform", "num_workers": 1})
    with torch.no_grad():
        drawn = 0
        while drawn < args.shots:
            n = min(args.chunk, args.shots - drawn)
            err, _ = emodel.inject_error(
                torch.zeros([n, shape[1]], dtype=probe.dtype), n
            )
            synd = syndrome.measure_syndrome(err, probe)
            chunks.append(
                {
                    "synd": synd,
                    "truth": syndrome.observable_flips.clone(),
                    "llr0": emodel.get_llr(err),
                }
            )
            drawn += n
    log.info(
        "drew %d shots in %d chunks; H %s, DEM detectors %d errors %d",
        drawn,
        len(chunks),
        tuple(shape),
        dem.num_detectors,
        dem.num_errors,
    )

    from syndrilla.matrix.matrix import STIM_CIRCUIT_CACHE

    cache = STIM_CIRCUIT_CACHE[(str(iface.circuit), True)]
    rows_def = {
        "a": (
            "syndrilla mwpm_gpu uniform, cpu, num_workers %d" % args.num_workers,
            lambda: syndrilla_runner(
                make_syndrilla(
                    bundle,
                    "cpu",
                    {"weights": "uniform", "num_workers": args.num_workers},
                ),
                "cpu",
                l_matrix,
            ),
        ),
        "b": (
            "syndrilla mwpm_gpu weights prior, cpu, num_workers %d" % args.num_workers,
            lambda: syndrilla_runner(
                make_syndrilla(
                    bundle,
                    "cpu",
                    {
                        "weights": "prior",
                        "num_workers": args.num_workers,
                    },
                ),
                "cpu",
                l_matrix,
            ),
        ),
        "c": (
            "syndrilla mwpm_gpu uniform, cuda device",
            lambda: syndrilla_runner(cuda_dec(), "cuda", l_matrix),
        ),
        "d": (
            "PyMatching from_detector_error_model (decomposed DEM), decode_batch, single process",
            lambda: pymatching_runner(
                pymatching.Matching.from_detector_error_model(dem)
            ),
        ),
        "e": (
            "PyMatching from_check_matrix (syndrilla H, observable matrix), no weights = uniform",
            lambda: pymatching_runner(
                pymatching.Matching.from_check_matrix(
                    cache["H"], faults_matrix=cache["obs_mat"]
                )
            ),
        ),
        # agreement checks: the same PyMatching matcher on syndrilla's H with syndrilla's weights
        "f": (
            "PyMatching from_check_matrix (syndrilla H) with the integer grid mwpm_gpu normalizes to, round(|llr0| * (2^24 - 1) / max |llr0|)",
            lambda: pymatching_runner(
                pymatching.Matching.from_check_matrix(
                    cache["H"],
                    weights=normalized_weights(),
                    faults_matrix=cache["obs_mat"],
                )
            ),
        ),
        "g": (
            "PyMatching from_check_matrix (syndrilla H) with float weights |llr0|",
            lambda: pymatching_runner(
                pymatching.Matching.from_check_matrix(
                    cache["H"], weights=float_weights(), faults_matrix=cache["obs_mat"]
                )
            ),
        ),
    }
    path = {}

    def base_llr():
        import numpy as np

        llr0 = torch.cat([c["llr0"] for c in chunks]).to("cpu").double()
        assert bool(
            (llr0 == llr0[0]).all()
        ), "llr0 differs between shots; weights must be per shot"
        assert bool(
            (llr0[0] > 0).all()
        ), "a non-positive llr0 would be pre-flipped by syndrilla (mwpm_gpu.py:1872-1875)"
        return llr0[0].numpy()

    def normalized_weights():
        import numpy as np

        # mwpm_gpu's PyMatching-style normalization, mwpm_gpu.py:1494-1505: the largest
        # retained weight maps to 2^24 - 1, rounded half up (all-integer rows unscaled; the matcher then doubles)
        w = np.abs(base_llr())
        if not np.equal(w, np.floor(w)).all():
            w = w * ((1 << 24) - 1) / w.max()
        lower = np.floor(w)
        return lower + (w - lower >= 0.5)  # PyMatching doubles these itself; 2x would exceed its 2^24 - 1 limit

    def float_weights():
        import numpy as np

        return np.abs(base_llr())

    def cuda_dec():
        dec = make_syndrilla(
            bundle, "cuda", {"weights": "uniform", "num_workers": args.num_workers}
        )
        path["c"] = cuda_path(dec)
        log.info("row c path: %s", path["c"])
        return dec

    results = {}
    with torch.no_grad():
        for key in args.only.split(","):
            label, build = rows_def[key]
            before = len(cuda_lines)
            try:
                run = build()
                results[key] = time_row(key, run, chunks, lcheck, l_matrix, args)
            except Exception:
                log.exception("row %s failed", key)
                continue
            results[key]["label"] = label
            results[key]["log"] = sorted(set(cuda_lines[before:]))
            results[key]["path"] = path.get(key, "")
            if key == "c" and path.get("c"):
                results[key]["label"] += ": " + path["c"]
    # shot-by-shot agreement: number of shots whose predicted observable differs
    keys = [k for k in "abcdefg" if k in results]
    agree = {}
    for i in keys:
        for j in keys:
            agree[(i, j)] = int(
                (results[i]["pred"] != results[j]["pred"]).any(dim=1).sum()
            )
    for i, j in [("f", "b"), ("g", "d"), ("e", "a"), ("c", "a"), ("b", "d")]:
        if (i, j) in agree:
            log.info(
                "agreement %s vs %s: %d of %d shots differ",
                i,
                j,
                agree[(i, j)],
                results[i]["shots"],
            )
    write_md(out, results, env, commit, args, shape, dem, keys, agree)
    log.info("wrote %s", out)


def write_md(out, results, env, commit, args, shape, dem, keys, agree):
    ref = results.get("d")
    cols = [
        "row",
        "decoder",
        "median wall (s)",
        "shots/s",
        "logical errors",
        "LER",
        "Wilson 95% low",
        "Wilson 95% high",
        "differs vs d (shots)",
    ]
    lines = [
        f"# syndrilla mwpm_gpu vs PyMatching v2, stim surface code d{args.distance} ({date.today()})",
        "",
        f"- Code: stim surface_code:rotated_memory_x, distance {args.distance}, rounds {args.distance}, the DEM decomposed into graphlike components as H (interface decompose_errors true, the keys of examples/stim/stim_mwpm.interface.yaml at distance {args.distance}); H is {shape[0]} detectors by {shape[1]} component columns; the stim DEM has {dem.num_detectors} detectors and {dem.num_errors} error instructions",
        f"- Error model: stim_circuit with {', '.join(NOISE)} and measurement_error_rate all {args.p:g}",
        f"- Shots: {args.shots} drawn once by syndrilla's mechanism sampler (torch seed {args.seed}, chunks of {args.chunk}) and fed to every row; syndromes and observable truth from syndrilla (H times the drawn error, and the observable matrix times it)",
        f"- Timing: one warm-up pass, then {args.repeats} timed passes over all chunks; wall = median of the timed passes, sampling and host-side scoring excluded, CUDA synced before and after each chunk; shots/s = shots over the median wall; row b's timed region rebuilds a NativeMatcher per worker per chunk (the per-batch cache is cleared on each forward, mwpm_gpu.py:1673-1681, :1836) and pickles the {shape[1]}-weight tuple with every shot (mwpm_gpu.py:1837-1839), while rows d and e build their matcher once, outside the timed region",
        "- LER: scored on the warm-up pass with the stim logical check (rows a, b, c: predicted observable = e_v times the observable matrix mod 2, an unconverged shot counts as an error; rows d, e: the observable flips decode_batch returns); Wilson 95% with z = 1.959963984540054",
        "- differs vs d: shots whose predicted observable flip differs from row d; the raw corrections live on different column sets (syndrilla component columns vs PyMatching DEM edges), so the observable is the common comparison",
        f"- Machine: {env['cpu']}, {env['cores']} logical cores, GPU {env['gpu']}; python {env['python']}, torch {env['torch']}, pymatching {env['pymatching']}, stim {env['stim']}; pymatching is a declared dependency (pyproject.toml)",
        f"- Provenance: git HEAD {commit} plus the uncommitted working tree, one invocation of zoo/script/mwpm_vs_pymatching.py (the last start line of the .log, which is appended across invocations)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for k in "abcdefg":
        r = results.get(k)
        if r is None:
            lines.append(f"| {k} | not run or failed (see .log) | | | | | | | |")
            continue
        diff = (
            "-"
            if ref is None
            else str(int((r["pred"] != ref["pred"]).any(dim=1).sum()))
        )
        lines.append(
            f"| {k} | {r['label']} | {r['median wall (s)']:.3f} | {r['shots/s']:.0f} | {r['logical errors']} | {r['LER']:.4g} | {r['lo']:.4g} | {r['hi']:.4g} | {diff} |"
        )
    lines += [""]
    for k in "abcdefg":
        r = results.get(k)
        if r:
            lines.append(
                f"- {k} timed passes (s): {', '.join(f'{t:.3f}' for t in r['times'])}"
                + (f"; unconverged shots {r['unconverged']}" if k in "abc" else "")
            )
    if results.get("c"):
        msgs = results["c"]["log"]
        lines.append(
            f"- c path, read from the decoder's `_use_kernel` and `N` after create: {results['c']['path']}"
        )
        lines.append(
            "- c mwpm_gpu_cuda log lines during create and the passes: "
            + ("; ".join(f'"{m}"' for m in msgs) if msgs else "none captured")
            + '. The "mwpm_gpu_cuda decoder ready (CUDA blossom kernel, ...)" line is logged unconditionally (mwpm_gpu_cuda.py:124) and does not mean the kernel ran'
        )
    lines += [
        "",
        "## Shot-by-shot agreement",
        "",
        "Number of the shots whose predicted observable flip differs between the two rows (rows f and g are PyMatching on syndrilla's H with syndrilla's weights, run only for this check; the .log holds the same counts on its `agreement` lines):",
        "",
        "| | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 1),
    ]
    for i in keys:
        lines.append(f"| {i} | " + " | ".join(str(agree[(i, j)]) for j in keys) + " |")
    lines += [
        "",
        "## Reading",
        "",
        "(filled in by hand after the run; a rerun overwrites it)",
    ]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
