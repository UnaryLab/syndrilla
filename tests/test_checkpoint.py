"""
Realistic checkpoint round-trip test, mirroring main.py's per-100-batch save.

Runs the same flow under several (error model, channels, rounds) combinations
to verify save/load works for every dimension layout the framework supports:

  - 1-channel BSC + perfect-syndrome (baseline)
  - 1-channel BSC + phenomenological-syndrome (rounds > 1)
  - 2-channel depolarizing + perfect-syndrome (bp4 decoder)

For each config:
  Phase 1 — run N batches with per-batch progress, save at end (done=0)
  Phase 2 — drop in-memory state, reload from disk, validate
  Phase 3 — confirm averaged LER survives the round-trip
  Phase 4 — resume + run a few more batches on top of loaded state

Run:
    python tests/test_checkpoint.py
"""

import os
import sys
import time

import torch
from loguru import logger

sys.path.append(os.getcwd())
logger.remove()  # silence syndrilla's INFO chatter

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.metric import report_metric, save_metric, MetricState, BatchTracker
from syndrilla.logical_check import create_check
from syndrilla.matrix import load_matrices
from syndrilla.utils import read_yaml, get_path, parse_device_dtype


# --------------------------------------------------------------------------
# (label, decoder_yaml, error_yaml, syndrome_yaml, check_yaml, check_type)
# --------------------------------------------------------------------------
CONFIGS = [
    (
        "1ch BSC + perfect",
        "examples/alist/bp_hx.decoder.yaml",
        "examples/alist/bsc.error.yaml",
        "examples/alist/perfect.syndrome.yaml",
        "examples/alist/lx.check.yaml",
        "hx",
    ),
    (
        "1ch BSC + phenomenological (10 rounds)",
        "examples/alist/bp_hx.decoder.yaml",
        "examples/alist/bsc.error.yaml",
        "examples/alist/phenomenological.syndrome.yaml",
        "examples/alist/lx.check.yaml",
        "hx",
    ),
    (
        "2ch depol + perfect (bp4)",
        "examples/alist/bp4.decoder.yaml",
        "examples/alist/depol.error.yaml",
        "examples/alist/perfect.syndrome.yaml",
        "examples/alist/lx.check.yaml",
        "hx",
    ),
]


def build_pipeline(
    decoder_yaml,
    error_yaml,
    syndrome_yaml,
    logical_yaml,
    matrix_yaml="examples/alist/surface_11.matrix.yaml",
):
    decoder_cfg = read_yaml(get_path(decoder_yaml))["decoder"]
    matrix_cfg = read_yaml(get_path(matrix_yaml))["matrix"]
    if not torch.cuda.is_available():
        decoder_cfg["device"] = {"device_type": "cpu", "device_idx": 0}
    bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
    decoders = create_decoder(cfg=decoder_cfg, bundle=bundle)
    error_model = create_error_model(error_yaml)
    syndrome_generator = create_syndrome(syndrome_yaml)
    logical_check = create_check(logical_yaml)
    return decoders, error_model, syndrome_generator, logical_check, bundle, decoder_cfg


def run_one_batch(
    metrics,
    decoders,
    em,
    sg,
    lc,
    bundle,
    batch_size,
    number_channel,
    check_type,
    num_err,
):
    """Run a single decode batch with the same shape as main.py's inner loop."""
    nd = len(decoders)
    dtype = decoders[0].dtype
    dev = decoders[0].device
    shape = bundle.Hx_matrix.get_index()[0]
    H = bundle.select(check_type)[3]
    l_mat = bundle.get_l_matrix(check_type, number_channel)
    nmi = [getattr(d, "num_max_iter", 0) for d in decoders]

    rounds = getattr(sg, "rounds", 1)
    bt = BatchTracker(nd, number_channel, shape, dtype, dev, rounds=rounds)
    zq = torch.zeros([batch_size, shape[1]], dtype=dtype)
    _, dl = em.inject_error(zq, batch_size)
    for err, llr, _ in dl:
        bt.record_error(err)
        synd = sg.measure_syndrome(err, decoders[0])
        io = {"synd": synd, "llr0": llr, "H_matrix": H}
        for i, d in enumerate(decoders):
            t0 = time.time()
            io = d(io)
            bt.record_decoder(i, io, time.time() - t0)
        check = [
            lc.check(bt.e_v_all[i], bt.e_all, l_mat, bt.converge_all[i + 1])
            for i in range(nd)
        ]
        num_err += int(torch.sum(check[-1]))
        if number_channel == 1:
            bt.e_v_all = [t.unsqueeze(1) for t in bt.e_v_all]
            check = [t.unsqueeze(1) for t in check]
        for i in range(nd):
            br = report_metric(
                nmi[i],
                bt.e_all,
                bt.e_v_all[i],
                bt.iter_all[i],
                bt.time_iter_all[i],
                check[i],
                bt.converge_all[i],
                bt.converge_all[i + 1],
                i,
            )
            metrics.accumulate(i, br)
    return num_err


def snapshot(m, number_channel):
    """Capture every accumulator field for later comparison."""
    snap = {}
    for field in (
        "logical_error_rate",
        "data_qubit_acc",
        "data_frame_error_rate",
        "synd_frame_error_rate",
        "correction_acc",
        "converge_fail",
        "converge_succ",
    ):
        snap[field] = [[float(v) for v in row] for row in getattr(m, field)]
    for field in (
        "total_time",
        "average_time_sample",
        "average_iter",
        "average_time_sample_iter",
        "invoke_rate",
    ):
        snap[field] = [float(v) for v in getattr(m, field)]
    snap["distribution"] = [
        d.clone() if hasattr(d, "clone") else d for d in m.distribution
    ]
    return snap


def compare(snap, m_loaded, number_channel, num_decoders):
    """Return list of (field, decoder_idx, channel_idx, pre, post) for any mismatch."""
    issues = []
    tol = 1e-6
    for field in (
        "logical_error_rate",
        "data_qubit_acc",
        "data_frame_error_rate",
        "synd_frame_error_rate",
        "correction_acc",
        "converge_fail",
        "converge_succ",
    ):
        loaded = getattr(m_loaded, field)
        for d in range(num_decoders):
            for c in range(number_channel):
                pre = snap[field][d][c]
                post = float(loaded[d][c])
                if abs(pre - post) > max(tol, abs(pre) * 1e-9):
                    issues.append((field, d, c, pre, post))
    for field in (
        "total_time",
        "average_time_sample",
        "average_iter",
        "average_time_sample_iter",
        "invoke_rate",
    ):
        loaded = getattr(m_loaded, field)
        for d in range(num_decoders):
            pre = snap[field][d]
            post = float(loaded[d])
            if abs(pre - post) > max(tol, abs(pre) * 1e-9):
                issues.append((field, d, "-", pre, post))
    for d in range(num_decoders):
        if not torch.equal(
            snap["distribution"][d].int(), m_loaded.distribution[d].int()
        ):
            issues.append(("distribution", d, "-", "<tensor>", "<tensor>"))
    return issues


def run_one_config(
    label,
    decoder_yaml,
    error_yaml,
    syndrome_yaml,
    check_yaml,
    check_type,
    tmp_dir,
    batch_size=200,
    save_every=20,
    resume_extra=2,
    progress_every=5,
):
    print(f"\n========================================================================")
    print(f"CONFIG: {label}")
    print(f"  decoder_yaml = {decoder_yaml}")
    print(f"  error_yaml   = {error_yaml}")
    print(f"  syndrome_yaml= {syndrome_yaml}")
    print(f"========================================================================")

    decoders, em, sg, lc, bundle, decoder_cfg = build_pipeline(
        decoder_yaml, error_yaml, syndrome_yaml, check_yaml
    )
    nd, dtype, dev = len(decoders), decoders[0].dtype, decoders[0].device
    nc = em.number_channel
    H_file = bundle.get_H_file_name(check_type, nc)
    chk_num = bundle.get_check_num(check_type, nc)
    algo = [d.algo for d in decoders]
    target_error = 10**9
    rounds = getattr(sg, "rounds", 1)
    print(
        f"  num_decoders={nd}  number_channel={nc}  rounds={rounds}  "
        f"algo={algo}  H_file={H_file}"
    )

    # ---------- Phase 1: run save_every batches with per-batch progress ----------
    print(f"\n  Phase 1: run {save_every} batches and save (done=0)")
    torch.manual_seed(0)
    m = MetricState(nd, nc, dev)
    ne = 0
    t_start = time.time()
    for b in range(1, save_every + 1):
        t_b = time.time()
        ne = run_one_batch(
            m, decoders, em, sg, lc, bundle, batch_size, nc, check_type, ne
        )
        if b % progress_every == 0 or b == 1 or b == save_every:
            print(
                f"    batch {b:>3d}/{save_every}  num_err={ne:>7d}  "
                f"batch_time={time.time()-t_b:.3f}s  "
                f"total={time.time()-t_start:.1f}s"
            )

    out = m.get_all_metrics(save_every, algo)
    save_metric(
        out,
        tmp_dir + "/",
        batch_size,
        target_error,
        str(dtype),
        em.rate,
        save_every,
        ne,
        H_file,
        chk_num,
    )
    ckpt_path = os.path.join(tmp_dir, f"result_phy_err_{em.rate}.yaml")
    print(f"    → saved ckpt: {ckpt_path}")

    pre = snapshot(m, nc)
    pre_ne = ne
    pre_avg_ler = m.logical_error_rate[0][0] / save_every

    # ---------- Phase 2: drop, reload, validate ----------
    print(f"  Phase 2: drop state, reload ckpt, validate")
    del m
    m_loaded, meta = MetricState.from_checkpoint(ckpt_path, nc, dev)
    m_loaded.validate_checkpoint(meta, batch_size, target_error, dtype, em.rate, H_file)
    issues = compare(pre, m_loaded, nc, nd)
    nb_ok = meta["batch_count"] == save_every
    ne_ok = meta["num_err"] == pre_ne

    if issues:
        print(f"    BROKEN — {len(issues)} field(s) mismatched:")
        for f, d, c, p, q in issues[:5]:
            print(f"      {f}[{d}][{c}]  pre={p}  post={q}")
    else:
        print(f"    all {nd} decoder(s) × {nc} channel(s) accumulators match exactly")
    print(f'    meta.batch_count={meta["batch_count"]} (expect {save_every}: {nb_ok})')
    print(f'    meta.num_err={meta["num_err"]} (expect {pre_ne}: {ne_ok})')

    # ---------- Phase 3: every averaged metric round-trips through the YAML ----------
    print(f"  Phase 3: every averaged metric round-trips")
    pre_avg = {f"d{i}": pre_out for i, pre_out in enumerate(out)}
    post_avg = {
        f"d{i}": v
        for i, v in enumerate(m_loaded.get_all_metrics(meta["batch_count"], algo))
    }

    def _eq(a, b, tol=1e-9):
        if isinstance(a, str) and isinstance(b, str):
            return a == b
        try:
            af, bf = float(a), float(b)
            return abs(af - bf) <= max(tol, abs(af) * 1e-9)
        except (TypeError, ValueError):
            try:
                return list(a) == list(b)
            except TypeError:
                return a == b

    def _is_numeric(v):
        if isinstance(v, (int, float)):
            return True
        if hasattr(v, "shape") and getattr(v, "shape", None) == torch.Size([]):
            return True
        if hasattr(v, "item"):
            try:
                _ = v.item()
                return True
            except Exception:
                return False
        return False

    fmt = "    {:42s} pre={:>14}  post={:>14}  match={}"
    avg_issues = []
    # ordering: scalar fields first, per-channel rate fields next (one row per
    # channel), tensor fields (distribution) last as a single equality check.
    for d in range(nd):
        key = f"d{d}"
        for field, pre_val in pre_avg[key].items():
            post_val = post_avg[key].get(field)
            ok = _eq(pre_val, post_val)
            if not ok:
                avg_issues.append((d, field, pre_val, post_val))

            # scalar (str / number / 0-D tensor)
            if isinstance(pre_val, str) or _is_numeric(pre_val):
                if isinstance(pre_val, str):
                    pre_disp, post_disp = str(pre_val), str(post_val)
                else:
                    pre_disp = f"{float(pre_val):.6f}"
                    post_disp = f"{float(post_val):.6f}"
                print(fmt.format(f"d{d}.{field}", pre_disp, post_disp, ok))
            # per-channel list (logical_error_rate, data_qubit_acc, etc.) —
            # entries can be Python floats OR 0-D tensors.
            elif isinstance(pre_val, list) and pre_val and _is_numeric(pre_val[0]):
                for ch, cv in enumerate(pre_val):
                    pv = post_val[ch] if ch < len(post_val) else None
                    cok = _eq(cv, pv)
                    if not cok:
                        avg_issues.append((d, f"{field}[ch{ch}]", cv, pv))
                    pre_disp = f"{float(cv):.6f}"
                    post_disp = f"{float(pv):.6f}" if pv is not None else "None"
                    print(fmt.format(f"d{d}.{field}[ch{ch}]", pre_disp, post_disp, cok))
            # tensor with shape (iteration distribution): summarize as equality
            elif hasattr(pre_val, "shape"):
                eq = bool(
                    torch.equal(
                        pre_val.int() if hasattr(pre_val, "int") else pre_val,
                        post_val.int() if hasattr(post_val, "int") else post_val,
                    )
                )
                if not eq:
                    avg_issues.append((d, field, "<tensor>", "<tensor>"))
                print(
                    fmt.format(
                        f"d{d}.{field}",
                        f"shape={tuple(pre_val.shape)}",
                        f"shape={tuple(post_val.shape)}",
                        eq,
                    )
                )
            # nested dict (hx/hz) — should not appear at this layer, but be safe
            elif isinstance(pre_val, dict):
                for ck, cv in pre_val.items():
                    pv = post_val.get(ck)
                    cok = _eq(cv, pv)
                    if not cok:
                        avg_issues.append((d, f"{field}.{ck}", cv, pv))
                    pre_disp = f"{float(cv):.6f}"
                    post_disp = f"{float(pv):.6f}"
                    print(fmt.format(f"d{d}.{field}.{ck}", pre_disp, post_disp, cok))
    ler_ok = not avg_issues
    if avg_issues:
        print(f"    BROKEN — {len(avg_issues)} averaged field(s) mismatched")

    # ---------- Phase 4: resume ----------
    print(f"  Phase 4: resume + {resume_extra} more batches")
    nb = meta["batch_count"]
    ne = meta["num_err"]
    try:
        for b in range(1, resume_extra + 1):
            ne = run_one_batch(
                m_loaded, decoders, em, sg, lc, bundle, batch_size, nc, check_type, ne
            )
            nb += 1
        print(
            f"    resumed OK — final batches={nb}  num_err={ne}  "
            f"LER={m_loaded.logical_error_rate[0][0]/nb:.6f}"
        )
        resume_ok = True
    except Exception as e:
        print(f"    RESUME FAILED — {type(e).__name__}: {str(e)[:120]}")
        resume_ok = False

    return not issues and nb_ok and ne_ok and ler_ok and resume_ok


def main():
    results = {}
    out_dir = "tests/test_outputs"
    os.makedirs(out_dir, exist_ok=True)
    for cfg in CONFIGS:
        label = cfg[0]
        try:
            ok = run_one_config(*cfg, tmp_dir=out_dir)
        except Exception as e:
            print(f"\n  CONFIG ERROR — {type(e).__name__}: {e}")
            ok = False
        results[label] = ok

    print(f"\n========================================================================")
    print(f"OVERALL SUMMARY")
    print(f"========================================================================")
    for label, ok in results.items():
        print(f'  {"PASS" if ok else "FAIL"}  {label}')


if __name__ == "__main__":
    main()
