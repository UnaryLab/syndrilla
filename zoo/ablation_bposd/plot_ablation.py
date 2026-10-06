"""Plot ablation results written by zoo/ablation_bposd/ablation.py.

Usage: python zoo/ablation_bposd/plot_ablation.py results.csv [--outdir DIR]

Writes, next to the CSV (or into --outdir), with the CSV stem as prefix:
  <stem>-total-vs-d-p<p>.pdf  speedup_total vs distance, one figure per p
  <stem>-total-vs-p-d<d>.pdf  speedup_total vs error rate, one figure per d
  <stem>-step-vs-d-p<p>.pdf   speedup_step bars grouped by distance, one figure per p
  <stem>-stage-vs-d-p<p>.pdf  bp_ms + osd_ms stacked bars grouped by distance, one
                              figure per p (skipped if the CSV has no bp_ms/osd_ms)
  <stem>-ler-vs-p.pdf         logical error rate vs p, rows (path, dtype), columns d
Configs follow the add-one-in ladder order (LADDER); +rebatch_opt is all groups on.
Panels are per (path, dtype). Rows with status != ok are skipped, except timeout
rows with a time_ms: "timeout, N of S shots" rows (time extrapolated from a
partial pass) are drawn hollow or hatched, "timeout after K timed passes" rows
(median of K full passes) are drawn as ok rows. A speedup whose reference row
(all_off for speedup_total, the previous step for speedup_step) timed out divides
by that row's extrapolated time and carries no mark of its own. Nonpositive
values are not drawn on the log axes.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.6,
    }
)

LADDER = [
    "all_off",
    "+memory_opt",
    "+pruning_opt",
    "+fusion_opt",
    "+mapping_opt",
    "+gather_opt",
    "+rebatch_opt",
]
ALL_ON = "+rebatch_opt"
# Okabe-Ito; black is reserved for ALL_ON
COLORS = [
    "#999999",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
    "#CC79A7",
    "#F0E442",
]
MARKERS = ["s", "^", "v", "D", "P", "X", "<", ">", "h", "*"]
LINESTYLES = ["--", ":", "-.", (0, (5, 1))]
TIMEOUT_LABEL = "timeout (extrapolated)"


def num(s):
    return float(s) if s not in ("", None) else np.nan


def ladder_key(c):
    return (LADDER.index(c), "") if c in LADDER else (len(LADDER), c)


def config_styles(configs):
    """Fixed style per config so a config looks the same in every figure."""
    styles = {}
    others = [c for c in sorted(configs, key=ladder_key) if c != ALL_ON]
    for i, c in enumerate(others):
        styles[c] = dict(
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
        )
    if ALL_ON in configs:
        # larger marker so ALL_ON stays visible under an identical config
        styles[ALL_ON] = dict(
            color="black", marker="o", markersize=5.5, linestyle="-", linewidth=2.0
        )
    return styles


def draw(ax, rows, config, x, y, s):
    """Line for one config; timeout rows get a hollow marker. Returns True if drawn."""
    pts = sorted(
        (r[x], r[y], r["timeout"]) for r in rows if r["config"] == config and r[y] > 0
    )
    if not pts:
        return False
    xs, ys, to = (np.array(v) for v in zip(*pts))
    ax.plot(xs, ys, **s)
    if to.any():
        ax.plot(
            xs[to],
            ys[to],
            linestyle="none",
            marker=s["marker"],
            markersize=s.get("markersize"),
            color=s["color"],
            markerfacecolor="white",
            zorder=3,
        )
    return True


def finish(fig, handles, labels, out):
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


def line_legend(styles, timeout):
    handles = [plt.Line2D([], [], **s) for s in styles.values()]
    labels = list(styles)
    if timeout:
        handles.append(
            plt.Line2D(
                [],
                [],
                color="0.3",
                marker="o",
                linestyle="none",
                markerfacecolor="white",
            )
        )
        labels.append(TIMEOUT_LABEL)
    return handles, labels


def grid(nrows, ncols, sharey):
    return plt.subplots(
        nrows,
        ncols,
        figsize=(min(7.0, 2.3 * ncols), 1.9 * nrows),
        squeeze=False,
        sharey=sharey,
    )


def log_xticks(ax, ticks):
    """Log x axis with ticks only at the given (value, label) pairs."""
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([v for v, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks])


def log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def no_data(ax):
    ax.text(
        0.5,
        0.5,
        "no data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="0.5",
    )


def total_figure(rows, combos, styles, x, xlabel, title, out, ticks):
    fig, axes = grid(1, len(combos), sharey=False)
    for ax, (path, dtype) in zip(axes[0], combos):
        sub = [r for r in rows if r["path"] == path and r["dtype"] == dtype]
        drawn = [draw(ax, sub, c, x, "speedup_total", s) for c, s in styles.items()]
        if any(drawn):
            ax.axhline(1.0, color="0.6", linewidth=0.6, zorder=0)
            log_y(ax)
        else:
            no_data(ax)
        log_xticks(ax, ticks)
        ax.set_title(f"{path}, {dtype}, {title}")
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel("all_off time / config time\n(higher is faster)")
    finish(fig, *line_legend(styles, any(r["timeout"] for r in rows)), out)


def step_figure(rows, combos, styles, ds, title, out):
    """speedup_step as bars: groups are distances, one bar per step."""
    width = 0.8 / max(len(styles), 1)
    fig, axes = grid(1, len(combos), sharey=False)
    for ax, (path, dtype) in zip(axes[0], combos):
        sub = [
            r
            for r in rows
            if r["path"] == path and r["dtype"] == dtype and r["speedup_step"] > 0
        ]
        if not sub:
            no_data(ax)
        else:
            vals = [r["speedup_step"] for r in sub]
            # linear y, bars anchored at 1: a slowdown reads as a downward bar
            bottom = 1.0
            for k, (c, s) in enumerate(styles.items()):
                for r in sub:
                    if r["config"] != c:
                        continue
                    xpos = ds.index(r["d"]) - 0.4 + (k + 0.5) * width
                    h = r["speedup_step"] - bottom
                    if r["timeout"]:
                        ax.bar(
                            xpos,
                            h,
                            width,
                            bottom=bottom,
                            facecolor="white",
                            edgecolor=s["color"],
                            hatch="////",
                            linewidth=0.6,
                        )
                    else:
                        ax.bar(xpos, h, width, bottom=bottom, color=s["color"])
            ax.axhline(1.0, color="0.3", linewidth=0.6, zorder=3)
            lo, hi = min(min(vals), 1.0), max(max(vals), 1.0)
            pad = 0.1 * (hi - lo) or 0.1
            ax.set_ylim(lo - pad if lo < 1.0 else 0.0, hi + pad)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlim(-0.6, len(ds) - 0.4)
        ax.set_title(f"{path}, {dtype}, {title}")
        ax.set_xlabel("distance d")
    axes[0][0].set_ylabel("previous step time /\nthis step time")
    handles = [Patch(color=s["color"]) for s in styles.values()]
    labels = list(styles)
    if any(r["timeout"] for r in rows):
        handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="////"))
        labels.append(TIMEOUT_LABEL)
    finish(fig, handles, labels, out)


def lighter(color):
    return tuple(0.5 + 0.5 * v for v in to_rgb(color))


def stage_figure(rows, combos, styles, ds, title, out):
    """bp_ms (solid) with osd_ms (lighter, dotted) stacked on top: groups are
    distances, one bar per ladder step."""
    fig, axes = grid(1, len(combos), sharey=False)
    for ax, (path, dtype) in zip(axes[0], combos):
        sub = [
            r
            for r in rows
            if r["path"] == path and r["dtype"] == dtype and r["bp_ms"] > 0
        ]
        if not sub:
            no_data(ax)
        else:
            configs = [c for c in styles if any(r["config"] == c for r in sub)]
            width = 0.8 / len(configs)
            # linear y from 0, so segment heights are the BP and OSD times
            bottom = 0.0
            for k, c in enumerate(configs):
                color = styles[c]["color"]
                for r in sub:
                    if r["config"] != c:
                        continue
                    xpos = ds.index(r["d"]) - 0.4 + (k + 0.5) * width
                    osd = r["osd_ms"] if r["osd_ms"] > 0 else 0.0
                    if r["timeout"]:
                        bp_kw = dict(facecolor="white", edgecolor=color, hatch="////")
                        osd_kw = dict(
                            facecolor="white", edgecolor=lighter(color), hatch="...."
                        )
                    else:
                        bp_kw = dict(facecolor=color, edgecolor=color)
                        osd_kw = dict(
                            facecolor=lighter(color), edgecolor=color, hatch="...."
                        )
                    ax.bar(
                        xpos, r["bp_ms"] - bottom, width, bottom=bottom,
                        linewidth=0.6, **bp_kw,
                    )
                    if osd:
                        ax.bar(
                            xpos, osd, width, bottom=r["bp_ms"],
                            linewidth=0.6, **osd_kw,
                        )
            ax.set_ylim(bottom=0)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlim(-0.6, len(ds) - 0.4)
        ax.set_title(f"{path}, {dtype}, {title}")
        ax.set_xlabel("distance d")
    axes[0][0].set_ylabel("decode time (ms)")
    handles = [Patch(color=s["color"]) for s in styles.values()]
    labels = list(styles)
    handles += [
        Patch(facecolor="0.3", edgecolor="0.3"),
        Patch(facecolor=lighter("0.3"), edgecolor="0.3", hatch="...."),
    ]
    labels += ["BP (bottom, solid)", "OSD (top, light dotted)"]
    if any(r["timeout"] for r in rows):
        handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="////"))
        labels.append(TIMEOUT_LABEL)
    finish(fig, handles, labels, out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", type=Path)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        raw = list(reader)
    has_stage = {"bp_ms", "osd_ms"} <= set(reader.fieldnames or ())
    if not has_stage:
        print(
            "note: CSV has no bp_ms/osd_ms columns, skipping the stage figure",
            file=sys.stderr,
        )
    rows = []
    for r in raw:
        timed = r["time_ms"] != ""
        # "timeout, N of S shots": time extrapolated from a partial pass
        r["timeout"] = r["status"].startswith("timeout, ") and timed
        # "timeout after K timed passes" with a time: median of K full passes
        if r["status"] == "ok" or (r["status"].startswith("timeout") and timed):
            rows.append(r)
    n_to = sum(r["timeout"] for r in rows)
    if len(raw) > len(rows):
        print(
            f"note: skipped {len(raw) - len(rows)} row(s) with status != ok",
            file=sys.stderr,
        )
    if n_to:
        print(
            f"note: plotted {n_to} timeout row(s) with extrapolated time, drawn hollow",
            file=sys.stderr,
        )
    if not rows:
        sys.exit("no rows to plot")
    for r in rows:
        r["d"] = int(r["d"])
        r["pf"] = float(r["p"])  # r["p"] stays the CSV text for file names
        for k in ("speedup_step", "speedup_total", "ler_mean", "ler_min", "ler_max"):
            r[k] = num(r[k])
        for k in ("bp_ms", "osd_ms"):
            r[k] = num(r.get(k))

    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / args.csv.stem

    combos = sorted({(r["path"], r["dtype"]) for r in rows})
    styles = config_styles({r["config"] for r in rows})
    steps = {c: s for c, s in styles.items() if c != "all_off"}
    ds = sorted({r["d"] for r in rows})
    ps = sorted({r["p"] for r in rows}, key=float)
    d_ticks = [(d, str(d)) for d in ds]
    p_ticks = [(float(p), p) for p in ps]

    # a. speedup_total vs d, one figure per p
    for p in ps:
        total_figure(
            [r for r in rows if r["p"] == p],
            combos,
            steps,
            "d",
            "distance d",
            f"p = {p}",
            f"{stem}-total-vs-d-p{p}.pdf",
            d_ticks,
        )

    # b. speedup_total vs p, one figure per d
    for d in ds:
        total_figure(
            [r for r in rows if r["d"] == d],
            combos,
            steps,
            "pf",
            "physical error rate p",
            f"d = {d}",
            f"{stem}-total-vs-p-d{d}.pdf",
            p_ticks,
        )

    # c. speedup_step bars vs d, one figure per p
    for p in ps:
        step_figure(
            [r for r in rows if r["p"] == p],
            combos,
            steps,
            ds,
            f"p = {p}",
            f"{stem}-step-vs-d-p{p}.pdf",
        )

    # d. bp_ms + osd_ms stacked bars vs d, one figure per p
    if has_stage:
        for p in ps:
            stage_figure(
                [r for r in rows if r["p"] == p],
                combos,
                styles,
                ds,
                f"p = {p}",
                f"{stem}-stage-vs-d-p{p}.pdf",
            )

    # e. LER vs p, rows (path, dtype), columns d
    nonpos = sum(1 for r in rows if not r["ler_mean"] > 0)
    if nonpos:
        print(
            f"note: {nonpos} row(s) with ler_mean <= 0 or empty are not drawn",
            file=sys.stderr,
        )
    fig, axes = grid(len(combos), len(ds), sharey=True)
    for i, (path, dtype) in enumerate(combos):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            sub = [
                r
                for r in rows
                if r["path"] == path and r["dtype"] == dtype and r["d"] == d
            ]
            for c, s in styles.items():
                pts = sorted(
                    (r["pf"], r["ler_mean"], r["ler_min"], r["ler_max"])
                    for r in sub
                    if r["config"] == c
                )
                if not pts:
                    continue
                x, mean, lo, hi = (np.array(v) for v in zip(*pts))
                mean = np.where(mean > 0, mean, np.nan)
                hi = np.where(hi > 0, hi, np.nan)
                # ponytail: a nonpositive ler_min clips the band at the mean, not at 0
                lo = np.where(lo > 0, lo, mean)
                draw(ax, sub, c, "pf", "ler_mean", s)
                ax.fill_between(x, lo, hi, color=s["color"], alpha=0.15, linewidth=0)
            log_xticks(ax, p_ticks)
            ax.set_yscale("log")
            ax.set_title(f"{path}, {dtype}, d = {d}")
            if i == len(combos) - 1:
                ax.set_xlabel("physical error rate p")
            if j == 0:
                ax.set_ylabel("logical error rate")
    finish(fig, *line_legend(styles, n_to > 0), f"{stem}-ler-vs-p.pdf")


if __name__ == "__main__":
    main()
