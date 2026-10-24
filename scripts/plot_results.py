#!/usr/bin/env python3
"""Turn results/benchmark.csv and results/convergence.csv into charts and tables.

Outputs (in the results directory):
  speedup.png          speedup of every implementation vs the serial Gauss-Seidel baseline
  thread_scaling.png   OpenMP speedup vs thread count (relative to serial Jacobi)
  convergence.png      fixed-point residual vs iteration, and vs measured time
  tables.md            markdown tables generated from the CSVs (pasted into the README)

Every number in the outputs comes from the CSV files; nothing is estimated
except where a chart says so explicitly (residual vs time multiplies the
iteration index by the measured mean time per iteration).

Usage: python scripts/plot_results.py [--results results]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Validated reference palette (light mode): categorical slots in fixed order,
# blue ordinal ramp for resolutions, neutral ink/grid tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RAMP = ["#86b6ef", "#3987e5", "#184f95"]  # small -> large resolution
REFERENCE = "#8c8b86"

BAR_IMPLS = ["serial-jacobi", "serial-rb", "omp", "omp-opt", "cuda-v1", "cuda-v2", "cuda-v3", "cuda-rb"]
LABELS = {
    "serial-gs": "serial GS",
    "serial-jacobi": "serial Jacobi",
    "serial-rb": "serial red-black",
    "omp": "OpenMP (orig.)",
    "omp-opt": "OpenMP (opt.)",
    "cuda-v1": "CUDA v1",
    "cuda-v2": "CUDA v2",
    "cuda-v3": "CUDA v3",
    "cuda-rb": "CUDA red-black",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "font.size": 9.5,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def read_bench(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("width", "height", "iterations", "runs"):
                r[k] = int(r[k])
            r["threads"] = int(r["threads"]) if r["threads"] else None
            for k in ("h2d_ms", "compute_ms", "d2h_ms", "total_ms", "total_min_ms", "total_max_ms"):
                r[k] = float(r[k]) if r[k] else None
            rows.append(r)
    return rows


def max_threads(rows, impl):
    t = [r["threads"] for r in rows if r["impl"] == impl and r["threads"]]
    return max(t) if t else None


def lookup(rows, impl, w, h, iters, threads=None):
    for r in rows:
        if (r["impl"], r["width"], r["height"], r["iterations"]) == (impl, w, h, iters) and \
                (threads is None or r["threads"] == threads):
            return r
    return None


def resolutions(rows):
    return sorted({(r["width"], r["height"]) for r in rows}, key=lambda s: s[0] * s[1])


def times_fmt(x, _):
    return f"{x:g}×"


def plot_speedup(rows, out):
    iters_list = sorted({r["iterations"] for r in rows})
    res = resolutions(rows)
    ramp = RAMP if len(res) <= 3 else plt.get_cmap("Blues")([0.35 + 0.6 * i / (len(res) - 1) for i in range(len(res))])
    impls = [i for i in BAR_IMPLS if any(r["impl"] == i for r in rows)]
    tmax = {i: max_threads(rows, i) for i in ("omp", "omp-opt")}

    fig, axes = plt.subplots(1, len(iters_list), figsize=(4.6 * len(iters_list), 4.4), sharey=True, squeeze=False)
    group_w = 0.8
    bar_w = group_w / len(res)
    for ax, iters in zip(axes[0], iters_list):
        for k, (w, h) in enumerate(res):
            base = lookup(rows, "serial-gs", w, h, iters)
            if not base:
                continue
            xs, ys = [], []
            for j, impl in enumerate(impls):
                r = lookup(rows, impl, w, h, iters, tmax.get(impl))
                if r:
                    xs.append(j - group_w / 2 + (k + 0.5) * bar_w)
                    ys.append(base["total_ms"] / r["total_ms"])
            # 2px surface gap between adjacent bars
            ax.bar(xs, ys, width=bar_w, color=ramp[k], edgecolor=SURFACE, linewidth=1.5,
                   label=f"{w}×{h}", zorder=2)
            # direct-label the largest resolution only (selective labels)
            if k == len(res) - 1:
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.0f}×" if y >= 10 else f"{y:.1f}×", (x, y), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=7.5, color=INK_2)
        ax.axhline(1.0, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(times_fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(len(impls)))
        ax.set_xticklabels([LABELS[i] for i in impls], rotation=35, ha="right")
        ax.set_title(f"{iters} iterations")
    axes[0][0].set_ylabel("speedup vs serial Gauss–Seidel (log scale)")
    axes[0][-1].legend(title="resolution", title_fontsize=8.5, loc="upper left")
    fig.suptitle("End-to-end speedup over the serial Gauss–Seidel baseline", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.905, "GPU times include H2D + derivatives + iterations + D2H. OpenMP at max threads. "
             "Labels: largest resolution.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "speedup.png", dpi=160)
    plt.close(fig)


def physical_cores(results_dir):
    p = results_dir / "system_info.txt"
    if p.exists():
        m = re.search(r"\((\d+) cores", p.read_text())
        if m:
            return int(m.group(1))
    return None


def plot_thread_scaling(rows, out, cores):
    iters = max(r["iterations"] for r in rows)
    res = resolutions(rows)
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.8), sharey=True, squeeze=False)
    for ax, (w, h) in zip(axes[0], res):
        base = lookup(rows, "serial-jacobi", w, h, iters)
        if not base:
            continue
        all_t = []
        for color, impl in zip(SERIES, ("omp", "omp-opt")):
            pts = sorted((r["threads"], base["total_ms"] / r["total_ms"]) for r in rows
                         if r["impl"] == impl and (r["width"], r["height"], r["iterations"]) == (w, h, iters))
            if not pts:
                continue
            t, s = zip(*pts)
            all_t += t
            ax.plot(t, s, color=color, linewidth=2, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[impl], zorder=3)
            ax.annotate(f"{s[-1]:.1f}×", (t[-1], s[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        if all_t:
            tt = sorted(set(all_t))
            ax.plot(tt, tt, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)), label="ideal (linear)", zorder=2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(tt)
            ax.set_xticklabels([str(x) for x in tt])
            ax.set_xlim(min(tt) / 1.3, max(tt) * 1.6)
        if cores:
            ax.axvline(cores, color=GRID, linewidth=1.5, zorder=1)
            ax.annotate(f"{cores} physical cores", (cores, ax.get_ylim()[1]), xytext=(-4, -4),
                        textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK_2)
        ax.set_title(f"{w}×{h}")
        ax.set_xlabel("OpenMP threads")
    axes[0][0].set_ylabel("speedup vs serial Jacobi")
    axes[0][0].legend(loc="upper left")
    fig.suptitle(f"OpenMP thread scaling ({iters} iterations)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "thread_scaling.png", dpi=160)
    plt.close(fig)


def read_convergence(path):
    curves = defaultdict(list)
    size = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            curves[r["impl"]].append((int(r["iteration"]), float(r["residual"])))
            size = (int(r["width"]), int(r["height"]))
    return {k: sorted(v) for k, v in curves.items()}, size


def plot_convergence(curves, size, rows, out):
    # One curve per iteration scheme; GPU runs of the same scheme are checked
    # to coincide with their CPU reference and reported, not drawn twice.
    schemes = [("serial-gs", "Gauss–Seidel (lexicographic)"),
               ("serial-jacobi", "Jacobi"),
               ("serial-rb", "red-black Gauss–Seidel")]
    for gpu, cpu in (("cuda-v3", "serial-jacobi"), ("cuda-rb", "serial-rb")):
        if gpu in curves and cpu in curves:
            rel = max(abs(a[1] - b[1]) / b[1] for a, b in zip(curves[gpu], curves[cpu]) if b[1] > 0)
            print(f"convergence: {gpu} vs {cpu}: max relative residual difference {rel:.2e}")

    scheme_color = {impl: color for color, (impl, _) in zip(SERIES, schemes)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ends = []
    for impl, label in schemes:
        if impl not in curves:
            continue
        it, res = zip(*curves[impl])
        ax1.plot(it, res, color=scheme_color[impl], linewidth=2, label=label)
        ends.append([res[-1], label, it[-1]])
    # Direct end labels, nudged apart in log space so they never overlap.
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] * 1.7)
    for y, label, x in ends:
        ax1.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8, color=INK_2)
    ax1.set_yscale("log")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("fixed-point residual (RMS, px)")
    ax1.set_title("Residual vs iteration")
    ax1.set_xlim(0, max(i for c in curves.values() for i, _ in c) * 1.42)
    ax1.legend(loc="upper right")

    # Residual vs time: iteration k plotted at k * (measured compute time /
    # measured iterations) for the same resolution and largest iteration count.
    # Color = iteration scheme (as on the left), dash = implementation.
    timed = [("serial-gs", "serial-gs", None, "-"), ("serial-jacobi", "omp-opt", "max", (0, (1.5, 1.5))),
             ("serial-jacobi", "cuda-v3", None, "-"), ("serial-rb", "cuda-rb", None, "-")]
    drew = False
    if rows and size:
        w, h = size
        iters = max((r["iterations"] for r in rows if (r["width"], r["height"]) == (w, h)), default=None)
        for scheme, impl, thr, dash in timed:
            t = max_threads(rows, impl) if thr == "max" else None
            r = lookup(rows, impl, w, h, iters, t) if iters else None
            if not r or scheme not in curves:
                continue
            per_iter = r["compute_ms"] / r["iterations"]
            pts = [(k * per_iter, res) for k, res in curves[scheme] if k > 0]
            name = LABELS[impl] + (f" ({t} threads)" if t else "")
            ax2.plot(*zip(*pts), color=scheme_color[scheme], linewidth=2, linestyle=dash, label=name)
            drew = True
    if drew:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("time (ms) = iteration × measured time per iteration\n"
                       f"(per-iteration time from the {iters}-iteration benchmark)")
        ax2.set_title("Residual vs time to reach it")
        ax2.legend(loc="lower left")
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5, f"no benchmark rows at {size[0]}×{size[1]}\nto convert iterations to time",
                 ha="center", va="center", color=INK_2, transform=ax2.transAxes)

    fig.suptitle(f"Convergence at {size[0]}×{size[1]}", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "convergence.png", dpi=160)
    plt.close(fig)


def fmt(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def write_tables(rows, out):
    lines = []
    tmax = {i: max_threads(rows, i) for i in ("omp", "omp-opt")}
    order = ["serial-gs", "serial-jacobi", "serial-rb", "omp", "omp-opt", "cuda-v1", "cuda-v2", "cuda-v3", "cuda-rb"]
    runs = sorted({r["runs"] for r in rows})
    lines.append(f"Median of {', '.join(map(str, runs))} timed runs per configuration, after warm-up. "
                 "All times in ms. Speedup = serial-gs total / total.\n")
    for (w, h) in resolutions(rows):
        lines.append(f"#### {w}×{h}\n")
        lines.append("| implementation | threads | iterations | total | compute | H2D | D2H | speedup vs serial-gs | speedup vs serial-jacobi |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for iters in sorted({r["iterations"] for r in rows}):
            gs = lookup(rows, "serial-gs", w, h, iters)
            sj = lookup(rows, "serial-jacobi", w, h, iters)
            for impl in order:
                r = lookup(rows, impl, w, h, iters, tmax.get(impl))
                if not r:
                    continue
                sp1 = f"{gs['total_ms'] / r['total_ms']:.1f}×" if gs else ""
                sp2 = f"{sj['total_ms'] / r['total_ms']:.1f}×" if sj else ""
                lines.append(f"| {impl} | {r['threads'] or '–'} | {iters} | {fmt(r['total_ms'])} | "
                             f"{fmt(r['compute_ms'])} | {fmt(r['h2d_ms'])} | {fmt(r['d2h_ms'])} | {sp1} | {sp2} |")
        lines.append("")

    lines.append("#### OpenMP thread scaling (total ms; speedup vs serial-jacobi in parentheses)\n")
    iters = max(r["iterations"] for r in rows)
    threads = sorted({r["threads"] for r in rows if r["impl"] in ("omp", "omp-opt") and r["threads"]})
    lines.append("| implementation | resolution | " + " | ".join(f"{t} thr" for t in threads) + " |")
    lines.append("|---|---|" + "---:|" * len(threads))
    for (w, h) in resolutions(rows):
        sj = lookup(rows, "serial-jacobi", w, h, iters)
        for impl in ("omp", "omp-opt"):
            cells = []
            for t in threads:
                r = lookup(rows, impl, w, h, iters, t)
                cells.append("" if not r else f"{r['total_ms']:.1f}" + (f" ({sj['total_ms'] / r['total_ms']:.1f}×)" if sj else ""))
            if any(cells):
                lines.append(f"| {impl} | {w}×{h} | " + " | ".join(cells) + " |")
    lines.append(f"\n({iters} iterations)\n")
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", type=Path)
    args = ap.parse_args()
    style()

    rows = []
    bench = args.results / "benchmark.csv"
    if bench.exists():
        rows = read_bench(bench)
        plot_speedup(rows, args.results)
        plot_thread_scaling(rows, args.results, physical_cores(args.results))
        write_tables(rows, args.results)
        print(f"wrote speedup.png, thread_scaling.png, tables.md from {bench}")
    else:
        print(f"skipping benchmark charts: {bench} not found")

    conv = args.results / "convergence.csv"
    if conv.exists():
        curves, size = read_convergence(conv)
        plot_convergence(curves, size, rows, args.results)
        print(f"wrote convergence.png from {conv}")
    else:
        print(f"skipping convergence chart: {conv} not found")


if __name__ == "__main__":
    main()
