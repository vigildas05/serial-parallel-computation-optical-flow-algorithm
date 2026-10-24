#!/usr/bin/env python3
"""Summarize Nsight Compute raw CSV exports (results/profile/ncu_<impl>.csv)
into a markdown table (results/profile/summary.md).

Each CSV holds several launches of one iteration kernel; the table reports the
median over launches. Derived column:
  DRAM bytes / pixel = (DRAM bytes read + written) / interior pixels,
  per kernel launch (red-black runs two launches per iteration).

Usage: python scripts/summarize_ncu.py results/profile --size 1920x1080
"""
import argparse
import csv
import statistics
from pathlib import Path

IMPLS = ["cuda-v1", "cuda-v2", "cuda-v3", "cuda-rb"]

# ncu prints either long ("nsecond", "Kbyte/second") or short ("ns", "KB/s",
# "byte/s") unit names depending on version; accept both.
TIME = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6,
        "ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}  # -> microseconds
BYTES = {}
for long_name, short_name, factor in (("byte", "B", 1), ("Kbyte", "KB", 1e3), ("Mbyte", "MB", 1e6),
                                      ("Gbyte", "GB", 1e9), ("Tbyte", "TB", 1e12)):
    BYTES[long_name] = BYTES[short_name] = BYTES[long_name.replace("byte", "Byte")] = factor
RATE = {}  # -> GB/s
for name, factor in BYTES.items():
    RATE[f"{name}/second"] = RATE[f"{name}/s"] = factor / 1e9


def load(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        raise ValueError(f"{path}: no ncu raw CSV header found (did ncu fail? check the file)")
    reader = csv.reader(lines[start:])
    header = next(reader)
    units = next(reader)
    rows = [dict(zip(header, r)) for r in reader if len(r) == len(header)]
    return rows, dict(zip(header, units))


def value(row, units, metric, table=None):
    x = float(row[metric].replace(",", ""))
    if table is not None:
        unit = units[metric]
        if unit not in table:
            raise KeyError(f"unknown unit '{unit}' for {metric}")
        x *= table[unit]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--size", default="1920x1080")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    interior = (w - 2) * (h - 2)

    out = [f"Nsight Compute, {args.size}, iteration kernel, median over sampled launches "
           "(GPU at its normal clocks).\n",
           "| version | duration (µs) | achieved occupancy | theoretical occupancy | "
           "DRAM throughput (% peak) | DRAM GB/s | L1 throughput (% peak) | L2 throughput (% peak) | "
           "memory SOL (% peak) | SM SOL (% peak) | DRAM bytes / pixel | L1 hit | L2 hit |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for impl in IMPLS:
        path = args.profile_dir / f"ncu_{impl}.csv"
        if not path.exists():
            continue
        rows, units = load(path)
        if not rows:
            print(f"{path}: no kernel launches recorded")
            continue

        def med(metric, table=None):
            return statistics.median(value(r, units, metric, table) for r in rows)

        dram_bytes = med("dram__bytes_read.sum", BYTES) + med("dram__bytes_write.sum", BYTES)
        out.append(
            f"| {impl} | {med('gpu__time_duration.sum', TIME):.1f} | "
            f"{med('sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}% | "
            f"{med('sm__maximum_warps_per_active_cycle_pct'):.1f}% | "
            f"{med('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{med('dram__bytes.sum.per_second', RATE):.1f} | "
            f"{med('l1tex__throughput.avg.pct_of_peak_sustained_active'):.1f}% | "
            f"{med('lts__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{med('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{med('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{dram_bytes / interior:.1f} | "
            f"{med('l1tex__t_sector_hit_rate.pct'):.1f}% | {med('lts__t_sector_hit_rate.pct'):.1f}% |")
    out.append(f"\nSampled launches per kernel: "
               + ", ".join(f"{i}: {len(load(args.profile_dir / f'ncu_{i}.csv')[0])}"
                           for i in IMPLS if (args.profile_dir / f"ncu_{i}.csv").exists()))
    (args.profile_dir / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
