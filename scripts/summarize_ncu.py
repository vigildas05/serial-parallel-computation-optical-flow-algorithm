#!/usr/bin/env python3
"""Summarize Nsight Compute raw CSV exports (results/profile/ncu_<impl>.csv)
into a markdown table (results/profile/summary.md).

Each CSV holds several launches of one iteration kernel; the table reports the
median over launches. Derived columns:
  DRAM bytes / pixel = (DRAM bytes read + written) / interior pixels,
  per kernel launch (red-black runs two launches per iteration).

Usage: python scripts/summarize_ncu.py results/profile --size 1920x1080
"""
import argparse
import csv
import statistics
from pathlib import Path

IMPLS = ["cuda-v1", "cuda-v2", "cuda-v3", "cuda-rb"]

TIME = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6}  # -> microseconds
BYTES = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "KByte": 1e3, "MByte": 1e6, "GByte": 1e9}
RATE = {f"{k}/second": v / 1e9 for k, v in BYTES.items()}  # -> GB/s


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
    raw = row[metric].replace(",", "")
    x = float(raw)
    if table is not None:
        x *= table[units[metric]]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--size", default="1920x1080")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    interior = (w - 2) * (h - 2)

    out = [f"Nsight Compute, {args.size}, median over sampled launches of the iteration kernel.\n",
           "| version | kernel | duration (µs) | achieved occupancy | theoretical occupancy | "
           "DRAM throughput (% peak) | DRAM GB/s | memory SOL (% peak) | SM SOL (% peak) | "
           "DRAM bytes / pixel | L1 hit | L2 hit | launches |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
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
        kernel = rows[0].get("Kernel Name", "?")
        out.append(
            f"| {impl} | `{kernel}` | {med('gpu__time_duration.sum', TIME):.1f} | "
            f"{med('sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}% | "
            f"{med('sm__maximum_warps_per_active_cycle_pct'):.1f}% | "
            f"{med('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{med('dram__bytes.sum.per_second', RATE):.1f} | "
            f"{med('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{med('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | "
            f"{dram_bytes / interior:.1f} | "
            f"{med('l1tex__t_sector_hit_rate.pct'):.1f}% | {med('lts__t_sector_hit_rate.pct'):.1f}% | "
            f"{len(rows)} |")
    (args.profile_dir / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
