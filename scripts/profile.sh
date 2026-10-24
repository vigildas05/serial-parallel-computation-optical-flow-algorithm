#!/usr/bin/env bash
# Profile the CUDA kernels with Nsight Compute and the full pipeline with
# Nsight Systems.
#
#   scripts/profile.sh [path/to/opt-flow-parallel]
#
# Environment overrides: SIZE (default 1920x1080), NCU, NSYS.
# Outputs in results/profile/:
#   ncu_<impl>.csv       raw per-launch metrics for the iteration kernel
#   ncu_<impl>.ncu-rep   full Nsight Compute report (one launch) for the GUI
#   summary.md           table built from the CSVs by scripts/summarize_ncu.py
#   nsys_pipeline.*      Nsight Systems timeline + CSV summaries. The .nsys-rep
#                        embeds the process environment (may contain tokens):
#                        it is git-ignored, do not share it publicly.
#
# Nsight Compute needs access to GPU performance counters: on Linux run as
# root or set NVreg_RestrictProfilingToAdminUsers=0; on Windows enable
# "Allow access to the GPU performance counters to all users" in the NVIDIA
# Control Panel (Developer > Manage GPU Performance Counters) or run as admin.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXE="${1:-}"
if [[ -z "$EXE" ]]; then
  for cand in build/Release/opt-flow-parallel.exe build/Release/opt-flow-parallel build/opt-flow-parallel; do
    if [[ -x "$cand" ]]; then EXE="$cand"; break; fi
  done
fi
[[ -x "$EXE" ]] || { echo "error: executable not found; build first or pass its path" >&2; exit 1; }

find_tool() { # name, glob of fallback locations
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return; fi
  for p in "$@"; do
    # Git Bash does not mark Windows .bat launchers as executable.
    if [[ -x "$p" || ( "$p" == *.bat && -f "$p" ) ]]; then echo "$p"; return; fi
  done
  return 1
}
shopt -s nullglob
NCU="${NCU:-$(find_tool ncu /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu \
  "/c/Program Files/NVIDIA Corporation/Nsight Compute "*/ncu.bat || true)}"
NSYS="${NSYS:-$(find_tool nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys \
  "/c/Program Files/NVIDIA Corporation/Nsight Systems "*/target-windows-x64/nsys.exe || true)}"
shopt -u nullglob

SIZE="${SIZE:-1920x1080}"
OUT=results/profile
mkdir -p "$OUT"

# Metrics (Nsight Compute names):
#   duration, achieved/theoretical occupancy, DRAM throughput as % of peak and
#   in bytes/s, L1 / L2 / overall memory and SM throughput as % of peak
#   (speed-of-light), DRAM bytes moved, and L1/L2 hit rates.
# Collected with --clock-control none, i.e. at the same (boost) clocks the
# benchmarks run at; ncu's default locks base clocks, which lowers the SM/L1
# clock relative to DRAM and shifts the bottleneck.
METRICS="gpu__time_duration.sum"
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct"
METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second"
METRICS+=",dram__bytes_read.sum,dram__bytes_write.sum"
METRICS+=",gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS+=",l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"

declare -A KERNEL=(
  [cuda-v1]=jacobiNaiveKernel
  [cuda-v2]=jacobiSharedKernel
  [cuda-v3]=jacobiFusedKernel
  [cuda-rb]=redBlackKernel
)

if [[ -n "$NCU" ]]; then
  echo "== Nsight Compute: $NCU"
  for impl in cuda-v1 cuda-v2 cuda-v3 cuda-rb; do
    k="${KERNEL[$impl]}"
    # Skip the first launches (cold caches / clocks ramping), sample 10.
    "$NCU" --clock-control none --kernel-name-base function --kernel-name "regex:$k" \
      --launch-skip 10 --launch-count 10 --metrics "$METRICS" --csv --page raw \
      --log-file "$OUT/ncu_$impl.csv" \
      "$EXE" bench --impl "$impl" --size "$SIZE" --iters 30 --runs 1 --warmup 0 >/dev/null
    # One fully detailed launch for interactive inspection in ncu-ui.
    "$NCU" --clock-control none --kernel-name-base function --kernel-name "regex:$k" \
      --launch-skip 10 --launch-count 1 --set full --import-source yes -f -o "$OUT/ncu_$impl" \
      "$EXE" bench --impl "$impl" --size "$SIZE" --iters 30 --runs 1 --warmup 0 >/dev/null
    echo "  $impl -> $OUT/ncu_$impl.csv, $OUT/ncu_$impl.ncu-rep"
  done
  PY=""
  for cand in python3 python; do
    if "$cand" -c "import csv" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
  [[ -n "$PY" ]] && "$PY" scripts/summarize_ncu.py "$OUT" --size "$SIZE"
else
  echo "ncu not found - skipping kernel metrics (set NCU=/path/to/ncu)" >&2
fi

if [[ -n "$NSYS" ]]; then
  echo "== Nsight Systems: $NSYS"
  # OS runtime tracing (osrt) exists only on Linux.
  TRACE=cuda
  [[ "$(uname -s)" == Linux ]] && TRACE=cuda,osrt
  "$NSYS" profile --trace="$TRACE" --force-overwrite true -o "$OUT/nsys_pipeline" \
    "$EXE" bench --impl cuda-v1,cuda-v2,cuda-v3,cuda-rb --size "$SIZE" --iters 100 --runs 3 --warmup 1 \
    > "$OUT/nsys_run.txt"
  "$NSYS" stats --force-export true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    --format csv --output "$OUT/nsys_pipeline" "$OUT/nsys_pipeline.nsys-rep" >/dev/null
  echo "  timeline -> $OUT/nsys_pipeline.nsys-rep (open in nsys-ui), summaries -> $OUT/nsys_pipeline_*.csv"
else
  echo "nsys not found - skipping timeline (set NSYS=/path/to/nsys)" >&2
fi
