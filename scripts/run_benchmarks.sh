#!/usr/bin/env bash
# Run the full validation + benchmark matrix and regenerate charts.
#
#   scripts/run_benchmarks.sh [path/to/opt-flow-parallel]
#
# Environment overrides: SIZES, ITERS, THREADS, RUNS, FRAME1, FRAME2.
# Outputs: results/{validation.txt,benchmark.csv,system_info.txt,convergence.csv,*.png,tables.md}
#          output/flow_hsv_<impl>.png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXE="${1:-}"
if [[ -z "$EXE" ]]; then
  for cand in build/Release/opt-flow-parallel.exe build/Release/opt-flow-parallel build/opt-flow-parallel; do
    if [[ -x "$cand" ]]; then EXE="$cand"; break; fi
  done
fi
[[ -x "$EXE" ]] || { echo "error: executable not found; build first (see README) or pass its path" >&2; exit 1; }

SIZES="${SIZES:-640x480,1280x720,1920x1080}"
ITERS="${ITERS:-100,500,1000}"
THREADS="${THREADS:-1,2,4,8,max}"
RUNS="${RUNS:-10}"
FRAMES=("${FRAME1:-data/rm1.jpg}" "${FRAME2:-data/rm2.jpg}")

# Implementations present in this build (CUDA ones only if compiled in).
CPU_IMPLS="serial-gs,serial-jacobi,serial-rb,omp,omp-opt"
GPU_IMPLS="cuda-v1,cuda-v2,cuda-v3,cuda-rb"
if "$EXE" validate "${FRAMES[@]}" --impl cuda-v1 --size 37x5 >/dev/null 2>&1; then HAVE_CUDA=1; else HAVE_CUDA=0; fi

mkdir -p results output

echo "== validation"
if [[ $HAVE_CUDA == 1 ]]; then VAL_IMPLS="all"; else VAL_IMPLS="serial-jacobi,omp-opt"; fi
"$EXE" validate "${FRAMES[@]}" --impl "$VAL_IMPLS" --size "37x5,641x479,$SIZES" --iters "$ITERS" \
  | tee results/validation.txt

echo "== benchmark: CPU"
"$EXE" bench "${FRAMES[@]}" --impl "$CPU_IMPLS" --size "$SIZES" --iters "$ITERS" \
  --threads "$THREADS" --runs "$RUNS" --warmup 1 --csv results/benchmark.csv

if [[ $HAVE_CUDA == 1 ]]; then
  echo "== benchmark: GPU"
  "$EXE" bench "${FRAMES[@]}" --impl "$GPU_IMPLS" --size "$SIZES" --iters "$ITERS" \
    --runs "$RUNS" --warmup 3 --csv results/benchmark.csv --append
fi

echo "== convergence"
CONV_IMPLS="serial-gs,serial-jacobi,serial-rb"
[[ $HAVE_CUDA == 1 ]] && CONV_IMPLS="$CONV_IMPLS,cuda-v3,cuda-rb"
"$EXE" convergence "${FRAMES[@]}" --impl "$CONV_IMPLS" --size 640x480 --iters 1000 \
  --csv results/convergence.csv

echo "== flow images"
FLOW_IMPLS="serial-gs,omp"
[[ $HAVE_CUDA == 1 ]] && FLOW_IMPLS="$FLOW_IMPLS,cuda-v3,cuda-rb"
"$EXE" flow "${FRAMES[@]}" --impl "$FLOW_IMPLS" --iters 1000 --out output

echo "== charts"
PY=""
for cand in python3 python; do
  if "$cand" -c "import matplotlib" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [[ -n "$PY" ]]; then
  "$PY" scripts/plot_results.py --results results
else
  echo "matplotlib not found: run scripts/plot_results.py manually" >&2
fi
