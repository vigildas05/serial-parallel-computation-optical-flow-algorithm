# Horn–Schunck Optical Flow: Serial, OpenMP and CUDA

Dense optical flow with the Horn–Schunck (HS) method in C++17 / OpenCV, implemented
four ways and measured against each other:

| family | versions | iteration scheme |
|---|---|---|
| serial CPU | `serial-gs`, `serial-jacobi`, `serial-rb` | Gauss–Seidel (lexicographic), Jacobi, red-black Gauss–Seidel |
| OpenMP CPU | `omp` (original), `omp-opt` (restructured) | Jacobi |
| CUDA GPU | `cuda-v1`, `cuda-v2`, `cuda-v3` | Jacobi (naive → shared-memory tiling → fused data layout) |
| CUDA GPU | `cuda-rb` | red-black Gauss–Seidel |

Every implementation solves exactly the same discrete problem. Each one is checked
automatically against a CPU reference before it's benchmarked (see [Correctness](#correctness)).

**Headline:** TODO: fill after run

---

## Contents
- [Algorithm](#algorithm)
- [Architecture: how each version parallelizes](#architecture-how-each-version-parallelizes)
- [Correctness](#correctness)
- [Build](#build)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Profiling](#profiling)
- [Flow visualizations](#flow-visualizations)
- [Caveats](#caveats)
- [Files](#files)

---

## Algorithm

HS assumes:
1. **Brightness constancy:** a point keeps its intensity between frames.
2. **Smooth flow:** neighbouring pixels move similarly.

It minimizes a data term (brightness constancy, via image derivatives Ix, Iy, It)
plus α² times a smoothness term. The resulting equations are solved iteratively.

**Discretization used here** (identical in all versions):

- Frames are grayscale, float32 in [0, 1].
- Both frames are smoothed with a 5×5 Gaussian (σ = 1, replicated border).
- `Ix`, `Iy` = 3×3 Sobel of the smoothed first frame, scaled by 0.5. `It` = I2s − I1s.
- Neighbour average ū, v̄ = **4-neighbour mean** (weights ¼; not the original HS 1/6 + 1/12 kernel).
- Update for every interior pixel:

  ```
  t  = (Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
  u' = ū − Ix·t
  v' = v̄ − Iy·t
  ```
- **Boundary:** the one-pixel border is never updated and stays at zero flow.
- Defaults: α = 0.5, 100 iterations, zero initial flow.

**Iteration schemes**

| scheme | update order | parallelism |
|---|---|---|
| Jacobi | every pixel reads only the previous iteration (two buffers, swapped) | fully parallel, deterministic |
| Gauss–Seidel (lexicographic) | in place, row by row; uses values already updated this sweep | inherently sequential |
| Red-black Gauss–Seidel | in place, checkerboard: all "red" pixels, then all "black" | each half-sweep is fully parallel |

Gauss–Seidel variants converge in fewer iterations than Jacobi. Only Jacobi and red-black
can use many cores. Which one gives the solution sooner in **wall-clock time** is a question
for measurement; see [Convergence](#convergence).

---

## Architecture: how each version parallelizes

### OpenMP (CPU)
- **`omp` (original):** `#pragma omp parallel for collapse(2)` over the interior (y, x) loop
  of each Jacobi sweep, using `cv::Mat::at<float>`. The border rows and columns are copied
  every iteration.
- **`omp-opt`:** the same arithmetic, with bit-identical results, restructured for throughput:
  - `parallel for` over **rows only**. Each thread gets a contiguous block of rows, so
    every thread streams through memory in order.
  - The inner loop uses raw row pointers (`ptr<float>(y-1/y/y+1)`). With no per-pixel
    index division from `collapse` and no `Mat` indirection, the compiler can keep the
    pointers in registers and vectorize.
  - No per-iteration border copies (the zero border never changes).
- The thread count is set per run with `omp_set_num_threads`. OpenCV's internal threading
  (used for the Gaussian/Sobel step) is set to the same count, so "1 thread" really is serial.

### CUDA (GPU)

The pipeline runs entirely on the device. The only host↔device traffic is one upload of the
two frames and one download of (u, v):

```
H2D: I1, I2 ─► Gaussian row/col (×2) ─► derivatives: Ix, Iy, It, α²+Ix²+Iy²
          ─► [v3/rb: fold coefficients] ─► N × iteration kernel (ping-pong or in place)
          ─► [v3/rb: split u,v] ─► D2H: u, v
```

- **Derivatives on the GPU** mirror OpenCV's separable filters: symmetric taps are evaluated
  in the same order, and `__fmul_rn`/`__fadd_rn` stop FMA contraction. The 5 Gaussian taps
  come from `cv::getGaussianKernel` and are stored in `__constant__` memory.
- **Thread mapping:** one thread per pixel, 32×8 blocks, `threadIdx.x` along columns. A warp
  therefore reads one contiguous 128-byte row segment (coalesced).
- **Ping-pong buffers:** Jacobi kernels read buffer *cur* and write *nxt*. The host swaps
  indices, so nothing is copied between iterations. Border threads write nothing, and both
  buffers start zeroed, which reproduces the CPU's fixed zero border.
- **`hs::gpu::Solver`** owns all device buffers for one image size, so benchmarks never time
  `cudaMalloc`.
- **Error handling:** every runtime call is wrapped in `CUDA_CHECK`, and every launch is
  followed by `CUDA_CHECK_LAUNCH` (`cudaGetLastError`, plus `cudaDeviceSynchronize` in debug builds).

| version | what changes | idea |
|---|---|---|
| `cuda-v1` naive | all neighbour and coefficient reads straight from global memory | baseline: correct and simple |
| `cuda-v2` shared | each block stages its 32×8 u/v tile + 1-pixel halo in shared memory | read each u/v value from global memory once per block instead of up to 5× |
| `cuda-v3` fused layout | u,v interleaved as `float2`; coefficients folded to p,q,r = (Ix,Iy,It)/√(α²+Ix²+Iy²) so `t = p·ū + q·v̄ + r; u' = ū − p·t; v' = v̄ − q·t`; `__restrict__` + `__ldg` | the same math with no division and 3 instead of 4 coefficient arrays: minimum DRAM traffic per pixel per sweep drops from 32 B to 28 B |
| `cuda-rb` red-black | v3 layout, updated **in place** in two launches per iteration (red, then black); one thread per pixel of the active colour | Gauss–Seidel convergence with GPU parallelism; no race, because a pixel's 4 neighbours all have the other colour |

A note on "fused": the update for u and v share `t`, so every version (even v1) updates u and
v in one kernel. Splitting them would only recompute `t` twice. In v3, "fused" refers to the
**data**: u and v in one `float2` stream, and four coefficient arrays folded into three.

---

## Correctness

`opt-flow-parallel validate` runs each implementation on the same input and iteration count
and compares it with a reference:

| implementation | reference | criterion |
|---|---|---|
| `serial-jacobi`, `omp-opt` | `omp` (original OpenMP Jacobi) | **bit-identical** |
| `cuda-v1`, `cuda-v2`, `cuda-v3` | `omp` | max \|Δu\|, \|Δv\| ≤ 1e-4 px and mean endpoint error ≤ 1e-6 px |
| `cuda-rb` | `serial-rb` (same update order on the CPU) | same float tolerance |

The CUDA tolerance allows for float32 evaluation-order and FMA differences. It sits about 3×
above the worst error measured for v1, so a real indexing or boundary bug would still fail.
Partial-block sizes (37×5, 641×479) are included to exercise grid and halo edges. The
refactor of the original serial and OpenMP code was also checked to be bit-identical to the
pre-refactor output.

Latest validation output: [`results/validation.txt`](results/validation.txt).
Worst-case errors: TODO: fill after run

---

## Build

Requirements: C++17 compiler, CMake ≥ 3.24, OpenCV 4 (core, imgproc, imgcodecs, highgui),
OpenMP. Optional: CUDA Toolkit (tested with 13.0) for the GPU versions.

CUDA is **optional**. CMake uses `check_language(CUDA)` and builds CPU-only when no CUDA
compiler is found.

### Linux

```bash
sudo apt install build-essential cmake libopencv-dev    # + CUDA toolkit for GPU versions
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build -j
```

The executable is `build/opt-flow-parallel`.

### Windows (MSVC)

nvcc on Windows requires MSVC as its host compiler, so build everything with Visual Studio
2022 (Build Tools are enough). Use the official OpenCV Windows package (`opencv-4.x-windows.exe`,
which contains MSVC `vc16` libraries).

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/tools/opencv/build
```

```bash
cmake --build build --config Release
```

The executable is `build/Release/opt-flow-parallel.exe`; the OpenCV DLL is copied next to it.
MSVC's LLVM OpenMP runtime (`/openmp:llvm`) is selected automatically, because the default
MSVC runtime only supports OpenMP 2.0 and ignores `collapse`.

### CPU-only, or a specific GPU architecture

```bash
cmake -S . -B build -DOFP_ENABLE_CUDA=OFF
```

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the GPUs in the build machine).

### No NVIDIA GPU?

[`colab/run_on_colab.ipynb`](colab/run_on_colab.ipynb) clones the repo, builds it with CMake +
nvcc on a Colab GPU, runs validation, benchmarks and profiling, and shows the charts.

---

## Usage

```bash
./run.sh
```

`./run.sh` is the original demo: serial + OpenMP on `data/rm1.jpg`, `data/rm2.jpg`, with
arrow plots and a GUI window.

```text
opt-flow-parallel <command> [frame1 frame2] [options]

  bench        time every (impl, size, iterations, threads) combination
  validate     compare Jacobi variants against OpenMP Jacobi, cuda-rb against serial-rb
  convergence  record the fixed-point residual after every iteration
  flow         save HSV color-coded flow images

  --impl LIST     serial-gs, serial-jacobi, serial-rb, omp, omp-opt, cuda-v1, cuda-v2, cuda-v3, cuda-rb, all
  --size LIST     WxH or native            --iters LIST     iteration counts
  --threads LIST  OpenMP threads, max = omp_get_max_threads()
  --runs N        timed runs (median)      --warmup N       untimed warm-up runs
  --alpha A       regularization           --csv FILE       write CSV   --append
```

Example:

```bash
build/Release/opt-flow-parallel.exe bench --impl omp-opt,cuda-v3 --size 1920x1080 --iters 1000 --threads 1,8,max
```

Everything in [Benchmarks](#benchmarks) is regenerated by:

```bash
scripts/run_benchmarks.sh
```

The profiling data comes from:

```bash
scripts/profile.sh
```

---

## Benchmarks

### Method
- **GPU timing:** `cudaEvent`s on the solver's stream. Reported separately:
  - **H2D**: upload of both float32 frames
  - **compute**: Gaussian, derivatives, all iterations, layout conversion
  - **D2H**: download of u and v
  - **total**: first upload to last download
  Device allocation happens once per image size and is not timed. Host buffers are
  ordinary pageable memory (no pinned staging).
- **CPU timing:** `std::chrono::steady_clock` around the whole solver call
  (derivatives + iterations).
- Every configuration does warm-up runs (1 for CPU, 3 for GPU) and then reports the
  **median of 10 timed runs** (min and max are also in the CSV).
- **Input:** `data/rm1.jpg` → `data/rm2.jpg` (1600×999), resized to each resolution
  (INTER_AREA when shrinking, bilinear when enlarging to 1920×1080). α = 0.5.
- Raw data: [`results/benchmark.csv`](results/benchmark.csv). Tables and charts are
  generated from it by [`scripts/plot_results.py`](scripts/plot_results.py).

### Hardware and software

| | |
|---|---|
| CPU | Intel Core i7-11800H @ 2.30 GHz, 8 cores / 16 threads (laptop) |
| GPU | NVIDIA GeForce RTX 2050 (laptop), sm_86, 16 SMs, 4 GiB, 64-bit bus, 112 GB/s theoretical peak DRAM bandwidth (from reported memory clock × bus width), 1 MiB L2 |
| Driver / CUDA | driver reports 13.3; built with CUDA Toolkit 13.0 |
| Host compiler | MSVC 19.41 (Visual Studio 2022 Build Tools), `/O2`, `/openmp:llvm` |
| OpenCV | 4.14.0 |
| OS | Windows 11 (WDDM driver model) |

The program prints this information at the start of every benchmark and writes it to
[`results/system_info.txt`](results/system_info.txt).

### Results

TODO: fill after run

### Speedup

TODO: fill after run

### OpenMP thread scaling

TODO: fill after run

### Convergence

TODO: fill after run

---

## Profiling

[`scripts/profile.sh`](scripts/profile.sh) runs:

- **Nsight Compute** on the iteration kernel of each CUDA version at 1920×1080. It skips the
  first 10 launches and samples the next 10. Metrics: duration, achieved and theoretical
  occupancy, DRAM throughput (% of peak and GB/s), memory and SM throughput as % of peak
  (speed-of-light), DRAM bytes moved, L1/L2 hit rates. It also saves one launch with
  `--set full` for the Nsight Compute GUI.
- **Nsight Systems** on the full pipeline, giving a timeline plus kernel, memcpy and API
  summaries.

[`scripts/summarize_ncu.py`](scripts/summarize_ncu.py) turns the CSVs into the table below.

TODO: fill after run

### Analysis (from the measured metrics only)

TODO: fill after run

---

## Flow visualizations

HSV coding: hue = flow direction, brightness = magnitude. All images share the colour scale
of the first one, so they can be compared directly.

TODO: fill after run

---

## Caveats
- **Laptop hardware.** Clocks depend on power source and temperature, and runs over many
  minutes can throttle. Medians over 10 runs reduce but don't remove this.
  Power state during the runs: TODO: confirm (plugged in / power mode).
- **Windows/WDDM.** Kernel launch and transfer overheads under WDDM are higher than under
  Linux or TCC. They matter most for the small, low-iteration configurations.
- **Enlarged input at 1920×1080.** The source frames are 1600×999, so the 1080p case is
  upsampled. The cost per pixel doesn't depend on content; the flow field itself is smoother.
- **Derivative scale.** `Sobel(..., scale = 0.5)` produces about 4× the central-difference
  gradient, so the effective α is smaller than its nominal value. Kept as in the original
  code so that results are unchanged.
- **Pageable transfers.** H2D/D2H use pageable host memory, as a typical caller would.
  Pinned buffers would make them faster.

---

## Files

| path | contents |
|---|---|
| `main.cpp` | CLI: demo, `bench`, `validate`, `convergence`, `flow` |
| `include/hs_common.hpp`, `src/hs_common.cpp` | shared derivatives, denominator, residual |
| `src/opt_flow_serial.cpp` | serial Gauss–Seidel, Jacobi, red-black |
| `src/opt_flow_parallel.cpp` | OpenMP Jacobi (`omp`, `omp-opt`) |
| `include/opt_flow_cuda.hpp`, `src/cuda/opt_flow_cuda.cpp` | CUDA solver: buffers, timing, iteration loop |
| `src/cuda/hs_kernels.cu` | all CUDA kernels |
| `include/cuda_check.hpp` | `CUDA_CHECK` / `CUDA_CHECK_LAUNCH` |
| `src/flow_viz.cpp`, `src/sysinfo.cpp` | HSV/arrow visualization, CPU/core detection |
| `scripts/` | benchmark driver, plotting, profiling |
| `colab/run_on_colab.ipynb` | build + run everything on Google Colab |
| `results/` | CSVs, charts, tables, profiling output |

## Reference
B. K. P. Horn and B. G. Schunck, "Determining Optical Flow," *Artificial Intelligence*, 17(1–3), 185–203, 1981.
