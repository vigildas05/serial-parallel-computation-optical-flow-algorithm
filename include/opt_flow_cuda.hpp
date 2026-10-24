#ifndef OPT_FLOW_CUDA_HPP
#define OPT_FLOW_CUDA_HPP

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include "hs_common.hpp"

// GPU Horn–Schunck. This header has no CUDA dependencies so it can be included
// from host-only translation units.
namespace hs::gpu {

enum class Kernel {
    Naive,    // v1: one thread per pixel, all reads from global memory
    Shared,   // v2: u/v tiles with a 1-pixel halo staged in shared memory
    Fused,    // v3: interleaved (u,v) + folded coefficients, read-only loads
    RedBlack, // red-black Gauss–Seidel on the v3 data layout
};

const char *kernelName(Kernel k);

// Milliseconds, measured with cudaEvents on the solver's stream.
struct Timings {
    float h2d_ms = 0.f;     // upload of both frames
    float compute_ms = 0.f; // derivatives + all iterations (+ layout conversion)
    float d2h_ms = 0.f;     // download of u and v
    float total_ms = 0.f;   // first upload to last download
};

// One-line description of the current device (name, compute capability, ...).
std::string deviceSummary();

// Owns all device buffers for one image size, so repeated runs (benchmarks)
// do not pay for cudaMalloc. Not thread-safe.
class Solver {
public:
    Solver(int rows, int cols);
    ~Solver();
    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;

    // I1, I2: CV_32F single channel, rows x cols. u, v: CV_32F outputs.
    // When onIteration is set, the flow is downloaded after every iteration
    // (slow - for convergence studies only, never for timing).
    void run(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &u, cv::Mat &v,
             float alpha, int iterations, Kernel kernel,
             Timings *timings = nullptr,
             const IterationCallback &onIteration = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hs::gpu

#endif
