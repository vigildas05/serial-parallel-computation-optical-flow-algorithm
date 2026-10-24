#include "opt_flow_cuda.hpp"
#include "cuda_check.hpp"
#include "hs_kernels.cuh"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <sstream>
#include <utility>

namespace hs::gpu {

using namespace detail;

const char *kernelName(Kernel k)
{
    switch (k) {
    case Kernel::Naive: return "cuda-v1";
    case Kernel::Shared: return "cuda-v2";
    case Kernel::Fused: return "cuda-v3";
    case Kernel::RedBlack: return "cuda-rb";
    }
    return "?";
}

std::string deviceSummary()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int memClockKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

    // Theoretical peak DRAM bandwidth: 2 transfers/clock * clock * bus width.
    const double peakGBs = 2.0 * memClockKHz * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;

    std::ostringstream os;
    os << p.name << " | sm_" << p.major << p.minor << " | " << p.multiProcessorCount << " SMs | "
       << (p.totalGlobalMem >> 20) << " MiB | " << p.memoryBusWidth << "-bit bus, "
       << peakGBs << " GB/s peak | L2 " << (p.l2CacheSize >> 10) << " KiB | driver "
       << driver / 1000 << "." << (driver % 1000) / 10 << ", runtime " << runtime / 1000 << "."
       << (runtime % 1000) / 10;
    return os.str();
}

struct Solver::Impl {
    int rows, cols;
    size_t bytes;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[4] = {};

    // Inputs and derivative scratch
    float *I1 = nullptr, *I2 = nullptr, *tmp = nullptr, *I1s = nullptr, *I2s = nullptr;
    float *Ix = nullptr, *Iy = nullptr, *It = nullptr, *denom = nullptr;
    // Ping-pong flow buffers
    float *u[2] = {}, *v[2] = {};

    float **allFloatBuffers[13] = {&I1, &I2, &tmp, &I1s, &I2s, &Ix, &Iy, &It, &denom,
                                   &u[0], &u[1], &v[0], &v[1]};

    Impl(int r, int c) : rows(r), cols(c), bytes(size_t(r) * c * sizeof(float))
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (auto &e : ev)
            CUDA_CHECK(cudaEventCreate(&e));
        for (float **b : allFloatBuffers)
            CUDA_CHECK(cudaMalloc(b, bytes));

        // Same 5-tap Gaussian OpenCV uses for GaussianBlur(5x5, sigma 1) on CV_32F.
        cv::Mat g = cv::getGaussianKernel(5, 1.0, CV_32F);
        setGaussianKernel(g.ptr<float>());
    }

    ~Impl()
    {
        // No CUDA_CHECK here: a destructor must not exit, and errors at
        // teardown have nothing left to corrupt.
        for (float **b : allFloatBuffers)
            cudaFree(*b);
        for (auto &e : ev)
            cudaEventDestroy(e);
        cudaStreamDestroy(stream);
    }

    void downloadFlow(const float *du, const float *dv, cv::Mat &hu, cv::Mat &hv)
    {
        CUDA_CHECK(cudaMemcpyAsync(hu.ptr<float>(), du, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hv.ptr<float>(), dv, bytes, cudaMemcpyDeviceToHost, stream));
    }

    void run(const cv::Mat &hI1, const cv::Mat &hI2, cv::Mat &hu, cv::Mat &hv,
             float alpha, int iterations, Kernel kernel, Timings *timings,
             const IterationCallback &onIteration)
    {
        hu.create(rows, cols, CV_32F);
        hv.create(rows, cols, CV_32F);

        CUDA_CHECK(cudaEventRecord(ev[0], stream));
        CUDA_CHECK(cudaMemcpyAsync(I1, hI1.ptr<float>(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(I2, hI2.ptr<float>(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(ev[1], stream));

        launchGaussianBlur(I1, tmp, I1s, rows, cols, stream);
        launchGaussianBlur(I2, tmp, I2s, rows, cols, stream);
        launchDerivatives(I1s, I2s, alpha, Ix, Iy, It, denom, rows, cols, stream);

        // Zero initial flow; the border is never written so it stays zero.
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMemsetAsync(u[b], 0, bytes, stream));
            CUDA_CHECK(cudaMemsetAsync(v[b], 0, bytes, stream));
        }

        int cur = 0;
        for (int it = 0; it < iterations; ++it) {
            const int nxt = cur ^ 1;
            switch (kernel) {
            case Kernel::Naive:
                launchJacobiNaive(u[cur], v[cur], Ix, Iy, It, denom, u[nxt], v[nxt],
                                  rows, cols, stream);
                break;
            case Kernel::Shared:
                launchJacobiShared(u[cur], v[cur], Ix, Iy, It, denom, u[nxt], v[nxt],
                                   rows, cols, stream);
                break;
            default:
                CV_Error(cv::Error::StsNotImplemented, kernelName(kernel));
            }
            cur = nxt;

            if (onIteration) {
                downloadFlow(u[cur], v[cur], hu, hv);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                onIteration(it + 1, hu, hv);
            }
        }
        CUDA_CHECK(cudaEventRecord(ev[2], stream));

        downloadFlow(u[cur], v[cur], hu, hv);
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        CUDA_CHECK(cudaEventSynchronize(ev[3]));

        if (timings) {
            CUDA_CHECK(cudaEventElapsedTime(&timings->h2d_ms, ev[0], ev[1]));
            CUDA_CHECK(cudaEventElapsedTime(&timings->compute_ms, ev[1], ev[2]));
            CUDA_CHECK(cudaEventElapsedTime(&timings->d2h_ms, ev[2], ev[3]));
            CUDA_CHECK(cudaEventElapsedTime(&timings->total_ms, ev[0], ev[3]));
        }
    }
};

Solver::Solver(int rows, int cols) : impl_(std::make_unique<Impl>(rows, cols)) {}
Solver::~Solver() = default;

void Solver::run(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &u, cv::Mat &v,
                 float alpha, int iterations, Kernel kernel, Timings *timings,
                 const IterationCallback &onIteration)
{
    CV_Assert(I1.size() == I2.size());
    CV_Assert(I1.rows == impl_->rows && I1.cols == impl_->cols);
    // Device buffers are dense, so host inputs must be dense float too.
    cv::Mat a = hs::toFloatGray(I1), b = hs::toFloatGray(I2);
    if (!a.isContinuous()) a = a.clone();
    if (!b.isContinuous()) b = b.clone();
    impl_->run(a, b, u, v, alpha, iterations, kernel, timings, onIteration);
}

} // namespace hs::gpu
