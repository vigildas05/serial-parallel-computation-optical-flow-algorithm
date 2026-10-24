#include "hs_kernels.cuh"
#include "cuda_check.hpp"

namespace hs::gpu::detail {

namespace {

__constant__ float c_gauss[5];

__device__ __forceinline__ int clampi(int x, int lo, int hi)
{
    return min(max(x, lo), hi);
}

dim3 gridFor(int rows, int cols)
{
    return dim3((cols + kBlockX - 1) / kBlockX, (rows + kBlockY - 1) / kBlockY);
}

// ---------------------------------------------------------------------------
// Derivatives
//
// These mirror OpenCV's separable filters for CV_32F (cv::GaussianBlur and
// cv::Sobel with BORDER_REPLICATE): symmetric kernels are evaluated as
// k0*center + k1*(left+right) + k2*(...), and explicit _rn intrinsics stop
// nvcc from contracting into FMAs, so results agree with the CPU to within
// float rounding.
// ---------------------------------------------------------------------------

__global__ void gaussianRowKernel(const float *__restrict__ src, float *__restrict__ dst,
                                  int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    const float *row = src + y * cols;
    float s = __fmul_rn(c_gauss[2], row[x]);
    s = __fadd_rn(s, __fmul_rn(c_gauss[1], __fadd_rn(row[clampi(x - 1, 0, cols - 1)],
                                                     row[clampi(x + 1, 0, cols - 1)])));
    s = __fadd_rn(s, __fmul_rn(c_gauss[0], __fadd_rn(row[clampi(x - 2, 0, cols - 1)],
                                                     row[clampi(x + 2, 0, cols - 1)])));
    dst[y * cols + x] = s;
}

__global__ void gaussianColKernel(const float *__restrict__ src, float *__restrict__ dst,
                                  int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    auto at = [&](int yy) { return src[clampi(yy, 0, rows - 1) * cols + x]; };
    float s = __fmul_rn(c_gauss[2], at(y));
    s = __fadd_rn(s, __fmul_rn(c_gauss[1], __fadd_rn(at(y - 1), at(y + 1))));
    s = __fadd_rn(s, __fmul_rn(c_gauss[0], __fadd_rn(at(y - 2), at(y + 2))));
    dst[y * cols + x] = s;
}

__global__ void derivativesKernel(const float *__restrict__ I1s, const float *__restrict__ I2s,
                                  float alpha2,
                                  float *__restrict__ Ix, float *__restrict__ Iy,
                                  float *__restrict__ It, float *__restrict__ denom,
                                  int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    const int xm = clampi(x - 1, 0, cols - 1), xp = clampi(x + 1, 0, cols - 1);
    const int ym = clampi(y - 1, 0, rows - 1), yp = clampi(y + 1, 0, rows - 1);
    auto I = [&](int yy, int xx) { return I1s[yy * cols + xx]; };

    // d/dx: row kernel [-1 0 1], column kernel [0.5 1 0.5]
    const float dxm = __fsub_rn(I(ym, xp), I(ym, xm));
    const float dx0 = __fsub_rn(I(y, xp), I(y, xm));
    const float dxp = __fsub_rn(I(yp, xp), I(yp, xm));
    const float ix = __fadd_rn(dx0, __fmul_rn(0.5f, __fadd_rn(dxm, dxp)));

    // d/dy: row kernel [0.5 1 0.5], column kernel [-1 0 1]
    auto smoothRow = [&](int yy) {
        return __fadd_rn(I(yy, x), __fmul_rn(0.5f, __fadd_rn(I(yy, xm), I(yy, xp))));
    };
    const float iy = __fsub_rn(smoothRow(yp), smoothRow(ym));

    const int i = y * cols + x;
    Ix[i] = ix;
    Iy[i] = iy;
    It[i] = __fsub_rn(I2s[i], I1s[i]);
    denom[i] = __fadd_rn(__fadd_rn(__fmul_rn(ix, ix), __fmul_rn(iy, iy)), alpha2);
}

// ---------------------------------------------------------------------------
// v1: naive Jacobi sweep. One thread per pixel; every neighbour read goes to
// global memory (served by L1/L2 when it hits). Same 4-neighbour average and
// same fixed zero border as the CPU versions: border threads write nothing,
// and both ping-pong buffers start zeroed.
// ---------------------------------------------------------------------------
__global__ void jacobiNaiveKernel(const float *u, const float *v,
                                  const float *Ix, const float *Iy,
                                  const float *It, const float *denom,
                                  float *u_out, float *v_out,
                                  int rows, int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= cols - 1 || y >= rows - 1)
        return;

    const int i = y * cols + x;
    const float u_avg = 0.25f * (u[i - cols] + u[i + cols] + u[i - 1] + u[i + 1]);
    const float v_avg = 0.25f * (v[i - cols] + v[i + cols] + v[i - 1] + v[i + 1]);

    const float ix = Ix[i];
    const float iy = Iy[i];
    const float t = (ix * u_avg + iy * v_avg + It[i]) / denom[i];
    u_out[i] = u_avg - ix * t;
    v_out[i] = v_avg - iy * t;
}

// ---------------------------------------------------------------------------
// v2: shared-memory tiling. Each block stages its kBlockX x kBlockY tile of u
// and v plus a 1-pixel halo (edges only - the 4-neighbour stencil needs no
// corners) in shared memory, so each u/v value is read from global memory
// about once per block instead of up to five times.
// ---------------------------------------------------------------------------
__global__ void jacobiSharedKernel(const float *__restrict__ u, const float *__restrict__ v,
                                   const float *__restrict__ Ix, const float *__restrict__ Iy,
                                   const float *__restrict__ It, const float *__restrict__ denom,
                                   float *__restrict__ u_out, float *__restrict__ v_out,
                                   int rows, int cols)
{
    __shared__ float su[kBlockY + 2][kBlockX + 2];
    __shared__ float sv[kBlockY + 2][kBlockX + 2];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * kBlockX + tx;
    const int y = blockIdx.y * kBlockY + ty;
    const int sx = tx + 1, sy = ty + 1;

    // Out-of-image cells are only ever neighbours of border pixels, which
    // are not updated, so their value does not matter.
    auto load = [&](int gy, int gx, int ly, int lx) {
        const bool inside = gx >= 0 && gx < cols && gy >= 0 && gy < rows;
        su[ly][lx] = inside ? u[gy * cols + gx] : 0.f;
        sv[ly][lx] = inside ? v[gy * cols + gx] : 0.f;
    };
    load(y, x, sy, sx);
    if (tx == 0) load(y, x - 1, sy, 0);
    if (tx == kBlockX - 1) load(y, x + 1, sy, kBlockX + 1);
    if (ty == 0) load(y - 1, x, 0, sx);
    if (ty == kBlockY - 1) load(y + 1, x, kBlockY + 1, sx);
    __syncthreads();

    if (x < 1 || y < 1 || x >= cols - 1 || y >= rows - 1)
        return;

    const float u_avg = 0.25f * (su[sy - 1][sx] + su[sy + 1][sx] + su[sy][sx - 1] + su[sy][sx + 1]);
    const float v_avg = 0.25f * (sv[sy - 1][sx] + sv[sy + 1][sx] + sv[sy][sx - 1] + sv[sy][sx + 1]);

    const int i = y * cols + x;
    const float ix = Ix[i];
    const float iy = Iy[i];
    const float t = (ix * u_avg + iy * v_avg + It[i]) / denom[i];
    u_out[i] = u_avg - ix * t;
    v_out[i] = v_avg - iy * t;
}

} // namespace

void setGaussianKernel(const float taps[5])
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, taps, 5 * sizeof(float)));
}

void launchGaussianBlur(const float *src, float *tmp, float *dst,
                        int rows, int cols, cudaStream_t stream)
{
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid = gridFor(rows, cols);
    gaussianRowKernel<<<grid, block, 0, stream>>>(src, tmp, rows, cols);
    CUDA_CHECK_LAUNCH();
    gaussianColKernel<<<grid, block, 0, stream>>>(tmp, dst, rows, cols);
    CUDA_CHECK_LAUNCH();
}

void launchDerivatives(const float *I1s, const float *I2s, float alpha,
                       float *Ix, float *Iy, float *It, float *denom,
                       int rows, int cols, cudaStream_t stream)
{
    derivativesKernel<<<gridFor(rows, cols), dim3(kBlockX, kBlockY), 0, stream>>>(
        I1s, I2s, alpha * alpha, Ix, Iy, It, denom, rows, cols);
    CUDA_CHECK_LAUNCH();
}

void launchJacobiNaive(const float *u, const float *v,
                       const float *Ix, const float *Iy, const float *It, const float *denom,
                       float *u_out, float *v_out,
                       int rows, int cols, cudaStream_t stream)
{
    jacobiNaiveKernel<<<gridFor(rows, cols), dim3(kBlockX, kBlockY), 0, stream>>>(
        u, v, Ix, Iy, It, denom, u_out, v_out, rows, cols);
    CUDA_CHECK_LAUNCH();
}

void launchJacobiShared(const float *u, const float *v,
                        const float *Ix, const float *Iy, const float *It, const float *denom,
                        float *u_out, float *v_out,
                        int rows, int cols, cudaStream_t stream)
{
    jacobiSharedKernel<<<gridFor(rows, cols), dim3(kBlockX, kBlockY), 0, stream>>>(
        u, v, Ix, Iy, It, denom, u_out, v_out, rows, cols);
    CUDA_CHECK_LAUNCH();
}

} // namespace hs::gpu::detail
