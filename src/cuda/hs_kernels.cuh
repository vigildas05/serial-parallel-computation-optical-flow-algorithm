#ifndef HS_KERNELS_CUH
#define HS_KERNELS_CUH

#include <cuda_runtime.h>

// Host-side launchers for the Horn–Schunck kernels. All images are dense,
// row-major (pitch == cols) float buffers in device memory. Every launcher
// checks its launch with CUDA_CHECK_LAUNCH.
namespace hs::gpu::detail {

// Thread block shape for all 2D kernels: 32 threads along x (columns) so a
// warp reads one contiguous 128-byte row segment.
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

// 5-tap Gaussian (must be called once before launchGaussianBlur).
void setGaussianKernel(const float taps[5]);

// dst = separable 5x5 Gaussian of src with replicated borders. tmp: scratch.
void launchGaussianBlur(const float *src, float *tmp, float *dst,
                        int rows, int cols, cudaStream_t stream);

// Ix, Iy = Sobel(I1s) * 0.5, It = I2s - I1s, denom = alpha^2 + Ix^2 + Iy^2.
void launchDerivatives(const float *I1s, const float *I2s, float alpha,
                       float *Ix, float *Iy, float *It, float *denom,
                       int rows, int cols, cudaStream_t stream);

// v1: one Jacobi sweep, all reads straight from global memory.
void launchJacobiNaive(const float *u, const float *v,
                       const float *Ix, const float *Iy, const float *It, const float *denom,
                       float *u_out, float *v_out,
                       int rows, int cols, cudaStream_t stream);

// v2: same sweep, u/v neighbours staged in shared memory with a 1-pixel halo.
void launchJacobiShared(const float *u, const float *v,
                        const float *Ix, const float *Iy, const float *It, const float *denom,
                        float *u_out, float *v_out,
                        int rows, int cols, cudaStream_t stream);

// v3 layout: p, q, r = Ix, Iy, It divided by sqrt(denom); n = rows * cols.
void launchFoldCoefficients(const float *Ix, const float *Iy, const float *It, const float *denom,
                            float *p, float *q, float *r, int n, cudaStream_t stream);

// v3: Jacobi sweep on interleaved (u, v) with folded coefficients.
void launchJacobiFused(const float2 *uv, const float *p, const float *q, const float *r,
                       float2 *uv_out, int rows, int cols, cudaStream_t stream);

// De-interleave (u, v) into separate planes for download.
void launchSplitFlow(const float2 *uv, float *u, float *v, int n, cudaStream_t stream);

} // namespace hs::gpu::detail

#endif
