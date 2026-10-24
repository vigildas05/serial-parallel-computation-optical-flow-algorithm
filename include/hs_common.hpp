#ifndef HS_COMMON_HPP
#define HS_COMMON_HPP

#include <opencv2/core.hpp>
#include <functional>

// Building blocks shared by every Horn–Schunck implementation (CPU and GPU),
// so that all of them solve exactly the same discrete problem.
namespace hs {

// Called after every iteration with the current flow field. Only used for
// convergence studies; timed runs pass an empty callback.
using IterationCallback = std::function<void(int iteration, const cv::Mat &u, const cv::Mat &v)>;

// Returns a CV_32F single-channel view/copy of m.
cv::Mat toFloatGray(const cv::Mat &m);

// Ix, Iy: 3x3 Sobel (scale 0.5) of the Gaussian-smoothed first frame.
// It:     difference of the Gaussian-smoothed frames (I2s - I1s).
// Smoothing is a 5x5 Gaussian, sigma = 1, replicated borders.
void computeDerivatives(const cv::Mat &I1, const cv::Mat &I2,
                        cv::Mat &Ix, cv::Mat &Iy, cv::Mat &It);

// denom = alpha^2 + Ix^2 + Iy^2
cv::Mat computeDenominator(const cv::Mat &Ix, const cv::Mat &Iy, float alpha);

// Scheme-independent convergence measure: RMS over interior pixels of the
// change one Jacobi sweep would make to (u, v), i.e. how far (u, v) is from
// satisfying the discrete Horn–Schunck equations. Zero at the solution.
double fixedPointResidual(const cv::Mat &u, const cv::Mat &v,
                          const cv::Mat &Ix, const cv::Mat &Iy,
                          const cv::Mat &It, const cv::Mat &denom);

} // namespace hs

#endif
