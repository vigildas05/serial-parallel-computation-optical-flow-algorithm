#include "hs_common.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>

namespace hs {

cv::Mat toFloatGray(const cv::Mat &m)
{
    CV_Assert(m.channels() == 1);
    if (m.type() == CV_32F)
        return m;
    cv::Mat f;
    m.convertTo(f, CV_32F);
    return f;
}

// Compute spatial and temporal derivatives (with slight smoothing for stability)
void computeDerivatives(const cv::Mat &I1, const cv::Mat &I2,
                        cv::Mat &Ix, cv::Mat &Iy, cv::Mat &It)
{
    cv::Mat I1s, I2s;
    cv::GaussianBlur(I1, I1s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(I2, I2s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);

    cv::Sobel(I1s, Ix, CV_32F, 1, 0, 3, 0.5, 0, cv::BORDER_REPLICATE);
    cv::Sobel(I1s, Iy, CV_32F, 0, 1, 3, 0.5, 0, cv::BORDER_REPLICATE);

    It = I2s - I1s;
}

cv::Mat computeDenominator(const cv::Mat &Ix, const cv::Mat &Iy, float alpha)
{
    cv::Mat Ix2, Iy2;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::Mat denom = Ix2 + Iy2;
    denom += alpha * alpha;
    return denom;
}

double fixedPointResidual(const cv::Mat &u, const cv::Mat &v,
                          const cv::Mat &Ix, const cv::Mat &Iy,
                          const cv::Mat &It, const cv::Mat &denom)
{
    const int rows = u.rows;
    const int cols = u.cols;
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int y = 1; y < rows - 1; ++y)
    {
        for (int x = 1; x < cols - 1; ++x)
        {
            float u_avg = 0.25f * (u.at<float>(y - 1, x) + u.at<float>(y + 1, x) +
                                   u.at<float>(y, x - 1) + u.at<float>(y, x + 1));
            float v_avg = 0.25f * (v.at<float>(y - 1, x) + v.at<float>(y + 1, x) +
                                   v.at<float>(y, x - 1) + v.at<float>(y, x + 1));
            float ix = Ix.at<float>(y, x);
            float iy = Iy.at<float>(y, x);
            float t = (ix * u_avg + iy * v_avg + It.at<float>(y, x)) / denom.at<float>(y, x);
            double du = (u_avg - ix * t) - u.at<float>(y, x);
            double dv = (v_avg - iy * t) - v.at<float>(y, x);
            sum += du * du + dv * dv;
        }
    }
    const double n = static_cast<double>(rows - 2) * (cols - 2);
    return std::sqrt(sum / n);
}

} // namespace hs
