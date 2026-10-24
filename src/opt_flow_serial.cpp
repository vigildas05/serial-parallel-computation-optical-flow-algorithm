#include "opt_flow_serial.hpp"

#include <opencv2/core.hpp>
#include <utility>

void hornSchunckSerial(const cv::Mat &I1, const cv::Mat &I2,
                       cv::Mat &u, cv::Mat &v,
                       float alpha, int iterations,
                       const hs::IterationCallback &onIteration)
{
    CV_Assert(!I1.empty() && !I2.empty());
    CV_Assert(I1.size() == I2.size());

    // Ensure float32 grayscale
    cv::Mat I1f = hs::toFloatGray(I1);
    cv::Mat I2f = hs::toFloatGray(I2);

    cv::Mat Ix, Iy, It;
    hs::computeDerivatives(I1f, I2f, Ix, Iy, It);

    // Precompute denominator: alpha^2 + Ix^2 + Iy^2
    cv::Mat denom = hs::computeDenominator(Ix, Iy, alpha);

    // Initialize flows
    u = cv::Mat::zeros(I1f.size(), CV_32F);
    v = cv::Mat::zeros(I1f.size(), CV_32F);

    const int rows = I1f.rows;
    const int cols = I1f.cols;

    // In-place GaussSeidel updates
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int y = 1; y < rows - 1; ++y)
        {
            for (int x = 1; x < cols - 1; ++x)
            {
                // 4-neighbor average (consistent with parallel)
                float u_avg = 0.25f * (u.at<float>(y - 1, x) + u.at<float>(y + 1, x) +
                                       u.at<float>(y, x - 1) + u.at<float>(y, x + 1));
                float v_avg = 0.25f * (v.at<float>(y - 1, x) + v.at<float>(y + 1, x) +
                                       v.at<float>(y, x - 1) + v.at<float>(y, x + 1));

                float ix = Ix.at<float>(y, x);
                float iy = Iy.at<float>(y, x);
                float it = It.at<float>(y, x);
                float den = denom.at<float>(y, x);

                float t = (ix * u_avg + iy * v_avg + it) / den;
                u.at<float>(y, x) = u_avg - ix * t;
                v.at<float>(y, x) = v_avg - iy * t;
            }
        }
        // borders remain unchanged

        if (onIteration)
            onIteration(iter + 1, u, v);
    }
}

void hornSchunckSerialJacobi(const cv::Mat &I1, const cv::Mat &I2,
                             cv::Mat &u, cv::Mat &v,
                             float alpha, int iterations,
                             const hs::IterationCallback &onIteration)
{
    CV_Assert(!I1.empty() && !I2.empty());
    CV_Assert(I1.size() == I2.size());

    cv::Mat I1f = hs::toFloatGray(I1);
    cv::Mat I2f = hs::toFloatGray(I2);

    cv::Mat Ix, Iy, It;
    hs::computeDerivatives(I1f, I2f, Ix, Iy, It);
    cv::Mat denom = hs::computeDenominator(Ix, Iy, alpha);

    u = cv::Mat::zeros(I1f.size(), CV_32F);
    v = cv::Mat::zeros(I1f.size(), CV_32F);

    // Jacobi buffers. Only interior pixels are ever written, so the zero
    // border of both buffers stays fixed, matching the other versions.
    cv::Mat u_next = u.clone();
    cv::Mat v_next = v.clone();

    const int rows = I1f.rows;
    const int cols = I1f.cols;

    for (int iter = 0; iter < iterations; ++iter)
    {
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
                float it = It.at<float>(y, x);
                float den = denom.at<float>(y, x);

                float t = (ix * u_avg + iy * v_avg + it) / den;
                u_next.at<float>(y, x) = u_avg - ix * t;
                v_next.at<float>(y, x) = v_avg - iy * t;
            }
        }

        std::swap(u, u_next);
        std::swap(v, v_next);

        if (onIteration)
            onIteration(iter + 1, u, v);
    }
}
