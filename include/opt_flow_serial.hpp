#ifndef OPT_FLOW_SERIAL_HPP
#define OPT_FLOW_SERIAL_HPP

#include <opencv2/opencv.hpp>
#include "hs_common.hpp"

// Serial Horn–Schunck, in-place Gauss–Seidel updates
void hornSchunckSerial(const cv::Mat &I1, const cv::Mat &I2,
                       cv::Mat &u, cv::Mat &v,
                       float alpha, int iterations,
                       const hs::IterationCallback &onIteration = {});

// Serial Horn–Schunck, Jacobi updates (double buffering). Same arithmetic as
// the OpenMP version, so it is the single-threaded baseline for it.
void hornSchunckSerialJacobi(const cv::Mat &I1, const cv::Mat &I2,
                             cv::Mat &u, cv::Mat &v,
                             float alpha, int iterations,
                             const hs::IterationCallback &onIteration = {});

// Serial Horn–Schunck, red-black Gauss–Seidel ordering. Reference for the
// CUDA red-black kernel (same update order).
void hornSchunckSerialRedBlack(const cv::Mat &I1, const cv::Mat &I2,
                               cv::Mat &u, cv::Mat &v,
                               float alpha, int iterations,
                               const hs::IterationCallback &onIteration = {});

#endif
