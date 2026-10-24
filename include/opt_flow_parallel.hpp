#ifndef OPT_FLOW_PARALLEL_HPP
#define OPT_FLOW_PARALLEL_HPP

#include <opencv2/opencv.hpp>
#include "hs_common.hpp"

// Parallel Horn–Schunck (OpenMP, Jacobi double buffering).
// Uses the OpenMP default thread count; set it with omp_set_num_threads().
void hornSchunckParallel(const cv::Mat &I1, const cv::Mat &I2,
                         cv::Mat &u, cv::Mat &v,
                         float alpha, int iterations,
                         const hs::IterationCallback &onIteration = {});

#endif
