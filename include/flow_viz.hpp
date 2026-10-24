#ifndef FLOW_VIZ_HPP
#define FLOW_VIZ_HPP

#include <opencv2/core.hpp>

namespace viz {

// Sparse arrow plot (every 5th pixel), as produced by the original program.
cv::Mat flowArrows(const cv::Mat &u, const cv::Mat &v);

// Dense HSV color coding: hue = flow direction, value = magnitude / maxMag
// (clipped to 1), full saturation. Pass the same maxMag for every image that
// should be compared side by side.
cv::Mat flowToHSV(const cv::Mat &u, const cv::Mat &v, float maxMag);

float maxMagnitude(const cv::Mat &u, const cv::Mat &v);

} // namespace viz

#endif
