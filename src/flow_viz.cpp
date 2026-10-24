#include "flow_viz.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>

namespace viz {

cv::Mat flowArrows(const cv::Mat &u, const cv::Mat &v)
{
    cv::Mat output = cv::Mat::zeros(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; y += 5)
    {
        for (int x = 0; x < u.cols; x += 5)
        {
            cv::Point2f pt(static_cast<float>(x), static_cast<float>(y));
            cv::Point2f flow(u.at<float>(y, x), v.at<float>(y, x));
            cv::line(output, pt, pt + flow, cv::Scalar(0, 255, 0));
            cv::circle(output, pt, 1, cv::Scalar(0, 0, 255), -1);
        }
    }
    return output;
}

float maxMagnitude(const cv::Mat &u, const cv::Mat &v)
{
    cv::Mat mag;
    cv::magnitude(u, v, mag);
    double maxVal = 0.0;
    cv::minMaxLoc(mag, nullptr, &maxVal);
    return static_cast<float>(maxVal);
}

cv::Mat flowToHSV(const cv::Mat &u, const cv::Mat &v, float maxMag)
{
    cv::Mat mag, angle;
    cv::cartToPolar(u, v, mag, angle, true); // degrees in [0, 360)

    const float scale = maxMag > 0.f ? 1.f / maxMag : 0.f;
    cv::Mat hue = angle * 0.5f; // OpenCV 8-bit hue range is [0, 180)
    cv::Mat val = cv::min(mag * scale, 1.0f) * 255.0f;
    cv::Mat sat(u.size(), CV_32F, cv::Scalar(255.0f));

    std::vector<cv::Mat> ch{hue, sat, val};
    cv::Mat hsv, hsv8, bgr;
    cv::merge(ch, hsv);
    hsv.convertTo(hsv8, CV_8UC3);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

} // namespace viz
