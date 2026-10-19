// Page detection: port of code/corners.py (paper_mask, curved_corners, ransac_quadratic,
// intersect) and photo_pipeline.detect.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <vector>

#include "nprng.h"
#include "nputil.h"

namespace fs {

struct Side {
    cv::Point2d P0, d, n;
    double len = 0;
    Vec p;                       // quadratic o(s), highest power first
    std::vector<char> inl;
    cv::Point2d point(double s) const;
    double signed_dist(const cv::Point2d& X) const;
};

struct Detection {
    bool ok = false;
    std::array<cv::Point2d, 4> corners;          // TL, TR, BR, BL (photo px)
    std::array<Side, 4> sides;                   // top, right, bottom, left
    std::array<std::vector<cv::Point2d>, 4> side_pts;   // outline points assigned to each side
    std::array<double, 4> fold_d{};              // contour gap at each virtual corner
    std::vector<cv::Point2d> contour;
    cv::Mat paper;                               // filled page outline (255 = paper), analysis size
    bool landscape = false;
};

// img: 8-bit BGR. rng: corners.py's module-level default_rng(0); the reference creates it
// once per process and every call continues its stream.
Detection detect_page(const cv::Mat& bgr, NpRng& rng);

}  // namespace fs
