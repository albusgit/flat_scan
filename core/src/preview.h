// Camera guidance (feature round 1): what a small live preview frame shows, for the camera screen's
// hints (app CameraGuide.kt). The page from the page finder's 256 x 256 logits directly (select_page at the
// net's own size, no 1000-px working copy, no edge snapping) and the quad step of the detection
// (corners_from_contour); the frame's luma, glare and sharpness. Nothing here is used by the photo path.
#pragma once
#include <opencv2/core.hpp>
#include <array>

namespace fs {

struct PreviewPage {
    bool found = false;                  // a page-like component (>= 2 % of the frame) with a quad
    std::array<cv::Point2d, 4> pts;      // TL, TR, BR, BL in frame px (may lie outside the frame: a cut-off page)
    double area = 0;                     // the paper's share of the frame (both pages of a spread)
    double luma = 0;                     // mean luma of the frame, 0..1
    double glare = 0;                    // share of the page's pixels at luma >= 250 (0 without a page)
    double sharpness = 0;                // variance of the frame's Laplacian (8-bit luma; scene dependent)
};

// bgr: the upright frame; logits: the page finder's output for it (2 x 256 x 256), or null (no page search:
// only luma and sharpness).
PreviewPage preview_page(const cv::Mat& bgr, const float* logits);

}  // namespace fs
