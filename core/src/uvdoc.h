// UVDoc page flattening (Verhoeven et al. 2023, github.com/tanguymagne/UVDoc, MIT licence):
// a network that predicts, for a 45 x 31 grid over the flat page, where each grid point sits in
// the photo. The app runs it on the page's bounding box (from the corners, plus a margin), turned
// so that the page's long side is vertical (it was trained on upright portrait documents).
//
// The net itself runs in the app (ONNX Runtime, assets/models/uvdoc.onnx); this file makes its
// input, reads its output back into photo pixels and checks it against the page finder.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <vector>

namespace fs {

constexpr int UV_IN_W = 488, UV_IN_H = 712;          // network input (RGB, 0..1, NCHW)
constexpr int UV_GW = 31, UV_GH = 45;                // output grid: 2 x 45 x 31, (x, y) in [-1, 1]
constexpr int UV_INPUT_LEN = 3 * UV_IN_W * UV_IN_H;
constexpr int UV_OUTPUT_LEN = 2 * UV_GW * UV_GH;

// The part of the photo the net sees: the bounding box of the page corners with a 6% margin;
// turned: the box is turned 90 degrees clockwise first (the page is landscape in the photo).
struct UvCrop {
    cv::Rect box;
    bool turned = false;
};

// corners: TL, TR, BR, BL in photo px (the crop screen's / detection's order)
UvCrop uv_crop(cv::Size photo, const std::array<cv::Point2d, 4>& corners);

// Network input for the crop: cv::INTER_AREA resize to 488 x 712, RGB / 255, NCHW float.
std::vector<float> uv_input(const cv::Mat& photo_bgr, const UvCrop& c);

// The grid in photo pixels, rows running down the page as the net saw it (portrait).
struct UvGrid {
    bool ok = false;
    bool turned = false;
    std::vector<cv::Point2d> pts;    // UV_GH x UV_GW, row major
    // backward map of a flat page of W x H px (landscape when turned): page pixel -> photo pixel
    cv::Point2d at(double x, double y, int W, int H) const;
    // the page outline (the grid's border), photo px
    std::vector<cv::Point2f> outline() const;
};

UvGrid uv_grid(const float* out, const UvCrop& c);

// IoU of the grid's outline with a page mask (CV_8U, non-zero = page) whose pixels are photo
// pixels * scale (the analysis copy), measured at <= 800 px.
double uv_agreement(const UvGrid& g, const cv::Mat& mask, double scale);
// the same against the corner quad (no mask: corners set by hand)
double uv_agreement(const UvGrid& g, const std::array<cv::Point2d, 4>& quad, cv::Size photo);

}  // namespace fs
