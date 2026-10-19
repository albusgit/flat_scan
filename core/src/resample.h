// Backward maps on a coarse grid and the single final resample.
//
// Contract (PHONE_MILESTONE.md): maps hold float32 source-pixel coordinates, top-left
// origin, integer pixel centres. Sample (i, j) of the grid is the map value at output pixel
// (j*stride, i*stride). Dense maps are rebuilt by bilinear interpolation with clamp to the
// last grid sample (cv2.remap(..., BORDER_REPLICATE) with x/stride, y/stride, as in
// warpmap.load_maps). The image is sampled with OpenCV's INTER_CUBIC semantics: positions
// quantised to 1/32 px, 15-bit fixed-point weights, constant white border.
#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace fs {

struct GridMap {
    int out_w = 0, out_h = 0;   // dense output size
    int gw = 0, gh = 0;         // grid samples per row / column
    int stride = 4;
    std::vector<float> x, y;    // gh*gw, row major

    static GridMap make(int out_w, int out_h, int stride);
    float& X(int i, int j) { return x[(size_t)i * gw + j]; }
    float& Y(int i, int j) { return y[(size_t)i * gw + j]; }
};

// Dense rows [y0, y1) of the expanded map (CV_32FC1 each).
void expand_rows(const GridMap& g, int y0, int y1, cv::Mat& mx, cv::Mat& my);
// Full dense maps (host harness / parity export only; the app never calls this).
void expand_full(const GridMap& g, cv::Mat& mx, cv::Mat& my);

// The same backward map for a page rendered at another size (output resolution setting):
// output pixel p' maps to analysis pixel (p' + 0.5) * out_w / new_w - 0.5, interpolated
// bilinearly on the grid. Geometry is unchanged; the photo is still sampled once.
GridMap rescale_grid(const GridMap& g, int new_w, int new_h);

// CPU path: cv::remap INTER_CUBIC in horizontal bands, so no full-resolution float map is
// ever allocated. src is CV_8UC3 or CV_8UC4.
cv::Mat resample_cpu(const cv::Mat& src, const GridMap& g, int band_rows = 64);

// OpenCV's BicubicTab_i (1024 x 16 shorts), recomputed with x86 float semantics.
const int16_t* bicubic_itab();
// Scalar model of the GPU kernel (same integer arithmetic as the shader). Used to prove the
// table and border semantics against cv::remap on the host.
cv::Mat resample_model(const cv::Mat& src, const GridMap& g);

}  // namespace fs
