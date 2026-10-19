// Photometric cleanup of the unwarped page (APP_BUILD_BRIEF.md section 3).
//
// Auto color:
//   1. background at 1/4 resolution: 4x4 block mean -> 3x3 dilate -> 11x11 median, per
//      channel (the darkness_fast idea, kept in colour)                      [GPU / CPU]
//   2. paper illumination: background pixels that are low-chroma and close to the local
//      maximum brightness are paper; their luminance is filled into a smooth field
//      (pull-push), so coloured or grey cells are not mistaken for shadow    [CPU, 1/4 res]
//   3. white point from the paper pixels (per-channel balance), black point from ink
//   4. one tone pass: divide by illumination * balance, map paper to white, light contrast
//      stretch, remove isolated specks (ink counts on the 1/4 grid)          [GPU / CPU]
// Grayscale: the same, then desaturate (Rec.601 luma).
// All images are 8-bit BGRA (OpenCV channel order).
// Solid colors (optional): see solid.h.
#pragma once
#include <opencv2/core.hpp>

namespace fs {

struct CleanParams {
    bool grayscale = false;
    cv::Mat illum;            // CV_32F, quarter grid: paper luminance (0..255)
    float wb[3] = {1, 1, 1};  // per-channel paper colour / paper luminance
    float black = 0.f;        // normalised value mapped to 0
    float white = 0.9f;       // normalised value mapped to 1 (paper)
    int speck_max = 4;        // isolated ink with at most this many pixels nearby is removed
};

struct CleanStats {
    bool gpu_background = false, gpu_tone = false;
    double ms_background = 0, ms_params = 0, ms_tone = 0;
};

// Steps 1 (background) and 4 (tone) on the CPU; the GPU versions live in gpu.cpp and must
// produce the same bytes up to float rounding in the final tone mapping (<= 1 level).
cv::Mat background_quarter_cpu(const cv::Mat& rgba);
cv::Mat tone_cpu(const cv::Mat& rgba, const CleanParams& p);
// Step 2-3: parameters from the quarter-resolution background.
CleanParams clean_params(const cv::Mat& bg4, const cv::Mat& rgba, bool grayscale);

// Whole cleanup; use_gpu=false forces the CPU path.
cv::Mat cleanup(const cv::Mat& rgba, bool grayscale, bool use_gpu, CleanStats* st = nullptr);

}  // namespace fs
