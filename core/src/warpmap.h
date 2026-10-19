// Warp map from page features: port of code/warpmap.py (unwarp_image with every optional
// switch at its default: UNWARP_PROP_*, UNWARP_LSD and UNWARP_R3_* off, G=8, GC=16,
// REG='biharmonic', TEXT_W=1) and the helpers it imports from dewarp.py and clean.py.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <vector>

#include "nputil.h"
#include "resample.h"

namespace fs {

using Line = std::array<double, 5>;   // [pos, start, end, thickness, length], as warpmap.py
using Lines = std::vector<Line>;

// A field stored on a g-px grid (gh x gw, row major).
struct Field {
    int gw = 0, gh = 0, g = 8;
    std::vector<double> v;
    // warpmap.grid_sample: bilinear, clamped (map_coordinates order=1 mode='nearest'), float32
    float sample(double x, double y) const;
};

struct Samples {      // collect(): rows (along, ideal, disp)
    std::vector<std::array<double, 3>> h, v;
};

struct WarpDebug {
    int n_ctrl_raw = 0, n_ctrl = 0, n_text = 0;
    Lines H, V;
    Samples S1, S2;
    std::vector<std::array<double, 3>> T;
    std::vector<std::array<double, 4>> G_lines;   // global-stage grid lines: [horizontal (1/0), straightened
                                                  // position, start, end] (stage-1 frame)
    int n_ignored = 0;                            // samples dropped by the ignore mask
    Field GX, GY, Fx1, Fy1, Fx, Fy;
    cv::Mat A_img, D;
    std::vector<std::pair<std::string, double>> ms;   // stage timings
    int cg_iters[6] = {0, 0, 0, 0, 0, 0};
};

// warpmap.unwarp_image(src) -> backward map on the 4-px grid of src (mx4, my4).
// src: 8-bit BGR. dbg (optional) receives the intermediates for the parity checks.
// ignore (optional, app only): CV_8U of src's size; line samples and grid controls whose
// straightened position falls on a non-zero pixel are left out of the fit (the user marked
// those lines as wrong). Null: the reference behaviour. share_rows (app): pieces of one ruled
// line share their ideal position (see share_ideals in warpmap.cpp); false: the reference.
GridMap unwarp_map(const cv::Mat& src_bgr, WarpDebug* dbg = nullptr, const cv::Mat* ignore = nullptr, bool share_rows = false);

// Building blocks exposed for the stage tests.
cv::Mat darkness_fast(const cv::Mat& bgr);
void get_lines(const cv::Mat& D, Lines& H, Lines& V);
// clean.segments / clean.short_segments (morphological axis-aligned line segments)
Lines clean_segments(const cv::Mat& ink, bool hz, int klen = 100, int minlen = 150);
Lines clean_short_segments(const cv::Mat& ink, bool hz, const Lines& long_segs, const Lines& perp);

}  // namespace fs
