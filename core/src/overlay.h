// "Wrinkle map": the deformation the unwarp removed, drawn over the result (render_map.py idea).
#pragma once
#include <opencv2/core.hpp>

#include "resample.h"

namespace fs {

// BGRA overlay (premultiplied-free, alpha in channel 3) of size disp.out_w x disp.out_h:
// a heat map of |d| (TURBO colours, 0 .. legend_px) and the deformed grid of the flat page
// (lines every `spacing` px drawn at p + d(p), i.e. where they sat on the before page).
// legend_px receives the scale's top value (99.5th percentile of |d|, rounded up).
// The legend's top value alone (what wrinkle_overlay reports), without drawing.
float wrinkle_legend_px(const GridMap& disp);

cv::Mat wrinkle_overlay(const GridMap& disp, bool heat, bool grid, float* legend_px, int spacing = 80);

}  // namespace fs
