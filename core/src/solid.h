// "Solid colors": port of the cell-fill part of code/clean.py clean() (v1).
//
// Table grid lines are found as in clean.py (segments, short_segments, infer_borders,
// extend_wobbly, snap) and rasterised with draw_lines() ONLY as region separators; the
// regions they enclose (4-connected, not touching the page border, < 30% of the page) get
// their median background colour, measured on the eroded core with text ink excluded.
// Low-chroma regions (HSV s <= 12 and v >= 238, or v <= 110) become white; similar colours
// (Lab distance < 8) are merged area-weighted and lifted by *1.03 + 3, as in clean.py.
// Unlike v1 nothing is redrawn: the result is composited as fill * (1 - A) + pixel * A with
// A from the local darkness of the unwarped page, so lines and text keep their own pixels.
// The v1 "coloured areas outside the grid -> rectangles" heuristic is not ported (it snaps
// rectangles to the redrawn line positions).
#pragma once
#include <opencv2/core.hpp>

namespace fs {

struct SolidStats {
    int regions = 0, coloured = 0, colour_groups = 0;
    double ms = 0;
};

// img: 8-bit BGRA (the cleaned page, before grayscale). Returns a new BGRA image.
cv::Mat solid_colors(const cv::Mat& img_bgra, SolidStats* st = nullptr);

}  // namespace fs
