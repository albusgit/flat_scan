// "Solid colors": fills table cells with one clean colour each (improved port of the cell-fill
// part of code/clean.py).
//  a. cells are the regions enclosed by the grid lines (clean.py's segments, short segments,
//     inferred borders, wobbly ends, snap), after closing collinear gaps of up to ~15 px
//  b. all cell colours of the page are clustered in Lab (merge while dE76 < 7, area weighted)
//     and every cell gets its cluster's colour, so same-colour cells match exactly
//  c. a cell is filled only if its colour cluster clearly differs from the page's paper colour
//     (dE76 >= 5 from the low-chroma, light cells); otherwise it becomes pure white. Decided per
//     cluster, so one colour is never split. (A fixed "chroma > 10" cut loses the light blue and
//     lavender cells of scan p3, chroma 3-7 but clearly tinted; a cast on a white cell after
//     Auto color stays below dE 5.)
//  d. the colour is measured only on pixels > 3 px away from any ink or line; ink is re-mixed
//     with its own coverage (unmixing against the old background), so no halos remain
//  e. large uniform coloured areas not enclosed by lines are found as blobs at 1/4
//     resolution; their rectangle edges snap to grid lines within 30 px
//  f. regions with local detail after removing ink (pictures, logos: high-pass 10-90% range
//     > 18 levels) or a large smooth range (designed gradients: > 45 levels), regions of at
//     least 60x60 px with > 10% darker areas at least 15 px wide (the darker half of a photo
//     or a gradient, which the ink exclusion would otherwise drop from the measurement), and
//     regions mostly inside a protected rectangle are left untouched
//  g. ink is never recoloured: its colour is the darkest nearby pixel's own colour
//  h. the fill reaches the line centres (line pixels take the nearest cell's fill as background)
// Grid lines and text are never redrawn.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace fs {

struct Rect01 { float x0, y0, x1, y1; };   // protected (picture) area, page fractions

struct SolidStats {
    int regions = 0, filled = 0, white = 0, skipped = 0, skipped_varied = 0, blobs = 0, clusters = 0;
    double ms = 0;
};

// img: 8-bit BGRA (the filtered page). Returns a new BGRA image.
cv::Mat solid_colors(const cv::Mat& img_bgra, const std::vector<Rect01>& protect, SolidStats* st = nullptr);

}  // namespace fs
