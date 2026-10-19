// Page surface model for photos: port of code/surface.py (curled strip + camera pose,
// fitted to the page outline; unrolled to A4 by arc length).
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <vector>

#include "nputil.h"

namespace fs {

struct SurfaceModel {
    bool ok = false;
    char axis = 'u';            // bend axis: 'u' (across the width) or 'v'
    double c[4] = {0, 0, 0, 0}; // bend coefficients (radians)
    double rvec[3] = {0, 0, 0}, tvec[3] = {0, 0, 0};
    double f = 0, cx = 0, cy = 0, W = 210, H = 297;
    double cost = 0, median_px = 0, p90_px = 0;

    // backward map: page coordinates (mm) -> photo pixels
    cv::Point2d project_uv(double u_mm, double v_mm) const;
    // unroll_map at output pixel (x, y) of an out_w-wide page (float64, before float32 rounding)
    cv::Point2d unroll_at(double x, double y, int out_w, int out_h) const;
    void precompute();          // profile tables + rotation, call after setting the parameters
    Vec s_, X_, Z_;
    double R_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

// surface.fit(side_pts, corners, img_shape, landscape, f=None)
// focal_px <= 0 selects the reference default 0.72 * max(image side).
SurfaceModel fit_surface(const std::array<std::vector<cv::Point2d>, 4>& side_pts,
                         const std::array<cv::Point2d, 4>& corners, int img_w, int img_h, bool landscape,
                         double focal_px = 0, int threads = 4);

}  // namespace fs
