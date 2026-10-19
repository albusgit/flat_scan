// End-to-end photo / scan pipeline (code/photo_pipeline.py, mode 'surface', with the
// homography ('v8') fallback) built on the stage ports.
//
// Parity notes:
//  * the unroll map U (surface or homography) is evaluated exactly per pixel, in bands, for
//    the stage-1 page (no full-resolution float map is stored);
//  * the final map is composed on the 4-px grid only: F(p) = U(expand(warp4)(p)), with
//    cv2.remap's 1/32-px quantisation and BORDER_REPLICATE, evaluated in the rotated frame
//    exactly as photo_pipeline stores warpmap.npz; the photo is then sampled once.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "corners.h"
#include "resample.h"
#include "surface.h"
#include "warpmap.h"

namespace fs {

enum class GeomMode { Surface, Homography };

struct PageCorners {
    bool ok = false;
    std::array<cv::Point2d, 4> pts;   // TL, TR, BR, BL in full-resolution photo pixels
    Detection det;                    // detection on the analysis copy
    double scale = 1.0;               // analysis copy = photo * scale
};

struct PipelineOptions {
    bool corners_edited = false;      // user moved a corner -> homography path
    std::array<cv::Point2d, 4> corners;   // used when corners_edited (full-res px)
    double focal_px = 0;              // full-res focal length from EXIF / camera; 0 = 0.72*max side
    bool auto_orient = true;          // decide 0/90/180/270 on the homography page
    bool orient_on_stage1 = false;    // reference-code behaviour (decide on stage 1); off = brief
    int out_w = 1600;                 // A4 page width (portrait); landscape swaps
    bool use_gpu = true;
    double surface_fallback_px = 3.0; // outline residual median above this -> homography
    int max_analysis_side = 2400;     // detection / fit run on a copy no larger than this (reference inputs: 2212-2300 px)
    std::function<void(int, const char*)> progress;   // percent, stage name
};

struct PipelineResult {
    bool ok = false;
    std::string error;
    GeomMode mode = GeomMode::Surface;
    bool fell_back = false;           // surface fit rejected (median residual)
    double fit_median_px = 0, fit_p90_px = 0;
    SurfaceModel model;
    int rot = -1;                     // applied cv::ROTATE_* code
    double orient_sh = 0, orient_sv = 0, orient_upright = 0;
    GridMap final_map;                // backward map (photo px), rotated frame, 4-px grid
    cv::Mat page;                     // BGRA, unwarped (before cleanup)
    cv::Mat stage1;                   // stage-1 page the warp map was estimated on (BGR)
    bool gpu_resample = false;
    std::vector<std::pair<std::string, double>> ms;
    WarpDebug warp;
};

// Page detection for the crop screen (and the pipeline): runs on a copy scaled down to
// opt.max_analysis_side, returns corners in full-resolution pixels.
PageCorners detect_corners(const cv::Mat& photo_bgr, int max_analysis_side = 2400);

PipelineResult run_photo(const cv::Mat& photo_bgr, const PipelineOptions& opt);
// Flatbed scan: warp map only (warpmap.py on the scan itself).
PipelineResult run_scan(const cv::Mat& scan_bgr, const PipelineOptions& opt);

}  // namespace fs
