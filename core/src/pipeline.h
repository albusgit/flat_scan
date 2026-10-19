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
#include "pagefind.h"
#include "resample.h"
#include "surface.h"
#include "warpmap.h"

namespace fs {

enum class GeomMode { Surface, Homography };

enum class Finder { ColourThreshold, Net };

struct PageCorners {
    bool ok = false;
    std::array<cv::Point2d, 4> pts;   // TL, TR, BR, BL in full-resolution photo pixels
    Detection det;                    // detection on the analysis copy
    double scale = 1.0;               // analysis copy = photo * scale
    // page finder (photo_pipeline.find_page)
    Finder finder = Finder::ColourThreshold;
    int n_pages = 1;                  // 2: book spread (this is page `page_index`)
    int page_index = 0;
    bool whole_frame = false;         // no page found: the whole frame (net: info['fallback'])
    bool spread = false;              // net: a straight gutter split the page (pages may still be 1)
    double net_margin = -1, edge_strength = -1, edge_at_limit = -1;   // net quality signals (-1: n/a)
    cv::Point2d centroid;             // page outline centroid, photo px (book page order)
};

// photo_pipeline.quality_gate: reasons not to trust the automatic crop (empty = trusted).
// fit_median_full_px: surface-fit median residual in full-resolution photo px (<0: not fitted).
std::vector<std::string> quality_gate(const PageCorners& pc, double fit_median_full_px, cv::Size photo);

// The automatic crop of one page as the crop screen sees it: corners, surface fit, gate.
struct PageAssessment {
    PageCorners pc;
    double fit_median_px = -1;        // full-resolution px; -1: no fit
    std::vector<std::string> reasons; // quality_gate
};

struct PipelineOptions {
    bool corners_edited = false;      // user moved a corner -> homography path
    bool force_homography = false;    // the quality gate flagged the crop: homography path
    const float* seg_logits = nullptr;   // page-finder net output for this photo (2 x 256 x 256);
                                         // null: colour-threshold paper_mask (the feature flag is off)
    int page_index = 0;               // book spread: which page of the photo (0 = the larger one)
    std::array<cv::Point2d, 4> corners;   // used when corners_edited (full-res px)
    double focal_px = 0;              // full-res focal length from EXIF / camera; 0 = 0.72*max side
    bool auto_orient = true;          // decide 0/90/180/270 on the homography page
    bool orient_on_stage1 = false;    // reference-code behaviour (decide on stage 1); off = brief
    double orient_min_conf = 0.2;     // sideways pages: CW only if the upright cue is below -this (it is
                                      // Hebrew-specific: ~0 on other or low-confidence pages)
    bool reference_orient = false;    // ungated decision, exactly the reference (parity tests)
    int out_w = 1600;                 // analysis page width (portrait; landscape swaps) - the reference's
    int render_w = 0;                 // output page width (portrait) if different from out_w; 0 = same
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
    std::vector<std::string> needs_manual_crop;   // quality_gate reasons (homography path when any)
    PageCorners detection;            // the automatic detection (finder info); ok = false if not run
    SurfaceModel model;
    int rot = -1;                     // applied cv::ROTATE_* code
    double orient_sh = 0, orient_sv = 0, orient_upright = 0;
    GridMap final_map;                // backward map (photo px), rotated frame, 4-px grid, analysis size
    GridMap render_map;               // the same map at the output resolution (used for the resample)
    cv::Mat page;                     // BGRA, unwarped (before cleanup)
    cv::Mat stage1;                   // stage-1 page the warp map was estimated on (BGR)
    cv::Mat paper;                    // CV_8U, render size: 255 = paper per the detection outline (soft
                                      // 1-2 px edge); 0 = folded / torn / missing -> filled white. Empty = no mask
    cv::Mat before;                   // BGRA, render size: perspective-corrected only (homography), same frame
    GridMap disp;                     // render size, 4-px grid: (dx, dy) in output px from each page point to
                                      // where it sat on the `before` page (scans: on the scan)
    bool gpu_resample = false;
    std::vector<std::pair<std::string, double>> ms;
    WarpDebug warp;
};

// Page detection for the crop screen (and the pipeline): the corner fit runs on a copy scaled
// down to max_analysis_side, corners are returned in full-resolution pixels. With seg_logits the
// page finder net (+ edge snapping) runs on the full photo, as the reference; otherwise the
// colour-threshold paper_mask. page_index picks the page of a book spread.
PageCorners detect_corners(const cv::Mat& photo_bgr, int max_analysis_side = 2400, const float* seg_logits = nullptr,
                           int page_index = 0);
// Every page of the photo (1, or 2 for a book spread) with its surface fit and quality gate.
std::vector<PageAssessment> assess_pages(const cv::Mat& photo_bgr, const float* seg_logits, double focal_px,
                                         int max_analysis_side = 2400);

PipelineResult run_photo(const cv::Mat& photo_bgr, const PipelineOptions& opt);
// Flatbed scan: warp map only (warpmap.py on the scan itself).
PipelineResult run_scan(const cv::Mat& scan_bgr, const PipelineOptions& opt);

}  // namespace fs
