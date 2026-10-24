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
#include "uvdoc.h"
#include "warpmap.h"

namespace fs {

// stage 1: the outline's surface model, the 4-corner homography, or the UVDoc grid (app)
enum class GeomMode { Surface, Homography, Neural };
inline const char* geom_mode_name(GeomMode m) {
    return m == GeomMode::Surface ? "surface" : m == GeomMode::Neural ? "uvdoc" : "homography";
}

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

// The flattening nets' grids (PipelineOptions::flat_nets) for the uv_out / uv_crop, uv_out2 / uv_crop2
// and uv_out_pass1 / uv_crop_pass1 slots; an empty vector leaves its slot as the options have it.
struct FlatNetGrids {
    std::vector<float> grid1, grid2, grid_pass1;
    UvCrop crop1, crop2, crop_pass1;
};
// The page corners for the nets (TL, TR, BR, BL, photo px): run_photo's detection, or the corners set
// by hand. It may wait for the detection; null: no page was found.
using NetCornersFn = std::function<const std::array<cv::Point2d, 4>*()>;
using FlatNetsFn = std::function<FlatNetGrids(const NetCornersFn& corners)>;

// Finger detection rules on top of the base test (skin share, reach, smoothness), PipelineOptions::finger_rules
const int kFingerContact = 1;       // the candidate lies along the page border (not only its bounding box)
const int kFingerAdaptiveSkin = 2;  // skin colour sampled from the hand outside the page, not a fixed box
const int kFingerShape = 4;         // finger width: 2-16 % of the page width
const int kFingerSkinRegion = 8;    // also skin areas the page mask counted as paper (needs the hand outside the page)
const int kFingerHandColour = 16;   // redder / less blue than the paper on average; from at most two sides

struct PipelineOptions {
    bool corners_edited = false;      // user moved a corner -> homography path
    bool force_homography = false;    // the quality gate flagged the crop: homography path
    bool perspective_only = false;    // app: the user turned the unwarp off for this page (homography, no warp map)
    const float* uv_out = nullptr;    // app: the flattening net's grid (uv_spec(uv_crop.model)) for uv_crop of this
    UvCrop uv_crop;                   // photo; null: not run
    const float* uv_out2 = nullptr;   // a second net's grid (e.g. UVDoc next to SyntheticDoc) for uv_crop2: both
    UvCrop uv_crop2;                  // are tried, the straighter stage-1 page wins (see uv_prefer_margin)
    double uv_prefer_margin = 3.0;    // the second net is chosen only if its text-row angle spread is lower by this (deg)
                                      // (DocUNet: UVDoc's straighter rows mostly came with uneven stretch; 0.3 lost 0.45 LD)
    const float* uv_shading = nullptr;   // SyntheticDoc's shading map (3 x in_h x in_w) for uv_shading_crop
    UvCrop uv_shading_crop;
    bool remove_shadows = true;       // divide the page by the shading map
    bool shadow_full_strength = true;    // ... by all of it (the reference weakens it on bright photos)
    double uv_min_agreement = 0.9;    // UVDoc outline vs the page finder's mask (IoU); below: not used
    double uv_min_agreement_hr = 0.93;    // SyntheticDoc high-res (model 2): its fine grid traces a right page at >= 0.957
                                          // on the 35 photos; a wrong crop (docres__208_in) passed 0.9 at 0.914
    double uv_min_agreement_quad = 0.85;   // ... vs the corners when they were set by hand (no mask)
    // The first pass (uv_crop_on_grid): a net's grid on a rough region whose outline gave uv_crop /
    // uv_crop2. A grid whose outline matches it (the same page found twice, from two framings) at
    // >= uv_min_consistency is used even when the page finder disagrees (a page finder that split a
    // folded page, cut a curled one or lost a coloured band), down to uv_consistent_floor; the page
    // corners and paper outline then come from the grid.
    const float* uv_out_pass1 = nullptr;
    UvCrop uv_crop_pass1;
    double uv_min_consistency = 0.95;
    double uv_consistent_floor = 0.5;
    // The flattening nets run by run_photo itself (nets.h flat_nets_fn), instead of uv_out* given in
    // advance; their grids fill those slots. overlap_nets: on a worker thread, started before the
    // detection, while the CPU stages that do not need a grid run (detection, surface fit, quality
    // gate, the orientation page and a speculative orientation decision); run_photo waits for them
    // before the grid check and redoes what a grid changed (new page corners: the orientation page;
    // a different page shape: the decision). The callback must not need the calling thread (no JNI
    // env, no GL context). Off: the same callback on the calling thread after the detection, nothing
    // speculative. The outputs are bit-identical either way (Gate D).
    FlatNetsFn flat_nets;
    bool overlap_nets = true;
    bool line_stage = true;           // the warp map (line refinement) on top of stage 1
    bool share_rows = true;           // UVDoc pages: pieces of one ruled line share one ideal (warpmap)
    bool robust_solve = true;         // IRLS in the warp map: samples the field does not follow lose weight (warpmap)
    bool colour_edges = true;         // neural pages: long straight boundaries between colour areas are lines too (warpmap colour_edges)
    bool second_line_pass = true;     // neural pages: the line stage again on its own output, kept if the rules come out straighter
    bool faint_lines = true;          // also find and trace printed rules too pale for the darkness-45 threshold (warpmap faint_rules)
    bool erase_fingers = true;        // fill skin-coloured non-paper areas at the page border with paper
    int finger_rules = kFingerSkinRegion | kFingerHandColour;   // finger tests (kFinger* bits; tools/fingers/eval.py measures each)
    bool keep_unerased = false;       // keep the page before the finger fill (PipelineResult::page_unerased; the app's brush)
    double max_local_rescale = 0.35;  // neural pages: line corrections that rescale a spot by more than this are faded out there (0: off;
                                      // 0.15 also faded real crease corrections, 0.5 left part of a bump)
    double max_line_distortion = 0.2;        // UVDoc pages: line refinement kept only below these
    double max_line_distortion_frac = 0.01;  // (local area change p99, share of cells above 0.2)
    const float* seg_logits = nullptr;   // page-finder net output for this photo (2 x 256 x 256);
                                         // null: colour-threshold paper_mask (the feature flag is off)
    int page_index = 0;               // book spread: which page of the photo (0 = the larger one)
    std::array<cv::Point2d, 4> corners;   // used when corners_edited (full-res px)
    double focal_px = 0;              // full-res focal length from EXIF / camera; 0 = 0.72*max side
    bool auto_orient = true;          // decide 0/90/180/270 on the homography page
    bool orient_on_stage1 = false;    // reference-code behaviour (decide on stage 1); off = brief
    bool orient_true_aspect = true;   // decide on the page stretched to its measured height / width (page_hw_ratio)
    double orient_min_ratio = 2.0;    // a page turns only if its vertical text scores this many times the horizontal
                                      // (sideways pages 3.8-5.9; upright pages up to 1.0, receipts 1.7)
    double orient_min_conf = 0.2;     // sideways pages: CW only if the upright cue is below -this (it is
                                      // Hebrew-specific: ~0 on other or low-confidence pages)
    bool reference_orient = false;    // ungated decision, exactly the reference (parity tests)
    int out_w = 1600;                 // analysis page width (portrait; landscape swaps) - the reference's
    int render_w = 0;                 // output page width (portrait) if different from out_w; 0 = same
    bool use_gpu = true;
    double surface_fallback_px = 3.0; // outline residual median above this -> homography
    int max_analysis_side = 2400;     // detection / fit run on a copy no larger than this (reference inputs: 2212-2300 px)
    std::vector<std::array<float, 4>> ignore;   // app: areas whose lines the warp map must not use, as
                                                // fractions [x0, y0, x1, y1] of the output page (rotated frame)
    std::function<void(int, const char*)> progress;   // percent, stage name
    // The shadow net (ShadeNet, tools/shadow_net): called with the flattened page at shade_long_side
    // (portrait; planar BGR in [0, 1], 3 x h x w, sides multiples of 32), it replaces the buffer with
    // the page's shading map (same layout) and returns true; the page is divided by the map
    // (brighten only). Empty / false: not run. Runs after SyntheticDoc's map when both are given
    // and shade_after_uv; otherwise it replaces it.
    std::function<bool(std::vector<float>& io, int w, int h)> shade_net;
    // The rule net (RuleNet, tools/rule_net): called in the line stage with the stage-1 page at half
    // size (planar BGR in [0, 1], 3 x h x w, sides multiples of 32), it replaces the buffer with
    // 2 x h x w rule probabilities [horizontal, vertical] (RuleNetFn). Empty / false: the line stage
    // finds rules from the darkness map alone.
    RuleNetFn rule_net;
    // The flattening nets for the zoomed second pass (uv_zoom): net(model, io) replaces the net's
    // input with its grid. Empty: no zoomed pass.
    UvNetFn uv_net;
    bool zoom_pass = false;           // off: the two parts' grids disagree by a few px and the blend shears the overlap band (receipt table); text-row spread does not see it
    int zoom_model = 0;               // net for the zoomed pass: UVDoc (on the receipt UVDoc halves 2.22 -> 1.58 deg, SyntheticDoc 2.50; -1: the page's net)
    int zoom_nu = 1, zoom_nv = 2;     // parts across / down: halves along the long side (quarters: 2.32 deg, two cut edges each)
    double zoom_max_worse = 0.1;      // deg of text-row spread the zoomed grid may lose and still be kept
    int shade_long_side = 512;
    bool scan_flattened = false;      // run_scan: the input is a page another method flattened (neural-page line stage + guards)
    bool shade_after_uv = false;
};

struct PipelineResult {
    bool ok = false;
    std::string error;
    GeomMode mode = GeomMode::Surface;
    bool fell_back = false;           // surface fit rejected (median residual)
    double fit_median_px = 0, fit_p90_px = 0;
    std::vector<std::string> needs_manual_crop;   // quality_gate reasons (homography path when any)
    PageCorners detection;            // the automatic detection (finder info); ok = false if not run
    double uv_agreement = -1;         // UVDoc outline vs the page (IoU); -1: UVDoc not run
    double uv_consistency = -1;       // ... vs the first pass's outline (-1: no first pass)
    bool uv_defines_page = false;     // the grid was used over the page finder's disagreement: corners and paper from it
    int uv_model = -1;                // the net whose grid was used (UvCrop::model); -1: none
    double uv_spread[2] = {-1, -1};   // text-row angle spread of each net's stage-1 page (deg; -1: not measured)
    bool uv_rejected = false;         // UVDoc ran but disagreed with the page finder (fallback used)
    int uv_zoom_quarters = -1;        // zoomed pass: quarters whose grid was used (-1: not run)
    double uv_zoom_spread[2] = {-1, -1};   // ... text-row spread with the first / the zoomed grid
    bool uv_zoom = false;             // ... the zoomed grid was kept
    int nets_overlap = -1;            // flat_nets: -1 not run, 0 after the detection, 1 beside the CPU stages
    bool orient_page_reused = false;  // ... the orientation page made while they ran was kept (same corners)
    bool orient_reused = false;       // ... and its orientation decision (the same page-shape test)
    bool before_reused = false;       // ... and the before page made then (the same corners and turn)
    double nets_ms = -1;              // ... their own wall time (ms)
    double nets_wait_ms = -1;         // ... how long run_photo waited for them after its own stages
    double nets_join_ms = -1;         // ... run_photo's start to the grids in hand
    double line_distortion = -1;      // UVDoc pages: the line refinement's local area change (p99)
    bool lines_rejected = false;      // ... too large: the UVDoc page was kept without it
    bool lines_spots_dropped = false; // ... too large at first: kept without the lines through the overstretched spots
    double second_pass[3] = {-1, -1, -1};   // ... the second pass: its distortion, its rules' deviation before / after (px)
    bool second_pass_kept = false;
    double second_pass_ms = -1;       // ... its time (resample, rule net, trace and solve; -1: not run)
    double line_retry_ms = -1;        // ... the retries after a first map that overstretched the page (-1: none)
    double line_straightness[2] = {-1, -1};    // ... the traced rules' mean p95 deviation (px): on the page as it is / through the kept map
    double bumps_faded = 0;           // ... share of the page where a local rescale was faded out (soften_bumps)
    double fingers_erased = 0;        // share of the page filled where fingers held it
    cv::Mat finger_mask;              // CV_8U page size, 255 = filled as a finger (empty: none)
    bool fingers_over_text = false;   // ... and text runs into it (outlined, not invented)
    cv::Mat page_unerased, paper_unerased;   // with keep_unerased: page and paper before the finger fill
    bool shadows_removed = false;     // the shading map was applied to the page
    bool shade_net = false;           // ... the shadow net's map (ShadeNet)
    bool rule_net = false;            // the rule net ran in the line stage (RuleNet)
    int net_dropped = 0, net_added = 0, net_joined = 0, net_kept = 0;   // ... its effect (NetLineStats, Samples::n_net_kept)
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
    // the lines the warp map was fitted to, for the app's line editor: [kind, x0, y0, x1, y1] with
    // the ends as fractions of the output page; kind 0 = ruled line, 1 = text row, 2 = grid line
    // seen by the global stage only. Positions are the straightened ones (on the output page).
    std::vector<std::array<float, 5>> lines;
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

// Fills `mask` (CV_8U, page size, non-zero = fill) of a BGRA page with the paper around it (finger
// erase, the app's Erase tool): a smooth fill, not invented content. paper (optional, CV_8U page
// size): marked as paper there.
void fill_with_paper(cv::Mat& page_bgra, const cv::Mat& mask, cv::Mat* paper);
// Fingers: [mask] filled with the paper around it, print taken out of that paper first (not inpainted: no
// stroke is continued into the hole); with [mark], where text runs into the area it is outlined dashed grey.
void fill_finger(cv::Mat& page_bgra, const cv::Mat& mask, cv::Mat* paper, bool mark = true);
// Whether print runs into [mask] from the page around it (ink in a ring outside it, away from the border).
bool finger_over_text(const cv::Mat& page_bgra, const cv::Mat& mask, const cv::Mat& paper);
// Flatbed scan: warp map only (warpmap.py on the scan itself).
PipelineResult run_scan(const cv::Mat& scan_bgr, const PipelineOptions& opt);

}  // namespace fs
