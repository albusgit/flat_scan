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
#include "orient.h"
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
    double area_frac = -1;            // net: share of the photo its paper covers (-1: n/a)
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
    SurfaceModel model;               // the surface fit (when pc.ok), as run_photo makes it (PipelineOptions::assessed)
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
// kFingerSkinRegion's hand colour, also: within 4 (not 3) spreads of the hand's sampled colour, or in the general skin
// box with Cb >= 95 (orange juice: Cb 79) (still redder / less blue than the paper): a thumb on the page lit unlike
// the hand below it (Cr 148 vs 138). Not on a page whose paper is dark (Y < 120: the finder framed a scene)
const int kFingerLooseSkin = 32;
// a finger found also takes the smooth, even-coloured, non-paper area joined to it within 0.6 x its half width, up to
// half its own area: a painted nail, its rim. Not an area that mostly lies further away (a magazine's red border along
// the edge) or of several colours (a logo strip beside the thumb)
const int kFingerGrow = 64;

struct PipelineOptions {
    bool corners_edited = false;      // user moved a corner -> homography path
    bool force_homography = false;    // the quality gate flagged the crop: homography path
    bool perspective_only = false;    // app: the user turned the unwarp off for this page (homography, no warp map)
    const float* uv_out = nullptr;    // app: the flattening net's grid (uv_spec(uv_crop.model)) for uv_crop of this
    UvCrop uv_crop;                   // photo; null: not run
    const float* uv_out2 = nullptr;   // a second net's grid (e.g. UVDoc next to SyntheticDoc) for uv_crop2: both
    UvCrop uv_crop2;                  // are tried, the straighter stage-1 page wins (see uv_prefer_margin)
    double uv_prefer_margin = 1.0;    // the second net is chosen only if its text-row angle spread is lower by this (deg)
                                      // (DocUNet: UVDoc's straighter rows mostly came with uneven stretch; 0.3 lost 0.45 LD.
                                      // Round 18, fine-tuned SD-HR: 3 -> 1 straightened a long-ruled table it bent,
                                      // dir300 +0.004 MS-SSIM, DocReal -0.003 with the 36k checkpoint)
    // SyntheticDoc-HR (model 2) may outline its own page against the page finder (uv_min_consistency with pass 1).
    // Off: only the page finder, the user's corners or UVDoc define the page (round 18: the fine-tuned SD-HR framed a
    // whole cluttered scene, and pass 1, which also gives its crop, is no independent check)
    bool uv_hr_own = true;
    // a grid with more than this share of folded-over triangles / collapsed cells (uv_grid_validity) is not used,
    // and a composed grid (after fusion and the border snap) that breaks it falls back to the page before the nets.
    // 0: off. (Round 17 logs: no chosen grid folds on 121 + dir300; collapsed share of chosen grids <= 0.009 there,
    // up to 0.024 on DocReal)
    double uv_max_folded = 0;
    double uv_max_collapsed = 0;
    // Edge doubt (round 18): where the two nets (SyntheticDoc-HR and UVDoc) both drew the page and their borders lie
    // more than edge_gap x the page's size apart along at least edge_doubt_frac of one side, nobody knows where that
    // edge is (a strip of desk, a cut corner): PipelineResult::edge_doubt asks the app to have the corners checked.
    // edge_doubt_frac 0: off.
    double edge_gap = 0.015;
    double edge_doubt_frac = 0.3;
    // flat paper (round 16): a net can invent a curl on a flat sheet (a dollar bill on a table came out bowed). When
    // every side of the page finder's outline is straight (side_bow at most flat_edge_bow: the bill 0.010, a curled
    // magazine page 0.15, a cookbook 0.06 - 0.09, a curled coupon strip 0.056), the homography replaces the grid
    // (0: off)
    double flat_edge_bow = 0.012;
    // ... judged on the sides that are the paper's own edge (side_edge <= side_edge_max): with fewer than
    // flat_min_sides of them the paper is not called flat (a page that fills the photo has only the photo's straight
    // edges, which say nothing about its curl). 0: every side counts, the photo's edge too (before round 19)
    int flat_min_sides = 2;
    double flat_frame_side = 0.9;   // ... a side is the photo's edge from this share of its points on it (side_edge)
    // a page that fills the photo, where neither net's grid agreed with the frame (they learnt pages on a background
    // and drew its edges inside the photo): SD-HR once more on the whole photo with a grey margin past its edges
    // (uv_crop pad), through uv_net; its grid is used when it agrees with the frame as any grid must
    // (PipelineResult::uv_full_frame). Off, or no uv_net: the homography, as before round 19.
    bool full_frame_net = true;
    // ... and only when its half-size page's text rows are straighter than the homography's by this (text-row angle
    // spread, deg; both pages with >= 5 rows): a flat form filling the photo is right as the homography has it,
    // and the net's grid only bends it (round 19: the wagering form's ruled box came out wavy)
    double full_frame_min_gain = 0.5;
    // both nets accepted: where the chosen grid's border misses the paper by more than fuse_miss x the page's size
    // and the other's fits it, the other grid is blended in around there (uv_fuse_border). 0: off.
    double fuse_miss = 0.015;
    // ... and only when the other net's whole outline fits the paper (mask IoU) better by at least this: where it
    // fits worse overall, its border near the paper's edge has followed something else (a shadow, the table)
    double fuse_min_gain = 0.003;
    // the surface model is fitted to the page finder's outline only: where fewer than surface_min_sides of its sides
    // are the paper's own edge (the rest lie on the photo's edge: the paper runs out of the photo, side_edge >
    // side_edge_max), or every such side is straight (flat_edge_bow), it has nothing to model and the homography
    // is stage 1 (a page that fills the photo came out with an empty band where the model curled it). 0: off.
    int surface_min_sides = 2;
    double side_edge_max = 0.5;
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
    // ... and only when at least this share of the grid's page is paper by the page finder's mask (round 16: a
    // self-consistent net drew a receipt's page into the coffee cups beside it; 0: off)
    double uv_own_min_paper = 0;   // off: at 0.9 the receipt improved, 5 of 6 other pages got the page finder's worse crop
    // Border snap (round 16, uvdoc.h uv_snap_border): the grid's border onto the page finder's paper edge, searched
    // within snap_search x the page's size; displacements below snap_dead_px (photo px) are left; a side most of
    // whose points need more than snap_lost x the page's size: the net lost the page (the page before the nets)
    bool snap_border = false;   // off: on 121 photos 2 pages better (coupon, bill a little), 12 worse (the mask edge is too rough: wavy text)
    // Frame snap (uvdoc.h uv_extend_to_frame): where the page runs off the photo, the page extended past the grid's border
    // to the photo's edge, the grid continued straight on (the nets put the end of a page that runs off the photo inside
    // it: a flyer lost its top line, a receipt its number), searched within frame_snap_search x the page's size
    bool frame_snap = true;
    double frame_snap_search = 0.08;
    double frame_snap_side = 0.3;   // the outline side's share of points on the photo's edge for its grid side to go out
    double snap_search = 0.08;
    double snap_dead_px = 3.0;
    double snap_lost = 0.06;
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
    bool auto_lines = true;           // ... straightens every printed line it finds; off: only the lines in the user's
                                      // Straighten paint (line_paint) and the lines the user drew (user_lines)
    bool share_rows = true;           // UVDoc pages: pieces of one ruled line share one ideal (warpmap)
    bool robust_solve = true;         // IRLS in the warp map: samples the field does not follow lose weight (warpmap)
    bool colour_edges = true;         // neural pages: long straight boundaries between colour areas are lines too (warpmap colour_edges)
    bool second_line_pass = false;    // neural pages: the line stage again on its own output, kept if the rules come out straighter
                                      // (off since the text-line net: real35 kept it on 2 of 24 pages, straightness the same without it,
                                      // the warp map 24 % faster; 1.6 s of a 5.4 s page on the S24 FE)
    bool faint_lines = true;          // also find and trace printed rules too pale for the darkness-45 threshold (warpmap faint_rules)
    bool erase_fingers = true;        // fill skin-coloured non-paper areas at the page border with paper
    int finger_rules = kFingerSkinRegion | kFingerHandColour | kFingerLooseSkin | kFingerGrow;   // finger tests (kFinger* bits; tools/fingers/eval.py measures each)
    bool keep_unerased = false;       // keep the page before the finger fill (PipelineResult::page_unerased; the app's brush)
    double max_local_rescale = 0.35;  // neural pages: line corrections that rescale a spot by more than this are faded out there (0: off;
                                      // 0.15 also faded real crease corrections, 0.5 left part of a bump)
    double max_line_distortion = 0.2;        // UVDoc pages: line refinement kept only below these
    double max_line_distortion_frac = 0.01;  // (local area change p99, share of cells above 0.2)
    // ... with edge_stretch_allow more allowed near the border: a cell within edge_stretch_band x the page's shorter
    // side of it counts its local area change less that much in both (and in the overstretched spots the retry drops
    // lines through) - it never counts more than before. A page curls at its edges and
    // corners, where the nets leave it compressed and straightening its lines needs a real stretch: the coupon's
    // curled corner (its dashed border, header and title rows) was dropped as "overstretched" (round 19). Inside,
    // where the nets have flattened it, a stretch is a tracing failure (music staves, a rule traced into text).
    // 0: the whole page measured (before round 19).
    double edge_stretch_band = 0.1;
    double edge_stretch_allow = 0.6;
    // Neural pages: photos and drawings printed on the page (product shots, illustrations) give the line stage no lines
    // (their edges and the text on them are curved by what they show, not by the paper): picture_mask in pipeline.cpp,
    // its areas joined to the ignore areas. Blocks of 32 px whose colour spread (with the text strokes taken out) is
    // above picture_spread and whose share of paper is below picture_paper; the text-line net vetoes blocks it marks as
    // text more than picture_text_veto (mean probability; 0: no veto). Off until measured (round 19: the poster's
    // bottles were bent to straighten their caps and label bands).
    bool picture_guard = false;
    double picture_spread = 18, picture_paper = 0.5, picture_text_veto = 0;
    int picture_mode = 1;             // 2: picture_mask_strict (colour that neither shading nor one ink on one paper explains)
    // The shape lock: where the user painted Keep as is (and where picture_guard found pictures) the line stage may move
    // the page only rigidly - a rotation and a shift, fitted to the correction just around the area (the paper there),
    // blended into the rest over lock_band x the page's shorter side. Leaving the lines there out (the ignore path) is
    // not enough: the smooth solve still bends an area with no lines in it to follow the lines beside it (round 19, the
    // poster's bottles with Keep as is over them: local area -1.2 % / +2.5 % of the nets' page).
    bool lock_shapes = true;
    double lock_band = 0.06;
    const float* seg_logits = nullptr;   // page-finder net output for this photo (2 x 256 x 256);
                                         // null: colour-threshold paper_mask (the feature flag is off)
    int page_index = 0;               // book spread: which page of the photo (0 = the larger one)
    // automatic corners: this photo's page page_index as assess_pages found it with the same seg_logits, focal_px
    // and max_analysis_side (the crop screen's detection): its corners and surface fit are taken instead of made
    // again (the same values). Null: made here.
    const PageAssessment* assessed = nullptr;
    std::array<cv::Point2d, 4> corners;   // used when corners_edited (full-res px)
    // with corners_edited: the user's outline between the corners (the crop screen's edge points), full-res px;
    // edge_pts[k] lies on side k, from corners[k] to corners[k + 1]. A point on any side makes the user's outline
    // the page's as the page finder's would be (outline_corners: its surface model, the nets checked against its
    // paper), instead of the 4-corner homography of a plain corner edit.
    std::array<std::vector<cv::Point2d>, 4> edge_pts;
    bool user_outline() const { return corners_edited && (!edge_pts[0].empty() || !edge_pts[1].empty() || !edge_pts[2].empty() || !edge_pts[3].empty()); }
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
    bool render_true_shape = true;    // the output page at the paper's measured shape (page_hw; A4 / US Letter when
                                      // within true_shape_snap of one), same pixel count; off = A4 (reference parity).
                                      // The analysis page and final_map stay A4 either way
    double true_shape_snap = 0.04;    // = the app's PageLook kSnap
    bool use_gpu = true;
    double surface_fallback_px = 3.0; // outline residual median above this -> homography
    int max_analysis_side = 2400;     // detection / fit run on a copy no larger than this (reference inputs: 2212-2300 px)
    std::vector<std::array<float, 4>> ignore;   // app: areas whose lines the warp map must not use, as
                                                // fractions [x0, y0, x1, y1] of the output page (rotated frame)
    // app: lines the user drew along rules the line stage missed (the line editor's Add line): points as
    // fractions of the output page (rotated frame) as the user saw it. Group k was drawn on the page the
    // groups before it made; each is straightened in turn after the line stage (apply_user_lines).
    struct UserLine { int group = 0; std::vector<cv::Point2f> pts; };
    std::vector<UserLine> user_lines;
    // app: the line brush (the Flatten tab's paint): strokes as fractions of the output page (rotated frame) as the user
    // saw it, radius a fraction of that page's width. Straighten (mode 0): when any is painted, the line stage uses only
    // the lines there, auto_lines or not. Keep as is (mode 1): the lines there are left out (as the ignore areas), so a
    // picture (a product photo, a drawing) moves with the paper around it and is not bent to straighten its edges.
    struct LinePaint { int mode = 0; float radius = 0.02f; std::vector<cv::Point2f> pts; };
    std::vector<LinePaint> line_paint;
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
    // The orientation net (orient.h OrientNetFn, orientnet.onnx): set, it decides the page's turn (any of the four)
    // when its margin is at least orient_net_margin; below that, or unset, the text-axis rule decides as before.
    OrientNetFn orient_net;
    double orient_net_margin = 2.0;   // 0 wrong turns in 1200 tests (dir300 scans x 4 turns)
    // The flattening nets for the zoomed second pass (uv_zoom): net(model, io) replaces the net's
    // input with its grid. Empty: no zoomed pass.
    UvNetFn uv_net;
    bool zoom_pass = false;           // off: the two parts' grids disagree by a few px and the blend shears the overlap band (receipt table); text-row spread does not see it
    int zoom_model = 0;               // net for the zoomed pass: UVDoc (on the receipt UVDoc halves 2.22 -> 1.58 deg, SyntheticDoc 2.50; -1: the page's net)
    int zoom_nu = 1, zoom_nv = 2;     // parts across / down: halves along the long side (quarters: 2.32 deg, two cut edges each)
    double zoom_max_worse = 0.1;      // deg of text-row spread the zoomed grid may lose and still be kept
    int shade_long_side = 512;
    // the map's clamp: a pixel is brightened by at most 1 / shade_floor (the net of round 16 is trained for 0.12)
    float shade_floor = 0.3f;
    bool shade_tint = true;      // a coloured cast shadow's colour taken out after the net (pipeline.cpp neutralise_shadow_tint)
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
    double uv_grid_check[2][3] = {{-1, -1, -1}, {-1, -1, -1}};   // each net's grid: folded, collapsed, outside (uv_grid_validity)
    bool uv_grid_rejected[2] = {false, false};   // ... set aside by uv_max_folded / uv_max_collapsed
    bool uv_composed_broken = false;             // the composed grid broke them: the page before the nets was kept
    // the two nets' borders: per side (top, right, bottom, left of the page) the share of its points where they lie
    // more than PipelineOptions::edge_gap x the page's size apart (-1: not both nets)
    double net_edge_gap[4] = {-1, -1, -1, -1};
    bool edge_doubt = false;   // a neural page whose edge the nets disagree on (net_edge_gap >= edge_doubt_frac)
    double uv_agree_quad[2] = {-1, -1}, uv_agree_mask[2] = {-1, -1};   // each net's outline against the corners, the mask (IoU)
    double uv_paper_share[2] = {-1, -1};   // share of each net's page that the page finder's mask calls paper (-1: not measured)
    double snap_px[2] = {-1, -1};     // border snap: median / max displacement (photo px; -1: not run)
    int snap_sides = -1;              // ... sides moved
    double frame_snap_px = -1;        // frame snap: the largest move out to the photo's edge (photo px; -1: none)
    int frame_snap_sides = 0;         // ... sides moved
    double side_bow[4] = {-1, -1, -1, -1};   // page finder's sides (top, right, bottom, left): p95 bow from the chord / its length
    int side_n[4] = {0, 0, 0, 0};             // the outline points on each side
    double side_edge[4] = {-1, -1, -1, -1};  // their share on the photo's own edge (the paper runs out of the photo there)
    double uv_flat_resid[3] = {-1, -1, -1};   // the grid's distance from its best homography (page px: p50, p95, max)
    bool uv_lost_page = false;        // ... a side was far off the paper's edge: the net was set aside
    bool uv_defines_page = false;     // the grid was used over the page finder's disagreement: corners and paper from it
    int uv_model = -1;                // the net whose grid was used (UvCrop::model); -1: none
    double uv_spread[2] = {-1, -1};   // text-row angle spread of each net's stage-1 page (deg; -1: not measured)
    int fuse_runs = 0;                 // border runs taken from the other net (uv_fuse_border)
    double fuse_worst[2] = {-1, -1};   // the chosen, the other grid's largest border miss / page size
    double fuse_moved_px = 0, fuse_area = 0;
    bool fuse_rejected = false;        // the blend folded the grid: not used
    bool surface_unfounded = false;   // the surface model was set aside: no paper edge to fit, or all of them straight
    bool flat_page = false;           // the net's grid was set aside: the paper is flat (PipelineOptions::flat_edge_bow)
    bool uv_rejected = false;         // UVDoc ran but disagreed with the page finder (fallback used)
    bool uv_full_frame = false;       // the grid is SD-HR's on the padded whole photo (PipelineOptions::full_frame_net)
    double uv_full_frame_agree = -1;  // ... its outline's agreement with the frame (-1: not run)
    double uv_full_frame_spread[2] = {-1, -1};   // ... text-row spread of the homography's, the grid's half-size page (deg)
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
    bool edges_bent = false;          // a line-stage candidate bent a straight edge of the page (and was not kept)
    bool lines_spots_dropped = false; // ... too large at first: kept without the lines through the overstretched spots
    double second_pass[3] = {-1, -1, -1};   // ... the second pass: its distortion, its rules' deviation before / after (px)
    bool second_pass_kept = false;
    double second_pass_ms = -1;       // ... its time (resample, rule net, trace and solve; -1: not run)
    double line_retry_ms = -1;        // ... the retries after a first map that overstretched the page (-1: none)
    double line_straightness[2] = {-1, -1};    // ... the traced rules' mean p95 deviation (px): on the page as it is / through the kept map
    double bumps_faded = 0;           // ... share of the page where a local rescale was faded out (soften_bumps)
    double fingers_erased = 0;        // share of the page filled where fingers held it
    cv::Mat finger_mask;              // CV_8U page size, 255 = filled as a finger (empty: none)
    bool fingers_over_text = false;   // ... and text runs into it
    int user_lines_used = 0;          // the user's lines (user_lines) that found ink and were straightened
    bool lines_painted = false;       // the line stage ran on the Straighten paint only (line_paint)
    double picture_share = -1;        // picture_guard: the share of the page it found pictures on (-1: not run)
    cv::Mat picture_mask;             // ... and where (CV_8U, the stage-1 page, 255 there; empty if none)
    double shape_locked = -1;         // lock_shapes: the share of the page held rigid (-1: nothing to lock)
    cv::Mat page_unerased, paper_unerased;   // with keep_unerased: page and paper before the finger fill
    bool shadows_removed = false;     // the shading map was applied to the page
    bool shade_net = false;           // ... the shadow net's map (ShadeNet)
    bool rule_net = false;            // the rule net ran in the line stage (RuleNet)
    int net_dropped = 0, net_added = 0, net_joined = 0, net_kept = 0;   // ... its effect (NetLineStats, Samples::n_net_kept)
    SurfaceModel model;
    int rot = -1;                     // applied cv::ROTATE_* code
    double orient_sh = 0, orient_sv = 0, orient_upright = 0;
    int orient_net_turn = -1, orient_net_patches = 0;   // the orientation net's answer (k x 90 deg cw), its patches
    double orient_net_margin = -1;
    bool orient_by_net = false;
    GridMap final_map;                // backward map (photo px), rotated frame, 4-px grid, analysis size
    double page_hw = -1;              // height / width of the paper itself, measured along final_map (-1: not measured)
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
    // seen by the global stage only, 3 = a line the user drew (user_lines). Positions are the
    // straightened ones (on the output page).
    std::vector<std::array<float, 5>> lines;
};

// Page detection for the crop screen (and the pipeline): the corner fit runs on a copy scaled
// down to max_analysis_side, corners are returned in full-resolution pixels. With seg_logits the
// page finder net (+ edge snapping) runs on the full photo, as the reference; otherwise the
// colour-threshold paper_mask. page_index picks the page of a book spread.
// The page as the user outlined it (PipelineOptions::corners / edge_pts, full-res px) in the page finder's form: its
// outline, the points of each side, the paper inside it, on the analysis copy of max_analysis_side.
PageCorners outline_corners(const cv::Mat& photo_bgr, const std::array<cv::Point2d, 4>& corners,
                            const std::array<std::vector<cv::Point2d>, 4>& edge_pts, int max_analysis_side = 2400);
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
// stroke is continued into the hole).
void fill_finger(cv::Mat& page_bgra, const cv::Mat& mask, cv::Mat* paper);
// Whether print runs into [mask] from the page around it (ink in a ring outside it, away from the border).
bool finger_over_text(const cv::Mat& page_bgra, const cv::Mat& mask, const cv::Mat& paper);
// The shape lock (PipelineOptions::lock_shapes) on a line-stage map (output -> page, 4-px grid): each piece of lock
// (CV_8U, page size) only moves rigidly with the page around it, blended back over band px; the share held rigid.
double lock_shapes(GridMap& m, const cv::Mat& lock, double band);
// Flatbed scan: warp map only (warpmap.py on the scan itself).
PipelineResult run_scan(const cv::Mat& scan_bgr, const PipelineOptions& opt);

}  // namespace fs
