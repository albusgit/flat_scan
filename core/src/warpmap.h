// Warp map from page features: port of code/warpmap.py (unwarp_image with every optional
// switch at its default: UNWARP_PROP_*, UNWARP_LSD and UNWARP_R3_* off, G=8, GC=16,
// REG='biharmonic', TEXT_W=1) and the helpers it imports from dewarp.py and clean.py.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <functional>
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

// App: the rule net (tools/rule_net, RuleNet). io in: the page at half size, 3 x h x w planar BGR
// in [0, 1] (w, h multiples of 32); out (the same vector, resized): 2 x h x w probabilities that a
// pixel belongs to a printed straight rule, [horizontal, vertical]. False: not available.
using RuleNetFn = std::function<bool(std::vector<float>& io, int w, int h)>;

// get_lines' use of the rule net (counts, for the processing details)
struct NetLineStats {
    int dropped = 0;   // found segments the net does not mark as rules (text rows, letters, edges)
    int added = 0;     // rules only the net saw (too faint, too bent for the straight-run search)
    int too_faint = 0; // ... not added: fainter than a faint printed rule (seen through the paper)
    int joined = 0;    // found pieces joined into one line along a rule the net sees whole
    int dropped_global = 0;   // global-stage grid lines the net does not mark as rules
};

struct Samples {      // collect(): rows (along, ideal, disp)
    std::vector<std::array<double, 3>> h, v;
    int n_bowed = 0;  // traces left out as curves (share_rows)
    // app: the stretch along each line (H, then V, by index) its samples cover, extensions included
    // ([NaN, NaN]: the line gave no samples); n_extended: line ends continued past the detected segment
    std::vector<std::array<double, 2>> span_h, span_v;
    int n_extended = 0;
    int n_net_kept = 0;   // bowed traces kept because the rule net marks them as a rule (a warped rule)
    // app: those extensions, [ideal, start, end] along the line (bump guard support, pipeline.cpp)
    std::vector<std::array<double, 3>> ext_h, ext_v;
    // app: each line's trace (along, across; sorted along) and its ideal, for corner_junctions
    struct Trace { double ideal = 0; size_t line = 0; std::vector<double> t, c; };   // line: index in H / V
    std::vector<Trace> tr_h, tr_v;
    int n_corners = 0;   // rule ends pinned to the corner they meet (corner_junctions)
};

struct WarpDebug {
    int n_ctrl_raw = 0, n_ctrl = 0, n_text = 0;
    Lines H, V;
    Samples S1, S2;
    std::vector<std::array<double, 3>> T;
    std::vector<std::array<double, 3>> M;          // app: aligned text-column edges (along y, ideal x, disp)
    int n_margins = 0;
    std::vector<std::array<double, 3>> EH, EV;     // app: colour-area boundaries traced as lines (colour_edges)
    Lines E_lines;                                 // their extents, stage-1 frame: [pos, start, end, width, length], horizontal first
    int n_edges_h = 0, n_edge_bowed = 0;
    int n_downweighted = 0;                        // robust solve: samples left at < half weight (last pass)
    std::vector<std::array<double, 4>> G_lines;   // global-stage grid lines: [horizontal (1/0), straightened
                                                  // position, start, end] (stage-1 frame)
    int n_ignored = 0;                            // samples dropped by the ignore mask (and drop_lines)
    int n_lines_dropped = 0;                      // lines left out whole by drop_lines
    Field GX, GY, Fx1, Fy1, Fx, Fy;
    cv::Mat A_img, D;
    cv::Mat PH, PV;                               // app: rule net probabilities on A_img (CV_32F), empty without it
    NetLineStats net;
    std::vector<std::pair<std::string, double>> ms;   // stage timings
    int cg_iters[6] = {0, 0, 0, 0, 0, 0};
};

// warpmap.unwarp_image(src) -> backward map on the 4-px grid of src (mx4, my4).
// src: 8-bit BGR. dbg (optional) receives the intermediates for the parity checks.
// ignore (optional, app only): CV_8U of src's size; line samples and grid controls whose
// straightened position falls on a non-zero pixel are left out of the fit (the user marked
// those lines as wrong). Null: the reference behaviour. share_rows (app): pieces of one ruled
// line share their ideal position (see share_ideals in warpmap.cpp); false: the reference.
// faint (app): printed rules too pale for the reference's darkness threshold are found and traced
// too (thin, isolated runs at darkness > 25; per-line tracing threshold); false: the reference.
// robust (app): IRLS - samples the solved field does not follow are down-weighted and the fields
// solved again, so one wrong line cannot bend the area around it.
// edges (app, with share_rows): long straight boundaries between colour areas are lines too
// (colour_edges in warpmap.cpp).
// drop_lines (app): CV_8U of src's size; every line (samples sharing one ideal) with a sample on a
// non-zero pixel is left out whole (the line stage's local retry, neural_line_stage).
// rule_net (app, with faint): which found lines are rules, which rules were missed, how far a
// rule runs and whether a bent trace is a rule (see net_lines in warpmap.cpp); null: no net.
GridMap unwarp_map(const cv::Mat& src_bgr, WarpDebug* dbg = nullptr, const cv::Mat* ignore = nullptr, bool share_rows = false,
                   bool faint = false, bool robust = false, bool edges = false, const RuleNetFn* rule_net = nullptr,
                   const cv::Mat* drop_lines = nullptr);

// Report-only option (docs/results/accel/cg_rtol.txt): the relative-residual tolerance of the
// warp map's conjugate-gradient field solves. Default 1e-8, the reference; a looser value is
// faster and changes the map. Process-wide; FS_CG_RTOL=x in the environment sets it at start
// (desktop harnesses); values <= 0 restore the default. The app never sets it.
void set_warp_cg_rtol(double rtol);
double warp_cg_rtol();

// Building blocks exposed for the stage tests.
cv::Mat darkness_fast(const cv::Mat& bgr);
void get_lines(const cv::Mat& D, Lines& H, Lines& V, bool faint = false, const cv::Mat* PH = nullptr, const cv::Mat* PV = nullptr,
               NetLineStats* stats = nullptr);
// the rule net's probabilities on a page (CV_32F, page size); false if the net did not run
bool rule_probs(const cv::Mat& bgr, const RuleNetFn& net, cv::Mat& PH, cv::Mat& PV);
// clean.segments / clean.short_segments (morphological axis-aligned line segments)
Lines clean_segments(const cv::Mat& ink, bool hz, int klen = 100, int minlen = 150);
Lines clean_short_segments(const cv::Mat& ink, bool hz, const Lines& long_segs, const Lines& perp);

}  // namespace fs
