// Page orientation (0/90/180/270): port of code/orient.py (_text_axis_score,
// _upright_hebrew_score) and the decision rule of photo_pipeline.run.
#pragma once
#include <opencv2/core.hpp>
#include <functional>
#include <vector>

namespace fs {

struct Orientation {
    int rot = -1;              // cv::ROTATE_* code, or -1 for none
    double sh = 0, sv = 0;     // text-axis scores (horizontal, vertical)
    double upright = 0;        // Hebrew top-bar score of the chosen candidate (>= 0 upright)
    int net_turn = -1;         // the orientation net's answer (0..3 x 90 deg clockwise; -1: not run / no text)
    double net_margin = -1;    // its mean log-probability lead over the runner-up
    int net_patches = 0;
    bool by_net = false;       // the net's answer was used
};

// The orientation net (tools/orientnet, orientnet.onnx): in = n x 1 x 128 x 128 patches, out = n x 4 logits
using OrientNetFn = std::function<bool(const std::vector<float>& in, int n, std::vector<float>& logits)>;
// The net's answer for a page (tools/orientnet/orientnet.py page_patches + decide): the page in grey at two scales,
// its text-like 128-px patches (the 64 richest in ink), the mean log-softmax over them. turn: k x 90 deg clockwise
// brings the page upright; margin: the lead of the best over the runner-up. false: no text patches or no net.
bool orient_net_turn(const cv::Mat& page, const OrientNetFn& net, int* turn, double* margin, int* patches = nullptr);

double text_axis_score(const cv::Mat& gray, double* sv);
double upright_hebrew_score(const cv::Mat& gray);
// Decision rule: sideways pages (vertical text axis) are turned 90 deg; there is no automatic
// 180-degree flip (round 4). min_conf: the upright cue (Hebrew-specific, near 0 on other pages)
// must be below -min_conf to choose CW instead of CCW for a sideways page. min_conf = 0 is
// exactly photo_pipeline.run (reference behaviour). `upright` is still reported for upright pages.
// min_ratio: sideways only when the vertical text score exceeds the horizontal one by this factor.
Orientation decide_orientation(const cv::Mat& bgr_page, double min_conf = 0.0, double min_ratio = 1.0);

}  // namespace fs
