// Page orientation (0/90/180/270): port of code/orient.py (_text_axis_score,
// _upright_hebrew_score) and the decision rule of photo_pipeline.run.
#pragma once
#include <opencv2/core.hpp>

namespace fs {

struct Orientation {
    int rot = -1;              // cv::ROTATE_* code, or -1 for none
    double sh = 0, sv = 0;     // text-axis scores (horizontal, vertical)
    double upright = 0;        // Hebrew top-bar score of the chosen candidate (>= 0 upright)
};

double text_axis_score(const cv::Mat& gray, double* sv);
double upright_hebrew_score(const cv::Mat& gray);
// Decision rule. min_conf: the upright cue (Hebrew-specific, near 0 on other pages) must be
// below -min_conf to choose the flipped candidate: 180 deg for an upright-axis page, CW instead of
// CCW for a sideways one. min_conf = 0 is exactly photo_pipeline.run (reference behaviour).
Orientation decide_orientation(const cv::Mat& bgr_page, double min_conf = 0.0);

}  // namespace fs
