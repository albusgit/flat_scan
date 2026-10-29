// Upright for the finished page (the app's Page > Perspective, as Lightroom's Transform > Upright): the perspective
// that makes the page's lines level and plumb, as the Transform sliders of the app's PageTransform.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace fs {

// A line of the page that should come out horizontal or vertical: its ends as page fractions, and how much it counts
struct UprightLine { double x0 = 0, y0 = 0, x1 = 0, y1 = 0, w = 1; };

// The page's own straight cues on a finished page (BGR or BGRA): text rows (their run, from the ink along them) and long
// straight edges (ruled lines, table borders, the sides of pictures), each within 30 degrees of horizontal or vertical,
// weighted by its length
std::vector<UprightLine> upright_lines(const cv::Mat& page);

enum class UprightMode { Level = 0, Vertical = 1, Full = 2, Auto = 3, Guided = 4 };

// The Transform sliders that make [lines] (page fractions of a page of height / width hw) horizontal or vertical,
// whichever each is nearer: out[0] rotate (degrees, -10..10), out[1] vertical, out[2] horizontal (-1..1 = -30..30
// degrees of tilt about the page's horizontal / vertical axis). Level turns only (by the horizontals, else the
// verticals); Vertical also tilts about the horizontal axis (converging verticals made parallel); Full tilts about
// both; Auto is Full held nearer to no change; Guided is Full on the user's guides, each fully. Robust to a few wrong
// lines (Huber, 1 degree). False when there is nothing to go by (out unchanged).
bool upright_solve(const std::vector<UprightLine>& lines, UprightMode mode, double hw, double out[3]);

// The sliders' map of a page point (fractions) on a page of height / width hw: PageTransform.matrix with only rotate,
// vertical and horizontal set (the same model: tilts seen from one page width, then the turn)
cv::Point2d upright_apply(double rotate, double vertical, double horizontal, double hw, cv::Point2d p);

}  // namespace fs
