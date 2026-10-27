// The line editor's Add line (app): the printed line a rough stroke was drawn along, found on the page.
#pragma once
#include <opencv2/core.hpp>

#include <vector>

namespace fs {

// The printed line (a rule, a table or box edge: dark and thin, paler on both sides) that [stroke] (page px,
// in drawing order) was drawn roughly along on [page] (BGR, the output page): its middle as a polyline (page
// px, every 4 px along it), followed past the stroke's ends while it goes on. The stroke only has to come within
// about 3 % of the page width of it; of several lines in reach, the strongest smooth one nearest the stroke.
// Empty when no line lies under most of the stroke.
std::vector<cv::Point2f> find_drawn_line(const cv::Mat& page, const std::vector<cv::Point2f>& stroke);

}  // namespace fs
