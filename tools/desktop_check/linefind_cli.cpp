// The line editor's Add line on a page image (core/src/linefind.h): each stroke's found line drawn over the page.
// usage: linefind_cli PAGE OUT.png "x0,y0,x1,y1,..." ["x0,y0,..." ...]   (strokes in page px)
// draws each stroke in red and the line found for it in yellow (nothing: no line), prints one JSON line per stroke
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "linefind.h"

using namespace fs;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: linefind_cli PAGE OUT.png \"x0,y0,x1,y1,...\" [...]\n");
        return 2;
    }
    cv::Mat page = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (page.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    cv::Mat vis = page.clone();
    for (int s = 3; s < argc; ++s) {
        std::vector<float> v;
        std::stringstream ss(argv[s]);
        for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stof(t));
        std::vector<cv::Point2f> stroke;
        for (size_t i = 0; i + 1 < v.size(); i += 2) stroke.emplace_back(v[i], v[i + 1]);
        auto t0 = Clock::now();
        std::vector<cv::Point2f> line = find_drawn_line(page, stroke);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        for (size_t i = 1; i < stroke.size(); ++i) cv::line(vis, stroke[i - 1], stroke[i], cv::Scalar(40, 40, 230), 2, cv::LINE_AA);
        for (size_t i = 1; i < line.size(); ++i) cv::line(vis, line[i - 1], line[i], cv::Scalar(0, 215, 255), 3, cv::LINE_AA);
        std::printf("{\"stroke\": %d, \"found\": %s, \"points\": %d, \"from\": [%.0f, %.0f], \"to\": [%.0f, %.0f], \"ms\": %.1f}\n", s - 2,
                    line.empty() ? "false" : "true", (int)line.size(), line.empty() ? 0.f : line.front().x, line.empty() ? 0.f : line.front().y,
                    line.empty() ? 0.f : line.back().x, line.empty() ? 0.f : line.back().y, ms);
    }
    cv::imwrite(argv[2], vis);
    return 0;
}
