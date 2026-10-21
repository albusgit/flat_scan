#include "overlay.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "nputil.h"

namespace fs {

namespace {
// |d| on the grid and the legend top (99.5th percentile, rounded up, >= 1 px)
float magnitudes(const GridMap& d, cv::Mat& mag) {
    mag.create(d.gh, d.gw, CV_32F);
    std::vector<float> all;
    all.reserve((size_t)d.gh * d.gw);
    for (int i = 0; i < d.gh; ++i)
        for (int j = 0; j < d.gw; ++j) {
            float v = hypot_xy(d.x[(size_t)i * d.gw + j], d.y[(size_t)i * d.gw + j]);
            mag.at<float>(i, j) = v;
            all.push_back(v);
        }
    if (all.empty()) return 1.f;
    std::nth_element(all.begin(), all.begin() + (size_t)(all.size() * 0.995), all.end());
    return std::max(1.f, std::ceil(all[(size_t)(all.size() * 0.995)]));
}
}  // namespace

float wrinkle_legend_px(const GridMap& d) {
    cv::Mat mag;
    return magnitudes(d, mag);
}

cv::Mat wrinkle_overlay(const GridMap& d, bool heat, bool grid, float* legend_px, int spacing) {
    int w = d.out_w, h = d.out_h;
    cv::Mat mag;
    float top = magnitudes(d, mag);
    if (legend_px) *legend_px = top;
    cv::Mat out(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    if (heat) {
        cv::Mat m8, col, full;
        mag.convertTo(m8, CV_8U, 255.0 / top);
        cv::applyColorMap(m8, col, cv::COLORMAP_TURBO);
        cv::resize(col, full, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        cv::Mat magf;
        cv::resize(mag, magf, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                cv::Vec3b c = full.at<cv::Vec3b>(y, x);
                // light: 0 px is almost transparent, the largest moves ~32% opaque
                float a = 0.06f + 0.26f * std::min(1.f, magf.at<float>(y, x) / top);
                out.at<cv::Vec4b>(y, x) = cv::Vec4b(c[0], c[1], c[2], (uint8_t)(a * 255));
            }
    }
    if (grid) {
        auto D = [&](double x, double y) {
            double gx = std::min(std::max(x / d.stride, 0.0), d.gw - 1.0), gy = std::min(std::max(y / d.stride, 0.0), d.gh - 1.0);
            int x0 = std::min((int)gx, std::max(d.gw - 2, 0)), y0 = std::min((int)gy, std::max(d.gh - 2, 0));
            int x1 = std::min(x0 + 1, d.gw - 1), y1 = std::min(y0 + 1, d.gh - 1);
            double a = gx - x0, b = gy - y0;
            auto L = [&](const std::vector<float>& f) {
                return (f[(size_t)y0 * d.gw + x0] * (1 - a) + f[(size_t)y0 * d.gw + x1] * a) * (1 - b) +
                       (f[(size_t)y1 * d.gw + x0] * (1 - a) + f[(size_t)y1 * d.gw + x1] * a) * b;
            };
            return cv::Point2d(x + L(d.x), y + L(d.y));
        };
        const cv::Scalar ink(216, 78, 31, 210);   // BGRA, the app's primary blue
        int th = std::max(1, w / 900);
        for (int y0 = spacing / 2; y0 < h; y0 += spacing) {
            std::vector<cv::Point> pts;
            for (int x = 0; x <= w; x += 6) { cv::Point2d p = D(x, y0); pts.emplace_back((int)std::lround(p.x * 16), (int)std::lround(p.y * 16)); }
            cv::polylines(out, pts, false, ink, th, cv::LINE_AA, 4);
        }
        for (int x0 = spacing / 2; x0 < w; x0 += spacing) {
            std::vector<cv::Point> pts;
            for (int y = 0; y <= h; y += 6) { cv::Point2d p = D(x0, y); pts.emplace_back((int)std::lround(p.x * 16), (int)std::lround(p.y * 16)); }
            cv::polylines(out, pts, false, ink, th, cv::LINE_AA, 4);
        }
    }
    return out;
}

}  // namespace fs
