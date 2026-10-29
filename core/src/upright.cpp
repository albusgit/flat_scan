#include "upright.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace fs {

namespace {

constexpr double kCamera = 1.0;                     // PageTransform.CAMERA
constexpr double kMaxTilt = 30.0 * CV_PI / 180.0;   // PageTransform.MAX_TILT
constexpr double kAxisTol = 30.0;                   // a line further than this from both axes says nothing (degrees)

double angle_deg(double dx, double dy) { return std::atan2(dy, dx) * 180.0 / CV_PI; }

// how far a direction is from the nearest axis (degrees, signed), and whether that axis is the horizontal
double off_axis(double dx, double dy, bool* horizontal) {
    double a = angle_deg(dx, dy);
    while (a > 90) a -= 180;
    while (a <= -90) a += 180;
    if (std::fabs(a) <= 45) { if (horizontal) *horizontal = true; return a; }
    if (horizontal) *horizontal = false;
    return a > 0 ? a - 90 : a + 90;
}

}  // namespace

cv::Point2d upright_apply(double rotate, double vertical, double horizontal, double hw, cv::Point2d p) {
    // page fractions -> metric (page widths) around the centre
    double x = p.x - 0.5, y = (p.y - 0.5) * hw;
    const double f = kCamera, sv = std::sin(std::clamp(vertical, -1.0, 1.0) * kMaxTilt), sh = std::sin(std::clamp(horizontal, -1.0, 1.0) * kMaxTilt);
    // the tilt about the horizontal axis, then about the vertical one (PageTransform.matrix's order)
    double w = sv * y + f;
    x = f * x / w; y = f * y / w;
    w = sh * x + f;
    x = f * x / w; y = f * y / w;
    const double a = std::clamp(rotate, -10.0, 10.0) * CV_PI / 180.0;
    const double xr = std::cos(a) * x - std::sin(a) * y, yr = std::sin(a) * x + std::cos(a) * y;
    return {xr + 0.5, yr / hw + 0.5};
}

std::vector<UprightLine> upright_lines(const cv::Mat& page) {
    std::vector<UprightLine> out;
    if (page.empty()) return out;
    cv::Mat g;
    if (page.channels() == 4) cv::cvtColor(page, g, cv::COLOR_BGRA2GRAY);
    else if (page.channels() == 3) cv::cvtColor(page, g, cv::COLOR_BGR2GRAY);
    else g = page;
    const double s = std::min(1.0, 1200.0 / std::max(g.cols, g.rows));
    if (s < 1) cv::resize(g, g, cv::Size(), s, s, cv::INTER_AREA);
    const int W = g.cols, H = g.rows;
    // ink: darker than the paper around it
    cv::Mat bg, ink;
    cv::dilate(g, bg, cv::Mat::ones(5, 5, CV_8U));
    cv::medianBlur(bg, bg, 21);
    cv::subtract(bg, g, ink);
    cv::threshold(ink, ink, 40, 255, cv::THRESH_BINARY);
    // text rows: the ink of a row merged along it; its run is the line through the middle of each 8-px column's ink
    cv::Mat rows;
    cv::morphologyEx(ink, rows, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(9, W / 60), 1)));
    cv::morphologyEx(rows, rows, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat lab, st, cen;
    const int n = cv::connectedComponentsWithStats(rows, lab, st, cen, 8, CV_32S);
    for (int i = 1; i < n; ++i) {
        const int x = st.at<int>(i, cv::CC_STAT_LEFT), y = st.at<int>(i, cv::CC_STAT_TOP);
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH), h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        if (w < 0.12 * W || h < 4 || h > H / 15 || w < 6 * h) continue;
        std::vector<cv::Point2f> pts;
        for (int cx = x; cx + 8 <= x + w; cx += 8) {
            double sy = 0; int k = 0;
            for (int yy = y; yy < y + h; ++yy)
                for (int xx = cx; xx < cx + 8; ++xx)
                    if (lab.at<int>(yy, xx) == i && ink.at<uint8_t>(yy, xx)) { sy += yy; ++k; }
            if (k > 0) pts.emplace_back(cx + 4.f, (float)(sy / k));
        }
        if (pts.size() < 6) continue;
        cv::Vec4f L;
        cv::fitLine(pts, L, cv::DIST_HUBER, 0, 0.01, 0.01);
        if (std::fabs(L[0]) < 1e-6) continue;
        const double k = L[1] / L[0], x0 = pts.front().x, x1 = pts.back().x;
        const double y0 = L[3] + k * (x0 - L[2]), y1 = L[3] + k * (x1 - L[2]);
        if (std::fabs(angle_deg(x1 - x0, y1 - y0)) > kAxisTol) continue;
        out.push_back({x0 / W, y0 / H, x1 / W, y1 / H, (x1 - x0) / W});
    }
    // long straight edges: rules, table borders, the sides of pictures
    cv::Mat e;
    cv::Canny(g, e, 50, 150);
    std::vector<cv::Vec4i> segs;
    cv::HoughLinesP(e, segs, 1, CV_PI / 360, 60, 0.15 * std::min(W, H), 6);
    for (const auto& q : segs) {
        const double dx = q[2] - q[0], dy = q[3] - q[1];
        if (std::fabs(off_axis(dx, dy, nullptr)) > kAxisTol) continue;
        out.push_back({q[0] / (double)W, q[1] / (double)H, q[2] / (double)W, q[3] / (double)H, std::hypot(dx, dy) / std::max(W, H)});
    }
    return out;
}

bool upright_solve(const std::vector<UprightLine>& lines, UprightMode mode, double hw, double out[3]) {
    if (hw <= 0) return false;
    struct L { cv::Point2d a, b; double w; bool hz; };
    std::vector<L> ls;
    double wh = 0, wv = 0;
    for (const auto& l : lines) {
        // directions measured on the page as it is shown (y scaled to page widths)
        const double dx = l.x1 - l.x0, dy = (l.y1 - l.y0) * hw;
        if (std::hypot(dx, dy) < 1e-4) continue;
        bool hz = true;
        const double off = off_axis(dx, dy, &hz);
        if (mode != UprightMode::Guided && std::fabs(off) > kAxisTol) continue;
        ls.push_back({{l.x0, l.y0}, {l.x1, l.y1}, mode == UprightMode::Guided ? 1.0 : l.w, hz});
        (hz ? wh : wv) += ls.back().w;
    }
    if (ls.empty()) return false;
    // which sliders move: Level turns (by the horizontals when there are any, else the verticals)
    const bool use_v = mode != UprightMode::Level, use_h = mode == UprightMode::Full || mode == UprightMode::Auto || mode == UprightMode::Guided;
    const bool level_by_h = wh > 0;
    // pulls toward no change (per degree of turn or tilt, against the lines' squared degrees)
    const double lambda = mode == UprightMode::Auto ? 3e-3 : mode == UprightMode::Guided ? 1e-6 : 2e-4;
    double p[3] = {0, 0, 0};
    std::vector<double> hub(ls.size(), 1.0);   // Huber weights at 1 degree, fixed within a round (IRLS)
    auto offsets = [&](const double* q, std::vector<double>& off) {
        off.clear();
        for (const auto& l : ls) {
            const cv::Point2d a = upright_apply(q[0], q[1], q[2], hw, l.a), b = upright_apply(q[0], q[1], q[2], hw, l.b);
            bool hz = l.hz;
            double o = off_axis(b.x - a.x, (b.y - a.y) * hw, &hz);
            off.push_back(hz != l.hz ? 45.0 : o);   // turned past the diagonal: as far off as it gets
        }
    };
    auto residuals = [&](const double* q, std::vector<double>& r) {
        std::vector<double> off;
        offsets(q, off);
        r.clear();
        for (size_t i = 0; i < ls.size(); ++i) {
            if (mode == UprightMode::Level && ls[i].hz != level_by_h) continue;
            r.push_back(std::sqrt(ls[i].w * hub[i]) * off[i]);
        }
        const double total = std::max(1e-9, wh + wv), k = std::sqrt(lambda * total);
        r.push_back(k * q[0]);
        r.push_back(k * q[1] * 30);
        r.push_back(k * q[2] * 30);
    };
    const bool free_[3] = {true, use_v, use_h};
    std::vector<double> r0, r1;
    for (int round = 0; round < 6; ++round) {   // IRLS: the Huber weights from the last round's sliders
        if (round > 0) {
            std::vector<double> off;
            offsets(p, off);
            for (size_t i = 0; i < ls.size(); ++i) hub[i] = std::fabs(off[i]) <= 1 ? 1.0 : 1.0 / std::fabs(off[i]);
        }
        double mu = 1e-3;
        residuals(p, r0);
        double cost = 0;
        for (double v : r0) cost += v * v;
        for (int it = 0; it < 40; ++it) {
            // the Jacobian by central differences, in the moving sliders only
            cv::Mat J((int)r0.size(), 3, CV_64F, cv::Scalar(0));
            const double step[3] = {0.01, 0.001, 0.001};
            for (int k = 0; k < 3; ++k) {
                if (!free_[k]) continue;
                double q[3] = {p[0], p[1], p[2]};
                q[k] += step[k]; residuals(q, r1);
                std::vector<double> rp = r1;
                q[k] -= 2 * step[k]; residuals(q, r1);
                for (int i = 0; i < (int)r0.size() && i < (int)rp.size() && i < (int)r1.size(); ++i) J.at<double>(i, k) = (rp[i] - r1[i]) / (2 * step[k]);
            }
            cv::Mat A = J.t() * J, g = J.t() * cv::Mat(r0);
            for (int k = 0; k < 3; ++k) A.at<double>(k, k) += free_[k] ? mu * (1 + A.at<double>(k, k)) : 1.0;
            cv::Mat d;
            if (!cv::solve(A, -g, d, cv::DECOMP_CHOLESKY)) break;
            double q[3] = {p[0] + d.at<double>(0), p[1] + d.at<double>(1), p[2] + d.at<double>(2)};
            q[0] = std::clamp(q[0], -10.0, 10.0); q[1] = std::clamp(q[1], -1.0, 1.0); q[2] = std::clamp(q[2], -1.0, 1.0);
            residuals(q, r1);
            double c1 = 0;
            for (double v : r1) c1 += v * v;
            if (c1 < cost) {
                const double moved = std::fabs(q[0] - p[0]) / 10 + std::fabs(q[1] - p[1]) + std::fabs(q[2] - p[2]);
                std::copy(q, q + 3, p); r0 = r1; cost = c1; mu = std::max(1e-6, mu * 0.3);
                if (moved < 1e-6) break;
            } else {
                mu *= 10;
                if (mu > 1e6) break;
            }
        }
    }
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
    return true;
}

}  // namespace fs
