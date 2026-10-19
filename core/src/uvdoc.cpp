#include "uvdoc.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace fs {

UvCrop uv_crop(cv::Size photo, const std::array<cv::Point2d, 4>& c) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const auto& p : c) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
    double m = 0.06 * std::max(x1 - x0, y1 - y0);
    int bx0 = std::max(0, (int)std::floor(x0 - m)), by0 = std::max(0, (int)std::floor(y0 - m));
    int bx1 = std::min(photo.width, (int)std::ceil(x1 + m) + 1), by1 = std::min(photo.height, (int)std::ceil(y1 + m) + 1);
    UvCrop u;
    u.box = cv::Rect(bx0, by0, std::max(1, bx1 - bx0), std::max(1, by1 - by0));
    // the same rule as run_photo's landscape flag: top edge longer than left edge
    u.turned = cv::norm(c[1] - c[0]) > cv::norm(c[3] - c[0]);
    return u;
}

std::vector<float> uv_input(const cv::Mat& photo, const UvCrop& c) {
    cv::Mat crop = photo(c.box & cv::Rect(0, 0, photo.cols, photo.rows)), turned, small, rgb;
    if (c.turned) cv::rotate(crop, turned, cv::ROTATE_90_CLOCKWISE);
    else turned = crop;
    cv::resize(turned, small, cv::Size(UV_IN_W, UV_IN_H), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> x((size_t)UV_INPUT_LEN);
    const size_t plane = (size_t)UV_IN_W * UV_IN_H;
    for (int y = 0; y < UV_IN_H; ++y) {
        const uint8_t* s = rgb.ptr<uint8_t>(y);
        for (int xx = 0; xx < UV_IN_W; ++xx)
            for (int k = 0; k < 3; ++k) x[k * plane + (size_t)y * UV_IN_W + xx] = s[3 * xx + k] / 255.f;
    }
    return x;
}

UvGrid uv_grid(const float* out, const UvCrop& c) {
    UvGrid g;
    if (!out) return g;
    g.turned = c.turned;
    // normalised coords refer to the (turned) crop at full resolution, align_corners = True
    int iw = c.turned ? c.box.height : c.box.width, ih = c.turned ? c.box.width : c.box.height;
    const size_t plane = (size_t)UV_GW * UV_GH;
    g.pts.resize(plane);
    bool finite = true;
    for (size_t i = 0; i < plane; ++i) {
        double u = (out[i] + 1.0) / 2.0 * (iw - 1), v = (out[plane + i] + 1.0) / 2.0 * (ih - 1);
        finite &= std::isfinite(u) && std::isfinite(v);
        // the turned crop: turned (u, v) = crop (v, ch - 1 - u)
        cv::Point2d p = c.turned ? cv::Point2d(v, (c.box.height - 1) - u) : cv::Point2d(u, v);
        g.pts[i] = p + cv::Point2d(c.box.x, c.box.y);
    }
    g.ok = finite;
    return g;
}

cv::Point2d UvGrid::at(double x, double y, int W, int H) const {
    // the page as the net saw it (portrait Wp x Hp): a landscape page turned clockwise
    double u = x, v = y, Wp = W, Hp = H;
    if (turned) { u = H - 1 - y; v = x; Wp = H; Hp = W; }
    double gx = u * (UV_GW - 1) / std::max(1.0, Wp - 1), gy = v * (UV_GH - 1) / std::max(1.0, Hp - 1);
    gx = std::min(std::max(gx, 0.0), (double)(UV_GW - 1));
    gy = std::min(std::max(gy, 0.0), (double)(UV_GH - 1));
    int x0 = std::min((int)gx, UV_GW - 2), y0 = std::min((int)gy, UV_GH - 2);
    double a = gx - x0, b = gy - y0;
    const cv::Point2d& p00 = pts[(size_t)y0 * UV_GW + x0];
    const cv::Point2d& p01 = pts[(size_t)y0 * UV_GW + x0 + 1];
    const cv::Point2d& p10 = pts[(size_t)(y0 + 1) * UV_GW + x0];
    const cv::Point2d& p11 = pts[(size_t)(y0 + 1) * UV_GW + x0 + 1];
    return p00 * ((1 - a) * (1 - b)) + p01 * (a * (1 - b)) + p10 * ((1 - a) * b) + p11 * (a * b);
}

std::vector<cv::Point2f> UvGrid::outline() const {
    std::vector<cv::Point2f> o;
    auto P = [&](int r, int c) { const cv::Point2d& p = pts[(size_t)r * UV_GW + c]; o.emplace_back((float)p.x, (float)p.y); };
    for (int c = 0; c < UV_GW; ++c) P(0, c);
    for (int r = 1; r < UV_GH; ++r) P(r, UV_GW - 1);
    for (int c = UV_GW - 2; c >= 0; --c) P(UV_GH - 1, c);
    for (int r = UV_GH - 2; r > 0; --r) P(r, 0);
    return o;
}

namespace {

double iou_with(const std::vector<cv::Point2f>& poly_photo, const cv::Mat& mask01, double s) {
    std::vector<std::vector<cv::Point>> p(1);
    for (const auto& q : poly_photo) p[0].emplace_back((int)std::lround(q.x * s), (int)std::lround(q.y * s));
    cv::Mat g = cv::Mat::zeros(mask01.size(), CV_8U);
    cv::fillPoly(g, p, cv::Scalar(1));
    double inter = cv::countNonZero(g & mask01), uni = cv::countNonZero(g | mask01);
    return uni > 0 ? inter / uni : 0.0;
}

}  // namespace

double uv_agreement(const UvGrid& g, const cv::Mat& mask, double scale) {
    if (!g.ok || mask.empty()) return 0.0;
    double k = std::min(1.0, 800.0 / std::max(mask.cols, mask.rows));
    cv::Mat m;
    cv::resize(mask, m, cv::Size(), k, k, cv::INTER_AREA);
    m = m > 127;
    m /= 255;
    return iou_with(g.outline(), m, scale * k);
}

double uv_agreement(const UvGrid& g, const std::array<cv::Point2d, 4>& quad, cv::Size photo) {
    if (!g.ok) return 0.0;
    double k = std::min(1.0, 800.0 / std::max(photo.width, photo.height));
    cv::Mat m = cv::Mat::zeros((int)std::lround(photo.height * k), (int)std::lround(photo.width * k), CV_8U);
    std::vector<std::vector<cv::Point>> p(1);
    for (const auto& q : quad) p[0].emplace_back((int)std::lround(q.x * k), (int)std::lround(q.y * k));
    cv::fillPoly(m, p, cv::Scalar(1));
    return iou_with(g.outline(), m, k);
}

}  // namespace fs
