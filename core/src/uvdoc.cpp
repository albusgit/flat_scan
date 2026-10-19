#include "uvdoc.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace fs {

UvCrop uv_crop(cv::Size photo, const std::array<cv::Point2d, 4>& c, int model) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const auto& p : c) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
    double m = 0.06 * std::max(x1 - x0, y1 - y0);
    int bx0 = std::max(0, (int)std::floor(x0 - m)), by0 = std::max(0, (int)std::floor(y0 - m));
    int bx1 = std::min(photo.width, (int)std::ceil(x1 + m) + 1), by1 = std::min(photo.height, (int)std::ceil(y1 + m) + 1);
    UvCrop u;
    u.box = cv::Rect(bx0, by0, std::max(1, bx1 - bx0), std::max(1, by1 - by0));
    // the same rule as run_photo's landscape flag: top edge longer than left edge
    u.turned = cv::norm(c[1] - c[0]) > cv::norm(c[3] - c[0]);
    u.model = model;
    return u;
}

std::vector<float> uv_input(const cv::Mat& photo, const UvCrop& c, const cv::Mat& blank) {
    const UvSpec& sp = uv_spec(c.model);
    cv::Rect box = c.box & cv::Rect(0, 0, photo.cols, photo.rows);
    cv::Mat crop = photo(box), turned, small, rgb;
    if (!blank.empty() && blank.size() == photo.size()) {
        cv::Mat hide;
        cv::dilate(blank(box) > 0, hide, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
        if (cv::countNonZero(hide) > 0) {
            // the background colour: the crop's border ring outside the hidden page
            cv::Mat ring = cv::Mat::zeros(crop.size(), CV_8U);
            cv::rectangle(ring, cv::Rect(0, 0, crop.cols, crop.rows), cv::Scalar(255), std::max(4, crop.cols / 40));
            ring.setTo(0, hide);
            cv::Scalar bg = cv::countNonZero(ring) > 0 ? cv::mean(crop, ring) : cv::Scalar(90, 90, 90);
            crop = crop.clone();
            crop.setTo(bg, hide);
        }
    }
    if (c.turned) cv::rotate(crop, turned, cv::ROTATE_90_CLOCKWISE);
    else turned = crop;
    cv::resize(turned, small, cv::Size(sp.in_w, sp.in_h), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> x((size_t)sp.input_len());
    const size_t plane = (size_t)sp.in_w * sp.in_h;
    for (int y = 0; y < sp.in_h; ++y) {
        const uint8_t* s = rgb.ptr<uint8_t>(y);
        for (int xx = 0; xx < sp.in_w; ++xx)
            for (int k = 0; k < 3; ++k) x[k * plane + (size_t)y * sp.in_w + xx] = s[3 * xx + k] / 255.f;
    }
    return x;
}

UvGrid uv_grid(const float* out, const UvCrop& c) {
    UvGrid g;
    if (!out) return g;
    const UvSpec& sp = uv_spec(c.model);
    g.turned = c.turned;
    g.gw = sp.gw;
    g.gh = sp.gh;
    // normalised coords refer to the (turned) crop at full resolution, align_corners = True
    int iw = c.turned ? c.box.height : c.box.width, ih = c.turned ? c.box.width : c.box.height;
    const size_t plane = (size_t)sp.gw * sp.gh;
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
    double gx = u * (gw - 1) / std::max(1.0, Wp - 1), gy = v * (gh - 1) / std::max(1.0, Hp - 1);
    gx = std::min(std::max(gx, 0.0), (double)(gw - 1));
    gy = std::min(std::max(gy, 0.0), (double)(gh - 1));
    int x0 = std::min((int)gx, gw - 2), y0 = std::min((int)gy, gh - 2);
    double a = gx - x0, b = gy - y0;
    const cv::Point2d& p00 = pts[(size_t)y0 * gw + x0];
    const cv::Point2d& p01 = pts[(size_t)y0 * gw + x0 + 1];
    const cv::Point2d& p10 = pts[(size_t)(y0 + 1) * gw + x0];
    const cv::Point2d& p11 = pts[(size_t)(y0 + 1) * gw + x0 + 1];
    return p00 * ((1 - a) * (1 - b)) + p01 * (a * (1 - b)) + p10 * ((1 - a) * b) + p11 * (a * b);
}

std::vector<cv::Point2f> UvGrid::outline() const {
    std::vector<cv::Point2f> o;
    auto P = [&](int r, int c) { const cv::Point2d& p = pts[(size_t)r * gw + c]; o.emplace_back((float)p.x, (float)p.y); };
    for (int c = 0; c < gw; ++c) P(0, c);
    for (int r = 1; r < gh; ++r) P(r, gw - 1);
    for (int c = gw - 2; c >= 0; --c) P(gh - 1, c);
    for (int r = gh - 2; r > 0; --r) P(r, 0);
    return o;
}

void uv_apply_shading(cv::Mat& page, int y0, const cv::Mat& mx, const cv::Mat& my, const UvShading& s) {
    if (!s.data) return;
    const UvSpec& sp = uv_spec(s.crop.model);
    const cv::Rect& b = s.crop.box;
    const int iw = s.crop.turned ? b.height : b.width, ih = s.crop.turned ? b.width : b.height;
    const double k = s.full_strength ? 1.0 : std::min(1.0, 0.4 / std::max(s.brightness, 1e-6));
    const size_t plane = (size_t)sp.in_w * sp.in_h;
    const double fx = (double)sp.in_w / iw, fy = (double)sp.in_h / ih;
    for (int r = 0; r < mx.rows; ++r) {
        uint8_t* p = page.ptr<uint8_t>(y0 + r);
        const float* X = mx.ptr<float>(r);
        const float* Y = my.ptr<float>(r);
        for (int x = 0; x < page.cols; ++x) {
            double cx = X[x] - b.x, cy = Y[x] - b.y;
            double u = cx, v = cy;            // position in the (turned) crop
            if (s.crop.turned) { u = (b.height - 1) - cy; v = cx; }
            // the map was resized to the net size with align_corners = False
            double sx = (u + 0.5) * fx - 0.5, sy = (v + 0.5) * fy - 0.5;
            sx = std::min(std::max(sx, 0.0), sp.in_w - 1.0);
            sy = std::min(std::max(sy, 0.0), sp.in_h - 1.0);
            int x0 = std::min((int)sx, sp.in_w - 2), yy0 = std::min((int)sy, sp.in_h - 2);
            double a = sx - x0, bb = sy - yy0;
            for (int ch = 0; ch < 3; ++ch) {   // shading channels are RGB, the page is BGR(A)
                const float* m = s.data + (size_t)ch * plane;
                size_t i = (size_t)yy0 * sp.in_w + x0;
                double sh = m[i] * (1 - a) * (1 - bb) + m[i + 1] * a * (1 - bb) + m[i + sp.in_w] * (1 - a) * bb + m[i + sp.in_w + 1] * a * bb;
                double blend = std::max(0.1, sh * k + (1 - k));
                int bi = 2 - ch;
                p[4 * x + bi] = (uint8_t)std::min(255.0, std::round(p[4 * x + bi] / blend));
            }
        }
    }
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
