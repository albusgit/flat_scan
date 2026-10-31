#include "uvdoc.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace fs {

UvCrop uv_crop(cv::Size photo, const std::array<cv::Point2d, 4>& c, int model, bool pad) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const auto& p : c) { x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y); }
    double m = 0.06 * std::max(x1 - x0, y1 - y0);
    int bx0 = (int)std::floor(x0 - m), by0 = (int)std::floor(y0 - m), bx1 = (int)std::ceil(x1 + m) + 1, by1 = (int)std::ceil(y1 + m) + 1;
    if (!pad) {
        bx0 = std::max(0, bx0);
        by0 = std::max(0, by0);
        bx1 = std::min(photo.width, bx1);
        by1 = std::min(photo.height, by1);
    }
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
    if (box != c.box) {   // a padded crop: the photo inside its box, kUvPadColour where the box reaches past it
        cv::Mat padded(c.box.size(), photo.type(), kUvPadColour);
        crop.copyTo(padded(box - c.box.tl()));
        crop = padded;
    }
    if (!blank.empty() && blank.size() == photo.size()) {
        cv::Mat hide = cv::Mat::zeros(c.box.size(), CV_8U), hide_in = hide(box - c.box.tl());
        cv::dilate(blank(box) > 0, hide_in, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
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

GridValidity uv_grid_validity(const UvGrid& g, cv::Size photo) {
    GridValidity v;
    if (!g.ok || g.gw < 2 || g.gh < 2) return v;
    auto P = [&](int i, int j) { return g.pts[(size_t)i * g.gw + j]; };
    std::vector<double> tri, cell;
    for (int i = 0; i + 1 < g.gh; ++i)
        for (int j = 0; j + 1 < g.gw; ++j) {
            const cv::Point2d a = P(i, j), b = P(i, j + 1), c = P(i + 1, j + 1), d = P(i + 1, j);
            const double t1 = (b - a).cross(c - a), t2 = (c - a).cross(d - a);
            tri.push_back(t1);
            tri.push_back(t2);
            cell.push_back(std::fabs(t1 + t2));
        }
    // the grid's own orientation: the sign of most triangles (a turned crop does not mirror)
    int pos = 0;
    for (double t : tri) pos += t > 0;
    const double sgn = 2 * pos >= (int)tri.size() ? 1 : -1;
    int flip = 0;
    for (double t : tri) flip += t * sgn <= 0;
    v.folded = (double)flip / tri.size();
    std::vector<double> s = cell;
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    const double med = s[s.size() / 2];
    int small = 0;
    for (double a : cell) small += a < 0.1 * med;
    v.collapsed = (double)small / cell.size();
    int out = 0;
    for (const auto& p : g.pts) out += p.x < -1 || p.y < -1 || p.x > photo.width || p.y > photo.height;
    v.outside = (double)out / g.pts.size();
    return v;
}

double uv_paper_share(const UvGrid& g, const cv::Mat& mask, double scale, const cv::Mat& photo) {
    if (!g.ok || mask.empty()) return 0.0;
    double k = std::min(1.0, 800.0 / std::max(mask.cols, mask.rows));
    cv::Mat m;
    cv::resize(mask, m, cv::Size(), k, k, cv::INTER_AREA);
    m = m > 127;
    std::vector<std::vector<cv::Point>> p(1);
    for (const auto& q : g.outline()) p[0].emplace_back((int)std::lround(q.x * scale * k), (int)std::lround(q.y * scale * k));
    cv::Mat in = cv::Mat::zeros(m.size(), CV_8U);
    cv::fillPoly(in, p, cv::Scalar(255));
    const double area = cv::countNonZero(in);
    if (area <= 0) return 0.0;
    const double paper = cv::countNonZero(in & m);
    cv::Mat extra = in & ~m;
    if (photo.empty() || paper < 0.2 * area || cv::countNonZero(extra) == 0) return paper / area;
    // the photo at the mask's size, in Lab; the paper's colour: the bright half of the agreed part (not its ink)
    cv::Mat small, lab;
    cv::resize(photo.channels() == 4 ? photo : photo, small, m.size(), 0, 0, cv::INTER_AREA);
    if (small.channels() == 4) cv::cvtColor(small, small, cv::COLOR_BGRA2BGR);
    cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Vec3b> ref;
    cv::Mat agreed = in & m;
    for (int y = 0; y < lab.rows; ++y)
        for (int x = 0; x < lab.cols; ++x)
            if (agreed.at<uint8_t>(y, x)) ref.push_back(lab.at<cv::Vec3b>(y, x));
    std::sort(ref.begin(), ref.end(), [](const cv::Vec3b& a, const cv::Vec3b& b) { return a[0] < b[0]; });
    ref.erase(ref.begin(), ref.begin() + ref.size() / 2);
    std::vector<int> la, aa, ba;
    for (const auto& v : ref) { la.push_back(v[0]); aa.push_back(v[1]); ba.push_back(v[2]); }
    auto med = [](std::vector<int>& v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return (double)v[v.size() / 2]; };
    const double L0 = med(la), A0 = med(aa), B0 = med(ba);
    // blocks of the extra part whose bright pixels have the paper's colour (lighter than 55 % of its
    // lightness: shading; within 12 of its hue in a, b) count as paper
    const int bs = 8;
    double like = 0;
    for (int y0 = 0; y0 < lab.rows; y0 += bs)
        for (int x0 = 0; x0 < lab.cols; x0 += bs) {
            cv::Rect bb(x0, y0, std::min(bs, lab.cols - x0), std::min(bs, lab.rows - y0));
            const int n = cv::countNonZero(extra(bb));
            if (n == 0) continue;
            std::vector<cv::Vec3b> px;
            for (int y = bb.y; y < bb.y + bb.height; ++y)
                for (int x = bb.x; x < bb.x + bb.width; ++x) px.push_back(lab.at<cv::Vec3b>(y, x));
            std::sort(px.begin(), px.end(), [](const cv::Vec3b& a, const cv::Vec3b& b) { return a[0] < b[0]; });
            const cv::Vec3b& q = px[px.size() * 3 / 4];
            if (q[0] >= 0.55 * L0 && std::hypot(q[1] - A0, q[2] - B0) <= 12) like += n;
        }
    return (paper + like) / area;
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

namespace {

// IoU of two polygons, rasterised at <= 800 px
double poly_iou(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    cv::Rect2f r = cv::boundingRect(a) | cv::boundingRect(b);
    double s = 800.0 / std::max(1.f, std::max(r.width, r.height));
    cv::Size sz((int)std::ceil(r.width * s) + 2, (int)std::ceil(r.height * s) + 2);
    auto raster = [&](const std::vector<cv::Point2f>& p) {
        std::vector<cv::Point> q;
        for (const auto& v : p) q.emplace_back((int)std::lround((v.x - r.x) * s) + 1, (int)std::lround((v.y - r.y) * s) + 1);
        cv::Mat m = cv::Mat::zeros(sz, CV_8U);
        cv::fillPoly(m, std::vector<std::vector<cv::Point>>{q}, cv::Scalar(255));
        return m;
    };
    cv::Mat A = raster(a), B = raster(b);
    double inter = cv::countNonZero(A & B), uni = cv::countNonZero(A | B);
    return uni > 0 ? inter / uni : 0;
}

// bilinear point of a grid at fractional index (gx, gy)
cv::Point2d grid_pt(const UvGrid& g, double gx, double gy) {
    gx = std::min(std::max(gx, 0.0), (double)(g.gw - 1));
    gy = std::min(std::max(gy, 0.0), (double)(g.gh - 1));
    int x0 = std::min((int)gx, g.gw - 2), y0 = std::min((int)gy, g.gh - 2);
    double a = gx - x0, b = gy - y0;
    const cv::Point2d& p00 = g.pts[(size_t)y0 * g.gw + x0];
    const cv::Point2d& p01 = g.pts[(size_t)y0 * g.gw + x0 + 1];
    const cv::Point2d& p10 = g.pts[(size_t)(y0 + 1) * g.gw + x0];
    const cv::Point2d& p11 = g.pts[(size_t)(y0 + 1) * g.gw + x0 + 1];
    return p00 * ((1 - a) * (1 - b)) + p01 * (a * (1 - b)) + p10 * ((1 - a) * b) + p11 * (a * b);
}

}  // namespace

double uv_agreement(const UvGrid& a, const UvGrid& b) {
    if (!a.ok || !b.ok) return 0.0;
    return poly_iou(a.outline(), b.outline());
}

std::array<cv::Point2d, 4> uv_corners(const UvGrid& g) {
    const size_t tl = 0, tr = (size_t)g.gw - 1, br = g.pts.size() - 1, bl = (size_t)(g.gh - 1) * g.gw;
    // turned: the net saw the photo turned clockwise, so its top-right is the page's top-left
    if (g.turned) return {g.pts[tr], g.pts[br], g.pts[bl], g.pts[tl]};
    return {g.pts[tl], g.pts[tr], g.pts[br], g.pts[bl]};
}

UvGrid uv_snap_border(const UvGrid& g, const cv::Mat& mask, double scale, BorderSnap* st, double search, double dead_px, double lost) {
    BorderSnap s;
    if (st) *st = s;
    if (!g.ok || mask.empty() || g.gw < 4 || g.gh < 4) return g;
    const int gw = g.gw, gh = g.gh;
    auto P = [&](int r, int c) { return g.pts[(size_t)r * gw + c]; };
    const double size = 0.5 * (cv::norm(P(0, 0) - P(gh - 1, gw - 1)) + cv::norm(P(0, gw - 1) - P(gh - 1, 0))) / std::sqrt(2.0);
    const double D = search * size;
    // the 4 sides: top (row 0), bottom (row gh-1) along the columns; left (col 0), right (col gw-1) along the
    // rows. in(i): the next point inwards
    struct SideRun { int n; std::function<cv::Point2d(int)> at, in; };
    const std::array<SideRun, 4> sides = {
        SideRun{gw, [&](int i) { return P(0, i); }, [&](int i) { return P(1, i); }},
        SideRun{gw, [&](int i) { return P(gh - 1, i); }, [&](int i) { return P(gh - 2, i); }},
        SideRun{gh, [&](int i) { return P(i, 0); }, [&](int i) { return P(i, 1); }},
        SideRun{gh, [&](int i) { return P(i, gw - 1); }, [&](int i) { return P(i, gw - 2); }}};
    std::array<std::vector<cv::Point2d>, 4> disp;
    std::vector<double> mags;
    for (int k = 0; k < 4; ++k) {
        const SideRun& sr = sides[k];
        std::vector<cv::Point2d> d(sr.n);
        std::vector<char> has(sr.n, 0);
        int found = 0, far = 0;
        for (int i = 0; i < sr.n; ++i) {
            const cv::Point2d p = sr.at(i);
            const cv::Point2d tg = sr.at(std::min(sr.n - 1, i + 1)) - sr.at(std::max(0, i - 1));
            cv::Point2d nrm(-tg.y, tg.x);
            const double nl = cv::norm(nrm);
            if (nl < 1e-9) continue;
            nrm *= 1.0 / nl;
            if (nrm.dot(sr.in(i) - p) > 0) nrm = -nrm;   // outwards
            // along the normal in mask px: paper -> background crossings, the nearest to the border
            const int steps = std::max(2, (int)std::ceil(D * scale));
            double best = 1e300;
            int prev = -1;
            for (int t = -steps; t <= steps; ++t) {
                const cv::Point2d q = (p + nrm * (t / scale)) * scale;
                const int x = (int)std::lround(q.x), y = (int)std::lround(q.y);
                if (x < 2 || y < 2 || x >= mask.cols - 2 || y >= mask.rows - 2) { prev = -1; continue; }
                const int v = mask.at<uint8_t>(y, x) > 127;
                if (prev == 1 && v == 0 && std::fabs(t - 0.5) < std::fabs(best)) best = t - 0.5;
                prev = v;
            }
            if (best > 1e299) continue;
            d[i] = nrm * (best / scale);
            has[i] = 1;
            ++found;
            if (std::fabs(best / scale) > lost * size) ++far;
        }
        if (found < 0.5 * sr.n) { disp[k].assign(sr.n, cv::Point2d(0, 0)); continue; }
        if (far > 0.5 * found) ++s.sides_lost;
        // the gaps filled from their neighbours along the side (at the ends: the nearest found)
        for (int i = 0; i < sr.n; ++i) {
            if (has[i]) continue;
            int a = i - 1, b = i + 1;
            while (a >= 0 && !has[a]) --a;
            while (b < sr.n && !has[b]) ++b;
            if (a < 0) d[i] = d[b];
            else if (b >= sr.n) d[i] = d[a];
            else d[i] = d[a] + (d[b] - d[a]) * ((double)(i - a) / (b - a));
        }
        // median then Gaussian along the side, per coordinate; then the dead band
        const int w = std::max(2, sr.n / 40);
        std::vector<cv::Point2d> m(sr.n);
        for (int i = 0; i < sr.n; ++i) {
            std::vector<double> xs, ys;
            for (int j = std::max(0, i - w); j <= std::min(sr.n - 1, i + w); ++j) { xs.push_back(d[j].x); ys.push_back(d[j].y); }
            std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
            std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
            m[i] = cv::Point2d(xs[xs.size() / 2], ys[ys.size() / 2]);
        }
        const double sg = std::max(1.0, sr.n / 40.0);
        std::vector<cv::Point2d> sm(sr.n);
        for (int i = 0; i < sr.n; ++i) {
            cv::Point2d acc(0, 0);
            double ws = 0;
            for (int j = std::max(0, i - (int)(3 * sg)); j <= std::min(sr.n - 1, i + (int)(3 * sg)); ++j) {
                const double wj = std::exp(-0.5 * (j - i) * (j - i) / (sg * sg));
                acc += m[j] * wj;
                ws += wj;
            }
            sm[i] = acc * (1.0 / ws);
            const double l = cv::norm(sm[i]);
            sm[i] = l > dead_px ? sm[i] * ((l - dead_px) / l) : cv::Point2d(0, 0);
            mags.push_back(cv::norm(sm[i]));
        }
        disp[k] = sm;
        ++s.sides_moved;
    }
    if (!mags.empty()) {
        std::vector<double> mm = mags;
        std::nth_element(mm.begin(), mm.begin() + mm.size() / 2, mm.end());
        s.median_px = mm[mm.size() / 2];
        s.max_px = *std::max_element(mags.begin(), mags.end());
    }
    if (st) *st = s;
    if (s.sides_moved == 0) return g;
    // corners: the mean of the two sides meeting there; each side then bent linearly to meet them
    const cv::Point2d c00 = (disp[0][0] + disp[2][0]) * 0.5, c10 = (disp[0][gw - 1] + disp[3][0]) * 0.5;
    const cv::Point2d c01 = (disp[1][0] + disp[2][gh - 1]) * 0.5, c11 = (disp[1][gw - 1] + disp[3][gh - 1]) * 0.5;
    auto fit_ends = [](std::vector<cv::Point2d>& v, cv::Point2d a, cv::Point2d b) {
        const int n = (int)v.size();
        const cv::Point2d ea = a - v[0], eb = b - v[n - 1];
        for (int i = 0; i < n; ++i) v[i] += ea * (1.0 - (double)i / (n - 1)) + eb * ((double)i / (n - 1));
    };
    fit_ends(disp[0], c00, c10);
    fit_ends(disp[1], c01, c11);
    fit_ends(disp[2], c00, c01);
    fit_ends(disp[3], c10, c11);
    UvGrid o = g;
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) {
            const double u = (double)c / (gw - 1), v = (double)r / (gh - 1);
            const cv::Point2d dd = disp[0][c] * (1 - v) + disp[1][c] * v + disp[2][r] * (1 - u) + disp[3][r] * u -
                                   (c00 * ((1 - u) * (1 - v)) + c10 * (u * (1 - v)) + c01 * ((1 - u) * v) + c11 * (u * v));
            o.pts[(size_t)r * gw + c] += dd;
        }
    return o;
}

UvGrid uv_extend_to_frame(const UvGrid& g, const cv::Mat& mask, double scale, double search, FrameExtend* st, const bool* allow) {
    FrameExtend fe;
    if (st) *st = fe;
    if (!g.ok || mask.empty() || g.gw < 4 || g.gh < 4) return g;
    const int gw = g.gw, gh = g.gh;
    auto P = [&](int r, int c) { return g.pts[(size_t)r * gw + c]; };
    const double size = 0.5 * (cv::norm(P(0, 0) - P(gh - 1, gw - 1)) + cv::norm(P(0, gw - 1) - P(gh - 1, 0))) / std::sqrt(2.0);
    const int steps = std::max(2, (int)std::ceil(search * size * scale));
    const double fw = mask.cols / scale - 1.0, fh = mask.rows / scale - 1.0;
    // per side: the border point, the next one inwards; the extension in grid steps (one step = the last cell)
    double ext[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        if (allow && !allow[k]) continue;
        const int n = k < 2 ? gw : gh;
        std::vector<double> reach;   // grid steps to the frame, where the way there is paper
        for (int i = 0; i < n; ++i) {
            const cv::Point2d p = k == 0 ? P(0, i) : k == 1 ? P(gh - 1, i) : k == 2 ? P(i, 0) : P(i, gw - 1);
            const cv::Point2d q = k == 0 ? P(1, i) : k == 1 ? P(gh - 2, i) : k == 2 ? P(i, 1) : P(i, gw - 2);
            cv::Point2d dir = p - q;   // outwards, one grid step long
            const double step = cv::norm(dir);
            if (step < 1e-6) continue;
            const cv::Point2d u = dir * (1.0 / step);
            int paper = 0, cnt = 0;
            bool hit = false;
            for (int t = 0; t <= steps; ++t) {
                const cv::Point2d m = (p + u * (t / scale)) * scale;
                const int x = (int)std::lround(m.x), y = (int)std::lround(m.y);
                if (x < 1 || y < 1 || x >= mask.cols - 1 || y >= mask.rows - 1) { hit = cnt > 0 && paper >= 0.9 * cnt; break; }
                paper += mask.at<uint8_t>(y, x) > 127;
                ++cnt;
            }
            if (!hit) continue;
            // the distance to the photo's edge along u, in photo px, then in grid steps
            double tmax = 1e300;
            if (u.x < -1e-9) tmax = std::min(tmax, (0.0 - p.x) / u.x);
            if (u.x > 1e-9) tmax = std::min(tmax, (fw - p.x) / u.x);
            if (u.y < -1e-9) tmax = std::min(tmax, (0.0 - p.y) / u.y);
            if (u.y > 1e-9) tmax = std::min(tmax, (fh - p.y) / u.y);
            if (tmax < 1e299 && tmax > 0) reach.push_back(tmax / step);
        }
        // a side with a few points on the frame (a corner off the photo) is not one running off it
        if ((int)reach.size() < std::max(3, n / 5)) continue;
        const size_t q = (size_t)(0.2 * (reach.size() - 1));
        std::nth_element(reach.begin(), reach.begin() + q, reach.end());
        ext[k] = reach[q];
        // below 2 px: nothing to add
        const double px_step = cv::norm(k == 0 ? P(0, n / 2) - P(1, n / 2) : k == 1 ? P(gh - 1, n / 2) - P(gh - 2, n / 2)
                                         : k == 2 ? P(n / 2, 0) - P(n / 2, 1) : P(n / 2, gw - 1) - P(n / 2, gw - 2));
        fe.px[k] = ext[k] * px_step;
        if (fe.px[k] < 2.0) { ext[k] = 0; fe.px[k] = 0; continue; }
        ++fe.sides;
    }
    if (st) *st = fe;
    if (fe.sides == 0) return g;
    // the old grid at (r, c) in its own steps, continued straight on past its border (the last cell's direction)
    auto sample = [&](double r, double c) {
        const double rc = std::min(std::max(r, 0.0), gh - 1.0), cc = std::min(std::max(c, 0.0), gw - 1.0);
        const int r0 = std::min((int)rc, gh - 2), c0 = std::min((int)cc, gw - 2);
        const double fr = rc - r0, fc = cc - c0;
        const cv::Point2d a = P(r0, c0), b = P(r0, c0 + 1), c2 = P(r0 + 1, c0), d = P(r0 + 1, c0 + 1);
        cv::Point2d q = (a * (1 - fc) + b * fc) * (1 - fr) + (c2 * (1 - fc) + d * fc) * fr;
        // past a border: along that border's last cell
        const cv::Point2d dr = (c2 * (1 - fc) + d * fc) - (a * (1 - fc) + b * fc);   // one step down
        const cv::Point2d dc = (b * (1 - fr) + d * fr) - (a * (1 - fr) + c2 * fr);   // one step right
        q += dr * (r - rc) + dc * (c - cc);
        return q;
    };
    UvGrid o = g;
    const double r0 = -ext[0], r1 = gh - 1 + ext[1], c0 = -ext[2], c1 = gw - 1 + ext[3];
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c)
            o.pts[(size_t)r * gw + c] = sample(r0 + (r1 - r0) * r / (gh - 1.0), c0 + (c1 - c0) * c / (gw - 1.0));
    return o;
}

UvCrop uv_crop_on_grid(cv::Size photo, const UvGrid& g, int model) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const auto& p : g.outline()) {
        double x = std::min(std::max((double)p.x, 0.0), photo.width - 1.0), y = std::min(std::max((double)p.y, 0.0), photo.height - 1.0);
        x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
    }
    std::array<cv::Point2d, 4> c = {cv::Point2d(x0, y0), cv::Point2d(x1, y0), cv::Point2d(x1, y1), cv::Point2d(x0, y1)};
    UvCrop u = uv_crop(photo, c, model);
    std::array<cv::Point2d, 4> k = uv_corners(g);
    u.turned = (cv::norm(k[1] - k[0]) + cv::norm(k[2] - k[3])) > (cv::norm(k[3] - k[0]) + cv::norm(k[2] - k[1]));
    return u;
}

namespace {

double smooth01(double t) {
    t = std::min(1.0, std::max(0.0, t));
    return t * t * (3 - 2 * t);
}

}  // namespace

UvGrid uv_zoom(const cv::Mat& photo, const UvGrid& g, int model, const UvNetFn& net, int* n_used, int nu, int nv) {
    if (n_used) *n_used = 0;
    if (!g.ok || g.gw < 8 || g.gh < 8 || !net) return g;
    const UvSpec& sp = uv_spec(model);
    // nu x nv parts in the page frame [0, 1]^2 (u across, v down, as the net saw the page), cut
    // evenly with an overlap of 0.1 on each side of a cut
    const double ov = 0.1;
    struct Q { double u0, u1, v0, v1; UvGrid g; bool ok = false; };
    std::vector<Q> qs;
    for (int qi = 0; qi < nv; ++qi)
        for (int qj = 0; qj < nu; ++qj)
            qs.push_back({qj ? (double)qj / nu - ov : 0.0, qj + 1 < nu ? (double)(qj + 1) / nu + ov : 1.0,
                          qi ? (double)qi / nv - ov : 0.0, qi + 1 < nv ? (double)(qi + 1) / nv + ov : 1.0, UvGrid(), false});
    int used = 0;
    for (auto& q : qs) {
        // the quarter's outline in the photo, along the first grid
        std::vector<cv::Point2f> poly;
        const int n = 24;
        auto P = [&](double u, double v) {
            cv::Point2d p = grid_pt(g, u * (g.gw - 1), v * (g.gh - 1));
            poly.emplace_back((float)p.x, (float)p.y);
        };
        for (int k = 0; k < n; ++k) P(q.u0 + (q.u1 - q.u0) * k / n, q.v0);
        for (int k = 0; k < n; ++k) P(q.u1, q.v0 + (q.v1 - q.v0) * k / n);
        for (int k = 0; k < n; ++k) P(q.u1 - (q.u1 - q.u0) * k / n, q.v1);
        for (int k = 0; k < n; ++k) P(q.u0, q.v1 - (q.v1 - q.v0) * k / n);
        cv::Rect2f bb = cv::boundingRect(poly);
        double m = 0.06 * std::max(bb.width, bb.height);
        int bx0 = std::max(0, (int)std::floor(bb.x - m)), by0 = std::max(0, (int)std::floor(bb.y - m));
        int bx1 = std::min(photo.cols, (int)std::ceil(bb.x + bb.width + m) + 1), by1 = std::min(photo.rows, (int)std::ceil(bb.y + bb.height + m) + 1);
        if (bx1 - bx0 < 16 || by1 - by0 < 16) continue;
        UvCrop crop;
        crop.box = cv::Rect(bx0, by0, bx1 - bx0, by1 - by0);
        crop.turned = g.turned;   // the same frame as the first pass
        crop.model = model;
        cv::Mat hide(photo.size(), CV_8U, cv::Scalar(255));
        std::vector<cv::Point> ip;
        for (const auto& p : poly) ip.emplace_back((int)std::lround(p.x), (int)std::lround(p.y));
        cv::fillPoly(hide, std::vector<std::vector<cv::Point>>{ip}, cv::Scalar(0));
        std::vector<float> io = uv_input(photo, crop, hide);
        if (!net(model, io) || (int)io.size() != sp.output_len()) continue;
        q.g = uv_grid(io.data(), crop);
        if (!q.g.ok || poly_iou(q.g.outline(), poly) < 0.9) continue;
        q.ok = true;
        ++used;
    }
    if (n_used) *n_used = used;
    if (used == 0) return g;
    UvGrid f;
    f.ok = true;
    f.turned = g.turned;
    f.gw = 2 * g.gw - 1;
    f.gh = 2 * g.gh - 1;
    f.pts.resize((size_t)f.gw * f.gh);
    // weight of a quarter at (u, v): 1 inside, fading to 0 across the overlap towards a cut edge
    auto ramp = [&](double x, double a0, double a1, bool cut0, bool cut1) {
        double w = 1;
        if (cut0) w *= smooth01((x - a0) / (2 * ov));
        if (cut1) w *= smooth01((a1 - x) / (2 * ov));
        return w;
    };
    for (int i = 0; i < f.gh; ++i)
        for (int j = 0; j < f.gw; ++j) {
            double u = (double)j / (f.gw - 1), v = (double)i / (f.gh - 1);
            cv::Point2d sum(0, 0);
            double ws = 0;
            for (const auto& q : qs) {
                if (u < q.u0 - 1e-9 || u > q.u1 + 1e-9 || v < q.v0 - 1e-9 || v > q.v1 + 1e-9) continue;
                double w = ramp(u, q.u0, q.u1, q.u0 > 0, q.u1 < 1) * ramp(v, q.v0, q.v1, q.v0 > 0, q.v1 < 1);
                if (w <= 0) continue;
                cv::Point2d p = q.ok ? grid_pt(q.g, (u - q.u0) / (q.u1 - q.u0) * (q.g.gw - 1), (v - q.v0) / (q.v1 - q.v0) * (q.g.gh - 1))
                                     : grid_pt(g, u * (g.gw - 1), v * (g.gh - 1));
                sum += p * w;
                ws += w;
            }
            f.pts[(size_t)i * f.gw + j] = ws > 0 ? sum / ws : grid_pt(g, u * (g.gw - 1), v * (g.gh - 1));
        }
    return f;
}

namespace {

const int kFuseSamples = 64;

// one side of a page grid at kFuseSamples points: (u, v) in [0, 1]^2 of the page as the net saw it
struct SideSample { double u, v; };

std::array<std::vector<SideSample>, 4> fuse_sides() {
    std::array<std::vector<SideSample>, 4> s;
    for (int i = 0; i < kFuseSamples; ++i) {
        const double t = (double)i / (kFuseSamples - 1);
        s[0].push_back({t, 0.0});        // top
        s[1].push_back({1.0, t});        // right
        s[2].push_back({1.0 - t, 1.0});  // bottom
        s[3].push_back({0.0, 1.0 - t});  // left
    }
    return s;
}

cv::Point2d grid_uv(const UvGrid& g, double u, double v) { return grid_pt(g, u * (g.gw - 1), v * (g.gh - 1)); }

// the signed distance (photo px) from the grid's border at (u, v) to the mask's edge along the outward normal:
// + the paper goes on beyond the border; no edge within reach: +-reach by the mask at the border point (on paper:
// the paper goes on, on the background: the border is out on it); NaN: the search leaves the mask image first
double border_miss(const UvGrid& g, double u, double v, const cv::Mat& mask, double scale, double reach) {
    const double e = 1.0 / 200;
    const cv::Point2d p = grid_uv(g, u, v);
    // the tangent along the side and the inward direction (towards the page's centre in (u, v))
    const bool horiz = v <= 0.0 || v >= 1.0;
    const cv::Point2d tg = horiz ? grid_uv(g, std::min(1.0, u + e), v) - grid_uv(g, std::max(0.0, u - e), v)
                                 : grid_uv(g, u, std::min(1.0, v + e)) - grid_uv(g, u, std::max(0.0, v - e));
    const cv::Point2d inw = grid_uv(g, u + (0.5 - u) * 0.02, v + (0.5 - v) * 0.02) - p;
    cv::Point2d nrm(-tg.y, tg.x);
    const double nl = cv::norm(nrm);
    if (nl < 1e-9) return std::nan("");
    nrm *= 1.0 / nl;
    if (nrm.dot(inw) > 0) nrm = -nrm;   // outwards
    const int steps = std::max(2, (int)std::ceil(reach * scale));
    double best = 1e300;
    int prev = -1;
    for (int t = -steps; t <= steps; ++t) {
        const cv::Point2d q = (p + nrm * (t / scale)) * scale;
        const int x = (int)std::lround(q.x), y = (int)std::lround(q.y);
        if (x < 1 || y < 1 || x >= mask.cols - 1 || y >= mask.rows - 1) {
            if (std::fabs((double)t) < std::fabs(best)) return std::nan("");   // the photo's edge comes first
            prev = -1;
            continue;
        }
        const int val = mask.at<uint8_t>(y, x) > 127;
        if (prev == 1 && val == 0 && std::fabs(t - 0.5) < std::fabs(best)) best = t - 0.5;
        prev = val;
    }
    if (best > 1e299) {
        const cv::Point2d q = p * scale;
        const int x = (int)std::lround(q.x), y = (int)std::lround(q.y);
        if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) return std::nan("");
        return mask.at<uint8_t>(y, x) > 127 ? reach : -reach;
    }
    return best / scale;
}

std::vector<double> median5(const std::vector<double>& d) {
    std::vector<double> o(d.size(), std::nan(""));
    for (size_t i = 0; i < d.size(); ++i) {
        std::vector<double> w;
        for (int j = (int)i - 2; j <= (int)i + 2; ++j)
            if (j >= 0 && j < (int)d.size() && std::isfinite(d[j])) w.push_back(d[j]);
        if (w.size() < 3) continue;
        std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
        o[i] = w[w.size() / 2];
    }
    return o;
}

}  // namespace

UvGrid uv_fuse_border(const UvGrid& a, const UvGrid& b, const cv::Mat& mask, double scale, FuseStats* st, double miss, int min_run,
                      double min_radius) {
    FuseStats s;
    if (st) *st = s;
    if (!a.ok || !b.ok || a.turned != b.turned || mask.empty() || a.gw < 4 || a.gh < 4 || b.gw < 4 || b.gh < 4) return a;
    const double size = 0.5 * (cv::norm(grid_uv(a, 0, 0) - grid_uv(a, 1, 1)) + cv::norm(grid_uv(a, 1, 0) - grid_uv(a, 0, 1))) / std::sqrt(2.0);
    const double reach = 0.15 * size, thr = miss * size;
    const auto sides = fuse_sides();
    // the runs taken from b: their samples' (u, v); the border samples where b itself misses the paper (no bump
    // reaches them: a bump that did would carry b's miss into the page)
    std::vector<std::vector<SideSample>> runs;
    std::vector<SideSample> b_bad;
    s.worst_a = s.worst_b = 0;
    // every side's misses first: each grid's own baseline (the median over its border: the mask's edge runs a few px
    // outside both nets' borders) is taken off before they are compared
    std::array<std::vector<double>, 4> MA, MB;
    std::vector<double> all_a, all_b;
    for (int k = 0; k < 4; ++k) {
        std::vector<double> ma(kFuseSamples), mb(kFuseSamples);
        for (int i = 0; i < kFuseSamples; ++i) {
            ma[i] = border_miss(a, sides[k][i].u, sides[k][i].v, mask, scale, reach);
            mb[i] = border_miss(b, sides[k][i].u, sides[k][i].v, mask, scale, reach);
        }
        MA[k] = median5(ma);
        MB[k] = median5(mb);
        for (double x : MA[k]) if (std::isfinite(x)) all_a.push_back(x);
        for (double x : MB[k]) if (std::isfinite(x)) all_b.push_back(x);
    }
    auto median_of = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    const double base_a = median_of(all_a), base_b = median_of(all_b);
    for (int k = 0; k < 4; ++k) {
        std::vector<double>& ma = MA[k];
        std::vector<double>& mb = MB[k];
        for (double& x : ma) x -= base_a;
        for (double& x : mb) x -= base_b;
        if (std::getenv("FS_FUSE_DEBUG")) {
            std::fprintf(stderr, "fuse side %d a:", k);
            for (double x : ma) std::fprintf(stderr, " %.0f", std::isfinite(x) ? 1000 * x / size : -999.0);
            std::fprintf(stderr, "\nfuse side %d b:", k);
            for (double x : mb) std::fprintf(stderr, " %.0f", std::isfinite(x) ? 1000 * x / size : -999.0);
            std::fprintf(stderr, "\n");
        }
        std::vector<char> take(kFuseSamples, 0);
        for (int i = 0; i < kFuseSamples; ++i) {
            if (std::isfinite(ma[i])) s.worst_a = std::max(s.worst_a, std::fabs(ma[i]) / size);
            if (std::isfinite(mb[i])) s.worst_b = std::max(s.worst_b, std::fabs(mb[i]) / size);
            if (std::isfinite(mb[i]) && std::fabs(mb[i]) > 1.5 * thr) b_bad.push_back(sides[k][i]);
            if (!std::isfinite(ma[i]) || !std::isfinite(mb[i])) continue;
            const double ea = std::fabs(ma[i]), eb = std::fabs(mb[i]);
            take[i] = ea > thr && eb <= 0.5 * ea && ea - eb >= 0.75 * thr;
        }
        for (int i = 0; i < kFuseSamples;) {
            if (!take[i]) { ++i; continue; }
            int j = i;
            while (j < kFuseSamples && take[j]) ++j;
            if (j - i >= min_run) runs.emplace_back(sides[k].begin() + i, sides[k].begin() + j);
            i = j;
        }
    }
    s.runs = (int)runs.size();
    if (runs.empty()) {
        if (st) *st = s;
        return a;
    }
    // each run's bump radius: its length (at least min_radius), but short of b's own misses
    std::vector<double> radius;
    for (const auto& run : runs) {
        const double len = std::hypot(run.back().u - run.front().u, run.back().v - run.front().v);
        double rad = std::max(min_radius, len), bad = 1e300;
        for (const auto& q : b_bad)
            for (const auto& p : run) bad = std::min(bad, std::hypot(p.u - q.u, p.v - q.v));
        radius.push_back(std::min(rad, 0.8 * bad));
    }
    // the weight of b at each of a's grid points: a bump around each run, in (u, v)
    UvGrid o = a;
    int heavy = 0;
    for (int r = 0; r < a.gh; ++r)
        for (int c = 0; c < a.gw; ++c) {
            const double u = (double)c / (a.gw - 1), v = (double)r / (a.gh - 1);
            double w = 0;
            for (size_t ri = 0; ri < runs.size(); ++ri) {
                const auto& run = runs[ri];
                const double rad = radius[ri];
                if (rad < 0.03) continue;
                double d = 1e300;
                for (const auto& q : run) d = std::min(d, std::hypot(u - q.u, v - q.v));
                w = std::max(w, smooth01(1.0 - d / rad));
            }
            if (w <= 0) continue;
            heavy += w > 0.5;
            cv::Point2d& p = o.pts[(size_t)r * a.gw + c];
            const cv::Point2d delta = (grid_uv(b, u, v) - p) * w;
            s.moved_px = std::max(s.moved_px, cv::norm(delta));
            p += delta;
        }
    s.area = (double)heavy / a.pts.size();
    if (st) *st = s;
    return o;
}

}  // namespace fs
