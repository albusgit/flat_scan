#include "uvdoc.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <functional>

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

}  // namespace fs
