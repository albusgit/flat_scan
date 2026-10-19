#include "corners.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

#pragma STDC FP_CONTRACT OFF

namespace fs {

static double dot(const cv::Point2d& a, const cv::Point2d& b) { return a.x * b.x + a.y * b.y; }
// np.linalg.norm of a 2-vector: sqrt(x*x + y*y) (not hypot, which rounds differently)
static double norm(const cv::Point2d& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

cv::Point2d Side::point(double s) const {
    double o = polyval(p, s);
    return cv::Point2d(P0.x + s * d.x + o * n.x, P0.y + s * d.y + o * n.y);
}

double Side::signed_dist(const cv::Point2d& X) const {
    cv::Point2d r = X - P0;
    double s = dot(r, d), o = dot(r, n);
    return o - polyval(p, s);
}

// corners.paper_mask (round 4): tries progressively looser definitions of paper and keeps the
// first whose largest component covers 10-99% of the frame; otherwise the whole frame, inset 2 px
bool paper_mask(const cv::Mat& img, std::vector<cv::Point2d>& contour, cv::Mat& full, bool* whole_frame) {
    cv::Mat blur, lab8;
    cv::GaussianBlur(img, blur, cv::Size(7, 7), 0);
    cv::cvtColor(blur, lab8, cv::COLOR_BGR2Lab);
    cv::Mat chroma8(img.size(), CV_8U), L8(img.size(), CV_8U);
    for (int y = 0; y < img.rows; ++y) {
        const uint8_t* q = lab8.ptr<uint8_t>(y);
        for (int x = 0; x < img.cols; ++x) {
            float a = (float)q[3 * x + 1] - 128.f, b = (float)q[3 * x + 2] - 128.f;
            float c = std::hypot(a, b) * 8.f;                       // float32, as numpy
            c = std::min(std::max(c, 0.f), 255.f);
            chroma8.at<uint8_t>(y, x) = (uint8_t)c;                 // astype(uint8) truncates
            L8.at<uint8_t>(y, x) = q[3 * x];
        }
    }
    cv::Mat chroma_med, tmp;
    cv::medianBlur(chroma8, chroma_med, 5);
    double otsu = cv::threshold(L8, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const double area = (double)img.rows * img.cols;
    if (whole_frame) *whole_frame = false;
    for (int cand = 0; cand < 3; ++cand) {
        cv::Mat m(img.size(), CV_8U);
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                double L = L8.at<uint8_t>(y, x), ch = chroma_med.at<uint8_t>(y, x) / 8.0;
                bool v = cand == 0 ? (L > 150.0 && ch < 7.0)          // white, neutral paper (original rule)
                       : cand == 1 ? (L > otsu && ch < 25.0)          // dimmer / tinted light
                                   : (L > otsu);                      // coloured paper (e.g. blue)
                m.at<uint8_t>(y, x) = v ? 1 : 0;
            }
        cv::morphologyEx(m, m, cv::MORPH_CLOSE, cv::Mat::ones(25, 25, CV_8U));
        cv::morphologyEx(m, m, cv::MORPH_OPEN, cv::Mat::ones(9, 9, CV_8U));
        cv::Mat labels, stats, cent;
        int n = cv::connectedComponentsWithStats(m, labels, stats, cent, 8);
        if (n < 2) continue;
        int k = 1;
        for (int i = 2; i < n; ++i)
            if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(k, cv::CC_STAT_AREA)) k = i;
        double frac = stats.at<int>(k, cv::CC_STAT_AREA) / area;
        if (!(0.10 <= frac && frac <= 0.99)) continue;
        cv::Mat mk = (labels == k) / 255;
        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(mk, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (cs.empty()) continue;
        size_t best = 0;
        double ba = -1;
        for (size_t i = 0; i < cs.size(); ++i) {           // max(cs, key=contourArea): first maximum
            double a = cv::contourArea(cs[i]);
            if (a > ba) { ba = a; best = i; }
        }
        contour.clear();
        for (auto& p : cs[best]) contour.emplace_back(p.x, p.y);
        full = cv::Mat::zeros(img.size(), CV_8U);                // corners.paper_mask's `full`
        cv::drawContours(full, cs, (int)best, cv::Scalar(255), cv::FILLED);
        return contour.size() >= 20;
    }
    // fallback: whole frame (inset 2 px so the outline has four clean sides)
    if (whole_frame) *whole_frame = true;
    full = cv::Mat::zeros(img.size(), CV_8U);
    if (img.rows <= 4 || img.cols <= 4) return false;
    full(cv::Rect(2, 2, img.cols - 4, img.rows - 4)).setTo(255);
    cv::Mat m01 = full / 255;
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m01, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    contour.clear();
    if (cs.empty()) return false;
    for (auto& p : cs[0]) contour.emplace_back(p.x, p.y);
    return contour.size() >= 20;
}

static std::array<cv::Point2d, 4> order(const std::array<cv::Point2d, 4>& q) {
    int imin_s = 0, imax_d = 0, imax_s = 0, imin_d = 0;
    for (int i = 1; i < 4; ++i) {
        double s = q[i].x + q[i].y, d = q[i].x - q[i].y;
        if (s < q[imin_s].x + q[imin_s].y) imin_s = i;
        if (s > q[imax_s].x + q[imax_s].y) imax_s = i;
        if (d > q[imax_d].x - q[imax_d].y) imax_d = i;
        if (d < q[imin_d].x - q[imin_d].y) imin_d = i;
    }
    return {q[imin_s], q[imax_d], q[imax_s], q[imin_d]};
}

static double ptp(const Vec& v) {
    auto mm = std::minmax_element(v.begin(), v.end());
    return *mm.second - *mm.first;
}

// corners.ransac_quadratic(s, o, thr=2.5, iters=400, rng=<shared>)
static void ransac_quadratic(const Vec& s, const Vec& o, NpRng& rng, Vec& p, std::vector<char>& best) {
    const double thr = 2.5;
    size_t n = s.size();
    double span = ptp(s);
    best.clear();
    size_t best_cnt = 0;
    bool have = false;
    std::vector<char> inl(n);
    for (int it = 0; it < 400; ++it) {
        auto idx = rng.choice_no_replace((int64_t)n, 3);
        double a = s[idx[0]], b = s[idx[1]], c = s[idx[2]];
        if (std::max({a, b, c}) - std::min({a, b, c}) < 0.2 * span) continue;
        double xs[3] = {a, b, c}, ys[3] = {o[idx[0]], o[idx[1]], o[idx[2]]};
        Vec q = polyfit(xs, ys, 3, 2);
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) cnt += (inl[i] = std::fabs(polyval(q, s[i]) - o[i]) < thr);
        if (!have || cnt > best_cnt) { best = inl; best_cnt = cnt; have = true; }
    }
    if (!have) best.assign(n, 1);
    for (int r = 0; r < 3; ++r) {
        Vec ss, oo;
        for (size_t i = 0; i < n; ++i)
            if (best[i]) { ss.push_back(s[i]); oo.push_back(o[i]); }
        p = polyfit(ss, oo, 2);
        for (size_t i = 0; i < n; ++i) best[i] = std::fabs(polyval(p, s[i]) - o[i]) < thr;
    }
}

static Side make_side(const cv::Point2d& P0, const cv::Point2d& P1, const std::vector<cv::Point2d>& pts, NpRng& rng) {
    Side sd;
    sd.P0 = P0;
    cv::Point2d v = P1 - P0;
    double nv = norm(v);
    sd.d = cv::Point2d(v.x / nv, v.y / nv);
    sd.n = cv::Point2d(-sd.d.y, sd.d.x);
    sd.len = nv;
    Vec s(pts.size()), o(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) { s[i] = dot(pts[i] - P0, sd.d); o[i] = dot(pts[i] - P0, sd.n); }
    ransac_quadratic(s, o, rng, sd.p, sd.inl);
    return sd;
}

static int sgn(double v) { return (v > 0) - (v < 0); }

// corners.intersect(A, B, ext=0.25)
static cv::Point2d intersect(const Side& A, const Side& B) {
    const int N = 4000;
    Vec s = linspace(-0.25 * A.len, 1.25 * A.len, N);
    std::vector<cv::Point2d> X(N);
    Vec f(N);
    for (int i = 0; i < N; ++i) { X[i] = A.point(s[i]); f[i] = B.signed_dist(X[i]); }
    int bi = -1;
    double bd = 0;
    for (int i = 0; i + 1 < N; ++i)
        if (sgn(f[i]) != sgn(f[i + 1])) {
            double dd = std::fabs(s[i] - std::min(std::max(s[i], 0.0), A.len));
            if (bi < 0 || dd < bd) { bi = i; bd = dd; }
        }
    if (bi < 0) {
        int k = 0;
        for (int i = 1; i < N; ++i)
            if (std::fabs(f[i]) < std::fabs(f[k])) k = i;
        return X[k];
    }
    double t = f[bi] / (f[bi] - f[bi + 1]);
    return X[bi] + t * (X[bi + 1] - X[bi]);
}

Detection detect_page(const cv::Mat& img, NpRng& rng) {
    Detection D;
    bool whole = false;
    if (!paper_mask(img, D.contour, D.paper, &whole)) return D;
    D.whole_frame = whole;
    corners_from_contour(D, rng);
    return D;
}

// corners.curved_corners on D.contour (any page finder) -> sides, corners, fold gaps, landscape
void corners_from_contour(Detection& D, NpRng& rng) {
    D.ok = false;
    const auto& contour = D.contour;
    if (contour.size() < 20) return;
    std::vector<cv::Point2f> c32(contour.size());
    for (size_t i = 0; i < contour.size(); ++i) c32[i] = cv::Point2f((float)contour[i].x, (float)contour[i].y);
    std::vector<cv::Point2f> hull;
    cv::convexHull(c32, hull);
    cv::Point2f bp[4];
    cv::minAreaRect(hull).points(bp);
    std::array<cv::Point2d, 4> box;
    for (int i = 0; i < 4; ++i) box[i] = cv::Point2d(bp[i].x, bp[i].y);   // boxPoints -> float64
    box = order(box);
    std::array<std::vector<cv::Point2d>, 4> pts;
    for (const auto& c : contour) {
        int best = 0;
        double bd = 0;
        for (int k = 0; k < 4; ++k) {
            cv::Point2d P0 = box[k], P1 = box[(k + 1) % 4], v = P1 - P0;
            double dk = std::fabs((c.x - P0.x) * (-v.y) + (c.y - P0.y) * v.x) / norm(v);
            if (k == 0 || dk < bd) { bd = dk; best = k; }
        }
        pts[best].push_back(c);
    }
    for (int k = 0; k < 4; ++k) {
        if (pts[k].size() < 3) return;
        D.sides[k] = make_side(box[k], box[(k + 1) % 4], pts[k], rng);
    }
    for (int k = 0; k < 4; ++k) D.corners[k] = intersect(D.sides[(k + 3) % 4], D.sides[k]);
    for (int k = 0; k < 4; ++k) {
        double m = 1e300;
        for (const auto& c : contour) m = std::min(m, norm(c - D.corners[k]));
        D.fold_d[k] = m;
    }
    D.side_pts = pts;
    cv::Point2d w = D.corners[1] - D.corners[0], h = D.corners[3] - D.corners[0];
    D.landscape = norm(w) > norm(h);
    D.ok = true;
}

}  // namespace fs
