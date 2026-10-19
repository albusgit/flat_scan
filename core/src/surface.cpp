#include "surface.h"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <thread>

#include "lsq.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

static const int NS = 160;            // samples along the profile
static const double W_CORNER = 0.5;   // weight of the extrapolated corners

// surface._profile(c, L)
static void profile(const double* c, double L, Vec& s, Vec& X, Vec& Z) {
    Vec t = linspace(0, 1, NS);
    Vec th(NS);
    for (int i = 0; i < NS; ++i)
        th[i] = ((c[0] * t[i] + c[1] * (t[i] * t[i])) + c[2] * std::pow(t[i], 3.0)) + c[3] * std::pow(t[i], 4.0);
    double ds = L / (NS - 1);
    s.resize(NS); X.resize(NS); Z.resize(NS);
    X[0] = 0; Z[0] = 0;
    double ax = 0, az = 0;
    for (int i = 1; i < NS; ++i) {
        ax += 0.5 * (std::cos(th[i]) + std::cos(th[i - 1])) * ds;
        az += 0.5 * (std::sin(th[i]) + std::sin(th[i - 1])) * ds;
        X[i] = ax; Z[i] = az;
    }
    for (int i = 0; i < NS; ++i) s[i] = t[i] * L;
}

struct Model {
    char axis;
    double W, H, f, cx, cy;
    Vec s, X, Z;
    double R[9], t[3];
    void set(const double* x) {
        profile(x, axis == 'u' ? W : H, s, X, Z);
        cv::Mat rv = (cv::Mat_<double>(3, 1) << x[4], x[5], x[6]), Rm;
        cv::Rodrigues(rv, Rm);
        for (int i = 0; i < 9; ++i) R[i] = Rm.at<double>(i / 3, i % 3);
        t[0] = x[7]; t[1] = x[8]; t[2] = x[9];
    }
    // surface(u, v) then project
    cv::Point2d proj(double u, double v) const {
        double P[3];
        if (axis == 'u') { P[0] = interp(u, s, X); P[1] = v; P[2] = interp(u, s, Z); }
        else { P[0] = u; P[1] = interp(v, s, X); P[2] = interp(v, s, Z); }
        double Pc[3];
        for (int i = 0; i < 3; ++i) Pc[i] = ((P[0] * R[3 * i] + P[1] * R[3 * i + 1]) + P[2] * R[3 * i + 2]) + t[i];
        return cv::Point2d(f * Pc[0] / Pc[2] + cx, f * Pc[1] / Pc[2] + cy);
    }
};

// surface._pt_to_poly: distance from each point to the polyline Q
static void pt_to_poly(const std::vector<cv::Point2d>& P, const std::vector<cv::Point2d>& Q, Vec& out) {
    size_t ns = Q.size() - 1;
    std::vector<double> abx(ns), aby(ns), den(ns);
    for (size_t k = 0; k < ns; ++k) {
        abx[k] = Q[k + 1].x - Q[k].x; aby[k] = Q[k + 1].y - Q[k].y;
        den[k] = (abx[k] * abx[k] + aby[k] * aby[k]) + 1e-9;
    }
    for (const auto& p : P) {
        double best = 1e300;
        for (size_t k = 0; k < ns; ++k) {
            double px = p.x - Q[k].x, py = p.y - Q[k].y;
            double t = (px * abx[k] + py * aby[k]) / den[k];
            t = std::min(std::max(t, 0.0), 1.0);
            double dx = p.x - (Q[k].x + t * abx[k]), dy = p.y - (Q[k].y + t * aby[k]);
            best = std::min(best, dx * dx + dy * dy);      // sqrt is monotone: min first
        }
        out.push_back(std::sqrt(best));
    }
}

struct Problem {
    char axis;
    double W, H, f, cx, cy;
    int n_edge;
    std::array<std::vector<cv::Point2d>, 4> sub;
    std::array<cv::Point2d, 4> corners;
    Vec operator()(const Vec& x) const {
        Model M{axis, W, H, f, cx, cy, {}, {}, {}, {}, {}};
        M.set(x.data());
        Vec a = linspace(0, 1, n_edge);
        Vec r;
        r.reserve(sub[0].size() * 4 + 8);
        std::vector<cv::Point2d> Q(n_edge);
        for (int e = 0; e < 4; ++e) {   // top, right, bottom, left (surface._edges)
            for (int i = 0; i < n_edge; ++i) {
                double ai = a[i], ar = a[n_edge - 1 - i];
                double u, v;
                if (e == 0) { u = ai * W; v = 0 * ai; }
                else if (e == 1) { u = W + 0 * ai; v = ai * H; }
                else if (e == 2) { u = ar * W; v = H + 0 * ai; }
                else { u = 0 * ai; v = ar * H; }
                Q[i] = M.proj(u, v);
            }
            pt_to_poly(sub[e], Q, r);
        }
        const double obj[4][2] = {{0, 0}, {W, 0}, {W, H}, {0, H}};
        for (int k = 0; k < 4; ++k) {
            cv::Point2d Cm = M.proj(obj[k][0], obj[k][1]);
            double dx = Cm.x - corners[k].x, dy = Cm.y - corners[k].y;
            r.push_back(W_CORNER * std::sqrt(dx * dx + dy * dy));
        }
        for (int k = 0; k < 4; ++k) r.push_back(0.5 * x[k]);
        return r;
    }
};

static Problem make_problem(char axis, int n_obs, int n_edge, const std::array<std::vector<cv::Point2d>, 4>& pts,
                            const std::array<cv::Point2d, 4>& corners, double W, double H, double f, double cx, double cy) {
    Problem p{axis, W, H, f, cx, cy, n_edge, {}, corners};
    for (int k = 0; k < 4; ++k) {
        int len = (int)pts[k].size(), cnt = std::min(len, n_obs);
        Vec idx = linspace(0, len - 1, cnt);
        for (double v : idx) p.sub[k].push_back(pts[k][(size_t)(int64_t)v]);
    }
    return p;
}

SurfaceModel fit_surface(const std::array<std::vector<cv::Point2d>, 4>& side_pts, const std::array<cv::Point2d, 4>& corners,
                         int w, int h, bool landscape, double focal_px, int threads) {
    SurfaceModel best;
    double f = focal_px > 0 ? focal_px : 0.72 * std::max(h, w);
    double cx = w / 2.0, cy = h / 2.0;
    double W = landscape ? 297.0 : 210.0, H = landscape ? 210.0 : 297.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    std::vector<cv::Point3d> obj = {{0, 0, 0}, {W, 0, 0}, {W, H, 0}, {0, H, 0}};
    std::vector<cv::Point2d> img(corners.begin(), corners.end());
    cv::Mat rvec0, tvec0;
    if (!cv::solvePnP(obj, img, K, cv::noArray(), rvec0, tvec0, false, cv::SOLVEPNP_IPPE)) return best;
    const double starts[5][4] = {{0, 0, 0, 0}, {0, .6, 0, 0}, {0, -.6, 0, 0}, {-.6, 0, 0, 0}, {.6, 0, 0, 0}};
    struct Cand { double cost; char ax; Vec x; int order; };
    std::vector<Cand> cands(10);
    // coarse multi-start: 2 axes x 5 starts, independent -> parallel
    auto run_start = [&](int i) {
        char ax = i < 5 ? 'u' : 'v';
        Problem P = make_problem(ax, 50, 70, side_pts, corners, W, H, f, cx, cy);
        Vec x0(10);
        for (int k = 0; k < 4; ++k) x0[k] = starts[i % 5][k];
        for (int k = 0; k < 3; ++k) { x0[4 + k] = rvec0.at<double>(k); x0[7 + k] = tvec0.at<double>(k); }
        LsqResult r = least_squares_cauchy(P, x0, 2.0, 150, 1);
        cands[i] = Cand{r.cost, ax, r.x, i};
    };
    {
        std::vector<std::thread> ts;
        int nt = std::max(1, std::min(threads, 10));
        for (int t = 0; t < nt; ++t) ts.emplace_back([&, t] { for (int i = t; i < 10; i += nt) run_start(i); });
        for (auto& t : ts) t.join();
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.cost < b.cost; });
    // the (at most 2) distinct basins to refine; this choice does not depend on the refinement
    std::vector<Cand> seen;
    for (const auto& c : cands) {
        bool dup = false;
        for (const auto& s : seen) {
            if (s.ax != c.ax) continue;
            bool close = true;
            for (int k = 0; k < 4; ++k) close &= std::fabs(c.x[k] - s.x[k]) <= 0.05 + 1e-5 * std::fabs(s.x[k]);
            dup |= close;
        }
        if (dup) continue;
        seen.push_back(c);
        if (seen.size() == 2) break;
    }
    std::vector<LsqResult> ref(seen.size());
    std::vector<Vec> rr(seen.size());
    {
        std::vector<std::thread> ts;
        int jt = std::max(1, threads / (int)seen.size());
        for (size_t i = 0; i < seen.size(); ++i)
            ts.emplace_back([&, i] {
                Problem P = make_problem(seen[i].ax, 150, 300, side_pts, corners, W, H, f, cx, cy);
                ref[i] = least_squares_cauchy(P, seen[i].x, 2.0, 200, jt);
                rr[i] = P(ref[i].x);
                rr[i].resize(rr[i].size() - 4);
            });
        for (auto& t : ts) t.join();
    }
    for (size_t i = 0; i < seen.size(); ++i) {
        if (best.ok && !(ref[i].cost < best.cost)) continue;
        best.ok = true;
        best.axis = seen[i].ax;
        for (int k = 0; k < 4; ++k) best.c[k] = ref[i].x[k];
        for (int k = 0; k < 3; ++k) { best.rvec[k] = ref[i].x[4 + k]; best.tvec[k] = ref[i].x[7 + k]; }
        best.f = f; best.cx = cx; best.cy = cy; best.W = W; best.H = H;
        best.cost = ref[i].cost;
        Vec a(rr[i].size());
        for (size_t k = 0; k < a.size(); ++k) a[k] = std::fabs(rr[i][k]);
        best.median_px = median(a);
        best.p90_px = percentile(a, 90);
    }
    if (best.ok) best.precompute();
    return best;
}

void SurfaceModel::precompute() {
    double x[10] = {c[0], c[1], c[2], c[3], rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]};
    profile(x, axis == 'u' ? W : H, s_, X_, Z_);
    cv::Mat rv = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]), Rm;
    cv::Rodrigues(rv, Rm);
    for (int i = 0; i < 9; ++i) R_[i] = Rm.at<double>(i / 3, i % 3);
}

cv::Point2d SurfaceModel::project_uv(double u, double v) const {
    double P[3];
    if (axis == 'u') { P[0] = interp(u, s_, X_); P[1] = v; P[2] = interp(u, s_, Z_); }
    else { P[0] = u; P[1] = interp(v, s_, X_); P[2] = interp(v, s_, Z_); }
    double Pc[3];
    for (int i = 0; i < 3; ++i) Pc[i] = ((P[0] * R_[3 * i] + P[1] * R_[3 * i + 1]) + P[2] * R_[3 * i + 2]) + tvec[i];
    return cv::Point2d(f * Pc[0] / Pc[2] + cx, f * Pc[1] / Pc[2] + cy);
}

cv::Point2d SurfaceModel::unroll_at(double x, double y, int out_w, int out_h) const {
    return project_uv(x * W / (out_w - 1), y * H / (out_h - 1));
}

}  // namespace fs
