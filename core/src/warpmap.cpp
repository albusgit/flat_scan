#include "warpmap.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <atomic>
#include <cstdlib>
#include <thread>

#include "nprng.h"
#include "sparse.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Python int(x): truncation toward zero
inline long pyint(double x) { return (long)std::trunc(x); }
// Python round(x) for floats: half to even
inline long pyround(double x) { return (long)std::nearbyint(x); }

// numpy floor_divide for floats (npy_divmod)
double np_floordiv(double a, double b) {
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) div -= 1.0;
    }
    double fl;
    if (div != 0) {
        fl = std::floor(div);
        if (div - fl > 0.5) fl += 1.0;
    } else {
        fl = std::copysign(0.0, a / b);
    }
    return fl;
}

double np_mean(const Vec& v) { return np_sum(v) / (double)v.size(); }

// ------------------------------------------------------------------ morphology helpers
cv::Mat rect(int w, int h) { return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h)); }

// ------------------------------------------------------------------ dewarp.py (global stage)
struct DLine {
    std::vector<double> t, v;   // t: integer positions (stored as double), concatenated as Python does
    Vec p;
    double t0, t1, target = 0;
};

Vec robust_fit(const std::vector<double>& t, const std::vector<double>& v, int deg) {
    deg = std::min(deg, std::max(1, (int)(t.size() / 300)));
    size_t n = t.size();
    std::vector<char> m(n, 1);
    Vec p;
    for (int it = 0; it < 4; ++it) {
        Vec tt, vv, rm;
        for (size_t i = 0; i < n; ++i)
            if (m[i]) { tt.push_back(t[i]); vv.push_back(v[i]); }
        p = polyfit(tt, vv, deg);
        Vec r(n);
        for (size_t i = 0; i < n; ++i) r[i] = std::fabs(polyval(p, t[i]) - v[i]);
        for (size_t i = 0; i < n; ++i)
            if (m[i]) rm.push_back(r[i]);
        double thr = std::max(1.5, 2.5 * median(rm));
        for (size_t i = 0; i < n; ++i) m[i] = r[i] < thr;
    }
    return p;
}

// comp_samples(mask, horizontal, min_len)
std::vector<std::pair<std::vector<double>, std::vector<double>>> comp_samples(const cv::Mat& mask, bool hz, int min_len) {
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(mask, lab, st, cen, 8);
    std::vector<std::pair<std::vector<double>, std::vector<double>>> out;
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), w = st.at<int>(i, 2), h = st.at<int>(i, 3);
        int L = hz ? w : h;
        if (L < min_len) continue;
        std::vector<double> t, v;
        if (hz) {
            for (int c = 0; c < w; ++c) {
                long cnt = 0, s = 0;
                for (int r = 0; r < h; ++r)
                    if (lab.at<int>(y + r, x + c) == i) { ++cnt; s += r; }
                if (cnt > 0) { t.push_back(c + x); v.push_back((double)s / (double)std::max(cnt, 1L) + y); }
            }
        } else {
            for (int r = 0; r < h; ++r) {
                long cnt = 0, s = 0;
                for (int c = 0; c < w; ++c)
                    if (lab.at<int>(y + r, x + c) == i) { ++cnt; s += c; }
                if (cnt > 0) { t.push_back(r + y); v.push_back((double)s / (double)std::max(cnt, 1L) + x); }
            }
        }
        out.emplace_back(std::move(t), std::move(v));
    }
    return out;
}

double vmin(const std::vector<double>& a) { return *std::min_element(a.begin(), a.end()); }
double vmax(const std::vector<double>& a) { return *std::max_element(a.begin(), a.end()); }

std::vector<DLine> merge_lines(std::vector<std::pair<std::vector<double>, std::vector<double>>> samples, double tol = 7) {
    std::stable_sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    std::vector<DLine> lines;
    for (auto& sm : samples) {
        const auto& t = sm.first;
        const auto& v = sm.second;
        Vec p = robust_fit(t, v, 3);
        double tmin = vmin(t), tmax = vmax(t);
        bool merged = false;
        for (auto& L : lines) {
            double a = L.t0 < tmax ? std::max(L.t0, tmin) : tmin;
            double b = L.t1 > tmin ? std::min(L.t1, tmax) : tmax;
            Vec tc = linspace(a, b, 5);
            tc.push_back(tmin); tc.push_back(tmax);
            double d = 0;
            for (double q : tc) d = std::max(d, std::fabs(polyval(L.p, q) - polyval(p, q)));
            double overlap = std::min(L.t1, tmax) - std::max(L.t0, tmin);
            if (d < tol && overlap < 0.5 * (tmax - tmin)) {
                L.t.insert(L.t.end(), t.begin(), t.end());
                L.v.insert(L.v.end(), v.begin(), v.end());
                L.p = robust_fit(L.t, L.v, 3);
                L.t0 = vmin(L.t); L.t1 = vmax(L.t);
                merged = true;
                break;
            }
            if (d < tol && overlap >= 0.5 * (tmax - tmin)) { merged = true; break; }
        }
        if (!merged) lines.push_back(DLine{t, v, p, tmin, tmax});
    }
    for (auto& L : lines) {
        Vec tt = linspace(L.t0, L.t1, 50), y(50);
        for (int i = 0; i < 50; ++i) y[i] = polyval(L.p, tt[i]);
        L.target = np_mean(y);
    }
    return lines;
}

// App only (faint_lines): long, thin, isolated runs of faint ink along one axis - printed rules too
// light for the darkness-45 threshold (pale grey form rules, blurry low-resolution photos blown up
// to page size, where a rule's darkness is 40 - 55). At a low threshold text rows merge into long
// runs too; those are thick, or have ink on both sides of the run, and are left out. A rule needs
// paper on at least one side (text may sit on it). Df: darkness (float, background - gray).
cv::Mat faint_rules(const cv::Mat& Df, bool hz, int klen, double lo = 25, double maxw = 7) {
    cv::Mat ink = Df > lo, m;
    cv::morphologyEx(ink, m, cv::MORPH_OPEN, hz ? rect(klen, 1) : rect(1, klen));
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(m, lab, st, cen, 8);
    cv::Mat out = cv::Mat::zeros(m.size(), CV_8U);
    auto Dat = [&](int along, double across) {
        int a = (int)std::lround(across);
        int r = hz ? a : along, c = hz ? along : a;
        if (r < 0 || c < 0 || r >= Df.rows || c >= Df.cols) return 0.f;
        return Df.at<float>(r, c);
    };
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), w = st.at<int>(i, 2), h = st.at<int>(i, 3);
        int L = hz ? w : h, X = hz ? h : w;
        if (L < klen) continue;
        Vec widths, centre, side;
        for (int k = 0; k < L; ++k) {
            long cnt = 0, s = 0;
            for (int q = 0; q < X; ++q) {
                int r = hz ? y + q : y + k, c = hz ? x + k : x + q;
                if (lab.at<int>(r, c) == i) { ++cnt; s += q; }
            }
            if (cnt == 0) continue;
            double ctr = (double)s / cnt + (hz ? y : x), off = cnt / 2.0 + 4;
            int along = hz ? x + k : y + k;
            widths.push_back((double)cnt);
            centre.push_back(Dat(along, ctr));
            side.push_back(std::min(Dat(along, ctr - off), Dat(along, ctr + off)));
        }
        // and 2+ px: a 1-px run at this threshold is dots merged by the blur (a chart's dotted
        // gridline), whose jittery trace bends the text beside it
        if (widths.empty() || median(widths) > maxw || median(widths) < 2 || median(side) >= 0.4 * median(centre)) continue;
        cv::Mat roi = out(cv::Rect(x, y, w, h));
        cv::Mat sel = lab(cv::Rect(x, y, w, h)) == i;
        roi.setTo(255, sel);
    }
    return out;
}

void dewarp_detect(const cv::Mat& img, std::vector<DLine>& Hs, std::vector<DLine>& Vs, bool faint = false) {
    cv::Mat g, bg;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    cv::dilate(g, bg, cv::Mat::ones(9, 9, CV_8U));
    cv::medianBlur(bg, bg, 41);
    cv::Mat ink(g.size(), CV_8U);
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x) ink.at<uint8_t>(y, x) = ((int)bg.at<uint8_t>(y, x) - (int)g.at<uint8_t>(y, x)) > 45 ? 255 : 0;
    cv::Mat hm, vm;
    cv::morphologyEx(ink, hm, cv::MORPH_OPEN, rect(120, 1));
    cv::morphologyEx(ink, vm, cv::MORPH_OPEN, rect(1, 120));
    if (faint) {
        cv::Mat Df;
        cv::subtract(bg, g, Df, cv::noArray(), CV_32F);
        hm |= faint_rules(Df, true, 120);
        vm |= faint_rules(Df, false, 120);
    }
    cv::morphologyEx(hm, hm, cv::MORPH_CLOSE, rect(25, 3));
    cv::morphologyEx(vm, vm, cv::MORPH_CLOSE, rect(3, 25));
    Hs = merge_lines(comp_samples(hm, true, 250));
    Vs = merge_lines(comp_samples(vm, false, 250));
}

void dl_trace(const DLine& L, Vec& tc, Vec& vc, double step = 20) {
    size_t n = L.t.size();
    std::vector<long> bins(n);
    long nb = 0;
    for (size_t i = 0; i < n; ++i) { bins[i] = (long)std::floor((L.t[i] - L.t0) / step); nb = std::max(nb, bins[i] + 1); }
    std::vector<Vec> bt(nb), bv(nb);
    for (size_t i = 0; i < n; ++i) { bt[bins[i]].push_back(L.t[i]); bv[bins[i]].push_back(L.v[i]); }
    tc.clear(); vc.clear();
    for (long b = 0; b < nb; ++b) {
        if ((double)bt[b].size() < step * 0.4) continue;
        tc.push_back(np_mean(bt[b]));   // integer positions: the float64 sum is exact in any order
        vc.push_back(median(bv[b]));
    }
    if (vc.size() >= 5) {
        Vec rm = median_filter1d_nearest(vc, 5);
        Vec t2, v2;
        for (size_t i = 0; i < vc.size(); ++i)
            if (std::fabs(vc[i] - rm[i]) < 3) { t2.push_back(tc[i]); v2.push_back(vc[i]); }
        tc = t2; vc = v2;
    }
}

void build_controls(const std::vector<DLine>& Hs, const std::vector<DLine>& Vs, std::vector<cv::Point2d>& src,
                    std::vector<cv::Point2d>& dst, double pad = 15) {
    std::vector<std::vector<std::pair<double, double>>> ch(Hs.size()), cvv(Vs.size());
    for (size_t i = 0; i < Hs.size(); ++i)
        for (size_t j = 0; j < Vs.size(); ++j) {
            double y = Hs[i].target, x = 0;
            for (int it = 0; it < 10; ++it) { x = polyval(Vs[j].p, y); y = polyval(Hs[i].p, x); }
            if (Hs[i].t0 - pad <= x && x <= Hs[i].t1 + pad && Vs[j].t0 - pad <= y && y <= Vs[j].t1 + pad) {
                src.emplace_back(x, y);
                dst.emplace_back(Vs[j].target, Hs[i].target);
                ch[i].emplace_back(x, Vs[j].target);
                cvv[j].emplace_back(y, Hs[i].target);
            }
        }
    auto along = [&](const DLine& L, std::vector<std::pair<double, double>> c, bool hz) {
        std::sort(c.begin(), c.end());
        Vec tc, vc;
        dl_trace(L, tc, vc);
        Vec xs, Xs;
        for (auto& q : c) { xs.push_back(q.first); Xs.push_back(q.second); }
        for (size_t k = 0; k < tc.size(); ++k) {
            double a = tc[k], X;
            if (c.size() >= 2) {
                X = interp(a, xs, Xs);
                if (a < xs[0]) X = Xs[0] + (a - xs[0]);
                if (a > xs.back()) X = Xs.back() + (a - xs.back());
            } else if (c.size() == 1) {
                X = c[0].second + (a - c[0].first);
            } else {
                continue;
            }
            if (hz) { src.emplace_back(a, vc[k]); dst.emplace_back(X, L.target); }
            else { src.emplace_back(vc[k], a); dst.emplace_back(L.target, X); }
        }
    };
    for (size_t i = 0; i < Hs.size(); ++i) along(Hs[i], ch[i], true);
    for (size_t j = 0; j < Vs.size(); ++j) along(Vs[j], cvv[j], false);
}

// ------------------------------------------------------------------ regulariser + solve
const double LAM_TPS = 0.5, LAM_MEM = 0.01;

// The normal matrix has a fixed 13-point stencil (biharmonic + membrane + bilinear data
// terms): offsets (dy, dx) with |dy| + |dx| <= 2, accumulated directly per node.
const int NOFF = 13;
const int OFF_DY[NOFF] = {-2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2};
const int OFF_DX[NOFF] = {0, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1, 0};
inline int off_index(int dy, int dx) {
    static const int lut[5][5] = {{-1, -1, 0, -1, -1}, {-1, 1, 2, 3, -1}, {4, 5, 6, 7, 8}, {-1, 9, 10, 11, -1}, {-1, -1, 12, -1, -1}};
    return lut[dy + 2][dx + 2];
}

struct Stencil {
    int gw, gh;
    std::vector<double> a;   // n x 13
    Stencil(int w, int h) : gw(w), gh(h), a((size_t)w * h * NOFF, 0.0) {}
    // add w_i * w_j for all node pairs of one row of a least-squares operator
    void add_row(const int* iy, const int* ix, const double* w, int k) {
        for (int p = 0; p < k; ++p)
            for (int q = 0; q < k; ++q)
                a[((size_t)iy[p] * gw + ix[p]) * NOFF + off_index(iy[q] - iy[p], ix[q] - ix[p])] += w[p] * w[q];
    }
    Csr to_csr() const {
        Csr M;
        int n = gw * gh;
        M.n = n;
        M.ptr.assign(n + 1, 0);
        M.idx.reserve((size_t)n * NOFF);
        M.val.reserve((size_t)n * NOFF);
        for (int iy = 0; iy < gh; ++iy)
            for (int ix = 0; ix < gw; ++ix) {
                for (int o = 0; o < NOFF; ++o) {
                    int y = iy + OFF_DY[o], x = ix + OFF_DX[o];
                    if (y < 0 || y >= gh || x < 0 || x >= gw) continue;
                    M.idx.push_back(y * gw + x);
                    M.val.push_back(a[((size_t)iy * gw + ix) * NOFF + o]);
                }
                M.ptr[iy * gw + ix + 1] = (int)M.idx.size();
            }
        return M;
    }
};

// Lm^T Lm + 1e-9 I for warpmap.regularizer(gw, gh): thin-plate (second differences in x, y
// and the mixed term, weight LAM_TPS) + membrane (LAM_MEM)
void add_regularizer(Stencil& S) {
    int gw = S.gw, gh = S.gh;
    double s1 = std::sqrt(LAM_TPS), s2 = std::sqrt(2 * LAM_TPS), s3 = std::sqrt(LAM_MEM);
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix + 2 < gw; ++ix) {
            int yy[3] = {iy, iy, iy}, xx[3] = {ix + 2, ix + 1, ix};
            double w[3] = {1 * s1, -2 * s1, 1 * s1};
            S.add_row(yy, xx, w, 3);
        }
    for (int iy = 0; iy + 2 < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            int yy[3] = {iy + 2, iy + 1, iy}, xx[3] = {ix, ix, ix};
            double w[3] = {1 * s1, -2 * s1, 1 * s1};
            S.add_row(yy, xx, w, 3);
        }
    for (int iy = 0; iy + 1 < gh; ++iy)
        for (int ix = 0; ix + 1 < gw; ++ix) {
            int yy[4] = {iy + 1, iy + 1, iy, iy}, xx[4] = {ix + 1, ix, ix + 1, ix};
            double w[4] = {1 * s2, -1 * s2, -1 * s2, 1 * s2};
            S.add_row(yy, xx, w, 4);
        }
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix + 1 < gw; ++ix) {
            int yy[2] = {iy, iy}, xx[2] = {ix + 1, ix};
            double w[2] = {1 * s3, -1 * s3};
            S.add_row(yy, xx, w, 2);
        }
    for (int iy = 0; iy + 1 < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            int yy[2] = {iy + 1, iy}, xx[2] = {ix, ix};
            double w[2] = {1 * s3, -1 * s3};
            S.add_row(yy, xx, w, 2);
        }
    for (size_t i = 0; i < (size_t)gw * gh; ++i) S.a[i * NOFF + 6] += 1e-9;
}

// The fields' CG tolerance (warpmap.h: set_warp_cg_rtol). 1e-8 is the reference.
static std::atomic<double>& cg_rtol_store() {
    static std::atomic<double> v{[] {
        const char* e = std::getenv("FS_CG_RTOL");
        double r = e ? std::atof(e) : 0;
        return r > 0 ? r : 1e-8;
    }()};
    return v;
}

// warpmap.solve: regularised least squares on a G-px grid
Field solve_field(const std::vector<std::array<double, 2>>& xy, const Vec& values, const Vec* weights, int gw, int gh,
                  int G, const Field* warm, int* iters) {
    int64_t n = (int64_t)gw * gh;
    Stencil S(gw, gh);
    add_regularizer(S);
    std::vector<double> rhs(n, 0.0);
    for (size_t s = 0; s < xy.size(); ++s) {
        double gx = xy[s][0] / G, gy = xy[s][1] / G;
        long x0 = std::min(std::max((long)std::floor(gx), 0L), (long)gw - 2);
        long y0 = std::min(std::max((long)std::floor(gy), 0L), (long)gh - 2);
        double fx = std::min(std::max(gx - x0, 0.0), 1.0), fy = std::min(std::max(gy - y0, 0.0), 1.0);
        int yy[4] = {(int)y0, (int)y0, (int)y0 + 1, (int)y0 + 1}, xx[4] = {(int)x0, (int)x0 + 1, (int)x0, (int)x0 + 1};
        double w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        double ws = weights ? (*weights)[s] : 1.0;
        for (int a = 0; a < 4; ++a) {
            int ia = yy[a] * gw + xx[a];
            rhs[ia] += w[a] * (ws * values[s]);
            for (int b = 0; b < 4; ++b) S.a[(size_t)ia * NOFF + off_index(yy[b] - yy[a], xx[b] - xx[a])] += w[a] * ws * w[b];
        }
    }
    Csr M = S.to_csr();
    std::vector<double> x0;
    if (warm) x0 = warm->v;
    SolveStats st;
    Field F;
    F.gw = gw; F.gh = gh; F.g = G;
    F.v = solve_spd_grid(M, rhs, gw, gh, warm ? &x0 : nullptr, warp_cg_rtol(), 1000, &st);
    if (iters) *iters = st.iterations;
    return F;
}

// warpmap._grid_to_grid: cv2.remap(F.astype(f32), gx*g_to/g_from, ..., LINEAR, REPLICATE)
Field grid_to_grid(const Field& F, int g_to, int gw, int gh) {
    Field o;
    o.gw = gw; o.gh = gh; o.g = g_to;
    o.v.resize((size_t)gw * gh);
    std::vector<float> f(F.v.begin(), F.v.end());
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            float X = (float)ix * (float)g_to / (float)F.g, Y = (float)iy * (float)g_to / (float)F.g;
            int Xi = (int)std::lrint(X * 32.f), Yi = (int)std::lrint(Y * 32.f);
            int sx = Xi >> 5, sy = Yi >> 5;
            float ax = (Xi & 31) * (1.f / 32), ay = (Yi & 31) * (1.f / 32);
            float vx0 = 1.f - ax, vy0 = 1.f - ay;
            auto at = [&](int y, int x) {
                x = x >= 0 ? (x < F.gw ? x : F.gw - 1) : 0;
                y = y >= 0 ? (y < F.gh ? y : F.gh - 1) : 0;
                return f[(size_t)y * F.gw + x];
            };
            o.v[(size_t)iy * gw + ix] = at(sy, sx) * (vy0 * vx0) + at(sy, sx + 1) * (vy0 * ax) + at(sy + 1, sx) * (ay * vx0) +
                                        at(sy + 1, sx + 1) * (ay * ax);
        }
    return o;
}

// ------------------------------------------------------------------ clean.py line segments
Lines segments(const cv::Mat& ink, bool hz, int klen = 100, int minlen = 150, bool merge = true, int close = 31) {
    cv::Mat m0, m;
    cv::morphologyEx(ink, m0, cv::MORPH_OPEN, hz ? rect(klen, 1) : rect(1, klen));
    cv::morphologyEx(m0, m, cv::MORPH_CLOSE, hz ? rect(close, 1) : rect(1, close));
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(m, lab, st, cen, 8);
    Lines segs;
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), w = st.at<int>(i, 2), h = st.at<int>(i, 3);
        int L = hz ? w : h;
        if (L < minlen) continue;
        long tot = 0, ps = 0;
        std::vector<long> cnt(hz ? w : h, 0);
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                if (lab.at<int>(y + r, x + c) == i && m0.at<uint8_t>(y + r, x + c) > 0) {
                    ++tot;
                    ps += hz ? r : c;
                    cnt[hz ? c : r]++;
                }
        if (tot == 0) continue;
        Vec cc;
        for (long q : cnt)
            if (q > 0) cc.push_back((double)q);
        double pos = (double)ps / (double)tot + (hz ? y : x);
        if (hz) segs.push_back({pos, (double)x, (double)(x + w - 1), median(cc), (double)L});
        else segs.push_back({pos, (double)y, (double)(y + h - 1), median(cc), (double)L});
    }
    std::stable_sort(segs.begin(), segs.end(), [](const Line& a, const Line& b) { return a[0] < b[0]; });
    if (!merge) return segs;
    Lines out;
    for (const auto& s : segs) {
        bool done = false;
        for (auto& o : out) {
            if (std::fabs(o[0] - s[0]) < 12 && s[1] <= o[2] + 40 && s[2] >= o[1] - 40) {
                double wsum = o[4] + s[4];
                o[0] = (o[0] * o[4] + s[0] * s[4]) / wsum;
                o[3] = (o[3] * o[4] + s[3] * s[4]) / wsum;
                o[1] = std::min(o[1], s[1]); o[2] = std::max(o[2], s[2]); o[4] = wsum;
                done = true;
                break;
            }
        }
        if (!done) out.push_back(s);
    }
    return out;
}

bool touches(const Line& seg, const Lines& perp, double tol, int need = 2) {
    int ok = 0;
    for (double end : {seg[1], seg[2]})
        for (const auto& p : perp)
            if (std::fabs(p[0] - end) < tol && p[1] - tol <= seg[0] && seg[0] <= p[2] + tol) { ++ok; break; }
    return ok >= need;
}

// one_end (app): a segment of >= 45 px touching a perpendicular line at one end also counts - box
// sides whose other edge is printed too faintly to be found (text strokes are shorter)
Lines short_segments(const cv::Mat& ink, bool hz, const Lines& long_segs, const Lines& perp, bool one_end = false) {
    Lines segs = segments(ink, hz, 35, 35), add;
    for (const auto& s : segs) {
        if (s[4] >= 150 && !one_end) continue;   // the long pass has it (unless it leans: app)
        bool dup = false;
        for (const auto& o : long_segs)
            if (std::fabs(o[0] - s[0]) < 12 && s[1] <= o[2] + 10 && s[2] >= o[1] - 10) { dup = true; break; }
        if (dup) continue;
        if (touches(s, perp, 7) || (one_end && s[4] >= 45 && touches(s, perp, 7, 1))) add.push_back(s);
    }
    return add;
}

// app: small open boxes ("┌", a form's discount column) have no side that reaches a found rule, so
// neither short pass takes them, and a box top tilted by a local curl stayed tilted. A horizontal and
// a vertical stroke of >= 45 px each whose ends meet within 10 px form a corner; both count. Horizontal
// ink is widened by 2 px here (a 3-px rule leaning 5 deg breaks into ~35-px runs). Strokes at most
// 9 px thick, widening included: a poster's bold letters ("B", "4") have such corners, printed
// rules are thin. And the horizontal one must continue a found rule of its row across a gap < 80 px
// (the boxes sit in a form's rows; a product's label outline on a flyer does not).
void corner_pairs(const cv::Mat& ink_h, const cv::Mat& ink_v, const Lines& H, const Lines& V, const Lines& Hs, const Lines& Vs,
                  Lines& Hc, Lines& Vc) {
    cv::Mat ih;
    cv::dilate(ink_h, ih, rect(1, 5));
    Lines hs = segments(ih, true, 25, 45, false, 7), vs = segments(ink_v, false, 25, 45, false, 7);   // unjoined: a box top 20 px from a column's rule is its own piece
    auto known = [](const Line& s, const Lines& A, const Lines& B) {
        for (const Lines* L : {&A, &B})
            for (const auto& o : *L)
                if (std::fabs(o[0] - s[0]) < 12 && s[1] <= o[2] + 10 && s[2] >= o[1] - 10) return true;
        return false;
    };
    std::vector<char> uh(hs.size(), 0), uv(vs.size(), 0);
    for (size_t i = 0; i < hs.size(); ++i) {
        if (hs[i][4] > 300 || hs[i][3] > 9 || known(hs[i], H, Hs)) continue;
        bool row = false;   // continues a table rule of its row (a box top in a form's column)
        for (const Lines* L : {&H, &Hs})
            for (const auto& o : *L)
                row |= std::fabs(o[0] - hs[i][0]) <= 8 && std::max(o[1], hs[i][1]) - std::min(o[2], hs[i][2]) < 80;
        if (!row) continue;
        for (size_t j = 0; j < vs.size(); ++j) {
            if (vs[j][4] > 300 || vs[j][3] > 9 || known(vs[j], V, Vs)) continue;
            const Line &h = hs[i], &v = vs[j];
            bool xe = std::fabs(v[0] - h[1]) < 10 || std::fabs(v[0] - h[2]) < 10;
            bool ye = std::fabs(h[0] - v[1]) < 10 || std::fabs(h[0] - v[2]) < 10;
            if (xe && ye) uh[i] = uv[j] = 1;
        }
    }
    for (size_t i = 0; i < hs.size(); ++i) if (uh[i]) Hc.push_back(hs[i]);
    for (size_t j = 0; j < vs.size(); ++j) if (uv[j]) Vc.push_back(vs[j]);
}

Lines merge_gap(Lines L, double tol = 8, double maxgap = 60) {
    std::stable_sort(L.begin(), L.end(), [](const Line& a, const Line& b) { return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]); });
    Lines out;
    for (const auto& s : L) {
        bool done = false;
        for (auto& o : out)
            if (std::fabs(o[0] - s[0]) < tol && s[1] <= o[2] + maxgap && s[2] >= o[1] - maxgap) {
                o[1] = std::min(o[1], s[1]); o[2] = std::max(o[2], s[2]); o[4] = o[2] - o[1];
                done = true;
                break;
            }
        if (!done) out.push_back(s);
    }
    return out;
}

// ------------------------------------------------------------------ tracing (numba kernels)
struct DView {
    const float* p;
    int rows, cols;
    float at(int r, int c) const { return p[(size_t)r * cols + c]; }
};

inline void measure(const DView& D, bool hz, long col, double guess, double w, double thr, long maxrun, long n_c,
                    double& v, double& ww) {
    long lo = pyint(std::max(0.0, guess - w)), hi = pyint(std::min((double)n_c, guess + w + 1));
    v = NaN; ww = NaN;
    if (hi - lo < 3) return;
    auto val = [&](long r) -> double { return hz ? D.at((int)r, (int)col) : D.at((int)col, (int)r); };
    double mx = -1.0;
    for (long r = lo; r < hi; ++r) mx = std::max(mx, val(r));
    if (mx < thr) return;
    double half = thr * 0.5, best_d = 1e18;
    long bs = -1, be = -1, r = lo;
    while (r < hi) {
        if (val(r) > half) {
            long s0 = r;
            while (r < hi && val(r) > half) ++r;
            double d = std::fabs((s0 + r) / 2.0 - guess);
            if (d < best_d) { best_d = d; bs = s0; be = r; }
        } else {
            ++r;
        }
    }
    if (be - bs > maxrun) return;
    double sw = 0, swy = 0;
    for (long q = bs; q < be; ++q) { double x = val(q); sw += x; swy += x * (double)(q - bs); }
    v = bs + swy / sw;
    ww = (double)(be - bs);
}

// warpmap.trace_w2 -> t (integers), c, wd
void trace_w2(const DView& D, double pos, double a, double b, bool hz, std::vector<long>& t, Vec& c, Vec& wd,
              long maxrun, const Vec* prior, double win = 16, double thr = 60, long max_gap = 90) {
    long t0 = pyint(a), t1 = pyint(b);
    t.clear();
    for (long k = t0; k <= t1; ++k) t.push_back(k);
    long n = (long)t.size(), n_c = hz ? D.rows : D.cols;
    c.assign(n, NaN); wd.assign(n, NaN);
    Vec ht(n), hc(n);
    long mid = n / 2;
    for (int direction : {1, -1}) {
        long nh = 0, miss = 0;
        double last = pos, guess;
        long k = direction > 0 ? mid : mid - 1;
        while (0 <= k && k < n) {
            if (miss > max_gap) break;
            if (miss == 0 || nh < 8) {
                guess = last;
            } else {
                long m0 = std::max(0L, nh - 25), m = nh - m0;
                double sx = 0, sy = 0, sxx = 0, sxy = 0;
                for (long i = m0; i < nh; ++i) { sx += ht[i]; sy += hc[i]; sxx += ht[i] * ht[i]; sxy += ht[i] * hc[i]; }
                double den = m * sxx - sx * sx;
                if (den != 0.0) {
                    double sl = (m * sxy - sx * sy) / den;
                    guess = (sy - sl * sx) / m + sl * t[k];
                } else {
                    guess = sy / m;
                }
            }
            double w = std::min(30.0, win + 0.3 * miss), v, ww;
            measure(D, hz, t[k], guess, w, thr, maxrun, n_c, v, ww);
            bool ok = !std::isnan(v) && ((miss == 0 && std::fabs(v - last) < 4) || std::fabs(v - guess) < 3 + 0.25 * miss);
            if (!ok && prior && miss > 2) {
                double g2 = (*prior)[k], v2, ww2;
                measure(D, hz, t[k], g2, 8.0, thr, maxrun, n_c, v2, ww2);
                if (!std::isnan(v2) && std::fabs(v2 - g2) < 3) { v = v2; ww = ww2; ok = true; }
            }
            if (ok) {
                c[k] = v; wd[k] = ww;
                ht[nh] = (double)t[k]; hc[nh] = v; ++nh;
                last = v; miss = 0;
            } else {
                ++miss;
            }
            k += direction;
        }
    }
}

// warpmap.chain_trace
std::vector<char> chain_trace(const std::vector<long>& t, const Vec& c, const Vec& wd, const Vec* prior) {
    size_t n = t.size();
    std::vector<char> ok(n);
    size_t nok = 0;
    for (size_t i = 0; i < n; ++i) nok += (ok[i] = !std::isnan(c[i]));
    if (nok < 20) return ok;
    Vec w;
    for (size_t i = 0; i < n; ++i)
        if (ok[i]) w.push_back(wd[i]);
    double mw = median(w);
    for (size_t i = 0; i < n; ++i) ok[i] = ok[i] && (wd[i] >= 0.6 * mw) && (wd[i] <= 1.7 * mw + 1);
    std::vector<size_t> idx;
    for (size_t i = 0; i < n; ++i)
        if (ok[i]) idx.push_back(i);
    if (idx.size() < 20) return ok;
    std::vector<std::vector<size_t>> pieces(1);
    pieces[0].push_back(idx[0]);
    for (size_t k = 1; k < idx.size(); ++k) {
        double gap = (double)(idx[k] - idx[k - 1]);
        if (gap > 3 || std::fabs(c[idx[k]] - c[idx[k - 1]]) > 2 + 0.2 * gap) pieces.emplace_back();
        pieces.back().push_back(idx[k]);
    }
    size_t seed = 0;   // longest piece (first on ties)
    long best = LONG_MIN;
    for (size_t j = 0; j < pieces.size(); ++j) {
        long len = t[pieces[j].back()] - t[pieces[j].front()];
        if (len > best) { best = len; seed = j; }
    }
    std::vector<char> keep(n, 0);
    for (size_t q : pieces[seed]) keep[q] = 1;
    for (int direction : {1, -1}) {
        long j = (long)seed + direction;
        while (0 <= j && j < (long)pieces.size()) {
            std::vector<size_t> ch;
            for (size_t i = 0; i < n; ++i)
                if (keep[i]) ch.push_back(i);
            std::vector<size_t> ref;
            double tp, cp, gap;
            const auto& pj = pieces[j];
            if (direction > 0) {
                for (size_t q : ch) if ((double)t[q] >= (double)t[ch.back()] - 40) ref.push_back(q);
                tp = (double)t[pj.front()]; cp = c[pj.front()];
                gap = tp - (double)t[ch.back()];
            } else {
                for (size_t q : ch) if ((double)t[q] <= (double)t[ch.front()] + 40) ref.push_back(q);
                tp = (double)t[pj.back()]; cp = c[pj.back()];
                gap = (double)t[ch.front()] - tp;
            }
            Vec pf;
            Vec rt, rc;
            for (size_t q : ref) { rt.push_back((double)t[q]); rc.push_back(c[q]); }
            if (ref.size() >= 5) pf = polyfit(rt, rc, 1);
            else pf = {0.0, np_mean(rc)};
            bool okp = std::fabs(cp - polyval(pf, tp)) < 2 + 0.1 * gap;
            if (!okp && prior) {
                Vec d;
                for (size_t q : pj) d.push_back(std::fabs(c[q] - (*prior)[q]));
                okp = median(d) < 2.5;
            }
            if (okp)
                for (size_t q : pj) keep[q] = 1;
            j += direction;
        }
    }
    return keep;
}

// free_ext (app): how far a trace may run past an end with no perpendicular rule to meet; there it
// only walks into what lies beside the line (a chart's dotted gridline into the text next to it)
void trace_bounds(const Line& L, const Lines& perp, double lo, double hi, double& a2, double& b2, double ext = 200, double pad = 6,
                  double free_ext = 200) {
    double p = L[0], a = L[1], b = L[2];
    bool hl = false, hr = false;
    double ml = -1e300, mr = 1e300;
    for (const auto& P : perp)
        if (P[1] - 15 <= p && p <= P[2] + 15) {
            if (P[0] <= a + 10) { hl = true; ml = std::max(ml, P[0]); }
            if (P[0] >= b - 10) { hr = true; mr = std::min(mr, P[0]); }
        }
    a2 = std::max({lo, a - (hl ? ext : free_ext), hl ? ml - pad : -1e9});
    b2 = std::min({hi, b + (hr ? ext : free_ext), hr ? mr + pad : 1e9});
}

}  // namespace

Lines clean_segments(const cv::Mat& ink, bool hz, int klen, int minlen) { return segments(ink, hz, klen, minlen); }
Lines clean_short_segments(const cv::Mat& ink, bool hz, const Lines& long_segs, const Lines& perp) {
    return short_segments(ink, hz, long_segs, perp);
}

float Field::sample(double x, double y) const {
    double gy = y / g, gx = x / g;
    gy = std::min(std::max(gy, 0.0), (double)(gh - 1));
    gx = std::min(std::max(gx, 0.0), (double)(gw - 1));
    long y0 = std::min((long)std::floor(gy), (long)gh - 2), x0 = std::min((long)std::floor(gx), (long)gw - 2);
    y0 = std::max(y0, 0L); x0 = std::max(x0, 0L);
    double ty = gy - y0, tx = gx - x0;
    const double* r0 = &v[(size_t)y0 * gw], *r1 = &v[(size_t)std::min(y0 + 1, (long)gh - 1) * gw];
    long x1 = std::min(x0 + 1, (long)gw - 1);
    double a = (1 - ty) * ((1 - tx) * r0[x0] + tx * r0[x1]) + ty * ((1 - tx) * r1[x0] + tx * r1[x1]);
    return (float)a;
}

cv::Mat darkness_fast(const cv::Mat& img) {
    cv::Mat g;
    if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    else g = img;
    int h = g.rows, w = g.cols;
    cv::Mat sm, bg;
    cv::resize(g, sm, cv::Size(w / 4, h / 4), 0, 0, cv::INTER_AREA);
    cv::dilate(sm, sm, cv::Mat::ones(3, 3, CV_8U));
    cv::medianBlur(sm, bg, 11);
    cv::resize(bg, bg, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::Mat D(h, w, CV_32F);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            D.at<float>(y, x) = std::min(std::max((float)bg.at<uint8_t>(y, x) - (float)g.at<uint8_t>(y, x), 0.f), 255.f);
    return D;
}

void get_lines(const cv::Mat& D, Lines& H, Lines& V, bool faint) {
    cv::Mat ink = D > 45.f;   // 0 / 255
    cv::Mat ink_h = ink, ink_v = ink;
    if (faint) {   // app: faint printed rules too (faint_rules)
        ink_h = ink | faint_rules(D, true, 100);
        ink_v = ink | faint_rules(D, false, 100);
        // ... and leaning vertical ones: a 3-px rule leaning 5 deg leaves a 1-px column after
        // ~30 px, so the short box sides where a page curls (the receipt's totals boxes) were never
        // seen. Widened by 2 px across the line, rules up to ~7 deg give long runs; the tracer then
        // follows the line itself (the widening is symmetric, so positions are unchanged). Not
        // for horizontal rules: there it joins the letters of a text row into false rules.
        cv::dilate(ink_v, ink_v, rect(5, 1));
    }
    H = segments(ink_h, true);
    V = segments(ink_v, false);
    Lines Vs = short_segments(ink_v, false, V, H, faint);
    Lines HV = V;
    HV.insert(HV.end(), Vs.begin(), Vs.end());
    Lines Hs = short_segments(ink_h, true, H, HV, faint);
    Lines Hc, Vc;
    if (faint) corner_pairs(ink_h, ink_v, H, V, Hs, Vs, Hc, Vc);
    Lines h2 = H; h2.insert(h2.end(), Hs.begin(), Hs.end());
    Lines v2 = V; v2.insert(v2.end(), Vs.begin(), Vs.end());
    H = merge_gap(h2);
    V = merge_gap(v2);
    // corner pieces stay their own lines: merged with a column's rule beside them, the one trace
    // follows the rule and drops the tilted piece (share_ideals still gives them one height)
    H.insert(H.end(), Hc.begin(), Hc.end());
    V.insert(V.end(), Vc.begin(), Vc.end());
    auto clip = [faint](Lines& A, const Lines& B, double tol) {
        for (auto& L : A) {
            bool any = false;
            double mn = 1e300, mx = -1e300;
            int cnt = 0;
            for (const auto& P : B)
                if (P[1] - 15 <= L[0] && L[0] <= P[2] + 15 && P[4] > 80) { mn = std::min(mn, P[0]); mx = std::max(mx, P[0]); ++cnt; any = true; }
            if (any && cnt >= 2) {
                double a = std::max(L[1], mn - tol), b = std::min(L[2], mx + tol);
                // app: a box side whose far edge is too faint to be found keeps its length
                if (faint && b - a < 0.5 * (L[2] - L[1])) continue;
                L[1] = a;
                L[2] = b;
                L[4] = L[2] - L[1];
            }
        }
    };
    clip(H, V, 20);
    clip(V, H, 20);
    auto snap = [](Lines& A, const Lines& B, double tol) {
        for (auto& L : A) {
            Vec cr;
            for (const auto& P : B)
                if (P[1] - 15 <= L[0] && L[0] <= P[2] + 15) cr.push_back(P[0]);
            for (int end : {1, 2}) {
                bool found = false;
                double best = 0;
                for (double c : cr)
                    if (std::fabs(c - L[end]) < tol && (!found || std::fabs(c - L[end]) < std::fabs(best - L[end]))) { best = c; found = true; }
                if (found) L[end] = end == 1 ? std::min(L[end], best) : std::max(L[end], best);
            }
        }
    };
    snap(V, H, 70);
    snap(H, V, 50);
}

namespace {

const int BIN = 4;

// warpmap.collect
// Pieces of one ruled line (split at table rules, gaps) traced separately each get their own
// ideal position, so a slightly tilted rule becomes a staircase. share_rows (app, UVDoc pages):
// pieces that continue each other (ideals within 6 px, gap along the line < 80 px) share one
// ideal, the sample-weighted mean of theirs (RDGR's "same row, same height" constraint).
void share_ideals(std::vector<std::vector<std::array<double, 3>>>& per) {
    size_t n = per.size();
    std::vector<size_t> parent(n);
    for (size_t i = 0; i < n; ++i) parent[i] = i;
    std::function<size_t(size_t)> find = [&](size_t i) { return parent[i] == i ? i : parent[i] = find(parent[i]); };
    std::vector<double> ideal(n, 0), t0(n, 0), t1(n, 0), c0(n, 0), c1(n, 0);   // c0 / c1: traced position at t0 / t1
    for (size_t i = 0; i < n; ++i) {
        if (per[i].empty()) continue;
        ideal[i] = per[i][0][1];
        t0[i] = 1e300; t1[i] = -1e300;
        for (const auto& r : per[i]) {
            if (r[0] < t0[i]) { t0[i] = r[0]; c0[i] = r[1] + r[2]; }
            if (r[0] > t1[i]) { t1[i] = r[0]; c1[i] = r[1] + r[2]; }
        }
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            if (per[i].empty() || per[j].empty()) continue;
            double gap = std::max(t0[i], t0[j]) - std::min(t1[i], t1[j]);
            if (std::fabs(ideal[i] - ideal[j]) <= 6 && gap < 80) { parent[find(i)] = find(j); continue; }
            // pieces that physically continue each other (their traces meet within 5 px across a
            // gap < 40 px) are one line even when a curl moved one piece's ideal (box sides
            // stacked down a curled page edge)
            if (std::fabs(ideal[i] - ideal[j]) <= 20 && gap < 40 && gap > -10) {
                double meet = t1[i] <= t0[j] + 10 ? std::fabs(c1[i] - c0[j]) : std::fabs(c1[j] - c0[i]);
                if (meet <= 5) parent[find(i)] = find(j);
            }
        }
    std::vector<double> sum(n, 0), cnt(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (!per[i].empty()) { sum[find(i)] += ideal[i] * per[i].size(); cnt[find(i)] += per[i].size(); }
    for (size_t i = 0; i < n; ++i) {
        if (per[i].empty()) continue;
        double g = sum[find(i)] / cnt[find(i)];
        for (auto& r : per[i]) { double c = r[1] + r[2]; r[1] = g; r[2] = c - g; }
    }
}

Samples collect(const DView& D, const Lines& H, const Lines& V, const Field* prior_dx, const Field* prior_dy, int threads,
                bool share_rows = false, bool faint = false) {
    Samples out;
    for (int pass = 0; pass < 2; ++pass) {
        bool hz = pass == 0;
        const Lines& lines = hz ? H : V;
        const Lines& perp = hz ? V : H;
        auto& rows_out = hz ? out.h : out.v;
        if (lines.empty()) continue;
        double lo = 1e300, hi = -1e300;
        for (const auto& L : lines) { lo = std::min(lo, L[1]); hi = std::max(hi, L[2]); }
        std::vector<std::vector<std::array<double, 3>>> per(lines.size());
        std::atomic<int> bowed{0};
        auto one = [&](size_t li) {
            const Line& L = lines[li];
            double p = L[0], a = L[1], b = L[2], th = L[3];
            double a2, b2;
            trace_bounds(L, perp, lo, hi, a2, b2, 200, 6, faint ? 40 : 200);
            std::vector<long> t;
            Vec c, wd, prior_v;
            const Vec* prior = nullptr;
            long t0 = pyint(a2), t1 = pyint(b2);
            if (prior_dy) {
                prior_v.resize(t1 - t0 + 1 > 0 ? t1 - t0 + 1 : 0);
                for (long k = t0; k <= t1; ++k)
                    prior_v[k - t0] = hz ? p + (double)prior_dy->sample((double)k, p) : p + (double)prior_dx->sample(p, (double)k);
                prior = &prior_v;
            }
            long maxrun = pyround(std::max(7.0, 1.5 * th));
            // faint (app): the tracing threshold from the line's own darkness (0.75 x its median
            // peak across +-3 px, 25 .. 60), so a pale rule is followed; dark rules keep 60
            double thr = 60;
            if (faint) {
                Vec pk;
                long n_c = hz ? D.rows : D.cols, n_a = hz ? D.cols : D.rows;
                for (long k = std::max(0L, pyint(a)); k <= std::min(n_a - 1, pyint(b)); k += 4) {
                    float m = 0;
                    for (long q = std::max(0L, pyint(p) - 3); q <= std::min(n_c - 1, pyint(p) + 3); ++q)
                        m = std::max(m, hz ? D.at((int)q, (int)k) : D.at((int)k, (int)q));
                    pk.push_back(m);
                }
                if (!pk.empty()) thr = std::min(60.0, std::max(25.0, 0.75 * median(pk)));
            }
            trace_w2(D, p, a2, b2, hz, t, c, wd, maxrun, prior, 16, thr);
            std::vector<char> ok = chain_trace(t, c, wd, prior);
            size_t n = t.size();
            auto count = [&] { size_t s = 0; for (char q : ok) s += q; return s; };
            if (count() < 20) return;
            for (const auto& P : perp)
                if (P[1] - 12 <= p && p <= P[2] + 12)
                    for (size_t i = 0; i < n; ++i) ok[i] = ok[i] && std::fabs((double)t[i] - P[0]) > 10;
            if (count() < 20) return;
            Vec tok, cok;
            for (size_t i = 0; i < n; ++i)
                if (ok[i]) { tok.push_back((double)t[i]); cok.push_back(c[i]); }
            Vec cf = c;
            for (size_t i = 0; i < n; ++i)
                if (!ok[i]) cf[i] = interp((double)t[i], tok, cok);
            Vec mf = median_filter1d_nearest(cf, 31);
            for (size_t i = 0; i < n; ++i) ok[i] = ok[i] && std::fabs(c[i] - mf[i]) < 1.5;
            Vec sel;
            for (size_t i = 0; i < n; ++i)
                if (ok[i]) sel.push_back(c[i]);
            double ideal = median(sel);
            for (size_t i = 0; i < n; ++i) ok[i] = ok[i] && std::fabs(c[i] - ideal) < 40;
            if (count() < 20) return;
            sel.clear();
            for (size_t i = 0; i < n; ++i)
                if (ok[i]) sel.push_back(c[i]);
            ideal = median(sel);
            Vec tv, cv;
            for (size_t i = 0; i < n; ++i)
                if (ok[i]) { tv.push_back((double)t[i]); cv.push_back(c[i]); }
            // share_rows (app, neural pages): the stage-1 page is already flat, so its ruled lines
            // are nearly straight; a trace that bows (p95 distance from its straight fit above
            // 3 px + 1% of its length) is a curve - an ellipse's top, a stamp, a rule traced into
            // text - and straightening it would bend everything around it
            if (share_rows && tv.size() >= 8) {
                Vec pf = polyfit(tv, cv, 1);
                Vec dev;
                for (size_t i = 0; i < tv.size(); ++i) dev.push_back(std::fabs(cv[i] - polyval(pf, tv[i])));
                std::sort(dev.begin(), dev.end());
                if (dev[(size_t)(0.95 * (dev.size() - 1))] > 3 + 0.01 * (tv.back() - tv.front())) { ++bowed; return; }
            }
            // 4-px bins of consecutive samples
            Vec tt, cc;
            {
                size_t i = 0;
                while (i < tv.size()) {
                    long bin = (long)np_floordiv(tv[i], BIN);
                    size_t j = i;
                    double st = 0, sc = 0;
                    while (j < tv.size() && (long)np_floordiv(tv[j], BIN) == bin) { st += tv[j]; sc += cv[j]; ++j; }
                    tt.push_back(st / (double)(j - i)); cc.push_back(sc / (double)(j - i));
                    i = j;
                }
            }
            // bridges at junctions and line ends
            Vec bt, bc;
            for (const auto& P : perp) {
                if (!(P[1] - 12 <= p && p <= P[2] + 12 && a - 15 <= P[0] && P[0] <= b + 15)) continue;
                double e = P[0];
                if (e < tv.front() - 2 || e > tv.back() + 2) {
                    bool left = e < tv.front();
                    Vec st_, sc_;
                    for (size_t i = 0; i < tv.size(); ++i)
                        if (left ? tv[i] <= tv.front() + 40 : tv[i] >= tv.back() - 40) { st_.push_back(tv[i]); sc_.push_back(cv[i]); }
                    if (st_.size() < 8 || std::fabs(e - (left ? tv.front() : tv.back())) > 30) continue;
                    Vec pf = polyfit(st_, sc_, 1);
                    double z0 = left ? std::min(e, tv.front()) : tv.back() + 1, z1 = left ? tv.front() : e + 1;
                    long nz = (long)std::ceil((z1 - z0) / 1.0);
                    for (long k = 0; k < nz; ++k) { double z = z0 + k * 1.0; bt.push_back(z); bc.push_back(polyval(pf, z)); }
                } else {
                    double z0 = e - 10;
                    long nz = (long)std::ceil(((e + 11) - z0) / 1.0);
                    for (long k = 0; k < nz; ++k) { double z = z0 + k * 1.0; bt.push_back(z); bc.push_back(interp(z, tv, cv)); }
                }
            }
            for (size_t k = 0; k < bt.size(); k += 3) { tt.push_back(bt[k]); cc.push_back(bc[k]); }
            auto& rows = per[li];
            for (size_t k = 0; k < tt.size(); ++k) rows.push_back({tt[k], ideal, cc[k] - ideal});
        };
        int nt = std::max(1, threads);
        std::vector<std::thread> ts;
        for (int k = 0; k < nt; ++k)
            ts.emplace_back([&, k] { for (size_t li = k; li < lines.size(); li += nt) one(li); });
        for (auto& q : ts) q.join();
        if (share_rows) share_ideals(per);
        out.n_bowed += bowed;
        for (auto& r : per) rows_out.insert(rows_out.end(), r.begin(), r.end());
    }
    return out;
}

// warpmap.text_rows
std::vector<std::array<double, 3>> text_rows(const cv::Mat& Dm, const Lines& H, const Lines& V, int& nrows, double step = 6) {
    int h = Dm.rows, w = Dm.cols;
    cv::Mat ink = (Dm > 70.f) / 255;
    cv::Mat lines = cv::Mat::zeros(h, w, CV_8U);
    auto fill = [&](long r0, long r1, long c0, long c1) {
        r0 = std::max(0L, r0); c0 = std::max(0L, c0);
        r1 = std::min((long)h, r1); c1 = std::min((long)w, c1);
        if (r1 > r0 && c1 > c0) lines(cv::Range((int)r0, (int)r1), cv::Range((int)c0, (int)c1)).setTo(1);
    };
    for (const auto& L : H) fill(pyint(std::max(0.0, L[0] - L[3] / 2 - 4)), pyint(L[0] + L[3] / 2 + 5), pyint(L[1]), pyint(L[2]) + 1);
    for (const auto& L : V) fill(pyint(L[1]), pyint(L[2]) + 1, pyint(std::max(0.0, L[0] - L[3] / 2 - 4)), pyint(L[0] + L[3] / 2 + 5));
    ink.setTo(0, lines > 0);
    cv::Mat words;
    cv::morphologyEx(ink, words, cv::MORPH_CLOSE, rect(31, 1));
    cv::morphologyEx(words, words, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    words.setTo(0, lines > 0);
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(words, lab, st, cen, 8);
    std::vector<std::array<double, 3>> rows;
    nrows = 0;
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), ww = st.at<int>(i, 2), hh = st.at<int>(i, 3);
        if (ww < 140 || !(10 <= hh && hh <= 55) || ww < 5 * hh) continue;
        Vec xs, cs;
        for (int cx = 0; cx < ww - 16; cx += 8) {
            long ms = 0, my = 0;
            for (int r = 0; r < hh; ++r)
                for (int c = cx; c < cx + 16 && c < ww; ++c)
                    if (lab.at<int>(y + r, x + c) == i && ink.at<uint8_t>(y + r, x + c) > 0) { ++ms; my += r; }
            if (ms < 12) continue;
            xs.push_back(x + cx + 8);
            cs.push_back(y + (double)my / (double)ms);
        }
        if (xs.size() < 10) continue;
        int deg = ww < 500 ? 1 : 2;
        std::vector<char> keep(xs.size(), 1);
        Vec p;
        for (int it = 0; it < 3; ++it) {
            Vec kx, kc;
            for (size_t k = 0; k < xs.size(); ++k)
                if (keep[k]) { kx.push_back(xs[k]); kc.push_back(cs[k]); }
            p = polyfit(kx, kc, deg);
            Vec r(xs.size());
            for (size_t k = 0; k < xs.size(); ++k) r[k] = std::fabs(polyval(p, xs[k]) - cs[k]);
            double thr = std::max(1.5, 2.0 * median(r));
            for (size_t k = 0; k < xs.size(); ++k) keep[k] = r[k] < thr;
        }
        long nt = (long)std::ceil(((xs.back() + 1) - xs.front()) / step);
        Vec tt(nt), yc(nt);
        for (long k = 0; k < nt; ++k) { tt[k] = xs.front() + k * step; yc[k] = polyval(p, tt[k]); }
        double ideal = np_mean(yc), mxd = 0;
        for (double v : yc) mxd = std::max(mxd, std::fabs(v - ideal));
        if (mxd > 15) continue;
        for (long k = 0; k < nt; ++k) rows.push_back({tt[k], ideal, yc[k] - ideal});
        ++nrows;
    }
    return rows;
}

// App: the aligned edges of text columns as vertical lines. A justified (or left / right aligned)
// column's edge is straight on the page, and on a magazine page without vertical rules it is the
// only vertical cue (a bowed column edge was left as it was). Text-line blobs as in text_rows; their
// left (and right) ends are chained down the page - each step at most 6 px across, at most ~3.5
// median line heights down, so an indented first line or a short last line is skipped - and a chain of
// >= 8 lines over >= 250 px whose ends sit within 2 px (rms) of a smooth curve (a polynomial of
// degree 2 / 3 / 4 for chains under 500 / 900 / over 900 px - a page's bow is no parabola; ragged
// edges fail) gives samples (along y, ideal x, disp) every 6 px, like a traced vertical rule.
std::vector<std::array<double, 3>> text_margins(const cv::Mat& Dm, const Lines& H, const Lines& V, int& nchains, double step = 6) {
    int h = Dm.rows, w = Dm.cols;
    cv::Mat ink = (Dm > 70.f) / 255;
    cv::Mat lines = cv::Mat::zeros(h, w, CV_8U);
    auto fill = [&](long r0, long r1, long c0, long c1) {
        r0 = std::max(0L, r0); c0 = std::max(0L, c0);
        r1 = std::min((long)h, r1); c1 = std::min((long)w, c1);
        if (r1 > r0 && c1 > c0) lines(cv::Range((int)r0, (int)r1), cv::Range((int)c0, (int)c1)).setTo(1);
    };
    for (const auto& L : H) fill(pyint(std::max(0.0, L[0] - L[3] / 2 - 4)), pyint(L[0] + L[3] / 2 + 5), pyint(L[1]), pyint(L[2]) + 1);
    for (const auto& L : V) fill(pyint(L[1]), pyint(L[2]) + 1, pyint(std::max(0.0, L[0] - L[3] / 2 - 4)), pyint(L[0] + L[3] / 2 + 5));
    ink.setTo(0, lines > 0);
    cv::Mat words;
    cv::morphologyEx(ink, words, cv::MORPH_CLOSE, rect(31, 1));
    cv::morphologyEx(words, words, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    words.setTo(0, lines > 0);
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(words, lab, st, cen, 8);
    struct Blob { double x0, x1, y, hh; };
    std::vector<Blob> B;
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), ww = st.at<int>(i, 2), hh = st.at<int>(i, 3);
        if (ww < 140 || !(10 <= hh && hh <= 55) || ww < 5 * hh) continue;
        B.push_back({(double)x, (double)(x + ww - 1), cen.at<double>(i, 1), (double)hh});
    }
    std::sort(B.begin(), B.end(), [](const Blob& a, const Blob& b) { return a.y < b.y; });
    Vec hs;
    for (const auto& b : B) hs.push_back(b.hh);
    const double hh_med = hs.empty() ? 20 : median(hs);   // line height (a short last line's blob is lower)
    std::vector<std::array<double, 3>> out;
    nchains = 0;
    for (int side = 0; side < 2; ++side) {
        auto e = [&](size_t k) { return side == 0 ? B[k].x0 : B[k].x1; };
        std::vector<char> used(B.size(), 0);
        for (size_t s0 = 0; s0 < B.size(); ++s0) {
            if (used[s0]) continue;
            std::vector<size_t> ch{s0};
            size_t last = s0;
            for (size_t k = s0 + 1; k < B.size(); ++k) {
                if (B[k].y - B[last].y > 3.5 * hh_med + 4) break;
                if (used[k] || B[k].y <= B[last].y + 0.5 * hh_med) continue;
                if (std::fabs(e(k) - e(last)) <= 6) { ch.push_back(k); last = k; }
            }
            if (ch.size() < 8 || B[ch.back()].y - B[ch.front()].y < 250) continue;
            Vec ys, xs;
            for (size_t k : ch) { ys.push_back(B[k].y); xs.push_back(e(k)); }
            const double y_mid = 0.5 * (ys.front() + ys.back()), y_sc = std::max(1.0, 0.5 * (ys.back() - ys.front()));
            const double span = ys.back() - ys.front();
            const int deg = span > 900 ? 4 : span > 500 ? 3 : 2;
            auto U = [&](double y) { return (y - y_mid) / y_sc; };   // fit in [-1, 1]
            std::vector<char> keep(ys.size(), 1);
            Vec p;
            double rms = 0;
            for (int it = 0; it < 3; ++it) {
                Vec ky, kx;
                for (size_t k = 0; k < ys.size(); ++k)
                    if (keep[k]) { ky.push_back(U(ys[k])); kx.push_back(xs[k]); }
                if ((int)ky.size() < std::max(8, 3 * (deg + 1))) { p.clear(); break; }
                p = polyfit(ky, kx, deg);
                double ss = 0; int m = 0;
                for (size_t k = 0; k < ys.size(); ++k) {
                    double r = std::fabs(polyval(p, U(ys[k])) - xs[k]);
                    keep[k] = r <= 4;
                    if (keep[k]) { ss += r * r; ++m; }
                }
                rms = m ? std::sqrt(ss / m) : 1e9;
            }
            if (p.empty() || rms > 2) continue;
            Vec ky;
            for (size_t k = 0; k < ys.size(); ++k)
                if (keep[k]) ky.push_back(ys[k]);
            if (ky.size() < 8 || ky.back() - ky.front() < 250) continue;
            for (size_t k : ch) used[k] = 1;
            long nt = (long)std::ceil((ky.back() + 1 - ky.front()) / step);
            Vec tt(nt), xc(nt);
            for (long k = 0; k < nt; ++k) { tt[k] = ky.front() + k * step; xc[k] = polyval(p, U(tt[k])); }
            double ideal = np_mean(xc), mxd = 0;
            for (double v : xc) mxd = std::max(mxd, std::fabs(v - ideal));
            if (mxd > 40) continue;
            for (long k = 0; k < nt; ++k) out.push_back({tt[k], ideal, xc[k] - ideal});
            ++nchains;
        }
    }
    return out;
}

// App, neural pages: long straight boundaries between colour areas (a flyer's cells, a can's side)
// as lines. The darkness map sees only dark ink, so on a page of colour blocks a cell boundary the
// net left wavy had no constraint at all. The page is closed (grey 9x9, per channel: text and rules
// go, colour areas stay), and the step across each pixel is the Lab distance of the 8-px means on
// either side (x 2.5; a photo's soft edge is ~10 px wide). Steps above 30 make segments as ink does (long
// pass only, widened 2 px across for gentle waves); a segment beside a found rule is left out, as
// are the outer 25 px (the paper's own edge). Each is traced like a rule, with the line's own
// threshold. A trace must be near the axis (chord within 2 deg: content tilted on the paper - a
// leaning bottle - is no residual warp) and straight like a rule (3 px + 1 % of its length), except
// that one over 500 px may bow by 4 % of its length: that long a step is a cell or column boundary,
// and its wave is what the net left.
void colour_edges(const cv::Mat& img, const Lines& H, const Lines& V, std::vector<std::array<double, 3>>& eh,
                  std::vector<std::array<double, 3>>& ev, Lines& EH, Lines& EV, int& nbowed) {
    const int h = img.rows, w = img.cols, border = 25;
    cv::Mat cl, lab;
    cv::morphologyEx(img, cl, cv::MORPH_CLOSE, cv::Mat::ones(9, 9, CV_8U));
    cv::cvtColor(cl, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32FC3);
    for (int pass = 0; pass < 2; ++pass) {
        const bool hz = pass == 0;   // horizontal edges: steps along y
        cv::Mat B;
        cv::blur(lab, B, hz ? cv::Size(5, 8) : cv::Size(8, 5));
        cv::Mat S(h, w, CV_32F, cv::Scalar(0));
        for (int y = border; y < h - border; ++y)
            for (int x = border; x < w - border; ++x) {
                const cv::Vec3f &q1 = hz ? B.at<cv::Vec3f>(y + 5, x) : B.at<cv::Vec3f>(y, x + 5),
                                &q0 = hz ? B.at<cv::Vec3f>(y - 5, x) : B.at<cv::Vec3f>(y, x - 5);
                cv::Vec3f d = q1 - q0;
                S.at<float>(y, x) = std::min(255.f, 2.5f * std::sqrt(d.dot(d)));
            }
        cv::Mat mask = S > 30.f;
        cv::dilate(mask, mask, hz ? rect(1, 5) : rect(5, 1));
        // pieces unjoined: a column boundary is a step only where the cells on either side differ,
        // and one trace from the middle of the whole column stops at the first such gap
        Lines segs = segments(mask, hz, 100, 150, false, 7);
        const Lines& ink = hz ? H : V;
        Lines keep;
        for (const auto& s : segs) {
            bool dup = false;
            for (const auto& o : ink)
                dup |= std::fabs(o[0] - s[0]) < 15 && std::min(o[2], s[2]) - std::max(o[1], s[1]) > 0.5 * (s[2] - s[1]);
            if (!dup) keep.push_back(s);
        }
        DView D{S.ptr<float>(), S.rows, S.cols};
        const long n_a = hz ? w : h, n_c = hz ? h : w;
        struct Piece { Vec t, c; double t0, t1, c0, c1, med, wd; };
        std::vector<Piece> pieces;
        for (size_t li = 0; li < keep.size(); ++li) {
            const Line& L = keep[li];
            double p = L[0], a = L[1], b = L[2];
            Vec pk;
            for (long k = std::max(0L, pyint(a)); k <= std::min(n_a - 1, pyint(b)); k += 4) {
                float m = 0;
                for (long q = std::max(0L, pyint(p) - 3); q <= std::min(n_c - 1, pyint(p) + 3); ++q)
                    m = std::max(m, hz ? D.at((int)q, (int)k) : D.at((int)k, (int)q));
                pk.push_back(m);
            }
            double thr = pk.empty() ? 60 : std::min(60.0, std::max(25.0, 0.75 * median(pk)));
            std::vector<long> t;
            Vec c, wd;
            trace_w2(D, p, std::max(0.0, a - 40), std::min((double)n_a - 1, b + 40), hz, t, c, wd, 25, nullptr, 16, thr);   // a step's response is ~16 px wide however thin the mask
            std::vector<char> ok = chain_trace(t, c, wd, nullptr);
            Vec tv, cv;
            for (size_t i = 0; i < t.size(); ++i)
                if (ok[i]) { tv.push_back((double)t[i]); cv.push_back(c[i]); }
            if (tv.size() < 20) continue;
            Vec mf = median_filter1d_nearest(cv, 31);
            // where the darkness map found the same boundary as a line, that line has it: two traces
            // of one edge a few px apart tear the page between them
            auto inked = [&](double tq, double cq) {
                for (const auto& o : ink)
                    if (std::fabs(o[0] - cq) < 15 && o[1] - 20 <= tq && tq <= o[2] + 20) return true;
                return false;
            };
            Piece P;
            for (size_t i = 0; i < tv.size(); ++i)
                if (std::fabs(cv[i] - mf[i]) < 1.5 && !inked(tv[i], cv[i])) { P.t.push_back(tv[i]); P.c.push_back(cv[i]); }
            if (P.t.size() < 20 || P.t.back() - P.t.front() < 60) continue;
            P.t0 = P.t.front(); P.t1 = P.t.back(); P.c0 = P.c.front(); P.c1 = P.c.back();
            P.med = median(P.c); P.wd = L[3];
            pieces.push_back(std::move(P));
        }
        // a boundary's pieces (where the cells on either side differ) chain into one line when they
        // continue each other: medians within 20 px, a gap < 100 px, facing ends within 12 px + 10 %
        // of the gap. The tests apply to the chain - a piece through the wave alone leans or bows.
        const size_t np_ = pieces.size();
        std::vector<size_t> parent(np_);
        for (size_t i = 0; i < np_; ++i) parent[i] = i;
        std::function<size_t(size_t)> find = [&](size_t i) { return parent[i] == i ? i : parent[i] = find(parent[i]); };
        for (size_t i = 0; i < np_; ++i)
            for (size_t j = i + 1; j < np_; ++j) {
                const Piece &A = pieces[i], &Bq = pieces[j];
                if (std::fabs(A.med - Bq.med) > 20) continue;
                const Piece &lo = A.t0 <= Bq.t0 ? A : Bq, &hi = A.t0 <= Bq.t0 ? Bq : A;
                double gap = hi.t0 - lo.t1;
                bool link;
                if (gap < 0) {   // overlapping: within 6 px of each other where both run
                    double s = 0;
                    int m = 0;
                    for (size_t k = 0; k < hi.t.size() && hi.t[k] <= lo.t1; k += 4) { s += std::fabs(hi.c[k] - interp(hi.t[k], lo.t, lo.c)); ++m; }
                    link = m > 0 && s / m <= 6;
                } else {
                    link = gap < 100 && std::fabs(hi.c0 - lo.c1) <= 12 + 0.1 * gap;
                }
                if (link) parent[find(i)] = find(j);
            }
        std::map<size_t, std::vector<size_t>> chains;
        for (size_t i = 0; i < np_; ++i) chains[find(i)].push_back(i);
        for (const auto& kv : chains) {
            std::vector<std::pair<double, double>> s;
            double wd = 0;
            for (size_t i : kv.second) {
                for (size_t k = 0; k < pieces[i].t.size(); ++k) s.push_back({pieces[i].t[k], pieces[i].c[k]});
                wd = std::max(wd, pieces[i].wd);
            }
            std::sort(s.begin(), s.end());
            Vec tv2, cv2;
            for (const auto& q : s) { tv2.push_back(q.first); cv2.push_back(q.second); }
            const double len = tv2.back() - tv2.front();
            if (tv2.size() < 40 || len < 150) continue;
            Vec pf = polyfit(tv2, cv2, 1), dev;
            for (size_t i = 0; i < tv2.size(); ++i) dev.push_back(std::fabs(cv2[i] - polyval(pf, tv2[i])));
            std::sort(dev.begin(), dev.end());
            const double p95 = dev[(size_t)(0.95 * (dev.size() - 1))];
            const double lim = len > 500 ? std::min(25.0, 3 + 0.04 * len) : 3 + 0.01 * len;
            if (std::fabs(pf[0]) > 0.035 || p95 > lim) { ++nbowed; continue; }
            const double ideal = median(cv2);
            auto& out = hz ? eh : ev;
            size_t i = 0;
            while (i < tv2.size()) {   // 4-px bins
                long bin = (long)np_floordiv(tv2[i], BIN);
                size_t j = i;
                double st = 0, sc = 0;
                while (j < tv2.size() && (long)np_floordiv(tv2[j], BIN) == bin) { st += tv2[j]; sc += cv2[j]; ++j; }
                out.push_back({st / (double)(j - i), ideal, sc / (double)(j - i) - ideal});
                i = j;
            }
            (hz ? EH : EV).push_back({ideal, tv2.front(), tv2.back(), wd, len});
        }
    }
}

Field zero_field(int gw, int gh, int g) {
    Field f;
    f.gw = gw; f.gh = gh; f.g = g;
    f.v.assign((size_t)gw * gh, 0.0);
    return f;
}

// compose global o local on the 4-px grid
GridMap compose_maps(const WarpDebug& d, int w, int h) {
    GridMap g = GridMap::make(w, h, 4);
    for (int i = 0; i < g.gh; ++i)
        for (int j = 0; j < g.gw; ++j) {
            float X4 = (float)(j * 4), Y4 = (float)(i * 4);
            float lx = X4 + d.Fx.sample(X4, Y4), ly = Y4 + d.Fy.sample(X4, Y4);
            g.X(i, j) = lx + d.GX.sample(lx, ly);
            g.Y(i, j) = ly + d.GY.sample(lx, ly);
        }
    return g;
}

}  // namespace

void set_warp_cg_rtol(double rtol) { cg_rtol_store() = rtol > 0 ? rtol : 1e-8; }
double warp_cg_rtol() { return cg_rtol_store().load(); }

GridMap unwarp_map(const cv::Mat& src, WarpDebug* dbg, const cv::Mat* ignore, bool share_rows, bool faint, bool robust, bool edges) {
    WarpDebug local;
    WarpDebug& d = dbg ? *dbg : local;
    const bool masked = ignore && !ignore->empty() && ignore->size() == src.size();
    auto ignored = [&](double x, double y) {
        if (!masked) return false;
        int xi = (int)std::lround(x), yi = (int)std::lround(y);
        if (xi < 0 || yi < 0 || xi >= ignore->cols || yi >= ignore->rows) return false;
        return ignore->at<uint8_t>(yi, xi) != 0;
    };
    // samples (along, ideal, disp) of a horizontal (or vertical) trace sit at (along, ideal + disp)
    auto drop = [&](std::vector<std::array<double, 3>>& rows, bool hz) {
        if (!masked) return;
        size_t n0 = rows.size();
        rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const std::array<double, 3>& r) {
                       return hz ? ignored(r[0], r[1] + r[2]) : ignored(r[1] + r[2], r[0]);
                   }), rows.end());
        d.n_ignored += (int)(n0 - rows.size());
    };
    const int threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    int h = src.rows, w = src.cols;
    auto t0 = Clock::now();
    // ---- global stage (global_grid)
    std::vector<DLine> Hs, Vs;
    std::vector<cv::Point2d> sp, dp;
    try {
        dewarp_detect(src, Hs, Vs, faint);
        build_controls(Hs, Vs, sp, dp);
    } catch (const std::exception&) {
        sp.clear(); dp.clear();
    }
    for (const auto& L : Hs) d.G_lines.push_back({1.0, L.target, L.t0, L.t1});
    for (const auto& L : Vs) d.G_lines.push_back({0.0, L.target, L.t0, L.t1});
    if (masked) {   // controls at their straightened (destination) position
        std::vector<cv::Point2d> s2, d2;
        for (size_t i = 0; i < dp.size(); ++i)
            if (!ignored(dp[i].x, dp[i].y)) { s2.push_back(sp[i]); d2.push_back(dp[i]); }
        d.n_ignored += (int)(dp.size() - d2.size());
        sp = s2; dp = d2;
    }
    d.n_ctrl_raw = (int)dp.size();
    const int GG = 32;
    int gw = w / GG + 2, gh = h / GG + 2;
    // no usable grid lines (plain text, photos, projector): skip the global stage (round 4 fix)
    const bool skip_global = dp.size() < 20;
    {
        std::map<long, size_t> first;
        for (size_t i = 0; i < dp.size(); ++i) {
            long key = (long)np_floordiv(dp[i].x, 60) * 10000 + (long)np_floordiv(dp[i].y, 60);
            first.emplace(key, i);
        }
        std::vector<cv::Point2d> s2, d2;
        for (auto& kv : first) { s2.push_back(sp[kv.second]); d2.push_back(dp[kv.second]); }
        sp = s2; dp = d2;
        if (dp.size() > 700) {
            NpRng rng = NpRng::seed0();
            auto sel = rng.choice_no_replace((int64_t)dp.size(), 700);
            std::vector<cv::Point2d> s3, d3;
            for (auto i : sel) { s3.push_back(sp[i]); d3.push_back(dp[i]); }
            sp = s3; dp = d3;
        }
    }
    d.n_ctrl = (int)dp.size();
    std::vector<std::array<double, 2>> cxy;
    Vec vx, vy;
    for (size_t i = 0; i < dp.size(); ++i) { cxy.push_back({dp[i].x, dp[i].y}); vx.push_back(sp[i].x - dp[i].x); vy.push_back(sp[i].y - dp[i].y); }
    if (skip_global) {
        d.n_ctrl = 0;
        d.GX = zero_field(gw, gh, GG);
        d.GY = zero_field(gw, gh, GG);
    } else {   // the x and y fields are independent solves
        std::thread tx([&] { d.GX = solve_field(cxy, vx, nullptr, gw, gh, GG, nullptr, &d.cg_iters[0]); });
        d.GY = solve_field(cxy, vy, nullptr, gw, gh, GG, nullptr, &d.cg_iters[1]);
        tx.join();
    }
    d.ms.emplace_back("global", ms_since(t0));
    t0 = Clock::now();
    // A_img = remap(src, xx + GX(xx/32, yy/32), ...): exact bilinear at x/32 (no border taps needed)
    std::vector<float> gx(d.GX.v.begin(), d.GX.v.end()), gy(d.GY.v.begin(), d.GY.v.end());
    if (skip_global) {
        d.A_img = src.clone();
    } else if (gpu_stages_enabled() && remap_offset_gpu(src, gx, gy, gw, gh, GG, d.A_img)) {
        // the same map and bicubic on the GPU, byte-identical (resample.h)
    } else {
        d.A_img.create(h, w, src.type());
        const int band = 64;
        cv::Mat mx(band, w, CV_32F), my(band, w, CV_32F);
        for (int y0 = 0; y0 < h; y0 += band) {
            int y1 = std::min(h, y0 + band);
            for (int y = y0; y < y1; ++y) {
                int iy = y >> 5;
                float ay = (y & 31) * (1.f / 32), vy0 = 1.f - ay;
                for (int x = 0; x < w; ++x) {
                    int ix = x >> 5;
                    float ax = (x & 31) * (1.f / 32), vx0 = 1.f - ax;
                    size_t i00 = (size_t)iy * gw + ix;
                    float ddx = gx[i00] * (vy0 * vx0) + gx[i00 + 1] * (vy0 * ax) + gx[i00 + gw] * (ay * vx0) + gx[i00 + gw + 1] * (ay * ax);
                    float ddy = gy[i00] * (vy0 * vx0) + gy[i00 + 1] * (vy0 * ax) + gy[i00 + gw] * (ay * vx0) + gy[i00 + gw + 1] * (ay * ax);
                    mx.at<float>(y - y0, x) = (float)x + ddx;
                    my.at<float>(y - y0, x) = (float)y + ddy;
                }
            }
            cv::Mat dst = d.A_img.rowRange(y0, y1);
            cv::remap(src, dst, mx.rowRange(0, y1 - y0), my.rowRange(0, y1 - y0), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
        }
    }
    d.D = darkness_fast(d.A_img);
    d.ms.emplace_back("global_remap+darkness", ms_since(t0));
    t0 = Clock::now();
    // ---- local stage (local_field)
    get_lines(d.D, d.H, d.V, faint);
    const int GC = 16, GF = 8;
    if (d.H.empty() || d.V.empty()) {
        // no table lines: nothing to straighten locally; a zero field (round 4 fix). Text-only
        // pages then get the surface model only.
        d.n_text = 0;
        d.Fx = d.Fy = d.Fx1 = d.Fy1 = zero_field(w / GF + 2, h / GF + 2, GF);
        d.ms.emplace_back("lines (none: zero field)", ms_since(t0));
        return compose_maps(d, w, h);
    }
    DView D{d.D.ptr<float>(), d.D.rows, d.D.cols};
    d.S1 = collect(D, d.H, d.V, nullptr, nullptr, threads, share_rows, faint);
    d.T = text_rows(d.D, d.H, d.V, d.n_text);
    if (faint) {   // app: aligned text-column edges as vertical lines (text_margins)
        d.M = text_margins(d.D, d.H, d.V, d.n_margins);
        drop(d.M, false);
    }
    if (edges && share_rows) {
        Lines eh, ev;
        colour_edges(d.A_img, d.H, d.V, d.EH, d.EV, eh, ev, d.n_edge_bowed);
        d.n_edges_h = (int)eh.size();
        d.E_lines = eh;
        d.E_lines.insert(d.E_lines.end(), ev.begin(), ev.end());
        drop(d.EH, true);
        drop(d.EV, false);
    }
    drop(d.S1.h, true);
    drop(d.S1.v, false);
    drop(d.T, true);
    d.ms.emplace_back("lines+trace pass1+text", ms_since(t0));
    t0 = Clock::now();
    Field Fx_prev, Fy_prev;
    for (int it = 0; it < 2; ++it) {
        int G = it == 1 ? GF : GC;
        int gw2 = w / G + 2, gh2 = h / G + 2;
        const Samples& S = it == 0 ? d.S1 : d.S2;
        if (it == 1) {
            auto tc = Clock::now();
            d.S2 = collect(D, d.H, d.V, &Fx_prev, &Fy_prev, threads, share_rows, faint);
            drop(d.S2.h, true);
            drop(d.S2.v, false);
            d.ms.emplace_back("trace pass2", ms_since(tc));
        }
        const Samples& SS = it == 0 ? d.S1 : d.S2;
        (void)S;
        std::vector<std::array<double, 2>> xy;
        Vec val, wts;
        for (const auto& r : SS.h) {
            double xs = r[0] - (it == 1 ? (double)Fx_prev.sample(r[0], r[1]) : 0.0);
            xy.push_back({xs, r[1]}); val.push_back(r[2]); wts.push_back(1.0);
        }
        for (const auto& r : d.T) {
            double xs = r[0] - (it == 1 ? (double)Fx_prev.sample(r[0], r[1]) : 0.0);
            xy.push_back({xs, r[1]}); val.push_back(r[2]); wts.push_back(1.0);   // TEXT_W = 1
        }
        for (const auto& r : d.EH) {
            double xs = r[0] - (it == 1 ? (double)Fx_prev.sample(r[0], r[1]) : 0.0);
            xy.push_back({xs, r[1]}); val.push_back(r[2]); wts.push_back(1.0);
        }
        Field warm_y, warm_x;
        if (it == 1) { warm_y = grid_to_grid(Fy_prev, G, gw2, gh2); warm_x = grid_to_grid(Fx_prev, G, gw2, gh2); }
        std::vector<std::array<double, 2>> xyv;
        Vec valv;
        for (const std::vector<std::array<double, 3>>* src : {&SS.v, (const std::vector<std::array<double, 3>>*)&d.M,
                                                              (const std::vector<std::array<double, 3>>*)&d.EV})
            for (const auto& r : *src) {
                double ys = r[0] - (it == 1 ? (double)Fy_prev.sample(r[1], r[0]) : 0.0);
                xyv.push_back({r[1], ys}); valv.push_back(r[2]);
            }
        Field Fy, Fx;
        // every sample of one direction ignored: nothing to straighten that way
        std::thread ty([&] {
            Fy = xy.empty() ? zero_field(gw2, gh2, G) : solve_field(xy, val, &wts, gw2, gh2, G, it == 1 ? &warm_y : nullptr, &d.cg_iters[2 + 2 * it]);
        });
        Fx = xyv.empty() ? zero_field(gw2, gh2, G) : solve_field(xyv, valv, nullptr, gw2, gh2, G, it == 1 ? &warm_x : nullptr, &d.cg_iters[3 + 2 * it]);
        ty.join();
        // robust (app): a line the solved field does not follow - one that is not straight on the
        // paper (a curve, a mis-trace, two pieces forced to one height) - is down-weighted as a
        // whole, w = 1 / (1 + (median |residual| / 3 px)^2), and the fields solved again (IRLS:
        // 2 reweights on the coarse pass, 1 on the fine one). Whole lines, not samples: a correct
        // line with one sharp bend keeps its weight. A line's samples share its ideal position
        // (the y of a horizontal sample, the x of a vertical one), which groups them.
        if (robust) {
            const double c = 3.0;
            auto line_weights = [&](const std::vector<std::array<double, 2>>& pos, const Vec& v, const Field& F, int key) {
                std::map<double, std::vector<size_t>> groups;
                for (size_t i = 0; i < pos.size(); ++i) groups[pos[i][key]].push_back(i);
                Vec w(pos.size(), 1.0);
                for (const auto& kv : groups) {
                    Vec r;
                    for (size_t i : kv.second) r.push_back(std::fabs(v[i] - F.sample(pos[i][0], pos[i][1])));
                    double m = median(r) / c, q = 1 / (1 + m * m);
                    for (size_t i : kv.second) w[i] = q;
                }
                return w;
            };
            for (int rw = 0; rw < (it == 0 ? 2 : 1); ++rw) {
                Vec wy = line_weights(xy, val, Fy, 1), wx = line_weights(xyv, valv, Fx, 0);
                for (size_t i = 0; i < wy.size(); ++i) wy[i] *= wts[i];
                Field Fy0 = Fy, Fx0 = Fx;
                std::thread t2([&] { if (!xy.empty()) Fy = solve_field(xy, val, &wy, gw2, gh2, G, &Fy0, nullptr); });
                if (!xyv.empty()) Fx = solve_field(xyv, valv, &wx, gw2, gh2, G, &Fx0, nullptr);
                t2.join();
                double down = 0;
                for (double q : wx) down += q < 0.5;
                for (size_t i = 0; i < wy.size(); ++i) down += wy[i] < 0.5 * wts[i];
                d.n_downweighted = (int)down;
            }
        }
        if (it == 0) { d.Fx1 = Fx; d.Fy1 = Fy; }
        Fy_prev = Fy; Fx_prev = Fx;
    }
    d.Fx = Fx_prev; d.Fy = Fy_prev;
    d.ms.emplace_back("solves", ms_since(t0));
    return compose_maps(d, w, h);
}

}  // namespace fs
