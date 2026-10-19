#include "warpmap.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
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

void dewarp_detect(const cv::Mat& img, std::vector<DLine>& Hs, std::vector<DLine>& Vs) {
    cv::Mat g, bg;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    cv::dilate(g, bg, cv::Mat::ones(9, 9, CV_8U));
    cv::medianBlur(bg, bg, 41);
    cv::Mat ink(g.size(), CV_8U);
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x) ink.at<uint8_t>(y, x) = ((int)bg.at<uint8_t>(y, x) - (int)g.at<uint8_t>(y, x)) > 45 ? 255 : 0;
    cv::Mat hm, vm;
    cv::morphologyEx(ink, hm, cv::MORPH_OPEN, rect(120, 1));
    cv::morphologyEx(hm, hm, cv::MORPH_CLOSE, rect(25, 3));
    cv::morphologyEx(ink, vm, cv::MORPH_OPEN, rect(1, 120));
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

// Lm^T Lm + 1e-9 I for warpmap.regularizer(gw, gh) (biharmonic + membrane), as triplets
void add_regularizer(int gw, int gh, std::vector<int64_t>& keys, std::vector<double>& vals) {
    int64_t n = (int64_t)gw * gh;
    auto row = [&](const int* id, const double* w, int k) {
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) { keys.push_back((int64_t)id[a] * n + id[b]); vals.push_back(w[a] * w[b]); }
    };
    auto I = [&](int iy, int ix) { return iy * gw + ix; };
    double s1 = std::sqrt(LAM_TPS), s2 = std::sqrt(2 * LAM_TPS), s3 = std::sqrt(LAM_MEM);
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix + 2 < gw; ++ix) {
            int id[3] = {I(iy, ix + 2), I(iy, ix + 1), I(iy, ix)};
            double w[3] = {1 * s1, -2 * s1, 1 * s1};
            row(id, w, 3);
        }
    for (int iy = 0; iy + 2 < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            int id[3] = {I(iy + 2, ix), I(iy + 1, ix), I(iy, ix)};
            double w[3] = {1 * s1, -2 * s1, 1 * s1};
            row(id, w, 3);
        }
    for (int iy = 0; iy + 1 < gh; ++iy)
        for (int ix = 0; ix + 1 < gw; ++ix) {
            int id[4] = {I(iy + 1, ix + 1), I(iy + 1, ix), I(iy, ix + 1), I(iy, ix)};
            double w[4] = {1 * s2, -1 * s2, -1 * s2, 1 * s2};
            row(id, w, 4);
        }
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix + 1 < gw; ++ix) {
            int id[2] = {I(iy, ix + 1), I(iy, ix)};
            double w[2] = {1 * s3, -1 * s3};
            row(id, w, 2);
        }
    for (int iy = 0; iy + 1 < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            int id[2] = {I(iy + 1, ix), I(iy, ix)};
            double w[2] = {1 * s3, -1 * s3};
            row(id, w, 2);
        }
    for (int64_t i = 0; i < n; ++i) { keys.push_back(i * n + i); vals.push_back(1e-9); }
}

// warpmap.solve: regularised least squares on a G-px grid
Field solve_field(const std::vector<std::array<double, 2>>& xy, const Vec& values, const Vec* weights, int gw, int gh,
                  int G, const Field* warm, int* iters) {
    int64_t n = (int64_t)gw * gh;
    std::vector<int64_t> keys;
    std::vector<double> vals;
    keys.reserve(xy.size() * 16 + n * 30);
    vals.reserve(keys.capacity());
    add_regularizer(gw, gh, keys, vals);
    std::vector<double> rhs(n, 0.0);
    for (size_t s = 0; s < xy.size(); ++s) {
        double gx = xy[s][0] / G, gy = xy[s][1] / G;
        long x0 = std::min(std::max((long)std::floor(gx), 0L), (long)gw - 2);
        long y0 = std::min(std::max((long)std::floor(gy), 0L), (long)gh - 2);
        double fx = std::min(std::max(gx - x0, 0.0), 1.0), fy = std::min(std::max(gy - y0, 0.0), 1.0);
        int id[4] = {(int)(y0 * gw + x0), (int)(y0 * gw + x0 + 1), (int)((y0 + 1) * gw + x0), (int)((y0 + 1) * gw + x0 + 1)};
        double w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        double ws = weights ? (*weights)[s] : 1.0;
        for (int a = 0; a < 4; ++a) {
            rhs[id[a]] += w[a] * (ws * values[s]);
            for (int b = 0; b < 4; ++b) { keys.push_back((int64_t)id[a] * n + id[b]); vals.push_back(w[a] * ws * w[b]); }
        }
    }
    Csr M = csr_from_triplets((int)n, (int)n, keys, vals);
    std::vector<double> x0;
    if (warm) x0 = warm->v;
    SolveStats st;
    Field F;
    F.gw = gw; F.gh = gh; F.g = G;
    F.v = solve_spd_grid(M, rhs, gw, gh, warm ? &x0 : nullptr, 1e-8, 1000, &st);
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
Lines segments(const cv::Mat& ink, bool hz, int klen = 100, int minlen = 150) {
    cv::Mat m0, m;
    cv::morphologyEx(ink, m0, cv::MORPH_OPEN, hz ? rect(klen, 1) : rect(1, klen));
    cv::morphologyEx(m0, m, cv::MORPH_CLOSE, hz ? rect(31, 1) : rect(1, 31));
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

bool touches(const Line& seg, const Lines& perp, double tol) {
    int ok = 0;
    for (double end : {seg[1], seg[2]})
        for (const auto& p : perp)
            if (std::fabs(p[0] - end) < tol && p[1] - tol <= seg[0] && seg[0] <= p[2] + tol) { ++ok; break; }
    return ok == 2;
}

Lines short_segments(const cv::Mat& ink, bool hz, const Lines& long_segs, const Lines& perp) {
    Lines segs = segments(ink, hz, 35, 35), add;
    for (const auto& s : segs) {
        if (s[4] >= 150) continue;
        bool dup = false;
        for (const auto& o : long_segs)
            if (std::fabs(o[0] - s[0]) < 12 && s[1] <= o[2] + 10 && s[2] >= o[1] - 10) { dup = true; break; }
        if (dup) continue;
        if (touches(s, perp, 7)) add.push_back(s);
    }
    return add;
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

void trace_bounds(const Line& L, const Lines& perp, double lo, double hi, double& a2, double& b2, double ext = 200, double pad = 6) {
    double p = L[0], a = L[1], b = L[2];
    bool hl = false, hr = false;
    double ml = -1e300, mr = 1e300;
    for (const auto& P : perp)
        if (P[1] - 15 <= p && p <= P[2] + 15) {
            if (P[0] <= a + 10) { hl = true; ml = std::max(ml, P[0]); }
            if (P[0] >= b - 10) { hr = true; mr = std::min(mr, P[0]); }
        }
    a2 = std::max({lo, a - ext, hl ? ml - pad : -1e9});
    b2 = std::min({hi, b + ext, hr ? mr + pad : 1e9});
}

}  // namespace

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

void get_lines(const cv::Mat& D, Lines& H, Lines& V) {
    cv::Mat ink = D > 45.f;   // 0 / 255
    H = segments(ink, true);
    V = segments(ink, false);
    Lines Vs = short_segments(ink, false, V, H);
    Lines HV = V;
    HV.insert(HV.end(), Vs.begin(), Vs.end());
    Lines Hs = short_segments(ink, true, H, HV);
    Lines h2 = H; h2.insert(h2.end(), Hs.begin(), Hs.end());
    Lines v2 = V; v2.insert(v2.end(), Vs.begin(), Vs.end());
    H = merge_gap(h2);
    V = merge_gap(v2);
    auto clip = [](Lines& A, const Lines& B, double tol) {
        for (auto& L : A) {
            bool any = false;
            double mn = 1e300, mx = -1e300;
            int cnt = 0;
            for (const auto& P : B)
                if (P[1] - 15 <= L[0] && L[0] <= P[2] + 15 && P[4] > 80) { mn = std::min(mn, P[0]); mx = std::max(mx, P[0]); ++cnt; any = true; }
            if (any && cnt >= 2) {
                L[1] = std::max(L[1], mn - tol);
                L[2] = std::min(L[2], mx + tol);
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
Samples collect(const DView& D, const Lines& H, const Lines& V, const Field* prior_dx, const Field* prior_dy, int threads) {
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
        auto one = [&](size_t li) {
            const Line& L = lines[li];
            double p = L[0], a = L[1], b = L[2], th = L[3];
            double a2, b2;
            trace_bounds(L, perp, lo, hi, a2, b2);
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
            trace_w2(D, p, a2, b2, hz, t, c, wd, maxrun, prior);
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

}  // namespace

GridMap unwarp_map(const cv::Mat& src, WarpDebug* dbg) {
    WarpDebug local;
    WarpDebug& d = dbg ? *dbg : local;
    const int threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    int h = src.rows, w = src.cols;
    auto t0 = Clock::now();
    // ---- global stage (global_grid)
    std::vector<DLine> Hs, Vs;
    dewarp_detect(src, Hs, Vs);
    std::vector<cv::Point2d> sp, dp;
    build_controls(Hs, Vs, sp, dp);
    d.n_ctrl_raw = (int)dp.size();
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
    const int GG = 32;
    int gw = w / GG + 2, gh = h / GG + 2;
    std::vector<std::array<double, 2>> cxy;
    Vec vx, vy;
    for (size_t i = 0; i < dp.size(); ++i) { cxy.push_back({dp[i].x, dp[i].y}); vx.push_back(sp[i].x - dp[i].x); vy.push_back(sp[i].y - dp[i].y); }
    d.GX = solve_field(cxy, vx, nullptr, gw, gh, GG, nullptr, &d.cg_iters[0]);
    d.GY = solve_field(cxy, vy, nullptr, gw, gh, GG, nullptr, &d.cg_iters[1]);
    d.ms.emplace_back("global", ms_since(t0));
    t0 = Clock::now();
    // A_img = remap(src, xx + GX(xx/32, yy/32), ...): exact bilinear at x/32 (no border taps needed)
    {
        std::vector<float> gx(d.GX.v.begin(), d.GX.v.end()), gy(d.GY.v.begin(), d.GY.v.end());
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
    get_lines(d.D, d.H, d.V);
    DView D{d.D.ptr<float>(), d.D.rows, d.D.cols};
    d.S1 = collect(D, d.H, d.V, nullptr, nullptr, threads);
    d.T = text_rows(d.D, d.H, d.V, d.n_text);
    d.ms.emplace_back("lines+trace pass1+text", ms_since(t0));
    t0 = Clock::now();
    const int GC = 16, GF = 8;
    Field Fx_prev, Fy_prev;
    for (int it = 0; it < 2; ++it) {
        int G = it == 1 ? GF : GC;
        int gw2 = w / G + 2, gh2 = h / G + 2;
        const Samples& S = it == 0 ? d.S1 : d.S2;
        if (it == 1) {
            auto tc = Clock::now();
            d.S2 = collect(D, d.H, d.V, &Fx_prev, &Fy_prev, threads);
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
        Field warm_y, warm_x;
        if (it == 1) { warm_y = grid_to_grid(Fy_prev, G, gw2, gh2); warm_x = grid_to_grid(Fx_prev, G, gw2, gh2); }
        Field Fy = solve_field(xy, val, &wts, gw2, gh2, G, it == 1 ? &warm_y : nullptr, &d.cg_iters[2 + 2 * it]);
        xy.clear(); val.clear();
        for (const auto& r : SS.v) {
            double ys = r[0] - (it == 1 ? (double)Fy_prev.sample(r[1], r[0]) : 0.0);
            xy.push_back({r[1], ys}); val.push_back(r[2]);
        }
        Field Fx = solve_field(xy, val, nullptr, gw2, gh2, G, it == 1 ? &warm_x : nullptr, &d.cg_iters[3 + 2 * it]);
        if (it == 0) { d.Fx1 = Fx; d.Fy1 = Fy; }
        Fy_prev = Fy; Fx_prev = Fx;
    }
    d.Fx = Fx_prev; d.Fy = Fy_prev;
    d.ms.emplace_back("solves", ms_since(t0));
    // ---- compose global o local on the 4-px grid
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

}  // namespace fs
