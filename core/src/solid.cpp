#include "solid.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "nputil.h"
#include "warpmap.h"

namespace fs {

namespace {

long pyround(double x) { return (long)std::nearbyint(x); }

// numpy slice bounds [start, stop) on an axis of length n (negative indices wrap)
void np_slice(long start, long stop, long n, long& a, long& b) {
    if (start < 0) start = std::max(0L, start + n);
    if (stop < 0) stop = std::max(0L, stop + n);
    a = std::min(start, n);
    b = std::min(stop, n);
}

// clean.draw_lines(shape, H, V, extra, scale)
cv::Mat draw_lines(int h, int w, const Lines& H, const Lines& V, int extra = 0, double scale = 0.62) {
    cv::Mat m = cv::Mat::zeros(h, w, CV_8U);
    auto fill = [&](long r0, long r1, long c0, long c1) {
        long a, b, c, d;
        np_slice(r0, r1, h, a, b);
        np_slice(c0, c1, w, c, d);
        if (b > a && d > c) m(cv::Range((int)a, (int)b), cv::Range((int)c, (int)d)).setTo(255);
    };
    double hi = scale < 1 ? 6 : 12;
    for (const auto& L : H) {
        long t = pyround(std::min(std::max(L[3] * scale, 3.0), hi)) + 2 * extra;
        long y0 = pyround(L[0] - t / 2.0);
        fill(std::max(0L, y0), y0 + t, pyround(L[1] - t / 2.0), pyround(L[2] + t / 2.0) + 1);
    }
    for (const auto& L : V) {
        long t = pyround(std::min(std::max(L[3] * scale, 3.0), hi)) + 2 * extra;
        long x0 = pyround(L[0] - t / 2.0);
        fill(pyround(L[1] - t / 2.0), pyround(L[2] + t / 2.0) + 1, std::max(0L, x0), x0 + t);
    }
    return m;
}

// clean.infer_borders: vertical border pieces where several horizontal lines end at one x
void infer_borders(const Lines& H, Lines& V, const cv::Mat& ink, int min_members = 3, double tol = 20, double maxgap = 160) {
    for (int endi : {1, 2}) {
        std::vector<std::pair<double, double>> ends;
        for (const auto& L : H)
            if (L[4] > 300) ends.emplace_back(L[endi], L[0]);
        std::sort(ends.begin(), ends.end());
        std::vector<std::vector<std::pair<double, double>>> clusters;
        for (auto& ep : ends) {
            if (!clusters.empty()) {
                Vec es;
                for (auto& c : clusters.back()) es.push_back(c.first);
                if (std::fabs(ep.first - median(es)) < tol) { clusters.back().push_back(ep); continue; }
            }
            clusters.push_back({ep});
        }
        for (auto& c : clusters) {
            if ((int)c.size() < min_members) continue;
            Vec es, ps;
            for (auto& q : c) { es.push_back(q.first); ps.push_back(q.second); }
            double pos = median(es);
            std::sort(ps.begin(), ps.end());
            Lines ex;
            for (const auto& L : V)
                if (std::fabs(L[0] - pos) < 15) ex.push_back(L);
            Vec th;
            for (const auto& L : V) th.push_back(L[3]);
            double thick = V.empty() ? 5.0 : median(th);
            for (size_t k = 0; k + 1 < ps.size(); ++k) {
                double p0 = ps[k], p1 = ps[k + 1];
                if (p1 - p0 > maxgap) continue;
                double cov = 0;
                for (const auto& L : ex) cov += std::max(0.0, std::min(L[2], p1) - std::max(L[1], p0));
                if (cov > 0.8 * (p1 - p0)) continue;
                long x0 = (long)std::max(0.0, pos - 12), x1 = (long)(pos + 13);
                long ra, rb, ca, cb;
                np_slice((long)p0 + 4, (long)p1 - 3, ink.rows, ra, rb);
                np_slice(x0, x1, ink.cols, ca, cb);
                if (rb <= ra || cb <= ca) continue;
                int present = 0;
                for (long r = ra; r < rb; ++r) {
                    bool any = false;
                    for (long q = ca; q < cb; ++q) any |= ink.at<uint8_t>((int)r, (int)q) > 0;
                    present += any;
                }
                if ((double)present / (double)(rb - ra) < 0.7) continue;
                V.push_back({pos, p0, p1, thick, p1 - p0});
            }
        }
    }
}

// clean.extend_wobbly: follow long vertical lines past their detected ends
void extend_wobbly(Lines& V, const Lines& H, const cv::Mat& ink, int band = 12, int win = 40, double frac = 0.9) {
    int h = ink.rows, w = ink.cols;
    for (auto& L : V) {
        if (L[4] < 200) continue;
        int x0 = (int)std::max(0.0, L[0] - band), x1 = (int)std::min((double)w, L[0] + band + 1);
        std::vector<char> pres(h, 0);
        for (int y = 0; y < h; ++y)
            for (int x = x0; x < x1; ++x)
                if (ink.at<uint8_t>(y, x) > 0) { pres[y] = 1; break; }
        for (int d : {1, -1}) {
            double endv = d > 0 ? L[2] : L[1];
            long y = (long)endv, last = y, miss = 0;
            std::vector<char> hist;
            while (0 < y + d && y + d < h - 1) {
                y += d;
                hist.push_back(pres[y]);
                if ((int)hist.size() > win) hist.erase(hist.begin());
                if (pres[y]) { last = y; miss = 0; }
                else ++miss;
                double mean = 0;
                for (char q : hist) mean += q;
                mean /= hist.size();
                if (miss >= 6 || ((int)hist.size() == win && mean < frac)) break;
            }
            double lastd = (double)last;
            if (std::fabs(lastd - endv) > 30) {
                if (!H.empty()) {
                    size_t j = 0;
                    for (size_t k = 1; k < H.size(); ++k)
                        if (std::fabs(H[k][0] - lastd) < std::fabs(H[j][0] - lastd)) j = k;
                    if (std::fabs(H[j][0] - lastd) < 25) lastd = H[j][0];
                }
                Vec cross;
                for (const auto& Hh : H)
                    if (Hh[1] - 15 <= L[0] && L[0] <= Hh[2] + 15) cross.push_back(Hh[0]);
                if (!cross.empty())
                    lastd = std::min(std::max(lastd, *std::min_element(cross.begin(), cross.end())), *std::max_element(cross.begin(), cross.end()));
                if (d > 0) L[2] = std::max(L[2], lastd);
                else L[1] = std::min(L[1], lastd);
            }
        }
    }
}

// clean.snap
void snap(Lines& H, Lines& V, double tol = 30) {
    auto one = [&](Lines& A, const Lines& B) {
        for (auto& a : A)
            for (int end : {1, 2}) {
                bool have = false;
                double best = 0;
                for (const auto& b : B)
                    if (b[1] - tol <= a[0] && a[0] <= b[2] + tol && std::fabs(b[0] - a[end]) < tol)
                        if (!have || std::fabs(b[0] - a[end]) < std::fabs(best - a[end])) { best = b[0]; have = true; }
                if (have) a[end] = best;
            }
    };
    one(H, V);
    one(V, H);
}

cv::Vec3f hsv_of(const cv::Vec3f& bgr) {
    cv::Mat m(1, 1, CV_8UC3, cv::Scalar((uint8_t)bgr[0], (uint8_t)bgr[1], (uint8_t)bgr[2])), o;
    cv::cvtColor(m, o, cv::COLOR_BGR2HSV);
    cv::Vec3b v = o.at<cv::Vec3b>(0, 0);
    return cv::Vec3f(v[0], v[1], v[2]);
}

}  // namespace

cv::Mat solid_colors(const cv::Mat& img, SolidStats* st) {
    auto t0 = std::chrono::steady_clock::now();
    int h = img.rows, w = img.cols;
    cv::Mat bgr, g, bg;
    cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::dilate(g, bg, cv::Mat::ones(9, 9, CV_8U));
    cv::medianBlur(bg, bg, 41);
    cv::Mat diff;
    cv::subtract(bg, g, diff, cv::noArray(), CV_16S);
    cv::Mat ink = diff > 45;
    Lines H = clean_segments(ink, true), V = clean_segments(ink, false);
    Lines Vs = clean_short_segments(ink, false, V, H);
    Lines HV = V;
    HV.insert(HV.end(), Vs.begin(), Vs.end());
    Lines Hs = clean_short_segments(ink, true, H, HV);
    V.insert(V.end(), Vs.begin(), Vs.end());
    H.insert(H.end(), Hs.begin(), Hs.end());
    infer_borders(H, V, ink);
    extend_wobbly(V, H, ink);
    snap(H, V);
    cv::Mat lines = draw_lines(h, w, H, V);              // region separators only, never drawn
    cv::Mat textish;
    cv::dilate(diff > 22, textish, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat lab, stats, cen;
    int n = cv::connectedComponentsWithStats(lines == 0, lab, stats, cen, 4);
    struct Region { int i; cv::Rect r; cv::Vec3f col; int area; bool coloured; };
    std::vector<Region> regions;
    for (int i = 1; i < n; ++i) {
        int x = stats.at<int>(i, 0), y = stats.at<int>(i, 1), ww = stats.at<int>(i, 2), hh = stats.at<int>(i, 3), ar = stats.at<int>(i, 4);
        if (x == 0 || y == 0 || x + ww >= w || y + hh >= h || ar > 0.3 * h * w) continue;
        cv::Rect R(x, y, ww, hh);
        cv::Mat sl = lab(R) == i, core;
        cv::erode(sl, core, cv::Mat::ones(9, 9, CV_8U));
        if (cv::countNonZero(core) < 30) core = sl;
        cv::Mat pc = core & ~textish(R);
        if (cv::countNonZero(pc) < 30) pc = core;
        cv::Vec3f col;
        for (int c = 0; c < 3; ++c) {
            Vec v;
            for (int yy = 0; yy < hh; ++yy)
                for (int xx = 0; xx < ww; ++xx)
                    if (pc.at<uint8_t>(yy, xx)) v.push_back(bgr.at<cv::Vec3b>(y + yy, x + xx)[c]);
            col[c] = (float)median(v);
        }
        cv::Vec3f hsv = hsv_of(col);
        bool coloured = (hsv[1] > 12 || hsv[2] < 238) && hsv[2] > 110;
        regions.push_back({i, R, col, ar, coloured});
    }
    // merge similar colours (Lab distance < 8, greedy, area-weighted), lift as clean.py
    std::vector<int> grp(regions.size(), -1);
    std::vector<cv::Vec3f> fillc;
    std::vector<cv::Vec3f> labc(regions.size());
    for (size_t k = 0; k < regions.size(); ++k) {
        cv::Mat m(1, 1, CV_8UC3, cv::Scalar((uint8_t)regions[k].col[0], (uint8_t)regions[k].col[1], (uint8_t)regions[k].col[2])), o;
        cv::cvtColor(m, o, cv::COLOR_BGR2Lab);
        cv::Vec3b v = o.at<cv::Vec3b>(0, 0);
        labc[k] = cv::Vec3f(v[0], v[1], v[2]);
    }
    int ngroups = 0, ncol = 0;
    for (size_t k = 0; k < regions.size(); ++k) {
        if (!regions[k].coloured || grp[k] >= 0) continue;
        double wsum = 0;
        cv::Vec3d acc(0, 0, 0);
        std::vector<size_t> members;
        for (size_t q = 0; q < regions.size(); ++q)
            if (regions[q].coloured && grp[q] < 0 && cv::norm(labc[q] - labc[k]) < 8) { grp[q] = ngroups; members.push_back(q); }
        for (size_t q : members) { acc += cv::Vec3d(regions[q].col) * regions[q].area; wsum += regions[q].area; }
        cv::Vec3f c(acc / wsum);
        for (int i = 0; i < 3; ++i) c[i] = std::min(255.f, std::max(0.f, c[i] * 1.03f + 3.f));
        fillc.push_back(c);
        ++ngroups;
        ncol += (int)members.size();
    }
    // composite: fill under the page's own ink, lines and text untouched
    cv::Mat out = img.clone();
    for (size_t k = 0; k < regions.size(); ++k) {
        const Region& R = regions[k];
        cv::Vec3f fc = R.coloured ? fillc[grp[k]] : cv::Vec3f(255, 255, 255);
        for (int yy = R.r.y; yy < R.r.y + R.r.height; ++yy)
            for (int xx = R.r.x; xx < R.r.x + R.r.width; ++xx) {
                if (lab.at<int>(yy, xx) != R.i) continue;
                float a = std::min(1.f, std::max(0.f, (diff.at<short>(yy, xx) - 20.f) / 40.f));
                cv::Vec4b& o = out.at<cv::Vec4b>(yy, xx);
                for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::lrint(fc[c] * (1.f - a) + o[c] * a);
            }
    }
    if (st) {
        st->regions = (int)regions.size();
        st->coloured = ncol;
        st->colour_groups = ngroups;
        st->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return out;
}

}  // namespace fs
