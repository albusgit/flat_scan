#include "solid.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nputil.h"
#include "warpmap.h"

namespace fs {

namespace {

const float kDarkArea = 0.1f;   // measure(): a larger share of dark areas -> picture, not a cell

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

// Lab (L 0..100, a, b) of an 8-bit BGR colour, in float (not the 8-bit-scaled Lab)
cv::Vec3f lab_of(const cv::Vec3f& bgr) {
    cv::Mat m(1, 1, CV_32FC3, cv::Scalar(bgr[0] / 255.f, bgr[1] / 255.f, bgr[2] / 255.f)), o;
    cv::cvtColor(m, o, cv::COLOR_BGR2Lab);
    return o.at<cv::Vec3f>(0, 0);
}
cv::Vec3f bgr_of(const cv::Vec3f& lab) {
    cv::Mat m(1, 1, CV_32FC3, cv::Scalar(lab[0], lab[1], lab[2])), o;
    cv::cvtColor(m, o, cv::COLOR_Lab2BGR);
    cv::Vec3f v = o.at<cv::Vec3f>(0, 0);
    for (int c = 0; c < 3; ++c) v[c] = std::min(255.f, std::max(0.f, v[c] * 255.f));
    return v;
}
inline float chroma(const cv::Vec3f& lab) { return std::hypot(lab[1], lab[2]); }
inline float lum8(const uint8_t* p) { return 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]; }

struct Region {
    int label;                 // in the region label image (enclosed cells) or -1 (blob rectangle)
    cv::Rect box;
    cv::Vec3f bgr, lab;        // measured background colour (median, ink excluded)
    int area = 0, core = 0;
    float noise = 0;           // 10-90% luminance spread of the raw cell texture
    bool fill = false, white = false;
    int cluster = -1;
};

double median_of(std::vector<float>& v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// median colour and luminance spread of the pixels where `use` is set inside box
// Median colour and luminance spread of a cell, measured only well away from its ink: a first
// median over the candidate pixels, then everything clearly darker than that (ink, relative to
// the cell's own colour) is dilated by 3 px and excluded, and the median is taken again.
// `bgr` is the lightly smoothed page (no halftone texture), `raw` the page itself.
// spread: 10-90% luminance range (smooth unevenness counts); detail: the same range of the
// high-pass L - blur(L) (pictures, logos and halftone images have detail, uneven print does not).
// dark_area (regions at least 60 px in both directions): share of the candidate pixels that
// are darker than the median - 14 in areas at least 15 px wide. Ink is strokes and lines of
// text (gone after a 15x15 opening); the darker half of a photo or a smooth gradient is an
// area. Without it a picture lost its darker half to the ink exclusion below, measured as
// uniform and was filled with one flat colour.
bool measure(const cv::Mat& bgr, const cv::Mat& use0, const cv::Rect& box, cv::Vec3f& col, float& spread, int& n,
             const cv::Mat* raw = nullptr, float* raw_spread = nullptr, float* detail = nullptr, float* dark_area = nullptr) {
    cv::Mat use = use0(box).clone(), L(box.size(), CV_32F);
    std::vector<float> l0;
    for (int y = 0; y < box.height; ++y)
        for (int x = 0; x < box.width; ++x) {
            float v = lum8(bgr.ptr<uint8_t>(box.y + y) + 3 * (box.x + x));
            L.at<float>(y, x) = v;
            if (use.at<uint8_t>(y, x)) l0.push_back(v);
        }
    if (l0.size() < 30) return false;
    float m0 = (float)median_of(l0);
    cv::Mat inkm = (L < m0 - 14.f), inkd;
    if (dark_area) *dark_area = 0.f;
    if (dark_area && std::min(box.width, box.height) >= 60) {   // thin rows are filled by their text
        cv::Mat big;
        cv::morphologyEx(inkm, big, cv::MORPH_OPEN, cv::Mat::ones(15, 15, CV_8U));
        *dark_area = (float)cv::countNonZero(big & use) / (float)l0.size();
    }
    cv::dilate(inkm, inkd, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    use.setTo(0, inkd);
    std::vector<float> ch[3], Ls;
    for (int y = 0; y < box.height; ++y)
        for (int x = 0; x < box.width; ++x) {
            if (!use.at<uint8_t>(y, x)) continue;
            const uint8_t* p = bgr.ptr<uint8_t>(box.y + y) + 3 * (box.x + x);
            for (int c = 0; c < 3; ++c) ch[c].push_back(p[c]);
            Ls.push_back(L.at<float>(y, x));
        }
    n = (int)Ls.size();
    if (n < 30) return false;
    for (int c = 0; c < 3; ++c) col[c] = (float)median_of(ch[c]);
    std::sort(Ls.begin(), Ls.end());
    spread = Ls[(size_t)(0.9 * (n - 1))] - Ls[(size_t)(0.1 * (n - 1))];
    if (detail) {
        // blur only over the measured pixels (normalised convolution), so ink does not leak in
        cv::Mat m, Lm, bl, bm;
        use.convertTo(m, CV_32F, 1 / 255.0);
        Lm = L.mul(m);
        cv::GaussianBlur(Lm, bl, cv::Size(0, 0), 6.0);
        cv::GaussianBlur(m, bm, cv::Size(0, 0), 6.0);
        std::vector<float> hp;
        for (int y = 0; y < box.height; ++y)
            for (int x = 0; x < box.width; ++x)
                if (use.at<uint8_t>(y, x) && bm.at<float>(y, x) > 0.2f) hp.push_back(L.at<float>(y, x) - bl.at<float>(y, x) / bm.at<float>(y, x));
        std::sort(hp.begin(), hp.end());
        *detail = hp.size() < 30 ? 0.f : hp[(size_t)(0.9 * (hp.size() - 1))] - hp[(size_t)(0.1 * (hp.size() - 1))];
    }
    if (raw && raw_spread) {   // texture of the unsmoothed pixels: the unmixing noise floor
        std::vector<float> R;
        for (int y = 0; y < box.height; ++y)
            for (int x = 0; x < box.width; ++x)
                if (use.at<uint8_t>(y, x)) R.push_back(lum8(raw->ptr<uint8_t>(box.y + y) + 3 * (box.x + x)));
        std::sort(R.begin(), R.end());
        *raw_spread = R[(size_t)(0.9 * (R.size() - 1))] - R[(size_t)(0.1 * (R.size() - 1))];
    }
    return true;
}

}  // namespace

cv::Mat solid_colors(const cv::Mat& img, const std::vector<Rect01>& protect, SolidStats* st) {
    auto t0 = std::chrono::steady_clock::now();
    const int h = img.rows, w = img.cols;
    cv::Mat bgr, g, bg, diff;
    cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::dilate(g, bg, cv::Mat::ones(9, 9, CV_8U));
    cv::medianBlur(bg, bg, 41);
    cv::subtract(bg, g, diff, cv::noArray(), CV_16S);
    cv::Mat bgr_s;
    cv::GaussianBlur(bgr, bgr_s, cv::Size(0, 0), 1.5);   // colour / spread measured without halftone texture
    // ---- grid lines (clean.py): segments, short segments, inferred borders, wobbly ends, snap
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
    // (a) region separators: line centres only, never drawn; collinear gaps up to ~15 px closed
    cv::Mat mh = draw_lines(h, w, H, {}), mv = draw_lines(h, w, {}, V);
    cv::morphologyEx(mh, mh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 1)));
    cv::morphologyEx(mv, mv, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 17)));
    cv::Mat lines = mh | mv;
    // protected (picture) areas
    cv::Mat prot = cv::Mat::zeros(h, w, CV_8U);
    for (const auto& r : protect)
        cv::rectangle(prot, cv::Point((int)(r.x0 * w), (int)(r.y0 * h)), cv::Point((int)(r.x1 * w), (int)(r.y1 * h)), cv::Scalar(255), cv::FILLED);
    // (d) colour measurement only well away from lines (here) and ink (per cell, in measure())
    cv::Mat away;
    cv::dilate(lines, away, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    away = (away == 0) & (prot == 0);
    cv::Mat lab, stats, cen;
    int n = cv::connectedComponentsWithStats(lines == 0, lab, stats, cen, 4);
    std::vector<Region> regs;
    std::vector<int> reg_of_label(n, -1);
    SolidStats S;
    for (int i = 1; i < n; ++i) {
        int x = stats.at<int>(i, 0), y = stats.at<int>(i, 1), ww = stats.at<int>(i, 2), hh = stats.at<int>(i, 3), ar = stats.at<int>(i, 4);
        if (x == 0 || y == 0 || x + ww >= w || y + hh >= h || ar > 0.3 * h * w || ar < 60) continue;
        cv::Rect R(x, y, ww, hh);
        cv::Mat sl = lab(R) == i, core;
        cv::erode(sl, core, cv::Mat::ones(9, 9, CV_8U));
        cv::Mat use = cv::Mat::zeros(h, w, CV_8U);
        use(R).setTo(255, core & away(R));
        if (cv::countNonZero(use(R)) < 30) use(R).setTo(255, sl & away(R));
        int pin = cv::countNonZero(prot(R) & sl);
        Region rg{i, R, {}, {}, ar};
        float spread = 0;
        S.regions++;
        float detail = 0, dark = 0;
        if (pin > 0.3 * ar || !measure(bgr_s, use, R, rg.bgr, spread, rg.core, &bgr, &rg.noise, &detail, &dark)) { S.skipped++; continue; }
        // (f) pictures, gradients, logos: too much colour variation once ink is removed
        rg.lab = lab_of(rg.bgr);
        if (getenv("FS_SOLID_DEBUG"))
            std::fprintf(stderr, "region %d box %d,%d %dx%d area %d core %d  L %.1f a %.1f b %.1f C %.1f spread %.1f detail %.1f noise %.1f dark %.2f\n", i, x, y,
                         ww, hh, ar, rg.core, rg.lab[0], rg.lab[1], rg.lab[2], chroma(rg.lab), spread, detail, rg.noise, dark);
        // (f) pictures / logos (local detail), designed gradients (large smooth range) and pictures
        // or gradients with large darker areas (measure()) stay as they are
        if (detail > 18.f || spread > 45.f || dark > kDarkArea) { S.skipped_varied++; continue; }
        reg_of_label[i] = (int)regs.size();
        regs.push_back(rg);
    }
    // paper colour: the low-chroma cells near the top of the lightness range (after Auto color
    // this is pure white; with Original it is the page's own paper)
    cv::Vec3d paper(100, 0, 0);
    bool have_paper = false;
    {
        // area-weighted 75th percentile of L, and median a, b, over the low-chroma light cells:
        // the paper itself, not pulled down by grey cells
        std::vector<std::pair<float, double>> Ls, as, bs;
        double wsum = 0;
        for (auto& r : regs)
            if (chroma(r.lab) < 3 && r.lab[0] > 85) {
                Ls.push_back({r.lab[0], (double)r.area}); as.push_back({r.lab[1], (double)r.area}); bs.push_back({r.lab[2], (double)r.area});
                wsum += r.area;
            }
        auto wq = [&](std::vector<std::pair<float, double>> v, double q) {
            std::sort(v.begin(), v.end());
            double acc = 0;
            for (auto& e : v) { acc += e.second; if (acc >= q * wsum) return (double)e.first; }
            return (double)v.back().first;
        };
        if (wsum > 0) { paper = cv::Vec3d(wq(Ls, 0.75), wq(as, 0.5), wq(bs, 0.5)); have_paper = true; }
    }
    // 1/4-resolution background (ink removed) in Lab: the blob search (e) and, on a page
    // without light low-chroma cells, the paper colour. That used to default to pure white,
    // but below strength 100% (or with Original) the paper is not white: a light colour area
    // and the slightly darker paper around it all counted as "coloured" and were merged into
    // one rectangle filled with their greyish average.
    cv::Mat ql;
    {
        cv::Mat q, qd, qm, qf;
        cv::resize(bgr, q, cv::Size(w / 4, h / 4), 0, 0, cv::INTER_AREA);
        cv::dilate(q, qd, cv::Mat::ones(3, 3, CV_8U));
        cv::medianBlur(qd, qm, 11);
        qm.convertTo(qf, CV_32FC3, 1 / 255.0);
        cv::cvtColor(qf, ql, cv::COLOR_BGR2Lab);
    }
    if (!have_paper) {
        std::vector<float> Ls, as, bs;
        for (int y = 0; y < ql.rows; ++y)
            for (int x = 0; x < ql.cols; ++x) {
                cv::Vec3f c = ql.at<cv::Vec3f>(y, x);
                if (chroma(c) < 3 && c[0] > 85) { Ls.push_back(c[0]); as.push_back(c[1]); bs.push_back(c[2]); }
            }
        if (Ls.size() > 0.05 * ql.total()) {
            std::sort(Ls.begin(), Ls.end());
            paper = cv::Vec3d(Ls[(size_t)(0.75 * (Ls.size() - 1))], median_of(as), median_of(bs));
        }
    }
    auto dE_paper = [&](const cv::Vec3f& lab) { return cv::norm(cv::Vec3d(lab) - paper); };
    // (c) candidates differ from paper at all; the fill / white decision is taken per colour
    // cluster below, so cells of one colour are never split between filled and white
    for (auto& r : regs) {
        r.fill = dE_paper(r.lab) >= 3.0;
        r.white = !r.fill;
    }
    // (e) coloured areas not enclosed by lines: large uniform coloured blobs at 1/4 resolution
    {
        cv::Mat taken = cv::Mat::zeros(h, w, CV_8U);
        for (auto& r : regs)
            if (r.fill) taken(r.box).setTo(255, lab(r.box) == r.label);
        cv::Mat takenq, linesq;
        cv::resize(taken, takenq, ql.size(), 0, 0, cv::INTER_AREA);
        cv::Mat col(ql.size(), CV_8U);
        for (int y = 0; y < ql.rows; ++y)
            for (int x = 0; x < ql.cols; ++x) {
                cv::Vec3f c = ql.at<cv::Vec3f>(y, x);
                // a colour (a, b off the paper's), or clearly darker than the paper; a slightly
                // darker paper (a shadow left at strength < 100%) is neither
                double dab = std::hypot(c[1] - paper[1], c[2] - paper[2]);
                bool coloured = c[0] > 35.f && (dab >= 6.0 || (cv::norm(cv::Vec3d(c) - paper) >= 8.0 && c[0] < paper[0] - 10.0));
                col.at<uint8_t>(y, x) = coloured && takenq.at<uint8_t>(y, x) < 64 ? 255 : 0;
            }
        cv::morphologyEx(col, col, cv::MORPH_OPEN, cv::Mat::ones(4, 4, CV_8U));
        cv::morphologyEx(col, col, cv::MORPH_CLOSE, cv::Mat::ones(9, 9, CV_8U));
        cv::Mat bl, bs, bc;
        int nb = cv::connectedComponentsWithStats(col, bl, bs, bc, 4);
        Vec vx, hy;
        for (auto& L : V) vx.push_back(L[0]);
        for (auto& L : H) hy.push_back(L[0]);
        auto snapv = [](double v, const Vec& c, double tol) {
            double best = v, bd = tol;
            for (double q : c) if (std::fabs(q - v) < bd) { bd = std::fabs(q - v); best = q; }
            return (int)std::lround(best);
        };
        for (int i = 1; i < nb; ++i) {
            int x = bs.at<int>(i, 0) * 4, y = bs.at<int>(i, 1) * 4, ww = bs.at<int>(i, 2) * 4, hh = bs.at<int>(i, 3) * 4, ar = bs.at<int>(i, 4) * 16;
            if (ar < 4000 || ar < 0.5 * ww * hh) continue;
            int x0 = snapv(x, vx, 30), x1 = snapv(x + ww, vx, 30), y0 = snapv(y, hy, 30), y1 = snapv(y + hh, hy, 30);
            cv::Rect R = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)) & cv::Rect(0, 0, w, h);
            if (R.area() < 4000) continue;
            cv::Mat use = cv::Mat::zeros(h, w, CV_8U);
            use(R).setTo(255, away(R) & (taken(R) == 0));
            Region rg{-1, R, {}, {}, R.area()};
            float spread = 0, detail = 0, dark = 0;
            if (!measure(bgr_s, use, R, rg.bgr, spread, rg.core, &bgr, &rg.noise, &detail, &dark) || detail > 18.f || spread > 45.f ||
                dark > kDarkArea)
                continue;
            rg.lab = lab_of(rg.bgr);
            if (dE_paper(rg.lab) < 5.0) continue;
            rg.fill = true;
            regs.push_back(rg);
            S.blobs++;
        }
    }
    // (b) one colour per cluster: agglomerative merge in Lab while the closest pair is < dE 7
    struct Cl { cv::Vec3d lab; double wgt; std::vector<int> mem; };
    std::vector<Cl> cl;
    for (size_t k = 0; k < regs.size(); ++k)
        if (regs[k].fill) cl.push_back({cv::Vec3d(regs[k].lab), (double)regs[k].area, {(int)k}});
    while (cl.size() > 1) {
        double bd = 1e30;
        size_t ba = 0, bb = 0;
        for (size_t a = 0; a < cl.size(); ++a)
            for (size_t b = a + 1; b < cl.size(); ++b) {
                double d = cv::norm(cl[a].lab - cl[b].lab);
                if (d < bd) { bd = d; ba = a; bb = b; }
            }
        if (bd >= 7.0) break;
        double W = cl[ba].wgt + cl[bb].wgt;
        cl[ba].lab = (cl[ba].lab * cl[ba].wgt + cl[bb].lab * cl[bb].wgt) / W;
        cl[ba].wgt = W;
        cl[ba].mem.insert(cl[ba].mem.end(), cl[bb].mem.begin(), cl[bb].mem.end());
        cl.erase(cl.begin() + bb);
    }
    std::vector<cv::Vec3f> fillc(regs.size(), cv::Vec3f(255, 255, 255));
    int ncl = 0;
    for (size_t c = 0; c < cl.size(); ++c) {
        // a cluster is filled only if it clearly differs from paper (dE >= 5); else pure white
        bool coloured = cv::norm(cl[c].lab - paper) >= 5.0 &&
                        !(cl[c].lab[0] >= paper[0] - 1.0 && std::hypot(cl[c].lab[1], cl[c].lab[2]) < 5.0);   // whiter than paper
        ncl += coloured;
        for (int k : cl[c].mem) {
            regs[k].cluster = (int)c;
            if (coloured) fillc[k] = bgr_of(cv::Vec3f(cl[c].lab));
            else { regs[k].fill = false; regs[k].white = true; }
        }
    }
    S.clusters = ncl;
    // ---- fill map: region index per pixel, propagated into the line-centre pixels (h)
    cv::Mat owner(h, w, CV_32F, cv::Scalar(0));     // region index + 1
    for (size_t k = 0; k < regs.size(); ++k) {
        const Region& r = regs[k];
        if (!r.fill && !r.white) continue;
        if (r.label >= 0) owner(r.box).setTo((float)(k + 1), lab(r.box) == r.label);
        else owner(r.box).setTo((float)(k + 1), owner(r.box) == 0);
        if (r.fill) S.filled++;
        else S.white++;
    }
    for (int it = 0; it < 4; ++it) {
        cv::Mat d;
        cv::dilate(owner, d, cv::Mat::ones(3, 3, CV_8U));
        d.copyTo(owner, (owner == 0) & (lines > 0));
    }
    // ---- composite (d, g): unmix each pixel into ink colour I (darkest pixel nearby) and the
    // cell's original background B, then re-mix I with the new fill. Anti-aliased ink keeps its
    // coverage, coloured ink keeps its colour, and no trace of the old background is left.
    cv::Mat out = img.clone();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int k = (int)owner.at<float>(y, x) - 1;
            if (k < 0 || prot.at<uint8_t>(y, x)) continue;
            const Region& r = regs[k];
            const cv::Vec3f& F = fillc[k];
            const uint8_t* P = bgr.ptr<uint8_t>(y) + 3 * x;
            float lb = 0.114f * r.bgr[0] + 0.587f * r.bgr[1] + 0.299f * r.bgr[2], lp = lum8(P);
            const uint8_t* I = P;
            float li = lp;
            for (int dy = -2; dy <= 2; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (int dx = -2; dx <= 2; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    const uint8_t* Q = bgr.ptr<uint8_t>(yy) + 3 * xx;
                    float lq = lum8(Q);
                    if (lq < li) { li = lq; I = Q; }
                }
            }
            // below the cell's own texture level a pixel is background (flattened to the fill)
            float f0 = std::max(6.f, 0.6f * r.noise);
            float t = 0.f;
            if (lb - li >= f0 + 12.f && lb - lp > f0) t = std::min(1.f, std::max(0.f, (lb - lp - f0) / (lb - li - f0)));
            uint8_t* o = out.ptr<uint8_t>(y) + 4 * x;
            for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::lrint(t * I[c] + (1.f - t) * F[c]);
        }
    S.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (st) *st = S;
    return out;
}

}  // namespace fs
