#include "linefind.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "warpmap.h"

namespace fs {

namespace {

constexpr int kStep = 4;           // px between samples along the line
constexpr int kMaxShift = 3;       // px the line may move across per sample (slopes to ~37 degrees)
constexpr float kMinRidge = 6.f;   // a line stands out this much from both its sides (darkness, 0..255)
constexpr float kMinDark = 8.f;    // and is at least this much darker than the paper
constexpr int kMaxGap = 3;         // samples without the line (12 px) before following it stops
constexpr double kMaxJitter = 0.7;  // px: smoother than this is a line, rougher a text row (see below)

}  // namespace

std::vector<cv::Point2f> find_drawn_line(const cv::Mat& page, const std::vector<cv::Point2f>& stroke) {
    std::vector<cv::Point2f> out;
    if (page.empty() || stroke.size() < 2) return out;
    const int W = page.cols, H = page.rows;
    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
    for (const auto& p : stroke) {
        x0 = std::min(x0, p.x); x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y); y1 = std::max(y1, p.y);
    }
    const bool hz = x1 - x0 >= y1 - y0;   // the line runs along x
    const int len = hz ? W : H, wid = hz ? H : W;
    auto along = [hz](const cv::Point2f& p) { return hz ? p.x : p.y; };
    auto across = [hz](const cv::Point2f& p) { return hz ? p.y : p.x; };
    // how far off the stroke may be, and where a line's sides are (both scale with the page: 1600 px wide)
    const double k = std::min(W, H) / 1600.0;
    const int reach = std::max(20, (int)std::lround(48 * k));
    const int s0 = std::max(3, (int)std::lround(4 * k)), s1 = s0 + std::max(3, (int)std::lround(4 * k));

    // the darkness of the strip the search can reach: the stroke's band across, the whole page along (to follow
    // the line past the stroke's ends)
    const float c_lo = hz ? y0 : x0, c_hi = hz ? y1 : x1;
    const int lo = std::max(0, (int)std::floor(c_lo) - reach - s1 - 2), hi = std::min(wid - 1, (int)std::ceil(c_hi) + reach + s1 + 2);
    if (hi - lo < 2 * s1 + 4) return out;
    const cv::Rect strip = hz ? cv::Rect(0, lo, W, hi - lo + 1) : cv::Rect(lo, 0, hi - lo + 1, H);
    cv::Mat bgr = page(strip);
    if (bgr.channels() == 4) cv::cvtColor(bgr, bgr, cv::COLOR_BGRA2BGR);
    const cv::Mat D = darkness_fast(bgr);
    // darkness at (along a, across c), averaged over 3 px along
    auto dark = [&](int a, int c) {
        float s = 0;
        for (int d = -1; d <= 1; ++d) {
            const int aa = std::min(std::max(a + d, 0), len - 1);
            s += hz ? D.at<float>(c - lo, aa) : D.at<float>(aa, c - lo);
        }
        return s / 3;
    };
    // how much a line at c stands out from both its sides (an edge of a dark area stands out from one only)
    auto ridge = [&](int a, int c, float* centre = nullptr) {
        const float m = dark(a, c);
        if (centre) *centre = m;
        float l = 0, r = 0;
        for (int d = s0; d <= s1; ++d) {
            l += dark(a, std::max(lo, c - d));
            r += dark(a, std::min(hi, c + d));
        }
        const int n = s1 - s0 + 1;
        return m - std::max(l, r) / n;
    };
    const int c_min = lo + s1, c_max = hi - s1;   // where the sides are inside the strip

    // the stroke as a function of `along` (sorted: drawn either way), sampled every kStep px
    std::vector<cv::Point2f> pts = stroke;
    std::sort(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) { return along(a) < along(b); });
    const int a0 = std::max(0, (int)std::ceil(along(pts.front()))), a1 = std::min(len - 1, (int)std::floor(along(pts.back())));
    std::vector<int> A, base;
    size_t seg = 0;
    for (int a = a0; a <= a1; a += kStep) {
        while (seg + 2 < pts.size() && along(pts[seg + 1]) < a) ++seg;
        const cv::Point2f &p = pts[seg], &q = pts[seg + 1];
        const float t = along(q) - along(p) > 1e-6f ? std::min(1.f, std::max(0.f, (a - along(p)) / (along(q) - along(p)))) : 0.f;
        A.push_back(a);
        base.push_back((int)std::lround(across(p) + (across(q) - across(p)) * t));
    }
    const int n = (int)A.size();
    if (n < 6) return out;

    // the smoothest strong path through the band (Viterbi): each sample gains the line's ridge there (a little less
    // the farther from the stroke), each step loses 1 per px it moves across; a letter beside the rule gives a
    // ridge too, but leaving the rule for it costs more than it gains
    const int S = 2 * reach + 1;
    const float NEG = -std::numeric_limits<float>::infinity();
    std::vector<float> score((size_t)n * S, NEG);
    std::vector<int> from((size_t)n * S, -1);
    auto unary = [&](int i, int j) {
        const int c = base[i] - reach + j;
        if (c < c_min || c > c_max) return NEG;
        return std::min(ridge(A[i], c), 60.f) - 0.1f * std::abs(j - reach);
    };
    for (int j = 0; j < S; ++j) score[j] = unary(0, j);
    for (int i = 1; i < n; ++i) {
        const int shift = base[i] - base[i - 1];   // the band moves with the stroke
        for (int j = 0; j < S; ++j) {
            const float u = unary(i, j);
            if (u == NEG) continue;
            float best = NEG;
            int arg = -1;
            for (int d = -kMaxShift; d <= kMaxShift; ++d) {
                const int jp = j + shift + d;   // the same absolute position, d px across
                if (jp < 0 || jp >= S) continue;
                const float v = score[(size_t)(i - 1) * S + jp];
                if (v == NEG) continue;
                const float cand = v - std::abs(d);
                if (cand > best) { best = cand; arg = jp; }
            }
            if (arg < 0) continue;
            score[(size_t)i * S + j] = best + u;
            from[(size_t)i * S + j] = arg;
        }
    }
    int j_end = -1;
    float best = NEG;
    for (int j = 0; j < S; ++j)
        if (score[(size_t)(n - 1) * S + j] > best) { best = score[(size_t)(n - 1) * S + j]; j_end = j; }
    if (j_end < 0) return out;
    std::vector<int> C(n);
    for (int i = n - 1, j = j_end; i >= 0; --i) {
        C[i] = base[i] - reach + j;
        if (i > 0) j = from[(size_t)i * S + j];
        if (j < 0) return out;
    }

    // on a line where it stands out
    auto on_line = [&](int a, int c) {
        float m = 0;
        const float r = ridge(a, c, &m);
        return r >= kMinRidge && m >= kMinDark;
    };
    std::vector<char> on(n);
    for (int i = 0; i < n; ++i) on[i] = on_line(A[i], C[i]);
    // where the path cuts across (more than 1 px in 2 over 16 px: from one line to another, or into text) it is cut;
    // of the pieces, the one on a line for most of its length, running smoothly, near the stroke and covering most
    // of it. Smooth: a path through a text row jumps from letter to letter, its mean distance from its own running
    // mean (9 samples, 36 px) is 1.3-1.5 px on text, under 0.15 px on rules, dotted ones too.
    std::vector<int> smooth_base(n);
    for (int i = 0; i < n; ++i) {
        long m = 0;
        int c = 0;
        for (int d = -4; d <= 4; ++d)
            if (i + d >= 0 && i + d < n) { m += base[i + d]; ++c; }
        smooth_base[i] = (int)(m / c);
    }
    std::vector<char> cut(n, 0);
    for (int i = 4; i < n; ++i)
        if (std::abs(C[i] - C[i - 4]) >= 2 * kStep)
            for (int d = 0; d <= 4; ++d) cut[i - d] = 1;
    int seg_a = -1, seg_b = -1, seg_hits = 0;
    for (int a = 0; a < n;) {
        if (cut[a]) { ++a; continue; }
        int b = a;
        while (b + 1 < n && !cut[b + 1]) ++b;
        const int m = b - a + 1;
        int hits = 0;
        double jitter = 0, offset = 0;
        for (int i = a; i <= b; ++i) {
            hits += on[i];
            double mean = 0;
            int c = 0;
            for (int d = -4; d <= 4; ++d)
                if (i + d >= a && i + d <= b) { mean += C[i + d]; ++c; }
            jitter += std::fabs(C[i] - mean / c);
            offset += std::abs(C[i] - smooth_base[i]);
        }
        const bool ok = m >= 6 && hits >= 6 && hits * 2 >= m && jitter / m <= kMaxJitter && offset / m <= 0.8 * reach && m * 10 >= 4 * n;
        if (ok && hits > seg_hits) { seg_a = a; seg_b = b; seg_hits = hits; }
        a = b + 1;
    }
    if (seg_a < 0) return out;
    // a running median (5) evens out a sample pulled by a letter touching the rule
    std::vector<int> Cm = C;
    for (int i = seg_a; i <= seg_b; ++i) {
        int w[5], m = 0;
        for (int d = -2; d <= 2; ++d)
            if (i + d >= seg_a && i + d <= seg_b) w[m++] = C[i + d];
        std::nth_element(w, w + m / 2, w + m);
        Cm[i] = w[m / 2];
    }
    // the line starts and ends where its ink does
    int first = seg_a, last = seg_b;
    while (first < last && !on[first]) ++first;
    while (last > first && !on[last]) --last;

    // followed past the stroke's ends while it goes on (a sweep over part of a rule finds all of it)
    auto follow = [&](int a, int c, int dir) {
        std::vector<cv::Point2i> ext;
        int gap = 0;
        for (a += dir * kStep; a >= 0 && a < len; a += dir * kStep) {
            int bc = -1;
            float bv = NEG;
            for (int d = -kMaxShift + 1; d <= kMaxShift - 1; ++d) {
                const int cc = c + d;
                if (cc < c_min || cc > c_max) continue;
                const float v = ridge(a, cc) - 0.5f * std::abs(d);
                if (v > bv) { bv = v; bc = cc; }
            }
            if (bc >= 0 && on_line(a, bc)) {
                c = bc;
                gap = 0;
                ext.push_back({a, c});
            } else if (++gap > kMaxGap) {
                break;
            } else {
                ext.push_back({a, -1});   // a gap: kept only if the line comes back
            }
        }
        while (!ext.empty() && ext.back().y < 0) ext.pop_back();
        // through a gap at the position the line came back at (straight enough over 12 px)
        for (int i = (int)ext.size() - 2; i >= 0; --i)
            if (ext[i].y < 0) ext[i].y = ext[i + 1].y;
        return ext;
    };
    std::vector<cv::Point2i> line;
    std::vector<cv::Point2i> head = follow(A[first], Cm[first], -1);
    for (auto it = head.rbegin(); it != head.rend(); ++it) line.push_back(*it);
    for (int i = first; i <= last; ++i) line.push_back({A[i], Cm[i]});
    for (const auto& e : follow(A[last], Cm[last], +1)) line.push_back(e);
    for (const auto& q : line) out.push_back(hz ? cv::Point2f((float)q.x, (float)q.y) : cv::Point2f((float)q.y, (float)q.x));
    return out;
}

}  // namespace fs
