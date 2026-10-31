#include "cleanup.h"
#include "board.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "gpu.h"
#include "solid.h"
#include "nputil.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// images are BGRA (OpenCV order): channel 0 is blue
inline float luma(float b, float g, float r) { return 0.114f * b + 0.587f * g + 0.299f * r; }

inline float smoothstep(float e0, float e1, float x) {
    float t = std::min(std::max((x - e0) / (e1 - e0), 0.f), 1.f);
    return t * t * (3.f - 2.f * t);
}

// pull-push fill of values where mask > 0 (fallback: the value where the coarsest level has too few)
cv::Mat fill_masked(const cv::Mat& val, const cv::Mat& mask, float fallback0 = 200.f) {
    std::vector<cv::Mat> V{val.mul(mask)}, W{mask.clone()};
    while (V.back().cols > 2 && V.back().rows > 2) {
        cv::Mat v2, w2;
        cv::Size s((V.back().cols + 1) / 2, (V.back().rows + 1) / 2);
        cv::resize(V.back(), v2, s, 0, 0, cv::INTER_AREA);
        cv::resize(W.back(), w2, s, 0, 0, cv::INTER_AREA);
        V.push_back(v2); W.push_back(w2);
    }
    cv::Mat est;
    for (int k = (int)V.size() - 1; k >= 0; --k) {
        cv::Mat e(V[k].size(), CV_32F), up;
        if (!est.empty()) cv::resize(est, up, V[k].size(), 0, 0, cv::INTER_LINEAR);
        for (int i = 0; i < e.rows; ++i)
            for (int j = 0; j < e.cols; ++j) {
                float w = W[k].at<float>(i, j);
                float fallback = up.empty() ? fallback0 : up.at<float>(i, j);
                float a = std::min(1.f, w * 4.f);   // blend towards the coarser estimate where coverage is thin
                e.at<float>(i, j) = w > 0 ? a * (V[k].at<float>(i, j) / w) + (1 - a) * fallback : fallback;
            }
        est = e;
    }
    return est;
}

// Paper samples come from the local brightness maximum (31x31 quarter px), so the inside of any
// neutral area larger than that window - a grey or dark photo, the light middle of a picture,
// a grey table cell - is its own local maximum and is taken for paper: the illumination dips
// there and the area turns white (a hole). Such samples form islands that are not connected
// to the page's main paper (the largest component). An island is kept only if
//   * its level is that of the paper around it (median of lum / S0 >= kIslandLevel, S0 = the
//     illumination filled in from the main paper alone): paper cut off by a band or a frame, or
//   * it looks like a cast shadow: it reaches the page edge (shadows come from outside the
//     page; pictures and cells are inside it) and it is plain paper at full resolution (at
//     least kIslandFlat of its pixels within 8% of the local background; text on paper is
//     sparse, photos and halftones are not).
// Everything else is left to the interpolation from the paper around it.
const float kIslandLevel = 0.85f, kIslandFlat = 0.7f;
const int kIslandBridge = 11;

void drop_dark_islands(const cv::Mat& page, const cv::Mat& lum, cv::Mat& mask) {
    // pieces of paper split only by a thin seam that is not a paper sample (the colour-edge
    // ring of paper_mask along a crease or a shadow edge, ~2 kEdgeR px) are one piece: the
    // components are taken on the mask dilated by kIslandBridge
    cv::Mat m8, md, lab, stats, cen;
    mask.convertTo(m8, CV_8U);
    cv::dilate(m8, md, cv::Mat::ones(kIslandBridge, kIslandBridge, CV_8U));
    int n = cv::connectedComponentsWithStats(md, lab, stats, cen, 8, CV_32S);
    lab.setTo(0, m8 == 0);
    if (n <= 2) return;
    int main_c = 1;
    for (int k = 2; k < n; ++k)
        if (stats.at<int>(k, cv::CC_STAT_AREA) > stats.at<int>(main_c, cv::CC_STAT_AREA)) main_c = k;
    cv::Mat mmain(mask.size(), CV_32F);
    for (int i = 0; i < mask.rows; ++i)
        for (int j = 0; j < mask.cols; ++j) mmain.at<float>(i, j) = lab.at<int>(i, j) == main_c ? 1.f : 0.f;
    cv::Mat S0 = fill_masked(lum, mmain);
    std::vector<std::vector<float>> ratio(n);
    std::vector<int> flat(n, 0), tot(n, 0);
    for (int i = 0; i < mask.rows; ++i)
        for (int j = 0; j < mask.cols; ++j) {
            int k = lab.at<int>(i, j);
            if (k == 0 || k == main_c) continue;
            float L = lum.at<float>(i, j);
            ratio[k].push_back(L / std::max(S0.at<float>(i, j), 1.f));
            for (int dy = 0; dy < 4; dy += 2)        // every other full-resolution pixel of the block
                for (int dx = 0; dx < 4; dx += 2) {
                    const uint8_t* q = page.ptr<uint8_t>(4 * i + dy) + 4 * (4 * j + dx);
                    float v = luma(q[0], q[1], q[2]) / std::max(L, 1.f);
                    flat[k] += v > 0.92f && v < 1.08f;
                    tot[k]++;
                }
        }
    std::vector<char> keep(n, 1);
    for (int k = 1; k < n; ++k) {
        if (k == main_c) continue;
        std::vector<float>& r = ratio[k];
        std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
        int x = stats.at<int>(k, cv::CC_STAT_LEFT), y = stats.at<int>(k, cv::CC_STAT_TOP);
        int w = stats.at<int>(k, cv::CC_STAT_WIDTH), h = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        bool edge = x == 0 || y == 0 || x + w >= mask.cols || y + h >= mask.rows;
        keep[k] = r[r.size() / 2] >= kIslandLevel || (edge && flat[k] >= kIslandFlat * tot[k]);
    }
    for (int i = 0; i < mask.rows; ++i)
        for (int j = 0; j < mask.cols; ++j)
            if (!keep[lab.at<int>(i, j)]) mask.at<float>(i, j) = 0.f;
}

// Colour of the (ink-free) background relative to the paper, independent of the lighting: the
// chromaticity bg / luma(bg), divided by the paper's chromaticity ref, scaled to a fixed light
// level (L ~ 94) and converted to Lab. A, B: its a, b; paper is (0, 0) whatever its own tint.
void rel_ab(const cv::Mat& bg4, const cv::Mat& lum, const float* ref, cv::Mat& A, cv::Mat& B) {
    cv::Mat f(bg4.size(), CV_32FC3), lab;
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
            float l = std::max(lum.at<float>(i, j), 1.f);
            cv::Vec3f& o = f.at<cv::Vec3f>(i, j);
            for (int c = 0; c < 3; ++c) o[c] = std::min(1.f, 0.85f * (float)q[c] / (l * ref[c]));
        }
    cv::cvtColor(f, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    A = ch[1];
    B = ch[2];
}

// The candidates are bright only next to their own neighbourhood (a 31 px window of the quarter
// grid). Inside a large picture that is the picture itself: the light sky of an illustration is its
// own local maximum, and where the paper is a little blue or grey too (cool light) its colour
// passed for paper, so the illumination took the sky's brightness and Auto color whitened it, with
// a halo where the sky darkens. Such a candidate is dimmer than the paper of its part of the page
// (below kPageBright of the 90th percentile of the candidates in cells of an eighth of the short
// side, the largest of them within kPageCells cells; a percentile: glare and stickers are small)
// AND a colour of its own (more than dim_c in a, b from the paper colour; the sky 5-7) AND in a
// picture: within kPicCells cells of a saturated mid or dark colour area (chroma above kPicC, below
// kPicLevel of the paper level: the deep blue of the same sky, the ground). A shadow is dim and,
// under mixed light, tinted too (pink, blue: measured on the shadow sets), but lies on paper, so
// shadowed paper stays a sample and the filter still evens it out where the shadow net left some.
const float kPageBright = 0.92f, kDimC = 4.f, kPicC = 12.f, kPicLevel = 0.85f;
const int kPageCells = 2, kPicCells = 2;

void drop_dim_candidates(const cv::Mat& lum, const cv::Mat& A, const cv::Mat& B, float dim_c, cv::Mat& cand) {
    const int cs = std::max(8, std::min(lum.rows, lum.cols) / 8);
    const int gh = (lum.rows + cs - 1) / cs, gw = (lum.cols + cs - 1) / cs;
    cv::Mat level(gh, gw, CV_32F, cv::Scalar(0.f));
    std::vector<float> v;
    for (int gi = 0; gi < gh; ++gi)
        for (int gj = 0; gj < gw; ++gj) {
            v.clear();
            for (int i = gi * cs; i < std::min(lum.rows, (gi + 1) * cs); ++i)
                for (int j = gj * cs; j < std::min(lum.cols, (gj + 1) * cs); ++j)
                    if (cand.at<uint8_t>(i, j)) v.push_back(lum.at<float>(i, j));
            if (v.size() < 16) continue;
            std::nth_element(v.begin(), v.begin() + v.size() * 9 / 10, v.end());
            level.at<float>(gi, gj) = v[v.size() * 9 / 10];
        }
    cv::dilate(level, level, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kPageCells + 1, 2 * kPageCells + 1)));
    cv::Mat full;
    cv::resize(level, full, cv::Size(gw * cs, gh * cs), 0, 0, cv::INTER_LINEAR);
    // saturated mid / dark areas (opened by a third of a cell: not headline letters or a rule), widened by kPicCells cells
    cv::Mat pic(lum.size(), CV_8U);
    for (int i = 0; i < lum.rows; ++i)
        for (int j = 0; j < lum.cols; ++j)
            pic.at<uint8_t>(i, j) = hypot_xy(A.at<float>(i, j), B.at<float>(i, j)) > kPicC && lum.at<float>(i, j) < kPicLevel * full.at<float>(i, j) ? 1 : 0;
    cv::morphologyEx(pic, pic, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(std::max(5, cs / 3) | 1, std::max(5, cs / 3) | 1)));
    cv::dilate(pic, pic, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4 * kPicCells * cs + 1, 4 * kPicCells * cs + 1)));
    for (int i = 0; i < lum.rows; ++i)
        for (int j = 0; j < lum.cols; ++j)
            if (cand.at<uint8_t>(i, j) && pic.at<uint8_t>(i, j) && lum.at<float>(i, j) < kPageBright * full.at<float>(i, j) &&
                hypot_xy(A.at<float>(i, j), B.at<float>(i, j)) > dim_c)
                cand.at<uint8_t>(i, j) = 0;
}

// Paper samples, by colour. The old test (chroma max - min < 24 levels, absolute) took every
// bright, smooth area of low chroma for paper: light yellow / blue / pink boxes, pale photos
// (whose own brightness then became the illumination, so they were flattened to white), and
// on a page under warm light it rejected the paper itself. Paper is near-neutral or uniformly
// tinted; a highlighter or a pastel box is a distinct colour relative to the paper around it:
//  1. the paper colour is the most common colour of the bright candidates (a 2-D histogram of
//     a, b relative to grey, votes weighted by brightness^4: paper is the brightest surface);
//  2. seeds: candidates within kSeedC (dE in a, b) of it;
//  3. breadth-first growth from the seeds over 4-neighbour candidates. Every grown sample
//     carries the paper colour of its path (a running average, weight kPathAlpha); a
//     neighbour joins if its colour is within kStepC of it and within kMaxC of the paper
//     colour, and it is not on a colour edge (a, b change > kEdgeC across +-kEdgeR px). A
//     colour cast that drifts slowly across the page (mixed lighting) is followed; the edge
//     of a pastel box, a highlighted line or a pale photo is not crossed;
//  4. pieces not connected to the main paper: see below.
// A, B: background colour relative to the paper (rel_ab). Returns the mask (CV_32F 0 / 1).
const float kSeedC = 3.f, kStepC = 2.5f, kPathAlpha = 0.1f, kMaxC = 40.f, kEdgeC = 4.f, kLocalC = 5.f, kRefSigma = 8.f;
const float kNoiseEdge = 2.5f, kNoiseStep = 1.5f, kNoiseTint = 2.f;   // thresholds >= these x colour noise
const int kEdgeR = 3;

cv::Mat paper_mask(const cv::Mat& bg4, const cv::Mat& lum, const cv::Mat& cand0, cv::Mat& A, cv::Mat& B, float& noise) {
    cv::Mat cand = cand0.clone();
    const float one[3] = {1.f, 1.f, 1.f};
    rel_ab(bg4, lum, one, A, B);
    float lmax = 1.f;
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j)
            if (cand.at<uint8_t>(i, j)) lmax = std::max(lmax, lum.at<float>(i, j));
    const int NB = 80;   // a, b in [-40, 40), 1 unit bins
    cv::Mat hist = cv::Mat::zeros(NB, NB, CV_32F);
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            if (!cand.at<uint8_t>(i, j)) continue;
            int ia = (int)std::floor(A.at<float>(i, j)) + NB / 2, ib = (int)std::floor(B.at<float>(i, j)) + NB / 2;
            if (ia < 0 || ib < 0 || ia >= NB || ib >= NB) continue;
            float r = lum.at<float>(i, j) / lmax;
            hist.at<float>(ib, ia) += r * r * r * r;
        }
    cv::GaussianBlur(hist, hist, cv::Size(0, 0), 2.5);
    cv::Point mx;
    cv::minMaxLoc(hist, nullptr, nullptr, nullptr, &mx);
    float a0 = (float)(mx.x - NB / 2) + 0.5f, b0 = (float)(mx.y - NB / 2) + 0.5f;
    // paper chromaticity: median of the candidates near the mode
    float ref[3] = {1.f, 1.f, 1.f};
    {
        std::vector<float> r[3];
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                if (!cand.at<uint8_t>(i, j) || hypot_xy(A.at<float>(i, j) - a0, B.at<float>(i, j) - b0) >= 4.f) continue;
                const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
                float l = std::max(lum.at<float>(i, j), 1.f);
                for (int c = 0; c < 3; ++c) r[c].push_back(q[c] / l);
            }
        if (!r[0].empty())
            for (int c = 0; c < 3; ++c) {
                std::nth_element(r[c].begin(), r[c].begin() + r[c].size() / 2, r[c].end());
                ref[c] = std::min(1.5f, std::max(0.6f, r[c][r[c].size() / 2]));
            }
    }
    rel_ab(bg4, lum, ref, A, B);
    // colour noise of the background (texture of magazine paper, JPEG chroma blocks, sensor
    // noise in the dark): 90th percentile of the a, b difference between candidates 2 px
    // apart. The colour thresholds grow with it, so noisy paper is not split into pieces.
    {
        std::vector<float> dn;
        for (int i = 0; i < bg4.rows; i += 2)
            for (int j = 0; j + 2 < bg4.cols; j += 2)
                if (cand.at<uint8_t>(i, j) && cand.at<uint8_t>(i, j + 2))
                    dn.push_back(hypot_xy(A.at<float>(i, j + 2) - A.at<float>(i, j), B.at<float>(i, j + 2) - B.at<float>(i, j)));
        noise = 0.f;
        if (dn.size() > 100) {
            std::nth_element(dn.begin(), dn.begin() + dn.size() * 9 / 10, dn.end());
            noise = dn[dn.size() * 9 / 10];
        }
    }
    drop_dim_candidates(lum, A, B, std::max(kDimC, kNoiseTint * noise), cand);
    const float edge_c = std::max(kEdgeC, kNoiseEdge * noise), step_c = std::max(kStepC, kNoiseStep * noise);
    // colour edges: the a, b difference across +-kEdgeR px (4 directions, lightly smoothed
    // first) above edge_c. A ring of them surrounds every coloured area, corners included.
    cv::Mat edge(bg4.size(), CV_8U);
    {
        cv::Mat As, Bs;
        cv::GaussianBlur(A, As, cv::Size(0, 0), 1.5);
        cv::GaussianBlur(B, Bs, cv::Size(0, 0), 1.5);
        const int ex[4] = {kEdgeR, 0, kEdgeR, kEdgeR}, ey[4] = {0, kEdgeR, kEdgeR, -kEdgeR};
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                float g = 0.f;
                for (int k = 0; k < 4; ++k) {
                    int i0 = std::min(std::max(i - ey[k], 0), bg4.rows - 1), j0 = std::min(std::max(j - ex[k], 0), bg4.cols - 1);
                    int i1 = std::min(std::max(i + ey[k], 0), bg4.rows - 1), j1 = std::min(std::max(j + ex[k], 0), bg4.cols - 1);
                    g = std::max(g, hypot_xy(As.at<float>(i1, j1) - As.at<float>(i0, j0), Bs.at<float>(i1, j1) - Bs.at<float>(i0, j0)));
                }
                edge.at<uint8_t>(i, j) = g > edge_c ? 1 : 0;
            }
    }
    // seeds, then breadth-first growth over candidates that are not colour edges; every grown
    // sample carries the paper colour of its path (R, a running average), and a neighbour
    // joins if its colour is within kStepC of that
    cv::Mat grown = cv::Mat::zeros(bg4.size(), CV_8U), RA(bg4.size(), CV_32F), RB(bg4.size(), CV_32F);
    std::vector<cv::Point> queue;
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j)
            if (cand.at<uint8_t>(i, j) && hypot_xy(A.at<float>(i, j), B.at<float>(i, j)) < kSeedC) {
                grown.at<uint8_t>(i, j) = 1;
                RA.at<float>(i, j) = A.at<float>(i, j);
                RB.at<float>(i, j) = B.at<float>(i, j);
                queue.emplace_back(j, i);
            }
    const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    for (size_t h = 0; h < queue.size(); ++h) {
        cv::Point p = queue[h];
        float ra = RA.at<float>(p), rb = RB.at<float>(p);
        for (int k = 0; k < 4; ++k) {
            cv::Point q(p.x + dx[k], p.y + dy[k]);
            if (q.x < 0 || q.y < 0 || q.x >= bg4.cols || q.y >= bg4.rows) continue;
            if (grown.at<uint8_t>(q) || !cand.at<uint8_t>(q) || edge.at<uint8_t>(q)) continue;
            float aq = A.at<float>(q), bq = B.at<float>(q);
            if (hypot_xy(aq - ra, bq - rb) < step_c && hypot_xy(aq, bq) < kMaxC) {
                grown.at<uint8_t>(q) = 1;
                RA.at<float>(q) = ra + kPathAlpha * (aq - ra);
                RB.at<float>(q) = rb + kPathAlpha * (bq - rb);
                queue.push_back(q);
            }
        }
    }
    // 4. pieces of the grown area that are not connected to the main paper (the largest piece)
    // are kept only if their colour (median a, b) is within kLocalC of the main paper's colour
    // filled in at their place. Under a colour cast, a pastel area can match the global paper
    // colour somewhere (a blue box under warm light) and seed itself; next to the paper
    // around it, it is still a distinct colour. Paper cut off by a picture or a band is kept.
    cv::Mat lab, stats, cen;
    int n = cv::connectedComponentsWithStats(grown, lab, stats, cen, 8, CV_32S);
    if (n > 2) {
        int main_c = 1;
        for (int k = 2; k < n; ++k)
            if (stats.at<int>(k, cv::CC_STAT_AREA) > stats.at<int>(main_c, cv::CC_STAT_AREA)) main_c = k;
        cv::Mat mmain(bg4.size(), CV_32F);
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) mmain.at<float>(i, j) = lab.at<int>(i, j) == main_c ? 1.f : 0.f;
        cv::Mat Ar = fill_masked(A, mmain, 0.f), Br = fill_masked(B, mmain, 0.f);
        cv::GaussianBlur(Ar, Ar, cv::Size(0, 0), kRefSigma);
        cv::GaussianBlur(Br, Br, cv::Size(0, 0), kRefSigma);
        std::vector<std::vector<float>> d(n);
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                int k = lab.at<int>(i, j);
                if (k > 0 && k != main_c)
                    d[k].push_back(hypot_xy(A.at<float>(i, j) - Ar.at<float>(i, j), B.at<float>(i, j) - Br.at<float>(i, j)));
            }
        std::vector<char> keep(n, 1);
        for (int k = 1; k < n; ++k) {
            if (k == main_c || d[k].empty()) continue;
            std::nth_element(d[k].begin(), d[k].begin() + d[k].size() / 2, d[k].end());
            keep[k] = d[k][d[k].size() / 2] < kLocalC;
        }
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j)
                if (!keep[lab.at<int>(i, j)]) grown.at<uint8_t>(i, j) = 0;
    }
    cv::Mat mask;
    grown.convertTo(mask, CV_32F);
    return mask;
}

}  // namespace

const char* filter_name(Filter f) {
    switch (f) {
        case Filter::Original: return "Original";
        case Filter::Lighten: return "Lighten";
        case Filter::AutoColor: return "Auto color";
        case Filter::Grayscale: return "Grayscale";
        case Filter::HighContrast: return "High contrast";
        case Filter::BlackWhite: return "Black & white";
        case Filter::Scan: return "Scan";
        case Filter::Whiteboard: return "Whiteboard";
        case Filter::ProjectorDark: return "Dark slide";
    }
    return "?";
}

ScanTune& scan_tune() {
    static ScanTune t;
    return t;
}

Curve filter_curve(Filter f, float sigma, float ink) {
    Curve c;
    switch (f) {
        case Filter::Original: c.kind = 0; break;
        case Filter::Lighten: c.kind = 1; c.wp = 1.f - 1.5f * sigma; break;
        case Filter::AutoColor: c.kind = 2; c.wp = 1.f - 2.5f * sigma; c.gamma = 1.25f; c.specks = true; break;
        case Filter::Grayscale: c.kind = 2; c.wp = 1.f - 2.5f * sigma; c.gamma = 1.25f; c.gray = true; c.specks = true; break;
        case Filter::HighContrast: c.kind = 2; c.wp = 1.f - 3.f * sigma; c.gamma = 1.8f; c.specks = true; break;
        // on tinted cells (noisier than white paper) the white point drops to 0.92 so their texture
        // does not turn into speckle; faint strokes (~0.85) still map to visible grey
        case Filter::BlackWhite: c.kind = 3; c.wp = 1.f - 3.f * sigma; c.wp_tint = std::min(c.wp, 0.92f); c.gamma = 2.5f; c.gray = true; c.specks = true; break;
        // Scan: lighting flattened at any strength (its base is Lighten), then a luminance curve
        // from a black point under the page's own ink level to the white point; colours are scaled
        // with their luminance, so hue and saturation are kept (cleanup.h)
        case Filter::Scan: {
            const ScanTune& t = scan_tune();
            c.kind = 4; c.wp = 1.f - std::min(t.wp_k * sigma, t.wp_cap); c.wp_base = 1.f - std::min(1.5f * sigma, t.wp_cap); c.gamma = t.gamma;
            c.black = std::min(t.black_max, std::max(0.f, t.black_k * ink));
            c.sharpen = t.sharpen; c.specks = true;
            break;
        }
    }
    // specks are removed where the local background renders white: at or above the white point
    // in the tone curves; anywhere outside pictures in B&W (it maps the local background to white)
    c.speck_bg = c.kind == 3 ? 0.f : c.wp;
    return c;
}

cv::Mat background_quarter_cpu(const cv::Mat& bgra) {
    CV_Assert(bgra.type() == CV_8UC4);
    int qw = bgra.cols / 4, qh = bgra.rows / 4;
    cv::Mat small(qh, qw, CV_8UC4);
    for (int i = 0; i < qh; ++i)
        for (int j = 0; j < qw; ++j)
            for (int c = 0; c < 4; ++c) {
                int s = 0;
                for (int r = 0; r < 4; ++r) {
                    const uint8_t* p = bgra.ptr<uint8_t>(4 * i + r) + 16 * j + c;
                    s += p[0] + p[4] + p[8] + p[12];
                }
                small.ptr<uint8_t>(i)[4 * j + c] = (uint8_t)((s + 8) >> 4);
            }
    cv::Mat dil, med;
    cv::dilate(small, dil, cv::Mat::ones(3, 3, CV_8U));   // default border: in-bounds max
    cv::medianBlur(dil, med, 11);                          // BORDER_REPLICATE
    return med;
}

// ---- per-pixel kernel, mirrored operation for operation by kToneCS in gpu.cpp
static inline float illum_at(const cv::Mat& I, float px, float py) {
    float fx = (px + 0.5f) * 0.25f - 0.5f, fy = (py + 0.5f) * 0.25f - 0.5f;
    fx = std::min(std::max(fx, 0.f), (float)(I.cols - 1));
    fy = std::min(std::max(fy, 0.f), (float)(I.rows - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = std::min(x0 + 1, I.cols - 1), y1 = std::min(y0 + 1, I.rows - 1);
    float a = fx - (float)x0, b = fy - (float)y0;
    return (I.at<float>(y0, x0) * (1.f - a) + I.at<float>(y0, x1) * a) * (1.f - b) +
           (I.at<float>(y1, x0) * (1.f - a) + I.at<float>(y1, x1) * a) * b;
}

// paper colour per channel at a page pixel (PageParams::paper, bilinear like illum_at)
static inline void paper_at(const cv::Mat& P, float px, float py, float* o) {
    float fx = (px + 0.5f) * 0.25f - 0.5f, fy = (py + 0.5f) * 0.25f - 0.5f;
    fx = std::min(std::max(fx, 0.f), (float)(P.cols - 1));
    fy = std::min(std::max(fy, 0.f), (float)(P.rows - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = std::min(x0 + 1, P.cols - 1), y1 = std::min(y0 + 1, P.rows - 1);
    float a = fx - (float)x0, b = fy - (float)y0;
    const cv::Vec4f &p00 = P.at<cv::Vec4f>(y0, x0), &p01 = P.at<cv::Vec4f>(y0, x1), &p10 = P.at<cv::Vec4f>(y1, x0), &p11 = P.at<cv::Vec4f>(y1, x1);
    for (int c = 0; c < 3; ++c) o[c] = (p00[c] * (1.f - a) + p01[c] * a) * (1.f - b) + (p10[c] * (1.f - a) + p11[c] * a) * b;
}

static inline void normalise(const uint8_t* q, const float* P, float* v) {
    for (int c = 0; c < 3; ++c) v[c] = (float)q[c] / std::max(P[c], 1.f);
}

// ink on paper: a paper sample within 2 quarter px (text sits on paper, between the samples)
static inline bool dens_paper(const cv::Mat& mask, int i, int j) {
    for (int di = -2; di <= 2; di += 2)
        for (int dj = -2; dj <= 2; dj += 2) {
            int ii = std::min(std::max(i + di, 0), mask.rows - 1), jj = std::min(std::max(j + dj, 0), mask.cols - 1);
            if (mask.at<float>(ii, jj) > 0) return true;
        }
    return false;
}

static const float kInk = 0.55f;   // normalised luminance below which a pixel counts as ink
static const float kSpeckLight = 0.75f;   // speck removal: pixels below this are the speck or its halo
static const float kBgChroma0 = 0.06f, kBgChroma1 = 0.15f;   // background cleanup: chroma max - min of t, fade-out
static const float kPicGamma = 1.25f;   // B&W tone curve inside pictures (= Auto color / Grayscale)
static const float kTint0 = 3.5f, kTint1 = 6.5f;   // light-colour weight: a, b distance from the local paper colour

// wp: the filter's white point, raised towards 1 on light colours (tint)
static inline float curve1(const Curve& c, float v, float wp) {
    if (c.kind == 1) return std::min(v / wp, 1.f);
    return v >= wp ? 1.f : std::pow(v / wp, c.gamma);
}

PageParams page_params(const cv::Mat& page, bool use_gpu) {
    auto t0 = std::chrono::steady_clock::now();
    PageParams p;
    p.page_w = page.cols; p.page_h = page.rows;
    cv::Mat bg4;
    p.gpu_background = use_gpu && Gpu::instance().background_quarter(page, bg4);
    if (!p.gpu_background) bg4 = background_quarter_cpu(page);
    cv::Mat lum(bg4.size(), CV_32F);
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
            lum.at<float>(i, j) = luma(q[0], q[1], q[2]);
        }
    cv::Mat locmax;
    cv::dilate(lum, locmax, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31)));
    // bright candidates (near the local brightness maximum); which of them are paper is decided
    // by their colour relative to the paper's own colour (paper_mask)
    cv::Mat cand(bg4.size(), CV_8U), A, B;
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            float l = lum.at<float>(i, j);
            cand.at<uint8_t>(i, j) = l >= 0.88f * locmax.at<float>(i, j) && l > 40.f ? 1 : 0;
        }
    float cnoise = 0.f;
    cv::Mat mask = paper_mask(bg4, lum, cand, A, B, cnoise);
    int npaper = cv::countNonZero(mask);
    if (npaper < (int)(0.01 * bg4.total()))
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) mask.at<float>(i, j) = lum.at<float>(i, j) >= 0.9f * locmax.at<float>(i, j);
    drop_dark_islands(page, lum, mask);
    p.samples = mask;
    cv::Mat ill = fill_masked(lum, mask);
    cv::GaussianBlur(ill, p.illum, cv::Size(0, 0), 2.0);
    // white balance from the paper samples of the (ink-free) background
    for (int c = 0; c < 3; ++c) {
        std::vector<float> r;
        for (int i = 0; i < bg4.rows; i += 2)
            for (int j = 0; j < bg4.cols; j += 2)
                if (mask.at<float>(i, j) > 0) r.push_back(bg4.ptr<uint8_t>(i)[4 * j + c] / std::max(1.f, lum.at<float>(i, j)));
        if (!r.empty()) {
            std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
            p.wb[c] = std::min(1.5f, std::max(0.6f, r[r.size() / 2]));
        }
    }
    // paper colour per channel, P = S x local white balance: the chromaticity of the paper
    // samples, filled in and smoothed (kRefSigma) like the paper colour reference of the
    // mask. A colour cast that changes across the page (a lamp on one side, daylight on the
    // other) is removed everywhere, not only on average; coloured areas are not paper samples
    // and get the white balance of the paper around them.
    p.paper.create(bg4.size(), CV_32FC4);
    {
        cv::Mat r[3];
        for (int c = 0; c < 3; ++c) {
            r[c].create(bg4.size(), CV_32F);
            for (int i = 0; i < bg4.rows; ++i)
                for (int j = 0; j < bg4.cols; ++j) r[c].at<float>(i, j) = bg4.ptr<uint8_t>(i)[4 * j + c] / std::max(1.f, lum.at<float>(i, j));
            r[c] = fill_masked(r[c], mask, p.wb[c]);
            cv::GaussianBlur(r[c], r[c], cv::Size(0, 0), kRefSigma);
        }
        // far from any paper sample the filled-in colour is a guess (the light may change
        // colour across a shadow edge); there the global white balance is used
        cv::Mat dens;
        cv::GaussianBlur(mask, dens, cv::Size(0, 0), kRefSigma);
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                float S = p.illum.at<float>(i, j), wl = smoothstep(0.05f, 0.3f, dens.at<float>(i, j));
                cv::Vec4f& P = p.paper.at<cv::Vec4f>(i, j);
                for (int c = 0; c < 3; ++c) P[c] = S * std::min(1.5f, std::max(0.6f, p.wb[c] + (r[c].at<float>(i, j) - p.wb[c]) * wl));
                P[3] = 0.f;
            }
    }
    // paper noise: MAD of normalised luminance on paper pixels away from ink
    {
        std::vector<float> v;
        for (int y = 1; y < page.rows; y += 3)
            for (int x = 1; x < page.cols; x += 3) {
                int i = std::min(y / 4, mask.rows - 1), j = std::min(x / 4, mask.cols - 1);
                if (mask.at<float>(i, j) <= 0) continue;
                float vv[3], P[3];
                paper_at(p.paper, (float)x, (float)y, P);
                normalise(page.ptr<uint8_t>(y) + 4 * x, P, vv);
                float l = luma(vv[0], vv[1], vv[2]);
                if (l > 0.8f) v.push_back(l);   // ink excluded
            }
        if (v.size() > 100) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            float med = v[v.size() / 2];
            for (float& q : v) q = std::fabs(q - med);
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            p.sigma = std::min(0.05f, std::max(0.008f, 1.4826f * v[v.size() / 2]));
        }
    }
    // ink level (Scan's black point): median luma(v) of ink pixels (below kInk) on paper
    {
        std::vector<float> v;
        for (int y = 1; y < page.rows; y += 3)
            for (int x = 1; x < page.cols; x += 3) {
                int i = std::min(y / 4, mask.rows - 1), j = std::min(x / 4, mask.cols - 1);
                float vv[3], P[3];
                paper_at(p.paper, (float)x, (float)y, P);
                normalise(page.ptr<uint8_t>(y) + 4 * x, P, vv);
                float l = luma(vv[0], vv[1], vv[2]);
                if (l < 0.55f && dens_paper(mask, i, j)) v.push_back(l);
            }
        if (v.size() > 200) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            p.ink = v[v.size() / 2];
        }
    }
    // local background of the normalised page (B&W reference), quarter grid
    p.bgv.create(bg4.size(), CV_32F);
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
            float vv[3];
            normalise(q, &p.paper.at<cv::Vec4f>(i, j)[0], vv);
            p.bgv.at<float>(i, j) = luma(vv[0], vv[1], vv[2]);
        }
    // picture weight (B&W), quarter grid. B&W thresholds every pixel against its local
    // background, which inside a dark or coloured area is that area itself: a photo comes out
    // as white with outlines, a dark box with light text as noise. Where the local background
    // (bg4, ink-free) is dark (normalised luminance below 0.45 -> 0.3) or a saturated mid or dark
    // colour (chroma of the normalised background above 0.12 -> 0.25 and luminance below 0.85
    // -> 0.65), B&W renders the page's own tones instead. Text never sets the weight (bg4 has
    // no ink; coloured text on paper has a paper background); pale tints and shadows on paper
    // that the illumination left over stay thresholded (white).
    {
        cv::Mat pic(bg4.size(), CV_32F);
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                float vv[3];
                normalise(bg4.ptr<uint8_t>(i) + 4 * j, &p.paper.at<cv::Vec4f>(i, j)[0], vv);
                float b = p.bgv.at<float>(i, j);
                float ch = std::max({vv[0], vv[1], vv[2]}) - std::min({vv[0], vv[1], vv[2]});
                pic.at<float>(i, j) = std::max(smoothstep(0.45f, 0.3f, b), smoothstep(0.12f, 0.25f, ch) * smoothstep(0.85f, 0.65f, b));
            }
        cv::blur(pic, p.pic, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    }
    // light-colour weight (tint), quarter grid: the local background is a distinct colour
    // relative to the paper around it (a, b distance from the paper colour filled in from the
    // paper samples: kTint0 -> kTint1) and light (bgv 0.7 -> 0.85). There the tone curves do not
    // clip the paper-noise band to white, background cleanup does not whiten and specks are
    // not removed: a light yellow, blue or pink area or a pale photo keeps its colour.
    {
        cv::Mat Af = fill_masked(A, mask, 0.f), Bf = fill_masked(B, mask, 0.f);
        cv::GaussianBlur(Af, Af, cv::Size(0, 0), kRefSigma);
        cv::GaussianBlur(Bf, Bf, cv::Size(0, 0), kRefSigma);
        cv::Mat tint(bg4.size(), CV_32F);
        const float tint0 = std::max(kTint0, kNoiseTint * cnoise);   // noisy paper: a larger distance
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                float d = hypot_xy(A.at<float>(i, j) - Af.at<float>(i, j), B.at<float>(i, j) - Bf.at<float>(i, j));
                tint.at<float>(i, j) = smoothstep(tint0, tint0 + (kTint1 - kTint0), d) * smoothstep(0.7f, 0.85f, p.bgv.at<float>(i, j));
            }
        cv::blur(tint, p.tint, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    }
    p.ms = ms_since(t0);
    return p;
}

// The paper alpha comes from the page outline, which follows light, low-chroma paper. Besides
// the thin slivers of background along the outline (what the fill is for), it also leaves out
// page content that reaches the edge of the page: coloured or dark pictures, colour bands,
// shadowed paper with text. Filling those would erase them in every filter. Only the thin
// parts of the non-paper area are filled: whatever survives a morphological opening with a
// disc of radius 2.5% of the short page side (a region more than 5% of the page thick) is
// kept as it is.
static cv::Mat fill_alpha(const cv::Mat& paper) {
    int qw = std::max(1, paper.cols / 4), qh = std::max(1, paper.rows / 4);
    cv::Mat q, thick;
    cv::resize(paper, q, cv::Size(qw, qh), 0, 0, cv::INTER_AREA);
    cv::Mat nonpaper = q < 128;
    int r = std::max(1, (int)std::lround(0.025 * std::min(paper.cols, paper.rows) / 4.0));
    // outside the image counts as paper, so a sliver along the border stays thin
    cv::morphologyEx(nonpaper, thick, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1)),
                     cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    if (cv::countNonZero(thick) == 0) return paper;
    cv::Mat up, a;
    cv::resize(thick, up, paper.size(), 0, 0, cv::INTER_LINEAR);
    cv::max(paper, up, a);
    return a;
}

cv::Mat apply_paper(const cv::Mat& page, const cv::Mat& paper0, const PageParams& p) {
    if (paper0.empty()) return page;
    CV_Assert(paper0.size() == page.size());
    cv::Mat paper = fill_alpha(paper0);
    cv::Mat out = page.clone();
    for (int y = 0; y < out.rows; ++y) {
        uint8_t* o = out.ptr<uint8_t>(y);
        const uint8_t* a = paper.ptr<uint8_t>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (a[x] == 255) continue;
            float al = a[x] / 255.f, P[3];
            paper_at(p.paper, (float)x, (float)y, P);
            for (int c = 0; c < 3; ++c) {
                float pc = std::min(255.f, P[c] * 1.01f);   // clean paper (slightly above, so filters clip it)
                o[4 * x + c] = (uint8_t)std::lrint(o[4 * x + c] * al + pc * (1.f - al));
            }
        }
    }
    return out;
}

// per 4x4 block: ink pixels (luma(v) < kInk) + (pixels below kSpeckLight) << 16
static cv::Mat ink_counts(const cv::Mat& img, const PageParams& p) {
    int qw = std::max(1, img.cols / 4), qh = std::max(1, img.rows / 4);
    cv::Mat cnt(qh, qw, CV_32S, cv::Scalar(0));
    for (int i = 0; i < qh; ++i)
        for (int j = 0; j < qw; ++j) {
            int n = 0;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    int y = 4 * i + r, x = 4 * j + c;
                    if (y >= img.rows || x >= img.cols) continue;
                    float v[3], P[3];
                    paper_at(p.paper, (float)x, (float)y, P);
                    normalise(img.ptr<uint8_t>(y) + 4 * x, P, v);
                    float l = luma(v[0], v[1], v[2]);
                    n += (l < kInk ? 1 : 0) + (l < kSpeckLight ? 65536 : 0);
                }
            cnt.at<int>(i, j) = n;
        }
    return cnt;
}

Adjust Adjust::clamped() const {
    Adjust a;
    auto cl = [](float v, float lo, float hi, float def) { return std::isfinite(v) ? std::min(hi, std::max(lo, v)) : def; };
    a.bg_cleanup = cl(bg_cleanup, 0.f, 1.f, 0.f);
    a.black = cl(black, 0.f, 0.5f, 0.f);
    a.white = cl(white, 0.5f, 1.f, 1.f);
    a.gamma = cl(gamma, 0.3f, 3.f, 1.f);
    if (a.white - a.black < 0.02f) a.white = std::min(1.f, a.black + 0.02f);
    return a;
}

AdjustK adjust_constants(const Adjust& a0, bool specks) {
    Adjust a = a0.clamped();
    AdjustK k;
    k.levels = !a.levels_identity();
    k.black = a.black;
    k.inv_range = 1.f / (a.white - a.black);
    k.inv_gamma = 1.f / a.gamma;
    k.bg = a.bg_cleanup > 0.f;
    float kk = 0.02f + 0.16f * a.bg_cleanup;
    k.knee_lo = 1.f - 2.f * kk;
    k.knee_hi = 1.f - kk;
    k.speck_max = specks ? (k.bg ? 2 + (int)std::lround(10.f * a.bg_cleanup) : 2) : 0;
    k.speck_light = 4 * k.speck_max + 4;
    return k;
}

static inline void adjust_px(const AdjustK& k, float* t, float tint) {
    if (k.levels)
        for (int c = 0; c < 3; ++c) {
            float u = std::min(std::max((t[c] - k.black) * k.inv_range, 0.f), 1.f);
            t[c] = std::pow(u, k.inv_gamma);
        }
    if (k.bg) {
        float l = luma(t[0], t[1], t[2]);
        float e = std::min(std::max((l - k.knee_lo) / (k.knee_hi - k.knee_lo), 0.f), 1.f);
        float w = e * e * (3.f - 2.f * e);
        // only near-neutral pixels (grey artefacts on paper): light colours (yellow, pastel
        // areas, the light parts of pictures) would turn into white holes
        float ch = std::max({t[0], t[1], t[2]}) - std::min({t[0], t[1], t[2]});
        float f = std::min(std::max((ch - kBgChroma0) / (kBgChroma1 - kBgChroma0), 0.f), 1.f);
        w = w * (1.f - f * f * (3.f - 2.f * f));
        // nor where the local background is a light colour (tint): a pale pastel area has a
        // per-pixel chroma near the noise level, but it is not a grey artefact on paper
        w = w * (1.f - tint);
        for (int c = 0; c < 3; ++c) t[c] = t[c] + (1.f - t[c]) * w;
    }
}

// Scan (kind 4), mirrored by kToneCS in gpu.cpp. v: the normalised pixel (paper ~ 1), l = luma(v),
// tint: light-colour weight, pic: picture weight, s: strength. Output t (0..1) per channel:
//   base  the Lighten curve min(v / wb, 1) (lighting flattened, paper white, ink untouched)
//   enh   luminance curve f(l) = ((l - bk) / (wp - bk))^gamma (1 above wp), bk = black x (1 - pic)
//         (pictures keep their shadows), applied by scaling the pixel: v x f(l) / l keeps hue and
//         saturation; near white (l from wp - 2 sigma.. wp) a near-neutral pixel (chroma below
//         kScanC0 -> kScanC1) fades to neutral f(l) except on light colours (tint), so paper noise and
//         colour casts turn white and pastels stay
//   t     base + s (enh - base)
static const float kScanC0 = 0.04f, kScanC1 = 0.10f;
static const float kPaperWhite0 = 0.96f;   // the filtered value from which the strength blend fades to full (gpu.cpp)   // Scan: chroma (max - min of v) of the neutral fade-out
static inline void scan_px(const Curve& c, const float* v, float l, float tint, float pic, float s, float* t) {
    const float wpe = c.wp + (1.f - c.wp) * tint, wbe = c.wp_base + (1.f - c.wp_base) * tint;
    const float bk = c.black * (1.f - pic);
    float f = l >= wpe ? 1.f : std::pow(std::min(std::max((l - bk) / (wpe - bk), 0.f), 1.f), c.gamma);
    float h = f / std::max(l, 1e-3f);
    float e[3], mx = 0.f;
    for (int k = 0; k < 3; ++k) e[k] = v[k] * h, mx = std::max(mx, e[k]);
    if (mx > 1.f)
        for (int k = 0; k < 3; ++k) e[k] /= mx;
    float band = std::max(0.02f, 2.f * (1.f - c.wp) / 2.5f);   // 2 sigma
    // ... only near-neutral pixels: a light colour that the tint weight missed keeps its colour
    float ch = std::max({v[0], v[1], v[2]}) - std::min({v[0], v[1], v[2]});
    float wn = smoothstep(wpe - band, wpe, l) * (1.f - tint) * (1.f - smoothstep(kScanC0, kScanC1, ch));
    for (int k = 0; k < 3; ++k) {
        float en = e[k] + (f - e[k]) * wn;
        float b = std::min(v[k] / wbe, 1.f);
        t[k] = b + s * (en - b);
    }
}

cv::Mat filter_cpu(const cv::Mat& img, const PageParams& p, Filter f, float s, const Adjust& adj) {
    if (is_board(f)) return board_filter(img, f == Filter::ProjectorDark, s);
    if (f == Filter::Original && adj.identity()) return img.clone();
    Curve c = filter_curve(f, p.sigma, p.ink);
    AdjustK ak = adjust_constants(adj, c.specks);
    bool full = img.cols == p.page_w && img.rows == p.page_h;
    float sx = (float)p.page_w / img.cols, sy = (float)p.page_h / img.rows;
    cv::Mat cnt = full && c.specks ? ink_counts(img, p) : cv::Mat();
    cv::Mat out(img.size(), CV_8UC4);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            const uint8_t* q = img.ptr<uint8_t>(y) + 4 * x;
            float t[3], tint = 0.f;
            if (c.kind == 0) {
                for (int k = 0; k < 3; ++k) t[k] = (float)q[k] * (1.f / 255.f);
            } else {
                float px = full ? (float)x : (x + 0.5f) * sx - 0.5f, py = full ? (float)y : (y + 0.5f) * sy - 0.5f;
                float v[3], P[3];
                paper_at(p.paper, px, py, P);
                normalise(q, P, v);
                float l = luma(v[0], v[1], v[2]);
                float b = std::max(illum_at(p.bgv, px, py), 0.05f), w = 0.f;
                tint = illum_at(p.tint, px, py);
                if (c.kind == 4) {
                    w = illum_at(p.pic, px, py);
                    scan_px(c, v, l, tint, w, s, t);
                } else if (c.kind == 3) {
                    float r = l / b;
                    float wp = b < 0.97f ? c.wp_tint : c.wp;
                    float u = std::min(std::max((r - 0.5f) / (wp - 0.5f), 0.f), 1.f);
                    float tr = std::pow(u, c.gamma);
                    // pictures and dark areas: their own tones (the Auto color curve) instead of a
                    // threshold against their own background, which turns them white
                    w = illum_at(p.pic, px, py);
                    float ta = l >= c.wp ? 1.f : std::pow(l / c.wp, kPicGamma);
                    t[0] = t[1] = t[2] = tr + (ta - tr) * w;
                } else {
                    // light colours: no clipping band below white, so their lighter channels are
                    // not flattened to 1 along with the paper noise
                    float wpe = c.wp + (1.f - c.wp) * tint;
                    for (int k = 0; k < 3; ++k) t[k] = curve1(c, v[k], wpe);
                }
                // a speck (a few ink pixels with little else around them: a light stroke such as
                // coloured text has many pixels below kSpeckLight) becomes the white of the paper
                // around it, only where that background renders white (not in pictures, colour
                // areas or tinted cells)
                if (!cnt.empty() && l < kSpeckLight && b >= c.speck_bg && w < 0.5f && tint < 0.5f) {
                    int bi = std::min(y / 4, cnt.rows - 1), bj = std::min(x / 4, cnt.cols - 1), n = 0;
                    for (int di = -4; di <= 4; ++di)
                        for (int dj = -4; dj <= 4; ++dj) {
                            int ii = bi + di, jj = bj + dj;
                            if (ii >= 0 && ii < cnt.rows && jj >= 0 && jj < cnt.cols) n += cnt.at<int>(ii, jj);
                        }
                    int ink = n & 65535, light = n >> 16;
                    if (ink >= 1 && ink <= ak.speck_max && light <= ak.speck_light) t[0] = t[1] = t[2] = 1.f;
                }
            }
            adjust_px(ak, t, tint);
            if (c.gray) t[0] = t[1] = t[2] = luma(t[0], t[1], t[2]);
            uint8_t* o = out.ptr<uint8_t>(y) + 4 * x;
            float se = c.kind == 4 ? 1.f : s;   // Scan applies the strength inside its curve
            // paper the curve made white is white at any strength (round 18: at 0.7 the paper stayed grey,
            // a 200 paper level rendered 238; CamScanner's is white); ink and colours keep the strength blend
            if (c.kind != 0 && c.kind != 4 && se < 1.f) se += (1.f - se) * smoothstep(kPaperWhite0, 1.f, std::min({t[0], t[1], t[2]}));
            for (int k = 0; k < 3; ++k) {
                float orig = (float)q[k];
                o[k] = (uint8_t)std::lrint(std::min(255.f, std::max(0.f, orig + se * (t[k] * 255.f - orig))));
            }
            o[3] = 255;
        }
    post_filter(out, p, c, s);
    return out;
}

void post_filter(cv::Mat& out, const PageParams& p, const Curve& c, float s) {
    if (c.kind != 4 || c.sharpen <= 0.f || s <= 0.f) return;
    float sigma = scan_tune().sharpen_sigma * (float)out.cols / (float)std::max(1, p.page_w);
    if (sigma < 0.4f) return;   // small thumbnails: nothing to sharpen at this scale
    sharpen_luma(out, c.sharpen * s, sigma);
}

// Unsharp mask of the luminance, limited to the 3x3 range of the pixel's neighbours (no overshoot
// beyond the darkest / lightest neighbour: no halos, no white rims around text on colour); the
// luminance change is added to every channel, so the chroma is unchanged.
void sharpen_luma(cv::Mat& bgra, float amount, float sigma) {
    CV_Assert(bgra.type() == CV_8UC4);
    cv::Mat L(bgra.size(), CV_32F);
    for (int y = 0; y < bgra.rows; ++y) {
        const uint8_t* q = bgra.ptr<uint8_t>(y);
        float* l = L.ptr<float>(y);
        for (int x = 0; x < bgra.cols; ++x, q += 4) l[x] = luma(q[0], q[1], q[2]);
    }
    cv::Mat B, mn, mx;
    cv::GaussianBlur(L, B, cv::Size(0, 0), sigma, sigma, cv::BORDER_REPLICATE);
    cv::erode(L, mn, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    cv::dilate(L, mx, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    for (int y = 0; y < bgra.rows; ++y) {
        uint8_t* q = bgra.ptr<uint8_t>(y);
        const float *l = L.ptr<float>(y), *b = B.ptr<float>(y), *lo = mn.ptr<float>(y), *hi = mx.ptr<float>(y);
        for (int x = 0; x < bgra.cols; ++x, q += 4) {
            float l2 = std::min(hi[x], std::max(lo[x], l[x] + amount * (l[x] - b[x])));
            float d = l2 - l[x];
            if (d == 0.f) continue;
            for (int k = 0; k < 3; ++k) q[k] = (uint8_t)std::lrint(std::min(255.f, std::max(0.f, (float)q[k] + d)));
        }
    }
}

cv::Mat render_page(const cv::Mat& page, const cv::Mat& paper, const PageParams& p, Filter f, float strength, bool solid,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st) {
    RenderParams rp;
    rp.filter = f; rp.strength = strength; rp.solid = solid;
    return render_page(page, paper, p, rp, protect, use_gpu, st);
}

std::vector<int> luma_histogram(const cv::Mat& img) {
    std::vector<int> h(256, 0);
    for (int y = 0; y < img.rows; ++y) {
        const uint8_t* q = img.ptr<uint8_t>(y);
        for (int x = 0; x < img.cols; ++x, q += 4) h[std::min(255, (int)std::lrint(luma(q[0], q[1], q[2])))]++;
    }
    return h;
}

// the paper luminance of the brightest paper (97th percentile of S): where the page is not in shadow
static float paper_ref(const PageParams& p) {
    std::vector<float> v;
    v.reserve(p.illum.total());
    for (int y = 0; y < p.illum.rows; ++y) {
        const float* r = p.illum.ptr<float>(y);
        for (int x = 0; x < p.illum.cols; ++x) if (r[x] > 1.f) v.push_back(r[x]);
    }
    if (v.empty()) return 0.f;
    auto k = v.begin() + (ptrdiff_t)(0.97 * (v.size() - 1));
    std::nth_element(v.begin(), k, v.end());
    return *k;
}

cv::Mat deshadow_page(const cv::Mat& img, const PageParams& p, float amount, PageParams* out_pp) {
    const float a = std::min(std::max(amount, 0.f), 1.f);
    const float sref = paper_ref(p);
    if (out_pp) *out_pp = p;
    if (a <= 0.f || sref <= 0.f || p.paper.empty()) return img;
    const float Pm[3] = {sref * p.wb[0], sref * p.wb[1], sref * p.wb[2]};
    // the gain on the quarter grid: (Pm / P)^a, at most 4x (a deep shadow's noise) and at least 0.8 (paper above the reference)
    cv::Mat G(p.paper.size(), CV_32FC4, cv::Scalar(1, 1, 1, 1));
    for (int i = 0; i < G.rows; ++i)
        for (int j = 0; j < G.cols; ++j) {
            const cv::Vec4f P = p.paper.at<cv::Vec4f>(i, j);
            cv::Vec4f& g = G.at<cv::Vec4f>(i, j);
            for (int c = 0; c < 3; ++c) g[c] = std::min(4.f, std::max(0.8f, std::pow(Pm[c] / std::max(P[c], 1.f), a)));
        }
    if (out_pp) {
        // P' = P x gain and S' = S^(1 - a) Sref^a: v = I' / P' is the filters' v as before
        for (int i = 0; i < G.rows; ++i)
            for (int j = 0; j < G.cols; ++j) {
                cv::Vec4f& P = out_pp->paper.at<cv::Vec4f>(i, j);
                const cv::Vec4f g = G.at<cv::Vec4f>(i, j);
                for (int c = 0; c < 3; ++c) P[c] *= g[c];
            }
        out_pp->paper = out_pp->paper.clone();
        cv::Mat S = p.illum.clone();
        for (int i = 0; i < S.rows; ++i) {
            float* r = S.ptr<float>(i);
            for (int j = 0; j < S.cols; ++j) r[j] = std::pow(std::max(r[j], 1.f), 1.f - a) * std::pow(sref, a);
        }
        out_pp->illum = S;
    }
    cv::Mat out(img.size(), CV_8UC4);
    const bool full = img.cols == p.page_w && img.rows == p.page_h;
    const float sx = (float)p.page_w / img.cols, sy = (float)p.page_h / img.rows;
    cv::parallel_for_(cv::Range(0, img.rows), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; ++y) {
            const uint8_t* q = img.ptr<uint8_t>(y);
            uint8_t* o = out.ptr<uint8_t>(y);
            const float py = full ? (float)y : (y + 0.5f) * sy - 0.5f;
            for (int x = 0; x < img.cols; ++x, q += 4, o += 4) {
                const float px = full ? (float)x : (x + 0.5f) * sx - 0.5f;
                float g[3];
                paper_at(G, px, py, g);
                for (int c = 0; c < 3; ++c) o[c] = cv::saturate_cast<uint8_t>(q[c] * g[c]);
                o[3] = q[3];
            }
        }
    });
    return out;
}

void deepen_shadows(cv::Mat& out, const PageParams& p, float contrast) {
    const float k = 2.f * std::min(std::max(contrast, 0.f), 1.f);
    const float sref = paper_ref(p);
    if (k <= 0.f || sref <= 0.f || p.illum.empty()) return;
    const bool full = out.cols == p.page_w && out.rows == p.page_h;
    const float sx = (float)p.page_w / out.cols, sy = (float)p.page_h / out.rows;
    cv::parallel_for_(cv::Range(0, out.rows), [&](const cv::Range& rg) {
        for (int y = rg.start; y < rg.end; ++y) {
            uint8_t* o = out.ptr<uint8_t>(y);
            const float py = full ? (float)y : (y + 0.5f) * sy - 0.5f;
            for (int x = 0; x < out.cols; ++x, o += 4) {
                const float px = full ? (float)x : (x + 0.5f) * sx - 0.5f;
                // the shadow's depth: 0 on paper at the reference brightness, 1 at 40 % of it or darker
                const float d = std::min(std::max((sref - illum_at(p.illum, px, py)) / (0.6f * sref), 0.f), 1.f);
                if (d <= 0.f) continue;
                // ink only: what renders near paper (above 85 %) keeps its level, the deepening grows below 55 %
                const float l = luma(o[0], o[1], o[2]) * (1.f / 255.f);
                const float ink = 1.f - std::min(std::max((l - 0.55f) / 0.30f, 0.f), 1.f);
                const float e = 1.f + k * d * ink;
                if (e <= 1.f) continue;
                for (int c = 0; c < 3; ++c) o[c] = cv::saturate_cast<uint8_t>(255.f * std::pow(o[c] * (1.f / 255.f), e));
            }
        }
    });
}

cv::Mat filter_with_shadows(const cv::Mat& img, const PageParams& p, const RenderParams& rp) {
    const bool sh = !rp.shadow.identity() && rp.filter != Filter::Original && !is_board(rp.filter);
    if (!sh) return filter_cpu(img, p, rp.filter, rp.strength, rp.adj);
    PageParams q;
    cv::Mat src = deshadow_page(img, p, rp.shadow.amount, &q);
    cv::Mat out = filter_cpu(src, q, rp.filter, rp.strength, rp.adj);
    deepen_shadows(out, p, rp.shadow.contrast);
    return out;
}

cv::Mat render_page(const cv::Mat& page, const cv::Mat& paper, const PageParams& p_in, const RenderParams& rp,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st) {
    RenderStats local;
    RenderStats& s = st ? *st : local;
    const Filter f = rp.filter;
    const float strength = rp.strength;
    auto t0 = std::chrono::steady_clock::now();
    if (is_board(f)) {
        // whiteboard mode: no paper fill (a board has no paper outline), no Solid colors, CPU
        cv::Mat out = board_filter(page, f == Filter::ProjectorDark, strength);
        s.ms_filter = ms_since(t0);
        return out;
    }
    cv::Mat src = apply_paper(page, paper, p_in);
    // the shadows' amount: the page and its params lit flat together (ShadowAdjust); Original shows the photo as it is
    const bool shadows = !rp.shadow.identity() && f != Filter::Original;
    PageParams p_sh;
    if (shadows && rp.shadow.amount > 0.f) src = deshadow_page(src, p_in, rp.shadow.amount, &p_sh);
    const PageParams& p = shadows && rp.shadow.amount > 0.f ? p_sh : p_in;
    bool sol = rp.solid && f != Filter::BlackWhite;
    // Solid colors in Grayscale: fill in colour, then desaturate
    Filter ff = (sol && f == Filter::Grayscale) ? Filter::AutoColor : f;
    cv::Mat out;
    bool trivial = ff == Filter::Original && rp.adj.identity();
    s.gpu_filter = !trivial && use_gpu && Gpu::instance().filter(src, p, ff, strength, rp.adj, out);
    if (!s.gpu_filter) out = filter_cpu(src, p, ff, strength, rp.adj);
    else post_filter(out, p, filter_curve(ff, p.sigma, p.ink), strength);   // CPU stages after the GPU curve (Scan's sharpening)
    if (shadows) deepen_shadows(out, p_in, rp.shadow.contrast);
    s.ms_filter = ms_since(t0);
    if (sol) {
        t0 = std::chrono::steady_clock::now();
        SolidStats ss;
        out = solid_colors(out, protect, &ss);
        s.solid_regions = ss.regions;
        s.solid_clusters = ss.clusters;
        s.solid = ss;
        if (f == Filter::Grayscale)
            for (int y = 0; y < out.rows; ++y)
                for (int x = 0; x < out.cols; ++x) {
                    cv::Vec4b& o = out.at<cv::Vec4b>(y, x);
                    uint8_t g = (uint8_t)std::lrint(luma(o[0], o[1], o[2]));
                    o[0] = o[1] = o[2] = g;
                }
        s.ms_solid = ms_since(t0);
    }
    return out;
}

}  // namespace fs
