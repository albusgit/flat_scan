#include "cleanup.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "gpu.h"
#include "solid.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// images are BGRA (OpenCV order): channel 0 is blue
inline float luma(float b, float g, float r) { return 0.114f * b + 0.587f * g + 0.299f * r; }

// pull-push fill of values where mask > 0
cv::Mat fill_masked(const cv::Mat& val, const cv::Mat& mask) {
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
                float fallback = up.empty() ? 200.f : up.at<float>(i, j);
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

void drop_dark_islands(const cv::Mat& page, const cv::Mat& lum, cv::Mat& mask) {
    cv::Mat m8, lab, stats, cen;
    mask.convertTo(m8, CV_8U);
    int n = cv::connectedComponentsWithStats(m8, lab, stats, cen, 8, CV_32S);
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

}  // namespace

const char* filter_name(Filter f) {
    switch (f) {
        case Filter::Original: return "Original";
        case Filter::Lighten: return "Lighten";
        case Filter::AutoColor: return "Auto color";
        case Filter::Grayscale: return "Grayscale";
        case Filter::HighContrast: return "High contrast";
        case Filter::BlackWhite: return "Black & white";
    }
    return "?";
}

Curve filter_curve(Filter f, float sigma) {
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

static inline void normalise(const uint8_t* q, float S, const float* wb, float* v) {
    for (int c = 0; c < 3; ++c) v[c] = (float)q[c] / std::max(S * wb[c], 1.f);
}

static const float kInk = 0.55f;   // normalised luminance below which a pixel counts as ink
static const float kSpeckLight = 0.75f;   // speck removal: pixels below this are the speck or its halo
static const float kBgChroma0 = 0.06f, kBgChroma1 = 0.15f;   // background cleanup: chroma max - min of t, fade-out
static const float kPicGamma = 1.25f;   // B&W tone curve inside pictures (= Auto color / Grayscale)

static inline float curve1(const Curve& c, float v) {
    if (c.kind == 1) return std::min(v / c.wp, 1.f);
    return v >= c.wp ? 1.f : std::pow(v / c.wp, c.gamma);
}

PageParams page_params(const cv::Mat& page, bool use_gpu) {
    auto t0 = std::chrono::steady_clock::now();
    PageParams p;
    p.page_w = page.cols; p.page_h = page.rows;
    cv::Mat bg4;
    p.gpu_background = use_gpu && Gpu::instance().background_quarter(page, bg4);
    if (!p.gpu_background) bg4 = background_quarter_cpu(page);
    cv::Mat lum(bg4.size(), CV_32F), chroma(bg4.size(), CV_32F);
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
            lum.at<float>(i, j) = luma(q[0], q[1], q[2]);
            chroma.at<float>(i, j) = (float)(std::max({q[0], q[1], q[2]}) - std::min({q[0], q[1], q[2]}));
        }
    cv::Mat locmax;
    cv::dilate(lum, locmax, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31)));
    cv::Mat mask(bg4.size(), CV_32F);
    int npaper = 0;
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            float l = lum.at<float>(i, j);
            bool m = chroma.at<float>(i, j) < 24.f && l >= 0.88f * locmax.at<float>(i, j) && l > 40.f;
            mask.at<float>(i, j) = m ? 1.f : 0.f;
            npaper += m;
        }
    if (npaper < (int)(0.01 * bg4.total()))
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) mask.at<float>(i, j) = lum.at<float>(i, j) >= 0.9f * locmax.at<float>(i, j);
    drop_dark_islands(page, lum, mask);
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
    // paper noise: MAD of normalised luminance on paper pixels away from ink
    {
        std::vector<float> v;
        for (int y = 1; y < page.rows; y += 3)
            for (int x = 1; x < page.cols; x += 3) {
                int i = std::min(y / 4, mask.rows - 1), j = std::min(x / 4, mask.cols - 1);
                if (mask.at<float>(i, j) <= 0) continue;
                float vv[3];
                normalise(page.ptr<uint8_t>(y) + 4 * x, illum_at(p.illum, (float)x, (float)y), p.wb, vv);
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
    // local background of the normalised page (B&W reference), quarter grid
    p.bgv.create(bg4.size(), CV_32F);
    for (int i = 0; i < bg4.rows; ++i)
        for (int j = 0; j < bg4.cols; ++j) {
            const uint8_t* q = bg4.ptr<uint8_t>(i) + 4 * j;
            float S = p.illum.at<float>(i, j), vv[3];
            normalise(q, S, p.wb, vv);
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
        auto smooth = [](float e0, float e1, float x) {
            float t = std::min(std::max((x - e0) / (e1 - e0), 0.f), 1.f);
            return t * t * (3.f - 2.f * t);
        };
        cv::Mat pic(bg4.size(), CV_32F);
        for (int i = 0; i < bg4.rows; ++i)
            for (int j = 0; j < bg4.cols; ++j) {
                float vv[3];
                normalise(bg4.ptr<uint8_t>(i) + 4 * j, p.illum.at<float>(i, j), p.wb, vv);
                float b = p.bgv.at<float>(i, j);
                float ch = std::max({vv[0], vv[1], vv[2]}) - std::min({vv[0], vv[1], vv[2]});
                pic.at<float>(i, j) = std::max(smooth(0.45f, 0.3f, b), smooth(0.12f, 0.25f, ch) * smooth(0.85f, 0.65f, b));
            }
        cv::blur(pic, p.pic, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);
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
            float al = a[x] / 255.f, S = illum_at(p.illum, (float)x, (float)y);
            for (int c = 0; c < 3; ++c) {
                float pc = std::min(255.f, S * p.wb[c] * 1.01f);   // clean paper (slightly above, so filters clip it)
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
                    float v[3];
                    normalise(img.ptr<uint8_t>(y) + 4 * x, illum_at(p.illum, (float)x, (float)y), p.wb, v);
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

static inline void adjust_px(const AdjustK& k, float* t) {
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
        for (int c = 0; c < 3; ++c) t[c] = t[c] + (1.f - t[c]) * w;
    }
}

cv::Mat filter_cpu(const cv::Mat& img, const PageParams& p, Filter f, float s, const Adjust& adj) {
    if (f == Filter::Original && adj.identity()) return img.clone();
    Curve c = filter_curve(f, p.sigma);
    AdjustK ak = adjust_constants(adj, c.specks);
    bool full = img.cols == p.page_w && img.rows == p.page_h;
    float sx = (float)p.page_w / img.cols, sy = (float)p.page_h / img.rows;
    cv::Mat cnt = full && c.specks ? ink_counts(img, p) : cv::Mat();
    cv::Mat out(img.size(), CV_8UC4);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            const uint8_t* q = img.ptr<uint8_t>(y) + 4 * x;
            float t[3];
            if (c.kind == 0) {
                for (int k = 0; k < 3; ++k) t[k] = (float)q[k] * (1.f / 255.f);
            } else {
                float px = full ? (float)x : (x + 0.5f) * sx - 0.5f, py = full ? (float)y : (y + 0.5f) * sy - 0.5f;
                float v[3];
                normalise(q, illum_at(p.illum, px, py), p.wb, v);
                float l = luma(v[0], v[1], v[2]);
                float b = std::max(illum_at(p.bgv, px, py), 0.05f), w = 0.f;
                if (c.kind == 3) {
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
                    for (int k = 0; k < 3; ++k) t[k] = curve1(c, v[k]);
                }
                // a speck (a few ink pixels with little else around them: a light stroke such as
                // coloured text has many pixels below kSpeckLight) becomes the white of the paper
                // around it, only where that background renders white (not in pictures, colour
                // areas or tinted cells)
                if (!cnt.empty() && l < kSpeckLight && b >= c.speck_bg && w < 0.5f) {
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
            adjust_px(ak, t);
            if (c.gray) t[0] = t[1] = t[2] = luma(t[0], t[1], t[2]);
            uint8_t* o = out.ptr<uint8_t>(y) + 4 * x;
            for (int k = 0; k < 3; ++k) {
                float orig = (float)q[k];
                o[k] = (uint8_t)std::lrint(std::min(255.f, std::max(0.f, orig + s * (t[k] * 255.f - orig))));
            }
            o[3] = 255;
        }
    return out;
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

cv::Mat render_page(const cv::Mat& page, const cv::Mat& paper, const PageParams& p, const RenderParams& rp,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st) {
    RenderStats local;
    RenderStats& s = st ? *st : local;
    const Filter f = rp.filter;
    const float strength = rp.strength;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat src = apply_paper(page, paper, p);
    bool sol = rp.solid && f != Filter::BlackWhite;
    // Solid colors in Grayscale: fill in colour, then desaturate
    Filter ff = (sol && f == Filter::Grayscale) ? Filter::AutoColor : f;
    cv::Mat out;
    bool trivial = ff == Filter::Original && rp.adj.identity();
    s.gpu_filter = !trivial && use_gpu && Gpu::instance().filter(src, p, ff, strength, rp.adj, out);
    if (!s.gpu_filter) out = filter_cpu(src, p, ff, strength, rp.adj);
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
