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

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

cv::Mat background_quarter_cpu(const cv::Mat& rgba) {
    CV_Assert(rgba.type() == CV_8UC4);
    int qw = rgba.cols / 4, qh = rgba.rows / 4;
    cv::Mat small(qh, qw, CV_8UC4);
    for (int i = 0; i < qh; ++i)
        for (int j = 0; j < qw; ++j)
            for (int c = 0; c < 4; ++c) {
                int s = 0;
                for (int r = 0; r < 4; ++r) {
                    const uint8_t* p = rgba.ptr<uint8_t>(4 * i + r) + 16 * j + c;
                    s += p[0] + p[4] + p[8] + p[12];
                }
                small.ptr<uint8_t>(i)[4 * j + c] = (uint8_t)((s + 8) >> 4);
            }
    cv::Mat dil, med;
    cv::dilate(small, dil, cv::Mat::ones(3, 3, CV_8U));   // default border: in-bounds max
    cv::medianBlur(dil, med, 11);                          // BORDER_REPLICATE
    return med;
}

// images are BGRA (OpenCV order): channel 0 is blue
static inline float luma(float b, float g, float r) { return 0.114f * b + 0.587f * g + 0.299f * r; }

// pull-push fill of values where mask > 0
static cv::Mat fill_masked(const cv::Mat& val, const cv::Mat& mask) {
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
        cv::Mat e(V[k].size(), CV_32F);
        cv::Mat up;
        if (!est.empty()) cv::resize(est, up, V[k].size(), 0, 0, cv::INTER_LINEAR);
        for (int i = 0; i < e.rows; ++i)
            for (int j = 0; j < e.cols; ++j) {
                float w = W[k].at<float>(i, j);
                float fallback = up.empty() ? 200.f : up.at<float>(i, j);
                // blend towards the coarser estimate where coverage is thin
                float a = std::min(1.f, w * 4.f);
                e.at<float>(i, j) = w > 0 ? a * (V[k].at<float>(i, j) / w) + (1 - a) * fallback : fallback;
            }
        est = e;
    }
    return est;
}

CleanParams clean_params(const cv::Mat& bg4, const cv::Mat& rgba, bool grayscale) {
    CleanParams p;
    p.grayscale = grayscale;
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
    cv::Mat ill = fill_masked(lum, mask);
    cv::GaussianBlur(ill, p.illum, cv::Size(0, 0), 2.0);
    // white balance: paper colour relative to its luminance
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
    // black point: 0.5th percentile of normalised luminance (ink), capped
    std::vector<float> v;
    for (int y = 0; y < rgba.rows; y += 3)
        for (int x = 0; x < rgba.cols; x += 3) {
            const uint8_t* q = rgba.ptr<uint8_t>(y) + 4 * x;
            float S = p.illum.at<float>(std::min(p.illum.rows - 1, y / 4), std::min(p.illum.cols - 1, x / 4));
            v.push_back(luma(q[0] / (S * p.wb[0]), q[1] / (S * p.wb[1]), q[2] / (S * p.wb[2])));
        }
    if (!v.empty()) {
        size_t k = v.size() / 200;
        std::nth_element(v.begin(), v.begin() + k, v.end());
        p.black = std::min(0.3f, std::max(0.f, v[k] - 0.02f));
    }
    return p;
}

// ---- tone pass: must match kToneCS in gpu.cpp operation for operation
static inline float illum_at(const cv::Mat& I, int x, int y) {
    float fx = (x + 0.5f) * 0.25f - 0.5f, fy = (y + 0.5f) * 0.25f - 0.5f;
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
static const float kDarkish = 0.75f;

cv::Mat ink_counts_cpu(const cv::Mat& rgba, const CleanParams& p) {
    int qw = std::max(1, rgba.cols / 4), qh = std::max(1, rgba.rows / 4);
    cv::Mat cnt(qh, qw, CV_8U, cv::Scalar(0));
    for (int i = 0; i < qh; ++i)
        for (int j = 0; j < qw; ++j) {
            int n = 0;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    int y = 4 * i + r, x = 4 * j + c;
                    if (y >= rgba.rows || x >= rgba.cols) continue;
                    float v[3];
                    normalise(rgba.ptr<uint8_t>(y) + 4 * x, illum_at(p.illum, x, y), p.wb, v);
                    n += luma(v[0], v[1], v[2]) < kInk;
                }
            cnt.at<uint8_t>(i, j) = (uint8_t)n;
        }
    return cnt;
}

cv::Mat tone_cpu(const cv::Mat& rgba, const CleanParams& p) {
    cv::Mat cnt = ink_counts_cpu(rgba, p);
    cv::Mat out(rgba.size(), CV_8UC4);
    const float scale = 1.f / (p.white - p.black);
    for (int y = 0; y < rgba.rows; ++y)
        for (int x = 0; x < rgba.cols; ++x) {
            const uint8_t* q = rgba.ptr<uint8_t>(y) + 4 * x;
            float v[3];
            normalise(q, illum_at(p.illum, x, y), p.wb, v);
            float l = luma(v[0], v[1], v[2]);
            bool speck = false;
            if (l < kDarkish) {
                int bi = std::min(y / 4, cnt.rows - 1), bj = std::min(x / 4, cnt.cols - 1), n = 0;
                for (int di = -4; di <= 4; ++di)
                    for (int dj = -4; dj <= 4; ++dj) {
                        int ii = bi + di, jj = bj + dj;
                        if (ii >= 0 && ii < cnt.rows && jj >= 0 && jj < cnt.cols) n += cnt.at<uint8_t>(ii, jj);
                    }
                speck = n >= 1 && n <= p.speck_max;
            }
            float t[3];
            for (int c = 0; c < 3; ++c) t[c] = speck ? 1.f : std::min(std::max((v[c] - p.black) * scale, 0.f), 1.f);
            uint8_t* o = out.ptr<uint8_t>(y) + 4 * x;
            if (p.grayscale) {
                uint8_t g = (uint8_t)std::lrint(luma(t[0], t[1], t[2]) * 255.f);
                o[0] = o[1] = o[2] = g;
            } else {
                for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::lrint(t[c] * 255.f);
            }
            o[3] = 255;
        }
    return out;
}

cv::Mat cleanup(const cv::Mat& rgba, bool grayscale, bool use_gpu, CleanStats* st) {
    CleanStats local;
    CleanStats& s = st ? *st : local;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat bg4;
    s.gpu_background = use_gpu && Gpu::instance().background_quarter(rgba, bg4);
    if (!s.gpu_background) bg4 = background_quarter_cpu(rgba);
    s.ms_background = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    CleanParams p = clean_params(bg4, rgba, grayscale);
    s.ms_params = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    cv::Mat out;
    s.gpu_tone = use_gpu && Gpu::instance().tone(rgba, p, out);
    if (!s.gpu_tone) out = tone_cpu(rgba, p);
    s.ms_tone = ms_since(t0);
    return out;
}

cv::Mat finish_page(const cv::Mat& page, bool grayscale, bool solid, bool use_gpu, CleanStats* st) {
    if (!solid) return cleanup(page, grayscale, use_gpu, st);
    cv::Mat out = solid_colors(cleanup(page, false, use_gpu, st));
    if (grayscale) {
        for (int y = 0; y < out.rows; ++y)
            for (int x = 0; x < out.cols; ++x) {
                cv::Vec4b& o = out.at<cv::Vec4b>(y, x);
                uint8_t g = (uint8_t)std::lrint(luma(o[0] / 255.f, o[1] / 255.f, o[2] / 255.f) * 255.f);
                o[0] = o[1] = o[2] = g;
            }
    }
    return out;
}

}  // namespace fs
