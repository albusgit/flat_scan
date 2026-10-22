#include "screen.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace fs {

const float kScreenThreshold = 2.4f;   // docs/results/r11/filters.md: no paper page of 217 at or above it

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// numpy's fftshift (DC to (cols / 2, rows / 2)) or ifftshift (inverse): a circular roll
void fftshift(cv::Mat& m, bool inverse = false) {
    int w = m.cols, h = m.rows;
    int sx = inverse ? w - w / 2 : w / 2, sy = inverse ? h - h / 2 : h / 2;   // element i moves to (i + s) mod n
    cv::Mat out(m.size(), m.type());
    int ax = w - sx, ay = h - sy;   // source blocks [0, ax) and [ax, w)
    if (ax > 0 && ay > 0) m(cv::Rect(0, 0, ax, ay)).copyTo(out(cv::Rect(sx, sy, ax, ay)));
    if (sx > 0 && ay > 0) m(cv::Rect(ax, 0, sx, ay)).copyTo(out(cv::Rect(0, sy, sx, ay)));
    if (ax > 0 && sy > 0) m(cv::Rect(0, ay, ax, sy)).copyTo(out(cv::Rect(sx, 0, ax, sy)));
    if (sx > 0 && sy > 0) m(cv::Rect(ax, ay, sx, sy)).copyTo(out(cv::Rect(0, 0, sx, sy)));
    m = out;
}

// log(1 + |F|) of a real image, centred (DC at (cols / 2, rows / 2))
cv::Mat log_spectrum(const cv::Mat& f32) {
    cv::Mat F, planes[2], mag;
    cv::dft(f32, F, cv::DFT_COMPLEX_OUTPUT);
    cv::split(F, planes);
    cv::magnitude(planes[0], planes[1], mag);
    mag += 1.f;
    cv::log(mag, mag);
    fftshift(mag);
    return mag;
}

// strongest peak of log magnitude above its k x k median, at radius >= rmin (in bins), off the axes
float peak_prominence(const cv::Mat& L, int ksize, double rmin_frac_of_n, float scale) {
    cv::Mat q, med;
    L.convertTo(q, CV_8U, scale);
    cv::medianBlur(q, med, ksize);
    int h = L.rows, w = L.cols;
    float best = 0.f;
    for (int y = 0; y < h; ++y) {
        const float* l = L.ptr<float>(y);
        const uint8_t* m = med.ptr<uint8_t>(y);
        int dy = y - h / 2;
        if (std::abs(dy) <= 1) continue;
        for (int x = 0; x < w; ++x) {
            int dx = x - w / 2;
            if (std::abs(dx) <= 1) continue;
            double fy = (double)dy / h, fx = (double)dx / w;
            double r = std::sqrt(fy * fy + fx * fx);
            if (r < rmin_frac_of_n || std::fabs(fy) >= 0.5 - 2.0 / h || std::fabs(fx) >= 0.5 - 2.0 / w) continue;
            // the grid of JPEG blocks and 2x2-subsampled chroma (an unwarped JPEG page): peaks at (k / 16, l / 16) cycles per px
            double jx = fx * 16 - std::round(fx * 16), jy = fy * 16 - std::round(fy * 16);
            if (std::fabs(jx) * w <= 16.0 * 1.5 && std::fabs(jy) * h <= 16.0 * 1.5) continue;
            best = std::max(best, l[x] - m[x] / scale);
        }
    }
    return best;
}

cv::Mat box(const cv::Mat& a, int r) {
    cv::Mat o;
    cv::boxFilter(a, o, -1, cv::Size(2 * r + 1, 2 * r + 1), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    return o;
}

// He et al.'s guided filter, guide I, input p (both CV_32F)
cv::Mat guided(const cv::Mat& I, const cv::Mat& p, int r, float eps) {
    cv::Mat mI = box(I, r), mp = box(p, r);
    cv::Mat cov = box(I.mul(p), r) - mI.mul(mp);
    cv::Mat var = box(I.mul(I), r) - mI.mul(mI);
    cv::Mat a = cov / (var + eps);
    cv::Mat b = mp - a.mul(mI);
    return box(a, r).mul(I) + box(b, r);
}

}  // namespace

ScreenInfo detect_screen(const cv::Mat& page) {
    auto t0 = std::chrono::steady_clock::now();
    ScreenInfo si;
    CV_Assert(page.type() == CV_8UC4);
    int h = page.rows, w = page.cols;
    // pages below 1024 px (thumbnails, small imports) are never detected: too little of the page per
    // tile, and resampled small images carry their own periodic patterns
    const int n = 512;
    if (std::min(h, w) < 1024) return si;
    cv::Mat bgr, ycc;
    cv::cvtColor(page, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    cv::Mat win;
    cv::createHanningWindow(win, cv::Size(n, n), CV_32F);
    std::vector<float> sc, sy;
    const double fr[3] = {0.3, 0.5, 0.7};
    for (double fyr : fr)
        for (double fxr : fr) {
            int y0 = std::min(std::max(0, (int)(h * fyr) - n / 2), h - n), x0 = std::min(std::max(0, (int)(w * fxr) - n / 2), w - n);
            cv::Mat tile = ycc(cv::Rect(x0, y0, n, n)), ch[3];
            cv::split(tile, ch);
            float best_c = 0.f, best_y = 0.f;
            for (int c = 1; c < 3; ++c) {   // Cr, Cb (the luminance is not used)
                cv::Mat f;
                ch[c].convertTo(f, CV_32F);
                f -= cv::mean(f)[0];
                f = f.mul(win);
                // periods below 16 px: radius >= n / 16 bins
                float pk = peak_prominence(log_spectrum(f), 15, 1.0 / 16, 20.f);
                if (c == 0) best_y = pk;
                else best_c = std::max(best_c, pk);
            }
            sc.push_back(best_c);
            sy.push_back(best_y);
        }
    std::nth_element(sc.begin(), sc.begin() + sc.size() / 2, sc.end());
    std::nth_element(sy.begin(), sy.begin() + sy.size() / 2, sy.end());
    si.score = sc[sc.size() / 2];
    si.score_y = sy[sy.size() / 2];
    si.screen = si.score >= kScreenThreshold;
    si.ms = ms_since(t0);
    return si;
}

DescreenTune& descreen_tune() {
    static DescreenTune t;
    return t;
}

cv::Mat descreen(const cv::Mat& page, DescreenStats* st) {
    const DescreenTune& tn = descreen_tune();
    auto t0 = std::chrono::steady_clock::now();
    CV_Assert(page.type() == CV_8UC4);
    int h = page.rows, w = page.cols;
    cv::Mat bgr, f, ycc;
    cv::cvtColor(page, bgr, cv::COLOR_BGRA2BGR);
    bgr.convertTo(f, CV_32F);
    cv::cvtColor(f, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch;
    cv::split(ycc, ch);
    // 1. notch filter on an optimal DFT size (the page reflected into the padding)
    int H = cv::getOptimalDFTSize(h), W = cv::getOptimalDFTSize(w);
    std::vector<cv::Mat> spec(3);
    std::vector<double> means(3);
    for (int c = 0; c < 3; ++c) {
        means[c] = cv::mean(ch[c])[0];
        cv::Mat p;
        cv::copyMakeBorder(ch[c] - means[c], p, 0, H - h, 0, W - w, cv::BORDER_REFLECT);
        cv::dft(p, spec[c], cv::DFT_COMPLEX_OUTPUT);
    }
    cv::Mat mag, planes[2];
    cv::split(spec[0], planes);
    cv::magnitude(planes[0], planes[1], mag);
    mag += 1.f;
    cv::log(mag, mag);
    fftshift(mag);
    cv::Mat q, med;
    mag.convertTo(q, CV_8U, 16.0);
    cv::medianBlur(q, med, 21);
    cv::Mat pk(H, W, CV_8U, cv::Scalar(0));
    int npk = 0;
    const double rmin = 1.0 / tn.min_period;
    for (int y = 0; y < H; ++y) {
        const float* l = mag.ptr<float>(y);
        const uint8_t* m = med.ptr<uint8_t>(y);
        uint8_t* o = pk.ptr<uint8_t>(y);
        double fy = (double)(y - H / 2) / H;
        for (int x = 0; x < W; ++x) {
            double fx = (double)(x - W / 2) / W;
            if (fy * fy + fx * fx <= rmin * rmin) continue;
            if (l[x] - m[x] / 16.f > tn.peak_thr) o[x] = 1, ++npk;
        }
    }
    if (npk > 0) {
        cv::Mat mk;
        cv::dilate(pk, mk, cv::Mat::ones(5, 5, CV_8U));
        mk.convertTo(mk, CV_32F);
        cv::GaussianBlur(mk, mk, cv::Size(0, 0), 1.5);
        cv::Mat M = 1.f - cv::min(mk * 1.5f, 1.f);
        fftshift(M, true);
        cv::Mat M2;
        cv::merge(std::vector<cv::Mat>{M, M}, M2);
        for (int c = 0; c < 3; ++c) {
            cv::Mat back;
            cv::multiply(spec[c], M2, spec[c]);
            cv::dft(spec[c], back, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            ch[c] = back(cv::Rect(0, 0, w, h)) + means[c];
        }
    }
    // 2. chroma guided by the luminance
    if (tn.chroma_r > 0)
        for (int c = 1; c < 3; ++c) ch[c] = guided(ch[0], ch[c], tn.chroma_r, tn.chroma_eps * tn.chroma_eps);
    cv::merge(ch, ycc);
    cv::cvtColor(ycc, f, cv::COLOR_YCrCb2BGR);
    cv::Mat out8, out;
    f.convertTo(out8, CV_8U);   // rounds and saturates
    cv::cvtColor(out8, out, cv::COLOR_BGR2BGRA);
    if (st) st->peaks = npk, st->ms = ms_since(t0);
    return out;
}

}  // namespace fs
