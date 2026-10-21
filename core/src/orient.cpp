#include "orient.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "nputil.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

// numpy float32 pairwise sum of a contiguous run
float pw32(const float* a, size_t n) {
    if (n < 8) {
        float r = 0.f;
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pw32(a, n2) + pw32(a + n2, n - n2);
}
// .sum() of a strided 2-D float32 view: one pairwise reduction per row, rows accumulated
float sum_view(const cv::Mat& m) {
    if (m.isContinuous()) return pw32(m.ptr<float>(), m.total());
    float acc = 0.f;
    for (int r = 0; r < m.rows; ++r) acc += pw32(m.ptr<float>(r), m.cols);
    return acc;
}

}  // namespace

double text_axis_score(const cv::Mat& gray, double* sv_out) {
    int H = gray.rows, W = gray.cols;
    double sc = 1400.0 / std::max(H, W);
    cv::Mat g, bg;
    cv::resize(gray, g, cv::Size((int)(W * sc), (int)(H * sc)), 0, 0, cv::INTER_AREA);
    cv::dilate(g, bg, cv::Mat::ones(5, 5, CV_8U));
    cv::medianBlur(bg, bg, 31);
    cv::Mat ink(g.size(), CV_8U);
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x) ink.at<uint8_t>(y, x) = ((int)bg.at<uint8_t>(y, x) - (int)g.at<uint8_t>(y, x)) > 45;
    cv::Mat lh, lv;
    cv::morphologyEx(ink, lh, cv::MORPH_OPEN, cv::Mat::ones(1, 60, CV_8U));
    cv::morphologyEx(ink, lv, cv::MORPH_OPEN, cv::Mat::ones(60, 1, CV_8U));
    cv::Mat lines = lh | lv, ld;
    cv::dilate(lines, ld, cv::Mat::ones(3, 3, CV_8U));
    ink.setTo(0, ld > 0);
    auto run = [&](const cv::Mat& k, bool horiz) {
        cv::Mat m, lab, st, cen;
        cv::morphologyEx(ink, m, cv::MORPH_CLOSE, k);
        int n = cv::connectedComponentsWithStats(m, lab, st, cen, 8);
        double s = 0;
        for (int i = 1; i < n; ++i) {
            double w = st.at<int>(i, 2), h = st.at<int>(i, 3);
            if (horiz && w >= 3 * h && h >= 4) s += w;
            if (!horiz && h >= 3 * w && w >= 4) s += h;
        }
        return s;
    };
    double sh = run(cv::Mat::ones(1, 9, CV_8U), true);
    double sv = run(cv::Mat::ones(9, 1, CV_8U), false);
    if (sv_out) *sv_out = sv;
    return sh;
}

double upright_hebrew_score(const cv::Mat& gray) {
    int H = gray.rows, W = gray.cols;
    double scale = 1000.0 / std::max(H, W);
    cv::Mat small, smallf, bg;
    cv::resize(gray, small, cv::Size((int)(W * scale), (int)(H * scale)), 0, 0, cv::INTER_AREA);
    small.convertTo(smallf, CV_32F);
    cv::GaussianBlur(smallf, bg, cv::Size(0, 0), 15.0);
    cv::Mat ink(small.size(), CV_32F);
    for (int y = 0; y < small.rows; ++y)
        for (int x = 0; x < small.cols; ++x) {
            float b = bg.at<float>(y, x), s = smallf.at<float>(y, x);
            ink.at<float>(y, x) = std::min(std::max((b - s) / std::max(b, 20.f), 0.f), 1.f);
        }
    cv::Mat bbin = ink > 0.18f, hl, vl;
    bbin /= 255;
    cv::morphologyEx(bbin, hl, cv::MORPH_OPEN, cv::Mat::ones(1, 35, CV_8U));
    cv::morphologyEx(bbin, vl, cv::MORPH_OPEN, cv::Mat::ones(35, 1, CV_8U));
    cv::Mat text_ink = ink.clone();
    text_ink.setTo(0.f, (hl > 0) | (vl > 0));
    cv::Mat t02 = (text_ink > 0.2f) / 255, bars8, bars;
    cv::morphologyEx(t02, bars8, cv::MORPH_OPEN, cv::Mat::ones(1, 6, CV_8U));
    bars8.convertTo(bars, CV_32F);
    bars = bars.mul(text_ink);
    cv::Mat rows_mask, lab, st, cen;
    cv::morphologyEx(t02, rows_mask, cv::MORPH_CLOSE, cv::Mat::ones(3, 19, CV_8U));
    int num = cv::connectedComponentsWithStats(rows_mask, lab, st, cen, 8);
    Vec scores;
    for (int i = 1; i < num; ++i) {
        int x0 = st.at<int>(i, 0), y0 = st.at<int>(i, 1), w = st.at<int>(i, 2), h = st.at<int>(i, 3);
        if (w < 25 || h < 6 || h > 28 || (w / (double)h) < 2.0) continue;
        cv::Mat pi = text_ink(cv::Rect(x0, y0, w, h)), pb = bars(cv::Rect(x0, y0, w, h));
        float si = sum_view(pi), sb = sum_view(pb);
        if (si < 5.0f || sb < 1.0f) continue;
        cv::Mat wi(h, w, CV_32F), wb(h, w, CV_32F);   // (patch * ys): new contiguous arrays
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) {
                wi.at<float>(r, c) = pi.at<float>(r, c) * (float)r;
                wb.at<float>(r, c) = pb.at<float>(r, c) * (float)r;
            }
        double yci = (double)(sum_view(wi) / si), ycb = (double)(sum_view(wb) / sb);
        scores.push_back((yci - ycb) / (double)h);
    }
    double gag = scores.empty() ? 0.0 : median(scores) * 10.0;
    int hs = small.rows;
    auto band_mean = [&](double a, double b) {
        cv::Mat band = text_ink.rowRange((int)(a * hs), (int)(b * hs));
        return (double)(sum_view(band) / (float)band.total());
    };
    double top = band_mean(0.04, 0.32), bot = band_mean(0.68, 0.96);
    double header = (top - bot) / (top + bot + 1e-5);
    return gag + 0.35 * header;
}

Orientation decide_orientation(const cv::Mat& page, double min_conf, double min_ratio) {
    Orientation o;
    cv::Mat g1;
    cv::cvtColor(page, g1, page.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    o.sh = text_axis_score(g1, &o.sv);
    if (o.sv > o.sh * min_ratio) {
        cv::Mat ccw;
        cv::rotate(g1, ccw, cv::ROTATE_90_COUNTERCLOCKWISE);
        o.upright = upright_hebrew_score(ccw);
        o.rot = o.upright < -min_conf ? cv::ROTATE_90_CLOCKWISE : cv::ROTATE_90_COUNTERCLOCKWISE;
    } else {
        // no automatic 180-degree flip (round 4): in the 34-photo test every auto-180 was wrong,
        // the upright score is Hebrew-only. The result screen has a one-tap rotate instead.
        o.upright = upright_hebrew_score(g1);
        o.rot = -1;
    }
    return o;
}

}  // namespace fs
