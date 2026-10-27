#include "preview.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "corners.h"
#include "nprng.h"
#include "pagefind.h"

namespace fs {

namespace {

// the largest outer contour of a 0/1 mask
std::vector<cv::Point> largest_contour(const cv::Mat& m01) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m01, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    size_t best = 0;
    double ba = -1;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        if (a > ba) { ba = a; best = i; }
    }
    return cs.empty() ? std::vector<cv::Point>() : cs[best];
}

// TL, TR, BR, BL of four points (by x + y and y - x)
std::array<cv::Point2d, 4> order4(const cv::Point2f* p) {
    std::array<cv::Point2d, 4> o;
    int tl = 0, br = 0, tr = 0, bl = 0;
    for (int i = 1; i < 4; ++i) {
        if (p[i].x + p[i].y < p[tl].x + p[tl].y) tl = i;
        if (p[i].x + p[i].y > p[br].x + p[br].y) br = i;
        if (p[i].y - p[i].x < p[tr].y - p[tr].x) tr = i;
        if (p[i].y - p[i].x > p[bl].y - p[bl].x) bl = i;
    }
    o[0] = p[tl]; o[1] = p[tr]; o[2] = p[br]; o[3] = p[bl];
    return o;
}

}  // namespace

PreviewPage preview_page(const cv::Mat& bgr, const float* logits) {
    PreviewPage r;
    if (bgr.empty()) return r;
    const int W = bgr.cols, H = bgr.rows;
    cv::Mat gray, lap;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    r.luma = cv::mean(gray)[0] / 255.0;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar mu, sd;
    cv::meanStdDev(lap, mu, sd);
    r.sharpness = sd[0] * sd[0];
    if (!logits) return r;
    // the page and gutter probabilities at the net's size (the whole frame squashed to 256 x 256)
    cv::Mat pg[2];
    for (int c = 0; c < 2; ++c) {
        cv::Mat l(SEG_SIZE, SEG_SIZE, CV_32F, const_cast<float*>(logits) + (size_t)c * SEG_SIZE * SEG_SIZE);
        cv::exp(-l, pg[c]);
        pg[c] = 1.0 / (1.0 + pg[c]);
    }
    PageSelection sel = select_page(pg[0], pg[1]);
    if (sel.empty) return r;
    cv::Mat m = sel.mask.clone();
    if (!sel.other.empty()) m |= sel.other;   // a book spread: both pages are in view or not
    const double share = cv::countNonZero(m) / (double)m.total();
    if (share < 0.02) return r;               // nothing page-like (find_page_net's whole-frame test)
    // the outline in frame px, every 2 px, then the detection's quad step
    const double kx = (double)W / SEG_SIZE, ky = (double)H / SEG_SIZE;
    std::vector<cv::Point2d> outline;
    for (const cv::Point& p : largest_contour(m)) outline.emplace_back((p.x + 0.5) * kx - 0.5, (p.y + 0.5) * ky - 0.5);
    if (outline.size() < 3) return r;
    Detection D;
    D.contour = resample_closed(outline, 2.0);
    NpRng rng = NpRng::seed0();
    corners_from_contour(D, rng);
    if (D.ok) {
        r.pts = D.corners;
    } else {   // too short an outline for the side fits: its rotated bounding box
        std::vector<cv::Point2f> c32;
        for (const auto& q : D.contour) c32.emplace_back((float)q.x, (float)q.y);
        cv::Point2f bp[4];
        cv::minAreaRect(c32).points(bp);
        r.pts = order4(bp);
    }
    r.found = true;
    r.area = share;
    // glare: blown-out pixels on the page (a glossy sheet or a lamp's reflection)
    cv::Mat page;
    cv::resize(m, page, bgr.size(), 0, 0, cv::INTER_NEAREST);
    const int n = cv::countNonZero(page);
    if (n > 0) r.glare = cv::countNonZero((gray >= 250) & (page > 0)) / (double)n;
    return r;
}

}  // namespace fs
