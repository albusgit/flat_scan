#include "pipeline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <thread>

#include "gpu.h"
#include "log.h"
#include "orient.h"
#include "nputil.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Backward unroll map U: output page pixel -> full-resolution photo pixel, float32 as the
// reference's ux / uy arrays.
struct Unroll {
    GeomMode mode = GeomMode::Homography;
    SurfaceModel m;
    UvGrid uv;
    double Minv[9];
    int W = 0, H = 0;              // page size (px)
    bool scaled = false;           // model fitted on the analysis copy
    double kx = 1, ky = 1;         // full / analysis pixel ratio

    cv::Point2f at(int x, int y) const {
        if (mode == GeomMode::Homography) {   // cv::perspectiveTransform, float32 points
            float xf = (float)x, yf = (float)y;
            double w = xf * Minv[6] + yf * Minv[7] + Minv[8];
            if (std::fabs(w) <= FLT_EPSILON) return cv::Point2f(0, 0);
            w = 1. / w;
            return cv::Point2f((float)((xf * Minv[0] + yf * Minv[1] + Minv[2]) * w), (float)((xf * Minv[3] + yf * Minv[4] + Minv[5]) * w));
        }
        if (mode == GeomMode::Neural) {
            cv::Point2d q = uv.at((double)x, (double)y, W, H);
            return cv::Point2f((float)q.x, (float)q.y);
        }
        cv::Point2d p = m.unroll_at((double)x, (double)y, W, H);
        if (scaled) p = cv::Point2d((p.x + 0.5) * kx - 0.5, (p.y + 0.5) * ky - 0.5);
        return cv::Point2f((float)p.x, (float)p.y);
    }
};

// The page's height / width as it sits in the photo: its outline (U along the output page's
// border) measured in photo px, opposite sides averaged (perspective shortens one and lengthens the
// other). The output page is A4-shaped whatever the paper; a receipt squeezed to A4 has rows closer
// than its letters, which reads as sideways text.
double page_hw_ratio(const Unroll& U) {
    auto len = [&](int x0, int y0, int x1, int y1) {
        const int n = 64;
        double L = 0;
        cv::Point2f a = U.at(x0, y0);
        for (int k = 1; k <= n; ++k) {
            cv::Point2f b = U.at((int)std::lround(x0 + (x1 - x0) * (double)k / n), (int)std::lround(y0 + (y1 - y0) * (double)k / n));
            L += cv::norm(b - a);
            a = b;
        }
        return L;
    };
    double w = len(0, 0, U.W - 1, 0) + len(0, U.H - 1, U.W - 1, U.H - 1), h = len(0, 0, 0, U.H - 1) + len(U.W - 1, 0, U.W - 1, U.H - 1);
    return w > 0 ? std::min(5.0, std::max(0.2, h / w)) : (double)U.H / U.W;
}

// The orientation decision's input on the orientation page op: the page stretched to its measured
// shape (orient_true_aspect: rows > 0 = resized to that many rows, area = INTER_AREA) and the ratio
// a turn needs. Two equal inputs on the same page give the same decision.
struct OrientInput {
    int rows = 0;
    bool area = false;
    double min_ratio = 0;
    bool operator==(const OrientInput& o) const { return rows == o.rows && area == o.area && min_ratio == o.min_ratio; }
};

OrientInput orient_input(const cv::Mat& op, const Unroll& U, const PipelineOptions& opt) {
    OrientInput in;
    in.min_ratio = opt.orient_min_ratio;
    if (opt.orient_true_aspect) {
        // decided on the page at its own shape (a long receipt is not A4)
        double hw = page_hw_ratio(U), k = hw * op.cols / op.rows;
        if (std::fabs(k - 1) > 0.15) { in.rows = (int)std::lround(op.rows * k); in.area = k < 1; }
        // a long narrow upright page (receipt, ticket) reads along its width: monospaced, widely
        // spaced print scores as vertical text there (DocUNet 65_1: 1.7x, 65_2: 2.9x)
        if (hw >= 1.8) in.min_ratio = 4.0;
    }
    return in;
}

Orientation orient_decide(cv::Mat op, const OrientInput& in, const PipelineOptions& opt) {
    if (in.rows > 0) cv::resize(op, op, cv::Size(op.cols, in.rows), 0, 0, in.area ? cv::INTER_AREA : cv::INTER_LINEAR);
    return decide_orientation(op, opt.reference_orient ? 0.0 : opt.orient_min_conf, opt.reference_orient ? 1.0 : in.min_ratio);
}

// two homography unrolls with the same page and matrix (the same page image)
bool same_homography(const Unroll& a, const Unroll& b) {
    if (a.mode != GeomMode::Homography || b.mode != GeomMode::Homography || a.W != b.W || a.H != b.H) return false;
    for (int i = 0; i < 9; ++i)
        if (a.Minv[i] != b.Minv[i]) return false;
    return true;
}

// stage-1 page: cv2.remap(photo, U) with U evaluated exactly per pixel, in row bands
// (gpu: the same map, the bicubic on the GPU, byte-identical; CPU when that fails)
cv::Mat remap_unroll(const cv::Mat& photo, const Unroll& U, bool gpu = false) {
    cv::Mat out(U.H, U.W, photo.type());
    if (gpu && remap_cubic_gpu(photo, U.W, U.H, [&U](int x, int y) { return U.at(x, y); }, out)) return out;
    const int band = 32;
    int nb = (U.H + band - 1) / band;
    int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            cv::Mat mx(band, U.W, CV_32F), my(band, U.W, CV_32F);
            for (int b = t; b < nb; b += nt) {
                int y0 = b * band, y1 = std::min(U.H, y0 + band);
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < U.W; ++x) {
                        cv::Point2f p = U.at(x, y);
                        mx.at<float>(y - y0, x) = p.x;
                        my.at<float>(y - y0, x) = p.y;
                    }
                cv::Mat dst = out.rowRange(y0, y1);
                cv::remap(photo, dst, mx.rowRange(0, y1 - y0), my.rowRange(0, y1 - y0), cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                          cv::Scalar::all(255));
            }
        });
    for (auto& t : ts) t.join();
    return out;
}

// dense warp map at an integer pixel (cv2.remap(m4, x/4, y/4, LINEAR, REPLICATE))
inline cv::Point2f warp_at(const GridMap& g, int x, int y) {
    int iy = y / g.stride, ix = x / g.stride;
    float a = (float)(x % g.stride) * (1.f / g.stride), b = (float)(y % g.stride) * (1.f / g.stride);
    int x0 = std::min(ix, g.gw - 1), x1 = std::min(ix + 1, g.gw - 1), y0 = std::min(iy, g.gh - 1), y1 = std::min(iy + 1, g.gh - 1);
    float w0 = (1.f - b) * (1.f - a), w1 = (1.f - b) * a, w2 = b * (1.f - a), w3 = b * a;
    size_t i00 = (size_t)y0 * g.gw + x0, i01 = (size_t)y0 * g.gw + x1, i10 = (size_t)y1 * g.gw + x0, i11 = (size_t)y1 * g.gw + x1;
    return cv::Point2f(g.x[i00] * w0 + g.x[i01] * w1 + g.x[i10] * w2 + g.x[i11] * w3,
                       g.y[i00] * w0 + g.y[i01] * w1 + g.y[i10] * w2 + g.y[i11] * w3);
}

// cv2.remap(U, mx, my, INTER_LINEAR, BORDER_REPLICATE) at one position
inline cv::Point2f sample_unroll(const Unroll& U, float mx, float my) {
    int X = (int)std::lrint(mx * 32.f), Y = (int)std::lrint(my * 32.f);
    int sx = std::max(-32768, std::min(32767, X >> 5)), sy = std::max(-32768, std::min(32767, Y >> 5));
    float ax = (float)(X & 31) * (1.f / 32), ay = (float)(Y & 31) * (1.f / 32);
    float vx0 = 1.f - ax, vy0 = 1.f - ay;
    float w0 = vy0 * vx0, w1 = vy0 * ax, w2 = ay * vx0, w3 = ay * ax;
    auto cl = [](int v, int n) { return v >= 0 ? (v < n ? v : n - 1) : 0; };
    int x0 = cl(sx, U.W), x1 = cl(sx + 1, U.W), y0 = cl(sy, U.H), y1 = cl(sy + 1, U.H);
    cv::Point2f v0 = U.at(x0, y0), v1 = U.at(x1, y0), v2 = U.at(x0, y1), v3 = U.at(x1, y1);
    return cv::Point2f(v0.x * w0 + v1.x * w1 + v2.x * w2 + v3.x * w3, v0.y * w0 + v1.y * w1 + v2.y * w2 + v3.y * w3);
}

// final 4-px grid in the (possibly rotated) output frame
GridMap compose(const Unroll& U, const GridMap& warp4, int rot, bool compose_unroll) {
    int W = U.W, H = U.H;
    bool swap = rot == cv::ROTATE_90_CLOCKWISE || rot == cv::ROTATE_90_COUNTERCLOCKWISE;
    GridMap g = GridMap::make(swap ? H : W, swap ? W : H, 4);
    int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            for (int i = t; i < g.gh; i += nt)
                for (int j = 0; j < g.gw; ++j) {
                    int xr = 4 * j, yr = 4 * i, x, y;
                    if (rot == cv::ROTATE_90_CLOCKWISE) { x = yr; y = H - 1 - xr; }
                    else if (rot == cv::ROTATE_180) { x = W - 1 - xr; y = H - 1 - yr; }
                    else if (rot == cv::ROTATE_90_COUNTERCLOCKWISE) { x = W - 1 - yr; y = xr; }
                    else { x = xr; y = yr; }
                    cv::Point2f m = warp_at(warp4, x, y);
                    cv::Point2f f = compose_unroll ? sample_unroll(U, m.x, m.y) : m;
                    g.X(i, j) = f.x;
                    g.Y(i, j) = f.y;
                }
        });
    for (auto& t : ts) t.join();
    return g;
}

void report(const PipelineOptions& o, int pct, const char* stage) {
    if (o.progress) o.progress(pct, stage);
}

cv::Mat to_bgra(const cv::Mat& m) {
    if (m.channels() == 4) return m;
    cv::Mat o;
    cv::cvtColor(m, o, cv::COLOR_BGR2BGRA);
    return o;
}

void final_resample(const cv::Mat& photo, PipelineResult& r, const PipelineOptions& opt) {
    auto t0 = Clock::now();
    cv::Mat src = to_bgra(photo);
    r.render_map = r.final_map;
    if (opt.render_w > 0 && opt.render_w != opt.out_w) {
        bool portrait = r.final_map.out_w <= r.final_map.out_h;
        int long_side = (int)std::lround(opt.render_w * 297.0 / 210.0);
        r.render_map = rescale_grid(r.final_map, portrait ? opt.render_w : long_side, portrait ? long_side : opt.render_w);
    }
    r.gpu_resample = opt.use_gpu && Gpu::instance().resample(src, r.render_map, r.page);
    if (!r.gpu_resample) r.page = resample_cpu(src, r.render_map);
    r.ms.emplace_back(r.gpu_resample ? "final resample (GPU)" : "final resample (CPU)", ms_since(t0));
}

// paper mask (analysis-copy pixels, 255 = paper) sampled through the render map: photo
// pixel P -> mask pixel (P + 0.5) * s - 0.5. Eroded 2 px first so the dark fringe of the
// outline is filled too, then blurred to a 1-2 px soft edge.
cv::Mat paper_alpha(const cv::Mat& mask, double s, const GridMap& g) {
    cv::Mat m;
    cv::erode(mask, m, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat out(g.out_h, g.out_w, CV_8U), mx, my;
    for (int y0 = 0; y0 < g.out_h; y0 += 64) {
        int y1 = std::min(g.out_h, y0 + 64);
        expand_rows(g, y0, y1, mx, my);
        if (s != 1.0) {
            mx = (mx + 0.5f) * (float)s - 0.5f;
            my = (my + 0.5f) * (float)s - 0.5f;
        }
        cv::Mat dst = out.rowRange(y0, y1);
        cv::remap(m, dst, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    cv::GaussianBlur(out, out, cv::Size(0, 0), 0.8);
    return out;
}

GridMap identity_grid(int W, int H) {
    GridMap g = GridMap::make(W, H, 4);
    for (int i = 0; i < g.gh; ++i)
        for (int j = 0; j < g.gw; ++j) { g.X(i, j) = (float)(4 * j); g.Y(i, j) = (float)(4 * i); }
    return g;
}

GridMap at_render_size(const GridMap& g, const PipelineOptions& opt) {
    if (opt.render_w <= 0 || opt.render_w == opt.out_w) return g;
    bool portrait = g.out_w <= g.out_h;
    int long_side = (int)std::lround(opt.render_w * 297.0 / 210.0);
    return rescale_grid(g, portrait ? opt.render_w : long_side, portrait ? long_side : opt.render_w);
}

void resample_any(const cv::Mat& src_bgra, const GridMap& g, bool gpu, cv::Mat& out) {
    if (!(gpu && Gpu::instance().resample(src_bgra, g, out))) out = resample_cpu(src_bgra, g);
}

// unrotated page pixel (analysis W x H) -> rotated output frame
inline cv::Point2d rotate_pt(double x, double y, int W, int H, int rot) {
    if (rot == cv::ROTATE_90_CLOCKWISE) return {H - 1 - y, x};
    if (rot == cv::ROTATE_180) return {W - 1 - x, H - 1 - y};
    if (rot == cv::ROTATE_90_COUNTERCLOCKWISE) return {y, W - 1 - x};
    return {x, y};
}

// rotated output frame -> unrotated page pixel (analysis W x H), the inverse of rotate_pt
inline cv::Point2d unrotate_pt(double xr, double yr, int W, int H, int rot) {
    if (rot == cv::ROTATE_90_CLOCKWISE) return {yr, H - 1 - xr};
    if (rot == cv::ROTATE_180) return {W - 1 - xr, H - 1 - yr};
    if (rot == cv::ROTATE_90_COUNTERCLOCKWISE) return {W - 1 - yr, xr};
    return {xr, yr};
}

// the user's ignore areas (output-page fractions, rotated frame) as a mask on the stage-1 page
// Where the line stage's map reaches past the stage-1 page (a W x H frame) by more than 2 px: CV_8U,
// 255 there, in the output frame (turned by rot), W x H before the turn; empty if nowhere. Those
// pixels have no page to show (the remap repeats the page's edge pixels into streaks: a curled
// corner pulled straight brings in the margin beyond it), so the paper fill paints them.
cv::Mat beyond_page(const GridMap& warp4, int W, int H, int rot) {
    cv::Mat q(warp4.gh, warp4.gw, CV_8U, cv::Scalar(0));
    int n = 0;
    for (int i = 0; i < warp4.gh; ++i)
        for (int j = 0; j < warp4.gw; ++j) {
            float x = warp4.x[(size_t)i * warp4.gw + j], y = warp4.y[(size_t)i * warp4.gw + j];
            if (x < -2 || y < -2 || x > W + 1 || y > H + 1) { q.at<uint8_t>(i, j) = 255; ++n; }
        }
    if (n == 0) return cv::Mat();
    cv::Mat m;
    cv::resize(q, m, cv::Size(warp4.gw * warp4.stride, warp4.gh * warp4.stride), 0, 0, cv::INTER_NEAREST);
    m = m(cv::Rect(0, 0, std::min(W, m.cols), std::min(H, m.rows))).clone();
    if (rot >= 0) cv::rotate(m, m, rot);
    return m;
}

cv::Mat ignore_mask(const std::vector<std::array<float, 4>>& rects, int W, int H, int rot) {
    if (rects.empty()) return cv::Mat();
    bool swap = rot == cv::ROTATE_90_CLOCKWISE || rot == cv::ROTATE_90_COUNTERCLOCKWISE;
    int Wr = swap ? H : W, Hr = swap ? W : H;
    cv::Mat m = cv::Mat::zeros(H, W, CV_8U);
    for (const auto& q : rects) {
        double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
        for (int k = 0; k < 4; ++k) {
            cv::Point2d p = unrotate_pt((k % 3 == 0 ? q[0] : q[2]) * Wr, (k < 2 ? q[1] : q[3]) * Hr, W, H, rot);
            x0 = std::min(x0, p.x); y0 = std::min(y0, p.y); x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
        }
        cv::Rect r((int)std::floor(x0), (int)std::floor(y0), (int)std::ceil(x1 - x0) + 1, (int)std::ceil(y1 - y0) + 1);
        r &= cv::Rect(0, 0, W, H);
        if (r.area() > 0) m(r).setTo(255);
    }
    return m;
}

// The shadow net on the flattened page (PipelineOptions::shade_net): the page at long_side,
// portrait (landscape pages turned for the net and back), its shading map bilinearly back at page
// size, the page divided by it: brighten only, by at most 1 / 0.3 (cast shadows on paper keep
// >= ~0.45 of the light; anything darker is ink, which a misread map must not blow out).
bool apply_shade_net(cv::Mat& page_bgra, const std::function<bool(std::vector<float>&, int, int)>& fn, int long_side) {
    if (page_bgra.empty() || !fn) return false;
    cv::Mat bgr;
    cv::cvtColor(page_bgra, bgr, cv::COLOR_BGRA2BGR);
    const bool turned = bgr.cols > bgr.rows;
    if (turned) cv::rotate(bgr, bgr, cv::ROTATE_90_CLOCKWISE);
    double s = (double)long_side / std::max(bgr.cols, bgr.rows);
    int nw = std::max(32, (int)std::lround(bgr.cols * s / 32) * 32), nh = std::max(32, (int)std::lround(bgr.rows * s / 32) * 32);
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    const size_t plane = (size_t)nw * nh;
    std::vector<float> buf(3 * plane);
    for (int y = 0; y < nh; ++y) {
        const cv::Vec3b* p = small.ptr<cv::Vec3b>(y);
        for (int x = 0; x < nw; ++x)
            for (int c = 0; c < 3; ++c) buf[c * plane + (size_t)y * nw + x] = p[x][c] * (1.f / 255);
    }
    if (!fn(buf, nw, nh) || buf.size() != 3 * plane) return false;
    cv::Mat S(nh, nw, CV_32FC3);
    for (int y = 0; y < nh; ++y) {
        cv::Vec3f* q = S.ptr<cv::Vec3f>(y);
        for (int x = 0; x < nw; ++x)
            for (int c = 0; c < 3; ++c) q[x][c] = std::min(1.f, std::max(0.3f, buf[c * plane + (size_t)y * nw + x]));
    }
    if (turned) cv::rotate(S, S, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::resize(S, S, page_bgra.size(), 0, 0, cv::INTER_LINEAR);
    cv::parallel_for_(cv::Range(0, page_bgra.rows), [&](const cv::Range& R) {
        for (int y = R.start; y < R.end; ++y) {
            cv::Vec4b* p = page_bgra.ptr<cv::Vec4b>(y);
            const cv::Vec3f* q = S.ptr<cv::Vec3f>(y);
            for (int x = 0; x < page_bgra.cols; ++x)
                for (int c = 0; c < 3; ++c) p[x][c] = cv::saturate_cast<uint8_t>(p[x][c] / q[x][c]);
        }
    });
    return true;
}

// Fingers / thumbs holding the page: on the unwarped page they are the non-paper areas (the page
// finder leaves them out of the page) that reach the page border and are mostly skin coloured
// (YCrCb skin box). They are filled with the paper colour around them (a smooth Telea fill at
// 1/4 size, so a line crossing the edge fades instead of ending in a hard cut) and marked as
// paper. Text under a finger is not invented: it is paper. Returns the share of the page filled.
double erase_fingers(cv::Mat& page_bgra, cv::Mat& paper) {
    const int w = page_bgra.cols, h = page_bgra.rows;
    cv::Mat hole = paper < 128;
    int n_hole = cv::countNonZero(hole);
    if (n_hole == 0 || n_hole > 0.25 * w * h) return 0;
    cv::Mat bgr, ycc;
    cv::cvtColor(page_bgra, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    // skin: in the YCrCb skin box AND clearly redder / less blue than this page's own paper
    // (beige or yellowed paper sits inside the box too)
    cv::Mat paper_px = paper > 200;
    if (cv::countNonZero(paper_px) < 0.2 * w * h) return 0;
    cv::Scalar pc = cv::mean(ycc, paper_px);
    cv::Mat skin;
    cv::inRange(ycc, cv::Scalar(40, std::max(140.0, pc[1] + 7), 77), cv::Scalar(255, 180, std::min(125.0, pc[2] - 4)), skin);
    // fingers are smooth: ink strokes (text, rules) inside a candidate mean it is page content
    cv::Mat gray, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 60, 150);
    cv::Mat lab, st, cen;
    int n = cv::connectedComponentsWithStats(hole, lab, st, cen, 8);
    cv::Mat fill = cv::Mat::zeros(h, w, CV_8U);
    const int border = std::max(4, w / 100);
    for (int i = 1; i < n; ++i) {
        int x = st.at<int>(i, 0), y = st.at<int>(i, 1), bw = st.at<int>(i, 2), bh = st.at<int>(i, 3), area = st.at<int>(i, 4);
        bool at_border = x <= border || y <= border || x + bw >= w - border || y + bh >= h - border;
        if (!at_border || area < 0.0005 * w * h) continue;
        cv::Mat comp = lab(cv::Rect(x, y, bw, bh)) == i;
        double skin_frac = (double)cv::countNonZero(skin(cv::Rect(x, y, bw, bh)) & comp) / area;
        // a finger reaches into the page: it is not a thin sliver along the border
        int reach = std::min(std::min(x + bw, w - x), std::min(y + bh, h - y));
        // ink inside the candidate, its own outline (finger edge, nail, the page rule beside it) left out
        cv::Mat inner;
        cv::erode(comp, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * border + 1, 2 * border + 1)));
        int n_inner = cv::countNonZero(inner);
        double edge_frac = n_inner > 0.2 * area ? (double)cv::countNonZero(edges(cv::Rect(x, y, bw, bh)) & inner) / n_inner : 1.0;
        if (std::getenv("FS_DEBUG_FINGERS")) {
            cv::Scalar cm = cv::mean(ycc(cv::Rect(x, y, bw, bh)), comp);
            double whole = (double)cv::countNonZero(edges(cv::Rect(x, y, bw, bh)) & comp) / area;
            std::fprintf(stderr, "finger cand %d,%d %dx%d area %d skin %.3f reach %d edge %.4f whole %.4f dY %.1f dCr %.1f dCb %.1f\n", x, y, bw, bh, area,
                         skin_frac, reach, edge_frac, whole, cm[0] - pc[0], cm[1] - pc[1], cm[2] - pc[2]);
        }
        // skin >= 35%: the hole often takes in the background strip beside the finger too (receipt: 0.499)
        if (skin_frac < 0.35 || reach < 2 * border || edge_frac > 0.03) continue;
        fill(cv::Rect(x, y, bw, bh)).setTo(255, comp);
    }
    int n_fill = cv::countNonZero(fill);
    if (n_fill == 0) return 0;
    // the fringe of the finger (shadow, blurred edge) goes with it
    cv::dilate(fill, fill, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * border + 1, 2 * border + 1)));
    fill_with_paper(page_bgra, fill, &paper);
    return (double)n_fill / (w * h);
}

// Text-row angle spread of a page (degrees, p90 - p10 of the rows' slopes): word blobs merged
// along a row, the centroid track of each row fitted with a line; the page is also measured turned
// 90 degrees (sideways text) and the orientation with more rows counts. rows: rows measured.
// Scaled for a page about 1100 px wide (a half-size stage-1 page).
double text_row_spread(const cv::Mat& page_bgr, int* rows_out) {
    double best = -1;
    int best_rows = -1;
    for (int turn = 0; turn < 2; ++turn) {
        cv::Mat img = page_bgr;
        if (turn) cv::rotate(page_bgr, img, cv::ROTATE_90_CLOCKWISE);
        cv::Mat g, bg, ink, lines, words;
        cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
        cv::dilate(g, bg, cv::Mat::ones(5, 5, CV_8U));
        cv::medianBlur(bg, bg, 15);
        cv::subtract(bg, g, ink);
        ink = ink > 40;
        cv::Mat lh, lv;
        cv::morphologyEx(ink, lh, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(60, 1)));
        cv::morphologyEx(ink, lv, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 60)));
        cv::dilate(lh | lv, lines, cv::Mat::ones(3, 3, CV_8U));
        ink.setTo(0, lines);
        cv::morphologyEx(ink, words, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 1)));
        cv::morphologyEx(words, words, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
        cv::Mat lab, st, cen;
        int n = cv::connectedComponentsWithStats(words, lab, st, cen, 8);
        std::vector<double> ang;
        for (int i = 1; i < n; ++i) {
            int x = st.at<int>(i, 0), y = st.at<int>(i, 1), w = st.at<int>(i, 2), h = st.at<int>(i, 3);
            if (w < 75 || h < 4 || h > 30 || w < 5 * h) continue;
            std::vector<double> xs, cs;
            for (int cx = 0; cx + 4 <= w; cx += 4) {
                double sy = 0;
                int m = 0;
                for (int yy = 0; yy < h; ++yy)
                    for (int xx = cx; xx < cx + 4; ++xx)
                        if (lab.at<int>(y + yy, x + xx) == i && ink.at<uint8_t>(y + yy, x + xx)) { sy += yy; ++m; }
                if (m < 3) continue;
                xs.push_back(x + cx + 2);
                cs.push_back(y + sy / m);
            }
            if (xs.size() < 12) continue;
            double mx = 0, my = 0;
            for (size_t k = 0; k < xs.size(); ++k) { mx += xs[k]; my += cs[k]; }
            mx /= xs.size(); my /= xs.size();
            double sxx = 0, sxy = 0;
            for (size_t k = 0; k < xs.size(); ++k) { sxx += (xs[k] - mx) * (xs[k] - mx); sxy += (xs[k] - mx) * (cs[k] - my); }
            if (sxx > 0) ang.push_back(std::atan(sxy / sxx) * 180.0 / CV_PI);
        }
        if ((int)ang.size() > best_rows) {
            best_rows = (int)ang.size();
            if (ang.size() >= 3) {
                std::sort(ang.begin(), ang.end());
                auto q = [&](double f) { return ang[(size_t)std::lround(f * (ang.size() - 1))]; };
                best = q(0.9) - q(0.1);
            } else {
                best = -1;
            }
        }
    }
    if (rows_out) *rows_out = best_rows;
    return best;
}

// local area change of the warp map's line stage, |det(I + dF) - 1| per grid cell (the outer 2
// cells left out): its 99th percentile; frac = the share of cells above 0.2
double local_distortion(const WarpDebug& d, double* frac) {
    const Field &Fx = d.Fx, &Fy = d.Fy;
    std::vector<double> v;
    for (int i = 2; i + 3 < Fx.gh; ++i)
        for (int j = 2; j + 3 < Fx.gw; ++j) {
            auto V = [&](const Field& F, int a, int b) { return F.v[(size_t)a * F.gw + b]; };
            double xx = (V(Fx, i, j + 1) - V(Fx, i, j)) / Fx.g, xy = (V(Fx, i + 1, j) - V(Fx, i, j)) / Fx.g;
            double yx = (V(Fy, i, j + 1) - V(Fy, i, j)) / Fy.g, yy = (V(Fy, i + 1, j) - V(Fy, i, j)) / Fy.g;
            v.push_back(std::fabs((1 + xx) * (1 + yy) - xy * yx - 1));
        }
    *frac = 0;
    if (v.empty()) return 0;
    size_t n = 0;
    for (double q : v) n += q > 0.2;
    *frac = (double)n / v.size();
    size_t k = (size_t)(0.99 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// the spots where the line stage overstretches the page (|det J - 1| > 0.2 per field cell, the outer
// 2 cells left out as in local_distortion), widened by 24 px: CV_8U, w x h, 255 there
cv::Mat distortion_spots(const WarpDebug& d, int w, int h) {
    const Field &Fx = d.Fx, &Fy = d.Fy;
    cv::Mat m = cv::Mat::zeros(h, w, CV_8U);
    for (int i = 2; i + 3 < Fx.gh; ++i)
        for (int j = 2; j + 3 < Fx.gw; ++j) {
            auto V = [&](const Field& F, int a, int b) { return F.v[(size_t)a * F.gw + b]; };
            double xx = (V(Fx, i, j + 1) - V(Fx, i, j)) / Fx.g, xy = (V(Fx, i + 1, j) - V(Fx, i, j)) / Fx.g;
            double yx = (V(Fy, i, j + 1) - V(Fy, i, j)) / Fy.g, yy = (V(Fy, i + 1, j) - V(Fy, i, j)) / Fy.g;
            if (std::fabs((1 + xx) * (1 + yy) - xy * yx - 1) > 0.2)
                cv::rectangle(m, cv::Rect(j * Fx.g, i * Fx.g, Fx.g, Fx.g), 255, -1);
        }
    cv::dilate(m, m, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(49, 49)));
    return m;
}

// Neural pages: the stage-1 page is already flat, so a line correction that locally rescales it by
// more than thr (|det J - 1| of a 4-px cell) is a wrong constraint's doing (a mis-traced line pulling
// the text beside it), never real flattening. The correction is faded out around such cells
// (widened by ~24 px, feathered over ~16 px); the rest of the page keeps it. g: the warp map on the
// 4-px grid (absolute stage-1 px). Returns the share of grid nodes touched.
// Exception (support, from extension_support): a stretch between two neighbouring rules that were
// both followed through it (extend_trace) is measured, not invented - rows fanning together at a
// curled corner, which straighten only if the page there is stretched by up to 2x. Those cells are
// neither flagged nor faded (a fade spreading in from beside them bent the rows it reached).
double soften_bumps(GridMap& g, double thr, const cv::Mat& support = cv::Mat()) {
    const int gw = g.gw, gh = g.gh;
    const double s = g.stride;
    if (gw < 3 || gh < 3) return 0;
    cv::Mat bad = cv::Mat::zeros(gh, gw, CV_8U);
    int n_bad = 0;
    for (int i = 0; i + 1 < gh; ++i)
        for (int j = 0; j + 1 < gw; ++j) {
            double xj = (g.X(i, j + 1) - g.X(i, j)) / s, xi = (g.X(i + 1, j) - g.X(i, j)) / s;
            double yj = (g.Y(i, j + 1) - g.Y(i, j)) / s, yi = (g.Y(i + 1, j) - g.Y(i, j)) / s;
            if (!support.empty() && support.at<uint8_t>(i, j)) continue;
            if (std::fabs(xj * yi - xi * yj - 1) > thr) { bad.at<uint8_t>(i, j) = 255; ++n_bad; }
        }
    if (n_bad == 0) return 0;
    cv::dilate(bad, bad, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
    cv::Mat m;
    bad.convertTo(m, CV_32F, 1.0 / 255);
    cv::GaussianBlur(m, m, cv::Size(0, 0), 4);
    int touched = 0;
    for (int i = 0; i < gh; ++i)
        for (int j = 0; j < gw; ++j) {
            double keep = 1 - std::min(1.0, 2.0 * m.at<float>(i, j));
            if (keep >= 1 || (!support.empty() && support.at<uint8_t>(i, j))) continue;
            ++touched;
            g.X(i, j) = (float)(s * j + keep * (g.X(i, j) - s * j));
            g.Y(i, j) = (float)(s * i + keep * (g.Y(i, j) - s * i));
        }
    return (double)touched / ((double)gw * gh);
}

// soften_bumps' support on the warp map's grid (gw x gh, stride s; output frame, where a traced
// line sits at its ideal): where two extensions of one direction overlap along the lines and their
// ideals are <= 150 px apart, the band between the ideals, widened by their spacing across (the
// paper beyond the outermost rule of a fan is stretched with it, up to the page edge) and by 16 px
// along
cv::Mat extension_support(const WarpDebug& d, int gw, int gh, double s) {
    cv::Mat m = cv::Mat::zeros(gh, gw, CV_8U);
    for (int hz = 0; hz < 2; ++hz) {
        std::vector<std::array<double, 3>> E = hz ? d.S2.ext_h : d.S2.ext_v;
        if (E.empty()) E = hz ? d.S1.ext_h : d.S1.ext_v;
        std::sort(E.begin(), E.end());
        for (size_t i = 0; i < E.size(); ++i)
            for (size_t j = i + 1; j < E.size() && E[j][0] - E[i][0] <= 150; ++j) {
                double a = std::max(E[i][1], E[j][1]), b = std::min(E[i][2], E[j][2]);
                if (b - a < 20 || E[j][0] - E[i][0] < 4) continue;
                // the nearest overlapping neighbour only: the band ends at the next extended rule
                bool between = false;
                for (size_t k = i + 1; k < j; ++k) between |= std::min(b, E[k][2]) - std::max(a, E[k][1]) >= 20;
                if (between) continue;
                double gap = E[j][0] - E[i][0];
                int c0 = std::max(0, (int)std::floor((E[i][0] - gap) / s)), c1 = (int)std::ceil((E[j][0] + gap) / s);
                int t0 = std::max(0, (int)std::floor((a - 16) / s)), t1 = (int)std::ceil((b + 16) / s);
                for (int c = c0; c <= c1; ++c)
                    for (int t = t0; t <= t1; ++t) {
                        int r = hz ? c : t, q = hz ? t : c;
                        if (r < gh && q < gw) m.at<uint8_t>(r, q) = 1;
                    }
            }
    }
    // ... and where two neighbouring rules' traces both run > 4 px off their ideals the same way (a
    // curl both rules show: rows fanning together), the band between them along that stretch; with
    // the rule net confirming them, the curled rows are traced whole and have no extensions
    for (int hz = 0; hz < 2; ++hz) {
        const auto& TR = hz ? d.S2.tr_h : d.S2.tr_v;
        std::vector<const Samples::Trace*> T;
        for (const auto& t : TR) if (t.t.size() >= 20) T.push_back(&t);
        std::sort(T.begin(), T.end(), [](const Samples::Trace* a, const Samples::Trace* b) { return a->ideal < b->ideal; });
        for (size_t i = 0; i < T.size(); ++i) {
            const auto& A = *T[i];
            for (size_t j = i + 1; j < T.size() && T[j]->ideal - A.ideal <= 150; ++j) {
                const auto& B = *T[j];
                double a = std::max(A.t.front(), B.t.front()), b = std::min(A.t.back(), B.t.back());
                if (b - a < 20 || B.ideal - A.ideal < 4) continue;
                bool between = false;
                for (size_t k = i + 1; k < j; ++k) between |= std::min(b, T[k]->t.back()) - std::max(a, T[k]->t.front()) >= 20;
                if (between) continue;
                double gap = B.ideal - A.ideal;
                for (double t = a; t <= b; t += s) {
                    double da = interp(t, A.t, A.c) - A.ideal, db = interp(t, B.t, B.c) - B.ideal;
                    if (std::fabs(da) <= 4 || std::fabs(db) <= 4 || da * db <= 0) continue;
                    int c0 = std::max(0, (int)std::floor((A.ideal - gap) / s)), c1 = (int)std::ceil((B.ideal + gap) / s);
                    int t0 = std::max(0, (int)std::floor((t - 16) / s)), t1 = (int)std::ceil((t + 16) / s);
                    for (int c = c0; c <= c1; ++c)
                        for (int u = t0; u <= t1; ++u) {
                            int r = hz ? c : u, q = hz ? u : c;
                            if (r < gh && q < gw) m.at<uint8_t>(r, q) = 1;
                        }
                }
            }
        }
    }
    return m;
}

// the lines the warp map used, as output-page fractions (PipelineResult::lines)
std::vector<std::array<float, 5>> fitted_lines(const WarpDebug& d, int W, int H, int rot) {
    bool swap = rot == cv::ROTATE_90_CLOCKWISE || rot == cv::ROTATE_90_COUNTERCLOCKWISE;
    double Wr = swap ? H : W, Hr = swap ? W : H;
    std::vector<std::array<float, 5>> out;
    auto add = [&](int kind, double x0, double y0, double x1, double y1) {
        cv::Point2d a = rotate_pt(x0, y0, W, H, rot), b = rotate_pt(x1, y1, W, H, rot);
        out.push_back({(float)kind, (float)(a.x / Wr), (float)(a.y / Hr), (float)(b.x / Wr), (float)(b.y / Hr)});
    };
    // ruled lines over the stretch their traces covered (both passes; extensions past the found
    // segment included); a line that gave no samples (a curve left out, a failed trace) is not drawn
    auto ruled = [&](const Lines& A, bool hz) {
        const auto &s1 = hz ? d.S1.span_h : d.S1.span_v, &s2 = hz ? d.S2.span_h : d.S2.span_v;
        for (size_t i = 0; i < A.size(); ++i) {
            const Line& L = A[i];
            double a = L[1], b = L[2];
            if (s1.size() == A.size() || s2.size() == A.size()) {
                a = 1e300; b = -1e300;
                for (const auto* s : {&s1, &s2})
                    if (s->size() == A.size() && !std::isnan((*s)[i][0])) { a = std::min(a, (*s)[i][0]); b = std::max(b, (*s)[i][1]); }
                if (a > b) continue;
            }
            if (hz) add(0, a, L[0], b, L[0]);
            else add(0, L[0], a, L[0], b);
        }
    };
    ruled(d.H, true);
    ruled(d.V, false);
    for (size_t i = 0; i < d.E_lines.size(); ++i) {   // colour-area boundaries, drawn as grid lines
        const auto& L = d.E_lines[i];
        if ((int)i < d.n_edges_h) add(2, L[1], L[0], L[2], L[0]);
        else add(2, L[0], L[1], L[0], L[2]);
    }
    // text rows: consecutive samples of one row share its ideal position
    for (size_t i = 0; i < d.T.size();) {
        size_t j = i;
        while (j + 1 < d.T.size() && d.T[j + 1][1] == d.T[i][1] && d.T[j + 1][0] > d.T[j][0]) ++j;
        add(1, d.T[i][0], d.T[i][1], d.T[j][0], d.T[i][1]);
        i = j + 1;
    }
    // grid lines of the global stage that the local stage did not find again
    for (const auto& G : d.G_lines) {
        bool hz = G[0] > 0.5, seen = false;
        for (const auto& L : hz ? d.H : d.V)
            seen |= std::fabs(L[0] - G[1]) < 12 && std::min(L[2], G[3]) - std::max(L[1], G[2]) > 0.5 * (G[3] - G[2]);
        if (seen) continue;
        if (hz) add(2, G[2], G[1], G[3], G[1]);
        else add(2, G[1], G[2], G[1], G[3]);
    }
    return out;
}

}  // namespace

namespace {

// the analysis copy the corner fit and the surface fit run on
cv::Mat analysis_copy(const cv::Mat& photo, int max_side) {
    double s = std::min(1.0, (double)max_side / std::max(photo.cols, photo.rows));
    cv::Mat small = photo;
    if (s < 1.0) cv::resize(photo, small, cv::Size((int)std::lround(photo.cols * s), (int)std::lround(photo.rows * s)), 0, 0, cv::INTER_AREA);
    return small;
}

// corners of page `idx` from the net finder's result (f) or, without it, the colour threshold
PageCorners make_corners(const cv::Mat& photo, const cv::Mat& small, const PageFindResult* f, int idx) {
    PageCorners pc;
    NpRng rng = NpRng::seed0();
    pc.scale = (double)small.cols / photo.cols;
    double kx = (double)photo.cols / small.cols, ky = (double)photo.rows / small.rows;
    bool scaled = small.cols != photo.cols || small.rows != photo.rows;
    if (f) {
        pc.finder = Finder::Net;
        pc.n_pages = (int)f->pages.size();
        pc.page_index = std::min(std::max(idx, 0), pc.n_pages - 1);
        pc.whole_frame = f->whole_frame;
        pc.spread = f->spread;
        pc.net_margin = f->net_margin;
        const FoundPage& fp = f->pages[pc.page_index];
        pc.edge_strength = fp.edge_strength;
        pc.edge_at_limit = fp.edge_at_limit;
        Detection& D = pc.det;
        D.whole_frame = f->whole_frame;
        std::vector<std::vector<cv::Point>> poly(1);
        cv::Point2d sum(0, 0);
        for (const auto& p : fp.contour) {
            cv::Point2d q = scaled ? cv::Point2d((p.x + 0.5) / kx - 0.5, (p.y + 0.5) / ky - 0.5) : p;
            D.contour.push_back(q);
            poly[0].emplace_back((int)std::lround(q.x * 4), (int)std::lround(q.y * 4));
            sum += p;
        }
        if (!fp.contour.empty()) pc.centroid = sum / (double)fp.contour.size();
        D.paper = cv::Mat::zeros(small.size(), CV_8U);
        if (!poly[0].empty()) cv::fillPoly(D.paper, poly, cv::Scalar(255), cv::LINE_8, 2);
        corners_from_contour(D, rng);
    } else {
        pc.det = detect_page(small, rng);
        pc.whole_frame = pc.det.whole_frame;
        cv::Point2d sum(0, 0);
        for (const auto& q : pc.det.contour) sum += q;
        if (!pc.det.contour.empty()) {
            cv::Point2d c = sum / (double)pc.det.contour.size();
            pc.centroid = scaled ? cv::Point2d((c.x + 0.5) * kx - 0.5, (c.y + 0.5) * ky - 0.5) : c;
        }
    }
    pc.ok = pc.det.ok;
    for (int k = 0; k < 4; ++k)
        pc.pts[k] = scaled ? cv::Point2d((pc.det.corners[k].x + 0.5) * kx - 0.5, (pc.det.corners[k].y + 0.5) * ky - 0.5) : pc.det.corners[k];
    return pc;
}

SurfaceModel fit_detected(const cv::Mat& photo, const PageCorners& pc, double focal_px) {
    const Detection& D = pc.det;
    double f_low = focal_px > 0 ? focal_px * pc.scale : 0;
    int sw = (int)std::lround(photo.cols * pc.scale), sh = (int)std::lround(photo.rows * pc.scale);
    if (pc.scale >= 1.0) { sw = photo.cols; sh = photo.rows; }
    int threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    return fit_surface(D.side_pts, D.corners, sw, sh, D.landscape, f_low, threads);
}

std::string fmt0(const char* f, double v) {
    char b[96];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

}  // namespace

PageCorners detect_corners(const cv::Mat& photo, int max_side, const float* seg_logits, int page_index) {
    cv::Mat small = analysis_copy(photo, max_side);
    if (!seg_logits) return make_corners(photo, small, nullptr, 0);
    PageFindResult f = find_page_net(photo, seg_logits);
    return make_corners(photo, small, &f, page_index);
}

std::vector<std::string> quality_gate(const PageCorners& pc, double fit_median_full_px, cv::Size photo) {
    const int h = photo.height, w = photo.width;
    std::vector<std::string> reasons;
    if (fit_median_full_px > 15) reasons.push_back(fmt0("surface fit residual %.0f px", fit_median_full_px));
    bool outside = false;
    for (const auto& c : pc.pts) {
        double o = std::max(std::max(-c.x, c.x - w), std::max(-c.y, c.y - h));
        outside |= o > 0.02 * std::max(h, w);
    }
    if (outside) reasons.push_back("corner outside the photo");
    std::vector<cv::Point2f> q;
    for (const auto& c : pc.pts) q.emplace_back((float)c.x, (float)c.y);
    double area = cv::contourArea(q) / ((double)h * w);
    if (area < 0.05 || area > 1.2) reasons.push_back(fmt0("page area %.0f%% of the photo", 100 * area));
    // the reference only reports a fallback for the net finder (info['fallback'] == 'whole_frame')
    if (pc.finder == Finder::Net && pc.whole_frame) reasons.push_back("no page found");
    return reasons;
}

std::vector<PageAssessment> assess_pages(const cv::Mat& photo, const float* seg_logits, double focal_px, int max_side) {
    cv::Mat small = analysis_copy(photo, max_side);
    PageFindResult f;
    if (seg_logits) f = find_page_net(photo, seg_logits);
    int n = seg_logits ? (int)f.pages.size() : 1;
    std::vector<PageAssessment> out;
    for (int i = 0; i < n; ++i) {
        PageAssessment a;
        a.pc = make_corners(photo, small, seg_logits ? &f : nullptr, i);
        if (!a.pc.ok) {
            a.reasons.push_back("no page found");
        } else {
            SurfaceModel m = fit_detected(photo, a.pc, focal_px);
            a.fit_median_px = m.ok ? m.median_px / a.pc.scale : -1;
            a.reasons = quality_gate(a.pc, a.fit_median_px, photo.size());
            if (!m.ok) a.reasons.push_back("surface fit failed");
        }
        out.push_back(std::move(a));
    }
    return out;
}

// The traced rules of a line stage (its S2 samples, grouped by ideal) as points on the page it ran
// on: A_img's p is the page's p + G(p) (the global field; zero when the global stage was skipped).
// Every 2nd sample; lines with fewer than 10 points are left out.
struct PageTraces {
    std::vector<std::vector<cv::Point2d>> pts;
    std::vector<char> hz;
};
PageTraces page_traces(const WarpDebug& d) {
    PageTraces T;
    const bool glob = d.n_ctrl > 0 && !d.GX.v.empty();
    for (int hz = 1; hz >= 0; --hz) {
        std::map<double, std::vector<cv::Point2d>> lines;
        size_t k = 0;
        for (const auto& r : hz ? d.S2.h : d.S2.v) {
            if (k++ % 2) continue;
            double x = hz ? r[0] : r[1] + r[2], y = hz ? r[1] + r[2] : r[0];
            if (glob) { double gx = d.GX.sample(x, y), gy = d.GY.sample(x, y); x += gx; y += gy; }
            lines[r[1]].push_back({x, y});
        }
        for (auto& kv : lines)
            if (kv.second.size() >= 10) { T.pts.push_back(std::move(kv.second)); T.hz.push_back((char)hz); }
    }
    // the text-line net's lines count too (a page of text has no rules to judge a map by)
    if (d.text_net)
        for (int hz = 1; hz >= 0; --hz) {
            std::map<double, std::vector<cv::Point2d>> lines;
            size_t k = 0;
            for (const auto& r : hz ? d.T : d.TVs) {
                if (k++ % 2) continue;
                double x = hz ? r[0] : r[1] + r[2], y = hz ? r[1] + r[2] : r[0];
                if (glob) { double gx = d.GX.sample(x, y), gy = d.GY.sample(x, y); x += gx; y += gy; }
                lines[r[1]].push_back({x, y});
            }
            for (auto& kv : lines)
                if (kv.second.size() >= 10) { T.pts.push_back(std::move(kv.second)); T.hz.push_back((char)hz); }
        }
    return T;
}

// a backward map (output -> page, 4-px grid) at an output point, bilinear, clamped to the grid
cv::Point2d map_at(const GridMap& g, double x, double y) {
    double gx = std::min(std::max(x / g.stride, 0.0), g.gw - 1.0), gy = std::min(std::max(y / g.stride, 0.0), g.gh - 1.0);
    int j = std::min((int)gx, g.gw - 2), i = std::min((int)gy, g.gh - 2);
    double tx = gx - j, ty = gy - i;
    auto at = [&](const std::vector<float>& v, int a, int b) { return (double)v[(size_t)a * g.gw + b]; };
    auto bil = [&](const std::vector<float>& v) {
        return (1 - ty) * ((1 - tx) * at(v, i, j) + tx * at(v, i, j + 1)) + ty * ((1 - tx) * at(v, i + 1, j) + tx * at(v, i + 1, j + 1));
    };
    return {bil(g.x), bil(g.y)};
}

// How straight those rules come out through map m (null: the page as it is): each point's output
// position (m inverted by fixed-point steps from the point itself; the line stage moves the page
// by tens of px at most), per line the p95 distance from its straight fit across the line,
// averaged over the lines (px)
double straightness_through(const PageTraces& T, const GridMap* m) {
    double sum = 0;
    for (size_t l = 0; l < T.pts.size(); ++l) {
        Vec t, c;
        for (const auto& s : T.pts[l]) {
            cv::Point2d p = s;
            if (m)
                for (int it = 0; it < 10; ++it) { cv::Point2d q = map_at(*m, p.x, p.y); p += s - q; }
            t.push_back(T.hz[l] ? p.x : p.y); c.push_back(T.hz[l] ? p.y : p.x);
        }
        Vec f = polyfit(t, c, 1), d;
        for (size_t i = 0; i < t.size(); ++i) d.push_back(std::fabs(c[i] - polyval(f, t[i])));
        std::sort(d.begin(), d.end());
        sum += d[(size_t)(0.95 * (d.size() - 1))];
    }
    return T.pts.empty() ? 0.0 : sum / T.pts.size();
}

// The line stage on a page the nets flattened, with its guards. The page is already flat, so a
// correction that stretches or squeezes it locally (|det J - 1| p99 above max_line_distortion, or
// more than max_line_distortion_frac of the cells above 0.2) is a tracing failure (dense staves,
// broken rules, a rule traced into text). Then, in turn: the same lines without those that pass
// through the overstretched spots (the nets' page stays as it is there), the reference lines alone,
// and no line stage at all. Corrections that rescale a spot by more than max_local_rescale are
// faded out there (soften_bumps), and a candidate must leave the rules the first attempt traced
// straighter than the page as it is (straightness_through, by 0.1 px). ignore: the user's areas
// (CV_8U, page size) or empty. Sets r.warp, r.line_distortion, r.line_straightness,
// r.lines_spots_dropped, r.lines_rejected, r.bumps_faded; returns the map on the 4-px grid.
GridMap neural_line_stage(const cv::Mat& page, const cv::Mat& ignore, const PipelineOptions& opt, PipelineResult& r) {
    // the rule net's output is kept for the further unwarp_map runs on the same page (the retry)
    std::vector<float> net_out;
    int net_w = 0, net_h = 0;
    RuleNetFn net = [&](std::vector<float>& io, int w, int h) {
        if (!net_out.empty() && w == net_w && h == net_h) { io = net_out; return true; }
        if (!opt.rule_net(io, w, h)) return false;
        net_out = io; net_w = w; net_h = h;
        return true;
    };
    const RuleNetFn* netp = opt.rule_net ? &net : nullptr;
    const cv::Mat* ig = ignore.empty() ? nullptr : &ignore;
    WarpDebug w0;
    GridMap m0 = unwarp_map(page, &w0, ig, opt.share_rows, opt.faint_lines, opt.robust_solve, opt.colour_edges, netp);
    // the yardstick for every candidate: the rules the first attempt traced, all of them
    const PageTraces T = page_traces(w0);
    const double s_page = straightness_through(T, nullptr);
    r.line_straightness[0] = s_page;
    // a candidate is kept if it neither overstretches the page nor leaves those rules less straight
    // than the page as it is (a fallback that drops some lines can bend the rules they were on)
    auto take = [&](WarpDebug& w, GridMap& m) {
        double f = 0, q = local_distortion(w, &f);
        if (r.line_distortion < 0) r.line_distortion = q;
        if (q > opt.max_line_distortion || f > opt.max_line_distortion_frac) return false;
        if (opt.max_local_rescale > 0) r.bumps_faded = soften_bumps(m, opt.max_local_rescale, extension_support(w, m.gw, m.gh, m.stride));
        double st = straightness_through(T, &m);
        if (T.pts.size() >= 3 && st > s_page - 0.1) return false;
        r.line_straightness[1] = st;
        r.line_distortion = q;
        r.warp = std::move(w);
        return true;
    };
    // a second trace-and-solve pass on the page the first one straightened (Marior, ArbDR: run again
    // on the own output, keep it only if it helps): rules too bent to be found or traced on the
    // first page are found there. Its map composes with the first (the page is resampled once, at
    // the end). Kept if it stays within the distortion guard, straightens the rules it traced by
    // > 0.1 px on average and leaves the first pass's rules no less straight.
    auto second_pass = [&](GridMap& m) {
        if (!opt.second_line_pass) return;
        // no lines on the first page (plain text, photos): the second finds none either (34 real photos: 11 such
        // pages, the pass ran ~0.9 s on each for nothing)
        if (r.warp.H.empty() && r.warp.V.empty()) return;
        auto t2 = Clock::now();
        struct Done { PipelineResult& r; Clock::time_point t; ~Done() { r.second_pass_ms = ms_since(t); } } done{r, t2};
        cv::Mat page2 = resample_cpu(page, m);
        if (page2.channels() == 4) cv::cvtColor(page2, page2, cv::COLOR_BGRA2BGR);
        RuleNetFn net2 = opt.rule_net;   // the page moved: the net runs again
        WarpDebug w;
        GridMap m2 = unwarp_map(page2, &w, nullptr, opt.share_rows, opt.faint_lines, opt.robust_solve, opt.colour_edges,
                                opt.rule_net ? &net2 : nullptr);
        double f = 0, q = local_distortion(w, &f);
        r.second_pass[0] = q;
        if (q > opt.max_line_distortion || f > opt.max_line_distortion_frac) return;
        const PageTraces T2 = page_traces(w);
        double b2 = straightness_through(T2, nullptr), a2 = straightness_through(T2, &m2);
        r.second_pass[1] = b2; r.second_pass[2] = a2;
        if (T2.pts.size() < 3 || a2 > b2 - 0.1) return;
        GridMap c = m2;   // composite: output -> page2 (m2) -> page (m)
        for (size_t i = 0; i < c.x.size(); ++i) {
            cv::Point2d q2 = map_at(m, m2.x[i], m2.y[i]);
            c.x[i] = (float)q2.x; c.y[i] = (float)q2.y;
        }
        if (straightness_through(T, &c) > r.line_straightness[1] + 0.05) return;
        r.line_straightness[1] = straightness_through(T, &c);
        m = std::move(c);
        r.second_pass_kept = true;
    };
    if (take(w0, m0)) { second_pass(m0); return m0; }
    auto tr = Clock::now();
    struct Done { PipelineResult& r; Clock::time_point t; ~Done() { r.line_retry_ms = ms_since(t) - std::max(0.0, r.second_pass_ms); } } retry_done{r, tr};   // without a second pass inside
    if (opt.faint_lines) {
        // the same lines without those that pass through the overstretched spots (left out whole:
        // dropping only their samples there straightened a rule on one side of a spot and not on
        // the other); the nets' page stays as it is there
        cv::Mat spots = distortion_spots(w0, page.cols, page.rows);
        WarpDebug w1;
        GridMap m1 = unwarp_map(page, &w1, ig, opt.share_rows, opt.faint_lines, opt.robust_solve, opt.colour_edges, netp, &spots);
        if (take(w1, m1)) { r.lines_spots_dropped = true; second_pass(m1); return m1; }
        // the reference lines alone: the faint rules may be what overstretches it (picture edges, a
        // rule traced into text)
        WarpDebug w2;
        GridMap m2 = unwarp_map(page, &w2, ig, opt.share_rows, false, opt.robust_solve);
        if (take(w2, m2)) { second_pass(m2); return m2; }
    }
    r.warp = std::move(w0);
    r.lines_rejected = true;
    r.bumps_faded = 0;
    return identity_grid(page.cols, page.rows);
}

PipelineResult run_photo(const cv::Mat& photo, const PipelineOptions& opt_in) {
    GpuStageScope gpu_stages(opt_in.use_gpu);   // the global stage's remap (warpmap.cpp)
    PipelineResult r;
    auto t0 = Clock::now(), tall = Clock::now();
    // flat_nets: their grids go into a copy of the options (uv_out* point into `grids`)
    std::unique_ptr<PipelineOptions> with_grids;
    if (opt_in.flat_nets) with_grids.reset(new PipelineOptions(opt_in));
    const PipelineOptions& opt = with_grids ? *with_grids : opt_in;
    const bool nets_on = with_grids && !opt.perspective_only;
    const bool overlap = nets_on && opt.overlap_nets;
    FlatNetGrids grids;
    auto take_grids = [&](FlatNetGrids g) {
        grids = std::move(g);
        if (!grids.grid1.empty()) { with_grids->uv_out = grids.grid1.data(); with_grids->uv_crop = grids.crop1; }
        if (!grids.grid2.empty()) { with_grids->uv_out2 = grids.grid2.data(); with_grids->uv_crop2 = grids.crop2; }
        if (!grids.grid_pass1.empty()) { with_grids->uv_out_pass1 = grids.grid_pass1.data(); with_grids->uv_crop_pass1 = grids.crop_pass1; }
    };
    // the nets' corners, handed over once (net_corners is not written after that). Declared before
    // the nets' future and that before the promise: on an exception the promise breaks first, the
    // nets' thread ends and the future's destructor joins it while what it reads still exists.
    std::array<cv::Point2d, 4> net_corners;
    double nets_ms = 0;
    std::future<FlatNetGrids> nets_fut;
    std::promise<const std::array<cv::Point2d, 4>*> corners_promise;
    std::shared_future<const std::array<cv::Point2d, 4>*> corners_fut = corners_promise.get_future().share();
    bool corners_given = false;
    auto give_corners = [&](const std::array<cv::Point2d, 4>* c) {
        if (!nets_on || corners_given) return;
        if (c) net_corners = *c;
        corners_promise.set_value(c ? &net_corners : nullptr);
        corners_given = true;
    };
    // a failed net leaves its slot empty (the pipeline runs without it), as a failed run_flat_nets
    auto run_nets = [&opt, &nets_ms, corners_fut]() {
        auto tn = Clock::now();
        FlatNetGrids g;
        try {
            g = opt.flat_nets([corners_fut]() { return corners_fut.get(); });
        } catch (const std::exception& e) {
            LOGI("flattening nets failed: %s", e.what());
            g = FlatNetGrids();
        }
        nets_ms = ms_since(tn);
        return g;
    };
    if (overlap) {
        if (opt.corners_edited) give_corners(&opt.corners);
        nets_fut = std::async(std::launch::async, run_nets);
    }
    report(opt, 2, "Finding the page");
    Unroll U;
    std::array<cv::Point2d, 4> corners;
    PageCorners pc;
    if (!opt.corners_edited) {
        pc = detect_corners(photo, opt.max_analysis_side, opt.seg_logits, opt.page_index);
        r.detection = pc;
        if (!pc.ok) {
            give_corners(nullptr);   // the nets that need corners do not run; the future's destructor waits for the rest
            r.error = "no page found";
            return r;
        }
        corners = pc.pts;
    } else {
        corners = opt.corners;
    }
    r.ms.emplace_back("detect", ms_since(t0));
    give_corners(&corners);
    if (nets_on && !overlap) {
        report(opt, 5, "Flattening nets");
        take_grids(run_nets());
        r.nets_overlap = 0;
        r.nets_ms = nets_ms;
        r.nets_wait_ms = 0;
        r.nets_join_ms = ms_since(tall);
        r.ms.emplace_back("flattening nets", nets_ms);
    }
    bool landscape = false;
    int W = 0, H = 0;
    cv::Mat M;
    Unroll Uh;
    // page size and the homography from the 4 corners (always: v8 path and the orientation page)
    auto set_corners = [&](const std::array<cv::Point2d, 4>& c) {
        corners = c;
        cv::Point2d wv = corners[1] - corners[0], hv = corners[3] - corners[0];
        landscape = std::sqrt(wv.x * wv.x + wv.y * wv.y) > std::sqrt(hv.x * hv.x + hv.y * hv.y);
        W = opt.out_w; H = (int)std::lround(opt.out_w * 297.0 / 210.0);
        if (landscape) std::swap(W, H);
        std::vector<cv::Point2f> srcq, dstq = {{0, 0}, {(float)(W - 1), 0}, {(float)(W - 1), (float)(H - 1)}, {0, (float)(H - 1)}};
        for (auto& q : corners) srcq.emplace_back((float)q.x, (float)q.y);
        cv::Mat Mi;
        M = cv::getPerspectiveTransform(srcq, dstq);
        cv::invert(M, Mi, cv::DECOMP_LU);
        Uh = Unroll();
        Uh.W = W; Uh.H = H;
        Uh.mode = GeomMode::Homography;
        for (int i = 0; i < 9; ++i) Uh.Minv[i] = Mi.at<double>(i / 3, i % 3);
    };
    set_corners(corners);
    r.mode = GeomMode::Homography;
    U = Uh;
    if (!opt.corners_edited) {
        t0 = Clock::now();
        report(opt, 8, "Fitting the page surface");
        r.model = fit_detected(photo, pc, opt.focal_px);
        r.ms.emplace_back("surface fit", ms_since(t0));
        r.fit_median_px = r.model.median_px;
        r.fit_p90_px = r.model.p90_px;
        // quality gate (photo_pipeline.quality_gate, residual in full-resolution px): a doubtful
        // crop takes the homography path; the app shows it on the crop screen first
        r.needs_manual_crop = quality_gate(pc, r.model.ok ? r.model.median_px / pc.scale : -1, photo.size());
        if (!r.model.ok) r.needs_manual_crop.push_back("surface fit failed");
        bool trusted = r.needs_manual_crop.empty() && !opt.force_homography && !opt.perspective_only;
        if (r.model.ok && r.model.median_px <= opt.surface_fallback_px && trusted) {
            U.mode = GeomMode::Surface;
            U.m = r.model;
            U.scaled = pc.scale < 1.0;
            int sw = (int)std::lround(photo.cols * pc.scale), sh = (int)std::lround(photo.rows * pc.scale);
            if (pc.scale >= 1.0) { sw = photo.cols; sh = photo.rows; }
            U.kx = (double)photo.cols / sw;
            U.ky = (double)photo.rows / sh;
            r.mode = GeomMode::Surface;
        } else {
            r.fell_back = true;
        }
    }
    // overlap: while the nets run, the orientation page (the homography page of these corners) and its
    // decision for the current unroll. Kept below when the grid check leaves the corners as they are
    // (uv_defines_page moves them) and the page-shape test gives the same input; otherwise redone.
    // The before page (the homography page at render size, turned as decided) too: it needs only these
    // corners and that turn, and the nets leave the CPU waiting (a GPU phone: 0.75 s); kept when both stay.
    struct {
        bool ok = false;
        Unroll Uh;
        cv::Mat page;
        OrientInput in;
        Orientation o;
        cv::Mat before;
    } spec;
    if (overlap && opt.auto_orient && !opt.orient_on_stage1) {
        auto ts = Clock::now();
        spec.Uh = Uh;
        spec.page = remap_unroll(photo, Uh, opt.use_gpu);
        spec.in = orient_input(spec.page, U, opt);
        spec.o = orient_decide(spec.page, spec.in, opt);
        spec.ok = true;
        r.ms.emplace_back("orientation, beside the nets", ms_since(ts));
        ts = Clock::now();
        resample_any(to_bgra(photo), at_render_size(compose(Uh, identity_grid(W, H), spec.o.rot, true), opt), opt.use_gpu, spec.before);
        r.ms.emplace_back("before page, beside the nets", ms_since(ts));
    }
    if (overlap) {
        auto tw = Clock::now();
        take_grids(nets_fut.get());   // run_nets catches every exception
        r.nets_overlap = 1;
        r.nets_ms = nets_ms;
        r.nets_wait_ms = ms_since(tw);
        r.nets_join_ms = ms_since(tall);
        r.ms.emplace_back("flattening nets, beside the CPU stages", nets_ms);
        r.ms.emplace_back("waiting for the flattening nets", r.nets_wait_ms);
    }
    // Flattening nets (app): a grid replaces the surface model / homography as stage 1 when its page
    // outline agrees with the page finder (or with corners set by hand); with two nets, the first is
    // kept unless the second's stage-1 page has far straighter text rows (uv_prefer_margin). The line
    // stage still runs on top.
    if (opt.uv_out && !opt.perspective_only) {
        auto t_uv = Clock::now();
        const bool use_mask = !opt.corners_edited && pc.ok && !pc.det.paper.empty();
        const double need = use_mask ? opt.uv_min_agreement : opt.uv_min_agreement_quad;
        struct Cand { UvGrid g; double agree = 0, consist = -1; int model = 0; bool own = false; };
        std::vector<Cand> ok;
        double best_agree = 0;
        UvGrid pass1;
        if (opt.uv_out_pass1 && !opt.corners_edited) pass1 = uv_grid(opt.uv_out_pass1, opt.uv_crop_pass1);
        for (int k = 0; k < 2; ++k) {
            const float* out = k == 0 ? opt.uv_out : opt.uv_out2;
            if (!out) continue;
            const UvCrop& crop = k == 0 ? opt.uv_crop : opt.uv_crop2;
            Cand c;
            c.g = uv_grid(out, crop);
            c.model = crop.model;
            // the page finder's mask can miss a coloured band or a dark page its corners still frame
            c.agree = uv_agreement(c.g, corners, photo.size());
            if (use_mask) c.agree = std::max(c.agree, uv_agreement(c.g, pc.det.paper, pc.scale));
            best_agree = std::max(best_agree, c.agree);
            if (!c.g.ok) continue;
            bool by_finder = c.g.turned == landscape && c.agree >= (c.model == 2 && use_mask ? opt.uv_min_agreement_hr : need);
            if (pass1.ok) c.consist = uv_agreement(c.g, pass1);
            c.own = !by_finder && c.consist >= opt.uv_min_consistency && c.agree >= opt.uv_consistent_floor;
            if (by_finder || c.own) ok.push_back(std::move(c));
        }
        r.uv_agreement = best_agree;
        // grids that define the page themselves: one page frame for all candidates
        bool any_finder = false;
        for (const auto& c : ok) any_finder |= !c.own;
        if (!ok.empty() && !any_finder) set_corners(uv_corners(ok[0].g));
        ok.erase(std::remove_if(ok.begin(), ok.end(), [&](const Cand& c) { return c.g.turned != landscape; }), ok.end());
        if (!ok.empty()) {
            size_t pick = 0;
            if (ok.size() == 2) {
                // a half-size stage-1 page of each: text-row angle spread
                for (size_t k = 0; k < 2; ++k) {
                    Unroll Us = Uh;
                    Us.mode = GeomMode::Neural;
                    Us.uv = ok[k].g;
                    Us.W = W / 2;
                    Us.H = H / 2;
                    int rows = 0;
                    r.uv_spread[k] = text_row_spread(remap_unroll(photo, Us, opt.use_gpu), &rows);
                    if (rows < 5) r.uv_spread[k] = -1;
                }
                if (r.uv_spread[0] >= 0 && r.uv_spread[1] >= 0 && r.uv_spread[1] < r.uv_spread[0] - opt.uv_prefer_margin) pick = 1;
            }
            if (ok[pick].own) {
                set_corners(uv_corners(ok[pick].g));
                r.uv_defines_page = true;
            }
            r.uv_consistency = ok[pick].consist;
            U = Uh;
            U.mode = GeomMode::Neural;
            U.uv = std::move(ok[pick].g);
            r.uv_agreement = ok[pick].agree;
            r.uv_model = ok[pick].model;
            r.mode = GeomMode::Neural;
            r.fell_back = false;
            // the zoomed second pass (uv_zoom): kept when its outline still agrees with the page and
            // its half-size stage-1 page has text rows no less straight (within zoom_max_worse)
            if (opt.zoom_pass && opt.uv_net) {
                auto tz = Clock::now();
                report(opt, 14, "Flattening: zoomed pass");
                int used = 0;
                UvGrid fine = uv_zoom(photo, U.uv, opt.zoom_model >= 0 ? opt.zoom_model : r.uv_model, opt.uv_net, &used, opt.zoom_nu, opt.zoom_nv);
                r.uv_zoom_quarters = used;
                if (used > 0) {
                    double agree = use_mask ? uv_agreement(fine, pc.det.paper, pc.scale) : uv_agreement(fine, corners, photo.size());
                    auto spread = [&](const UvGrid& gg) {
                        Unroll Us = Uh;
                        Us.mode = GeomMode::Neural;
                        Us.uv = gg;
                        Us.W = W / 2;
                        Us.H = H / 2;
                        int rows = 0;
                        double s = text_row_spread(remap_unroll(photo, Us), &rows);
                        return rows < 5 ? -1.0 : s;
                    };
                    r.uv_zoom_spread[0] = spread(U.uv);
                    r.uv_zoom_spread[1] = spread(fine);
                    if (agree >= need && r.uv_zoom_spread[0] >= 0 && r.uv_zoom_spread[1] >= 0 &&
                        r.uv_zoom_spread[1] <= r.uv_zoom_spread[0] + opt.zoom_max_worse) {
                        U.uv = std::move(fine);
                        r.uv_zoom = true;
                    }
                }
                r.ms.emplace_back("zoomed flattening pass", ms_since(tz));
            }
        } else {
            r.uv_rejected = true;
        }
        r.ms.emplace_back("flattening net check", ms_since(t_uv));
    }
    t0 = Clock::now();
    report(opt, 20, "Unrolling the page");
    cv::Mat stage1;
    // the homography page made beside the nets is this stage-1 page when no grid or surface replaced it
    if (spec.ok && same_homography(U, spec.Uh)) {
        stage1 = spec.page;
        r.orient_page_reused = true;
    } else {
        stage1 = remap_unroll(photo, U, opt.use_gpu);
    }
    r.ms.emplace_back("stage-1 page", ms_since(t0));
    if (opt.auto_orient) {
        t0 = Clock::now();
        cv::Mat op;
        if (opt.orient_on_stage1 || r.mode == GeomMode::Homography) {
            op = stage1;
        } else if (spec.ok && same_homography(Uh, spec.Uh)) {
            op = spec.page;
            r.orient_page_reused = true;
        } else {
            op = remap_unroll(photo, Uh, opt.use_gpu);
        }
        OrientInput in = orient_input(op, U, opt);
        Orientation o;
        if (spec.ok && op.data == spec.page.data && in == spec.in) {
            o = spec.o;   // the same page, the same test: the decision made beside the nets
            r.orient_reused = true;
        } else {
            o = orient_decide(op, in, opt);
        }
        r.rot = o.rot; r.orient_sh = o.sh; r.orient_sv = o.sv; r.orient_upright = o.upright;
        r.ms.emplace_back("orientation", ms_since(t0));
    }
    t0 = Clock::now();
    report(opt, 30, "Tracing lines and solving the warp map");
    GridMap warp4;
    if (opt.perspective_only || !opt.line_stage) {
        warp4 = identity_grid(W, H);
    } else {
        cv::Mat ignore = ignore_mask(opt.ignore, W, H, r.rot);
        if (r.mode == GeomMode::Neural) {
            warp4 = neural_line_stage(stage1, ignore, opt, r);
        } else {
            warp4 = unwarp_map(stage1, &r.warp, ignore.empty() ? nullptr : &ignore, false, opt.faint_lines, opt.robust_solve, opt.colour_edges,
                               opt.rule_net ? &opt.rule_net : nullptr);
        }
        r.lines = fitted_lines(r.warp, W, H, r.rot);
    }
    r.rule_net = !r.warp.PH.empty();
    r.net_dropped = r.warp.net.dropped; r.net_added = r.warp.net.added; r.net_joined = r.warp.net.joined; r.net_kept = r.warp.S2.n_net_kept;
    r.stage1 = stage1;
    r.ms.emplace_back("warp map", ms_since(t0));
    for (auto& m : r.warp.ms) r.ms.emplace_back("  " + m.first, m.second);
    if (r.line_retry_ms >= 0) r.ms.emplace_back("  retries (map overstretched)", r.line_retry_ms);
    if (r.second_pass_ms >= 0) r.ms.emplace_back(r.second_pass_kept ? "  second line pass (kept)" : "  second line pass (not kept)", r.second_pass_ms);
    t0 = Clock::now();
    report(opt, 85, "Composing the map");
    r.final_map = compose(U, warp4, r.rot, true);
    cv::Mat beyond = beyond_page(warp4, W, H, r.rot);
    r.ms.emplace_back("compose (4-px grid)", ms_since(t0));
    report(opt, 90, "Resampling");
    final_resample(photo, r, opt);
    // shadows: SyntheticDoc's shading map (sampled through the same map) and / or the shadow net
    // on the flattened page; the net replaces the map unless shade_after_uv
    auto shade = [&] {
        auto ts = Clock::now();
        report(opt, 92, "Removing shadows");
        r.shade_net = apply_shade_net(r.page, opt.shade_net, opt.shade_long_side);
        r.shadows_removed |= r.shade_net;
        r.ms.emplace_back("shadow net", ms_since(ts));
    };
    const bool net = opt.remove_shadows && opt.shade_net;
    if (net && !opt.shade_after_uv) shade();
    if (opt.uv_shading && opt.remove_shadows && !r.shade_net) {
        auto ts = Clock::now();
        UvShading sh;
        sh.data = opt.uv_shading;
        sh.crop = opt.uv_shading_crop;
        sh.full_strength = opt.shadow_full_strength;
        cv::Rect box = opt.uv_shading_crop.box & cv::Rect(0, 0, photo.cols, photo.rows);
        cv::Scalar mean = cv::mean(photo(box));
        sh.brightness = (mean[0] + mean[1] + mean[2]) / (3 * 255.0);
        cv::Mat mx, my;
        for (int y0 = 0; y0 < r.render_map.out_h; y0 += 64) {
            int y1 = std::min(r.render_map.out_h, y0 + 64);
            expand_rows(r.render_map, y0, y1, mx, my);
            uv_apply_shading(r.page, y0, mx, my, sh);
        }
        r.shadows_removed = true;
        r.ms.emplace_back("shadow removal", ms_since(ts));
    }
    if (net && opt.shade_after_uv) shade();
    t0 = Clock::now();
    report(opt, 95, "Finishing");
    // paper mask: from this run's detection, or - when the corners were edited - from a fresh
    // detection if its corners agree with the user's (otherwise the outline is not trusted)
    if (opt.corners_edited) {
        PageCorners pd = detect_corners(photo, opt.max_analysis_side, opt.seg_logits, opt.page_index);
        double diag = hypot_xy((double)photo.cols, (double)photo.rows), worst = 0;
        for (int k = 0; k < 4 && pd.ok; ++k) worst = std::max(worst, cv::norm(pd.pts[k] - corners[k]));
        if (pd.ok && worst < 0.03 * diag) pc = pd;
    }
    if (r.uv_defines_page && pc.ok && !pc.det.paper.empty()) {
        // the page finder's outline was overruled: the grid's outline is the paper
        cv::Mat m = cv::Mat::zeros(pc.det.paper.size(), CV_8U);
        std::vector<cv::Point> poly;
        for (const auto& p : U.uv.outline()) poly.emplace_back((int)std::lround(p.x * pc.scale * 4), (int)std::lround(p.y * pc.scale * 4));
        cv::fillPoly(m, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255), cv::LINE_8, 2);
        pc.det.paper = m;
    }
    if (pc.ok && !pc.det.paper.empty()) r.paper = paper_alpha(pc.det.paper, pc.scale, r.render_map);
    if (!beyond.empty() && r.page.cols > 0) {   // the margin the line stage brought in from beyond the stage-1 page
        cv::Mat b;
        cv::resize(beyond, b, r.page.size(), 0, 0, cv::INTER_NEAREST);
        if (r.paper.empty()) r.paper = cv::Mat(r.page.size(), CV_8U, cv::Scalar(255));
        r.paper.setTo(0, b);
    }
    if (opt.erase_fingers && !r.paper.empty()) r.fingers_erased = erase_fingers(r.page, r.paper);
    // before page: perspective only, same rotation and size (made beside the nets when they kept both)
    if (!spec.before.empty() && same_homography(Uh, spec.Uh) && r.rot == spec.o.rot) {
        r.before = spec.before;
        r.before_reused = true;
    } else {
        GridMap hb = at_render_size(compose(Uh, identity_grid(W, H), r.rot, true), opt);
        resample_any(to_bgra(photo), hb, opt.use_gpu, r.before);
    }
    // deformation: page point p -> where it sat on the before page
    {
        const GridMap& g = r.render_map;
        bool swap = r.rot == cv::ROTATE_90_CLOCKWISE || r.rot == cv::ROTATE_90_COUNTERCLOCKWISE;
        int Wa = swap ? H : W;
        double k = (double)g.out_w / Wa;
        r.disp = GridMap::make(g.out_w, g.out_h, 4);
        const double* m = M.ptr<double>();
        for (int i = 0; i < g.gh; ++i)
            for (int j = 0; j < g.gw; ++j) {
                double X = g.x[(size_t)i * g.gw + j], Y = g.y[(size_t)i * g.gw + j];
                double w = X * m[6] + Y * m[7] + m[8];
                double u = (X * m[0] + Y * m[1] + m[2]) / w, v = (X * m[3] + Y * m[4] + m[5]) / w;
                cv::Point2d q = rotate_pt(u, v, W, H, r.rot);
                r.disp.X(i, j) = (float)((q.x + 0.5) * k - 0.5 - 4 * j);
                r.disp.Y(i, j) = (float)((q.y + 0.5) * k - 0.5 - 4 * i);
            }
    }
    r.ms.emplace_back("paper mask + before page", ms_since(t0));
    r.ms.emplace_back("total", ms_since(tall));
    r.ok = true;
    return r;
}

void fill_with_paper(cv::Mat& page_bgra, const cv::Mat& mask, cv::Mat* paper) {
    const int w = page_bgra.cols, h = page_bgra.rows;
    if (mask.empty() || mask.size() != page_bgra.size() || cv::countNonZero(mask) == 0) return;
    // a Telea fill from the area's surroundings at 1/4 size: paper colour and shading continue
    // smoothly across; at full size it would be slow and smear text strokes into the hole
    cv::Mat bgr, small, small_mask, filled;
    cv::cvtColor(page_bgra, bgr, cv::COLOR_BGRA2BGR);
    cv::Size ss(std::max(1, w / 4), std::max(1, h / 4));
    cv::resize(bgr, small, ss, 0, 0, cv::INTER_AREA);
    cv::resize(mask, small_mask, ss, 0, 0, cv::INTER_AREA);
    small_mask = small_mask > 0;
    cv::dilate(small_mask, small_mask, cv::Mat::ones(3, 3, CV_8U));
    cv::inpaint(small, small_mask, filled, 5, cv::INPAINT_TELEA);
    cv::resize(filled, filled, bgr.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat soft;
    cv::GaussianBlur(mask, soft, cv::Size(0, 0), 1.5);
    soft = cv::max(soft, mask);
    for (int y = 0; y < h; ++y) {
        uint8_t* p = page_bgra.ptr<uint8_t>(y);
        const uint8_t* f = filled.ptr<uint8_t>(y);
        const uint8_t* a = soft.ptr<uint8_t>(y);
        uint8_t* pa = paper && !paper->empty() ? paper->ptr<uint8_t>(y) : nullptr;
        for (int x = 0; x < w; ++x) {
            if (!a[x]) continue;
            float t = a[x] / 255.f;
            for (int k = 0; k < 3; ++k) p[4 * x + k] = (uint8_t)std::lround(p[4 * x + k] * (1 - t) + f[3 * x + k] * t);
            if (pa) pa[x] = std::max(pa[x], a[x]);
        }
    }
}

PipelineResult run_scan(const cv::Mat& scan, const PipelineOptions& opt) {
    GpuStageScope gpu_stages(opt.use_gpu);   // the global stage's remap (warpmap.cpp)
    PipelineResult r;
    auto t0 = Clock::now();
    report(opt, 5, "Tracing lines and solving the warp map");
    if (opt.scan_flattened) {
        // an already flattened page (another method's output): the neural-page line stage and its guards
        r.final_map = neural_line_stage(scan, cv::Mat(), opt, r);
        r.rule_net = !r.warp.PH.empty();
        r.net_dropped = r.warp.net.dropped; r.net_added = r.warp.net.added; r.net_joined = r.warp.net.joined; r.net_kept = r.warp.S2.n_net_kept;
    } else {
        r.final_map = unwarp_map(scan, &r.warp);
    }
    r.ms.emplace_back("warp map", ms_since(t0));
    for (auto& m : r.warp.ms) r.ms.emplace_back("  " + m.first, m.second);
    report(opt, 90, "Resampling");
    final_resample(scan, r, opt);
    r.before = to_bgra(scan).clone();
    if (r.before.cols != r.page.cols || r.before.rows != r.page.rows) cv::resize(r.before, r.before, r.page.size(), 0, 0, cv::INTER_AREA);
    const GridMap& g = r.render_map;
    double k = (double)g.out_w / scan.cols;
    r.disp = GridMap::make(g.out_w, g.out_h, 4);
    for (size_t i = 0; i < g.x.size(); ++i) {
        r.disp.x[i] = (float)((g.x[i] + 0.5) * k - 0.5 - 4.0 * (i % g.gw));
        r.disp.y[i] = (float)((g.y[i] + 0.5) * k - 0.5 - 4.0 * (i / g.gw));
    }
    r.ms.emplace_back("total", ms_since(t0));
    r.ok = true;
    return r;
}

}  // namespace fs
