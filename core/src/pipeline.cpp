#include "pipeline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#include "gpu.h"
#include "log.h"
#include "orient.h"

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

// stage-1 page: cv2.remap(photo, U) with U evaluated exactly per pixel, in row bands
cv::Mat remap_unroll(const cv::Mat& photo, const Unroll& U) {
    cv::Mat out(U.H, U.W, photo.type());
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
        if (skin_frac < 0.5 || reach < 2 * border || edge_frac > 0.03) continue;
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

// the lines the warp map used, as output-page fractions (PipelineResult::lines)
std::vector<std::array<float, 5>> fitted_lines(const WarpDebug& d, int W, int H, int rot) {
    bool swap = rot == cv::ROTATE_90_CLOCKWISE || rot == cv::ROTATE_90_COUNTERCLOCKWISE;
    double Wr = swap ? H : W, Hr = swap ? W : H;
    std::vector<std::array<float, 5>> out;
    auto add = [&](int kind, double x0, double y0, double x1, double y1) {
        cv::Point2d a = rotate_pt(x0, y0, W, H, rot), b = rotate_pt(x1, y1, W, H, rot);
        out.push_back({(float)kind, (float)(a.x / Wr), (float)(a.y / Hr), (float)(b.x / Wr), (float)(b.y / Hr)});
    };
    for (const auto& L : d.H) add(0, L[1], L[0], L[2], L[0]);
    for (const auto& L : d.V) add(0, L[0], L[1], L[0], L[2]);
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

PipelineResult run_photo(const cv::Mat& photo, const PipelineOptions& opt) {
    PipelineResult r;
    auto t0 = Clock::now(), tall = Clock::now();
    report(opt, 2, "Finding the page");
    Unroll U;
    std::array<cv::Point2d, 4> corners;
    PageCorners pc;
    if (!opt.corners_edited) {
        pc = detect_corners(photo, opt.max_analysis_side, opt.seg_logits, opt.page_index);
        r.detection = pc;
        if (!pc.ok) { r.error = "no page found"; return r; }
        corners = pc.pts;
    } else {
        corners = opt.corners;
    }
    r.ms.emplace_back("detect", ms_since(t0));
    cv::Point2d wv = corners[1] - corners[0], hv = corners[3] - corners[0];
    bool landscape = std::sqrt(wv.x * wv.x + wv.y * wv.y) > std::sqrt(hv.x * hv.x + hv.y * hv.y);
    int W = opt.out_w, H = (int)std::lround(opt.out_w * 297.0 / 210.0);
    if (landscape) std::swap(W, H);
    U.W = W; U.H = H;
    // homography from the 4 corners (always: v8 path and the orientation page)
    std::vector<cv::Point2f> srcq, dstq = {{0, 0}, {(float)(W - 1), 0}, {(float)(W - 1), (float)(H - 1)}, {0, (float)(H - 1)}};
    for (auto& c : corners) srcq.emplace_back((float)c.x, (float)c.y);
    cv::Mat M = cv::getPerspectiveTransform(srcq, dstq), Mi;
    cv::invert(M, Mi, cv::DECOMP_LU);
    Unroll Uh = U;
    Uh.mode = GeomMode::Homography;
    for (int i = 0; i < 9; ++i) Uh.Minv[i] = Mi.at<double>(i / 3, i % 3);
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
    // Flattening nets (app): a grid replaces the surface model / homography as stage 1 when its page
    // outline agrees with the page finder (or with corners set by hand); with two nets, the one whose
    // stage-1 page has the straighter text rows wins. The line stage still runs on top.
    if (opt.uv_out && !opt.perspective_only) {
        auto t_uv = Clock::now();
        const bool use_mask = !opt.corners_edited && pc.ok && !pc.det.paper.empty();
        const double need = use_mask ? opt.uv_min_agreement : opt.uv_min_agreement_quad;
        struct Cand { UvGrid g; double agree = 0; int model = 0; };
        std::vector<Cand> ok;
        double best_agree = 0;
        for (int k = 0; k < 2; ++k) {
            const float* out = k == 0 ? opt.uv_out : opt.uv_out2;
            if (!out) continue;
            const UvCrop& crop = k == 0 ? opt.uv_crop : opt.uv_crop2;
            Cand c;
            c.g = uv_grid(out, crop);
            c.model = crop.model;
            c.agree = use_mask ? uv_agreement(c.g, pc.det.paper, pc.scale) : uv_agreement(c.g, corners, photo.size());
            best_agree = std::max(best_agree, c.agree);
            if (c.g.ok && c.g.turned == landscape && c.agree >= need) ok.push_back(std::move(c));
        }
        r.uv_agreement = best_agree;
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
                    r.uv_spread[k] = text_row_spread(remap_unroll(photo, Us), &rows);
                    if (rows < 5) r.uv_spread[k] = -1;
                }
                if (r.uv_spread[0] >= 0 && r.uv_spread[1] >= 0 && r.uv_spread[1] < r.uv_spread[0] - opt.uv_prefer_margin) pick = 1;
            }
            U = Uh;
            U.mode = GeomMode::Neural;
            U.uv = std::move(ok[pick].g);
            r.uv_agreement = ok[pick].agree;
            r.uv_model = ok[pick].model;
            r.mode = GeomMode::Neural;
            r.fell_back = false;
        } else {
            r.uv_rejected = true;
        }
        r.ms.emplace_back("flattening net check", ms_since(t_uv));
    }
    t0 = Clock::now();
    report(opt, 20, "Unrolling the page");
    cv::Mat stage1 = remap_unroll(photo, U);
    r.ms.emplace_back("stage-1 page", ms_since(t0));
    if (opt.auto_orient) {
        t0 = Clock::now();
        cv::Mat op = (opt.orient_on_stage1 || r.mode == GeomMode::Homography) ? stage1 : remap_unroll(photo, Uh);
        Orientation o = decide_orientation(op, opt.reference_orient ? 0.0 : opt.orient_min_conf);
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
        warp4 = unwarp_map(stage1, &r.warp, ignore.empty() ? nullptr : &ignore, r.mode == GeomMode::Neural && opt.share_rows, opt.faint_lines);
        r.lines = fitted_lines(r.warp, W, H, r.rot);
        // on a UVDoc page the line refinement only corrects what is left: a correction that
        // stretches or squeezes the page locally is a tracing failure (dense staves, broken table
        // rules), and the UVDoc page is kept as it is
        if (r.mode == GeomMode::Neural) {
            double frac = 0, p99 = local_distortion(r.warp, &frac);
            r.line_distortion = p99;
            bool bad = p99 > opt.max_line_distortion || frac > opt.max_line_distortion_frac;
            if (bad && opt.faint_lines) {
                // the faint rules may be what overstretches it (picture edges, a rule traced into
                // text): the reference lines alone
                WarpDebug w2;
                GridMap m2 = unwarp_map(stage1, &w2, ignore.empty() ? nullptr : &ignore, opt.share_rows, false);
                double f2 = 0, q2 = local_distortion(w2, &f2);
                if (q2 <= opt.max_line_distortion && f2 <= opt.max_line_distortion_frac) {
                    warp4 = std::move(m2);
                    r.warp = std::move(w2);
                    r.lines = fitted_lines(r.warp, W, H, r.rot);
                    r.line_distortion = q2;
                    bad = false;
                }
            }
            if (bad) {
                warp4 = identity_grid(W, H);
                r.lines_rejected = true;
            }
        }
    }
    r.stage1 = stage1;
    r.ms.emplace_back("warp map", ms_since(t0));
    for (auto& m : r.warp.ms) r.ms.emplace_back("  " + m.first, m.second);
    t0 = Clock::now();
    report(opt, 85, "Composing the map");
    r.final_map = compose(U, warp4, r.rot, true);
    r.ms.emplace_back("compose (4-px grid)", ms_since(t0));
    report(opt, 90, "Resampling");
    final_resample(photo, r, opt);
    if (opt.uv_shading && opt.remove_shadows) {
        // shadows: the page divided by the net's shading map, sampled through the same map
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
    t0 = Clock::now();
    report(opt, 95, "Finishing");
    // paper mask: from this run's detection, or - when the corners were edited - from a fresh
    // detection if its corners agree with the user's (otherwise the outline is not trusted)
    if (opt.corners_edited) {
        PageCorners pd = detect_corners(photo, opt.max_analysis_side, opt.seg_logits, opt.page_index);
        double diag = std::hypot(photo.cols, photo.rows), worst = 0;
        for (int k = 0; k < 4 && pd.ok; ++k) worst = std::max(worst, cv::norm(pd.pts[k] - corners[k]));
        if (pd.ok && worst < 0.03 * diag) pc = pd;
    }
    if (pc.ok && !pc.det.paper.empty()) r.paper = paper_alpha(pc.det.paper, pc.scale, r.render_map);
    if (opt.erase_fingers && !r.paper.empty()) r.fingers_erased = erase_fingers(r.page, r.paper);
    // before page: perspective only, same rotation and size
    cv::Mat src = to_bgra(photo);
    GridMap hb = at_render_size(compose(Uh, identity_grid(W, H), r.rot, true), opt);
    resample_any(src, hb, opt.use_gpu, r.before);
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
    PipelineResult r;
    auto t0 = Clock::now();
    report(opt, 5, "Tracing lines and solving the warp map");
    r.final_map = unwarp_map(scan, &r.warp);
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
