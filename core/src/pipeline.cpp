#include "pipeline.h"

#include <opencv2/imgproc.hpp>
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
        bool trusted = r.needs_manual_crop.empty() && !opt.force_homography;
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
    GridMap warp4 = unwarp_map(stage1, &r.warp);
    r.stage1 = stage1;
    r.ms.emplace_back("warp map", ms_since(t0));
    for (auto& m : r.warp.ms) r.ms.emplace_back("  " + m.first, m.second);
    t0 = Clock::now();
    report(opt, 85, "Composing the map");
    r.final_map = compose(U, warp4, r.rot, true);
    r.ms.emplace_back("compose (4-px grid)", ms_since(t0));
    report(opt, 90, "Resampling");
    final_resample(photo, r, opt);
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
