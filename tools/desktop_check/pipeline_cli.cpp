// The app's photo pipeline (core/src/pipeline.cpp, CPU paths) on one photo, for the round-4
// checks: crash fixes, quality gate, orientation, page finder. Driven by tools/round4_check.py.
//   pipeline_cli PHOTO OUT.jpg [LOGITS.npz]
// FS_IGNORE="x0,y0,x1,y1;..." (output-page fractions): areas whose lines the warp map ignores;
// FS_PERSPECTIVE_ONLY=1: homography only, no warp map. With OUT.jpg, OUT.jpg.lines.jpg shows the fitted lines (red: ruled, green: text, blue: grid).
// Prints one JSON line: the crop-screen assessment of every page (assess_pages) and the
// processing result of page 0 (run_photo) without user edits; OUT.jpg = the page (or "-").
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>

#include "npz.h"
#include "cleanup.h"
#include "pipeline.h"

using namespace fs;

static std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

static std::string strs(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + v[i] + "\"";
    return o + "]";
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: pipeline_cli PHOTO OUT.jpg [LOGITS.npz]\n"); return 2; }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    std::vector<float> lg;
    if (argc > 3) {
        Npz z = read_npz(argv[3]);
        const NpyArray& a = z.at("logits");
        lg.assign(a.as<float>(), a.as<float>() + a.count());
    }
    const float* logits = lg.empty() ? nullptr : lg.data();
    std::ostringstream f;
    f << "{\"photo\": \"" << argv[1] << "\", \"assess\": [";
    try {
        auto pages = assess_pages(img, logits, 0);
        for (size_t i = 0; i < pages.size(); ++i) {
            const auto& a = pages[i];
            f << (i ? ", " : "") << "{\"ok\": " << (a.pc.ok ? "true" : "false") << ", \"fit_median_px\": " << num(a.fit_median_px)
              << ", \"reasons\": " << strs(a.reasons) << ", \"whole_frame\": " << (a.pc.whole_frame ? "true" : "false")
              << ", \"n_pages\": " << a.pc.n_pages << ", \"corners\": [";
            for (int k = 0; k < 4; ++k) f << (k ? ", " : "") << num(a.pc.pts[k].x) << ", " << num(a.pc.pts[k].y);
            f << "]}";
        }
        f << "]";
    } catch (const std::exception& e) {
        f << "], \"assess_error\": \"" << e.what() << "\"";
    }
    PipelineOptions o;
    o.use_gpu = false;
    o.seg_logits = logits;
    o.perspective_only = std::getenv("FS_PERSPECTIVE_ONLY") != nullptr;
    o.line_stage = std::getenv("FS_NO_LINES") == nullptr;
    o.share_rows = std::getenv("FS_NO_SHARE") == nullptr;
    o.erase_fingers = std::getenv("FS_NO_FINGERS") == nullptr;
    o.shadow_full_strength = std::getenv("FS_SHADE_REF") == nullptr;   // FS_SHADE_REF: the reference strength
    if (const char* ig = std::getenv("FS_IGNORE")) {
        std::string spec = ig;
        size_t p = 0;
        while (p < spec.size()) {
            size_t q = spec.find(';', p);
            std::array<float, 4> a{};
            if (std::sscanf(spec.substr(p, q - p).c_str(), "%f,%f,%f,%f", &a[0], &a[1], &a[2], &a[3]) == 4) o.ignore.push_back(a);
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    // UVDoc in two steps (the net runs in Python, tools/uvdoc_check.py): FS_UVIN=X.npz writes the
    // net input for the detected corners and exits; FS_UVOUT=Y.npz ("out": 2 x 45 x 31) uses it
    std::vector<float> uv, uv_sh, uv2;
    if (std::getenv("FS_UVIN") || std::getenv("FS_UVOUT")) {
        PageCorners pc = detect_corners(img, o.max_analysis_side, logits, 0);
        if (!pc.ok) { std::printf("%s, \"ok\": false, \"error\": \"no page found\"}\n", f.str().c_str()); return 0; }
        const char* um = std::getenv("FS_UVMODEL");
        o.uv_crop = uv_crop(img.size(), pc.pts, um ? std::atoi(um) : 0);
        if (const char* in = std::getenv("FS_UVIN")) {
            cv::Mat blank;
            if (logits) {
                PageFindResult pf = find_page_net(img, logits);
                if (pf.pages.size() == 2) blank = pf.pages[1].mask;
            }
            std::vector<float> x = uv_input(img, o.uv_crop, blank);
            const UvSpec& sp = uv_spec(o.uv_crop.model);
            write_npz(in, {{"x", NpyArray::f32({1, 3, (size_t)sp.in_h, (size_t)sp.in_w}, x.data())}});
            return 0;
        }
        Npz z = read_npz(std::getenv("FS_UVOUT"));
        const NpyArray& a = z.at("out");
        uv.assign(a.as<float>(), a.as<float>() + a.count());
        if (uv.size() == (size_t)uv_spec(o.uv_crop.model).output_len()) o.uv_out = uv.data();
        // FS_UVOUT2=Z.npz ("out"): the other net's grid for the same corners, the second candidate
        Npz z2;
        if (const char* o2 = std::getenv("FS_UVOUT2")) {
            z2 = read_npz(o2);
            const NpyArray& a2 = z2.at("out");
            uv2.assign(a2.as<float>(), a2.as<float>() + a2.count());
            o.uv_crop2 = uv_crop(img.size(), pc.pts, 1 - o.uv_crop.model);
            if (uv2.size() == (size_t)uv_spec(o.uv_crop2.model).output_len()) o.uv_out2 = uv2.data();
        }
        // the shading map of whichever net has one (SyntheticDoc)
        for (int k = 0; k < 2; ++k) {
            const Npz& zz = k == 0 ? z : z2;
            const UvCrop& cr = k == 0 ? o.uv_crop : o.uv_crop2;
            if (!zz.count("shading") || std::getenv("FS_NO_SHADOWS")) continue;
            const NpyArray& s = zz.at("shading");
            uv_sh.assign(s.as<float>(), s.as<float>() + s.count());
            if (uv_sh.size() == (size_t)uv_spec(cr.model).input_len()) { o.uv_shading = uv_sh.data(); o.uv_shading_crop = cr; }
        }
    }
    PipelineResult r;
    try {
        // FS_SCAN=1: the warp map on the image itself (run_scan), e.g. on an already flattened page
        r = std::getenv("FS_SCAN") ? run_scan(img, o) : run_photo(img, o);
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("exception: ") + e.what();
    }
    f << ", \"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << r.error << "\", \"mode\": \""
      << geom_mode_name(r.mode) << "\", \"fit_median_full_px\": "
      << num(r.detection.ok ? r.fit_median_px / r.detection.scale : -1) << ", \"needs_manual_crop\": " << strs(r.needs_manual_crop)
      << ", \"rot\": " << r.rot << ", \"n_ctrl\": " << r.warp.n_ctrl << ", \"lines\": [" << r.warp.H.size() << ", " << r.warp.V.size()
      << "], \"text_rows\": " << r.warp.n_text << ", \"fitted_lines\": " << r.lines.size() << ", \"n_ignored\": " << r.warp.n_ignored << ", \"uv_agreement\": " << num(r.uv_agreement) << ", \"uv_rejected\": " << (r.uv_rejected ? "true" : "false")
      << ", \"line_distortion\": " << num(r.line_distortion) << ", \"lines_rejected\": " << (r.lines_rejected ? "true" : "false") << ", \"fingers_erased\": " << num(r.fingers_erased) << ", \"uv_model\": " << r.uv_model
      << ", \"uv_spread\": [" << num(r.uv_spread[0]) << ", " << num(r.uv_spread[1]) << "]}";
    // warp-map field size and roughness: max |displacement| and max change between grid neighbours per px
    auto field_stats = [&](const char* name, const Field& a, const Field& b) {
        double mx = 0, gr = 0;
        for (const Field* F : {&a, &b}) {
            if (F->v.empty()) continue;
            for (int i = 0; i < F->gh; ++i)
                for (int j = 0; j < F->gw; ++j) {
                    double v = F->v[(size_t)i * F->gw + j];
                    mx = std::max(mx, std::fabs(v));
                    if (j + 1 < F->gw) gr = std::max(gr, std::fabs(F->v[(size_t)i * F->gw + j + 1] - v) / F->g);
                    if (i + 1 < F->gh) gr = std::max(gr, std::fabs(F->v[(size_t)(i + 1) * F->gw + j] - v) / F->g);
                }
        }
        f << ", \"" << name << "_max\": " << num(mx) << ", \"" << name << "_grad\": " << num(gr);
    };
    if (r.ok) {
        std::string s = f.str();
        s.pop_back();   // reopen the object
        f.str(s);
        f.seekp(0, std::ios_base::end);
        field_stats("global", r.warp.GX, r.warp.GY);
        field_stats("local", r.warp.Fx, r.warp.Fy);
        // local area change: |det(I + dF) - 1| per grid cell, the outer 2 cells left out
        const Field &Fx = r.warp.Fx, &Fy = r.warp.Fy;
        std::vector<double> dets;
        for (int i = 2; i + 3 < Fx.gh; ++i)
            for (int j = 2; j + 3 < Fx.gw; ++j) {
                auto V = [&](const Field& F, int a, int b) { return F.v[(size_t)a * F.gw + b]; };
                double xx = (V(Fx, i, j + 1) - V(Fx, i, j)) / Fx.g, xy = (V(Fx, i + 1, j) - V(Fx, i, j)) / Fx.g;
                double yx = (V(Fy, i, j + 1) - V(Fy, i, j)) / Fy.g, yy = (V(Fy, i + 1, j) - V(Fy, i, j)) / Fy.g;
                dets.push_back(std::fabs((1 + xx) * (1 + yy) - xy * yx - 1));
            }
        if (!dets.empty()) {
            std::sort(dets.begin(), dets.end());
            size_t n20 = 0;
            for (double d : dets) n20 += d > 0.2;
            f << ", \"det_p99\": " << num(dets[(size_t)(0.99 * (dets.size() - 1))]) << ", \"det_frac20\": " << num((double)n20 / dets.size());
        }
        f << "}";
    }
    std::printf("%s\n", f.str().c_str());
    if (r.ok && std::string(argv[2]) != "-") {
        cv::Mat bgr;
        cv::cvtColor(r.page, bgr, cv::COLOR_BGRA2BGR);
        cv::imwrite(argv[2], bgr, {cv::IMWRITE_JPEG_QUALITY, 85});
        const cv::Scalar col[3] = {{0, 0, 255}, {0, 160, 0}, {255, 0, 0}};
        for (const auto& l : r.lines)
            cv::line(bgr, cv::Point((int)(l[1] * bgr.cols), (int)(l[2] * bgr.rows)), cv::Point((int)(l[3] * bgr.cols), (int)(l[4] * bgr.rows)),
                     col[std::min(2, std::max(0, (int)l[0]))], 3);
        for (const auto& a : o.ignore)
            cv::rectangle(bgr, cv::Point((int)(a[0] * bgr.cols), (int)(a[1] * bgr.rows)), cv::Point((int)(a[2] * bgr.cols), (int)(a[3] * bgr.rows)),
                          cv::Scalar(0, 140, 255), 4);
        cv::imwrite(std::string(argv[2]) + ".lines.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 80});
        // FS_RENDER=1: the page as the app shows it by default (Auto color 0.7) -> OUT.jpg.auto.jpg
        if (std::getenv("FS_RENDER")) {
            PageParams pp = page_params(r.page, false);
            cv::Mat shown = render_page(r.page, r.paper, pp, Filter::AutoColor, 0.7f, false, {}, false);
            cv::cvtColor(shown, bgr, cv::COLOR_BGRA2BGR);
            cv::imwrite(std::string(argv[2]) + ".auto.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 88});
        }
    }
    return 0;
}
