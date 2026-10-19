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
#include <cstdlib>
#include <sstream>
#include <string>

#include "npz.h"
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
    PipelineResult r;
    try {
        r = run_photo(img, o);
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("exception: ") + e.what();
    }
    f << ", \"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << r.error << "\", \"mode\": \""
      << (r.mode == GeomMode::Surface ? "surface" : "homography") << "\", \"fit_median_full_px\": "
      << num(r.detection.ok ? r.fit_median_px / r.detection.scale : -1) << ", \"needs_manual_crop\": " << strs(r.needs_manual_crop)
      << ", \"rot\": " << r.rot << ", \"n_ctrl\": " << r.warp.n_ctrl << ", \"lines\": [" << r.warp.H.size() << ", " << r.warp.V.size()
      << "], \"text_rows\": " << r.warp.n_text << ", \"fitted_lines\": " << r.lines.size() << ", \"n_ignored\": " << r.warp.n_ignored << "}";
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
    }
    return 0;
}
