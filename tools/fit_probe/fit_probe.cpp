// Cross-build check of the surface fit (core/src/surface.cpp): runs the crop-screen detection and
// fit_surface on one photo and prints every intermediate value as hex floats, so that a desktop
// (MSVC) and an Android (NDK clang) build can be diffed line by line.
//   fit_probe PHOTO LOGITS.npz [THREADS]
//   fit_probe --list LIST   (LIST: "name photo logits" lines: each page's fit residual, corners, reasons)
// LOGITS.npz: the page finder's "logits" (as pagefind_cli / gate_b.py write them).
#include "surface.cpp"   // the fit's static helpers (make_problem, Problem)

#include <opencv2/imgcodecs.hpp>
#include <cstdio>

#include "npz.h"
#include "pagefind.h"
#include "corners.h"
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <string>
#include "pipeline.h"

using namespace fs;

static void hx(const char* tag, double v) { std::printf("%s %a %.17g\n", tag, v, v); }

static void fit_trace(const std::array<std::vector<cv::Point2d>, 4>& side_pts, const std::array<cv::Point2d, 4>& corners,
                      int w, int h, bool landscape, double focal_px, int threads) {
    double f = focal_px > 0 ? focal_px : 0.72 * std::max(h, w);
    double cx = w / 2.0, cy = h / 2.0;
    double W = landscape ? 297.0 : 210.0, H = landscape ? 210.0 : 297.0;
    hx("f", f);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    std::vector<cv::Point3d> obj = {{0, 0, 0}, {W, 0, 0}, {W, H, 0}, {0, H, 0}};
    std::vector<cv::Point2d> img(corners.begin(), corners.end());
    cv::Mat rvec0, tvec0;
    cv::solvePnP(obj, img, K, cv::noArray(), rvec0, tvec0, false, cv::SOLVEPNP_IPPE);
    for (int k = 0; k < 3; ++k) { hx("rvec0", rvec0.at<double>(k)); hx("tvec0", tvec0.at<double>(k)); }
    // libm on a few values the profile uses
    for (double t : {0.1, 0.3, 0.6, 1.0, 1.7}) { hx("cos", std::cos(t)); hx("sin", std::sin(t)); hx("pow3", std::pow(t, 3.0)); hx("pow4", std::pow(t, 4.0)); }
    const double starts[5][4] = {{0, 0, 0, 0}, {0, .6, 0, 0}, {0, -.6, 0, 0}, {-.6, 0, 0, 0}, {.6, 0, 0, 0}};
    for (int i = 0; i < 10; ++i) {
        char ax = i < 5 ? 'u' : 'v';
        Problem P = make_problem(ax, 50, 70, side_pts, corners, W, H, f, cx, cy);
        Vec x0(10);
        for (int k = 0; k < 4; ++k) x0[k] = starts[i % 5][k];
        for (int k = 0; k < 3; ++k) { x0[4 + k] = rvec0.at<double>(k); x0[7 + k] = tvec0.at<double>(k); }
        Vec r0 = P(x0);
        double s = 0;
        for (double v : r0) s += v;
        std::printf("start %d res0_sum %a\n", i, s);
        LsqResult r = least_squares_cauchy(P, x0, 2.0, 150, 1);
        std::printf("coarse %d %c cost %a %.17g nfev %d status %d x", i, ax, r.cost, r.cost, r.nfev, r.status);
        for (double v : r.x) std::printf(" %a", v);
        std::printf("\n");
    }
    SurfaceModel m = fit_surface(side_pts, corners, w, h, landscape, focal_px, threads);
    std::printf("final ok %d axis %c cost %a %.17g\n", m.ok, m.axis, m.cost, m.cost);
    hx("median_px", m.median_px);
    hx("p90_px", m.p90_px);
    for (int k = 0; k < 4; ++k) hx("c", m.c[k]);
}


static uint64_t hmat(const cv::Mat& m) {
    uint64_t h = 1469598103934665603ull;
    for (int y = 0; y < m.rows; ++y) {
        const uint8_t* q = m.ptr<uint8_t>(y);
        for (size_t i = 0; i < (size_t)m.cols * m.elemSize(); ++i) h = (h ^ q[i]) * 1099511628211ull;
    }
    return h;
}
static uint64_t hpts(const std::vector<cv::Point2d>& v) {
    uint64_t h = 1469598103934665603ull;
    for (auto& p : v) { uint64_t a, b; std::memcpy(&a, &p.x, 8); std::memcpy(&b, &p.y, 8); h = (h ^ a) * 1099511628211ull; h = (h ^ b) * 1099511628211ull; }
    return h;
}
// the stages of find_page_net + corners_from_contour, each hashed
static void trace_find(const cv::Mat& img, const float* logits) {
    const int H = img.rows, W = img.cols;
    const double s = (double)PAGEFIND_WORK / std::max(H, W);
    const int sw = (int)std::nearbyint(W * s), sh = (int)std::nearbyint(H * s);
    cv::Mat sm;
    cv::resize(img, sm, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
    std::printf("stage sm %dx%d %016llx\n", sw, sh, (unsigned long long)hmat(sm));
    cv::Mat pg[2];
    for (int c = 0; c < 2; ++c) {
        cv::Mat l(SEG_SIZE, SEG_SIZE, CV_32F, const_cast<float*>(logits) + (size_t)c * SEG_SIZE * SEG_SIZE), up;
        cv::resize(l, up, cv::Size(sw, sh), 0, 0, cv::INTER_CUBIC);
        std::printf("stage up%d %016llx\n", c, (unsigned long long)hmat(up));
        for (int y = 0; y < up.rows; ++y) {
            float* q = up.ptr<float>(y);
            for (int x = 0; x < up.cols; ++x) q[x] = 1.f / (1.f + (float)std::exp(-(double)q[x]));
        }
        std::printf("stage sig%d %016llx\n", c, (unsigned long long)hmat(up));
        pg[c] = up;
    }
    PageSelection sel = select_page(pg[0], pg[1]);
    std::printf("stage sel mask %016llx\n", (unsigned long long)hmat(sel.mask));
    PageFindResult f = find_page_net(img, logits);
    for (auto& fp : f.pages) std::printf("stage page contour n %zu %016llx mask %016llx edge %a\n", fp.contour.size(), (unsigned long long)hpts(fp.contour), (unsigned long long)hmat(fp.mask), fp.edge_strength);
    PageFindResult g = find_page_net(img, logits, false);
    for (auto& fp : g.pages) std::printf("stage page(norefine) contour n %zu %016llx\n", fp.contour.size(), (unsigned long long)hpts(fp.contour));
}

static void trace_refine(const cv::Mat& img) {
    const double s = (double)PAGEFIND_WORK / std::max(img.rows, img.cols);
    cv::Mat sm0;
    cv::resize(img, sm0, cv::Size((int)std::nearbyint(img.cols * s), (int)std::nearbyint(img.rows * s)), 0, 0, cv::INTER_AREA);
    cv::Mat blur, lab8, lab;
    cv::GaussianBlur(sm0, blur, cv::Size(0, 0), 1.0);
    std::printf("stage r.blur %016llx\n", (unsigned long long)hmat(blur));
    cv::cvtColor(blur, lab8, cv::COLOR_BGR2Lab);
    std::printf("stage r.lab8 %016llx\n", (unsigned long long)hmat(lab8));
    lab8.convertTo(lab, CV_32F);
    std::vector<cv::Mat> ch(3);
    cv::split(lab, ch);
    for (int k = 0; k < 3; ++k) {
        cv::Mat gx, gy;
        cv::Scharr(ch[k], gx, CV_32F, 1, 0);
        cv::Scharr(ch[k], gy, CV_32F, 0, 1);
        std::printf("stage r.scharr%d %016llx %016llx\n", k, (unsigned long long)hmat(gx), (unsigned long long)hmat(gy));
    }
    cv::Mat k1 = cv::getGaussianKernel(33, 4, CV_64F);
    std::printf("stage r.gk64 %016llx\n", (unsigned long long)hmat(k1));
    cv::Mat col(200, 1, CV_64F), bl;
    for (int i = 0; i < 200; ++i) col.at<double>(i) = 100 + 37.3 * std::sin(i * 0.1) + i * 0.731;
    cv::GaussianBlur(col, bl, cv::Size(1, 0), 4);
    std::printf("stage r.gblur64 %016llx\n", (unsigned long long)hmat(bl));
    cv::Mat src(64, 64, CV_32F), Xf(10, 10, CV_32F), Yf(10, 10, CV_32F), out;
    for (int i = 0; i < 64 * 64; ++i) src.at<float>(i) = (float)std::sqrt(i * 1.37);
    for (int i = 0; i < 100; ++i) { Xf.at<float>(i) = 3.3f + i * 0.537f; Yf.at<float>(i) = 7.1f + i * 0.291f; }
    cv::remap(src, out, Xf, Yf, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    std::printf("stage r.remap %016llx\n", (unsigned long long)hmat(out));
}

// --list LIST: one "name photo logits" per line -> one line per photo: the crop-screen assessment
// (assess_pages) of every page: fit_median_px, corners and the quality-gate reasons
static int run_list(const char* list) {
    std::FILE* f = std::fopen(list, "r");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", list); return 1; }
    char name[512], photo[1024], logits[1024];
    while (std::fscanf(f, "%511s %1023s %1023s", name, photo, logits) == 3) {
        cv::Mat img = cv::imread(photo, cv::IMREAD_COLOR);
        if (img.empty()) { std::printf("%s unreadable\n", name); continue; }
        Npz z = read_npz(logits);
        auto pages = assess_pages(img, z.at("logits").as<float>(), 0);
        std::printf("%s", name);
        for (auto& a : pages) {
            std::printf(" | %a %.9g", a.fit_median_px, a.fit_median_px);
            for (int k = 0; k < 4; ++k) std::printf(" %a %a", a.pc.pts[k].x, a.pc.pts[k].y);
            for (auto& r : a.reasons) std::printf(" [%s]", r.c_str());
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    std::fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--list") return run_list(argv[2]);
    if (argc < 3) { std::fprintf(stderr, "usage: fit_probe PHOTO LOGITS.npz [THREADS] | --list LIST\n"); return 2; }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    Npz z = read_npz(argv[2]);
    const NpyArray& lg = z.at("logits");
    trace_find(img, lg.as<float>());
    trace_refine(img);
    int threads = argc > 3 ? std::atoi(argv[3]) : 8;
    PageCorners pc = detect_corners(img, 2400, lg.as<float>(), 0);
    std::printf("ok %d scale %a landscape %d\n", pc.ok, pc.scale, pc.det.landscape);
    for (auto& c : pc.det.corners) std::printf("corner %a %a\n", c.x, c.y);
    for (int s = 0; s < 4; ++s) {
        double hs = 0;
        for (auto& p : pc.det.side_pts[s]) hs = hs * 1.000001 + p.x * 3 + p.y;
        std::printf("side %d n %zu hash %a\n", s, pc.det.side_pts[s].size(), hs);
    }
    const Detection& D = pc.det;
    int sw = (int)std::lround(img.cols * pc.scale), sh = (int)std::lround(img.rows * pc.scale);
    if (pc.scale >= 1.0) { sw = img.cols; sh = img.rows; }
    std::printf("size %d %d\n", sw, sh);
    fit_trace(D.side_pts, D.corners, sw, sh, D.landscape, 0, threads);
    auto pages = assess_pages(img, lg.as<float>(), 0);
    for (auto& a : pages) hx("assess_fit_median_px", a.fit_median_px);
    return 0;
}
