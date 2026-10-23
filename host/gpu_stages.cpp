// GPU image stages against their CPU counterparts (Stream E): the dense-map bicubic (stage-1 and
// orientation pages), the global stage's offset remap (warpmap.cpp), and the whole photo
// pipeline with use_gpu on and off.
//   selftest-resample (fs_cli.cpp) also runs gpu_stage_fixtures on each of its fixtures
//   gpu-stages OUT.json PHOTO [PHOTO ...] [--reps N] [--default-policy]
//              [--logits P] [--uvpass1 P] [--uvout P] [--uvout2 P]
//     runs run_photo with use_gpu = false and = true on each photo and compares every image the
//     two produce (stage-1 page, A_img, darkness, orientation scores, map, page, before, paper),
//     byte for byte. The nets come from cached outputs, paths with {n} = the photo's file stem, as
//     pipeline_cli's env (FS_UVPASS1 + FS_UVPASS1_FRAME + FS_UVRECROP, FS_UVOUT with FS_UVMODEL=2,
//     FS_UVOUT2 with FS_UVMODEL2=0): --logits the page finder's output ("logits"); --uvpass1 UVDoc
//     on the whole photo; --uvout SyntheticDoc-HR, --uvout2 UVDoc on the re-crop ("out").
#include "gpu_stages.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "gpu.h"
#include "npz.h"
#include "pipeline.h"
#include "resample.h"
#include "warpmap.h"

namespace fs {

static int max_diff(const cv::Mat& a, const cv::Mat& b, int* n_diff = nullptr) {
    if (a.size() != b.size() || a.type() != b.type()) {
        if (n_diff) *n_diff = -1;
        return 999;
    }
    if (a.empty()) {
        if (n_diff) *n_diff = 0;
        return 0;
    }
    cv::Mat d;
    cv::absdiff(a, b, d);
    double mx;
    cv::minMaxLoc(d.reshape(1), nullptr, &mx);
    if (n_diff) *n_diff = cv::countNonZero(d.reshape(1));
    return (int)std::ceil(mx);
}

static int report(const char* name, const cv::Mat& ref, const cv::Mat& out) {
    int n = 0, mx = max_diff(ref, out, &n);
    std::printf("  %-28s max |diff| %3d   values>0: %8d\n", name, mx, n);
    return mx;
}

// warpmap.cpp's global-stage remap (unwarp_map, A_img), the CPU code as it is there
static cv::Mat offset_remap_cpu(const cv::Mat& src, const std::vector<float>& gx, const std::vector<float>& gy, int gw) {
    const int h = src.rows, w = src.cols, band = 64;
    cv::Mat out(h, w, src.type()), mx(band, w, CV_32F), my(band, w, CV_32F);
    for (int y0 = 0; y0 < h; y0 += band) {
        int y1 = std::min(h, y0 + band);
        for (int y = y0; y < y1; ++y) {
            int iy = y >> 5;
            float ay = (y & 31) * (1.f / 32), vy0 = 1.f - ay;
            for (int x = 0; x < w; ++x) {
                int ix = x >> 5;
                float ax = (x & 31) * (1.f / 32), vx0 = 1.f - ax;
                size_t i00 = (size_t)iy * gw + ix;
                float ddx = gx[i00] * (vy0 * vx0) + gx[i00 + 1] * (vy0 * ax) + gx[i00 + gw] * (ay * vx0) + gx[i00 + gw + 1] * (ay * ax);
                float ddy = gy[i00] * (vy0 * vx0) + gy[i00 + 1] * (vy0 * ax) + gy[i00 + gw] * (ay * vx0) + gy[i00 + gw + 1] * (ay * ax);
                mx.at<float>(y - y0, x) = (float)x + ddx;
                my.at<float>(y - y0, x) = (float)y + ddy;
            }
        }
        cv::Mat dst = out.rowRange(y0, y1);
        cv::remap(src, dst, mx.rowRange(0, y1 - y0), my.rowRange(0, y1 - y0), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    }
    return out;
}

// every GPU stage on, whatever the default policy (GpuStagePolicy), while alive
struct AllGpuStages {
    GpuStagePolicy prev = gpu_stage_policy();
    AllGpuStages() { set_gpu_stage_policy({true, true}); }
    ~AllGpuStages() { set_gpu_stage_policy(prev); }
};

int gpu_stage_fixtures(const cv::Mat& src, const cv::Mat& mx, const cv::Mat& my, std::mt19937& rng) {
    AllGpuStages all;
    int worst = 0;
    cv::Mat bgr, ref, out;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    // dense maps (stage-1 / orientation page): 4 and 3 channels
    cv::remap(src, ref, mx, my, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    if (!remap_dense_gpu(src, mx, my, out)) { std::printf("  GPU dense remap FAILED\n"); return 999; }
    worst = std::max(worst, report("gpu dense (BGRA) vs cv::remap", ref, out));
    cv::remap(bgr, ref, mx, my, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    if (!remap_dense_gpu(bgr, mx, my, out)) { std::printf("  GPU dense remap FAILED\n"); return 999; }
    worst = std::max(worst, report("gpu dense (BGR) vs cv::remap", ref, out));
    // ... with NaN, infinite and huge map values (white, as cvRound's INT_MIN / saturation)
    cv::Mat mx2 = mx.clone(), my2 = my.clone();
    const float bad[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), 1e9f, -1e9f, 40000.5f, -32769.f};
    std::uniform_int_distribution<int> px(0, (int)mx.total() - 1);
    for (int k = 0; k < std::min<int>(64, (int)mx.total()); ++k) {
        float v = bad[k % 7];
        if (k & 1) mx2.at<float>(px(rng)) = v;
        else my2.at<float>(px(rng)) = v;
    }
    cv::remap(src, ref, mx2, my2, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    if (!remap_dense_gpu(src, mx2, my2, out)) { std::printf("  GPU dense remap FAILED\n"); return 999; }
    worst = std::max(worst, report("gpu dense, NaN/inf/huge", ref, out));
    // the global stage's offset remap on an image of the output size: small and large offsets
    cv::Mat img;
    cv::resize(bgr, img, mx.size(), 0, 0, cv::INTER_NEAREST);
    const int gw = img.cols / 32 + 2, gh = img.rows / 32 + 2;
    for (float amp : {4.f, 150.f}) {
        std::uniform_real_distribution<float> U(-amp, amp);
        std::vector<float> gx((size_t)gw * gh), gy((size_t)gw * gh);
        for (size_t i = 0; i < gx.size(); ++i) { gx[i] = U(rng); gy[i] = U(rng); }
        ref = offset_remap_cpu(img, gx, gy, gw);
        if (!remap_offset_gpu(img, gx, gy, gw, gh, 32, out)) { std::printf("  GPU offset remap FAILED\n"); return 999; }
        worst = std::max(worst, report(amp < 10 ? "gpu offset (4 px) vs cpu" : "gpu offset (150 px) vs cpu", ref, out));
    }
    return worst;
}

// ---- gpu-stages: the photo pipeline, use_gpu off vs on

namespace {

struct Nets {
    std::vector<float> lg, p1, out, out2;
};

std::vector<float> load_out(const std::string& path, const char* key) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Npz z = read_npz(path);
    const NpyArray& a = z.at(key);
    return std::vector<float>(a.as<float>(), a.as<float>() + a.count());
}

std::string stem(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    std::string n = s == std::string::npos ? path : path.substr(s + 1);
    size_t d = n.find_last_of('.');
    return d == std::string::npos ? n : n.substr(0, d);
}

std::string subst(std::string p, const std::string& n) {
    for (size_t k; (k = p.find("{n}")) != std::string::npos;) p.replace(k, 3, n);
    return p;
}

// pipeline_cli's FS_UVPASS1 (frame) + FS_UVRECROP + FS_UVOUT (model 2) + FS_UVOUT2 (model 0)
void set_nets(const cv::Mat& img, Nets& n, PipelineOptions& o) {
    o.seg_logits = n.lg.empty() ? nullptr : n.lg.data();
    if (n.out.empty()) return;
    PageCorners pc = detect_corners(img, o.max_analysis_side, o.seg_logits, 0);
    if (!pc.ok) return;
    o.uv_crop = uv_crop(img.size(), pc.pts, 2);
    o.uv_crop2 = uv_crop(img.size(), pc.pts, 0);
    std::array<cv::Point2d, 4> fr = {cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 0), cv::Point2d(img.cols - 1, img.rows - 1),
                                     cv::Point2d(0, img.rows - 1)};
    o.uv_crop_pass1 = uv_crop(img.size(), fr, 0);
    if (n.p1.size() == (size_t)uv_spec(0).output_len()) {
        o.uv_out_pass1 = n.p1.data();
        UvGrid g1 = uv_grid(o.uv_out_pass1, o.uv_crop_pass1);
        if (g1.ok) {
            o.uv_crop = uv_crop_on_grid(img.size(), g1, 2);
            o.uv_crop2 = uv_crop_on_grid(img.size(), g1, 0);
        }
    }
    if (n.out.size() == (size_t)uv_spec(2).output_len()) o.uv_out = n.out.data();
    if (n.out2.size() == (size_t)uv_spec(0).output_len()) o.uv_out2 = n.out2.data();
}

double ms_of(const PipelineResult& r, const std::string& prefix) {
    double t = 0;
    for (const auto& m : r.ms)
        if (m.first.compare(0, prefix.size(), prefix) == 0) t += m.second;
    return t;
}

cv::Mat grid_mat(const GridMap& g) {
    cv::Mat m((int)g.x.size(), 2, CV_32F);
    for (size_t i = 0; i < g.x.size(); ++i) { m.at<float>((int)i, 0) = g.x[i]; m.at<float>((int)i, 1) = g.y[i]; }
    return m;
}

}  // namespace

// gpu-bench PHOTO [--reps N]: each bicubic stage alone at the pipeline's sizes (1600 x 2263 page
// from the full photo; the 4-px grid map; the global remap on the page), CPU vs GPU, median ms
int cmd_gpu_bench(int argc, char** argv) {
    cv::Mat photo = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (photo.empty()) return 1;
    int reps = 5;
    for (int i = 3; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--reps")) reps = std::max(1, std::atoi(argv[i + 1]));
    if (!Gpu::instance().available()) return 2;
    AllGpuStages all;
    std::printf("GPU: %s\nphoto %dx%d\n", Gpu::instance().info().c_str(), photo.cols, photo.rows);
    const int W = 1600, H = 2263;
    // a mild perspective + curl, as a stage-1 unroll map
    auto at = [&](int x, int y) {
        double u = (double)x / (W - 1), v = (double)y / (H - 1);
        double px = photo.cols * (0.08 + 0.84 * u + 0.03 * v * u) + 25 * std::sin(3.1 * v);
        double py = photo.rows * (0.06 + 0.88 * v) + 18 * std::sin(2.3 * u + v);
        return cv::Point2f((float)px, (float)py);
    };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
    std::map<std::string, std::vector<double>> t;
    cv::Mat ref, out, bgra;
    for (int rep = -1; rep < reps; ++rep) {
        auto keep = [&](const char* k, double v) { if (rep >= 0) t[k].push_back(v); };
        // stage-1 page, CPU path: map + cv::remap in 32-row bands on up to 8 threads (pipeline.cpp remap_unroll)
        auto t0 = now();
        ref.create(H, W, photo.type());
        {
            const int band = 32, nb = (H + band - 1) / band;
            int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
            std::vector<std::thread> ts;
            for (int k = 0; k < nt; ++k)
                ts.emplace_back([&, k] {
                    cv::Mat mx(band, W, CV_32F), my(band, W, CV_32F);
                    for (int b = k; b < nb; b += nt) {
                        int y0 = b * band, y1 = std::min(H, y0 + band);
                        for (int y = y0; y < y1; ++y)
                            for (int x = 0; x < W; ++x) {
                                cv::Point2f p = at(x, y);
                                mx.at<float>(y - y0, x) = p.x;
                                my.at<float>(y - y0, x) = p.y;
                            }
                        cv::Mat dst = ref.rowRange(y0, y1);
                        cv::remap(photo, dst, mx.rowRange(0, y1 - y0), my.rowRange(0, y1 - y0), cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                                  cv::Scalar::all(255));
                    }
                });
            for (auto& th : ts) th.join();
        }
        keep("stage1 cpu", ms(t0));
        t0 = now();
        bool ok = remap_cubic_gpu(photo, W, H, at, out);
        keep("stage1 gpu", ms(t0));
        keep("stage1 gpu: upload", Gpu::instance().last_times().upload_ms);
        keep("stage1 gpu: dispatch+readback", Gpu::instance().last_times().run_ms);
        if (!ok || max_diff(ref, out) != 0) { std::printf("stage1: GPU %s\n", ok ? "DIFFERS" : "FAILED"); return 1; }
        t0 = now();
        cv::cvtColor(photo, bgra, cv::COLOR_BGR2BGRA);
        keep("stage1 gpu: BGR->BGRA", ms(t0));
        // the final resample (4-px grid of the same map), CPU resample_cpu vs GPU
        GridMap g = GridMap::make(W, H, 4);
        for (int i = 0; i < g.gh; ++i)
            for (int j = 0; j < g.gw; ++j) {
                cv::Point2f p = at(4 * j, 4 * i);
                g.X(i, j) = p.x;
                g.Y(i, j) = p.y;
            }
        t0 = now();
        cv::Mat rc = resample_cpu(bgra, g);
        keep("final cpu", ms(t0));
        t0 = now();
        ok = Gpu::instance().resample(bgra, g, out);
        keep("final gpu", ms(t0));
        keep("final gpu: upload", Gpu::instance().last_times().upload_ms);
        keep("final gpu: dispatch+readback", Gpu::instance().last_times().run_ms);
        if (!ok || max_diff(rc, out) != 0) { std::printf("final: GPU %s\n", ok ? "DIFFERS" : "FAILED"); return 1; }
        // the global remap on the stage-1 page: smooth offsets up to ~12 px
        const int gw = W / 32 + 2, gh = H / 32 + 2;
        std::vector<float> gx((size_t)gw * gh), gy((size_t)gw * gh);
        for (int i = 0; i < gh; ++i)
            for (int j = 0; j < gw; ++j) {
                gx[(size_t)i * gw + j] = (float)(8 * std::sin(0.11 * i + 0.05 * j));
                gy[(size_t)i * gw + j] = (float)(12 * std::cos(0.07 * j - 0.03 * i));
            }
        t0 = now();
        cv::Mat oc = offset_remap_cpu(ref, gx, gy, gw);
        keep("global remap cpu", ms(t0));
        t0 = now();
        ok = remap_offset_gpu(ref, gx, gy, gw, gh, 32, out);
        keep("global remap gpu", ms(t0));
        keep("global remap gpu: upload", Gpu::instance().last_times().upload_ms);
        keep("global remap gpu: dispatch+readback", Gpu::instance().last_times().run_ms);
        if (!ok || max_diff(oc, out) != 0) { std::printf("global: GPU %s\n", ok ? "DIFFERS" : "FAILED"); return 1; }
    }
    for (const auto& kv : t) std::printf("  %-38s %8.1f ms\n", kv.first.c_str(), median(kv.second));
    std::printf("all GPU outputs identical to CPU\n");
    return 0;
}

// cg-tol OUT.txt RTOL PHOTO... [--reps N] [--logits P] [--uvpass1 P] [--uvout P] [--uvout2 P]
// Stream E step 3, report only: run_photo (CPU) with the warp map's CG tolerance at the
// reference 1e-8 and at RTOL; warp-map time, CG iterations and how far map and page move.
int cmd_cg_tol(int argc, char** argv) {
    std::string out_txt = argv[2], p_logits, p_pass1, p_out, p_out2;
    double rtol = std::atof(argv[3]);
    int reps = 3;
    std::vector<std::string> photos;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--logits") && i + 1 < argc) p_logits = argv[++i];
        else if (!std::strcmp(argv[i], "--uvpass1") && i + 1 < argc) p_pass1 = argv[++i];
        else if (!std::strcmp(argv[i], "--uvout") && i + 1 < argc) p_out = argv[++i];
        else if (!std::strcmp(argv[i], "--uvout2") && i + 1 < argc) p_out2 = argv[++i];
        else photos.push_back(argv[i]);
    }
    if (!(rtol > 0)) return 2;
    std::ofstream f(out_txt);
    char line[512];
    std::snprintf(line, sizeof line, "%-28s %9s %9s %7s %7s %10s %9s %9s\n", "photo", "warp 1e-8", "warp tol", "iters", "iters", "map max px",
                  "page max", "page n>0");
    std::printf("CG rtol 1e-8 vs %g, CPU, median of %d\n%s", rtol, reps, line);
    f << "CG rtol 1e-8 vs " << rtol << ", CPU path, warp-map ms median of " << reps << "\n" << line;
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    double sum_ref = 0, sum_tol = 0;
    for (const std::string& path : photos) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); return 1; }
        std::string n = stem(path);
        Nets nets;
        if (!p_logits.empty()) nets.lg = load_out(subst(p_logits, n), "logits");
        if (!p_pass1.empty()) nets.p1 = load_out(subst(p_pass1, n), "out");
        if (!p_out.empty()) nets.out = load_out(subst(p_out, n), "out");
        if (!p_out2.empty()) nets.out2 = load_out(subst(p_out2, n), "out");
        PipelineResult r[2];
        std::vector<double> t[2];
        int iters[2] = {0, 0};
        for (int rep = 0; rep < reps; ++rep)
            for (int k = 0; k < 2; ++k) {
                set_warp_cg_rtol(k ? rtol : 1e-8);
                PipelineOptions o;
                o.use_gpu = false;
                set_nets(img, nets, o);
                r[k] = run_photo(img, o);
                t[k].push_back(ms_of(r[k], "warp map"));
            }
        set_warp_cg_rtol(0);   // the default again
        for (int k = 0; k < 2; ++k)
            for (int i = 0; i < 6; ++i) iters[k] += r[k].warp.cg_iters[i];
        double map_px = 0;
        if (r[0].final_map.x.size() == r[1].final_map.x.size())
            for (size_t i = 0; i < r[0].final_map.x.size(); ++i)
                map_px = std::max(map_px, (double)std::abs(r[0].final_map.x[i] - r[1].final_map.x[i]) + std::abs(r[0].final_map.y[i] - r[1].final_map.y[i]));
        else map_px = -1;
        int nd = 0, md = max_diff(r[0].page, r[1].page, &nd);
        double a = median(t[0]), b = median(t[1]);
        sum_ref += a;
        sum_tol += b;
        std::snprintf(line, sizeof line, "%-28s %9.1f %9.1f %7d %7d %10.4f %9d %9d\n", n.c_str(), a, b, iters[0], iters[1], map_px, md, nd);
        std::printf("%s", line);
        f << line;
    }
    std::snprintf(line, sizeof line, "sum warp-map ms: 1e-8 %.1f, %g %.1f (%.1f%%)\n", sum_ref, rtol, sum_tol, sum_ref > 0 ? 100 * (sum_tol - sum_ref) / sum_ref : 0);
    std::printf("%s", line);
    f << line;
    return 0;
}

int cmd_gpu_stages(int argc, char** argv) {
    std::string out_json = argv[2], p_logits, p_pass1, p_out, p_out2;
    int reps = 1;
    bool default_policy = false;   // --default-policy: GpuStagePolicy as the app has it; else every GPU stage on
    std::vector<std::string> photos;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--logits") && i + 1 < argc) p_logits = argv[++i];
        else if (!std::strcmp(argv[i], "--uvpass1") && i + 1 < argc) p_pass1 = argv[++i];
        else if (!std::strcmp(argv[i], "--uvout") && i + 1 < argc) p_out = argv[++i];
        else if (!std::strcmp(argv[i], "--uvout2") && i + 1 < argc) p_out2 = argv[++i];
        else if (!std::strcmp(argv[i], "--default-policy")) default_policy = true;
        else photos.push_back(argv[i]);
    }
    bool gpu = Gpu::instance().available();
    std::printf("GPU: %s\n", gpu ? Gpu::instance().info().c_str() : "unavailable");
    if (!gpu) return 2;
    GpuStagePolicy policy = gpu_stage_policy();
    if (!default_policy) policy = {true, true};
    set_gpu_stage_policy(policy);
    std::printf("GPU stages: dense %s, offset %s\n", policy.dense ? "on" : "off", policy.offset ? "on" : "off");
    // stage timers (r.ms names) reported for both paths
    const std::vector<std::pair<const char*, const char*>> stages = {
        {"stage1", "stage-1 page"}, {"orient", "orientation"}, {"global_remap", "  global_remap+darkness"},
        {"final", "final resample"}, {"before", "paper mask + before page"}, {"netcheck", "flattening net check"}, {"total", "total"}};
    std::ofstream js(out_json);
    js << "[\n";
    int n_bad = 0, worst = 0;
    std::map<std::string, std::vector<double>> sum_cpu, sum_gpu;
    for (size_t pi = 0; pi < photos.size(); ++pi) {
        const std::string& path = photos[pi];
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); return 1; }
        std::string n = stem(path);
        Nets nets;
        if (!p_logits.empty()) nets.lg = load_out(subst(p_logits, n), "logits");
        if (!p_pass1.empty()) nets.p1 = load_out(subst(p_pass1, n), "out");
        if (!p_out.empty()) nets.out = load_out(subst(p_out, n), "out");
        if (!p_out2.empty()) nets.out2 = load_out(subst(p_out2, n), "out");
        PipelineResult rc, rg;
        std::map<std::string, std::vector<double>> tc, tg;
        GpuStageCounts c0 = gpu_stage_counts();
        const int warm = pi == 0 ? 1 : 0;   // the first photo once more first: the first GPU dispatches
        for (int rep = -warm; rep < reps; ++rep)
            for (int k = 0; k < 2; ++k) {
                PipelineOptions o;
                o.use_gpu = k == 1;
                set_nets(img, nets, o);
                PipelineResult r = run_photo(img, o);
                if (rep >= 0)
                    for (const auto& s : stages) (k ? tg : tc)[s.first].push_back(ms_of(r, s.second));
                (k ? rg : rc) = std::move(r);
            }
        GpuStageCounts c1 = gpu_stage_counts();
        long n_dense = (c1.dense - c0.dense) / (reps + warm), n_offset = (c1.offset - c0.offset) / (reps + warm);
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        struct Cmp { const char* name; cv::Mat a, b; };
        std::vector<Cmp> cmp = {{"stage1", rc.stage1, rg.stage1}, {"A_img", rc.warp.A_img, rg.warp.A_img}, {"darkness", rc.warp.D, rg.warp.D},
                                {"map", grid_mat(rc.final_map), grid_mat(rg.final_map)}, {"page", rc.page, rg.page},
                                {"before", rc.before, rg.before}, {"paper", rc.paper, rg.paper}};
        bool same = rc.ok == rg.ok && rc.rot == rg.rot && rc.orient_sh == rg.orient_sh && rc.orient_sv == rg.orient_sv &&
                    rc.orient_upright == rg.orient_upright && rc.uv_spread[0] == rg.uv_spread[0] && rc.uv_spread[1] == rg.uv_spread[1] &&
                    rc.mode == rg.mode && rc.uv_model == rg.uv_model;
        std::printf("%-36s mode %-10s rot %2d gpu calls %ld+%ld+%d %s", n.c_str(), geom_mode_name(rg.mode), rg.rot, n_dense, n_offset,
                    (int)rg.gpu_resample, same ? "decisions =" : "DECISIONS DIFFER");
        js << (pi ? ",\n" : "") << "{\"photo\": \"" << n << "\", \"mode\": \"" << geom_mode_name(rg.mode) << "\", \"gpu_resample\": "
           << (rg.gpu_resample ? "true" : "false") << ", \"gpu_dense_calls\": " << n_dense << ", \"gpu_offset_calls\": " << n_offset
           << ", \"rot\": " << rg.rot << ", \"uv_model\": " << rg.uv_model << ", \"uv_spread\": [" << rg.uv_spread[0] << ", " << rg.uv_spread[1]
           << "], \"uv_defines_page\": " << (rg.uv_defines_page ? "true" : "false") << ", \"n_ctrl\": " << rg.warp.n_ctrl
           << ", \"decisions_equal\": " << (same ? "true" : "false") << ", \"diff\": {";
        for (size_t i = 0; i < cmp.size(); ++i) {
            int nd = 0, md = max_diff(cmp[i].a, cmp[i].b, &nd);
            worst = std::max(worst, md);
            same &= md == 0;
            std::printf(" | %s %d", cmp[i].name, md);
            js << (i ? ", " : "") << "\"" << cmp[i].name << "\": [" << md << ", " << nd << "]";
        }
        js << "}, \"ms_cpu\": {";
        for (size_t i = 0; i < stages.size(); ++i) {
            double v = median(tc[stages[i].first]);
            sum_cpu[stages[i].first].push_back(v);
            js << (i ? ", " : "") << "\"" << stages[i].first << "\": " << std::lround(v * 10) / 10.0;
        }
        js << "}, \"ms_gpu\": {";
        for (size_t i = 0; i < stages.size(); ++i) {
            double v = median(tg[stages[i].first]);
            sum_gpu[stages[i].first].push_back(v);
            js << (i ? ", " : "") << "\"" << stages[i].first << "\": " << std::lround(v * 10) / 10.0;
        }
        js << "}}";
        std::printf(" | total cpu %.0f gpu %.0f ms\n", median(tc["total"]), median(tg["total"]));
        if (!same) ++n_bad;
    }
    js << "\n]\n";
    std::printf("\nstage (median over photos, ms)      CPU      GPU\n");
    for (const auto& s : stages) {
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
        std::printf("  %-30s %8.1f %8.1f\n", s.first, median(sum_cpu[s.first]), median(sum_gpu[s.first]));
    }
    std::printf("photos %zu, not identical %d, worst |diff| %d (gate: 0)\n", photos.size(), n_bad, worst);
    return n_bad == 0 ? 0 : 1;
}

}  // namespace fs
