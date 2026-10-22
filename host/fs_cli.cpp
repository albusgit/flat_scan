// Host harness for the parity / acceptance tests. Every command reads and writes plain files
// (images, NPZ) so the Python reference tools can score the results.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu.h"
#include "npz.h"
#include "resample.h"
#include "cli_ext.h"
#include "gpu_stages.h"

using namespace fs;

static bool has(int argc, char** argv, const char* flag) {
    for (int i = 0; i < argc; ++i)
        if (!std::strcmp(argv[i], flag)) return true;
    return false;
}
static const char* opt(int argc, char** argv, const char* flag, const char* def = nullptr) {
    for (int i = 0; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}

GridMap load_grid(const std::string& path) {
    Npz z = read_npz(path);
    int stride = (int)z.at("stride").at(0);
    int h = (int)z.at("shape").at(0), w = (int)z.at("shape").at(1);
    GridMap g = GridMap::make(w, h, stride);
    const NpyArray &ax = z.at("map_x"), &ay = z.at("map_y");
    if ((int)ax.shape[0] != g.gh || (int)ax.shape[1] != g.gw) throw std::runtime_error("grid shape mismatch");
    for (size_t i = 0; i < g.x.size(); ++i) { g.x[i] = (float)ax.at(i); g.y[i] = (float)ay.at(i); }
    return g;
}

void save_grid(const std::string& path, const GridMap& g, Npz extra) {
    int64_t shape[2] = {g.out_h, g.out_w};
    extra["stride"] = NpyArray::scalar_i64(g.stride);
    extra["shape"] = NpyArray::i64({2}, shape);
    extra["map_x"] = NpyArray::f32({(size_t)g.gh, (size_t)g.gw}, g.x.data());
    extra["map_y"] = NpyArray::f32({(size_t)g.gh, (size_t)g.gw}, g.y.data());
    write_npz(path, extra);
}

static void save_dense(const std::string& path, const cv::Mat& mx, const cv::Mat& my) {
    Npz z;
    cv::Mat a = mx.clone(), b = my.clone();
    z["map_x"] = NpyArray::f32({(size_t)a.rows, (size_t)a.cols}, a.ptr<float>());
    z["map_y"] = NpyArray::f32({(size_t)b.rows, (size_t)b.cols}, b.ptr<float>());
    write_npz(path, z);
}

static cv::Mat to_bgra(const cv::Mat& m) {
    cv::Mat o;
    if (m.channels() == 4) return m;
    cv::cvtColor(m, o, m.channels() == 3 ? cv::COLOR_BGR2BGRA : cv::COLOR_GRAY2BGRA);
    return o;
}

static int diff_stats(const cv::Mat& a, const cv::Mat& b, const char* name) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    double mx;
    cv::minMaxLoc(d.reshape(1), nullptr, &mx);
    cv::Mat gt1 = d.reshape(1) > 1, gt0 = d.reshape(1) > 0;
    std::printf("  %-28s max |diff| %3.0f   pixels>0: %8d   pixels>1: %8d\n", name, mx, cv::countNonZero(gt0), cv::countNonZero(gt1));
    return (int)mx;
}

// ---- M1: random + structured fixtures, every path against cv::remap
static int cmd_selftest_resample(int argc, char** argv) {
    bool gpu = Gpu::instance().available();
    std::printf("GPU: %s\n", gpu ? Gpu::instance().info().c_str() : "unavailable");
    std::mt19937 rng(1234);
    int worst = 0, worst_stages = 0, cases = 0;
    struct Case { const char* name; int sw, sh, ow, oh; };
    std::vector<Case> cs = {{"tiny 2x2 src", 2, 2, 9, 7}, {"3x3 src", 3, 3, 13, 11}, {"5x4 src", 5, 4, 17, 21},
                            {"medium", 211, 157, 123, 181}, {"large", 1203, 1609, 801, 1131}};
    const char* fields[] = {"constant field", "impulse", "checkerboard", "subpixel shift", "negative/outside", "random warp"};
    for (auto& c : cs)
        for (int f = 0; f < 6; ++f) {
            cv::Mat src(c.sh, c.sw, CV_8UC4);
            if (f == 0) src.setTo(cv::Scalar(37, 140, 222, 255));
            else if (f == 1) { src.setTo(cv::Scalar(0, 0, 0, 255)); src.at<cv::Vec4b>(c.sh / 2, c.sw / 2) = cv::Vec4b(255, 255, 255, 255); }
            else if (f == 2) { for (int y = 0; y < c.sh; ++y) for (int x = 0; x < c.sw; ++x) src.at<cv::Vec4b>(y, x) = ((x + y) & 1) ? cv::Vec4b(255, 0, 255, 255) : cv::Vec4b(0, 255, 0, 255); }
            else cv::randu(src, 0, 256);
            GridMap g = GridMap::make(c.ow, c.oh, 4);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            float sx = (float)c.sw / c.ow, sy = (float)c.sh / c.oh;
            for (int i = 0; i < g.gh; ++i)
                for (int j = 0; j < g.gw; ++j) {
                    float x = j * 4 * sx, y = i * 4 * sy;
                    if (f == 3) { x += 0.37f; y -= 0.61f; }
                    if (f == 4) { x = x * 1.6f - 0.3f * c.sw; y = y * 1.6f - 0.3f * c.sh; }
                    if (f == 5) { x += 3.f * U(rng); y += 3.f * U(rng); }
                    g.X(i, j) = x; g.Y(i, j) = y;
                }
            cv::Mat mx, my, ref;
            expand_full(g, mx, my);
            cv::remap(src, ref, mx, my, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
            std::printf("%s / %s (%dx%d -> %dx%d)\n", c.name, fields[f], c.sw, c.sh, c.ow, c.oh);
            worst = std::max(worst, diff_stats(ref, resample_cpu(src, g, 7), "cpu banded vs cv::remap"));
            worst = std::max(worst, diff_stats(ref, resample_model(src, g), "kernel model vs cv::remap"));
            if (gpu) {
                cv::Mat out;
                if (!Gpu::instance().resample(src, g, out)) { std::printf("  GPU resample FAILED\n"); worst = 999; }
                else worst = std::max(worst, diff_stats(ref, out, "gpu vs cv::remap"));
                cv::Mat gx, gy;
                Gpu::instance().expand_maps(g, gx, gy);
                cv::Mat e = cv::abs(gx - mx) + cv::abs(gy - my);
                double emax; cv::minMaxLoc(e, nullptr, &emax);
                std::printf("  %-28s max |dx|+|dy| %.3g px\n", "gpu map expansion vs cpu", emax);
                // the other bicubic stages (stage-1 / orientation page, global remap): byte-exact
                worst_stages = std::max(worst_stages, gpu_stage_fixtures(src, mx, my, rng));
            }
            ++cases;
        }
    std::printf("cases %d, worst |diff| %d (gate: <= 1)\n", cases, worst);
    if (gpu) std::printf("GPU stages (dense / offset remap): worst |diff| %d (gate: 0)\n", worst_stages);
    return worst <= 1 && worst_stages == 0 ? 0 : 1;
}

// resample IMG GRID.npz OUT.png [--gpu|--model] [--dense OUT.npz]
static int cmd_resample(int argc, char** argv) {
    cv::Mat src = to_bgra(cv::imread(argv[2], cv::IMREAD_COLOR));
    GridMap g = load_grid(argv[3]);
    cv::Mat out, mx, my;
    auto t0 = std::chrono::steady_clock::now();
    bool gpu = has(argc, argv, "--gpu");
    if (gpu) {
        if (!Gpu::instance().resample(src, g, out)) { std::fprintf(stderr, "GPU resample failed: %s\n", Gpu::instance().info().c_str()); return 2; }
    } else if (has(argc, argv, "--model")) out = resample_model(src, g);
    else out = resample_cpu(src, g);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cv::Mat bgr;
    cv::cvtColor(out, bgr, cv::COLOR_BGRA2BGR);
    cv::imwrite(argv[4], bgr);
    if (const char* d = opt(argc, argv, "--dense")) {
        if (gpu) Gpu::instance().expand_maps(g, mx, my);
        else expand_full(g, mx, my);
        save_dense(d, mx, my);
    }
    std::printf("{\"path\": \"%s\", \"ms\": %.1f}\n", gpu ? "gpu" : "cpu", ms);
    return 0;
}

static int cmd_gpuinfo(int, char**) {
    bool ok = Gpu::instance().available();
    std::printf("%s: %s, max texture %d\n", ok ? "available" : "unavailable", Gpu::instance().info().c_str(), Gpu::instance().max_texture_size());
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fs_cli gpuinfo | selftest-resample | resample IMG GRID.npz OUT.png [--gpu] [--dense D.npz] | gpu-stages OUT.json PHOTO... | gpu-bench PHOTO | cg-tol OUT.txt RTOL PHOTO... | ...\n");
        return 2;
    }
    std::string c = argv[1];
    try {
        if (c == "gpuinfo") return cmd_gpuinfo(argc, argv);
        if (c == "selftest-resample") return cmd_selftest_resample(argc, argv);
        if (c == "resample" && argc >= 5) return cmd_resample(argc, argv);
        if (c == "gpu-stages" && argc >= 4) return cmd_gpu_stages(argc, argv);
        if (c == "gpu-bench" && argc >= 3) return cmd_gpu_bench(argc, argv);
        if (c == "cg-tol" && argc >= 5) return cmd_cg_tol(argc, argv);
        return cli_ext(c, argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 3;
    }
}
