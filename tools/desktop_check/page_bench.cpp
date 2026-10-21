// Per-page timing of the app's steps with warm sessions (docs/accel/PLAN.md Gate F), one process:
//   page_bench [--backend B] [--reps N] [--gpu] [--dense] [--save DIR] OUT.json PHOTO...
// Per photo, as pipeline_cli with FS_NETS=1, "net" and ShadeNet after the flattening nets (FS_SHADE_AFTER;
// gate_b.py --shade): page finder -> detect_corners -> run_flat_nets -> run_photo (ShadeNet inside).
// One warm-up pass (session creation, GPU context and shader compile, CUDA JIT / autotune: "first_ms"),
// then N timed passes (default 5); the medians go to OUT.json (one JSON object per line) and a table to
// stdout. --gpu: PipelineOptions::use_gpu on (the GPU stage policy's defaults; --dense adds the dense
// stage); off by default as pipeline_cli. Every pass must give the same page (else "stable": false).
// --save DIR: the last pass's page as DIR/<photo name>.jpg (quality 85, as pipeline_cli).
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gpu.h"
#include "nets.h"
#include "pipeline.h"
#include "resample.h"

using namespace fs;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double med(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static std::string stem(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    std::string b = s == std::string::npos ? p : p.substr(s + 1);
    size_t d = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

int main(int argc, char** argv) {
    std::string backend = "cpu", save;
    int reps = 5;
    bool gpu = false, dense = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--save" && i + 1 < argc) save = argv[++i];
        else if (a == "--gpu") gpu = true;
        else if (a == "--dense") dense = true;
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::fprintf(stderr, "usage: page_bench [--backend B] [--reps N] [--gpu] [--dense] [--save DIR] OUT.json PHOTO...\n");
        return 2;
    }
    const std::string models = std::getenv("FS_MODELS") ? std::getenv("FS_MODELS") : std::string(FS_REPO_DIR) + "/app/src/main/assets/models";
    InferConfig cfg;
    if (!parse_backends(backend, cfg.preference)) { std::fprintf(stderr, "unknown backend in \"%s\"\n", backend.c_str()); return 2; }
    cfg.load = [models](const std::string& file) { return read_file(models + "/" + file); };
    if (const char* t = std::getenv("FS_ORT_THREADS")) cfg.threads = std::atoi(t);
    std::unique_ptr<Inference> inf = Inference::create(cfg);
    if (gpu) {
        GpuStagePolicy gp = gpu_stage_policy();
        if (dense) gp.dense = true;
        set_gpu_stage_policy(gp);
    }
    std::ofstream js(pos[0]);
    std::printf("%-34s %8s %8s %8s %8s %8s %8s %8s %9s %9s  %s\n", "photo", "pf", "detect", "uv_p1", "uvdoc", "sd_hr", "shade", "run_ph",
                "page", "first", "backends");
    std::vector<double> all_page;
    for (size_t pi = 1; pi < pos.size(); ++pi) {
        cv::Mat img = cv::imread(pos[pi], cv::IMREAD_COLOR);
        if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", pos[pi].c_str()); return 1; }
        std::map<std::string, std::vector<double>> t;
        std::map<std::string, std::string> used;
        double first_ms = 0;
        bool stable = true, ok = true;
        cv::Mat page0;
        for (int rep = 0; rep <= reps; ++rep) {
            auto t0 = Clock::now();
            std::vector<float> lg;
            const bool have_lg = page_finder_logits(img, *inf, lg);
            const double pf = inf->last_ms(NetId::PageFinder);
            used["pf"] = backend_name(inf->backend(NetId::PageFinder));
            const float* logits = have_lg ? lg.data() : nullptr;
            PipelineOptions o;
            o.use_gpu = gpu;
            o.seg_logits = logits;
            o.zoom_pass = false;
            double shade_ms = 0;
            o.shade_net = shade_net_fn(*inf, &shade_ms);
            o.shade_after_uv = true;
            auto td = Clock::now();
            PageCorners pc = detect_corners(img, o.max_analysis_side, logits, 0);
            const double det = ms_since(td);
            NetsResult nets;
            std::map<std::string, double> net_ms;
            if (pc.ok) {
                NetsRequest rq;
                rq.corners = pc.pts;
                rq.seg_logits = logits;
                nets = run_flat_nets(img, rq, *inf);
                nets.apply(o);
                for (const NetRun& n : nets.runs) {
                    net_ms[n.name] = n.ms;
                    used[n.name] = backend_name(n.backend);
                }
            }
            auto tr = Clock::now();
            PipelineResult r = run_photo(img, o);
            const double run = ms_since(tr), page = ms_since(t0);
            used["shade"] = backend_name(inf->backend(NetId::ShadeNet));
            ok = ok && r.ok;
            if (rep == 0) {
                first_ms = page;
                page0 = r.page.clone();
                continue;
            }
            if (r.page.size() != page0.size() || r.page.type() != page0.type() || cv::norm(r.page, page0, cv::NORM_INF) != 0) stable = false;
            t["pf"].push_back(pf);
            t["detect"].push_back(det);
            for (const char* n : {"uv_pass1", "uvdoc", "synthdoc_hr"}) t[n].push_back(net_ms.count(n) ? net_ms[n] : 0);
            t["shade"].push_back(shade_ms);
            t["run_photo"].push_back(run);
            t["page"].push_back(page);
            if (rep == reps && !save.empty() && r.ok) {
                cv::Mat bgr;
                cv::cvtColor(r.page, bgr, cv::COLOR_BGRA2BGR);
                cv::imwrite(save + "/" + stem(pos[pi]) + ".jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 85});
            }
        }
        all_page.push_back(med(t["page"]));
        std::string be;
        for (const char* n : {"pf", "uv_pass1", "uvdoc", "synthdoc_hr", "shade"})
            if (used.count(n)) be += (be.empty() ? "" : ",") + used[n];
        std::printf("%-34s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %9.1f %9.1f  %s%s\n", stem(pos[pi]).c_str(), med(t["pf"]), med(t["detect"]),
                    med(t["uv_pass1"]), med(t["uvdoc"]), med(t["synthdoc_hr"]), med(t["shade"]), med(t["run_photo"]),
                    med(t["page"]), first_ms, be.c_str(), stable ? "" : "  UNSTABLE");
        js << "{\"photo\": \"" << stem(pos[pi]) << "\", \"backend\": \"" << backend << "\", \"gpu\": " << (gpu ? "true" : "false")
           << ", \"dense\": " << (dense ? "true" : "false") << ", \"reps\": " << reps << ", \"ok\": " << (ok ? "true" : "false")
           << ", \"stable\": " << (stable ? "true" : "false") << ", \"first_ms\": " << first_ms;
        for (const auto& kv : t) js << ", \"" << kv.first << "_ms\": " << med(kv.second);
        js << ", \"used\": {";
        bool sep = false;
        for (const auto& kv : used) js << (sep ? ", " : "") << "\"" << kv.first << "\": \"" << kv.second << "\"", sep = true;
        js << "}}\n";
    }
    std::printf("median page over %zu photos: %.1f ms  (backend %s, GPU stages %s%s)\n", all_page.size(), med(all_page), backend.c_str(),
                gpu ? "on" : "off", gpu && dense ? " + dense" : "");
    if (gpu) std::printf("GPU: %s\n", Gpu::instance().info().c_str());
    return 0;
}
