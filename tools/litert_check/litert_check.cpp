// One net on the LiteRT GPU backend (core/src/infer.cpp, Backend::LiteRtGpu) against ONNX Runtime's CPU
// on the same input, without the app (Android command line; README.md beside this file):
//   litert_check LIB_DIR MODEL_DIR NET [--runs N] [--input FILE] [--cache DIR] [--tol T]
// LIB_DIR: libtensorflowlite_jni.so + libtensorflowlite_gpu_jni.so (InferConfig::litert_dir).
// MODEL_DIR: the GPU model and the CPU's ONNX, as the app's loader finds them: uvdoc_static.tflite +
// uvdoc.onnx, syntheticdoc_hr_static.tflite + syntheticdoc_hr.onnx, shadenet_512x352.tflite + shadenet.onnx.
// NET: uvdoc (input 1x3x712x488), synthdoc_hr (1x3x1440x1024), shadenet (1x3x512x352).
// FILE: the NCHW float32 input, raw little-endian (default: a fixed pseudo-random image in [0, 1]).
// Prints the backend each side ran on, the first (session set-up included) and the median run time, and
// |GPU - CPU| over the first output: max, p99, mean, with the CPU output's range. Exit 0: the GPU ran and
// (with --tol) max |GPU - CPU| <= T; 1: the net fell back to the CPU or missed T; 2: usage / input.
// FS_LITERT_TEST_XNNPACK=1 (this build of infer.cpp only): LiteRT's XNNPACK CPU delegate stands in for
// the GPU delegate, to run the backend's own code where no GPU can (README.md).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "infer.h"

using namespace fs;

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static const char* name(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::LiteRtGpu: return "gpu";
        default: return "other";
    }
}

// runs net `runs` times; false when a run fails. first / median: ms of the first and the median run
// (Inference::last_ms: input, run, outputs)
static bool time_runs(Inference& inf, NetId net, const Tensor& x, int runs, std::vector<Tensor>& y, double& first, double& median,
                      std::string& err) {
    std::vector<double> ms;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (!inf.run(net, x, y, &err)) return false;
        // the first run's wall time includes the session (GPU: delegate compile or kernel cache load)
        if (i == 0) first = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        else ms.push_back(inf.last_ms(net));
    }
    if (ms.empty()) ms.push_back(inf.last_ms(net));
    std::sort(ms.begin(), ms.end());
    median = ms[ms.size() / 2];
    return true;
}

static int check(int argc, char** argv) {
    std::vector<std::string> pos;
    int runs = 5;
    std::string input, cache;
    double tol = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--input" && i + 1 < argc) input = argv[++i];
        else if (a == "--cache" && i + 1 < argc) cache = argv[++i];
        else if (a == "--tol" && i + 1 < argc) tol = std::atof(argv[++i]);
        else pos.push_back(a);
    }
    if (pos.size() != 3) {
        std::fprintf(stderr, "usage: litert_check LIB_DIR MODEL_DIR uvdoc|synthdoc_hr|shadenet [--runs N] [--input FILE] [--cache DIR] [--tol T]\n");
        return 2;
    }
    NetId net;
    Tensor x;
    if (pos[2] == "uvdoc") { net = NetId::UVDoc; x.shape = {1, 3, 712, 488}; }
    else if (pos[2] == "synthdoc_hr") { net = NetId::SynthDocHR; x.shape = {1, 3, 1440, 1024}; }
    else if (pos[2] == "shadenet") { net = NetId::ShadeNet; x.shape = {1, 3, 512, 352}; }
    else { std::fprintf(stderr, "unknown net %s\n", pos[2].c_str()); return 2; }
    const size_t n = (size_t)(x.shape[1] * x.shape[2] * x.shape[3]);
    if (!input.empty()) {
        std::vector<uint8_t> b = read_file(input);
        if (b.size() != n * sizeof(float)) { std::fprintf(stderr, "%s: %zu bytes, want %zu\n", input.c_str(), b.size(), n * sizeof(float)); return 2; }
        x.data.resize(n);
        std::memcpy(x.data.data(), b.data(), b.size());
    } else {
        // a smooth image with noise, so convolutions see structure; fixed LCG, the same on every build
        x.data.resize(n);
        uint32_t s = 12345;
        const int64_t h = x.shape[2], w = x.shape[3];
        for (size_t i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            const int64_t c = (int64_t)(i / (h * w)), yx = (int64_t)(i % (h * w)), yy = yx / w, xx = yx % w;
            const double v = 0.5 + 0.3 * std::sin(0.02 * yy + 1.3 * c) * std::cos(0.03 * xx) + 0.2 * ((s >> 8) / 16777216.0 - 0.5);
            x.data[i] = (float)std::min(1.0, std::max(0.0, v));
        }
    }
    const std::string models = pos[1];
    InferConfig gc;
    gc.load = [models](const std::string& file) { return read_file(models + "/" + file); };
    gc.preference = {Backend::LiteRtGpu};
    gc.litert_dir = pos[0];
    gc.cache_dir = cache;
    InferConfig cc;
    cc.load = gc.load;
    std::unique_ptr<Inference> gpu = Inference::create(gc), cpu = Inference::create(cc);

    std::vector<Tensor> yg, yc;
    std::string err;
    double g_first = 0, g_med = 0, c_first = 0, c_med = 0;
    if (!time_runs(*gpu, net, x, runs, yg, g_first, g_med, err)) { std::printf("GPU side FAILED: %s\n", err.c_str()); return 1; }
    const Backend gb = gpu->backend(net);
    if (!time_runs(*cpu, net, x, runs, yc, c_first, c_med, err)) { std::printf("CPU FAILED: %s\n", err.c_str()); return 1; }
    std::printf("%s: gpu side ran on %s%s: first %.1f ms, median %.2f ms of %d\n", pos[2].c_str(), name(gb),
                std::getenv("FS_LITERT_TEST_XNNPACK") ? " (XNNPACK standing in for the GPU delegate)" : "", g_first, g_med, runs);
    std::printf("%s: cpu ran on %s: first %.1f ms, median %.2f ms of %d\n", pos[2].c_str(), name(cpu->backend(net)), c_first, c_med, runs);
    if (yg.empty() || yc.empty() || yg[0].shape != yc[0].shape || yg[0].data.size() != yc[0].data.size()) {
        std::printf("output shapes differ\n");
        return 1;
    }
    std::string sh;
    for (int64_t d : yg[0].shape) sh += (sh.empty() ? "" : "x") + std::to_string(d);
    std::vector<double> d(yg[0].data.size());
    double lo = 1e300, hi = -1e300, sum = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = std::fabs((double)yg[0].data[i] - (double)yc[0].data[i]);
        sum += d[i];
        lo = std::min(lo, (double)yc[0].data[i]);
        hi = std::max(hi, (double)yc[0].data[i]);
    }
    std::vector<double> s = d;
    std::sort(s.begin(), s.end());
    const double mx = s.back(), p99 = s[std::min(s.size() - 1, (size_t)(0.99 * (s.size() - 1) + 0.5))];
    std::printf("%s: output %s, |gpu - cpu| max %.3g p99 %.3g mean %.3g (cpu output range %.4g .. %.4g)\n", pos[2].c_str(), sh.c_str(), mx, p99,
                sum / d.size(), lo, hi);
    if (gb != Backend::LiteRtGpu) {
        std::printf("the net did not run on the LiteRT GPU (logcat -s flatscan has why)\n");
        return 1;
    }
    if (tol >= 0 && !(mx <= tol)) {
        std::printf("max |gpu - cpu| %.3g over --tol %.3g\n", mx, tol);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    const int rc = check(argc, argv);
    // without exit()'s static destructors: ONNX Runtime 1.22.0's abort there on Android ("pthread_mutex_lock
    // called on a destroyed mutex", inside libonnxruntime.so's __cxa_finalize, also with the CPU alone)
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(rc);
}
