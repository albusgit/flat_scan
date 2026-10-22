// ONNX Runtime inference for the page nets (infer.h). Built with FS_WITH_ORT (core/CMakeLists.txt:
// FS_ORT_ROOT); without it every run() fails and the caller takes its no-net path.
//
// Per net, the backends are tried in InferConfig::preference order with CPU appended last. A
// backend whose execution provider cannot be created (not compiled in, not in this ORT build, no
// such hardware, model rejected) or whose run fails is dropped for that net, and the same input is
// run on the next one; only a failing CPU run makes run() return false. The CPU session is ORT's
// default CPU EP on the original fp32 model with default session options (graph optimisation
// ORT_ENABLE_ALL, as the app's Kotlin sessions and Python's InferenceSession), so its outputs are
// those of the Python / Kotlin path of the same ORT version. Accelerators load the fixed-shape
// models of tools/accel when the loader has them (`uvdoc_static.onnx`, `syntheticdoc_hr_static.onnx`,
// `shadenet_<H>x<W>.onnx`), else the original.
//
// Backends, each compiled in only with its CMake option (all OFF by default):
//   QNN_HTP  FS_ORT_QNN     libQnnHtp.so (QnnHtp.dll on Windows), fp16 precision, burst mode; the compiled
//                           graph is cached in cache_dir as an EP-context model keyed by the model's
//                           SHA-256, the SoC and the QNN / ORT versions
//   NNAPI    FS_ORT_NNAPI   Android; USE_FP16 (cfg.fp16) and CPU_DISABLED; looked up at run time
//                           (the onnxruntime-android-qnn AAR has no NNAPI EP)
//   XNNPACK  FS_ORT_XNNPACK fp32
//   CUDA, TensorRT, DirectML, OpenVINO, CoreML: FS_ORT_CUDA, FS_ORT_TENSORRT, FS_ORT_DML,
//                           FS_ORT_OPENVINO, FS_ORT_COREML (desktop / Apple builds with a matching ORT package)
// The page finder never runs on QNN or NNAPI (PLAN: 34 NNAPI partitions; kept on the CPU). The rule net
// (rulenet.onnx, any page size) has no fixed-shape model: never on QNN / NNAPI either.
#include "infer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>

#include "log.h"

#ifdef FS_WITH_ORT
#include <onnxruntime_cxx_api.h>
#if defined(__ANDROID__) || defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#endif

namespace fs {


#ifndef FS_WITH_ORT

struct Inference::Impl {
    InferConfig cfg;
};

Inference::Inference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Inference::~Inference() = default;

std::unique_ptr<Inference> Inference::create(const InferConfig& cfg) {
    auto impl = std::make_unique<Impl>();
    impl->cfg = cfg;
    return std::unique_ptr<Inference>(new Inference(std::move(impl)));
}

bool Inference::run(NetId, const Tensor&, std::vector<Tensor>& out, std::string* err) {
    out.clear();
    if (err) *err = "no inference backend in this build (FS_WITH_ORT not set)";
    return false;
}

Backend Inference::backend(NetId) const { return Backend::CPU; }
double Inference::last_ms(NetId) const { return 0.0; }

#else  // FS_WITH_ORT

namespace {

const char* backend_str(Backend b) {
    switch (b) {
        case Backend::CPU: return "CPU";
        case Backend::XNNPACK: return "XNNPACK";
        case Backend::QNN_HTP: return "QNN_HTP";
        case Backend::NNAPI: return "NNAPI";
        case Backend::CUDA: return "CUDA";
        case Backend::TensorRT: return "TensorRT";
        case Backend::DirectML: return "DirectML";
        case Backend::OpenVINO: return "OpenVINO";
        case Backend::CoreML: return "CoreML";
    }
    return "?";
}

const char* net_str(NetId n) {
    switch (n) {
        case NetId::PageFinder: return "pagefinder";
        case NetId::UVDoc: return "uvdoc";
        case NetId::SynthDocHR: return "syntheticdoc_hr";
        case NetId::ShadeNet: return "shadenet";
        case NetId::RuleNet: return "rulenet";
    }
    return "?";
}

constexpr int N_NETS = 5;

}  // namespace

namespace {

using Clock = std::chrono::steady_clock;

Ort::Env& ort_env() {
    // never destroyed: sessions held in static objects may outlive any other order of teardown
    static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "flatscan");
    return *env;
}

#ifdef FS_ORT_QNN
// SHA-256 (FIPS 180-4), for the compiled-model cache key
std::string sha256_hex(const std::vector<uint8_t>& data) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::vector<uint8_t> m(data);
    const uint64_t bits = (uint64_t)data.size() * 8;
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t)(bits >> (8 * i)));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t)m[off + 4 * i] << 24 | (uint32_t)m[off + 4 * i + 1] << 16 | (uint32_t)m[off + 4 * i + 2] << 8 | m[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
    }
    char s[65];
    for (int i = 0; i < 8; ++i) std::snprintf(s + 8 * i, 9, "%08x", h[i]);
    return s;
}

// The SoC for the compiled-model cache key (Android 12+: ro.soc.model, else the board platform).
std::string soc_name() {
#ifdef __ANDROID__
    char v[PROP_VALUE_MAX] = {0};
    if (__system_property_get("ro.soc.model", v) > 0) return v;
    if (__system_property_get("ro.board.platform", v) > 0) return v;
    return "android";
#else
    return "host";
#endif
}

// The QNN HTP backend's API versions (core and backend) from QnnInterface_getProviders, which the
// library exports; only the leading fields of QnnInterface_t (QnnInterface.h) are read.
std::string qnn_version(const std::string& dir) {
#if defined(__ANDROID__) || defined(__linux__)
    struct Ver { uint32_t major, minor, patch; };
    struct IfacePrefix { uint32_t backend_id; const char* provider_name; Ver core, backend; };
    using GetProviders = uint64_t (*)(const IfacePrefix*** list, uint32_t* n);
    void* h = dlopen((dir.empty() ? std::string("libQnnHtp.so") : dir + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return "qnn-none";
    std::string v = "qnn-unknown";
    if (auto fn = (GetProviders)dlsym(h, "QnnInterface_getProviders")) {
        const IfacePrefix** list = nullptr;
        uint32_t n = 0;
        if ((uint32_t)fn(&list, &n) == 0 && list && n > 0 && list[0]) {
            const IfacePrefix& p = *list[0];
            char b[96];
            std::snprintf(b, sizeof b, "qnn%u.%u.%u-htp%u.%u.%u", p.core.major, p.core.minor, p.core.patch, p.backend.major, p.backend.minor,
                          p.backend.patch);
            v = b;
        }
    }
    dlclose(h);
    return v;
#else
    (void)dir;
    return "qnn";
#endif
}

bool file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.insert(out.end(), buf, buf + n);
    std::fclose(f);
    return !out.empty();
}
#endif

struct Sess {
    std::unique_ptr<Ort::Session> s;
    std::string in_name;
    std::vector<std::string> out_names;
};

// One net: its sessions on the current backend (ShadeNet on an accelerator: one per fixed input
// shape, as the static models are per width; otherwise key {}), where it is in the backend order.
struct Slot {
    std::mutex mu;
    size_t pos = 0;                   // index into Impl::order of the backend in use / to try
    std::map<std::vector<int64_t>, Sess> sess;
    bool cpu_failed = false;          // the CPU session could not be created (model missing / unreadable)
    Backend ran = Backend::CPU;
    double ms = 0;
};

}  // namespace

struct Inference::Impl {
    InferConfig cfg;
    std::vector<Backend> order;       // preference without repeats, CPU last
    std::array<Slot, N_NETS> slots;

    static bool accelerator_allowed(NetId net, Backend b) {
        return !(net == NetId::PageFinder && (b == Backend::QNN_HTP || b == Backend::NNAPI));
    }

    // model bytes for net on backend b and an input of shape `shape`; file: the name used
    std::vector<uint8_t> model_bytes(NetId net, Backend b, const std::vector<int64_t>& shape, std::string& file) const {
        if (!cfg.load) return {};
        std::vector<uint8_t> bytes;
        if (b != Backend::CPU) {
            if (net == NetId::UVDoc) file = "uvdoc_static.onnx";
            else if (net == NetId::SynthDocHR) file = "syntheticdoc_hr_static.onnx";
            else if (net == NetId::ShadeNet && shape.size() == 4) file = "shadenet_" + std::to_string(shape[2]) + "x" + std::to_string(shape[3]) + ".onnx";
            else file.clear();
            if (!file.empty()) {
                try { bytes = cfg.load(file); } catch (...) { bytes.clear(); }
                if (!bytes.empty()) return bytes;
            }
            // QNN / NNAPI cannot run the dynamic-shape ShadeNet or rule net as one graph: no static model, no accelerator
            if ((net == NetId::ShadeNet || net == NetId::RuleNet) && (b == Backend::QNN_HTP || b == Backend::NNAPI)) return {};
        }
        file = std::string(net_str(net)) + ".onnx";
        try { bytes = cfg.load(file); } catch (...) { bytes.clear(); }
        return bytes;
    }

    // Adds backend b's execution provider to `so`; false (with why) when this build or device lacks it.
    bool add_ep(Ort::SessionOptions& so, Backend b, std::string& why) const {
        switch (b) {
            case Backend::CPU:
                return true;
            case Backend::XNNPACK:
#ifdef FS_ORT_XNNPACK
                // XNNPACK's own thread pool; ORT's intra-op pool then only runs what XNNPACK does not take
                so.SetIntraOpNumThreads(1);
                so.AddConfigEntry("session.intra_op.allow_spinning", "0");
                so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(cfg.threads > 0 ? cfg.threads : 4)}});
                return true;
#else
                break;
#endif
            case Backend::QNN_HTP:
#ifdef FS_ORT_QNN
                // the whole graph on the NPU or no session: with ORT's CPU fallback a QNN backend that fails to
                // set up (no DSP, wrong skeleton) leaves every node on the CPU EP and the net would be reported
                // as "qnn" (seen on the emulator); this way it moves to the next backend, reported as such
                so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
#ifdef _WIN32
                so.AppendExecutionProvider("QNN", {{"backend_path", "QnnHtp.dll"}, {"htp_performance_mode", "burst"},
                                                   {"enable_htp_fp16_precision", cfg.fp16 ? "1" : "0"}});
#else
                so.AppendExecutionProvider("QNN", {{"backend_path", cfg.qnn_dir.empty() ? std::string("libQnnHtp.so") : cfg.qnn_dir + "/libQnnHtp.so"},
                                                   {"htp_performance_mode", "burst"},
                                                   {"enable_htp_fp16_precision", cfg.fp16 ? "1" : "0"}});
#endif
                return true;
#else
                break;
#endif
            case Backend::NNAPI:
#if defined(FS_ORT_NNAPI) && defined(__ANDROID__)
            {
                // nnapi_provider_factory.h: OrtSessionOptionsAppendExecutionProvider_Nnapi(options, flags);
                // NNAPI_FLAG_USE_FP16 = 0x001, NNAPI_FLAG_CPU_DISABLED = 0x004
                using AppendNnapi = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
                void* h = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_NOLOAD);
                auto fn = h ? (AppendNnapi)dlsym(h, "OrtSessionOptionsAppendExecutionProvider_Nnapi") : nullptr;
                if (!fn) {
                    why = "this ONNX Runtime build has no NNAPI execution provider";
                    return false;
                }
                OrtStatus* st = fn(so, (cfg.fp16 ? 0x001u : 0u) | 0x004u);
                if (st) {
                    why = Ort::GetApi().GetErrorMessage(st);
                    Ort::GetApi().ReleaseStatus(st);
                    return false;
                }
                return true;
            }
#else
                break;
#endif
            case Backend::CUDA:
#ifdef FS_ORT_CUDA
            {
                const OrtApi& api = Ort::GetApi();
                OrtCUDAProviderOptionsV2* o = nullptr;
                Ort::ThrowOnError(api.CreateCUDAProviderOptions(&o));
                std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)> hold(o, api.ReleaseCUDAProviderOptions);
                // fp32 convolutions and matmuls: ORT's default TF32 (Ampere and later) moves SyntheticDoc-HR's
                // grid by p99 0.50 px of the page (RTX 5060, 5 photos), over Gate A's 0.25 px
                const char* k[] = {"use_tf32"};
                const char* v[] = {"0"};
                Ort::ThrowOnError(api.UpdateCUDAProviderOptions(o, k, v, 1));
                so.AppendExecutionProvider_CUDA_V2(*o);
                return true;
            }
#else
                break;
#endif
            case Backend::TensorRT:
#ifdef FS_ORT_TENSORRT
            {
                const OrtApi& api = Ort::GetApi();
                OrtTensorRTProviderOptionsV2* o = nullptr;
                Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&o));
                std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2*)> hold(o, api.ReleaseTensorRTProviderOptions);
                std::vector<const char*> k = {"trt_fp16_enable"}, v = {cfg.fp16 ? "1" : "0"};
                if (!cfg.cache_dir.empty()) {
                    k.push_back("trt_engine_cache_enable"); v.push_back("1");
                    k.push_back("trt_engine_cache_path"); v.push_back(cfg.cache_dir.c_str());
                }
                Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(o, k.data(), v.data(), k.size()));
                so.AppendExecutionProvider_TensorRT_V2(*o);
                return true;
            }
#else
                break;
#endif
            case Backend::DirectML:
#ifdef FS_ORT_DML
                // the DirectML EP needs sequential execution without memory patterns
                so.DisableMemPattern();
                so.SetExecutionMode(ORT_SEQUENTIAL);
                so.AppendExecutionProvider("DML", {{"device_id", "0"}});
                return true;
#else
                break;
#endif
            case Backend::OpenVINO:
#ifdef FS_ORT_OPENVINO
                so.AppendExecutionProvider("OpenVINO", {{"device_type", "AUTO:GPU,CPU"}});
                return true;
#else
                break;
#endif
            case Backend::CoreML:
#ifdef FS_ORT_COREML
                so.AppendExecutionProvider("CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"},
                                                      {"AllowLowPrecisionAccumulationOnGPU", cfg.fp16 ? "1" : "0"}});
                return true;
#else
                break;
#endif
        }
        why = std::string(backend_str(b)) + " is not compiled into this build";
        return false;
    }

    // A session for net on backend b, or false with why.
    bool make_session(NetId net, Backend b, const std::vector<int64_t>& shape, Sess& out, std::string& why) const {
        std::string file;
        std::vector<uint8_t> bytes = model_bytes(net, b, shape, file);
        if (bytes.empty()) {
            why = "model not available (" + (file.empty() ? std::string(net_str(net)) : file) + ")";
            return false;
        }
        auto t0 = Clock::now();
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (cfg.threads > 0) so.SetIntraOpNumThreads(cfg.threads);
            if (!add_ep(so, b, why)) return false;
            bool from_cache = false;
#ifdef FS_ORT_QNN
            // QNN: the graph compiled for this SoC is kept as an EP-context model (QNN context binary
            // embedded); a cached one that fails to load is rebuilt
            if (b == Backend::QNN_HTP && !cfg.cache_dir.empty()) {
                std::string key = sha256_hex(bytes).substr(0, 16) + "_" + soc_name() + "_" + qnn_version(cfg.qnn_dir) + "_ort" + Ort::GetVersionString() +
                                  (cfg.fp16 ? "_fp16" : "_fp32");
                for (char& c : key)
                    if (!(std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_')) c = '_';
                std::string ctx = cfg.cache_dir + "/" + net_str(net) + (shape.empty() ? "" : "_" + std::to_string(shape.back())) + "_" + key + "_ctx.onnx";
                std::vector<uint8_t> cached;
                if (file_bytes(ctx, cached)) {
                    try {
                        out.s = std::make_unique<Ort::Session>(ort_env(), cached.data(), cached.size(), so);
                        from_cache = true;
                    } catch (const Ort::Exception& e) {
                        LOGI("infer: cached QNN context %s unusable (%s); rebuilding", ctx.c_str(), e.what());
                        std::remove(ctx.c_str());
                    }
                }
                if (!from_cache) {
                    so.AddConfigEntry("ep.context_enable", "1");
                    so.AddConfigEntry("ep.context_embed_mode", "1");
                    so.AddConfigEntry("ep.context_file_path", ctx.c_str());
                }
            }
#endif
            if (!from_cache) out.s = std::make_unique<Ort::Session>(ort_env(), bytes.data(), bytes.size(), so);
            Ort::AllocatorWithDefaultOptions alloc;
            if (out.s->GetInputCount() != 1 || out.s->GetOutputCount() < 1) {
                why = "unexpected model signature";
                out = Sess();
                return false;
            }
            out.in_name = out.s->GetInputNameAllocated(0, alloc).get();
            out.out_names.clear();
            for (size_t i = 0; i < out.s->GetOutputCount(); ++i) out.out_names.push_back(out.s->GetOutputNameAllocated(i, alloc).get());
            LOGI("infer: %s on %s (%s%s) ready in %.0f ms", net_str(net), backend_str(b), file.c_str(), from_cache ? ", cached context" : "",
                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            return true;
        } catch (const Ort::Exception& e) {
            why = e.what();
        } catch (const std::exception& e) {
            why = e.what();
        }
        out = Sess();
        return false;
    }
};

Inference::Inference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Inference::~Inference() = default;

std::unique_ptr<Inference> Inference::create(const InferConfig& cfg) {
    auto impl = std::make_unique<Impl>();
    impl->cfg = cfg;
    for (Backend b : cfg.preference)
        if (b != Backend::CPU && std::find(impl->order.begin(), impl->order.end(), b) == impl->order.end()) impl->order.push_back(b);
    impl->order.push_back(Backend::CPU);
    return std::unique_ptr<Inference>(new Inference(std::move(impl)));
}

bool Inference::run(NetId net, const Tensor& in, std::vector<Tensor>& out, std::string* err) {
    out.clear();
    Impl& I = *impl_;
    Slot& S = I.slots[(int)net];
    std::lock_guard<std::mutex> lk(S.mu);
    size_t n = 1;
    for (int64_t d : in.shape) n *= d < 0 ? 0 : (size_t)d;
    if (in.shape.empty() || n != in.data.size()) {
        if (err) *err = "input shape does not match its data";
        return false;
    }
    std::string why;
    while (S.pos < I.order.size()) {
        const Backend b = I.order[S.pos];
        const bool cpu = b == Backend::CPU;
        if (!cpu && !Impl::accelerator_allowed(net, b)) { ++S.pos; continue; }
        if (cpu && S.cpu_failed) break;
        // ShadeNet's fixed-shape models on accelerators: one session per input shape
        std::vector<int64_t> key = (!cpu && net == NetId::ShadeNet) ? in.shape : std::vector<int64_t>();
        auto it = S.sess.find(key);
        if (it == S.sess.end()) {
            Sess s;
            if (!I.make_session(net, b, key, s, why)) {
                LOGI("infer: %s: %s unavailable: %s", net_str(net), backend_str(b), why.c_str());
                if (cpu) { S.cpu_failed = true; break; }
                S.sess.clear();
                ++S.pos;
                continue;
            }
            it = S.sess.emplace(key, std::move(s)).first;
        }
        Sess& s = it->second;
        try {
            auto t0 = Clock::now();
            Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            // ORT does not write to inputs
            Ort::Value x = Ort::Value::CreateTensor<float>(mi, const_cast<float*>(in.data.data()), in.data.size(), in.shape.data(), in.shape.size());
            const char* in_name = s.in_name.c_str();
            std::vector<const char*> names;
            for (const auto& o : s.out_names) names.push_back(o.c_str());
            std::vector<Ort::Value> y = s.s->Run(Ort::RunOptions{nullptr}, &in_name, &x, 1, names.data(), names.size());
            out.resize(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                auto info = y[i].GetTensorTypeAndShapeInfo();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("output is not float32");
                out[i].shape = info.GetShape();
                const float* p = y[i].GetTensorData<float>();
                out[i].data.assign(p, p + info.GetElementCount());
            }
            S.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            S.ran = b;
            return true;
        } catch (const std::exception& e) {
            why = e.what();
        }
        out.clear();
        LOGI("infer: %s on %s failed: %s", net_str(net), backend_str(b), why.c_str());
        if (cpu) break;   // the CPU session stays; this input failed
        S.sess.clear();
        ++S.pos;
    }
    if (err) *err = why.empty() ? "no backend could run the net" : why;
    return false;
}

Backend Inference::backend(NetId net) const {
    Slot& S = impl_->slots[(int)net];
    std::lock_guard<std::mutex> lk(S.mu);
    return S.ran;
}

double Inference::last_ms(NetId net) const {
    Slot& S = impl_->slots[(int)net];
    std::lock_guard<std::mutex> lk(S.mu);
    return S.ms;
}

#endif  // FS_WITH_ORT

}  // namespace fs
