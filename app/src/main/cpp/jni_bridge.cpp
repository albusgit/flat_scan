// JNI bridge: Android Bitmaps (RGBA_8888) <-> the C++ core (BGR / BGRA cv::Mat).
// Implements app/src/main/java/com/albus/flatscan/NativeLib.kt exactly.
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cleanup.h"
#include "gpu.h"
#include "infer.h"
#include "log.h"
#include "nets.h"
#include "npz.h"
#include "overlay.h"
#include "pagefind.h"
#include "pipeline.h"
#include "uvdoc.h"

#ifdef FS_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

using namespace fs;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// One processed page. Everything a re-render needs is computed once in process(): the page,
// its paper alpha, the before page, the deformation grid and the filter analysis (PageParams).
// The stage-1 page and the warp-map debug data are freed right after processing.
struct Session {
    PipelineResult result;    // page, paper, before, disp, maps, timings (stage1 / warp freed)
    PageParams pp;
    float legend_px = 1.f;
    std::mutex mu;            // guards the downscaled-source cache
    std::vector<std::pair<cv::Size, cv::Mat>> small;   // paper-filled page, INTER_AREA, by size
    cv::Mat page_orig, paper_orig;   // the page before the Erase tool (kept from its first use)
    bool use_gpu = true;
    std::string accel;        // the nets' timings and backends: the members of info()'s "accel" object

    // paper-filled page at `sz` (the full page itself when sz is the page size)
    cv::Mat filled_at(cv::Size sz) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& e : small)
            if (e.first == sz) return e.second;
        cv::Mat full = apply_paper(result.page, result.paper, pp), out;
        if (sz == full.size()) out = full;
        else cv::resize(full, out, sz, 0, 0, cv::INTER_AREA);
        if (small.size() >= 3) small.erase(small.begin());
        small.emplace_back(sz, out);
        return out;
    }
};

Session* session(jlong h) { return reinterpret_cast<Session*>(h); }

// What a session needs to come back without re-running the geometry (DocumentStore keeps it
// next to the page): the page (JPEG q95; its alpha is always opaque), the paper alpha (PNG),
// the before page (JPEG q90), the deformation grid and the fitted lines (raw float32) and the
// wrinkle legend scale. The Mats share the session's pixels (they are never written after
// process()), so a snapshot is cheap to take and can be written on another thread.
struct Snapshot {
    cv::Mat page, paper, before;
    GridMap disp;
    std::vector<std::array<float, 5>> lines;
    float legend_px = 1.f;
};

bool write_floats(const std::string& path, const std::vector<int32_t>& head, const std::vector<const std::vector<float>*>& parts) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(head.data(), sizeof(int32_t), head.size(), f) == head.size();
    for (auto* p : parts) ok = ok && (p->empty() || std::fwrite(p->data(), sizeof(float), p->size(), f) == p->size());
    return std::fclose(f) == 0 && ok;
}

bool read_floats(const std::string& path, std::vector<int32_t>& head, size_t nhead, std::vector<float>& data) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    head.assign(nhead, 0);
    bool ok = std::fread(head.data(), sizeof(int32_t), nhead, f) == nhead;
    data.clear();
    float buf[4096];
    size_t n;
    while (ok && (n = std::fread(buf, sizeof(float), 4096, f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    return ok;
}

bool write_snapshot(const Snapshot& s, const std::string& dir) {
    cv::Mat bgr;
    if (s.page.channels() == 4) cv::cvtColor(s.page, bgr, cv::COLOR_BGRA2BGR);
    else bgr = s.page;
    if (!cv::imwrite(dir + "/page.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 95})) return false;
    if (!s.paper.empty() && !cv::imwrite(dir + "/paper.png", s.paper, {cv::IMWRITE_PNG_COMPRESSION, 1})) return false;
    if (!s.before.empty()) {
        if (s.before.channels() == 4) cv::cvtColor(s.before, bgr, cv::COLOR_BGRA2BGR);
        else bgr = s.before;
        if (!cv::imwrite(dir + "/before.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 90})) return false;
    }
    const GridMap& g = s.disp;
    if (!write_floats(dir + "/disp.bin", {g.out_w, g.out_h, g.gw, g.gh, g.stride}, {&g.x, &g.y})) return false;
    std::vector<float> lf;
    for (const auto& l : s.lines) lf.insert(lf.end(), l.begin(), l.end());
    lf.push_back(s.legend_px);
    return write_floats(dir + "/lines.bin", {(int32_t)s.lines.size()}, {&lf});
}

cv::Mat read_bgra(const std::string& path) {
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR), o;
    if (m.empty()) return m;
    cv::cvtColor(m, o, cv::COLOR_BGR2BGRA);
    return o;
}

std::string jstr(JNIEnv* env, jstring js) {
    const char* c = env->GetStringUTFChars(js, nullptr);
    std::string s = c;
    env->ReleaseStringUTFChars(js, c);
    return s;
}

bool lock(JNIEnv* env, jobject bmp, cv::Mat& m, bool* premul = nullptr) {
    if (!bmp) return false;
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return false;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return false;
    void* px = nullptr;
    if (AndroidBitmap_lockPixels(env, bmp, &px) != ANDROID_BITMAP_RESULT_SUCCESS) return false;
    m = cv::Mat((int)info.height, (int)info.width, CV_8UC4, px, info.stride);
    if (premul) *premul = (info.flags & ANDROID_BITMAP_FLAGS_ALPHA_MASK) == ANDROID_BITMAP_FLAGS_ALPHA_PREMUL;
    return true;
}

cv::Size bitmap_size(JNIEnv* env, jobject bmp) {
    AndroidBitmapInfo info;
    if (!bmp || AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return cv::Size();
    return cv::Size((int)info.width, (int)info.height);
}

cv::Mat bitmap_to_bgr(JNIEnv* env, jobject bmp) {
    cv::Mat rgba, bgr;
    if (!lock(env, bmp, rgba)) return bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bmp);
    return bgr;
}

// BGRA (opaque or straight alpha) -> RGBA bitmap; resized if the bitmap has another size.
bool bgra_to_bitmap(JNIEnv* env, const cv::Mat& bgra, jobject bmp, bool straight_alpha = false) {
    cv::Mat rgba;
    bool premul = true;
    if (bgra.empty() || !lock(env, bmp, rgba, &premul)) return false;
    cv::Mat src = bgra;
    if (src.size() != rgba.size())
        cv::resize(bgra, src, rgba.size(), 0, 0, rgba.cols < bgra.cols ? cv::INTER_AREA : cv::INTER_LINEAR);
    if (straight_alpha && premul) {
        cv::Mat pm(src.size(), CV_8UC4);
        for (int y = 0; y < src.rows; ++y) {
            const uint8_t* s = src.ptr<uint8_t>(y);
            uint8_t* d = pm.ptr<uint8_t>(y);
            for (int x = 0; x < src.cols; ++x, s += 4, d += 4) {
                int a = s[3];
                d[0] = (uint8_t)((s[0] * a + 127) / 255);
                d[1] = (uint8_t)((s[1] * a + 127) / 255);
                d[2] = (uint8_t)((s[2] * a + 127) / 255);
                d[3] = (uint8_t)a;
            }
        }
        src = pm;
    }
    cv::cvtColor(src, rgba, cv::COLOR_BGRA2RGBA);   // writes into the locked pixels (same size / type)
    AndroidBitmap_unlockPixels(env, bmp);
    return true;
}

std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if ((unsigned char)c < 0x20) { o += ' '; continue; }
        o += c;
    }
    return o;
}

// RenderParams.toArray(): [filter, strength, solid, bgCleanup, black, white, gamma]
RenderParams parse_params(JNIEnv* env, jfloatArray a) {
    RenderParams rp;
    if (!a) return rp;
    jsize n = env->GetArrayLength(a);
    float v[7] = {(float)Filter::AutoColor, 0.7f, 0.f, 0.f, 0.f, 1.f, 1.f};
    env->GetFloatArrayRegion(a, 0, std::min<jsize>(n, 7), v);
    int f = (int)std::lround(v[0]);
    rp.filter = (Filter)std::min(5, std::max(0, f));
    rp.strength = std::isfinite(v[1]) ? std::min(1.f, std::max(0.f, v[1])) : 0.7f;
    rp.solid = v[2] > 0.5f;
    rp.adj.bg_cleanup = v[3];
    rp.adj.black = v[4];
    rp.adj.white = v[5];
    rp.adj.gamma = v[6];
    rp.adj = rp.adj.clamped();
    return rp;
}

std::vector<Rect01> parse_protect(JNIEnv* env, jfloatArray a) {
    std::vector<Rect01> r;
    if (!a) return r;
    jsize n = env->GetArrayLength(a) / 4;
    std::vector<float> v((size_t)n * 4);
    if (n > 0) env->GetFloatArrayRegion(a, 0, n * 4, v.data());
    for (jsize i = 0; i < n; ++i) {
        float x0 = v[4 * i], y0 = v[4 * i + 1], x1 = v[4 * i + 2], y1 = v[4 * i + 3];
        auto c = [](float q) { return std::min(1.f, std::max(0.f, q)); };
        Rect01 q{c(std::min(x0, x1)), c(std::min(y0, y1)), c(std::max(x0, x1)), c(std::max(y0, y1))};
        if (q.x1 > q.x0 && q.y1 > q.y0) r.push_back(q);
    }
    return r;
}

// page-finder logits (2 x 256 x 256) from a Java float[]; empty when absent or the wrong size
std::vector<float> read_logits(JNIEnv* env, jfloatArray a) {
    std::vector<float> v;
    if (!a || env->GetArrayLength(a) != SEG_OUTPUT_LEN) return v;
    v.resize(SEG_OUTPUT_LEN);
    env->GetFloatArrayRegion(a, 0, SEG_OUTPUT_LEN, v.data());
    return v;
}

// flattening-net crop from NativeLib.uvCrop's IntArray [x, y, w, h, turned, model]
bool read_uv_crop(JNIEnv* env, jintArray a, UvCrop& c) {
    if (!a || env->GetArrayLength(a) != 6) return false;
    jint v[6];
    env->GetIntArrayRegion(a, 0, 6, v);
    if (v[2] <= 0 || v[3] <= 0) return false;
    c.box = cv::Rect(v[0], v[1], v[2], v[3]);
    c.turned = v[4] != 0;
    c.model = v[5] >= 0 && v[5] <= 2 ? v[5] : 0;   // 0 UVDoc, 1 SyntheticDoc, 2 SyntheticDoc high-res
    return true;
}

// a Java float[] of exactly n values, or empty
std::vector<float> read_floats_exact(JNIEnv* env, jfloatArray a, size_t n) {
    std::vector<float> v;
    if (!a || (size_t)env->GetArrayLength(a) != n) return v;
    v.resize(n);
    env->GetFloatArrayRegion(a, 0, (jsize)n, v.data());
    return v;
}

std::string json_num(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream f;
    f << v;
    return f.str();
}

std::string json_strings(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + escape(v[i]) + "\"";
    return o + "]";
}

// finder fields shared by detect() and info()
std::string finder_json(const PageCorners& pc) {
    std::ostringstream f;
    f << "\"finder\": \"" << (pc.finder == Finder::Net ? "net" : "colour_threshold") << "\", \"n_pages\": " << pc.n_pages
      << ", \"page_index\": " << pc.page_index << ", \"whole_frame\": " << (pc.whole_frame ? "true" : "false")
      << ", \"spread\": " << (pc.spread ? "true" : "false") << ", \"net_margin\": " << json_num(pc.net_margin)
      << ", \"edge_strength\": " << json_num(pc.edge_strength) << ", \"edge_at_limit\": " << json_num(pc.edge_at_limit)
      << ", \"centroid\": [" << json_num(pc.centroid.x) << ", " << json_num(pc.centroid.y) << "]";
    return f.str();
}

struct Progress {
    JNIEnv* env;
    jobject listener;
    jmethodID mid = nullptr;
    Progress(JNIEnv* e, jobject l) : env(e), listener(l) {
        if (!l) return;
        jclass lc = env->GetObjectClass(l);
        mid = env->GetMethodID(lc, "onProgress", "(ILjava/lang/String;)V");
        env->DeleteLocalRef(lc);
    }
    void operator()(int pct, const std::string& stage) const {
        if (!mid) return;
        jstring js = env->NewStringUTF(stage.c_str());
        env->CallVoidMethod(listener, mid, (jint)pct, js);
        env->DeleteLocalRef(js);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
};

// ---- the nets (core/src/nets.h on core/src/infer.h, ONNX Runtime from the AAR). One Inference per
// group of nets, so that Settings > Acceleration (or Auto's choice) can give each group its own
// backends: the page finder, UVDoc (both passes), SyntheticDoc high-res (its own group: its fp16 grid
// misses the accelerator tolerance where UVDoc's meets it, Gate A), the shadow net and the rule net (CPU
// only: no fixed-shape model, not in Settings > Acceleration). An Inference keeps its sessions for the
// life of the process and moves a net to its next backend (CPU last) on any failure, so a net that
// failed on an accelerator stays on the CPU until the next configureNets with other backends.
struct NetGroup {
    std::unique_ptr<Inference> inf;
    std::string pref;             // backend names, comma separated ("" or "cpu": CPU only)
    std::mutex mu;
    std::set<int> used;           // nets that have run on this Inference (the first run creates the session)
    int gen = 0;                  // g_qnn_gen when it was made
    bool first_use(NetId n) {
        std::lock_guard<std::mutex> lk(mu);
        return used.insert((int)n).second;
    }
};

std::mutex g_nets_mu;             // guards everything below
std::shared_ptr<NetGroup> g_finder, g_flat, g_hr, g_shade, g_rule;
jobject g_assets = nullptr;       // global reference: keeps the AAssetManager alive
AAssetManager* g_mgr = nullptr;
std::string g_cache_dir;          // QNN context binaries (noBackupFilesDir/qnn_cache)
int g_threads = 0;
bool g_static_models = false;     // the CPU loads the fixed-shape UVDoc / SyntheticDoc copies (HUMAN GATE A-1)
std::string g_qnn_dir;            // the QNN libraries (the qnn_htp pack's lib/, loadQnn); empty: not loaded
std::string g_pack_models;        // the static models (the qnn_htp pack's models/); empty: from the APK's assets
int g_qnn_gen = 0;                // bumped by loadQnn: groups made before it are remade

// assets/models/<file>, or empty (missing: that net or backend is unavailable)
std::vector<uint8_t> load_model(std::string file) {
    AAssetManager* mgr;
    {
        std::lock_guard<std::mutex> lk(g_nets_mu);
        mgr = g_mgr;
        if (g_static_models && file == "uvdoc.onnx") file = "uvdoc_static.onnx";
        if (g_static_models && file == "syntheticdoc_hr.onnx") file = "syntheticdoc_hr_static.onnx";
    }
    std::vector<uint8_t> v;
    std::string pack;
    {
        std::lock_guard<std::mutex> lk(g_nets_mu);
        pack = g_pack_models;
    }
    // the static models come from the qnn_htp pack (not in the base APK with the packs)
    if (!pack.empty()) {
        std::ifstream f(pack + "/" + file, std::ios::binary);
        if (f) {
            v.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            if (!v.empty()) return v;
        }
    }
    if (!mgr) return v;
    AAsset* a = AAssetManager_open(mgr, ("models/" + file).c_str(), AASSET_MODE_BUFFER);
    if (!a) return v;
    const off_t n = AAsset_getLength(a);
    if (n > 0) {
        v.resize((size_t)n);
        const void* p = AAsset_getBuffer(a);
        if (p) std::memcpy(v.data(), p, v.size());
        else if (AAsset_read(a, v.data(), v.size()) != (int)v.size()) v.clear();
    }
    AAsset_close(a);
    return v;
}

// caller holds g_nets_mu
std::shared_ptr<NetGroup> make_group(const std::string& pref) {
    auto g = std::make_shared<NetGroup>();
    g->pref = pref;
    InferConfig c;
    c.load = load_model;
    if (!parse_backends(pref, c.preference)) {
        LOGI("nets: unknown backend in \"%s\": CPU only", pref.c_str());
        c.preference.clear();
    }
    c.cache_dir = g_cache_dir;
    c.threads = g_threads;
    c.qnn_dir = g_qnn_dir;
    g->inf = Inference::create(c);
    g->gen = g_qnn_gen;
    return g;
}

// the group's current Inference (CPU only until configureNets chooses)
std::shared_ptr<NetGroup> group(std::shared_ptr<NetGroup>& slot) {
    std::lock_guard<std::mutex> lk(g_nets_mu);
    if (!slot) slot = make_group("");
    return slot;
}

std::string shape_str(std::initializer_list<int64_t> s) {
    std::string o;
    for (int64_t d : s) o += (o.empty() ? "" : "x") + std::to_string(d);
    return o;
}

// one net's entry in the processing details' "accel" nets (Phase 0 keys): run_ms the net's run
// (Inference::last_ms), input_ms everything else the call took (bitmap, input, checks; on the net's
// first use also creating its session), backend the one that ran
std::string net_entry(bool ok, double run_ms, double total_ms, bool first, Backend b, const std::string& shape, int calls = 0) {
    std::ostringstream f;
    f << "{\"ok\": " << (ok ? "true" : "false") << ", \"run_ms\": " << json_num(std::round(run_ms * 10) / 10)
      << ", \"input_ms\": " << json_num(std::round(std::max(0.0, total_ms - run_ms) * 10) / 10) << ", \"first_use\": " << (first ? "true" : "false")
      << ", \"backend\": \"" << backend_name(b) << "\", \"shape\": \"" << shape << "\"";
    if (calls > 0) f << ", \"calls\": " << calls;
    f << "}";
    return f.str();
}

// "uv_pass1" (nets.cpp) is "uvdoc_pass1" in the processing details, as Phase 0 named it
const char* net_key(const std::string& run_name) { return run_name == "uv_pass1" ? "uvdoc_pass1" : run_name.c_str(); }

std::string run_shape(const std::string& run_name) {
    const UvSpec& sp = uv_spec(run_name == "synthdoc_hr" ? 2 : 0);
    return shape_str({1, 3, sp.in_h, sp.in_w});
}

// the shadow net or the rule net through Inference (nets.cpp shade_net_fn, rule_net_fn) with its timing
struct NetCall {
    std::shared_ptr<NetGroup> g;
    NetId id = NetId::ShadeNet;
    double run_ms = 0, total_ms = 0;
    int calls = 0;
    bool first = false, ok = false;
    std::string shape;
    std::function<bool(std::vector<float>&, int, int)> fn() {
        auto net = id == NetId::RuleNet ? rule_net_fn(*g->inf, &run_ms) : shade_net_fn(*g->inf, &run_ms);
        return [this, net](std::vector<float>& io, int w, int h) {
            if (calls++ == 0) first = g->first_use(id);
            auto t0 = Clock::now();
            bool r = net(io, w, h);
            total_ms += ms_since(t0);
            ok = ok || r;
            shape = shape_str({1, 3, h, w});
            return r;
        };
    }
    std::string entry() const { return net_entry(ok, run_ms, total_ms, first, g->inf->backend(id), shape, calls); }
};

// Gate A's grid measure (tools/accel/compare_outputs.py grid_px): a grid point's move between grids a
// and b (2 x gh x gw, normalised [-1, 1]) over a's own extent, times the 1600 x 2263 analysis page, px
std::vector<double> grid_px(const std::vector<float>& a, const std::vector<float>& b, int gw, int gh) {
    std::vector<double> d;
    const size_t n = (size_t)gw * gh;
    if (a.size() != 2 * n || b.size() != 2 * n || gw < 2 || gh < 2) return d;
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t m = v.size() / 2;
        return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
    };
    std::vector<double> eu, ev;
    for (int i = 0; i < gh; ++i) eu.push_back((double)a[(size_t)i * gw + gw - 1] - a[(size_t)i * gw]);
    for (int j = 0; j < gw; ++j) ev.push_back((double)a[n + (size_t)(gh - 1) * gw + j] - a[n + j]);
    const double ext_u = median(eu) / 2, ext_v = median(ev) / 2;
    const double page_w = 1600, page_h = std::round(1600 * 297.0 / 210.0);
    for (size_t k = 0; k < n; ++k) {
        double du = ((double)b[k] - a[k]) / 2 / ext_u * page_w, dv = ((double)b[n + k] - a[n + k]) / 2 / ext_v * page_h;
        d.push_back(std::hypot(du, dv));
    }
    return d;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_gpuInfo(JNIEnv* env, jclass) {
    bool ok = Gpu::instance().available();
    std::string s = (ok ? "GPU (OpenGL ES 3.1 compute): " : "CPU fallback (GPU unavailable): ") + Gpu::instance().info();
    return env->NewStringUTF(s.c_str());
}

// Settings > Acceleration: the backends of each net group (finder, flat: UVDoc, hr: SyntheticDoc
// high-res, shade; names as nets.h parse_backends, comma separated; "" = CPU only), the QNN compiled-model cache, intra-op threads (0: ORT's default).
// staticModels: the CPU loads uvdoc_static / syntheticdoc_hr_static instead of the originals (HUMAN
// GATE A-1). A group whose settings are unchanged keeps its Inference and sessions.
JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_configureNets(JNIEnv* env, jclass, jobject assets, jstring cache_dir, jstring finder,
                                                                       jstring flat, jstring hr, jstring shade, jint threads, jboolean static_models) {
    std::string cd = cache_dir ? jstr(env, cache_dir) : "", pf = finder ? jstr(env, finder) : "", fl = flat ? jstr(env, flat) : "",
                hs = hr ? jstr(env, hr) : "", sh = shade ? jstr(env, shade) : "";
    std::lock_guard<std::mutex> lk(g_nets_mu);
    if (!g_assets && assets) {
        g_assets = env->NewGlobalRef(assets);
        g_mgr = AAssetManager_fromJava(env, g_assets);
    }
    const bool same = cd == g_cache_dir && threads == g_threads && (bool)static_models == g_static_models;
    g_cache_dir = cd;
    g_threads = threads;
    g_static_models = static_models;
    auto set = [&](std::shared_ptr<NetGroup>& slot, const std::string& pref) {
        if (!slot || slot->pref != pref || !same || slot->gen != g_qnn_gen) slot = make_group(pref);
    };
    set(g_finder, pf);
    set(g_flat, fl);
    set(g_hr, hs);
    set(g_shade, sh);
    set(g_rule, "");
}

// The QNN libraries from the asset packs (QnnPacks.kt): lib_dir has libQnnHtp, libQnnHtpPrepare and
// libQnnSystem, htp_dirs (';'-separated) one Hexagon generation's libQnnHtpV<n>Stub (CPU side) and Skel
// (DSP side) each. The DSP finds the skeleton through ADSP_LIBRARY_PATH; the stubs and the common
// libraries are loaded here by path, so that QNN's own dlopen of them by name finds them already loaded
// (they are not in the app's native library directory). models_dir: the static models (load_model).
// The groups made before are remade on their next configureNets. JSON {"ok", "loaded": [...], "error"}.
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_loadQnn(JNIEnv* env, jclass, jstring lib_dir, jstring models_dir, jstring htp_dirs) {
    const std::string lib = jstr(env, lib_dir), models = jstr(env, models_dir), htp = jstr(env, htp_dirs);
    std::vector<std::string> gens;
    for (size_t p = 0; p <= htp.size();) {
        size_t q = htp.find(';', p);
        if (q == std::string::npos) q = htp.size();
        if (q > p) gens.push_back(htp.substr(p, q - p));
        p = q + 1;
    }
    std::string adsp;
    for (const auto& g : gens) adsp += g + ";";
    adsp += "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
    std::vector<std::string> order = {lib + "/libQnnSystem.so"};
    for (const auto& g : gens) {
        const std::string v = g.substr(g.find_last_of('/') + 1);   // "v73"
        order.push_back(g + "/libQnnHtpV" + v.substr(1) + "Stub.so");
    }
    order.push_back(lib + "/libQnnHtpPrepare.so");
    order.push_back(lib + "/libQnnHtp.so");
    std::ostringstream f;
    std::vector<std::string> loaded;
    std::string err;
    for (const auto& path : order) {
        if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
            loaded.push_back(path.substr(path.find_last_of('/') + 1));
        } else {
            const char* e = dlerror();
            err += (err.empty() ? "" : "; ") + std::string(e ? e : path.c_str());
        }
    }
    // libQnnHtp is what the QNN EP opens; without it there is no NPU (a missing stub only loses that generation)
    const bool ok = !loaded.empty() && loaded.back() == "libQnnHtp.so";
    if (ok) {
        std::lock_guard<std::mutex> lk(g_nets_mu);
        g_qnn_dir = lib;
        g_pack_models = models;
        ++g_qnn_gen;
    }
    LOGI("loadQnn: %s, loaded %zu of %zu%s%s", ok ? "ok" : "failed", loaded.size(), order.size(), err.empty() ? "" : ", ", err.c_str());
    f << "{\"ok\": " << (ok ? "true" : "false") << ", \"loaded\": " << json_strings(loaded) << ", \"adsp_library_path\": \"" << escape(adsp)
      << "\", \"error\": \"" << escape(err) << "\"}";
    return env->NewStringUTF(f.str().c_str());
}

// JSON: {"with_ort", "ort" (version), "providers": [the execution providers this ONNX Runtime build has]}
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_ortInfo(JNIEnv* env, jclass) {
    std::ostringstream f;
#ifdef FS_WITH_ORT
    try {
        std::vector<std::string> p = Ort::GetAvailableProviders();
        f << "{\"with_ort\": true, \"ort\": \"" << escape(Ort::GetVersionString()) << "\", \"providers\": " << json_strings(p) << "}";
    } catch (const std::exception& e) {
        f.str("");
        f << "{\"with_ort\": true, \"ort\": \"unknown\", \"providers\": [], \"error\": \"" << escape(e.what()) << "\"}";
    }
#else
    f << "{\"with_ort\": false, \"ort\": \"none\", \"providers\": []}";
#endif
    return env->NewStringUTF(f.str().c_str());
}

// The page finder's logits for the whole photo (float[2 * 256 * 256]), or null (net unavailable: the
// caller uses the colour threshold). timing (optional, 1 element): its "accel" nets entry (net_entry).
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_pageFinder(JNIEnv* env, jclass, jobject bmp, jobjectArray timing) {
    auto t0 = Clock::now();
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return nullptr;
    auto g = group(g_finder);
    const bool first = g->first_use(NetId::PageFinder);
    std::vector<float> lg;
    const bool ok = page_finder_logits(bgr, *g->inf, lg);
    if (!ok) return nullptr;
    if (timing && env->GetArrayLength(timing) >= 1) {
        std::string e = net_entry(true, g->inf->last_ms(NetId::PageFinder), ms_since(t0), first, g->inf->backend(NetId::PageFinder),
                                  shape_str({1, 3, SEG_SIZE, SEG_SIZE}));
        jstring js = env->NewStringUTF(e.c_str());
        env->SetObjectArrayElement(timing, 0, js);
        env->DeleteLocalRef(js);
    }
    jfloatArray a = env->NewFloatArray((jsize)lg.size());
    env->SetFloatArrayRegion(a, 0, (jsize)lg.size(), lg.data());
    return a;
}

// -> JSON {"ok", "error", "finder", ..., "pages": [{"corners": [x0, y0, ... x3, y3] (bitmap px, TL TR BR BL),
//    "ok", "fit_median_px", "needs_manual_crop": [reasons], finder fields}, ...]} (2 pages: book spread)
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_detect(JNIEnv* env, jclass, jobject bmp, jfloatArray logits,
                                                                   jfloat focal_px) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return env->NewStringUTF("{\"ok\": false, \"error\": \"could not read the photo bitmap\", \"pages\": []}");
    std::vector<float> lg = read_logits(env, logits);
    std::vector<PageAssessment> pages;
    try {
        pages = assess_pages(bgr, lg.empty() ? nullptr : lg.data(), focal_px);
    } catch (const std::exception& e) {
        LOGI("detect failed: %s", e.what());
        return env->NewStringUTF((std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\", \"pages\": []}").c_str());
    }
    std::ostringstream f;
    bool any = false;
    for (auto& a : pages) any |= a.pc.ok;
    f << "{\"ok\": " << (any ? "true" : "false") << ", \"error\": \"" << (any ? "" : "no page found") << "\", \"pages\": [";
    for (size_t i = 0; i < pages.size(); ++i) {
        const PageAssessment& a = pages[i];
        f << (i ? ", " : "") << "{\"ok\": " << (a.pc.ok ? "true" : "false") << ", \"corners\": [";
        for (int k = 0; k < 4; ++k) f << (k ? ", " : "") << json_num(a.pc.pts[k].x) << ", " << json_num(a.pc.pts[k].y);
        f << "], \"fit_median_px\": " << json_num(a.fit_median_px) << ", \"needs_manual_crop\": " << json_strings(a.reasons) << ", "
          << finder_json(a.pc) << "}";
    }
    f << "]}";
    return env->NewStringUTF(f.str().c_str());
}

// flatten: Settings > neural flattening (the flattening nets run unless perspectiveOnly); flatNets:
// 0 both, 1 UVDoc only, 2 SyntheticDoc high-res only; shadows: the shadow net; rules: the rule net in the
// line stage (Settings > Table rules). The nets run here,
// through the groups' Inference (core/src/nets.cpp, as the Kotlin FlatNet / ShadeNet calls did).
JNIEXPORT jlong JNICALL Java_com_albus_flatscan_NativeLib_process(JNIEnv* env, jclass, jobject bmp, jfloatArray corners,
                                                                  jboolean edited, jfloat focal_px, jboolean auto_orient,
                                                                  jint render_w, jboolean use_gpu, jfloatArray logits,
                                                                  jint page_index, jboolean force_homography, jboolean perspective_only, jfloatArray ignore,
                                                                  jboolean flatten, jint flat_nets, jboolean shadows, jboolean rules,
                                                                  jobject listener) {
    auto s = std::make_unique<Session>();
    s->use_gpu = use_gpu;
    PipelineResult& r = s->result;
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) {
        r.error = "could not read the photo bitmap (RGBA_8888 required)";
        return (jlong)s.release();
    }
    PipelineOptions o;
    o.corners_edited = edited;
    if (corners && env->GetArrayLength(corners) == 8) {
        jfloat v[8];
        env->GetFloatArrayRegion(corners, 0, 8, v);
        for (int k = 0; k < 4; ++k) o.corners[k] = cv::Point2d(v[2 * k], v[2 * k + 1]);
    } else {
        o.corners_edited = false;
    }
    o.focal_px = focal_px;
    std::vector<float> lg = read_logits(env, logits);
    o.seg_logits = lg.empty() ? nullptr : lg.data();
    o.page_index = page_index;
    o.force_homography = force_homography;
    o.perspective_only = perspective_only;
    for (const Rect01& q : parse_protect(env, ignore)) o.ignore.push_back({q.x0, q.y0, q.x1, q.y1});
    o.auto_orient = auto_orient;
    o.render_w = render_w;
    o.use_gpu = use_gpu;
    Progress progress(env, listener);
    // the flattening nets, run by run_photo beside its CPU stages (PipelineOptions::flat_nets, Gate D:
    // the same outputs as before them): with automatic corners a first UVDoc pass on the whole photo,
    // then UVDoc and SyntheticDoc high-res on the page it found (the high-res grid in the first slot).
    // They run on a worker thread, which must not touch JNI: their entries are made after run_photo.
    std::ostringstream acc;   // the "accel" nets entries (info)
    NetsResult nets;
    std::shared_ptr<NetGroup> flat_g, hr_g;
    if (flatten && !perspective_only) {
        flat_g = group(g_flat);
        hr_g = group(g_hr);
        NetsRequest rq;
        rq.corners_edited = o.corners_edited;
        rq.nets = (FlatNets)std::min(2, std::max(0, (int)flat_nets));
        rq.seg_logits = o.seg_logits;
        rq.page_index = page_index;
        o.flat_nets = flat_nets_fn(bgr, rq, *flat_g->inf, &nets, hr_g->inf.get());
    }
    // the shadow net, after SyntheticDoc's grid (as with the Kotlin ShadeNet)
    NetCall shade;
    if (shadows) {
        shade.g = group(g_shade);
        o.shade_after_uv = true;
        o.shade_net = shade.fn();
    }
    // the rule net, in the line stage on the run_photo thread (PipelineOptions::rule_net)
    NetCall rule;
    rule.id = NetId::RuleNet;
    if (rules) {
        rule.g = group(g_rule);
        o.rule_net = rule.fn();
    }
    o.progress = [&](int pct, const char* stage) { progress(pct, stage); };
    auto t1 = Clock::now();
    try {
        r = run_photo(bgr, o);
        bgr.release();
        if (r.ok) {
            // free what re-renders never need
            r.stage1.release();
            r.warp = WarpDebug();
            auto t0 = Clock::now();
            s->pp = page_params(r.page, use_gpu);
            r.ms.emplace_back(s->pp.gpu_background ? "page params (GPU)" : "page params (CPU)", ms_since(t0));
            s->legend_px = r.disp.x.empty() ? 1.f : wrinkle_legend_px(r.disp);
        }
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = e.what();
    }
    for (const NetRun& n : nets.runs) {
        const bool hr_net = n.name == "synthdoc_hr";
        const bool first = (hr_net ? hr_g : flat_g)->first_use(hr_net ? NetId::SynthDocHR : NetId::UVDoc);
        acc << (acc.tellp() > 0 ? ", " : "") << "\"" << net_key(n.name) << "\": " << net_entry(n.ok, n.ms, n.total_ms, first, n.backend, run_shape(n.name));
    }
    if (shade.calls > 0) acc << (acc.tellp() > 0 ? ", " : "") << "\"shadenet\": " << shade.entry();
    if (rule.calls > 0) acc << (acc.tellp() > 0 ? ", " : "") << "\"rulenet\": " << rule.entry();
    std::ostringstream a;
    a << "\"nets\": {" << acc.str() << "}, \"flattening_nets_ms\": " << std::lround(std::max(0.0, r.nets_ms)) << ", \"nets_overlap\": " << r.nets_overlap
      << ", \"nets_wait_ms\": " << std::lround(std::max(0.0, r.nets_wait_ms)) << ", \"process_ms\": " << std::lround(ms_since(t1));
    s->accel = a.str();
    if (!r.ok) LOGI("pipeline failed: %s", r.error.c_str());
    progress(100, "Done");
    return (jlong)s.release();
}

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_info(JNIEnv* env, jclass, jlong h) {
    if (!h) return env->NewStringUTF("{\"ok\": false, \"error\": \"no session\"}");
    Session* s = session(h);
    const PipelineResult& r = s->result;
    std::ostringstream f;
    f << "{\"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << escape(r.error) << "\", \"mode\": \""
      << geom_mode_name(r.mode) << "\", \"fell_back\": " << (r.fell_back ? "true" : "false")
      << ", \"fit_median_px\": " << r.fit_median_px << ", \"rot\": " << r.rot << ", \"gpu_resample\": " << (r.gpu_resample ? "true" : "false")
      << ", \"legend_px\": " << s->legend_px << ", \"has_paper_mask\": " << (r.paper.empty() ? "false" : "true")
      << ", \"needs_manual_crop\": " << json_strings(r.needs_manual_crop)
      << ", \"fit_median_full_px\": " << json_num(r.detection.ok ? r.fit_median_px / r.detection.scale : -1)
      << ", \"uv_agreement\": " << json_num(r.uv_agreement) << ", \"uv_rejected\": " << (r.uv_rejected ? "true" : "false")
      << ", \"line_distortion\": " << json_num(r.line_distortion) << ", \"lines_rejected\": " << (r.lines_rejected ? "true" : "false")
      << ", \"fingers_erased\": " << json_num(r.fingers_erased) << ", \"uv_model\": " << r.uv_model
      << ", \"uv_spread\": [" << json_num(r.uv_spread[0]) << ", " << json_num(r.uv_spread[1]) << "]"
      << ", \"shadows_removed\": " << (r.shadows_removed ? "true" : "false") << ", \"shade_net\": " << (r.shade_net ? "true" : "false")
      << ", \"rule_net\": " << (r.rule_net ? "true" : "false") << ", \"net\": [" << r.net_dropped << ", " << r.net_added << ", " << r.net_joined
      << ", " << r.net_kept << "], \"spots_dropped\": " << (r.lines_spots_dropped ? "true" : "false");
    if (r.detection.ok) f << ", " << finder_json(r.detection);
    if (!s->accel.empty()) f << ", \"accel\": {" << s->accel << "}";
    f << ", \"ms\": {";
    for (size_t i = 0; i < r.ms.size(); ++i) f << (i ? ", " : "") << "\"" << escape(r.ms[i].first) << "\": " << (long)r.ms[i].second;
    f << "}}";
    return env->NewStringUTF(f.str().c_str());
}

JNIEXPORT jintArray JNICALL Java_com_albus_flatscan_NativeLib_pageSize(JNIEnv* env, jclass, jlong h) {
    jint v[2] = {0, 0};
    if (h) {
        const PipelineResult& r = session(h)->result;
        v[0] = r.page.cols;
        v[1] = r.page.rows;
    }
    jintArray a = env->NewIntArray(2);
    env->SetIntArrayRegion(a, 0, 2, v);
    return a;
}

// full-size render with RenderParams into `out` (pageSize); protect: page-fraction rectangles
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_render(JNIEnv* env, jclass, jlong h, jfloatArray params,
                                                                  jfloatArray protect, jboolean use_gpu, jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return env->NewStringUTF("{\"ok\": false, \"error\": \"no page\"}");
    RenderParams rp = parse_params(env, params);
    std::vector<Rect01> prot = parse_protect(env, protect);
    auto t0 = Clock::now();
    RenderStats st;
    cv::Mat page;
    try {
        page = render_page(s->result.page, s->result.paper, s->pp, rp, prot, use_gpu, &st);
    } catch (const std::exception& e) {
        return env->NewStringUTF((std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}").c_str());
    }
    double ms_render = ms_since(t0);
    cv::Size bs = bitmap_size(env, out);
    bool ok = bgra_to_bitmap(env, page, out);
    std::ostringstream f;
    f << "{\"ok\": " << (ok ? "true" : "false") << ", \"filter\": \"" << filter_name(rp.filter) << "\", \"gpu_filter\": "
      << (st.gpu_filter ? "true" : "false") << ", \"filter_ms\": " << (long)st.ms_filter << ", \"solid_ms\": " << (long)st.ms_solid
      << ", \"render_ms\": " << (long)ms_render << ", \"total_ms\": " << (long)ms_since(t0)
      << ", \"resized\": " << (bs != page.size() ? "true" : "false") << ", \"solid\": {\"regions\": " << st.solid.regions
      << ", \"clusters\": " << st.solid.clusters << ", \"filled\": " << st.solid.filled << ", \"white\": " << st.solid.white
      << ", \"skipped\": " << st.solid.skipped + st.solid.skipped_varied << ", \"blobs\": " << st.solid.blobs << "}}";
    return env->NewStringUTF(f.str().c_str());
}

// live preview of any size: CPU filter on an INTER_AREA downscale of the paper-filled page, no Solid colors
JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_renderThumb(JNIEnv* env, jclass, jlong h, jfloatArray params, jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return;
    cv::Size bs = bitmap_size(env, out);
    if (bs.width <= 0 || bs.height <= 0) return;
    RenderParams rp = parse_params(env, params);
    try {
        cv::Mat src = s->filled_at(bs);
        cv::Mat t = filter_cpu(src, s->pp, rp.filter, rp.strength, rp.adj);
        bgra_to_bitmap(env, t, out);
    } catch (const std::exception& e) {
        LOGI("renderThumb failed: %s", e.what());
    }
}

JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_renderBefore(JNIEnv* env, jclass, jlong h, jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return;
    try {
        bgra_to_bitmap(env, s->result.before.empty() ? s->result.page : s->result.before, out);
    } catch (const std::exception& e) {
        LOGI("renderBefore failed: %s", e.what());
    }
}

// wrinkle map overlay (transparent): heat map and / or deformed grid; returns the legend top in px
JNIEXPORT jfloat JNICALL Java_com_albus_flatscan_NativeLib_renderOverlay(JNIEnv* env, jclass, jlong h, jboolean heat, jboolean grid,
                                                                        jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return 0.f;
    float top = s->legend_px;
    try {
        cv::Mat ov;
        if (s->result.disp.x.empty()) ov = cv::Mat(s->result.page.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
        else ov = wrinkle_overlay(s->result.disp, heat, grid, &top);
        bgra_to_bitmap(env, ov, out, true);
    } catch (const std::exception& e) {
        LOGI("renderOverlay failed: %s", e.what());
    }
    return top;
}

// 256-bin luma histogram of the filtered page (about 1/4 scale, CPU, no Solid colors)
JNIEXPORT jintArray JNICALL Java_com_albus_flatscan_NativeLib_histogram(JNIEnv* env, jclass, jlong h, jfloatArray params) {
    std::vector<int> hist(256, 0);
    Session* s = session(h);
    if (s && s->result.ok) {
        RenderParams rp = parse_params(env, params);
        try {
            const cv::Mat& pg = s->result.page;
            cv::Mat src = s->filled_at(cv::Size(std::max(1, pg.cols / 4), std::max(1, pg.rows / 4)));
            hist = luma_histogram(filter_cpu(src, s->pp, rp.filter, rp.strength, rp.adj));
        } catch (const std::exception& e) {
            LOGI("histogram failed: %s", e.what());
        }
    }
    jintArray a = env->NewIntArray(256);
    env->SetIntArrayRegion(a, 0, 256, reinterpret_cast<const jint*>(hist.data()));
    return a;
}

JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_release(JNIEnv*, jclass, jlong h) { delete session(h); }

// Erase tool: the page as processed, then every rectangle (page fractions [x0, y0, x1, y1, ...])
// filled with the paper around it. An empty array restores the page.
JNIEXPORT jboolean JNICALL Java_com_albus_flatscan_NativeLib_setErase(JNIEnv* env, jclass, jlong h, jfloatArray rects) {
    Session* s = session(h);
    if (!s || !s->result.ok) return JNI_FALSE;
    std::vector<Rect01> rs = parse_protect(env, rects);
    PipelineResult& r = s->result;
    try {
        if (s->page_orig.empty()) {
            if (rs.empty()) return JNI_TRUE;
            s->page_orig = r.page.clone();
            s->paper_orig = r.paper.clone();
        }
        r.page = s->page_orig.clone();
        r.paper = s->paper_orig.clone();
        if (!rs.empty()) {
            cv::Mat mask = cv::Mat::zeros(r.page.size(), CV_8U);
            for (const Rect01& q : rs) {
                cv::Rect box((int)std::floor(q.x0 * r.page.cols), (int)std::floor(q.y0 * r.page.rows),
                             (int)std::ceil((q.x1 - q.x0) * r.page.cols), (int)std::ceil((q.y1 - q.y0) * r.page.rows));
                box &= cv::Rect(0, 0, r.page.cols, r.page.rows);
                if (box.area() > 0) mask(box).setTo(255);
            }
            fill_with_paper(r.page, mask, r.paper.empty() ? nullptr : &r.paper);
        }
        s->pp = page_params(r.page, s->use_gpu);
        std::lock_guard<std::mutex> lk(s->mu);
        s->small.clear();
    } catch (const std::exception& e) {
        LOGI("setErase failed: %s", e.what());
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

// lines the warp map was fitted to: [kind, x0, y0, x1, y1, ...] as page fractions (PipelineResult::lines)
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_lines(JNIEnv* env, jclass, jlong h) {
    std::vector<float> v;
    Session* s = session(h);
    if (s && s->result.ok)
        for (const auto& l : s->result.lines) v.insert(v.end(), l.begin(), l.end());
    jfloatArray a = env->NewFloatArray((jsize)v.size());
    if (!v.empty()) env->SetFloatArrayRegion(a, 0, (jsize)v.size(), v.data());
    return a;
}

// a cheap copy of what restoreSession needs (shares the pixels); writeSnapshot frees it
JNIEXPORT jlong JNICALL Java_com_albus_flatscan_NativeLib_snapshot(JNIEnv*, jclass, jlong h) {
    Session* s = session(h);
    if (!s || !s->result.ok) return 0;
    auto* sn = new Snapshot();
    sn->page = s->result.page;
    sn->paper = s->result.paper;
    sn->before = s->result.before;
    sn->disp = s->result.disp;
    sn->lines = s->result.lines;
    sn->legend_px = s->legend_px;
    return (jlong)sn;
}

// writes a snapshot into the existing directory `dir` and frees it (any thread)
JNIEXPORT jboolean JNICALL Java_com_albus_flatscan_NativeLib_writeSnapshot(JNIEnv* env, jclass, jlong h, jstring dir) {
    std::unique_ptr<Snapshot> sn(reinterpret_cast<Snapshot*>(h));
    if (!sn) return JNI_FALSE;
    try {
        return write_snapshot(*sn, jstr(env, dir)) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGI("writeSnapshot failed: %s", e.what());
        return JNI_FALSE;
    }
}

// a session from a written snapshot (no geometry is re-run); 0 when it cannot be read
JNIEXPORT jlong JNICALL Java_com_albus_flatscan_NativeLib_restoreSession(JNIEnv* env, jclass, jstring dir, jboolean use_gpu) {
    std::string path = jstr(env, dir);
    auto s = std::make_unique<Session>();
    s->use_gpu = use_gpu;
    PipelineResult& r = s->result;
    try {
        r.page = read_bgra(path + "/page.jpg");
        if (r.page.empty()) return 0;
        if (std::ifstream(path + "/paper.png").good()) r.paper = cv::imread(path + "/paper.png", cv::IMREAD_GRAYSCALE);
        if (!r.paper.empty() && r.paper.size() != r.page.size()) r.paper.release();
        r.before = read_bgra(path + "/before.jpg");
        if (!r.before.empty() && r.before.size() != r.page.size()) r.before.release();
        std::vector<int32_t> head;
        std::vector<float> data;
        if (read_floats(path + "/disp.bin", head, 5, data) && head[2] > 0 && head[3] > 0 &&
            data.size() == 2 * (size_t)head[2] * head[3]) {
            GridMap& g = r.disp;
            g.out_w = head[0]; g.out_h = head[1]; g.gw = head[2]; g.gh = head[3]; g.stride = head[4];
            g.x.assign(data.begin(), data.begin() + data.size() / 2);
            g.y.assign(data.begin() + data.size() / 2, data.end());
        }
        if (read_floats(path + "/lines.bin", head, 1, data) && head[0] >= 0 && data.size() == 5 * (size_t)head[0] + 1) {
            for (int i = 0; i < head[0]; ++i)
                r.lines.push_back({data[5 * i], data[5 * i + 1], data[5 * i + 2], data[5 * i + 3], data[5 * i + 4]});
            s->legend_px = data.back();
        } else {
            s->legend_px = r.disp.x.empty() ? 1.f : wrinkle_legend_px(r.disp);
        }
        auto t0 = Clock::now();
        s->pp = page_params(r.page, use_gpu);
        r.ms.emplace_back("restored; page params", ms_since(t0));
        r.ok = true;
    } catch (const std::exception& e) {
        LOGI("restoreSession failed: %s", e.what());
        return 0;
    }
    return (jlong)s.release();
}

// ---- on-device self-test and benchmark (brief section 5.1 / 5.2 / 5.6 on a real phone)
static long peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::atol(line.c_str() + 6);
    return -1;
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = q / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] + (v[lo + 1] - v[lo]) * (idx - lo);
}

static int max_abs_diff(const cv::Mat& a, const cv::Mat& b, int* n_gt1) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    double mx;
    cv::minMaxLoc(d.reshape(1), nullptr, &mx);
    if (n_gt1) *n_gt1 = cv::countNonZero(d.reshape(1) > 1);
    return (int)mx;
}

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_selfTest(JNIEnv* env, jclass, jobject bmp, jstring ref_path, jint runs,
                                                                    jobject listener) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    const char* rp = env->GetStringUTFChars(ref_path, nullptr);
    std::string ref = rp;
    env->ReleaseStringUTFChars(ref_path, rp);
    Progress progress(env, listener);
    std::ostringstream o;
    try {
        if (bgr.empty()) throw std::runtime_error("could not read the test photo");
        std::map<std::string, std::vector<double>> ms;
        std::vector<std::string> order;
        PipelineResult r;
        int total = runs + 1;
        for (int k = 0; k < total; ++k) {
            progress(100 * k / (total + 1), k == 0 ? "Warm-up run" : "Run " + std::to_string(k) + " of " + std::to_string(runs));
            PipelineOptions opt;
            r = run_photo(bgr, opt);
            if (!r.ok) throw std::runtime_error(r.error);
            if (k == 0) continue;   // warm-up
            for (auto& m : r.ms) {
                if (!ms.count(m.first)) order.push_back(m.first);
                ms[m.first].push_back(m.second);
            }
        }
        progress(95, "Parity checks");
        // map parity against the reference grid (max over grid nodes == max over all pixels)
        Npz z = read_npz(ref);
        const NpyArray &ax = z.at("map_x"), &ay = z.at("map_y");
        double max_epe = 0, sum = 0;
        size_t n = 0;
        bool shape_ok = (int)ax.shape[0] == r.final_map.gh && (int)ax.shape[1] == r.final_map.gw;
        if (shape_ok)
            for (size_t i = 0; i < r.final_map.x.size(); ++i) {
                double e = std::hypot(r.final_map.x[i] - ax.at(i), r.final_map.y[i] - ay.at(i));
                max_epe = std::max(max_epe, e);
                sum += e;
                ++n;
            }
        cv::Mat bgra;
        cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
        cv::Mat cpu_page = resample_cpu(bgra, r.render_map);
        int g1 = 0, c1 = 0;
        int dres = max_abs_diff(r.page, cpu_page, &g1);
        // filter: render_page with Auto color 0.7 (the default), GPU vs CPU
        PageParams pp = page_params(r.page, true);
        RenderParams rpar;
        rpar.filter = Filter::AutoColor;
        rpar.strength = 0.7f;
        RenderStats sg, sc;
        auto t0 = Clock::now();
        cv::Mat cg = render_page(r.page, r.paper, pp, rpar, {}, true, &sg);
        double ms_cg = ms_since(t0);
        t0 = Clock::now();
        cv::Mat cc = render_page(r.page, r.paper, pp, rpar, {}, false, &sc);
        double ms_cc = ms_since(t0);
        int dclean = max_abs_diff(cg, cc, &c1);
        o << "{\"ok\": true, \"runs\": " << runs << ", \"gpu\": \"" << escape(Gpu::instance().info()) << "\""
          << ", \"gpu_resample\": " << (r.gpu_resample ? "true" : "false")
          << ", \"map_parity\": {\"shape_ok\": " << (shape_ok ? "true" : "false") << ", \"max_epe_px\": " << max_epe
          << ", \"mean_epe_px\": " << (n ? sum / n : 0) << ", \"passed\": " << (shape_ok && max_epe <= 0.1 ? "true" : "false") << "}"
          << ", \"gpu_vs_cpu_resample\": {\"max_levels\": " << dres << ", \"values_gt1\": " << g1 << "}"
          << ", \"gpu_vs_cpu_filter\": {\"filter\": \"Auto color\", \"strength\": 0.7, \"max_levels\": " << dclean
          << ", \"values_gt1\": " << c1 << ", \"gpu_ms\": " << (long)ms_cg << ", \"cpu_ms\": " << (long)ms_cc
          << ", \"gpu_used\": " << (sg.gpu_filter ? "true" : "false") << ", \"page_params_ms\": " << (long)pp.ms << "}"
          << ", \"peak_rss_mb\": " << peak_rss_kb() / 1024 << ", \"stages_ms\": [";
        for (size_t i = 0; i < order.size(); ++i) {
            auto& v = ms[order[i]];
            o << (i ? ", " : "") << "{\"stage\": \"" << escape(order[i]) << "\", \"median\": " << (long)pct(v, 50) << ", \"p95\": " << (long)pct(v, 95) << "}";
        }
        o << "]}";
    } catch (const std::exception& e) {
        o.str("");
        o << "{\"ok\": false, \"error\": \"" << escape(e.what()) << "\"}";
    }
    progress(100, "Done");
    return env->NewStringUTF(o.str().c_str());
}


// ---- Settings > Acceleration > Auto: the nets on the bundled self-test photo, on the CPU and on every
// candidate backend, timed and compared with the CPU on the same inputs (made as automatic processing
// makes them; the shadow net gets the photo at the A4 page's net size, 352 x 512). A backend is
// acceptable for a group of nets (UVDoc's two passes; SyntheticDoc high-res; the shadow net) when each
// net of the group ran on it (no fallback to the next backend) within the accelerator tolerances below;
// Auto takes the fastest acceptable backend per group, the CPU when none is faster. The page finder stays on the CPU (PLAN: 34 NNAPI partitions).
// Accelerator tolerances (the CPU path itself stays bit-identical):
//   flattening nets: Gate A's grid measure (grid_px), p99 <= 0.25 px (Gate A's target) and max <= 1 px
//                    of the 1600 x 2263 analysis page;
//   shadow net: |shading difference| x 255, p99 <= 0.5 and max <= 2 levels (Gate A's fp16 stand-in: p99 0.07).
constexpr double AUTO_GRID_P99_PX = 0.25, AUTO_GRID_MAX_PX = 1.0, AUTO_SHADE_P99 = 0.5, AUTO_SHADE_MAX = 2.0;

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_benchBackends(JNIEnv* env, jclass, jobject bmp, jstring candidates, jint runs,
                                                                         jobject listener) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    const std::string cand_s = candidates ? jstr(env, candidates) : "";
    Progress progress(env, listener);
    std::ostringstream o;
    try {
        if (bgr.empty()) throw std::runtime_error("could not read the test photo");
        runs = std::max(1, (int)runs);
        std::vector<Backend> all = {Backend::CPU}, cands;
        if (!parse_backends(cand_s, cands)) throw std::runtime_error("unknown backend in \"" + cand_s + "\"");
        for (Backend b : cands)
            if (std::find(all.begin(), all.end(), b) == all.end()) all.push_back(b);
        auto make = [](Backend b) {
            InferConfig c;
            c.load = load_model;
            if (b != Backend::CPU) c.preference = {b};
            std::lock_guard<std::mutex> lk(g_nets_mu);
            c.cache_dir = g_cache_dir;
            c.threads = g_threads;
            c.qnn_dir = g_qnn_dir;
            return Inference::create(c);
        };
        // the inputs, from CPU runs as automatic processing makes them
        progress(1, "Auto: preparing the inputs (CPU)");
        std::unique_ptr<Inference> cpu = make(Backend::CPU);
        std::vector<float> lg;
        const float* logits = page_finder_logits(bgr, *cpu, lg) ? lg.data() : nullptr;
        PageCorners pc = detect_corners(bgr, 2400, logits, 0);
        NetsRequest rq;
        const double w1 = bgr.cols - 1.0, h1 = bgr.rows - 1.0;
        rq.corners = pc.ok ? pc.pts : std::array<cv::Point2d, 4>{cv::Point2d(0, 0), cv::Point2d(w1, 0), cv::Point2d(w1, h1), cv::Point2d(0, h1)};
        rq.seg_logits = logits;
        NetsResult ref = run_flat_nets(bgr, rq, *cpu);
        cv::Mat blank;
        if (logits) {
            PageFindResult f = find_page_net(bgr, logits);
            if (f.pages.size() == 2) blank = f.pages[1].mask;
        }
        struct In { std::string key; NetId id; int model; Tensor x; };
        std::vector<In> in;
        auto add_uv = [&](const std::string& key, const UvCrop& c) {
            const UvSpec& sp = uv_spec(c.model);
            in.push_back({key, c.model == 2 ? NetId::SynthDocHR : NetId::UVDoc, c.model, Tensor{{1, 3, sp.in_h, sp.in_w}, uv_input(bgr, c, blank)}});
        };
        if (!ref.grid_pass1.empty()) add_uv("uvdoc_pass1", ref.crop_pass1);
        if (!ref.grid2.empty()) add_uv("uvdoc", ref.crop2);
        if (!ref.grid1.empty()) add_uv(ref.crop1.model == 2 ? "synthdoc_hr" : "uvdoc", ref.crop1);
        {
            const int sw = 352, sh = 512;
            cv::Mat sm;
            cv::resize(bgr, sm, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
            Tensor t{{1, 3, sh, sw}, std::vector<float>((size_t)3 * sw * sh)};
            for (int y = 0; y < sh; ++y)
                for (int x = 0; x < sw; ++x)
                    for (int c = 0; c < 3; ++c) t.data[((size_t)c * sh + y) * sw + x] = sm.at<cv::Vec3b>(y, x)[c] / 255.f;
            in.push_back({"shadenet", NetId::ShadeNet, -1, std::move(t)});
        }
        // every backend on every input: a first run (session creation; QNN: the graph compile or the
        // cached context) and `runs` timed runs
        struct Res { bool ok = false, ran = false, pass = false; double first_ms = 0, median = 0, p95 = 0, p99 = 0, max = 0; std::vector<float> out; };
        std::vector<std::vector<Res>> res(all.size(), std::vector<Res>(in.size()));
        for (size_t bi = 0; bi < all.size(); ++bi) {
            std::unique_ptr<Inference> own;
            Inference* inf = cpu.get();
            if (all[bi] != Backend::CPU) inf = (own = make(all[bi])).get();
            for (size_t k = 0; k < in.size(); ++k) {
                progress((int)(5 + 90 * (bi * in.size() + k) / (all.size() * in.size())),
                         std::string("Auto: ") + in[k].key + " on " + backend_name(all[bi]));
                Res& r = res[bi][k];
                std::vector<Tensor> y;
                auto t0 = Clock::now();
                r.ok = inf->run(in[k].id, in[k].x, y) && !y.empty();
                r.first_ms = ms_since(t0);
                r.ran = r.ok && inf->backend(in[k].id) == all[bi];
                if (!r.ran) continue;
                std::vector<double> t;
                for (int i = 0; i < runs && r.ok; ++i) {
                    r.ok = inf->run(in[k].id, in[k].x, y) && !y.empty() && inf->backend(in[k].id) == all[bi];
                    t.push_back(inf->last_ms(in[k].id));
                }
                r.ran = r.ok;
                if (!r.ok) continue;
                r.median = pct(t, 50);
                r.p95 = pct(t, 95);
                r.out = std::move(y[0].data);
            }
        }
        // the differences from the CPU
        for (size_t bi = 0; bi < all.size(); ++bi)
            for (size_t k = 0; k < in.size(); ++k) {
                Res& r = res[bi][k];
                const Res& c = res[0][k];
                if (!r.ran || !c.ran || r.out.size() != c.out.size()) continue;
                std::vector<double> d;
                if (in[k].model < 0) {
                    for (size_t i = 0; i < r.out.size(); ++i) d.push_back(std::fabs((double)r.out[i] - c.out[i]) * 255);
                } else {
                    const UvSpec& sp = uv_spec(in[k].model);
                    d = grid_px(c.out, r.out, sp.gw, sp.gh);
                }
                if (d.empty()) continue;
                r.p99 = pct(d, 99);
                r.max = *std::max_element(d.begin(), d.end());
                r.pass = in[k].model < 0 ? r.p99 <= AUTO_SHADE_P99 && r.max <= AUTO_SHADE_MAX : r.p99 <= AUTO_GRID_P99_PX && r.max <= AUTO_GRID_MAX_PX;
            }
        // the fastest acceptable backend per group (the CPU's own runs are the reference: acceptable)
        // model: 0 UVDoc, 2 SyntheticDoc high-res, -1 the shadow net
        auto choose = [&](int model) {
            size_t best = 0;
            double best_ms = -1;
            for (size_t bi = 0; bi < all.size(); ++bi) {
                double sum = 0;
                bool okb = true, any = false;
                for (size_t k = 0; k < in.size(); ++k) {
                    if (in[k].model != model) continue;
                    any = true;
                    const Res& r = res[bi][k];
                    okb = okb && r.ran && (bi == 0 || r.pass);
                    sum += r.median;
                }
                if (!any || !okb) continue;
                if (best_ms < 0 || sum < best_ms) best = bi, best_ms = sum;
            }
            return all[best];
        };
        o << "{\"ok\": true, \"runs\": " << runs << ", \"corners_found\": " << (pc.ok ? "true" : "false")
          << ", \"tolerance\": {\"grid_p99_px\": " << AUTO_GRID_P99_PX << ", \"grid_max_px\": " << AUTO_GRID_MAX_PX
          << ", \"shade_p99_levels\": " << AUTO_SHADE_P99 << ", \"shade_max_levels\": " << AUTO_SHADE_MAX << "}, \"backends\": [";
        for (size_t bi = 0; bi < all.size(); ++bi) {
            o << (bi ? ", " : "") << "{\"backend\": \"" << backend_name(all[bi]) << "\", \"nets\": {";
            for (size_t k = 0; k < in.size(); ++k) {
                const Res& r = res[bi][k];
                o << (k ? ", " : "") << "\"" << in[k].key << "\": {\"ran\": " << (r.ran ? "true" : "false") << ", \"first_ms\": " << std::lround(r.first_ms);
                if (r.ran)
                    o << ", \"median_ms\": " << json_num(std::round(r.median * 10) / 10) << ", \"p95_ms\": " << json_num(std::round(r.p95 * 10) / 10)
                      << ", \"p99_diff\": " << json_num(r.p99) << ", \"max_diff\": " << json_num(r.max) << ", \"pass\": " << (bi == 0 || r.pass ? "true" : "false");
                o << "}";
            }
            o << "}}";
        }
        o << "], \"choice\": {\"finder\": \"cpu\", \"flat\": \"" << backend_name(choose(0)) << "\", \"hr\": \"" << backend_name(choose(2))
          << "\", \"shade\": \"" << backend_name(choose(-1)) << "\"}}";
    } catch (const std::exception& e) {
        o.str("");
        o << "{\"ok\": false, \"error\": \"" << escape(e.what()) << "\", \"choice\": {\"finder\": \"cpu\", \"flat\": \"cpu\", \"hr\": \"cpu\", \"shade\": \"cpu\"}}";
    }
    progress(100, "Done");
    return env->NewStringUTF(o.str().c_str());
}

// ---- debug batch mode (ScanViewModel.runBatch): one photo file exactly as tools/desktop_check/pipeline_cli
// runs it with the nets in-process (`FS_NETS=1 FS_NET_DUMP=OUT pipeline_cli PHOTO OUT.ext net`, with
// shade: FS_SHADE_NET + FS_SHADE_AFTER; rules: FS_RULE_NET=1), on the groups' backends: the same JSON line (returned, and
// written to OUT.json), the page (OUT + ext: JPEG q85, or PNG) and the nets' outputs (OUT_logits.npz,
// _p1.npz, _rc0.npz, _rc2.npz), so tools/desktop_check/gate_b.py cmp and the DocUNet scorer read a phone's
// run like a desktop one. CPU image stages (use_gpu off), automatic corners, as pipeline_cli.
static std::string cli_num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

static std::string cli_strs(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + v[i] + "\"";
    return o + "]";
}

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_batchPhoto(JNIEnv* env, jclass, jstring photo_path, jstring out_base, jstring page_ext,
                                                                      jboolean shade, jboolean rules) {
    const std::string photo = jstr(env, photo_path), out = jstr(env, out_base), ext = jstr(env, page_ext);
    std::ostringstream f;
    try {
        cv::Mat img = cv::imread(photo, cv::IMREAD_COLOR);
        if (img.empty()) throw std::runtime_error("cannot read " + photo);
        std::ostringstream net_json;
        auto net_sep = [&]() { return net_json.tellp() > 0 ? ", " : ""; };
        // the page finder (pipeline_cli LOGITS "net")
        std::vector<float> lg;
        {
            auto g = group(g_finder);
            auto t0 = Clock::now();
            bool ok = page_finder_logits(img, *g->inf, lg);
            net_json << "\"page_finder\": {\"ok\": " << (ok ? "true" : "false") << ", \"backend\": \"" << backend_name(g->inf->backend(NetId::PageFinder))
                     << "\", \"ms\": " << cli_num(g->inf->last_ms(NetId::PageFinder)) << ", \"total_ms\": " << cli_num(ms_since(t0)) << "}";
            if (ok) write_npz(out + "_logits.npz", {{"logits", NpyArray::f32({2, (size_t)SEG_SIZE, (size_t)SEG_SIZE}, lg.data())}});
        }
        const float* logits = lg.empty() ? nullptr : lg.data();
        f << "{\"photo\": \"" << photo << "\", \"assess\": [";
        try {
            auto pages = assess_pages(img, logits, 0);
            for (size_t i = 0; i < pages.size(); ++i) {
                const auto& a = pages[i];
                f << (i ? ", " : "") << "{\"ok\": " << (a.pc.ok ? "true" : "false") << ", \"fit_median_px\": " << cli_num(a.fit_median_px)
                  << ", \"reasons\": " << cli_strs(a.reasons) << ", \"whole_frame\": " << (a.pc.whole_frame ? "true" : "false")
                  << ", \"n_pages\": " << a.pc.n_pages << ", \"corners\": [";
                for (int k = 0; k < 4; ++k) f << (k ? ", " : "") << cli_num(a.pc.pts[k].x) << ", " << cli_num(a.pc.pts[k].y);
                f << "]}";
            }
            f << "]";
        } catch (const std::exception& e) {
            f << "], \"assess_error\": \"" << e.what() << "\"";
        }
        PipelineOptions o;
        o.use_gpu = false;
        o.seg_logits = logits;
        o.zoom_pass = false;
        NetCall sr;
        if (shade) {
            sr.g = group(g_shade);
            o.shade_net = sr.fn();
            o.shade_after_uv = true;
        }
        NetCall rr;
        rr.id = NetId::RuleNet;
        if (rules) {
            rr.g = group(g_rule);
            o.rule_net = rr.fn();
        }
        // the flattening nets as the app runs them (FS_NETS=1): inside run_photo, beside its CPU stages,
        // on its detection's corners
        NetsRequest rq;
        rq.seg_logits = logits;
        NetsResult nets;
        auto flat_g = group(g_flat), hr_g = group(g_hr);
        o.flat_nets = flat_nets_fn(img, rq, *flat_g->inf, &nets, hr_g->inf.get());
        PipelineResult r;
        auto t_run = Clock::now();
        try {
            r = run_photo(img, o);
        } catch (const std::exception& e) {
            r.ok = false;
            r.error = std::string("exception: ") + e.what();
        }
        const double run_ms = ms_since(t_run);
        if (!r.ok && r.error == "no page found") {
            f << ", \"ok\": false, \"error\": \"no page found\"}";
        } else {
            for (const NetRun& n : nets.runs)
                net_json << net_sep() << "\"" << n.name << "\": {\"ok\": " << (n.ok ? "true" : "false") << ", \"backend\": \"" << backend_name(n.backend)
                         << "\", \"ms\": " << cli_num(n.ms) << ", \"total_ms\": " << cli_num(n.total_ms) << "}";
            net_json << net_sep() << "\"overlap\": {\"mode\": \"" << (r.nets_overlap == 1 ? "overlap" : r.nets_overlap == 0 ? "sequential" : "none")
                     << "\", \"orient_page_reused\": " << (r.orient_page_reused ? "true" : "false") << ", \"orient_reused\": " << (r.orient_reused ? "true" : "false")
                     << ", \"nets_ms\": " << cli_num(r.nets_ms) << ", \"wait_ms\": " << cli_num(r.nets_wait_ms) << ", \"join_ms\": " << cli_num(r.nets_join_ms) << ", \"stages\": [";
            for (size_t i = 0; i < r.ms.size(); ++i) net_json << (i ? ", " : "") << "[\"" << r.ms[i].first << "\", " << cli_num(r.ms[i].second) << "]";
            net_json << "]}";
            auto put = [&](const char* tag, const std::vector<float>& gr, const UvCrop& c) {
                if (gr.empty()) return;
                const UvSpec& sp = uv_spec(c.model);
                write_npz(out + tag, {{"out", NpyArray::f32({2, (size_t)sp.gh, (size_t)sp.gw}, gr.data())}});
            };
            put("_p1.npz", nets.grid_pass1, nets.crop_pass1);
            put(nets.crop1.model == 2 ? "_rc2.npz" : "_rc0.npz", nets.grid1, nets.crop1);
            put(nets.crop2.model == 2 ? "_rc2.npz" : "_rc0.npz", nets.grid2, nets.crop2);
            if (o.shade_net)
                net_json << net_sep() << "\"shadenet\": {\"ok\": " << (r.shade_net ? "true" : "false") << ", \"backend\": \""
                         << backend_name(sr.g->inf->backend(NetId::ShadeNet)) << "\", \"ms\": " << cli_num(sr.run_ms) << "}";
            if (o.rule_net && rr.calls > 0)
                net_json << net_sep() << "\"rulenet\": {\"ok\": " << (r.rule_net ? "true" : "false") << ", \"backend\": \""
                         << backend_name(rr.g->inf->backend(NetId::RuleNet)) << "\", \"ms\": " << cli_num(rr.run_ms) << ", \"calls\": " << rr.calls << "}";
            // the processing result: pipeline_cli's fields, in its order and format
            f << ", \"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << r.error << "\", \"mode\": \"" << geom_mode_name(r.mode)
              << "\", \"fit_median_full_px\": " << cli_num(r.detection.ok ? r.fit_median_px / r.detection.scale : -1)
              << ", \"needs_manual_crop\": " << cli_strs(r.needs_manual_crop) << ", \"rot\": " << r.rot << ", \"n_ctrl\": " << r.warp.n_ctrl
              << ", \"lines\": [" << r.warp.H.size() << ", " << r.warp.V.size() << "], \"text_rows\": " << r.warp.n_text
              << ", \"fitted_lines\": " << r.lines.size() << ", \"n_ignored\": " << r.warp.n_ignored << ", \"uv_agreement\": " << cli_num(r.uv_agreement)
              << ", \"uv_rejected\": " << (r.uv_rejected ? "true" : "false") << ", \"line_distortion\": " << cli_num(r.line_distortion)
              << ", \"lines_rejected\": " << (r.lines_rejected ? "true" : "false") << ", \"fingers_erased\": " << cli_num(r.fingers_erased)
              << ", \"uv_model\": " << r.uv_model << ", \"uv_consistency\": " << cli_num(r.uv_consistency)
              << ", \"uv_defines_page\": " << (r.uv_defines_page ? "true" : "false") << ", \"orient\": [" << cli_num(r.orient_sh) << ", "
              << cli_num(r.orient_sv) << ", " << cli_num(r.orient_upright) << "]" << ", \"uv_spread\": [" << cli_num(r.uv_spread[0]) << ", "
              << cli_num(r.uv_spread[1]) << "], \"shade_net\": " << (r.shade_net ? "true" : "false") << ", \"bowed\": [" << r.warp.S1.n_bowed
              << ", " << r.warp.S2.n_bowed << "], \"extended\": [" << r.warp.S1.n_extended << ", " << r.warp.S2.n_extended << "], \"second\": ["
              << cli_num(r.second_pass[0]) << ", " << cli_num(r.second_pass[1]) << ", " << cli_num(r.second_pass[2]) << ", " << (r.second_pass_kept ? 1 : 0)
              << "], \"corners\": [" << r.warp.S1.n_corners << ", " << r.warp.S2.n_corners << "], \"rule_net\": " << (r.rule_net ? "true" : "false")
              << ", \"net\": [" << r.warp.net.dropped << ", " << r.warp.net.added << ", " << r.warp.net.joined << ", " << r.warp.S2.n_net_kept
              << "], \"spots_dropped\": " << (r.lines_spots_dropped ? "true" : "false") << ", \"straightness\": [" << cli_num(r.line_straightness[0])
              << ", " << cli_num(r.line_straightness[1]) << "], \"margins\": " << r.warp.n_margins << ", \"edges\": [" << r.warp.n_edges_h << ", "
              << (r.warp.E_lines.size() - r.warp.n_edges_h) << ", " << r.warp.n_edge_bowed << "]" << ", \"downweighted\": " << r.warp.n_downweighted
              << ", \"bumps_faded\": " << cli_num(r.bumps_faded) << ", \"zoom\": [" << r.uv_zoom_quarters << ", " << cli_num(r.uv_zoom_spread[0])
              << ", " << cli_num(r.uv_zoom_spread[1]) << ", " << (r.uv_zoom ? 1 : 0) << "]";
            if (r.ok) {
                // warp-map field size and roughness, local area change (pipeline_cli field_stats, det_p99)
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
                    f << ", \"" << name << "_max\": " << cli_num(mx) << ", \"" << name << "_grad\": " << cli_num(gr);
                };
                field_stats("global", r.warp.GX, r.warp.GY);
                field_stats("local", r.warp.Fx, r.warp.Fy);
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
                    f << ", \"det_p99\": " << cli_num(dets[(size_t)(0.99 * (dets.size() - 1))]) << ", \"det_frac20\": " << cli_num((double)n20 / dets.size());
                }
            }
            f << ", \"nets\": {" << net_json.str() << "}, \"pipeline_ms\": " << cli_num(run_ms) << "}";
            if (r.ok) {
                cv::Mat bgr;
                cv::cvtColor(r.page, bgr, cv::COLOR_BGRA2BGR);
                cv::imwrite(out + ext, bgr, {cv::IMWRITE_JPEG_QUALITY, 85});
            }
        }
    } catch (const std::exception& e) {
        f.str("");
        f << "{\"photo\": \"" << photo << "\", \"ok\": false, \"error\": \"" << escape(e.what()) << "\"}";
    }
    std::ofstream(out + ".json") << f.str();
    return env->NewStringUTF(f.str().c_str());
}

}  // extern "C"
