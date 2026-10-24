// JNI bridge: Android Bitmaps (RGBA_8888) <-> the C++ core (BGR / BGRA cv::Mat).
// Implements app/src/main/java/com/albus/flatscan/NativeLib.kt exactly.
#include <android/asset_manager_jni.h>
#include "board.h"
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cleanup.h"
#include "screen.h"
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

cv::Mat read_bgra(const std::string& path);

// One processed page. Everything a re-render needs is computed once in process(): the page,
// its paper alpha, the before page, the deformation grid and the filter analysis (PageParams).
// The stage-1 page and the warp-map debug data are freed right after processing.
struct Session {
    PipelineResult result;    // page, paper, before, disp, maps, timings (stage1 / warp freed)
    PageParams pp;
    float legend_px = 1.f;
    std::mutex mu;            // guards the downscaled-source cache and the before page's loading
    std::string before_file;  // restored sessions: the before page, read on first use (before_page)
    std::vector<std::pair<cv::Size, cv::Mat>> small;   // paper-filled page, INTER_AREA, by size
    cv::Mat page_orig, paper_orig;   // the page before the Erase tool (kept from its first use)
    cv::Mat ai_mask, ai_page;        // the finger mask the AI fill last painted, and the page it made (setEdits)
    bool use_gpu = true;
    std::string accel;        // the nets' timings and backends: the members of info()'s "accel" object

    // Screen mode (screen.h): detected once per page (page_changed); the descreened page and its
    // params are made on first use and cached like the page's own
    ScreenInfo screen;
    bool scr_built = false;
    cv::Mat scr_page;
    PageParams scr_pp;
    std::vector<std::pair<cv::Size, cv::Mat>> scr_small;

    // after result.page and pp changed (process, erase, restore): screen detection (or `known`: made
    // beside the page params, or stored), caches dropped
    void page_changed(const ScreenInfo* known = nullptr) {
        ScreenInfo si = known ? *known : detect_screen(result.page);
        std::lock_guard<std::mutex> lk(mu);
        screen = si;
        scr_built = false;
        scr_page.release();
        scr_small.clear();
        small.clear();
    }
    // the before page (a restored session reads it now: opening a page does not wait for it); may be empty
    cv::Mat before_page() {
        std::lock_guard<std::mutex> lk(mu);
        if (!before_file.empty()) {
            cv::Mat b = read_bgra(before_file);
            before_file.clear();
            if (!b.empty() && b.size() == result.page.size()) result.before = b;
        }
        return result.before;
    }
    // Original always shows the page as photographed (and flattened)
    bool screen_on(const RenderParams& rp) const {
        if (rp.filter == Filter::Original) return false;
        return rp.screen == ScreenMode::On || (rp.screen == ScreenMode::Auto && screen.screen);
    }
    void ensure_screen() {   // called with mu held
        if (scr_built) return;
        scr_page = descreen(result.page);
        scr_pp = page_params(scr_page, use_gpu);
        scr_built = true;
    }
    // the page (and its params) a render with `scr` uses
    cv::Mat page_for(bool scr) {
        if (!scr) return result.page;
        std::lock_guard<std::mutex> lk(mu);
        ensure_screen();
        return scr_page;
    }
    PageParams pp_for(bool scr) {
        if (!scr) return pp;
        std::lock_guard<std::mutex> lk(mu);
        ensure_screen();
        return scr_pp;
    }

    // paper-filled page at `sz` (the full page itself when sz is the page size); scr: descreened
    cv::Mat filled_at(cv::Size sz, bool scr = false) {
        std::lock_guard<std::mutex> lk(mu);
        if (scr) ensure_screen();
        auto& cache = scr ? scr_small : small;
        for (auto& e : cache)
            if (e.first == sz) return e.second;
        cv::Mat full = apply_paper(scr ? scr_page : result.page, result.paper, scr ? scr_pp : pp), out;
        if (sz == full.size()) out = full;
        else cv::resize(full, out, sz, 0, 0, cv::INTER_AREA);
        if (cache.size() >= 3) cache.erase(cache.begin());
        cache.emplace_back(sz, out);
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
    cv::Mat page_unerased, paper_unerased, finger_mask;   // the finger brush's base (empty: no finger found)
    GridMap disp;
    std::vector<std::array<float, 5>> lines;
    float legend_px = 1.f;
    PageParams pp;            // the filters' page analysis and the screen detection: a restore does not redo them
    ScreenInfo screen;
};

// params.bin: the page params (scalars, then the quarter-grid Mats as rows, cols, type and data) and the
// screen detection, for page_w x page_h. Version 1.
constexpr int32_t kParamsMagic = 0x31505346;   // "FSP1"

bool write_params(const std::string& path, const PageParams& p, const ScreenInfo& sc) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const int32_t head[5] = {kParamsMagic, p.page_w, p.page_h, p.gpu_background ? 1 : 0, sc.screen ? 1 : 0};
    const float sc_f[7] = {p.wb[0], p.wb[1], p.wb[2], p.sigma, p.ink, sc.score, sc.score_y};
    bool ok = std::fwrite(head, sizeof head, 1, f) == 1 && std::fwrite(sc_f, sizeof sc_f, 1, f) == 1;
    for (const cv::Mat* m : {&p.illum, &p.paper, &p.bgv, &p.pic, &p.tint}) {
        const cv::Mat c = m->isContinuous() ? *m : m->clone();
        const int32_t mh[3] = {c.rows, c.cols, c.type()};
        ok = ok && std::fwrite(mh, sizeof mh, 1, f) == 1;
        const size_t n = c.total() * c.elemSize();
        ok = ok && (n == 0 || std::fwrite(c.data, 1, n, f) == n);
    }
    return std::fclose(f) == 0 && ok;
}

// false when missing, of another version, or not for a page_w x page_h page
bool read_params(const std::string& path, int page_w, int page_h, PageParams& p, ScreenInfo& sc) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t head[5];
    float sc_f[7];
    bool ok = std::fread(head, sizeof head, 1, f) == 1 && head[0] == kParamsMagic && head[1] == page_w && head[2] == page_h &&
              std::fread(sc_f, sizeof sc_f, 1, f) == 1;
    PageParams q;
    for (cv::Mat* m : {&q.illum, &q.paper, &q.bgv, &q.pic, &q.tint}) {
        int32_t mh[3];
        ok = ok && std::fread(mh, sizeof mh, 1, f) == 1 && mh[0] >= 0 && mh[1] >= 0 && mh[0] <= 4096 && mh[1] <= 4096;
        if (!ok) break;
        if (mh[0] == 0 || mh[1] == 0) continue;
        *m = cv::Mat(mh[0], mh[1], mh[2]);
        const size_t n = m->total() * m->elemSize();
        ok = std::fread(m->data, 1, n, f) == n;
    }
    std::fclose(f);
    if (!ok) return false;
    q.page_w = head[1]; q.page_h = head[2]; q.gpu_background = head[3] != 0;
    q.wb[0] = sc_f[0]; q.wb[1] = sc_f[1]; q.wb[2] = sc_f[2]; q.sigma = sc_f[3]; q.ink = sc_f[4];
    p = q;
    sc = ScreenInfo();
    sc.screen = head[4] != 0; sc.score = sc_f[5]; sc.score_y = sc_f[6];
    return true;
}

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
    if (!s.finger_mask.empty() && !s.page_unerased.empty()) {
        cv::cvtColor(s.page_unerased, bgr, cv::COLOR_BGRA2BGR);
        if (!cv::imwrite(dir + "/page_unerased.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 95})) return false;
        if (!cv::imwrite(dir + "/fingers.png", s.finger_mask, {cv::IMWRITE_PNG_COMPRESSION, 1})) return false;
        if (!s.paper_unerased.empty() && !cv::imwrite(dir + "/paper_unerased.png", s.paper_unerased, {cv::IMWRITE_PNG_COMPRESSION, 1})) return false;
    }
    const GridMap& g = s.disp;
    if (!write_floats(dir + "/disp.bin", {g.out_w, g.out_h, g.gw, g.gh, g.stride}, {&g.x, &g.y})) return false;
    std::vector<float> lf;
    for (const auto& l : s.lines) lf.insert(lf.end(), l.begin(), l.end());
    lf.push_back(s.legend_px);
    if (!write_floats(dir + "/lines.bin", {(int32_t)s.lines.size()}, {&lf})) return false;
    // optional (older sessions have none: the restore makes them)
    if (s.pp.page_w == s.page.cols && s.pp.page_h == s.page.rows) write_params(dir + "/params.bin", s.pp, s.screen);
    return true;
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

// RenderParams.toArray(): [filter, strength, solid, bgCleanup, black, white, gamma, screen]
// (screen: 0 off, 1 on, 2 auto; arrays without it are auto)
RenderParams parse_params(JNIEnv* env, jfloatArray a) {
    RenderParams rp;
    if (!a) return rp;
    jsize n = env->GetArrayLength(a);
    float v[8] = {(float)Filter::AutoColor, 0.7f, 0.f, 0.f, 0.f, 1.f, 1.f, (float)ScreenMode::Auto};
    env->GetFloatArrayRegion(a, 0, std::min<jsize>(n, 8), v);
    int f = (int)std::lround(v[0]);
    rp.filter = (Filter)std::min(kFilterCount - 1, std::max(0, f));
    rp.strength = std::isfinite(v[1]) ? std::min(1.f, std::max(0.f, v[1])) : 0.7f;
    rp.solid = v[2] > 0.5f;
    rp.adj.bg_cleanup = v[3];
    rp.adj.black = v[4];
    rp.adj.white = v[5];
    rp.adj.gamma = v[6];
    rp.adj = rp.adj.clamped();
    int sm = std::isfinite(v[7]) ? (int)std::lround(v[7]) : 2;
    rp.screen = (ScreenMode)std::min(2, std::max(0, sm));
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

// ---- the nets (core/src/nets.h on core/src/infer.h: LiteRT, CPU kernels and GPU delegate, its libraries
// and the .tflite models in the APK). One Inference per
// group of nets, so that Settings > Acceleration (or Auto's choice) can give each group its own
// backends: the page finder, UVDoc (both passes), SyntheticDoc high-res (its own group: its fp16 grid
// misses the accelerator tolerance where UVDoc's meets it, Gate A), the shadow net and the rule net (the
// GPU from its fixed-shape copies of the A4 stage-1 page, infer.cpp). An Inference keeps its sessions for the
// life of the process and moves a net to its next backend (CPU last) on any failure, so a net that
// failed on an accelerator stays on the CPU until the next configureNets with other backends.
struct NetGroup {
    std::unique_ptr<Inference> inf;
    std::string pref;             // backend names, comma separated ("" or "cpu": CPU only)
    std::mutex mu;
    std::set<int> used;           // nets that have run on this Inference (the first run creates the session)
    bool first_use(NetId n) {
        std::lock_guard<std::mutex> lk(mu);
        return used.insert((int)n).second;
    }
};

std::mutex g_nets_mu;             // guards everything below
std::shared_ptr<NetGroup> g_finder, g_flat, g_hr, g_shade, g_rule;
std::shared_ptr<NetGroup> g_inpaint;   // the finger AI fill (MI-GAN): CPU only, its own Inference
jobject g_assets = nullptr;       // global reference: keeps the AAssetManager alive
AAssetManager* g_mgr = nullptr;
std::string g_cache_dir;          // compiled-model cache (the LiteRT GPU's serialised kernels)
int g_threads = 0;
bool g_static_models = false;     // the CPU loads the fixed-shape UVDoc / SyntheticDoc copies (HUMAN GATE A-1)

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

// assets/models/<file> in place: the APK's uncompressed asset (noCompress tflite), mapped while the view lives;
// no data (missing, or the static-model .onnx names): load_model copies it instead
ModelView map_model(const std::string& file) {
    AAssetManager* mgr;
    {
        std::lock_guard<std::mutex> lk(g_nets_mu);
        mgr = g_mgr;
    }
    ModelView v;
    if (!mgr) return v;
    AAsset* a = AAssetManager_open(mgr, ("models/" + file).c_str(), AASSET_MODE_BUFFER);
    if (!a) return v;
    const void* p = AAsset_getBuffer(a);
    const off_t n = AAsset_getLength(a);
    if (!p || n <= 0 || (uintptr_t)p % 4 != 0) {   // TFLite wants a 4-byte aligned buffer
        AAsset_close(a);
        return v;
    }
    v.data = (const uint8_t*)p;
    v.size = (size_t)n;
    v.keep = std::shared_ptr<const void>(a, [](const void* x) { AAsset_close((AAsset*)x); });
    return v;
}

// caller holds g_nets_mu
std::shared_ptr<NetGroup> make_group(const std::string& pref) {
    auto g = std::make_shared<NetGroup>();
    g->pref = pref;
    InferConfig c;
    c.load = load_model;
    c.map = map_model;
    if (!parse_backends(pref, c.preference)) {
        LOGI("nets: unknown backend in \"%s\": CPU only", pref.c_str());
        c.preference.clear();
    }
    c.cache_dir = g_cache_dir;
    c.threads = g_threads;
    g->inf = Inference::create(c);
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

// ---- the flattening nets ahead of processing (precomputeNets): while the crop screen is open. With
// automatic corners they depend only on the photo, the page finder's output, the page of a spread and
// the nets setting (a first UVDoc pass on the whole photo finds the page; the detection's corners only
// where that pass fails, and the detection is the same function of the photo). process() takes them
// when it runs with the same inputs on the same net groups (configureNets makes new ones), so the page
// is the same; otherwise they are cancelled and process runs its own.
uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, b + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    for (; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

uint64_t nets_key(const cv::Mat& bgr, const std::vector<float>& logits, int page_index, int flat_nets) {
    uint64_t h = 1469598103934665603ull;
    const int head[4] = {bgr.cols, bgr.rows, page_index, flat_nets};
    h = fnv(h, head, sizeof head);
    for (int y = 0; y < bgr.rows; ++y) h = fnv(h, bgr.ptr(y), bgr.cols * bgr.elemSize());
    return fnv(h, logits.data(), logits.size() * sizeof(float));
}

// their grids, and whether every net that was to run ran (false: cancelled, process then runs its own)
struct AheadResult {
    NetsResult nets;
    bool complete = false;
};

struct AheadNets {
    uint64_t key = 0;
    std::shared_ptr<NetGroup> flat, hr;   // the groups they run on
    cv::Mat bgr;                          // the inputs, until run_ahead takes them
    std::vector<float> logits;
    int page_index = 0, nets = 0;
    bool started = false;
    std::shared_ptr<std::promise<AheadResult>> done;
    std::shared_future<AheadResult> result;
    std::shared_ptr<std::atomic<bool>> cancel;
};
std::mutex g_ahead_mu;
AheadNets g_ahead;

// caller holds g_ahead_mu. A set that never started is answered at once (incomplete); one that runs stops
// before its next net and answers then.
void drop_ahead() {
    if (g_ahead.cancel) *g_ahead.cancel = true;
    if (g_ahead.done && !g_ahead.started) g_ahead.done->set_value(AheadResult());
    g_ahead = AheadNets();
}

// Runs the registered set on the calling thread; returns when its nets are done or cancelled (at once if
// there is none, or it runs or ran elsewhere).
void run_ahead() {
    AheadNets a;
    {
        std::lock_guard<std::mutex> lk(g_ahead_mu);
        if (!g_ahead.done || g_ahead.started) return;
        g_ahead.started = true;
        a = g_ahead;
        g_ahead.bgr.release();
        g_ahead.logits.clear();
    }
    AheadResult res;
    try {
        NetsRequest rq;
        rq.nets = (FlatNets)a.nets;
        rq.seg_logits = a.logits.empty() ? nullptr : a.logits.data();
        rq.page_index = a.page_index;
        rq.cancel = a.cancel.get();
        // as run_photo's: the detection's corners, asked for only without a first-pass grid
        PageCorners pc;
        bool detected = false;
        rq.corners_fn = [&]() -> const std::array<cv::Point2d, 4>* {
            if (!detected) {
                pc = detect_corners(a.bgr, PipelineOptions().max_analysis_side, rq.seg_logits, a.page_index);
                detected = true;
            }
            return pc.ok ? &pc.pts : nullptr;
        };
        res.nets = run_flat_nets(a.bgr, rq, *a.flat->inf, a.hr->inf.get());
        res.complete = !*a.cancel;
    } catch (const std::exception& e) {
        LOGI("flattening nets ahead failed: %s", e.what());
        res = AheadResult();
    }
    a.done->set_value(std::move(res));
}

}  // namespace

extern "C" {

// The flattening nets for this photo with automatic corners (see AheadNets), in two steps: this call
// registers them (from here on a process() of the same inputs waits for them instead of running its own),
// runPrecompute runs them. Replaces (and cancels) an earlier set.
JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_precomputeNets(JNIEnv* env, jclass, jobject bmp, jfloatArray logits,
                                                                        jint page_index, jint flat_nets) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    std::vector<float> lg = read_logits(env, logits);
    std::lock_guard<std::mutex> lk(g_ahead_mu);
    drop_ahead();
    if (bgr.empty()) return;
    AheadNets& a = g_ahead;
    a.nets = std::min(2, std::max(0, (int)flat_nets));
    a.key = nets_key(bgr, lg, page_index, a.nets);
    a.flat = group(g_flat);
    a.hr = group(g_hr);
    a.bgr = bgr;
    a.logits = std::move(lg);
    a.page_index = page_index;
    a.cancel = std::make_shared<std::atomic<bool>>(false);
    a.done = std::make_shared<std::promise<AheadResult>>();
    a.result = a.done->get_future().share();
}

// Runs the registered set on the calling thread (run_ahead).
JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_runPrecompute(JNIEnv*, jclass) { run_ahead(); }

JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_cancelPrecompute(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lk(g_ahead_mu);
    drop_ahead();
}

// Whiteboard mode: the board or projector screen in the photo (board.cpp find_board). Returns
// [x0, y0, ... x3, y3 (TL, TR, BR, BL, photo px), confidence, area share, dark (1: a dark slide)] or null
// when no reliable quadrilateral is found (the app then offers the whole frame for manual corners).
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_detectBoard(JNIEnv* env, jclass, jobject bmp) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return nullptr;
    BoardQuad q = find_board(bgr);
    if (!q.ok) return nullptr;
    bool dark = board_is_dark(bgr, q);
    float out[11];
    for (int i = 0; i < 4; ++i) { out[2 * i] = (float)q.pts[i].x; out[2 * i + 1] = (float)q.pts[i].y; }
    out[8] = (float)q.confidence;
    out[9] = (float)q.area;
    out[10] = dark ? 1.f : 0.f;
    jfloatArray a = env->NewFloatArray(11);
    env->SetFloatArrayRegion(a, 0, 11, out);
    return a;
}

// Export (feature 3): a one-channel (grayscale) JPEG of an RGBA_8888 bitmap at [quality]; Android's own
// encoder writes only three-channel JPEGs. Luma as in the BT.601 conversion of cv::cvtColor. Null on failure.
JNIEXPORT jbyteArray JNICALL Java_com_albus_flatscan_NativeLib_encodeGrayJpeg(JNIEnv* env, jclass, jobject bitmap, jint quality) {
    AndroidBitmapInfo info;
    void* px = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &px) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;
    std::vector<uchar> buf;
    bool ok = false;
    try {
        cv::Mat rgba((int)info.height, (int)info.width, CV_8UC4, px, info.stride), gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        ok = cv::imencode(".jpg", gray, buf, {cv::IMWRITE_JPEG_QUALITY, (int)quality});
    } catch (const std::exception& e) {
        LOGI("encodeGrayJpeg failed: %s", e.what());
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    if (!ok) return nullptr;
    jbyteArray out = env->NewByteArray((jsize)buf.size());
    env->SetByteArrayRegion(out, 0, (jsize)buf.size(), reinterpret_cast<const jbyte*>(buf.data()));
    return out;
}

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_gpuInfo(JNIEnv* env, jclass) {
    bool ok = Gpu::instance().available();
    std::string s = (ok ? "GPU (OpenGL ES 3.1 compute): " : "CPU fallback (GPU unavailable): ") + Gpu::instance().info();
    return env->NewStringUTF(s.c_str());
}

// Settings > Acceleration: the backends of each net group (finder, flat: UVDoc, hr: SyntheticDoc
// high-res, shade, rule: the rule net; names as nets.h parse_backends, comma separated; "" = CPU only), the compiled-model cache (LiteRT GPU kernels), CPU threads (0: the default).
// staticModels: the CPU loads uvdoc_static / syntheticdoc_hr_static instead of the originals (HUMAN
// GATE A-1). A group whose settings are unchanged keeps its Inference and sessions.
JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_configureNets(JNIEnv* env, jclass, jobject assets, jstring cache_dir, jstring finder,
                                                                       jstring flat, jstring hr, jstring shade, jstring rule, jint threads,
                                                                       jboolean static_models) {
    std::string cd = cache_dir ? jstr(env, cache_dir) : "", pf = finder ? jstr(env, finder) : "", fl = flat ? jstr(env, flat) : "",
                hs = hr ? jstr(env, hr) : "", sh = shade ? jstr(env, shade) : "", rl = rule ? jstr(env, rule) : "";
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
        if (!slot || slot->pref != pref || !same) slot = make_group(pref);
    };
    set(g_finder, pf);
    set(g_flat, fl);
    set(g_hr, hs);
    set(g_shade, sh);
    set(g_rule, rl);
}

// JSON: {"cpu" (the CPU runtime and version), "litert" (LiteRT's version), "gpu" (the LiteRT GPU delegate can be
// tried: its library loads and the device has an OpenCL library), "gpu_why", "with_ort", "ort" (version),
// "providers": [the execution providers of the ONNX Runtime build, if any]}
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_runtimeInfo(JNIEnv* env, jclass) {
    std::ostringstream f;
    const RuntimeInfo r = runtime_info();
    f << "{\"cpu\": \"" << escape(r.cpu) << "\", \"litert\": \"" << escape(r.litert) << "\", \"gpu\": " << (r.gpu ? "true" : "false")
      << ", \"gpu_why\": \"" << escape(r.gpu_why) << "\", ";
#ifdef FS_WITH_ORT
    try {
        std::vector<std::string> p = Ort::GetAvailableProviders();
        f << "\"with_ort\": true, \"ort\": \"" << escape(Ort::GetVersionString()) << "\", \"providers\": " << json_strings(p) << "}";
    } catch (const std::exception& e) {
        f << "\"with_ort\": true, \"ort\": \"unknown\", \"providers\": [], \"error\": \"" << escape(e.what()) << "\"}";
    }
#else
    f << "\"with_ort\": false, \"ort\": \"none\", \"providers\": []}";
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
                                                                  jboolean board, jboolean remove_shadows, jfloatArray user_lines, jobject listener) {
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
    // the line editor's drawn lines: [group, n, x0, y0, ... x(n-1), y(n-1)] repeated, output-page fractions
    if (user_lines) {
        jsize n = env->GetArrayLength(user_lines);
        std::vector<float> v(n);
        if (n) env->GetFloatArrayRegion(user_lines, 0, n, v.data());
        size_t i = 0;
        while (i + 2 <= v.size()) {
            PipelineOptions::UserLine l;
            l.group = (int)std::lround(v[i]);
            size_t np = (size_t)std::max(0.f, v[i + 1]);
            i += 2;
            if (i + 2 * np > v.size()) break;
            for (size_t k = 0; k < np; ++k) l.pts.emplace_back(v[i + 2 * k], v[i + 2 * k + 1]);
            i += 2 * np;
            if (l.pts.size() >= 2) o.user_lines.push_back(std::move(l));
        }
    }
    // the page's Remove shadows switch: off, neither the shading map nor the shadow net
    o.remove_shadows = remove_shadows;
    o.auto_orient = auto_orient;
    // whiteboard mode: a board is not held by fingers (and skin-coloured drawings must stay)
    if (board) o.erase_fingers = false;
    o.keep_unerased = true;   // the finger brush (setEdits) starts from the page before the fill
    o.render_w = render_w;
    o.use_gpu = use_gpu;
    Progress progress(env, listener);
    // the flattening nets, run by run_photo beside its CPU stages (PipelineOptions::flat_nets, Gate D:
    // the same outputs as before them): with automatic corners a first UVDoc pass on the whole photo,
    // then UVDoc and SyntheticDoc high-res on the page it found (the high-res grid in the first slot).
    // They run on a worker thread, which must not touch JNI: their entries are made after run_photo.
    std::ostringstream acc;   // the "accel" nets entries (info)
    NetsResult nets;
    bool nets_ahead = false;   // the grids came from precomputeNets
    std::shared_ptr<NetGroup> flat_g, hr_g;
    if (flatten && !perspective_only) {
        flat_g = group(g_flat);
        hr_g = group(g_hr);
        NetsRequest rq;
        rq.corners_edited = o.corners_edited;
        rq.nets = (FlatNets)std::min(2, std::max(0, (int)flat_nets));
        rq.seg_logits = o.seg_logits;
        rq.page_index = page_index;
        // the nets started on the crop screen, if they ran on this photo with these inputs (precomputeNets)
        std::shared_future<AheadResult> ahead;
        {
            std::lock_guard<std::mutex> lk(g_ahead_mu);
            if (g_ahead.result.valid()) {
                if (!o.corners_edited && g_ahead.flat == flat_g && g_ahead.hr == hr_g && g_ahead.key == nets_key(bgr, lg, page_index, (int)rq.nets))
                    ahead = g_ahead.result;
                else
                    drop_ahead();
            }
        }
        if (ahead.valid()) {
            nets_ahead = true;
            FlatNetsFn own = flat_nets_fn(bgr, rq, *flat_g->inf, &nets, hr_g->inf.get());
            o.flat_nets = [ahead, own, &nets, &nets_ahead](const NetCornersFn& corners) {
                run_ahead();   // not started yet: here (this set, or a replaced one, already answered)
                const AheadResult& a = ahead.get();
                if (!a.complete) {   // cancelled before it ran every net: this run's own
                    nets_ahead = false;
                    return own(corners);
                }
                nets = a.nets;
                FlatNetGrids g;
                g.grid1 = nets.grid1;
                g.grid2 = nets.grid2;
                g.grid_pass1 = nets.grid_pass1;
                g.crop1 = nets.crop1;
                g.crop2 = nets.crop2;
                g.crop_pass1 = nets.crop_pass1;
                return g;
            };
        } else {
            o.flat_nets = flat_nets_fn(bgr, rq, *flat_g->inf, &nets, hr_g->inf.get());
        }
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
            // the page params (GPU) and the screen detection (CPU) side by side
            auto t0 = Clock::now();
            double ms_screen = 0;
            std::future<ScreenInfo> scr = std::async(std::launch::async, [&r, &ms_screen] {
                auto ts = Clock::now();
                ScreenInfo si = detect_screen(r.page);
                ms_screen = ms_since(ts);
                return si;
            });
            s->pp = page_params(r.page, use_gpu);
            r.ms.emplace_back(s->pp.gpu_background ? "page params (GPU)" : "page params (CPU)", ms_since(t0));
            const ScreenInfo si = scr.get();
            s->page_changed(&si);
            r.ms.emplace_back("screen detection, beside the page params", ms_screen);
            r.ms.emplace_back("page params + screen detection", ms_since(t0));
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
      << ", \"nets_wait_ms\": " << std::lround(std::max(0.0, r.nets_wait_ms)) << ", \"nets_ahead\": " << (nets_ahead ? "true" : "false")
      << ", \"process_ms\": " << std::lround(ms_since(t1));
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
      << ", \"fingers_erased\": " << json_num(r.fingers_erased) << ", \"fingers_over_text\": " << (r.fingers_over_text ? "true" : "false") << ", \"user_lines_used\": " << r.user_lines_used << ", \"uv_model\": " << r.uv_model
      << ", \"uv_spread\": [" << json_num(r.uv_spread[0]) << ", " << json_num(r.uv_spread[1]) << "]"
      << ", \"shadows_removed\": " << (r.shadows_removed ? "true" : "false") << ", \"shade_net\": " << (r.shade_net ? "true" : "false")
      << ", \"screen\": {\"detected\": " << (s->screen.screen ? "true" : "false") << ", \"score\": " << json_num(s->screen.score)
      << ", \"threshold\": " << json_num(kScreenThreshold) << "}"
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
        const bool scr = s->screen_on(rp);
        page = render_page(s->page_for(scr), s->result.paper, s->pp_for(scr), rp, prot, use_gpu, &st);
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
        const bool scr = s->screen_on(rp);
        cv::Mat src = s->filled_at(bs, scr);
        cv::Mat t = filter_cpu(src, s->pp_for(scr), rp.filter, rp.strength, rp.adj);
        bgra_to_bitmap(env, t, out);
    } catch (const std::exception& e) {
        LOGI("renderThumb failed: %s", e.what());
    }
}

JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_renderBefore(JNIEnv* env, jclass, jlong h, jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return;
    try {
        cv::Mat b = s->before_page();
        bgra_to_bitmap(env, b.empty() ? s->result.page : b, out);
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
            const bool scr = s->screen_on(rp);
            cv::Mat src = s->filled_at(cv::Size(std::max(1, pg.cols / 4), std::max(1, pg.rows / 4)), scr);
            hist = luma_histogram(filter_cpu(src, s->pp_for(scr), rp.filter, rp.strength, rp.adj));
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
// Finger brush strokes: [mode (1 add, 0 remove), radius (page-width fraction), n, x0, y0, ... x(n-1), y(n-1)]
// repeated, points as page fractions; drawn into [mask] (255 add, 0 remove) in order. Modes 2 (add) and 3
// (remove): the points outline an area (smart select), filled; the radius is not used.
void draw_strokes(JNIEnv* env, jfloatArray strokes, cv::Mat& mask) {
    if (!strokes) return;
    jsize n = env->GetArrayLength(strokes);
    std::vector<float> v(n);
    if (n) env->GetFloatArrayRegion(strokes, 0, n, v.data());
    size_t i = 0;
    while (i + 3 <= v.size()) {
        const int mode = (int)std::lround(v[i]);
        const bool add = mode == 1 || mode == 2;
        int th = std::max(1, (int)std::lround(2 * v[i + 1] * mask.cols));
        size_t np = (size_t)std::max(0.f, v[i + 2]);
        i += 3;
        if (i + 2 * np > v.size()) break;
        cv::Scalar c(add ? 255 : 0);
        if (mode >= 2) {
            std::vector<cv::Point> poly;
            for (size_t k = 0; k < np; ++k) poly.emplace_back((int)std::lround(v[i + 2 * k] * mask.cols), (int)std::lround(v[i + 2 * k + 1] * mask.rows));
            if (poly.size() >= 3) cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, c, cv::LINE_8);
            i += 2 * np;
            continue;
        }
        for (size_t k = 0; k < np; ++k) {
            cv::Point p((int)std::lround(v[i + 2 * k] * mask.cols), (int)std::lround(v[i + 2 * k + 1] * mask.rows));
            cv::circle(mask, p, th / 2, c, cv::FILLED);
            if (k + 1 < np) {
                cv::Point q((int)std::lround(v[i + 2 * k + 2] * mask.cols), (int)std::lround(v[i + 2 * k + 3] * mask.rows));
                cv::line(mask, p, q, c, th);
            }
        }
        i += 2 * np;
    }
}

// The finger mask a render uses: the automatic one (unless [auto_on] is false) with the brush strokes.
cv::Mat finger_mask_for(JNIEnv* env, const PipelineResult& r, cv::Size size, bool auto_on, jfloatArray strokes) {
    cv::Mat m = auto_on && !r.finger_mask.empty() && r.finger_mask.size() == size ? r.finger_mask.clone() : cv::Mat::zeros(size, CV_8U);
    draw_strokes(env, strokes, m);
    return m;
}

// The page's edits after processing: the finger mask (the automatic one when [fingers_auto], plus brush
// [strokes]) filled with paper (fill_finger) - [fill] 1: then painted in by the inpainting net (inpaint_fill) -,
// then the Erase tool's rectangles [rects] (page fractions).
// Starts from the page before any finger fill, so removing a stroke or turning the automatic mask off
// brings the photo's pixels back.
JNIEXPORT jboolean JNICALL Java_com_albus_flatscan_NativeLib_setEdits(JNIEnv* env, jclass, jlong h, jfloatArray rects, jboolean fingers_auto, jfloatArray strokes,
                                                                   jint fill) {
    Session* s = session(h);
    if (!s || !s->result.ok) return JNI_FALSE;
    std::vector<Rect01> rs = parse_protect(env, rects);
    PipelineResult& r = s->result;
    try {
        if (s->page_orig.empty()) {
            s->page_orig = r.page.clone();
            s->paper_orig = r.paper.clone();
        }
        const bool has_base = !r.page_unerased.empty();
        r.page = (has_base ? r.page_unerased : s->page_orig).clone();
        r.paper = (has_base ? r.paper_unerased : s->paper_orig).clone();
        if (has_base || (strokes && env->GetArrayLength(strokes) > 0)) {
            cv::Mat fm = finger_mask_for(env, r, r.page.size(), fingers_auto, strokes);
            if (cv::countNonZero(fm) > 0) {
                fill_finger(r.page, fm, r.paper.empty() ? nullptr : &r.paper);
                // fill 1: the AI fill over the paper fill (the same mask again: its last result)
                if (fill == 1) {
                    cv::Mat diff;
                    if (!s->ai_mask.empty() && s->ai_mask.size() == fm.size() && s->ai_page.size() == r.page.size()) cv::bitwise_xor(fm, s->ai_mask, diff);
                    if (!diff.empty() && cv::countNonZero(diff) == 0) {
                        s->ai_page.copyTo(r.page);
                    } else {
                        auto g = group(g_inpaint);
                        if (inpaint_fill(r.page, fm, *g->inf, r.paper_unerased.empty() ? nullptr : &r.paper_unerased)) {
                            s->ai_mask = fm.clone();
                            s->ai_page = r.page.clone();
                        }
                    }
                }
            }
        }
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
        s->page_changed();   // screen detection; drops the downscaled and descreened caches
    } catch (const std::exception& e) {
        LOGI("setEdits failed: %s", e.what());
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

// The finger brush's view: [out] (page size, ARGB) cleared, the mask a render with these settings fills
// painted translucent orange. Returns the share of the page in the mask.
JNIEXPORT jfloat JNICALL Java_com_albus_flatscan_NativeLib_fingerOverlay(JNIEnv* env, jclass, jlong h, jboolean fingers_auto, jfloatArray strokes, jobject out) {
    Session* s = session(h);
    if (!s || !s->result.ok) return 0.f;
    AndroidBitmapInfo info;
    void* px = nullptr;
    if (AndroidBitmap_getInfo(env, out, &info) != ANDROID_BITMAP_RESULT_SUCCESS || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return 0.f;
    cv::Mat m = finger_mask_for(env, s->result, s->result.page.size(), fingers_auto, strokes);
    if (m.cols != (int)info.width || m.rows != (int)info.height) cv::resize(m, m, cv::Size(info.width, info.height), 0, 0, cv::INTER_NEAREST);
    if (AndroidBitmap_lockPixels(env, out, &px) != ANDROID_BITMAP_RESULT_SUCCESS) return 0.f;
    cv::Mat o((int)info.height, (int)info.width, CV_8UC4, px, info.stride);
    o.setTo(cv::Scalar(0, 0, 0, 0));
    o.setTo(cv::Scalar(255, 140, 40, 120), m);   // RGBA memory order
    AndroidBitmap_unlockPixels(env, out);
    return (float)cv::countNonZero(m) / std::max(1, m.cols * m.rows);
}

// The finger tool's smart select: the area around page point (fx, fy) (fractions) that belongs with it - a
// finger, a thumb - on the page before any finger fill: GrabCut in a box around the tap, the tap itself
// certain, the colour-alike area flood-filled from it likely, the rest of the box unsure. The piece joined
// to the tap, holes filled, 2 px wider (the finger's soft edge). Returns its outline as page fractions
// [x0, y0, x1, y1, ...]; empty when nothing was found or it would cover more than a quarter of the page
// (a tap on blank paper).
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_fingerSelect(JNIEnv* env, jclass, jlong h, jfloat fx, jfloat fy) {
    std::vector<float> out;
    Session* s = session(h);
    if (s && s->result.ok) {
        try {
            const PipelineResult& r = s->result;
            const cv::Mat& src = !r.page_unerased.empty() ? r.page_unerased : (!s->page_orig.empty() ? s->page_orig : r.page);
            cv::Mat bgr;
            if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
            else bgr = src.clone();
            const double k = std::min(1.0, 700.0 / std::max(bgr.cols, bgr.rows));
            if (k < 1) cv::resize(bgr, bgr, cv::Size(), k, k, cv::INTER_AREA);
            const int W = bgr.cols, H = bgr.rows;
            cv::Point seed(std::min(W - 1, std::max(0, (int)(fx * W))), std::min(H - 1, std::max(0, (int)(fy * H))));
            cv::Mat lab, smooth;
            cv::GaussianBlur(bgr, smooth, cv::Size(5, 5), 0);
            cv::cvtColor(smooth, lab, cv::COLOR_BGR2Lab);
            cv::Mat ff = cv::Mat::zeros(H + 2, W + 2, CV_8U);
            cv::floodFill(lab, ff, seed, cv::Scalar(), nullptr, cv::Scalar(5, 4, 4), cv::Scalar(5, 4, 4), 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8));
            cv::Mat alike = ff(cv::Rect(1, 1, W, H)).clone();
            if (cv::countNonZero(alike) > 0.25 * W * H) return env->NewFloatArray(0);   // blank paper
            const int half = std::max(W, H) / 7;
            cv::Rect box = cv::Rect(seed.x - half, seed.y - half, 2 * half, 2 * half) | cv::boundingRect(alike);
            box &= cv::Rect(0, 0, W, H);
            cv::Mat gc(H, W, CV_8U, cv::Scalar(cv::GC_BGD));
            gc(box).setTo(cv::GC_PR_BGD);
            gc.setTo(cv::GC_PR_FGD, alike);
            cv::circle(gc, seed, std::max(2, W / 120), cv::Scalar(cv::GC_FGD), cv::FILLED);
            cv::Mat bgd, fgd;
            cv::grabCut(bgr, gc, box, bgd, fgd, 4, cv::GC_INIT_WITH_MASK);
            cv::Mat fg = (gc == cv::GC_FGD) | (gc == cv::GC_PR_FGD);
            cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            cv::Mat lbl;
            cv::connectedComponents(fg, lbl, 8, CV_32S);
            const int keep = lbl.at<int>(seed);
            if (keep == 0) return env->NewFloatArray(0);
            fg = lbl == keep;
            std::vector<std::vector<cv::Point>> cs;
            cv::findContours(fg, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (cs.empty()) return env->NewFloatArray(0);
            cv::Mat filled = cv::Mat::zeros(H, W, CV_8U);
            cv::drawContours(filled, cs, -1, cv::Scalar(255), cv::FILLED);
            cv::dilate(filled, filled, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            if (cv::countNonZero(filled) > 0.25 * W * H) return env->NewFloatArray(0);
            cs.clear();
            cv::findContours(filled, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            size_t big = 0;
            for (size_t i = 1; i < cs.size(); ++i)
                if (cv::contourArea(cs[i]) > cv::contourArea(cs[big])) big = i;
            std::vector<cv::Point> poly;
            cv::approxPolyDP(cs[big], poly, 1.0, true);
            for (const auto& q : poly) { out.push_back((q.x + 0.5f) / W); out.push_back((q.y + 0.5f) / H); }
        } catch (const std::exception& e) {
            LOGI("fingerSelect failed: %s", e.what());
            out.clear();
        }
    }
    jfloatArray a = env->NewFloatArray((jsize)out.size());
    if (!out.empty()) env->SetFloatArrayRegion(a, 0, (jsize)out.size(), out.data());
    return a;
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
    sn->before = s->before_page();
    sn->page_unerased = s->result.page_unerased;
    sn->paper_unerased = s->result.paper_unerased;
    sn->finger_mask = s->result.finger_mask;
    sn->disp = s->result.disp;
    sn->lines = s->result.lines;
    sn->legend_px = s->legend_px;
    sn->pp = s->pp;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        sn->screen = s->screen;
    }
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
    const auto t_all = Clock::now();
    std::string path = jstr(env, dir);
    auto s = std::make_unique<Session>();
    s->use_gpu = use_gpu;
    PipelineResult& r = s->result;
    try {
        r.page = read_bgra(path + "/page.jpg");
        if (r.page.empty()) return 0;
        if (std::ifstream(path + "/paper.png").good()) r.paper = cv::imread(path + "/paper.png", cv::IMREAD_GRAYSCALE);
        if (!r.paper.empty() && r.paper.size() != r.page.size()) r.paper.release();
        // the before page: on first use (compare, the next snapshot)
        if (std::ifstream(path + "/before.jpg").good()) s->before_file = path + "/before.jpg";
        if (std::ifstream(path + "/fingers.png").good()) {
            r.finger_mask = cv::imread(path + "/fingers.png", cv::IMREAD_GRAYSCALE);
            r.page_unerased = read_bgra(path + "/page_unerased.jpg");
            if (std::ifstream(path + "/paper_unerased.png").good()) r.paper_unerased = cv::imread(path + "/paper_unerased.png", cv::IMREAD_GRAYSCALE);
            if (r.finger_mask.size() != r.page.size() || r.page_unerased.size() != r.page.size()) {
                r.finger_mask.release(); r.page_unerased.release(); r.paper_unerased.release();
            }
            if (!r.paper_unerased.empty() && r.paper_unerased.size() != r.page.size()) r.paper_unerased.release();
        }
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
        ScreenInfo si;
        if (read_params(path + "/params.bin", r.page.cols, r.page.rows, s->pp, si)) {
            s->page_changed(&si);
            r.ms.emplace_back("restored; page params and screen detection read", ms_since(t0));
        } else {
            s->pp = page_params(r.page, use_gpu);
            r.ms.emplace_back("restored; page params", ms_since(t0));
            s->page_changed();
        }
        r.ok = true;
        LOGI("restoreSession: %s %.0f ms, all %.0f ms", r.ms.back().first.c_str(), r.ms.back().second, ms_since(t_all));
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
// makes them; the shadow net gets the photo at the A4 page's net size, 352 x 512; the rule net the page
// (the photo through the page finder's corners) at the A4 stage-1 page's half size, 800 x 1152). A backend is
// acceptable for a group of nets (UVDoc's two passes; SyntheticDoc high-res; the shadow net; the rule net) when each
// net of the group ran on it (no fallback to the next backend) within the accelerator tolerances below;
// Auto takes the fastest acceptable backend per group, the CPU when none is faster. The page finder stays on the CPU (PLAN: 34 NNAPI partitions).
// Accelerator tolerances (the CPU path itself stays bit-identical):
//   flattening nets: Gate A's grid measure (grid_px), p99 <= 0.25 px (Gate A's target) and max <= 1 px
//                    of the 1600 x 2263 analysis page;
//   shadow net: |shading difference| x 255, p99 <= 0.5 and max <= 2 levels (Gate A's fp16 stand-in: p99 0.07);
//   rule net: |rule probability difference|, p99 <= 0.01 and max <= 0.1 (the line stage reads it against
//             thresholds of 0.3 - 0.5: a pixel's few-percent move changes no rule).
constexpr double AUTO_GRID_P99_PX = 0.25, AUTO_GRID_MAX_PX = 1.0, AUTO_SHADE_P99 = 0.5, AUTO_SHADE_MAX = 2.0, AUTO_RULE_P99 = 0.01,
                 AUTO_RULE_MAX = 0.1;
constexpr int BENCH_SHADE = -1, BENCH_RULE = -2;   // In::model of the shadow net's and the rule net's inputs

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
            return Inference::create(c);
        };
        // Memory: only one backend's sessions at a time. The app's own sessions go first (each group keeps
        // its backends and makes its sessions again on its next run), and the CPU's before the GPU's: with
        // the CPU and GPU copies of every net (SyntheticDoc high-res at 1440 x 1024) and the app's rule net
        // all alive, an 8 GB phone killed the app (low memory) during this benchmark.
        {
            std::lock_guard<std::mutex> lk(g_nets_mu);
            for (std::shared_ptr<NetGroup>* s : {&g_finder, &g_flat, &g_hr, &g_shade, &g_rule})
                if (*s) *s = make_group((*s)->pref);
        }
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
            in.push_back({"shadenet", NetId::ShadeNet, BENCH_SHADE, std::move(t)});
        }
        {
            // the rule net's GPU shape (1152 x 800, infer.cpp kRuleGpuHw): the page through the found corners, else the photo
            const int rw = 800, rh = 1152;
            cv::Mat pg;
            if (pc.ok) {
                std::vector<cv::Point2f> src, dst = {{0, 0}, {(float)rw - 1, 0}, {(float)rw - 1, (float)rh - 1}, {0, (float)rh - 1}};
                for (const auto& q : pc.pts) src.emplace_back((float)q.x, (float)q.y);
                cv::warpPerspective(bgr, pg, cv::getPerspectiveTransform(src, dst), cv::Size(rw, rh), cv::INTER_AREA, cv::BORDER_REPLICATE);
            } else {
                cv::resize(bgr, pg, cv::Size(rw, rh), 0, 0, cv::INTER_AREA);
            }
            Tensor t{{1, 3, rh, rw}, std::vector<float>((size_t)3 * rw * rh)};
            for (int y = 0; y < rh; ++y)
                for (int x = 0; x < rw; ++x)
                    for (int c = 0; c < 3; ++c) t.data[((size_t)c * rh + y) * rw + x] = pg.at<cv::Vec3b>(y, x)[c] / 255.f;
            in.push_back({"rulenet", NetId::RuleNet, BENCH_RULE, std::move(t)});
        }
        // every backend on every input: a first run (session creation; the GPU: its kernels' compile or
        // the cached ones) and `runs` timed runs. The CPU (the reference, its outputs are what the GPU is
        // compared with) skips the first run of the nets the inputs' step ran (their sessions exist) and
        // times one run: it only has to show whether the GPU is faster (SyntheticDoc high-res: 2 CPU runs
        // instead of 5)
        struct Res { bool ok = false, ran = false, pass = false; double first_ms = 0, median = 0, p95 = 0, p99 = 0, max = 0; std::vector<float> out; };
        std::vector<std::vector<Res>> res(all.size(), std::vector<Res>(in.size()));
        for (size_t bi = 0; bi < all.size(); ++bi) {
            const bool on_cpu = all[bi] == Backend::CPU;
            std::unique_ptr<Inference> own;
            Inference* inf = cpu.get();
            if (!on_cpu) {
                cpu.reset();   // the CPU's outputs are kept (res[0]): its sessions can go
                inf = (own = make(all[bi])).get();
            }
            for (size_t k = 0; k < in.size(); ++k) {
                progress((int)(5 + 90 * (bi * in.size() + k) / (all.size() * in.size())),
                         std::string("Auto: ") + in[k].key + " on " + backend_name(all[bi]));
                Res& r = res[bi][k];
                std::vector<Tensor> y;
                r.ok = true;
                if (!on_cpu || in[k].model < 0) {
                    auto t0 = Clock::now();
                    r.ok = inf->run(in[k].id, in[k].x, y) && !y.empty();
                    r.first_ms = ms_since(t0);
                    r.ran = r.ok && inf->backend(in[k].id) == all[bi];
                    if (!r.ran) continue;
                }
                std::vector<double> t;
                for (int i = 0; i < (on_cpu ? 1 : runs) && r.ok; ++i) {
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
                    const double unit = in[k].model == BENCH_SHADE ? 255 : 1;   // the shadow net in levels, the rule net in probability
                    for (size_t i = 0; i < r.out.size(); ++i) d.push_back(std::fabs((double)r.out[i] - c.out[i]) * unit);
                } else {
                    const UvSpec& sp = uv_spec(in[k].model);
                    d = grid_px(c.out, r.out, sp.gw, sp.gh);
                }
                if (d.empty()) continue;
                r.p99 = pct(d, 99);
                r.max = *std::max_element(d.begin(), d.end());
                r.pass = in[k].model == BENCH_SHADE  ? r.p99 <= AUTO_SHADE_P99 && r.max <= AUTO_SHADE_MAX
                         : in[k].model == BENCH_RULE ? r.p99 <= AUTO_RULE_P99 && r.max <= AUTO_RULE_MAX
                                                     : r.p99 <= AUTO_GRID_P99_PX && r.max <= AUTO_GRID_MAX_PX;
            }
        // the fastest acceptable backend per group (the CPU's own runs are the reference: acceptable)
        // model: 0 UVDoc, 2 SyntheticDoc high-res, BENCH_SHADE the shadow net, BENCH_RULE the rule net
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
          << ", \"shade_p99_levels\": " << AUTO_SHADE_P99 << ", \"shade_max_levels\": " << AUTO_SHADE_MAX << ", \"rule_p99\": " << AUTO_RULE_P99
          << ", \"rule_max\": " << AUTO_RULE_MAX << "}, \"backends\": [";
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
          << "\", \"shade\": \"" << backend_name(choose(BENCH_SHADE)) << "\", \"rule\": \"" << backend_name(choose(BENCH_RULE)) << "\"}}";
    } catch (const std::exception& e) {
        o.str("");
        o << "{\"ok\": false, \"error\": \"" << escape(e.what()) << "\", \"choice\": {\"finder\": \"cpu\", \"flat\": \"cpu\", \"hr\": \"cpu\", \"shade\": \"cpu\", \"rule\": \"cpu\"}}";
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
              << ", \"lines\": [" << r.warp.H.size() << ", " << r.warp.V.size() << "], \"text_rows\": " << r.warp.n_text << ", \"text_rows_v\": " << r.warp.n_text_v << ", \"text_net\": " << (r.warp.text_net ? "true" : "false")
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
