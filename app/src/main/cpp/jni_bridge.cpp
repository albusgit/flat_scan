// JNI bridge: Android Bitmaps (RGBA_8888) <-> the C++ core (BGR / BGRA cv::Mat).
// Implements app/src/main/java/com/albus/flatscan/NativeLib.kt exactly.
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "cleanup.h"
#include "gpu.h"
#include "log.h"
#include "npz.h"
#include "overlay.h"
#include "pagefind.h"
#include "pipeline.h"
#include "uvdoc.h"

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
    c.model = v[5] == 1 ? 1 : 0;
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

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_gpuInfo(JNIEnv* env, jclass) {
    bool ok = Gpu::instance().available();
    std::string s = (ok ? "GPU (OpenGL ES 3.1 compute): " : "CPU fallback (GPU unavailable): ") + Gpu::instance().info();
    return env->NewStringUTF(s.c_str());
}

// segnet.to_input of the whole photo: float[3 * 256 * 256] (NCHW, ImageNet-normalised RGB), or null
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_segInput(JNIEnv* env, jclass, jobject bmp) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return nullptr;
    std::vector<float> x = seg_input(bgr);
    jfloatArray a = env->NewFloatArray((jsize)x.size());
    env->SetFloatArrayRegion(a, 0, (jsize)x.size(), x.data());
    return a;
}

// -> JSON {"ok", "error", "finder", ..., "pages": [{"corners": [x0, y0, ... x3, y3] (bitmap px, TL TR BR BL),
//    "ok", "fit_median_px", "needs_manual_crop": [reasons], finder fields}, ...]} (2 pages: book spread)
// flattening net (0 UVDoc, 1 SyntheticDoc): the part of the photo it sees for these corners
// -> [x, y, w, h, turned, model]
JNIEXPORT jintArray JNICALL Java_com_albus_flatscan_NativeLib_uvCrop(JNIEnv* env, jclass, jint w, jint h, jfloatArray corners,
                                                                    jint model) {
    std::array<cv::Point2d, 4> c;
    jfloat v[8] = {0};
    if (corners && env->GetArrayLength(corners) == 8) env->GetFloatArrayRegion(corners, 0, 8, v);
    for (int k = 0; k < 4; ++k) c[k] = cv::Point2d(v[2 * k], v[2 * k + 1]);
    UvCrop u = uv_crop(cv::Size(w, h), c, model);
    jint o[6] = {u.box.x, u.box.y, u.box.width, u.box.height, u.turned ? 1 : 0, u.model};
    jintArray a = env->NewIntArray(6);
    env->SetIntArrayRegion(a, 0, 6, o);
    return a;
}

// flattening-net input for a crop: float[3 * in_h * in_w] (RGB 0..1, NCHW), or null. With the page finder's
// logits, the other page of a book spread (not pageIndex) is hidden from the net.
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_uvInput(JNIEnv* env, jclass, jobject bmp, jintArray crop,
                                                                       jfloatArray logits, jint page_index) {
    UvCrop c;
    if (!read_uv_crop(env, crop, c)) return nullptr;
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return nullptr;
    cv::Mat blank;
    std::vector<float> lg = read_logits(env, logits);
    if (!lg.empty()) {
        try {
            PageFindResult f = find_page_net(bgr, lg.data());
            if (f.pages.size() == 2) blank = f.pages[page_index == 1 ? 0 : 1].mask;
        } catch (const std::exception& e) {
            LOGI("uvInput: page finder failed: %s", e.what());
        }
    }
    std::vector<float> x = uv_input(bgr, c, blank);
    jfloatArray a = env->NewFloatArray((jsize)x.size());
    env->SetFloatArrayRegion(a, 0, (jsize)x.size(), x.data());
    return a;
}

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

JNIEXPORT jlong JNICALL Java_com_albus_flatscan_NativeLib_process(JNIEnv* env, jclass, jobject bmp, jfloatArray corners,
                                                                  jboolean edited, jfloat focal_px, jboolean auto_orient,
                                                                  jint render_w, jboolean use_gpu, jfloatArray logits,
                                                                  jint page_index, jboolean force_homography, jboolean perspective_only, jfloatArray ignore,
                                                                  jfloatArray uv_out, jintArray uv_crop_a, jfloatArray uv_out2,
                                                                  jintArray uv_crop2_a, jfloatArray uv_shading, jobject listener) {
    auto s = std::make_unique<Session>();
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
    // flattening nets: up to two grids (the first preferred) and SyntheticDoc's shading map
    std::vector<float> uv, uv2, shading;
    UvCrop c1, c2;
    bool has1 = read_uv_crop(env, uv_crop_a, c1), has2 = read_uv_crop(env, uv_crop2_a, c2);
    if (has1) uv = read_floats_exact(env, uv_out, uv_spec(c1.model).output_len());
    if (has2) uv2 = read_floats_exact(env, uv_out2, uv_spec(c2.model).output_len());
    if (uv.empty() && !uv2.empty()) { std::swap(uv, uv2); std::swap(c1, c2); }
    if (!uv.empty()) { o.uv_out = uv.data(); o.uv_crop = c1; }
    if (!uv2.empty()) { o.uv_out2 = uv2.data(); o.uv_crop2 = c2; }
    for (const UvCrop* c : {&c1, &c2}) {
        if (c->model != 1 || !shading.empty()) continue;
        if ((c == &c1 && !has1) || (c == &c2 && !has2)) continue;
        shading = read_floats_exact(env, uv_shading, uv_spec(1).input_len());
        if (!shading.empty()) { o.uv_shading = shading.data(); o.uv_shading_crop = *c; }
    }
    for (const Rect01& q : parse_protect(env, ignore)) o.ignore.push_back({q.x0, q.y0, q.x1, q.y1});
    o.auto_orient = auto_orient;
    o.render_w = render_w;
    o.use_gpu = use_gpu;
    Progress progress(env, listener);
    o.progress = [&](int pct, const char* stage) { progress(pct, stage); };
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
      << ", \"shadows_removed\": " << (r.shadows_removed ? "true" : "false");
    if (r.detection.ok) f << ", " << finder_json(r.detection);
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

}  // extern "C"
