// JNI bridge: Android Bitmaps (RGBA_8888) <-> the C++ core (BGR / BGRA cv::Mat).
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include "cleanup.h"
#include "gpu.h"
#include "log.h"
#include "pipeline.h"
#include "npz.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <vector>

using namespace fs;

namespace {

struct Session {
    PipelineResult result;
};

// Locks a bitmap and wraps it (no copy). Caller must unlock.
bool lock(JNIEnv* env, jobject bmp, cv::Mat& m) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return false;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return false;
    void* px = nullptr;
    if (AndroidBitmap_lockPixels(env, bmp, &px) != ANDROID_BITMAP_RESULT_SUCCESS) return false;
    m = cv::Mat((int)info.height, (int)info.width, CV_8UC4, px, info.stride);
    return true;
}

cv::Mat bitmap_to_bgr(JNIEnv* env, jobject bmp) {
    cv::Mat rgba, bgr;
    if (!lock(env, bmp, rgba)) return bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bmp);
    return bgr;
}

bool bgra_to_bitmap(JNIEnv* env, const cv::Mat& bgra, jobject bmp) {
    cv::Mat rgba;
    if (!lock(env, bmp, rgba)) return false;
    bool ok = rgba.rows == bgra.rows && rgba.cols == bgra.cols;
    if (ok) cv::cvtColor(bgra, rgba, cv::COLOR_BGRA2RGBA);
    AndroidBitmap_unlockPixels(env, bmp);
    return ok;
}

std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_gpuInfo(JNIEnv* env, jclass) {
    bool ok = Gpu::instance().available();
    std::string s = (ok ? "GPU (OpenGL ES 3.1 compute): " : "CPU fallback (GPU unavailable): ") + Gpu::instance().info();
    return env->NewStringUTF(s.c_str());
}

// -> float[8] corners TL, TR, BR, BL in bitmap pixels, or null if no page was found
JNIEXPORT jfloatArray JNICALL Java_com_albus_flatscan_NativeLib_detect(JNIEnv* env, jclass, jobject bmp) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return nullptr;
    PageCorners pc;
    try {
        pc = detect_corners(bgr);
    } catch (const std::exception& e) {
        LOGI("detect failed: %s", e.what());
        return nullptr;
    }
    if (!pc.ok) return nullptr;
    jfloat v[8];
    for (int k = 0; k < 4; ++k) { v[2 * k] = (jfloat)pc.pts[k].x; v[2 * k + 1] = (jfloat)pc.pts[k].y; }
    jfloatArray a = env->NewFloatArray(8);
    env->SetFloatArrayRegion(a, 0, 8, v);
    return a;
}

JNIEXPORT jlong JNICALL Java_com_albus_flatscan_NativeLib_process(JNIEnv* env, jclass, jobject bmp, jfloatArray corners,
                                                                  jboolean edited, jfloat focal_px, jboolean auto_orient,
                                                                  jint render_w, jboolean use_gpu, jobject listener) {
    cv::Mat bgr = bitmap_to_bgr(env, bmp);
    if (bgr.empty()) return 0;
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
    o.auto_orient = auto_orient;
    o.render_w = render_w;
    o.use_gpu = use_gpu;
    jclass lc = listener ? env->GetObjectClass(listener) : nullptr;
    jmethodID mid = lc ? env->GetMethodID(lc, "onProgress", "(ILjava/lang/String;)V") : nullptr;
    o.progress = [&](int pct, const char* stage) {
        if (!mid) return;
        jstring js = env->NewStringUTF(stage);
        env->CallVoidMethod(listener, mid, (jint)pct, js);
        env->DeleteLocalRef(js);
    };
    auto s = std::make_unique<Session>();
    try {
        s->result = run_photo(bgr, o);
    } catch (const std::exception& e) {
        s->result.ok = false;
        s->result.error = e.what();
    }
    if (!s->result.ok) LOGI("pipeline failed: %s", s->result.error.c_str());
    return (jlong)s.release();
}

JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_info(JNIEnv* env, jclass, jlong h) {
    const PipelineResult& r = reinterpret_cast<Session*>(h)->result;
    std::ostringstream f;
    f << "{\"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << escape(r.error) << "\", \"mode\": \""
      << (r.mode == GeomMode::Surface ? "surface" : "homography") << "\", \"fell_back\": " << (r.fell_back ? "true" : "false")
      << ", \"fit_median_px\": " << r.fit_median_px << ", \"rot\": " << r.rot << ", \"gpu_resample\": " << (r.gpu_resample ? "true" : "false")
      << ", \"ms\": {";
    for (size_t i = 0; i < r.ms.size(); ++i) f << (i ? ", " : "") << "\"" << escape(r.ms[i].first) << "\": " << (long)r.ms[i].second;
    f << "}}";
    return env->NewStringUTF(f.str().c_str());
}

JNIEXPORT jintArray JNICALL Java_com_albus_flatscan_NativeLib_pageSize(JNIEnv* env, jclass, jlong h) {
    const PipelineResult& r = reinterpret_cast<Session*>(h)->result;
    jint v[2] = {r.page.cols, r.page.rows};
    jintArray a = env->NewIntArray(2);
    env->SetIntArrayRegion(a, 0, 2, v);
    return a;
}

// cleanup into `out` (pageSize); raw = true copies the unwarped page without cleanup
JNIEXPORT jstring JNICALL Java_com_albus_flatscan_NativeLib_render(JNIEnv* env, jclass, jlong h, jboolean gray, jboolean solid,
                                                                  jboolean raw, jboolean use_gpu, jobject out) {
    const PipelineResult& r = reinterpret_cast<Session*>(h)->result;
    auto t0 = std::chrono::steady_clock::now();
    CleanStats st;
    cv::Mat page;
    try {
        page = raw ? r.page : finish_page(r.page, gray, solid, use_gpu, &st);
    } catch (const std::exception& e) {
        return env->NewStringUTF((std::string("{\"error\": \"") + escape(e.what()) + "\"}").c_str());
    }
    bool ok = bgra_to_bitmap(env, page, out);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::ostringstream f;
    f << "{\"ok\": " << (ok ? "true" : "false") << ", \"cleanup_ms\": " << (long)ms << ", \"gpu_background\": " << st.gpu_background
      << ", \"gpu_tone\": " << st.gpu_tone << "}";
    return env->NewStringUTF(f.str().c_str());
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
    jclass lc = listener ? env->GetObjectClass(listener) : nullptr;
    jmethodID mid = lc ? env->GetMethodID(lc, "onProgress", "(ILjava/lang/String;)V") : nullptr;
    auto progress = [&](int p, const std::string& s) {
        if (!mid) return;
        jstring js = env->NewStringUTF(s.c_str());
        env->CallVoidMethod(listener, mid, (jint)p, js);
        env->DeleteLocalRef(js);
    };
    std::ostringstream o;
    try {
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
        CleanStats sg, sc;
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat cg = finish_page(r.page, false, false, true, &sg);
        double ms_cg = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        cv::Mat cc = finish_page(r.page, false, false, false, &sc);
        double ms_cc = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int dclean = max_abs_diff(cg, cc, &c1);
        o << "{\"ok\": true, \"runs\": " << runs << ", \"gpu\": \"" << escape(Gpu::instance().info()) << "\""
          << ", \"gpu_resample\": " << (r.gpu_resample ? "true" : "false")
          << ", \"map_parity\": {\"shape_ok\": " << (shape_ok ? "true" : "false") << ", \"max_epe_px\": " << max_epe
          << ", \"mean_epe_px\": " << (n ? sum / n : 0) << ", \"passed\": " << (shape_ok && max_epe <= 0.1 ? "true" : "false") << "}"
          << ", \"gpu_vs_cpu_resample\": {\"max_levels\": " << dres << ", \"values_gt1\": " << g1 << "}"
          << ", \"gpu_vs_cpu_cleanup\": {\"max_levels\": " << dclean << ", \"values_gt1\": " << c1 << ", \"gpu_ms\": " << (long)ms_cg
          << ", \"cpu_ms\": " << (long)ms_cc << ", \"gpu_used\": " << (sg.gpu_tone ? "true" : "false") << "}"
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

JNIEXPORT void JNICALL Java_com_albus_flatscan_NativeLib_release(JNIEnv*, jclass, jlong h) { delete reinterpret_cast<Session*>(h); }

}  // extern "C"
