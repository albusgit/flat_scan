// OpenGL ES 3.1 compute backend: map expansion + bicubic resample, and the cleanup pass.
// One EGL context with a 1x1 pbuffer (or surfaceless), made current only for the duration of
// each call, so any worker thread can use it. Capabilities are queried at runtime; callers
// fall back to the CPU path when available() is false or a call returns false.
#pragma once
#include <opencv2/core.hpp>
#include <mutex>
#include <string>

#include "resample.h"

namespace fs {

struct CleanParams;

class Gpu {
public:
    static Gpu& instance();
    bool available();                 // lazily initialises EGL/GLES on first use
    const std::string& info() const { return info_; }
    int max_texture_size() const { return max_tex_; }

    // Same output as resample_cpu (byte-exact on the parity path). src: CV_8UC4.
    bool resample(const cv::Mat& src_rgba, const GridMap& g, cv::Mat& out_rgba);
    // Parity/debug: dense maps as reconstructed on the GPU (float32).
    bool expand_maps(const GridMap& g, cv::Mat& mx, cv::Mat& my);
    // Cleanup steps 1 and 4 (cleanup.h); same bytes as background_quarter_cpu / tone_cpu
    // up to float rounding in the tone mapping.
    bool background_quarter(const cv::Mat& img_rgba, cv::Mat& bg4_rgba);
    bool tone(const cv::Mat& img_rgba, const CleanParams& p, cv::Mat& out_rgba);

private:
    Gpu() = default;
    bool init_locked();
    bool make_current();
    void release();
    unsigned program(const char* name, const char* src);
    bool init_cleanup_programs();
    bool run_banded(unsigned prog, int w, int h, cv::Mat& out);

    std::mutex mu_;
    bool tried_ = false, ok_ = false;
    std::string info_;
    int max_tex_ = 0;
    void* dpy_ = nullptr;
    void* ctx_ = nullptr;
    void* surf_ = nullptr;
    unsigned prog_resample_ = 0, prog_expand_ = 0, prog_down_ = 0, prog_dilate_ = 0, prog_median_ = 0, prog_count_ = 0, prog_tone_ = 0;
};

}  // namespace fs
