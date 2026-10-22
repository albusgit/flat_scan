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

struct PageParams;
struct Adjust;
enum class Filter : int;

class Gpu {
public:
    static Gpu& instance();
    bool available();                 // lazily initialises EGL/GLES on first use
    const std::string& info() const { return info_; }
    int max_texture_size() const { return max_tex_; }
    // the map arithmetic is compiled `precise` (no FMA contraction): GLSL ES 3.2 or gpu_shader5.
    // Without it a driver may fuse it, and the float-map stages (resample, remap_offset) can then
    // differ from the CPU by a 1/32-px rounding step; remap_dense has no such arithmetic.
    bool precise() const { return precise_; }
    // CPU-side time of the last bicubic call (any thread): texture / buffer uploads, then dispatch
    // + readback (which waits for the GPU). For the harness's timing table.
    struct Times { double upload_ms = 0, run_ms = 0; };
    Times last_times() const { return times_; }

    // Same output as resample_cpu (byte-exact on the parity path). src: CV_8UC4.
    bool resample(const cv::Mat& src_rgba, const GridMap& g, cv::Mat& out_rgba);
    // The same bicubic for the other image stages (resample.h: remap_dense_gpu, remap_offset_gpu),
    // byte-exact against cv::remap(INTER_CUBIC, BORDER_CONSTANT white). src: CV_8UC4.
    //   remap_offset: map(x, y) = (x, y) + the bilinear offset field `off` (warpmap's global stage)
    //   remap_dense:  dense float32 maps from the CPU (the stage-1 / orientation unroll map)
    bool remap_offset(const cv::Mat& src_rgba, const GridMap& off, cv::Mat& out_rgba);
    bool remap_dense(const cv::Mat& src_rgba, const cv::Mat& mx, const cv::Mat& my, cv::Mat& out_rgba);
    // Parity/debug: dense maps as reconstructed on the GPU (float32).
    bool expand_maps(const GridMap& g, cv::Mat& mx, cv::Mat& my);
    // Filter passes (cleanup.h): the 1/4-resolution background (same bytes as
    // background_quarter_cpu) and the filter (same as filter_cpu up to float rounding).
    bool background_quarter(const cv::Mat& img_rgba, cv::Mat& bg4_rgba);
    bool filter(const cv::Mat& img_rgba, const PageParams& p, Filter f, float strength, const Adjust& adj, cv::Mat& out_rgba);

private:
    Gpu() = default;
    bool init_locked();
    bool make_current();
    void release();
    unsigned program(const char* name, const char* src);
    bool init_cleanup_programs();
    bool run_banded(unsigned prog, int w, int h, cv::Mat& out);
    bool run_grid(unsigned prog, const cv::Mat& src, const GridMap& g, cv::Mat& out);

    std::mutex mu_;
    Times times_;
    bool tried_ = false, ok_ = false, precise_ = false;
    std::string info_;
    int max_tex_ = 0;
    void* dpy_ = nullptr;
    void* ctx_ = nullptr;
    void* surf_ = nullptr;
    unsigned prog_resample_ = 0, prog_expand_ = 0, prog_offset_ = 0, prog_dense_ = 0, prog_down_ = 0, prog_dilate_ = 0, prog_median_ = 0, prog_count_ = 0, prog_tone_ = 0;
};

}  // namespace fs
