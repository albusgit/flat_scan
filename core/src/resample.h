// Backward maps on a coarse grid and the single final resample.
//
// Contract (PHONE_MILESTONE.md): maps hold float32 source-pixel coordinates, top-left
// origin, integer pixel centres. Sample (i, j) of the grid is the map value at output pixel
// (j*stride, i*stride). Dense maps are rebuilt by bilinear interpolation with clamp to the
// last grid sample (cv2.remap(..., BORDER_REPLICATE) with x/stride, y/stride, as in
// warpmap.load_maps). The image is sampled with OpenCV's INTER_CUBIC semantics: positions
// quantised to 1/32 px, 15-bit fixed-point weights, constant white border.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace fs {

struct GridMap {
    int out_w = 0, out_h = 0;   // dense output size
    int gw = 0, gh = 0;         // grid samples per row / column
    int stride = 4;
    std::vector<float> x, y;    // gh*gw, row major

    static GridMap make(int out_w, int out_h, int stride);
    float& X(int i, int j) { return x[(size_t)i * gw + j]; }
    float& Y(int i, int j) { return y[(size_t)i * gw + j]; }
};

// Dense rows [y0, y1) of the expanded map (CV_32FC1 each).
void expand_rows(const GridMap& g, int y0, int y1, cv::Mat& mx, cv::Mat& my);
// Full dense maps (host harness / parity export only; the app never calls this).
void expand_full(const GridMap& g, cv::Mat& mx, cv::Mat& my);

// The same backward map for a page rendered at another size (output resolution setting):
// output pixel p' maps to analysis pixel (p' + 0.5) * out_w / new_w - 0.5, interpolated
// bilinearly on the grid. Geometry is unchanged; the photo is still sampled once.
GridMap rescale_grid(const GridMap& g, int new_w, int new_h);

// CPU path: cv::remap INTER_CUBIC in horizontal bands, so no full-resolution float map is
// ever allocated. src is CV_8UC3 or CV_8UC4.
cv::Mat resample_cpu(const cv::Mat& src, const GridMap& g, int band_rows = 64);

// OpenCV's BicubicTab_i (1024 x 16 shorts), recomputed with x86 float semantics.
const int16_t* bicubic_itab();
// Scalar model of the GPU kernel (same integer arithmetic as the shader). Used to prove the
// table and border semantics against cv::remap on the host.
cv::Mat resample_model(const cv::Mat& src, const GridMap& g);

// ---- GPU paths of the other bicubic image stages (Gpu, GLES 3.1 compute). Each returns false,
// leaving `out` unspecified, when the GPU is unavailable or a call fails; the caller then runs its
// CPU code unchanged. On success `out` is byte-identical to that CPU code: cv::remap INTER_CUBIC,
// BORDER_CONSTANT white, through the same float32 map. src: CV_8UC3 or CV_8UC4; out: src's type.

// Whether a GPU stage can take this source and output size (cheap; no GPU work). Lets callers
// skip CPU-side preparation for a GPU call that would be refused.
bool gpu_stage_ready(const cv::Mat& src, int out_w, int out_h);

// Which of these stages use the GPU when the caller allows it (process-wide). Outputs are the same
// either way; this is speed only. Defaults from docs/results/accel/gpu_stages.txt (desktop RTX 5060
// through Mesa d3d12; not measured on a phone): the dense stage-1 / orientation path is off (the
// CPU still evaluates the exact per-pixel unroll map, so only the bicubic moves and the transfers
// cost more: stage-1 page 21-35 ms CPU vs 66-79 ms GPU); the global remap is on (single-threaded
// on the CPU: global_remap+darkness 82-104 ms CPU vs 52-59 ms GPU).
struct GpuStagePolicy {
    bool dense = false;    // remap_cubic_gpu / remap_dense_gpu
    bool offset = true;    // remap_offset_gpu
};
void set_gpu_stage_policy(const GpuStagePolicy& p);
GpuStagePolicy gpu_stage_policy();

// Dense maps (CV_32F, output size), e.g. made on the CPU.
bool remap_dense_gpu(const cv::Mat& src, const cv::Mat& mx, const cv::Mat& my, cv::Mat& out);

// A w x h output whose map at (x, y) is at(x, y) (cv::Point2f), evaluated on the CPU in parallel
// row bands exactly as the CPU path evaluates it; only the bicubic runs on the GPU. The stage-1
// and orientation pages (pipeline.cpp: remap_unroll).
template <class At>
bool remap_cubic_gpu(const cv::Mat& src, int w, int h, const At& at, cv::Mat& out) {
    if (!gpu_stage_policy().dense || !gpu_stage_ready(src, w, h)) return false;
    cv::Mat mx(h, w, CV_32F), my(h, w, CV_32F);
    const int band = 32, nb = (h + band - 1) / band;
    int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            for (int b = t; b < nb; b += nt)
                for (int y = b * band; y < std::min(h, (b + 1) * band); ++y) {
                    float* px = mx.ptr<float>(y);
                    float* py = my.ptr<float>(y);
                    for (int x = 0; x < w; ++x) {
                        cv::Point2f p = at(x, y);
                        px[x] = p.x;
                        py[x] = p.y;
                    }
                }
        });
    for (auto& t : ts) t.join();
    return remap_dense_gpu(src, mx, my, out);
}

// The global stage's A_img (warpmap.cpp): map(x, y) = (x + GX, y + GY), the offsets interpolated
// bilinearly on their stride-px grid of gw x gh samples (row major) without clamping. Only on a
// context with `precise` (Gpu::precise): the map is float arithmetic the driver must not fuse.
bool remap_offset_gpu(const cv::Mat& src, const std::vector<float>& gx, const std::vector<float>& gy, int gw, int gh, int stride,
                      cv::Mat& out);

// GPU image stages that sit inside CPU code without a use_gpu argument of their own (the global
// stage's remap in warpmap.cpp) run on the GPU only while a GpuStageScope(true) is alive on the
// calling thread: run_photo / run_scan open one from PipelineOptions::use_gpu. Off by default.
class GpuStageScope {
public:
    explicit GpuStageScope(bool on);
    ~GpuStageScope();
    GpuStageScope(const GpuStageScope&) = delete;
    GpuStageScope& operator=(const GpuStageScope&) = delete;
private:
    bool prev_;
};
bool gpu_stages_enabled();

// Calls of remap_dense_gpu / remap_offset_gpu that ran on the GPU, process-wide (harness checks).
struct GpuStageCounts { long dense = 0, offset = 0; };
GpuStageCounts gpu_stage_counts();

}  // namespace fs
