// Desktop check tools only: no EGL / GLES here, so every GPU call reports "unavailable" and the
// core takes its CPU paths (the same fallback the app uses when the GPU is missing).
#include "gpu.h"

namespace fs {

Gpu& Gpu::instance() {
    static Gpu g;
    return g;
}
bool Gpu::available() {
    info_ = "no GPU (desktop check build)";
    return false;
}
bool Gpu::resample(const cv::Mat&, const GridMap&, cv::Mat&) { return false; }
bool Gpu::expand_maps(const GridMap&, cv::Mat&, cv::Mat&) { return false; }
bool Gpu::background_quarter(const cv::Mat&, cv::Mat&) { return false; }
bool Gpu::filter(const cv::Mat&, const PageParams&, Filter, float, const Adjust&, cv::Mat&) { return false; }

}  // namespace fs
