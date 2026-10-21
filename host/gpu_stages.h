// GPU image stages against their CPU counterparts (gpu_stages.cpp).
#pragma once
#include <opencv2/core.hpp>
#include <random>

namespace fs {

// The stage-1 / orientation dense-map remap and the global stage's offset remap on one fixture
// (src BGRA, dense maps of the output size), GPU vs CPU. Returns the worst |diff|.
int gpu_stage_fixtures(const cv::Mat& src_bgra, const cv::Mat& mx, const cv::Mat& my, std::mt19937& rng);
// gpu-stages OUT.json PHOTO... [--reps N] [--default-policy] [--logits P] [--uvpass1 P] [--uvout P] [--uvout2 P]
int cmd_gpu_stages(int argc, char** argv);
// cg-tol OUT.txt RTOL PHOTO... [--reps N] [nets as gpu-stages]: the warp map's CG tolerance, report only
int cmd_cg_tol(int argc, char** argv);
// gpu-bench PHOTO [--reps N]: each bicubic stage alone, CPU vs GPU, median ms
int cmd_gpu_bench(int argc, char** argv);

}  // namespace fs
