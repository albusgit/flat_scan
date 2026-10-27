// The self-test's map-parity reference (app/src/debug/assets/selftest/photo1_reference_warpmap.npz, debug builds only): the
// final map of run_photo with PipelineOptions defaults (no nets), as NativeLib.selfTest makes it.
//   selftest_ref PHOTO OUT.npz   (PHOTO: app/src/debug/assets/selftest/photo1.webp)
// Keys: map_x, map_y (gh x gw float32, photo px), stride, shape (out_h, out_w). Rebuild it whenever the
// core's map changes on purpose; the self-test's limit is 0.1 px.
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

#include "npz.h"
#include "pipeline.h"

using namespace fs;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: selftest_ref PHOTO OUT.npz\n"); return 2; }
    cv::Mat bgr = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (bgr.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    PipelineOptions opt;
    PipelineResult r = run_photo(bgr, opt);
    if (!r.ok) { std::fprintf(stderr, "run_photo: %s\n", r.error.c_str()); return 1; }
    const GridMap& m = r.final_map;
    const std::vector<size_t> sh = {(size_t)m.gh, (size_t)m.gw};
    const int64_t shape[2] = {m.out_h, m.out_w};
    write_npz(argv[2], {{"map_x", NpyArray::f32(sh, m.x.data())}, {"map_y", NpyArray::f32(sh, m.y.data())},
                        {"stride", NpyArray::scalar_i64(m.stride)}, {"shape", NpyArray::i64({2}, shape)}});
    std::printf("{\"photo\": \"%s\", \"w\": %d, \"h\": %d, \"grid\": [%d, %d], \"out\": [%d, %d]}\n", argv[1], bgr.cols, bgr.rows,
                m.gh, m.gw, m.out_h, m.out_w);
    return 0;
}
