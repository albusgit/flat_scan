// The page nets around the pipeline, run through Inference (infer.h): what the app's Kotlin did
// before (ScanViewModel.process, FlatNet, PageFinder, ShadeNet), now in the core so the app and
// the desktop harness run the same orchestration.
//  * page finder: seg_input -> PageFinder -> logits (PipelineOptions::seg_logits);
//  * flattening nets: with automatic corners, a first UVDoc pass on the whole photo (the other page
//    of a book spread hidden); UVDoc and SyntheticDoc high-res then run on the re-crop around the
//    page that pass found (uv_crop_on_grid), or on the corners' crop (uv_crop) without one. The
//    high-res grid fills the first slot, UVDoc's the second (run_photo prefers the first);
//  * ShadeNet: PipelineOptions::shade_net through Inference;
//  * the rule net (RuleNet, tools/rule_net, rulenet.onnx): PipelineOptions::rule_net through Inference.
// Every net failure leaves its slot empty, as before: the pipeline then runs without that net.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <vector>

#include "infer.h"
#include "pipeline.h"
#include "uvdoc.h"

namespace fs {

const char* backend_name(Backend b);
// "cpu", "xnnpack", "qnn", "nnapi", "cuda", "tensorrt", "dml", "openvino", "coreml" (any case,
// comma separated) -> backends; false on an unknown name
bool parse_backends(const std::string& s, std::vector<Backend>& out);

// The page finder's logits for the whole photo (2 x 256 x 256); false: net unavailable (the
// caller uses the colour-threshold finder).
bool page_finder_logits(const cv::Mat& photo_bgr, Inference& inf, std::vector<float>& logits);

// Settings > flattening nets
enum class FlatNets { Both = 0, UVDocOnly = 1, SynthDocOnly = 2 };

struct NetsRequest {
    std::array<cv::Point2d, 4> corners;   // TL, TR, BR, BL photo px (detection, or set by hand)
    bool corners_edited = false;          // set by hand: no first pass
    bool perspective_only = false;        // unwarp off: no flattening nets
    bool flatten = true;                  // Settings > neural flattening
    FlatNets nets = FlatNets::Both;
    const float* seg_logits = nullptr;    // page finder output (hides the other page of a spread); may be null
    int page_index = 0;
    // set: the corners come from here instead of `corners`, asked for only when a net needs them (no
    // first-pass grid); null result: no page found, those nets do not run (flat_nets_fn)
    NetCornersFn corners_fn;
};

struct NetRun {
    std::string name;                     // "uv_pass1", "uvdoc", "synthdoc_hr"
    bool ok = false;
    Backend backend = Backend::CPU;
    double ms = 0;                        // the net's run (Inference::last_ms)
    double total_ms = 0;                  // input, run and checks
};

struct NetsResult {
    std::vector<float> grid1, grid2, grid_pass1;   // empty: slot not filled
    UvCrop crop1, crop2, crop_pass1;
    std::vector<NetRun> runs;
    // PipelineOptions::uv_out / uv_crop, uv_out2 / uv_crop2, uv_out_pass1 / uv_crop_pass1 (pointers into this)
    void apply(PipelineOptions& o) const;
};

// hr (optional): SyntheticDoc high-res runs on this Inference instead of inf (its own backends: the
// app keeps it on the CPU where its fp16 grid misses the accelerator tolerance)
NetsResult run_flat_nets(const cv::Mat& photo_bgr, const NetsRequest& rq, Inference& inf, Inference* hr = nullptr);
// PipelineOptions::flat_nets: run_flat_nets on this photo with rq, the corners from run_photo (the
// detection, or rq.corners_edited's hand-set corners: set rq.corners_edited as the options'
// corners_edited). It may run on a worker thread (overlap_nets); keep (optional) receives the runs
// and grids, to be read after run_photo returned.
FlatNetsFn flat_nets_fn(const cv::Mat& photo_bgr, const NetsRequest& rq, Inference& inf, NetsResult* keep = nullptr, Inference* hr = nullptr);

// PipelineOptions::shade_net: ShadeNet on the planar BGR page (3 x h x w), replaced by its shading map.
// ms (optional): accumulates the run time.
std::function<bool(std::vector<float>&, int, int)> shade_net_fn(Inference& inf, double* ms = nullptr);
// PipelineOptions::rule_net: the rule net on the half-size stage-1 page (planar BGR [0, 1], 3 x h x w,
// sides multiples of 32), replaced by its 2 x h x w rule probabilities. ms (optional): accumulates the run time.
RuleNetFn rule_net_fn(Inference& inf, double* ms = nullptr);
// PipelineOptions::uv_net (the zoomed pass): model 0 UVDoc, 2 SyntheticDoc high-res.
UvNetFn uv_net_fn(Inference& inf);

}  // namespace fs
