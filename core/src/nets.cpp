#include "nets.h"

#include <algorithm>
#include <cctype>
#include <chrono>

#include "log.h"
#include "pagefind.h"

namespace fs {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

}  // namespace

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::XNNPACK: return "xnnpack";
        case Backend::QNN_HTP: return "qnn";
        case Backend::NNAPI: return "nnapi";
        case Backend::CUDA: return "cuda";
        case Backend::TensorRT: return "tensorrt";
        case Backend::DirectML: return "dml";
        case Backend::OpenVINO: return "openvino";
        case Backend::CoreML: return "coreml";
        case Backend::LiteRtGpu: return "gpu";
    }
    return "?";
}

bool parse_backends(const std::string& s, std::vector<Backend>& out) {
    out.clear();
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        std::string t = s.substr(p, q - p);
        for (char& c : t) c = (char)std::tolower((unsigned char)c);
        if (!t.empty()) {
            bool found = false;
            for (Backend b : {Backend::CPU, Backend::XNNPACK, Backend::QNN_HTP, Backend::NNAPI, Backend::CUDA, Backend::TensorRT, Backend::DirectML,
                              Backend::OpenVINO, Backend::CoreML, Backend::LiteRtGpu})
                if (t == backend_name(b) || (b == Backend::QNN_HTP && t == "qnn_htp") || (b == Backend::DirectML && t == "directml") ||
                    (b == Backend::LiteRtGpu && t == "litert")) {
                    out.push_back(b);
                    found = true;
                }
            if (!found) return false;
        }
        p = q + 1;
    }
    return true;
}

bool page_finder_logits(const cv::Mat& photo, Inference& inf, std::vector<float>& logits) {
    logits.clear();
    try {
        Tensor x{{1, 3, SEG_SIZE, SEG_SIZE}, seg_input(photo)};
        std::vector<Tensor> y;
        std::string err;
        if (!inf.run(NetId::PageFinder, x, y, &err)) {
            LOGI("page finder: %s", err.c_str());
            return false;
        }
        if (y.empty() || y[0].data.size() != (size_t)SEG_OUTPUT_LEN) return false;
        logits = std::move(y[0].data);
        return true;
    } catch (const std::exception& e) {
        LOGI("page finder failed: %s", e.what());
        return false;
    }
}

void NetsResult::apply(PipelineOptions& o) const {
    if (!grid1.empty()) { o.uv_out = grid1.data(); o.uv_crop = crop1; }
    if (!grid2.empty()) { o.uv_out2 = grid2.data(); o.uv_crop2 = crop2; }
    if (!grid_pass1.empty()) { o.uv_out_pass1 = grid_pass1.data(); o.uv_crop_pass1 = crop_pass1; }
}

NetsResult run_flat_nets(const cv::Mat& photo, const NetsRequest& rq, Inference& uv_inf, Inference* hr) {
    NetsResult r;
    if (!rq.flatten || rq.perspective_only || photo.empty()) return r;
    // the other page of a book spread is hidden from every net (uvInput)
    cv::Mat blank;
    if (rq.seg_logits) {
        try {
            PageFindResult f = find_page_net(photo, rq.seg_logits);
            if (f.pages.size() == 2) blank = f.pages[rq.page_index == 1 ? 0 : 1].mask;
        } catch (const std::exception& e) {
            LOGI("flattening nets: page finder failed: %s", e.what());
        }
    }
    // one net on one crop; false: no grid (input, net or output size failed)
    auto run = [&](const char* name, const UvCrop& c, std::vector<float>& grid) {
        auto t0 = Clock::now();
        const NetId id = c.model == 2 ? NetId::SynthDocHR : NetId::UVDoc;
        Inference& inf = c.model == 2 && hr ? *hr : uv_inf;
        const UvSpec& sp = uv_spec(c.model);
        NetRun nr;
        nr.name = name;
        grid.clear();
        try {
            if (c.model == 0 || c.model == 2) {
                Tensor x{{1, 3, sp.in_h, sp.in_w}, uv_input(photo, c, blank)};
                std::vector<Tensor> y;
                std::string err;
                if (x.data.size() == (size_t)sp.input_len() && inf.run(id, x, y, &err)) {
                    if (!y.empty() && y[0].data.size() == (size_t)sp.output_len()) grid = std::move(y[0].data);
                } else if (!err.empty()) {
                    LOGI("flattening net %s: %s", name, err.c_str());
                }
            }
        } catch (const std::exception& e) {
            LOGI("flattening net %s failed: %s", name, e.what());
            grid.clear();
        }
        nr.ok = !grid.empty();
        nr.backend = inf.backend(id);
        nr.ms = inf.last_ms(id);
        nr.total_ms = ms_since(t0);
        r.runs.push_back(nr);
        return nr.ok;
    };
    // automatic corners: a first UVDoc pass on the whole photo finds the page; the nets then run on its
    // outline (a page finder that split a folded page or cut a curled one does not decide what they see)
    if (!rq.corners_edited) {
        const double w1 = photo.cols - 1.0, h1 = photo.rows - 1.0;
        std::array<cv::Point2d, 4> frame = {cv::Point2d(0, 0), cv::Point2d(w1, 0), cv::Point2d(w1, h1), cv::Point2d(0, h1)};
        r.crop_pass1 = uv_crop(photo.size(), frame, 0);
        run("uv_pass1", r.crop_pass1, r.grid_pass1);
    }
    // false: no first-pass grid and no page corners (corners_fn found no page)
    auto crop_for = [&](int model, UvCrop& c) {
        if (!r.grid_pass1.empty()) {
            UvGrid g = uv_grid(r.grid_pass1.data(), r.crop_pass1);
            if (g.ok) { c = uv_crop_on_grid(photo.size(), g, model); return true; }
        }
        const std::array<cv::Point2d, 4>* q = rq.corners_fn ? rq.corners_fn() : &rq.corners;
        if (!q) return false;
        c = uv_crop(photo.size(), *q, model);
        return true;
    };
    std::vector<float> uv, sd;
    UvCrop cu, cs;
    if (rq.nets != FlatNets::SynthDocOnly && crop_for(0, cu)) run("uvdoc", cu, uv);
    if (rq.nets != FlatNets::UVDocOnly && crop_for(2, cs)) run("synthdoc_hr", cs, sd);
    // SyntheticDoc high-res fills the first slot (run_photo keeps it unless UVDoc's rows are far
    // straighter), UVDoc the second; UVDoc alone the first
    if (!sd.empty()) {
        r.grid1 = std::move(sd);
        r.crop1 = cs;
        if (!uv.empty()) { r.grid2 = std::move(uv); r.crop2 = cu; }
    } else if (!uv.empty()) {
        r.grid1 = std::move(uv);
        r.crop1 = cu;
    }
    return r;
}

FlatNetsFn flat_nets_fn(const cv::Mat& photo, const NetsRequest& rq, Inference& inf, NetsResult* keep, Inference* hr) {
    return [photo, rq, &inf, keep, hr](const NetCornersFn& corners) {
        NetsRequest q = rq;
        q.corners_fn = corners;
        NetsResult n = run_flat_nets(photo, q, inf, hr);
        FlatNetGrids g;
        g.grid1 = n.grid1;
        g.grid2 = n.grid2;
        g.grid_pass1 = n.grid_pass1;
        g.crop1 = n.crop1;
        g.crop2 = n.crop2;
        g.crop_pass1 = n.crop_pass1;
        if (keep) *keep = std::move(n);
        return g;
    };
}

std::function<bool(std::vector<float>&, int, int)> shade_net_fn(Inference& inf, double* ms) {
    return [&inf, ms](std::vector<float>& io, int w, int h) {
        if (w <= 0 || h <= 0 || io.size() != (size_t)3 * w * h) return false;
        Tensor x{{1, 3, h, w}, io};
        std::vector<Tensor> y;
        std::string err;
        if (!inf.run(NetId::ShadeNet, x, y, &err)) {
            LOGI("shadow net: %s", err.c_str());
            return false;
        }
        if (y.empty() || y[0].data.size() != io.size()) return false;
        io = std::move(y[0].data);
        if (ms) *ms += inf.last_ms(NetId::ShadeNet);
        return true;
    };
}

RuleNetFn rule_net_fn(Inference& inf, double* ms) {
    return [&inf, ms](std::vector<float>& io, int w, int h) {
        if (w <= 0 || h <= 0 || io.size() != (size_t)3 * w * h) return false;
        Tensor x{{1, 3, h, w}, io};
        std::vector<Tensor> y;
        std::string err;
        if (!inf.run(NetId::RuleNet, x, y, &err)) {
            LOGI("rule net: %s", err.c_str());
            return false;
        }
        if (y.empty() || y[0].data.size() != (size_t)2 * w * h) return false;
        io = std::move(y[0].data);
        if (ms) *ms += inf.last_ms(NetId::RuleNet);
        return true;
    };
}

UvNetFn uv_net_fn(Inference& inf) {
    return [&inf](int model, std::vector<float>& io) {
        if (model != 0 && model != 2) return false;   // the standard SyntheticDoc model (1) is no longer shipped
        const UvSpec& sp = uv_spec(model);
        if (io.size() != (size_t)sp.input_len()) return false;
        Tensor x{{1, 3, sp.in_h, sp.in_w}, io};
        std::vector<Tensor> y;
        if (!inf.run(model == 2 ? NetId::SynthDocHR : NetId::UVDoc, x, y) || y.empty() || y[0].data.size() != (size_t)sp.output_len()) return false;
        io = std::move(y[0].data);
        return true;
    };
}

}  // namespace fs
