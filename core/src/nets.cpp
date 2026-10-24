#include "nets.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <future>

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
    // one net on one crop into nr; false: no grid (input, net or output size failed)
    auto run_into = [&](const char* name, const UvCrop& c, std::vector<float>& grid, NetRun& nr) {
        auto t0 = Clock::now();
        const NetId id = c.model == 2 ? NetId::SynthDocHR : NetId::UVDoc;
        Inference& inf = c.model == 2 && hr ? *hr : uv_inf;
        const UvSpec& sp = uv_spec(c.model);
        nr.name = name;
        grid.clear();
        if (rq.cancel && *rq.cancel) return false;
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
        return nr.ok;
    };
    auto run = [&](const char* name, const UvCrop& c, std::vector<float>& grid) {
        NetRun nr;
        const bool ok = run_into(name, c, grid, nr);
        if (!nr.name.empty()) r.runs.push_back(nr);
        return ok;
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
    const bool want_uv = rq.nets != FlatNets::SynthDocOnly && crop_for(0, cu);
    const bool want_sd = rq.nets != FlatNets::UVDocOnly && crop_for(2, cs);
    // the two depend only on their crops (and run on their own Inference): side by side
    if (want_uv && want_sd) {
        NetRun nu, ns;
        auto f = std::async(std::launch::async, [&] { run_into("synthdoc_hr", cs, sd, ns); });
        run_into("uvdoc", cu, uv, nu);
        f.get();
        for (NetRun* n : {&nu, &ns})
            if (!n->name.empty()) r.runs.push_back(*n);
    } else if (want_uv) {
        run("uvdoc", cu, uv);
    } else if (want_sd) {
        run("synthdoc_hr", cs, sd);
    }
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

bool inpaint_fill(cv::Mat& page_bgra, const cv::Mat& mask, Inference& inf, const cv::Mat* paper, double* ms) {
    if (mask.empty() || mask.size() != page_bgra.size() || page_bgra.channels() != 4) return false;
    const int S = 512, W = page_bgra.cols, H = page_bgra.rows;
    // nearby areas share a crop
    cv::Mat grown;
    cv::dilate(mask, grown, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31)));
    cv::Mat lbl, stats, cent;
    const int n = cv::connectedComponentsWithStats(grown, lbl, stats, cent, 8, CV_32S);
    bool any = false;
    for (int k = 1; k < n; ++k) {
        cv::Rect hole(stats.at<int>(k, cv::CC_STAT_LEFT), stats.at<int>(k, cv::CC_STAT_TOP), stats.at<int>(k, cv::CC_STAT_WIDTH), stats.at<int>(k, cv::CC_STAT_HEIGHT));
        // a square crop twice the hole's size (at least 256 px): the page around it is what the net reads
        int side = std::min(std::max(W, H), std::max(256, 2 * std::max(hole.width, hole.height)));
        int cw = std::min(side, W), ch = std::min(side, H);
        int x0 = std::min(std::max(0, hole.x + hole.width / 2 - cw / 2), W - cw), y0 = std::min(std::max(0, hole.y + hole.height / 2 - ch / 2), H - ch);
        cv::Rect crop(x0, y0, cw, ch);
        cv::Mat m = mask(crop) & (lbl(crop) == k);
        if (cv::countNonZero(m) == 0) continue;
        cv::Mat rgb, rs, ms_;
        cv::cvtColor(page_bgra(crop), rgb, cv::COLOR_BGRA2RGB);
        if (paper && paper->size() == page_bgra.size()) {
            // beyond the paper: the paper's own colour (its median in the crop, outside the hole)
            cv::Mat off = (*paper)(crop) < 128, on = ((*paper)(crop) >= 128) & (m == 0);
            if (cv::countNonZero(off) > 0 && cv::countNonZero(on) > 0) {
                cv::Vec3b med;
                for (int q = 0; q < 3; ++q) {
                    std::vector<uint8_t> v;
                    for (int y = 0; y < ch; ++y)
                        for (int xx = 0; xx < cw; ++xx)
                            if (on.at<uint8_t>(y, xx)) v.push_back(rgb.at<cv::Vec3b>(y, xx)[q]);
                    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                    med[q] = v[v.size() / 2];
                }
                rgb.setTo(cv::Scalar(med[0], med[1], med[2]), off);
            }
        }
        cv::resize(rgb, rs, cv::Size(S, S), 0, 0, cw > S || ch > S ? cv::INTER_AREA : cv::INTER_LINEAR);
        cv::resize(m, ms_, cv::Size(S, S), 0, 0, cv::INTER_NEAREST);
        cv::dilate(ms_, ms_, cv::Mat::ones(3, 3, CV_8U));   // no finger edge pixels left as "known"
        // input (MI-GAN's demo): channel 0 = known - 0.5 (+0.5 page, -0.5 hole), 1-3 = RGB in [-1, 1] times known
        Tensor x{{1, 4, S, S}, std::vector<float>((size_t)4 * S * S)};
        float* d = x.data.data();
        for (int y = 0; y < S; ++y)
            for (int xx = 0; xx < S; ++xx) {
                const size_t i = (size_t)y * S + xx;
                const float known = ms_.at<uint8_t>(y, xx) ? 0.f : 1.f;
                const cv::Vec3b c = rs.at<cv::Vec3b>(y, xx);
                d[i] = known - 0.5f;
                for (int q = 0; q < 3; ++q) d[(size_t)(q + 1) * S * S + i] = (c[q] * (2.f / 255.f) - 1.f) * known;
            }
        std::vector<Tensor> out;
        std::string err;
        if (!inf.run(NetId::Inpaint, x, out, &err)) {
            LOGI("inpaint net: %s", err.c_str());
            return any;
        }
        if (ms) *ms += inf.last_ms(NetId::Inpaint);
        if (out.empty() || out[0].data.size() != (size_t)3 * S * S) return any;
        cv::Mat res(S, S, CV_8UC3);
        const float* o = out[0].data.data();
        for (int y = 0; y < S; ++y)
            for (int xx = 0; xx < S; ++xx)
                for (int q = 0; q < 3; ++q) {
                    float v = (o[(size_t)q * S * S + (size_t)y * S + xx] * 0.5f + 0.5f) * 255.f;
                    res.at<cv::Vec3b>(y, xx)[2 - q] = (uint8_t)std::min(255.f, std::max(0.f, std::round(v)));   // RGB -> BGR
                }
        cv::Mat back;
        cv::resize(res, back, crop.size(), 0, 0, cv::INTER_LINEAR);
        // the net's pixels in the hole, feathered over 1 px (MI-GAN's pipeline: 3x3 dilate, 5x5 Gaussian)
        cv::Mat a;
        cv::dilate(m, a, cv::Mat::ones(3, 3, CV_8U));
        cv::GaussianBlur(a, a, cv::Size(5, 5), 1.0);
        a = cv::max(a, m);
        for (int y = 0; y < ch; ++y) {
            uint8_t* p = page_bgra.ptr<uint8_t>(y0 + y) + 4 * x0;
            const uint8_t* b = back.ptr<uint8_t>(y);
            const uint8_t* w = a.ptr<uint8_t>(y);
            for (int xx = 0; xx < cw; ++xx) {
                if (!w[xx]) continue;
                const float t = w[xx] / 255.f;
                for (int q = 0; q < 3; ++q) p[4 * xx + q] = (uint8_t)std::lround(p[4 * xx + q] * (1 - t) + b[3 * xx + q] * t);
            }
        }
        any = true;
    }
    return any;
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
        if (y.empty() || (y[0].data.size() != (size_t)2 * w * h && y[0].data.size() != (size_t)4 * w * h)) return false;   // rules (+ text lines)
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
