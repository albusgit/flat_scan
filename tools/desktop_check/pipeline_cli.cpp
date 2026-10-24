// The app's photo pipeline (core/src/pipeline.cpp, CPU paths) on one photo, for the round-4
// checks: crash fixes, quality gate, orientation, page finder. Driven by tools/round4_check.py.
//   pipeline_cli [--backend B] PHOTO OUT.jpg [LOGITS.npz | net]
// The nets run in-process through core/src/infer.cpp (ONNX Runtime) on the repo's models
// (FS_MODELS=DIR overrides app/src/main/assets/models): LOGITS "net" = the page finder;
// FS_NETS=1 (both) | uv | hr = the flattening nets as the app runs them (core/src/nets.cpp: a first
// UVDoc pass on the whole photo unless FS_FRAME_CORNERS, then the nets on its re-crop);
// FS_NET_DUMP=PREFIX writes their outputs (PREFIX_logits.npz, _p1.npz, _rc0.npz, _rc2.npz, as the
// Python caches). --backend / FS_BACKEND=cpu,dml,...: the backends to try in order (CPU is always
// last); FS_ORT_THREADS=n: intra-op threads (default: ORT's); FS_ORT_CACHE=DIR: compiled-model
// cache. The JSON then has "nets" (per-net backend and times) and "pipeline_ms".
// FS_NETS runs the flattening nets inside run_photo (PipelineOptions::flat_nets), beside its CPU
// stages; FS_SEQUENTIAL=1: after the detection, nothing speculative (Gate D: the same outputs).
// FS_NET_DELAY_MS=n: the nets take n ms longer without CPU load (an accelerator's time). "nets" then
// has "overlap" (mode, what was reused, the nets' and the wait's ms), and "pipeline_ms" includes them.
// FS_IGNORE="x0,y0,x1,y1;..." (output-page fractions): areas whose lines the warp map ignores;
// FS_GPU=1 (a build with FS_DESKTOP_GPU): the GPU image stages on (PipelineOptions::use_gpu; the
// GPU stage policy's defaults, FS_GPU_DENSE=1 adds the dense stage); the JSON then has "gpu".
// FS_RULE_NET=1 | MODEL.onnx: the rule net in the line stage, in-process as the other nets (JSON "nets" has "rulenet");
// FS_RULE_NET_PY=1 with it: through tools/shadow_net/run_onnx.py and FS_PYTHON instead (Round 10's way).
// FS_PERSPECTIVE_ONLY=1: homography only, no warp map. With OUT.jpg, OUT.jpg.lines.jpg shows the fitted lines (red: ruled, green: text, blue: grid).
// Prints one JSON line: the crop-screen assessment of every page (assess_pages) and the
// processing result of page 0 (run_photo) without user edits; OUT.jpg = the page (or "-").
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "npz.h"
#include "cleanup.h"
#include "nets.h"
#include "pipeline.h"
#include "board.h"
#ifdef FS_DESKTOP_GPU
#include "gpu.h"
#include "resample.h"
#endif

using namespace fs;

static std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

static std::string strs(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + v[i] + "\"";
    return o + "]";
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// a command line for std::system: cmd /c on Windows needs the whole line quoted once more
static std::string shell_line(const std::string& py, const std::string& script, const std::vector<std::string>& args) {
    std::string c = "\"" + py + "\" -I \"" + script + "\"";
    for (const auto& a : args) c += " \"" + a + "\"";
#ifdef _WIN32
    return "\"" + c + "\"";
#else
    return c;
#endif
}

int main(int argc, char** argv) {
    // --backend B (anywhere): the inference backends, as FS_BACKEND
    std::string backend_arg = std::getenv("FS_BACKEND") ? std::getenv("FS_BACKEND") : "cpu";
    {
        int k = 1;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--backend" && i + 1 < argc) backend_arg = argv[++i];
            else argv[k++] = argv[i];
        }
        argc = k;
    }
    if (argc < 3) { std::fprintf(stderr, "usage: pipeline_cli [--backend B] PHOTO OUT.jpg [LOGITS.npz | net]\n"); return 2; }
    // in-process inference, created on first use
    std::unique_ptr<Inference> inf_store;
    const std::string models = std::getenv("FS_MODELS") ? std::getenv("FS_MODELS") : std::string(FS_REPO_DIR) + "/app/src/main/assets/models";
    const char* shade_model = std::getenv("FS_SHADE_NET");
    // FS_RULE_NET: "1" the repo's rulenet.onnx, else a model file
    const char* rule_model = std::getenv("FS_RULE_NET");
    const std::string rule_file = rule_model && std::string(rule_model) != "1" ? rule_model : "";
    auto inference = [&]() -> Inference& {
        if (!inf_store) {
            InferConfig cfg;
            if (!parse_backends(backend_arg, cfg.preference)) std::fprintf(stderr, "unknown backend in \"%s\"; using cpu\n", backend_arg.c_str());
            // FS_SHADE_NET=MODEL.onnx is the CPU ShadeNet model
            std::string shade = shade_model ? shade_model : "";
            cfg.load = [models, shade, rule_file](const std::string& file) {
                if (file == "rulenet.onnx" && !rule_file.empty()) return read_file(rule_file);
                return read_file(file == "shadenet.onnx" && !shade.empty() ? shade : models + "/" + file);
            };
            if (const char* t = std::getenv("FS_ORT_THREADS")) cfg.threads = std::atoi(t);
            if (const char* c = std::getenv("FS_ORT_CACHE")) cfg.cache_dir = c;
            inf_store = Inference::create(cfg);
        }
        return *inf_store;
    };
    std::ostringstream net_json;   // "nets" timings
    auto net_sep = [&]() { return net_json.tellp() > 0 ? ", " : ""; };
    const char* dump = std::getenv("FS_NET_DUMP");
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    // FS_RENDER_ONLY=1: PHOTO is an already flattened page; write its Auto color 0.7 render to OUT.jpg
    if (std::getenv("FS_RENDER_ONLY")) {
        cv::Mat bgra, shown, bgr;
        cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
        PageParams pp = page_params(bgra, false);
        shown = render_page(bgra, cv::Mat(), pp, Filter::AutoColor, 0.7f, false, {}, false);
        cv::cvtColor(shown, bgr, cv::COLOR_BGRA2BGR);
        cv::imwrite(argv[2], bgr, {cv::IMWRITE_JPEG_QUALITY, 88});
        return 0;
    }
    std::vector<float> lg;
    if (argc > 3 && std::string(argv[3]) == "net") {
        auto t0 = Clock::now();
        Inference& inf = inference();
        bool ok = page_finder_logits(img, inf, lg);
        net_json << "\"page_finder\": {\"ok\": " << (ok ? "true" : "false") << ", \"backend\": \"" << backend_name(inf.backend(NetId::PageFinder))
                 << "\", \"ms\": " << num(inf.last_ms(NetId::PageFinder)) << ", \"total_ms\": " << num(ms_since(t0)) << "}";
        if (ok && dump) write_npz(std::string(dump) + "_logits.npz", {{"logits", NpyArray::f32({2, (size_t)SEG_SIZE, (size_t)SEG_SIZE}, lg.data())}});
    } else if (argc > 3) {
        Npz z = read_npz(argv[3]);
        const NpyArray& a = z.at("logits");
        lg.assign(a.as<float>(), a.as<float>() + a.count());
    }
    const float* logits = lg.empty() ? nullptr : lg.data();
    std::ostringstream f;
    f << "{\"photo\": \"" << argv[1] << "\", \"assess\": [";
    try {
        auto pages = assess_pages(img, logits, 0);
        for (size_t i = 0; i < pages.size(); ++i) {
            const auto& a = pages[i];
            f << (i ? ", " : "") << "{\"ok\": " << (a.pc.ok ? "true" : "false") << ", \"fit_median_px\": " << num(a.fit_median_px)
              << ", \"reasons\": " << strs(a.reasons) << ", \"whole_frame\": " << (a.pc.whole_frame ? "true" : "false")
              << ", \"n_pages\": " << a.pc.n_pages << ", \"corners\": [";
            for (int k = 0; k < 4; ++k) f << (k ? ", " : "") << num(a.pc.pts[k].x) << ", " << num(a.pc.pts[k].y);
            f << "]}";
        }
        f << "]";
    } catch (const std::exception& e) {
        f << "], \"assess_error\": \"" << e.what() << "\"";
    }
    PipelineOptions o;
    o.use_gpu = false;
#ifdef FS_DESKTOP_GPU
    const bool gpu_on = std::getenv("FS_GPU") != nullptr;
    if (gpu_on) {
        o.use_gpu = true;
        GpuStagePolicy gp = gpu_stage_policy();
        if (std::getenv("FS_GPU_DENSE")) gp.dense = true;
        set_gpu_stage_policy(gp);
    }
#endif
    o.seg_logits = logits;
    o.perspective_only = std::getenv("FS_PERSPECTIVE_ONLY") != nullptr;
    o.line_stage = std::getenv("FS_NO_LINES") == nullptr;
    o.share_rows = std::getenv("FS_NO_SHARE") == nullptr;
    o.faint_lines = std::getenv("FS_NO_FAINT") == nullptr;
    o.second_line_pass = std::getenv("FS_SECOND") != nullptr;
    o.robust_solve = std::getenv("FS_NO_ROBUST") == nullptr;
    o.colour_edges = std::getenv("FS_NO_EDGES") == nullptr;
    if (const char* md = std::getenv("FS_MAX_DIST")) o.max_line_distortion = std::atof(md), o.max_line_distortion_frac = 1;
    if (std::getenv("FS_NO_SOFTEN")) o.max_local_rescale = 0;
    if (const char* st = std::getenv("FS_SOFTEN_THR")) o.max_local_rescale = std::atof(st);
    o.erase_fingers = std::getenv("FS_NO_FINGERS") == nullptr;
    if (const char* fr = std::getenv("FS_FINGER_RULES")) o.finger_rules = std::atoi(fr);   // kFinger* bits
    o.shadow_full_strength = std::getenv("FS_SHADE_REF") == nullptr;   // FS_SHADE_REF: the reference strength
    // FS_FRAME_CORNERS=1: the photo's frame is the page, as if set by hand (benchmarks whose photos are
    // cropped to the page, e.g. DocUNet); FS_NO_ORIENT=1: no 0/90/180/270 decision
    const bool frame_corners = std::getenv("FS_FRAME_CORNERS") != nullptr;
    if (frame_corners) {
        o.corners_edited = true;
        o.corners = {cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 0), cv::Point2d(img.cols - 1, img.rows - 1), cv::Point2d(0, img.rows - 1)};
    }
    o.auto_orient = std::getenv("FS_NO_ORIENT") == nullptr;
    // FS_BOARD=1 (or =dark): whiteboard mode as the app runs it: the board's corners (find_board; the whole
    // frame when none is reliable), perspective only, no finger erase, no orientation decision; with
    // FS_RENDER the Whiteboard (Dark slide) filter at 0.7 goes to OUT.jpg.board.jpg
    const char* board_env = std::getenv("FS_BOARD");
    BoardQuad board;
    if (board_env) {
        const int64 t_board = cv::getTickCount();
        board = find_board(img);
        const double board_ms = (cv::getTickCount() - t_board) * 1000.0 / cv::getTickFrequency();
        o.corners_edited = true;
        o.corners = board.ok ? board.pts
                             : std::array<cv::Point2d, 4>{cv::Point2d(1, 1), cv::Point2d(img.cols - 2, 1), cv::Point2d(img.cols - 2, img.rows - 2), cv::Point2d(1, img.rows - 2)};
        o.perspective_only = true;
        o.erase_fingers = false;
        o.auto_orient = false;
        std::fprintf(stderr, "board: %.1f ms\n", board_ms);
        std::fprintf(stderr, "board: ok %d method %s confidence %.2f area %.2f dark %d corners %.0f,%.0f %.0f,%.0f %.0f,%.0f %.0f,%.0f\n", board.ok, board.method,
                     board.confidence, board.area, (int)board_is_dark(img, board), o.corners[0].x, o.corners[0].y, o.corners[1].x, o.corners[1].y,
                     o.corners[2].x, o.corners[2].y, o.corners[3].x, o.corners[3].y);
    }
    o.scan_flattened = std::getenv("FS_SCAN_FLAT") != nullptr;   // with FS_SCAN=1: the input was flattened by a net
    o.orient_true_aspect = std::getenv("FS_ORIENT_A4") == nullptr;   // FS_ORIENT_A4=1: decide on the A4-shaped page (round 7)
    // FS_UV_MIN_AGREE=x: the page-outline agreement a net's grid needs (0: always used)
    if (const char* ma = std::getenv("FS_UV_MIN_AGREE")) o.uv_min_agreement = o.uv_min_agreement_quad = std::atof(ma);
    // FS_UV_PREFER=x: the text-row spread margin (deg) by which the second net must beat the first
    if (const char* pm = std::getenv("FS_UV_PREFER")) o.uv_prefer_margin = std::atof(pm);
    // FS_ZOOM=1: the zoomed second flattening pass (uv_zoom), the repo's ONNX models in-process
    if (std::getenv("FS_ZOOM")) {
        o.zoom_pass = true;
        o.uv_net = uv_net_fn(inference());
        if (const char* zw = std::getenv("FS_ZOOM_WORSE")) o.zoom_max_worse = std::atof(zw);
        if (const char* zm = std::getenv("FS_ZOOM_MODEL")) o.zoom_model = std::atoi(zm);
        if (const char* zs = std::getenv("FS_ZOOM_SPLIT")) { o.zoom_nu = zs[0] - '0'; o.zoom_nv = zs[1] - '0'; }   // e.g. 22, 12
    } else {
        o.zoom_pass = false;
    }
    // FS_SHADE_NET=MODEL.onnx: the shadow net, in-process; FS_SHADE_AFTER: after SyntheticDoc's map;
    // FS_SHADE_LS: long side
    double shade_ms = 0;
    if (shade_model) {
        o.shade_net = shade_net_fn(inference(), &shade_ms);
        o.shade_after_uv = std::getenv("FS_SHADE_AFTER") != nullptr;
        if (const char* ls = std::getenv("FS_SHADE_LS")) o.shade_long_side = std::atoi(ls);
    }
    // FS_RULE_NET=1 | MODEL.onnx: the rule net (PipelineOptions::rule_net), in-process (nets.cpp
    // rule_net_fn; MODEL.onnx replaces the repo's rulenet.onnx). FS_RULE_NET_PY=1: run by
    // tools/shadow_net/run_onnx.py with FS_PYTHON instead, as Round 10 did (raw float32 files next to
    // OUT), to compare the two runtimes.
    double rule_ms = 0;
    int rule_calls = 0;
    if (rule_model) {
        if (std::getenv("FS_RULE_NET_PY")) {
            std::string model = rule_file.empty() ? models + "/rulenet.onnx" : rule_file;
            std::string py = std::getenv("FS_PYTHON") ? std::getenv("FS_PYTHON") : "python";
            std::string base = std::string(argv[2]) == "-" ? std::string(argv[1]) : std::string(argv[2]);
            std::string fin = base + ".rnin.f32", fout = base + ".rnout.f32";
            std::string script = std::string(FS_REPO_DIR) + "/tools/shadow_net/run_onnx.py";
            o.rule_net = [=](std::vector<float>& io, int w, int h) {
                { std::ofstream f(fin, std::ios::binary); f.write((const char*)io.data(), io.size() * 4); }
                std::string cmd = shell_line(py, script, {model, fin, fout, std::to_string(w), std::to_string(h)});
                if (std::system(cmd.c_str()) != 0) return false;
                io.assign((size_t)2 * w * h, 0.f);
                std::ifstream f(fout, std::ios::binary);
                f.read((char*)io.data(), io.size() * 4);
                bool ok = (size_t)f.gcount() == io.size() * 4;
                std::remove(fin.c_str()); std::remove(fout.c_str());
                return ok;
            };
        } else {
            RuleNetFn net = rule_net_fn(inference(), &rule_ms);
            o.rule_net = [net, &rule_calls](std::vector<float>& io, int w, int h) {
                ++rule_calls;
                return net(io, w, h);
            };
        }
    }
    if (const char* ig = std::getenv("FS_IGNORE")) {
        std::string spec = ig;
        size_t p = 0;
        while (p < spec.size()) {
            size_t q = spec.find(';', p);
            std::array<float, 4> a{};
            if (std::sscanf(spec.substr(p, q - p).c_str(), "%f,%f,%f,%f", &a[0], &a[1], &a[2], &a[3]) == 4) o.ignore.push_back(a);
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    // UVDoc in two steps (the net runs in Python, tools/uvdoc_check.py): FS_UVIN=X.npz writes the
    // net input for the detected corners and exits; FS_UVOUT=Y.npz ("out": 2 x 45 x 31) uses it
    std::vector<float> uv, uv_sh, uv2, p1_store;
    NetsResult nets;
    const char* flat_nets = std::getenv("FS_NETS");
    if (flat_nets) {
        // the flattening nets in-process, as the app runs them (nets.cpp), inside run_photo: their
        // corners are its detection (the same call detect_corners made here before)
        NetsRequest rq;
        rq.corners_edited = frame_corners;
        rq.perspective_only = o.perspective_only;
        rq.nets = std::string(flat_nets) == "uv" ? FlatNets::UVDocOnly : std::string(flat_nets) == "hr" ? FlatNets::SynthDocOnly : FlatNets::Both;
        rq.seg_logits = logits;
        FlatNetsFn fn = flat_nets_fn(img, rq, inference(), &nets);
        if (const char* d = std::getenv("FS_NET_DELAY_MS")) {
            const int delay = std::atoi(d);
            fn = [fn, delay](const NetCornersFn& c) {
                FlatNetGrids g = fn(c);
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                return g;
            };
        }
        o.flat_nets = fn;
        o.overlap_nets = std::getenv("FS_SEQUENTIAL") == nullptr;
    } else if (std::getenv("FS_UVIN") || std::getenv("FS_UVOUT")) {
        PageCorners pc;
        if (frame_corners) pc.ok = true, pc.pts = o.corners;
        else pc = detect_corners(img, o.max_analysis_side, logits, 0);
        if (!pc.ok) { std::printf("%s, \"ok\": false, \"error\": \"no page found\"}\n", f.str().c_str()); return 0; }
        const char* um = std::getenv("FS_UVMODEL");
        o.uv_crop = uv_crop(img.size(), pc.pts, um ? std::atoi(um) : 0);
        // FS_UVPASS1=P.npz ("out"): a first pass of net FS_UVPASS1_MODEL (default 0) on the corners' crop,
        // or on the whole photo with FS_UVPASS1_FRAME=1; FS_UVRECROP=1: the nets' crops are the re-crop on it

        bool recrop = false;
        UvGrid g1;
        if (const char* pp = std::getenv("FS_UVPASS1")) {
            Npz zp = read_npz(pp);
            const NpyArray& ap = zp.at("out");
            p1_store.assign(ap.as<float>(), ap.as<float>() + ap.count());
            const char* pm = std::getenv("FS_UVPASS1_MODEL");
            std::array<cv::Point2d, 4> fr = {cv::Point2d(0, 0), cv::Point2d(img.cols - 1, 0), cv::Point2d(img.cols - 1, img.rows - 1), cv::Point2d(0, img.rows - 1)};
            o.uv_crop_pass1 = uv_crop(img.size(), std::getenv("FS_UVPASS1_FRAME") ? fr : pc.pts, pm ? std::atoi(pm) : 0);
            if (p1_store.size() == (size_t)uv_spec(o.uv_crop_pass1.model).output_len()) {
                o.uv_out_pass1 = p1_store.data();
                g1 = uv_grid(o.uv_out_pass1, o.uv_crop_pass1);
                recrop = std::getenv("FS_UVRECROP") != nullptr && g1.ok;
            }
        }
        if (recrop) o.uv_crop = uv_crop_on_grid(img.size(), g1, o.uv_crop.model);
        if (const char* in = std::getenv("FS_UVIN")) {
            cv::Mat blank;
            if (logits) {
                PageFindResult pf = find_page_net(img, logits);
                if (pf.pages.size() == 2) blank = pf.pages[1].mask;
            }
            std::vector<float> x = uv_input(img, o.uv_crop, blank);
            const UvSpec& sp = uv_spec(o.uv_crop.model);
            write_npz(in, {{"x", NpyArray::f32({1, 3, (size_t)sp.in_h, (size_t)sp.in_w}, x.data())}});
            return 0;
        }
        Npz z = read_npz(std::getenv("FS_UVOUT"));
        const NpyArray& a = z.at("out");
        uv.assign(a.as<float>(), a.as<float>() + a.count());
        if (uv.size() == (size_t)uv_spec(o.uv_crop.model).output_len()) o.uv_out = uv.data();
        // FS_UVOUT2=Z.npz ("out"): the other net's grid for the same corners, the second candidate
        Npz z2;
        if (const char* o2 = std::getenv("FS_UVOUT2")) {
            z2 = read_npz(o2);
            const NpyArray& a2 = z2.at("out");
            uv2.assign(a2.as<float>(), a2.as<float>() + a2.count());
            const char* um2 = std::getenv("FS_UVMODEL2");   // the second net's model (default: the other of 0 / 1)
            o.uv_crop2 = uv_crop(img.size(), pc.pts, um2 ? std::atoi(um2) : 1 - o.uv_crop.model);
            if (recrop) o.uv_crop2 = uv_crop_on_grid(img.size(), g1, o.uv_crop2.model);
            if (uv2.size() == (size_t)uv_spec(o.uv_crop2.model).output_len()) o.uv_out2 = uv2.data();
        }
        // the shading map of whichever net has one (SyntheticDoc)
        for (int k = 0; k < 2; ++k) {
            const Npz& zz = k == 0 ? z : z2;
            const UvCrop& cr = k == 0 ? o.uv_crop : o.uv_crop2;
            if (!zz.count("shading") || std::getenv("FS_NO_SHADOWS")) continue;
            const NpyArray& s = zz.at("shading");
            uv_sh.assign(s.as<float>(), s.as<float>() + s.count());
            if (uv_sh.size() == (size_t)uv_spec(cr.model).input_len()) { o.uv_shading = uv_sh.data(); o.uv_shading_crop = cr; }
        }
    }
    PipelineResult r;
    auto t_run = Clock::now();
    try {
        // FS_SCAN=1: the warp map on the image itself (run_scan), e.g. on an already flattened page
        r = std::getenv("FS_SCAN") ? run_scan(img, o) : run_photo(img, o);
    } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("exception: ") + e.what();
    }
    const double run_ms = ms_since(t_run);
    if (flat_nets) {
        // no page: no nets and this short line, as before the nets moved into run_photo
        if (!r.ok && r.error == "no page found" && !frame_corners) { std::printf("%s, \"ok\": false, \"error\": \"no page found\"}\n", f.str().c_str()); return 0; }
        for (const NetRun& n : nets.runs)
            net_json << net_sep() << "\"" << n.name << "\": {\"ok\": " << (n.ok ? "true" : "false") << ", \"backend\": \"" << backend_name(n.backend)
                     << "\", \"ms\": " << num(n.ms) << ", \"total_ms\": " << num(n.total_ms) << "}";
        net_json << net_sep() << "\"overlap\": {\"mode\": \"" << (r.nets_overlap == 1 ? "overlap" : r.nets_overlap == 0 ? "sequential" : "none")
                 << "\", \"orient_page_reused\": " << (r.orient_page_reused ? "true" : "false") << ", \"orient_reused\": " << (r.orient_reused ? "true" : "false")
                 << ", \"nets_ms\": " << num(r.nets_ms) << ", \"wait_ms\": " << num(r.nets_wait_ms) << ", \"join_ms\": " << num(r.nets_join_ms) << ", \"stages\": [";
        for (size_t i = 0; i < r.ms.size(); ++i) net_json << (i ? ", " : "") << "[\"" << r.ms[i].first << "\", " << num(r.ms[i].second) << "]";
        net_json << "]}";
        if (dump) {
            auto put = [&](const char* tag, const std::vector<float>& g, const UvCrop& c) {
                if (g.empty()) return;
                const UvSpec& sp = uv_spec(c.model);
                write_npz(std::string(dump) + tag, {{"out", NpyArray::f32({2, (size_t)sp.gh, (size_t)sp.gw}, g.data())}});
            };
            put("_p1.npz", nets.grid_pass1, nets.crop_pass1);
            put(nets.crop1.model == 2 ? "_rc2.npz" : "_rc0.npz", nets.grid1, nets.crop1);
            put(nets.crop2.model == 2 ? "_rc2.npz" : "_rc0.npz", nets.grid2, nets.crop2);
        }
    }
    if (o.shade_net) {
        Inference& inf = inference();
        net_json << net_sep() << "\"shadenet\": {\"ok\": " << (r.shade_net ? "true" : "false") << ", \"backend\": \""
                 << backend_name(inf.backend(NetId::ShadeNet)) << "\", \"ms\": " << num(shade_ms) << "}";
    }
    if (o.rule_net && rule_calls > 0) {
        Inference& inf = inference();
        net_json << net_sep() << "\"rulenet\": {\"ok\": " << (r.rule_net ? "true" : "false") << ", \"backend\": \""
                 << backend_name(inf.backend(NetId::RuleNet)) << "\", \"ms\": " << num(rule_ms) << ", \"calls\": " << rule_calls << "}";
    }
    f << ", \"ok\": " << (r.ok ? "true" : "false") << ", \"error\": \"" << r.error << "\", \"mode\": \""
      << geom_mode_name(r.mode) << "\", \"fit_median_full_px\": "
      << num(r.detection.ok ? r.fit_median_px / r.detection.scale : -1) << ", \"needs_manual_crop\": " << strs(r.needs_manual_crop)
      << ", \"rot\": " << r.rot << ", \"n_ctrl\": " << r.warp.n_ctrl << ", \"lines\": [" << r.warp.H.size() << ", " << r.warp.V.size()
      << "], \"text_rows\": " << r.warp.n_text << ", \"text_rows_v\": " << r.warp.n_text_v << ", \"text_net\": " << (r.warp.text_net ? "true" : "false") << ", \"fitted_lines\": " << r.lines.size() << ", \"n_ignored\": " << r.warp.n_ignored << ", \"uv_agreement\": " << num(r.uv_agreement) << ", \"uv_rejected\": " << (r.uv_rejected ? "true" : "false")
      << ", \"line_distortion\": " << num(r.line_distortion) << ", \"lines_rejected\": " << (r.lines_rejected ? "true" : "false") << ", \"fingers_erased\": " << num(r.fingers_erased) << ", \"uv_model\": " << r.uv_model
      << ", \"uv_consistency\": " << num(r.uv_consistency) << ", \"uv_defines_page\": " << (r.uv_defines_page ? "true" : "false")
      << ", \"orient\": [" << num(r.orient_sh) << ", " << num(r.orient_sv) << ", " << num(r.orient_upright) << "]"
      << ", \"uv_spread\": [" << num(r.uv_spread[0]) << ", " << num(r.uv_spread[1]) << "], \"shade_net\": " << (r.shade_net ? "true" : "false") << ", \"bowed\": [" << r.warp.S1.n_bowed << ", " << r.warp.S2.n_bowed << "], \"extended\": [" << r.warp.S1.n_extended << ", " << r.warp.S2.n_extended << "], \"second\": [" << num(r.second_pass[0]) << ", " << num(r.second_pass[1]) << ", " << num(r.second_pass[2]) << ", " << (r.second_pass_kept ? 1 : 0) << "], \"corners\": [" << r.warp.S1.n_corners << ", " << r.warp.S2.n_corners << "], \"rule_net\": " << (r.rule_net ? "true" : "false")
      << ", \"net\": [" << r.warp.net.dropped << ", " << r.warp.net.added << ", " << r.warp.net.joined << ", " << r.warp.S2.n_net_kept << "], \"spots_dropped\": "
      << (r.lines_spots_dropped ? "true" : "false") << ", \"straightness\": [" << num(r.line_straightness[0]) << ", " << num(r.line_straightness[1]) << "], \"margins\": " << r.warp.n_margins << ", \"edges\": [" << r.warp.n_edges_h << ", " << (r.warp.E_lines.size() - r.warp.n_edges_h) << ", " << r.warp.n_edge_bowed << "]" << ", \"downweighted\": " << r.warp.n_downweighted << ", \"bumps_faded\": " << num(r.bumps_faded) << ", \"zoom\": [" << r.uv_zoom_quarters << ", " << num(r.uv_zoom_spread[0]) << ", " << num(r.uv_zoom_spread[1]) << ", " << (r.uv_zoom ? 1 : 0) << "]}";
    // warp-map field size and roughness: max |displacement| and max change between grid neighbours per px
    auto field_stats = [&](const char* name, const Field& a, const Field& b) {
        double mx = 0, gr = 0;
        for (const Field* F : {&a, &b}) {
            if (F->v.empty()) continue;
            for (int i = 0; i < F->gh; ++i)
                for (int j = 0; j < F->gw; ++j) {
                    double v = F->v[(size_t)i * F->gw + j];
                    mx = std::max(mx, std::fabs(v));
                    if (j + 1 < F->gw) gr = std::max(gr, std::fabs(F->v[(size_t)i * F->gw + j + 1] - v) / F->g);
                    if (i + 1 < F->gh) gr = std::max(gr, std::fabs(F->v[(size_t)(i + 1) * F->gw + j] - v) / F->g);
                }
        }
        f << ", \"" << name << "_max\": " << num(mx) << ", \"" << name << "_grad\": " << num(gr);
    };
    if (r.ok) {
        std::string s = f.str();
        s.pop_back();   // reopen the object
        f.str(s);
        f.seekp(0, std::ios_base::end);
        field_stats("global", r.warp.GX, r.warp.GY);
        field_stats("local", r.warp.Fx, r.warp.Fy);
        // local area change: |det(I + dF) - 1| per grid cell, the outer 2 cells left out
        const Field &Fx = r.warp.Fx, &Fy = r.warp.Fy;
        std::vector<double> dets;
        for (int i = 2; i + 3 < Fx.gh; ++i)
            for (int j = 2; j + 3 < Fx.gw; ++j) {
                auto V = [&](const Field& F, int a, int b) { return F.v[(size_t)a * F.gw + b]; };
                double xx = (V(Fx, i, j + 1) - V(Fx, i, j)) / Fx.g, xy = (V(Fx, i + 1, j) - V(Fx, i, j)) / Fx.g;
                double yx = (V(Fy, i, j + 1) - V(Fy, i, j)) / Fy.g, yy = (V(Fy, i + 1, j) - V(Fy, i, j)) / Fy.g;
                dets.push_back(std::fabs((1 + xx) * (1 + yy) - xy * yx - 1));
            }
        if (!dets.empty()) {
            std::sort(dets.begin(), dets.end());
            size_t n20 = 0;
            for (double d : dets) n20 += d > 0.2;
            f << ", \"det_p99\": " << num(dets[(size_t)(0.99 * (dets.size() - 1))]) << ", \"det_frac20\": " << num((double)n20 / dets.size());
        }
        f << "}";
    }
#ifdef FS_DESKTOP_GPU
    if (gpu_on) {
        std::string s = f.str();
        s.pop_back();
        f.str(s);
        f.seekp(0, std::ios_base::end);
        const bool av = Gpu::instance().available();
        std::string info = Gpu::instance().info();
        for (char& c : info)
            if (c == '"' || c == '\\') c = '\'';
        f << ", \"gpu\": {\"available\": " << (av ? "true" : "false") << ", \"info\": \"" << info << "\", \"resample\": "
          << (r.gpu_resample ? "true" : "false") << "}}";
    }
#endif
    if (inf_store) {
        // in-process nets: their timings, and run_photo's own time (which includes ShadeNet)
        std::string s = f.str();
        s.pop_back();
        f.str(s);
        f.seekp(0, std::ios_base::end);
        f << ", \"nets\": {" << net_json.str() << "}, \"pipeline_ms\": " << num(run_ms) << "}";
    }
    std::printf("%s\n", f.str().c_str());
    // FS_DUMP=PREFIX: the stage-1 page (PREFIX_stage1.png) and the line tracer's segments on it
    // (PREFIX_segs.jpg: red horizontal, blue vertical; PREFIX_segs.txt: kind pos start end width length)
    if (const char* dp = std::getenv("FS_DUMP"); dp && r.ok && !r.stage1.empty()) {
        std::string p = dp;
        cv::imwrite(p + "_stage1.png", r.stage1);
        cv::Mat v = r.stage1.clone();
        std::ofstream t(p + "_segs.txt");
        for (const auto& L : r.warp.H) {
            cv::line(v, cv::Point((int)L[1], (int)L[0]), cv::Point((int)L[2], (int)L[0]), {0, 0, 255}, 2);
            t << "H " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " " << L[4] << "\n";
        }
        for (const auto& L : r.warp.V) {
            cv::line(v, cv::Point((int)L[0], (int)L[1]), cv::Point((int)L[0], (int)L[2]), {255, 0, 0}, 2);
            t << "V " << L[0] << " " << L[1] << " " << L[2] << " " << L[3] << " " << L[4] << "\n";
        }
        cv::imwrite(p + "_segs.jpg", v, {cv::IMWRITE_JPEG_QUALITY, 85});
        // PREFIX_rules.jpg: the rule net's probabilities over the stage-1 page (red horizontal, blue vertical)
        if (!r.warp.PH.empty() && r.warp.PH.size() == r.stage1.size()) {
            cv::Mat rv = r.stage1.clone();
            for (int y = 0; y < rv.rows; ++y)
                for (int x = 0; x < rv.cols; ++x) {
                    float a = r.warp.PH.at<float>(y, x), b = r.warp.PV.at<float>(y, x);
                    cv::Vec3b& q = rv.at<cv::Vec3b>(y, x);
                    if (a > 0.5f) q = {0, 0, 255};
                    if (b > 0.5f) q = {255, 0, 0};
                }
            cv::imwrite(p + "_rules.jpg", rv, {cv::IMWRITE_JPEG_QUALITY, 85});
        }
        // PREFIX_det.png: |det J - 1| of the line stage per 8-px cell (x 1000, 0.2 = 200), stage-1 size / 8
        const fs::Field &Fx = r.warp.Fx, &Fy = r.warp.Fy;
        if (Fx.gw > 3 && Fx.gh > 3) {
            cv::Mat dm(Fx.gh, Fx.gw, CV_8U, cv::Scalar(0));
            for (int i = 0; i + 1 < Fx.gh; ++i)
                for (int j = 0; j + 1 < Fx.gw; ++j) {
                    auto V = [&](const fs::Field& F, int a, int b) { return F.v[(size_t)a * F.gw + b]; };
                    double xx = (V(Fx, i, j + 1) - V(Fx, i, j)) / Fx.g, xy = (V(Fx, i + 1, j) - V(Fx, i, j)) / Fx.g;
                    double yx = (V(Fy, i, j + 1) - V(Fy, i, j)) / Fy.g, yy = (V(Fy, i + 1, j) - V(Fy, i, j)) / Fy.g;
                    dm.at<uint8_t>(i, j) = (uint8_t)std::min(255.0, 1000 * std::fabs((1 + xx) * (1 + yy) - xy * yx - 1));
                }
            cv::imwrite(p + "_det.png", dm);
        }
    }
    // FS_DUMP_SAMPLES=P.npz: the line stage's fine-pass samples ("h", "v": n x 3, along / ideal / disp) and
    // its found lines ("H", "V": n x 5), stage-1 frame after the global stage
    if (const char* ds = std::getenv("FS_DUMP_SAMPLES"); ds && r.ok) {
        auto rows = [](const std::vector<std::array<double, 3>>& a) {
            std::vector<float> f;
            for (const auto& q : a) for (double v : q) f.push_back((float)v);
            return f;
        };
        auto lines = [](const Lines& a) {
            std::vector<float> f;
            for (const auto& q : a) for (double v : q) f.push_back((float)v);
            return f;
        };
        std::vector<float> h = rows(r.warp.S2.h), v = rows(r.warp.S2.v), H = lines(r.warp.H), V = lines(r.warp.V);
        write_npz(ds, {{"h", NpyArray::f32({h.size() / 3, 3}, h.data())}, {"v", NpyArray::f32({v.size() / 3, 3}, v.data())},
                       {"H", NpyArray::f32({H.size() / 5, 5}, H.data())}, {"V", NpyArray::f32({V.size() / 5, 5}, V.data())}});
    }
    // FS_DUMP_MAP=P.npz: the final backward map on its 4-px grid ("x", "y": gh x gw, photo / scan px)
    if (const char* dm = std::getenv("FS_DUMP_MAP"); dm && r.ok) {
        const GridMap& g = r.final_map;
        write_npz(dm, {{"x", NpyArray::f32({(size_t)g.gh, (size_t)g.gw}, g.x.data())}, {"y", NpyArray::f32({(size_t)g.gh, (size_t)g.gw}, g.y.data())}});
    }
    if (r.ok && std::string(argv[2]) != "-") {
        cv::Mat bgr;
        cv::cvtColor(r.page, bgr, cv::COLOR_BGRA2BGR);
        cv::imwrite(argv[2], bgr, {cv::IMWRITE_JPEG_QUALITY, 85});
        // FS_FINGER_OUT=PREFIX: for the finger evaluation (tools/fingers/eval.py), in the PHOTO's frame:
        // PREFIX_footprint.png (the photo pixels the page shows) and PREFIX_fingers.png (those it filled as a
        // finger), each page grid cell (4 px) drawn as the quad it comes from
        if (const char* fo = std::getenv("FS_FINGER_OUT")) {
            const GridMap& g = r.render_map;
            cv::Mat foot = cv::Mat::zeros(img.size(), CV_8U), fing = cv::Mat::zeros(img.size(), CV_8U);
            for (int i = 0; i + 1 < g.gh; ++i)
                for (int j = 0; j + 1 < g.gw; ++j) {
                    std::vector<cv::Point> q;
                    for (auto [di, dj] : {std::pair{0, 0}, {0, 1}, {1, 1}, {1, 0}}) {
                        int ii = i + di, jj = j + dj;
                        q.emplace_back((int)std::lround(g.x[(size_t)ii * g.gw + jj]), (int)std::lround(g.y[(size_t)ii * g.gw + jj]));
                    }
                    cv::fillConvexPoly(foot, q, cv::Scalar(255));
                    if (!r.finger_mask.empty()) {
                        int y0 = i * g.stride, x0 = j * g.stride;
                        cv::Rect cell(x0, y0, g.stride, g.stride);
                        cell &= cv::Rect(0, 0, r.finger_mask.cols, r.finger_mask.rows);
                        if (cell.area() > 0 && cv::countNonZero(r.finger_mask(cell)) * 2 >= cell.area()) cv::fillConvexPoly(fing, q, cv::Scalar(255));
                    }
                }
            cv::imwrite(std::string(fo) + "_footprint.png", foot);
            cv::imwrite(std::string(fo) + "_fingers.png", fing);
        }
        // FS_PAGE_PNG=PREFIX: the flattened page and its paper alpha, lossless (tools/filters/)
        if (const char* pg = std::getenv("FS_PAGE_PNG")) {
            cv::imwrite(std::string(pg) + "_page.png", bgr);
            if (!r.paper.empty()) cv::imwrite(std::string(pg) + "_paper.png", r.paper);
        }
        const cv::Scalar col[3] = {{0, 0, 255}, {0, 160, 0}, {255, 0, 0}};
        for (const auto& l : r.lines)
            cv::line(bgr, cv::Point((int)(l[1] * bgr.cols), (int)(l[2] * bgr.rows)), cv::Point((int)(l[3] * bgr.cols), (int)(l[4] * bgr.rows)),
                     col[std::min(2, std::max(0, (int)l[0]))], 3);
        for (const auto& a : o.ignore)
            cv::rectangle(bgr, cv::Point((int)(a[0] * bgr.cols), (int)(a[1] * bgr.rows)), cv::Point((int)(a[2] * bgr.cols), (int)(a[3] * bgr.rows)),
                          cv::Scalar(0, 140, 255), 4);
        cv::imwrite(std::string(argv[2]) + ".lines.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 80});
        // FS_RENDER=1: the page as the app shows it by default (Auto color 0.7) -> OUT.jpg.auto.jpg
        if (std::getenv("FS_RENDER") && board_env) {
            PageParams pp = page_params(r.page, false);
            RenderParams rp;
            // as the app: the Dark slide filter when the screen is mostly dark (FS_BOARD=dark forces it)
            bool dark = std::string(board_env) == "dark" || board_is_dark(img, board);
            rp.filter = dark ? Filter::ProjectorDark : Filter::Whiteboard;
            rp.strength = 0.7f;
            cv::Mat shown = render_page(r.page, r.paper, pp, rp, {}, false);
            cv::cvtColor(shown, bgr, cv::COLOR_BGRA2BGR);
            cv::imwrite(std::string(argv[2]) + ".board.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 88});
        } else if (std::getenv("FS_RENDER")) {
            PageParams pp = page_params(r.page, false);
            cv::Mat shown = render_page(r.page, r.paper, pp, Filter::AutoColor, 0.7f, false, {}, false);
            cv::cvtColor(shown, bgr, cv::COLOR_BGRA2BGR);
            cv::imwrite(std::string(argv[2]) + ".auto.jpg", bgr, {cv::IMWRITE_JPEG_QUALITY, 88});
        }
    }
    return 0;
}
