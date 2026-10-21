// Stage commands: each runs one ported stage (optionally on the reference's own input for that
// stage) and writes NPZ / PNG for tools/compare_stages.py.
#include "cli_ext.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include "corners.h"
#include "surface.h"
#include "sparse.h"
#include "warpmap.h"
#include "pipeline.h"
#include "cleanup.h"
#include "solid.h"
#include "overlay.h"
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
#include <fstream>
#include <chrono>

using namespace fs;

static NpyArray pts_arr(const std::vector<cv::Point2d>& v) {
    std::vector<double> d;
    for (auto& p : v) { d.push_back(p.x); d.push_back(p.y); }
    return NpyArray::f64({v.size(), 2}, d.data());
}

static int cmd_detect(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    NpRng rng = NpRng::seed0();
    Detection D = detect_page(img, rng);
    if (!D.ok) { std::fprintf(stderr, "no page found\n"); return 1; }
    Npz z;
    std::vector<cv::Point2d> c(D.corners.begin(), D.corners.end());
    z["corners"] = pts_arr(c);
    z["contour"] = pts_arr(D.contour);
    std::vector<double> sp, ninl;
    for (auto& s : D.sides) { for (double v : s.p) sp.push_back(v); int n = 0; for (char b : s.inl) n += b; ninl.push_back(n); }
    z["side_p"] = NpyArray::f64({4, 3}, sp.data());
    z["side_inl"] = NpyArray::f64({4}, ninl.data());
    z["fold_d"] = NpyArray::f64({4}, D.fold_d.data());
    write_npz(argv[3], z);
    return 0;
}

static int cmd_surface(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    NpRng rng = NpRng::seed0();
    Detection D = detect_page(img, rng);
    if (!D.ok) return 1;
    auto t0 = std::chrono::steady_clock::now();
    SurfaceModel m = fit_surface(D.side_pts, D.corners, img.cols, img.rows, D.landscape, 0, 4);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("{\"axis\": \"%c\", \"c\": [%.10g, %.10g, %.10g, %.10g], \"rvec\": [%.10g, %.10g, %.10g], \"tvec\": [%.10g, %.10g, %.10g], "
                "\"cost\": %.10g, \"median_px\": %.6g, \"p90_px\": %.6g, \"fit_ms\": %.0f}\n",
                m.axis, m.c[0], m.c[1], m.c[2], m.c[3], m.rvec[0], m.rvec[1], m.rvec[2], m.tvec[0], m.tvec[1], m.tvec[2],
                m.cost, m.median_px, m.p90_px, ms);
    return m.ok ? 0 : 1;
}

// solvetest SYSTEM.npz GW : solve an exported reference system, compare with its exact solution
static int cmd_solvetest(int argc, char** argv) {
    Npz z = read_npz(argv[2]);
    int gw = std::atoi(argv[3]);
    Csr M;
    auto ptr = z.at("indptr").to_f64(), idx = z.at("indices").to_f64();
    M.n = (int)ptr.size() - 1;
    for (double v : ptr) M.ptr.push_back((int)v);
    for (double v : idx) M.idx.push_back((int)v);
    M.val = z.at("data").to_f64();
    auto b = z.at("b").to_f64(), ex = z.at("exact").to_f64(), ref = z.at("ref").to_f64();
    int gh = M.n / gw;
    for (double tol : {1e-5, 1e-8, 1e-10}) {
        SolveStats st;
        auto t0 = std::chrono::steady_clock::now();
        auto x = solve_spd_grid(M, b, gw, gh, nullptr, tol, 2000, &st);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double de = 0, dr = 0;
        for (int i = 0; i < M.n; ++i) { de = std::max(de, std::fabs(x[i] - ex[i])); dr = std::max(dr, std::fabs(x[i] - ref[i])); }
        std::printf("  rtol %.0e: %d iterations, %d levels, rel.res %.1e, %.0f ms | max |x - exact| %.2e px, max |x - reference| %.2e px\n",
                    tol, st.iterations, st.levels, st.rel_residual, ms, de, dr);
    }
    return 0;
}

static NpyArray rows3(const std::vector<std::array<double, 3>>& r) {
    std::vector<double> d;
    for (auto& q : r) d.insert(d.end(), q.begin(), q.end());
    return NpyArray::f64({r.size(), 3}, d.data());
}
static NpyArray lines5(const Lines& L) {
    std::vector<double> d;
    for (auto& q : L) d.insert(d.end(), q.begin(), q.end());
    return NpyArray::f64({L.size(), 5}, d.data());
}
static NpyArray field(const Field& F) { return NpyArray::f64({(size_t)F.gh, (size_t)F.gw}, F.v.data()); }

// warp IMAGE OUT.npz [A_IMG.png]: warpmap.unwarp_image with all intermediates
static int cmd_warp(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    WarpDebug d;
    auto t0 = std::chrono::steady_clock::now();
    GridMap g = unwarp_map(img, &d);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    Npz z;
    z["H"] = lines5(d.H); z["V"] = lines5(d.V);
    z["S1_h"] = rows3(d.S1.h); z["S1_v"] = rows3(d.S1.v); z["S2_h"] = rows3(d.S2.h); z["S2_v"] = rows3(d.S2.v); z["T"] = rows3(d.T);
    z["GX"] = field(d.GX); z["GY"] = field(d.GY); z["Fx1"] = field(d.Fx1); z["Fy1"] = field(d.Fy1); z["Fx"] = field(d.Fx); z["Fy"] = field(d.Fy);
    z["D"] = NpyArray::f32({(size_t)d.D.rows, (size_t)d.D.cols}, d.D.ptr<float>());
    double nc[2] = {(double)d.n_ctrl_raw, (double)d.n_ctrl};
    z["n_ctrl"] = NpyArray::f64({2}, nc);
    save_grid(argv[3], g, z);
    if (argc >= 5) cv::imwrite(argv[4], d.A_img);
    std::printf("{\"total_ms\": %.0f", ms);
    for (auto& m : d.ms) std::printf(", \"%s\": %.0f", m.first.c_str(), m.second);
    std::printf(", \"cg_iters\": [%d, %d, %d, %d, %d, %d]}\n", d.cg_iters[0], d.cg_iters[1], d.cg_iters[2], d.cg_iters[3], d.cg_iters[4], d.cg_iters[5]);
    return 0;
}

static bool flag(int argc, char** argv, const char* f) {
    for (int i = 0; i < argc; ++i) if (!std::strcmp(argv[i], f)) return true;
    return false;
}

static void write_outputs(const PipelineResult& r, const std::string& dir, const char* img_name, bool photo) {
    mkdir(dir.c_str(), 0755);
    cv::Mat bgr;
    cv::cvtColor(r.page, bgr, cv::COLOR_BGRA2BGR);
    cv::imwrite(dir + "/" + img_name, bgr);
    save_grid(dir + "/warpmap.npz", r.final_map, {});
    if (!r.stage1.empty()) cv::imwrite(dir + "/stage1.png", r.stage1);
    if (!r.paper.empty()) cv::imwrite(dir + "/paper.png", r.paper);
    if (!r.before.empty()) { cv::Mat b; cv::cvtColor(r.before, b, cv::COLOR_BGRA2BGR); cv::imwrite(dir + "/before.png", b); }
    if (!r.disp.x.empty()) {
        float top = 0;
        cv::Mat ov = wrinkle_overlay(r.disp, true, true, &top);
        cv::imwrite(dir + "/wrinkle_overlay.png", ov);
        // overlay composited on the page, for inspection
        cv::Mat comp = r.page.clone();
        for (int y = 0; y < comp.rows; ++y)
            for (int x = 0; x < comp.cols; ++x) {
                cv::Vec4b o = ov.at<cv::Vec4b>(y, x);
                cv::Vec4b& c = comp.at<cv::Vec4b>(y, x);
                for (int k = 0; k < 3; ++k) c[k] = (uint8_t)((c[k] * (255 - o[3]) + o[k] * o[3]) / 255);
            }
        cv::Mat c3;
        cv::cvtColor(comp, c3, cv::COLOR_BGRA2BGR);
        cv::imwrite(dir + "/wrinkle_composite.png", c3);
        std::ofstream(dir + "/wrinkle_legend.txt") << top << "\n";
    }
    std::ofstream f(dir + "/info.json");
    f << "{\"mode\": \"" << geom_mode_name(r.mode) << "\"";
    if (photo) {
        f << ", \"fell_back\": " << (r.fell_back ? "true" : "false") << ", \"fit_median_px\": " << r.fit_median_px
          << ", \"fit_p90_px\": " << r.fit_p90_px << ", \"axis\": \"" << r.model.axis << "\", \"rot\": " << r.rot
          << ", \"orient\": [" << r.orient_sh << ", " << r.orient_sv << ", " << r.orient_upright << "]";
    }
    f << ", \"gpu_resample\": " << (r.gpu_resample ? "true" : "false") << ", \"ms\": {";
    for (size_t i = 0; i < r.ms.size(); ++i) f << (i ? ", " : "") << "\"" << r.ms[i].first << "\": " << (long)std::lround(r.ms[i].second);
    f << "}}\n";
}

// photo PHOTO OUTDIR [--v8] [--cpu] [--orient-stage1] [--reference-orient] [--no-orient]
static int cmd_photo(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    PipelineOptions o;
    o.use_gpu = !flag(argc, argv, "--cpu");
    o.orient_on_stage1 = flag(argc, argv, "--orient-stage1");
    o.reference_orient = flag(argc, argv, "--reference-orient");   // ungated CW/CCW cue for sideways pages (reference)
    o.auto_orient = !flag(argc, argv, "--no-orient");
    if (flag(argc, argv, "--v8")) o.surface_fallback_px = -1;   // force the homography path
    PipelineResult r = run_photo(img, o);
    if (!r.ok) { std::fprintf(stderr, "%s\n", r.error.c_str()); return 1; }
    write_outputs(r, argv[3], "out.png", true);
    std::printf("%s: mode %s rot %d gpu %d total %.0f ms\n", argv[2], geom_mode_name(r.mode), r.rot,
                (int)r.gpu_resample, r.ms.back().second);
    return 0;
}

static int cmd_scan(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    PipelineOptions o;
    o.use_gpu = !flag(argc, argv, "--cpu");
    PipelineResult r = run_scan(img, o);
    write_outputs(r, argv[3], "B_warpmap.png", false);
    std::printf("%s: gpu %d total %.0f ms\n", argv[2], (int)r.gpu_resample, r.ms.back().second);
    return 0;
}

// render PAGE.png OUTDIR [--paper PAPER.png] [--filters 0,1,2,3,4,5] [--strength 0.7] [--solid] [--cpu]
//        [--bg 0..1] [--black 0..0.5] [--white 0.5..1] [--gamma 0.3..3]
// writes OUTDIR/f<k>.png per filter and prints GPU vs CPU differences and timings
static int cmd_render(int argc, char** argv) {
    cv::Mat in = cv::imread(argv[2], cv::IMREAD_COLOR), page;
    cv::cvtColor(in, page, cv::COLOR_BGR2BGRA);
    std::string dir = argv[3];
    mkdir(dir.c_str(), 0755);
    cv::Mat paper;
    for (int i = 0; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--paper")) paper = cv::imread(argv[i + 1], cv::IMREAD_GRAYSCALE);
    float strength = 0.7f;
    std::string fl = "0,1,2,3,4,5";
    Adjust adj;
    for (int i = 0; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--strength")) strength = (float)std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--filters")) fl = argv[i + 1];
        if (!std::strcmp(argv[i], "--bg")) adj.bg_cleanup = (float)std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--black")) adj.black = (float)std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--white")) adj.white = (float)std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--gamma")) adj.gamma = (float)std::atof(argv[i + 1]);
    }
    bool solid = flag(argc, argv, "--solid"), gpu = !flag(argc, argv, "--cpu");
    PageParams pp = page_params(page, gpu);
    std::printf("{\"page_params_ms\": %.0f, \"gpu_background\": %d, \"sigma\": %.4f, \"wb\": [%.3f, %.3f, %.3f], \"filters\": [", pp.ms,
                (int)pp.gpu_background, pp.sigma, pp.wb[0], pp.wb[1], pp.wb[2]);
    bool first = true;
    for (size_t i = 0; i < fl.size(); i += 2) {
        int k = fl[i] - '0';
        Filter f = (Filter)k;
        RenderStats sg, sc;
        RenderParams rp;
        rp.filter = f; rp.strength = strength; rp.solid = solid; rp.adj = adj;
        cv::Mat og = render_page(page, paper, pp, rp, {}, gpu, &sg);
        cv::Mat oc = render_page(page, paper, pp, rp, {}, false, &sc);
        cv::Mat d;
        cv::absdiff(og, oc, d);
        double mx;
        cv::minMaxLoc(d.reshape(1), nullptr, &mx);
        cv::Mat o3;
        cv::cvtColor(og, o3, cv::COLOR_BGRA2BGR);
        cv::imwrite(dir + "/f" + std::to_string(k) + (solid ? "_solid" : "") + ".png", o3);
        std::printf("%s{\"filter\": \"%s\", \"gpu\": %d, \"gpu_ms\": %.0f, \"cpu_ms\": %.0f, \"solid_ms\": %.0f, \"gpu_vs_cpu_max\": %.0f, \"values_gt1\": %d, \"regions\": %d, \"clusters\": %d, \"filled\": %d, \"white\": %d, \"skipped\": %d, \"varied\": %d, \"blobs\": %d}",
                    first ? "" : ", ", filter_name(f), (int)sg.gpu_filter, sg.ms_filter, sc.ms_filter, sg.ms_solid, mx,
                    cv::countNonZero(d.reshape(1) > 1), sg.solid_regions, sg.solid_clusters, sg.solid.filled, sg.solid.white, sg.solid.skipped, sg.solid.skipped_varied, sg.solid.blobs);
        first = false;
    }
    std::printf("]}\n");
    return 0;
}

int cli_ext(const std::string& cmd, int argc, char** argv) {
    if (cmd == "render" && argc >= 4) return cmd_render(argc, argv);
    if (cmd == "photo" && argc >= 4) return cmd_photo(argc, argv);
    if (cmd == "scan" && argc >= 4) return cmd_scan(argc, argv);
    if (cmd == "warp" && argc >= 4) return cmd_warp(argc, argv);
    if (cmd == "solvetest" && argc >= 4) return cmd_solvetest(argc, argv);
    if (cmd == "surface" && argc >= 3) return cmd_surface(argc, argv);
    if (cmd == "detect" && argc >= 4) return cmd_detect(argc, argv);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
