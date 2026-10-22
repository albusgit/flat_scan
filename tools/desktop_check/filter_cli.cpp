// Filter checks on flattened pages (tools/filters/): renders filters of the core's cleanup stage.
// usage: filter_cli PAGE.png OUTDIR [--paper PAPER.png] [--filters 0,2,5] [--strength 0.7] [--bg 0..1]
//                   [--gpu] [--repeat N] [--jpg] [--thumb W]
// writes OUTDIR/f<k>.png (or .jpg) per filter, prints one JSON line with timings
// (--thumb W: also renders a W px wide live thumbnail as the app does, OUTDIR/t<k>.png)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cleanup.h"
#include "screen.h"

using namespace fs;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: filter_cli PAGE.png OUTDIR [options]\n"); return 2; }
    cv::Mat in = cv::imread(argv[1], cv::IMREAD_COLOR), page;
    if (in.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    cv::cvtColor(in, page, cv::COLOR_BGR2BGRA);
    std::string dir = argv[2];
    cv::Mat paper;
    float strength = 0.7f;
    std::string fl = "0,1,2,3,4,5";
    Adjust adj;
    bool gpu = false, jpg = false;
    int repeat = 1, thumb = 0, screen = 0;   // --screen 0 off, 1 on, 2 auto (detect_screen)
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)"0"; };
        if (a == "--paper") paper = cv::imread(nx(), cv::IMREAD_GRAYSCALE);
        else if (a == "--strength") strength = (float)std::atof(nx());
        else if (a == "--filters") fl = nx();
        else if (a == "--bg") adj.bg_cleanup = (float)std::atof(nx());
        else if (a == "--black") adj.black = (float)std::atof(nx());
        else if (a == "--white") adj.white = (float)std::atof(nx());
        else if (a == "--gamma") adj.gamma = (float)std::atof(nx());
        else if (a == "--gpu") gpu = true;
        else if (a == "--jpg") jpg = true;
        else if (a == "--repeat") repeat = std::max(1, std::atoi(nx()));
        else if (a == "--thumb") thumb = std::atoi(nx());
        else if (a == "--screen") screen = std::atoi(nx());
        else if (a == "--tune") {   // key=value,... (ScanTune fields)
            std::string spec = nx();
            size_t q = 0;
            while (q < spec.size()) {
                size_t e = spec.find(',', q);
                std::string kv = spec.substr(q, e == std::string::npos ? std::string::npos : e - q);
                size_t eq = kv.find('=');
                if (eq != std::string::npos) {
                    std::string k = kv.substr(0, eq);
                    float v = (float)std::atof(kv.c_str() + eq + 1);
                    ScanTune& t = scan_tune();
                    if (k == "wp_k") t.wp_k = v;
                    else if (k == "gamma") t.gamma = v;
                    else if (k == "black_k") t.black_k = v;
                    else if (k == "black_max") t.black_max = v;
                    else if (k == "sharpen") t.sharpen = v;
                    else if (k == "sharpen_sigma") t.sharpen_sigma = v;
                    else std::fprintf(stderr, "unknown tune key %s\n", k.c_str());
                }
                if (e == std::string::npos) break;
                q = e + 1;
            }
        }
    }
    if (!paper.empty() && paper.size() != page.size()) paper.release();
    ScreenInfo si;
    for (int r = 0; r < repeat; ++r) {
        ScreenInfo s1 = detect_screen(page);
        if (r == 0 || s1.ms < si.ms) si = s1;
    }
    DescreenStats ds;
    if (screen == 1 || (screen == 2 && si.screen)) {
        cv::Mat d;
        for (int r = 0; r < repeat; ++r) {
            DescreenStats d1;
            d = descreen(page, &d1);
            if (r == 0 || d1.ms < ds.ms) ds = d1;
        }
        page = d;
    }
    double pp_ms = 1e30;
    PageParams pp;
    for (int r = 0; r < repeat; ++r) {
        auto t0 = Clock::now();
        pp = page_params(page, gpu);
        pp_ms = std::min(pp_ms, ms_since(t0));
    }
    std::printf("{\"page\": \"%s\", \"w\": %d, \"h\": %d, \"page_params_ms\": %.1f, \"sigma\": %.4f, \"ink\": %.3f, \"screen_score\": %.3f, \"screen_score_y\": %.3f, \"screen\": %d, \"detect_ms\": %.1f, \"descreen_ms\": %.1f, \"peaks\": %d, \"filters\": [", argv[1], page.cols,
                page.rows, pp_ms, pp.sigma, pp.ink, si.score, si.score_y, (int)si.screen, si.ms, ds.ms, ds.peaks);
    bool first = true;
    for (size_t i = 0; i < fl.size(); ++i) {
        if (fl[i] < '0' || fl[i] > '9') continue;
        int k = std::atoi(fl.c_str() + i);
        while (i + 1 < fl.size() && fl[i + 1] >= '0' && fl[i + 1] <= '9') ++i;
        RenderParams rp;
        rp.filter = (Filter)k; rp.strength = strength; rp.adj = adj;
        RenderStats st;
        cv::Mat out;
        double best = 1e30;
        for (int r = 0; r < repeat; ++r) {
            auto t0 = Clock::now();
            out = render_page(page, paper, pp, rp, {}, gpu, &st);
            best = std::min(best, ms_since(t0));
        }
        cv::Mat o3;
        cv::cvtColor(out, o3, cv::COLOR_BGRA2BGR);
        cv::imwrite(dir + "/f" + std::to_string(k) + (jpg ? ".jpg" : ".png"), o3, {cv::IMWRITE_JPEG_QUALITY, 92});
        double tms = 0;
        if (thumb > 0) {
            cv::Mat filled = apply_paper(page, paper, pp), small, t3;
            int th = std::max(1, (int)std::lround((double)thumb * page.rows / page.cols));
            cv::resize(filled, small, cv::Size(thumb, th), 0, 0, cv::INTER_AREA);
            auto t0 = Clock::now();
            cv::Mat t = filter_cpu(small, pp, rp.filter, rp.strength, rp.adj);
            tms = ms_since(t0);
            cv::cvtColor(t, t3, cv::COLOR_BGRA2BGR);
            cv::imwrite(dir + "/t" + std::to_string(k) + ".png", t3);
        }
        std::printf("%s{\"filter\": %d, \"name\": \"%s\", \"ms\": %.1f, \"gpu\": %d, \"thumb_ms\": %.1f}", first ? "" : ", ", k,
                    filter_name(rp.filter), best, (int)st.gpu_filter, tms);
        first = false;
    }
    std::printf("]}\n");
    return 0;
}
