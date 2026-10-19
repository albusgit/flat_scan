// Host check of the page-finder port (core/src/pagefind.cpp) against seg/pagefind.py; driven by
// tools/pagefind_parity.py. The net runs in Python (onnxruntime), as in the app it runs in ONNX
// Runtime outside the core.
//   pagefind_cli input PHOTO OUT.npz             -> {"x": segnet.to_input(photo)} (1 x 3 x 256 x 256 float32)
//   pagefind_cli find PHOTO LOGITS.npz OUTPREFIX -> OUTPREFIX_page0.png (0 / 255, photo size), _page1.png for a
//                                                   book spread, OUTPREFIX_contours.npz, OUTPREFIX.json
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <fstream>
#include <string>

#include "npz.h"
#include "pagefind.h"

using namespace fs;

static std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: pagefind_cli input PHOTO OUT.npz | find PHOTO LOGITS.npz OUTPREFIX\n");
        return 2;
    }
    std::string cmd = argv[1];
    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    if (cmd == "input") {
        std::vector<float> x = seg_input(img);
        write_npz(argv[3], {{"x", NpyArray::f32({1, 3, (size_t)SEG_SIZE, (size_t)SEG_SIZE}, x.data())}});
        return 0;
    }
    if (cmd != "find" || argc < 5) return 2;
    Npz z = read_npz(argv[3]);
    const NpyArray& lg = z.at("logits");
    if (lg.dtype != "<f4" || lg.count() != (size_t)SEG_OUTPUT_LEN) { std::fprintf(stderr, "logits: float32 2x256x256 expected\n"); return 1; }
    PageFindResult r = find_page_net(img, lg.as<float>());
    std::string pre = argv[4];
    Npz cz;
    for (size_t i = 0; i < r.pages.size(); ++i) {
        cv::imwrite(pre + "_page" + std::to_string(i) + ".png", r.pages[i].mask * 255);
        std::vector<double> c;
        for (auto& p : r.pages[i].contour) { c.push_back(p.x); c.push_back(p.y); }
        cz["page" + std::to_string(i)] = NpyArray::f64({r.pages[i].contour.size(), 2}, c.data());
    }
    write_npz(pre + "_contours.npz", cz);
    std::ofstream f(pre + ".json");
    f << "{\"pages\": " << r.pages.size() << ", \"whole_frame\": " << (r.whole_frame ? "true" : "false")
      << ", \"empty\": " << (r.empty ? "true" : "false") << ", \"spread\": " << (r.spread ? "true" : "false")
      << ", \"area_frac\": " << num(r.area_frac) << ", \"net_margin\": " << num(r.net_margin)
      << ", \"edge_strength\": " << num(r.edge_strength) << ", \"edge_at_limit\": " << num(r.edge_at_limit)
      << ", \"halves\": [" << r.halves[0] << ", " << r.halves[1] << "]}\n";
    return 0;
}
