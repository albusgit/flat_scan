// Neural page flattening with UVDoc-type networks: a net that predicts, for a coarse grid over the
// flat page, where each grid point sits in the photo. Two models share this code:
//  * UVDoc (Verhoeven et al. 2023, github.com/tanguymagne/UVDoc, MIT): 488 x 712 in, 45 x 31 grid;
//  * SyntheticDoc (Woortmann, Magne, Sorkine-Hornung, ECCV 2026, github.com/tanguymagne/SyntheticDoc,
//    code MIT): the same network retrained on 1M simulated pages, 512 x 720 in, 31 x 45 grid, plus a
//    shading map (the page's illumination, 3 x 720 x 512) used to remove shadows;
//  * SyntheticDoc high-res (the same paper's 1024 x 1440 model): the same network at 2x the input
//    size, a 177 x 121 grid (one cell ~13 px of a page instead of ~50 px) and a 3 x 1440 x 1024
//    shading map.
// The app runs the net on the page's bounding box (from the corners, plus a margin), turned so that
// the page's long side is vertical (both were trained on upright portrait documents). The net runs
// in the app (ONNX Runtime, assets/models/*.onnx; inputs RGB in [0, 1]); this file makes its input,
// reads its output back into photo pixels and checks it against the page finder.
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <functional>
#include <vector>

namespace fs {

// a model's input size and output grid (rows x cols of its 2 x gh x gw output)
struct UvSpec {
    int in_w, in_h, gw, gh;
    int input_len() const { return 3 * in_w * in_h; }
    int output_len() const { return 2 * gw * gh; }
};
constexpr UvSpec UVDOC_SPEC{488, 712, 31, 45};
constexpr UvSpec SYNTHDOC_SPEC{512, 720, 45, 31};
constexpr UvSpec SYNTHDOC_HR_SPEC{1024, 1440, 177, 121};
// 0 UVDoc, 1 SyntheticDoc, 2 SyntheticDoc high-res
inline const UvSpec& uv_spec(int model) { return model == 2 ? SYNTHDOC_HR_SPEC : model == 1 ? SYNTHDOC_SPEC : UVDOC_SPEC; }

// The part of the photo the net sees: the bounding box of the page corners with a 6% margin;
// turned: the box is turned 90 degrees clockwise first (the page is landscape in the photo).
struct UvCrop {
    cv::Rect box;
    bool turned = false;
    int model = 0;
};

// corners: TL, TR, BR, BL in photo px (the crop screen's / detection's order)
UvCrop uv_crop(cv::Size photo, const std::array<cv::Point2d, 4>& corners, int model = 0);

// Network input for the crop: cv::INTER_AREA resize to the model's size, RGB / 255, NCHW float.
// blank (optional, CV_8U photo size, non-zero = hide): the other page of a book spread is
// painted with the crop's background colour, so the net sees one page.
std::vector<float> uv_input(const cv::Mat& photo_bgr, const UvCrop& c, const cv::Mat& blank = cv::Mat());

// The grid in photo pixels, rows running down the page as the net saw it (portrait).
struct UvGrid {
    bool ok = false;
    bool turned = false;
    int gw = 0, gh = 0;
    std::vector<cv::Point2d> pts;    // gh x gw, row major
    // backward map of a flat page of W x H px (landscape when turned): page pixel -> photo pixel
    cv::Point2d at(double x, double y, int W, int H) const;
    // the page outline (the grid's border), photo px
    std::vector<cv::Point2f> outline() const;
};

UvGrid uv_grid(const float* out, const UvCrop& c);

// The model's shading map (SyntheticDoc: 3 x in_h x in_w, RGB in [0, 1], over the turned crop)
// applied to a page resampled from the photo through `map_x / map_y` rows (photo px per page
// pixel): page / (shading * k + 1 - k), k = min(1, 0.4 / brightness of the net input), as the
// reference inference. page: BGRA, modified in place.
struct UvShading {
    const float* data = nullptr;
    UvCrop crop;
    double brightness = 0.5;
    bool full_strength = false;   // divide by the whole map (the reference scales it down on bright photos)
};
void uv_apply_shading(cv::Mat& page_bgra, int y0, const cv::Mat& map_x, const cv::Mat& map_y, const UvShading& s);

// The zoomed second pass (creases, curls, tight folds: features under ~2 grid cells, which the
// net's fixed grid smooths over). The page is cut along the first grid into 4 overlapping quarters;
// each is shown to the net as its own page (the rest of the photo painted with the background, as
// for a book spread), so its fixed grid is ~2x denser there. A quarter whose grid outline does not
// match its region (IoU < 0.9) keeps the first grid. The quarters are blended with weights that
// fade to 0 at their cut edges (the net takes a cut for a page edge, so the other quarter's grid is
// used there) into a grid of (2 gw - 1) x (2 gh - 1) points. net(model, io): the net's input in,
// its grid (2 x gh x gw) out; false: failed. n_used: quarters taken from the zoomed pass.
using UvNetFn = std::function<bool(int model, std::vector<float>& io)>;
// nu x nv: the parts across x down (2 x 2: quarters; 1 x 2: top and bottom halves).
UvGrid uv_zoom(const cv::Mat& photo_bgr, const UvGrid& g, int model, const UvNetFn& net, int* n_used = nullptr, int nu = 2, int nv = 2);

// IoU of the grid's outline with a page mask (CV_8U, non-zero = page) whose pixels are photo
// pixels * scale (the analysis copy), measured at <= 800 px.
double uv_agreement(const UvGrid& g, const cv::Mat& mask, double scale);
// the same against the corner quad (no mask: corners set by hand)
double uv_agreement(const UvGrid& g, const std::array<cv::Point2d, 4>& quad, cv::Size photo);

}  // namespace fs
