// Neural page flattening with UVDoc-type networks: a net that predicts, for a coarse grid over the
// flat page, where each grid point sits in the photo. Two models share this code:
//  * UVDoc (Verhoeven et al. 2023, github.com/tanguymagne/UVDoc, MIT): 488 x 712 in, 45 x 31 grid;
//  * SyntheticDoc (Woortmann, Magne, Sorkine-Hornung, ECCV 2026, github.com/tanguymagne/SyntheticDoc,
//    code MIT): the same network retrained on 1M simulated pages, 512 x 720 in, 31 x 45 grid, plus a
//    shading map (the page's illumination, 3 x 720 x 512) used to remove shadows.
// The app runs the net on the page's bounding box (from the corners, plus a margin), turned so that
// the page's long side is vertical (both were trained on upright portrait documents). The net runs
// in the app (ONNX Runtime, assets/models/*.onnx; inputs RGB in [0, 1]); this file makes its input,
// reads its output back into photo pixels and checks it against the page finder.
#pragma once
#include <opencv2/core.hpp>
#include <array>
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
inline const UvSpec& uv_spec(int model) { return model == 1 ? SYNTHDOC_SPEC : UVDOC_SPEC; }   // 0 UVDoc, 1 SyntheticDoc

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
};
void uv_apply_shading(cv::Mat& page_bgra, int y0, const cv::Mat& map_x, const cv::Mat& map_y, const UvShading& s);

// IoU of the grid's outline with a page mask (CV_8U, non-zero = page) whose pixels are photo
// pixels * scale (the analysis copy), measured at <= 800 px.
double uv_agreement(const UvGrid& g, const cv::Mat& mask, double scale);
// the same against the corner quad (no mask: corners set by hand)
double uv_agreement(const UvGrid& g, const std::array<cv::Point2d, 4>& quad, cv::Size photo);

}  // namespace fs
