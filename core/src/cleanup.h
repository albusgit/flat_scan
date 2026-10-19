// Photometric filters on the unwarped page (all images 8-bit BGRA, OpenCV channel order).
//
// Shared analysis, computed once per page (page_params):
//   * background at 1/4 resolution: 4x4 block mean -> 3x3 dilate -> exact 11x11 median, per
//     channel (darkness_fast idea, kept in colour)                                 [GPU / CPU]
//   * paper illumination S: background pixels that are low-chroma and near the local maximum
//     brightness are paper; their luminance is filled into a smooth field (pull-push)  [CPU]
//   * white balance wb: paper colour per channel, from those paper samples (never from the
//     brightest pixels); paper noise sigma: robust spread (MAD) of normalised paper luminance
//     on ink-free paper pixels
// Every filter starts from v = I / (S * wb): lighting flattened, paper ~ 1.0.
//   Original      unwarped page only
//   Lighten       v / (1 - 1.5 sigma), clipped at paper: paper slightly whiter, ink untouched
//   Auto color    v >= wp ? 1 : (v / wp)^1.25 with wp = 1 - 2.5 sigma: paper white, colours kept
//   Grayscale     Auto color, desaturated (Rec.601)
//   High contrast wp = 1 - 3 sigma, exponent 1.8: whiter paper, darker ink
//   Black & white adaptive: luminance relative to the LOCAL background (1/4-res median of the
//                 normalised page), soft ramp ((r - 0.5) / (wp - 0.5))^2.5, wp = 1 - 3 sigma
//                 (min(that, 0.92) where the local background is tinted, below 0.97)
// Only values within the paper-noise band above wp are clipped to white; below it the curves
// are soft power curves with exponent >= 1, so ink is never lightened towards white.
// strength s in [0, 1]: out = Original + s * (filter - Original); s = 0 is Original.
// Isolated specks (<= 2 ink pixels in a 36 px window) are removed in Auto, Grayscale, High
// contrast and B&W.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

#include "solid.h"

namespace fs {

enum class Filter : int { Original = 0, Lighten = 1, AutoColor = 2, Grayscale = 3, HighContrast = 4, BlackWhite = 5 };
const char* filter_name(Filter f);

struct PageParams {
    cv::Mat illum;            // CV_32F, quarter grid: paper luminance S (0..255)
    cv::Mat bgv;              // CV_32F, quarter grid: local background luminance of v (B&W)
    float wb[3] = {1, 1, 1};  // per-channel paper colour / paper luminance
    float sigma = 0.02f;      // paper noise of luma(v)
    int page_w = 0, page_h = 0;
    bool gpu_background = false;
    double ms = 0;
};

// Curve constants for a filter (shared by the CPU and GPU code).
struct Curve {
    int kind = 0;             // 0 original, 1 lighten, 2 power curve, 3 black & white
    float wp = 1, gamma = 1;
    float wp_tint = 1;        // B&W white point where the local background is tinted (< 0.97)
    bool gray = false, specks = false;
};
Curve filter_curve(Filter f, float sigma);

PageParams page_params(const cv::Mat& page_bgra, bool use_gpu);
cv::Mat background_quarter_cpu(const cv::Mat& bgra);

// Paper fill: page * a + paper * (1 - a), a = paper alpha (CV_8U, same size; no-op if empty).
// The fill is clean paper: the page's own paper colour there (illumination x white balance),
// which every filter except Original maps to pure white.
cv::Mat apply_paper(const cv::Mat& page_bgra, const cv::Mat& paper, const PageParams& p);

// CPU filter. img may be a downscaled copy of the page (live thumbnails): its pixels are
// mapped to page coordinates for the illumination lookup. Specks are only removed at full size.
cv::Mat filter_cpu(const cv::Mat& img_bgra, const PageParams& p, Filter f, float strength);

struct RenderStats {
    bool gpu_filter = false;
    double ms_filter = 0, ms_solid = 0;
    int solid_regions = 0, solid_clusters = 0;
    SolidStats solid;
};

// Paper fill, filter (GPU or CPU), then Solid colors (not with Black & white).
cv::Mat render_page(const cv::Mat& page_bgra, const cv::Mat& paper, const PageParams& p, Filter f, float strength, bool solid,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st = nullptr);

}  // namespace fs
