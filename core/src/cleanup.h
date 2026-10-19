// Photometric filters on the unwarped page (all images 8-bit BGRA, OpenCV channel order).
//
// Shared analysis, computed once per page (page_params):
//   * background at 1/4 resolution: 4x4 block mean -> 3x3 dilate -> exact 11x11 median, per
//     channel (darkness_fast idea, kept in colour)                                 [GPU / CPU]
//   * paper illumination S: background pixels that are low-chroma and near the local maximum
//     brightness are paper; their luminance is filled into a smooth field (pull-push)  [CPU]
//     Islands of such samples that are not connected to the main paper (the inside of a grey
//     or dark photo, a grey cell) are dropped unless they are at the paper's level or look
//     like a cast shadow (reach the page edge, plain paper inside); otherwise the area would
//     be flattened to white
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
//                 (min(that, 0.92) where the local background is tinted, below 0.97);
//                 where the local background is dark or a saturated colour (picture weight
//                 pic, page_params) it blends to the Auto color curve of the luminance, so
//                 photos and dark boxes keep their tones instead of turning white
// Only values within the paper-noise band above wp are clipped to white; below it the curves
// are soft power curves with exponent >= 1, so ink is never lightened towards white.
// strength s in [0, 1]: out = Original + s * (filter - Original); s = 0 is Original.
// Isolated specks (<= 2 ink pixels and <= 12 pixels below 0.75 in a 36 px window) are removed
// in Auto, Grayscale, High contrast and B&W, only where the local background renders white
// (bgv >= wp; B&W: outside pictures). Light strokes (coloured text) and mid-tone areas
// (pictures, colour blocks) are never specks.
//
// User adjustments (Adjust; NativeLib.kt RenderParams), per pixel and channel, in this order:
//   filter curve t -> speck removal -> levels t = clamp((t - black) / (white - black), 0, 1)^(1/gamma)
//   -> background cleanup (bg > 0): soft knee towards white, k = 0.02 + 0.16 bg: weight
//      w = smoothstep(1 - 2k, 1 - k, luma(t)), t += (1 - t) w (luma >= 1 - k -> white; hue kept),
//      faded out with the chroma of t (max - min, 0.06 -> 0.15): light colours are not grey
//      artefacts and stay; with bg > 0 the speck limit also grows to 2 + round(10 bg) ink px
//      (and 4 * that + 4 px below 0.75) per 36-px window
//   -> grayscale -> strength blend with Original.
// The Original filter uses t = page / 255 (no curve, no specks) when the adjustments are not
// identity; with identity adjustments Original is the page itself. Defaults are identity, and
// an identity step is skipped (bit-identical to the unadjusted filters).
#pragma once
#include <opencv2/core.hpp>
#include <vector>

#include "solid.h"

namespace fs {

enum class Filter : int { Original = 0, Lighten = 1, AutoColor = 2, Grayscale = 3, HighContrast = 4, BlackWhite = 5 };
const char* filter_name(Filter f);

struct PageParams {
    cv::Mat illum;            // CV_32F, quarter grid: paper luminance S (0..255)
    cv::Mat bgv;              // CV_32F, quarter grid: local background luminance of v (B&W, specks)
    cv::Mat pic;              // CV_32F, quarter grid: picture weight 0..1 (B&W tones, specks)
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
    float speck_bg = 1;       // speck removal only where the local background bgv >= speck_bg
    bool gray = false, specks = false;
};
Curve filter_curve(Filter f, float sigma);

// User adjustments (defaults: identity).
struct Adjust {
    float bg_cleanup = 0.f;   // 0..1
    float black = 0.f;        // 0..0.5
    float white = 1.f;        // 0.5..1
    float gamma = 1.f;        // 0.3..3 (> 1 brightens mid-tones)
    bool levels_identity() const { return black == 0.f && white == 1.f && gamma == 1.f; }
    bool identity() const { return bg_cleanup <= 0.f && levels_identity(); }
    Adjust clamped() const;   // ranges enforced, white - black >= 0.02
};

// Everything a render depends on besides the page (NativeLib.kt RenderParams).
struct RenderParams {
    Filter filter = Filter::AutoColor;
    float strength = 0.7f;
    bool solid = false;
    Adjust adj;
};

// Adjustment constants (shared by the CPU and GPU code).
struct AdjustK {
    bool levels = false, bg = false;
    float black = 0, inv_range = 1, inv_gamma = 1;
    float knee_lo = 1, knee_hi = 1;
    int speck_max = 2;        // speck removal: 1..speck_max ink px per 36-px window (0 = off)
    int speck_light = 12;     // ... and at most speck_light px below 0.75 (speck + halo) in it
};
AdjustK adjust_constants(const Adjust& a, bool specks);

PageParams page_params(const cv::Mat& page_bgra, bool use_gpu);
cv::Mat background_quarter_cpu(const cv::Mat& bgra);

// Paper fill: page * a + paper * (1 - a), a = paper alpha (CV_8U, same size; no-op if empty).
// The fill is clean paper: the page's own paper colour there (illumination x white balance),
// which every filter except Original maps to pure white. Only thin slivers of non-paper along
// the outline are filled; parts thicker than 5% of the short page side are page content the
// outline missed (pictures, colour bands, shadowed text at the edge) and are kept.
cv::Mat apply_paper(const cv::Mat& page_bgra, const cv::Mat& paper, const PageParams& p);

// CPU filter. img may be a downscaled copy of the page (live thumbnails): its pixels are
// mapped to page coordinates for the illumination lookup. Specks are only removed at full size.
cv::Mat filter_cpu(const cv::Mat& img_bgra, const PageParams& p, Filter f, float strength, const Adjust& adj = Adjust());

struct RenderStats {
    bool gpu_filter = false;
    double ms_filter = 0, ms_solid = 0;
    int solid_regions = 0, solid_clusters = 0;
    SolidStats solid;
};

// Paper fill, filter (GPU or CPU), then Solid colors (not with Black & white).
cv::Mat render_page(const cv::Mat& page_bgra, const cv::Mat& paper, const PageParams& p, Filter f, float strength, bool solid,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st = nullptr);
cv::Mat render_page(const cv::Mat& page_bgra, const cv::Mat& paper, const PageParams& p, const RenderParams& rp,
                    const std::vector<Rect01>& protect, bool use_gpu, RenderStats* st = nullptr);

// 256-bin histogram of the luma (Rec.601, rounded) of an 8-bit BGRA image.
std::vector<int> luma_histogram(const cv::Mat& bgra);

}  // namespace fs
