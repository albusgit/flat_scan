// Screen mode: photos of a monitor, laptop or phone screen (round 11, tools/filters/).
//
// A phone photo of an LCD has, besides the page content: the screen's pixel grid and RGB subpixel
// stripes, resolved or aliased by the camera's sensor into moire (colour rainbow bands and fine
// periodic patterns), refresh / PWM bands, and the glow of the screen's white. Paper has none of
// the periodic COLOUR patterns: its ink and paper are neutral at the pixel scale, text rows are
// periodic only at low frequencies and in the luminance.
//
// detect_screen: on the flattened page, 3 x 3 tiles of 512 px (256 on small pages); in each tile
//   the spectrum (Hann window) of the chroma channels Cr and Cb, log magnitude minus its 15 x 15
//   median; the tile's score is its strongest peak at periods below 16 px, off the two axes (page
//   edges, rules and text columns put energy on the axes). Score = median over the tiles of
//   max(Cr, Cb). A screen if score >= kScreenThreshold (screen.cpp; measured on paper and screen
//   sets in docs/results/r11/filters.md).
// descreen: removes what the screen added, before the filters run:
//   1. notch filter of the luminance: periodic peaks of the spectrum of the page's BACKGROUND (text
//      with a local contrast above 40 levels masked out and filled in, so the harmonics of regular
//      text are not taken for a pattern; log magnitude > 2.0 above its 21 x 21 median, 4.0 on the two
//      axes, periods below 12 px) are cut (soft Gaussian notches) - pixel grid, stripes, aliases;
//   2. chroma: Cr and Cb are replaced by a guided filter of themselves with the luminance as guide
//      (radius 8 px, eps 20^2): colour that follows the content's edges stays, colour moire that has
//      no luminance edge of its own is averaged away.
//   The filters then flatten the screen's glow and bands like paper lighting (page_params).
#pragma once
#include <opencv2/core.hpp>

namespace fs {

struct ScreenInfo {
    float score = 0.f;        // median over tiles of the strongest chroma peak (log magnitude above median)
    float score_y = 0.f;      // the same for the luminance (reported, not used for the decision)
    bool screen = false;      // score >= kScreenThreshold
    double ms = 0;
};

extern const float kScreenThreshold;

// page: 8-bit BGRA (the flattened page)
ScreenInfo detect_screen(const cv::Mat& page_bgra);

struct DescreenStats {
    int peaks = 0;            // spectral peaks notched
    double ms = 0;
};
// Descreen constants (defaults shipped; desktop tools override them for tuning)
struct DescreenTune {
    float peak_thr = 2.0f;    // notch: log magnitude above the 21 x 21 median
    float text_contrast = 40.f;   // notch: pixels this far from their sigma-6 surroundings are text (masked)
    float peak_thr_axis = 4.0f;   // ... on the two axes (+-2 bins): text rows and columns have harmonics there
    float min_period = 12.f;  // notch: only periods below this (px)
    int chroma_r = 8;         // guided chroma: radius (0 = off)
    float chroma_eps = 20.f;  // guided chroma: eps (levels)
};
DescreenTune& descreen_tune();
// Returns a new BGRA page (same size).
cv::Mat descreen(const cv::Mat& page_bgra, DescreenStats* st = nullptr);

}  // namespace fs
