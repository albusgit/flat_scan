// Page finder from the segmentation net: port of seg/pagefind.py (select_page, refine_contour,
// paper_mask) and seg/segnet.py (to_input, Predictor.probs). The network itself runs outside
// the core (ONNX Runtime in the app, onnxruntime in the parity tools); the core makes its
// input tensor and post-processes its logits.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace fs {

constexpr int SEG_SIZE = 256;                      // segnet.SIZE
constexpr int SEG_INPUT_LEN = 3 * SEG_SIZE * SEG_SIZE;
constexpr int SEG_OUTPUT_LEN = 2 * SEG_SIZE * SEG_SIZE;   // logits: [page paper, book gutter]
constexpr int PAGEFIND_WORK = 1000;                // pagefind.WORK (long side of the working image)

// segnet.to_input: the WHOLE photo squashed to 256 x 256 (INTER_AREA, aspect ignored), BGR -> RGB,
// / 255, ImageNet mean / std; NCHW float32 (1 x 3 x 256 x 256).
std::vector<float> seg_input(const cv::Mat& bgr);

// Page finder v2 (round 19, tools/train/pagefind2/net.py): the photo letterboxed into 384 x 384 (long side
// scaled to 384, INTER_AREA, centred, padded with the ImageNet mean), 3 outputs: paper, gutter, paper edge.
// Its output is turned into the round-4 layout above (paper and gutter logits of the whole photo squashed to
// 256 x 256), which everything downstream reads.
constexpr int SEG2_SIZE = 384;
constexpr int SEG2_INPUT_LEN = 3 * SEG2_SIZE * SEG2_SIZE;
constexpr int SEG2_OUTPUT_LEN = 3 * SEG2_SIZE * SEG2_SIZE;
struct SegLetterbox { double scale = 1; int x0 = 0, y0 = 0, w = 0, h = 0; };   // the photo inside the 384 x 384 input

// net.to_input: NCHW float32 (1 x 3 x 384 x 384); lb = where the photo went
std::vector<float> seg2_input(const cv::Mat& bgr, SegLetterbox* lb);
// the v2 output (3 x 384 x 384 logits) -> the round-4 layout (2 x 256 x 256: the photo's box resized, bilinear)
std::vector<float> seg2_to_seg(const float* out, const SegLetterbox& lb);

struct RefineInfo {
    std::vector<double> offset, strength;   // per outline point
    double edge_norm = 0, at_limit = 0;
};

// pagefind.refine_contour: snaps the outline to the strongest colour edge along its normals
// (+-R px, `step` px), Viterbi along the outline. img: BGR at the scale of pts.
std::vector<cv::Point2d> refine_contour(const cv::Mat& img, const std::vector<cv::Point2d>& pts, RefineInfo* info = nullptr,
                                        double R = 8.0, double step = 0.5, double alpha = 0.35, double beta = 0.6);

struct PageSelection {
    cv::Mat mask;                // CV_8U 0/1, working size: the page (the larger half of a spread)
    cv::Mat other;               // the second page of a book spread (pages == 2), else empty
    bool empty = false;          // no component above the threshold
    bool spread = false;         // a straight gutter line split the component
    int pages = 1;
    double area_frac = 0;
    cv::Point2d gutter_c, gutter_t;   // gutter line: centre, direction (working px)
    int halves[2] = {0, 0};
    double across = 0;           // spread: the paper's extent across the gutter / along it
    bool folded = false;         // the gutter is a fold of one page (kept whole)
};

// pagefind.select_page on the page / gutter probability maps (working size, float32)
PageSelection select_page(const cv::Mat& p, const cv::Mat& g, double thr = 0.5);

// pagefind._resample: a closed outline resampled every `step` px of arc length
std::vector<cv::Point2d> resample_closed(const std::vector<cv::Point2d>& c, double step);

struct FoundPage {
    cv::Mat mask;                         // CV_8U 0/1, photo size
    std::vector<cv::Point2d> contour;     // photo px, every 2 px (feeds curved_corners)
    double edge_strength = -1, edge_at_limit = -1;   // -1: not refined (whole-frame fallback)
};

struct PageFindResult {
    std::vector<FoundPage> pages;         // 1, or 2 for a book spread (pages[0] = the larger page)
    bool whole_frame = false;             // info['fallback'] == 'whole_frame'
    bool empty = false, spread = false;
    double area_frac = 0;                 // of the selected component (before a split), working px
    double net_margin = 0;                // how decided the net is in a band around the outline
    double edge_strength = -1, edge_at_limit = -1;   // of pages[0]; -1 when not refined
    cv::Point2d gutter_c, gutter_t;       // photo px
    int halves[2] = {0, 0};
    double across = 0;                    // spread: the paper's extent across the gutter / along it
    bool folded = false;                  // the gutter is a fold of one page (kept whole)
};

// pagefind.paper_mask with Predictor.probs: bgr = the photo the logits were computed on (any
// size; the reference uses the full photo), logits = the net output (2 x 256 x 256, NCHW).
// refine = edge snapping. The second page of a spread goes through the same outline / snapping /
// fill as the first (the reference drops it; the app processes it as a second document page).
PageFindResult find_page_net(const cv::Mat& bgr, const float* logits, bool refine = true);

}  // namespace fs
