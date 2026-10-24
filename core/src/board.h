// Whiteboards and projector screens (whiteboard mode, Phase 1 feature 4): their corners and their
// cleanup. Nothing here is used by the document path.
#pragma once
#include <array>
#include <opencv2/core.hpp>

namespace fs {

struct BoardQuad {
    bool ok = false;                       // a reliable quadrilateral was found
    std::array<cv::Point2d, 4> pts;        // TL, TR, BR, BL in photo px, inside the photo
    double confidence = 0;                 // share of the outline backed by edges (0..1)
    double area = 0;                       // share of the photo inside the quad
    const char* method = "";               // "bright" (a lit screen), "contour" or "lines"
};

// The board or screen in a photo (BGR). Candidates: the largest bright region (a projected screen in a
// darker room), contours of the edge map, and long straight edges (a board with a partial frame). A
// candidate must be convex, cover 15-97% of the photo, have corner angles of 45-135 degrees and its
// outline backed by edges; the most edge-backed large one wins. ok = false: none is reliable (the caller
// offers the whole frame for manual corners). Corners are clamped 1 px inside the photo, so the
// perspective correction never samples outside it.
BoardQuad find_board(const cv::Mat& bgr);

// A projected slide with a dark background: the median luminance inside the quad (or the whole photo
// when !q.ok) is below 70.
bool board_is_dark(const cv::Mat& bgr, const BoardQuad& q);

// Board cleanup on the perspective-corrected board (BGRA): the background (board surface or screen)
// estimated from its brightest pixels per tile, robust to marker strokes, sticky notes and pictures, and
// divided out (shadows, uneven light and colour casts go, the board turns white); the background is set
// to pure white; marker colours are kept and their saturation raised; glare that left no strokes turns
// white. dark: a projected slide with a dark background and bright content: the background is evened out
// to its own median colour instead, and the content's contrast raised. strength 0..1 blends with the page.
cv::Mat board_filter(const cv::Mat& bgra, bool dark, float strength);

}  // namespace fs
