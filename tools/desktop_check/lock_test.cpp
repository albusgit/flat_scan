// Tests of the shape lock (core/src/pipeline.cpp lock_shapes) on synthetic maps: run by `lock_test` (desktop_check
// CMake target); exit code 1 when a check fails.
//   * a map that bends a block (a bump in the area change, as a line stage straightening a bottle's label bands) and
//     turns the page 0.5 degrees: inside the locked block the local area is the page's within 0.1 %, and the block
//     turns with the paper around it (within 0.02 degrees)
//   * beyond the blend band, and along the page's outline, the map is the line stage's as it was
//   * an empty lock changes nothing
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "pipeline.h"

using namespace fs;
static int failures = 0;
static void check(bool ok, const char* what, double v = NAN) {
    std::printf("%s %s", ok ? "PASS" : "FAIL", what);
    if (!std::isnan(v)) std::printf(" (%.5f)", v);
    std::printf("\n");
    if (!ok) ++failures;
}

static const int W = 1200, H = 1600;
static const double TURN = 0.5 * CV_PI / 180;
static const cv::Rect BLOCK(600, 800, 360, 400);

// identity turned TURN about the page centre, plus a bump (a Gaussian push to the right) centred in the block
static GridMap bent() {
    GridMap g = GridMap::make(W, H, 4);
    const double cx = W / 2.0, cy = H / 2.0, bx = BLOCK.x + BLOCK.width / 2.0, by = BLOCK.y + BLOCK.height / 2.0;
    for (int i = 0; i < g.gh; ++i)
        for (int j = 0; j < g.gw; ++j) {
            const double x = 4.0 * j, y = 4.0 * i, ax = x - cx, ay = y - cy;
            const double b = 12 * std::exp(-((x - bx) * (x - bx) + (y - by) * (y - by)) / (2 * 90.0 * 90.0));
            g.X(i, j) = (float)(cx + std::cos(TURN) * ax - std::sin(TURN) * ay + b);
            g.Y(i, j) = (float)(cy + std::sin(TURN) * ax + std::cos(TURN) * ay);
        }
    return g;
}

int main() {
    const GridMap m0 = bent();
    cv::Mat lock = cv::Mat::zeros(H, W, CV_8U);
    lock(BLOCK).setTo(255);
    GridMap m = m0;
    const double band = 0.06 * std::min(W, H);
    const double share = lock_shapes(m, lock, band);
    check(share > 0.05 && share < 0.12, "the block's share of the grid is held", share);
    // local area (Jacobian determinant, central differences) inside the block, 2 nodes in from its edge
    double worst = 0, ang = 0;
    int n = 0;
    for (int i = BLOCK.y / 4 + 2; i < (BLOCK.y + BLOCK.height) / 4 - 2; ++i)
        for (int j = BLOCK.x / 4 + 2; j < (BLOCK.x + BLOCK.width) / 4 - 2; ++j) {
            const double xu = (m.X(i, j + 1) - m.X(i, j - 1)) / 8, yu = (m.Y(i, j + 1) - m.Y(i, j - 1)) / 8;
            const double xv = (m.X(i + 1, j) - m.X(i - 1, j)) / 8, yv = (m.Y(i + 1, j) - m.Y(i - 1, j)) / 8;
            worst = std::max(worst, std::fabs(xu * yv - xv * yu - 1));
            ang += std::atan2(yu, xu);
            ++n;
        }
    check(worst < 1e-3, "inside the block the local area is the page's (max |det J - 1|)", worst);
    check(std::fabs(ang / n - TURN) * 180 / CV_PI < 0.02, "the block turns with the page around it (degrees off)",
          std::fabs(ang / n - TURN) * 180 / CV_PI);
    // beyond the band and on the outline: unchanged
    double far = 0, edge = 0;
    for (int i = 0; i < m.gh; ++i)
        for (int j = 0; j < m.gw; ++j) {
            const double d = std::hypot(m.X(i, j) - m0.x[(size_t)i * m.gw + j], m.Y(i, j) - m0.y[(size_t)i * m.gw + j]);
            const int x = 4 * j, y = 4 * i;
            const int dx = std::max({BLOCK.x - x, x - (BLOCK.x + BLOCK.width - 1), 0}), dy = std::max({BLOCK.y - y, y - (BLOCK.y + BLOCK.height - 1), 0});
            if (std::hypot(dx, dy) > band + 8) far = std::max(far, d);
            if (i == 0 || j == 0 || i == m.gh - 1 || j == m.gw - 1) edge = std::max(edge, d);
        }
    check(far == 0, "beyond the blend band the map is the line stage's (max px moved)", far);
    check(edge == 0, "the page's outline is the line stage's (max px moved)", edge);
    GridMap e = m0;
    lock_shapes(e, cv::Mat::zeros(H, W, CV_8U), band);
    bool same = e.x == m0.x && e.y == m0.y;
    check(same, "an empty lock changes nothing");
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
