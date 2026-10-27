// Tests of core/src/board.cpp on synthetic photos (no test data needed): run by `board_test` (desktop_check
// CMake target); exit code 1 when a check fails.
//   * a board drawn in perspective on a textured wall is found within 2% of the photo diagonal, corners inside
//   * a lit projector screen in a dark room is found ("bright")
//   * a scene with no quadrilateral is not
//   * board_filter: the board turns white (> 245) under uneven light and a colour cast, marker strokes keep
//     their hue and get more saturated, a large sticky note keeps its colour; dark mode keeps a dark background
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include "board.h"

using namespace fs;
static int failures = 0;
static void check(bool ok, const char* what, double v = NAN) {
    std::printf("%s %s", ok ? "PASS" : "FAIL", what);
    if (!std::isnan(v)) std::printf(" (%.3f)", v);
    std::printf("\n");
    if (!ok) ++failures;
}

static cv::Mat scene(const std::vector<cv::Point2f>& quad, bool dark_room) {
    cv::Mat img(900, 1200, CV_8UC3);
    std::mt19937 rng(7);
    std::normal_distribution<float> n(0, 6);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            float base = dark_room ? 35 : 150 + 20 * std::sin(x * 0.05) * std::sin(y * 0.04);   // a textured wall
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(base + n(rng)), cv::saturate_cast<uchar>(base * 0.95f + n(rng)), cv::saturate_cast<uchar>(base * 0.9f + n(rng)));
        }
    std::vector<cv::Point> q;
    for (auto& p : quad) q.emplace_back((int)p.x, (int)p.y);
    cv::fillConvexPoly(img, q, dark_room ? cv::Scalar(235, 235, 240) : cv::Scalar(225, 228, 230));
    if (!dark_room) cv::polylines(img, std::vector<std::vector<cv::Point>>{q}, true, cv::Scalar(120, 120, 125), 10);   // the frame
    return img;
}

static double corner_err(const BoardQuad& b, const std::vector<cv::Point2f>& q) {
    double e = 0;
    for (int i = 0; i < 4; ++i) e = std::max(e, std::hypot(b.pts[i].x - q[i].x, b.pts[i].y - q[i].y));
    return e / std::hypot(1200.0, 900.0);
}

int main() {
    std::vector<cv::Point2f> quad = {{180, 140}, {1050, 90}, {1110, 780}, {130, 700}};
    BoardQuad b = find_board(scene(quad, false));
    check(b.ok, "board on a wall found");
    if (b.ok) check(corner_err(b, quad) < 0.02, "board corners within 2% of the diagonal", corner_err(b, quad));
    bool inside = true;
    for (auto& p : b.pts) inside &= p.x >= 1 && p.y >= 1 && p.x <= 1200 - 2 && p.y <= 900 - 2;
    check(inside, "corners inside the photo");

    std::vector<cv::Point2f> scr = {{300, 200}, {950, 230}, {930, 640}, {320, 660}};
    BoardQuad s = find_board(scene(scr, true));
    check(s.ok, "lit screen in a dark room found");
    if (s.ok) check(corner_err(s, scr) < 0.02, "screen corners within 2% of the diagonal", corner_err(s, scr));

    cv::Mat none(900, 1200, CV_8UC3);
    cv::randu(none, cv::Scalar::all(90), cv::Scalar::all(160));
    cv::GaussianBlur(none, none, cv::Size(0, 0), 3);
    check(!find_board(none).ok, "no quadrilateral in a noisy scene");

    // board_filter on a synthetic board: uneven light (left half 60% bright), a warm cast, red / blue strokes,
    // a large yellow sticky note
    cv::Mat board(800, 1200, CV_8UC4);
    for (int y = 0; y < board.rows; ++y)
        for (int x = 0; x < board.cols; ++x) {
            float l = 0.6f + 0.4f * x / board.cols;
            board.at<cv::Vec4b>(y, x) = cv::Vec4b((uchar)(200 * l), (uchar)(215 * l), (uchar)(232 * l), 255);
        }
    cv::line(board, {100, 200}, {1100, 220}, cv::Scalar(40, 40, 200, 255), 14);    // red marker (BGR)
    cv::line(board, {100, 500}, {1100, 480}, cv::Scalar(190, 80, 30, 255), 14);    // blue marker
    cv::rectangle(board, cv::Rect(700, 560, 260, 200), cv::Scalar(90, 215, 235, 255), cv::FILLED);   // sticky note
    cv::Mat out = board_filter(board, false, 1.f);
    auto px = [&](int x, int y) { return out.at<cv::Vec4b>(y, x); };
    auto minc = [](cv::Vec4b p) { return std::min(p[0], std::min(p[1], p[2])); };
    check(minc(px(60, 60)) > 245 && minc(px(1150, 380)) > 245 && minc(px(600, 700)) > 245, "board white on both the dim and the lit side");
    cv::Vec4b red = px(600, 210), blue = px(600, 490);
    auto sat = [](cv::Vec4b p) { int mx = std::max(p[0], std::max(p[1], p[2])), mn = std::min(p[0], std::min(p[1], p[2])); return mx ? (mx - mn) / (double)mx : 0.0; };
    check(red[2] > red[0] + 60 && red[2] > red[1] + 60, "red marker stays red");
    check(blue[0] > blue[2] + 60, "blue marker stays blue");
    cv::Vec4b red_in = board.at<cv::Vec4b>(210, 600);
    check(sat(red) > sat(red_in), "marker saturation raised", sat(red));
    cv::Vec4b note = px(830, 660);
    check(note[2] > note[0] + 60 && note[1] > note[0] + 40, "sticky note keeps its colour");
    // dark slide: background stays dark and even, bright text stays bright
    cv::Mat slide(600, 900, CV_8UC4);
    for (int y = 0; y < slide.rows; ++y)
        for (int x = 0; x < slide.cols; ++x) {
            float l = 0.7f + 0.3f * y / slide.rows;
            slide.at<cv::Vec4b>(y, x) = cv::Vec4b((uchar)(60 * l), (uchar)(30 * l), (uchar)(20 * l), 255);
        }
    cv::putText(slide, "Slide", {200, 330}, cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(240, 240, 240, 255), 12);
    cv::Mat d = board_filter(slide, true, 1.f);
    cv::Vec4b top = d.at<cv::Vec4b>(20, 20), bottom = d.at<cv::Vec4b>(580, 880);
    check(std::max(top[0], std::max(top[1], top[2])) < 90 && std::abs(top[0] - bottom[0]) < 12, "dark slide: background dark and even");
    cv::Mat g;
    cv::cvtColor(d, g, cv::COLOR_BGRA2GRAY);
    double mx;
    cv::minMaxLoc(g, nullptr, &mx);
    check(mx > 220, "dark slide: text bright");
    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
