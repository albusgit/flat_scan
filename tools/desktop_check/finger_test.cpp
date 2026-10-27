// Tests of the finger fill (core/src/pipeline.cpp fill_finger, finger_over_text) on synthetic pages; run by
// `finger_test` (desktop_check CMake target), exit code 1 on a failure.
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
#include "pipeline.h"

using namespace fs;
static int failures = 0;
static void check(bool ok, const char* what, double v = NAN) {
    std::printf("%s %s", ok ? "PASS" : "FAIL", what);
    if (!std::isnan(v)) std::printf(" (%.2f)", v);
    std::printf("\n");
    if (!ok) ++failures;
}

int main() {
    // a cream page with lines of "text" (dark bars), a thumb (skin colour) coming in from the left edge
    const int w = 1200, h = 1600;
    cv::Mat page(h, w, CV_8UC4, cv::Scalar(222, 238, 246, 255));
    for (int y = 200; y < 1400; y += 60) cv::line(page, {80, y}, {1120, y}, cv::Scalar(40, 40, 40, 255), 10);
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8U);
    cv::ellipse(mask, {60, 700}, {260, 90}, 0, 0, 360, cv::Scalar(255), cv::FILLED);
    page.setTo(cv::Scalar(120, 150, 205, 255), mask);   // the thumb over the text
    cv::Mat paper(h, w, CV_8U, cv::Scalar(255));
    paper.setTo(0, mask);

    check(finger_over_text(page, mask, paper), "text runs under the thumb");
    cv::Mat filled = page.clone(), pf = paper.clone();
    fill_finger(filled, mask, &pf);
    // inside, away from the outline: paper colour, no stroke continued into it
    cv::Mat inner;
    cv::erode(mask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41)));
    cv::Scalar m = cv::mean(filled, inner);
    check(std::abs(m[0] - 222) < 8 && std::abs(m[1] - 238) < 8 && std::abs(m[2] - 246) < 8, "filled with the paper's colour", m[2]);
    cv::Mat g;
    cv::cvtColor(filled, g, cv::COLOR_BGRA2GRAY);
    double mn;
    cv::minMaxLoc(g, &mn, nullptr, nullptr, nullptr, inner);
    check(mn > 180, "no text invented inside", mn);
    check(cv::countNonZero(pf > 200 & mask) > 0.9 * cv::countNonZero(mask), "marked as paper");
    // the outline: grey dashes somewhere on the inner edge (not on the page border)
    cv::Mat ring;
    cv::dilate(mask, ring, cv::Mat::ones(7, 7, CV_8U));
    ring &= ~inner;
    cv::Mat grey;
    cv::inRange(filled, cv::Scalar(120, 120, 120, 0), cv::Scalar(160, 160, 160, 255), grey);
    check(cv::countNonZero(grey & ring) > 100, "the area is outlined (text was under it)");
    // the same thumb in the empty margin: no text there, so no outline
    cv::Mat blank(h, w, CV_8UC4, cv::Scalar(222, 238, 246, 255));
    blank.setTo(cv::Scalar(120, 150, 205, 255), mask);
    check(!finger_over_text(blank, mask, paper), "no text: not reported");
    cv::Mat bf = blank.clone();
    fill_finger(bf, mask, nullptr);
    cv::inRange(bf, cv::Scalar(120, 120, 120, 0), cv::Scalar(160, 160, 160, 255), grey);
    check(cv::countNonZero(grey) == 0, "no text: no outline");
    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
