// Tests of core/src/upright.cpp on synthetic pages (no test data needed): run by `upright_test` (desktop_check CMake
// target); exit code 1 when a check fails.
//   * a level page: every mode leaves it as it is (sliders within 0.05 degrees / 0.005)
//   * a page turned 3 degrees: Level turns it back within 0.1 degrees
//   * a keystoned and turned page (vertical 0.2, horizontal -0.15, rotate 2.5): Full makes its rows and rules level
//     and plumb within 0.2 degrees (median), Auto at least halves the error
//   * Guided: two drawn guides along leaning verticals come out plumb within 0.1 degrees
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "upright.h"

using namespace fs;
static int failures = 0;
static void check(bool ok, const char* what, double v = NAN) {
    std::printf("%s %s", ok ? "PASS" : "FAIL", what);
    if (!std::isnan(v)) std::printf(" (%.4f)", v);
    std::printf("\n");
    if (!ok) ++failures;
}

static const int W = 900, H = 1270;
static const double HW = (double)H / W;

// a printed page: a title, justified text rows in two columns, a table with ruled lines, a framed picture
static cv::Mat page() {
    cv::Mat p(H, W, CV_8UC3, cv::Scalar(245, 245, 242));
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> word(18, 70), gap(8, 14);
    cv::rectangle(p, {90, 80, 520, 34}, cv::Scalar(30, 30, 30), -1);
    for (int col = 0; col < 2; ++col)
        for (int y = 160; y < 700; y += 26) {
            int x = 70 + col * 400, end = x + 360;
            while (x < end) {
                int w = std::min(word(rng), end - x);
                cv::rectangle(p, {x, y, w, 11}, cv::Scalar(40, 40, 40), -1);
                x += w + gap(rng);
            }
        }
    for (int r = 0; r <= 6; ++r) cv::line(p, {70, 760 + 50 * r}, {830, 760 + 50 * r}, cv::Scalar(20, 20, 20), 2);
    for (int c = 0; c <= 4; ++c) cv::line(p, {70 + 190 * c, 760}, {70 + 190 * c, 1060}, cv::Scalar(20, 20, 20), 2);
    cv::rectangle(p, {560, 1100, 270, 120}, cv::Scalar(90, 120, 160), -1);
    return p;
}

// the page seen through the sliders' map (upright_apply), as the app's PageTransform would show it
static cv::Mat warped(const cv::Mat& src, double rot, double v, double h) {
    std::vector<cv::Point2f> a, b;
    for (auto q : {cv::Point2d(0, 0), cv::Point2d(1, 0), cv::Point2d(1, 1), cv::Point2d(0, 1)}) {
        cv::Point2d o = upright_apply(rot, v, h, HW, q);
        a.emplace_back((float)(q.x * W), (float)(q.y * H));
        b.emplace_back((float)(o.x * W), (float)(o.y * H));
    }
    cv::Mat out;
    cv::warpPerspective(src, out, cv::getPerspectiveTransform(a, b), src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(245, 245, 242));
    return out;
}

// the median angle (degrees) of the page's lines from their nearest axis after the sliders s
static double median_off(const std::vector<UprightLine>& ls, const double s[3]) {
    std::vector<double> off;
    for (const auto& l : ls) {
        cv::Point2d a = upright_apply(s[0], s[1], s[2], HW, {l.x0, l.y0}), b = upright_apply(s[0], s[1], s[2], HW, {l.x1, l.y1});
        double ang = std::atan2((b.y - a.y) * HW, b.x - a.x) * 180 / CV_PI;
        while (ang > 45) ang -= 90;
        while (ang < -45) ang += 90;
        off.push_back(std::fabs(ang));
    }
    if (off.empty()) return 90;
    std::nth_element(off.begin(), off.begin() + off.size() / 2, off.end());
    return off[off.size() / 2];
}

int main() {
    const cv::Mat flat = page();
    const std::vector<UprightLine> lf = upright_lines(flat);
    check(lf.size() >= 20, "a printed page gives its rows and rules", (double)lf.size());
    for (UprightMode m : {UprightMode::Level, UprightMode::Vertical, UprightMode::Full, UprightMode::Auto}) {
        double s[3] = {9, 9, 9};
        bool ok = upright_solve(lf, m, HW, s);
        char what[96];
        std::snprintf(what, sizeof what, "level page, mode %d: no change", (int)m);
        check(ok && std::fabs(s[0]) < 0.05 && std::fabs(s[1]) < 0.005 && std::fabs(s[2]) < 0.005, what, std::fabs(s[0]) + std::fabs(s[1]) + std::fabs(s[2]));
    }

    const cv::Mat turned = warped(flat, 3.0, 0, 0);
    double s[3] = {0, 0, 0};
    upright_solve(upright_lines(turned), UprightMode::Level, HW, s);
    check(std::fabs(s[0] + 3.0) < 0.1 && s[1] == 0 && s[2] == 0, "turned 3 degrees: Level turns it back", s[0]);

    const cv::Mat keyed = warped(flat, 2.5, 0.2, -0.15);
    const std::vector<UprightLine> lk = upright_lines(keyed);
    const double none[3] = {0, 0, 0};
    const double before = median_off(lk, none);
    double full[3] = {0, 0, 0}, aut[3] = {0, 0, 0};
    upright_solve(lk, UprightMode::Full, HW, full);
    upright_solve(lk, UprightMode::Auto, HW, aut);
    std::printf("  keystoned page: median off-axis %.3f deg; Full -> %.3f (rot %.2f v %.3f h %.3f); Auto -> %.3f\n", before,
                median_off(lk, full), full[0], full[1], full[2], median_off(lk, aut));
    check(median_off(lk, full) < 0.2, "keystoned page: Full levels and plumbs it", median_off(lk, full));
    check(median_off(lk, aut) < 0.5 * before, "keystoned page: Auto at least halves the error", median_off(lk, aut));

    // two guides along the keystoned page's outer verticals (the table's first and last column rules)
    std::vector<UprightLine> guides;
    for (double x : {70.0, 830.0}) {
        cv::Point2d a = upright_apply(2.5, 0.2, -0.15, HW, {x / W, 760.0 / H}), b = upright_apply(2.5, 0.2, -0.15, HW, {x / W, 1060.0 / H});
        guides.push_back({a.x, a.y, b.x, b.y, 1});
    }
    double gs[3] = {0, 0, 0};
    upright_solve(guides, UprightMode::Guided, HW, gs);
    double worst = 0;
    for (const auto& gl : guides) {
        cv::Point2d a = upright_apply(gs[0], gs[1], gs[2], HW, {gl.x0, gl.y0}), b = upright_apply(gs[0], gs[1], gs[2], HW, {gl.x1, gl.y1});
        worst = std::max(worst, std::fabs(std::atan2(b.x - a.x, (b.y - a.y) * HW)) * 180 / CV_PI);
    }
    check(worst < 0.1, "Guided: two leaning verticals come out plumb", worst);

    std::printf("%s\n", failures ? "SOME CHECKS FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
