#include "board.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace fs {

namespace {

double angle_deg(cv::Point2d a, cv::Point2d b, cv::Point2d c) {   // angle at b
    cv::Point2d u = a - b, v = c - b;
    double d = u.dot(v) / std::max(1e-9, std::hypot(u.x, u.y) * std::hypot(v.x, v.y));
    return std::acos(std::max(-1.0, std::min(1.0, d))) * 180.0 / CV_PI;
}

// TL, TR, BR, BL from four points.
std::array<cv::Point2d, 4> order(std::vector<cv::Point2d> p) {
    std::array<cv::Point2d, 4> o;
    auto sum = [](const cv::Point2d& q) { return q.x + q.y; };
    auto diff = [](const cv::Point2d& q) { return q.x - q.y; };
    o[0] = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); });
    o[2] = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); });
    o[1] = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); });
    o[3] = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); });
    return o;
}

// Share of the quad's outline with an edge pixel within 3 px ([use]: the sides counted; side k runs from
// corner k to corner k + 1).
double edge_support(const cv::Mat& edges_dil, const std::array<cv::Point2d, 4>& q, const bool use[4] = nullptr) {
    int hit = 0, n = 0;
    for (int k = 0; k < 4; ++k) {
        if (use && !use[k]) continue;
        cv::Point2d a = q[k], b = q[(k + 1) % 4];
        int steps = std::max(8, (int)std::hypot(b.x - a.x, b.y - a.y));
        for (int i = 0; i <= steps; ++i) {
            cv::Point2d p = a + (b - a) * (double(i) / steps);
            int x = (int)std::lround(p.x), y = (int)std::lround(p.y);
            if (x < 0 || y < 0 || x >= edges_dil.cols || y >= edges_dil.rows) continue;
            ++n;
            if (edges_dil.at<uint8_t>(y, x)) ++hit;
        }
    }
    return n ? double(hit) / n : 0;
}

struct Cand {
    std::array<cv::Point2d, 4> q;
    double area, support;
    const char* method;
    int real_sides = 4;   // sides that are edges in the photo (not the photo's border)
};

bool valid(const std::array<cv::Point2d, 4>& q, double frame_area, double* area_share) {
    std::vector<cv::Point2f> pf;
    for (auto& p : q) pf.emplace_back((float)p.x, (float)p.y);
    if (!cv::isContourConvex(pf)) return false;
    double a = std::fabs(cv::contourArea(pf)) / frame_area;
    *area_share = a;
    if (a < 0.06 || a > 0.97) return false;
    for (int k = 0; k < 4; ++k) {
        double ang = angle_deg(q[(k + 3) % 4], q[k], q[(k + 1) % 4]);
        if (ang < 45 || ang > 135) return false;
    }
    return true;
}

void add_poly(const std::vector<cv::Point>& c, double frame_area, const cv::Mat& edges, const char* method, std::vector<Cand>& out) {
    std::vector<cv::Point> hull, poly;
    cv::convexHull(c, hull);
    double peri = cv::arcLength(hull, true);
    for (double eps : {0.02, 0.035, 0.05}) {
        cv::approxPolyDP(hull, poly, eps * peri, true);
        if (poly.size() == 4) break;
    }
    if (poly.size() != 4) return;
    std::vector<cv::Point2d> p;
    for (auto& v : poly) p.emplace_back(v.x, v.y);
    auto q = order(p);
    double area;
    if (!valid(q, frame_area, &area)) return;
    out.push_back({q, area, edge_support(edges, q), method});
}

// Intersection of the lines through a-b and c-d.
bool cross(cv::Vec4i l1, cv::Vec4i l2, cv::Point2d& out) {
    cv::Point2d p(l1[0], l1[1]), r(l1[2] - l1[0], l1[3] - l1[1]);
    cv::Point2d q(l2[0], l2[1]), s(l2[2] - l2[0], l2[3] - l2[1]);
    double d = r.x * s.y - r.y * s.x;
    if (std::fabs(d) < 1e-9) return false;
    double t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / d;
    out = p + r * t;
    return true;
}

}  // namespace

BoardQuad find_board(const cv::Mat& bgr) {
    BoardQuad res;
    if (bgr.empty()) return res;
    const double k = 640.0 / std::max(bgr.cols, bgr.rows);
    cv::Mat small, gray, blur, edges, edges_dil;
    cv::resize(bgr, small, cv::Size(), k, k, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::Mat otsu_bin;   // (a named output: a temporary cv::Mat() is a fixed-size output and throws)
    double otsu = cv::threshold(blur, otsu_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Canny(blur, edges, 0.4 * otsu, otsu);
    cv::dilate(edges, edges_dil, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    const double frame = double(small.cols) * small.rows;
    std::vector<Cand> cands;

    // 1. the largest bright region (a lit screen, a board under light in a darker room)
    {
        cv::Mat bright;
        cv::threshold(blur, bright, otsu, 255, cv::THRESH_BINARY);
        cv::morphologyEx(bright, bright, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(bright, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::sort(cs.begin(), cs.end(), [](auto& a, auto& b) { return cv::contourArea(a) > cv::contourArea(b); });
        for (size_t i = 0; i < std::min<size_t>(3, cs.size()); ++i) add_poly(cs[i], frame, edges_dil, "bright", cands);
    }
    // 2. closed outlines in the edge map (a framed board)
    {
        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(edges_dil, cs, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (auto& c : cs)
            if (cv::contourArea(c) > 0.15 * frame) add_poly(c, frame, edges_dil, "contour", cands);
    }
    // 3. long straight edges (a board whose frame is partly hidden): every pair of near-horizontal and
    //    near-vertical lines among the longest ones, kept when the quad they make is valid
    {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 60, 0.2 * std::min(small.cols, small.rows), 12);
        std::vector<cv::Vec4i> hz, vt;
        auto len = [](const cv::Vec4i& l) { return std::hypot(l[2] - l[0], l[3] - l[1]); };
        std::sort(lines.begin(), lines.end(), [&](auto& a, auto& b) { return len(a) > len(b); });
        for (auto& l : lines) {
            double a = std::fmod(std::atan2(l[3] - l[1], l[2] - l[0]) * 180 / CV_PI + 180, 180);
            if ((a < 25 || a > 155) && hz.size() < 10) hz.push_back(l);
            else if (a > 65 && a < 115 && vt.size() < 10) vt.push_back(l);
        }
        // a board wider (taller) than the photo: the photo's own edges stand in for its sides; they are not
        // counted as edge support, and at least two real sides are needed
        const size_t real_hz = hz.size(), real_vt = vt.size();
        hz.push_back(cv::Vec4i(0, 1, small.cols - 1, 1));
        hz.push_back(cv::Vec4i(0, small.rows - 2, small.cols - 1, small.rows - 2));
        vt.push_back(cv::Vec4i(1, 0, 1, small.rows - 1));
        vt.push_back(cv::Vec4i(small.cols - 2, 0, small.cols - 2, small.rows - 1));
        auto ymid = [](const cv::Vec4i& l) { return (l[1] + l[3]) / 2.0; };
        auto xmid = [](const cv::Vec4i& l) { return (l[0] + l[2]) / 2.0; };
        if (std::getenv("FS_BOARD_DEBUG")) {
            std::fprintf(stderr, "lines %zu: %zu horizontal, %zu vertical\n", lines.size(), hz.size(), vt.size());
            for (auto& l : hz) std::fprintf(stderr, "  h y=%.0f x %d..%d\n", ymid(l), std::min(l[0], l[2]), std::max(l[0], l[2]));
            for (auto& l : vt) std::fprintf(stderr, "  v x=%.0f y %d..%d\n", xmid(l), std::min(l[1], l[3]), std::max(l[1], l[3]));
        }
        for (size_t i = 0; i < hz.size(); ++i)
            for (size_t j = 0; j < hz.size(); ++j) {
                if (ymid(hz[j]) - ymid(hz[i]) < 0.3 * small.rows) continue;   // i top, j bottom
                for (size_t m = 0; m < vt.size(); ++m)
                    for (size_t n = 0; n < vt.size(); ++n) {
                        if (xmid(vt[n]) - xmid(vt[m]) < 0.3 * small.cols) continue;   // m left, n right
                        std::array<cv::Point2d, 4> q;
                        if (!(cross(hz[i], vt[m], q[0]) && cross(hz[i], vt[n], q[1]) && cross(hz[j], vt[n], q[2]) && cross(hz[j], vt[m], q[3]))) continue;
                        bool inside = true;
                        for (auto& p : q) inside &= p.x > -0.03 * small.cols && p.y > -0.03 * small.rows && p.x < 1.03 * small.cols && p.y < 1.03 * small.rows;
                        // sides: 0 top (i), 1 right (n), 2 bottom (j), 3 left (m)
                        const bool use[4] = {i < real_hz, n < real_vt, j < real_hz, m < real_vt};
                        if (use[0] + use[1] + use[2] + use[3] < 2) continue;
                        double area;
                        if (inside && valid(q, frame, &area)) cands.push_back({q, area, edge_support(edges_dil, q, use), "lines", use[0] + use[1] + use[2] + use[3]});
                    }
            }
    }
    // the best: well backed by edges, large, bright against its surroundings. A box drawn on the board or a
    // small framed board far away is not what a photo of a board usually shows: line and contour candidates must
    // cover a quarter of the photo, a bright region 15 %, unless they stand out (a lit screen in a dark room)
    const Cand* best = nullptr;
    // how much brighter the inside is than the rest of the photo: a board under light or a lit screen is the
    // brightest large surface in the photo; a quad that takes in the floor, people or a dark stage is not.
    // A small quad (a screen far away) is kept only when it stands out that clearly.
    // (measured on a 160 px copy: thousands of line quads are scored)
    cv::Mat tiny;
    const double kt = 160.0 / std::max(blur.cols, blur.rows);
    cv::resize(blur, tiny, cv::Size(), kt, kt, cv::INTER_AREA);
    auto contrast = [&](const Cand& c) {
        cv::Mat m = cv::Mat::zeros(tiny.size(), CV_8U);
        std::vector<cv::Point> pts;
        for (auto& p : c.q) pts.emplace_back((int)std::lround(p.x * kt), (int)std::lround(p.y * kt));
        cv::fillConvexPoly(m, pts, 255);
        const double n = cv::countNonZero(m), all = (double)tiny.total();
        if (n < 1 || all - n < 0.02 * all) return 0.0;   // (all of the photo: nothing to compare with)
        return cv::mean(tiny, m)[0] - cv::mean(tiny, m == 0)[0];
    };
    // a quad with all four sides in the photo beats one that leans on the photo's border
    auto base_score = [](const Cand& c) { const double r = c.real_sides / 4.0; return c.area * c.support * c.support * r * r; };
    auto score = [&](const Cand& c, double con) { return base_score(c) * std::max(0.1, 1.0 + 4.0 * con / 255.0); };
    const bool debug = std::getenv("FS_BOARD_DEBUG") != nullptr;
    double best_score = -1;
    for (auto& c : cands) {
        if (c.support < 0.6) continue;
        if (!debug && base_score(c) * 5.0 <= best_score) continue;   // cannot win even at full contrast
        const double con = contrast(c);
        if (debug) std::fprintf(stderr, "cand %s area %.2f support %.2f contrast %.0f\n", c.method, c.area, c.support, con);
        const bool stands_out = con >= 40;
        if (!stands_out && c.area < (std::string(c.method) == "bright" ? 0.15 : 0.25)) continue;
        const double sc = score(c, con);
        if (sc > best_score) best = &c, best_score = sc;
    }
    if (!best) return res;
    res.ok = true;
    res.confidence = best->support;
    res.area = best->area;
    res.method = best->method;
    for (int i = 0; i < 4; ++i) {
        cv::Point2d p = best->q[i] / k;
        res.pts[i] = cv::Point2d(std::min(bgr.cols - 2.0, std::max(1.0, p.x)), std::min(bgr.rows - 2.0, std::max(1.0, p.y)));
    }
    return res;
}

bool board_is_dark(const cv::Mat& bgr, const BoardQuad& q) {
    if (bgr.empty()) return false;
    cv::Mat small, gray;
    const double k = 320.0 / std::max(bgr.cols, bgr.rows);
    cv::resize(bgr, small, cv::Size(), k, k, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask(gray.size(), CV_8U, cv::Scalar(q.ok ? 0 : 255));
    if (q.ok) {
        std::vector<cv::Point> poly;
        for (auto& p : q.pts) poly.emplace_back((int)std::lround(p.x * k), (int)std::lround(p.y * k));
        cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    }
    std::vector<uint8_t> v;
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
            if (mask.at<uint8_t>(y, x)) v.push_back(gray.at<uint8_t>(y, x));
    if (v.empty()) return false;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2] < 70;
}

cv::Mat board_filter(const cv::Mat& bgra, bool dark, float strength) {
    CV_Assert(bgra.type() == CV_8UC4);
    const int w = bgra.cols, h = bgra.rows;
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    // background per channel from a small copy: per tile (about 4% of the long side) the 90th percentile
    // (the 10th for a dark slide) of each channel, tiles far from their neighbours' level dropped
    // (sticky notes, pictures, a person in front), then filled and smoothed
    const double ks = 256.0 / std::max(w, h);
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(std::max(8, (int)std::lround(w * ks)), std::max(8, (int)std::lround(h * ks))), 0, 0, cv::INTER_AREA);
    const int tile = 10;
    const int tw = (small.cols + tile - 1) / tile, th = (small.rows + tile - 1) / tile;
    cv::Mat grid(th, tw, CV_32FC3);
    std::vector<uint8_t> v[3];
    for (int ty = 0; ty < th; ++ty)
        for (int tx = 0; tx < tw; ++tx) {
            for (auto& c : v) c.clear();
            for (int y = ty * tile; y < std::min(small.rows, (ty + 1) * tile); ++y)
                for (int x = tx * tile; x < std::min(small.cols, (tx + 1) * tile); ++x) {
                    const cv::Vec3b& p = small.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < 3; ++c) v[c].push_back(p[c]);
                }
            cv::Vec3f g;
            for (int c = 0; c < 3; ++c) {
                size_t n = v[c].size();
                size_t at = dark ? n / 10 : (n * 9) / 10;
                std::nth_element(v[c].begin(), v[c].begin() + std::min(n - 1, at), v[c].end());
                g[c] = v[c][std::min(n - 1, at)];
            }
            grid.at<cv::Vec3f>(ty, tx) = g;
        }
    // robust: a tile darker (lighter, dark mode) than the median of its 5x5 neighbourhood by > 15% is content
    cv::Mat lum;
    cv::transform(grid, lum, cv::Matx13f(0.114f, 0.587f, 0.299f));
    cv::Mat med;
    {
        cv::Mat l8;
        lum.convertTo(l8, CV_8U);
        cv::medianBlur(l8, med, 5);
        med.convertTo(med, CV_32F);
    }
    // ... and so is a tile whose colour (chroma) is far from its neighbourhood's: a sticky note can be
    // brighter than the board
    cv::Mat chroma_t(th, tw, CV_32F);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            cv::Vec3f g = grid.at<cv::Vec3f>(y, x);
            chroma_t.at<float>(y, x) = std::max(g[0], std::max(g[1], g[2])) - std::min(g[0], std::min(g[1], g[2]));
        }
    cv::Mat cmed;
    {
        cv::Mat c8;
        chroma_t.convertTo(c8, CV_8U);
        cv::medianBlur(c8, cmed, 5);
        cmed.convertTo(cmed, CV_32F);
    }
    cv::Mat keep(th, tw, CV_8U);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            float l = lum.at<float>(y, x), m = med.at<float>(y, x);
            bool level = dark ? (l < m * 1.15f + 4) : (l > m * 0.85f - 4);
            keep.at<uint8_t>(y, x) = level && std::fabs(chroma_t.at<float>(y, x) - cmed.at<float>(y, x)) < 25;
        }
    // fill dropped tiles from kept neighbours (repeated 3x3 averaging), then smooth
    cv::Mat filled = grid.clone(), mask = keep.clone();
    for (int it = 0; it < std::max(tw, th) && cv::countNonZero(mask) < tw * th; ++it) {
        cv::Mat sum = cv::Mat::zeros(th, tw, CV_32FC3), cnt = cv::Mat::zeros(th, tw, CV_32F);
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
                if (!mask.at<uint8_t>(y, x)) continue;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int yy = y + dy, xx = x + dx;
                        if (yy < 0 || xx < 0 || yy >= th || xx >= tw) continue;
                        sum.at<cv::Vec3f>(yy, xx) += filled.at<cv::Vec3f>(y, x);
                        cnt.at<float>(yy, xx) += 1;
                    }
            }
        cv::Mat next = mask.clone();
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                if (!mask.at<uint8_t>(y, x) && cnt.at<float>(y, x) > 0) {
                    filled.at<cv::Vec3f>(y, x) = sum.at<cv::Vec3f>(y, x) / cnt.at<float>(y, x);
                    next.at<uint8_t>(y, x) = 1;
                }
        mask = next;
    }
    if (cv::countNonZero(keep) == 0) filled = grid;
    cv::GaussianBlur(filled, filled, cv::Size(0, 0), 1.2);
    cv::Mat bg;
    cv::resize(filled, bg, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);

    cv::Mat out(h, w, CV_8UC4);
    cv::Vec3f dark_ref;
    if (dark) {
        // the slide's own background colour, evened out across the screen
        std::vector<float> c[3];
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x)
                if (keep.at<uint8_t>(y, x)) for (int k = 0; k < 3; ++k) c[k].push_back(grid.at<cv::Vec3f>(y, x)[k]);
        for (int k = 0; k < 3; ++k) {
            if (c[k].empty()) { dark_ref[k] = 20; continue; }
            std::nth_element(c[k].begin(), c[k].begin() + c[k].size() / 2, c[k].end());
            dark_ref[k] = c[k][c[k].size() / 2];
        }
    }
    const float s = std::min(1.f, std::max(0.f, strength));
    // the colour decisions (is it white board? how vivid?) from a lightly smoothed copy: sensor and JPEG
    // noise on a pale wall must not turn into coloured speckle
    cv::Mat smooth;
    cv::GaussianBlur(bgr, smooth, cv::Size(0, 0), std::max(1.0, w / 800.0));
    cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& R) {
        for (int y = R.start; y < R.end; ++y) {
            const cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* sm = smooth.ptr<cv::Vec3b>(y);
            const cv::Vec3f* b = bg.ptr<cv::Vec3f>(y);
            cv::Vec4b* o = out.ptr<cv::Vec4b>(y);
            for (int x = 0; x < w; ++x) {
                float r[3];
                if (!dark) {
                    // board: divide by the background (1 = board), white above 0.9, strokes darkened a little
                    float rs[3];
                    for (int k = 0; k < 3; ++k) {
                        r[k] = p[x][k] / std::max(16.f, b[x][k]);
                        rs[k] = sm[x][k] / std::max(16.f, b[x][k]);
                    }
                    float l = 0.114f * r[0] + 0.587f * r[1] + 0.299f * r[2];
                    float ls = 0.114f * rs[0] + 0.587f * rs[1] + 0.299f * rs[2];
                    float chroma = std::max(rs[0], std::max(rs[1], rs[2])) - std::min(rs[0], std::min(rs[1], rs[2]));
                    // markers more vivid: saturation up only where there is a stroke (darker than the board)
                    // (or a coloured area: a sticky note, a filled shape)
                    float stroke = std::max(std::min(1.f, std::max(0.f, (0.93f - ls) / 0.2f)), std::min(1.f, std::max(0.f, (chroma - 0.1f) / 0.15f)));
                    for (int k = 0; k < 3; ++k) r[k] = l + (r[k] - l) * (1.f + 0.8f * stroke);
                    float t = std::min(1.f, std::max(0.f, (ls - 0.80f) / 0.12f));
                    float white = t * t * (3 - 2 * t) * (1.f - std::min(1.f, std::max(0.f, chroma - 0.06f) * 5.f));
                    for (int k = 0; k < 3; ++k) {
                        float v = std::pow(std::min(1.f, std::max(0.f, r[k])), 1.6f);
                        r[k] = (v + (1.f - v) * white) * 255.f;
                    }
                    // a dark background (the room around a board, a whole-frame photo) is not board: kept as it is
                    float bl = 0.114f * b[x][0] + 0.587f * b[x][1] + 0.299f * b[x][2];
                    float keep = std::min(1.f, std::max(0.f, (110.f - bl) / 50.f));
                    for (int k = 0; k < 3; ++k) r[k] = r[k] * (1 - keep) + p[x][k] * keep;
                } else {
                    // dark slide: content over an even background of the slide's own colour, contrast up
                    for (int k = 0; k < 3; ++k) r[k] = dark_ref[k] + (p[x][k] - b[x][k]) * 1.35f;
                    float l = 0.114f * r[0] + 0.587f * r[1] + 0.299f * r[2];
                    for (int k = 0; k < 3; ++k) r[k] = l + (r[k] - l) * 1.4f;
                }
                for (int k = 0; k < 3; ++k) {
                    float v = p[x][k] + s * (r[k] - p[x][k]);
                    o[x][k] = (uint8_t)std::lrint(std::min(255.f, std::max(0.f, v)));
                }
                o[x][3] = 255;
            }
        }
    });
    return out;
}

}  // namespace fs
