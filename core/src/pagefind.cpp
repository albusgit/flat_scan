#include "pagefind.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

#include "nputil.h"

#pragma STDC FP_CONTRACT OFF

namespace fs {

namespace {

const float MEAN[3] = {0.485f, 0.456f, 0.406f}, STD[3] = {0.229f, 0.224f, 0.225f};

// Python's int(round(v)): round half to even
int py_round(double v) { return (int)std::nearbyint(v); }

// pagefind._fill: the largest outer contour of m (holes filled), 0/1
cv::Mat fill_largest(const cv::Mat& m01) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m01, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat out = cv::Mat::zeros(m01.size(), CV_8U);
    if (cs.empty()) return out;
    size_t best = 0;
    double ba = -1;
    for (size_t i = 0; i < cs.size(); ++i) {     // max(cs, key=contourArea): first maximum
        double a = cv::contourArea(cs[i]);
        if (a > ba) { ba = a; best = i; }
    }
    cv::drawContours(out, cs, (int)best, cv::Scalar(1), cv::FILLED);
    return out;
}

std::vector<cv::Point2d> largest_outline(const cv::Mat& m01) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m01, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<cv::Point2d> c;
    if (cs.empty()) return c;
    size_t best = 0;
    double ba = -1;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        if (a > ba) { ba = a; best = i; }
    }
    for (auto& p : cs[best]) c.emplace_back(p.x, p.y);
    return c;
}

double ptp(const std::vector<double>& v) {
    if (v.empty()) return 0;
    auto mm = std::minmax_element(v.begin(), v.end());
    return *mm.second - *mm.first;
}

// cv::GaussianBlur(col, out, Size(1, 0), sigma) of a float64 column, written out as the desktop OpenCV
// computes it: OpenCV's kernel, k0 x0 + sum_j kj (x-j + xj) in that order, BORDER_REFLECT_101.
// OpenCV's Android build fuses these steps into FMAs (measured: 93 of 300 values differ), which moved
// the refined outline, the corners and the surface fit between the desktop and the phone.
cv::Mat gauss_col64(const cv::Mat& col, double sigma) {
    const int n = col.rows, ks = cvRound(sigma * 4 * 2 + 1) | 1, r = ks / 2;   // createGaussianKernels, non-8U
    cv::Mat kk = cv::getGaussianKernel(ks, sigma, CV_64F);                     // bit-exact on every build
    const double* k = kk.ptr<double>() + r;
    const double* x = col.ptr<double>();
    cv::Mat out(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        double s = k[0] * x[i];
        for (int j = 1; j <= r; ++j)
            s += k[j] * (x[cv::borderInterpolate(i - j, n, cv::BORDER_REFLECT_101)] + x[cv::borderInterpolate(i + j, n, cv::BORDER_REFLECT_101)]);
        out.at<double>(i) = s;
    }
    return out;
}

// outline (working px) -> refined outline, full-resolution mask and contour
FoundPage finish_page(const cv::Mat& sm, const cv::Mat& m, double s, int W, int H, bool refine) {
    FoundPage fp;
    std::vector<cv::Point2d> pts = resample_closed(largest_outline(m), 2.0);
    if (refine && pts.size() >= 3) {
        RefineInfo ri;
        pts = refine_contour(sm, pts, &ri);
        fp.edge_strength = median(ri.strength);
        fp.edge_at_limit = ri.at_limit;
    }
    fp.mask = cv::Mat::zeros(H, W, CV_8U);
    std::vector<cv::Point2d> P(pts.size());
    std::vector<std::vector<cv::Point>> poly(1);
    for (size_t i = 0; i < pts.size(); ++i) {
        P[i] = cv::Point2d(pts[i].x / s, pts[i].y / s);
        poly[0].emplace_back((int)std::nearbyint(P[i].x * 4), (int)std::nearbyint(P[i].y * 4));   // np.round
    }
    if (!poly[0].empty()) cv::fillPoly(fp.mask, poly, cv::Scalar(1), cv::LINE_AA, 2);
    fp.mask = (fp.mask > 0) / 255;
    fp.contour = resample_closed(P, 2.0);
    return fp;
}

}  // namespace

std::vector<float> seg_input(const cv::Mat& bgr) {
    cv::Mat x;
    cv::resize(bgr, x, cv::Size(SEG_SIZE, SEG_SIZE), 0, 0, cv::INTER_AREA);
    std::vector<float> out(SEG_INPUT_LEN);
    const int plane = SEG_SIZE * SEG_SIZE;
    for (int y = 0; y < SEG_SIZE; ++y) {
        const uint8_t* q = x.ptr<uint8_t>(y);
        for (int xx = 0; xx < SEG_SIZE; ++xx)
            for (int c = 0; c < 3; ++c) {          // RGB plane c <- BGR channel 2 - c
                float v = (float)q[3 * xx + 2 - c] / 255.f;
                out[(size_t)c * plane + y * SEG_SIZE + xx] = (v - MEAN[c]) / STD[c];
            }
    }
    return out;
}

std::vector<float> seg2_input(const cv::Mat& bgr, SegLetterbox* lb) {
    const int H = bgr.rows, W = bgr.cols;
    const double s = (double)SEG2_SIZE / std::max(W, H);
    const int w = std::max(1, py_round(W * s)), h = std::max(1, py_round(H * s));
    const int x0 = (SEG2_SIZE - w) / 2, y0 = (SEG2_SIZE - h) / 2;
    cv::Mat x(SEG2_SIZE, SEG2_SIZE, CV_8UC3, cv::Scalar(104, 116, 124));   // net.PAD_BGR: the ImageNet mean
    cv::Mat r;
    cv::resize(bgr, r, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    r.copyTo(x(cv::Rect(x0, y0, w, h)));
    if (lb) *lb = SegLetterbox{s, x0, y0, w, h};
    std::vector<float> out(SEG2_INPUT_LEN);
    const int plane = SEG2_SIZE * SEG2_SIZE;
    for (int y = 0; y < SEG2_SIZE; ++y) {
        const uint8_t* q = x.ptr<uint8_t>(y);
        for (int xx = 0; xx < SEG2_SIZE; ++xx)
            for (int c = 0; c < 3; ++c) {          // RGB plane c <- BGR channel 2 - c
                float v = (float)q[3 * xx + 2 - c] / 255.f;
                out[(size_t)c * plane + y * SEG2_SIZE + xx] = (v - MEAN[c]) / STD[c];
            }
    }
    return out;
}

std::vector<float> seg2_to_seg(const float* out, const SegLetterbox& lb) {
    std::vector<float> lg(SEG_OUTPUT_LEN);
    const size_t plane2 = (size_t)SEG2_SIZE * SEG2_SIZE, plane = (size_t)SEG_SIZE * SEG_SIZE;
    for (int c = 0; c < 2; ++c) {   // paper, gutter (the edge channel is not read downstream yet)
        cv::Mat l(SEG2_SIZE, SEG2_SIZE, CV_32F, const_cast<float*>(out) + c * plane2);
        cv::Mat dst(SEG_SIZE, SEG_SIZE, CV_32F, lg.data() + c * plane);
        cv::resize(l(cv::Rect(lb.x0, lb.y0, lb.w, lb.h)), dst, dst.size(), 0, 0, cv::INTER_LINEAR);
    }
    return lg;
}

std::vector<cv::Point2d> resample_closed(const std::vector<cv::Point2d>& c, double step) {
    std::vector<cv::Point2d> out;
    size_t n = c.size();
    if (n == 0) return out;
    Vec s(n + 1), cx(n + 1), cy(n + 1);
    s[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        const cv::Point2d &a = c[i], &b = c[(i + 1) % n];
        s[i + 1] = s[i] + hypot_xy(b.x - a.x, b.y - a.y);     // np.cumsum: sequential
        cx[i] = a.x; cy[i] = a.y;
    }
    cx[n] = c[0].x; cy[n] = c[0].y;
    double L = s[n];
    size_t m = L > 0 ? (size_t)std::ceil(L / step) : 0;      // np.arange(0, L, step)
    out.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        double t = (double)i * step;
        out.emplace_back(interp(t, s, cx), interp(t, s, cy));
    }
    return out;
}

std::vector<cv::Point2d> refine_contour(const cv::Mat& img, const std::vector<cv::Point2d>& pts, RefineInfo* info,
                                        double R, double step, double alpha, double beta) {
    const int h = img.rows, w = img.cols;
    const int n = (int)pts.size();
    cv::Mat blur, lab8, lab;
    cv::GaussianBlur(img, blur, cv::Size(0, 0), 1.0);
    cv::cvtColor(blur, lab8, cv::COLOR_BGR2Lab);
    lab8.convertTo(lab, CV_32F);
    std::vector<cv::Mat> ch(3), gx(3), gy(3);
    cv::split(lab, ch);
    for (int k = 0; k < 3; ++k) {
        cv::Scharr(ch[k], gx[k], CV_32F, 1, 0);
        cv::Scharr(ch[k], gy[k], CV_32F, 0, 1);
        gx[k] /= 32;
        gy[k] /= 32;
    }
    // tangents from the outline smoothed along its length (wrapped by 15 points)
    std::vector<cv::Point2d> sm(n);
    for (int k = 0; k < 2; ++k) {
        int pad = std::min(15, n);
        cv::Mat col(n + 2 * pad, 1, CV_64F), bl;
        for (int i = 0; i < pad; ++i) col.at<double>(i) = k ? pts[n - pad + i].y : pts[n - pad + i].x;
        for (int i = 0; i < n; ++i) col.at<double>(pad + i) = k ? pts[i].y : pts[i].x;
        for (int i = 0; i < pad; ++i) col.at<double>(pad + n + i) = k ? pts[i].y : pts[i].x;
        bl = gauss_col64(col, 4);
        for (int i = 0; i < n; ++i) (k ? sm[i].y : sm[i].x) = bl.at<double>(pad + i);
    }
    std::vector<cv::Point2d> nr(n);
    for (int i = 0; i < n; ++i) {
        cv::Point2d t = sm[(i + 1) % n] - sm[(i - 1 + n) % n];
        double l = std::sqrt(t.x * t.x + t.y * t.y) + 1e-9;
        t = cv::Point2d(t.x / l, t.y / l);
        nr[i] = cv::Point2d(t.y, -t.x);
    }
    const int K = (int)std::floor((2 * R + 1e-6) / step) + 1;   // np.arange(-R, R + 1e-6, step)
    std::vector<double> offs(K);
    for (int j = 0; j < K; ++j) offs[j] = -R + j * step;
    cv::Mat Xf(n, K, CV_32F), Yf(n, K, CV_32F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < K; ++j) {
            Xf.at<float>(i, j) = (float)(pts[i].x + nr[i].x * offs[j]);
            Yf.at<float>(i, j) = (float)(pts[i].y + nr[i].y * offs[j]);
        }
    std::vector<cv::Mat> gxs(3), gys(3);
    for (int k = 0; k < 3; ++k) {
        cv::remap(gx[k], gxs[k], Xf, Yf, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(gy[k], gys[k], Xf, Yf, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    std::vector<double> G((size_t)n * K), gmax(n);
    for (int i = 0; i < n; ++i) {
        double mx = -1;
        for (int j = 0; j < K; ++j) {
            double ss = 0;
            for (int k = 0; k < 3; ++k) {
                double p = (double)gxs[k].at<float>(i, j) * nr[i].x + (double)gys[k].at<float>(i, j) * nr[i].y;
                ss += p * p;
            }
            G[(size_t)i * K + j] = std::sqrt(ss);
            mx = std::max(mx, G[(size_t)i * K + j]);
        }
        gmax[i] = mx;
    }
    const double norm = percentile(gmax, 75) + 1e-6;
    std::vector<double> cost((size_t)n * K);
    for (int i = 0; i < n; ++i) {
        // points on the frame border stay put (the page runs off the photo there)
        bool border = pts[i].x < 3 || pts[i].y < 3 || pts[i].x > w - 4 || pts[i].y > h - 4;
        for (int j = 0; j < K; ++j) {
            double& c = cost[(size_t)i * K + j];
            if (border) c = std::fabs(offs[j]) < 1e-6 ? 0 : 10;
            else {
                double q = offs[j] / R;
                c = -std::min(G[(size_t)i * K + j] / norm, 1.5) + alpha * (q * q);
            }
        }
    }
    // Viterbi; start at the point with the clearest optimum
    int start = 0;
    double best_span = -1;
    for (int i = 0; i < n; ++i) {
        auto mm = std::minmax_element(cost.begin() + (size_t)i * K, cost.begin() + (size_t)(i + 1) * K);
        double span = *mm.second - *mm.first;
        if (span > best_span) { best_span = span; start = i; }
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = (start + i) % n;
    std::vector<double> acc(K), nacc(K);
    std::vector<int> back((size_t)n * K, 0);
    for (int j = 0; j < K; ++j) acc[j] = cost[(size_t)order[0] * K + j];
    for (int i = 1; i < n; ++i) {
        const double* Ci = &cost[(size_t)order[i] * K];
        for (int b = 0; b < K; ++b) {
            int ba = 0;
            double bv = acc[0] + beta * std::fabs(offs[0] - offs[b]);
            for (int a = 1; a < K; ++a) {
                double v = acc[a] + beta * std::fabs(offs[a] - offs[b]);
                if (v < bv) { bv = v; ba = a; }
            }
            back[(size_t)i * K + b] = ba;
            nacc[b] = bv + Ci[b];
        }
        acc.swap(nacc);
    }
    std::vector<int> sel(n);
    sel[n - 1] = (int)(std::min_element(acc.begin(), acc.end()) - acc.begin());
    for (int i = n - 1; i > 0; --i) sel[i - 1] = back[(size_t)i * K + sel[i]];
    std::vector<double> o(n), strength(n);
    for (int i = 0; i < n; ++i) {
        o[order[i]] = offs[sel[i]];
        strength[order[i]] = G[(size_t)order[i] * K + sel[i]] / norm;
    }
    std::vector<cv::Point2d> out(n);
    int lim = 0;
    for (int i = 0; i < n; ++i) {
        out[i] = cv::Point2d(pts[i].x + nr[i].x * o[i], pts[i].y + nr[i].y * o[i]);
        lim += std::fabs(o[i]) >= R - 1e-6;
    }
    if (info) {
        info->offset = o;
        info->strength = strength;
        info->edge_norm = norm;
        info->at_limit = n ? (double)lim / n : 0;
    }
    return out;
}

PageSelection select_page(const cv::Mat& p, const cv::Mat& g, double thr) {
    PageSelection r;
    const int h = p.rows, w = p.cols;
    cv::Mat b = (p > thr) / 255, lab, st, cen;
    int n = cv::connectedComponentsWithStats(b, lab, st, cen, 8, CV_32S);
    if (n < 2) {
        r.mask = cv::Mat::zeros(p.size(), CV_8U);
        r.empty = true;
        return r;
    }
    int k = 1;
    double best = -1e300;
    for (int i = 1; i < n; ++i) {
        double d = hypot_xy((cen.at<double>(i, 0) - w / 2.0) / w, (cen.at<double>(i, 1) - h / 2.0) / h);
        double sc = st.at<int>(i, cv::CC_STAT_AREA) * (1 - 0.6 * d);
        if (sc > best) { best = sc; k = i; }
    }
    cv::Mat m = fill_largest((lab == k) / 255);
    r.area_frac = cv::countNonZero(m) / ((double)h * w);
    // book spread: a long gutter line inside the page
    std::vector<int> xs, ys;
    std::vector<double> wts;
    for (int y = 0; y < h; ++y) {
        const float* gp = g.ptr<float>(y);
        const uint8_t* mp = m.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x)
            if (gp[x] > 0.5f && mp[x]) { xs.push_back(x); ys.push_back(y); wts.push_back(gp[x]); }
    }
    if (xs.size() > 30) {
        size_t N = xs.size();
        Vec wx(N), wy(N);
        for (size_t i = 0; i < N; ++i) { wx[i] = xs[i] * wts[i]; wy[i] = ys[i] * wts[i]; }
        double sw = np_sum(wts);
        cv::Point2d c(np_sum(wx) / sw, np_sum(wy) / sw);     // np.average(.., weights=wts)
        double sxx = 0, sxy = 0, syy = 0;
        for (size_t i = 0; i < N; ++i) {
            double px = xs[i] - c.x, py = ys[i] - c.y;
            sxx += px * wts[i] * px; sxy += px * wts[i] * py; syy += py * wts[i] * py;
        }
        sxx /= sw; sxy /= sw; syy /= sw;
        // eigh of [[sxx, sxy], [sxy, syy]]: ascending eigenvalues, t = major axis, nrm = minor
        double tr = 0.5 * (sxx + syy), dd = std::sqrt(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy);
        double ev0 = tr - dd, ev1 = tr + dd;
        cv::Point2d t;
        if (std::fabs(sxy) > 1e-12 * (std::fabs(sxx) + std::fabs(syy))) t = cv::Point2d(sxy, ev1 - sxx);
        else t = sxx >= syy ? cv::Point2d(1, 0) : cv::Point2d(0, 1);
        t /= std::sqrt(t.x * t.x + t.y * t.y);
        cv::Point2d nrm(-t.y, t.x);
        std::vector<double> pt(N);
        for (size_t i = 0; i < N; ++i) pt[i] = (xs[i] - c.x) * t.x + (ys[i] - c.y) * t.y;
        double ext = ptp(pt);
        std::vector<cv::Point> mpts;
        cv::findNonZero(m, mpts);
        std::vector<double> qt(mpts.size());
        for (size_t i = 0; i < mpts.size(); ++i) qt[i] = (mpts[i].x - c.x) * t.x + (mpts[i].y - c.y) * t.y;
        double mext = ptp(qt);
        for (size_t i = 0; i < mpts.size(); ++i) qt[i] = (mpts[i].x - c.x) * nrm.x + (mpts[i].y - c.y) * nrm.y;
        const double across = mext > 0 ? ptp(qt) / mext : 0;
        bool straight = std::sqrt(ev0) < 0.03 * std::max(h, w);
        if (ext > 0.55 * mext && straight) {
            cv::Mat A = cv::Mat::zeros(m.size(), CV_8U), B = cv::Mat::zeros(m.size(), CV_8U);
            for (auto& q : mpts) {
                bool side = (q.x - c.x) * nrm.x + (q.y - c.y) * nrm.y > 0;
                (side ? A : B).at<uint8_t>(q) = 1;
            }
            A = fill_largest(A);
            B = fill_largest(B);
            int a = cv::countNonZero(A), bb = cv::countNonZero(B);
            r.spread = true;
            r.gutter_c = c;
            r.gutter_t = t;
            r.halves[0] = a; r.halves[1] = bb;
            r.across = across;
            // the paper's share of each frame side (a band 1% of the frame wide)
            const int band = std::max(3, (int)std::lround(0.01 * std::max(h, w)));
            double touch = 0;
            for (int side = 0; side < 4; ++side) {
                bool horiz = side < 2;
                int len = horiz ? w : h, hit = 0;
                for (int i = 0; i < len; ++i) {
                    bool any = false;
                    for (int d = 0; d < band && !any; ++d) {
                        int x = horiz ? i : (side == 2 ? d : w - 1 - d), y = horiz ? (side == 0 ? d : h - 1 - d) : i;
                        any = m.at<uint8_t>(y, x) != 0;
                    }
                    hit += any;
                }
                touch = std::max(touch, (double)hit / len);
            }
            if (std::min(a, bb) < 0.3 * std::max(a, bb)) {
                m = a > bb ? A : B;   // the other page is only a sliver at the frame edge
                r.pages = 1;
            } else if (across < 0.87 && touch < 0.05) {
                // a page folded down the middle, all in the frame: two pages side by side are wider
                // across the gutter than along it (DocUNet folds 0.58-0.80; synthetic spreads in the
                // frame >= 0.94, below that only when the frame cuts the paper)
                r.pages = 1;
                r.folded = true;
            } else {
                r.pages = 2;
                r.other = a > bb ? B : A;
                m = a > bb ? A : B;
            }
        }
    }
    r.mask = m;
    return r;
}

PageFindResult find_page_net(const cv::Mat& img, const float* logits, bool refine) {
    PageFindResult res;
    const int H = img.rows, W = img.cols;
    const double s = (double)PAGEFIND_WORK / std::max(H, W);
    const int sw = py_round(W * s), sh = py_round(H * s);
    cv::Mat sm;
    cv::resize(img, sm, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
    // Predictor.probs: logits upsampled (bicubic) before the sigmoid: sub-pixel edges
    cv::Mat pg[2];
    for (int c = 0; c < 2; ++c) {
        cv::Mat l(SEG_SIZE, SEG_SIZE, CV_32F, const_cast<float*>(logits) + (size_t)c * SEG_SIZE * SEG_SIZE), up;
        cv::resize(l, up, cv::Size(sw, sh), 0, 0, cv::INTER_CUBIC);
        for (int y = 0; y < up.rows; ++y) {
            float* q = up.ptr<float>(y);
            for (int x = 0; x < up.cols; ++x) q[x] = 1.f / (1.f + (float)std::exp(-(double)q[x]));   // expf rounds differently per libm, this does not
        }
        pg[c] = up;
    }
    const cv::Mat& p = pg[0];
    PageSelection sel = select_page(p, pg[1]);
    res.empty = sel.empty;
    res.spread = sel.spread;
    res.area_frac = sel.area_frac;
    res.gutter_c = sel.gutter_c / s;
    res.gutter_t = sel.gutter_t;
    res.halves[0] = sel.halves[0]; res.halves[1] = sel.halves[1];
    res.across = sel.across;
    res.folded = sel.folded;
    cv::Mat m = sel.mask;
    if (cv::countNonZero(m) < 0.02 * m.total()) {   // nothing page-like: whole frame (projector screen, close-up)
        res.whole_frame = true;
        m = cv::Mat::zeros(sm.size(), CV_8U);
        if (sw > 4 && sh > 4) m(cv::Rect(2, 2, sw - 4, sh - 4)).setTo(1);
        sel.other.release();
    }
    // confidence: how decided the net is in a band around the outline
    cv::Mat band;
    cv::morphologyEx(m, band, cv::MORPH_GRADIENT, cv::Mat::ones(9, 9, CV_8U));
    double acc = 0;
    size_t cnt = 0;
    for (int y = 0; y < band.rows; ++y)
        for (int x = 0; x < band.cols; ++x)
            if (band.at<uint8_t>(y, x)) { acc += std::fabs(p.at<float>(y, x) - 0.5f); ++cnt; }
    res.net_margin = cnt ? acc / cnt * 2 : 0.0;
    bool snap = refine && !res.whole_frame;
    res.pages.push_back(finish_page(sm, m, s, W, H, snap));
    if (!sel.other.empty()) res.pages.push_back(finish_page(sm, sel.other, s, W, H, snap));
    res.edge_strength = res.pages[0].edge_strength;
    res.edge_at_limit = res.pages[0].edge_at_limit;
    return res;
}

}  // namespace fs
