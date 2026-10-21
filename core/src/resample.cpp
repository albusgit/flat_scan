#include "resample.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>

#pragma STDC FP_CONTRACT OFF

#include "gpu.h"

namespace fs {

GridMap GridMap::make(int out_w, int out_h, int stride) {
    GridMap g;
    g.out_w = out_w; g.out_h = out_h; g.stride = stride;
    g.gw = (out_w + stride - 1) / stride;   // samples at 0, s, 2s, ... < out_w  (np [::s])
    g.gh = (out_h + stride - 1) / stride;
    g.x.assign((size_t)g.gw * g.gh, 0.f);
    g.y.assign((size_t)g.gw * g.gh, 0.f);
    return g;
}

// Bilinear weights at x/stride exactly as cv::remap's float table (positions are multiples
// of 1/stride, exactly representable on the 1/32 grid for stride 4).
static inline float lerp4(const float* v, int gw, int gh, int ix, int iy, float a, float b) {
    int x0 = std::min(ix, gw - 1), x1 = std::min(ix + 1, gw - 1);
    int y0 = std::min(iy, gh - 1), y1 = std::min(iy + 1, gh - 1);
    float w0 = (1.f - b) * (1.f - a), w1 = (1.f - b) * a, w2 = b * (1.f - a), w3 = b * a;
    return v[(size_t)y0 * gw + x0] * w0 + v[(size_t)y0 * gw + x1] * w1 + v[(size_t)y1 * gw + x0] * w2 +
           v[(size_t)y1 * gw + x1] * w3;
}

void expand_rows(const GridMap& g, int y0, int y1, cv::Mat& mx, cv::Mat& my) {
    mx.create(y1 - y0, g.out_w, CV_32F);
    my.create(y1 - y0, g.out_w, CV_32F);
    const float inv = 1.f / (float)g.stride;
    for (int y = y0; y < y1; ++y) {
        int iy = y / g.stride;
        float b = (float)(y % g.stride) * inv;
        float* px = mx.ptr<float>(y - y0);
        float* py = my.ptr<float>(y - y0);
        for (int x = 0; x < g.out_w; ++x) {
            int ix = x / g.stride;
            float a = (float)(x % g.stride) * inv;
            px[x] = lerp4(g.x.data(), g.gw, g.gh, ix, iy, a, b);
            py[x] = lerp4(g.y.data(), g.gw, g.gh, ix, iy, a, b);
        }
    }
}

GridMap rescale_grid(const GridMap& g, int new_w, int new_h) {
    if (new_w == g.out_w && new_h == g.out_h) return g;
    GridMap o = GridMap::make(new_w, new_h, g.stride);
    double sx = (double)g.out_w / new_w, sy = (double)g.out_h / new_h;
    for (int i = 0; i < o.gh; ++i)
        for (int j = 0; j < o.gw; ++j) {
            double u = ((j * o.stride + 0.5) * sx - 0.5) / g.stride, v = ((i * o.stride + 0.5) * sy - 0.5) / g.stride;
            u = std::min(std::max(u, 0.0), (double)(g.gw - 1));
            v = std::min(std::max(v, 0.0), (double)(g.gh - 1));
            int x0 = std::min((int)u, g.gw - 2 < 0 ? 0 : g.gw - 2), y0 = std::min((int)v, g.gh - 2 < 0 ? 0 : g.gh - 2);
            int x1 = std::min(x0 + 1, g.gw - 1), y1 = std::min(y0 + 1, g.gh - 1);
            double a = u - x0, b = v - y0;
            auto L = [&](const std::vector<float>& f) {
                return (float)((f[(size_t)y0 * g.gw + x0] * (1 - a) + f[(size_t)y0 * g.gw + x1] * a) * (1 - b) +
                               (f[(size_t)y1 * g.gw + x0] * (1 - a) + f[(size_t)y1 * g.gw + x1] * a) * b);
            };
            o.X(i, j) = L(g.x);
            o.Y(i, j) = L(g.y);
        }
    return o;
}

void expand_full(const GridMap& g, cv::Mat& mx, cv::Mat& my) { expand_rows(g, 0, g.out_h, mx, my); }

cv::Mat resample_cpu(const cv::Mat& src, const GridMap& g, int band_rows) {
    cv::Mat out(g.out_h, g.out_w, src.type());
    cv::Scalar white = cv::Scalar::all(255);
    cv::Mat mx, my;
    for (int y0 = 0; y0 < g.out_h; y0 += band_rows) {
        int y1 = std::min(g.out_h, y0 + band_rows);
        expand_rows(g, y0, y1, mx, my);
        cv::Mat dst = out.rowRange(y0, y1);
        cv::remap(src, dst, mx, my, cv::INTER_CUBIC, cv::BORDER_CONSTANT, white);
    }
    return out;
}

// ---- OpenCV INTER_CUBIC fixed-point table (imgwarp.cpp: interpolateCubic, initInterTab2D)
static void interpolate_cubic(float x, float* c) {
    const float A = -0.75f;
    c[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    c[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    c[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];
}

const int16_t* bicubic_itab() {
    static int16_t itab[1024 * 16];
    static bool init = false;
    if (init) return itab;
    float tab1[32 * 4];
    for (int i = 0; i < 32; ++i) interpolate_cubic(i * (1.f / 32), tab1 + i * 4);
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 32; ++j) {
            int16_t* t = itab + (i * 32 + j) * 16;
            int isum = 0;
            for (int k1 = 0; k1 < 4; ++k1) {
                float vy = tab1[i * 4 + k1];
                for (int k2 = 0; k2 < 4; ++k2) {
                    float v = vy * tab1[j * 4 + k2];
                    int iv = (int)std::lrint(v * 32768.f);           // saturate_cast<short>(float)
                    iv = std::max(-32768, std::min(32767, iv));
                    t[k1 * 4 + k2] = (int16_t)iv;
                    isum += iv;
                }
            }
            if (isum != 32768) {                                     // same (off-centre) search as OpenCV
                int diff = isum - 32768, Mk1 = 2, Mk2 = 2, mk1 = 2, mk2 = 2;
                for (int k1 = 2; k1 < 4; ++k1)
                    for (int k2 = 2; k2 < 4; ++k2) {
                        if (t[k1 * 4 + k2] < t[mk1 * 4 + mk2]) mk1 = k1, mk2 = k2;
                        else if (t[k1 * 4 + k2] > t[Mk1 * 4 + Mk2]) Mk1 = k1, Mk2 = k2;
                    }
                if (diff < 0) t[Mk1 * 4 + Mk2] = (int16_t)(t[Mk1 * 4 + Mk2] - diff);
                else t[mk1 * 4 + mk2] = (int16_t)(t[mk1 * 4 + mk2] - diff);
            }
        }
    init = true;
    return itab;
}

cv::Mat resample_model(const cv::Mat& src, const GridMap& g) {
    const int16_t* itab = bicubic_itab();
    const int cn = src.channels(), W = src.cols, H = src.rows;
    cv::Mat out(g.out_h, g.out_w, src.type());
    cv::Mat mx, my;
    for (int y = 0; y < g.out_h; ++y) {
        expand_rows(g, y, y + 1, mx, my);
        uint8_t* D = out.ptr<uint8_t>(y);
        for (int x = 0; x < g.out_w; ++x, D += cn) {
            int X = (int)std::lrint(mx.at<float>(0, x) * 32.f), Y = (int)std::lrint(my.at<float>(0, x) * 32.f);
            int sx = std::max(-32768, std::min(32767, X >> 5)) - 1;
            int sy = std::max(-32768, std::min(32767, Y >> 5)) - 1;
            const int16_t* w = itab + ((Y & 31) * 32 + (X & 31)) * 16;
            bool inside = (unsigned)sx < (unsigned)std::max(W - 3, 0) && (unsigned)sy < (unsigned)std::max(H - 3, 0);
            if (!inside && (sx >= W || sx + 4 <= 0 || sy >= H || sy + 4 <= 0)) {
                for (int k = 0; k < cn; ++k) D[k] = 255;
                continue;
            }
            for (int k = 0; k < cn; ++k) {
                int sum;
                if (inside) {
                    sum = 0;
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c) sum += src.ptr<uint8_t>(sy + r)[(sx + c) * cn + k] * w[r * 4 + c];
                } else {
                    sum = 255 * 32768;
                    for (int r = 0; r < 4; ++r) {
                        int yy = sy + r;
                        if (yy < 0 || yy >= H) continue;
                        for (int c = 0; c < 4; ++c) {
                            int xx = sx + c;
                            if (xx < 0 || xx >= W) continue;
                            sum += (src.ptr<uint8_t>(yy)[xx * cn + k] - 255) * w[r * 4 + c];
                        }
                    }
                }
                D[k] = (uint8_t)std::max(0, std::min(255, (sum + (1 << 14)) >> 15));
            }
        }
    }
    return out;
}

// ---- GPU paths of the other bicubic stages
static thread_local bool t_gpu_stages = false;

GpuStageScope::GpuStageScope(bool on) : prev_(t_gpu_stages) { t_gpu_stages = on; }
GpuStageScope::~GpuStageScope() { t_gpu_stages = prev_; }
bool gpu_stages_enabled() { return t_gpu_stages; }

static std::atomic<bool> g_policy_dense{GpuStagePolicy().dense}, g_policy_offset{GpuStagePolicy().offset};
void set_gpu_stage_policy(const GpuStagePolicy& p) {
    g_policy_dense = p.dense;
    g_policy_offset = p.offset;
}
GpuStagePolicy gpu_stage_policy() {
    GpuStagePolicy p;
    p.dense = g_policy_dense.load();
    p.offset = g_policy_offset.load();
    return p;
}

static std::atomic<long> g_dense{0}, g_offset{0};
GpuStageCounts gpu_stage_counts() {
    GpuStageCounts c;
    c.dense = g_dense.load();
    c.offset = g_offset.load();
    return c;
}

bool gpu_stage_ready(const cv::Mat& src, int out_w, int out_h) {
    if (src.empty() || src.depth() != CV_8U || (src.channels() != 3 && src.channels() != 4) || out_w < 1 || out_h < 1) return false;
    Gpu& gpu = Gpu::instance();
    if (!gpu.available()) return false;
    int m = gpu.max_texture_size();
    return src.cols <= m && src.rows <= m && out_w <= m && out_h <= m;
}

// the GPU works on RGBA; the bicubic is per channel, so BGR in / BGR out is the same bytes
static cv::Mat as_rgba(const cv::Mat& src) {
    if (src.channels() == 4) return src;
    cv::Mat o;
    cv::cvtColor(src, o, cv::COLOR_BGR2BGRA);
    return o;
}
static void from_rgba(const cv::Mat& rgba, int channels, cv::Mat& out) {
    if (channels == 4) out = rgba;
    else cv::cvtColor(rgba, out, cv::COLOR_BGRA2BGR);
}

bool remap_dense_gpu(const cv::Mat& src, const cv::Mat& mx, const cv::Mat& my, cv::Mat& out) {
    if (!gpu_stage_policy().dense || !gpu_stage_ready(src, mx.cols, mx.rows)) return false;
    cv::Mat o;
    if (!Gpu::instance().remap_dense(as_rgba(src), mx, my, o)) return false;
    from_rgba(o, src.channels(), out);
    ++g_dense;
    return true;
}

bool remap_offset_gpu(const cv::Mat& src, const std::vector<float>& gx, const std::vector<float>& gy, int gw, int gh, int stride,
                      cv::Mat& out) {
    const int w = src.cols, h = src.rows;
    // the CPU path reads samples ix + 1, iy + 1 without clamping: the grid must cover them
    if (stride < 1 || gw < (w - 1) / stride + 2 || gh < (h - 1) / stride + 2) return false;
    if (!gpu_stage_policy().offset) return false;
    if ((size_t)gw * gh != gx.size() || gx.size() != gy.size() || !gpu_stage_ready(src, w, h)) return false;
    // the map is float arithmetic: without `precise` a driver may fuse it and round differently
    if (!Gpu::instance().precise()) return false;
    GridMap off;
    off.out_w = w; off.out_h = h; off.gw = gw; off.gh = gh; off.stride = stride;
    off.x = gx; off.y = gy;
    cv::Mat o;
    if (!Gpu::instance().remap_offset(as_rgba(src), off, o)) return false;
    from_rgba(o, src.channels(), out);
    ++g_offset;
    return true;
}

}  // namespace fs
