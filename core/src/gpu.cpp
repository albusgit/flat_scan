#include "gpu.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "cleanup.h"
#include "log.h"

namespace fs {

// ---------------------------------------------------------------- shaders
static const char* kResampleCS = R"(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 16, local_size_y = 8) in;
uniform highp usampler2D uSrc;      // RGBA8UI source photo (texelFetch only)
uniform highp sampler2D uGx;        // R32F backward map grid, x (source px)
uniform highp sampler2D uGy;        // R32F backward map grid, y
layout(std430, binding = 0) readonly buffer Tab { int tab[]; };      // OpenCV BicubicTab_i
layout(std430, binding = 1) writeonly buffer Dst { uint px[]; };     // packed RGBA8 band
uniform ivec2 uOut;                 // dense output size
uniform ivec2 uGrid;                // grid samples (gw, gh)
uniform int uStride;
uniform int uRow0;
uniform int uRows;
uniform ivec2 uSrcSize;

float grid(highp sampler2D t, int ix, int iy, float a, float b) {
    int x0 = min(ix, uGrid.x - 1), x1 = min(ix + 1, uGrid.x - 1);
    int y0 = min(iy, uGrid.y - 1), y1 = min(iy + 1, uGrid.y - 1);
    float w0 = (1.0 - b) * (1.0 - a), w1 = (1.0 - b) * a, w2 = b * (1.0 - a), w3 = b * a;
    return texelFetch(t, ivec2(x0, y0), 0).r * w0 + texelFetch(t, ivec2(x1, y0), 0).r * w1 +
           texelFetch(t, ivec2(x0, y1), 0).r * w2 + texelFetch(t, ivec2(x1, y1), 0).r * w3;
}

void main() {
    int x = int(gl_GlobalInvocationID.x), r = int(gl_GlobalInvocationID.y);
    if (x >= uOut.x || r >= uRows) return;
    int y = uRow0 + r;
    float inv = 1.0 / float(uStride);
    float a = float(x % uStride) * inv, b = float(y % uStride) * inv;
    int ix = x / uStride, iy = y / uStride;
    float mx = clamp(grid(uGx, ix, iy, a, b), -40000.0, 40000.0);
    float my = clamp(grid(uGy, ix, iy, a, b), -40000.0, 40000.0);
    int X = int(roundEven(mx * 32.0)), Y = int(roundEven(my * 32.0));
    int sx = clamp(X >> 5, -32768, 32767) - 1, sy = clamp(Y >> 5, -32768, 32767) - 1;
    int wb = ((Y & 31) * 32 + (X & 31)) * 16;
    int W = uSrcSize.x, H = uSrcSize.y;
    bool inside = uint(sx) < uint(max(W - 3, 0)) && uint(sy) < uint(max(H - 3, 0));
    uvec4 o;
    if (!inside && (sx >= W || sx + 4 <= 0 || sy >= H || sy + 4 <= 0)) {
        o = uvec4(255u);
    } else {
        ivec4 sum = inside ? ivec4(0) : ivec4(255 * 32768);
        for (int i = 0; i < 4; ++i) {
            int yy = sy + i;
            if (yy < 0 || yy >= H) continue;
            for (int j = 0; j < 4; ++j) {
                int xx = sx + j;
                if (xx < 0 || xx >= W) continue;
                ivec4 s = ivec4(texelFetch(uSrc, ivec2(xx, yy), 0));
                int w = tab[wb + i * 4 + j];
                sum += inside ? s * w : (s - 255) * w;
            }
        }
        o = uvec4(clamp((sum + 16384) >> 15, 0, 255));
    }
    px[r * uOut.x + x] = o.r | (o.g << 8) | (o.b << 16) | (o.a << 24);
}
)";

// Parity/debug only: the dense maps exactly as the resample shader reconstructs them.
static const char* kExpandCS = R"(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 16, local_size_y = 8) in;
uniform highp sampler2D uGx;
uniform highp sampler2D uGy;
layout(std430, binding = 1) writeonly buffer Dst { vec2 m[]; };
uniform ivec2 uOut;
uniform ivec2 uGrid;
uniform int uStride;
uniform int uRow0;
uniform int uRows;
float grid(highp sampler2D t, int ix, int iy, float a, float b) {
    int x0 = min(ix, uGrid.x - 1), x1 = min(ix + 1, uGrid.x - 1);
    int y0 = min(iy, uGrid.y - 1), y1 = min(iy + 1, uGrid.y - 1);
    float w0 = (1.0 - b) * (1.0 - a), w1 = (1.0 - b) * a, w2 = b * (1.0 - a), w3 = b * a;
    return texelFetch(t, ivec2(x0, y0), 0).r * w0 + texelFetch(t, ivec2(x1, y0), 0).r * w1 +
           texelFetch(t, ivec2(x0, y1), 0).r * w2 + texelFetch(t, ivec2(x1, y1), 0).r * w3;
}
void main() {
    int x = int(gl_GlobalInvocationID.x), r = int(gl_GlobalInvocationID.y);
    if (x >= uOut.x || r >= uRows) return;
    int y = uRow0 + r;
    float inv = 1.0 / float(uStride);
    float a = float(x % uStride) * inv, b = float(y % uStride) * inv;
    m[r * uOut.x + x] = vec2(grid(uGx, x / uStride, y / uStride, a, b), grid(uGy, x / uStride, y / uStride, a, b));
}
)";

// ---------------------------------------------------------------- context
Gpu& Gpu::instance() {
    static Gpu g;
    return g;
}

bool Gpu::available() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!tried_) {
        tried_ = true;
        ok_ = init_locked();
        if (!ok_) LOGI("GPU unavailable: %s", info_.c_str());
        else LOGI("GPU: %s", info_.c_str());
    }
    return ok_;
}

bool Gpu::make_current() {
    return eglMakeCurrent((EGLDisplay)dpy_, (EGLSurface)surf_, (EGLSurface)surf_, (EGLContext)ctx_) == EGL_TRUE;
}
void Gpu::release() { eglMakeCurrent((EGLDisplay)dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); }

bool Gpu::init_locked() {
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) { info_ = "no EGL display"; return false; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint cfg_pb[] = {EGL_RENDERABLE_TYPE, 0x40 /*EGL_OPENGL_ES3_BIT*/, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                             EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
    EGLConfig cfg = nullptr;
    EGLint n = 0;
    EGLSurface surf = EGL_NO_SURFACE;
    if (eglChooseConfig(dpy, cfg_pb, &cfg, 1, &n) && n == 1) {
        const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surf = eglCreatePbufferSurface(dpy, cfg, pb);
    }
    if (surf == EGL_NO_SURFACE) {   // surfaceless (EGL_KHR_surfaceless_context)
        const EGLint cfg_sl[] = {EGL_RENDERABLE_TYPE, 0x40, EGL_NONE};
        if (!eglChooseConfig(dpy, cfg_sl, &cfg, 1, &n) || n != 1) { info_ = "no ES3 EGL config"; return false; }
    }
    const EGLint ca[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ca);
    if (ctx == EGL_NO_CONTEXT) { info_ = "eglCreateContext failed"; return false; }
    dpy_ = dpy; ctx_ = ctx; surf_ = surf;
    if (!make_current()) { info_ = "eglMakeCurrent failed"; return false; }
    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    const char* ren = (const char*)glGetString(GL_RENDERER);
    const char* ver = (const char*)glGetString(GL_VERSION);
    info_ = std::string(ren ? ren : "?") + " | " + (ver ? ver : "?");
    if (major < 3 || (major == 3 && minor < 1)) { info_ += " (needs GLES 3.1)"; release(); return false; }
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex_);
    prog_resample_ = program("resample", kResampleCS);
    prog_expand_ = program("expand", kExpandCS);
    bool ok = prog_resample_ != 0 && prog_expand_ != 0 && init_cleanup_programs();
    release();
    return ok;
}

unsigned Gpu::program(const char* name, const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        info_ += std::string(" | compile ") + name + ": " + log;
        LOGI("shader %s: %s", name, log);
        glDeleteShader(sh);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, sh);
    glLinkProgram(p);
    glDeleteShader(sh);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        info_ += std::string(" | link ") + name + ": " + log;
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// ---------------------------------------------------------------- helpers
static GLuint tex_r32f(int w, int h, const float* data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, w, h);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

GLuint gpu_tex_rgba8ui(const cv::Mat& m) {
    CV_Assert(m.type() == CV_8UC4 && m.isContinuous());
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, m.cols, m.rows);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m.cols, m.rows, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, m.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static void set_i(GLuint p, const char* n, int v) { glUniform1i(glGetUniformLocation(p, n), v); }
static void set_i2(GLuint p, const char* n, int a, int b) { glUniform2i(glGetUniformLocation(p, n), a, b); }

// Dispatch `prog` over a w x h output in row bands; the shader writes packed RGBA8 to the
// SSBO at binding 1 (px[(y - uRow0) * uOut.x + x]). Bands keep the SSBO under the
// guaranteed GL_MAX_SHADER_STORAGE_BLOCK_SIZE.
bool Gpu::run_banded(unsigned prog, int w, int h, cv::Mat& out) {
    GLint max_block = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_block);
    int band = std::max(1, std::min(h, (int)(std::min<long long>(max_block, 8 << 20) / (4LL * w))));
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)band * w * 4, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf);
    glUseProgram(prog);
    set_i2(prog, "uOut", w, h);
    out.create(h, w, CV_8UC4);
    bool ok = true;
    for (int y0 = 0; y0 < h && ok; y0 += band) {
        int rows = std::min(band, h - y0);
        set_i(prog, "uRow0", y0);
        set_i(prog, "uRows", rows);
        glDispatchCompute((w + 15) / 16, (rows + 7) / 8, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        void* p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)rows * w * 4, GL_MAP_READ_BIT);
        if (!p) { ok = false; break; }
        std::memcpy(out.ptr(y0), p, (size_t)rows * w * 4);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glDeleteBuffers(1, &buf);
    return ok && glGetError() == GL_NO_ERROR;
}

bool Gpu::resample(const cv::Mat& src, const GridMap& g, cv::Mat& out) {
    if (!available()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (src.cols > max_tex_ || src.rows > max_tex_ || g.gw > max_tex_ || g.gh > max_tex_) return false;
    if (!make_current()) return false;
    cv::Mat s = src.isContinuous() ? src : src.clone();
    GLuint tsrc = gpu_tex_rgba8ui(s);
    GLuint tgx = tex_r32f(g.gw, g.gh, g.x.data()), tgy = tex_r32f(g.gw, g.gh, g.y.data());
    const int16_t* itab = bicubic_itab();
    std::vector<int32_t> tab(itab, itab + 1024 * 16);
    GLuint bufs[1];
    glGenBuffers(1, bufs);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, tab.size() * 4, tab.data(), GL_STATIC_DRAW);
    glUseProgram(prog_resample_);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tsrc); set_i(prog_resample_, "uSrc", 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tgx); set_i(prog_resample_, "uGx", 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tgy); set_i(prog_resample_, "uGy", 2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufs[0]);
    set_i2(prog_resample_, "uGrid", g.gw, g.gh);
    set_i(prog_resample_, "uStride", g.stride);
    set_i2(prog_resample_, "uSrcSize", s.cols, s.rows);
    bool ok = run_banded(prog_resample_, g.out_w, g.out_h, out);
    ok = ok && glGetError() == GL_NO_ERROR;
    glDeleteBuffers(1, bufs);
    GLuint ts[3] = {tsrc, tgx, tgy};
    glDeleteTextures(3, ts);
    glFinish();
    release();
    return ok;
}

// ---------------------------------------------------------------- cleanup
static const char* kDownCS = R"(#version 310 es
precision highp int;
layout(local_size_x = 16, local_size_y = 8) in;
uniform highp usampler2D uSrc;
layout(rgba8ui, binding = 0) writeonly uniform highp uimage2D uDst;
uniform ivec2 uSize;   // quarter size
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= uSize.x || q.y >= uSize.y) return;
    uvec4 s = uvec4(0u);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) s += texelFetch(uSrc, 4 * q + ivec2(c, r), 0);
    imageStore(uDst, q, (s + 8u) >> 4);
}
)";

static const char* kDilateCS = R"(#version 310 es
precision highp int;
layout(local_size_x = 16, local_size_y = 8) in;
uniform highp usampler2D uSrc;
layout(rgba8ui, binding = 0) writeonly uniform highp uimage2D uDst;
uniform ivec2 uSize;
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= uSize.x || q.y >= uSize.y) return;
    uvec4 m = uvec4(0u);
    for (int r = -1; r <= 1; ++r)
        for (int c = -1; c <= 1; ++c) m = max(m, texelFetch(uSrc, clamp(q + ivec2(c, r), ivec2(0), uSize - 1), 0));
    imageStore(uDst, q, m);
}
)";

// exact 11x11 median with replicated border: per channel, the 61st smallest value found by
// a bitwise search on the count of values below a candidate
static const char* kMedianCS = R"(#version 310 es
precision highp int;
layout(local_size_x = 16, local_size_y = 8) in;
uniform highp usampler2D uSrc;
layout(std430, binding = 1) writeonly buffer Dst { uint px[]; };
uniform ivec2 uOut;
uniform int uRow0;
uniform int uRows;
void main() {
    int x = int(gl_GlobalInvocationID.x), r = int(gl_GlobalInvocationID.y);
    if (x >= uOut.x || r >= uRows) return;
    ivec2 q = ivec2(x, uRow0 + r);
    uvec4 res = uvec4(0u);
    for (int bit = 7; bit >= 0; --bit) {
        uvec4 cand = res | uvec4(1u << uint(bit));
        uvec4 cnt = uvec4(0u);
        for (int dy = -5; dy <= 5; ++dy)
            for (int dx = -5; dx <= 5; ++dx) {
                uvec4 v = texelFetch(uSrc, clamp(q + ivec2(dx, dy), ivec2(0), uOut - 1), 0);
                cnt += uvec4(lessThan(v, cand));
            }
        res = mix(res, cand, lessThan(cnt, uvec4(61u)));
    }
    px[r * uOut.x + x] = res.r | (res.g << 8) | (res.b << 16) | (res.a << 24);
}
)";

// shared by the count and tone passes; mirrors tone_cpu in cleanup.cpp
#define TONE_COMMON R"(
precision highp float;
precision highp int;
uniform highp usampler2D uSrc;
uniform highp sampler2D uIllum;
uniform vec3 uWb;
uniform ivec2 uImg;
float luma(vec3 v) { return 0.114 * v.r + 0.587 * v.g + 0.299 * v.b; }   // BGRA order
float illumAt(int x, int y) {
    ivec2 n = textureSize(uIllum, 0);
    float fx = clamp((float(x) + 0.5) * 0.25 - 0.5, 0.0, float(n.x - 1));
    float fy = clamp((float(y) + 0.5) * 0.25 - 0.5, 0.0, float(n.y - 1));
    int x0 = int(fx), y0 = int(fy);
    int x1 = min(x0 + 1, n.x - 1), y1 = min(y0 + 1, n.y - 1);
    float a = fx - float(x0), b = fy - float(y0);
    return (texelFetch(uIllum, ivec2(x0, y0), 0).r * (1.0 - a) + texelFetch(uIllum, ivec2(x1, y0), 0).r * a) * (1.0 - b) +
           (texelFetch(uIllum, ivec2(x0, y1), 0).r * (1.0 - a) + texelFetch(uIllum, ivec2(x1, y1), 0).r * a) * b;
}
vec3 normalised(int x, int y) {
    vec3 q = vec3(texelFetch(uSrc, ivec2(x, y), 0).rgb);
    float S = illumAt(x, y);
    return vec3(q.r / max(S * uWb.r, 1.0), q.g / max(S * uWb.g, 1.0), q.b / max(S * uWb.b, 1.0));
}
)"

static const char* kCountCS = "#version 310 es\nlayout(local_size_x = 16, local_size_y = 8) in;\n" TONE_COMMON R"(
layout(r32ui, binding = 0) writeonly uniform highp uimage2D uCnt;
uniform ivec2 uQ;
void main() {
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (q.x >= uQ.x || q.y >= uQ.y) return;
    uint n = 0u;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            int y = 4 * q.y + r, x = 4 * q.x + c;
            if (y >= uImg.y || x >= uImg.x) continue;
            if (luma(normalised(x, y)) < 0.55) n++;
        }
    imageStore(uCnt, q, uvec4(n));
}
)";

static const char* kToneCS = "#version 310 es\nlayout(local_size_x = 16, local_size_y = 8) in;\n" TONE_COMMON R"(
uniform highp usampler2D uCnt;
layout(std430, binding = 1) writeonly buffer Dst { uint px[]; };
uniform ivec2 uOut;
uniform int uRow0;
uniform int uRows;
uniform float uBlack;
uniform float uScale;
uniform int uSpeckMax;
uniform int uGray;
void main() {
    int x = int(gl_GlobalInvocationID.x), r = int(gl_GlobalInvocationID.y);
    if (x >= uOut.x || r >= uRows) return;
    int y = uRow0 + r;
    vec3 v = normalised(x, y);
    float l = luma(v);
    bool speck = false;
    if (l < 0.75) {
        ivec2 nq = textureSize(uCnt, 0);
        int bi = min(y / 4, nq.y - 1), bj = min(x / 4, nq.x - 1);
        uint n = 0u;
        for (int di = -4; di <= 4; ++di)
            for (int dj = -4; dj <= 4; ++dj) {
                int ii = bi + di, jj = bj + dj;
                if (ii >= 0 && ii < nq.y && jj >= 0 && jj < nq.x) n += texelFetch(uCnt, ivec2(jj, ii), 0).r;
            }
        speck = n >= 1u && int(n) <= uSpeckMax;
    }
    vec3 t = speck ? vec3(1.0) : clamp((v - uBlack) * uScale, 0.0, 1.0);
    uvec3 o;
    if (uGray != 0) o = uvec3(uint(roundEven(luma(t) * 255.0)));
    else o = uvec3(roundEven(t * 255.0));
    px[r * uOut.x + x] = o.r | (o.g << 8) | (o.b << 16) | (255u << 24);
}
)";

bool Gpu::init_cleanup_programs() {
    prog_down_ = program("down4", kDownCS);
    prog_dilate_ = program("dilate3", kDilateCS);
    prog_median_ = program("median11", kMedianCS);
    prog_count_ = program("inkcount", kCountCS);
    prog_tone_ = program("tone", kToneCS);
    return prog_down_ && prog_dilate_ && prog_median_ && prog_count_ && prog_tone_;
}

static GLuint tex_storage(GLenum fmt, int w, int h) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, fmt, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static void bind_tex(GLuint prog, int unit, GLuint t, const char* name) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, t);
    set_i(prog, name, unit);
}

bool Gpu::background_quarter(const cv::Mat& img, cv::Mat& bg4) {
    if (!available()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    int qw = img.cols / 4, qh = img.rows / 4;
    if (img.cols > max_tex_ || img.rows > max_tex_ || qw < 1 || qh < 1) return false;
    if (!make_current()) return false;
    cv::Mat s = img.isContinuous() ? img : img.clone();
    GLuint tsrc = gpu_tex_rgba8ui(s);
    GLuint ta = tex_storage(GL_RGBA8UI, qw, qh), tb = tex_storage(GL_RGBA8UI, qw, qh);
    glUseProgram(prog_down_);
    bind_tex(prog_down_, 0, tsrc, "uSrc");
    glBindImageTexture(0, ta, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
    set_i2(prog_down_, "uSize", qw, qh);
    glDispatchCompute((qw + 15) / 16, (qh + 7) / 8, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glUseProgram(prog_dilate_);
    bind_tex(prog_dilate_, 0, ta, "uSrc");
    glBindImageTexture(0, tb, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
    set_i2(prog_dilate_, "uSize", qw, qh);
    glDispatchCompute((qw + 15) / 16, (qh + 7) / 8, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glUseProgram(prog_median_);
    bind_tex(prog_median_, 0, tb, "uSrc");
    bool ok = run_banded(prog_median_, qw, qh, bg4);
    GLuint ts[3] = {tsrc, ta, tb};
    glDeleteTextures(3, ts);
    glFinish();
    release();
    return ok;
}

bool Gpu::tone(const cv::Mat& img, const CleanParams& p, cv::Mat& out) {
    if (!available()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (img.cols > max_tex_ || img.rows > max_tex_) return false;
    if (!make_current()) return false;
    cv::Mat s = img.isContinuous() ? img : img.clone();
    cv::Mat ill = p.illum.isContinuous() ? p.illum : p.illum.clone();
    int qw = std::max(1, img.cols / 4), qh = std::max(1, img.rows / 4);
    GLuint tsrc = gpu_tex_rgba8ui(s);
    GLuint till = tex_r32f(ill.cols, ill.rows, ill.ptr<float>());
    GLuint tcnt = tex_storage(GL_R32UI, qw, qh);
    for (GLuint prog : {prog_count_, prog_tone_}) {
        glUseProgram(prog);
        bind_tex(prog, 0, tsrc, "uSrc");
        bind_tex(prog, 1, till, "uIllum");
        glUniform3f(glGetUniformLocation(prog, "uWb"), p.wb[0], p.wb[1], p.wb[2]);
        set_i2(prog, "uImg", img.cols, img.rows);
    }
    glUseProgram(prog_count_);
    glBindImageTexture(0, tcnt, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    set_i2(prog_count_, "uQ", qw, qh);
    glDispatchCompute((qw + 15) / 16, (qh + 7) / 8, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glUseProgram(prog_tone_);
    bind_tex(prog_tone_, 2, tcnt, "uCnt");
    glUniform1f(glGetUniformLocation(prog_tone_, "uBlack"), p.black);
    glUniform1f(glGetUniformLocation(prog_tone_, "uScale"), 1.f / (p.white - p.black));
    set_i(prog_tone_, "uSpeckMax", p.speck_max);
    set_i(prog_tone_, "uGray", p.grayscale ? 1 : 0);
    bool ok = run_banded(prog_tone_, img.cols, img.rows, out);
    GLuint ts[3] = {tsrc, till, tcnt};
    glDeleteTextures(3, ts);
    glFinish();
    release();
    return ok;
}

bool Gpu::expand_maps(const GridMap& g, cv::Mat& mx, cv::Mat& my) {
    if (!available()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (!make_current()) return false;
    GLuint tgx = tex_r32f(g.gw, g.gh, g.x.data()), tgy = tex_r32f(g.gw, g.gh, g.y.data());
    glUseProgram(prog_expand_);
    bind_tex(prog_expand_, 0, tgx, "uGx");
    bind_tex(prog_expand_, 1, tgy, "uGy");
    set_i2(prog_expand_, "uGrid", g.gw, g.gh);
    set_i(prog_expand_, "uStride", g.stride);
    set_i2(prog_expand_, "uOut", g.out_w, g.out_h);
    GLint max_block = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_block);
    int band = std::max(1, std::min(g.out_h, (int)(std::min<long long>(max_block, 8 << 20) / (8LL * g.out_w))));
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)band * g.out_w * 8, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf);
    mx.create(g.out_h, g.out_w, CV_32F);
    my.create(g.out_h, g.out_w, CV_32F);
    bool ok = true;
    for (int y0 = 0; y0 < g.out_h && ok; y0 += band) {
        int rows = std::min(band, g.out_h - y0);
        set_i(prog_expand_, "uRow0", y0);
        set_i(prog_expand_, "uRows", rows);
        glDispatchCompute((g.out_w + 15) / 16, (rows + 7) / 8, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        const float* p = (const float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)rows * g.out_w * 8, GL_MAP_READ_BIT);
        if (!p) { ok = false; break; }
        for (int r = 0; r < rows; ++r)
            for (int x = 0; x < g.out_w; ++x) {
                mx.at<float>(y0 + r, x) = p[2 * ((size_t)r * g.out_w + x)];
                my.at<float>(y0 + r, x) = p[2 * ((size_t)r * g.out_w + x) + 1];
            }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    ok = ok && glGetError() == GL_NO_ERROR;
    glDeleteBuffers(1, &buf);
    GLuint ts[2] = {tgx, tgy};
    glDeleteTextures(2, ts);
    glFinish();
    release();
    return ok;
}

}  // namespace fs
