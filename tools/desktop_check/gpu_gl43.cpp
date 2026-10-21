// Desktop GPU for the check tools (FS_DESKTOP_GPU=WGL, tools/desktop_check/CMakeLists.txt): the
// EGL + GLES 3.1 entry points that core/src/gpu.cpp calls, implemented over WGL on the machine's
// own OpenGL driver (System32 opengl32.dll -> the vendor ICD, e.g. NVIDIA). gpu.cpp itself is
// compiled unchanged, so the compute shaders are the app's.
//
// Context, first that works (FS_GL_PROFILE=es | core forces one):
//   es    an OpenGL ES 3.2 / 3.1 context through WGL_EXT_create_context_es2_profile (NVIDIA, AMD):
//         the shaders' own "#version 320 es" / "310 es" headers, as on a phone
//   core  a desktop OpenGL 4.6 .. 4.3 core context; the ES shaders then compile through
//         GL_ARB_ES3_1_compatibility / GL_ARB_ES3_2_compatibility (gpu.cpp picks "#version 320 es"
//         for any 4.x context, so the driver needs ES3_2_compatibility; otherwise the programs fail
//         to compile and the core takes its CPU paths, as everywhere else)
// The EGL part is the subset gpu.cpp uses: one display (a hidden window's DC), one config, a dummy
// pbuffer surface, one context; eglMakeCurrent = wglMakeCurrent on that DC (any thread; gpu.cpp
// serialises the calls). Any failure here makes Gpu::available() false and the core runs on the CPU.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// the GLES 3.1 functions gpu.cpp calls: return type, name, parameters, arguments
#define FS_GL_FUNCS(X)                                                                                                         \
    X(void, glActiveTexture, (GLenum texture), (texture))                                                                      \
    X(void, glAttachShader, (GLuint program, GLuint shader), (program, shader))                                                \
    X(void, glBindBuffer, (GLenum target, GLuint buffer), (target, buffer))                                                    \
    X(void, glBindBufferBase, (GLenum target, GLuint index, GLuint buffer), (target, index, buffer))                           \
    X(void, glBindImageTexture, (GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format), \
      (unit, texture, level, layered, layer, access, format))                                                                  \
    X(void, glBindTexture, (GLenum target, GLuint texture), (target, texture))                                                 \
    X(void, glBufferData, (GLenum target, GLsizeiptr size, const void* data, GLenum usage), (target, size, data, usage))       \
    X(void, glCompileShader, (GLuint shader), (shader))                                                                        \
    X(GLuint, glCreateProgram, (void), ())                                                                                     \
    X(GLuint, glCreateShader, (GLenum type), (type))                                                                           \
    X(void, glDeleteBuffers, (GLsizei n, const GLuint* buffers), (n, buffers))                                                 \
    X(void, glDeleteProgram, (GLuint program), (program))                                                                      \
    X(void, glDeleteShader, (GLuint shader), (shader))                                                                         \
    X(void, glDeleteTextures, (GLsizei n, const GLuint* textures), (n, textures))                                              \
    X(void, glDispatchCompute, (GLuint x, GLuint y, GLuint z), (x, y, z))                                                      \
    X(void, glFinish, (void), ())                                                                                              \
    X(void, glGenBuffers, (GLsizei n, GLuint* buffers), (n, buffers))                                                          \
    X(void, glGenTextures, (GLsizei n, GLuint* textures), (n, textures))                                                       \
    X(GLenum, glGetError, (void), ())                                                                                          \
    X(void, glGetIntegerv, (GLenum pname, GLint* data), (pname, data))                                                         \
    X(void, glGetProgramInfoLog, (GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog), (program, bufSize, length, infoLog)) \
    X(void, glGetProgramiv, (GLuint program, GLenum pname, GLint* params), (program, pname, params))                           \
    X(void, glGetShaderInfoLog, (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog), (shader, bufSize, length, infoLog)) \
    X(void, glGetShaderiv, (GLuint shader, GLenum pname, GLint* params), (shader, pname, params))                              \
    X(const GLubyte*, glGetString, (GLenum name), (name))                                                                      \
    X(const GLubyte*, glGetStringi, (GLenum name, GLuint index), (name, index))                                                \
    X(GLint, glGetUniformLocation, (GLuint program, const GLchar* name), (program, name))                                      \
    X(void, glLinkProgram, (GLuint program), (program))                                                                        \
    X(void*, glMapBufferRange, (GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access), (target, offset, length, access)) \
    X(void, glMemoryBarrier, (GLbitfield barriers), (barriers))                                                                \
    X(void, glPixelStorei, (GLenum pname, GLint param), (pname, param))                                                        \
    X(void, glShaderSource, (GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length), (shader, count, string, length)) \
    X(void, glTexParameteri, (GLenum target, GLenum pname, GLint param), (target, pname, param))                               \
    X(void, glTexStorage2D, (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height),             \
      (target, levels, internalformat, width, height))                                                                         \
    X(void, glTexSubImage2D, (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels), \
      (target, level, xoffset, yoffset, width, height, format, type, pixels))                                                  \
    X(void, glUniform1f, (GLint location, GLfloat v0), (location, v0))                                                         \
    X(void, glUniform1i, (GLint location, GLint v0), (location, v0))                                                           \
    X(void, glUniform1ui, (GLint location, GLuint v0), (location, v0))                                                         \
    X(void, glUniform2i, (GLint location, GLint v0, GLint v1), (location, v0, v1))                                             \
    X(GLboolean, glUnmapBuffer, (GLenum target), (target))                                                                     \
    X(void, glUseProgram, (GLuint program), (program))

#define FS_GL_PTR(ret, name, params, args) ret(GL_APIENTRY* p_##name) params = nullptr;
FS_GL_FUNCS(FS_GL_PTR)
#undef FS_GL_PTR

// WGL_ARB_create_context(_profile), WGL_EXT_create_context_es2_profile
constexpr int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091, WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092, WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126,
              WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x1, WGL_CONTEXT_ES2_PROFILE_BIT_EXT = 0x4;
using CreateContextAttribs = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
using GetExtensionsString = const char*(WINAPI*)(HDC);

struct Wgl {
    HMODULE gl = nullptr;
    HGLRC(WINAPI* create)(HDC) = nullptr;
    BOOL(WINAPI* destroy)(HGLRC) = nullptr;
    BOOL(WINAPI* make_current)(HDC, HGLRC) = nullptr;
    PROC(WINAPI* get_proc)(LPCSTR) = nullptr;
    HWND wnd = nullptr;
    HDC dc = nullptr;
    HGLRC ctx = nullptr;
    bool initialised = false;
};
Wgl W;

// A GL entry point of the current context: the ICD's (wglGetProcAddress), else opengl32's export (GL 1.1).
void* gl_proc(const char* name) {
    PROC p = W.get_proc(name);
    const intptr_t v = (intptr_t)p;
    if (v == 0 || v == 1 || v == 2 || v == 3 || v == -1) p = GetProcAddress(W.gl, name);
    return (void*)p;
}

bool load_wgl() {
    // the system's opengl32.dll (the vendor driver), never one next to the exe (e.g. Mesa's)
    char sys[MAX_PATH];
    UINT n = GetSystemDirectoryA(sys, MAX_PATH);
    if (n == 0 || n >= MAX_PATH - 16) return false;
    W.gl = LoadLibraryA((std::string(sys) + "\\opengl32.dll").c_str());
    if (!W.gl) return false;
    W.create = (HGLRC(WINAPI*)(HDC))GetProcAddress(W.gl, "wglCreateContext");
    W.destroy = (BOOL(WINAPI*)(HGLRC))GetProcAddress(W.gl, "wglDeleteContext");
    W.make_current = (BOOL(WINAPI*)(HDC, HGLRC))GetProcAddress(W.gl, "wglMakeCurrent");
    W.get_proc = (PROC(WINAPI*)(LPCSTR))GetProcAddress(W.gl, "wglGetProcAddress");
    return W.create && W.destroy && W.make_current && W.get_proc;
}

bool make_window() {
    HINSTANCE inst = GetModuleHandleA(nullptr);
    WNDCLASSA wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = inst;
    wc.lpszClassName = "flatscan_gl43";
    RegisterClassA(&wc);   // fails harmlessly when already registered
    W.wnd = CreateWindowExA(0, wc.lpszClassName, "flatscan_gl43", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr, inst, nullptr);
    if (!W.wnd) return false;
    W.dc = GetDC(W.wnd);
    if (!W.dc) return false;
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof pfd;
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(W.dc, &pfd);
    return pf != 0 && SetPixelFormat(W.dc, pf, &pfd);
}

bool has_word(const char* list, const char* word) {
    if (!list) return false;
    const size_t n = std::strlen(word);
    for (const char* p = list; (p = std::strstr(p, word)) != nullptr; p += n)
        if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == 0)) return true;
    return false;
}

// The context (see the top); nullptr when none can be made.
HGLRC create_context() {
    HGLRC legacy = W.create(W.dc);   // needed to reach wglCreateContextAttribsARB
    if (!legacy || !W.make_current(W.dc, legacy)) {
        if (legacy) W.destroy(legacy);
        return nullptr;
    }
    auto attribs = (CreateContextAttribs)gl_proc("wglCreateContextAttribsARB");
    auto exts = (GetExtensionsString)gl_proc("wglGetExtensionsStringARB");
    const char* ext = exts ? exts(W.dc) : nullptr;
    W.make_current(nullptr, nullptr);
    W.destroy(legacy);
    if (!attribs) return nullptr;
    const char* force = std::getenv("FS_GL_PROFILE");
    const bool try_es = (!force || !std::strcmp(force, "es")) && has_word(ext, "WGL_EXT_create_context_es2_profile");
    const bool try_core = !force || !std::strcmp(force, "core");
    struct Want { int profile, major, minor; };
    const Want wants[] = {{WGL_CONTEXT_ES2_PROFILE_BIT_EXT, 3, 2}, {WGL_CONTEXT_ES2_PROFILE_BIT_EXT, 3, 1},
                          {WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 4, 6}, {WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 4, 5},
                          {WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 4, 4}, {WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 4, 3}};
    for (const Want& w : wants) {
        const bool es = w.profile == WGL_CONTEXT_ES2_PROFILE_BIT_EXT;
        if (es ? !try_es : !try_core) continue;
        const int a[] = {WGL_CONTEXT_MAJOR_VERSION_ARB, w.major, WGL_CONTEXT_MINOR_VERSION_ARB, w.minor, WGL_CONTEXT_PROFILE_MASK_ARB, w.profile, 0};
        if (HGLRC c = attribs(W.dc, nullptr, a)) return c;
    }
    return nullptr;
}

const EGLDisplay kDisplay = (EGLDisplay)&W;
const EGLConfig kConfig = (EGLConfig)&W.dc;
const EGLSurface kSurface = (EGLSurface)&W.wnd;

}  // namespace

extern "C" {

EGLDisplay EGLAPIENTRY eglGetDisplay(EGLNativeDisplayType) { return kDisplay; }

EGLBoolean EGLAPIENTRY eglInitialize(EGLDisplay dpy, EGLint* major, EGLint* minor) {
    if (dpy != kDisplay) return EGL_FALSE;
    if (!W.initialised) {
        W.initialised = true;
        if (!load_wgl() || !make_window()) W.dc = nullptr;
    }
    if (!W.dc) return EGL_FALSE;
    if (major) *major = 1;
    if (minor) *minor = 4;
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY eglBindAPI(EGLenum api) { return api == EGL_OPENGL_ES_API ? EGL_TRUE : EGL_FALSE; }

EGLBoolean EGLAPIENTRY eglChooseConfig(EGLDisplay dpy, const EGLint*, EGLConfig* configs, EGLint config_size, EGLint* num_config) {
    if (dpy != kDisplay || !W.dc) return EGL_FALSE;
    if (configs && config_size > 0) configs[0] = kConfig;
    if (num_config) *num_config = 1;
    return EGL_TRUE;
}

EGLSurface EGLAPIENTRY eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig, const EGLint*) {
    return dpy == kDisplay && W.dc ? kSurface : EGL_NO_SURFACE;
}

EGLContext EGLAPIENTRY eglCreateContext(EGLDisplay dpy, EGLConfig, EGLContext, const EGLint*) {
    if (dpy != kDisplay || !W.dc || W.ctx) return EGL_NO_CONTEXT;
    HGLRC c = create_context();
    if (!c) return EGL_NO_CONTEXT;
    bool ok = W.make_current(W.dc, c) == TRUE;
#define FS_GL_LOAD(ret, name, params, args) ok = ok && (p_##name = (decltype(p_##name))gl_proc(#name)) != nullptr;
    FS_GL_FUNCS(FS_GL_LOAD)
#undef FS_GL_LOAD
    W.make_current(nullptr, nullptr);
    if (!ok) {
        W.destroy(c);
        return EGL_NO_CONTEXT;
    }
    W.ctx = c;
    return (EGLContext)c;
}

EGLBoolean EGLAPIENTRY eglMakeCurrent(EGLDisplay dpy, EGLSurface, EGLSurface, EGLContext ctx) {
    if (dpy != kDisplay || !W.dc) return EGL_FALSE;
    if (ctx == EGL_NO_CONTEXT) return W.make_current(nullptr, nullptr) ? EGL_TRUE : EGL_FALSE;
    if ((HGLRC)ctx != W.ctx) return EGL_FALSE;
    return W.make_current(W.dc, W.ctx) ? EGL_TRUE : EGL_FALSE;
}

#define FS_GL_DEF(ret, name, params, args) \
    ret GL_APIENTRY name params { return p_##name args; }
FS_GL_FUNCS(FS_GL_DEF)
#undef FS_GL_DEF

}  // extern "C"
