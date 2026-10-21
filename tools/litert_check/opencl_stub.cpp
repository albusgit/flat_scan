// libOpenCL.so with no platform, for litert_check on the emulator only (never shipped). The emulator has no
// OpenCL; infer.cpp then keeps the net off the GPU delegate (opencl_found: LiteRT 1.4.0's own OpenCL lookup
// crashes when it finds no library). With this library on LD_LIBRARY_PATH the lookup succeeds, the delegate's
// OpenCL set-up finds no platform and the delegate takes its OpenGL ES backend, which the emulator has.
#include <cstdint>

extern "C" __attribute__((visibility("default"))) int32_t clGetPlatformIDs(uint32_t, void** platforms, uint32_t* num_platforms) {
    if (platforms) *platforms = nullptr;
    if (num_platforms) *num_platforms = 0;
    return -1001;   // CL_PLATFORM_NOT_FOUND_KHR
}
