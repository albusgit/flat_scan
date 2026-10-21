# litert_check

The LiteRT GPU backend of `core/src/infer.cpp` (`Backend::LiteRtGpu`) against ONNX Runtime's CPU on the
same input, as an Android command-line program: no app, no OpenCV. Results: `docs/results/litert/runtime.txt`.

## Build (Windows, NDK 27.2.12479018)

Unpack `onnxruntime-android-1.22.0.aar` (`headers/`, `jni/<ABI>/libonnxruntime.so`) into `ORT`, then per ABI
(`arm64-v8a` for a phone, `x86_64` for the emulator):

    SDK=C:/Users/User/Android/sdk; CM=$SDK/cmake/3.22.1/bin
    $CM/cmake -S tools/litert_check -B build/$ABI -G Ninja -DCMAKE_MAKE_PROGRAM=$CM/ninja.exe \
      -DCMAKE_TOOLCHAIN_FILE=$SDK/ndk/27.2.12479018/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=$ABI -DANDROID_PLATFORM=26 -DCMAKE_BUILD_TYPE=Release -DFS_ORT_ROOT=$ORT
    $CM/cmake --build build/$ABI

## Run

Push to `/data/local/tmp/lrt`: `litert_check`, `libonnxruntime.so` (same ABI), `lib/` with
`libtensorflowlite_jni.so` and `libtensorflowlite_gpu_jni.so` (`jni/<ABI>/` of the
`com.google.ai.edge.litert:litert` and `litert-gpu` 1.4.0 AARs), and a model folder with the GPU model and
the CPU's ONNX as the app names them (`uvdoc_static.tflite` + `uvdoc.onnx`, `syntheticdoc_hr_static.tflite`
+ `syntheticdoc_hr.onnx`, `shadenet_512x352.tflite` + `shadenet.onnx`). Then

    adb shell "cd /data/local/tmp/lrt && LD_LIBRARY_PATH=. ./litert_check lib models uvdoc --runs 10 --cache cache"
    adb logcat -d -s flatscan tflite

prints the backend that ran (`gpu`, or `cpu` after a fallback; why is in logcat), the first run (set-up
included: the delegate's compile, or the kernel cache with `--cache`) and the median run, and
|GPU - CPU| max / p99 / mean over the output. `--input FILE` takes a raw float32 NCHW input
(default: a fixed synthetic image); `--tol T` makes a max difference over T exit 1.

## Without a usable GPU (the emulator)

The emulator's GLES translator (`-gpu host`, `swiftshader_indirect`, `angle_indirect`) has no OpenCL, and the
delegate's OpenGL ES backend fails there (`glGetBufferParameteri64v`: GL_INVALID_ENUM), so every net falls
back to the CPU. Two test aids, neither shipped:

* `build/<ABI>/stub/libOpenCL.so` (`opencl_stub.cpp`): an OpenCL with no platform. Without any OpenCL
  library infer.cpp keeps the net off the delegate (LiteRT 1.4.0's own OpenCL lookup crashes then:
  `opencl_found` in infer.cpp); with the stub on `LD_LIBRARY_PATH=.:stub` the delegate gets as far as its
  OpenGL ES set-up.
* `FS_LITERT_TEST_XNNPACK=1`: this program's build of infer.cpp (`FS_LITERT_TEST_XNNPACK` defined) puts
  LiteRT's XNNPACK CPU delegate in the GPU delegate's place, so the whole-graph check, the NCHW / NHWC
  conversions, the outputs and the LiteRT thread run as on the GPU. XNNPACK covers fewer ops than the GPU
  (no MIRROR_PAD), so the real models are refused there: test nets only.
