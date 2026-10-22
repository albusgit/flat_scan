# litert_check (litert_check.cpp): the LiteRT GPU backend of core/src/infer.cpp against ONNX Runtime's
# CPU, as an Android command-line program (NDK; arm64-v8a for phones, x86_64 for the emulator):
#   cmake -S tools/litert_check -B build/<ABI> -G Ninja -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake
#         -DANDROID_ABI=<ABI> -DANDROID_PLATFORM=26 -DCMAKE_BUILD_TYPE=Release -DFS_ORT_ROOT=<onnxruntime-android-1.22.0.aar unpacked>
# FS_ORT_ROOT: the AAR's headers/ and jni/<ABI>/libonnxruntime.so (the layout core/CMakeLists.txt reads).
# Only infer.cpp of the core is built (no OpenCV); the LiteRT libraries are loaded at run time.
# stub/libOpenCL.so (opencl_stub.cpp): an OpenCL with no platform, for the emulator only (README.md).
cmake_minimum_required(VERSION 3.18)
project(litert_check CXX)
set(CMAKE_CXX_STANDARD 17)
set(CORE ${CMAKE_CURRENT_LIST_DIR}/../../core)
set(FS_ORT_ROOT "" CACHE PATH "onnxruntime-android AAR unpacked (headers/, jni/<ABI>/)")
find_library(FS_ORT_LIB NAMES onnxruntime PATHS ${FS_ORT_ROOT}/jni/${ANDROID_ABI} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
add_executable(litert_check litert_check.cpp ${CORE}/src/infer.cpp)
target_include_directories(litert_check PRIVATE ${CORE}/src ${CORE}/third_party/litert/include ${FS_ORT_ROOT}/headers)
target_compile_definitions(litert_check PRIVATE FS_WITH_ORT FS_LITERT_TEST_XNNPACK)
target_compile_options(litert_check PRIVATE -ffp-contract=off -Wall -Wno-sign-compare)
target_link_libraries(litert_check PRIVATE ${FS_ORT_LIB} dl log)
add_library(opencl_stub SHARED opencl_stub.cpp)
set_target_properties(opencl_stub PROPERTIES OUTPUT_NAME OpenCL LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/stub)
