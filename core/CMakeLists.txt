# Portable C++ core: shared by the Android app (app/src/main/cpp) and the host parity harness.
# The caller provides OpenCV (Android: org.opencv:opencv 4.11.0 prefab; host: a 4.11.0 build
# without IPP) and the EGL / GLES libraries.
file(GLOB FS_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_library(flatscan_core STATIC ${FS_CORE_SOURCES})
target_include_directories(flatscan_core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_compile_features(flatscan_core PUBLIC cxx_std_17)
# No FMA contraction: the ports reproduce numpy/OpenCV float arithmetic step by step,
# and x86 and ARM must round identically.
target_compile_options(flatscan_core PRIVATE -ffp-contract=off -Wall -Wno-sign-compare)
# LiteRT's TFLite C API and GPU delegate headers (1.4.0, Apache-2.0) for infer.cpp's LiteRtGpu backend,
# whose libraries are loaded at run time (InferConfig::litert_dir); infer.cpp compiles it on Android / Linux
target_include_directories(flatscan_core PRIVATE ${CMAKE_CURRENT_LIST_DIR}/third_party/litert/include)

# ONNX Runtime for the nets (infer.cpp). FS_ORT_ROOT: an unpacked release (include/, lib/) or the
# headers/ and jni/<ABI>/ of an onnxruntime-android(-qnn) AAR. Unset: infer.cpp is a stub whose runs
# fail (the callers' no-net paths). The execution providers beyond the CPU are compiled in only
# when asked for, and need an ORT package that has them (all OFF by default).
set(FS_ORT_ROOT "" CACHE PATH "ONNX Runtime package (include/ + lib/, or AAR headers/ + jni/<ABI>/)")
option(FS_ORT_QNN "Qualcomm QNN HTP execution provider (onnxruntime-android-qnn)" OFF)
option(FS_ORT_NNAPI "Android NNAPI execution provider (onnxruntime-android)" OFF)
option(FS_ORT_XNNPACK "XNNPACK execution provider" OFF)
option(FS_ORT_CUDA "CUDA execution provider" OFF)
option(FS_ORT_TENSORRT "TensorRT execution provider" OFF)
option(FS_ORT_DML "DirectML execution provider" OFF)
option(FS_ORT_OPENVINO "OpenVINO execution provider" OFF)
option(FS_ORT_COREML "CoreML execution provider" OFF)
if(FS_ORT_ROOT)
  if(EXISTS ${FS_ORT_ROOT}/include/onnxruntime_cxx_api.h)
    set(FS_ORT_INCLUDE ${FS_ORT_ROOT}/include)
  else()
    set(FS_ORT_INCLUDE ${FS_ORT_ROOT}/headers)
  endif()
  find_library(FS_ORT_LIB NAMES onnxruntime
               PATHS ${FS_ORT_ROOT}/lib ${FS_ORT_ROOT}/jni/${ANDROID_ABI} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
  target_include_directories(flatscan_core PUBLIC ${FS_ORT_INCLUDE})
  target_link_libraries(flatscan_core PUBLIC ${FS_ORT_LIB})
  target_compile_definitions(flatscan_core PUBLIC FS_WITH_ORT)
  foreach(ep QNN NNAPI XNNPACK CUDA TENSORRT DML OPENVINO COREML)
    if(FS_ORT_${ep})
      target_compile_definitions(flatscan_core PRIVATE FS_ORT_${ep})
    endif()
  endforeach()
  if(ANDROID)
    target_link_libraries(flatscan_core PUBLIC dl)
  endif()
endif()
