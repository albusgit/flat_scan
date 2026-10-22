cmake_minimum_required(VERSION 3.22.1)
project(flatscan CXX)
# OpenCV 4.11.0, the same version as the reference:
#  * FS_OPENCV_MIN (app/build.gradle.kts: opencv.min.dir): the slim static build of tools/opencv_min (only the modules
#    the core uses, the AAR's HALs, dispatch and TBB), linked into libflatscan; its unused code is dropped at link
#    time (--gc-sections, and --exclude-libs so that the static libraries' symbols are not exported);
#  * otherwise the org.opencv:opencv AAR's libopencv_java4.so (prefab).
set(FS_OPENCV_MIN "" CACHE PATH "tools/opencv_min install folder (sdk/native/jni/OpenCVConfig.cmake)")
if(FS_OPENCV_MIN)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d photo
               PATHS ${FS_OPENCV_MIN}/sdk/native/jni NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  set(FS_OPENCV_TARGETS ${OpenCV_LIBS})
  message(STATUS "OpenCV ${OpenCV_VERSION} (static, ${FS_OPENCV_MIN}): ${OpenCV_LIBS}")
else()
  find_package(OpenCV REQUIRED CONFIG)
  set(FS_OPENCV_TARGETS OpenCV::opencv_java4)
endif()
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../core ${CMAKE_BINARY_DIR}/core)
target_link_libraries(flatscan_core PUBLIC ${FS_OPENCV_TARGETS} EGL GLESv3 log)
# LiteRT (infer.cpp) is dlopen'ed at run time: no ONNX Runtime and no link-time LiteRT dependency
target_link_libraries(flatscan_core PUBLIC dl)
add_library(flatscan SHARED jni_bridge.cpp)
target_link_libraries(flatscan PRIVATE flatscan_core jnigraphics android log)
if(FS_OPENCV_MIN)
  target_link_options(flatscan PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)
endif()
