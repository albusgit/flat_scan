cmake_minimum_required(VERSION 3.22.1)
project(flatscan CXX)
# OpenCV 4.11.0 from the org.opencv:opencv AAR (prefab), the same version as the reference.
find_package(OpenCV REQUIRED CONFIG)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../core ${CMAKE_BINARY_DIR}/core)
target_link_libraries(flatscan_core PUBLIC OpenCV::opencv_java4 EGL GLESv3 log)
add_library(flatscan SHARED jni_bridge.cpp)
target_link_libraries(flatscan PRIVATE flatscan_core jnigraphics android log)
