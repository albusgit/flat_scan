# Host (Linux) build of the core + CLI used for the acceptance tests. GPU paths run through
# EGL/GLES 3.1 (Mesa llvmpipe in CI-like environments; any GLES 3.1 driver works).
cmake_minimum_required(VERSION 3.18)
project(flatscan_host CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenCV 4.11 EXACT REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../core ${CMAKE_BINARY_DIR}/core)
target_link_libraries(flatscan_core PUBLIC ${OpenCV_LIBS} EGL GLESv2 pthread)
add_executable(fs_cli fs_cli.cpp cli_ext.cpp)
target_link_libraries(fs_cli PRIVATE flatscan_core)
