# Host (Linux, or Windows with Mesa) build of the core + CLI used for the acceptance tests. GPU paths run through
# EGL/GLES 3.1 (Mesa llvmpipe in CI-like environments; any GLES 3.1 driver works).
cmake_minimum_required(VERSION 3.18)
project(flatscan_host CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(WIN32)
  # Windows (MSVC): EGL + GLES 3.1 from Mesa (mesa-dist-win: release + devel packages; llvmpipe by
  # default, GALLIUM_DRIVER=d3d12 for the real GPU through Direct3D 12). MESA_DIR holds the
  # release's x64/ DLLs and the devel package's include/ and lib/x64/; the DLLs are copied next
  # to fs_cli.exe. The core is built here with MSVC flags (core/CMakeLists.txt is GCC/Clang).
  set(MESA_DIR "C:/Users/User/Android/mesa" CACHE PATH "mesa-dist-win: release/x64, devel/include, devel/lib/x64")
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d photo)
  file(GLOB FS_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../core/src/*.cpp)
  add_library(flatscan_core STATIC ${FS_CORE_SOURCES})
  target_include_directories(flatscan_core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../core/src ${MESA_DIR}/devel/include)
  # /fp:precise does not contract to FMA (VS 2022+); the core is built with -ffp-contract=off elsewhere
  target_compile_options(flatscan_core PUBLIC /fp:precise /utf-8 /EHsc /bigobj)
  target_compile_definitions(flatscan_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
  target_link_libraries(flatscan_core PUBLIC ${OpenCV_LIBS} ${MESA_DIR}/devel/lib/x64/libEGL.lib ${MESA_DIR}/devel/lib/x64/libGLESv2.lib)
  add_executable(fs_cli fs_cli.cpp cli_ext.cpp gpu_stages.cpp)
  target_link_libraries(fs_cli PRIVATE flatscan_core)
  foreach(dll libEGL.dll libGLESv2.dll libgallium_wgl.dll opengl32.dll dxil.dll)
    add_custom_command(TARGET fs_cli POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MESA_DIR}/release/x64/${dll} $<TARGET_FILE_DIR:fs_cli>)
  endforeach()
else()
  find_package(OpenCV 4.11 EXACT REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../core ${CMAKE_BINARY_DIR}/core)
  target_link_libraries(flatscan_core PUBLIC ${OpenCV_LIBS} EGL GLESv2 pthread)
  add_executable(fs_cli fs_cli.cpp cli_ext.cpp gpu_stages.cpp)
  target_link_libraries(fs_cli PRIVATE flatscan_core)
endif()
