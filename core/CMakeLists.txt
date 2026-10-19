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
