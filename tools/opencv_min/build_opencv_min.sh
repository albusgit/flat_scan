#!/bin/bash
# OpenCV 4.11.0 for the app with only the modules the core uses (core imgproc imgcodecs calib3d photo, and
# features2d + flann that calib3d needs), static, linked into libflatscan instead of the Maven AAR's
# libopencv_java4.so (22.7 MB, 16 modules, 10 image codecs).
#
# Everything else follows the Maven AAR's own build (its getBuildInformation(), and platforms/android/build_sdk.py):
# NDK 26.3.11579264 (clang 17), Release (-O3), API 21, c++_shared, WITH_OPENCL=OFF, WITH_TBB=ON (oneTBB 2021.11,
# built here), IPP on x86_64 only, the default custom HALs (KleidiCV + Carotene on arm64), the default CPU
# baseline / dispatch (arm64: NEON FP16 + NEON_DOTPROD / NEON_FP16 / NEON_BF16), 16 KB pages. So the kernels, and
# with them the numbers and the speed, are the AAR's. Dropped: codecs other than PNG and JPEG, ITT tracing,
# protobuf / ADE, MediaNDK (videoio is not built).
#
#   bash tools/opencv_min/build_opencv_min.sh SRC_DIR INSTALL_DIR [ABI ...]     (default ABIs: arm64-v8a x86_64)
# SRC_DIR: the opencv-4.11.0 source tree. INSTALL_DIR: one OpenCV Android SDK layout for all ABIs
# (INSTALL_DIR/sdk/native/jni/OpenCVConfig.cmake); point the app at it with opencv.min.dir in local.properties.
set -eu
SRC=$1; OUT=$2; shift 2
ABIS=${*:-arm64-v8a x86_64}
SDK=${ANDROID_SDK:-C:/Users/User/Android/sdk}
NDK=$SDK/ndk/26.3.11579264
CMAKE=$SDK/cmake/3.22.1/bin/cmake
NINJA=$SDK/cmake/3.22.1/bin/ninja
for ABI in $ABIS; do
  IPP=OFF; [ "$ABI" = x86_64 ] && IPP=ON
  B=$OUT/build-$ABI
  "$CMAKE" -S "$SRC" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=$ABI -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_STL=c++_shared -DANDROID_TOOLCHAIN=clang \
    -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,photo \
    -DWITH_OPENCL=OFF -DWITH_TBB=ON -DWITH_IPP=$IPP \
    -DBUILD_JAVA=OFF -DBUILD_opencv_java=OFF -DBUILD_KOTLIN_EXTENSIONS=OFF -DBUILD_ANDROID_PROJECTS=OFF \
    -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
    -DWITH_WEBP=OFF -DWITH_TIFF=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
    -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF \
    -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_ITT=OFF -DWITH_ANDROID_MEDIANDK=OFF \
    -DCMAKE_INSTALL_PREFIX="$OUT"
  "$CMAKE" --build "$B" -j 16
  "$CMAKE" --install "$B"
done
