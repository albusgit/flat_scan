@echo off
rem Desktop build of the core for the filter checks: build_dc.bat BUILD_DIR OFF|WGL
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0\..\desktop_check
cmake -S . -B %1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/User/Android/opencv/build/x64/vc16/lib -DFS_DESKTOP_GPU=%2 || exit /b 1
cmake --build %1 -j 6 || exit /b 1
