@echo off
rem Desktop check builds with a GPU and / or an ORT execution provider (CMakeLists.txt, gate_f.sh):
rem   build_gpu.bat BUILD_DIR OFF|WGL|MESA ORT_ROOT [EP [EP2]]     EP: DML, CUDA, TENSORRT, ... (-DFS_ORT_<EP>=ON)
rem e.g. build_gpu.bat build/f_dml WGL C:/Users/User/Android/ort/onnxruntime-win-x64-directml-1.22.0 DML
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
set EPS=
if not "%~4"=="" set EPS=-DFS_ORT_%~4:BOOL=ON
if not "%~5"=="" set EPS=%EPS% -DFS_ORT_%~5:BOOL=ON
cmake -S . -B %1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/User/Android/opencv/build/x64/vc16/lib -DFS_DESKTOP_GPU=%2 -DFS_ORT_ROOT=%3 %EPS% || exit /b 1
cmake --build %1 -j 8 || exit /b 1
