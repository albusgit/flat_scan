# Acceleration work: machine facts, resolved inputs, shared rules

Coordinator notes for `docs/accel/PLAN.md` (2026-10-21). Where these differ from the plan, these win.

## Machine (Windows 11, the only build machine)

- CPU i5-14400F (10 cores / 16 threads, no integrated GPU), 32 GB RAM, GPU NVIDIA RTX 5060 8 GB
  (Blackwell, driver 591.44). No CUDA toolkit (`nvcc`) installed. No QNN SDK, no Snapdragon or Intel GPU.
  So on this machine: DirectML is the practical GPU EP; CUDA EP only if an ORT build with sm_120
  (CUDA >= 12.8) runtime works from pip/NuGet redistributables; TensorRT, QNN, OpenVINO-GPU, CoreML are
  HUMAN GATES (OpenVINO CPU is possible).
- Shells: Git Bash and PowerShell. MSVC: `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`,
  Ninja, CMake. Desktop harness build (run through `cmd //c` from Git Bash):

      call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
      cd /d <worktree>\tools\desktop_check
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/User/Android/opencv/build/x64/vc16/lib
      cmake --build build -j 8

  Runtime: put `C:\Users\User\Android\opencv\build\x64\vc16\bin` on PATH.
- Android: SDK `C:/Users/User/Android/sdk` (emulator, system-images/android-35, NDK 27.2.12479018, CMake 3.22.1).
  `local.properties` is untracked: copy it from `C:\Users\User\Desktop\flat_scan\local.properties` into a worktree.
  Build: `JAVA_HOME=/c/Users/User/Android/jdk17 ./gradlew assembleDebug`. No AVD may exist yet; create one
  with the SDK's avdmanager from the android-35 x86_64 image if needed.
- Disk: C: has ~27 GB free. Each stream keeps its footprint under ~3 GB and deletes large temporaries.
  Download ORT packages once into `C:\Users\User\Android\ort\` (outside any repo) and share them.

## ONNX Runtime versions (measured 2026-10-21)

- The app uses ORT Android 1.22.0. The desktop Python reference venv
  `C:\Users\User\envs\flatscan_torch` (Python 3.11, torch 2.11 cu128) has ORT **1.31.0**.
  A Python ORT **1.22.0** venv exists at
  `C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-flat-scan\922e3e9d-4193-49a6-b4b4-d3421c7157d5\scratchpad\ort122`.
- uvdoc.onnx and syntheticdoc_hr.onnx: outputs bit-identical between ORT 1.22 and 1.31 and across
  intra-op thread counts (2 vs default), CPU EP, on real DocUNet net inputs.
- pagefinder.onnx: thread-count invariant, but ORT 1.22 vs 1.31 differ by up to ~1e-5.
  So the Gate B reference must be made with Python ORT **1.22.0** (the app's version), and the desktop
  C++ build links ORT **1.22.0** (`onnxruntime-win-x64-1.22.0` for CPU; GPU/DirectML variants of the same
  version for Stream F). Existing DocUNet net caches (uvdoc/SD-HR) stay valid; page-finder logits made with
  1.31 are not the 1.22 reference.

## Resolved external inputs

- `<APP_REF_DIR>` = `E:\fs_datasets\round4` (`code/`, `seg/`, `real_photos/`, `gt/`).
- `<PHOTOS_35_DIR>` = `E:\fs_datasets\round4\real_photos` (34 photos) + `user2` photo as
  `tools/round4_check.py` defines the set; `<GT_JSON>` = `E:\fs_datasets\round4\gt\gt.json`.
- `<DOCUNET_DIR>` = `C:\Users\User\Desktop\docunet_bench` (`data_crop/crop/{k}_{m} copy.png` photos,
  `data_scan/scan/{k}.png` scans). `<DOCUNET_SCORER>` = `docunet_bench/docunet_eval.py` run with the
  round4 venv python (`-I`). Harness: `run_ours.py` (see its docstring; RESULTS.md has the exact configs).
- **DocUNet baseline is now the Round 8 app after commit 22a6554, not the plan's 5.30:**
  MS-SSIM 0.5898, LD 5.162, AD 0.1893 (our scale, 128 photos, doc 64 excluded) = `work/new_fold2`.
  Config: `NETS=nets_rc_f`, `FS_UVPASS1={work}/nets_frame_f/{n}_uv0.npz FS_UVPASS1_FRAME=1 FS_UVRECROP=1`,
  `FS_UVOUT={nets}/{n}_uv2.npz FS_UVMODEL=2 FS_UVOUT2={nets}/{n}_uv0.npz FS_UVMODEL2=0`.
  Prefer comparing pipeline outputs image-by-image to `work/new_fold2/*.png` (bit-identical => metrics
  unchanged) over re-scoring; re-score only images that changed.
- `<QNN_SDK_DIR>`: not available. QNN work relies on the libraries in the onnxruntime-android-qnn AAR.

## Shared resource rules (several agents run at once on one machine)

- DocUNet scorer and DocUNet pipeline runs are RAM-heavy (two scorers at once crashed with "bad allocation").
  Take the machine-wide lock first: `mkdir C:\Users\User\Desktop\docunet_bench\.heavy_lock` (retry every
  30 s until it succeeds), release with `rmdir` when done, also on failure. Scorer `--jobs 4` at most.
  The same lock guards the Android emulator and any Gradle build.
- Never write into another stream's worktree or into `C:\Users\User\Desktop\flat_scan` (the user's
  checkout). Never touch branch `claude/android-document-scanner-fkka2i`. Do not push.
- Do not delete or overwrite existing `docunet_bench/work/*` folders; create new ones with a stream prefix.
- Commits: `git -c user.name=Claude -c user.email=noreply@anthropic.com commit`, message ending with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01RAnbCC9TcAHbsirQFejEFd`.
