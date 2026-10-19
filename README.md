# FlatScan: Android document scanner

Native Android app (Kotlin + C++/NDK). It photographs or imports a page, finds it, unwarps it with
the surface model and the warp map of the Python reference (`app_ref/code`), then cleans it up and
saves or shares it as JPG or PDF. The C++ core is a line-by-line port of the reference, with every
experimental option off: `UNWARP_PROP_*`, `UNWARP_LSD` and `UNWARP_R3_*`. The same core also builds
on Linux as a test harness. That harness produced all the parity numbers below.

**Debug APK:** `dist/flatscan-debug.apk` (arm64-v8a, minSdk 26, 34 MB).
It has **not been run on a phone or an emulator** (see [Not done / not verified](#not-done--not-verified-on-a-real-phone)).

## Result screen features (v2)

- **Filters:** a strip with live thumbnails: Original, Lighten, Auto color (default), Grayscale, High contrast, B&W.
  - Strength slider, default 70%. The last filter and strength are remembered.
  - Solid colors is a separate switch (off for B&W), with "Protect a picture" rectangles that Solid colors leaves alone.
  - Measured on the 5 test pages: 0 text marks lost at 70% and 100%; GPU and CPU differ by at most 1 level.
- **Adjust panel:**
  - Background cleanup pushes grey artefacts near paper colour to white.
  - Levels editor with a histogram and black, midtone and white handles.
  - At default settings the output is byte-identical to the unadjusted filters.
- **Hold to see the original:** press and hold shows the page with perspective correction only. A hint appears on first use.
- **Zoom:** pinch from 1x to 6x, pan, and double tap.
- **Wrinkle map:** a switch in the menu, off by default. It shows a heat map and grid of how far each area was moved, with a legend in px.
- **Re-crop:** goes back to the crop screen and keeps the corners, filter and adjustments.
- **Missing corners and gaps:** filled with clean paper using a soft-edged paper mask.
- **Multiple pages:**
  - Import several images from the gallery.
  - The Pages screen reorders, deletes and adds pages.
  - Export as a multi-page PDF, one JPG per page into a folder, or share them all.
- **Orientation:** pages are turned 180° only when the upright cue is confident (score below -0.2). This fixes the upside-down results: the bench pages scored -0.03 and -0.07, were turned 180° before, and now stay upright. The Auto-rotate setting turns this off.

## Build

Requirements: JDK 17+, Android SDK platform 35, NDK `27.2.12479018`, CMake `3.22.1`.
OpenCV `4.11.0` comes from Maven Central (`org.opencv:opencv`, prefab). This is the same OpenCV version as the reference (`requirements-round3.txt`).

```bash
echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew assembleDebug                                  # app/build/outputs/apk/debug/app-debug.apk
./gradlew assembleDebug -Pabis=arm64-v8a,x86_64          # also an emulator ABI
```

Host harness (Linux) for the acceptance tests:

```bash
# OpenCV 4.11.0 built WITHOUT IPP (as on Android): core, imgproc, imgcodecs, calib3d
cmake -S host -B build-host -G Ninja -DOpenCV_DIR=<ocv>/lib/cmake/opencv4 && ninja -C build-host
export EGL_PLATFORM=surfaceless              # headless GLES 3.1 through Mesa (libegl-mesa0, libgles-dev)
build-host/fs_cli selftest-resample          # resample: CPU / GPU / kernel model vs cv::remap
```

Reproducing the measurements (Python env from `app_ref/requirements-round3.txt`, `PY=` points to it):

```bash
python3 code/photobench.py test_images/flat_page.png bench_photo            # in app_ref/
python3 code/photo_pipeline.py <photo> out_ref/<name> surface                # reference outputs
python3 code/warpmap.py test_images/scan_pN.jpg out_ref/scan_pN 0.15
python3 -I tools/m1_fixtures.py ...; python3 -I tools/export_stages.py ...   # fixtures
tools/run_m1.sh        fs_cli app_ref/code out_ref app_ref/test_images fixtures out   # milestone 1
tools/run_stage_parity.sh fs_cli app_ref fixtures out                         # stage by stage
tools/run_acceptance.sh   fs_cli app_ref out_ref fixtures out                 # section 5
```

The raw output of those runs is in `docs/results/`.

## Architecture

```
core/src/    portable C++17, shared by app and host (compiled with -ffp-contract=off)
  corners      page detection (corners.py)            nprng    numpy PCG64 + Generator.choice, bit-exact
  surface,lsq  surface model + scipy TRF port          npz      NPY/NPZ I/O for the harness
  orient       orientation (orient.py)                 nputil   polyfit, interp, percentile, np.sum order
  warpmap      warp map (warpmap.py, dewarp.py, clean.py helpers)
  sparse       13-point stencil normal equations, MG-preconditioned CG
  pipeline     photo_pipeline.py orchestration, composition on the 4-px grid
  resample     map expansion + OpenCV-exact bicubic (CPU), table for the GPU
  gpu          OpenGL ES 3.1 compute: resample, background, ink count, tone pass
  cleanup,solid  Auto color / Grayscale / Solid colors
app/         Kotlin + Compose UI, CameraX, JNI bridge (app/src/main/cpp/jni_bridge.cpp)
host/        fs_cli: the stage and acceptance harness
tools/       Python fixture exporters, comparators, acceptance runners
```

| Stage | Where | Notes |
|---|---|---|
| Page detection (`paper_mask`, `curved_corners`) | CPU (C++) | RANSAC draws come from the same numpy `default_rng(0)` stream. Runs on a copy no larger than 2400 px; the reference inputs are 2212–2300 px. |
| Surface fit (10 unknowns) | CPU, multi-threaded | Port of `scipy.optimize.least_squares` (TRF, exact SVD trust region, 2-point Jacobian, Cauchy loss). The 10 coarse starts and the Jacobian columns run in parallel; this is deterministic. EXIF focal length is used when present. |
| Homography fallback | CPU | Used when the user moved a corner (> 0.3% of the image size), when detection failed, or when the outline residual median is above 3 px. |
| Stage-1 page, orientation page | CPU | `cv::remap` from the exact per-pixel unroll map, in row bands. |
| Orientation | CPU | Decided on the homography-rectified page, as the brief asks. Turned off if the user rotated the photo. |
| Global stage, line detection, tracing (`trace_w2`), chaining, text rows | CPU, lines in parallel | |
| Sparse solves (global 32 px, pass 1 16 px, pass 2 8 px warm-started) | CPU | CG preconditioned with a geometric multigrid V-cycle (Galerkin, symmetric Gauss-Seidel, dense Cholesky on the coarsest grid), rtol 1e-8. The x and y solves run in parallel. |
| Composition | CPU, 4-px grid only | `F(p) = U(expand(warp4)(p))`, with cv2.remap's 1/32-px quantisation and `BORDER_REPLICATE`, evaluated in the rotated output frame exactly as `photo_pipeline` stores `warpmap.npz`. No full-resolution float map is allocated on the CPU. |
| Map expansion + bicubic resample of the full photo | **GPU** (GLES 3.1 compute), CPU fallback | float32 maps; OpenCV `INTER_CUBIC` semantics: 1/32-px positions, its 15-bit weight table including its sum-correction quirk, white border. |
| Cleanup: ¼-resolution background (block mean, 3×3 dilate, exact 11×11 median), ink counts, one tone pass (illumination, white point, contrast, specks, grayscale) | **GPU**, CPU fallback | The paper-illumination fill and the black/white points are a small CPU step at ¼ resolution. |
| Solid colors | CPU | Port of the cell-fill part of `clean.py`. The grid lines only separate regions; nothing is redrawn; ink is composited on top. |

The GPU paths are chosen at runtime: EGL + GLES ≥ 3.1, texture size limits, and a fallback whenever a call fails.
Analysis runs at 1600 px page width. The output-resolution setting (1240 / 1600 / 2480 px) only rescales the final grid, so the photo is still sampled once.

**App flow:** Home (camera / gallery) → Camera → Crop → Processing → Result → Save/Share.
- **Crop:** auto-detected corners, 44 dp touch targets, a 3× magnifier loupe while dragging, and rotate 90°.
- **Processing:** progress bar with the stage name.
- **Result:** Before/After toggle; Auto color / Grayscale; Solid colors; Rotate; Save (JPG or one-page A4 PDF through the system file picker) and Share; a details dialog with per-stage times.
- **Settings:** default colour mode, solid colours, output resolution, and an on-device self-test.

Light and dark themes follow the system setting.

## Acceptance results (section 5 of the brief)

All numbers below were **measured**, on the Linux host harness: 4-core Xeon 2.1 GHz. The GPU results ran on Mesa llvmpipe, a software GLES 3.2 driver. **None were measured on a phone.**
The reference values were re-measured in this environment with the pinned Python environment (`docs/results/reference_metrics_measured.json`). Where they differ from the brief, both are shown.

### 1. Map parity (`port_compare.py reference candidate --limit 0.1`, app = full C++ pipeline)

| page | max EPE | p99 | result |
|---|---|---|---|
| photo 1 | 0.0886 px | 0.0151 px | pass |
| photo 2 | 0.0843 px | 0.0165 px | pass |
| scan p1 / p2 / p3 | 0.0054 / 0.0024 / 0.0036 px | ≤ 0.0024 px | pass |
| bench P1 / P2 | 0.0308 / 0.0267 px | ≤ 0.0001 px | pass |
| photo 1, homography (`v8`) path | 0.0425 px | 0.0 px | pass |

The GPU and CPU paths give identical maps. **The photo margins are small.** Stage by stage, the port matches the reference much more closely (`docs/results/stage_parity_host.txt`):
- detection: identical contours and RANSAC inlier sets; corners within 2e-12 px;
- surface-fit cost: equal to about 1e-9 relative;
- warp map, run on the reference's own stage-1 page: the global page is bit-identical, the detected lines are identical, the samples agree to 1e-11, and the final map error is ≤ 0.005 px.

The remaining photo error comes from where the two TRF runs stop. Both satisfy the same `ftol=1e-8` stopping test, but BLAS/LAPACK rounding differs. The resulting fit-parameter difference (≤ 4.6e-4 rad) moves the photo 1 unroll map by only 0.0003 px. It still changes 2,941 stage-1 pixels by up to 5 levels: Python's own stage 1 rendered with the C++ parameters differs from the reference in 2,941 pixels, and from the C++ stage 1 in 15. That shifts a few line centroids by 0.05 px, which becomes the 0.08 px above.

### 2. Image parity (8-bit)

- **Parity path (same map, milestone 1):** GPU and CPU resample are **identical to the reference** on all 5 pages: 0 levels, 0 pixels. They are also identical to `cv::remap` on 30 synthetic fixtures (constant, impulse, checkerboard, sub-pixel shift, out-of-range, random warp; 2×2 up to 1203×1609 sources).
- **GPU vs CPU, end to end:** identical on all pages. Cleanup is also identical, GPU vs CPU, in all four modes.
- **End to end against the reference image:** this cannot meet ≤ 1 level, and I did not relax the gate.
  - **Scans:** max 5–6 levels, in 0.004–0.013% of pixels. A 0.005 px map difference can flip OpenCV's 1/32-px position rounding, which is up to ~6 levels at a hard edge.
  - **Photos against the reference resampled from its own stored grid:** max 9 / 8 levels, in 0.12 / 0.13% of pixels.
  - **Photos against the published `out.png`:** max 118 / 140 levels, in 3.4 / 4.5% of pixels. The reference's own stored map differs from its `out.png` by the same amount (see finding 3).

### 3. Ground truth (`photobench.py` + `bench.score`, mean position error)

| case | app | reference (measured here) | brief | within +0.2 px |
|---|---|---|---|---|
| P1 | 1.082 px | 1.082 px | 0.97 px | yes |
| P2 | 1.199 px | 1.199 px | 1.21 px | yes |

### 4. Real-page straightness (`evaluate.py`)

| page | app: lines < 1 px, worst | reference measured here | brief |
|---|---|---|---|
| photo 1 | 98.6%, 1.41 px | 98.6%, 1.42 px | 97%, 1.42 px |
| photo 2 | 100%, 0.621 px | 100%, 0.572 px | 100%, 0.58 px |
| scans p1/p2/p3 | 100%, 0.463 / 0.773 / 0.572 px | identical | 100%, ≤ 0.8 px |

The photo 2 worst-line difference (0.621 vs 0.572 px) comes from composing on the coarse grid, which the brief requires. The reference's own map, resampled from its stored 4-px grid, scores **0.616 px**.

### 5. Photo 1 title border (`fixed_photo_feature.py` logic, on the pipeline output)

p98 **0.261 px**, end **0.191 px**; the reference measured 0.268 / 0.189 px. The gate is ≤ 0.5 px: **pass**.

### 6. Timing

**Not measured on any phone.** No device or emulator was available; the container has no KVM. The app has an on-device benchmark for this: Settings → *Run self-test*. It runs the bundled photo 1 six times (one warm-up) and reports:
- map parity against the bundled reference grid;
- GPU vs CPU resample and cleanup;
- per-stage median and p95;
- peak memory (VmHWM);
- device, SoC, GPU driver and thermal state.

It has a *Share report* button. Run it on two named devices to fill in this section.

For orientation only, single host runs of the whole C++ pipeline (not medians, not a phone) take 1.5–1.7 s per photo and 2.5–2.9 s per scan; per-stage times are in `docs/results/acceptance_host.txt`. The brief quotes about 15–20 s per photo and 10–13 s per scan for the Python reference (desktop, 2 cores).

## Findings about the reference (with evidence; nothing was changed silently)

1. **The orientation page differs between brief and code.** The brief says to decide orientation on the homography-rectified page. `photo_pipeline.py` decides it on the stage-1 page, which in surface mode is the surface unroll. On photo 1, the upright score is **0.0018** on the stage-1 page and **0.49** on the homography page. The app follows the brief. It reaches the same decision as the reference on all 4 photo inputs (`fixtures/*/info.json`), and the option `orient_on_stage1` reproduces the code's behaviour.
2. **The bench pages are flipped 180° by the reference.** Upright scores are −0.03 / −0.07 on the homography page and −0.06 / −0.03 on the stage-1 page. EPE is unaffected, because `bench.score` fits a similarity that includes rotation. But it reports `scale_x_p95` / `scale_y_p95` of about **200%** for every bench run for that reason. Those two columns are meaningless as long as the flip happens. The app reproduces the flip, for parity.
3. **The stored photo map is not the published image.** `photo_pipeline.py` resamples `out.png` from its dense composed map, but stores only every 4th sample in `warpmap.npz`. Within about 8 px of the border, the dense map has `BORDER_REPLICATE` kinks that the grid cannot hold: up to **0.97 px** (photo 1). Resampling the photo from the stored grid changes 3.4% (photo 1) to 4.5% (photo 2) of the pixels by more than one level. Photo 2's worst line is 0.572 px from the dense map and 0.616 px from the stored map.
4. **The pipeline is sensitive end to end.** A 0.0003 px change of the stage-1 unroll map moves the final photo map by up to 0.084 px, through thresholded morphology and line centroids (section 1 above). The 0.1 px gate passes, but other photos may need a looser gate or exact LAPACK.
5. **The brief's numbers differ from the measured reference.** Bench P1 measures 1.082 px here; the brief says 0.97. Photo 1 has 71 of 72 lines under 1 px, which is 98.6%; the brief says 97%. Everything else matches.
6. **IPP:** the reference's OpenCV wheel uses Intel IPP; Android OpenCV does not. Re-running the reference with IPP off changes its maps by at most **0.029 px** (photo 1) and 0.0005 px (scan p1), and about 3 pixels per 10 million in the images. Remap itself is not affected (`IPP_DISABLE_REMAP 1`).
7. **The AMG solve stops early, harmlessly.** Pass 2 stops at relative residual about 9e-6. Its field is within **0.005 px** of the exact solution (`docs/results/solver_audit.txt`), so the port solves to the exact solution; the C++ field is within 2e-6 px of it.
8. **Photo 2 has no text-row constraints.** Stage 1 is still sideways when the warp map runs (orientation is applied only at the end), so `text_rows` finds **0** rows there. The other pages have 81–174. I left this as it is, because it is reference behaviour.
9. **A possible platform dependence in `chain_trace`:** the seed piece is chosen with `np.argsort`, which is not a stable sort. With ties in the longest piece length, numpy's choice can depend on the CPU's SIMD sort. The port takes the first piece. This was not observed to matter on the test pages.
10. **numba works here.** The round-3 reviewer could not use it; in this environment it works, and the reference ran with it.

## Not done / not verified on a real phone

- **Nothing has run on Android:** no phone, no emulator (no KVM).
  - Not verified: the UI, CameraX capture, EXIF handling, JNI calls, save/share, and the PDF output.
  - GLES 3.1 shader compilation on Adreno or Mali drivers has not been seen; the shaders were compiled and checked only on Mesa llvmpipe.
  - Checked statically: the APK builds, and every Kotlin `external` function has a matching exported JNI symbol.
- **Timings on two named devices are not measured.** Use the in-app self-test.
- **Map parity on ARM is not verified.** Android OpenCV is built with clang, which may fuse multiply-adds inside OpenCV's own float code. The core itself is compiled with `-ffp-contract=off`. The self-test reports map parity on the device.
- **Phone photos above 2400 px are downscaled for detection and the surface fit.** These kernel and threshold choices are untested: the reference was only tuned on 2212–2300 px photos. Stage 1 and the final resample use the full photo, which is capped at 4096 px on decode.
- **Cleanup is my design** (the brief gives the steps, the reference has no implementation). It was checked visually on the two photos and checked GPU = CPU. It was not checked on colour-critical or low-light pages.
  - Solid colors does not port v1's "coloured areas outside the grid → rectangles" step, because that step snaps to the redrawn lines.
- **Device behaviour that is not verified:**
  - GPU memory on large photos: a 4096×3072 RGBA texture is about 50 MB; the app falls back to the CPU if a GPU call fails.
  - Behaviour under thermal throttling.
  - The camera focal length from Camera2 on multi-camera phones: only the first lens focal length is read.
- **Not done:** cancelling a running job, text-only pages, and pinch zoom on the crop screen. Edits are saved to the page cache when you leave the page or the app goes to the background; a process kill before that loses the last edits.
- **Solid colors GPU vs CPU on scan p3:** up to 22 levels on 33 values (the other pages are within 1 level); scan p3 also keeps 2207 px of (255,255,254) slivers that are already in the plain Auto color render.
- **Text-only pages:** Only table pages exist among the test inputs, as in the reference.
