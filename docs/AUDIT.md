# Phase 0 audit: FlatScan against the six-feature plan

Audited commit: `58908db` (v0.1.0-r11). Read: every Kotlin file in `app/`, the JNI bridge, the parts of
the C++ core that matter for the plan (`pipeline.{h,cpp}`, the finger erase, `fill_with_paper`), the
build files, `README.md` and `docs/results/r11/*.md`. The core's geometry code (`warpmap.cpp`,
`surface.cpp`, `pagefind.cpp`, `nets.cpp`, `infer.cpp`) was read only far enough to place each stage.

Nothing in this audit changed app code; this file is the only change. The `local.properties` written for the
build in section 3 is git-ignored.

---

## 1. Architecture map

### 1.1 Modules

One Gradle module, `:app`, plus a portable C++ core shared with desktop/host harnesses.

| Path | Language | What it is |
|---|---|---|
| `app/src/main/java/com/albus/flatscan/` | Kotlin, Compose | The whole Android app: one activity, one view model, screens in `ui/` |
| `app/src/main/cpp/jni_bridge.cpp` | C++ | JNI: `NativeLib.*` → the core; Bitmap ↔ `cv::Mat`; session handles |
| `core/src/` | C++17 | The image pipeline (≈ 15 k lines), compiled into `libflatscan.so` on Android and into `fs_cli` / `pipeline_cli` on desktops |
| `host/` | C++ | `fs_cli`: Linux/Windows acceptance and stage-parity harness (EGL/GLES via Mesa) |
| `tools/desktop_check/` | C++, Python, sh | `pipeline_cli` and the gate scripts used for every measured number in the README (Windows) |
| `tools/` (rest) | Python | Fixture exporters, comparators, dataset harness (`library/`), net training (`shadow_net/`, `rule_net/`, `flatnet/`), LiteRT conversion |
| `docs/results/` | text, md | Raw logs and reports of every round |

### 1.2 Main Kotlin classes

| Class / file | Role |
|---|---|
| `MainActivity` | Hosts `App()`: a `when (vm.screen)` switch, no Navigation library. Debug-only batch mode and self-test via intent extras. `onStop` → `vm.persist()` |
| `ScanViewModel` (2018 lines) | Everything: navigation state, import queue, crop state, page-finder cache, book spreads, processing, the render loop, edits, undo, export, self-test, benchmark, batch mode |
| `Document.kt`: `PageMeta`, `DocumentStore` | The one current document on disk (see 1.5); per-page processing parameters; `geometryKey()` decides whether a stored session can be reused |
| `NativeLib` | JNI surface: `pageFinder`, `detect`, `process`, `render`, `renderThumb`, `renderBefore`, `renderOverlay`, `setErase`, `lines`, `snapshot`/`writeSnapshot`/`restoreSession`, `selfTest`, `benchBackends`, `batchPhoto` |
| `RenderParams`, `FilterKind`, `ScreenMode` | Filter (Original, Scan, Lighten, Auto color, Grayscale, High contrast, B&W), strength, Solid colors, background cleanup, levels, screen-photo mode |
| `PageLook` | Kotlin-side brightness/contrast/saturation/warmth colour matrix and page shape (A4, measured, Letter, ID card) |
| `EditHistory` | Per-page undo/redo of render edits (60 steps, 450 ms debounce) |
| `Exporter`, `ExportOptions` (`PdfOptions`, `ExportPrefs`, `DocumentName`) | PDF via `android.graphics.pdf.PdfDocument`, JPG, share intents |
| `Settings`, `Features` | SharedPreferences: default filter, auto-rotate, output width, per-net feature flags, acceleration, crop review, book RTL |
| `Accel`, `NetEnv`, `CrashGuard` | Net backend choice (Auto/CPU/GPU on LiteRT), crash guard that disables the GPU after a native crash |
| `ImageIo.kt` | `loadPhoto`: EXIF-upright decode, capped at 4096 px, focal length from EXIF |
| `ui/` | `HomeScreen`, `CameraScreen` (CameraX), `CropScreen`, `ProcessingScreen`, `ResultScreen` (+ `AdjustTools`, `LevelsEditor`, `PageViewer`), `PagesScreen`, `ExportUi`, `SettingsScreen` |

### 1.3 Pipeline stages and where they run

From `README.md` (Architecture table) and `core/src/pipeline.cpp`, checked against the code:

| Stage | Main code | CPU / GPU |
|---|---|---|
| Page finder net (segmentation, 256×256 logits) | `nets.cpp`, `pagefind.cpp`, `infer.cpp` | CPU always (LiteRT CPU kernels + XNNPACK) |
| Colour-threshold page mask (fallback when the net is off/fails) | `corners.cpp` `paper_mask` | CPU |
| Corner fitting, surface model fit (TRF port), quality gate | `corners.cpp`, `surface.cpp`, `lsq.cpp`, `pipeline.cpp` `quality_gate` | CPU (multi-threaded fit) |
| Flattening nets UVDoc (two passes) + SyntheticDoc-HR | `nets.cpp`, `uvdoc.cpp`, `infer.cpp` | LiteRT: CPU by default; GPU delegate when Auto picks it or the user picks GPU. **The GPU delegate has never run on a real GPU** (README) |
| Homography / perspective-only path | `pipeline.cpp` (`perspective_only`, `force_homography`) | CPU |
| Stage-1 page, orientation page | `pipeline.cpp`, `orient.cpp` | CPU; GPU path exists, off by default |
| Line stage (global + local warp map, rule net, guards) | `warpmap.cpp`, `sparse.cpp`, rule net in `nets.cpp` | CPU; global remap on GPU when the GL context has `precise`; rule net CPU only |
| Composition, map expansion, final bicubic resample | `pipeline.cpp`, `resample.cpp`, `gpu.cpp` | GPU (GLES 3.1 compute), CPU fallback |
| Shadow removal (ShadeNet) | `pipeline.cpp` + `nets.cpp` | LiteRT CPU or GPU |
| **Finger erase** | `pipeline.cpp` `erase_fingers` → `fill_with_paper` | CPU |
| Cleanup filters (background, white point, tone, specks) | `cleanup.cpp`, `gpu.cpp` | GPU, CPU fallback |
| Solid colors | `solid.cpp` | CPU |
| Screen-photo descreen | `screen.cpp` | CPU |
| Look (colour matrix), page shape | `PageLook.kt` | Kotlin: Compose colour filter on screen, one Canvas pass on save |

### 1.4 Flow of one scan

1. **Input.** Camera: CameraX `ImageCapture` (max quality) writes `cacheDir/capture_<ms>.jpg` (older captures
   deleted first) → `vm.loadUri(Uri.fromFile(..))`. Gallery: system photo picker → `vm.importUris` (several
   photos run as a batch).
2. **Load.** `loadPhoto` decodes EXIF-upright, ≤ 4096 px long side, ARGB_8888. A new page id is created and the
   **decoded bitmap is re-encoded** to `document/<id>/source.jpg` (JPEG q95) on an IO coroutine.
3. **Detect.** `NativeLib.pageFinder` (net logits, cached per bitmap) → `NativeLib.detect` (corners, quality
   gate reasons, book-spread pages). A trusted crop goes straight to processing; a doubtful one (or every one
   with Settings › *Check every crop*) opens `CropScreen`.
4. **Process.** `NativeLib.process(...)` on the single `flatscan-native` thread runs `run_photo` (all of 1.3 up
   to and including finger erase and shadow removal) and returns a session handle. `writeSnapshot` stores the
   processed page in `document/<id>/session/` keyed by `PageMeta.geometryKey` so reopening skips the geometry.
5. **Render.** `NativeLib.render(handle, params, protect, useGpu, bitmap)` applies the filter; the Kotlin look
   is applied on top. Every edit re-renders from the session (not from the photo).
6. **Persist.** `commitCurrent()` writes `meta.json` and `out.jpg` (rendered, rotated page, JPEG q95) on leaving
   the page and in `onStop`.
7. **Export.** `exportPages()` returns the `out.jpg` files in order → `Exporter` → SAF `CreateDocument` /
   `OpenDocumentTree`, or a share intent via `FileProvider` (`cacheDir/shared/`).

### 1.5 Storage

All under **`context.cacheDir`** (not `filesDir`):

```
cacheDir/
  capture_<ms>.jpg                 last camera capture
  document/
    doc.json                       {"pages": [id, ...]}  — the one current document's page order
    name.txt                       the document's name (optional)
    <id>/source.jpg                upright photo, re-encoded JPEG q95, ≤ 4096 px
    <id>/meta.json                 PageMeta: corners, detected corners, crop rotation, focal, RenderParams,
                                   protect rects, rotation, spread info, ignore areas, perspectiveOnly,
                                   erase rects, PageLook
    <id>/out.jpg                   rendered page, JPEG q95 (thumbnail source and export source)
    <id>/session/                  processed page (JPEG q95), paper alpha, before page, deformation grid,
                                   fitted lines, ready.json {key, info, details}
  shared/                          files handed to share intents
```

Writes go through `*.tmp` + `renameTo`. There is **no database**: JSON files only. SharedPreferences files:
`settings`, `export`, `look`, plus the crash guard's.

### 1.6 Build

| Item | Value |
|---|---|
| Gradle wrapper | 8.14.3 |
| Android Gradle Plugin | 8.11.1 |
| Kotlin | 2.1.21 (Compose compiler plugin 2.1.21) |
| compileSdk / targetSdk / minSdk | 35 / 35 / 26 |
| NDK / CMake | 27.2.12479018 / 3.22.1 |
| JVM target | 17 |
| ABIs | `arm64-v8a` by default (`-Pabis=` adds `x86_64`) |
| applicationId / version | `com.albus.flatscan`, versionCode 1, versionName 0.1.0 |
| Release | R8 + resource shrinking, **signed with the debug key** |

### 1.7 External libraries

| Library | Version | Use |
|---|---|---|
| OpenCV | 4.11.0 (Maven AAR via prefab, or the slim static build `tools/opencv_min`) | Everything in the core |
| LiteRT (TensorFlow Lite runtime) + GPU delegate | 1.4.0 | All nets; `.so` copied out of the AARs and `dlopen`ed by `infer.cpp` (no Java API) |
| CameraX | 1.4.2 | Camera |
| Jetpack Compose BOM | 2025.05.00 (Material 3, icons-extended) | UI |
| AndroidX activity, core-ktx, lifecycle, exifinterface | 1.10.1, 1.16.0, 2.9.0, 1.4.1 | — |
| `android.graphics.pdf.PdfDocument` | platform | PDF writing |

Not present: Room/SQLite, WorkManager, any HTTP client, ML Kit, Tesseract or any OCR, a PDF library
(PdfBox etc.), DataStore, Navigation, Hilt. Bundled models (`.tflite`, 47 MB): pagefinder, uvdoc_static,
syntheticdoc_hr_static, shadenet (+ 512×352), rulenet. The `.onnx` copies in `assets/models/` are excluded
from the APK (desktop only).

### 1.8 Permissions

Manifest: `CAMERA` only. **No `INTERNET` permission.** `allowBackup="false"`. No `android:supportsRtl`
attribute (see 4.2).

---

## 2. Status of the six features

| Feature | Status | Where | What works | Gaps vs. the plan |
|---|---|---|---|---|
| 1. Scan history | **partial** (a single, non-persistent-by-design document) | `Document.kt` (`PageMeta`, `DocumentStore`), `ExportOptions.kt` (`DocumentName`), `ScanViewModel.init/openPage/newDocument`, `ui/Screens.kt` `HomeScreen`/`DocumentCard` | Pages survive app restart and process death (restored from `doc.json` in `ScanViewModel.init`). Per-page processing parameters are stored (`meta.json`) and **editing is already non-destructive**: re-crop, unwarp on/off, ignore areas re-run `process` from `source.jpg`; filter/look edits re-render from the stored session. Thumbnails come from `out.jpg` (`loadThumbnail`, 480 px). Rename exists (`DocumentName`) | Only **one** document; "Start a new document" **deletes** the previous one. No `Document` entity, no created/modified dates, tags, sync status. No database (plan: Room). No history screen, search, multi-select, trash. **Storage is `cacheDir`**, which Android may clear under storage pressure: the plan's "app-private storage" needs `filesDir`. The stored "original" is a **re-encoded, ≤ 4096 px** JPEG q95, not the captured file. No storage-used view. No migration test (nothing to migrate yet) |
| 2. Page rearrangement | **partial** | `ui/PagesScreen.kt`, `ScanViewModel.movePage/deletePage`, `ExportUi.kt` | Grid of page thumbnails; long-press drag to reorder (implemented as repeated `movePage(±1)`); per-page ⋮ menu: move left/right, delete (with confirm); add page from camera or gallery (appended at the end); rename; save/share | No **auto-scroll** while dragging (no scroll code in `PagesScreen`), no multi-select, no rotate/duplicate from the grid (rotate exists per page on the result screen), no **reverse all**, no insert at a position, no import of PDF pages, no move between documents / split (only one document exists). Undo/redo (`EditHistory`) covers one page's render edits only, not page order. No tests |
| 3. Better output | **partial** | `Exporter.kt`, `ExportOptions.kt`, `ui/ExportUi.kt` | Multi-page PDF (A4 / Letter / "same as the page", margins none/small/normal, image size full or ≤ 150 dpi); one JPG per page into a folder (SAF `OpenDocumentTree`); single JPG (SAF `CreateDocument`); share PDF or JPGs. File name = document name or `Scan yyyy-MM-dd HH.mm`. Page images are centred when the aspect differs | **No OCR at all**, so no text layer. `PdfDocument` cannot set Title/Creator/etc. and gives no control over image encoding (not measured what it embeds). No 1-bit B&W storage, no CCITT G4/Flate choice, no JPEG quality presets for the PDF (pages enter as already-decoded `out.jpg`), no size presets with an estimate, no PNG, no PDF+JPEG together, no filename template / date format, no default export folder (always asks). The B&W **filter** exists, but its output is still an 8-bit RGB JPEG |
| 4. Whiteboard mode | **missing** (building blocks exist) | — (`PipelineOptions::perspective_only`, `pagefind.cpp` whole-frame fallback, `ScreenMode`, `PageLook` saturation) | Perspective-only geometry exists per page (Unwarp off). The page finder falls back to the whole frame when < 2 % looks like a page. The final resample uses a constant **white** border (`resample.cpp` l. 84, `gpu.cpp`), so no black corners; but white is still filler, not photo pixels, so the plan's "never output pixels from outside the photo" needs the whiteboard quad clamped to the frame. A screen-photo descreen exists for photos of monitors | No mode switch on the camera screen; no processing profile. Cleanup filters are designed for paper (paper mask, white point), not for marker colours on a board; no glare reduction; no dark-background (projector) sub-option. The user's projector photo (`user__projector_screen`) is in the owner's 35-photo set on `E:\fs_datasets` / `model_zoo`, **not in this repo**. No whiteboard test photos in the repo |
| 5. Finger detection | **exists** (automatic detection + removal); manual tool is rectangles, not a brush | `core/src/pipeline.cpp` `erase_fingers` (l. 358) and `fill_with_paper` (l. 1342); `PipelineOptions::erase_fingers` (always true in the app); `PageMeta.erase` + `NativeLib.setErase` + Erase panel in `ResultScreen` | **Method:** runs on the *unwarped* page. Candidates = holes in the paper alpha (non-paper, `paper < 128`) that touch the page border (location prior already in place) and are ≥ 0.05 % of the page. Accept if ≥ 35 % of the candidate is skin, where skin = YCrCb box `Y ≥ 40, Cr ∈ [max(140, paperCr+7), 180], Cb ∈ [77, min(125, paperCb−4)]` (thresholds **relative to this page's paper colour**, not sampled from the hand); reach into the page ≥ 2 × border width (border = max(4, w/100) px); Canny edge density inside the eroded candidate ≤ 0.03 (fingers are smooth, text is not). Skipped when holes cover > 25 % of the page or paper < 20 %. **Removal:** mask dilated by the border width, Telea inpaint at ¼ size, soft-edged blend, area marked as paper (so filters whiten it). Text under a finger is not invented. **Manual:** Erase tool, rectangles in page fractions, stored in `meta.json`, applied with the same `fill_with_paper` | **No labelled test set and no metric** in the repo: the only evidence is README prose ("erased on 2 of 35 photos", "two earlier false positives rejected"); the 35 photos are off-repo. No IoU or false-positive numbers exist, so the baseline must be built first. No shape test. Not adaptive to the hand's colour. The automatic mask is **not stored** in the page's parameters and **cannot be turned off or edited** per page (no setting, no toggle): a false positive can only be fixed by re-cropping. A finger over text is filled with paper silently, **not marked**. Manual tool is boxes, not a brush, and only adds (cannot subtract from the automatic mask) |
| 6. Sync to own server | **missing** | `Exporter.shareIntent`, `Exporter.saveJpgsToFolder` (SAF tree) | Share sheet and SAF folder export exist, so "local folder via SAF" is close for a manual one-off | No `INTERNET` permission, no network code, no WorkManager, no credential storage, no per-document sync status (no documents), no persisted SAF tree permission (`takePersistableUriPermission` not used) |

---

## 3. Baseline numbers

### 3.1 Tests that exist in the repo

- **No unit tests and no instrumented tests**: there is no `app/src/test` or `app/src/androidTest`, and no test
  dependencies in `app/build.gradle.kts`.
- **Benchmarks / acceptance harnesses exist but depend on data outside the repo:**
  - `tools/run_acceptance.sh`, `tools/run_m1.sh`, `tools/run_stage_parity.sh` → need `app_ref/` (the Python
    reference with its test images) and its outputs. Not in the repo.
  - `tools/uvdoc_check.py`, `tools/desktop_check/gate_*.sh`, `tools/library/run_library.py` → need the 35 real
    photos, DocUNet and `E:\fs_datasets` (owner's Windows machine). Not in the repo.
  - `host/fs_cli selftest-resample` / `gpu-stages` → synthetic, self-contained, but need OpenCV **4.11 exact**,
    built without IPP, plus EGL/GLES.
  - On-device: Settings › *Run self-test* (bundled `assets/selftest/photo1.webp` + reference warp map).
- The **last recorded numbers** (README, measured on the owner's desktop, not re-run here): DocUNet 128
  photos, automatic: MS-SSIM 0.590, LD 5.16 (5.122 with the rule net in the core, R11), AD 0.189; 35 real
  photos text-row spread mean 1.27°; map parity with the reference ≤ 0.089 px; GT bench P1/P2 1.082 / 1.199 px.

### 3.2 What I ran in this container

| Check | Result |
|---|---|
| `./gradlew assembleDebug -Pabis=arm64-v8a` (SDK 35, NDK 27.2, CMake 3.22.1 installed for this audit; OpenCV from the Maven AAR) | **BUILD SUCCESSFUL** in 5 m 15 s (cold, 4 cores). One C++ warning (`warpmap.cpp:1391` unused `y`), one Kotlin deprecation (`CameraScreen.kt:100` `LocalLifecycleOwner`). Debug APK 101.8 MB (uncompressed native libs and models; the README's 54.8 MB is the Play download per phone) |
| `./gradlew lintDebug` | **0 errors, 65 warnings, 1 hint** (baseline for later PRs) |
| Unit / instrumented tests | none exist |
| Host harness (`fs_cli`) | **not run**: no OpenCV 4.11 in the container and its source download from GitHub is blocked by this environment's network policy |
| Document benchmark (DocUNet, 35 photos) | **not run**: the data is not in the repo |
| On a phone or emulator | **not run**: no device, no KVM |

**Consequence for Phase 1:** the plan's rule "run the benchmark before and after every image-processing
change" cannot be met from this environment as the repo stands. See 4.1, item 1.

---

## 4. Risks and conflicts

### 4.1 Things in the code that conflict with the plan

1. **No runnable benchmark in the repo.** Every image number comes from data on the owner's machine. For
   features 4 and 5 (the two that touch image processing) I need either the data, or permission to commit a
   small test set (licence-clean photos), or for the owner to run `pipeline_cli` / `uvdoc_check.py` before and
   after. A cheap guard I can add without data: a golden-output test of the core on the bundled self-test
   photo through `fs_cli` (needs OpenCV 4.11 in CI or here).
2. **`cacheDir` storage.** History must not live in the cache. Moving to `filesDir` is a one-time migration
   of the existing current document.
3. **"Start a new document" deletes the old one.** With history this becomes "close and keep".
4. **The "original" is not the original.** `source.jpg` is the decoded bitmap, downscaled to ≤ 4096 px and
   re-encoded at q95. The core's analysis caps at 4096 px anyway, so keeping the camera file byte-for-byte
   costs disk (≈ 3–8 MB per page) but makes "edits never lose quality" literally true. Needs a decision
   (4.3, D2).
5. **Single-threaded native session model.** `ScanViewModel` holds one live native session; history, export
   and sync need to work from stored files (`out.jpg`, `meta.json`) without a session. That is already how
   export works, so it fits, but re-rendering a page in another document needs `restoreSession` per page.
6. **`ScanViewModel` is 2018 lines** and owns navigation, documents, processing and export. Adding a history
   screen, a page editor with session-wide undo, and sync on top of it will need splitting (a document
   repository + navigation) first, or every feature will collide in this file.
7. **PDF writer.** `PdfDocument` cannot write metadata, an invisible text layer, 1-bit CCITT/Flate images or
   chosen JPEG quality. Feature 3 needs a different writer: either a small hand-written PDF writer (objects,
   xref, DCTDecode passthrough of JPEG bytes, Flate/CCITT for 1-bit, a `3 Tr` invisible text layer with a
   Unicode `ToUnicode` CMap for Hebrew) or a library such as PdfBox-Android (Apache-2.0, ≈ 2–3 MB after R8).
8. **Finger mask is not a stored parameter.** The plan stores "finger mask" per page and adds a brush; today
   the automatic result is recomputed inside `process` and not exposed. It must come out of the core as a
   mask (session file) and be overridable.
9. **README is partly stale.** The Acceleration/Build sections still describe ONNX Runtime, QNN/NNAPI, the
   on-demand asset packs and `QnnPacks.kt`, which Round 11 removed (`app/build.gradle.kts` is LiteRT only, no
   packs). Each Phase 1 PR updates its own section; the stale parts should get one clean-up PR.
10. **Release is signed with the debug key**; nothing in the plan needs Play, but sync adds network
    credentials to an app anyone can re-sign. Not a blocker; noted.

### 4.2 Hebrew / right-to-left

- Positives: book-spread order follows the locale (Settings › *Book spreads: right page first*); the
  orientation's upright cue is tuned on Hebrew pages (`orient_min_conf` comment).
- **The manifest has no `android:supportsRtl="true"`**, so Compose layouts are **not mirrored** in a Hebrew
  locale, and every UI string is hard-coded English in Kotlin (`strings.xml` holds only the app name). Not in
  the plan, but the history/search/page-grid UI added in Phase 1 will be built on this. Recommend adding
  `supportsRtl` early (user-visible: the existing UI mirrors).
- **OCR:** none exists. ML Kit has no Hebrew; Tesseract (`heb` + `eng`) does. Tesseract4Android (Apache-2.0)
  adds native libraries (≈ 3–5 MB per ABI) plus traineddata: `tessdata_fast` heb + eng ≈ 1 + 4 MB, `tessdata_best`
  ≈ 9 + 15 MB (sizes from memory, **to be measured**). Given the R11 effort to cut the download to 54.8 MB, the
  model choice (fast vs best, bundled vs downloaded on first use) is a user decision (4.3, D4). Downloading on
  first use would need `INTERNET` outside of sync, which the ground rules forbid unless approved.
- Text-layer alignment: OCR must run on the **final rendered page** (`out.jpg` after rotation and page
  shape), not on the session page, because `PageLook` resizes the page for non-A4 shapes after the core. Hebrew
  words go in the text layer in logical order with a `ToUnicode` map; viewers handle bidi on extraction.
  Acceptance needs a Hebrew page search test in two viewers.

### 4.3 Decisions for the user

| # | Decision | My default if you don't say |
|---|---|---|
| D1 | Commit a small labelled test set to the repo (whiteboard ≥ 13 photos, finger ≥ 30, control ≥ 30) or keep it off-repo and have you run the before/after numbers | Off-repo; I write the harness and you (or a session with the data) run it |
| D2 | Keep the camera file byte-for-byte as the original (more disk) or keep today's ≤ 4096 px q95 re-encode | Keep the camera file; gallery imports keep the picked file |
| D3 | PDF writer: hand-written minimal writer or PdfBox-Android | Hand-written (no size cost, full control of 1-bit and text layer) |
| D4 | Tesseract data: `fast` bundled, `best` bundled, or download on first use | `fast` heb+eng bundled; OCR optional per export |
| D5 | Add `android:supportsRtl="true"` now | Yes, in the History PR |
| D6 | Adding the `INTERNET` permission in the Sync PR (user-visible; the app is offline today) | Yes, only in that PR |

### 4.4 Plan items that are hard given how the code is built

- **"Re-editing corners on an old scan gives the same result as on a fresh scan."** Holds today only while
  the app version, net backend and settings are the same: the result depends on `Settings` (nets on/off, rule
  net, shadows, output width), the acceleration backend (GPU vs CPU differ within tolerance) and the app
  version. The test should pin those, and the stored page should record the settings it was made with.
- **"Drag page 5 to position 1 in a 20-page document" UI test.** Needs an instrumented test on an emulator;
  this container has no KVM, so it can be written but not run here.
- **Integration tests against WebDAV and paperless-ngx in Docker.** Need Docker (or a local WebDAV server in
  the JVM test) where the tests run; the `docker` CLI exists in this container (daemon and image pulls not
  checked). Offline-mid-upload "exactly
  once" needs an emulator (WorkManager + network toggling) or a Robolectric/WorkManager test harness with a
  fake transport.
- **Black-and-white output 3× smaller than colour.** Easy with 1-bit CCITT G4, but the core's B&W filter keeps
  "the tones of dark and saturated areas" (README): it is not a pure two-level image. A true 1-bit export needs
  a threshold step at export time, which can lose those picture tones. Measured trade-off needed.
- **Whiteboard "no regression on the document benchmark".** Achievable by keeping the whiteboard profile a
  separate branch in `run_photo` (perspective only, new cleanup) with the document path untouched, but it can
  only be *shown* with the off-repo benchmark (4.1, item 1).
- **Finger "mark the area if it covers text".** Today the fill is on the unwarped page before the filters, and
  the core does not know whether text was under the finger (it is occluded). "Mark" will have to mean: a
  visible outline or hatch in the output plus a flag in the page parameters, decided by whether text rows run
  into the filled area on both sides.

---

## 5. Recommended order

The plan's order (1 → 6) is sound because 2, 3 and 6 need documents. Two changes:

1. **History** (with the `filesDir` migration, `supportsRtl`, and a document repository split out of
   `ScanViewModel`).
2. **Page rearrangement.**
3. **Sync** before Better output: sync needs only the existing PDF export plus the filename template, and is
   independent of OCR; Better output is the largest item (PDF writer, OCR, Hebrew) and its size and
   app-size decisions (D3, D4) can be settled meanwhile. If you'd rather have searchable PDFs first, keep the
   plan's order; nothing else depends on this swap.
4. **Better output** (PDF writer, OCR, presets, templates). Sync then picks up the new writer for free.
5. **Finger detection**, starting with the labelled set and the baseline (blocked on D1 for data).
6. **Whiteboard mode** (blocked on D1 for its photos; it touches `run_photo` and must be measured against the
   document benchmark).

Finger detection and whiteboard both need data I do not have; doing them last gives time to collect it.
