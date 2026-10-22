# Round 11, RUNTIME stream: one runtime (LiteRT) in the app, and a smaller app

Branch `r11/runtime` (worktree `C:\Users\User\Desktop\fs_r11_runtime`), from `r11/main` db9ef56, with
`litert/main` fc0442c and `r11/main` ddfd6f0 (Round 10) merged. Measured builds: 322fdd7 (clean tree; this commit
adds only this report and its scripts).

## What changed
- Android: LiteRT 1.4.0 (AAR; the runtime reports itself as "LiteRT 2.20.0-dev0") is the only runtime. CPU = LiteRT CPU
  kernels + XNNPACK delegate (created explicitly: the C API does not apply it by itself), GPU = LiteRT GPU delegate
  (fp32) where OpenCL exists. ONNX Runtime, the .onnx models, QNN (flavour, packs, QnnPacks.kt) and the litert_gpu
  pack (Packs.kt) are gone from the Android build.
- Desktop (tools/desktop_check): ONNX Runtime unchanged (FS_WITH_ORT).
- Models (.tflite, app/src/main/litert/models, base APK, stored uncompressed): pagefinder (new, fp32), uvdoc_static,
  syntheticdoc_hr_static, shadenet (new, any H x W), shadenet_512x352 (GPU), rulenet (new, any H x W).
- OpenCV: slim static build (tools/opencv_min, with KleidiCV) linked into libflatscan, libc++ static
  (`-PopencvMin=aar` = the old OpenCV AAR + libc++_shared).
- Self-test photo: lossless WebP (same pixels).

## Size (release, arm64-v8a; baseline = flatscan_out/flatscan-release.* from 9f2c93e)
Play download per phone: `bundletool 1.18.1 build-apks`, then `get-size total --device-spec` (arm64-v8a, en, 420 dpi,
SDK 35; splits base-arm64_v8a, base-en, base-master, base-xxhdpi):

| | before (ORT) | after (LiteRT) |
|---|---|---|
| Play download per arm64 phone | **66.37 MB** | **54.59 MB** (-11.8 MB, -18 %) |
| release APK (arm64, default packaging) | 70.28 MB (libs compressed) | **66.86 MB** (libs stored uncompressed) |
| release APK with `-PcompressLibs=true` | - | 58.49 MB |
| release AAB | 139.05 MB (with the NPU packs) | 63.66 MB |
| release APK with `-PopencvMin=aar` (old OpenCV AAR) | - | 84.46 MB |

Per component, download bytes (each entry of the phone's splits deflated at level 9; the sums, 66.33 / 54.59 MB, match
bundletool's totals), MB:

| component | before | after |
|---|---|---|
| models | 42.96 (syntheticdoc_hr 15.68, uvdoc 14.62, pagefinder 11.69, shadenet 0.97) | 43.37 (syntheticdoc_hr 14.64, uvdoc 14.62, pagefinder 11.69, shadenet 0.97, shadenet_512x352 0.97, rulenet 0.48) |
| native libs | 16.28 (opencv_java4 8.89, onnxruntime 6.53, flatscan 0.46, c++_shared 0.40) | 5.85 (flatscan incl. slim OpenCV 3.08, tensorflowlite_jni 1.74, tensorflowlite_gpu_jni 1.03) |
| dex | 1.50 | 1.36 |
| assets other (self-test photo, ...) | 5.24 | 3.65 |
| res + other | 0.35 | 0.36 |

The saving is the runtime + OpenCV (-10.4 MB) and the WebP self-test photo (-1.6 MB); the models are as big as before
(fp32; the rule net adds 0.48 MB). Scripts: `runtime/scripts/comp.py`, `dl.py`.

## Parity on the emulator (AVD fs35, Android 15 x86_64, -gpu swiftshader_indirect, read-only)
One session (11:00-11:20 Oct 22): ORT CPU debug APK from r11/main ddfd6f0 vs LiteRT CPU debug APK from 322fdd7, batch
mode, accel cpu, no shadow net, PNG page; the self-test photo once, DocUNet 1_1 4_2 17_2 18_1 twice (`du`, `du2`).
`runtime/scripts/emu_cmp.sh` + `cmp_rt.py`; full output `runtime/emu_ort_vs_litert.txt`.

| page | net outputs max / p99 abs diff (logits; p1, rc0, rc2) | page pixels differing | max | s per page ORT / LiteRT (du; du2) |
|---|---|---|---|---|
| self-test photo | 4.1e-5 / 1.4e-5; <= 6e-7 / 3.6e-7 | 0.94 % | 9 levels | 55.9 (cold) / 18.0 |
| 1_1 | 4.2e-5 / 1.2e-5; <= 6.6e-7 / 3.6e-7 | 0.83 % | 14 | 10.5 / 14.0; 18.8 / 14.0 |
| 4_2 | 3.0e-5 / 1.4e-5; <= 7.2e-7 / 3.6e-7 | 0.02 % | 5 | 10.3 / 13.9; 11.0 / 14.0 |
| 17_2 | 2.7e-5 / 8.1e-6; <= 6.6e-7 / 4.2e-7 | 1.09 % | 15 | 16.1 / 17.8; 21.3 / 17.1 |
| 18_1 | 3.5e-5 / 1.0e-5; <= 7.2e-7 / 3.6e-7 | 0.21 % | 5 | 19.9 / 14.6; 21.1 / 14.5 |

- Pixels > 8 levels off: 0.00 % on every page. JSON: only float fields differ (corners <= 0.42 px, metrics <= 1.5e-3); no
  decision (net chosen, guard, rule, orientation) differs. du and du2 give the same outputs (both runtimes deterministic).
- Time: 8 DocUNet page runs, mean 16.1 s (ORT) vs 15.0 s (LiteRT); batch totals 57 / 72 s vs 60 / 60 s. LiteRT is
  steadier, but SyntheticDoc-HR is slower on x86_64 XNNPACK: 6.6-6.8 s vs ORT 1.8-7.3 s; the rule net 1.1-2.1 s vs
  0.6-1.6 s. An earlier session under heavier machine load (07:11-07:38) was 2.5-4x slower for both runtimes.
- Self-test "map parity vs reference" FAIL: max 6.4989 px, mean 0.325 px on the LiteRT build, the same in two sessions
  (07:37 and 11:19; the draft's 15.47 px was an earlier build). The ddfd6f0 ORT build could not run it: its MainActivity
  has no `--ez selftest` intent (600 s, no report). The map is made with PipelineOptions defaults (no nets: no uv_out,
  no flat_nets_fn), and core/ differs from ddfd6f0 only in infer.cpp, nets.cpp (net runtime / backend names) and
  CMakeLists, so the runtime cannot move it: the reference warpmap (2e90d55, 48 core/src commits older) is stale, not
  a regression. Inferred from the code, not measured on the ORT build. Check 2 (GPU vs CPU resample) PASS.

## GPU fallback and arm64 translation (11:31-11:33, `runtime/scripts/emu_arm.sh`)
- No OpenCL on the emulator: accel gpu logs "LiteRT_GPU unavailable: no OpenCL library" per net and runs on the CPU;
  JSON backends all "cpu"; no crash (x86_64 build in both sessions, arm64 build below).
- arm64 under translation (abilist x86_64,arm64-v8a, libndk_translation): the arm64 release APK (R8) installs, starts
  (1.8 s) and is alive after 20 s; the arm64 debug APK (lib/arm64-v8a/libflatscan.so loaded) runs the self-test photo
  with accel cpu (42 s) and gpu (35 s, CPU fallback), outputs bit for bit identical between the two; vs the x86_64
  build: net outputs <= 8.1e-6, page 3.14 % of pixels, <= 7 levels. 0 crash lines (FATAL/SIGSEGV/SIGILL/Abort
  message/UnsatisfiedLink).

## Decisions
- Default packaging keeps libs uncompressed (smaller download, mmap); `-PcompressLibs=true` for a side-loaded APK.
- Models stay fp32 (no quantisation in this stream: it is meant to be output-neutral).

## Not verified
- Not run on a phone: no arm64 CPU timing, no real GPU delegate run (OpenCL), no NPU (removed).
- LiteRT vs ORT on all 100 DocUNet pages not scored on Android (4 pages here; the desktop keeps ORT).
- Self-test map parity on the ORT build (no intent there); the reference warpmap should be regenerated.

## Reproduce
Release builds: `./gradlew assembleRelease bundleRelease` (+ `-PcompressLibs=true`, `-PopencvMin=aar`); copies in
`C:\Users\User\Desktop\flatscan_out\r11\runtime\` (SHA256SUMS.txt). Emulator, under the heavy lock:
`REPS=2 emu_cmp.sh OUT new_ort=ORT.apk newx=LRT.apk`, then `cmp_rt.py OUT/new_ort OUT/newx`; arm64:
`emu_arm.sh OUT RELEASE.apk ARM64_DEBUG.apk`.

## Notes
- 1st emulator run (built-in kernels, XNNPACK not applied): UVDoc 4-5 s vs ORT 1-2 s, SD-HR 16-17 s vs 6-11 s per
  call -> fixed by creating the XNNPACK delegate explicitly.
