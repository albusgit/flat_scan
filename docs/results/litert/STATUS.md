# LiteRT GPU backend: status

Integration of litert/models, litert/runtime and litert/app on `litert/main`, 2026-10-21. Base:
`claude/android-document-scanner-fkka2i` at ba245e8. **Speed mode** (owner's decision): builds and smoke checks
only, no benchmark runs, no DocUNet scoring. **Nothing has run on a phone, and the GPU delegate has not run
anywhere**: the emulator has no OpenCL. Every phone result is the HUMAN GATE below.

- `<SCR>` = `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad` (temporary).
- `<OUT>` = `C:/Users/User/Desktop/flatscan_out/litert`: the builds, `SHA256SUMS.txt`, and in `<OUT>/check/` the
  check scripts, their logs and the emulator outputs (copies, because the scratchpad is temporary).
- `PY` = `C:/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe` (always with `-I`).
- Stream reports: `docs/results/litert/models.txt` (conversion, parity), `docs/results/litert/runtime.txt`
  (infer.cpp, LiteRT 1.4.0 facts, the OpenCL crash).

## Why LiteRT (the owner's question)

NNAPI is deprecated since Android 15, and ONNX Runtime's NNAPI EP already aborted the app on the emulator
(accel STATUS.md). LiteRT (the renamed TensorFlow Lite) is Google's maintained on-device runtime. Its GPU
delegate reaches the GPU directly through OpenCL (OpenGL ES as its fallback) on Mali, Adreno and Samsung's
Xclipse, with no vendor driver layer between. So the GPU path is now LiteRT. ONNX Runtime 1.22.0 stays for
the CPU path (the bit-identical reference) and the Qualcomm NPU (QNN). NNAPI stays only in the debug batch
mode, as before.

## Merge log (litert/main)

| Commit | What | Branch head |
|---|---|---|
| 32609fa | litert/models: `app/src/main/litert/models/*.tflite` (fp16 weights, fp32 compute and I/O), `tools/litert/` | 35ffded |
| f00f968 | litert/runtime: `Backend::LiteRtGpu` in `core/src/infer.cpp`, vendored LiteRT 1.4.0 headers, `tools/litert_check` | eb15902 |
| 4586aa6 | litert/app: the `litert_gpu` pack, `LiteRtPacks`, `NativeLib.loadLiteRt`, GPU (LiteRT) in Settings and Auto | 89c81aa |
| 46f7161 | **Fix**: manifest declares the vendor OpenCL libraries | |
| (this commit) | README Acceleration / Build, this file | |

- Real merges, in the order models, runtime, app, with no conflicts. Both runtime and app carried the contract
  lines (C1) in `infer.h` and `nets.cpp`. They were identical, so git kept one copy (checked: one `LiteRtGpu`
  enum value, one `litert_dir`, one `backend_name` case, one `"litert"` alias).
- **Integration fix 46f7161** (`AndroidManifest.xml`): three optional `<uses-native-library>` entries,
  `libOpenCL.so`, `libOpenCL-pixel.so` and `libOpenCL-car.so`. The app targets API 35, so without them its linker
  namespace cannot open the vendor's OpenCL library. `infer.cpp`'s OpenCL check (and LiteRT's OpenCL backend)
  would then find nothing on any phone, and the GPU would never run. Both the models and the runtime stream
  reported this, and nobody owned the manifest. `required="false"`: phones without the library install as
  before. Checked in the built APK (`aapt2 dump xmltree`).
- A question from the runtime stream, answered: `uvdoc_static.tflite` (15.8 MB) is the same size as
  `uvdoc.onnx` (15.8 MB) because the ONNX files already store fp16 weights. The `.tflite` keeps them fp16
  (FLOAT16 constants + DEQUANTIZE), as contract C2 says.

## Checks run on litert/main (all passed)

1. **Desktop build** (`<OUT>/check/build_dc.bat`, MSVC 14.50 + Ninja, `tools/desktop_check`): it builds. The
   only warnings were already there (C4068 `#pragma STDC`, C4244 in `<xmemory>`). None is in infer.cpp or
   nets.cpp. On Windows the LiteRT backend compiles out, by design.
2. **CPU path unchanged** (`<OUT>/check/cpu_check.sh`, the accel coordinator's check pointed at this worktree;
   pipeline_cli `--backend cpu`, nets in-process, overlapped and sequential; 786 s):
   - 35 photos vs the Python ORT 1.22.0 reference (`<SCR>/accel_b/s35_ref`): **0 of 35 differ**, overlapped and
     sequential (net outputs, pages, JSON);
   - DocUNet 1_1..10_2 (heavy lock) vs `<SCR>/accel_b/du_ref` and `docunet_bench/work/new_fold2`: **0 of 20 differ**,
     overlapped and sequential. DocUNet was not re-scored: identical outputs keep LD 5.16.
3. **Model parity** (`tools/litert/compare.py`, LiteRT interpreter fp32 on the desktop vs ORT 1.22.0 CPU, 5 photos,
   134 s, `<OUT>/check/compare5.log`). The per-photo numbers are the same as in the models stream's 35-photo
   appendix.

   | Net | p99 | max | tolerance (p99 / max) |
   |---|---|---|---|
   | UVDoc pass 1 | 0.00072 px | 0.00135 px | 0.25 / 1 px |
   | UVDoc rc0 | 0.00061 px | 0.00148 px | 0.25 / 1 px |
   | SD-HR rc2 | 0.00233 px | 0.00340 px | 0.25 / 1 px |
   | ShadeNet | 0.00003 levels | 0.00011 levels | 0.5 / 2 levels |

   Photos: doctr__25, scanner_samples__tax (the worst for SD-HR), scanner_samples__receipt,
   dewarpnet__1_1_copy, scanner_samples__desk.
4. **tools/litert_check with the real models** on the emulator (AVD fs35, Android 15 x86_64,
   `-gpu swiftshader_indirect`, LiteRT 1.4.0 x86_64 libraries; log `<OUT>/check/heavy.log`). The emulator has no
   `libOpenCL*` and none of its GPU modes runs the delegate (runtime.txt), so this checks only that the
   delegate fails cleanly:
   - no OpenCL (as shipped): UVDoc, SD-HR and ShadeNet each log `LiteRT_GPU unavailable: no OpenCL library`.
     The net runs on the CPU, with output identical to ORT's CPU (max 0) and no crash;
   - with the test-only OpenCL stub (no platform): the real GPU delegate takes the **whole graph** of the real
     models, "Replacing 154 out of 154 node(s) ... yielding 1 partitions" (UVDoc) and "89 out of 89" (ShadeNet).
     It then fails in its OpenGL fallback (`clGetPlatformIDs -1001`, `GL_INVALID_ENUM`). infer.cpp retries
     once without the kernel cache, then falls back to the CPU. Output identical to the CPU, no crash.
     So the GPU delegate's own op check accepts every op of UVDoc and ShadeNet. SD-HR was not run with the
     stub.
   - The SD-HR run's logcat has TFLite lines ("XNNPACK ... 301 out of 301 node(s)") from another process. The
     emulator's Play services ML Kit runs TFLite (`tflite_dynamite`, `vision.ocr` in the app logcats). Our
     code creates no interpreter without OpenCL. The pid was not captured.
5. **App on the emulator**, debug batch mode, bundled self-test photo (`--ez shade false --es ext .png`):
   - x86_64 test build (`-Pabis=x86_64 -PlitertAbi=x86_64 -PlitertInBase=true`), `--es accel cpu`: compared
     with `<SCR>/coordF/st_desk` (`cmp_st.py`), logits, p1, rc0 and rc2 are the **same bit for bit**, the page
     is the same pixel for pixel and the JSON is the same;
   - same build, `--es accel gpu`: `litert pack: ready (APK, 3 GPU models)`, `loadLiteRt: ok, loaded 2 of 2`.
     UVDoc and SD-HR log `LiteRT_GPU unavailable: no OpenCL library` and run on the CPU.
     `photo1.json` reports the **real** backend of every net (`cpu`). The outputs are bit-identical to
     st_desk and to the cpu run;
   - the **S24 FE test APK** itself (arm64, under the emulator's arm64 translation), `--es accel gpu`: the pack
     loads from the APK, the nets fall back the same way, and there is no crash. The output is bit-identical
     to litert/app's run of the base commit's arm64 APK (`st_cpu_base`). Against st_desk it differs by the
     known arm64-translation amounts (logits 1.05e-05, page 14 levels on 10 % of pixels), as the base
     commit's arm64 APK does.
   - The batch status line (`done.txt`: "UVDoc gpu, SyntheticDoc high-res gpu ...") shows the requested plan,
     as it already did for QNN. The backend that actually ran is in `photo1.json` > `nets` > `backend`.
6. Copies of the emulator outputs: `<OUT>/check/emu/st_cpu`, `st_gpu`, `st_gpu_arm64`.

## Builds (`<OUT>`, heavy lock; sha256 in `SHA256SUMS.txt`)

| File | Bytes | What |
|---|---|---|
| `flatscan-gpu-test-arm64.apk` | 124,824,283 | **The test APK for the Galaxy S24 FE**: arm64-v8a, `assembleDebug -PlitertInBase=true -PcompressLibs=true`. LiteRT 1.4.0 runtime + GPU delegate + the three `.tflite` in the base (`assets/litert/`). No QNN libraries (Exynos phone). Debug-signed. |
| `flatscan-litert-debug.aab` | 183,305,642 | `bundleDebug`: base + `litert_gpu` + `qnn_htp` + `qnn_v68..v79` packs |
| `flatscan-litert-debug.apks` | 410,699,517 | `bundletool build-apks --local-testing` of the bundle (every split, for `install-apks`) |
| `litert_check_arm64/` | 27 MB | `litert_check` (arm64), `libonnxruntime.so`, the two LiteRT libraries, `phone_check.sh` (optional step 7) |

`bundletool get-size total --dimensions=ABI`, arm64-v8a (download, min..max over screen splits):

| Modules | Bytes |
|---|---|
| base | 81,001,396 .. 81,153,771 |
| base + litert_gpu | 114,024,218 .. 114,176,593 (pack: +33.0 MB, of which 2.8 MB the libraries) |
| base + qnn_htp + qnn_v73 (Snapdragon 8 Gen 2, NPU) | 137,362,781 .. 137,515,156 |
| base + qnn_htp + qnn_v73 + litert_gpu | 170,385,603 .. 170,537,978 |

The base is unchanged by this work (81 MB before). Only phones that choose GPU or Auto download the 33 MB pack.

## Not verified

- **The GPU delegate has never run**: no fp32 GPU parity, no op coverage on a real GPU, no timing, no kernel
  cache reuse. The emulator has no OpenCL, and the stub stops at the OpenGL set-up.
- Whether the S24 FE (Exynos 2400e, Xclipse 940) lets an app open `libOpenCL.so`. It works only if Samsung
  lists the library in the vendor's public libraries. If not, the GPU backend reports "no OpenCL library"
  and the nets stay on the CPU, with no crash. OpenGL-ES-only phones get no GPU backend (LiteRT 1.4.0
  crashes in its OpenCL lookup otherwise; runtime.txt).
- Play Store delivery of `litert_gpu` on a device. Only the sideload APK and bundletool `--local-testing` on
  the emulator (the latter by litert/app) have been checked.
- Auto with a working GPU candidate (on the emulator the gpu candidate cannot run).
- The Settings screen with the new choice was not looked at on screen.
- SD-HR through the real delegate with the OpenCL stub (only UVDoc and ShadeNet were run that way).

## Open issues

- ShadeNet is converted only at 512 x 352, the A4 page's size. Another page shape runs ShadeNet on the CPU
  for the rest of the session (as with QNN).
- LiteRT 1.4.0's OpenCL crash on phones without OpenCL: re-check 1.4.2 / 2.x before allowing OpenGL-only
  phones.
- The nets on the GPU run one at a time on one LiteRT thread (the OpenGL backend's EGL context is tied to
  it).
- Disk: C: had 4.3 GB free at the end (other agents share it). This worktree is 2.5 GB with build outputs;
  `<OUT>` is 0.76 GB.

## HUMAN GATE: Galaxy S24 FE, GPU (LiteRT), step by step

What you need: the phone, a USB cable, a PC with `adb` (Android SDK platform-tools), and
`C:/Users/User/Desktop/flatscan_out/litert/flatscan-gpu-test-arm64.apk`.

1. **Install.** On the phone: Settings > About phone > Software information, tap Build number 7 times, then
   Developer options > USB debugging on. Connect, accept the prompt, then on the PC:
   `adb install -r C:/Users/User/Desktop/flatscan_out/litert/flatscan-gpu-test-arm64.apk`.
   This is a debug build signed with this PC's debug key. If an older FlatScan with another signature is
   installed, uninstall it first (`adb uninstall com.albus.flatscan`).
   Without a PC: copy the APK to the phone and open it (allow "Install unknown apps" for the file manager).
2. **Start a log** (PC; it shows why a net leaves the GPU):
   `adb logcat -c` then `adb logcat -s flatscan:I FlatScan:I tflite:* > s24fe_logcat.txt` (leave it running).
3. **Choose the GPU.** Open FlatScan > Settings > Acceleration > **GPU (LiteRT)**. The text under the choices
   should end with "...on any phone: ready (APK, 3 GPU models)" (it appears after the first page, or after
   step 4).
4. **Benchmark.** In Settings, tap **Run the acceleration benchmark** and wait (a minute or two; the first GPU
   run also compiles the GPU kernels). Then in *Self-test and benchmark* tap **Share report** and send the
   text back.
5. **Process 2-3 photos** (camera or gallery, ideally a curved page, a receipt and a page with a shadow;
   Remove shadows on). On each result: menu (top right) > **Processing details**, and send a screenshot of the
   whole text (scroll). Also say how long each page felt compared with Acceleration = CPU (optional: process the
   same photo once with CPU).
6. Stop the log (Ctrl+C) and send `s24fe_logcat.txt`.
7. Optional, if step 4 shows the GPU did not run (needs only adb and this worktree):
   `bash C:/Users/User/Desktop/flatscan_out/litert/litert_check_arm64/phone_check.sh` and send
   `litert_check_arm64/phone_check_out/phone_check.txt`. It runs the same infer.cpp outside the app: shell
   linker namespace, no manifest, both cache passes.

**Pass:**
- Report: "LiteRT GPU pack: ready (APK, 3 GPU models)"; "candidates" includes gpu; a **gpu:** block where
  `uvdoc_pass1`, `uvdoc`, `synthdoc_hr` and `shadenet` each have a time line ending in **PASS** (grid
  p99 <= 0.25 px and max <= 1 px; shadow p99 <= 0.5 and max <= 2 levels). With fp32 the differences should
  be about 0.01 px or less. The gpu medians, compared with the cpu block, show the speed-up.
- Processing details: under the nets, `backend` is **gpu** for UVDoc (both passes), SyntheticDoc high-res
  and the shadow net, and **cpu** for the page finder; `litert_pack` says ready.
- logcat: per net, "Replacing N out of N node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1
  partitions" and "infer: <net> on LiteRT_GPU (... LiteRT 2.20.0-dev0+selfbuilt) ready in ... ms"; no crash.
- Pages look the same as with CPU.

**Fail, and what it means:**
- "gpu: ... did not run on this backend", or `backend` cpu in Processing details: the GPU fell back. The reason
  is in logcat, `infer: <net>: LiteRT_GPU unavailable: ...`:
  - "no OpenCL library": the S24 FE does not give apps OpenCL (or the manifest entry does not cover the name
    Samsung uses). Send the output of `adb shell ls -l /vendor/lib64/ | grep -i opencl` and
    `adb shell cat /vendor/etc/public.libraries.txt`;
  - "the GPU delegate takes only part of the graph": an op the Xclipse GPU delegate does not take; the line
    names the counts;
  - any other "GPU delegate failed: ..." message: send it.
- A time line ending in **FAIL**: the GPU ran but its output is outside the tolerance (Auto would keep the CPU;
  GPU in Settings still uses it). Send the report.
- The pack status not "ready": the APK's assets did not copy or load. The status text says why.
- The app closes during the benchmark or processing: send the logcat (and `adb bugreport` if possible).
