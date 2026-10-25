# Speed and size, round 3 (round 2's "Not done")

Round 2's list of what was not done (`round2.md`), worked through in order. Each item is its own commit, with what
was measured. Same check as round 2 for the items that keep the output: the emulator's debug batch of 4 photos
(`--es batch t4 --es accel cpu --ez rules true`, Android 15 x86_64 AVD, 4 cores, every net on LiteRT's CPU)
against round 2's outputs (pages, page-finder logits, net grids: 20 files byte-identical). Nothing here was
measured on a phone.

## 1. XNNPACK weight cache: measured, not kept

LiteRT 1.4.0's `xnnpack_delegate.h` is vendored now (`core/third_party/litert`), and the CPU interpreters'
options are its struct, with static asserts on its LP64 layout. The hand-written mirror called the second field
`flags`; in 1.4.0 that offset is `runtime_flags`, and `flags` is the third. Only `num_threads` was ever written,
so nothing changed (batch: 20 files identical), but a flag set through the old mirror would have gone to the
wrong field.

The weight cache itself was tried: `weight_cache_file_path` = `<cache dir>/xnn_<net>_<model sha256>_<LiteRT
version>.cache`, one file per model, held by one session at a time. Results:

- **UVDoc: the build hangs.** LiteRT logs `Unknown constant buffer passed to BuildPackIdentifier` and the batch
  never finished (10 min, process idle). Turning it on for every fixed-shape net would hang the CPU path.
- **Page finder: no gain.** It built and loaded correctly (12.1 MB file; outputs identical), but its session was
  ready in 280, 341 and 319 ms from the cache against 169 and 296 ms without it. The SHA-256 of the 12.6 MB
  model plus reading the cache cost as much as packing the weights.
- The models are fp32, not fp16, so a cache is about the size of its model (12.1 MB vs 12.6 MB), not twice
  it. Caching every net would put about 70 MB in `no_backup`.
- The sessions are cheap on the emulator anyway: UVDoc 180-414 ms, SyntheticDoc high-res 63-109 ms, both made
  beside the CPU stages.

The gain is well under the 0.3 s gate, so the cache path stays unset.

## 2. fp16 backends as Auto candidates, and a CPU-threads extra (debug)

`gpu16` (`Backend::LiteRtGpu16`: the GPU delegate with precision loss allowed, MIN_LATENCY first, its own
kernel-cache key `|fp16`) and `cpu16` (`Backend::LiteRtCpu16`: XNNPACK with
`TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16`, and no fallback to the built-in kernels, so a refusal moves the net to
the next backend) are backends now. They come in only when asked for:
`adb shell am start -S -n com.albus.flatscan/.MainActivity --es bench gpu,gpu16,cpu16`. Auto's tolerances judge
them per group as they judge `gpu`. A benchmark with an fp16 candidate is reported but not stored as Auto's choice.
The page finder never runs on them. Settings has no fp16 choice, and only groups that pass on a phone should get one.

`--ei threads N` (with any debug extra) gives the CPU nets N threads (`configureNets`' existing argument;
`run.json` records it).

Emulator (no OpenCL, so `gpu` and `gpu16` cannot run there; `bench_report.txt`):

- `cpu16`: XNNPACK refuses the forced fp16 for every net ("failed to create XNNPACK runtime"), and each net ran
  on the CPU instead. The arm64 app runs translated on the x86_64 host, which reports no fp16 arithmetic. The
  plan expected XNNPACK to fall back to fp32 kernels; on this runtime it fails instead, and the fallback above
  covers that.
- The CPU path is unchanged: batch t4, 20 files identical.
- `--ei threads 2`: the sessions report 2 threads. Batch 90 s against 84 s with the default 4.
  **The output depends on the thread count.** The page finder's logits are identical, but the UVDoc and
  SyntheticDoc grids move by 1-2 ulp (max |diff| 2.4e-7), and the pages then differ: max 8-20 levels, 0.5-9.6 %
  of pixels changed by at least 1, under 0.01 % by more than 8. So a phone with a different `min(cores, 4)`
  already gets slightly different pages from the same photo. Tuning the thread count is an fp-level change, not a
  same-output one.

Phone checklist (yours): `--es bench gpu,gpu16,cpu16` -> `files/batch_out/bench_report.txt`; batch s35 with
`--es accel cpu --ei threads 2`, then 4, then 6 -> per-stage ms.

## 4. Cold start

### a. The native library off the main thread; WorkManager on demand (same output)

- The first `configureNets` (in the ViewModel's init) was the app's first native call. It loaded
  `libflatscan.so` on the main thread: about 0.85 s on the emulator (logcat: nativeloader on the main thread's
  tid). It now runs on the native thread (`nativeCtx`), under its own lock. A net asked for before it runs waits
  for the asset manager (`jni_bridge.cpp` `assets_locked`, up to 10 s). It doesn't use `netsLock`, because a
  benchmark holding that lock may be the one waiting. `prewarmFinder` stays on `Dispatchers.IO`: on the
  native thread its 0.2-0.5 s session would hold up the renders queued behind it, and the wait covers the ordering.
- WorkManager: `FlatScanApp : Configuration.Provider`, and the manifest removes `WorkManagerInitializer` from
  androidx.startup. `WorkManager.getInstance` (OCR, sync) initialises it the first time it is called
  (`FlatScanAppTest`).

Emulator, `am start -W` TotalTime, 10 cold starts each (force-stop between):

| | runs (ms) | median |
|---|---|---|
| before | 3539 2913 3262 2899 2755 2758 2736 3174 2916 2607 | 2906 ms |
| after | 2638 2778 2884 2964 2688 2912 2751 2799 2659 2860 | 2789 ms |

That is -117 ms on the median and a tighter spread (max 3539 -> 2964). It's less than the library's 0.85 s,
because the library load now competes with the first frame for the emulator's cores. Batch t4: 20 files
identical. Unit tests: 64 run (the new one included), 10 skipped, 0 failures. Camera after a cold start: the
page finder warmed (ready in 467 ms), no errors.

### b. Baseline and startup profiles

A `:baselineprofile` module (`com.android.test`, `androidx.baselineprofile` 1.5.0: the current stable, works
with AGP 8.11.1; the plan named 1.4.1). It has one journey, `firstScan`: Home -> camera -> photo -> crop ->
flattening -> result (its tabs, the save sheet) -> Home -> Settings. It also has `StartupBenchmark`, a cold start
with no compilation and with the profile. The app gets `profileinstaller` and the generated
`app/src/release/generated/baselineProfiles/{baseline,startup}-prof.txt` (26,006 rules each); the release APK
carries `assets/dexopt/baseline.prof`. Regenerate with
`./gradlew :app:generateBaselineProfile -Pandroid.testInstrumentationRunnerArguments.androidx.benchmark.suppressErrors=EMULATOR`
(on a phone, without the flag).

- Windows: the plugin copies the release source set's directories into `nonMinifiedRelease` and
  `benchmarkRelease`, one of them as a provider whose path comes out as `provider(?)`. KSP 2.1.21-2.0.2 rejects
  that as an illegal path (KSP1 too), so the build fails. `app/build.gradle.kts` drops that entry; it is not a
  directory of this project.
- Writing the journey turned up two things on the emulator. "Flatten page" is disabled while the detection runs,
  so the journey waits for the enabled pill. And **a shutter tap in the first seconds after the camera opens is
  lost**: CameraX is still retrying its camera check (about 6 s here, because there is no front camera), the
  capture never completes, and the camera screen stays open with nothing happening. The journey waits 10 s. On
  a phone, CameraX starts much faster, but the app has no timeout or retry for this. Not changed here.

Macrobenchmark, release build, emulator, 10 cold starts each (`timeToInitialDisplayMs`):

| | runs | median |
|---|---|---|
| no compilation (a new install) | 1291 1172 1090 1151 1126 1076 1616 1240 1083 1048 | 1139 ms |
| baseline profile | 1048 964 1020 1031 1021 1074 1029 1060 1030 1064 | 1031 ms |

That is -108 ms (-9.5 %) on the median and -542 ms on the slowest start. The release build starts in about
1.1 s, against the debug build's 2.8 s (4a).

## 3. B2 int8 nets: no code (as planned)

The plan's checks need a phone: `gpu_check.sh` with the int8 files and the GPU delegate's verdict on the
DEQUANTIZE graph. Nothing was written for it.

## 6. security-crypto / Tink: measured, deferred

Release APK (R8), `apkanalyzer dex packages --defined-only --proguard-mappings`: `com.google.crypto.tink`
(its shaded protobuf included) is 174,422 bytes of dex, 1,246 methods; `androidx.security.crypto` is 4,290 bytes.
That is 4.8 % of the 3.73 MB of dex, and less on the wire after compression. Under the 0.5 MB bar, so it stays.
Dropping it would mean the two-release migration of the sync credentials, for under 0.2 MB.

## 5. PDF import: the first page right away, the pages encoded side by side (same bytes)

`renderPdfPages` still renders the pages one after another (PdfRenderer), but PNG-encodes up to
`min(cores, 3)` at a time beside that (a semaphore keeps at most that many page bitmaps in memory). `importPdf`
starts the import when page 1's file is written. `pending` still holds every page's file, so the batch logic
(`flatImport`, the counts, "Next photo") is unchanged; `loadUri` waits for a page whose file isn't written yet
(`pdfPages`). The PNG stays: it is the page's stored original.

Emulator, an 8-page PDF of the t4 photos, imported through Pages > Add page > PDF pages:

| | page 1 written | page 8 written | page 1's import starts |
|---|---|---|---|
| before | +2.16 s | +11.95 s | +11.97 s (after every page) |
| after | +2.17 s | +13.32 s | by +4.31 s (the screen showed "Photo 1 of 8") |

The 8 PNG files are byte-identical (md5) before and after. Rendering all 8 takes about as long as before
(13.3 s against 12.0 s), because it now shares the 4 cores with page 1's processing; the gain is the wait before
the first page, about 8 s here. The whole import's duration wasn't measured precisely (the emulator stopped
before the second run's page files were read).

## The first page after an install: minutes at "Auto: rulenet on gpu (86%)" (bug report, S24 FE)

Fixed as the report proposed, in its order:

1. **Fast GPU tuning.** LiteRT GPU sessions open with `FAST_SINGLE_ANSWER` instead of `SUSTAINED_SPEED`.
   With `SUSTAINED_SPEED`, LiteRT's OpenCL backend times every work-group size of every kernel (`kExhaustive`),
   which takes minutes for the rule net and SyntheticDoc high-res in fp32. Precision is unchanged: it comes from
   `is_precision_loss_allowed` and the priorities. The kernel cache key gets `|fast` / `|exhaustive`.
   Debug: `--ez gpu_exhaustive true` brings back the exhaustive tuning, to time the trade-off.
2. **Auto's benchmark at launch, in the background** (`setUpGpu`), when Auto has no benchmark for this SoC and app
   version and the phone has a GPU candidate. Processing that comes first waits for it (`netsLock`) and shows its
   stage: "Setting up the GPU for this phone (once per install): rulenet on gpu, 86 %". The processing screen
   hides the page percentage while that text shows, since it isn't the page's progress.
3. **One timed run per net** in the benchmark instead of 3.
4. and 5. **The app's GPU sessions made right after the benchmark** (`NativeLib.prewarmGpu`): one blank run of
   UVDoc, SyntheticDoc high-res, the shadow net (512 x 352, its only GPU shape; others run on the CPU) and the rule
   net portrait and landscape, for each group whose backends include the GPU. The landscape rule net no longer
   compiles mid-page. Settings > Run the benchmark does the same.
6. **The setup ms per net**, logged (`infer: ... on LiteRT_GPU (..., fast tuning, 1x3x1152x800) ready in N ms`)
   and from `--ez gpu_warm true` into `files/batch_out/gpu_warm.txt` (all groups on the GPU, a warm-up, then a
   second warm-up showing the run time alone).

Checked on the emulator (no OpenCL, so the GPU itself can't run there): the CPU path is unchanged (batch t4,
20 files identical); `--es bench gpu` runs with one timed run; `--ez gpu_warm true` runs every warm-up (on the
CPU fallback here: UVDoc 2504 ms, SyntheticDoc high-res 9300 ms, shadow net 221 ms, rule net 2074 / 1375 ms the
first time); unit tests 64 run, 0 failures. **Not measured: anything on the GPU.** On the phone:
`adb shell am start -S -n com.albus.flatscan/.MainActivity --ez gpu_warm true` (fast) and then with
`--ez gpu_exhaustive true --ez gpu_warm true`, and compare the two `gpu_warm.txt` files and the benchmark's
`first` column for `rulenet` and `synthdoc_hr`.
