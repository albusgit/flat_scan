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
