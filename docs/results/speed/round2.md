# Speed and size, round 2 (the plan after 046d1b3)

What was done from the speed-and-size plan, each a commit with its numbers. Everything on `main` keeps the
output: pages, paper masks, maps and pipeline JSON byte-identical on the desktop harness, and pages, page-finder
logits and net grids byte-identical on the emulator, against the build before these changes (9412f23).

## How it was checked

- Desktop: `pipeline_cli` (MSVC, ONNX Runtime CPU, 2 threads per net) over the 34 real photos of
  `E:/fs_datasets/round4/real_photos`, FS_NETS=1 FS_RULE_NET=1; automatic corners, and the frame as edited corners
  (FS_FRAME_CORNERS=1). Equality on 4 parallel jobs, timings on one job (stage medians below).
- Emulator (Android 15 x86_64, arm64 build, 4 cores, every net on LiteRT's CPU): the debug batch of 4 photos
  against the build before; the self-test (map parity max 0.0950 px before and after: the Android / desktop gap,
  unchanged); the debug open test and the camera flow for the app-side changes.
- Unit tests: 63 run, 10 skipped, 0 failures.
- Not measured: anything on a phone. The phone's nets run on the GPU, so its CPU stages weigh more there than on
  the desktop, where the page mostly waits for the CPU nets.

## Desktop, before -> after (34 photos, one job, medians)

| stage | before | after |
|---|---|---|
| page total | 3624 ms | 3105 ms (sum over the 34: 138.6 -> 110.2 s) |
| warp map | 1389 ms | 991 ms (sum 61.1 -> 38.3 s) |
| solves | 336 ms | 233 ms |
| lines + first traces (25 pages with lines) | 377 ms | 188 ms |
| retries, map overstretched (13 pages) | 1150 ms | 318 ms |
| global stage (25 pages) | 146 ms | 104 ms |
| global remap (5 pages with one) | 92 ms | 45 ms |
| flattening net check | 192 ms | 111 ms |
| flattening nets' wall time | 1843 ms | 1749 ms |
| detect + surface fit in processing | 55 + 191 ms | 0 (taken from the crop screen) |
| edited corners: paper mask + before page | 141 ms | 75 ms |

## The changes (main)

| commit | what |
|---|---|
| decb816 | global remap: its bands on all cores (A1) |
| 19af310 | the local retry resumes from the first attempt's global stage and traces (new: the largest single gain) |
| 8f79a05 | solves: multigrid structure once per grid size, one operator for the global x / y (A2) |
| f7fc971 | line search: colour-edge maps beside get_lines, both directions side by side |
| 449519c | the two candidates' text-row spreads side by side; edited corners' detection up front (A5) |
| 51b3562, 876a697 | the flattening nets start right after the page finder, beside the detection and the crop screen (A6) |
| a366c88 | UVDoc and SyntheticDoc high-res side by side (A4) |
| 81466a9 | LiteRT CPU: ShadeNet and the rule net built once, at their first input's shape (A4) |
| 9ef6ab7 | thumbnails decoded once per file version; Rotate off the main thread (A7) |
| 8a2cc50 | a release without the slim OpenCV fails instead of shipping the AAR (A8) |
| 40cd2f6 | ThinLTO + identical code folding: libflatscan.so -0.29 MB (A8) |
| bb3601f | the LiteRT backends read the models in place from the APK, not copied to the heap (A4) |
| 67ae722 | dewarp_detect's two directions side by side |
| 6a6046f | processing takes the crop screen's detection and surface fit |
| 904dff9 | the page finder's session is made while the camera is open (A4 prewarm) |
| 6135133 | export: pages encoded two at a time; no unused full-size and gray copies (A7) |

bb3601f's commit gives the self-test's peak memory after the nets as 1657 -> 1616 MB; a later run of the same
build gave 1653 MB, so VmHWM shows no measurable change (mapped model pages count in it too). The heap copies are
gone; their size was not measured separately.

## Measured, not on main

- B1, the page finder with fp16 weights (-6.2 MB): branch `speed/pagefinder-fp16`,
  `docs/results/speed/pagefinder_fp16.md`. The net matches round 11's numbers on the phone runtime, but the crop
  screen's corners move on unstable photos and the quality gate changed on 1 of 34.
- The least-squares Jacobian on OpenCV's pool instead of new threads per iteration (A3): no gain on the desktop
  (surface fit 191 -> 200 ms), not kept.

## Not done

- XNNPACK weight cache: the options struct's field offset needs LiteRT's header, which is not on this machine;
  writing at a guessed offset could crash phones.
- B2 (int8 UVDoc / SD-HR), B3 (fp16 compute in Auto), B4 (hierarchy reuse in IRLS, lossy), A3 thread-count tuning:
  they need a phone (GPU checks, timings) or DocUNet runs.
- Cold start (native library off the main thread, WorkManager initializer, baseline profile), PDF import as JPEG
  and export with inSampleSize (both change pixels), security-crypto replacement (migrates the sync credentials),
  font instancing, the smaller benchmark photo.
