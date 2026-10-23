# Speed round (after r11): processing and reopening a page

Starting point: a page on the owner's Galaxy S24 FE (Exynos 2400e, Xclipse 940; nets on the GPU) took 5.9 s,
63 % of it in the warp map (3.7 s): the rule net (CPU only, 444 ms) and a second trace-and-solve pass that was not
timed on its own (~1.8 s). Reopening a stored page took ~2 s.

## Changes

| what | where |
|---|---|
| The second line pass and the retries are timed on their own ("second line pass (kept / not kept)", "retries (map overstretched)") | `core/src/pipeline.cpp` neural_line_stage |
| No second line pass when the first pass found no lines (it never helped there) | same |
| The rule net on the GPU: fixed-shape copies `rulenet_1152x800.tflite` / `rulenet_800x1152.tflite` (the A4 stage-1 page at half size, portrait / landscape); other shapes run on the CPU for that call | `tools/litert/convert.py rulegpu`, `core/src/infer.cpp` |
| Auto's benchmark checks the rule net too (probability difference p99 <= 0.01, max <= 0.1) and gives it its own group (`rule`) | `jni_bridge.cpp` benchBackends, `Acceleration.kt` |
| Warp-map CG tolerance 1e-6 (was 1e-8) | `core/src/warpmap.cpp` |
| The before page is made while the nets run (kept when the corners and the turn stay) | `core/src/pipeline.cpp` run_photo |
| Page params (GPU) and screen detection (CPU) side by side | `jni_bridge.cpp` process |
| Stored sessions keep the page params and the screen detection (`params.bin`); the restore reads them instead of recomputing; the before page is read on first use | `jni_bridge.cpp` |
| Opening a stored page shows its last rendering (out.jpg) at once while the session is restored | `ScanViewModel.openPage`, `ProcessingScreen` |

## Measured

The converted rule net, LiteRT CPU interpreter vs ONNX Runtime on a real page (1152 x 800 and 800 x 1152): max
difference 3e-6, no pixel on the other side of 0.5. GPU op check (convert.py gpu_ops, DEPTHWISE_CONV_2D added): every
op in a form the GPU delegate takes. Its speed and GPU parity on a phone: not measured yet (Auto's benchmark reports
them).

Desktop, the 34 real photos (`tools/library/run_library.py`, real35, FS_NETS=1 FS_RULE_NET=1, CPU, 3 jobs; runs
sp_base -> sp_new in E:/fs_datasets/runs):

| | before | after |
|---|---|---|
| second line pass run / kept | 32 / 7 | 21 / 7 |
| warp map, sum over 34 photos | 123.8 s | 110.0 s (-11 %) |
| solves, sum | 18.7 s | 14.7 s (-22 %) |
| paper mask + before page, median | 185 ms | 90 ms |
| text-row straightness: row deviation p90 / angle spread | 6.7100 px / 1.7497 deg | 6.7103 px / 1.7502 deg |

Before the change the second pass took 1.9 s median where it ran (0.9 s on the 11 pages without lines). Totals on the
desktop are dominated by the nets on the CPU (median wait for them 1.9 s) and are noisy with 3 jobs.
CG 1e-6 vs 1e-8 on the self-test photo: map max 0.032 px, mean 1.2e-5 px (the self-test limit is 0.1 px); its
reference (`assets/selftest/photo1_reference_warpmap.npz`) is regenerated with `tools/desktop_check/selftest_ref`.
