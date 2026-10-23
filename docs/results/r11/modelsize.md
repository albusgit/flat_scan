# Round 11, stream MODELSIZE: smaller net files

Branch `r11/modelsize` (from `litert/main` fc0442c). `<OUT>` = `C:/Users/User/Desktop/flatscan_out/r11/modelsize`
(logs, variant models, e2e outputs). `<SCR>` = the shared scratchpad. Machine shared by ~7 agents: every time here
is indicative.

## Summary

- **Int8 weights, fp32 compute** ("weight-only int8"): the four nets shrink from 39.1 MB (fp16 .tflite, page finder
  fp16) to **21.6 MB** (-17.5 MB; deflate-compressed, as in an APK download: 35.1 -> 18.0 MB). Against today's
  ONNX set of the APK (46.5 MB with the fp32 page finder) it is -24.9 MB.
- Accuracy: see the table. Net-level the int8 grids move 1-4 px (p99) on hold-out photos; end to end the 35 photos
  keep their text-row straightness (within noise) and look the same by eye; DocUNet: [filled in below].
- Speed: the int8 weights are unpacked to fp32 once at load (XNNPACK, the builtin kernels and the GPU delegate), so
  the CPU time is unchanged (UVDoc, 1 thread, XNNPACK: 638 ms int8 vs 675 ms fp16, median of 8).
- **Finding for the runtime stream (important):** tensorflow 2.19.1's XNNPACK (the desktop) computes **wrong
  outputs** on the fp16-weight .tflite files of litert/main (UVDoc max |d| 0.99 in grid units, SD-HR and ShadeNet
  NaN; the fp16 page finder fails AllocateTensors). The same files on the builtin kernels match ONNX Runtime to 5e-7,
  and int8 files with **no fp16 constants** match their reference on XNNPACK to 1e-6. Not checked with LiteRT
  1.4.0's XNNPACK (phone / emulator). Note: `<OUT>/XNNPACK_FP16_FINDING.txt`.
- The page finder is not redundant: without it [filled in below].

## Inventory (tools/litert/quantize.py info)

| Net | File today | Bytes | Weights | Stored | Input in the pipeline | Runs on |
|---|---|---|---|---|---|---|
| Page finder (MobileNetV3-L + U-Net decoder, 79 convs, HardSwish/HardSigmoid, SE) | pagefinder.onnx | 12,590,589 | 3.14 M | **fp32** | 1x3x256x256, the whole photo squashed | CPU always |
| UVDoc (45 convs, 5x5) | uvdoc.onnx / uvdoc_static.tflite | 15,824,020 / 15,817,028 | 7.89 M | fp16 | 1x3x712x488: pass 1 on the whole photo, then the re-crop | CPU / GPU |
| SyntheticDoc-HR | syntheticdoc_hr.onnx / _static.tflite | 17,027,370 / 15,857,620 | 8.43 M (+0.53 M of the unused `shading` branch, only in the .onnx; the static/.tflite already drop it) | fp16 | 1x3x1440x1024, the re-crop | CPU / GPU |
| ShadeNet | shadenet.onnx / shadenet_512x352.tflite | 1,053,728 / 1,052,328 | 515 k | fp16 | 1x3x512x352 (A4) | CPU / GPU |
| Rule net (Round 10) | rulenet.onnx | 529,583 | 254 k | fp16 | 1x3xhxw, page at half size | CPU |

Dead parts: SD-HR's `shading` decoder (0.53 M weights, 1.1 MB) is the only one; prepare_models.py / the .tflite
already remove it. The page finder's two outputs (page, gutter) are both read (pagefind.cpp).

## Method

- **Format** (quantize.py `tflite`, form dq): each conv filter an INT8 constant (symmetric, per output channel)
  -> DEQUANTIZE -> CONV_2D. Checked in the TFLite sources (v2.20): XNNPACK unpacks a static int8 DEQUANTIZE to fp32 at
  load (`static_unpack_nodes_`), the builtin DEQUANTIZE runs once for a constant input, the GPU delegate turns a
  constant DEQUANTIZE into a constant (parser version <= 3, so the op is written version 2). Hence the compute is
  fp32 with the dequantized weights, and the outputs are exactly those of an fp32 ONNX holding them
  (`*.sim.onnx`; measured: sim vs .tflite p99 0.0003-0.0006 px). That sim ONNX is how the end-to-end runs use the
  variant (pipeline_cli FS_MODELS, nets in-process).
- Three layout fixes, each found by a failure on tensorflow 2.19.1's XNNPACK: conv biases fp32 (not fp16), the int8
  DEQUANTIZE placed right before its conv, and **no fp16 constant left in the file** (layers kept out of int8 are
  stored fp32). Depthwise filters stay unquantized (XNNPACK "failed to delegate DEPTHWISE_CONV_2D"; small).
- **GPTQ** (gptq.py, Frantar et al. 2022) for UVDoc and SD-HR: int8 values chosen to keep each conv's output, using
  the conv inputs of 18 of the 35 photos (calibration; never DocUNet). Plain rounding (RTN) was not good enough:
  hold-out p99 UVDoc 2.83 px, SD-HR 7.83 px (mixed RTN) vs GPTQ 1.12 / 1.89 px.
- Layers kept out of int8 (sensitivity.py, one layer int8 at a time): UVDoc `out_point_positions2D` (alone 2.85 px)
  and `bridge_concat` (1.37 px); SD-HR `bm_head.0` and `bridge_concat` (the same two layers of the same
  architecture); plus weights < 4096 values. Page finder and ShadeNet: all convs (RTN).
- Hold-out = the other 17 photos (`<OUT>/holdout.txt`); page finder / ShadeNet parity on all 35 (no calibration).

## Results

| Net | Variant | Bytes | Net parity vs fp16/fp32 reference (ONNX Runtime) | DocUNet delta | CPU ms (desktop, XNNPACK) |
|---|---|---|---|---|---|
| Page finder | fp32 .onnx (today) | 12,590,589 | reference | - | - |
| Page finder | fp16 .tflite | 6,365,844 | mask IoU min 0.9996, gutter F1 min 0.990, \|dp\| max 0.014 (35 photos) | not run (expected none) | fails on tf 2.19.1 XNNPACK (fp16 depthwise); builtin 375 |
| Page finder | **int8 .tflite** | **3,797,024** | IoU min 0.9965 (docres__208_in), gutter F1 min 0.889, \|dp\| max 0.082 | in V1/V2 | 141 (4 threads, loaded) |
| UVDoc | fp16 .tflite | 15,817,028 | p99 0.0005 px | - | 675 (1 thread) |
| UVDoc | **int8 GPTQ, 2 layers fp32** | **8,604,736** | re-crop p99 1.12 / max 1.53 px; pass 1 p99 3.82 / max 4.81 px (17 hold-out) | in V1/V2 | 638 (1 thread) |
| SD-HR | fp16 .tflite | 15,856,248 | p99 0.0003 px | - | |
| SD-HR | **int8 GPTQ, 2 layers fp32** | **8,631,960** | p99 1.89 / max 2.83 px (17 hold-out) | V2 only | same order (noisy) |
| ShadeNet | fp16 .tflite | 1,052,180 | 0.0001 levels | - | |
| ShadeNet | int8 RTN | 587,336 | worst photo p99 0.85, max 4.87 levels (docscanner__48_1) | (DocUNet runs without shadows) | same |

End to end, 35 photos (pipeline_cli, nets in-process, ShadeNet on; `<OUT>/e2e/s35_*`; 34 photos ran: the
hand-held receipt's photo path is gone from the scratchpad):
- V2 (all four nets int8): every page changes slightly (geometry shifts of a few px, mean |pixel diff| 0.7-15
  levels); text-row straightness (scratchpad straightness.py, 31 pages with rows): row deviation median
  4.329 -> 4.338 px (delta +0.009, s.e. 0.068), p90 6.93 -> 6.85, angle spread 1.775 -> 1.760; table lines (19
  pages) 2.38 -> 2.54 (+0.16, s.e. 0.11). By eye (`<OUT>/sheet_v2.jpg`, the 4 most changed pages): the same.

DocUNet (130 crops, doc 64 excluded, scorer docunet_bench/docunet_eval.py, only changed pages re-scored):
- Baseline in-process (`du_base`): 0 of 130 pages differ from `work/new_fold2`: LD 5.162 / MS 0.5898 / AD 0.1893.
[V1, V2, no page finder: below]

## Not verified

- The GPU delegate with the int8 files (no GPU here). By the sources it dequantizes constant int8 DEQUANTIZEs at
  build time; the conv may then see its weights as a runtime input (2-input convolution) - slower or refused: phone
  check `tools/litert/gpu_check.sh` with the int8 files (HUMAN GATE).
- LiteRT 1.4.0's XNNPACK (phone): neither the fp16 bug nor the int8 files were run there.
- The page finder's int8 sim (end-to-end runs) quantizes 8 filters (changed by onnx2tf) and the depthwise filters
  that the shipped file keeps unquantized: the e2e runs used a slightly *more* quantized page finder.

## Reproduce / apply to new models

```
PY=C:/Users/User/Android/litert_venv/Scripts/python.exe        # tensorflow 2.19.1, onnx2tf, onnxruntime 1.22.0
PT=C:/Users/User/envs/flatscan_torch/Scripts/python.exe   # torch (gptq.py only)
IN=<SCR>/accel_a/in    # 35-photo net inputs (tools/accel/net_inputs.py) + NAME_pf_in.npz (pagefind_cli input)
# 1. GPTQ values (UVDoc / SD-HR; ~10-20 min each on the CPU), calibration photos = <OUT>/calib.txt
$PT -I tools/litert/gptq.py NEW_uvdoc.onnx OUT/uv_gptq --net uv --inputs $IN --names $(cat <OUT>/calib.txt)
$PT -I tools/litert/gptq.py NEW_sdhr.onnx OUT/sd_gptq --net sdhr_rc2 --inputs $IN --names $(cat <OUT>/calib.txt) --samples 4096
# 2. the shipped .tflite + its report (fp16 file, int8 file, sim ONNX, parity on the hold-out photos)
$PY -I tools/litert/ship.py uvdoc NEW_uvdoc.onnx OUTDIR --qmap OUT/uv_gptq.npz --inputs $IN --names $(cat <OUT>/holdout.txt)
$PY -I tools/litert/ship.py sdhr NEW_sdhr.onnx OUTDIR --qmap OUT/sd_gptq.npz --inputs $IN --names $(cat <OUT>/holdout.txt)
$PY -I tools/litert/ship.py pagefinder pagefinder.onnx OUTDIR --inputs $IN --names ALL35
$PY -I tools/litert/ship.py shade shadenet.onnx OUTDIR --weights fp16      # recommended: ShadeNet stays fp16
# 3. end to end: a models folder with the *.sim.onnx renamed to the CPU names, then
bash tools/litert/du_variant.sh NAME MODELS_DIR <OUT>/e2e/du_base           # DocUNet, under the heavy lock
$PT -I tools/litert/quant_e2e.py run s35 OUT/s35_NAME --models MODELS_DIR --cli <OUT>/bin/pipeline_cli.exe
$PT -I tools/litert/quant_e2e.py cmp s35 <OUT>/e2e/s35_base OUT/s35_NAME
```
Without --qmap ship.py rounds to nearest (fine for the page finder, not for UVDoc / SD-HR). The keep list per net
is ship.py DEFAULT_KEEP (node-name substrings; a retrained net with the same architecture keeps the same names).
