# Round 11, stream MODELSIZE: smaller net files (WIP notes)

Branch `r11/modelsize` (from `litert/main` fc0442c). Output folder `<OUT>` = `C:/Users/User/Desktop/flatscan_out/r11/modelsize`.

## Inventory (measured: tools/litert/quantize.py info, ONNX files of litert/main + rulenet.onnx of Round 10)

| Net | File (today) | Bytes | Weights | Stored | Input (pipeline) | Where it runs |
|---|---|---|---|---|---|---|
| Page finder (MobileNetV3-L + U-Net decoder) | pagefinder.onnx | 12,590,589 | 3,136,360 | **fp32** | 1x3x256x256 (whole photo squashed) | CPU always |
| UVDoc | uvdoc.onnx / uvdoc_static.tflite | 15,824,020 / 15,817,028 | 7.89 M | fp16 | 1x3x712x488 (pass 1 on the whole photo, then the re-crop) | CPU / LiteRT GPU |
| SyntheticDoc-HR | syntheticdoc_hr.onnx / _static.tflite | 17,027,370 / 15,857,620 | 8.43 M (+0.53 M dead `shading` branch in the .onnx) | fp16 | 1x3x1440x1024 (re-crop) | CPU / LiteRT GPU |
| ShadeNet | shadenet.onnx / shadenet_512x352.tflite | 1,053,728 / 1,052,328 | 515 k | fp16 | 1x3x512xW (512x352 for A4) | CPU / LiteRT GPU |
| Rule net (Round 10) | rulenet.onnx | 529,583 | 254 k | fp16 | 1x3xhxw (page at half size, dynamic) | CPU |

(notes continue below as results come in)
