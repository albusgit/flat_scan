# MI-GAN inpainting net (content-aware finger fill)

An on-device inpainting net for the optional "content-aware (AI)" fill of a finger removed from a page.
The model is MI-GAN 512 Places2, the bare generator only: the crop, resize and blend of MI-GAN's ONNX pipeline are left out.

## Source and licences

- Code: https://github.com/Picsart-AI-Research/MI-GAN, MIT (`LICENSE`, Copyright (c) 2024 Picsart AI Research (PAIR)).
  `migan_net.py` is a copy of its `lib/model_zoo/migan_inference.py` and keeps that notice.
- Weights: `migan_512_places2.pt`, MIT (`LICENSE-WEIGHTS` in the same repo, same copyright). They come from the Google
  Drive folder that the repo README links:
  https://drive.google.com/drive/folders/1xNtvN2lto0p5yFKOEEg9RioMjGrYM74w, file id `1D_YCuCgo20S2256sqpedsmENNm2WMtVY`
  (29,553,256 bytes, sha256 `1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a`).
  Places2 is the only 512 model they publish; the other two models are 256 px (FFHQ faces and Places2).
- The file loads with `torch.load(weights_only=True)`. It is a plain state dict of 177 tensors and 5,973,366 parameters.

## Chain

```
# 1. PyTorch -> ONNX, plus the reference outputs (C:/Users/User/envs/flatscan_torch)
python -I tools/inpaint/export_onnx.py migan_512_places2.pt OUT/onnx [--orig MIGAN_REPO_CLONE]
# 2. ONNX -> TFLite (C:/Users/User/Android/litert_venv)
python -I tools/inpaint/convert.py OUT/onnx/migan_512.onnx OUT/tflite
# 3. TFLite vs PyTorch, PNGs, CPU time (C:/Users/User/Android/litert_venv)
python -I tools/inpaint/check.py OUT/onnx/cases.npz OUT/tflite OUT/compare
```

Each script's docstring covers what it makes and how.

## The model file: `migan_512.tflite`

- Input `input`: float32 NHWC `[1, 512, 512, 4]`.
- Output `output`: float32 NHWC `[1, 512, 512, 3]`.
- fp32 graph. Weights are stored fp16 with DEQUANTIZE. Conv biases are fp32. A depthwise filter that has an fp32 bias is
  also kept fp32, because XNNPACK will not delegate that pair otherwise.
- Ops: ADD, CONV_2D, DEPTHWISE_CONV_2D, DEQUANTIZE, LEAKY_RELU, MAXIMUM, MINIMUM, MUL, PAD and RESIZE_NEAREST_NEIGHBOR.
  All are LiteRT CPU builtins.
- The fixed noise of the StyleGAN-style synthesis is baked in as constants, so the output is deterministic.
- The file is meant for the CPU. Its upsamplers multiply by a `[1, H, W, 1]` zero-insertion mask and its noise adds use
  `[1, H, W, 1]` constants. The GPU delegate's ADD and MUL do not take these broadcasts.

## Input: pre-processing

Take a 512 x 512 RGB crop around the hole. Resize it bilinear or area if needed, and resize the mask with nearest.

| channel | value |
|---|---|
| 0 | `m - 0.5`: **+0.5 on known pixels, -0.5 in the hole** |
| 1, 2, 3 | `rgb * m`, where `rgb = pixel * 2 / 255 - 1` in [-1, 1], in **R, G, B order** (not BGR). It is 0 in the hole |

`m` is binary: 1 = known, 0 = hole (to fill). MI-GAN's own pipeline uses the same convention, with a uint8 mask where
255 = known and 0 = hole. That is the inverse of a "white = finger" mask, so invert such a mask first.

## Output: post-processing

- The output is RGB, roughly in [-1, 1] and not clamped. Measured ranges were about [-1.5, 2.0].
- Convert with `pixel = clamp((out * 0.5 + 0.5) * 255, 0, 255)`.
- Use the output only inside the hole and keep the original pixels outside, because the net does not reproduce known
  pixels exactly. MI-GAN's pipeline feathers the edge: it dilates the hole by 3 x 3 and blurs it with a 5 x 5 Gaussian
  (sigma 1).
- If the crop was resized, resize the output back to the crop's size before you blend.
