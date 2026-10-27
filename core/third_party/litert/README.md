# LiteRT headers

The TFLite C API, GPU delegate and XNNPACK delegate headers of LiteRT 1.4.0, unchanged, for `core/src/infer.cpp`'s
`LiteRtGpu` and LiteRT CPU backends (its libraries are loaded at run time; nothing here is linked). Licence: Apache-2.0
(`LICENSE`, the Apache Software Foundation's text).

* `include/tflite/**` (but `delegates/gpu`, `delegates/xnnpack`), `include/tensorflow/compiler/mlir/lite/core/c/*.h`: the
  `headers/` of `com.google.ai.edge.litert:litert:1.4.0` (dl.google.com/android/maven2, AAR sha256
  54b567572cb2d1295571c785198f87597809cbf22d6cb8acfeac23869041887d; `external/org_tensorflow/` dropped from
  the path, the NNAPI delegate header left out). The library reports `TfLiteVersion()` 2.20.0-dev0+selfbuilt.
* `include/tflite/delegates/gpu/*.h`: the `headers/` of `com.google.ai.edge.litert:litert-gpu:1.4.0` (AAR
  sha256 48f69a7809d82bdcc4d10d401a786c4124dc7932be6b1255eea806cd25f03274).
* `include/tflite/delegates/xnnpack/xnnpack_delegate.h`: not in the AARs' `headers/`; the file of the `v1.4.0` tag of
  github.com/google-ai-edge/LiteRT (`tflite/delegates/xnnpack/xnnpack_delegate.h`, sha256
  d01b38bc1ae87422c9fe96f5f0daee0ef284272a6020f69aab75a0b1cab07aed), the options struct the shipped
  `libtensorflowlite_jni.so` (litert-1.4.0.aar) fills in `TfLiteXNNPackDelegateOptionsDefault`.
