"""The shadow net on one raw buffer (pipeline_cli FS_SHADE_NET hook).
  python run_onnx.py MODEL.onnx IN.f32 OUT.f32 W H    (planar BGR float32, 3 x H x W, in and out)"""
import sys
import numpy as np, onnxruntime as ort

model, fin, fout, w, h = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
x = np.fromfile(fin, np.float32).reshape(1, 3, h, w)
s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
y = s.run(None, {s.get_inputs()[0].name: x})[0].astype(np.float32)
y.tofile(fout)
