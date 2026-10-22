"""A LiteRT (.tflite) copy of the rule net at a fixed size, with the litert branch's converter
(fs_lrt_main/tools/litert/convert.py: fp16 weights, GPU-delegate op forms, its checks), and parity with ONNX
Runtime on random pages and on real stage-1 pages.
Run with the litert venv (C:/Users/User/Android/litert_venv; see the litert branch's
docs/results/litert/models.txt):
  python to_tflite.py RULENET.onnx OUT_DIR [H W] [PAGE.png ...]     (default 1152 x 800: an A4 stage-1 page at half size)
Writes OUT_DIR/rulenet_static.onnx (the input / output fixed to 1 x 3 x H x W, same graph) and
OUT_DIR/rulenet_HxW.tflite (input page [1,H,W,3] NHWC, output rules [1,H,W,2] NHWC)."""
import os, sys
import numpy as np, onnx

src, out = sys.argv[1], os.path.abspath(sys.argv[2])
H, W = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[3].isdigit() else (1152, 800)
pages = [a for a in sys.argv[3:] if not a.isdigit()]
os.makedirs(out, exist_ok=True)
LRT = os.environ.get('LRT_REPO', r'C:\Users\User\Desktop\fs_lrt_main')
sys.path.insert(0, os.path.join(LRT, 'tools', 'litert'))
import convert   # noqa: E402

m = onnx.load(src)
for v in list(m.graph.input) + list(m.graph.output):
    d = v.type.tensor_type.shape.dim
    d[2].ClearField('dim_param'); d[2].dim_value = H
    d[3].ClearField('dim_param'); d[3].dim_value = W
m = onnx.shape_inference.infer_shapes(m)
static = os.path.join(out, 'rulenet_static.onnx')
onnx.save(m, static)
name = f'rulenet_{H}x{W}.tflite'
convert.SRC = out
convert.NETS = [('rulenet', 'rulenet_static.onnx', name, [1, H, W, 3], [1, H, W, 2])]
sys.argv = [sys.argv[0], out, 'rulenet', '--work', os.path.join(out, 'work')]
convert.main()

import onnxruntime as ort   # noqa: E402
from ai_edge_litert.interpreter import Interpreter   # noqa: E402
import cv2   # noqa: E402
s = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
it = Interpreter(model_path=os.path.join(out, name)); it.allocate_tensors()
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
xs = [np.random.default_rng(0).random((1, 3, H, W), dtype=np.float32)]
for p in pages:
    im = cv2.imread(p)
    sm = cv2.resize(im, ((im.shape[1] + 1) // 2, (im.shape[0] + 1) // 2), interpolation=cv2.INTER_AREA)[:H, :W]
    sm = cv2.copyMakeBorder(sm, 0, H - sm.shape[0], 0, W - sm.shape[1], cv2.BORDER_REPLICATE)
    xs.append((sm.astype(np.float32) / 255).transpose(2, 0, 1)[None])
for k, x in enumerate(xs):
    ref = s.run(None, {'page': x})[0]
    it.set_tensor(inp['index'], x.transpose(0, 2, 3, 1).copy()); it.invoke()
    y = it.get_tensor(outp['index']).transpose(0, 3, 1, 2)
    flip = ((ref > 0.5) != (y > 0.5)).mean()
    print(f'{"random" if k == 0 else os.path.basename(pages[k - 1])}: max |tflite - ort| {np.abs(y - ref).max():.2e}, '
          f'pixels on the other side of 0.5: {flip * 100:.4f} %', flush=True)
