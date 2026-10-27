"""MI-GAN 512 (Places2) generator -> fixed-shape ONNX, plus the PyTorch reference outputs for tools/inpaint/check.py.

  python -I tools/inpaint/export_onnx.py WEIGHTS OUT_DIR [--orig MIGAN_REPO]
      WEIGHTS: migan_512_places2.pt (state dict, loaded with torch.load(weights_only=True): no pickled objects)
      -> OUT_DIR/migan_512.onnx: the bare generator (tools/inpaint/migan_net.py), opset 17, input "input"
         [1, 4, 512, 512] float32 NCHW, output "output" [1, 3, 512, 512] float32 NCHW, fp32 weights,
         constants folded (the noise inputs noise_const * noise_strength become fixed [1, 1, R, R] constants)
      -> OUT_DIR/cases.npz: the check's test cases (CASES below: page photos of testdata/ cropped to 512 x 512 with a
         finger mask) with the NHWC net input, the PyTorch output and the ONNX Runtime output
      --orig: a clone of the MI-GAN repo; its lib/model_zoo/migan_inference.py Generator is loaded too and must give
         the same output as migan_net.py bit for bit (the copy is faithful)
Run it with C:/Users/User/envs/flatscan_torch (torch 2.11 CPU, onnx, onnxruntime, opencv-python). The legacy
TorchScript exporter (dynamo=False) is used: no onnxscript needed.

Pre / post-processing (MI-GAN's scripts/demo.py; tools/inpaint/README.md):
  rgb in [-1, 1] = pixel * 2 / 255 - 1 (RGB order), m = 1 known / 0 hole,
  input = concat(m - 0.5, rgb * m) along channels; output RGB in ~[-1, 1]: pixel = (out * 0.5 + 0.5) * 255, clamped;
  the result is out in the hole, the original outside (the net does not keep known pixels exactly)."""
import importlib.util
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from migan_net import Generator  # noqa: E402

RES = 512
# (name, testdata photo, mask source): 'blob' = a synthetic finger drawn by finger_blob() over text at the page's
# right edge, in the 512 x 512 crop (x, y, side) of the photo; a path = the photo's ground-truth finger mask (white =
# finger), cropped around its largest finger
CASES = [('receipt_blob', 'testdata/control/control_05.jpg', 'blob', (256, 560, 1024)),
         ('fingers_01_gt', 'testdata/fingers/fingers_01.jpg', 'testdata/fingers/masks/fingers_01.png', None),
         ('fingers_05_gt', 'testdata/fingers/fingers_05.jpg', 'testdata/fingers/masks/fingers_05.png', None)]


def finger_blob(res=RES):
    """A finger entering from the right edge: a round-capped band ~0.16 * res wide, slightly tilted, tip at ~0.45 res
    from the edge, plus a knuckle bulge. uint8 [res, res], 255 = hole."""
    m = np.zeros((res, res), np.uint8)
    s = res / 512
    cv2.line(m, (int(res + 40 * s), int(300 * s)), (int(290 * s), int(250 * s)), 255, int(84 * s), cv2.LINE_AA)
    cv2.ellipse(m, (int(430 * s), int(292 * s)), (int(70 * s), int(52 * s)), -8, 0, 360, 255, -1, cv2.LINE_AA)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def make_case(photo, mask_src, crop):
    """(rgb uint8 [512, 512, 3], hole uint8 [512, 512] 255 = hole)."""
    bgr = cv2.imread(os.path.join(REPO, photo), cv2.IMREAD_COLOR)
    if mask_src == 'blob':
        x, y, side = crop
        rgb = cv2.cvtColor(cv2.resize(bgr[y:y + side, x:x + side], (RES, RES), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        return rgb, finger_blob()
    gt = cv2.imread(os.path.join(REPO, mask_src), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    n, lab, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = st[k, :4]
    side = int(min(max(2.2 * max(bw, bh), RES), min(bgr.shape[:2])))
    cx, cy = bx + bw // 2, by + bh // 2
    x = int(np.clip(cx - side // 2, 0, bgr.shape[1] - side))
    y = int(np.clip(cy - side // 2, 0, bgr.shape[0] - side))
    rgb = cv2.cvtColor(cv2.resize(bgr[y:y + side, x:x + side], (RES, RES), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    hole = cv2.resize(gt[y:y + side, x:x + side].astype(np.uint8) * 255, (RES, RES), interpolation=cv2.INTER_NEAREST)
    # grow the hole a little (a finger's soft edge / shadow), as an app would before inpainting
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return rgb, np.where(hole > 127, 255, 0).astype(np.uint8)


def net_input(rgb, hole):
    """NCHW float32 [1, 4, 512, 512]: (m - 0.5, rgb * m), rgb in [-1, 1], m = 1 known / 0 hole."""
    x = rgb.astype(np.float32) * 2 / 255 - 1
    m = (hole == 0).astype(np.float32)[..., None]
    return np.concatenate([m - 0.5, x * m], axis=2).transpose(2, 0, 1)[None].copy()


def load(weights):
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    g = Generator(resolution=RES)
    g.load_state_dict(sd, strict=True)
    return g.eval(), sd


def main():
    args = sys.argv[1:]
    orig = None
    if '--orig' in args:
        k = args.index('--orig')
        orig = os.path.abspath(args[k + 1])
        del args[k:k + 2]
    weights, out = os.path.abspath(args[0]), os.path.abspath(args[1])
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    g, sd = load(weights)
    n_par = sum(p.numel() for p in g.parameters())
    print(f'torch {torch.__version__}; {weights}: {len(sd)} tensors, {n_par} parameters', flush=True)

    path = os.path.join(out, 'migan_512.onnx')
    torch.onnx.export(g, (torch.zeros(1, 4, RES, RES),), path, input_names=['input'], output_names=['output'],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ops = {}
    for nd in m.graph.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    print(f'{path}: {os.path.getsize(path)} bytes; ops ' + ', '.join(f'{k} {v}' for k, v in sorted(ops.items())))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    ref = None
    if orig:
        spec = importlib.util.spec_from_file_location('migan_inference_orig',
                                                      os.path.join(orig, 'lib', 'model_zoo', 'migan_inference.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ref = mod.Generator(resolution=RES)
        ref.load_state_dict(sd, strict=True)
        ref.eval()

    save = {}
    for name, photo, mask_src, crop in CASES:
        rgb, hole = make_case(photo, mask_src, crop)
        x = net_input(rgb, hole)
        y = g(torch.from_numpy(x)).numpy()
        yo = sess.run(None, {'input': x})[0]
        line = (f'{name}: hole {100 * (hole > 0).mean():.1f}% of the crop; torch out range [{y.min():.3f}, {y.max():.3f}]'
                f'; ONNX Runtime vs torch max {np.abs(yo - y).max():.2e} mean {np.abs(yo - y).mean():.2e}')
        if ref is not None:
            yr = ref(torch.from_numpy(x.copy())).numpy()
            line += f'; original migan_inference.py vs migan_net.py max {np.abs(yr - y).max():.2e}'
            if not np.array_equal(yr, y):
                print(line)
                raise SystemExit('migan_net.py differs from the original generator')
        print(line, flush=True)
        save[name + '/rgb'] = rgb
        save[name + '/hole'] = hole
        save[name + '/x_nhwc'] = x.transpose(0, 2, 3, 1).copy()
        save[name + '/y_torch_nhwc'] = y.transpose(0, 2, 3, 1).copy()
        save[name + '/y_onnx_nhwc'] = yo.transpose(0, 2, 3, 1).copy()
    np.savez_compressed(os.path.join(out, 'cases.npz'), names=np.array([c[0] for c in CASES]), **save)
    print(f'{os.path.join(out, "cases.npz")}: {len(CASES)} cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
