"""export_onnx.py CKPT OUT.onnx [--check PHOTO ...]: page finder v2 weights (train.py's step*.pt / best.pt) -> the
app's pagefinder.onnx: input `image` 1 x 3 x 384 x 384 (net.to_input), output `logits` 1 x 3 x 384 x 384 (paper,
gutter, paper edge), fp32, opset 17, the same names as the round-4 file (the core tells the two apart by size:
core/src/nets.cpp page_finder_logits). --check: ONNX Runtime vs torch on those photos (max |logit| difference)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import SIZE, SegNet2, to_input  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--check', nargs='*', default=[])
a = ap.parse_args()
net = SegNet2().eval()
net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
tmp = a.out + '.tmp'
torch.onnx.export(net, torch.zeros(1, 3, SIZE, SIZE), tmp, input_names=['image'], output_names=['logits'], opset_version=17,
                  do_constant_folding=True, dynamo=False)
os.replace(tmp, a.out)
print('wrote', a.out, os.path.getsize(a.out), 'bytes')
if a.check:
    import cv2
    import onnxruntime as ort
    s = ort.InferenceSession(a.out, providers=['CPUExecutionProvider'])
    for p in a.check:
        x, _ = to_input(cv2.imread(p))
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        y = s.run(None, {'image': x})[0]
        print(f'{os.path.basename(p)}: max |d logit| {np.abs(y - ref).max():.2e}, paper agree {((y[0, 0] > 0) == (ref[0, 0] > 0)).mean():.5f}')
