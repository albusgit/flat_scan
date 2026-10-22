"""SyntheticDoc-HR checkpoint -> ONNX as shipped (assets/models/syntheticdoc_hr.onnx): input 'image' 1x3x1440x1024
RGB in [0, 1]; outputs 'grid' 1x2x121x177 in [-1, 1] and 'shading' 1x3x1440x1024 in [0, 1]; fp16 weights with
fp32 compute (Cast nodes); checks it against torch. This is the Round 5 export (scratchpad sdoc_export_hr.py)
with the checkpoint as an argument.

  python export_onnx.py OUT.onnx [CKPT]      # CKPT: the original highres_121x177.ckpt (default) or a train.py epNN.pt"""
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_model import UVDocNet  # noqa: E402

ORIG = r'C:\Users\User\Desktop\model_zoo\SyntheticDoc\weights\highres_121x177.ckpt'
ck0 = torch.load(ORIG, map_location='cpu', weights_only=True)
dc, mc = ck0['hyper_parameters']['data'], ck0['hyper_parameters'].get('model', {})
ck = torch.load(sys.argv[2], map_location='cpu', weights_only=True) if len(sys.argv) > 2 else ck0
sd = {(k.replace('model.', '', 1) if k.startswith('model.') else k): v for k, v in ck['state_dict'].items()}
net = UVDocNet(num_filter=mc.get('num_filter', 32), kernel_size=mc.get('kernel_size', 5),
               bm_out_size=tuple(dc['bm_out_size']), output_size=tuple(dc['image_size']))
print(net.load_state_dict(sd, strict=True))
net.eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x * 2 - 1)                       # app input: RGB in [0, 1]
        return o['bm_map'], o['shadow_map'] * 0.5 + 0.5


H, W = dc['image_size']
tmp = sys.argv[1] + '.fp32.onnx'
torch.onnx.export(Wrap(net).eval(), torch.zeros(1, 3, H, W), tmp, input_names=['image'], output_names=['grid', 'shading'],
                  opset_version=17, dynamo=False, do_constant_folding=False)
m = onnx.load(tmp)
g = m.graph
inits, casts = [], []
for init in list(g.initializer):
    a = numpy_helper.to_array(init)
    if a.dtype != np.float32 or a.size < 16:
        inits.append(init)
        continue
    hh = numpy_helper.from_array(a.astype(np.float16), init.name + '_f16')
    inits.append(hh)
    casts.append(helper.make_node('Cast', [hh.name], [init.name], to=TensorProto.FLOAT, name=init.name + '_cast'))
del g.initializer[:]
g.initializer.extend(inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])
sf = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
x = np.random.default_rng(0).random((1, 3, H, W), dtype=np.float32)
with torch.no_grad():
    tg, ts = Wrap(net)(torch.from_numpy(x))
s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
og, os_ = s.run(None, {'image': x})
fg, fs = sf.run(None, {'image': x})
print('fp32 onnx vs torch', float(np.abs(fg - tg.numpy()).max()), float(np.abs(fs - ts.numpy()).max()),
      'fp16 vs fp32 onnx', float(np.abs(og - fg).max()), float(np.abs(os_ - fs).max()))
print('grid', og.shape, 'max diff', float(np.abs(og - tg.numpy()).max()), ' shading', os_.shape, 'max diff',
      float(np.abs(os_ - ts.numpy()).max()))
t = time.time()
for _ in range(3):
    s.run(None, {'image': x})
print('ORT CPU %.0f ms' % ((time.time() - t) / 3 * 1000), 'size %.1f MB' % (os.path.getsize(sys.argv[1]) / 1e6))
os.remove(tmp)
