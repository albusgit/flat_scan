"""ShadeNet checkpoint -> ONNX for the app (input 'page' 1 x 3 x H x W BGR [0, 1], output 'shading',
H and W dynamic multiples of 32), fp16 weights with fp32 compute (Cast nodes), like the other nets.
  python export.py CKPT.pt OUT.onnx"""
import os, sys
import numpy as np, torch, onnx
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ShadeNet

ck, out = sys.argv[1], sys.argv[2]
net = ShadeNet().eval()
net.load_state_dict(torch.load(ck, map_location='cpu'))
x = torch.rand(1, 3, 512, 384)
tmp = out + '.fp32.onnx'
torch.onnx.export(net, x, tmp, input_names=['page'], output_names=['shading'], opset_version=17,
                  dynamic_axes={'page': {2: 'h', 3: 'w'}, 'shading': {2: 'h', 3: 'w'}}, do_constant_folding=True, dynamo=False)
m = onnx.load(tmp)
# fp16 weights: every float initializer stored as fp16 and cast back to fp32 at load
g = m.graph
new_inits, casts = [], []
for init in list(g.initializer):
    a = numpy_helper.to_array(init)
    if a.dtype == np.float32 and a.size > 16:
        h = numpy_helper.from_array(a.astype(np.float16), init.name + '_fp16')
        new_inits.append(h)
        casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT))
        g.initializer.remove(init)
g.initializer.extend(new_inits)
for c in reversed(casts):
    g.node.insert(0, c)
onnx.checker.check_model(m)
onnx.save(m, out)
os.remove(tmp)
import onnxruntime as ort
s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
with torch.no_grad():
    ref = net(x).numpy()
y = s.run(None, {'page': x.numpy()})[0]
print(out, os.path.getsize(out) // 1024, 'KB, max |onnx - torch|', float(np.abs(y - ref).max()))
