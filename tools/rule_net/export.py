"""RuleNet checkpoint -> ONNX for the app: input 'page' 1 x 3 x H x W BGR [0, 1] (H, W dynamic
multiples of 32), output 'rules' 1 x 2 x H x W probabilities [horizontal, vertical] (the sigmoid is
in the graph), fp16 weights with fp32 compute (Cast nodes), like the other nets. The rule + text-line
net (tools/text_net checkpoints, n_out 4): 1 x 4 x H x W [rule h, rule v, text line h, text line v].
  python export.py CKPT.pt OUT.onnx    (train.py state dicts, train2.py or text_net checkpoints)"""
import os, sys
import numpy as np, torch, onnx
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RuleNet


class Probs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


ck, out = sys.argv[1], sys.argv[2]
sd = torch.load(ck, map_location='cpu')
if 'state' in sd:   # round 11 checkpoints (train2.py): {'ch': widths, 'state': state_dict}
    net = RuleNet(tuple(sd['ch']), sd.get('n_out', 2)).eval(); net.load_state_dict(sd['state'])
else:
    net = RuleNet().eval(); net.load_state_dict(sd)
m_ = Probs(net).eval()
x = torch.rand(1, 3, 576, 416)
tmp = out + '.fp32.onnx'
torch.onnx.export(m_, x, tmp, input_names=['page'], output_names=['rules'], opset_version=17,
                  dynamic_axes={'page': {2: 'h', 3: 'w'}, 'rules': {2: 'h', 3: 'w'}}, do_constant_folding=True, dynamo=False)
m = onnx.load(tmp)
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
for shape in ((1, 3, 576, 416), (1, 3, 1152, 800)):
    xx = torch.rand(*shape)
    with torch.no_grad():
        ref = m_(xx).numpy()
    y = s.run(None, {'page': xx.numpy()})[0]
    print(out, os.path.getsize(out) // 1024, 'KB', shape, 'max |onnx - torch|', float(np.abs(y - ref).max()))
