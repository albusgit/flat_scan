"""ShadeNet for the fine-tune: the shipped architecture (tools/shadow_net/model.py) with each
Conv + BatchNorm pair as one Conv with a bias, which is how the app's ONNX stores it (BN folded at
export). The training checkpoint was deleted; the ONNX holds the only copy of the weights.

  fused()                      ShadeNet with BN removed (convs with bias)
  from_onnx(path)              fused() loaded with the ONNX's weights (fp16-rounded, as shipped)
  python net.py ONNX           loads it and checks it against ONNX Runtime"""
import os, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shadow_net'))
from model import ShadeNet  # noqa: E402


def fused():
    net = ShadeNet()
    for m in net.modules():
        if isinstance(m, nn.Sequential):
            for i in range(len(m) - 1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i + 1], nn.BatchNorm2d):
                    c = m[i]
                    m[i] = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, bias=True)
                    m[i + 1] = nn.Identity()
    return net


def convs(net):
    """the convs in definition order, which is the forward order and the ONNX graph order"""
    return [m for m in net.modules() if isinstance(m, nn.Conv2d)]


def from_onnx(path):
    import onnx
    from onnx import numpy_helper
    g = onnx.load(path).graph
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def arr(name):
        while name not in init:   # through the fp16 -> fp32 Cast
            name = prod[name].input[0]
        return numpy_helper.to_array(init[name]).astype(np.float32)
    nodes = [n for n in g.node if n.op_type == 'Conv']
    net = fused()
    cs = convs(net)
    assert len(cs) == len(nodes), (len(cs), len(nodes))
    with torch.no_grad():
        for c, n in zip(cs, nodes):
            w, b = arr(n.input[1]), arr(n.input[2])
            assert tuple(c.weight.shape) == w.shape, (c, w.shape)
            c.weight.copy_(torch.from_numpy(w)); c.bias.copy_(torch.from_numpy(b))
    return net


if __name__ == '__main__':
    import onnxruntime as ort
    net = from_onnx(sys.argv[1]).eval()
    x = np.random.default_rng(0).random((1, 3, 512, 384), dtype=np.float32)
    ref = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).run(None, {'page': x})[0]
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    d = np.abs(out - ref)
    print(f'max |torch - onnx| {d.max():.2e}, p99 {np.percentile(d, 99):.2e} (x255: {d.max() * 255:.3f} levels)')
