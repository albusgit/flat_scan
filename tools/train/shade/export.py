"""A trained shadow net (train.py's best.pt) -> the app's two ONNX files, as tools/shadow_net/export.py
makes them: shadenet.onnx (dynamic H, W) and shadenet_512x352.onnx (fixed shape, what LiteRT converts).
fp16 weights with Cast nodes, opset 17, checked against torch.

  python export.py RUN/best.pt OUT_DIR
Then on the Windows machine: tools/litert/convert.py shade (litert_venv) -> app/src/main/litert/models/shadenet.tflite,
and the map floor in core/src/pipeline.cpp apply_shade_net set to the --floor the net was trained for."""
import os, sys

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'shadow_net'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'accel'))
from net import fused  # noqa: E402
from model import ShadeNet  # noqa: E402


def fp16_weights(m):
    g = m.graph
    new, casts = [], []
    for init in list(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size > 16:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + '_fp16')
            new.append(h)
            casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(new)
    for c in reversed(casts):
        g.node.insert(0, c)


def export(net, x, out, dynamic):
    tmp = out + '.fp32.onnx'
    torch.onnx.export(net, x, tmp, input_names=['page'], output_names=['shading'], opset_version=17,
                      dynamic_axes={'page': {2: 'h', 3: 'w'}, 'shading': {2: 'h', 3: 'w'}} if dynamic else None,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(tmp)
    os.remove(tmp)
    fp16_weights(m)
    if not dynamic:
        from prepare_models import make_static
        make_static(m, {'page': list(x.shape)}, ['shading'], log=lambda s: None)
    onnx.checker.check_model(m)
    onnx.save(m, out)
    import onnxruntime as ort
    y = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, {'page': x.numpy()})[0]
    with torch.no_grad():
        ref = net(x).numpy()
    print(out, os.path.getsize(out) // 1024, 'KB, max |onnx - torch| x255:', round(float(np.abs(y - ref).max()) * 255, 3))


def main():
    ck = torch.load(sys.argv[1], map_location='cpu')
    net = fused() if ck['arch'] == 'fused' else ShadeNet()
    net.load_state_dict(ck['state'])
    net.eval()
    os.makedirs(sys.argv[2], exist_ok=True)
    print('step', ck['step'], 'real err', ck.get('real_err'))
    export(net, torch.rand(1, 3, 512, 384), os.path.join(sys.argv[2], 'shadenet.onnx'), True)
    export(net, torch.rand(1, 3, 512, 352), os.path.join(sys.argv[2], 'shadenet_512x352.onnx'), False)


if __name__ == '__main__':
    main()
