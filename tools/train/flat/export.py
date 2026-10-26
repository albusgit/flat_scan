"""A fine-tuned UVDoc (train.py's best.pt) -> the app's uvdoc.onnx and uvdoc_static.onnx: input 'image'
1 x 3 x 712 x 488 RGB [0, 1], output 'grid' 1 x 2 x 45 x 31 (the 2D head only), BatchNorm folded, fp16
weights with Cast nodes, opset 17; checked against torch.

  python export.py RUN/best.pt OUT_DIR
Then on the Windows machine: tools/litert/convert.py uvdoc (litert_venv) -> app/src/main/litert/models/uvdoc_static.tflite."""
import os, sys

import numpy as np
import onnx
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'shade'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'accel'))
from uvdoc_model import UVDocnet  # noqa: E402
from export import fp16_weights  # noqa: E402  (tools/train/shade/export.py)


class Head2D(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    net = UVDocnet(num_filter=32, kernel_size=5)
    net.load_state_dict(ck.get('state', ck.get('model_state', ck)))
    m = Head2D(net).eval()
    os.makedirs(sys.argv[2], exist_ok=True)
    x = torch.rand(1, 3, 712, 488)
    with torch.no_grad():
        ref = m(x).numpy()
    for name, static in (('uvdoc.onnx', False), ('uvdoc_static.onnx', True)):
        out = os.path.join(sys.argv[2], name)
        tmp = out + '.fp32.onnx'
        torch.onnx.export(m, x, tmp, input_names=['image'], output_names=['grid'], opset_version=17, do_constant_folding=True, dynamo=False)
        g = onnx.load(tmp)
        os.remove(tmp)
        fp16_weights(g)
        if static:
            from prepare_models import make_static
            make_static(g, {'image': [1, 3, 712, 488]}, ['grid'], log=lambda s: None)
        onnx.checker.check_model(g)
        onnx.save(g, out)
        import onnxruntime as ort
        y = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, {'image': x.numpy()})[0]
        print(out, os.path.getsize(out) // 1024, 'KB, max |onnx - torch| (grid, [-1, 1]):', float(np.abs(y - ref).max()))


if __name__ == '__main__':
    main()
