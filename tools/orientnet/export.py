"""export.py CKPT.pt OUT.onnx [OUT.tflite]: the orientation net for the app, at a fixed batch of 32 patches (the C++
side pads: core/src/nets.cpp orient_net_fn): ONNX (desktop, ONNX Runtime) and, with OUT.tflite, LiteRT (the app;
run in C:/Users/User/Android/litert_venv: onnx2tf, fp32). For a 1-channel input NCHW [32,1,128,128] and NHWC
[32,128,128,1] are the same bytes, and the output [32,4] needs no transpose."""
import os, sys, tempfile

import numpy as np

B = 32


def to_onnx(ckpt, out):
    import torch
    import orientnet as on
    sd = torch.load(ckpt, map_location='cpu')
    net = on.OrientNet(on.widths_of(sd))
    net.load_state_dict(sd)
    net.eval()
    torch.onnx.export(net, torch.zeros(B, 1, on.PATCH, on.PATCH), out, input_names=['patches'], output_names=['logits'],
                      opset_version=17, dynamo=False)


def to_tflite(onnx_path, out):
    import onnx2tf
    work = tempfile.mkdtemp()
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=work, copy_onnx_input_output_names_to_tflite=True,
                    non_verbose=True)
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    import shutil
    shutil.move(os.path.join(work, name + '_float32.tflite'), out)


if __name__ == '__main__':
    if sys.argv[1].endswith('.pt'):
        to_onnx(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        to_tflite(sys.argv[2], sys.argv[3])
