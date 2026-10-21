"""ShadeNet checkpoint -> ONNX for the app (input 'page' 1 x 3 x H x W BGR [0, 1], output 'shading',
H and W dynamic multiples of 32), fp16 weights with fp32 compute (Cast nodes), like the other nets.
  python export.py CKPT.pt OUT.onnx
Fixed-shape models for the NPU / NNAPI (docs/accel/PLAN.md, Stream A), one per input size, exported
without dynamic_axes and with the shape arithmetic folded (tools/accel/prepare_models.py make_static):
  python export.py --static CKPT.pt OUT_DIR [W ...]    -> OUT_DIR/shadenet_512x<W>.onnx, H = 512
  (default W: 288, 320, ..., 512. The app's page is always A4-shaped, so apply_shade_net in
  core/src/pipeline.cpp gives the net 512 x 352 there; other widths come only from other page shapes.)
  The app ships only shadenet_512x352.onnx (`--static final_11k.pt app/src/main/assets/models 352`); any
  other width, and the CPU, use shadenet.onnx. On the ORT CPU EP the fixed-shape model is the same net but
  not bit-identical at the default optimisation level (|diff| <= 1.8e-7: with constant sizes ORT turns the
  Resizes into NCHWc Upsample; identical at ORT_ENABLE_BASIC), so it is for accelerators only.
The shipped app/src/main/assets/models/shadenet.onnx is `export.py final_11k.pt` (round 6,
C:/Users/User/Desktop/shadow_train/final_11k.pt) with torch 2.11, byte for byte."""
import os, sys
import numpy as np, torch, onnx
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ShadeNet


def fp16_weights(m):
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


def export(net, x, out, dynamic):
    tmp = out + '.fp32.onnx'
    torch.onnx.export(net, x, tmp, input_names=['page'], output_names=['shading'], opset_version=17,
                      dynamic_axes={'page': {2: 'h', 3: 'w'}, 'shading': {2: 'h', 3: 'w'}} if dynamic else None,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(tmp)
    os.remove(tmp)
    fp16_weights(m)
    if not dynamic:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'accel'))
        from prepare_models import make_static
        make_static(m, {'page': list(x.shape)}, ['shading'], log=lambda s: None)
    onnx.checker.check_model(m)
    onnx.save(m, out)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        ref = net(x).numpy()
    y = s.run(None, {'page': x.numpy()})[0]
    print(out, os.path.getsize(out) // 1024, 'KB, max |onnx - torch|', float(np.abs(y - ref).max()))


static = sys.argv[1] == '--static'
args = sys.argv[2:] if static else sys.argv[1:]
ck, out = args[0], args[1]
net = ShadeNet().eval()
net.load_state_dict(torch.load(ck, map_location='cpu'))
if not static:
    export(net, torch.rand(1, 3, 512, 384), out, True)
else:
    os.makedirs(out, exist_ok=True)
    for w in [int(v) for v in args[2:]] or list(range(288, 513, 32)):
        export(net, torch.rand(1, 3, 512, w), os.path.join(out, f'shadenet_512x{w}.onnx'), False)
