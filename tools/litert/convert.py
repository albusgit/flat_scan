"""LiteRT (TFLite) copies of the nets: the GPU delegate's fixed-shape models (contract C2) and the LiteRT CPU path's.

  python tools/litert/convert.py [OUT_DIR] [--work DIR] [NAME ...]
      OUT_DIR default app/src/main/litert/models; --work default a temp folder (deleted afterwards)
      NAME: uvdoc, sdhr, shade, finder, rule (default all; rule only when its source exists)
      --rulenet PATH: the rule net's ONNX file (default app/src/main/assets/models/rulenet.onnx)
      -> uvdoc_static.tflite, syntheticdoc_hr_static.tflite (GPU and CPU: fixed input), shadenet.tflite (shade:
         fixed at 512 x 352), pagefinder.tflite (finder: CPU only, fp32 weights, keep_fp32()), rulenet.tflite (rule:
         rulenet.onnx made static at 1152 x 800, the A4 stage-1 page at half size (tools/accel/prepare_models.py
         make_static), converted as ShadeNet, with the GPU op check), and NAME.analyzer.txt
         (tf.lite.experimental.Analyzer, gpu_compatibility=True) in the work folder
      One file per net: the app's loader rewrites shadenet.tflite and rulenet.tflite in memory for each input size
      (core/src/tflite_shape.cpp: the input and the resizes' sizes), on the CPU and on the GPU. For the check, the
      work folder also gets each one's any-size form (any_hw(): shadenet_any.tflite, rulenet_any.tflite), run at the
      pipeline's sizes; tools/litert/check_resize.py compares the app's rewrite with them bit for bit.
      The files are byte-identical when re-made by the full run (no NAME): onnx2tf's tensor names carry a counter
      of the models made in the process, so a run of fewer nets gives the same graphs and numbers under other names.
Run it with C:/Users/User/Android/litert_venv (tensorflow 2.19.1, onnx2tf 1.28.8: docs/results/litert/models.txt).
Sources: the static ONNX files of app/src/main/assets/models (tools/accel/prepare_models.py), and pagefinder.onnx.
The app's loader (core/src/infer.cpp) runs shadenet.tflite rewritten for the page's [1, 512, W, 3], rulenet.tflite
for [1, H, W, 3] (the half-size page padded to multiples of 32: core/src/warpmap.cpp rule_probs); every other file at
its fixed input. Swapping a model (e.g. a quantised one): a file of the same name, float32 NHWC input and output of
the same shapes.

Steps per net:
  1. fp32 copy of the ONNX graph: each fp16 weight's Cast -> fp32 node becomes an fp32 initializer. Exact (fp16
     -> fp32 is), and the graph ONNX Runtime runs after its own constant folding.
  2. onnx2tf: NCHW -> NHWC (input [1,H,W,3], output [1,gh,gw,2] / [1,512,352,3]), fp32 .tflite. Its onnxsim pass
     folds BatchNormalization into the Convs in fp32, as ONNX Runtime does at load time.
  3. rewrite(): the GPU form and fp16 weights, done here instead of the converter's float16 mode, which rounds
     every fp32 constant (the fp32 biases of the nets' last convs, BatchNorm-folded weights: ~1e-4 off the ONNX
     outputs) and marks the file `reduced_precision_support: fp16accfp32` (a hint that fp16 compute is fine):
       * a PRelu with one slope for all channels gets it per channel ([1,1,C], the same values), and ShadeNet's
         x [1,1,1,C] * ones [1,H,W,1] broadcast becomes a nearest resize of x to H x W (the same copies): the
         forms the GPU delegate's PRelu and two-input ops take;
       * a constant whose fp32 values are all fp16 numbers (the ONNX files' fp16 weights) is stored fp16 with
         a DEQUANTIZE (fp16 -> fp32) in front, as the converter's float16 mode stores it: exact, and the form
         the GPU delegate loads as fp32 weights;
       * a conv filter that is an ONNX fp16 weight times its BatchNorm scale s (per output channel, folded in
         fp32 by step 2) is unfolded again: the fp16 weight, bias / s, then a MUL by s that carries the conv's
         fused activation (a per-channel MUL after a conv, which the GPU delegate can merge into the conv
         weights in fp32 when it builds its graph);
       * every other constant (biases, BatchNorm scales, PRelu slopes, pads) stays fp32.
  4. checks: I/O shapes and types of contract C2, gpu_ops() (every op in a form the GPU delegate's op parsers
     take; the Analyzer's GPU check flags nothing in these builds, see gpu_ops), no reduced_precision_support
     metadata. tools/litert/compare.py checks the outputs against ONNX Runtime; tools/litert/gpu_check.sh runs
     the files through the GPU delegate itself on a device."""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models')
OUT = os.path.join(REPO, 'app', 'src', 'main', 'litert', 'models')
# (name, ONNX source, .tflite, NHWC input shape, NHWC output shape)
NETS = [('uvdoc', 'uvdoc_static.onnx', 'uvdoc_static.tflite', [1, 712, 488, 3], [1, 45, 31, 2]),
        ('sdhr', 'syntheticdoc_hr_static.onnx', 'syntheticdoc_hr_static.tflite', [1, 1440, 1024, 3], [1, 121, 177, 2]),
        ('shade', 'shadenet_512x352.onnx', 'shadenet.tflite', [1, 512, 352, 3], [1, 512, 352, 3]),
        # the page finder: LiteRT CPU only (it never runs on the GPU), fp32 weights as in the ONNX file
        ('finder', 'pagefinder.onnx', 'pagefinder.tflite', [1, 256, 256, 3], [1, 256, 256, 2])]
# ShadeNet at any input size, the check's reference (work folder): made from shadenet.tflite by any_hw() after 'shade'
SHADE_ANY = 'shadenet_any.tflite'
# the sizes the pipeline feeds it (core/src/pipeline.cpp apply_shade_net: portrait, long side 512, W a multiple of 32)
SHADE_WIDTHS = list(range(32, 513, 32))
# the rule net (Round 10, tools/rule_net): dynamic h, w in its ONNX file; converted at RULE_HW, the A4 stage-1 page
# (1600 x 2263) at half size padded to multiples of 32 (core/src/warpmap.cpp rule_probs; the GPU's portrait shape)
RULE_SRC = os.path.join(SRC, 'rulenet.onnx')
RULE_HW = [1152, 800]
RULE_OUT = 'rulenet.tflite'
# sizes check_any runs it at (H, W): the half-size pages of rule_probs are multiples of 32 in both
RULE_SIZES = [(32, 32), (64, 96), (512, 352), (352, 512), (1152, 800), (800, 1152), (1024, 1024), (1440, 1024), (96, 1440)]
# onnx2tf runs a sample inference on this file when it is in the working folder (else it downloads one)
SAMPLE = 'calibration_image_sample_data_20x128x128x3_float32.npy'


def fp32_onnx(src):
    """The ONNX graph with its fp16 weights as fp32 initializers, and {Conv weight name: fp16 array}."""
    m = onnx.load(src)
    inits = {i.name: i for i in m.graph.initializer}
    w16, keep = {}, []
    for n in m.graph.node:
        if n.op_type == 'Cast' and n.input[0] in inits and inits[n.input[0]].data_type == TensorProto.FLOAT16:
            a = numpy_helper.to_array(inits[n.input[0]])
            w16[n.output[0]] = a
            m.graph.initializer.append(numpy_helper.from_array(a.astype(np.float32), n.output[0]))
        else:
            keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    used = {i for n in m.graph.node for i in n.input}
    live = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(live)
    onnx.checker.check_model(m)
    return m, w16


def bn_convs(m, w16):
    """[(fp16 weight OHWI, BatchNorm scale s)] of every Conv followed by a BatchNormalization, s in fp32
    (gamma / sqrt(var + eps), the factor ONNX Runtime and onnxsim fold into the weights)."""
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    out = []
    for n in m.graph.node:
        if n.op_type != 'BatchNormalization':
            continue
        c = prod.get(n.input[0])
        if c is None or c.op_type != 'Conv' or c.input[1] not in w16:
            continue
        eps = next((a.f for a in n.attribute if a.name == 'epsilon'), 1e-5)
        g, var = init[n.input[1]].astype(np.float32), init[n.input[4]].astype(np.float32)
        s = (g / np.sqrt(var + np.float32(eps))).astype(np.float32)
        out.append((w16[c.input[1]].transpose(0, 2, 3, 1), s))
    return out


def rewrite(path, dst, bn, out_name, fp32_filters='unfold'):
    """fp32_filters: what to do with a conv filter that is not fp16 numbers: 'unfold' (a BatchNorm-folded ONNX fp16
    weight, the GPU nets) or 'round' (an fp32 source, the page finder: rounded to fp16, a change measured by
    tools/litert/ship.py's parity)."""
    """The fp32 .tflite at path in its GPU form with fp16 weights (module docstring, step 3); its output gets the
    ONNX output's name (onnx2tf names it Identity). Returns a summary line."""
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.read_model(path)
    g = m.subgraphs[0]
    B = s.BuiltinOperator

    def opcode(op, version):
        for i, c in enumerate(m.operatorCodes):
            if c.builtinCode == op:
                return i
        c = s.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = op
        c.version = version
        m.operatorCodes.append(c)
        return len(m.operatorCodes) - 1

    def const(t):
        d = m.buffers[g.tensors[t].buffer].data
        return None if d is None or len(d) == 0 else np.frombuffer(bytes(d), np.float32)

    def add_tensor(name, shape, ttype, data=None):
        b = s.BufferT()
        if data is not None:
            b.data = np.frombuffer(data.tobytes(), np.uint8)
        m.buffers.append(b)
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name, np.array(shape, np.int32), ttype, len(m.buffers) - 1
        g.tensors.append(t)
        return len(g.tensors) - 1

    def set_data(t, a):
        m.buffers[g.tensors[t].buffer].data = np.frombuffer(np.ascontiguousarray(a, np.float32).tobytes(), np.uint8)

    def shape(t):
        return [int(v) for v in g.tensors[t].shape]

    dq = opcode(B.DEQUANTIZE, 3)   # version 3: fp16 input, as the converter writes it
    mul = opcode(B.MUL, 1)
    resize = opcode(B.RESIZE_NEAREST_NEIGHBOR, 1)
    code = [c.builtinCode for c in m.operatorCodes]
    ops, unfolded, used, n_prelu, n_bcast, n_round = [], 0, set(), 0, 0, 0
    for o in g.operators:
        ops.append(o)
        op = code[o.opcodeIndex]
        if op == B.PRELU and const(o.inputs[1]) is not None and const(o.inputs[1]).size == 1:
            # one slope for all channels -> the slope per channel [1, 1, C] (the same values): the GPU delegate
            # takes a per-channel or an H x W x C slope, not a broadcast one
            c = shape(o.inputs[0])[-1]
            o.inputs[1] = add_tensor(g.tensors[o.inputs[1]].name + b'_per_channel', [1, 1, c], s.TensorType.FLOAT32,
                                     np.full(c, const(o.inputs[1])[0], np.float32))
            n_prelu += 1
        elif op == B.MUL and const(o.inputs[1]) is not None and np.all(const(o.inputs[1]) == 1) \
                and shape(o.inputs[0])[1:3] == [1, 1] and shape(o.outputs[0])[1:3] != [1, 1]:
            # x [1,1,1,C] * ones [1,H,W,1] (prepare_models.py's exact form of ShadeNet's Expand) -> nearest resize
            # of x to H x W: the same copies, and an op the GPU delegate has (it broadcasts only the second input)
            hw = shape(o.outputs[0])[1:3]
            o.opcodeIndex = resize
            o.inputs = [o.inputs[0], add_tensor(g.tensors[o.outputs[0]].name + b'_size', [2], s.TensorType.INT32,
                                                np.array(hw, np.int32))]
            o.builtinOptionsType = s.BuiltinOptions.ResizeNearestNeighborOptions
            o.builtinOptions = s.ResizeNearestNeighborOptionsT()
            n_bcast += 1
        if op != B.CONV_2D:
            continue
        # unfold BatchNorm-scaled filters (step 3, second case)
        f = const(o.inputs[1])
        if np.array_equal(f.astype(np.float16).astype(np.float32), f):
            continue
        if fp32_filters == 'round':
            set_data(o.inputs[1], f.astype(np.float16).astype(np.float32))
            n_round += 1
            continue
        if len(o.inputs) < 3 or o.inputs[2] < 0:
            raise SystemExit(f'conv filter {g.tensors[o.inputs[1]].name}: BatchNorm-folded but no bias')
        fs = shape(o.inputs[1])
        f = f.reshape(fs)
        hit = [k for k, (w, sc) in enumerate(bn) if k not in used and list(w.shape) == fs and
               np.abs(w.astype(np.float32) * sc[:, None, None, None] - f).max() <= 1e-6 * np.abs(f).max()]
        if len(hit) != 1:
            raise SystemExit(f'conv filter {g.tensors[o.inputs[1]].name} {fs}: {len(hit)} BatchNorm matches')
        used.add(hit[0])
        w, sc = bn[hit[0]]
        if not np.all(sc != 0):
            raise SystemExit('BatchNorm scale 0')
        set_data(o.inputs[1], w.astype(np.float32))
        set_data(o.inputs[2], const(o.inputs[2]) / sc)
        y = o.outputs[0]
        t = g.tensors[y]
        mid = add_tensor(t.name + b'_unscaled', t.shape, s.TensorType.FLOAT32)
        o.outputs = [mid]
        act = o.builtinOptions.fusedActivationFunction
        o.builtinOptions.fusedActivationFunction = s.ActivationFunctionType.NONE
        k = add_tensor(t.name + b'_bn_scale', [len(sc)], s.TensorType.FLOAT32, sc)
        p = s.OperatorT()
        p.opcodeIndex, p.inputs, p.outputs = mul, [mid, k], [y]
        p.builtinOptionsType = s.BuiltinOptions.MulOptions
        p.builtinOptions = s.MulOptionsT()
        p.builtinOptions.fusedActivationFunction = act
        ops.append(p)
        unfolded += 1
    if fp32_filters == 'unfold' and unfolded != len(bn):
        raise SystemExit(f'{unfolded} filters unfolded, {len(bn)} Conv + BatchNorm pairs in the ONNX graph')
    # fp16 storage of the constants that are fp16 numbers (step 3, first case)
    deq, done, n16, v16, n32, v32 = [], set(), 0, 0, 0, 0
    for o in ops:
        for ti in o.inputs:
            if ti < 0 or ti in done or g.tensors[ti].type != s.TensorType.FLOAT32:
                continue
            a = const(ti)
            if a is None:
                continue
            done.add(ti)
            h = a.astype(np.float16)
            if not np.array_equal(h.astype(np.float32), a):
                n32, v32 = n32 + 1, v32 + a.size
                continue
            n16, v16 = n16 + 1, v16 + a.size
            t = g.tensors[ti]
            t16 = add_tensor(t.name + b'_fp16', t.shape, s.TensorType.FLOAT16, h)
            m.buffers[t.buffer].data = None
            p = s.OperatorT()
            p.opcodeIndex, p.inputs, p.outputs = dq, [t16], [ti]
            p.builtinOptionsType = s.BuiltinOptions.DequantizeOptions
            p.builtinOptions = s.DequantizeOptionsT()
            deq.append(p)
    g.operators = deq + ops
    # constants no op reads any more (the replaced PRelu slope, the ones of the broadcast Mul): emptied
    read = {i for o in g.operators for i in o.inputs}
    for i, t in enumerate(g.tensors):
        if i not in read and i not in g.outputs and m.buffers[t.buffer].data is not None:
            m.buffers[t.buffer].data = None
    g.tensors[g.outputs[0]].name = out_name.encode()
    for md in m.metadata or []:
        if md.name == b'min_runtime_version':   # what the converter's float16 mode writes (fp16 DEQUANTIZE)
            m.buffers[md.buffer].data = np.frombuffer(b'1.15.0'.ljust(16, b'\0'), np.uint8)
    fu.write_model(m, dst)
    return (f'fp16 constants {n16} ({v16} values), fp32 constants {n32} ({v32} values), '
            f'{unfolded} BatchNorm-folded filters unfolded, {n_prelu} PRelu slope(s) per channel, '
            f'{n_bcast} broadcast Mul(s) -> nearest resize' + (f', {n_round} fp32 filters rounded to fp16' if n_round else ''))


def gpu_ops(path):
    """Problems the LiteRT GPU delegate's op parsers (tflite/delegates/gpu/common/model_builder.cc) would have
    with the graph, as a list of strings (empty: every op is one the delegate takes, in a form it takes).
    tf.lite.experimental.Analyzer(gpu_compatibility=True) is no check here: its builds for this venv flag nothing
    (it passes WHERE, TOPK_V2, int32 MUL and a Flex op; docs/results/litert/models.txt). The delegate's own
    verdict is its partition log on a device: tools/litert/gpu_check.sh."""
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.read_model(path)
    g = m.subgraphs[0]
    B, T = s.BuiltinOperator, s.TensorType
    name = {v: k for k, v in vars(B).items() if not k.startswith('_')}
    code = [c.builtinCode for c in m.operatorCodes]
    made = {o.outputs[0]: code[o.opcodeIndex] for o in g.operators}

    def is_const(t):
        d = m.buffers[g.tensors[t].buffer].data
        return (d is not None and len(d) > 0) or made.get(t) == B.DEQUANTIZE

    def data(t, dt):
        return np.frombuffer(bytes(m.buffers[g.tensors[t].buffer].data), dt)

    def shape(t):
        return [int(v) for v in g.tensors[t].shape]

    bad = []
    for k, o in enumerate(g.operators):
        op, ins, out = code[o.opcodeIndex], [i for i in o.inputs if i >= 0], o.outputs[0]
        what = f'op {k} {name.get(op, op)}'
        x = ins[0]
        if op == B.DEQUANTIZE:
            if g.tensors[x].type != T.FLOAT16 or not is_const(x):
                bad.append(what + ': not an fp16 constant')
            continue
        for t in (x, out):
            if g.tensors[t].type != T.FLOAT32 or len(shape(t)) != 4 or shape(t)[0] != 1:
                bad.append(f'{what}: tensor {g.tensors[t].name} {shape(t)} not float32 [1,H,W,C]')
        c = shape(x)[-1]
        if op == B.CONV_2D:
            if not all(is_const(i) for i in ins[1:]):
                bad.append(what + ': weights not constant')
        elif op == B.DEPTHWISE_CONV_2D:
            # the rule net's 1 x 15 / 15 x 1 line blocks: constant weights, one filter per channel
            if not all(is_const(i) for i in ins[1:]) or o.builtinOptions.depthMultiplier not in (0, 1):
                bad.append(what + ': weights not constant or depth multiplier > 1')
        elif op in (B.PAD, B.MIRROR_PAD):
            p = data(ins[1], np.int32).reshape(4, 2) if is_const(ins[1]) else None
            if p is None or p[0].any() or p[3].any():
                bad.append(what + ': paddings not constant H / W only')
            if op == B.MIRROR_PAD and o.builtinOptions.mode != s.MirrorPadMode.REFLECT:
                bad.append(what + ': not REFLECT')
        elif op in (B.ADD, B.SUB, B.MUL):
            y = ins[1]
            n = int(np.prod(shape(y)))
            ok = (not is_const(y) and shape(y) == shape(x)) or (is_const(y) and (n == 1 or (shape(y)[-1] == c and n == c)))
            if is_const(x) or not ok or shape(out) != shape(x):
                bad.append(f'{what}: operands {shape(x)} {shape(y)}{" const" if is_const(y) else ""} -> {shape(out)}')
        elif op == B.CONCATENATION:
            if o.builtinOptions.axis not in (3, -1) or any(is_const(i) for i in ins):
                bad.append(what + ': not along channels of runtime tensors')
        elif op == B.PRELU:
            a = ins[1]
            if not is_const(a) or shape(a)[-1] != c or int(np.prod(shape(a))) != c:
                bad.append(f'{what}: slope {shape(a)} for {c} channels')
        elif op in (B.RESIZE_BILINEAR, B.RESIZE_NEAREST_NEIGHBOR):
            if not is_const(ins[1]) or o.builtinOptions.alignCorners:
                bad.append(what + ': size not constant or align_corners')
        elif op == B.MEAN:
            if not is_const(ins[1]) or sorted(data(ins[1], np.int32).tolist()) != [1, 2]:
                bad.append(what + ': not over H, W')
        elif op not in (B.TANH, B.LOGISTIC):
            bad.append(what + ': op not taken by the GPU delegate (or not checked here)')
    return bad


def keep_fp32(path, dst, m):
    """The page finder: onnx2tf's fp32 .tflite as it is (its weights are fp32 in the ONNX file: no fp16 storage,
    nothing to rewrite for the GPU, which never runs it), the output named as in the ONNX graph. Checks that every
    CONV_2D / DEPTHWISE_CONV_2D filter is bit for bit an ONNX Conv weight (transposed), or for a Conv in front of a
    HardSigmoid the fp32 product weight x alpha (TF folds the gate's scale into the filter; ~1 ulp, the order of
    ONNX Runtime's own fusions). Returns a summary line."""
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    tm = fu.read_model(path)
    g = tm.subgraphs[0]
    code = [c.builtinCode for c in tm.operatorCodes]
    B = s.BuiltinOperator
    w, scaled = {}, {}
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    users = {}
    for nd in m.graph.node:
        for i in nd.input:
            users.setdefault(i, []).append(nd)
    for nd in m.graph.node:
        if nd.op_type != 'Conv':
            continue
        a = init[nd.input[1]]
        w.setdefault(a.shape, []).append(a)
        # a Conv feeding a HardSigmoid (the squeeze-excite gates): the converter folds alpha into the filter
        hs = [u for u in users.get(nd.output[0], []) if u.op_type == 'HardSigmoid']
        if len(hs) == 1:
            alpha = next((x.f for x in hs[0].attribute if x.name == 'alpha'), 0.2)
            scaled.setdefault(a.shape, []).append((a * np.float32(alpha)).astype(np.float32))
    n = n_scaled = 0
    for o in g.operators:
        if code[o.opcodeIndex] not in (B.CONV_2D, B.DEPTHWISE_CONV_2D):
            continue
        t = g.tensors[o.inputs[1]]
        f = np.frombuffer(bytes(tm.buffers[t.buffer].data), np.float32).reshape([int(v) for v in t.shape])
        # CONV_2D OHWI <- ONNX OIHW; DEPTHWISE_CONV_2D 1HWC <- ONNX C1HW
        cand = f.transpose(0, 3, 1, 2) if code[o.opcodeIndex] == B.CONV_2D else f.transpose(3, 0, 1, 2)
        if any(np.array_equal(cand, a) for a in w.get(cand.shape, [])):
            n += 1
        elif any(np.array_equal(cand, a) for a in scaled.get(cand.shape, [])):
            n_scaled += 1
        else:
            raise SystemExit(f'{t.name}: filter not bit for bit an ONNX weight')
    g.tensors[g.outputs[0]].name = m.graph.output[0].name.encode()
    fu.write_model(tm, dst)
    return (f'fp32 as converted: {n} conv filters bit for bit the ONNX weights, {n_scaled} bit for bit the fp32 '
            f'product ONNX weight x HardSigmoid alpha (the converter folds the gate scale 1/6 into the conv before it)')


def any_hw(path, dst):
    """A net at any input size on LiteRT's CPU kernels (the check's reference): the fixed-shape file (the same fp16 weights and ops)
    with each resize's constant output size replaced by the size of the tensor it is concatenated with (the skip
    connection, or the input for the last two): SHAPE -> STRIDED_SLICE [1:3] of that tensor. The input / output
    shape signature becomes [1, -1, -1, C]; after TfLiteInterpreterResizeInputTensor to [1, H, W, 3] (H, W multiples
    of 32) and AllocateTensors every op re-derives its shapes (no RESHAPE or other shape constant is left in the
    graph). The GPU keeps the static file: its delegate does not take SHAPE / STRIDED_SLICE. Returns a summary."""
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.read_model(path)
    g = m.subgraphs[0]
    B = s.BuiltinOperator
    code = [c.builtinCode for c in m.operatorCodes]

    def opcode(op, version):
        for i, c in enumerate(m.operatorCodes):
            if c.builtinCode == op:
                return i
        c = s.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = op
        c.version = version
        m.operatorCodes.append(c)
        return len(m.operatorCodes) - 1

    def add_tensor(name, shape, ttype, data=None):
        b = s.BufferT()
        if data is not None:
            b.data = np.frombuffer(data.tobytes(), np.uint8)
        m.buffers.append(b)
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = name, np.array(shape, np.int32), ttype, len(m.buffers) - 1
        g.tensors.append(t)
        return len(g.tensors) - 1

    shape_op, slice_op = opcode(B.SHAPE, 1), opcode(B.STRIDED_SLICE, 1)
    made = {o.outputs[0]: code[o.opcodeIndex] for o in g.operators}
    users = {}
    for k, o in enumerate(g.operators):
        for i in o.inputs:
            users.setdefault(i, []).append(k)

    def hw(t):
        return [int(v) for v in g.tensors[t].shape[1:3]]

    resizes = (B.RESIZE_BILINEAR, B.RESIZE_NEAREST_NEIGHBOR)
    sizes, new_ops = {}, []   # reference tensor -> its [H, W] tensor
    begin = add_tensor(b'any_hw_begin', [1], s.TensorType.INT32, np.array([1], np.int32))
    end = add_tensor(b'any_hw_end', [1], s.TensorType.INT32, np.array([3], np.int32))
    stride = add_tensor(b'any_hw_stride', [1], s.TensorType.INT32, np.array([1], np.int32))
    n = 0
    for o in g.operators:
        if code[o.opcodeIndex] not in resizes:
            new_ops.append(o)
            continue
        y = o.outputs[0]
        cat = [g.operators[k] for k in users.get(y, []) if code[g.operators[k].opcodeIndex] == B.CONCATENATION]
        refs = [i for c in cat for i in c.inputs if i != y and made.get(i) not in resizes and hw(i) == hw(y)]
        if len(cat) != 1 or not refs:
            raise SystemExit(f'resize {g.tensors[y].name}: no concatenated tensor of its size')
        ref = refs[0]
        if ref not in sizes:
            sh = add_tensor(g.tensors[ref].name + b'_shape', [4], s.TensorType.INT32)
            p = s.OperatorT()
            p.opcodeIndex, p.inputs, p.outputs = shape_op, [ref], [sh]
            p.builtinOptionsType = s.BuiltinOptions.ShapeOptions
            p.builtinOptions = s.ShapeOptionsT()
            p.builtinOptions.outType = s.TensorType.INT32
            sz = add_tensor(g.tensors[ref].name + b'_hw', [2], s.TensorType.INT32)
            q = s.OperatorT()
            q.opcodeIndex, q.inputs, q.outputs = slice_op, [sh, begin, end, stride], [sz]
            q.builtinOptionsType = s.BuiltinOptions.StridedSliceOptions
            q.builtinOptions = s.StridedSliceOptionsT()
            new_ops += [p, q]
            sizes[ref] = sz
        o.inputs[1] = sizes[ref]
        new_ops.append(o)
        n += 1
    g.operators = new_ops
    for t in (g.inputs[0], g.outputs[0]):
        g.tensors[t].shapeSignature = np.array([1, -1, -1, int(g.tensors[t].shape[3])], np.int32)
    # the constant sizes no op reads any more
    read = {i for o in g.operators for i in o.inputs}
    for i, t in enumerate(g.tensors):
        if i not in read and i not in g.outputs and t.type == s.TensorType.INT32 and m.buffers[t.buffer].data is not None:
            m.buffers[t.buffer].data = None
    fu.write_model(m, dst)
    return f'{n} resize sizes from {len(sizes)} SHAPE -> STRIDED_SLICE pairs, input / output signature [1, -1, -1, C]'


def check_any(path, ref_path, sizes=None, gpu_note='the app runs the fixed-shape file, rewritten per size'):
    """An any-size file (shadenet_any.tflite, rulenet_any.tflite): runs at every size the pipeline feeds (sizes: (H, W)
    list, default 512 x SHADE_WIDTHS), and at the size of the fixed-shape file ref_path gives its output bit for bit
    (the same graph otherwise). Returns summary lines."""
    from ai_edge_litert.interpreter import Interpreter
    sizes = sizes or [(512, w) for w in SHADE_WIDTHS]
    it = Interpreter(model_path=path)
    ref = Interpreter(model_path=ref_path)
    ref.allocate_tensors()
    rin = ref.get_input_details()[0]
    rh, rw = int(rin['shape'][1]), int(rin['shape'][2])
    i0 = it.get_input_details()[0]
    c_out = int(ref.get_output_details()[0]['shape'][3])
    rng = np.random.default_rng(1)
    ok = []
    for h, w in sorted(set(sizes) | {(rh, rw)}):
        it.resize_tensor_input(i0['index'], [1, h, w, 3], strict=True)
        it.allocate_tensors()
        x = np.ascontiguousarray(rng.random((1, h, w, 3), dtype=np.float32))
        it.set_tensor(i0['index'], x)
        it.invoke()
        y = it.get_tensor(it.get_output_details()[0]['index'])
        if list(y.shape) != [1, h, w, c_out] or not np.isfinite(y).all():
            raise SystemExit(f'{path}: {h} x {w} gives {list(y.shape)}')
        if (h, w) == (rh, rw):
            ref.set_tensor(rin['index'], x)
            ref.invoke()
            if not np.array_equal(y, ref.get_tensor(ref.get_output_details()[0]['index'])):
                raise SystemExit(f'{path}: {h} x {w} differs from {os.path.basename(ref_path)}')
        ok.append(f'{h}x{w}')
    ops = {}
    for o in it._get_ops_details():
        ops[o['op_name']] = ops.get(o['op_name'], 0) + 1
    bad = gpu_ops(path)
    return [f'in {i0["name"]} signature {list(map(int, i0["shape_signature"]))}; runs at {", ".join(ok)}; '
            f'{rh} x {rw} bit for bit {os.path.basename(ref_path)}',
            'ops ' + ', '.join(f'{k} {v}' for k, v in ops.items()),
            f'GPU op check (information only: {gpu_note}): {len(bad)} problems' + (f', e.g. {bad[0]}' if bad else '')]


def rule_static(src, work, hw, gpu):
    """The rule net made static at hw = [H, W] (make_static: shape arithmetic folded, the global branch's Expand as
    a Mul by ones), converted and rewritten as the other nets (fp16 weights + DEQUANTIZE): work/rulenet_HxW.tflite.
    gpu: the GPU op check too. Returns its path."""
    import onnx2tf
    sys.path.insert(0, os.path.join(REPO, 'tools', 'accel'))
    from prepare_models import make_static
    tag = '%dx%d' % tuple(hw)
    m = onnx.load(src)
    iname = m.graph.input[0].name
    make_static(m, {iname: [1, 3] + list(hw)}, log=lambda *a: None)
    st = os.path.join(work, 'rule_static_%s.onnx' % tag)
    onnx.save(m, st)
    m, w16 = fp32_onnx(st)
    f32 = os.path.join(work, 'rule_%s_fp32.onnx' % tag)
    onnx.save(m, f32)
    onnx2tf.convert(input_onnx_file_path=f32, output_folder_path=os.path.join(work, 'rule_' + tag),
                    copy_onnx_input_output_names_to_tflite=True, non_verbose=True)
    tfl32 = os.path.join(work, 'rule_' + tag, 'rule_%s_fp32_float32.tflite' % tag)
    static = os.path.join(work, 'rulenet_%s.tflite' % tag)
    summary = rewrite(tfl32, static, bn_convs(m, w16), m.graph.output[0].name)
    nout = m.graph.output[0].type.tensor_type.shape.dim[1].dim_value   # 2 (rules) or 4 (+ text lines: tools/text_net)
    lines, text = check(static, [1] + list(hw) + [3], [1] + list(hw) + [nout], gpu=gpu)
    print(f'rule net {src}: static {hw[0]} x {hw[1]} {os.path.getsize(static)} bytes; {summary}')
    for s in lines:
        print('  ' + s, flush=True)
    return static


def convert_rule(src, work, out):
    """rulenet.tflite: rule_static() at RULE_HW with the GPU op check, the app's one rule net file (rewritten per page
    size by the loader); its any_hw() form in the work folder (rulenet_any.tflite) is run at RULE_SIZES."""
    static = rule_static(src, work, RULE_HW, gpu=True)
    path = os.path.join(out, RULE_OUT)
    shutil.copyfile(static, path)
    print(f'{RULE_OUT}: {os.path.getsize(path)} bytes (CPU and GPU, fixed at {RULE_HW[0]} x {RULE_HW[1]})', flush=True)
    any_path = os.path.join(work, 'rulenet_any.tflite')
    summary = any_hw(static, any_path)
    print(f'rulenet_any.tflite (work folder, the check's reference): {summary}')
    for s in check_any(any_path, static, RULE_SIZES):
        print('  ' + s, flush=True)


def check(path, want_in, want_out, gpu=True):
    """Contract C2 I/O, the GPU op check (gpu), no fp16-compute hint. Returns (summary lines, analyzer text)."""
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter
    from tensorflow.lite.tools import flatbuffer_utils as fu
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ins, outs = it.get_input_details(), it.get_output_details()
    io_ok = (len(ins) == 1 and len(outs) == 1 and list(ins[0]['shape']) == want_in and list(outs[0]['shape']) == want_out
             and ins[0]['dtype'] == np.float32 and outs[0]['dtype'] == np.float32)
    ops = {}
    for o in it._get_ops_details():
        ops[o['op_name']] = ops.get(o['op_name'], 0) + 1
    bad = gpu_ops(path) if gpu else []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_path=path, gpu_compatibility=True)
    text = buf.getvalue()
    verdict = [ln for ln in text.splitlines() if 'GPU' in ln]
    meta = [md.name.decode() for md in fu.read_model(path).metadata or []]
    lines = [f'in {ins[0]["name"]} {list(map(int, ins[0]["shape"]))} {ins[0]["dtype"].__name__}, '
             f'out {outs[0]["name"]} {list(map(int, outs[0]["shape"]))} {outs[0]["dtype"].__name__}'
             f'{"" if io_ok else "  NOT AS CONTRACT C2"}',
             'ops ' + ', '.join(f'{k} {v}' for k, v in ops.items()),
             'GPU op check: ' + ('not run (CPU only)' if not gpu else
                                 'every op in a form the GPU delegate takes' if not bad else '; '.join(bad)),
             'analyzer (gpu_compatibility=True): ' + ' / '.join(verdict),
             'metadata ' + ', '.join(meta)]
    if not io_ok or bad or 'reduced_precision_support' in meta:
        print('\n'.join(lines))
        raise SystemExit(f'{path}: check failed')
    return lines, text


def main():
    args = sys.argv[1:]
    work = None
    if '--work' in args:
        k = args.index('--work')
        work = os.path.abspath(args[k + 1])
        del args[k:k + 2]
    rule_src = RULE_SRC
    if '--rulenet' in args:
        k = args.index('--rulenet')
        rule_src = os.path.abspath(args[k + 1])
        del args[k:k + 2]
    known = [n[0] for n in NETS] + ['rule']
    out = os.path.abspath(args[0]) if args and args[0] not in known else OUT
    names = [a for a in args if a in known] or [n[0] for n in NETS] + (['rule'] if os.path.exists(rule_src) else [])
    keep_work = work is not None
    work = work or tempfile.mkdtemp(prefix='litert_convert_')
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    # onnx2tf runs onnxsim as a program: the venv's Scripts folder on PATH
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    import onnx2tf
    import tensorflow as tf
    print(f'tensorflow {tf.__version__}, onnx2tf {onnx2tf.__version__}, onnx {onnx.__version__}', flush=True)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        np.save(SAMPLE, np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
        for name, src, dst, want_in, want_out in NETS:
            if name not in names:
                continue
            m, w16 = fp32_onnx(os.path.join(SRC, src))
            f32 = os.path.join(work, name + '_fp32.onnx')
            onnx.save(m, f32)
            onnx2tf.convert(input_onnx_file_path=f32, output_folder_path=os.path.join(work, name),
                            copy_onnx_input_output_names_to_tflite=True, non_verbose=True)
            tfl32 = os.path.join(work, name, name + '_fp32_float32.tflite')
            path = os.path.join(out, dst)
            if name == 'finder':
                summary = keep_fp32(tfl32, path, m)
                lines, text = check(path, want_in, want_out, gpu=False)
            else:
                summary = rewrite(tfl32, path, bn_convs(m, w16), m.graph.output[0].name)
                lines, text = check(path, want_in, want_out)
            with open(os.path.join(work, name + '.analyzer.txt'), 'w') as fh:
                fh.write(text)
            print(f'{dst}: {os.path.getsize(path)} bytes; {summary}')
            for s in lines:
                print('  ' + s, flush=True)
            if name == 'shade':
                any_path = os.path.join(work, SHADE_ANY)
                summary = any_hw(path, any_path)
                print(f'{SHADE_ANY} (work folder, the check's reference): {summary}')
                for s in check_any(any_path, path):
                    print('  ' + s, flush=True)
        if 'rule' in names:
            convert_rule(rule_src, work, out)
    finally:
        os.chdir(cwd)
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
