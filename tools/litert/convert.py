"""LiteRT (TFLite) copies of the fixed-shape nets for the LiteRT GPU delegate (contract C2).

  python tools/litert/convert.py [OUT_DIR] [--work DIR] [NAME ...]
      OUT_DIR default app/src/main/litert/models; --work default a temp folder (deleted afterwards)
      NAME: uvdoc, sdhr, shade (default all)
      -> uvdoc_static.tflite, syntheticdoc_hr_static.tflite, shadenet_512x352.tflite, and NAME.analyzer.txt
         (tf.lite.experimental.Analyzer, gpu_compatibility=True) in the work folder
Run it with C:/Users/User/Android/litert_venv (tensorflow 2.19.1, onnx2tf 1.28.8: docs/results/litert/models.txt).
Sources: the static ONNX files of app/src/main/assets/models (tools/accel/prepare_models.py).

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
        ('shade', 'shadenet_512x352.onnx', 'shadenet_512x352.tflite', [1, 512, 352, 3], [1, 512, 352, 3])]
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


def check(path, want_in, want_out):
    """Contract C2 I/O, the GPU op check, no fp16-compute hint. Returns (summary lines, analyzer text)."""
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
    bad = gpu_ops(path)
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
             'GPU op check: ' + ('every op in a form the GPU delegate takes' if not bad else '; '.join(bad)),
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
    out = os.path.abspath(args[0]) if args and args[0] not in [n[0] for n in NETS] else OUT
    names = [a for a in args if a in [n[0] for n in NETS]] or [n[0] for n in NETS]
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
            summary = rewrite(tfl32, path, bn_convs(m, w16), m.graph.output[0].name)
            lines, text = check(path, want_in, want_out)
            with open(os.path.join(work, name + '.analyzer.txt'), 'w') as fh:
                fh.write(text)
            print(f'{dst}: {os.path.getsize(path)} bytes; {summary}')
            for s in lines:
                print('  ' + s, flush=True)
    finally:
        os.chdir(cwd)
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
