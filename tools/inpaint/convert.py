"""LiteRT (TFLite) copy of the MI-GAN 512 inpainting generator: the optional "content-aware (AI)" fill of a removed
finger.

  python -I tools/inpaint/convert.py ONNX OUT_DIR [--work DIR]
      ONNX: migan_512.onnx made by tools/inpaint/export_onnx.py (the bare generator, NCHW [1, 4, 512, 512] ->
            [1, 3, 512, 512], fp32 weights)
      --work: onnx2tf's working folder (default a temp folder, deleted afterwards)
      -> OUT_DIR/migan_512.tflite: fixed input "input" [1, 512, 512, 4] float32 NHWC, output "output"
         [1, 512, 512, 3] float32 NHWC; fp32 graph, weights stored fp16 with a DEQUANTIZE (fp16 -> fp32) in front
         (the form of tools/litert/convert.py: what the converter's float16 mode stores), conv biases fp32
      -> OUT_DIR/migan_512_fp32.tflite: the same graph with every constant fp32 (the conversion's reference: it
         differs from PyTorch only by the kernels' arithmetic; tools/inpaint/check.py measures both)
Run it with C:/Users/User/Android/litert_venv (tensorflow 2.19.1, onnx2tf 1.28.8, as tools/litert/convert.py).
The whole chain and the net's input / output conventions: tools/inpaint/README.md.

Steps:
  1. fp16 weights in the ONNX graph: every Conv filter (input 1) and the synthesis blocks' fixed noise constants
     ([1, 1, R, R], noise_const * noise_strength folded by the export) rounded to the nearest fp16 number, stored
     fp32; Conv biases and the zero-insertion masks of the upsamplers (0 / 1, exact) as they are. MI-GAN's weights
     are fp32, so this is the one rounding of the chain (tools/inpaint/check.py: its effect on the output).
  2. onnx2tf: NCHW -> NHWC, fp32 .tflite of both graphs (rounded and not).
  3. store_fp16() on the rounded one (the fp16 storage of tools/litert/convert.py rewrite()): every constant whose
     fp32 values are all fp16 numbers is stored fp16 + DEQUANTIZE (exact), except a depthwise filter whose bias is
     fp32 (kept fp32, values unchanged: XNNPACK does not delegate that pair); both files' output named "output".
  4. checks: I/O names, shapes and types, the op list. The target is LiteRT's CPU: no GPU form is made (the
     upsamplers' x * [1, H, W, 1] zero-insertion mask and the [1, H, W, 1] noise adds are broadcasts the GPU
     delegate's ADD / MUL do not take, see tools/litert/convert.py gpu_ops()).
tools/litert/convert.py is not imported (its rewrite() would do step 3): two of its f-strings have a quote inside
(lines 561, 652, "the check's reference"), which needs Python 3.12; the venv's is 3.11.9."""
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper

IN_SHAPE, OUT_SHAPE = [1, 512, 512, 4], [1, 512, 512, 3]
# onnx2tf runs a sample inference on this file when it is in the working folder (else it downloads one)
SAMPLE = 'calibration_image_sample_data_20x128x128x3_float32.npy'


def round_fp16(m):
    """Step 1 on the ONNX model m (in place). Returns (tensors rounded, values rounded, tensors kept)."""
    use = {}
    for nd in m.graph.node:
        for k, i in enumerate(nd.input):
            use.setdefault(i, []).append((nd.op_type, k))
    nr = vr = nk = 0
    for idx, init in enumerate(m.graph.initializer):
        a = numpy_helper.to_array(init)
        u = use.get(init.name, [])
        weight = bool(u) and all(t == ('Conv', 1) for t in u)
        noise = bool(u) and all(t == ('Add', 1) for t in u) and a.ndim == 4 and a.shape[:2] == (1, 1)
        if a.dtype != np.float32 or not (weight or noise):
            nk += 1
            continue
        r = a.astype(np.float16).astype(np.float32)
        m.graph.initializer[idx].CopyFrom(numpy_helper.from_array(r, init.name))
        nr, vr = nr + 1, vr + a.size
    return nr, vr, nk


def to_tflite(onnx_path, work, tag):
    """Step 2: onnx2tf's fp32 .tflite of onnx_path (NHWC I/O)."""
    import onnx2tf
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=os.path.join(work, tag),
                    copy_onnx_input_output_names_to_tflite=True, non_verbose=True)
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    return os.path.join(work, tag, base + '_float32.tflite')


def store_fp16(path, dst, out_name, fp16=True):
    """Step 3: (fp16) every fp32 constant whose values are all fp16 numbers stored fp16 with a DEQUANTIZE (version 3,
    fp16 input) in front, as tools/litert/convert.py rewrite() and the converter's float16 mode store them; the
    output tensor renamed out_name. Returns a summary line."""
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.read_model(path)
    g = m.subgraphs[0]
    g.tensors[g.outputs[0]].name = out_name.encode()
    if not fp16:
        fu.write_model(m, dst)
        return 'all constants fp32'
    dq = next((i for i, c in enumerate(m.operatorCodes) if c.builtinCode == s.BuiltinOperator.DEQUANTIZE), None)
    if dq is None:
        c = s.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = s.BuiltinOperator.DEQUANTIZE
        c.version = 3
        m.operatorCodes.append(c)
        dq = len(m.operatorCodes) - 1
    def fp16_exact(ti):
        d = m.buffers[g.tensors[ti].buffer].data
        if d is None or len(d) == 0:
            return None
        a = np.frombuffer(bytes(d), np.float32)
        return np.array_equal(a.astype(np.float16).astype(np.float32), a)

    code = [c.builtinCode for c in m.operatorCodes]
    deq, done, n16, v16, n32, v32, ndw = [], set(), 0, 0, 0, 0, 0
    for o in g.operators:
        # XNNPACK (TF 2.19's interpreter) fails to delegate a DEPTHWISE_CONV_2D with an fp16 (DEQUANTIZE) filter and
        # an fp32 bias (measured: 8 of MI-GAN's 53, "failed to delegate DEPTHWISE_CONV_2D node #N" at
        # allocate_tensors); such a filter stays fp32 (3 x 3 x C, a few kB each)
        keep32 = set()
        if code[o.opcodeIndex] == s.BuiltinOperator.DEPTHWISE_CONV_2D and len(o.inputs) > 2 and o.inputs[2] >= 0 \
                and fp16_exact(o.inputs[2]) is False:
            keep32.add(o.inputs[1])
        for ti in o.inputs:
            if ti < 0 or ti in done or g.tensors[ti].type != s.TensorType.FLOAT32:
                continue
            t = g.tensors[ti]
            d = m.buffers[t.buffer].data
            if d is None or len(d) == 0:
                continue
            done.add(ti)
            a = np.frombuffer(bytes(d), np.float32)
            h = a.astype(np.float16)
            if ti in keep32 or not np.array_equal(h.astype(np.float32), a):
                n32, v32 = n32 + 1, v32 + a.size
                ndw += ti in keep32
                continue
            n16, v16 = n16 + 1, v16 + a.size
            b = s.BufferT()
            b.data = np.frombuffer(h.tobytes(), np.uint8)
            m.buffers.append(b)
            t16 = s.TensorT()
            t16.name, t16.shape, t16.type, t16.buffer = (t.name + b'_fp16', t.shape, s.TensorType.FLOAT16,
                                                         len(m.buffers) - 1)
            g.tensors.append(t16)
            m.buffers[t.buffer].data = None
            p = s.OperatorT()
            p.opcodeIndex, p.inputs, p.outputs = dq, [len(g.tensors) - 1], [ti]
            p.builtinOptionsType = s.BuiltinOptions.DequantizeOptions
            p.builtinOptions = s.DequantizeOptionsT()
            deq.append(p)
    g.operators = deq + list(g.operators)
    for md in m.metadata or []:
        if md.name == b'min_runtime_version':   # what the converter's float16 mode writes (fp16 DEQUANTIZE)
            m.buffers[md.buffer].data = np.frombuffer(b'1.15.0'.ljust(16, bytes(1)), np.uint8)
    fu.write_model(m, dst)
    return (f'fp16 constants {n16} ({v16} values), fp32 constants {n32} ({v32} values; {ndw} of them depthwise filters '
            f'kept fp32 for XNNPACK)')


def describe(path):
    """Step 4: the I/O check and the op list (a summary line)."""
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    i, o = it.get_input_details(), it.get_output_details()
    ok = (len(i) == 1 and len(o) == 1 and list(i[0]['shape']) == IN_SHAPE and list(o[0]['shape']) == OUT_SHAPE
          and i[0]['dtype'] == np.float32 and o[0]['dtype'] == np.float32 and i[0]['name'] == 'input'
          and o[0]['name'] == 'output')
    ops = {}
    for d in it._get_ops_details():
        ops[d['op_name']] = ops.get(d['op_name'], 0) + 1
    line = (f'in {i[0]["name"]} {list(map(int, i[0]["shape"]))} {i[0]["dtype"].__name__}, out {o[0]["name"]} '
            f'{list(map(int, o[0]["shape"]))} {o[0]["dtype"].__name__}; ops '
            + ', '.join(f'{k} {v}' for k, v in sorted(ops.items())))
    if not ok:
        raise SystemExit(f'{path}: I/O not as wanted: {line}')
    return line


def main():
    args = sys.argv[1:]
    work = None
    if '--work' in args:
        k = args.index('--work')
        work = os.path.abspath(args[k + 1])
        del args[k:k + 2]
    src, out = os.path.abspath(args[0]), os.path.abspath(args[1])
    keep_work = work is not None
    work = work or tempfile.mkdtemp(prefix='inpaint_convert_')
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
        m = onnx.load(src)
        f32 = os.path.join(work, 'migan_fp32w.onnx')
        onnx.save(m, f32)
        nr, vr, nk = round_fp16(m)
        onnx.checker.check_model(m)
        r16 = os.path.join(work, 'migan_fp16w.onnx')
        onnx.save(m, r16)
        print(f'step 1: {nr} initializers ({vr} values) rounded to fp16 numbers, {nk} kept', flush=True)
        for tag, onx, name, fp16 in (('fp32w', f32, 'migan_512_fp32.tflite', False),
                                     ('fp16w', r16, 'migan_512.tflite', True)):
            dst = os.path.join(out, name)
            summary = store_fp16(to_tflite(onx, work, tag), dst, 'output', fp16=fp16)
            print(f'{dst}: {os.path.getsize(dst)} bytes; {summary}')
            print('  ' + describe(dst), flush=True)
    finally:
        os.chdir(cwd)
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
