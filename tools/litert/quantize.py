"""Int8 weight copies of the nets (Round 11, stream MODELSIZE): half the bytes of the fp16 files, fp32 compute.

  python tools/litert/quantize.py tflite SRC.tflite DST.tflite [--form dq|conv] [--per-tensor] [--min N] [--keep T,...]
      int8 weights of an fp16-weight .tflite (tools/litert/convert.py's form: FLOAT16 constant -> DEQUANTIZE -> the
      filter input of CONV_2D / DEPTHWISE_CONV_2D / TRANSPOSE_CONV / FULLY_CONNECTED). Only those filters change;
      every other constant (biases, BatchNorm scales, PRelu slopes, small filters) stays as it is.
        --form dq (default): INT8 constant -> DEQUANTIZE -> the op (the fp16 layout with int8 storage). The op computes
            in fp32 with the dequantized weights: XNNPACK unpacks a static int8 DEQUANTIZE to fp32 at load time
            (xnnpack_delegate.cc, static_unpack_nodes_), the builtin kernel does it once (dequantize.cc, constant
            input), so CPU outputs are those of the fp32 graph with dequantized weights (`sim` below), bit for bit.
        --form conv: the INT8 filter on the op itself, no DEQUANTIZE (the converter's dynamic-range form). The GPU
            delegate dequantizes constant int8 filters when it builds its graph (model_builder_helper.cc,
            CreateVectorCopyData<float>), but LiteRT's CPU kernels then run the op "hybrid" (int8 activations,
            quantized per invocation): other numbers than `sim`, measured separately.
      --per-tensor: one scale per filter (default: one per output channel); --min N: filters with fewer than N
      values stay fp16 (default 4096); --keep: tensor names (substrings) that stay fp16.
  python tools/litert/quantize.py sim SRC.onnx DST.onnx [--per-tensor] [--min N]
      the ONNX graph with every Conv / ConvTranspose / Gemm / MatMul weight replaced by its int8 dequantized values
      (fp32 initializers): exactly the weights `tflite --form dq` computes with, for pipeline_cli (FS_MODELS) and
      ONNX Runtime. Same rule for which weights are quantized (size >= --min, fp16 / fp32 weights).
  python tools/litert/quantize.py info FILE.tflite|FILE.onnx
      bytes, parameters by storage type, ops.

Quantization (both commands, the same numbers): symmetric int8, zero point 0, per output channel (axis 0 of an
OIHW / OHWI filter, axis 1 of ConvTranspose's IOHW, the channel axis of a depthwise filter): scale = max|w| / 127
in float32, q = clip(rint(w / scale), -127, 127), w' = q * scale in float32 (the DEQUANTIZE kernels' arithmetic).
w is the fp16 weight as float32 (the ONNX files' and the .tflite files' fp16 values are the same numbers).
Run with C:/Users/User/Android/litert_venv (tensorflow 2.19.1 for the flatbuffer tools; onnx for `sim`)."""
import os
import sys

import numpy as np

QMAX = 127


def qparams(w, axis, per_tensor=False):
    """(q int8, scale float32 [C] or [1]) for float32 w quantized along axis."""
    w = np.asarray(w, np.float32)
    if per_tensor:
        amax = np.array([np.abs(w).max()], np.float32)
        shape = [1] * w.ndim
    else:
        other = tuple(i for i in range(w.ndim) if i != axis)
        amax = np.abs(w).max(axis=other).astype(np.float32)
        shape = [1] * w.ndim
        shape[axis] = -1
    scale = (amax / np.float32(QMAX)).astype(np.float32)
    scale[scale == 0] = np.float32(1)
    q = np.clip(np.rint(w / scale.reshape(shape)), -QMAX, QMAX).astype(np.int8)
    return q, scale


def dequant(q, scale, axis):
    shape = [1] * q.ndim
    if scale.size > 1:
        shape[axis] = -1
    return (q.astype(np.float32) * scale.reshape(shape)).astype(np.float32)


def fake_quant(w, axis, per_tensor=False):
    q, s = qparams(w, axis, per_tensor)
    return dequant(q, s, axis)


# ---------------------------------------------------------------------------------------------------- .tflite

def _tfl():
    from tensorflow.lite.python import schema_py_generated as s
    from tensorflow.lite.tools import flatbuffer_utils as fu
    return s, fu


def onnx_conv_weights(path):
    """[(ONNX Conv node name, weight as float32 OIHW, group)] of an ONNX file (fp16 + Cast or fp32 weights)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    out = []
    for n in m.graph.node:
        if n.op_type != 'Conv':
            continue
        w = n.input[1]
        src = w if w in inits else (prod[w].input[0] if w in prod and prod[w].op_type == 'Cast' else None)
        if src is None or src not in inits:
            continue
        group = next((a.i for a in n.attribute if a.name == 'group'), 1)
        out.append((n.name, w, numpy_helper.to_array(inits[src]).astype(np.float32), group))
    return out


def tflite_int8(src, dst, form='dq', per_tensor=False, min_size=4096, keep=(), onnx_src=None, qmap=None, strict=True, no_fp16=False):
    """Rewrites src (fp16 filters behind DEQUANTIZE) with int8 filters; returns a summary dict.
    onnx_src: the ONNX file the .tflite was made from; each filter is matched to its ONNX Conv weight by value
    (OIHW -> OHWI for CONV_2D, -> 1 x KH x KW x C for DEPTHWISE_CONV_2D), so that keep (ONNX Conv node name
    substrings that stay fp16) and qmap ({ONNX weight input name: (q int8 OIHW, scale)}, gptq.py) apply.
    Without onnx_src, keep is matched against the .tflite tensor names."""
    s, fu = _tfl()
    m = fu.read_model(src)
    g = m.subgraphs[0]
    B, T = s.BuiltinOperator, s.TensorType
    code = [c.builtinCode for c in m.operatorCodes]
    # filter input index and the channel axis of the filter tensor per op
    # DEPTHWISE_CONV_2D filters stay fp16: with an int8 DEQUANTIZE, tensorflow 2.19.1's XNNPACK "failed to delegate
    # DEPTHWISE_CONV_2D" (measured on the page finder) and they are small (KH x KW x C values)
    filt = {B.CONV_2D: (1, 0), B.TRANSPOSE_CONV: (1, 0), B.FULLY_CONNECTED: (1, 0)}
    made = {o.outputs[0]: o for o in g.operators}
    users = {}
    for o in g.operators:
        for i in o.inputs:
            users.setdefault(i, []).append(o)
    dq_op = None
    for i, c in enumerate(m.operatorCodes):
        if c.builtinCode == B.DEQUANTIZE:
            dq_op = i
    done, drop = {}, set()
    n_q = n_keep = v_q = v_keep = n_gptq = n_bias = n_fp32 = 0
    ows = onnx_conv_weights(onnx_src) if onnx_src else []
    matched, kept, unmatched = [], [], []
    for o in g.operators:
        op = code[o.opcodeIndex]
        if op not in filt:
            continue
        k, axis = filt[op]
        t_out = o.inputs[k]
        d = made.get(t_out)
        if d is None or code[d.opcodeIndex] != B.DEQUANTIZE:
            continue
        t16 = d.inputs[0]
        t = g.tensors[t16]
        if t.type != T.FLOAT16:
            continue
        name = t.name.decode()
        w = np.frombuffer(bytes(m.buffers[t.buffer].data), np.float16).astype(np.float32).reshape([int(v) for v in t.shape])
        if t16 in done:
            continue
        match = None
        if onnx_src is not None:
            for node, wname, W, group in ows:
                Wt = W.transpose(0, 2, 3, 1) if op != B.DEPTHWISE_CONV_2D else W.transpose(1, 2, 3, 0)
                if list(Wt.shape) == list(w.shape) and np.array_equal(Wt.astype(np.float16).astype(np.float32), w):
                    match = (node, wname)
                    break
            if match is None and strict:
                raise SystemExit(f'{name} {list(w.shape)}: no ONNX Conv weight with these values in {onnx_src}')
            if match is None:
                # a filter the converter changed (folded a following scale into it): rounded to nearest; the
                # sim ONNX then differs from the .tflite for this filter (counted in the summary)
                unmatched.append(name)
            else:
                matched.append(match[0])
        label = match[0] if match else name
        if w.size < min_size or any(kk in label for kk in keep):
            n_keep, v_keep = n_keep + 1, v_keep + w.size
            kept.append(label)
            continue
        # another reader of the fp32 weights than this filter input: leave it (not seen in these nets)
        if any(u is not o and code[u.opcodeIndex] not in filt for u in users.get(t_out, [])):
            n_keep, v_keep = n_keep + 1, v_keep + w.size
            continue
        if qmap is not None and match and match[1] in qmap:
            q, scale = qmap[match[1]]
            q = q.transpose(0, 2, 3, 1) if op != B.DEPTHWISE_CONV_2D else q.transpose(1, 2, 3, 0)
            q = np.ascontiguousarray(q)
            n_gptq += 1
        else:
            q, scale = qparams(w, axis, per_tensor)
        m.buffers[t.buffer].data = np.frombuffer(q.tobytes(), np.uint8)
        t.type = T.INT8
        qp = s.QuantizationParametersT()
        qp.scale = scale.astype(np.float32).tolist()
        qp.zeroPoint = [0] * scale.size
        qp.quantizedDimension = axis if scale.size > 1 else 0
        t.quantization = qp
        done[t16] = d
        n_q, v_q = n_q + 1, v_q + w.size
        # an fp16 bias (fp16 DEQUANTIZE) of this op becomes an fp32 constant: XNNPACK passes fp16 DEQUANTIZEs that
        # feed GEMMs to its fp16-weight path, and a conv with an int8-unpacked fp32 filter and an fp16 bias failed
        # to delegate (tensorflow 2.19.1: "failed to delegate CONV_2D", AllocateTensors fails). Same values.
        if len(o.inputs) > 2 and o.inputs[2] >= 0:
            bd = made.get(o.inputs[2])
            if bd is not None and code[bd.opcodeIndex] == B.DEQUANTIZE and g.tensors[bd.inputs[0]].type == T.FLOAT16:
                bt = g.tensors[bd.inputs[0]]
                bv = np.frombuffer(bytes(m.buffers[bt.buffer].data), np.float16).astype(np.float32)
                buf = s.BufferT()
                buf.data = np.frombuffer(bv.tobytes(), np.uint8)
                m.buffers.append(buf)
                nt = s.TensorT()
                nt.name, nt.shape, nt.type, nt.buffer = bt.name + b'_fp32', np.array(bt.shape, np.int32), T.FLOAT32, len(m.buffers) - 1
                g.tensors.append(nt)
                o.inputs[2] = len(g.tensors) - 1
                n_bias += 1
        if form == 'conv':
            # the op reads the int8 constant itself; the DEQUANTIZE goes
            for u in users.get(t_out, []):
                u.inputs = [t16 if i == t_out else i for i in u.inputs]
            drop.add(id(d))
    if form == 'conv':
        g.operators = [o for o in g.operators if id(o) not in drop]
        # the hybrid kernels' op versions (tensorflow/lite/tools/versioning/op_version.cc): CONV_2D 5 with
        # per-channel int8 filters and float input, DEPTHWISE_CONV_2D 6, FULLY_CONNECTED 3 / TRANSPOSE_CONV 1
        want = {B.CONV_2D: 5 if not per_tensor else 2, B.DEPTHWISE_CONV_2D: 6 if not per_tensor else 2,
                B.FULLY_CONNECTED: 3, B.TRANSPOSE_CONV: 1}
        for c in m.operatorCodes:
            if c.builtinCode in want:
                c.version = max(c.version, want[c.builtinCode])
    else:
        # one DEQUANTIZE opcode for the int8 ones, version 2 (int8 input): the GPU delegate's DEQUANTIZE parser
        # takes versions <= 3 (model_builder.cc CheckMaxSupportedOpVersion(registration, 3)); the per-channel
        # version 5 of op_version.cc would put every filter off the GPU. The CPU kernel reads the quantization
        # parameters, not the version (dequantize.h PerChannelDequantizeImpl).
        c = s.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = B.DEQUANTIZE
        c.version = 2
        m.operatorCodes.append(c)
        k8 = len(m.operatorCodes) - 1
        for d in done.values():
            d.opcodeIndex = k8
        del dq_op
        # each int8 DEQUANTIZE right before the op that reads it (convert.py puts every DEQUANTIZE first): the
        # XNNPACK delegate unpacks a static DEQUANTIZE only inside the partition of its consumer, and a filter
        # unpacked in an earlier partition (before a MIRROR_PAD / CONCATENATION XNNPACK leaves to the builtin
        # kernels) is a runtime tensor to the later one: "failed to delegate CONV_2D" and AllocateTensors fails
        # (measured, tensorflow 2.19.1's XNNPACK). fp16 DEQUANTIZEs are remapped by XNNPACK and stay where they are.
        mine = {id(d) for d in done.values()}
        rest = [o for o in g.operators if id(o) not in mine]
        first = {}
        for d in done.values():
            first[id(d)] = d
        by_out = {d.outputs[0]: d for d in done.values()}
        ordered = []
        for o in rest:
            for i in o.inputs:
                if i in by_out:
                    ordered.append(by_out.pop(i))
            ordered.append(o)
        ordered.extend(by_out.values())
        g.operators = ordered
    if no_fp16:
        # every fp16 DEQUANTIZE left (kept filters, biases, slopes) -> an fp32 constant, so that no fp16 weight reaches
        # XNNPACK's fp16-weight path: tensorflow 2.19.1's XNNPACK gave wrong outputs (UVDoc, max |d| 0.99) or NaN
        # (SD-HR, ShadeNet) on the fp16-weight files and "failed to delegate" the page finder's depthwise convs
        # (measured on the desktop; the builtin kernels match ONNX Runtime to 5e-7)
        code3 = [c.builtinCode for c in m.operatorCodes]
        repl = {}
        for o in g.operators:
            if code3[o.opcodeIndex] == B.DEQUANTIZE and g.tensors[o.inputs[0]].type == T.FLOAT16:
                bt = g.tensors[o.inputs[0]]
                bv = np.frombuffer(bytes(m.buffers[bt.buffer].data), np.float16).astype(np.float32)
                buf = s.BufferT()
                buf.data = np.frombuffer(bv.tobytes(), np.uint8)
                m.buffers.append(buf)
                nt = s.TensorT()
                nt.name, nt.shape, nt.type, nt.buffer = bt.name + b'_fp32', np.array(bt.shape, np.int32), T.FLOAT32, len(m.buffers) - 1
                g.tensors.append(nt)
                repl[o.outputs[0]] = len(g.tensors) - 1
        for o in g.operators:
            o.inputs = [repl.get(i, i) for i in o.inputs]
        n_fp32 = len(repl)
    read = {i for o in g.operators for i in o.inputs} | set(g.outputs)
    code2 = [c.builtinCode for c in m.operatorCodes]
    g.operators = [o for o in g.operators if not (code2[o.opcodeIndex] == B.DEQUANTIZE and o.outputs[0] not in read)]
    for i, t in enumerate(g.tensors):
        if i not in {j for o in g.operators for j in list(o.inputs) + list(o.outputs)} and i not in g.outputs                 and m.buffers[t.buffer].data is not None:
            m.buffers[t.buffer].data = None
    fu.write_model(m, dst)
    return dict(fp16_to_fp32=n_fp32, unmatched=unmatched, fp32_biases=n_bias, quantized=n_q, quantized_values=v_q, from_qmap=n_gptq, kept_fp16=n_keep, kept_values=v_keep,
                kept=kept, bytes=os.path.getsize(dst))


def tflite_info(path):
    s, fu = _tfl()
    m = fu.read_model(path)
    g = m.subgraphs[0]
    tn = {v: k for k, v in vars(s.TensorType).items() if not k.startswith('_')}
    on = {v: k for k, v in vars(s.BuiltinOperator).items() if not k.startswith('_')}
    by = {}
    for t in g.tensors:
        d = m.buffers[t.buffer].data
        if d is not None and len(d):
            k = tn[t.type]
            n = int(np.prod(t.shape)) if len(t.shape) else 1
            by[k] = by.get(k, 0) + n
    ops = {}
    code = [c.builtinCode for c in m.operatorCodes]
    for o in g.operators:
        ops[on[code[o.opcodeIndex]]] = ops.get(on[code[o.opcodeIndex]], 0) + 1
    ins = [(g.tensors[i].name.decode(), [int(v) for v in g.tensors[i].shape]) for i in g.inputs]
    outs = [(g.tensors[i].name.decode(), [int(v) for v in g.tensors[i].shape]) for i in g.outputs]
    return dict(bytes=os.path.getsize(path), values=by, ops=ops, inputs=ins, outputs=outs)


# ---------------------------------------------------------------------------------------------------- ONNX

def onnx_sim(src, dst, per_tensor=False, min_size=4096, keep=(), qmap=None):
    """fp32-weight ONNX with the int8 dequantized weights (module docstring). Returns a summary dict."""
    import onnx
    from onnx import TensorProto, numpy_helper
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    # (weight input name, channel axis) per consumer op
    want = {}
    for n in g.node:
        if any(k in n.name for k in keep):
            continue
        if n.op_type == 'Conv' and len(n.input) > 1:
            if next((a.i for a in n.attribute if a.name == 'group'), 1) != 1:
                continue   # grouped / depthwise: fp16 (tflite_int8 leaves DEPTHWISE_CONV_2D filters fp16)
            want.setdefault(n.input[1], 0)
        elif n.op_type == 'ConvTranspose' and len(n.input) > 1:
            group = next((a.i for a in n.attribute if a.name == 'group'), 1)
            want.setdefault(n.input[1], 1 if group == 1 else 0)
        elif n.op_type == 'Gemm' and len(n.input) > 1:
            tb = next((a.i for a in n.attribute if a.name == 'transB'), 0)
            want.setdefault(n.input[1], 0 if tb else 1)
        elif n.op_type == 'MatMul' and n.input[1] in inits or (n.op_type == 'MatMul' and n.input[1] in prod):
            want.setdefault(n.input[1], 1)
    n_q = v_q = n_keep = 0
    drop_nodes = set()
    for name, axis in want.items():
        if name in inits:
            init = inits[name]
            w = numpy_helper.to_array(init).astype(np.float32)
            src_kind = TensorProto.DataType.Name(init.data_type)
            target = name
        elif name in prod and prod[name].op_type == 'Cast' and prod[name].input[0] in inits:
            init = inits[prod[name].input[0]]
            w = numpy_helper.to_array(init).astype(np.float32)
            src_kind = TensorProto.DataType.Name(init.data_type) + '+Cast'
            target = name
        else:
            continue
        if w.size < min_size or w.ndim < 2:
            n_keep += 1
            continue
        if qmap is not None and name in qmap:
            fq = dequant(qmap[name][0], qmap[name][1], 0)
        else:
            fq = fake_quant(w, axis, per_tensor)
        if target in inits:
            inits[target].CopyFrom(numpy_helper.from_array(fq, target))
        else:
            # the Cast's output becomes an fp32 initializer
            drop_nodes.add(id(prod[target]))
            g.initializer.append(numpy_helper.from_array(fq, target))
        n_q, v_q = n_q + 1, v_q + w.size
    keep = [n for n in g.node if id(n) not in drop_nodes]
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    live = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(live)
    onnx.save(m, dst)
    return dict(quantized=n_q, quantized_values=v_q, kept=n_keep, bytes=os.path.getsize(dst))


def onnx_info(path):
    import onnx
    from onnx import TensorProto, numpy_helper
    m = onnx.load(path)
    by = {}
    for i in m.graph.initializer:
        k = TensorProto.DataType.Name(i.data_type)
        by[k] = by.get(k, 0) + int(np.prod(i.dims)) if len(i.dims) else by.get(k, 0) + 1
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    init_names = {i.name for i in m.graph.initializer}

    def shp(v):
        return [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    return dict(bytes=os.path.getsize(path), values=by, ops=ops,
                inputs=[(v.name, shp(v)) for v in m.graph.input if v.name not in init_names],
                outputs=[(v.name, shp(v)) for v in m.graph.output])


def load_qmap(path):
    """gptq.py's OUT.npz -> {ONNX weight input name: (q int8, scale)} (None for no path)."""
    if not path:
        return None
    z = np.load(path)
    return {k[:-3]: (z[k], z[k[:-3] + '::s']) for k in z.files if k.endswith('::q')}


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2

    def opt(flag, default=None, cast=str):
        if flag in a:
            k = a.index(flag)
            v = a[k + 1]
            del a[k:k + 2]
            return cast(v)
        return default

    def flag(f):
        if f in a:
            a.remove(f)
            return True
        return False
    cmd = a.pop(0)
    if cmd == 'info':
        for p in a:
            print(p, tflite_info(p) if p.endswith('.tflite') else onnx_info(p))
        return 0
    per_tensor = flag('--per-tensor')
    min_size = opt('--min', 4096, int)
    if cmd == 'tflite':
        form = opt('--form', 'dq')
        keep = tuple(x for x in (opt('--keep', '') or '').split(',') if x)
        onnx_src = opt('--onnx')
        qm = load_qmap(opt('--qmap'))
        print(tflite_int8(a[0], a[1], form, per_tensor, min_size, keep, onnx_src, qm, no_fp16=flag('--no-fp16')))
    elif cmd == 'sim':
        keep = tuple(x for x in (opt('--keep', '') or '').split(',') if x)
        print(onnx_sim(a[0], a[1], per_tensor, min_size, keep, load_qmap(opt('--qmap'))))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
