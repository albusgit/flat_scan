"""Per-layer int8 sensitivity of a net (Round 11, stream MODELSIZE): which weights cost the accuracy.

  python tools/litert/sensitivity.py NET MODEL.onnx INPUTS_DIR NAME,NAME,... [--per-tensor] [--mode one|allbut]
      NET: uv_p1 | uv_rc0 | sdhr_rc2 | shade | pf (as parity_q.py); MODEL: the fp16 / fp32 ONNX the CPU path runs.
      --mode one: only this weight int8 (default); allbut: every weight int8 except this one.
Prints per weight (size order): values, its effect (grid p99 / max px, shading levels, page finder 1 - IoU)
against MODEL on the photos' inputs. ONNX Runtime CPU with quantize.py's fake quantization (the same numbers as the
int8 .tflite, quantize.py `sim`)."""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_q  # noqa: E402
import quantize  # noqa: E402


def weights(m, min_size=4096):
    inits = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    out = []
    for n in m.graph.node:
        if n.op_type not in ('Conv', 'ConvTranspose'):
            continue
        w = n.input[1]
        src = w if w in inits else (prod[w].input[0] if w in prod and prod[w].op_type == 'Cast' else None)
        if src is None or src not in inits:
            continue
        a = numpy_helper.to_array(inits[src]).astype(np.float32)
        if a.size >= min_size:
            out.append((w, src, a, 1 if n.op_type == 'ConvTranspose' else 0, n.name))
    return out


def build(m, repl):
    """bytes of m with the Conv weights in repl {weight input name: fp32 array} as fp32 initializers."""
    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    g = m2.graph
    prod = {o: n for n in g.node for o in n.output}
    drop = set()
    inits = {i.name: i for i in g.initializer}
    for name, a in repl.items():
        if name in inits:
            inits[name].CopyFrom(numpy_helper.from_array(a, name))
        else:
            drop.add(id(prod[name]))
            g.initializer.append(numpy_helper.from_array(a, name))
    keep = [n for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(keep)
    return m2.SerializeToString()


def main():
    a = sys.argv[1:]
    per_tensor = '--per-tensor' in a
    a = [x for x in a if x != '--per-tensor']
    mode = 'one'
    if '--mode' in a:
        k = a.index('--mode')
        mode = a[k + 1]
        del a[k:k + 2]
    net, model, inputs, names = a[0], a[1], a[2], a[3].split(',')
    m = onnx.load(model)
    ws = weights(m)
    xs = [np.load(os.path.join(inputs, n + parity_q.SUFFIX[net]))['x'].astype(np.float32) for n in names]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s0 = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
    ref = [s0.run(None, {s0.get_inputs()[0].name: x})[0][0] for x in xs]
    fq = {w: quantize.fake_quant(arr, ax, per_tensor) for w, _, arr, ax, _ in ws}

    def score(repl):
        s = ort.InferenceSession(build(m, repl), so, providers=['CPUExecutionProvider'])
        rs = [parity_q.diff(net, r, s.run(None, {s.get_inputs()[0].name: x})[0][0]) for r, x in zip(ref, xs)]
        if net == 'pf':
            return f'1-IoU {max(1 - r["iou"] for r in rs):.5f}  |dp| max {max(r["max"] for r in rs):.4f}'
        return f'p99 {max(r["p99"] for r in rs):.4f}  max {max(r["max"] for r in rs):.4f} {rs[0]["unit"]}'
    print(f'{net} {model}: {len(ws)} weights >= 4096 values, {"per-tensor" if per_tensor else "per-channel"}, mode {mode}', flush=True)
    print(f'  all int8: {score(fq)}', flush=True)
    if mode == 'greedy':
        # keep the k most sensitive weights fp16 (k = 0..KMAX), sensitivity = its own effect (mode one)
        def num(r):
            return float(r.split()[1]) if net != 'pf' else float(r.split()[1])
        one = sorted(((num(score({w: fq[w]})), w, arr.size, node) for w, src, arr, ax, node in ws), reverse=True)
        total = sum(arr.size for _, _, arr, _, _ in ws)
        kept = 0
        for k in range(int(os.environ.get('KMAX', '8')) + 1):
            keep = {w for _, w, _, _ in one[:k]}
            kept = sum(n for _, w, n, _ in one[:k])
            last = one[k - 1] if k else None
            print(f'  keep {k} fp16 ({kept} of {total} values; +{kept / 1e6:.2f} MB over all-int8)'
                  f'{"  + " + last[3][:40] + f" (alone {last[0]:.4f})" if last else ""}: '
                  f'{score({w2: v for w2, v in fq.items() if w2 not in keep})}', flush=True)
        return 0
    for w, src, arr, ax, node in ws:
        repl = {w: fq[w]} if mode == 'one' else {k: v for k, v in fq.items() if k != w}
        rel = np.abs(fq[w] - arr).max() / max(1e-12, np.abs(arr).max())
        print(f'  {node[:40]:40s} {str(list(arr.shape)):20s} {arr.size:8d}  {score(repl)}  (max rel err {rel:.4f})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
