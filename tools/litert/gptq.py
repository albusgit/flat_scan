"""Int8 weights chosen to keep each conv's OUTPUT, not just its weights, close to the fp16 net (GPTQ, Frantar et
al. 2022, arXiv:2210.17323), calibrated on real photos' net inputs (Round 11, stream MODELSIZE).

  python tools/litert/gptq.py SRC.onnx OUT_PREFIX --net NET --inputs DIR --names A,B,... [--samples N] [--damp F]
                              [--keep NODE,...] [--min N] [--plain]
      SRC.onnx: the CPU path's ONNX model (fp16 weights + Cast, or fp32); NET as parity_q.py (picks DIR/NAME + suffix);
      writes OUT_PREFIX.npz (per Conv weight input name: q int8 in the ONNX layout, scale float32 per output channel;
      the .tflite step, quantize.py tflite --qmap, stores exactly these) and OUT_PREFIX_sim.onnx (fp32 initializers
      q * scale: what the int8 .tflite computes with, for pipeline_cli / ONNX Runtime).
      --keep: Conv node names (substrings) whose weights stay fp16; --min: weights with fewer values stay (4096);
      --plain: round to nearest (quantize.py's rule) instead of GPTQ (for comparison).
Per conv with weight W [O, C/g, KH, KW] (groups 1 only; grouped / depthwise convs are rounded to nearest):
  H = sum over sampled output positions of x x^T, x the conv's input patch (C*KH*KW, the ONNX OIHW order) taken from
  the fp16 net's activations on the calibration inputs (ONNX Runtime), N positions per input (default 8192);
  the scale per output channel is fixed first (max|W| / 127, as quantize.py); then the columns are quantized one by
  one in order of decreasing H diagonal, each column's rounding error spread over the columns not yet quantized
  through the inverse Hessian (Cholesky form), damping F * mean(diag H) (default 0.01). Convs are independent (the
  inputs are the fp16 net's, not the partly quantized one's)."""
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize  # noqa: E402

ort.set_default_logger_severity(3)
SUFFIX = {'uv_p1': ['_p1_in.npz'], 'uv_rc0': ['_rc0_in.npz'], 'uv': ['_p1_in.npz', '_rc0_in.npz'],
          'sdhr_rc2': ['_rc2_in.npz'], 'shade': ['_shade_in.npz'], 'pf': ['_pf_in.npz']}


def attr(n, name, default):
    for a in n.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else (a.i if a.type == onnx.AttributeProto.INT else a.s.decode())
    return default


def conv_weights(m, min_size, keep):
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
        a = numpy_helper.to_array(inits[src]).astype(np.float32)
        if a.size < min_size or any(k in n.name for k in keep):
            continue
        out.append(dict(node=n, w=w, W=a, group=attr(n, 'group', 1), pads=attr(n, 'pads', [0, 0, 0, 0]),
                        strides=attr(n, 'strides', [1, 1]), dil=attr(n, 'dilations', [1, 1]), x=n.input[0]))
    return out


def patches(x, c, rng, S):
    """[C*KH*KW, S] patches of x [C, H, W] at S random output positions of conv c (zero padding as the Conv)."""
    O, Cg, KH, KW = c['W'].shape
    pt, pl, pb, pr = c['pads']
    sh, sw = c['strides']
    dh, dw = c['dil']
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))
    Ho = (xp.shape[1] - dh * (KH - 1) - 1) // sh + 1
    Wo = (xp.shape[2] - dw * (KW - 1) - 1) // sw + 1
    n = Ho * Wo
    idx = rng.choice(n, size=min(S, n), replace=False)
    yy, xx = idx // Wo * sh, idx % Wo * sw
    P = np.empty((x.shape[0], KH, KW, len(idx)), np.float32)
    for kh in range(KH):
        for kw in range(KW):
            P[:, kh, kw, :] = xp[:, yy + kh * dh, xx + kw * dw]
    return P.reshape(-1, len(idx))


def gptq(W, H, damp):
    """W [O, K] float32, H [K, K]: (q int8 [O, K], scale [O]) with quantize.py's per-row scale."""
    O, K = W.shape
    q0, scale = quantize.qparams(W, 0)
    Wt = torch.tensor(W, dtype=torch.float64)
    Ht = torch.tensor(H, dtype=torch.float64)
    dead = torch.diag(Ht) == 0
    Ht[dead, dead] = 1
    Wt[:, dead] = 0
    perm = torch.argsort(torch.diag(Ht), descending=True)
    Wt, Ht = Wt[:, perm], Ht[perm][:, perm]
    Ht += damp * torch.mean(torch.diag(Ht)) * torch.eye(K, dtype=torch.float64)
    L = torch.linalg.cholesky(Ht)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    s = torch.tensor(scale, dtype=torch.float64)[:, None]
    Q = torch.zeros_like(Wt)
    B = 128
    for i0 in range(0, K, B):
        i1 = min(K, i0 + B)
        W1 = Wt[:, i0:i1].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i0:i1, i0:i1]
        for j in range(i1 - i0):
            w = W1[:, j]
            q = torch.clamp(torch.round(w / s[:, 0]), -127, 127)
            Q[:, i0 + j] = q
            err = (w - q * s[:, 0]) / U1[j, j]
            W1[:, j:] -= err[:, None] * U1[j, j:][None, :]
            E1[:, j] = err
        Wt[:, i1:] -= E1 @ U[i0:i1, i1:]
    inv = torch.argsort(perm)
    q = Q[:, inv].numpy().astype(np.int8)
    return q, scale


def main():
    a = sys.argv[1:]
    src, prefix = a[0], a[1]
    opts = {'--samples': '8192', '--damp': '0.01', '--keep': '', '--min': '4096', '--net': None, '--inputs': None, '--names': None}
    plain = '--plain' in a
    rest = [x for x in a[2:] if x != '--plain']
    while rest:
        k = rest.pop(0)
        opts[k] = rest.pop(0)
    S, damp, min_size = int(opts['--samples']), float(opts['--damp']), int(opts['--min'])
    keep = [k for k in opts['--keep'].split(',') if k]
    names = opts['--names'].split(',')
    m = onnx.load(src)
    convs = conv_weights(m, min_size, keep)
    print(f'{src}: {len(convs)} conv weights to int8 ({"round to nearest" if plain else "GPTQ"}), '
          f'{len(names)} calibration photos x {SUFFIX[opts["--net"]]}', flush=True)
    t0 = time.time()
    H = {}
    if not plain:
        # the conv inputs as extra graph outputs
        m2 = onnx.ModelProto()
        m2.CopyFrom(m)
        have = {o.name for o in m2.graph.output}
        xs = []
        for c in convs:
            if c['x'] not in have and c['x'] not in xs:
                xs.append(c['x'])
        for x in xs:
            m2.graph.output.append(onnx.helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None))
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get('GPTQ_THREADS', '4'))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(m2.SerializeToString(), so, providers=['CPUExecutionProvider'])
        outs = [o.name for o in sess.get_outputs()]
        rng = np.random.default_rng(0)
        n_in = 0
        for n in names:
            for suf in SUFFIX[opts['--net']]:
                f = os.path.join(opts['--inputs'], n + suf)
                if not os.path.exists(f):
                    continue
                x = np.load(f)['x'].astype(np.float32)
                ys = dict(zip(outs, sess.run(outs, {sess.get_inputs()[0].name: x})))
                for c in convs:
                    if c['group'] != 1:
                        continue
                    P = torch.from_numpy(patches(ys[c['x']][0], c, rng, S))
                    h = (P @ P.T).numpy().astype(np.float64)
                    H[c['w']] = H.get(c['w'], 0) + h
                del ys
                n_in += 1
                print(f'  {n}{suf}: {time.time() - t0:.0f} s', flush=True)
        print(f'calibration: {n_in} inputs, {time.time() - t0:.0f} s', flush=True)
    qmap = {}
    for c in convs:
        W = c['W']
        O = W.shape[0]
        if plain or c['group'] != 1 or c['w'] not in H:
            q, s = quantize.qparams(W, 0)
        else:
            q, s = gptq(W.reshape(O, -1), H[c['w']], damp)
            q = q.reshape(W.shape)
        qmap[c['w']] = (q, s)
        dq = quantize.dequant(q, s, 0)
        print(f'  {c["node"].name[:48]:48s} {str(list(W.shape)):18s} max |w - w\'| / max|w| {np.abs(dq - W).max() / np.abs(W).max():.4f}', flush=True)
    np.savez(prefix + '.npz', **{k + '::q': v[0] for k, v in qmap.items()}, **{k + '::s': v[1] for k, v in qmap.items()})
    # the sim ONNX
    inits = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    drop = set()
    for w, (q, s) in qmap.items():
        dq = quantize.dequant(q, s, 0)
        if w in inits:
            inits[w].CopyFrom(numpy_helper.from_array(dq, w))
        else:
            drop.add(id(prod[w]))
            m.graph.initializer.append(numpy_helper.from_array(dq, w))
    keepn = [n for n in m.graph.node if id(n) not in drop]
    del m.graph.node[:]
    m.graph.node.extend(keepn)
    used = {i for n in m.graph.node for i in n.input}
    live = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(live)
    onnx.save(m, prefix + '_sim.onnx')
    print(f'wrote {prefix}.npz, {prefix}_sim.onnx; {time.time() - t0:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
