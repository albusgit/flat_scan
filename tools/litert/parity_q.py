"""Net-output parity of size variants on the 35 photos' real net inputs (Round 11, stream MODELSIZE).

  python tools/litert/parity_q.py INPUTS_DIR --ref NET=MODEL [--var LABEL:NET=MODEL ...] [--log FILE] [--names A,B]
      NET: uv_p1 | uv_rc0 | sdhr_rc2 | shade | pf  (input files NAME_p1_in / _rc0_in / _rc2_in / _shade_in / _pf_in.npz)
      MODEL: an .onnx (ONNX Runtime CPU EP) or a .tflite (tensorflow 2.19.1's tf.lite interpreter with its default XNNPACK
      delegate, the phone's CPU path; PARITY_LITERT=1: ai-edge-litert's, which on Windows has no XNNPACK);
      --ref: the reference model per net (normally the shipped fp16 file's ONNX source, as compare.py);
      --var LABEL:NET=MODEL: a variant, compared with the reference on the same inputs.
      Also: --pair LABEL:NET=MODEL_A,MODEL_B  compares two models directly (e.g. the `sim` ONNX vs its .tflite).
Metrics: grids in px of the 1600 x 2263 analysis page (tools/accel/compare_outputs.py grid_px; p99 / max), the
shading in 8-bit levels (x 255), the page finder on its 256 x 256 logits: page mask (sigmoid > 0.5) IoU with the
reference, gutter mask F1, and max |probability difference|. Also the LiteRT CPU time per invoke (median, ms).
Run with C:/Users/User/Android/litert_venv (onnxruntime 1.22.0, ai-edge-litert 2.1.2)."""
import glob
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'accel'))
from compare_outputs import grid_px  # noqa: E402

SUFFIX = {'uv_p1': '_p1_in.npz', 'uv_rc0': '_rc0_in.npz', 'sdhr_rc2': '_rc2_in.npz', 'shade': '_shade_in.npz',
          'pf': '_pf_in.npz'}
_cache = {}
THREADS = int(os.environ.get('PARITY_THREADS', '4'))


def run(path, x):
    """NCHW float32 in, NCHW float32 out (batch dropped), and the invoke time in ms."""
    if path not in _cache:
        if path.endswith('.onnx'):
            import onnxruntime as ort
            o = ort.SessionOptions()
            o.intra_op_num_threads = THREADS
            _cache[path] = ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])
        else:
            # tensorflow's interpreter: its Windows build applies the XNNPACK delegate (the phone's CPU path);
            # ai-edge-litert 2.1.2's Windows wheel runs the builtin kernels (10x slower here, same fp32 math)
            if os.environ.get('PARITY_LITERT'):
                from ai_edge_litert.interpreter import Interpreter
            else:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            it = Interpreter(model_path=path, num_threads=THREADS)
            it.allocate_tensors()
            _cache[path] = it
    m = _cache[path]
    t0 = time.perf_counter()
    if path.endswith('.onnx'):
        y = m.run(None, {m.get_inputs()[0].name: x})[0][0]
    else:
        m.set_tensor(m.get_input_details()[0]['index'], np.ascontiguousarray(x.transpose(0, 2, 3, 1)))
        m.invoke()
        y = m.get_tensor(m.get_output_details()[0]['index'])[0].transpose(2, 0, 1)
    return np.asarray(y, np.float32), 1000 * (time.perf_counter() - t0)


def sig(x):
    return 1 / (1 + np.exp(-x.astype(np.float64)))


def diff(net, a, b):
    """dict of metrics of b against a."""
    if net == 'shade':
        d = np.abs(b.astype(np.float64) - a).ravel() * 255
        return {'p99': float(np.percentile(d, 99)), 'max': float(d.max()), 'unit': 'levels'}
    if net == 'pf':
        pa, pb = sig(a), sig(b)
        ma, mb = pa[0] > 0.5, pb[0] > 0.5
        iou = (ma & mb).sum() / max(1, (ma | mb).sum())
        ga, gb = pa[1] > 0.5, pb[1] > 0.5
        f1 = 2 * (ga & gb).sum() / max(1, ga.sum() + gb.sum()) if (ga.sum() + gb.sum()) else 1.0
        return {'iou': float(iou), 'gutter_f1': float(f1), 'max': float(np.abs(pb - pa).max()),
                'p99': float(np.percentile(np.abs(pb - pa), 99)), 'unit': 'prob'}
    d = grid_px(a, b)
    return {'p99': float(np.percentile(d, 99)), 'max': float(d.max()), 'unit': 'px', '_d': d}


def parse(spec):
    label, rest = spec.split(':', 1) if ':' in spec else (None, spec)
    net, model = rest.split('=', 1)
    return label, net, model


def main():
    a = sys.argv[1:]
    inputs = a.pop(0)
    refs, vars_, pairs, log, names = {}, [], [], None, None
    while a:
        f = a.pop(0)
        if f == '--ref':
            _, net, m = parse(a.pop(0))
            refs[net] = m
        elif f == '--var':
            vars_.append(parse(a.pop(0)))
        elif f == '--pair':
            label, net, m = parse(a.pop(0))
            pairs.append((label, net, m.split(',')))
        elif f == '--log':
            log = a.pop(0)
        elif f == '--names':
            names = a.pop(0).split(',')
    out = open(log, 'w') if log else None

    def say(s=''):
        print(s, flush=True)
        if out:
            out.write(s + '\n')
            out.flush()
    if names is None:
        names = sorted(os.path.basename(p)[:-len('_p1_in.npz')] for p in glob.glob(os.path.join(inputs, '*_p1_in.npz')))
    jobs = [(lab, net, refs[net], m) for lab, net, m in vars_] + [(lab, net, ms[0], ms[1]) for lab, net, ms in pairs]
    say(f'parity_q: {len(names)} photos, {len(jobs)} comparisons, {THREADS} threads; inputs {inputs}')
    for lab, net, ra, mb in jobs:
        say(f'  {lab}: {net}  ref {ra}  vs  {mb}')
    pooled = {j: [] for j in range(len(jobs))}
    per = {j: [] for j in range(len(jobs))}
    times = {}
    t0 = time.time()
    for n in names:
        cols = []
        for j, (lab, net, ra, mb) in enumerate(jobs):
            f = os.path.join(inputs, n + SUFFIX[net])
            if not os.path.exists(f):
                continue
            x = np.load(f)['x'].astype(np.float32)
            ya, ta = run(ra, x)
            yb, tb = run(mb, x)
            times.setdefault(ra, []).append(ta)
            times.setdefault(mb, []).append(tb)
            r = diff(net, ya, yb)
            if '_d' in r:
                pooled[j].append(r.pop('_d'))
            per[j].append((n, r))
            if net == 'pf':
                cols.append(f'{lab} iou {r["iou"]:.4f} f1 {r["gutter_f1"]:.3f}')
            else:
                cols.append(f'{lab} {r["max"]:.4f}/{r["p99"]:.4f}')
        say(f'{n:36s} ' + '  '.join(cols))
    say()
    say('pooled over the photos (grid px: p99 of all grid points, max; shading levels: worst photo; page finder: worst photo):')
    for j, (lab, net, ra, mb) in enumerate(jobs):
        rs = [r for _, r in per[j]]
        if not rs:
            continue
        if net == 'pf':
            w = min(per[j], key=lambda t: t[1]['iou'])
            say(f'  {lab:22s} {net:8s} IoU min {w[1]["iou"]:.4f} ({w[0]}), mean {np.mean([r["iou"] for r in rs]):.4f}; '
                f'gutter F1 min {min(r["gutter_f1"] for r in rs):.3f}; |dp| max {max(r["max"] for r in rs):.4f}, '
                f'p99 max {max(r["p99"] for r in rs):.4f}; photos {len(rs)}')
        elif pooled[j]:
            d = np.concatenate(pooled[j])
            w = max(per[j], key=lambda t: t[1]['max'])
            say(f'  {lab:22s} {net:8s} p99 {np.percentile(d, 99):.4f} px, max {d.max():.4f} px ({w[0]}); '
                f'worst photo p99 {max(r["p99"] for r in rs):.4f}; photos {len(rs)}')
        else:
            w = max(per[j], key=lambda t: t[1]['max'])
            say(f'  {lab:22s} {net:8s} p99 max {max(r["p99"] for r in rs):.4f} levels, max {w[1]["max"]:.4f} ({w[0]}); photos {len(rs)}')
    say()
    say(f'CPU time per invoke ({THREADS} threads, median after the first; shared, loaded machine: indicative):')
    for m, ts in times.items():
        ts = ts[1:] if len(ts) > 1 else ts
        say(f'  {np.median(ts):8.1f} ms  (n {len(ts)})  {m}')
    say(f'{time.time() - t0:.0f} s')
    if out:
        out.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
