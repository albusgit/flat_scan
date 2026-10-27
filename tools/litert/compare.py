"""LiteRT models vs the CPU path's ONNX models on the 35 photos' real net inputs (contract C2 parity).

  python tools/litert/compare.py INPUTS_DIR [TFLITE_DIR] [--log FILE] [--seg DIR] [--pages NPZ] [NAME ...]
      INPUTS_DIR: tools/accel/net_inputs.py's output (NAME_p1_in / _rc0_in / _rc2_in / _shade_in.npz, NCHW)
      TFLITE_DIR default app/src/main/litert/models (tools/litert/convert.py)
      --seg DIR: the page finder's inputs NAME_seg_in.npz (tools/desktop_check pagefind_cli input PHOTO OUT.npz;
                 default: INPUTS_DIR); compared on pagefinder.tflite
      --pages NPZ: tools/litert/shade_inputs.py's real pages at 512 x W, W = 32..512; compared on shadenet_any.tflite
                 (the any-size ShadeNet, which convert.py --work DIR leaves in DIR: copy it to TFLITE_DIR) at every width;
                 the app's rewrite of shadenet.tflite gives its output bit for bit (tools/litert/check_resize.py)
      NAME: only these photos (default: every photo in INPUTS_DIR)
  python tools/litert/compare.py raw INPUTS_DIR RAW_DIR NAME ...
      writes RAW_DIR/NAME_TAG.in.f32 (the net input, float32 NHWC bytes) for tools/litert/gpu_check.sh
  python tools/litert/compare.py device INPUTS_DIR RAW_DIR [--log FILE]
      scores gpu_check's outputs RAW_DIR/NAME_TAG.MODE.out.f32 (MODE gpu or cpu) against ONNX Runtime the same way
Run it with C:/Users/User/Android/litert_venv (onnxruntime 1.22.0, the app's version, and ai-edge-litert).
Exit status 1 when a net is outside the tolerance.

Per photo and net input: ONNX Runtime CPU EP, default session options, on the model the CPU path runs
(pagefinder.onnx, uvdoc.onnx, syntheticdoc_hr.onnx, shadenet.onnx) vs the LiteRT interpreter (CPU, fp32, its default XNNPACK
kernels) on the .tflite, input transposed to NHWC and output back to NCHW. The desktop has no LiteRT GPU
delegate: this checks the conversion (graph, layouts, weights); the GPU's own fp32 arithmetic is a phone check.
Grid differences in px of the 1600 x 2263 analysis page (tools/accel/compare_outputs.py grid_px), shading in
8-bit levels (x 255), page finder logits as they are plus the share of pixels whose page / gutter decision
(logit > 0, the core's sigmoid > 0.5) flips. Tolerance: grid p99 <= 0.25 px and max <= 1 px; shading p99 <= 0.5
and max <= 2 levels; page finder logits p99 <= 0.01 and max <= 0.1, flips <= 0.01 % of the 2 x 256 x 256 decisions
(the logits are bicubic-upsampled before that threshold, so a flip at a 0-crossing moves an edge by a fraction of a
256-grid pixel)."""
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'accel'))
from compare_outputs import grid_px  # noqa: E402

ONNX_DIR = os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models')
TFLITE_DIR = os.path.join(REPO, 'app', 'src', 'main', 'litert', 'models')
# (tag, input file suffix, CPU path's ONNX model, .tflite)
NETS = [('uv_p1', '_p1_in.npz', 'uvdoc.onnx', 'uvdoc_static.tflite'),
        ('uv_rc0', '_rc0_in.npz', 'uvdoc.onnx', 'uvdoc_static.tflite'),
        ('sdhr_rc2', '_rc2_in.npz', 'syntheticdoc_hr.onnx', 'syntheticdoc_hr_static.tflite'),
        ('shade', '_shade_in.npz', 'shadenet.onnx', 'shadenet.tflite'),
        ('shade_any', '_shade_in.npz', 'shadenet.onnx', 'shadenet_any.tflite'),
        ('finder', '_seg_in.npz', 'pagefinder.onnx', 'pagefinder.tflite')]
TOL = {'px': (0.25, 1.0), 'levels': (0.5, 2.0), 'logits': (0.01, 0.1)}   # (p99, max)
FLIP_TOL = 1e-4   # share of page finder decisions that may flip
UNIT = {'shade': 'levels', 'shade_any': 'levels', 'finder': 'logits'}

_sess, _tfl = {}, {}


def run_onnx(path, x):
    if path not in _sess:
        _sess[path] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    s = _sess[path]
    return s.run(None, {s.get_inputs()[0].name: x})[0][0]


def run_tflite(path, x):
    if path not in _tfl:
        it = Interpreter(model_path=path)
        it.allocate_tensors()
        _tfl[path] = it
    it = _tfl[path]
    nhwc = [x.shape[0], x.shape[2], x.shape[3], x.shape[1]]
    i0 = it.get_input_details()[0]
    if list(i0['shape']) != nhwc:   # an any-size model (shadenet.tflite): resized as infer.cpp does
        it.resize_tensor_input(i0['index'], nhwc, strict=True)
        it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]['index'], np.ascontiguousarray(x.transpose(0, 2, 3, 1)))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]['index'])[0].transpose(2, 0, 1)


def raw(inputs, raw_dir, names):
    os.makedirs(raw_dir, exist_ok=True)
    for n in names:
        for tag, suf, *_ in NETS:
            x = np.load(os.path.join(inputs, n + suf))['x'].astype(np.float32)
            np.ascontiguousarray(x.transpose(0, 2, 3, 1)).tofile(os.path.join(raw_dir, f'{n}_{tag}.in.f32'))
    return 0


def device(inputs, raw_dir, say):
    """gpu_check outputs vs ONNX Runtime; returns the failures as (mode, what): a missing output (the run failed)
    or an output outside the tolerance."""
    bad = []
    runs = [os.path.basename(f)[:-len('.in.f32')] for f in glob.glob(os.path.join(raw_dir, '*.in.f32'))]
    for mode in ('gpu', 'cpu'):  # gpu_check's nets: uvdoc, sdhr, shade (512 x 352)
        outs = sorted(glob.glob(os.path.join(raw_dir, f'*.{mode}.out.f32')))
        missing = [r for r in runs if not os.path.exists(os.path.join(raw_dir, f'{r}.{mode}.out.f32'))]
        if missing:
            bad.append((mode, f'no output for {len(missing)} of {len(runs)} inputs'))
        say(f'{mode}: LiteRT on the device ({len(outs)} outputs of {len(runs)} inputs) vs ONNX Runtime on the desktop')
        for f in outs:
            stem = os.path.basename(f)[:-len(f'.{mode}.out.f32')]
            tag = next(t for t, *_ in NETS[:4] if stem.endswith('_' + t))
            n = stem[:-len(tag) - 1]
            _, suf, onx, _ = next(e for e in NETS if e[0] == tag)
            a = run_onnx(os.path.join(ONNX_DIR, onx), np.load(os.path.join(inputs, n + suf))['x'].astype(np.float32))
            b = np.fromfile(f, np.float32).reshape(a.shape[1], a.shape[2], a.shape[0]).transpose(2, 0, 1)
            d = np.abs(b.astype(np.float64) - a).ravel() * 255 if tag == 'shade' else grid_px(a, b)
            unit = UNIT.get(tag, 'px')
            ok = np.percentile(d, 99) <= TOL[unit][0] and d.max() <= TOL[unit][1]
            if not ok:
                bad.append((mode, f'{n} {tag} outside tolerance'))
            say(f'  {n:36s} {tag:9s} max {d.max():.5f} p99 {np.percentile(d, 99):.5f} {unit}: {"ok" if ok else "OUTSIDE"}')
    return bad


def main():
    args = sys.argv[1:]
    log_path = None
    if '--log' in args:
        k = args.index('--log')
        log_path = args[k + 1]
        del args[k:k + 2]
    seg_dir = pages = None
    if '--seg' in args:
        k = args.index('--seg')
        seg_dir = args[k + 1]
        del args[k:k + 2]
    if '--pages' in args:
        k = args.index('--pages')
        pages = args[k + 1]
        del args[k:k + 2]
    if args and args[0] == 'raw':
        return raw(args[1], args[2], args[3:])
    if args and args[0] == 'device':
        out = open(log_path, 'w') if log_path else None

        def say_d(s=''):
            print(s, flush=True)
            if out:
                out.write(s + '\n')
        bad = device(args[1], args[2], say_d)
        say_d('all outputs present and within tolerance' if not bad else
              'FAILED: ' + ', '.join(f'{m} {t}' for m, t in bad))
        if out:
            out.close()
        return 1 if bad else 0
    inputs = args[0]
    tfl_dir = args[1] if len(args) > 1 and os.path.isdir(args[1]) else TFLITE_DIR
    names = [a for a in args[1:] if not os.path.isdir(a)] or sorted(
        os.path.basename(f)[:-len('_rc2_in.npz')] for f in glob.glob(os.path.join(inputs, '*_rc2_in.npz')))
    out = open(log_path, 'w') if log_path else None

    def say(s=''):
        print(s, flush=True)
        if out:
            out.write(s + '\n')

    say(f'compare.py: onnxruntime {ort.__version__} CPU EP vs LiteRT interpreter (CPU, fp32); models {tfl_dir}')
    say(f'{len(names)} photos; grid diffs in px of the 1600 x 2263 analysis page (max / p99), shading in 8-bit levels')
    pooled = {t: [] for t, *_ in NETS}
    flips = [0, 0]   # page finder decisions flipped, compared
    t0 = time.time()
    for n in names:
        cols = []
        for tag, suf, onx, tfl in NETS:
            f = os.path.join(seg_dir or inputs if tag == 'finder' else inputs, n + suf)
            if not os.path.exists(f) or not os.path.exists(os.path.join(tfl_dir, tfl)):
                cols.append(f'{tag} -')
                continue
            x = np.load(f)['x'].astype(np.float32)
            a = run_onnx(os.path.join(ONNX_DIR, onx), x)
            b = run_tflite(os.path.join(tfl_dir, tfl), x)
            if a.shape != b.shape:
                raise SystemExit(f'{n} {tag}: ONNX {a.shape} vs LiteRT {b.shape}')
            unit = UNIT.get(tag, 'px')
            d = (np.abs(b.astype(np.float64) - a).ravel() * (255 if unit == 'levels' else 1) if unit != 'px'
                 else grid_px(a, b))
            pooled[tag].append(d)
            extra = ''
            if tag == 'finder':
                fl = int(np.count_nonzero((a > 0) != (b > 0)))
                flips[0] += fl
                flips[1] += a.size
                extra = f' flips {fl}'
            cols.append(f'{tag} {d.max():.5f}/{np.percentile(d, 99):.5f}{extra}')
        say(f'{n:36s} ' + '  '.join(cols))
    by_width = {}
    if pages:
        # shadenet.tflite at every width on real pages (shade_inputs.py)
        z = np.load(pages)
        pn = sorted({k.split('@')[0] for k in z.files})
        say()
        say(f'shadenet.tflite (any size) vs shadenet.onnx on {len(pn)} real pages per width (max / p99 levels):')
        for w in sorted({int(k.split('@')[1]) for k in z.files}):
            ds = []
            for n in pn:
                x = np.ascontiguousarray((z[f'{n}@{w}'].astype(np.float32) * np.float32(1 / 255)).transpose(2, 0, 1)[None])
                a = run_onnx(os.path.join(ONNX_DIR, 'shadenet.onnx'), x)
                b = run_tflite(os.path.join(tfl_dir, 'shadenet.tflite'), x)
                if a.shape != b.shape:
                    raise SystemExit(f'{n} width {w}: ONNX {a.shape} vs LiteRT {b.shape}')
                ds.append(np.abs(b.astype(np.float64) - a).ravel() * 255)
            d = np.concatenate(ds)
            by_width[w] = d
            pooled['shade_any'].append(d)
            say(f'  512 x {w:3d}: max {d.max():.5f} p99 {np.percentile(d, 99):.5f}')
    say()
    say('LiteRT vs ONNX Runtime, pooled over all photos:')
    bad = []
    for tag, *_ in NETS:
        if not pooled[tag]:
            continue
        d = np.concatenate(pooled[tag])
        unit = UNIT.get(tag, 'px')
        p99, mx = np.percentile(d, 99), d.max()
        ok = p99 <= TOL[unit][0] and mx <= TOL[unit][1]
        extra = ''
        if tag == 'finder':
            share = flips[0] / max(flips[1], 1)
            ok = ok and share <= FLIP_TOL
            extra = f', decisions flipped {flips[0]} of {flips[1]} ({share * 100:.4f} %, tolerance {FLIP_TOL * 100:g} %)'
        if not ok:
            bad.append(tag)
        what = 'photos' if tag != 'shade_any' or not by_width else 'inputs (35 photos at 512 x 352 + the pages at each width)'
        say(f'  {tag:9s} p99 {p99:.5f} {unit}, max {mx:.5f} {unit} (tolerance p99 <= {TOL[unit][0]}, max <= '
            f'{TOL[unit][1]}){extra}: {"ok" if ok else "OUTSIDE"}, {what} {len(pooled[tag])}')
    say(f'{"all nets within tolerance" if not bad else "OUTSIDE TOLERANCE: " + ", ".join(bad)}; {time.time() - t0:.0f} s')
    if out:
        out.close()
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
