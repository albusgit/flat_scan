"""Original vs static nets on the 35 photos' real net inputs (docs/accel/PLAN.md, Stream A step 3 / Gate A).

  python tools/accel/compare_outputs.py INPUTS_DIR [STATIC_DIR] [--log FILE] [--dump DIR] [NAME ...]
      INPUTS_DIR: tools/accel/net_inputs.py's output (NAME_p1_in / _rc0_in / _rc2_in / _shade_in.npz)
      STATIC_DIR default app/src/main/assets/models (uvdoc_static, syntheticdoc_hr_static, shadenet_512x<W>)
      NAME: only these photos (default: every photo in INPUTS_DIR)
      --dump DIR: write the fp16 proxy's outputs as DIR/NAME_{p1,rc0,rc2,shade}_fp16.npz (key `out`, the
      format of net_inputs.py's NAME_rc2.npz), so pipeline_cli can run on them (FS_UVOUT / FS_UVOUT2)
Run it with the app's ONNX Runtime (1.22.0, C:/Users/User/Android/accel_venv) and onnxconverter-common.
Exit status 1 when a gate model's static copy is not bit-identical.

Per photo and net input, all on the ORT CPU execution provider:
  fp32: the original model vs its static copy, same session options. Gate A wants bit-identical (`same`)
        for uvdoc and syntheticdoc_hr. ShadeNet's fixed-shape model is for accelerators only (the CPU keeps
        shadenet.onnx): with constant sizes ORT's NCHWc pass turns its Resizes into nchwc Upsample, so at
        the default optimisation level it differs in the last bits (`diff`); it must then be bit-identical
        at ORT_ENABLE_BASIC (the same graph arithmetic).
  fp16: the static copy with fp16 compute as an fp16 proxy for the NPU: its ORT_ENABLE_BASIC graph (fp16
        weights' Casts folded, BatchNorm folded into Conv in fp32 as the QNN / NNAPI compilers do), then
        onnxconverter_common float16 (keep_io_types). The ORT CPU EP has few fp16
        kernels on x64, so it runs most ops in fp32 between fp16 tensors (casts it inserts itself): the
        proxy rounds every activation and weight to fp16, not the accumulation. Diff vs fp32 original.
Grid differences are in px of the analysis page (PipelineOptions::out_w = 1600, A4 portrait 1600 x 2263,
as the net sees the page): a grid point's move in normalised crop coords over the grid's own extent
(median width / height of the page in the grid) times the page size, Euclidean. Reported max and p99
over the grid points; the summary pools all points of all photos. ShadeNet: |shading| difference in
8-bit levels (x 255; the app divides the page by it). Photos without a static ShadeNet of their width
are listed as such (the app falls back to shadenet.onnx for them)."""
import glob
import os
import sys
import time
import warnings

import numpy as np
import onnx
import onnxruntime as ort

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS = os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models')
PAGE_W, PAGE_H = 1600, int(round(1600 * 297.0 / 210.0))
# (tag, input file suffix, original model, static model, fp32 bit-identity is a Gate A condition)
NETS = [('uv_p1', '_p1_in.npz', 'uvdoc.onnx', 'uvdoc_static.onnx', True),
        ('uv_rc0', '_rc0_in.npz', 'uvdoc.onnx', 'uvdoc_static.onnx', True),
        ('sdhr_rc2', '_rc2_in.npz', 'syntheticdoc_hr.onnx', 'syntheticdoc_hr_static.onnx', True),
        ('shade', '_shade_in.npz', 'shadenet.onnx', None, False)]


def fp16_model(path, tmp):
    """The fp16 proxy of a model: its ORT_ENABLE_BASIC fp32 graph (the fp16 weights' Casts folded, which is
    exact, and BatchNormalization folded into the Convs, as the NPU compilers do), then all fp16."""
    from onnxconverter_common import float16
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    o.optimized_model_filepath = tmp
    ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])
    m = onnx.load(tmp)
    os.remove(tmp)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # "float32 number ... truncated": tiny weights flushed, as fp16 would
        return float16.convert_float_to_float16(m, keep_io_types=True).SerializeToString()


_sess = {}


def run(key, src, x, basic=False):
    if (key, basic) not in _sess:
        o = ort.SessionOptions()
        if basic:
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _sess[key, basic] = ort.InferenceSession(src, o, providers=['CPUExecutionProvider'])
    s = _sess[key, basic]
    return s.run(None, {s.get_inputs()[0].name: x})[0][0]


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def grid_px(a, b):
    """Per grid point move between grids a, b (2 x gh x gw, normalised) in analysis-page px."""
    ext_u = np.median(a[0, :, -1] - a[0, :, 0]) / 2   # normalised [-1, 1] -> crop fraction
    ext_v = np.median(a[1, -1, :] - a[1, 0, :]) / 2
    du = (b[0] - a[0]).astype(np.float64) / 2 / ext_u * PAGE_W
    dv = (b[1] - a[1]).astype(np.float64) / 2 / ext_v * PAGE_H
    return np.hypot(du, dv).ravel()


def main():
    args = sys.argv[1:]
    opt = {}
    for flag in ('--log', '--dump'):
        if flag in args:
            k = args.index(flag)
            opt[flag] = args[k + 1]
            del args[k:k + 2]
    log_path, dump = opt.get('--log'), opt.get('--dump')
    if dump:
        os.makedirs(dump, exist_ok=True)
    inputs = args[0]
    tmp_dir = dump or inputs
    static_dir = args[1] if len(args) > 1 and os.path.isdir(args[1]) else MODELS
    names = [a for a in args[1:] if not os.path.isdir(a)] or sorted(
        os.path.basename(f)[:-len('_rc2_in.npz')] for f in glob.glob(os.path.join(inputs, '*_rc2_in.npz')))
    out = open(log_path, 'w') if log_path else None

    def say(s=''):
        print(s, flush=True)
        if out:
            out.write(s + '\n')

    say(f'compare_outputs.py: onnxruntime {ort.__version__} CPU EP, onnx {onnx.__version__}; static models {static_dir}')
    say(f'{len(names)} photos; grid diffs in px of the 1600 x {PAGE_H} analysis page (max / p99), shading in 8-bit levels')
    fp16 = {}
    pooled = {t: [] for t, *_ in NETS}
    not_same, no_static, accel_only = [], [], []
    runs = 0
    t0 = time.time()
    for n in names:
        cols = []
        for tag, suf, orig, stat, gate in NETS:
            f = os.path.join(inputs, n + suf)
            if not os.path.exists(f):
                cols.append(f'{tag} -')
                continue
            x = np.load(f)['x'].astype(np.float32)
            if stat is None:
                stat = f'shadenet_{x.shape[2]}x{x.shape[3]}.onnx'
            sp = os.path.join(static_dir, stat)
            if not os.path.exists(sp):
                no_static.append(f'{n} {tag} {stat}')
                cols.append(f'{tag} no {stat}')
                continue
            op = os.path.join(MODELS, orig)
            a = run(op, op, x)
            b = run(sp, sp, x)
            runs += 1
            eq = same(a, b)
            if not eq and gate:
                not_same.append(f'{n} {tag} max |diff| {float(np.abs(a - b).max()):.3g}')
            elif not eq:
                accel_only.append(float(np.abs(a - b).max()))
                if not same(run(op, op, x, True), run(sp, sp, x, True)):
                    not_same.append(f'{n} {tag} differs at ORT_ENABLE_BASIC too')
            if sp not in fp16:
                fp16[sp] = fp16_model(sp, os.path.join(tmp_dir, 'basic.onnx'))
            h = run(sp + '#fp16', fp16[sp], x)
            if dump:   # the fp16 proxy's output in net_inputs.py's format, for pipeline_cli FS_UVOUT runs
                np.savez(os.path.join(dump, f'{n}_{tag.split("_")[-1]}_fp16.npz'), out=h.astype(np.float32))
            d = np.abs(h.astype(np.float64) - a).ravel() * 255 if tag == 'shade' else grid_px(a, h)
            pooled[tag].append(d)
            cols.append(f'{tag} {"same" if eq else "DIFF" if gate else "diff"} fp16 {d.max():.3f}/{np.percentile(d, 99):.3f}')
        say(f'{n:36s} ' + '  '.join(cols))
    say()
    say(f'fp32 static vs original (bit-identical required): {"all identical" if not not_same else "NOT IDENTICAL"}'
        f' ({runs} runs compared)')
    for s in not_same:
        say('  ' + s)
    if accel_only:
        say(f'  shade (accelerators only, not a gate condition): {len(accel_only)} runs differ at the default '
            f'optimisation level, max |diff| {max(accel_only):.3g} ({max(accel_only) * 255:.2g} levels); '
            'bit-identical at ORT_ENABLE_BASIC unless listed above')
    for s in no_static:
        say('  no static model: ' + s)
    say('fp16 proxy vs fp32 original, pooled over all photos (target grid p99 < 0.25 px):')
    for tag, *_ in NETS:
        if pooled[tag]:
            d = np.concatenate(pooled[tag])
            per = [v.max() for v in pooled[tag]]
            unit = 'levels' if tag == 'shade' else 'px'
            say(f'  {tag:9s} p99 {np.percentile(d, 99):.4f} {unit}, max {d.max():.4f} {unit}, '
                f'worst photo max {max(per):.4f}, photos {len(per)}')
    say(f'{time.time() - t0:.0f} s')
    if out:
        out.close()
    return 1 if not_same else 0


if __name__ == '__main__':
    sys.exit(main())
