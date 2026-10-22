"""The shipped .tflite of a net from its ONNX file, with its parity report (Round 11, stream MODELSIZE).

  python tools/litert/ship.py NET SRC.onnx OUT_DIR [--weights int8|fp16] [--qmap Q.npz | --gptq] [--keep NODE,...]
                              [--inputs DIR --names A,B,...] [--work DIR]
      NET: uvdoc | sdhr | shade | pagefinder
      SRC.onnx: the net as the CPU path runs it (dynamic or static shapes, fp16 + Cast or fp32 weights)
      OUT_DIR gets: FILE.tflite (the shipped file: uvdoc_static / syntheticdoc_hr_static / shadenet_512x352 /
        pagefinder .tflite), FILE.fp16.tflite (the fp16-weight file before int8), FILE.sim.onnx (the fp32 ONNX of the
        int8 weights: what the .tflite computes with, for pipeline_cli FS_MODELS), FILE.report.txt (sizes, ops,
        checks, parity), FILE.qmap.npz (with --gptq).
      --weights int8 (default): int8 weights (quantize.py form dq), fp16 for the --keep layers (default per NET,
        DEFAULT_KEEP: the output heads and the bridge 1x1 conv of UVDoc / SD-HR, measured most sensitive) and for
        weights under 4096 values; fp16: the fp16-weight file only.
      --gptq: int8 values from gptq.py calibrated on --inputs/--names (needs the round4 venv's torch: run gptq.py
        yourself and pass --qmap; see docs/results/r11/modelsize.md); default: round to nearest.
      --inputs DIR --names: the 35 photos' net inputs (tools/accel/net_inputs.py, + pagefind_cli input for
        NAME_pf_in.npz) for the parity report (else no parity).
Run with C:/Users/User/Android/litert_venv (tensorflow 2.19.1, onnx2tf 1.28.8, onnxruntime 1.22.0).

Steps: (1) static ONNX (tools/accel/prepare_models.make_static at the NET's input size; outputs the app does not
read are dropped, e.g. SD-HR's `shading`); (2) an fp32-weight source (the page finder) gets its Conv weights rounded
to fp16 (the fp16 file's numbers; measured in the report); (3) tools/litert/convert.py's conversion (fp32 graph,
onnx2tf, rewrite: fp16 weights, GPU forms) and checks (I/O; GPU op forms for the GPU nets); (4) int8 filters
(quantize.tflite_int8, matched to the ONNX weights by value) and the sim ONNX; (5) parity: ONNX Runtime on SRC vs
the LiteRT interpreter (tensorflow's tf.lite, XNNPACK) on FILE.fp16.tflite and FILE.tflite, and sim vs FILE.tflite
(should be ~1e-3 px: fp32 arithmetic only)."""
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'tools', 'accel'))
import convert  # noqa: E402
import quantize  # noqa: E402

# NET: (file stem, NCHW input, NHWC input, NHWC output, parity tags, GPU net)
NETS = {'uvdoc': ('uvdoc_static', [1, 3, 712, 488], [1, 712, 488, 3], [1, 45, 31, 2], ['uv_p1', 'uv_rc0'], True),
        'sdhr': ('syntheticdoc_hr_static', [1, 3, 1440, 1024], [1, 1440, 1024, 3], [1, 121, 177, 2], ['sdhr_rc2'], True),
        'shade': ('shadenet_512x352', [1, 3, 512, 352], [1, 512, 352, 3], [1, 512, 352, 3], ['shade'], True),
        'pagefinder': ('pagefinder', [1, 3, 256, 256], [1, 256, 256, 3], [1, 256, 256, 2], ['pf'], False)}
DEFAULT_KEEP = {'uvdoc': ['out_point_positions2D', 'bridge_concat'], 'sdhr': ['bm_head.0', 'bridge_concat'],
                'shade': [], 'pagefinder': []}


def static_onnx(src, nchw, dst):
    import prepare_models
    m = onnx.load(src)
    name = [i.name for i in m.graph.input if i.name not in {x.name for x in m.graph.initializer}][0]
    prepare_models.make_static(m, {name: nchw}, [m.graph.output[0].name], log=lambda *a: None)
    onnx.save(m, dst)
    return dst


def round_fp32_weights(path, dst):
    """Conv weights stored fp32 (not fp16 numbers) rounded to fp16 values (kept fp32 storage); returns the count."""
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    n = 0
    for node in m.graph.node:
        if node.op_type != 'Conv' or node.input[1] not in inits:
            continue
        i = inits[node.input[1]]
        a = numpy_helper.to_array(i)
        if a.dtype == np.float32 and not np.array_equal(a.astype(np.float16).astype(np.float32), a):
            i.CopyFrom(numpy_helper.from_array(a.astype(np.float16).astype(np.float32), i.name))
            n += 1
    onnx.save(m, dst)
    return n


def to_tflite(net, onnx_static, dst, work, say):
    stem, nchw, nhwc, out_shape, tags, gpu = NETS[net]
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    import onnx2tf
    m, w16 = convert.fp32_onnx(onnx_static)
    f32 = os.path.join(work, net + '_fp32.onnx')
    onnx.save(m, f32)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        np.save(convert.SAMPLE, np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
        onnx2tf.convert(input_onnx_file_path=f32, output_folder_path=os.path.join(work, net),
                        copy_onnx_input_output_names_to_tflite=True, non_verbose=True)
    finally:
        os.chdir(cwd)
    tfl32 = os.path.join(work, net, net + '_fp32_float32.tflite')
    summary = convert.rewrite(tfl32, dst, convert.bn_convs(m, w16), m.graph.output[0].name,
                              'unfold' if gpu else 'round')
    say(f'  fp16 file: {os.path.getsize(dst)} bytes; {summary}')
    if gpu:
        lines, _ = convert.check(dst, nhwc, out_shape)
        for s in lines:
            say('    ' + s)
    else:
        info = quantize.tflite_info(dst)
        ok = info['inputs'][0][1] == nhwc and info['outputs'][0][1] == out_shape
        say(f'    in {info["inputs"]}, out {info["outputs"]}{"" if ok else "  NOT THE EXPECTED SHAPES"}')
        say(f'    ops {info["ops"]}')
        say('    CPU net (the page finder always runs on the CPU): no GPU op check')
        if not ok:
            raise SystemExit('I/O shapes')


def parity(net, src, files, inputs, names, say):
    import parity_q
    stem, nchw, nhwc, out_shape, tags, gpu = NETS[net]
    for tag in tags:
        say(f'  parity {tag} ({len(names)} photos), reference ONNX Runtime on {os.path.basename(src)}:')
        pooled = {k: [] for k in files}
        per = {k: [] for k in files}
        for n in names:
            f = os.path.join(inputs, n + parity_q.SUFFIX[tag])
            if not os.path.exists(f):
                continue
            x = np.load(f)['x'].astype(np.float32)
            ref, _ = parity_q.run(src, x)
            outs = {k: parity_q.run(p, x)[0] for k, p in files.items()}
            for k, y in outs.items():
                base = outs['int8'] if k == 'sim' else ref   # sim is checked against the int8 file
                r = parity_q.diff(tag, base, y)
                if '_d' in r:
                    pooled[k].append(r.pop('_d'))
                per[k].append((n, r))
        for k in files:
            rs = per[k]
            if not rs:
                continue
            what = {'sim': 'sim ONNX vs int8 file', 'int8_xnnpack': 'int8 file on XNNPACK vs ref'}.get(k, f'{k} file vs reference')
            if tag == 'pf':
                w = min(rs, key=lambda t: t[1]['iou'])
                say(f'    {what:28s} mask IoU min {w[1]["iou"]:.4f} ({w[0]}), gutter F1 min '
                    f'{min(r["gutter_f1"] for _, r in rs):.3f}, |dp| max {max(r["max"] for _, r in rs):.4f}')
            elif pooled[k]:
                d = np.concatenate(pooled[k])
                w = max(rs, key=lambda t: t[1]['max'])
                say(f'    {what:28s} p99 {np.percentile(d, 99):.4f} px, max {d.max():.4f} px ({w[0]})')
            else:
                w = max(rs, key=lambda t: t[1]['max'])
                say(f'    {what:28s} p99 (worst photo) {max(r["p99"] for _, r in rs):.4f} levels, max {w[1]["max"]:.4f} ({w[0]})')


def main():
    a = sys.argv[1:]
    net, src, out = a[0], os.path.abspath(a[1]), os.path.abspath(a[2])
    opts = {'--weights': 'int8', '--qmap': None, '--keep': None, '--inputs': None, '--names': None, '--work': None}
    rest = a[3:]
    while rest:
        k = rest.pop(0)
        opts[k] = rest.pop(0)
    stem, nchw, nhwc, out_shape, tags, gpu = NETS[net]
    keep = DEFAULT_KEEP[net] if opts['--keep'] is None else [k for k in opts['--keep'].split(',') if k]
    os.makedirs(out, exist_ok=True)
    work = opts['--work'] or tempfile.mkdtemp(prefix='ship_')
    os.makedirs(work, exist_ok=True)
    rep = open(os.path.join(out, stem + '.report.txt'), 'w')

    def say(s=''):
        print(s, flush=True)
        rep.write(s + '\n')
        rep.flush()
    say(f'ship.py {net} {src} -> {out} (weights {opts["--weights"]}, keep fp16 {keep}, qmap {opts["--qmap"]})')
    st = static_onnx(src, nchw, os.path.join(work, net + '_static.onnx'))
    n16 = round_fp32_weights(st, st)
    say(f'  static ONNX {os.path.getsize(st)} bytes; fp32 Conv weights rounded to fp16: {n16}')
    fp16 = os.path.join(out, stem + '.fp16.tflite')
    to_tflite(net, st, fp16, work, say)
    files = {'fp16': fp16}
    final = os.path.join(out, stem + '.tflite')
    if opts['--weights'] == 'int8':
        qm = quantize.load_qmap(opts['--qmap'])
        r = quantize.tflite_int8(fp16, final, keep=tuple(keep), onnx_src=st, qmap=qm, strict=gpu, no_fp16=True)
        if r['unmatched']:
            say(f'  {len(r["unmatched"])} filters changed by onnx2tf (no ONNX weight with their values): rounded to nearest, '
                'not in the sim ONNX (the parity below measures the .tflite itself)')
        say(f'  int8 file: {r["bytes"]} bytes; {r["quantized"]} filters int8 ({r["quantized_values"]} values, '
            f'{r["from_qmap"]} from the qmap), {r["kept_fp16"]} fp16 ({r["kept_values"]} values: {r["kept"]})')
        # the sim ONNX of the int8 weights, on the CPU path's (dynamic) source graph
        sim = os.path.join(out, stem + '.sim.onnx')
        if n16:
            round_fp32_weights(src, sim)
            s = quantize.onnx_sim(sim, sim, keep=tuple(keep), qmap=qm)
        else:
            s = quantize.onnx_sim(src, sim, keep=tuple(keep), qmap=qm)
        say(f'  sim ONNX: {s}')
        files.update(int8=final, sim=sim)
    else:
        shutil.copy(fp16, final)
    info = quantize.tflite_info(final)
    say(f'  shipped {os.path.basename(final)}: {info["bytes"]} bytes, constants by type {info["values"]}')
    if opts['--inputs'] and opts['--names']:
        if 'sim' in files:
            # sim vs int8: reorder so int8 runs first
            files = {'fp16': files['fp16'], 'int8': files['int8'], 'int8_xnnpack': files['int8'] + '#xnn', 'sim': files['sim']}
        parity(net, src, files, opts['--inputs'], opts['--names'].split(','), say)
    rep.close()
    if not opts['--work']:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
