"""The app's in-memory size rewrite of a fixed-shape .tflite (core/src/tflite_shape.cpp) against the converter's files.

  python -I tools/litert/check_resize.py CLI STATIC.tflite REF.tflite [--sizes H,W ...] [--work DIR]
      CLI: tools/desktop_check/build/tflite_shape_cli(.exe); STATIC: the shipped fixed-shape file (rulenet.tflite made
      at 1152 x 800, shadenet.tflite at 512 x 352); REF: the any-size file of the same net (convert.py any_hw), or a
      fixed-shape file of one size (then only that size is checked)
For each size: STATIC rewritten by CLI must load, run at that size without a resize call and give REF's output (REF
resized to that size) bit for bit on a random input, with the built-in kernels alone and with XNNPACK applied when the
interpreter is made (before AllocateTensors re-derives the shapes the file still holds at STATIC's size, as in the
app); STATIC resized at run time without the rewrite is shown for contrast. Exit code 1 on any mismatch.
Run it with C:/Users/User/Android/litert_venv (ai_edge_litert)."""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

ap = argparse.ArgumentParser()
ap.add_argument('cli')
ap.add_argument('static')
ap.add_argument('ref')
ap.add_argument('--sizes', nargs='*', default=None, help='H,W pairs (default: the rule net sizes of convert.py)')
ap.add_argument('--work', default=None)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(__file__))
from convert import RULE_SIZES  # noqa: E402

sizes = [tuple(int(v) for v in s.split(',')) for s in a.sizes] if a.sizes else RULE_SIZES
work = a.work or tempfile.mkdtemp(prefix='check_resize_')
os.makedirs(work, exist_ok=True)
MODES = {'builtin': OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES, 'xnnpack': OpResolverType.AUTO}
r_in = Interpreter(model_path=a.ref).get_input_details()[0]
dynamic = -1 in list(r_in['shape_signature'])
if not dynamic:
    sizes = [(int(r_in['shape'][1]), int(r_in['shape'][2]))]
rng = np.random.default_rng(7)
bad = 0


def run(path, mode, x, resize=False):
    it = Interpreter(model_path=path, experimental_op_resolver_type=MODES[mode])
    i0 = it.get_input_details()[0]
    if resize:
        it.resize_tensor_input(i0['index'], list(x.shape), strict=False)
    it.allocate_tensors()
    if list(it.get_input_details()[0]['shape']) != list(x.shape):
        raise RuntimeError('input %s' % list(it.get_input_details()[0]['shape']))
    it.set_tensor(i0['index'], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]['index'])


for h, w in sizes:
    out = os.path.join(work, 'patched_%dx%d.tflite' % (h, w))
    p = subprocess.run([a.cli, a.static, out, str(h), str(w)], capture_output=True, text=True)
    if p.returncode:
        print(f'{h} x {w}: rewrite failed: {p.stderr.strip()}')
        bad += 1
        continue
    x = np.ascontiguousarray(rng.random((1, h, w, 3), dtype=np.float32))
    notes = []
    for mode in MODES:
        try:
            y = run(out, mode, x)
            yr = run(a.ref, mode, x, resize=dynamic)
            same = y.shape == yr.shape and np.array_equal(y, yr)
            d = float(np.abs(y - yr).max()) if y.shape == yr.shape else float('nan')
            notes.append(f'{mode} {"bit for bit" if same else "MISMATCH max %.3g" % d}')
        except Exception as e:  # noqa: BLE001
            same = False
            notes.append(f'{mode} FAILED ({str(e).splitlines()[0][:80]})')
        bad += not same
    try:
        yp = run(a.static, 'builtin', x, resize=True)
        yr = run(a.ref, 'builtin', x, resize=dynamic)
        plain = 'same' if yp.shape == yr.shape and np.array_equal(yp, yr) else 'differs'
    except Exception as e:  # noqa: BLE001
        plain = 'fails'
    print(f'{h} x {w}: rewritten vs {os.path.basename(a.ref)}: {", ".join(notes)}; without the rewrite: {plain}', flush=True)
print('ALL MATCH' if not bad else f'{bad} FAILED')
sys.exit(1 if bad else 0)
