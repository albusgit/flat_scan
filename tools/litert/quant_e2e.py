"""End-to-end check of model-size variants: the desktop pipeline (pipeline_cli, nets in-process on ONNX Runtime CPU)
with a variant's models, on the 35 photos or the DocUNet crops (Round 11, stream MODELSIZE).

  python tools/litert/quant_e2e.py run SET OUT --models DIR [--cli PIPELINE_CLI] [--jobs N] [--no-pf]
      SET s35 | docunet (tools/desktop_check/gate_b.py's sets); DIR holds pagefinder.onnx, uvdoc.onnx,
      syntheticdoc_hr.onnx, shadenet.onnx (a variant: quantize.py `sim` files, the others copied); s35 runs
      ShadeNet (FS_SHADE_NET=DIR/shadenet.onnx, FS_SHADE_AFTER=1, as the app), DocUNet FS_NO_SHADOWS=1 and PNG
      (as docunet_bench/run_ours.py). --no-pf: no page finder (pipeline_cli without "net": the colour-threshold
      paper mask, the app's fallback when the page finder is off or fails).
  python tools/litert/quant_e2e.py cmp SET BASE OUT
      per photo: page size, max / mean |pixel difference|, share of pixels off by > 8 levels, the JSON fields
      that differ (mode, corners, flags); and the changed pages' list (for docunet_bench scoring).
  python tools/litert/quant_e2e.py compose BASE_METRICS.json OUT
      DocUNet metrics of OUT composed from BASE's per-image scores and OUT_sub's re-scored changed pages
      (identical page => identical score; doc 64 excluded, as docunet_bench/RESULTS.md).
The nets' outputs are dumped (FS_NET_DUMP: OUT/NAME_logits / _p1 / _rc0 / _rc2.npz) for net-level checks."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'desktop_check'))
import gate_b  # noqa: E402

CLI = os.path.join(REPO, 'tools', 'desktop_check', 'build', 'pipeline_cli.exe')


def one(job):
    cli, models, set_, out, n, photo, no_pf = job
    base = os.path.join(out, n)
    ext = '.png' if set_ == 'docunet' else '.jpg'
    e = dict(os.environ)
    e['PATH'] = gate_b.OPENCV_BIN + ';' + e['PATH']
    e.update(FS_NETS='1', FS_NET_DUMP=base, FS_MODELS=models, FS_ORT_THREADS=os.environ.get('FS_ORT_THREADS', '2'))
    if set_ == 'docunet':
        e['FS_NO_SHADOWS'] = '1'
    else:
        e.update(FS_SHADE_NET=os.path.join(models, 'shadenet.onnx'), FS_SHADE_AFTER='1')
    args = [cli, '--backend', 'cpu', photo, base + ext] + ([] if no_pf else ['net'])
    t0 = time.perf_counter()
    r = subprocess.run(args, env=e, capture_output=True, text=True)
    with open(base + '.json', 'w') as f:
        f.write(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({'ok': False, 'error': r.stderr[-300:]}))
    return n, 1000 * (time.perf_counter() - t0)


def run(a):
    out, models = a['out'], a['models']
    os.makedirs(out, exist_ok=True)
    jobs = [(a['cli'], models, a['set'], out, n, p, a['no_pf']) for n, p in gate_b.photo_set(a['set'])]
    t0 = time.time()
    with Pool(a['jobs']) as pool:
        times = dict(pool.imap_unordered(one, jobs))
    json.dump({'models': models, 'no_pf': a['no_pf'], 'cli': a['cli'], 'ms': times}, open(os.path.join(out, 'run.json'), 'w'), indent=1)
    print(f'{a["set"]}: {len(jobs)} photos in {time.time() - t0:.0f} s -> {out}')


def img(p):
    import cv2
    return cv2.imread(p, cv2.IMREAD_COLOR) if os.path.exists(p) else None


def jcore(p):
    if not os.path.exists(p):
        return {}
    j = json.loads(open(p).read().replace('\\', '/'))
    for k in ('nets', 'pipeline_ms', 'photo'):
        j.pop(k, None)
    return j


def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{pre}{k}.'))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f'{pre}{i}.'))
    else:
        out[pre[:-1]] = d
    return out


def cmp(set_, base, out):
    ext = '.png' if set_ == 'docunet' else '.jpg'
    changed, rows = [], []
    for n, _ in gate_b.photo_set(set_):
        a, b = img(os.path.join(base, n + ext)), img(os.path.join(out, n + ext))
        ja, jb = flat(jcore(os.path.join(base, n + '.json'))), flat(jcore(os.path.join(out, n + '.json')))
        # JSON keys that differ, numbers within 1e-6 relative counted equal
        dk = []
        for k in sorted(set(ja) | set(jb)):
            va, vb = ja.get(k), jb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                if abs(va - vb) <= 1e-6 * max(1, abs(va)):
                    continue
            elif va == vb:
                continue
            dk.append(k)
        if a is None or b is None:
            rows.append((n, 'missing', dk))
            changed.append(n)
            continue
        if a.shape != b.shape:
            rows.append((n, f'size {a.shape[1]}x{a.shape[0]} -> {b.shape[1]}x{b.shape[0]}', dk))
            changed.append(n)
            continue
        d = np.abs(a.astype(np.int16) - b).max(axis=2)
        if d.max() == 0:
            rows.append((n, 'same', dk))
            continue
        changed.append(n)
        rows.append((n, f'max {d.max():3d} mean {d.mean():.3f} >8: {100 * (d > 8).mean():.2f}%', dk))
    for n, s, dk in rows:
        top = [k for k in dk if not any(k.startswith(p) for p in ('timing', 'ms'))]
        print(f'{n:34s} {s:36s} json: {", ".join(top[:8]) + (" ..." if len(top) > 8 else "") if top else "same"}')
    print(f'pages differing: {len(changed)} of {len(rows)}')
    print('CHANGED ' + ' '.join(changed))
    return changed


def compose(base_metrics, out):
    names = [f'{k}_{m}' for k in range(1, 66) for m in (1, 2)]
    bp = json.load(open(base_metrics))['per_image']
    sub = out.rstrip('/\\') + '_sub'
    sp = json.load(open(os.path.join(sub, 'docunet_metrics.json')))['per_image'] if os.path.exists(os.path.join(sub, 'docunet_metrics.json')) else {}
    per = {n: dict(sp[n]) if n in sp else dict(bp[n]) for n in names}
    ks = [n for n in names if not n.startswith('64_')]
    mean = {m: sum(per[k][m] for k in ks) / len(ks) for m in ('ms', 'ld', 'ad')}
    bm = {m: sum(bp[k][m] for k in ks) / len(ks) for m in ('ms', 'ld', 'ad')}
    json.dump({'mean': mean, 'n': len(ks), 'per_image': per, 'composed_from': base_metrics, 'rescored': sorted(sp)},
              open(os.path.join(out, 'docunet_metrics.json'), 'w'), indent=1)
    print(f'{"":10s} {"MS-SSIM":>8s} {"LD":>7s} {"AD":>7s}   (n {len(ks)}, re-scored {len(sp)})')
    print(f'{"base":10s} {bm["ms"]:8.4f} {bm["ld"]:7.3f} {bm["ad"]:7.4f}')
    print(f'{"variant":10s} {mean["ms"]:8.4f} {mean["ld"]:7.3f} {mean["ad"]:7.4f}')
    print(f'{"delta":10s} {mean["ms"] - bm["ms"]:+8.4f} {mean["ld"] - bm["ld"]:+7.3f} {mean["ad"] - bm["ad"]:+7.4f}')
    rows = sorted((per[n]['ld'] - bp[n]['ld'], n) for n in sp if not n.startswith('64_'))
    for d, n in rows[:5] + rows[-5:]:
        print(f'  {n:6s} LD {bp[n]["ld"]:6.2f} -> {per[n]["ld"]:6.2f} ({d:+.2f})  MS {bp[n]["ms"]:.4f} -> {per[n]["ms"]:.4f}')
    better = sum(1 for d, _ in rows if d < -0.05)
    worse = sum(1 for d, _ in rows if d > 0.05)
    print(f'changed pages: LD better {better}, worse {worse}, within 0.05: {len(rows) - better - worse}')


def main():
    a = sys.argv[1:]
    cmd = a.pop(0)
    if cmd == 'cmp':
        cmp(a[0], a[1], a[2])
        return 0
    if cmd == 'compose':
        compose(a[0], a[1])
        return 0
    o = {'set': a[0], 'out': a[1], 'cli': CLI, 'jobs': 4, 'no_pf': False, 'models': None}
    rest = a[2:]
    while rest:
        f = rest.pop(0)
        if f == '--models':
            o['models'] = os.path.abspath(rest.pop(0))
        elif f == '--cli':
            o['cli'] = rest.pop(0)
        elif f == '--jobs':
            o['jobs'] = int(rest.pop(0))
        elif f == '--no-pf':
            o['no_pf'] = True
    run(o)
    return 0


if __name__ == '__main__':
    sys.exit(main())
