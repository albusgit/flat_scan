"""Screen detection scores (core detect_screen via filter_cli) over image sets.

  python -I screen_scores.py BIN OUT.json LABEL=GLOB [LABEL=GLOB ...]
LABEL starting with "pos" counts as screens, anything else as paper. Prints the score range per set and
the false-positive / miss counts at thresholds.
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

bin_, out = sys.argv[1], sys.argv[2]
env = dict(os.environ)
env['PATH'] = r'C:\Users\User\Android\opencv\build\x64\vc16\bin;' + env['PATH']
tmp = os.path.join(os.path.dirname(out), 'tmp_scores')
os.makedirs(tmp, exist_ok=True)


def one(p):
    r = subprocess.run([bin_, p.replace(os.sep, '/'), tmp.replace(os.sep, '/'), '--filters', 'x', '--screen', '0'], capture_output=True, text=True, env=env)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1])
        return p, j['screen_score'], j['detect_ms']
    except Exception:
        return p, None, None


res = {}
for spec in sys.argv[3:]:
    lab, g = spec.split('=', 1)
    files = sorted(glob.glob(g))
    with ThreadPoolExecutor(4) as ex:
        rr = list(ex.map(one, files))
    res[lab] = {os.path.basename(p): (s, ms) for p, s, ms in rr if s is not None}
    v = np.float64([s for s, _ in res[lab].values()])
    if len(v):
        print(f'{lab:14s} n={len(v):3d} min {v.min():.3f} p10 {np.percentile(v, 10):.3f} med {np.median(v):.3f} p90 {np.percentile(v, 90):.3f} max {v.max():.3f}  '
              f'ms med {np.median([m for _, m in res[lab].values()]):.0f}', flush=True)
json.dump(res, open(out, 'w'), indent=1)
pos = np.float64([s for l, d in res.items() if l.startswith('pos') for s, _ in d.values()])
neg = np.float64([s for l, d in res.items() if not l.startswith('pos') for s, _ in d.values()])
for t in np.arange(1.3, 2.6, 0.1):
    print(f'thr {t:.1f}: false positives {int((neg >= t).sum())}/{len(neg)}, screens found {int((pos >= t).sum())}/{len(pos)}')
