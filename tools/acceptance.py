"""Acceptance metrics from APP_BUILD_BRIEF.md section 5, computed with the reference code.

usage: python -I tools/acceptance.py REF_CODE_DIR OUT_DIR [--flat FLAT_PNG --bench BENCH_DIR]
OUT_DIR must contain photo1/, photo2/ (out.png, warpmap.npz), scan_p1..3/ (B_warpmap.png),
and optionally bench_P1/, bench_P2/ (warpmap.npz).
"""
import sys, os, json, argparse
p = argparse.ArgumentParser()
p.add_argument('code'); p.add_argument('out')
p.add_argument('--flat'); p.add_argument('--bench')
a = p.parse_args()
sys.path.insert(0, os.path.abspath(a.code))
import numpy as np, cv2
import warpmap as W, evaluate as E, bench as B
from scipy.ndimage import gaussian_filter1d

res = {}
if a.bench:
    for c in ('P1', 'P2'):
        f = f'{a.out}/bench_{c}/warpmap.npz'
        if os.path.exists(f):
            d = B.score(a.flat, f'{a.bench}/photo{c}_gt.npz', f)
            res[f'bench_{c}'] = {k: round(v, 3) for k, v in d.items()}
for name, img in (('photo1', 'out.png'), ('photo2', 'out.png'),
                  ('scan_p1', 'B_warpmap.png'), ('scan_p2', 'B_warpmap.png'), ('scan_p3', 'B_warpmap.png')):
    f = f'{a.out}/{name}/{img}'
    if not os.path.exists(f):
        continue
    n, m, p90, mx, f1, e90, emx = E.evaluate(f)
    res[name] = dict(lines=int(n), median=round(float(m), 3), p90=round(float(p90), 3), worst=round(float(mx), 3),
                     within_1px_pct=round(float(f1), 1), end_p90=round(float(e90), 3), end_worst=round(float(emx), 3))
# photo-1 title border: same logic and ROI as code/fixed_photo_feature.py, applied to the photo pipeline output
f = f'{a.out}/photo1/out.png'
if os.path.exists(f):
    im = cv2.imread(f); D = W.darkness_fast(im)
    band = D[35:96, 200:400].mean(1)
    seed = 35 + int(np.argmax(gaussian_filter1d(band, 1)))
    t, c, wd = W.trace_w2(D, seed, 100, 1550, True, win=22, maxrun=10, max_gap=8)
    valid = np.isfinite(c) & (wd > 0) & (wd <= 10)
    keep = valid & ~((t >= 1400) & (t <= 1490))
    med = np.median(c[keep]); end = keep & (t > 1490)
    res['photo1_title_border'] = dict(seed_y=seed, coverage=round(float(valid.mean()), 3),
        p98_px=round(float(np.percentile(abs(c[keep] - med), 98)), 3),
        end_max_px=round(float(np.max(abs(c[end] - med))), 3) if end.any() else None)
print(json.dumps(res, indent=1))
