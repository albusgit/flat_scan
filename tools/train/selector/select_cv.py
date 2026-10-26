"""select_cv.py: can a small model pick the best flattening method per photo from what the pipeline logs?

For photos run several ways (ft.sh runs: the app default, UVDoc only, no nets) and scored against flat scans
(gtscore.py -> RUN/_gt_<SET>.json: MS-SSIM, LD per page), the features are the app default run's JSON signals
(both nets' outline agreements, text-row spreads, the grid's distance from flat, the page finder's side bows and
fit, ...). A gradient-boosted classifier predicts which method wins (by MS-SSIM); grouped K-fold cross-validation
(every photo is scored by a model that never saw it, and DIR300 / DocReal documents never straddle folds) gives
the honest gain.

  python select_cv.py FT_DIR SET:DEFAULT_RUN:RUN2:RUN3[:...] [SET:...]   (names as in FT_DIR)
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

FEATS = ['uv_agree_quad', 'uv_agree_mask', 'uv_spread', 'uv_flat_resid', 'side_bow', 'side_edge', 'uv_paper_share',
         'uv_grid_check', 'fuse']
SCAL = ['fit_median_full_px', 'uv_agreement', 'uv_consistency', 'page_hw', 'line_distortion', 'uv_model', 'flat_page',
        'surface_unfounded', 'uv_defines_page', 'uv_rejected', 'text_rows', 'straightness']


def feats(j):
    v = []
    for k in SCAL:
        x = j.get(k, -1)
        if isinstance(x, list):
            v += [float(y) if isinstance(y, (int, float)) else -1 for y in x[:2]]
        else:
            v.append(float(x) if isinstance(x, (int, float, bool)) else -1)
    for k in FEATS:
        x = j.get(k, [])
        flat = np.array(x, dtype=float).ravel().tolist() if isinstance(x, list) else [float(x)]
        n = {'uv_flat_resid': 3, 'side_bow': 4, 'side_edge': 4, 'uv_grid_check': 6, 'fuse': 6}.get(k, 2)
        flat = (flat + [-1] * n)[:n]
        if k in ('side_bow', 'side_edge'):
            flat = sorted(flat)   # the sides' order is the photo's, not the page's
        v += flat
    return v


def main():
    D = sys.argv[1]
    X, Y, S, G, names = [], [], [], [], None
    for spec in sys.argv[2:]:
        parts = spec.split(':')
        st, runs = parts[0], parts[1:]
        names = runs if names is None else names
        sc = [json.load(open(os.path.join(D, r, f'_gt_{st}.json'))) for r in runs]
        for k in sc[0]:
            if not all(k in s for s in sc):
                continue
            try:
                j = json.load(open(os.path.join(D, runs[0], k + '.json')))
            except Exception:
                continue
            X.append(feats(j))
            ms = [s[k][0] for s in sc]
            S.append(ms)
            Y.append(int(np.argmax(ms)))
            doc = k.split('_')[0] if st == 'docreal' else k
            G.append(f'{st}:{doc}')
    X, Y, S = np.array(X), np.array(Y), np.array(S)
    print(f'{len(X)} photos, {X.shape[1]} features; methods {names}; wins per method {np.bincount(Y, minlength=S.shape[1])}')
    pred = np.zeros(len(X), int)
    for tr, te in GroupKFold(5).split(X, Y, G):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        m.fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
    picked = S[np.arange(len(S)), pred]
    for i, n in enumerate(names):
        print(f'  always {n:14s} MS-SSIM {S[:, i].mean():.4f}')
    print(f'  selector (5-fold CV)    MS-SSIM {picked.mean():.4f}   picks {np.bincount(pred, minlength=S.shape[1])}')
    print(f'  oracle                  MS-SSIM {S.max(1).mean():.4f}')
    sets = sorted({g.split(':')[0] for g in G})
    for st in sets:
        m = np.array([g.startswith(st + ':') for g in G])
        print(f'  {st:10s} n {m.sum():4d}  default {S[m, 0].mean():.4f}  selector {picked[m].mean():.4f}  oracle {S[m].max(1).mean():.4f}')


if __name__ == '__main__':
    main()
