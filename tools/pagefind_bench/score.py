"""score.py RUN_DIR GT_DIR [--real] [--out CSV]: score the app's page finder (pipeline_cli outline dumps, as written by
the fast-test runner one.sh: NAME.outline.npz + NAME.json) against ground-truth masks.

GT_DIR: make_synth.py output (NAME.png mask + NAME.json corners / side_outside), or with --real the round-4 pack's
gt folder (gt.json + masks/NAME.png).

Per photo:
  iou          the found outline's area vs the true paper mask
  bd_mean/p95  boundary distance, both ways, as a share of the page diagonal; boundary on the photo's own frame is left
               out (a page cut by the frame has no edge there to find)
  corner_err   largest corner miss / diagonal (synthetic, pages not cut by the frame)
  cut_ok       the sides cut by the frame marked as such (side_edge > 0.5 vs truth: > 30 % of the side outside)
  flagged      the app asked the user: needs_manual_crop or edge_doubt
  fail         bd_p95 > 3 % of the diagonal or iou < 0.9;   silent = fail and not flagged
Prints totals and per slice (synthetic: kind, cut, aspect)."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def boundary(mask, frame=3):
    """the mask's outline pixels; frame > 0 leaves out those within `frame` px of the photo's edge (the photo's own
    edge is then no outline), frame = 0 keeps them (the area beyond the photo counts as not paper)"""
    m = np.pad(mask.astype(np.uint8), 1)
    e = (m - cv2.erode(m, np.ones((3, 3), np.uint8)))[1:-1, 1:-1]
    if frame:
        e[:frame, :] = 0; e[-frame:, :] = 0; e[:, :frame] = 0; e[:, -frame:] = 0
    return e > 0


def score_one(pred_contour, gt_mask):
    H, W = gt_mask.shape
    pm = np.zeros((H, W), np.uint8)
    if pred_contour is not None and len(pred_contour) >= 3:
        cv2.fillPoly(pm, [np.round(pred_contour).astype(np.int32)], 1)
    g = gt_mask > 0
    p = pm > 0
    inter, union = np.logical_and(g, p).sum(), np.logical_or(g, p).sum()
    iou = inter / union if union else 0.0
    ys, xs = np.nonzero(g)
    diag = float(np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1)) if len(xs) else float(np.hypot(W, H))
    # samples: outline pixels off the photo's frame; distances: to the other outline including the frame, so a found
    # edge a few px inside a page that fills the photo counts its real miss, not "no edge to compare with"
    bg, bp = boundary(g), boundary(p)
    if not bg.any() and not bp.any():
        return iou, 0.0, 0.0, diag
    dg = cv2.distanceTransform((~boundary(g, 0)).astype(np.uint8), cv2.DIST_L2, 5)   # distance to the true outline
    dp = cv2.distanceTransform((~boundary(p, 0)).astype(np.uint8), cv2.DIST_L2, 5)
    a, b = dg[bp], dp[bg]
    both = np.concatenate([a, b])
    p95 = max(np.percentile(x, 95) for x in (a, b) if len(x))
    return iou, float(both.mean() / diag), float(p95 / diag), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('gt'); ap.add_argument('--real', action='store_true'); ap.add_argument('--out')
    a = ap.parse_args()
    rows = []
    real_gt = json.load(open(os.path.join(a.gt, 'gt.json'))) if a.real else None
    for oj in sorted(glob.glob(os.path.join(a.run, '*.json'))):
        name = os.path.basename(oj)[:-5]
        if name.startswith('_'):
            continue
        res = json.load(open(oj))
        on = os.path.join(a.run, name + '.outline.npz')
        o = np.load(on) if os.path.exists(on) else None
        if a.real:
            g = real_gt.get(name)
            if not g or not g.get('page'):
                continue
            gm = cv2.imread(os.path.join(a.gt, 'masks', name + '.png'), 0)
            meta = dict(kind='real', cut=None, aspect='', corners=None, side_outside=None)
        else:
            gm = cv2.imread(os.path.join(a.gt, name + '.png'), 0)
            meta = json.load(open(os.path.join(a.gt, name + '.json')))
        if gm is None:
            continue
        con = o['contour'] if o is not None else None   # photo px; the masks are stored at photo size
        iou, bm, b95, diag = score_one(con, gm > 127)
        flagged = bool(res.get('needs_manual_crop')) or bool(res.get('edge_doubt'))
        r = dict(name=name, kind=meta.get('kind'), cut=meta.get('cut'), aspect=meta.get('aspect'), iou=round(iou, 4),
                 bd_mean=round(bm, 4), bd_p95=round(b95, 4), flagged=flagged, mode=res.get('mode'),
                 reasons=';'.join(res.get('needs_manual_crop') or []))
        if o is not None and meta.get('corners') and not meta.get('cut'):
            pc = np.array(o['corners'], float).reshape(-1, 2)
            gc = np.array(meta['corners'], float)
            if len(pc) == 4:
                err = min(np.linalg.norm(np.roll(pc, k, 0) - gc, axis=1).max() for k in range(4))
                r['corner_err'] = round(err / diag, 4)
        if meta.get('side_outside') is not None and o is not None and res.get('side_edge') is not None:
            truth = [s > 0.3 for s in meta['side_outside']]
            pred = [s > 0.5 for s in res['side_edge']]
            # the found corners may be numbered from another corner: compare under the best rotation
            r['cut_ok'] = max(all(np.roll(pred, k) == np.array(truth)) for k in range(4))
        r['fail'] = r['bd_p95'] > 0.03 or r['iou'] < 0.9
        r['silent'] = r['fail'] and not flagged
        rows.append(r)
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)

    def summary(tag, rs):
        if not rs:
            return
        n = len(rs)
        ce = [r['corner_err'] for r in rs if 'corner_err' in r]
        co = [r['cut_ok'] for r in rs if 'cut_ok' in r]
        print(f"{tag:22s} n {n:4d}  IoU mean {np.mean([r['iou'] for r in rs]):.3f} (<0.9: {sum(r['iou'] < 0.9 for r in rs):3d})  "
              f"edge mean {100 * np.mean([r['bd_mean'] for r in rs]):.2f}% p95 {100 * np.median([r['bd_p95'] for r in rs]):.2f}% (median of p95)  "
              f"corner {'' if not ce else f'{100 * np.median(ce):.2f}% med, {100 * np.percentile(ce, 90):.1f}% p90'}  "
              f"cut-sides ok {'' if not co else f'{np.mean(co):.2f}'}  fail {sum(r['fail'] for r in rs):3d}  "
              f"flagged {sum(r['flagged'] for r in rs):3d}  SILENT {sum(r['silent'] for r in rs):3d} ({100 * np.mean([r['silent'] for r in rs]):.1f}%)")
    summary('ALL', rows)
    for key in ('kind', 'cut', 'aspect', 'mode'):
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(key)].append(r)
        if len(groups) > 1:
            for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
                summary(f'{key}={k}', rs)


if __name__ == '__main__':
    main()
