"""Stability of the page finder: the same photo, slightly changed, should give the same outline.

  perturb.py make LIST OUT_DIR       write perturbed copies of each photo in LIST (+ OUT_DIR/_transforms.json) and
                                     OUT_DIR/_list.txt (the originals and the copies, for the fast-test runner)
  perturb.py score RUN_DIR OUT_DIR   compare each copy's found outline with the original's, carried through the same
                                     transform: p95 distance as a share of the page diagonal, per change and overall

Changes (each alone): scale 0.95 / 1.05, rotate -3 / +3 deg, crop 3 % off one side, JPEG quality 60, brightness
-15 % / +15 %."""
import json
import os
import sys

import cv2
import numpy as np

CHANGES = ['scale0.95', 'scale1.05', 'rot-3', 'rot+3', 'crop3', 'jpeg60', 'dark15', 'bright15']


def transform(name, W, H):
    """2 x 3 affine (original px -> changed px) and the output size"""
    if name.startswith('scale'):
        s = float(name[5:])
        return np.float64([[s, 0, 0], [0, s, 0]]), (int(round(W * s)), int(round(H * s)))
    if name.startswith('rot'):
        return cv2.getRotationMatrix2D((W / 2, H / 2), float(name[3:]), 1.0), (W, H)
    if name == 'crop3':
        dx = int(round(0.03 * W))
        return np.float64([[1, 0, -dx], [0, 1, 0]]), (W - dx, H)
    return np.float64([[1, 0, 0], [0, 1, 0]]), (W, H)


def make(lst, out):
    os.makedirs(out, exist_ok=True)
    tr, names = {}, []
    for p in open(lst).read().split():
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        base = os.path.splitext(os.path.basename(p))[0]
        names.append(p)
        for c in CHANGES:
            A, (w, h) = transform(c, W, H)
            o = cv2.warpAffine(im, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) if c.startswith(('scale', 'rot', 'crop')) else im.copy()
            if c == 'dark15':
                o = np.clip(o.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
            if c == 'bright15':
                o = np.clip(o.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
            q = 60 if c == 'jpeg60' else 95
            fn = os.path.join(out, f'{base}__{c}.jpg')
            cv2.imwrite(fn, o, [cv2.IMWRITE_JPEG_QUALITY, q])
            tr[f'{base}__{c}'] = dict(orig=base, A=A.tolist(), size=[w, h])
            names.append(fn.replace('\\', '/'))
    json.dump(tr, open(os.path.join(out, '_transforms.json'), 'w'))
    open(os.path.join(out, '_list.txt'), 'w', newline='').write('\n'.join(names) + '\n')
    print(len(names), 'photos')


def outline_mask(con, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(con).astype(np.int32)], 1)
    return m


def score(run, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score import boundary
    tr = json.load(open(os.path.join(out, '_transforms.json')))
    per = {c: [] for c in CHANGES}
    worst = []
    for name, t in tr.items():
        fo, fc = os.path.join(run, t['orig'] + '.outline.npz'), os.path.join(run, name + '.outline.npz')
        if not (os.path.exists(fo) and os.path.exists(fc)):
            continue
        w, h = t['size']
        A = np.array(t['A'])
        co = np.load(fo)['contour'] @ A[:, :2].T + A[:, 2]       # the original's outline, carried into the copy
        cc = np.load(fc)['contour']
        mo, mc = outline_mask(co, w, h), outline_mask(cc, w, h)
        bo, bc = boundary(mo > 0), boundary(mc > 0)
        if not bo.any() and not bc.any():
            d = 0.0
        else:
            do = cv2.distanceTransform((~boundary(mo > 0, 0)).astype(np.uint8), cv2.DIST_L2, 5)
            dc = cv2.distanceTransform((~boundary(mc > 0, 0)).astype(np.uint8), cv2.DIST_L2, 5)
            a, b = do[bc], dc[bo]
            ys, xs = np.nonzero(mo)
            diag = float(np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1)) if len(xs) else float(np.hypot(w, h))
            d = max(np.percentile(x, 95) for x in (a, b) if len(x)) / diag
        c = name.split('__')[-1]
        per[c].append(d)
        worst.append((d, name))
    allv = [v for vs in per.values() for v in vs]
    print(f'ALL      n {len(allv):4d}  outline change p95: median {100 * np.median(allv):.2f}%  90th pct {100 * np.percentile(allv, 90):.2f}%  '
          f'> 1%: {sum(v > 0.01 for v in allv)}  > 3%: {sum(v > 0.03 for v in allv)}')
    for c, vs in per.items():
        if vs:
            print(f'{c:9s} n {len(vs):4d}  median {100 * np.median(vs):.2f}%  90th {100 * np.percentile(vs, 90):.2f}%  max {100 * max(vs):.2f}%  > 3%: {sum(v > 0.03 for v in vs)}')
    worst.sort(reverse=True)
    print('worst:', [(n, round(100 * d, 1)) for d, n in worst[:10]])


if __name__ == '__main__':
    {'make': make, 'score': score}[sys.argv[1]](*sys.argv[2:4])
