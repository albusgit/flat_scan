"""Prototype (round 19b fix 2c): straighten wobbly page outlines with a per-side shape prior plus edge evidence.

Each side of the found outline (side0..3 of the outline dump: top, right, bottom, left) is fitted with a robust
quadratic in the frame of its chord (corner to corner). A stretch that strays from the fit by more than tau is kept
only when the photo shows a paper edge there as strongly as along the rest of that side (a real curl, fold or lifted
corner); otherwise (the outline followed print, a page stack, a shadow, the background) it is replaced by the fit.

  smooth_outline.py RUN_DIR NAMES_FILE OUT_DIR    -> OUT_DIR/NAME.outline.npz (same keys, contour replaced) and
                                                     OUT_DIR/NAME.jpg (old outline red, new green) + a summary line"""
import json
import os
import sys

import cv2
import numpy as np


def edge_strength(lab, pts, nrm, inside_sign):
    """per point: Lab contrast across the outline (inside minus outside, 3-6 px each way), one value per point"""
    H, W = lab.shape[:2]
    def s(q):
        x = np.clip(q[:, 0], 0, W - 1).astype(np.float32).reshape(-1, 1)
        y = np.clip(q[:, 1], 0, H - 1).astype(np.float32).reshape(-1, 1)
        return cv2.remap(lab, x, y, cv2.INTER_LINEAR).reshape(-1, 3)
    n = nrm * inside_sign
    a = (s(pts + 3 * n) + s(pts + 6 * n)) / 2
    b = (s(pts - 3 * n) + s(pts - 6 * n)) / 2
    return np.linalg.norm(a - b, axis=1), (a - b)[:, 0]


def robust_quadratic(x, y, iters=8):
    w = np.ones_like(x)
    c = np.polyfit(x, y, 2)
    for _ in range(iters):
        r = y - np.polyval(c, x)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1 / np.maximum(1, np.abs(r) / (2 * s))           # Huber-like
        c = np.polyfit(x, y, 2, w=np.sqrt(w))
    return c


def smooth(photo, o, verbose_name=''):
    H, W = photo.shape[:2]
    lab = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)
    con = o['contour'].astype(np.float64)
    corners = np.array(o['corners'], float).reshape(4, 2)
    poly = con.astype(np.float32).reshape(-1, 1, 2)
    diag = float(np.hypot(*(con.max(0) - con.min(0))))
    tau = max(4.0, 0.004 * diag)
    new_sides, stats = [], []
    for k in range(4):
        P = o[f'side{k}'].astype(np.float64)
        if len(P) < 12:
            new_sides.append(P); stats.append(dict(side=k, n=len(P), replaced=0, kept_runs=0)); continue
        a, b = corners[k], corners[(k + 1) % 4]
        ex = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        ey = np.array([-ex[1], ex[0]])
        x, y = (P - a) @ ex, (P - a) @ ey
        order = np.argsort(x); P, x, y = P[order], x[order], y[order]
        on_frame = (P[:, 0] < 3) | (P[:, 1] < 3) | (P[:, 0] > W - 4) | (P[:, 1] > H - 4)
        use = ~on_frame
        if use.sum() < 8:
            new_sides.append(P); stats.append(dict(side=k, n=len(P), replaced=0, kept_runs=0, frame=True)); continue
        c = robust_quadratic(x[use], y[use])
        fit = np.polyval(c, x)
        res = y - fit
        # normals of the fitted curve, pointing into the page (test with the outline polygon)
        dy = np.polyval(np.polyder(c), x)
        t = np.stack([np.ones_like(dy), dy], 1); t /= np.linalg.norm(t, axis=1, keepdims=True)
        tang = t[:, :1] * ex[None] + t[:, 1:] * ey[None]
        nf = np.stack([-tang[:, 1], tang[:, 0]], 1)
        mid = a + ex * x[len(x) // 2] + ey * fit[len(x) // 2]
        sign = 1.0 if cv2.pointPolygonTest(poly, (float(mid[0] + 8 * nf[len(x) // 2, 0]), float(mid[1] + 8 * nf[len(x) // 2, 1])), False) > 0 else -1.0
        strength, dl = edge_strength(lab, P, nf, sign)
        inl = use & (np.abs(res) <= tau)
        ref = np.median(strength[inl]) if inl.sum() > 5 else np.median(strength[use])
        ref_sign = np.sign(np.median(dl[inl])) if inl.sum() > 5 else 0
        out = use & (np.abs(res) > tau)
        # runs of strays
        newP = P.copy()
        replaced = kept = 0
        i = 0
        n = len(P)
        while i < n:
            if not out[i]:
                i += 1; continue
            j = i
            while j < n and out[j]:
                j += 1
            run = slice(i, j)
            st = np.median(strength[run]); sg = np.mean(np.sign(dl[run]) == ref_sign) if ref_sign else 1.0
            real = st >= 0.7 * ref and sg >= 0.8 and (j - i) >= 6
            if real:
                kept += 1
            else:
                newP[run] = a + ex[None] * x[run, None] + ey[None] * fit[run, None]
                replaced += j - i
            i = j
        new_sides.append(newP)
        stats.append(dict(side=k, n=n, replaced=int(replaced), kept_runs=int(kept), tau=round(tau, 1),
                          max_stray=round(float(np.abs(res[use]).max()), 1)))
    # move the replaced points inside the outline itself, in its own order (rebuilding it from the sides, sorted along
    # their chords, scrambled the order wherever the outline doubles back)
    from scipy.spatial import cKDTree
    contour = con.copy()
    for k in range(4):
        P = o[f'side{k}'].astype(np.float64)
        if len(P) < 12 or len(new_sides[k]) != len(P):
            continue
        # new_sides[k] is in sorted order; undo the sort to match P
        a, b = corners[k], corners[(k + 1) % 4]
        ex = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        order = np.argsort((P - a) @ ex)
        moved = np.empty_like(P); moved[order] = new_sides[k]
        d, idx = cKDTree(P).query(contour)
        hit = d < 0.5
        contour[hit] = moved[idx[hit]]
    return contour, new_sides, stats


if __name__ == '__main__':
    run, names, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    for name in open(names).read().split():
        on = os.path.join(run, name + '.outline.npz')
        if not os.path.exists(on):
            continue
        o = dict(np.load(on))
        res = json.load(open(os.path.join(run, name + '.json')))
        photo = cv2.imread(res['photo'])
        contour, sides, stats = smooth(photo, o)
        o2 = dict(o); o2['contour'] = contour.astype(np.float32)
        for k in range(4):
            o2[f'side{k}'] = sides[k].astype(np.float32)
        np.savez(os.path.join(out, name + '.outline.npz'), **o2)
        json.dump(res, open(os.path.join(out, name + '.json'), 'w'))
        im = photo.copy()
        t = max(2, im.shape[1] // 500)
        cv2.polylines(im, [o['contour'].astype(np.int32)], True, (0, 0, 255), t)
        cv2.polylines(im, [contour.astype(np.int32)], True, (0, 220, 0), t)
        cv2.imwrite(os.path.join(out, name + '.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name[:40], ' '.join(f"s{s['side']}: repl {s['replaced']}/{s['n']} keep {s['kept_runs']} max {s.get('max_stray')}" for s in stats))
