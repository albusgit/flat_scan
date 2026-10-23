"""Straightness of a flattened page without ground truth (method-neutral), as Round 8 / Round 10 measured it
(scratchpad straightness.py, kept here so the library harness does not depend on the scratchpad).

Text rows: ink blobs merged along a row, centroid track per 8-px column, linear fit; per row the 95th
percentile deviation (px at 2263 px long side) and the row angle. A page's score:
  rows, row_dev_med, row_dev_p90 (px), angle_spread (deg, p90 - p10 of the row angles).
Table lines (optional, E:/fs_datasets/round4/code/evaluate.py): tlines, tline_med, tline_p90 (px, p98 deviation)."""
import os
import sys

import cv2
import numpy as np

R4 = r"E:\fs_datasets\round4\code"


def norm(img):
    s = 2263 / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def text_rows(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((7, 7), np.uint8)), 31)
    ink = ((bg.astype(np.int16) - g) > 40).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1))) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))
    ink[cv2.dilate(lines, np.ones((5, 5), np.uint8)) > 0] = 0
    words = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    words = cv2.morphologyEx(words, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(words, 8)
    devs, angs, lens = [], [], []
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        if w < 150 or not (8 <= h <= 60) or w < 5 * h:
            continue
        sub = (lab[y:y + h, x:x + w] == i) & (ink[y:y + h, x:x + w] > 0)
        xs, cs = [], []
        for cx in range(0, w - 8, 8):
            col = sub[:, cx:cx + 8]
            if col.sum() < 6:
                continue
            xs.append(x + cx + 4)
            cs.append(y + (np.nonzero(col)[0]).mean())
        if len(xs) < 12:
            continue
        xs, cs = np.array(xs, float), np.array(cs, float)
        pf = np.polyfit(xs, cs, 1)
        r = np.abs(np.polyval(pf, xs) - cs)
        devs.append(np.percentile(r, 95))
        angs.append(np.degrees(np.arctan(pf[0])))
        lens.append(w)
    return np.array(devs), np.array(angs), np.array(lens)


def score(path, table_lines=False):
    img = cv2.imread(path)
    if img is None:
        return None
    img = norm(img)
    best = None
    for k in (0, 1):    # sideways text: the orientation with more rows
        im = img if k == 0 else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        d, a, l = text_rows(im)
        if best is None or len(d) > len(best[0]):
            best = (d, a, l, im)
    d, a, l, im = best
    out = {"rows": int(len(d))}
    if len(d) >= 3:
        out.update(row_dev_med=float(np.median(d)), row_dev_p90=float(np.percentile(d, 90)),
                   angle_spread=float(np.percentile(a, 90) - np.percentile(a, 10)))
    if table_lines:
        if R4 not in sys.path:
            sys.path.insert(0, R4)
        try:
            import evaluate as EV  # noqa: E402
            tmp = path + ".norm.png"
            cv2.imwrite(tmp, im)
            try:
                n, m, p90, mx, f1, e90, emx = EV.evaluate(tmp)
                out.update(tlines=int(n), tline_med=float(m), tline_p90=float(p90))
            finally:
                os.remove(tmp)
        except Exception:  # noqa: BLE001
            out["tlines"] = 0
    return out
