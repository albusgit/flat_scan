"""Cache training pairs as small uint8 arrays so the data loader does almost no CPU work (the machine is shared).

  python cache_pairs.py DATA_DIR

For each pair with a pseudo-GT grid: the page's box + 14 % of its longer side on every side (clamped to the photo,
so clamping a crop to this region = clamping it to the photo, as the app does), scaled so its longer side is at
most 1600 px -> DATA_DIR/cache/<id>.npy (HxWx3 RGB) and <id>_bm.npy (grid in region px)."""
import json
import math
import os
import sys

import cv2
import numpy as np

data = sys.argv[1]
od = os.path.join(data, 'cache')
os.makedirs(od, exist_ok=True)
pairs = json.load(open(os.path.join(data, 'pairs.json')))
n = 0
for p in pairs:
    pg = os.path.join(data, 'pgt', p['id'] + '.npz')
    out = os.path.join(od, p['id'] + '.npy')
    if not os.path.exists(pg) or os.path.exists(out):
        continue
    bm = np.load(pg)['bm'].astype(np.float64)
    im = cv2.imread(os.path.join(data, p['photo']))
    h, w = im.shape[:2]
    c = np.array([bm[:, 0, 0], bm[:, 0, -1], bm[:, -1, -1], bm[:, -1, 0]])
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    m = 0.14 * max(x1 - x0, y1 - y0)
    bx0, by0 = max(0, int(math.floor(x0 - m))), max(0, int(math.floor(y0 - m)))
    bx1, by1 = min(w, int(math.ceil(x1 + m)) + 1), min(h, int(math.ceil(y1 + m)) + 1)
    reg = im[by0:by1, bx0:bx1]
    s = min(1.0, 1600.0 / max(reg.shape[:2]))
    if s < 1:
        reg = cv2.resize(reg, (max(1, round(reg.shape[1] * s)), max(1, round(reg.shape[0] * s))), interpolation=cv2.INTER_AREA)
    sx = reg.shape[1] / (bx1 - bx0)
    sy = reg.shape[0] / (by1 - by0)
    b = np.stack([(bm[0] - bx0 + 0.5) * sx - 0.5, (bm[1] - by0 + 0.5) * sy - 0.5]).astype(np.float32)
    np.save(out, np.ascontiguousarray(reg[:, :, ::-1]))
    np.save(os.path.join(od, p['id'] + '_bm.npy'), b)
    n += 1
print('cached', n)
