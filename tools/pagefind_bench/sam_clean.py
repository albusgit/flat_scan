"""sam_clean.py PACK.npz INDEX.jsonl OUT.npz [--wobble 1.0] [--show SHEET.jpg]: SAM outlines (sam_pack.py, sources
NAME_sam) checked and cleaned side by side before the page finder trusts them as exact.

SAM's outline wanders along low-contrast edges (white paper on a pale table, a card's rounded edge): small teeth and
waves of 1-7 px at 384 that the 4-corner gate (sam_label: within delta of the quad) lets through. Per photo:
  1. made whole (sam_amodal.make_whole: holes, and notches deeper than wide, where a finger or a pen lies over the paper);
  2. the outline cut into the paper's four sides at the label's corners (the photo's own border left out: there the
     mask is the label's fill), each side smoothed along its length with its ends pinned (a curl or a bow stays,
     teeth and waves go: Gaussian, sigma 4 % of the side, at least 3 px);
  3. wobble = the raw side's p95 distance from its smooth version, along the side's middle (the ends left out: a card's
     rounded corner is no wobble; there the outline stays as SAM drew it). Every side <= --wobble px: the smoothed
     outline, exact (NAME_sam); <= --rough px: the smoothed outline with a ~5 px tolerance (NAME_samr: pagefind2/
     train.py DELTA); else the label's 4-corner fill with its source's tolerance (NAME).
Other sources are copied as they are."""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'train', 'pagefind2'))
import data as D  # noqa: E402
from sam_amodal import make_whole  # noqa: E402

SIZE = 384
_PQ = {}


def photo_size(path):
    """(w, h) of an index photo as cv2 reads it (EXIF orientation applied), from its header (a file, or
    parquet:<file>#<row>)"""
    from PIL import Image
    if path.startswith('parquet:'):
        f, row = path[len('parquet:'):].rsplit('#', 1)
        if f not in _PQ:
            import pyarrow.parquet as pq
            _PQ.clear()
            _PQ[f] = pq.read_table(f, columns=['image']).column('image')
        src = io.BytesIO(_PQ[f][int(row)].as_py()['bytes'])
    else:
        src = path
    with Image.open(src) as im:
        w, h = im.size
        turned = im.getexif().get(0x0112, 1) in (5, 6, 7, 8)
    return (h, w) if turned else (w, h)


def smooth1d(v, sig):
    """v: N x 2 points along an open curve -> smoothed, ends pinned (reflected about the end points)"""
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    a, b = v[0], v[-1]
    lo = 2 * a - v[1:r + 1][::-1] if len(v) > r else np.repeat(a[None], r, 0)
    hi = 2 * b - v[-r - 1:-1][::-1] if len(v) > r else np.repeat(b[None], r, 0)
    if len(lo) < r:
        lo = np.concatenate([np.repeat(lo[:1], r - len(lo), 0), lo])
    if len(hi) < r:
        hi = np.concatenate([hi, np.repeat(hi[-1:], r - len(hi), 0)])
    ext = np.concatenate([lo, v, hi])
    return np.stack([np.convolve(ext[:, j], k, mode='valid') for j in range(2)], 1)


def clean(m, quad, box, wobble_px):
    """m: 384 x 384 bool SAM mask; quad: 4 x 2 label corners (letterbox px); box: the photo's box -> (mask, exact?, info)"""
    m, holes, notches = make_whole(m)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return m, False, dict(why='empty')
    c = max(cs, key=cv2.contourArea)[:, 0].astype(np.float64)
    n = len(c)
    x0, y0, x1, y1 = box
    on_frame = (c[:, 0] <= x0 + 1) | (c[:, 1] <= y0 + 1) | (c[:, 0] >= x1 - 2) | (c[:, 1] >= y1 - 2)
    # the contour's points nearest the corners (a corner outside the photo: the nearest point on the frame)
    idx = [int(np.argmin(((c - q) ** 2).sum(1))) for q in quad]
    order = np.argsort(idx)
    cut = [idx[i] for i in order]
    out = c.copy()
    worst, sides = 0.0, []
    for j in range(4):
        a, b = cut[j], cut[(j + 1) % 4]
        seg = np.arange(a, b + 1) if b > a else np.concatenate([np.arange(a, n), np.arange(0, b + 1)])
        if len(seg) < 8:
            continue
        pts = c[seg]
        free = ~on_frame[seg]
        L = len(seg)
        sig = max(3.0, 0.04 * L)
        sm = smooth1d(pts, sig)
        res = np.sqrt(((pts - sm) ** 2).sum(1))
        e = max(6, int(0.1 * L))                           # the ends: corners (rounded on cards), as SAM drew them
        pos = np.arange(L)
        mid = free & (pos >= e) & (pos < L - e)
        w = float(np.percentile(res[mid], 95)) if mid.sum() >= 8 else 0.0
        sides.append(round(w, 2))
        worst = max(worst, w)
        ramp = np.clip(np.minimum(pos - e / 2, L - 1 - e / 2 - pos) / (e / 2), 0, 1)[:, None]
        out[seg] = np.where(free[:, None], ramp * sm + (1 - ramp) * pts, pts)   # along the frame: the label's fill as it was
    poly = np.round(out * 8).astype(np.int32)              # sub-pixel polygon (8x fixed point)
    m2 = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m2, [poly], 1, lineType=cv2.LINE_8, shift=3)
    return m2.astype(bool), worst, dict(wobble=round(worst, 2), sides=sides, holes=holes, notches=notches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack'); ap.add_argument('index'); ap.add_argument('out')
    ap.add_argument('--wobble', type=float, default=1.0); ap.add_argument('--rough', type=float, default=4.0); ap.add_argument('--show', default='')
    a = ap.parse_args()
    z = D.load(a.pack)
    rows = {}
    for l in open(a.index):
        r = json.loads(l)
        rows[r['id']] = r
    srcs = list(z['sources'])
    names = [str(x) for x in z['name']]
    todo = [k for k in range(len(names)) if srcs[z['src'][k]].endswith('_sam')]

    def key(k):
        ph = rows[names[k]]['photo']
        return (ph.rsplit('#', 1)[0], int(ph.rsplit('#', 1)[1])) if ph.startswith('parquet:') else (ph, 0)
    todo.sort(key=key)                                       # parquet files read once each
    masks, src = z['mask'].copy(), z['src'].copy()
    new_sources = list(srcs)
    log, st = [], {'exact': 0, 'rough': 0, 'quad': 0, 'skipped': 0}
    shown = []
    for t, k in enumerate(todo):
        r = rows[names[k]]
        w, h = photo_size(r['photo'])
        bx = z['box'][k]
        s = (bx[2] - bx[0]) / w
        if abs(s - SIZE / max(w, h)) > 0.01 * s:
            st['skipped'] += 1
            continue
        quad = np.asarray(r['corners'], np.float64) * s + bx[:2]
        m = np.unpackbits(masks[k])[:SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
        m2, worst, info = clean(m, quad, bx, a.wobble)
        level = 'exact' if worst <= a.wobble else 'rough' if worst <= a.rough else 'quad'
        if level == 'quad':   # SAM's outline is no use here: the label's fill, clipped to the photo
            qm = np.zeros((SIZE, SIZE), np.uint8)
            cv2.fillPoly(qm, [np.round(quad * 8).astype(np.int32)], 1, shift=3)
            box = np.zeros((SIZE, SIZE), bool)
            box[bx[1]:bx[3], bx[0]:bx[2]] = True
            m2 = (qm > 0) & box
        masks[k] = np.packbits(m2.ravel())
        if level != 'exact':
            base = srcs[z['src'][k]][:-4] + ('_samr' if level == 'rough' else '')
            if base not in new_sources:
                new_sources.append(base)
            src[k] = new_sources.index(base)
        st[level] += 1
        info.update(id=names[k], level=level)
        log.append(info)
        if a.show:
            shown.append((info['wobble'], k, m, m2))
        if (t + 1) % 2000 == 0:
            print(f'{t + 1}/{len(todo)} {st}', flush=True)
    z['mask'], z['src'], z['sources'] = masks, src, np.array(new_sources)
    np.savez(a.out, **z)
    with open(a.out[:-4] + '.clean.jsonl', 'w') as f:
        for e in log:
            f.write(json.dumps(e) + '\n')
    wb = np.array([e['wobble'] for e in log])
    print(f'{len(todo)} SAM outlines: {st}; wobble px p50 {np.median(wb):.2f} p90 {np.percentile(wb, 90):.2f} p99 {np.percentile(wb, 99):.2f}')
    if a.show and shown:
        shown.sort(key=lambda x: -x[0])
        pick = [shown[int(q * (len(shown) - 1))] for q in (0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)]
        tiles = []
        for wbl, k, m, m2 in pick:
            im = cv2.imdecode(z['jpeg'][z['off'][k]:z['off'][k + 1]], cv2.IMREAD_COLOR)
            im = cv2.resize(im, (768, 768), interpolation=cv2.INTER_CUBIC)
            for mm, col in ((m, (40, 220, 40)), (m2, (230, 60, 230))):
                cs, _ = cv2.findContours(cv2.resize(mm.astype(np.uint8), (768, 768), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, col, 1)
            cv2.putText(im, f'wobble {wbl:.2f}  {names[k][-24:]}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(im)
        cv2.imwrite(a.show, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('wrote', a.show)


if __name__ == '__main__':
    main()
