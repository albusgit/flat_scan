"""sam_pack.py INDEX.jsonl OUT.npz [--split train --licence train --max_per_source N]: the real training photos' masks from
SAM 2.1 (sam_label.py) instead of their 4-corner fill, as a page finder v2 pack (pagefind2/data.py layout).

Per photo: SAM on the photo (long side 1600), prompted by the label's quad. Kept where it agrees with the label as
paper can (sam_label.label: outline within delta of the quad, corners met, IoU >= 0.85), with delta per source as
the trainer's (pagefind2/train.py source_delta: what a straight-sided label can miss on that kind of paper, at least
2 %). Where the quad runs along the photo's own edge, the mask is the quad's there (SAM bumps along the frame).
Kept photos get source NAME_sam (the trainer then trusts them as exact); rejected ones keep the quad's fill and the
source NAME (the interval loss). Resumable: every 500 photos a part file OUT.partNN.npz; done parts are skipped."""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'train', 'pagefind2'))
import sam_label as L  # noqa: E402
import data as D  # noqa: E402
from net import letterbox  # noqa: E402
from prepare import read_photo  # noqa: E402
from train import source_delta  # noqa: E402

PART = 500


def rows_of(index, split, licence, max_per_source, seed=0):
    rows = [json.loads(l) for l in open(index)]
    rows = [r for r in rows if r['split'] == split and r['licence'] == licence and not r['negative']]
    rng = np.random.default_rng(seed)
    if max_per_source:
        by = {}
        for r in rows:
            by.setdefault(r['source'], []).append(r)
        rows = []
        for s, rs in by.items():
            rows += [rs[i] for i in rng.permutation(len(rs))[:max_per_source]]

    def key(r):
        ph = r['photo']
        if ph.startswith('parquet:'):
            f, row = ph.rsplit('#', 1)
            return (f, int(row))
        return (ph, 0)
    return sorted(rows, key=key)


def one(pred, im, quad, delta):
    H, W = im.shape[:2]
    s = min(1.0, 1600 / max(H, W))
    small = cv2.resize(im, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else im
    q = np.asarray(quad, np.float64) * s
    with torch.autocast('cuda', dtype=torch.bfloat16):
        m, info = L.label(pred, small, q, delta=delta, sides=False)
    info.pop('raw', None)
    if m is None:
        return None, info
    qm = L.quad_mask(small.shape, q)
    band = max(3, int(0.004 * max(small.shape[:2])))
    frame = np.zeros_like(qm)
    frame[:band] = frame[-band:] = True
    frame[:, :band] = frame[:, -band:] = True
    m = np.where(frame, qm, m)                      # along the photo's own edge: the label's fill
    m = L.largest(m)
    if s < 1:
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR) > 0
    return m, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index'); ap.add_argument('out')
    ap.add_argument('--split', default='train'); ap.add_argument('--licence', default='train')
    ap.add_argument('--max_per_source', type=int, default=1500)
    a = ap.parse_args()
    rows = rows_of(a.index, a.split, a.licence, a.max_per_source)
    print(len(rows), 'photos', flush=True)
    pred = L.predictor()
    stats = {'kept': 0, 'quad': 0, 'unread': 0}
    t0 = time.time()
    for pi, lo in enumerate(range(0, len(rows), PART)):
        part = f'{a.out[:-4]}.part{pi:03d}.npz'
        if os.path.exists(part):
            continue
        p = D.Pack()
        log = []
        for r in rows[lo:lo + PART]:
            im = read_photo(r['photo'])
            if im is None:
                stats['unread'] += 1
                continue
            d = max(0.02, source_delta(r['source'] if r['set'] == 'doccorner' else r['set']))
            m, info = one(pred, im, r['corners'], d)
            src = r['source'] if r['set'] == 'doccorner' else r['set']
            if m is None and info.get('unsure'):
                stats['dropped'] = stats.get('dropped', 0) + 1
                log.append(dict(id=r['id'], src='dropped', **info))
                continue
            if m is None:
                m = L.quad_mask(im.shape, r['corners'])
                stats['quad'] += 1
            else:
                src += '_sam'
                stats['kept'] += 1
            lb, (s, x0, y0) = letterbox(im)
            mb, _ = letterbox((m * 255).astype(np.uint8))
            h, w = im.shape[:2]
            p.add(lb, mb, [x0, y0, x0 + int(round(w * s)), y0 + int(round(h * s))], src, r['id'])
            log.append(dict(id=r['id'], src=src, **info))
        p.save(part)
        with open(part[:-4] + '.jsonl', 'w') as f:
            for e in log:
                f.write(json.dumps(e) + '\n')
        n = lo + PART
        print(f'{min(n, len(rows))}/{len(rows)}  {stats}  {(time.time() - t0) / 60:.0f} min', flush=True)
    D.merge(a.out, sorted(glob.glob(f'{a.out[:-4]}.part*.npz')))
    print('done', stats, flush=True)


if __name__ == '__main__':
    main()
