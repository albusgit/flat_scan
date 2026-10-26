"""make_data.py PAGES OUT N [--workers W] [--seed S] [--kinds k1,k2] : pre-generate SD-HR training samples.

Each sample (synth.sample + synth.app_input) is stored as
  OUT/<shard>/<id>.jpg   the net's input as the app makes it: 1024 x 1440 RGB crop (JPEG q 93)
  OUT/<shard>/<id>.npy   the target grid, float16 2 x 121 x 177 in [-1, 1] of that input
  OUT/index.jsonl        {"id", "path", "kind", "aspect", "hw", "turned"} per sample
Generation is CPU-bound (~1-3 s a sample a core): run it with all cores while the GPU trains on an earlier part
(train_sdhr.py re-reads index.jsonl every epoch). PAGES: a folder of flat page images (textures.py)."""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def work(job):
    import synth
    synth.DEVICE = 'cpu'   # many processes: the CPU rasteriser each (the GPU is the trainer's)
    pages, out, seed, n, kinds = job
    rng = np.random.default_rng(seed)
    shard = os.path.join(out, f's{seed:06d}')
    os.makedirs(shard, exist_ok=True)
    rows = []
    i = 0
    while len(rows) < n and i < n * 4:
        i += 1
        kind = kinds[rng.integers(len(kinds))] if kinds else None
        try:
            sm = synth.sample(rng, pages, kind)
        except Exception as e:   # a degenerate shape: skip it
            print('sample failed:', e, flush=True)
            continue
        if sm is None:
            continue
        ai = synth.app_input(sm, rng)
        if ai is None:
            continue
        x, t, turned = ai
        sid = f'{seed:06d}_{len(rows):04d}'
        cv2.imwrite(os.path.join(shard, sid + '.jpg'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        np.save(os.path.join(shard, sid + '.npy'), t.astype(np.float16))
        rows.append({'id': sid, 'path': os.path.relpath(os.path.join(shard, sid), out).replace('\\', '/'), 'kind': sm['kind'],
                     'aspect': sm['aspect'], 'hw': round(float(sm['hw']), 4), 'turned': bool(turned)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages')
    ap.add_argument('out')
    ap.add_argument('n', type=int)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per_job', type=int, default=50)
    ap.add_argument('--kinds', default='')
    a = ap.parse_args()
    pages = sorted(glob.glob(os.path.join(a.pages, '**', '*.png'), recursive=True) +
                   glob.glob(os.path.join(a.pages, '**', '*.jpg'), recursive=True))
    os.makedirs(a.out, exist_ok=True)
    kinds = [k for k in a.kinds.split(',') if k]
    jobs = [(pages, a.out, a.seed + j, a.per_job, kinds) for j in range((a.n + a.per_job - 1) // a.per_job)]
    with Pool(a.workers) as pool, open(os.path.join(a.out, 'index.jsonl'), 'a') as idx:
        done = 0
        for rows in pool.imap_unordered(work, jobs):
            for r in rows:
                idx.write(json.dumps(r) + '\n')
            idx.flush()
            done += len(rows)
            print(f'{done} samples', flush=True)


if __name__ == '__main__':
    main()
