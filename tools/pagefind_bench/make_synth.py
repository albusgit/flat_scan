"""make_synth.py PAGES OUT N [--workers W] [--seed S] [--cut P]: synthetic photos with an exact paper mask, for the
page-finder benchmark (tools/pagefind_bench/bench.py). Uses the SD-HR generator (tools/train/flat2/synth.py): bent
sheets, desks, clutter, shadows, glare, JPEG; a share P of the photos (default 0.35) has the page cut by the frame.
Use held-out pages (E:/fs_flat2/pages2_held), never the training pages.

  OUT/<id>.jpg   the photo
  OUT/<id>.png   the paper mask (0 / 255; what the camera sees of the sheet)
  OUT/<id>.json  kind, aspect, hw, cut, the page's 4 corners in photo px (TL, TR, BR, BL; may lie outside the photo)
                 and per page side (top, right, bottom, left) the share of that side outside the photo"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT2 = os.path.join(HERE, '..', 'train', 'flat2')


def init(cut):
    os.environ['FS_CUT_P'] = str(cut)
    sys.path.insert(0, FLAT2)
    import cv2
    cv2.setNumThreads(1)


def job(a):
    import cv2
    import numpy as np
    import glob
    import synth
    pages_dir, out, seed, count = a
    pages = sorted(glob.glob(os.path.join(pages_dir, '**', '*.png'), recursive=True) +
                   glob.glob(os.path.join(pages_dir, '**', '*.jpg'), recursive=True))
    rng = np.random.default_rng(seed)
    done = 0
    while done < count:
        s = synth.sample(rng, pages)
        if s is None:
            continue
        sid = f'{seed:05d}_{done:03d}'
        g = s['grid']                      # 2 x GH x GW, photo px
        H, W = s['photo'].shape[:2]
        def outside(px):
            return float(np.mean((px[0] < 0) | (px[1] < 0) | (px[0] > W - 1) | (px[1] > H - 1)))
        sides = [outside(g[:, 0, :]), outside(g[:, :, -1]), outside(g[:, -1, :]), outside(g[:, :, 0])]
        corners = [g[:, 0, 0], g[:, 0, -1], g[:, -1, -1], g[:, -1, 0]]
        cv2.imwrite(os.path.join(out, sid + '.jpg'), s['photo'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out, sid + '.png'), (s['mask'] * 255).astype(np.uint8))
        json.dump(dict(kind=s['kind'], aspect=s['aspect'], hw=float(s['hw']), cut=s['cut'], visible=s['visible'],
                       corners=[[float(c[0]), float(c[1])] for c in corners], side_outside=sides),
                  open(os.path.join(out, sid + '.json'), 'w'))
        done += 1
    return seed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pages'); ap.add_argument('out'); ap.add_argument('n', type=int)
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--seed', type=int, default=9000)
    ap.add_argument('--cut', type=float, default=0.35)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = 20
    jobs = [(a.pages, a.out, a.seed + k, min(per, a.n - k * per)) for k in range((a.n + per - 1) // per)]
    with Pool(a.workers, initializer=init, initargs=(a.cut,)) as pool:
        for k, s in enumerate(pool.imap_unordered(job, jobs)):
            print(f'{(k + 1) * per}/{a.n}', flush=True)
