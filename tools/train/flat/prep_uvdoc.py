"""Index the UVDoc dataset (UVDoc_final, MIT licence) for train.py: every photo with its ground-truth
backward grid (where each point of the flat page lies in the photo).

  python prep_uvdoc.py UVDOC_DIR OUT_DIR [--val 0.05]
UVDOC_DIR holds img/*.png (712 x 488), metadata_sample/*.json and grid2d/*.mat (fetch_zip.py fetches only these).
OUT_DIR/index.json  [{"id", "img", "val"}]   (val: a held-out geometry - the page shapes, not just the photos, are new)
OUT_DIR/grids.npy   float32 N x 2 x 89 x 61  (x, y in photo px; rows = page top -> bottom)"""
import argparse, glob, hashlib, json, os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--val', type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items, grids = [], []
    for f in sorted(glob.glob(os.path.join(a.src, 'metadata_sample', '*.json'))):
        m = json.load(open(f))
        img = os.path.join(a.src, 'img', m['sample_id'] + '.png')
        g = os.path.join(a.src, 'grid2d', m['geom_name'] + '.mat')
        if not (os.path.exists(img) and os.path.exists(g)):
            continue
        with h5py.File(g, 'r') as h:
            grid = np.array(h['grid2d'][:]).T.transpose(2, 0, 1).astype(np.float32)   # as UVDoc's data_UVDoc.py
        base = '_'.join(m['geom_name'].split('_')[:3])   # the captured shape; its variants share it
        val = int(hashlib.md5(base.encode()).hexdigest(), 16) % 10000 < a.val * 10000
        items.append({'id': m['sample_id'], 'img': os.path.relpath(img, a.out), 'val': bool(val)})
        grids.append(grid)
    np.save(os.path.join(a.out, 'grids.npy'), np.stack(grids))
    json.dump(items, open(os.path.join(a.out, 'index.json'), 'w'))
    print(len(items), 'samples,', sum(i['val'] for i in items), 'held out')


if __name__ == '__main__':
    main()
