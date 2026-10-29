"""Page finder v2 data: a "pack" holds every sample in RAM, ready for the GPU.

  jpeg     the letterboxed photo (SIZE x SIZE, BGR) as JPEG bytes (q 95; ~30 KB), decoded on the GPU in batches
  mask     the paper mask, bit-packed (SIZE * SIZE / 8 bytes)
  box      x0, y0, x1, y1 of the photo inside the letterbox (the rest is padding)
  src      source id (names in `sources`), for per-source weights and reports

Packs are built once from (a) make_both.py page-finder shards (OUT/pf/*.npz) and (b) the real-photo index of
tools/pagefind_bench/prepare.py (a filled quad as the mask), and saved as one .npz (fast to load: E: is a spinning disk).

  python data.py synth OUT.npz SHARD_DIR [SHARD_DIR ...]
  python data.py real OUT.npz INDEX.jsonl --split train --licence train [--sets smartdoc,doccorner] [--max_per_source N]
  python data.py negcrops OUT.npz INDEX.jsonl --split train --licence train [--max_per_source N]
  python data.py merge OUT.npz A.npz B.npz ..."""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import SIZE, letterbox  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pagefind_bench"))
from prepare import read_photo  # noqa: E402

NBITS = SIZE * SIZE // 8


class Pack:
    def __init__(self):
        self.jpeg, self.mask, self.box, self.src, self.name, self.sources = [], [], [], [], [], []

    def add(self, img_lb, mask_lb, box, source, name):
        if source not in self.sources:
            self.sources.append(source)
        ok, b = cv2.imencode('.jpg', img_lb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.jpeg.append(b.ravel())
        self.mask.append(np.packbits((mask_lb > 127).ravel()))
        self.box.append(box)
        self.src.append(self.sources.index(source))
        self.name.append(name)

    def save(self, path):
        lens = np.array([len(b) for b in self.jpeg], np.int64)
        np.savez(path, jpeg=np.concatenate(self.jpeg) if self.jpeg else np.zeros(0, np.uint8),
                 off=np.concatenate([[0], np.cumsum(lens)]), mask=np.stack(self.mask), box=np.array(self.box, np.int32),
                 src=np.array(self.src, np.int16), name=np.array(self.name), sources=np.array(self.sources))
        print('saved', path, len(self.jpeg), 'samples', {s: int((np.array(self.src) == i).sum()) for i, s in enumerate(self.sources)})


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def box_of(valid):
    ys, xs = np.nonzero(valid)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def from_synth(dirs, out):
    p = Pack()
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            # each z[key] decompresses the whole array again: read every array once per shard, and copy the slices
            # kept (a view kept the shard's whole array alive: 12 GB for 9k photos, round 19)
            try:
                with np.load(f) as zf:
                    z = {k: zf[k] for k in zf.files}
                meta = json.loads(str(z['meta']))
            except Exception:   # a shard still being written
                continue
            if 'jpeg' in z:   # compact shards: already JPEG + packed bits
                for k in range(len(z['box'])):
                    src = 'synth_cut' if meta[k]['cut'] else 'synth'
                    if src not in p.sources:
                        p.sources.append(src)
                    p.jpeg.append(z['jpeg'][z['off'][k]:z['off'][k + 1]].copy()); p.mask.append(z['mask'][k].copy()); p.box.append(list(z['box'][k]))
                    p.src.append(p.sources.index(src)); p.name.append(f'{os.path.basename(f)[:-4]}_{k}')
                continue
            for k in range(len(z['img'])):
                src = 'synth_cut' if meta[k]['cut'] else 'synth'
                p.add(z['img'][k], z['mask'][k], box_of(z['valid'][k]), src, f'{os.path.basename(f)[:-4]}_{k}')
    p.save(out)


def from_real(index, out, split, licence, sets, max_per_source, seed=0):
    rows = [json.loads(l) for l in open(index)]
    rows = [r for r in rows if r['split'] == split and (licence == 'any' or r['licence'] == licence)
            and (not sets or r['set'] in sets)]
    rng = np.random.default_rng(seed)
    if max_per_source:
        by = {}
        for r in rows:
            by.setdefault(r['source'], []).append(r)
        rows = []
        for s, rs in by.items():
            idx = rng.permutation(len(rs))[:max_per_source]
            rows += [rs[i] for i in idx]
    # read in file order (DocCornerDataset photos sit inside parquet files: one file in memory at a time)
    def key(r):
        ph = r['photo']
        if ph.startswith('parquet:'):
            f, row = ph.rsplit('#', 1)
            return (f, int(row))
        return (ph, 0)
    rows.sort(key=key)
    p = Pack()
    for k, r in enumerate(rows):
        im = read_photo(r["photo"])
        if im is None:
            continue
        m = np.zeros(im.shape[:2], np.uint8)
        if not r['negative']:
            cv2.fillPoly(m, [np.round(np.array(r['corners'])).astype(np.int32)], 255)
        lb, (s, x0, y0) = letterbox(im)
        mb, _ = letterbox(m)
        h, w = im.shape[:2]
        box = [x0, y0, x0 + int(round(w * s)), y0 + int(round(h * s))]
        p.add(lb, mb, box, r['source'] if r['set'] == 'doccorner' else r['set'], r['id'])
        if k % 2000 == 0:
            print(k, '/', len(rows), flush=True)
    p.save(out)


def from_negcrops(index, out, split, licence, max_per_source, seed=0):
    """licence-clean "no page" photos: the largest strip of a training photo clear of its page (the quad's box plus a
    margin of 3 % of the photo's diagonal), whole height or width, at least a quarter of the photo each way, short side
    >= 160 px, sides at most 1 : 3. (DocCorner's COCO negatives are evaluation only: Flickr licences.)"""
    rows = [json.loads(l) for l in open(index)]
    rows = [r for r in rows if r['split'] == split and r['licence'] == licence and not r['negative']]
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    rows = []
    for s, rs in by.items():
        rows += [rs[i] for i in rng.permutation(len(rs))[:max_per_source]] if max_per_source else rs

    def key(r):
        ph = r['photo']
        if ph.startswith('parquet:'):
            f, row = ph.rsplit('#', 1)
            return (f, int(row))
        return (ph, 0)
    rows.sort(key=key)
    p = Pack()
    for k, r in enumerate(rows):
        im = read_photo(r['photo'])
        if im is None:
            continue
        H, W = im.shape[:2]
        q = np.array(r['corners'], np.float64)
        mg = 0.03 * np.hypot(W, H)
        qx0, qy0 = q.min(0) - mg
        qx1, qy1 = q.max(0) + mg
        cands = [(0, 0, int(qx0), H), (int(np.ceil(qx1)), 0, W, H), (0, 0, W, int(qy0)), (0, int(np.ceil(qy1)), W, H)]
        best = None
        for x0, y0, x1, y1 in cands:
            w, h = x1 - x0, y1 - y0
            if w < 0.25 * W or h < 0.25 * H or min(w, h) < 160 or max(w, h) > 3 * min(w, h):
                continue
            if best is None or w * h > (best[2] - best[0]) * (best[3] - best[1]):
                best = (x0, y0, x1, y1)
        if best is None:
            continue
        x0, y0, x1, y1 = best
        crop = np.ascontiguousarray(im[y0:y1, x0:x1])
        lb, (s, ox, oy) = letterbox(crop)
        h, w = crop.shape[:2]
        p.add(lb, np.zeros(lb.shape[:2], np.uint8), [ox, oy, ox + int(round(w * s)), oy + int(round(h * s))], 'negcrop', r['id'] + '_neg')
        if k % 2000 == 0:
            print(k, '/', len(rows), len(p.jpeg), 'kept', flush=True)
    p.save(out)


def merge(out, ins):
    p = Pack()
    for f in ins:
        z = load(f)
        srcs = list(z['sources'])
        for k in range(len(z['box'])):
            s = srcs[z['src'][k]]
            if s not in p.sources:
                p.sources.append(s)
            p.jpeg.append(z['jpeg'][z['off'][k]:z['off'][k + 1]])
            p.mask.append(z['mask'][k]); p.box.append(list(z['box'][k])); p.src.append(p.sources.index(s)); p.name.append(z['name'][k])
    p.save(out)


if __name__ == '__main__' and sys.argv[1] != 'maskdir':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('out'); ap.add_argument('ins', nargs='*')
    ap.add_argument('--split', default='train'); ap.add_argument('--licence', default='train'); ap.add_argument('--sets', default='')
    ap.add_argument('--max_per_source', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'synth':
        from_synth(a.ins, a.out)
    elif a.cmd == 'real':
        from_real(a.ins[0], a.out, a.split, a.licence, [s for s in a.sets.split(',') if s], a.max_per_source)
    elif a.cmd == 'negcrops':
        from_negcrops(a.ins[0], a.out, a.split, a.licence, a.max_per_source)
    else:
        merge(a.out, a.ins)


def from_maskdir(folder, out, limit=0):
    """make_synth.py output (NAME.jpg + NAME.png): exact masks"""
    p = Pack()
    names = sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.png'))
    if limit:
        names = names[:limit]
    for n in names:
        im = cv2.imread(os.path.join(folder, n + '.jpg'))
        m = cv2.imread(os.path.join(folder, n + '.png'), 0)
        lb, (s, x0, y0) = letterbox(im)
        mb, _ = letterbox(m)
        h, w = im.shape[:2]
        meta = json.load(open(os.path.join(folder, n + '.json')))
        p.add(lb, mb, [x0, y0, x0 + int(round(w * s)), y0 + int(round(h * s))], 'synth_cut' if meta['cut'] else 'synth', n)
    p.save(out)


if __name__ == '__main__' and sys.argv[1] == 'maskdir':
    from_maskdir(sys.argv[3], sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
