"""prepare.py ROOT: put the real labelled sets into one index, ROOT/unified/index.jsonl, one JSON object per photo:

  id         <set>/<name>
  set        smartdoc | receipts | doccorner | mendeley
  source     the original source inside the set (DocCornerDataset: from the file name, e.g. midv500, receipts_train)
  licence    train (cleared for training) | eval (evaluation only); see ROOT/LICENSES.md
  split      train | val | test (never mixes one document across splits where the set says which document it is)
  photo      path of the photo (absolute)
  corners    [[x, y] x 4] TL TR BR BL in photo px, ordered by where they are in the photo (clockwise from the corner
             nearest the photo's top-left); null for a photo with no document
  negative   true for a photo with no document

  python prepare.py E:/fs_datasets/real_pf [--sets smartdoc,receipts,doccorner] [--every 4]
SmartDoc: every --every-th frame (neighbouring frames are near copies); split by document model: member 5 of each of
the 6 types -> test, member 4 -> val, the rest train. DocCornerDataset: its own splits; images written out of the
parquet files to ROOT/doccorner/img/<split>/."""
import argparse
import csv
import gzip
import io
import json
import os
import re

import numpy as np

SOURCES_LICENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'doccorner_sources.json')


def order_corners(q):
    """4 points -> TL TR BR BL by position in the photo (clockwise, starting nearest the photo's top-left)"""
    q = np.asarray(q, np.float64).reshape(4, 2)
    c = q.mean(0)
    a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])     # image y down: increasing angle = clockwise
    q = q[np.argsort(a)]
    k = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -k, 0).tolist()


def smartdoc(root, every):
    d = os.path.join(root, 'smartdoc')
    rows = []
    with gzip.open(os.path.join(d, 'metadata.csv.gz'), 'rt') as f:
        for r in csv.DictReader(f):
            if int(r['frame_index']) % every:
                continue
            sub = int(r['model_subid'])          # 0..4 within its type
            split = 'test' if sub == 4 else 'val' if sub == 3 else 'train'
            q = [[float(r['tl_x']), float(r['tl_y'])], [float(r['tr_x']), float(r['tr_y'])],
                 [float(r['br_x']), float(r['br_y'])], [float(r['bl_x']), float(r['bl_y'])]]
            name = r['image_path'].replace('/', '_').rsplit('.', 1)[0]
            rows.append(dict(id=f'smartdoc/{name}', set='smartdoc', source=f"smartdoc_{r['bg_name']}", licence='train',
                             split=split, photo=os.path.join(d, r['image_path']).replace('\\', '/'),
                             corners=order_corners(q), negative=False, doc=r['model_name']))
    return rows


def receipts(root):
    d = os.path.join(root, 'receipts')
    rows = []
    with open(os.path.join(d, 'homography.csv')) as f:
        for r in csv.DictReader(f, delimiter=';'):
            p = os.path.join(d, f"{r['Image']}-receipt.jpg")
            if not os.path.exists(p):
                continue
            q = [[float(v) for v in r[k].split(',')] for k in ('P1', 'P2', 'P3', 'P4')]
            rows.append(dict(id=f"receipts/{r['Image']}", set='receipts', source='express_expense_srd', licence='eval',
                             split='test', photo=p.replace('\\', '/'), corners=order_corners(q), negative=False))
    return rows


def doccorner(root):
    import pyarrow.parquet as pq
    lic = json.load(open(SOURCES_LICENCE)) if os.path.exists(SOURCES_LICENCE) else {}
    d = os.path.join(root, 'doccorner')
    rows = []
    # the photos stay inside the parquet files (writing 48k files to the spinning disk E: stalled everything):
    # photo = "parquet:<file>#<row>", read with read_photo()
    from PIL import Image
    for split, folder in (('train', 'train'), ('val', 'val'), ('test', 'test')):
        for f in sorted(os.listdir(os.path.join(d, folder))):
            if not f.endswith('.parquet'):
                continue
            path = os.path.join(d, folder, f).replace('\\', '/')
            row = 0
            for b in pq.ParquetFile(path).iter_batches(batch_size=256):
                for rec in b.to_pylist():
                    fn = rec['filename']
                    src = re.sub(r'_\d+\.\w+$', '', fn)
                    neg = bool(rec['is_negative'])
                    q = None
                    if not neg:
                        w, h = Image.open(io.BytesIO(rec['image']['bytes'])).size
                        q = order_corners([[rec[f'corner_{k}_x'] * w, rec[f'corner_{k}_y'] * h] for k in ('tl', 'tr', 'br', 'bl')])
                    rows.append(dict(id=f'doccorner/{fn.rsplit(".", 1)[0]}', set='doccorner', source=src,
                                     licence=lic.get(src, {}).get('use', 'eval'), split=split, photo=f'parquet:{path}#{row}',
                                     corners=q, negative=neg))
                    row += 1
    return rows



def mendeley(root):
    """CC BY 4.0, 1,094 phone photos; NAME.csv: x1,y1,...,x4,y4,"phone model". No document ids: split by name, 80/10/10"""
    import hashlib
    d = os.path.join(root, 'mendeley', 'raw', 'All My Dataset')
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.jpg') or not os.path.exists(os.path.join(d, f[:-4] + '.csv')):
            continue
        v = open(os.path.join(d, f[:-4] + '.csv')).read().strip().split(',')
        q = [[float(v[i]), float(v[i + 1])] for i in range(0, 8, 2)]
        h = int(hashlib.md5(f.encode()).hexdigest(), 16) % 10
        split = 'test' if h == 0 else 'val' if h == 1 else 'train'
        name = f[:-4].strip('()')
        rows.append(dict(id=f'mendeley/{name}', set='mendeley', source='mendeley', licence='train', split=split,
                         photo=os.path.join(d, f).replace(os.sep, '/'), corners=order_corners(q), negative=False))
    return rows

_PQ = {}


def read_photo(path):
    """a photo path from the index (a file, or parquet:<file>#<row>) -> BGR image"""
    import cv2
    if not path.startswith('parquet:'):
        return cv2.imread(path)
    f, row = path[len('parquet:'):].rsplit('#', 1)
    if f not in _PQ:
        import pyarrow.parquet as pq
        _PQ.clear()   # one file in memory at a time: read in index order
        _PQ[f] = pq.read_table(f, columns=['image']).column('image')
    b = _PQ[f][int(row)].as_py()['bytes']
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--sets', default='smartdoc,receipts,mendeley,doccorner'); ap.add_argument('--every', type=int, default=4)
    a = ap.parse_args()
    out = os.path.join(a.root, 'unified')
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in a.sets.split(','):
        r = smartdoc(a.root, a.every) if s == 'smartdoc' else receipts(a.root) if s == 'receipts' else mendeley(a.root) if s == 'mendeley' else doccorner(a.root)
        print(s, len(r), 'photos;', {k: sum(x['split'] == k for x in r) for k in ('train', 'val', 'test')},
              'licence train:', sum(x['licence'] == 'train' for x in r), flush=True)
        rows += r
    with open(os.path.join(out, 'index.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(rows))
