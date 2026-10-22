"""WarpDoc (Xue et al., FDRNet, CVPR 2022; HF mirror 34data/WarpDoc_merged) -> working copy.

  python prep_warpdoc.py PARQUET OUT_DIR

Writes OUT_DIR/photo/<type>_<id>.jpg (camera photo, longest side <= 2000 px) and OUT_DIR/flat/<type>_<id>.png
(the digital original, height 1400 px, grayscale) plus OUT_DIR/pairs.json. 'incomplete' pairs use the
'digitalmargin' original (the part of the page that is in the photo)."""
import io
import json
import os
import sys

import cv2
import numpy as np

if os.environ.get('FS_PYLIB'):          # pyarrow installed with pip --target (the shared venv has none)
    sys.path.append(os.environ['FS_PYLIB'])
import pyarrow.parquet as pq  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, 'photo'), exist_ok=True)
os.makedirs(os.path.join(out, 'flat'), exist_ok=True)
pf = pq.ParquetFile(src)
photos, flats, flatm = {}, {}, {}
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=['image', 'filename']).to_pydict()
    for b, fn in zip(t['image'], t['filename']):
        stem = fn.rsplit('.', 1)[0]
        kind, rest = stem.split('_', 1)
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            print('bad', fn)
            continue
        if kind == 'image':
            s = 2000.0 / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out, 'photo', rest + '.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            photos[rest] = im.shape[:2]
        else:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            s = 1400.0 / g.shape[0]
            g = cv2.resize(g, (max(8, round(g.shape[1] * s)), 1400), interpolation=cv2.INTER_AREA)
            name = rest + ('__m' if kind == 'digitalmargin' else '')
            cv2.imwrite(os.path.join(out, 'flat', name + '.png'), g)
            (flatm if kind == 'digitalmargin' else flats)[rest] = g.shape[:2]
    print('row group', rg, len(photos), len(flats), len(flatm), flush=True)
pairs = []
for k in sorted(photos):
    f = (k + '__m') if k in flatm else (k if k in flats else None)
    if f is None:
        continue
    pairs.append({'id': k, 'type': k.split('_')[0], 'photo': f'photo/{k}.jpg', 'flat': f'flat/{f}.png'})
json.dump(pairs, open(os.path.join(out, 'pairs.json'), 'w'), indent=0)
print('pairs', len(pairs), {t: sum(p['type'] == t for p in pairs) for t in sorted({p['type'] for p in pairs})})
