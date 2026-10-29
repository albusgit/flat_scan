"""make_test.py DATA RUN OUT: the pilot's test set from an oracle run (held-out page sources): DATA from
make_oracle.py (photo + G at 641 points across), RUN the stage-1 maps (FS_DUMP_MAP). Same layout as make_data.py."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import targets  # noqa: E402

data, run, out = sys.argv[1:4]
for sub in ('in', 't', 'map'):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
with open(os.path.join(out, 'index.jsonl'), 'w') as index:
    for line in open(os.path.join(data, 'index.jsonl')):
        row = json.loads(line)
        sid = row['id']
        mp = os.path.join(run, sid + '.map.npz')
        if not os.path.exists(mp):
            row['lost'] = 'stage1'
        else:
            with np.load(mp) as B:
                bx, by = B['x'], B['y']
            G = np.load(os.path.join(data, sid + '.npz'))['G'][:, ::2, ::2]
            np.savez_compressed(os.path.join(out, 'map', sid + '.npz'), x=bx, y=by, G=G)
            r = targets.build(bx, by, G)
            if r is None:
                row['lost'] = 'page'
            else:
                photo = cv2.imread(os.path.join(data, sid + '.jpg'))
                cv2.imwrite(os.path.join(out, 'in', sid + '.jpg'), targets.render(photo, bx, by), [cv2.IMWRITE_JPEG_QUALITY, 92])
                np.savez_compressed(os.path.join(out, 't', sid + '.npz'), d=r['d'], valid=r['valid'], M=r['M'], frame=np.int32(r['frame']))
                row['stats'] = r['stats']
        index.write(json.dumps(row) + '\n')
print('done')
