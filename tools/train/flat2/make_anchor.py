"""make_anchor.py OUT RUN_DIR [RUN_DIR ...] [--copy]: the anchor set for train_sdhr.py (learning without forgetting).

RUN_DIR: pipeline_cli outputs with FS_DUMP_OUTLINE (scratchpad ft.sh runs: NAME.json + NAME.outline.npz) on photos
whose licence allows training: Wikimedia Commons sets (CC / PD, licences.csv per folder), WildReceipt (Apache-2.0),
CORD (CC BY 4.0), inv3d real (MIT). NOT dir300, DocReal, WarpDoc, DocUNet (validation / test, or no licence).
OUT/index.jsonl: {"photo", "corners"} (the page finder's corners TL TR BR BL, photo px); photos the page finder did not
trust (needs_manual_crop) are left out. --copy: copy the photos into OUT/photos (for packing to the GPU box)."""
import argparse
import glob
import json
import os
import shutil

import numpy as np

BLOCK = ('dir300', 'docreal', 'DocReal', 'warpdoc', 'WarpDoc', 'docunet', 'DocUNet', 'round4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--copy', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, 'index.jsonl'), 'w') as f:
        for run in a.runs:
            for jf in sorted(glob.glob(os.path.join(run, '*.json'))):
                if jf.endswith('_env.json') or os.path.basename(jf).startswith('_'):
                    continue
                try:
                    j = json.load(open(jf))
                except Exception:
                    continue
                photo = j.get('photo', '')
                if not photo or any(b in photo for b in BLOCK) or j.get('needs_manual_crop'):
                    continue
                on = jf[:-5] + '.outline.npz'
                if not os.path.exists(on):
                    continue
                c = np.load(on)['corners'].tolist()
                if a.copy:
                    os.makedirs(os.path.join(a.out, 'photos'), exist_ok=True)
                    dst = os.path.join(a.out, 'photos', f'{n:05d}' + os.path.splitext(photo)[1])
                    shutil.copy(photo, dst)
                    photo = os.path.relpath(dst, a.out).replace(os.sep, '/')   # relative: the folder moves to the box
                f.write(json.dumps({'photo': photo, 'corners': c}) + '\n')
                n += 1
    print(n, 'anchor photos')


if __name__ == '__main__':
    main()
