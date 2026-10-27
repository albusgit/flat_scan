"""anchor_uv.py OUT SET[:TAG] [SET[:TAG] ...] [--exclude LIST] [--max N]: the anchor set for train_sdhr.py from the
test library's licence-clean photo sets (E:/fs_datasets/sets/<SET>.json), framed the way the app frames SD-HR's
input: UVDoc pass 1 (the shipped ONNX) on the whole photo, its grid's corners. No pipeline runs needed (make_anchor.py
does the same from pipeline_cli outputs, with the page finder's corners).

Only sets whose licence allows training: commons_* (CC / PD, per file), wildreceipt (Apache-2.0), cord (CC BY 4.0).
NOT dir300, docreal, warpdoc, DocUNet, inv3dreal (kept for testing). --exclude: a list of photo paths (ft all.txt)
whose file names are left out too (the 121 review photos are copies of commons photos).
OUT/index.jsonl: {"photo" (absolute), "corners" TL TR BR BL photo px}."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'flat'))
import realval as rv  # noqa: E402

BLOCK = ('dir300', 'docreal', 'warpdoc', 'docunet', 'inv3d', 'round4', 'real35', 'uvdoc_bench')
ONNX = os.path.join(HERE, '..', '..', '..', 'app', 'src', 'main', 'assets', 'models', 'uvdoc.onnx')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('sets', nargs='+')
    ap.add_argument('--root', default='E:/fs_datasets')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--max', type=int, default=0, help='at most this many photos per set (0: all)')
    a = ap.parse_args()
    skip = set()
    if a.exclude:
        for line in open(a.exclude):
            b = os.path.splitext(os.path.basename(line.strip()))[0]
            skip.add(b[len('commons_'):] if b.startswith('commons_') else b)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2   # runs beside the data generator
    sess = ort.InferenceSession(ONNX, so, providers=['CPUExecutionProvider'])
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, 'index.jsonl'), 'w') as f:
        for spec in a.sets:
            name, _, tag = spec.partition(':')
            if any(b in name.lower() for b in BLOCK):
                sys.exit(f'{name}: not a training set')
            items = json.load(open(os.path.join(a.root, 'sets', name + '.json')))['items']
            k = 0
            for it in items:
                if tag and tag not in it.get('tags', []):
                    continue
                if os.path.splitext(os.path.basename(it['photo']))[0] in skip:
                    continue
                img = cv2.imread(it['photo'])
                if img is None or min(img.shape[:2]) < 400:
                    continue
                h, w = img.shape[:2]
                g = sess.run(None, {'image': rv.net_input(img)})[0][0]
                gx, gy = rv.to_px(g, w, h)
                c = [[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]]
                c = np.clip(np.array(c, np.float64), [0, 0], [w - 1, h - 1])
                # a degenerate frame (a sliver) is no use as a crop
                if cv2.contourArea(c.astype(np.float32)) < 0.04 * w * h:
                    continue
                f.write(json.dumps({'photo': it['photo'], 'corners': c.round(1).tolist(), 'set': name}) + '\n')
                n += 1
                k += 1
                if a.max and k >= a.max:
                    break
            print(name, k, flush=True)
    print(n, 'anchor photos')


if __name__ == '__main__':
    main()
