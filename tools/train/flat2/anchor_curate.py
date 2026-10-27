"""anchor_curate.py IN OUT [--min_iou 0.9] [--cap set=n ...] [--long 2000]: the anchor set for train_run2.py, curated and
copied for a GPU box.

Anchors teach the fine-tune to keep what the original net does (the teacher's grid is the target), so only photos
where the teacher is plausibly right are kept: the original SD-HR's page outline on the app's crop (UVDoc pass-1
corners + 6 % margin) must agree with pass 1's own page (IoU >= --min_iou). --cap limits a set (receipts were 70 %
of the run-1 anchors). Photos are copied into OUT/photos (long side <= --long) with corners scaled; OUT/index.jsonl
paths are relative, so the folder can move."""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'flatnet'), os.path.join(HERE, '..', 'flat')]
import common as C  # noqa: E402
from run2_core import bm_forward  # noqa: E402
from train_sdhr import IN_H, IN_W, app_crop  # noqa: E402


def outline_iou(a, b, size):
    w, h = size
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    return (ma & mb).sum() / max(1, (ma | mb).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--min_iou', type=float, default=0.9)
    ap.add_argument('--cap', nargs='*', default=[])
    ap.add_argument('--long', type=int, default=2000)
    a = ap.parse_args()
    caps = {k: int(v) for k, v in (c.split('=') for c in a.cap)}
    rows = [json.loads(l) for l in open(os.path.join(a.inp, 'index.jsonl'))]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    net = C.load_sdhr(C.SD_CKPT, 'cuda')
    os.makedirs(os.path.join(a.out, 'photos'), exist_ok=True)
    kept, seen, ious = [], collections.Counter(), []
    for r in rows:
        s = r.get('set', '')
        if s in caps and seen[s] >= caps[s]:
            continue
        img = cv2.imread(r['photo'] if os.path.isabs(r['photo']) else os.path.join(a.inp, r['photo']))
        if img is None:
            continue
        q = np.array(r['corners'], np.float32)
        sp = np.linspace(0, 1, C.GW)[None, :, None]
        tp = np.linspace(0, 1, C.GH)[:, None, None]
        g = (q[0] * (1 - sp) * (1 - tp) + q[1] * sp * (1 - tp) + q[2] * sp * tp + q[3] * (1 - sp) * tp).transpose(2, 0, 1)
        x, t, turned = app_crop(img, g.astype(np.float32), np.random.default_rng(1), margin=(0.06, 0.06), jitter=0)
        xt = (torch.from_numpy(x).permute(2, 0, 1).float().div(255) * 2 - 1)[None].cuda()
        with torch.no_grad():
            p = bm_forward(net, xt)[0].cpu().numpy()
        # the teacher's outline and pass 1's page, both in the net input's px
        border = lambda G: np.concatenate([G[:, 0, :].T, G[:, :, -1].T, G[:, -1, ::-1].T, G[:, ::-1, 0].T])
        px = lambda G: np.stack([(G[0] + 1) / 2 * (IN_W - 1), (G[1] + 1) / 2 * (IN_H - 1)])
        iou = outline_iou(border(px(p)), border(px(t)), (IN_W, IN_H))
        ious.append(iou)
        if iou < a.min_iou:
            continue
        k = min(1.0, a.long / max(img.shape[:2]))
        if k < 1:
            img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        name = f'photos/{len(kept):05d}.jpg'
        cv2.imwrite(os.path.join(a.out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        kept.append({'photo': name, 'corners': (q * k).round(1).tolist(), 'set': s, 'teacher_iou': round(float(iou), 4)})
        seen[s] += 1
    with open(os.path.join(a.out, 'index.jsonl'), 'w') as f:
        for r in kept:
            f.write(json.dumps(r) + '\n')
    print(f'{len(kept)} kept of {len(ious)} checked (teacher IoU median {np.median(ious):.3f}); per set {dict(seen)}')


if __name__ == '__main__':
    main()
