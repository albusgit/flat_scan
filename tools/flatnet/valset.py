"""Validation of a flattening-net checkpoint on photo / flat-scan sets (never DocUNet: it is the test set).

  python valset.py crops SET_JSON OUT_DIR [--limit N]          # fixed net crops from the ORIGINAL SD-HR (once)
  python valset.py run SET_JSON OUT_DIR TAG CKPT|orig|ONNX     # net-only unwarp -> OUT_DIR/TAG/<id>.png
  python valset.py score SET_JSON OUT_DIR TAG [--jobs 3]       # DocUNet scorer (docunet_bench/docunet_eval.py)

The crop is the app's rule (train.app_crop, margin 6 %, turned when wider than tall) around the page that the
original SD-HR finds in the whole photo, so every checkpoint sees exactly the same inputs. The unwarp is the
official SyntheticDoc inference: grid upsampled (bilinear, align_corners) to the crop size, grid_sample."""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from train import app_crop  # noqa: E402

BENCH = r'C:\Users\User\Desktop\docunet_bench'


def load_items(set_json, limit):
    s = json.load(open(set_json))
    items = [it for it in s['items'] if it.get('gt')]
    if limit:
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    return items


def crops(a):
    os.makedirs(a.out, exist_ok=True)
    net = C.load_sdhr()
    res = {}
    for it in load_items(a.set, a.limit):
        im = cv2.imread(it['photo'])[:, :, ::-1].copy()
        h, w = im.shape[:2]
        bm = C.bm_to_px(C.predict_bm(net, im), w, h)
        box, turned = app_crop(bm, w, h, 0.06)
        res[it['id']] = {'box': list(map(int, box)), 'turned': turned, 'photo': it['photo'], 'gt': it['gt']}
    json.dump(res, open(os.path.join(a.out, 'crops.json'), 'w'), indent=0)
    print('crops', len(res))


class OnnxNet:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

    def __call__(self, x01):
        return self.s.run(['grid'], {'image': x01})[0][0]


def get_net(spec):
    if spec.endswith('.onnx'):
        return OnnxNet(spec)
    net = C.load_sdhr(C.SD_CKPT if spec == 'orig' else spec)
    def f(x01):
        with torch.no_grad():
            return C.bm_only_forward(net, torch.from_numpy(x01).cuda() * 2 - 1)[0].float().cpu().numpy()
    return f


def run(a):
    cr = json.load(open(os.path.join(a.out, 'crops.json')))
    od = os.path.join(a.out, a.tag)
    os.makedirs(od, exist_ok=True)
    f = get_net(a.model)
    grids = {}
    for k, c in cr.items():
        im = cv2.imread(c['photo'])
        x0, y0, x1, y1 = c['box']
        crop = im[y0:y1, x0:x1]
        if c['turned']:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        small = cv2.resize(crop, (C.IN_W, C.IN_H), interpolation=cv2.INTER_AREA)
        x = (small[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()
        g = f(x)
        grids[k] = g.astype(np.float32)
        ch, cw = crop.shape[:2]
        px = C.bm_to_px(g, cw, ch)
        s = min(1.0, (1.5e6 / (ch * cw)) ** 0.5)   # the scorer works at 598400 px; keep the files small
        out = C.unwarp_px(crop, px, int(round(ch * s)), int(round(cw * s)))
        cv2.imwrite(os.path.join(od, f'{k}.png'), out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    np.savez_compressed(os.path.join(od, 'grids.npz'), **grids)
    print('run', a.tag, len(cr))


def score(a):
    sys.path.insert(0, BENCH)
    import docunet_eval as E
    from concurrent.futures import ProcessPoolExecutor
    cr = json.load(open(os.path.join(a.out, 'crops.json')))
    od = os.path.join(a.out, a.tag)
    keys = list(cr)
    jobs = [(cr[k]['gt'], os.path.join(od, f'{k}.png')) for k in keys]
    t = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(zip(keys, ex.map(E.eval_pair, jobs)))
    mean = {q: float(np.mean([r[q] for r in res.values()])) for q in ('ms', 'ld', 'ad')}
    json.dump({'mean': mean, 'n': len(res), 'per_image': res}, open(os.path.join(od, 'metrics.json'), 'w'), indent=0)
    print(f"{a.tag} n={len(res)} MS-SSIM {mean['ms']:.4f} LD {mean['ld']:.3f} AD {mean['ad']:.4f} ({time.time() - t:.0f}s)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('set')
    ap.add_argument('out')
    ap.add_argument('tag', nargs='?')
    ap.add_argument('model', nargs='?')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    {'crops': crops, 'run': run, 'score': score}[a.cmd](a)
