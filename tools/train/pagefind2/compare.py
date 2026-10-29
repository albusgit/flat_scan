"""compare.py OUT_DIR [--new CKPT] [--packs A.npz,B.npz] [--real33 LIST]: the app's page finder (round 4,
app/src/main/assets/models/pagefinder.onnx: the photo squashed to 256) vs a page finder v2 checkpoint (letterboxed
384), network masks only (logit > 0), on the held-out packs and the 33 hand-checked round-4 photos.

Per photo: IoU of each net's paper with the true paper (inside the photo). Prints per set and per DocCorner source:
mean IoU, photos below 0.9, and better / same / worse (IoU moved by more than 0.02). Writes OUT_DIR/results.json and
side-by-side images (app now | new, true outline thin green, found paper outlined) of the largest gains and every
loss: OUT_DIR/SET_gain_NN.jpg, SET_loss_NN.jpg, and contact sheets SET_sheet_gain.jpg / SET_sheet_loss.jpg."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
from net import MEAN, PAD_BGR, SIZE, STD, SegNet2, letterbox  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OLD = os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models', 'pagefinder.onnx')
GT33 = 'E:/fs_datasets/round4/gt'
MOVE = 0.02


class Old:
    def __init__(self):
        o = ort.SessionOptions(); o.intra_op_num_threads = 8
        self.s = ort.InferenceSession(OLD, o, providers=['CPUExecutionProvider'])

    def paper(self, bgr):
        """photo (any size) -> bool paper mask at the photo's size"""
        x = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        lg = self.s.run(None, {'image': x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        h, w = bgr.shape[:2]
        return cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR) > 0


class New:
    def __init__(self, ck):
        self.net = SegNet2().cuda().eval()
        self.net.load_state_dict(torch.load(ck, map_location='cuda'))

    @torch.no_grad()
    def paper_lb(self, lb):
        """letterboxed S x S BGR -> S x S bool paper mask"""
        x = (cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        out = self.net(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).cuda())
        return (out[0, 0] > 0).cpu().numpy()

    def paper(self, bgr):
        lb, (s, x0, y0) = letterbox(bgr)
        p = self.paper_lb(lb).astype(np.uint8)
        h, w = bgr.shape[:2]
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        return cv2.resize(p[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_NEAREST) > 0


def iou(p, t):
    u = np.logical_or(p, t).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, t).sum() / u)


def outline(img, m, col, th):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)


def panel(bgr, gt, old, new, io, inew, title, H=420):
    s = H / bgr.shape[0]
    W = int(round(bgr.shape[1] * s))
    base = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    r = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    g, o, n = r(gt), r(old), r(new)
    out = []
    for m, col, name, v in ((o, (40, 40, 230), 'app now', io), (n, (230, 120, 20), 'new', inew)):
        im = base.copy()
        tint = im.copy(); tint[m] = (0.55 * tint[m] + 0.45 * np.array(col)).astype(np.uint8)
        im = cv2.addWeighted(im, 0.3, tint, 0.7, 0)
        outline(im, m, col, 3)
        outline(im, g, (40, 200, 40), 1)
        bar = np.full((34, W, 3), 255, np.uint8)
        cv2.putText(bar, f'{name}  IoU {v:.3f}', (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        out.append(np.vstack([bar, im]))
    gap = np.full((H + 34, 6, 3), 255, np.uint8)
    top = np.full((26, 2 * W + 6, 3), 235, np.uint8)
    cv2.putText(top, title[:90], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([top, np.hstack([out[0], gap, out[1]])])


def sheet(images, path, cols=2, width=1400):
    if not images:
        return
    rows, cur = [], []
    for im in images:
        s = (width / cols - 10) / im.shape[1]
        cur.append(cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA))
        if len(cur) == cols:
            rows.append(cur); cur = []
    if cur:
        rows.append(cur)
    lines = []
    for row in rows:
        h = max(i.shape[0] for i in row)
        row = [np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)]) for i in row]
        row = sum([[i, np.full((h, 10, 3), 255, np.uint8)] for i in row], [])
        line = np.hstack(row)
        line = np.hstack([line, np.full((h, width - line.shape[1], 3), 255, np.uint8)]) if line.shape[1] < width else line[:, :width]
        lines += [line, np.full((10, width, 3), 255, np.uint8)]
    cv2.imwrite(path, np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])


def summarise(rows):
    a = np.array([r['old'] for r in rows]); b = np.array([r['new'] for r in rows]); d = b - a
    return dict(n=len(rows), old=round(float(a.mean()), 4), new=round(float(b.mean()), 4),
                old_below=int((a < 0.9).sum()), new_below=int((b < 0.9).sum()),
                better=int((d > MOVE).sum()), same=int((np.abs(d) <= MOVE).sum()), worse=int((d < -MOVE).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--new', default='C:/fs_train/pf2/r19/best.pt')
    ap.add_argument('--packs', default='C:/fs_train/pf2/val_smartdoc.npz,C:/fs_train/pf2/val_doccorner.npz,C:/fs_train/pf2/val_synth.npz')
    ap.add_argument('--real33', default='')
    ap.add_argument('--gains', type=int, default=12); ap.add_argument('--losses', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    old, new = Old(), New(a.new)
    sets = {}   # name -> list of (row, photo, gt, old mask, new mask) for the image pass; masks kept only for picks
    for path in [p for p in a.packs.split(',') if p]:
        name = os.path.basename(path)[:-4].replace('val_', '')
        z = D.load(path)
        srcs = list(z['sources'])
        rows = []
        for k in range(len(z['box'])):
            lb = cv2.imdecode(z['jpeg'][z['off'][k]:z['off'][k + 1]], cv2.IMREAD_COLOR)
            x0, y0, x1, y1 = [int(v) for v in z['box'][k]]
            lb[:y0] = PAD_BGR; lb[y1:] = PAD_BGR; lb[:, :x0] = PAD_BGR; lb[:, x1:] = PAD_BGR
            t = np.unpackbits(z['mask'][k]).reshape(SIZE, SIZE)[y0:y1, x0:x1] > 0
            photo = lb[y0:y1, x0:x1]
            po = old.paper(photo)
            pn = new.paper_lb(lb)[y0:y1, x0:x1]
            rows.append(dict(set=name, source=srcs[z['src'][k]], name=str(z['name'][k]), old=iou(po, t), new=iou(pn, t), k=k))
        sets[name] = (rows, ('pack', z))
    if a.real33:
        gtj = json.load(open(os.path.join(GT33, 'gt.json')))
        rows, ph = [], {}
        for p in open(a.real33).read().split():
            n = os.path.splitext(os.path.basename(p))[0]
            gm = cv2.imread(os.path.join(GT33, 'masks', n + '.png'), 0)
            im = cv2.imread(p)
            if gm is None or im is None or n not in gtj:
                continue
            if gm.shape != im.shape[:2]:
                gm = cv2.resize(gm, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            t = gm > 127
            rows.append(dict(set='real33', source='real33', name=n, old=iou(old.paper(im), t), new=iou(new.paper(im), t), path=p))
        sets['real33'] = (rows, ('files', None))

    report = {}
    for name, (rows, (kind, z)) in sets.items():
        report[name] = dict(all=summarise(rows))
        if name == 'doccorner':
            by = {}
            for r in rows:
                by.setdefault(r['source'], []).append(r)
            report[name]['by_source'] = {s: summarise(rs) for s, rs in sorted(by.items())}
        order = sorted(rows, key=lambda r: r['new'] - r['old'])
        losses = [r for r in order if r['new'] - r['old'] < -MOVE][:a.losses]
        gains = [r for r in order[::-1] if r['new'] - r['old'] > MOVE][:a.gains]
        for tag, pick in (('gain', gains), ('loss', losses)):
            ims = []
            for i, r in enumerate(pick):
                if kind == 'pack':
                    k = r['k']
                    lb = cv2.imdecode(z['jpeg'][z['off'][k]:z['off'][k + 1]], cv2.IMREAD_COLOR)
                    x0, y0, x1, y1 = [int(v) for v in z['box'][k]]
                    lb[:y0] = PAD_BGR; lb[y1:] = PAD_BGR; lb[:, :x0] = PAD_BGR; lb[:, x1:] = PAD_BGR
                    photo = lb[y0:y1, x0:x1].copy()
                    t = np.unpackbits(z['mask'][k]).reshape(SIZE, SIZE)[y0:y1, x0:x1] > 0
                    po, pn = old.paper(photo), new.paper_lb(lb)[y0:y1, x0:x1]
                else:
                    photo = cv2.imread(r['path'])
                    gm = cv2.imread(os.path.join(GT33, 'masks', r['name'] + '.png'), 0)
                    t = cv2.resize(gm, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                    po, pn = old.paper(photo), new.paper(photo)
                im = panel(photo, t, po, pn, r['old'], r['new'], f"{name} / {r['source']} / {r['name']}")
                cv2.imwrite(os.path.join(a.out, f'{name}_{tag}_{i:02d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
                ims.append(im)
            sheet(ims, os.path.join(a.out, f'{name}_sheet_{tag}.jpg'))
        for r in rows:
            r.pop('k', None)
        report[name]['rows'] = rows
        s = report[name]['all']
        print(f"{name:10s} n={s['n']:5d}  IoU app {s['old']:.4f} -> new {s['new']:.4f}   below 0.9: {s['old_below']} -> {s['new_below']}"
              f"   better / same / worse: {s['better']} / {s['same']} / {s['worse']}", flush=True)
        if 'by_source' in report[name]:
            for src, s in report[name]['by_source'].items():
                print(f"    {src:34s} n={s['n']:3d}  {s['old']:.3f} -> {s['new']:.3f}  <0.9: {s['old_below']:3d} -> {s['new_below']:3d}"
                      f"  +{s['better']} ={s['same']} -{s['worse']}")
    json.dump(report, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
