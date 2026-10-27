"""Real shadow photos with ground truth, scored as the app uses the net: the page turned portrait,
long side 512, sides multiples of 32, the map clamped at FLOOR, the page divided by it.

Sets (validation only - their licences allow research use, not training a shipped model):
  jung2018  filters_r11/shadow_gt/jung2018/{input,gt}   87
  osr2020   filters_r11/shadow_gt/osr2020/{input,gt}    91
  rdd       raw/rdd/test/{img,gt}                       545 (every RDD_EVERY-th)

Score per pair: e = log of the de-shadowed page's luminance over the shadow-free photo's (both blurred,
sigma 3 at 512 px), less its median (the overall exposure). err = mean |e| (shadow left behind AND page
content damaged both raise it); flat = p95 - p5 of e (how even the light is). Lower is better.

  load(root, every=3) -> list of (name, input uint8 HxWx3 BGR, gt uint8, blurred luminance of gt)
  score(pairs, run, floor) -> dict(err, flat, per-set err)  with run(batch float NCHW) -> map NCHW
  python realval.py ONNX [FLOOR]      the score of a model file (e.g. the shipped one)"""
import glob, os, sys

import numpy as np
import cv2

SETS = [('jung2018', 'filters_r11/shadow_gt/jung2018/input', 'filters_r11/shadow_gt/jung2018/gt'),
        ('osr2020', 'filters_r11/shadow_gt/osr2020/input', 'filters_r11/shadow_gt/osr2020/gt'),
        ('rdd', 'raw/rdd/test/img', 'raw/rdd/test/gt')]


def fit(img, long_side=512):
    """the app's net input: portrait, long side 512, multiples of 32"""
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def lum(img):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)


def load(root, every=3, rdd_every=3):
    pairs = []
    for name, ind, gtd in SETS:
        files = sorted(glob.glob(os.path.join(root, ind, '*')))
        files = files[::rdd_every if name == 'rdd' else every]
        for f in files:
            g = os.path.join(root, gtd, os.path.basename(f))
            if not os.path.exists(g):
                c = glob.glob(os.path.splitext(g)[0] + '.*')
                if not c:
                    continue
                g = c[0]
            a, b = cv2.imread(f), cv2.imread(g)
            if a is None or b is None:
                continue
            a = fit(a)
            b = fit(b)
            if b.shape != a.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            # some GT pages are turned half a turn: keep the turn that matches the photo
            la, lb, lr = lum(a), lum(b), lum(cv2.rotate(b, cv2.ROTATE_180))
            if np.corrcoef(la.ravel(), lr.ravel())[0, 1] > np.corrcoef(la.ravel(), lb.ravel())[0, 1] + 0.1:
                b, lb = cv2.rotate(b, cv2.ROTATE_180), lr
            pairs.append((f'{name}/{os.path.basename(f)}', a, b, lb))
    return pairs


def score(pairs, run, floor=0.3, batch=8):
    errs, flats, per = [], [], {}
    for k in range(0, len(pairs), batch):
        chunk = pairs[k:k + batch]
        for name, a, b, lb in chunk:   # pages differ in size: one at a time inside the chunk
            x = (a.astype(np.float32) / 255).transpose(2, 0, 1)[None]
            S = np.clip(run(x)[0].transpose(1, 2, 0), floor, 1)
            out = np.clip(a.astype(np.float32) / S, 0, 255).astype(np.uint8)
            e = np.log((lum(out) + 4) / (lb + 4))
            m = 16
            e = e[m:-m, m:-m]
            e -= np.median(e)
            err = float(np.abs(e).mean()); flat = float(np.percentile(e, 95) - np.percentile(e, 5))
            errs.append(err); flats.append(flat)
            per.setdefault(name.split('/')[0], []).append(err)
    out = {'err': float(np.mean(errs)), 'flat': float(np.mean(flats))}
    out.update({f'err_{k}': float(np.mean(v)) for k, v in per.items()})
    return out


def identity_score(pairs):
    """no shadow removal at all: the floor of what a model must beat"""
    return score(pairs, lambda x: np.ones_like(x))


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from fast import data_root
    import onnxruntime as ort
    sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    pairs = load(data_root())
    print(len(pairs), 'pairs')
    print('no removal', identity_score(pairs))
    print(os.path.basename(sys.argv[1]), f'floor {floor}', score(pairs, lambda x: sess.run(None, {'page': x})[0], floor))
