"""Pseudo ground-truth backward maps for real photo / flat-original pairs.

  python pseudo_gt.py teacher DATA_DIR            # GPU: SD-HR on each photo (whole photo, then a crop around its page)
  python pseudo_gt.py refine DATA_DIR [JOBS]      # CPU: dense optical flow between the unwarped photo and the
                                                  # flat original corrects the teacher grid (2 iterations)

DATA_DIR/pairs.json lists {id, photo, flat}. Output DATA_DIR/pgt/<id>.npz:
  bm_t   2x121x177  teacher grid, photo pixel coords (x, y)
  bm     2x121x177  refined grid, photo pixel coords
  r0, r1           mean flow residual (px at flat height 1400) of the teacher and of the refined grid
  ncc0, ncc1       normalised cross-correlation of the unwarped photo with the original, before / after
The refined grid maps each point of the flat original to the photo point showing it: the label a dewarping
net should output. The flow is regularised (DIS + variational refinement, then a Gaussian at grid scale)."""
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FH = 1400  # flat original height used for the flow


def norm_gray(g):
    """Illumination-normalised grayscale uint8: paper ~ 230, ink dark."""
    g = g.astype(np.float32)
    k = max(31, (g.shape[0] // 20) | 1)
    bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    n = np.clip(g / np.maximum(bg, 1) * 230, 0, 255)
    return n.astype(np.uint8)


def crop_box(bm_px, w, h, margin=0.06):
    x0, x1 = bm_px[0].min(), bm_px[0].max()
    y0, y1 = bm_px[1].min(), bm_px[1].max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = max(0, int(np.floor(x0 - mx))), min(w, int(np.ceil(x1 + mx)))
    y0, y1 = max(0, int(np.floor(y0 - my))), min(h, int(np.ceil(y1 + my)))
    return x0, y0, x1, y1


def teacher(data):
    pairs = json.load(open(os.path.join(data, 'pairs.json')))
    os.makedirs(os.path.join(data, 'pgt'), exist_ok=True)
    net = C.load_sdhr()
    for i, p in enumerate(pairs):
        out = os.path.join(data, 'pgt', p['id'] + '_t.npz')
        if os.path.exists(out):
            continue
        im = cv2.imread(os.path.join(data, p['photo']))[:, :, ::-1].copy()
        h, w = im.shape[:2]
        bm = C.bm_to_px(C.predict_bm(net, im), w, h)
        x0, y0, x1, y1 = crop_box(bm, w, h)
        crop = im[y0:y1, x0:x1]
        bm2 = C.bm_to_px(C.predict_bm(net, crop), x1 - x0, y1 - y0)
        bm2[0] += x0
        bm2[1] += y0
        np.savez(out, bm_t=bm2.astype(np.float32), box=np.array([x0, y0, x1, y1]))
        if i % 100 == 0:
            print('teacher', i, len(pairs), flush=True)


def _flow(a, b):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setPatchSize(12)
    dis.setPatchStride(4)
    dis.setGradientDescentIterations(25)
    dis.setVariationalRefinementIterations(5)
    dis.setVariationalRefinementAlpha(40.0)   # smooth: text is repetitive
    return dis.calc(a, b, None)


def _ncc(a, b, mask):
    a = a[mask].astype(np.float32)
    b = b[mask].astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def smooth_flow(f, w, sig):
    """Confidence-weighted (normalised convolution) smoothing at two scales: grid scale where the original
    has texture, a broad fit (8 grid cells) across blank paper and margins, blended by local confidence."""
    out = np.empty_like(f)
    ws = cv2.GaussianBlur(w, (0, 0), sig)
    wl = cv2.GaussianBlur(w, (0, 0), 8 * sig)
    a = np.clip(ws / 0.3, 0, 1)
    for c in range(2):
        fs = cv2.GaussianBlur(f[:, :, c] * w, (0, 0), sig) / np.maximum(ws, 1e-4)
        fl = cv2.GaussianBlur(f[:, :, c] * w, (0, 0), 8 * sig) / np.maximum(wl, 1e-4)
        if (wl < 1e-3).any():   # nothing near: global mean
            fl[wl < 1e-3] = (f[:, :, c] * w).sum() / max(w.sum(), 1e-4)
        out[:, :, c] = a * fs + (1 - a) * fl
    return out


def sample_grid(bm, sx, sy, dx, dy, pad=12):
    """Bilinear value of the grid (as upsample_field, align_corners=True) at flat px (sx, sy), extrapolated
    linearly (odd reflection) up to pad cells beyond the page."""
    g = np.pad(bm, ((0, 0), (pad, pad), (pad, pad)), mode='reflect', reflect_type='odd')
    mx = (sx / dx + pad).astype(np.float32)
    my = (sy / dy + pad).astype(np.float32)
    return np.stack([cv2.remap(g[0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                     cv2.remap(g[1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)]).astype(np.float32)


def refine_one(args):
    data, p = args
    tp = os.path.join(data, 'pgt', p['id'] + '_t.npz')
    out = os.path.join(data, 'pgt', p['id'] + '.npz')
    if os.path.exists(out) or not os.path.exists(tp):
        return p['id'], None
    t = np.load(tp)
    bm_t = t['bm_t']
    photo = cv2.imread(os.path.join(data, p['photo']), cv2.IMREAD_GRAYSCALE)
    flat = cv2.imread(os.path.join(data, p['flat']), cv2.IMREAD_GRAYSCALE)
    fh, fw = flat.shape
    fn = norm_gray(flat)
    pn = norm_gray(photo)
    sig = 0.5 * fh / C.GH
    dx, dy = (fw - 1) / (C.GW - 1), (fh - 1) / (C.GH - 1)
    ys = np.linspace(0, fh - 1, C.GH)
    xs = np.linspace(0, fw - 1, C.GW)
    gx, gy = [a.astype(np.float32) for a in np.meshgrid(xs, ys)]
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    # flow confidence: texture of the original (blank paper gives no flow), inside the photo
    gm = cv2.GaussianBlur(np.abs(cv2.Laplacian(fn.astype(np.float32), cv2.CV_32F, ksize=3)), (0, 0), 4)
    tex = np.clip(gm / 6.0, 0, 1)
    bm = bm_t.copy()
    stats = {'tex': float(tex.mean())}
    for it in range(3):
        dense = sample_grid(bm, xx, yy, dx, dy)                  # flat px -> photo px
        u = cv2.remap(pn, dense[0], dense[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = u > 0
        stats[f'ncc{it}'] = _ncc(u, fn, valid)
        f = _flow(fn, u)                                         # fn(x) ~ u(x + f(x))
        f[~valid] = 0
        conf = tex * valid
        mag = np.sqrt((f ** 2).sum(-1))
        stats[f'r{it}'] = float(mag[valid].mean())
        stats[f'p90_{it}'] = float(np.percentile(mag[valid], 90))
        if it == 2:
            break
        f = smooth_flow(f, conf, sig)
        # new grid point x -> photo px dense(x + f(x)); the teacher grid is extrapolated beyond the page
        fx = cv2.remap(f[:, :, 0], gx, gy, cv2.INTER_LINEAR)
        fy = cv2.remap(f[:, :, 1], gx, gy, cv2.INTER_LINEAR)
        bm = sample_grid(bm, gx + fx, gy + fy, dx, dy)
    stats['valid'] = float(valid.mean())
    np.savez(out, bm_t=bm_t, bm=bm, box=t['box'], **{k: np.float32(v) for k, v in stats.items()})
    return p['id'], stats


def refine(data, jobs):
    pairs = json.load(open(os.path.join(data, 'pairs.json')))
    res = {}
    with Pool(jobs) as pool:
        for i, (k, s) in enumerate(pool.imap_unordered(refine_one, [(data, p) for p in pairs])):
            if s is not None:
                res[k] = s
            if i % 50 == 0:
                print('refine', i, len(pairs), k, s, flush=True)
    old = {}
    sp = os.path.join(data, 'pgt_stats.json')
    if os.path.exists(sp):
        old = json.load(open(sp))
    old.update(res)
    json.dump(old, open(sp, 'w'), indent=0)


if __name__ == '__main__':
    if sys.argv[1] == 'teacher':
        teacher(sys.argv[2])
    else:
        refine(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
