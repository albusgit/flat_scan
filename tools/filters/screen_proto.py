"""Prototype screen-photo features (ported to core/src/screen.cpp once chosen).

  python -I screen_proto.py feats OUT.json DIR_OR_GLOB...   features per image
"""
import glob, json, os, sys
import numpy as np, cv2


def tiles(h, w, n=512):
    ys = [int(h * f) - n // 2 for f in (0.3, 0.5, 0.7)]
    xs = [int(w * f) - n // 2 for f in (0.3, 0.5, 0.7)]
    return [(max(0, y), max(0, x)) for y in ys for x in xs if h >= n and w >= n]


def peak_score(t, rmin):
    """prominence of the strongest isolated spectral peak (log magnitude above its 15x15 median), radius >= rmin"""
    n = t.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2((t - t.mean()) * win)))
    L = np.log1p(F).astype(np.float32)
    med = cv2.medianBlur(np.clip(L * 20, 0, 255).astype(np.uint8), 15).astype(np.float32) / 20
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    r = np.hypot(yy, xx)
    m = (r >= rmin) & (r < n // 2 - 2) & (np.abs(yy) > 1) | (r >= rmin) & (np.abs(xx) > 1) & (r < n // 2 - 2)
    m &= ~((np.abs(yy) <= 1) | (np.abs(xx) <= 1))   # the axes: page edges, text rows and columns
    d = (L - med)[m]
    return float(np.percentile(d, 99.99)), float(d.max())


def feats(img):
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    # background: near the local maximum brightness, no ink nearby
    bgY = cv2.dilate(cv2.resize(Y, (w // 4, h // 4), interpolation=cv2.INTER_AREA), np.ones((15, 15), np.uint8))
    bgY = cv2.resize(bgY, (w, h))
    bg = (Y >= 0.85 * bgY)
    ink = cv2.dilate((Y < 0.7 * bgY).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bg &= ~ink
    out = {}
    # chroma band-pass energy on the background (moire rainbow bands, subpixel fringes): 2..12 px
    for name, C in (('cr', Cr), ('cb', Cb)):
        bp = cv2.GaussianBlur(C, (0, 0), 1.5) - cv2.GaussianBlur(C, (0, 0), 10)
        out[f'bp_{name}'] = float(np.sqrt((bp[bg] ** 2).mean())) if bg.sum() > 1000 else 0.0
    bpY = cv2.GaussianBlur(Y, (0, 0), 1.5) - cv2.GaussianBlur(Y, (0, 0), 10)
    out['bp_y'] = float(np.sqrt((bpY[bg] ** 2).mean()) / max(1, np.median(Y[bg]))) * 100 if bg.sum() > 1000 else 0.0
    out['bg_frac'] = float(bg.mean())
    # spectral peaks (tiles): luma and chroma, periods < 16 px (radius >= 32 of 512)
    pk = {'y': [], 'c': []}
    for (y0, x0) in tiles(h, w):
        ty = Y[y0:y0 + 512, x0:x0 + 512]
        tc = Cr[y0:y0 + 512, x0:x0 + 512] + Cb[y0:y0 + 512, x0:x0 + 512] * 0.0
        pk['y'].append(peak_score(ty, 32)[1])
        pk['c'].append(max(peak_score(Cr[y0:y0 + 512, x0:x0 + 512], 32)[1], peak_score(Cb[y0:y0 + 512, x0:x0 + 512], 32)[1]))
    out['pk_y'] = float(np.median(pk['y'])) if pk['y'] else 0.0
    out['pk_c'] = float(np.median(pk['c'])) if pk['c'] else 0.0
    # background colour (white balance of the "paper"): b - r of the bright background, normalised
    if bg.sum() > 1000:
        m = f[bg].mean(0)
        out['bg_blue'] = float((m[0] - m[2]) / max(1, m.mean()))
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'feats':
        res = {}
        files = []
        for a in sys.argv[3:]:
            files += sorted(glob.glob(a)) if any(c in a for c in '*?') else [a]
        for p in files:
            img = cv2.imread(p)
            if img is None:
                continue
            res[p] = feats(img)
            print(os.path.basename(p), {k: round(v, 3) for k, v in res[p].items()}, flush=True)
        json.dump(res, open(sys.argv[2], 'w'), indent=1)
