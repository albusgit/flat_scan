"""Prototype of the screen-mode pre-filter (ported to core/src/screen.cpp).

  python -I descreen_proto.py SETDIR OUTSET [--notch 0/1] [--chroma R] [--eps E] [--thr T]
writes OUTSET/NNN_in.png (+ copies NNN_gt.png) or NAME_page.png (+ paper)
"""
import glob, os, shutil, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_filters import pages, arg  # noqa: E402


def box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided(I, p, r, eps):
    mI, mp = box(I, r), box(p, r)
    cov = box(I * p, r) - mI * mp
    var = box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def notch_mask(Y, thr, rmin_period=12.0):
    h, w = Y.shape
    F = np.fft.fft2(Y - Y.mean())
    L = np.log1p(np.abs(np.fft.fftshift(F))).astype(np.float32)
    med = cv2.medianBlur(np.clip(L * 16, 0, 255).astype(np.uint8), 21).astype(np.float32) / 16
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fy, fx = (yy - h // 2) / h, (xx - w // 2) / w   # cycles per px
    r = np.hypot(fy, fx)
    pk = ((L - med) > thr) & (r > 1.0 / rmin_period)
    pk = cv2.dilate(pk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    pk = cv2.GaussianBlur(pk, (0, 0), 1.5)
    return np.fft.ifftshift(1 - np.clip(pk * 1.5, 0, 1)), int(((L - med) > thr)[r > 1.0 / rmin_period].sum())


def descreen(img, notch=True, chroma_r=8, eps=40.0, thr=2.0):
    f = img.astype(np.float32)
    ycc = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    npk = 0
    if notch:
        M, npk = notch_mask(ycc[..., 0], thr)
        for c in range(3):
            ch = ycc[..., c]
            m = ch.mean()
            ycc[..., c] = np.real(np.fft.ifft2(np.fft.fft2(ch - m) * M)) + m
    if chroma_r > 0:
        Y = ycc[..., 0]
        for c in (1, 2):
            ycc[..., c] = guided(Y, ycc[..., c], chroma_r, eps * eps)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), npk


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    notch = int(arg('--notch', 1)) == 1
    cr, eps, thr = int(arg('--chroma', 8)), float(arg('--eps', 40)), float(arg('--thr', 2.0))
    for n, p, pa, g in pages(src):
        img = cv2.imread(p)
        t0 = time.time()
        o, npk = descreen(img, notch, cr, eps, thr)
        print(n, f'{(time.time() - t0) * 1000:.0f} ms, peaks {npk}', flush=True)
        if g:
            cv2.imwrite(os.path.join(dst, n + '_in.png'), o)
            shutil.copy(g, os.path.join(dst, os.path.basename(g)))
        else:
            cv2.imwrite(os.path.join(dst, n + '_page.png'), o)
            if pa:
                shutil.copy(pa, os.path.join(dst, n + '_paper.png'))
