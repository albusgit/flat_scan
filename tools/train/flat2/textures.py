"""textures.py SRC OUT N [--seed S]: colour page images for synth.py, made from grey text pages (tools/orientnet
render_pages.py output, or any flat page images in SRC) by adding what real scans have and the nets get wrong:
pictures (smooth colour fields with shapes, photos-like noise), coloured title bars and bands, a coupon with a dashed
border (the "SAVE $1" coupon), barcodes, tinted or coloured paper, a highlighter, stamps. Licence-clean: everything
is drawn here or comes from the fonts the text pages were rendered with."""
import argparse
import glob
import os

import cv2
import numpy as np


def picture(rng, h, w):
    """a picture-like block: a smooth colour field, a few shapes, texture"""
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    a = rng.uniform(0, np.pi)
    t = ((xx * np.cos(a) + yy * np.sin(a)) / max(h, w))[..., None]
    img = c0 * (1 - t) + c1 * t
    for _ in range(int(rng.integers(2, 8))):
        col = tuple(float(v) for v in rng.uniform(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(0.05, 0.4) * min(h, w)), col, -1)
        else:
            pts = (rng.uniform(0, 1, (int(rng.integers(3, 7)), 2)) * [w, h]).astype(np.int32)
            cv2.fillPoly(img, [pts], col)
    n = cv2.resize(rng.normal(0, 1, (h // 8 + 1, w // 8 + 1, 3)), (w, h)) * rng.uniform(5, 25)
    return np.clip(cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 3)) + n, 0, 255)


def barcode(rng, h, w):
    img = np.full((h, w, 3), 255.0)
    x = 0
    while x < w:
        bw = int(rng.integers(1, 5))
        if rng.random() < 0.5:
            img[:, x:x + bw] = 0
        x += bw
    return img


def decorate(rng, page):
    img = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR).astype(np.float64) if page.ndim == 2 else page.astype(np.float64)
    H, W = img.shape[:2]
    if rng.random() < 0.3:   # coloured or tinted paper
        img = img / 255 * rng.uniform(170, 255, 3)
    for _ in range(int(rng.integers(0, 4))):   # pictures
        h, w = int(rng.uniform(0.12, 0.45) * H), int(rng.uniform(0.2, 0.9) * W)
        y, x = int(rng.uniform(0, H - h)), int(rng.uniform(0, W - w))
        img[y:y + h, x:x + w] = picture(rng, h, w)
    if rng.random() < 0.4:   # a coloured bar / band with knocked-out text area
        h = int(rng.uniform(0.03, 0.12) * H)
        y = int(rng.uniform(0, H - h))
        img[y:y + h] = rng.uniform(0, 255, 3)
    if rng.random() < 0.3:   # a coupon with a dashed border
        h, w = int(rng.uniform(0.2, 0.5) * H), int(rng.uniform(0.2, 0.45) * W)
        y, x = int(rng.uniform(0, H - h)), int(rng.uniform(0, W - w))
        img[y:y + h, x:x + w] = 245
        dash = int(rng.integers(6, 18))
        for k in range(0, 2 * (w + h), dash * 2):
            for (a, b) in [((x + k, y), (x + k + dash, y)), ((x + k, y + h), (x + k + dash, y + h)),
                           ((x, y + k), (x, y + k + dash)), ((x + w, y + k), (x + w, y + k + dash))]:
                cv2.line(img, a, b, (20, 20, 20), int(rng.integers(2, 5)))
        bh = int(h * 0.25)
        img[y + h - bh - 10:y + h - 10, x + 10:x + w - 10] = barcode(rng, bh, w - 20)
    if rng.random() < 0.2:   # highlighter strokes
        for _ in range(int(rng.integers(1, 5))):
            y = int(rng.uniform(0, H))
            x0, x1 = sorted(rng.uniform(0, W, 2).astype(int))
            col = np.array([rng.choice([0, 60]), 255, 255]) if rng.random() < 0.6 else np.array([255, 120, 255])
            band = img[max(0, y - 12):y + 12, x0:x1]
            img[max(0, y - 12):y + 12, x0:x1] = band * (col / 255)
    if rng.random() < 0.15:   # a stamp
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        cv2.circle(img, c, int(rng.uniform(0.04, 0.1) * W), (rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(150, 255)), 4)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = sorted(glob.glob(os.path.join(a.src, '*.png')) + glob.glob(os.path.join(a.src, '*.jpg')))
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        page = cv2.imread(src[rng.integers(len(src))], cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(a.out, f'tex_{a.seed}_{i:05d}.jpg'), decorate(rng, page), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == '__main__':
    main()
