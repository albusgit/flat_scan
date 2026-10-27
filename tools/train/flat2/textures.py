"""textures.py SRC OUT N [--seed S]: colour page images for synth.py, made from grey text pages (tools/orientnet
render_pages.py output, or any flat page images in SRC) by adding what real scans have and the nets get wrong:
pictures (smooth colour fields with shapes, photos-like noise), coloured title bars and bands, a coupon with a dashed
border (the "SAVE $1" coupon), barcodes, tinted or coloured paper, a highlighter, stamps; and (--forms) tables, forms,
ruled paper and sheet music with the text pages' print in them. Licence-clean: everything
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


def snippet(rng, src, h, w):
    """a h x w piece of the text page src (grey) with print on it, or None"""
    H, W = src.shape[:2]
    if h < 4 or w < 4 or h >= H or w >= W:
        return None
    rows = np.nonzero((src < 128).sum(1) > 2)[0]   # rows with print: start a piece on a text line
    for _ in range(16):
        y = int(rows[rng.integers(len(rows))]) - int(rng.integers(0, max(1, h // 4))) if len(rows) else int(rng.integers(0, H - h))
        y = min(max(0, y), H - h)
        x = int(rng.integers(0, W - w))
        piece = src[y:y + h, x:x + w]
        if piece.min() < 90 and piece.mean() < 245:
            return piece
    return None


def paste_text(rng, img, src, x0, y0, x1, y1, scale):
    """text from src (a line's height scaled by `scale`) into the box, darkening (print on paper)"""
    bh, bw = y1 - y0, x1 - x0
    if bh < 6 or bw < 10:
        return
    sh = int(min(bh, rng.uniform(0.5, 0.9) * bh) / scale)
    sw = int(bw * rng.uniform(0.3, 0.95) / scale)
    piece = snippet(rng, src, sh, sw)
    if piece is None:
        return
    piece = cv2.resize(piece, (max(1, int(sw * scale)), max(1, int(sh * scale))), interpolation=cv2.INTER_AREA)
    piece = np.clip(piece.astype(np.float64) * 255 / max(1.0, np.percentile(piece, 90)), 0, 255)   # its paper white
    ph, pw = piece.shape[:2]
    oy, ox = y0 + (bh - ph) // 2, x0 + int(rng.uniform(0.02, 0.1) * bw)
    ph, pw = min(ph, img.shape[0] - oy), min(pw, img.shape[1] - ox)
    if ph <= 0 or pw <= 0:
        return
    reg = img[oy:oy + ph, ox:ox + pw]
    img[oy:oy + ph, ox:ox + pw] = np.minimum(reg, piece[:ph, :pw, None].astype(np.float64))


def structured_page(rng, src):
    """a table, a form, ruled paper or sheet music on src's page size, with text from src (real fonts and scripts):
    long straight rules and much blank paper (round 18: the fine-tune bent a long-ruled table; the generator had
    almost no rules)"""
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    H, W = src.shape[:2]
    img = np.full((H, W, 3), float(rng.uniform(232, 255)))
    ink = np.array([rng.uniform(0, 90)] * 3) if rng.random() < 0.7 else np.array([rng.uniform(90, 200), rng.uniform(40, 120), rng.uniform(0, 60)])
    col = tuple(float(v) for v in ink)
    lw = lambda: int(rng.choice([1, 1, 2, 2, 3]))
    mx0, mx1 = int(rng.uniform(0.04, 0.15) * W), int((1 - rng.uniform(0.04, 0.15)) * W)
    my0, my1 = int(rng.uniform(0.04, 0.2) * H), int((1 - rng.uniform(0.04, 0.15)) * H)
    scale = rng.uniform(0.6, 1.2)
    kind = rng.choice(['table', 'form', 'ruled', 'music'], p=[0.5, 0.25, 0.15, 0.10])
    if kind == 'table':
        nr, nc = int(rng.integers(5, 32)), int(rng.integers(2, 9))
        hts = rng.uniform(0.7, 1.6, nr)
        hts[0] *= rng.uniform(1, 1.8)
        ys = (my0 + np.concatenate([[0], np.cumsum(hts)]) / hts.sum() * (my1 - my0)).astype(int)
        wds = rng.uniform(0.4, 2.0, nc)
        xs = (mx0 + np.concatenate([[0], np.cumsum(wds)]) / wds.sum() * (mx1 - mx0)).astype(int)
        if rng.random() < 0.4:   # a shaded header row
            img[ys[0]:ys[1], xs[0]:xs[-1]] *= rng.uniform(0.75, 0.92)
        if rng.random() < 0.25:   # zebra rows
            for k in range(1, nr, 2):
                img[ys[k]:ys[k + 1], xs[0]:xs[-1]] *= rng.uniform(0.9, 0.97)
        verticals = rng.random() < 0.7
        t = lw()
        for k, y in enumerate(ys):
            if k in (0, 1, nr) or rng.random() < 0.85:
                cv2.line(img, (xs[0], y), (xs[-1], y), col, t, cv2.LINE_AA)
        if verticals:
            for k, x in enumerate(xs):
                if k in (0, nc) or rng.random() < 0.9:
                    cv2.line(img, (x, ys[0]), (x, ys[-1]), col, t, cv2.LINE_AA)
        fill = rng.uniform(0.2, 0.8)
        for r in range(nr):
            for c in range(nc):
                if rng.random() < fill:
                    paste_text(rng, img, src, xs[c] + 2, ys[r] + 2, xs[c + 1] - 2, ys[r + 1] - 2, scale)
    elif kind == 'form':
        y = my0
        while y < my1:
            gap = int(rng.uniform(0.025, 0.07) * H)
            x = mx0
            lab = int(rng.uniform(0.15, 0.4) * (mx1 - mx0))
            paste_text(rng, img, src, x, y - gap // 2, x + lab, y, scale)
            if rng.random() < 0.2:   # tick boxes
                b = int(gap * 0.4)
                for k in range(int(rng.integers(2, 6))):
                    bx = x + lab + 10 + k * 3 * b
                    if bx + b < mx1:
                        cv2.rectangle(img, (bx, y - b), (bx + b, y), col, lw())
            else:
                cv2.line(img, (x + lab + 8, y), (int(rng.uniform(0.6, 1.0) * (mx1 - x - lab) + x + lab), y), col, lw(), cv2.LINE_AA)
            y += gap
        if rng.random() < 0.5:
            cv2.rectangle(img, (mx0 - 10, my0 - 30), (mx1 + 10, my1 + 10), col, lw())
    elif kind == 'ruled':
        sp = int(rng.uniform(0.015, 0.03) * H)
        blue = (float(rng.uniform(180, 230)), float(rng.uniform(140, 190)), float(rng.uniform(90, 140)))
        for y in range(my0, H - 5, sp):
            cv2.line(img, (0, y), (W - 1, y), blue, 1, cv2.LINE_AA)
            if rng.random() < 0.3:
                paste_text(rng, img, src, mx0, y - sp, int(rng.uniform(0.4, 1.0) * W), y - 1, scale)
        if rng.random() < 0.7:
            x = int(rng.uniform(0.08, 0.15) * W)
            cv2.line(img, (x, 0), (x, H - 1), (90.0, 90.0, 220.0), 1, cv2.LINE_AA)
    else:   # staves
        sp = int(rng.uniform(0.006, 0.012) * H)
        y = my0
        while y + 5 * sp < my1:
            for k in range(5):
                cv2.line(img, (mx0, y + k * sp), (mx1, y + k * sp), col, 1, cv2.LINE_AA)
            for _ in range(int(rng.integers(10, 40))):   # notes
                cv2.ellipse(img, (int(rng.uniform(mx0, mx1)), int(y + rng.integers(0, 9) * sp / 2)), (int(sp * 0.65), int(sp * 0.45)), -20, 0, 360, col, -1)
            y += int(5 * sp * rng.uniform(2.2, 3.5))
    return np.clip(img, 0, 255).astype(np.uint8)


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
        t = int(rng.integers(2, 5))
        for k in range(0, w, dash * 2):   # each side's dashes along that side only (they ran past the coupon)
            for yy in (y, y + h):
                cv2.line(img, (x + k, yy), (x + min(k + dash, w), yy), (20, 20, 20), t)
        for k in range(0, h, dash * 2):
            for xx in (x, x + w):
                cv2.line(img, (xx, y + k), (xx, y + min(k + dash, h)), (20, 20, 20), t)
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
    ap.add_argument('--forms', type=float, default=0.3, help='share of tables / forms / ruled paper / sheet music')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = sorted(glob.glob(os.path.join(a.src, '*.png')) + glob.glob(os.path.join(a.src, '*.jpg')))
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        page = cv2.imread(src[rng.integers(len(src))], cv2.IMREAD_UNCHANGED)
        if rng.random() < a.forms:
            out = structured_page(rng, page)
            if rng.random() < 0.25:   # tinted paper
                out = np.clip(out * rng.uniform(0.85, 1.0, 3), 0, 255).astype(np.uint8)
        else:
            out = decorate(rng, page)
        cv2.imwrite(os.path.join(a.out, f'tex_{a.seed}_{i:05d}.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == '__main__':
    main()
