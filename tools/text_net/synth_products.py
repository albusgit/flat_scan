"""Round 19: product shots and drawings for the rule + text-line net's pages, as hard negatives (never a rule, a
picture edge or a text line). The round 12 net marked a poster's vitamin bottles (cap edges, bottle sides, label bands)
as rules and the logos wrapped round them as text rows, and the line stage bent the bottles to straighten them; its
pictures were textured rectangles whose straight sides were labelled.

  block(pg, x0, y0, x1, y1)   a panel (plain, tinted or a gradient) with bottles, boxes or a line drawing on it; every
                              label in the block cleared
  python synth_products.py N OUT.jpg [SEED]   contact sheet of blocks

Bottles: a cap (plain or ridged), a neck, a rounded shoulder and a body, lit as a cylinder (a soft highlight), with a
label band whose edges and words follow the cylinder (seen from a little above: the band sags at its middle). Boxes:
two faces in perspective, words on them warped with the face. Drawings: outlines, spikes and hatching in one ink.
Everything is drawn by this code (no photo is used): licence-clean."""
import math
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rule_net'))
import synth  # noqa: E402
from synth2 import HEAVY_LAT, LAT_FONTS  # noqa: E402

WORDS = ['MULTI', 'GUMMIES', 'ORIGINAL', 'VITAMIN', 'DAILY', 'FRESH', 'NATURAL', 'FORMULA', 'EXTRA', 'STRENGTH', 'SOFT', 'CLEAN',
         'ACTIVE', 'DRY', 'SPORT', 'CARE', 'PLUS', 'ULTRA', 'MEN', 'WOMEN', 'KIDS', 'ONE', 'DAY', 'MAX', 'PURE', 'CLASSIC', 'NEW',
         'Lemon', 'Mint', 'Berry', 'Ocean', 'Cool', 'Sensitive', 'Relief', 'Fast', 'Long lasting', '24h', '150 ct', '48 oz']


def _font(path, size):
    try:
        return ImageFont.truetype(path, max(6, int(size)))
    except OSError:
        return ImageFont.load_default()


def _colour(rng, light=None):
    h = rng.randrange(180)
    s = rng.randint(60, 255)
    v = rng.randint(60, 255) if light is None else (rng.randint(200, 255) if light else rng.randint(30, 120))
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255


def _words_image(rng, w, h, n_lines):
    """a flat label: n_lines of words (the first big), white on black alpha, w x h"""
    im = Image.new('L', (max(8, int(w)), max(8, int(h))), 0)
    d = ImageDraw.Draw(im)
    y = h * rng.uniform(0.02, 0.15)
    for k in range(n_lines):
        sz = h / (n_lines + 1) * (rng.uniform(1.0, 1.6) if k == 0 else rng.uniform(0.45, 0.9))
        f = _font(rng.choice(HEAVY_LAT if k == 0 or rng.random() < 0.5 else LAT_FONTS), sz)
        s = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(1, 3)))
        d.text((w * rng.uniform(0.02, 0.2), y), s, font=f, fill=255)
        y += sz * rng.uniform(1.0, 1.4)
        if y > h:
            break
    return np.asarray(im).astype(np.float32) / 255


def _over(dst, alpha, col):
    a = alpha[..., None]
    dst[:] = dst * (1 - a) + a * np.asarray(col, np.float32)


def bottle(rng, ink, cx, y_bot, wb, hb):
    """one bottle standing at (cx, y_bot), wb wide and hb high, drawn into ink (H x W x 3 float32)"""
    H, W = ink.shape[:2]
    r = wb / 2
    hc = hb * rng.uniform(0.1, 0.2)            # cap
    wc = wb * rng.uniform(0.45, 0.85)
    hn = hb * rng.uniform(0.0, 0.06)           # neck
    hs = hb * rng.uniform(0.05, 0.16)          # shoulder
    y_top = y_bot - hb
    y_cap1 = y_top + hc
    y_body = y_cap1 + hn + hs
    # the body's half width at each row: the neck (the cap's width), the shoulder's quarter ellipse, the body, a rounded foot
    ys = np.arange(int(max(0, y_cap1)), int(min(H, y_bot)))
    half = np.full(len(ys), r, np.float32)
    for i, y in enumerate(ys):
        if y < y_cap1 + hn:
            half[i] = wc / 2 * 0.95
        elif y < y_body:
            t = (y - y_cap1 - hn) / max(1.0, hs)
            half[i] = wc / 2 * 0.95 + (r - wc / 2 * 0.95) * math.sqrt(max(0.0, 1 - (1 - t) ** 2))
        elif y > y_bot - r * 0.25:
            t = (y - (y_bot - r * 0.25)) / (r * 0.25)
            half[i] = r * math.sqrt(max(0.0, 1 - t * t * 0.6))
    body_col = _colour(rng)
    cap_col = _colour(rng, light=rng.random() < 0.3)
    x0, x1 = int(max(0, cx - r - 2)), int(min(W, cx + r + 2))
    if x1 <= x0 or len(ys) == 0:
        return
    X = np.arange(x0, x1, dtype=np.float32)
    light_a = rng.uniform(-0.6, 0.2)
    for i, y in enumerate(ys):
        hw = half[i]
        u = (X - cx) / hw
        inside = np.abs(u) < 1
        if not inside.any():
            continue
        th = np.arcsin(np.clip(u, -1, 1))
        shade = 0.55 + 0.45 * np.cos(th - light_a) + 0.35 * np.exp(-((th - light_a - 0.25) / 0.12) ** 2)
        a = np.clip((1 - np.abs(u)) * hw / 1.5, 0, 1) * inside   # a crisp outline (1.5 px)
        col = np.clip(body_col[None, :] * shade[:, None], 0, 1)
        row = ink[y, x0:x1]
        row[:] = row * (1 - a[:, None]) + col * a[:, None]
    # the label: a band on the body, its edges and words following the cylinder (seen from a little above)
    e = r * rng.uniform(0.04, 0.16)
    yl0 = y_body + (y_bot - y_body) * rng.uniform(0.05, 0.3)
    yl1 = y_bot - (y_bot - y_body) * rng.uniform(0.05, 0.25)
    lab_col = _colour(rng)
    flat = _words_image(rng, math.pi * r * 0.95, yl1 - yl0, rng.randint(1, 4))
    txt_col = _colour(rng, light=rng.random() < 0.5)
    yy, xx = np.mgrid[int(max(0, yl0 - 2)):int(min(H, yl1 + e + 2)), x0:x1].astype(np.float32)
    u = np.clip((xx - cx) / r, -0.999, 0.999)
    th = np.arcsin(u)
    sag = e * (1 - np.cos(th))
    fy = yy - sag - yl0
    in_band = (np.abs((xx - cx) / r) < 0.985) & (fy >= 0) & (fy < yl1 - yl0)
    shade = (0.6 + 0.4 * np.cos(th - light_a))[..., None]
    reg = ink[int(max(0, yl0 - 2)):int(min(H, yl1 + e + 2)), x0:x1]
    band = np.clip(lab_col[None, None, :] * shade, 0, 1)
    reg[in_band] = band[in_band]
    fx = (th * r + math.pi * r * 0.95 / 2).astype(np.float32)
    words = cv2.remap(flat, fx, fy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * in_band
    _over(reg, words, np.clip(txt_col * 0.9, 0, 1))
    # the cap: lit the same way, ridged now and then, a dark rim at its foot
    cy0, cy1 = int(max(0, y_top)), int(min(H, y_cap1))
    cx0, cx1 = int(max(0, cx - wc / 2)), int(min(W, cx + wc / 2))
    if cy1 > cy0 and cx1 > cx0:
        X = np.arange(cx0, cx1, dtype=np.float32)
        th = np.arcsin(np.clip((X - cx) / (wc / 2), -0.999, 0.999))
        shade = 0.5 + 0.5 * np.cos(th - light_a)
        if rng.random() < 0.5:
            shade = shade * (0.85 + 0.15 * np.sign(np.sin(th * rng.randint(10, 30))))
        ink[cy0:cy1, cx0:cx1] = np.clip(cap_col[None, None, :] * shade[None, :, None], 0, 1)
        ink[max(0, cy1 - 2):cy1, cx0:cx1] *= 0.6


def box(rng, ink, x0, y_bot, wb, hb):
    """a box in perspective: its front and one side, words warped with the front"""
    H, W = ink.shape[:2]
    d = wb * rng.uniform(0.15, 0.4)                 # the side's depth
    lean = hb * rng.uniform(-0.08, 0.08)
    sy = hb * rng.uniform(0.04, 0.15)               # the top's rise
    front = np.float32([[x0, y_bot - hb], [x0 + wb, y_bot - hb + lean], [x0 + wb, y_bot + lean * 0.3], [x0, y_bot]])
    side = np.float32([front[1], front[1] + [d, -sy], front[2] + [d, -sy * 0.6], front[2]])
    top = np.float32([front[0], front[0] + [d, -sy], side[1], front[1]])
    col = _colour(rng)
    for poly, k in ((front, 1.0), (side, 0.7), (top, 0.85)):
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 255)
        sel = m > 0
        ink[sel] = np.clip(col * k * rng.uniform(0.9, 1.1), 0, 1)
    fw, fh = int(wb), int(hb)
    if fw < 16 or fh < 16:
        return
    flat = _words_image(rng, fw, fh, rng.randint(1, 4))
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), front)
    a = cv2.warpPerspective(flat, Hm, (W, H))
    _over(ink, a, np.clip(_colour(rng, light=rng.random() < 0.5), 0, 1))


def drawing(rng, ink, x0, y0, x1, y1):
    """a line drawing: outlines of rounded shapes, spikes (as an illustration's peaks) and hatching, in one ink"""
    col = np.float32([rng.uniform(0.05, 0.3)] * 3)
    m = np.zeros(ink.shape[:2], np.uint8)
    w, h = x1 - x0, y1 - y0
    for _ in range(rng.randint(2, 6)):
        cx, cy = x0 + rng.uniform(0.1, 0.9) * w, y0 + rng.uniform(0.2, 0.9) * h
        rx, ry = rng.uniform(0.05, 0.3) * w, rng.uniform(0.05, 0.4) * h
        ts = np.linspace(0, 2 * math.pi, 60)
        wob = 1 + 0.15 * np.sin(ts * rng.randint(2, 6) + rng.uniform(0, 6))
        pts = np.stack([cx + rx * wob * np.cos(ts), cy + ry * wob * np.sin(ts)], 1)
        cv2.polylines(m, [np.round(pts).astype(np.int32)], True, 255, rng.randint(1, 4), cv2.LINE_AA)
    base = y0 + h * rng.uniform(0.7, 0.95)
    x = x0
    pts = [(x, base)]
    while x < x1:
        x += rng.uniform(0.03, 0.12) * w
        pts.append((x - rng.uniform(0.01, 0.05) * w, base - rng.uniform(0.2, 0.75) * h))
        pts.append((x, base - rng.uniform(0, 0.08) * h))
    cv2.polylines(m, [np.round(np.float32(pts)).astype(np.int32)], False, 255, rng.randint(1, 4), cv2.LINE_AA)
    for _ in range(rng.randint(10, 60)):
        hx, hy = x0 + rng.uniform(0, 1) * w, y0 + rng.uniform(0, 1) * h
        ln, an = rng.uniform(0.02, 0.08) * w, rng.uniform(-1.2, 1.2)
        cv2.line(m, (int(hx), int(hy)), (int(hx + ln * math.cos(an)), int(hy + ln * math.sin(an))), 255, 1, cv2.LINE_AA)
    _over(ink, m.astype(np.float32) / 255, col)


def block(pg, x0, y0, x1, y1):
    """a product shot or a drawing in the page's block (x0, y0) - (x1, y1); every label there cleared"""
    rng = pg.rng
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(pg.w, x1)), int(min(pg.h, y1))
    if x1 - x0 < 60 or y1 - y0 < 60:
        return
    reg = pg.ink[y0:y1, x0:x1]
    r = rng.random()
    if r < 0.35:      # a tinted panel
        reg[:] = _colour(rng, light=True)
    elif r < 0.6:     # a gradient
        a, b = _colour(rng), _colour(rng, light=True)
        t = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None] if rng.random() < 0.5 else np.linspace(0, 1, x1 - x0, dtype=np.float32)[None, :, None]
        reg[:] = a * (1 - t) + b * t
    w, h = x1 - x0, y1 - y0
    kind = rng.random()
    if kind < 0.55:
        n = rng.randint(1, 3)
        for k in range(n):
            wb = w / n * rng.uniform(0.45, 0.8)
            hb = min(h * rng.uniform(0.65, 0.95), wb * rng.uniform(1.3, 3.0))
            cx = x0 + w / n * (k + 0.5) + rng.uniform(-0.05, 0.05) * w
            yb = y1 - h * rng.uniform(0.02, 0.1)
            # a soft shadow under it
            sh = np.zeros(pg.ink.shape[:2], np.float32)
            cv2.ellipse(sh, (int(cx), int(yb)), (int(wb * 0.6), int(max(2, wb * 0.08))), 0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, wb * 0.05))
            pg.ink *= (1 - 0.35 * sh)[..., None]
            bottle(rng, pg.ink, cx, yb, wb, hb)
    elif kind < 0.8:
        wb = w * rng.uniform(0.3, 0.6)
        box(rng, pg.ink, x0 + w * rng.uniform(0.05, 0.3), y1 - h * rng.uniform(0.05, 0.15), wb, min(h * 0.8, wb * rng.uniform(0.8, 1.6)))
    else:
        drawing(rng, pg.ink, x0, y0, x1, y1)
    for lab in ('mh', 'mv', 'th', 'tv', 'eh', 'ev'):
        m = getattr(pg, lab, None)
        if m is not None:
            m[y0:y1, x0:x1] = 0


if __name__ == '__main__':
    n, out = int(sys.argv[1]), sys.argv[2]
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)

    class P:   # a bare page as synth.Page has it
        def __init__(self, h, w):
            self.rng, self.h, self.w = rng, h, w
            self.ink = np.ones((h, w, 3), np.float32)

    tiles = []
    for _ in range(n):
        p = P(420, 640)
        block(p, 10, 10, 630, 410)
        tiles.append((np.clip(p.ink, 0, 1) * 255).astype(np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - 2, 3)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
