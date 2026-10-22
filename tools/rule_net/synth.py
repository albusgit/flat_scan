"""Synthetic training samples for the rule net: a stage-1 page (already flattened by the nets, so
mostly flat with residual warps) and the masks of its printed straight rules.

A sample is drawn at stage-1 scale (pages 1600 px wide: rules 1-6 px, text 10-40 px): forms and
tables (full grids, row rules only, boxed headers, merged cells, double borders, shaded and coloured
cells, rules in colour or pale grey, rule pieces, text touching or crossing rules), underlines and
fill-in lines, chart gridlines, form boxes. Negatives, never labelled: text in Hebrew and Latin of
every size and weight (justified columns too), headings, ellipses, circles, arcs and the round
corners of boxes, diagonal lines, hand-drawn and wavy strokes, signatures, stamps, dashed and dotted
lines, barcodes, colour blocks and highlighter, textures standing in for photos, the page edge with
the table beside it, and shadows.

The page is warped as a stage-1 page still is (a smooth residual field, a curled corner whose rows
fan together, a fold crease, a small turn), shaded, blurred, noised and JPEG'd; the masks get the
same warp. Returned at half scale (the net's input), masks as float in [0, 1].

  sample(rng, size=448) -> (img uint8 HxWx3 BGR, mask float32 2xHxW: [horizontal, vertical])
  python synth.py N OUT.jpg    contact sheet of N samples (page | rules: red H, blue V)
"""
import math, random, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

import os
FONT_DIRS = '/usr/share/fonts/truetype/'
if os.path.isdir(FONT_DIRS + 'dejavu'):
    FONTS = [FONT_DIRS + f for f in (
        'dejavu/DejaVuSans.ttf', 'dejavu/DejaVuSans-Bold.ttf',
        'liberation/LiberationSans-Regular.ttf', 'liberation/LiberationSans-Bold.ttf', 'liberation/LiberationSerif-Regular.ttf',
        'liberation/LiberationSerif-Bold.ttf', 'liberation/LiberationMono-Regular.ttf',
        'freefont/FreeSans.ttf', 'freefont/FreeSansBold.ttf', 'freefont/FreeSerif.ttf', 'freefont/FreeSerifBold.ttf',
        'freefont/FreeMono.ttf', 'freefont/FreeMonoBold.ttf')]   # all with Hebrew glyphs (DejaVu Serif / Mono have none)
else:   # Windows (round 11): fonts with Hebrew glyphs; 1, 3, 8, 10 are bold (headings)
    FONTS = ['C:/Windows/Fonts/' + f for f in (
        'arial.ttf', 'arialbd.ttf', 'calibri.ttf', 'calibrib.ttf', 'times.ttf', 'timesbd.ttf', 'cour.ttf',
        'tahoma.ttf', 'tahomabd.ttf', 'david.ttf', 'davidbd.ttf', 'frank.ttf', 'courbd.ttf', 'gisha.ttf',
        'mriam.ttf', 'segoeui.ttf', 'micross.ttf', 'rod.ttf')]
_font_cache = {}
HEB = 'אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
LAT = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lat = LAT.lower()


def font(path, size):
    k = (path, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]


def word(rng):
    r = rng.random()
    n = rng.randint(1, 9)
    if r < 0.45:
        return ''.join(rng.choice(HEB) for _ in range(n)) + (rng.choice(['"', "'", '.', ',', '']) if rng.random() < 0.2 else '')
    if r < 0.65:   # SAP-like codes
        parts = [''.join(rng.choice(LAT) for _ in range(rng.randint(2, 6))) for _ in range(rng.randint(1, 3))]
        return rng.choice(['', '=/', '/', 'S_']) + rng.choice(['_', '/', '']).join(parts) + (str(rng.randint(0, 99999)) if rng.random() < 0.3 else '')
    if r < 0.8:
        return ''.join(rng.choice(lat) for _ in range(n)).capitalize() if rng.random() < 0.3 else ''.join(rng.choice(lat) for _ in range(n))
    if r < 0.93:
        return str(rng.randint(0, 10 ** rng.randint(1, 7))) + rng.choice(['', '.00', ',5', '%'])
    return rng.choice(['EMF', 'PM', 'ERP', 'BW', '(', ')', '-', '–', ':', '/', '&', '#', 'No.', '...'])


def phrase(rng, maxw, fnt):
    s = word(rng)
    for _ in range(rng.randint(0, 8)):
        t = s + ' ' + word(rng)
        if fnt.getlength(t) > maxw:
            break
        s = t
    return s


class Page:
    """Drawing surface at stage-1 scale: ink (float BGR, multiplicative over the paper), the two
    rule masks, and a PIL layer for text."""

    def __init__(self, rng, h, w):
        self.rng, self.h, self.w = rng, h, w
        self.ink = np.ones((h, w, 3), np.float32)
        self.mh = np.zeros((h, w), np.uint8)
        self.mv = np.zeros((h, w), np.uint8)

    def colour(self, kind='ink'):
        rng = self.rng
        r = rng.random()
        if kind == 'rule':
            if r < 0.55: v = rng.uniform(0.0, 0.25); return (v, v, v)
            if r < 0.8: v = rng.uniform(0.3, 0.72); return (v, v, v)            # pale grey form rules
            if r < 0.9: return (rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.4), rng.uniform(0.0, 0.2))   # blue (BGR)
            return (rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3), rng.uniform(0.4, 0.8))                # red
        if r < 0.75: v = rng.uniform(0.0, 0.2); return (v, v, v)
        if r < 0.88: v = rng.uniform(0.25, 0.6); return (v, v, v)
        if r < 0.94: return (rng.uniform(0.4, 0.8), rng.uniform(0.1, 0.4), rng.uniform(0.0, 0.2))
        return (rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3), rng.uniform(0.5, 0.9))

    def paint(self, mask, col):
        """multiply the ink layer by col where mask (uint8 0..255) is set"""
        a = mask.astype(np.float32)[..., None] / 255
        self.ink *= 1 - a * (1 - np.float32(col))

    def rect(self, x0, y0, x1, y1, col, lab):
        """a filled axis-aligned rule rectangle (inclusive px), painted in place and labelled in lab"""
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w - 1, int(x1)), min(self.h - 1, int(y1))
        if x1 < x0 or y1 < y0:
            return
        self.ink[y0:y1 + 1, x0:x1 + 1] *= np.float32(col)
        if lab is not None:
            lab[y0:y1 + 1, x0:x1 + 1] = 255

    def hline(self, y, x0, x1, t, col, label=True):
        y0 = int(round(y - t / 2))
        self.rect(x0, y0, x1, y0 + max(1, int(round(t))) - 1, col, self.mh if label else None)

    def vline(self, x, y0, y1, t, col, label=True):
        x0 = int(round(x - t / 2))
        self.rect(x0, y0, x0 + max(1, int(round(t))) - 1, y1, col, self.mv if label else None)

    def fill(self, x0, y0, x1, y1, col):
        self.ink[int(y0):int(y1), int(x0):int(x1)] *= np.float32(col)

    def text(self, x, y, s, fnt, col, anchor='la'):
        im = Image.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(im).text((x, y), s, font=fnt, fill=255, anchor=anchor)
        self.paint(np.asarray(im), col)

    def texts(self, items):
        """many text items in one PIL pass: (x, y, s, fnt, anchor) with one colour each"""
        by_col = {}
        for x, y, s, fnt, col, anchor in items:
            by_col.setdefault(col, []).append((x, y, s, fnt, anchor))
        for col, its in by_col.items():
            im = Image.new('L', (self.w, self.h), 0)
            d = ImageDraw.Draw(im)
            for x, y, s, fnt, anchor in its:
                d.text((x, y), s, font=fnt, fill=255, anchor=anchor)
            self.paint(np.asarray(im), col)


def pastel(rng):
    h = rng.randrange(180)
    hsv = np.uint8([[[h, int(rng.uniform(0.05, 0.5) * 255), int(rng.uniform(0.8, 1) * 255)]]])
    c = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] / 255.0
    if rng.random() < 0.3:
        v = rng.uniform(0.75, 0.95); return (v, v, v)
    return tuple(float(q) for q in c)


def table(pg, x0, y0, x1, y1):
    """a table or form grid inside the box; returns nothing (draws text, fills, rules, masks)"""
    rng = pg.rng
    style = rng.choice(['grid', 'grid', 'grid', 'rows', 'rows', 'boxed', 'cols'])
    t = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5, 6]) * rng.uniform(0.8, 1.2)
    tout = t * rng.choice([1, 1, 1, 1.5, 2])
    col = pg.colour('rule')
    fsz = rng.randint(11, 30)
    fp = rng.choice(FONTS)
    fnt = font(fp, fsz)
    rh = fsz * rng.uniform(1.25, 2.6)
    ys = [y0]
    while ys[-1] + rh < y1:
        ys.append(ys[-1] + rh * (rng.choice([1, 1, 1, 1, 2, 3]) if rng.random() < 0.25 else 1) * rng.uniform(0.95, 1.05))
    ys = [y for y in ys if y <= y1]
    if len(ys) < 2:
        return
    ncol = rng.randint(1, 6)
    cuts = sorted(rng.uniform(0.08, 0.92) for _ in range(ncol - 1))
    xs = [x0] + [x0 + c * (x1 - x0) for c in cuts] + [x1]
    xs = [xs[0]] + [xs[i] for i in range(1, len(xs)) if xs[i] - xs[i - 1] > 3 * fsz]
    if xs[-1] != x1:
        xs[-1] = x1
    # header fill, shaded cells, coloured column
    if rng.random() < 0.4:
        pg.fill(x0, ys[0], x1, ys[1], pastel(rng))
    for _ in range(rng.choice([0, 0, 0, 1, 2, 4])):
        i, j = rng.randrange(len(ys) - 1), rng.randrange(len(xs) - 1)
        pg.fill(xs[j], ys[i], xs[j + 1], ys[i + 1], pastel(rng))
    # text
    items = []
    rtl = rng.random() < 0.6
    tcol = pg.colour()
    tight = rng.random() < 0.25   # text set against the rule (right-aligned Hebrew ending at its column rule)
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            if rng.random() < 0.25:
                continue
            cw = xs[j + 1] - xs[j]
            f2 = font(rng.choice(FONTS), fsz) if rng.random() < 0.15 else fnt
            s = phrase(rng, cw - 8, f2)
            cy = (ys[i] + min(ys[i + 1], ys[i] + rh)) / 2
            pad = rng.uniform(1, 4) if tight else rng.uniform(4, 12)
            al = rng.random()
            if al < 0.4:
                items.append((xs[j + 1] - pad if rtl else xs[j] + pad, cy, s, f2, tcol, 'rm' if rtl else 'lm'))
            elif al < 0.8:
                items.append(((xs[j] + xs[j + 1]) / 2, cy, s, f2, tcol, 'mm'))
            else:
                items.append((xs[j] + pad if rtl else xs[j + 1] - pad, cy, s, f2, tcol, 'lm' if rtl else 'rm'))
            if ys[i + 1] - ys[i] > 2.2 * rh and rng.random() < 0.6:   # a second line in a tall cell
                items.append(((xs[j] + xs[j + 1]) / 2, cy + fsz * 1.2, phrase(rng, cw - 8, f2), f2, tcol, 'mm'))
    pg.texts(items)
    # rules
    merged = set()
    for _ in range(rng.choice([0, 0, 1, 3, 6])):   # merged cells: a rule piece left out
        merged.add((rng.randrange(1, max(2, len(ys) - 1)), rng.randrange(len(xs) - 1)))
    for i, y in enumerate(ys):
        outer = i in (0, len(ys) - 1)
        if style == 'cols' and not outer and rng.random() < 0.9:
            continue
        for j in range(len(xs) - 1):
            if (i, j) in merged and not outer:
                continue
            gap = rng.random() < 0.03
            xa, xb = xs[j] - t / 2, xs[j + 1] + t / 2
            if gap:
                xb = xa + (xb - xa) * rng.uniform(0.2, 0.8)
            pg.hline(y, xa, xb, tout if outer else t, col)
    for j, x in enumerate(xs):
        outer = j in (0, len(xs) - 1)
        if style == 'rows' and (not outer or rng.random() < 0.5):
            continue
        if style == 'boxed' and not outer:
            pg.vline(x, ys[0], ys[1], t, col)   # header box divisions only
            continue
        pg.vline(x, ys[0] - tout / 2, ys[-1] + tout / 2, tout if outer else t, col)
    if rng.random() < 0.12:   # double outer border
        d = rng.uniform(3, 8)
        pg.hline(ys[0] - d - t, xs[0] - d - t, xs[-1] + d + t, t, col); pg.hline(ys[-1] + d + t, xs[0] - d - t, xs[-1] + d + t, t, col)
        pg.vline(xs[0] - d - t, ys[0] - d - t, ys[-1] + d + t, t, col); pg.vline(xs[-1] + d + t, ys[0] - d - t, ys[-1] + d + t, t, col)


def paragraph(pg, x0, y0, x1, y1):
    rng = pg.rng
    fsz = rng.randint(10, 34) if rng.random() < 0.85 else rng.randint(35, 70)
    fnt = font(rng.choice(FONTS), fsz)
    lh = fsz * rng.uniform(1.1, 1.8)
    col = pg.colour()
    items = []
    rtl = rng.random() < 0.5
    y = y0 + fsz
    just = rng.random() < 0.4
    while y < y1 - fsz / 2:
        s = phrase(rng, x1 - x0, fnt)
        if just:   # justified: words spread over the full width (a straight column edge, not a rule)
            ws = s.split(' ')
            if len(ws) > 1:
                tot = sum(fnt.getlength(q) for q in ws)
                gap = max(0.0, (x1 - x0 - tot) / (len(ws) - 1))
                gap = min(gap, fsz * 2.0)
                x = x0
                for q in ws:
                    items.append((x, y, q, fnt, col, 'ls'))
                    x += fnt.getlength(q) + gap
                y += lh
                continue
        items.append((x1 if rtl else x0, y, s, fnt, col, 'rs' if rtl else 'ls'))
        if rng.random() < 0.08:   # underlined words: a short straight rule
            w0 = fnt.getlength(s.split(' ')[0])
            ux0 = x1 - w0 if rtl else x0
            pg.hline(y + fsz * 0.15, ux0, ux0 + w0, max(1, fsz / 14), col)
        y += lh
    pg.texts(items)


def negatives(pg):
    """shapes that are not rules"""
    rng = pg.rng
    h, w = pg.h, pg.w
    for _ in range(rng.choice([0, 1, 1, 2, 3, 5])):
        k = rng.random()
        col = pg.colour('rule')
        t = max(1, int(rng.choice([1, 2, 3, 4, 6]) * rng.uniform(0.8, 1.3)))
        m = np.zeros((h, w), np.uint8)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        if k < 0.25:   # ellipses, circles, arcs
            ax = (int(rng.uniform(15, 300)), int(rng.uniform(15, 220)))
            a0 = rng.choice([0, 0, rng.uniform(0, 360)]); a1 = a0 + rng.choice([360, rng.uniform(40, 300)])
            cv2.ellipse(m, (int(cx), int(cy)), ax, rng.uniform(0, 180), a0, a1, 255, t, cv2.LINE_AA)
        elif k < 0.4:   # round-cornered box: straight sides are rules, the corners are not
            bw, bh, r = rng.uniform(80, 500), rng.uniform(40, 300), rng.uniform(8, 40)
            x0, y0 = cx, cy
            for (ex, ey, a0) in ((x0 + r, y0 + r, 180), (x0 + bw - r, y0 + r, 270), (x0 + bw - r, y0 + bh - r, 0), (x0 + r, y0 + bh - r, 90)):
                cv2.ellipse(m, (int(ex), int(ey)), (int(r), int(r)), 0, a0, a0 + 90, 255, t, cv2.LINE_AA)
            pg.hline(y0 + t / 2, x0 + r, x0 + bw - r, t, col); pg.hline(y0 + bh - t / 2, x0 + r, x0 + bw - r, t, col)
            pg.vline(x0 + t / 2, y0 + r, y0 + bh - r, t, col); pg.vline(x0 + bw - t / 2, y0 + r, y0 + bh - r, t, col)
        elif k < 0.52:   # diagonal lines (straight, but not along an axis)
            ang = rng.choice([rng.uniform(8, 82), rng.uniform(98, 172)]) * math.pi / 180
            L = rng.uniform(60, 600)
            cv2.line(m, (int(cx), int(cy)), (int(cx + L * math.cos(ang)), int(cy + L * math.sin(ang))), 255, t, cv2.LINE_AA)
        elif k < 0.67:   # hand-drawn / wavy strokes, signatures
            n = rng.randint(3, 14)
            pts = [(cx, cy)]
            for _ in range(n):
                pts.append((pts[-1][0] + rng.uniform(-80, 120), pts[-1][1] + rng.uniform(-50, 50)))
            pts = np.array(pts, np.float32)
            tt = np.linspace(0, 1, 30 * n)
            xs = np.interp(tt, np.linspace(0, 1, len(pts)), pts[:, 0]); ys = np.interp(tt, np.linspace(0, 1, len(pts)), pts[:, 1])
            xs = cv2.GaussianBlur(xs[None], (1, 0), 6).ravel() if len(xs) > 9 else xs
            ys = cv2.GaussianBlur(ys[None], (1, 0), 6).ravel() if len(ys) > 9 else ys
            cv2.polylines(m, [np.stack([xs, ys], 1).astype(np.int32)], False, 255, t, cv2.LINE_AA)
        elif k < 0.79:   # dashed and dotted lines along an axis (not traced as rules)
            hz = rng.random() < 0.6
            L = rng.uniform(100, 900); dash, gap = rng.choice([(2, 4), (3, 3), (6, 4), (10, 6), (15, 8)])
            s = 0
            while s < L:
                if hz: cv2.rectangle(m, (int(cx + s), int(cy)), (int(cx + s + dash), int(cy + t - 1)), 255, -1)
                else: cv2.rectangle(m, (int(cx), int(cy + s)), (int(cx + t - 1), int(cy + s + dash)), 255, -1)
                s += dash + gap
        elif k < 0.86:   # barcode
            x = cx; bh = rng.uniform(30, 120)
            for _ in range(rng.randint(20, 60)):
                bw = rng.choice([1, 2, 3, 4]); cv2.rectangle(m, (int(x), int(cy)), (int(x + bw - 1), int(cy + bh)), 255, -1); x += bw + rng.choice([1, 2, 3, 4])
        elif k < 0.93:   # stamp: circles and text
            r = int(rng.uniform(40, 110)); col = (rng.uniform(0.4, 0.9), rng.uniform(0.1, 0.4), rng.uniform(0.0, 0.3)) if rng.random() < 0.6 else (0.2, 0.2, 0.8)
            cv2.circle(m, (int(cx), int(cy)), r, 255, t, cv2.LINE_AA); cv2.circle(m, (int(cx), int(cy)), int(r * 0.75), 255, max(1, t - 1), cv2.LINE_AA)
            pg.text(cx, cy, word(rng), font(rng.choice(FONTS), int(r * 0.35)), col, 'mm')
        else:   # big bold heading
            pg.text(cx, cy, phrase(rng, w / 2, font(FONTS[1], 40)), font(rng.choice([FONTS[1], FONTS[3], FONTS[8], FONTS[10]]), rng.randint(36, 90)), pg.colour(), 'mm')
        pg.paint(m, col)
    for _ in range(rng.choice([0, 0, 0, 1, 2])):   # colour blocks and highlighter: their edges are not rules
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        bw, bh = rng.uniform(40, 600), rng.uniform(15, 400)
        if rng.random() < 0.5:
            bh = rng.uniform(15, 40)
        pg.fill(max(0, x0), max(0, y0), min(w, x0 + bw), min(h, y0 + bh), pastel(rng) if rng.random() < 0.7 else
                tuple(float(q) for q in cv2.cvtColor(np.uint8([[[rng.randrange(180), rng.randint(120, 255), rng.randint(60, 230)]]]), cv2.COLOR_HSV2BGR)[0, 0] / 255.0))
    if rng.random() < 0.15:   # a picture: smooth random texture
        x0, y0 = int(rng.uniform(0, w * 0.7)), int(rng.uniform(0, h * 0.7))
        pw, ph = int(rng.uniform(100, w / 2)), int(rng.uniform(100, h / 2))
        g = np.random.default_rng(rng.randrange(1 << 30))
        tex = cv2.resize(g.uniform(0, 1, (rng.randint(2, 12), rng.randint(2, 12), 3)).astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        tex = np.clip(tex + g.normal(0, 0.08, tex.shape).astype(np.float32), 0.02, 1)
        sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
        pg.ink[sl] = tex[:pg.ink[sl].shape[0], :pg.ink[sl].shape[1]]
        pg.mh[sl] = 0; pg.mv[sl] = 0   # the picture covers the rules under it
        if hasattr(pg, 'edge_rect'):   # round 11 (synth2): a picture's straight sides
            pg.edge_rect(x0, y0, x0 + pw, y0 + ph)


def standalone_rules(pg):
    """rules outside tables: separators, fill-in lines, boxes, chart gridlines"""
    rng = pg.rng
    h, w = pg.h, pg.w
    for _ in range(rng.choice([0, 0, 1, 2, 4])):
        col = pg.colour('rule'); t = max(1, rng.choice([1, 2, 3, 4]) * rng.uniform(0.8, 1.3))
        k = rng.random()
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        if k < 0.35:   # separator or fill-in line
            L = rng.uniform(60, w)
            pg.hline(y0, x0, x0 + L, t, col)
        elif k < 0.55:   # vertical separator
            L = rng.uniform(60, h)
            pg.vline(x0, y0, y0 + L, t, col)
        elif k < 0.75:   # boxes (form fields, check boxes)
            for _ in range(rng.randint(1, 8)):
                bw, bh = rng.uniform(14, 300), rng.uniform(14, 90)
                pg.hline(y0, x0, x0 + bw, t, col); pg.hline(y0 + bh, x0, x0 + bw, t, col)
                pg.vline(x0, y0, y0 + bh, t, col); pg.vline(x0 + bw, y0, y0 + bh, t, col)
                if rng.random() < 0.5: x0 += bw + rng.uniform(5, 40)
                else: y0 += bh + rng.uniform(5, 40)
        else:   # chart: gridlines (rules) with a data curve and bars (not rules)
            cw, ch = rng.uniform(200, 700), rng.uniform(150, 500)
            n = rng.randint(3, 10)
            gcol = pg.colour('rule') if rng.random() < 0.5 else (0.7, 0.7, 0.7)
            for i in range(n + 1):
                pg.hline(y0 + ch * i / n, x0, x0 + cw, max(1, t / 2), gcol)
            if rng.random() < 0.5:
                for i in range(n + 1):
                    pg.vline(x0 + cw * i / n, y0, y0 + ch, max(1, t / 2), gcol)
            m = np.zeros((h, w), np.uint8)
            xs = np.linspace(x0, x0 + cw, 40); ys = y0 + ch * (0.5 + 0.4 * np.sin(np.linspace(0, rng.uniform(2, 9), 40) + rng.uniform(0, 6)))
            cv2.polylines(m, [np.stack([xs, ys], 1).astype(np.int32)], False, 255, int(t) + 1, cv2.LINE_AA)
            pg.paint(m, pg.colour())


def layout(rng, h, w, page_cls=None):
    pg = (page_cls or Page)(rng, h, w)
    # blocks stacked down the page: tables, paragraphs; sometimes two columns
    y = rng.uniform(-0.3, 0.05) * h
    while y < h:
        bh = rng.uniform(0.15, 0.9) * h if rng.random() < 0.6 else rng.uniform(0.8, 1.6) * h
        r = rng.random()
        mx = rng.uniform(-0.2, 0.15) * w
        if r < 0.72:
            table(pg, mx, y, w - rng.uniform(-0.2, 0.15) * w, y + bh)
        elif r < 0.85:
            if rng.random() < 0.3:
                paragraph(pg, mx, y, w / 2 - 15, y + bh); paragraph(pg, w / 2 + 15, y, w - mx, y + bh)
            else:
                paragraph(pg, mx, y, w - mx, y + bh)
        y += bh + (rng.uniform(10, 80) if rng.random() < 0.8 else rng.uniform(100, 400))
    standalone_rules(pg)
    negatives(pg)
    return pg


def warp_field(rng, h, w):
    """backward map (map_x, map_y) of a stage-1 page's residual warp"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.random.default_rng(rng.randrange(1 << 30))
    dx = np.zeros((h, w), np.float32); dy = np.zeros((h, w), np.float32)
    amp = rng.choice([0, 1, 2, 4, 6, 10])
    for d in (dx, dy):
        c = rng.randint(2, 6)
        d += cv2.resize(g.normal(0, 1, (c, c)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * amp
    if rng.random() < 0.35:   # curled corner / edge: displacement growing toward it, rows fan together
        cx, cy = rng.choice([0, w]), rng.choice([0, h])
        R = rng.uniform(0.25, 0.7) * max(h, w)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        s = np.clip(1 - dist / R, 0, 1) ** rng.uniform(1.5, 3)
        A = rng.uniform(15, 70)
        dy += s * A * (1 if cy == 0 else -1) * rng.uniform(0.3, 1)
        dx += s * A * (1 if cx == 0 else -1) * rng.uniform(0, 0.6)
    if rng.random() < 0.2:   # fold crease: a kink along a line
        ang = rng.uniform(0, math.pi)
        nx, ny = math.cos(ang), math.sin(ang)
        off = rng.uniform(0.2, 0.8) * (w * abs(nx) + h * abs(ny))
        dd = xx * nx + yy * ny - off
        k = rng.uniform(-0.03, 0.03)
        f = np.abs(dd) * k
        dx += f * -ny; dy += f * nx
    ang = rng.uniform(-1.2, 1.2) * math.pi / 180   # a small turn the nets left
    cx, cy = w / 2, h / 2
    mx = cx + (xx - cx) * math.cos(ang) - (yy - cy) * math.sin(ang) + dx
    my = cy + (xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang) + dy
    return mx, my


def surround(rng, img, mh, mv):
    """the page edge: part of the frame beyond the paper (table, background), never rules"""
    h, w = img.shape[:2]
    if rng.random() > 0.2:
        return
    side = rng.randrange(4)
    d = int(rng.uniform(0.03, 0.25) * (w if side % 2 else h))
    g = np.random.default_rng(rng.randrange(1 << 30))
    base = np.float32([rng.uniform(0.1, 0.7), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.5)])
    wav = (np.sin(np.arange(max(h, w)) * rng.uniform(0.002, 0.02)) * rng.uniform(0, 12)).astype(np.int32)
    for i in range(max(h, w)):
        if side == 0 and i < w: img[:max(0, d + wav[i]), i] = base; mh[:max(0, d + wav[i]), i] = 0; mv[:max(0, d + wav[i]), i] = 0
        if side == 1 and i < h: img[i, w - max(0, d + wav[i]):] = base; mh[i, w - max(0, d + wav[i]):] = 0; mv[i, w - max(0, d + wav[i]):] = 0
        if side == 2 and i < w: img[h - max(0, d + wav[i]):, i] = base; mh[h - max(0, d + wav[i]):, i] = 0; mv[h - max(0, d + wav[i]):, i] = 0
        if side == 3 and i < h: img[i, :max(0, d + wav[i])] = base; mh[i, :max(0, d + wav[i])] = 0; mv[i, :max(0, d + wav[i])] = 0
    img *= 1 + g.normal(0, 0.05, img.shape).astype(np.float32)


def degrade(rng, img):
    h, w = img.shape[:2]
    g = np.random.default_rng(rng.randrange(1 << 30))
    paper = np.float32([rng.uniform(0.9, 1), rng.uniform(0.9, 1), rng.uniform(0.9, 1)]) if rng.random() < 0.8 else \
        np.float32([rng.uniform(0.8, 1), rng.uniform(0.8, 1), rng.uniform(0.8, 1)])
    paper = paper / paper.max() * rng.uniform(0.75, 1.0)
    shade = 1 + cv2.resize(g.normal(0, 1, (4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(0, 0.12)
    out = img * paper * shade[..., None]
    if rng.random() < 0.25:   # hard cast shadow
        m = np.zeros((h, w), np.float32)
        pts = np.array([[rng.uniform(-w, 2 * w), rng.uniform(-h, 2 * h)] for _ in range(rng.randint(3, 6))], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(3, 25))
        out *= (1 - m * rng.uniform(0.2, 0.55))[..., None]
    s = rng.choice([0, 0.4, 0.7, 1.0, 1.4, 2.0])
    if s > 0:
        out = cv2.GaussianBlur(out, (0, 0), s)
    if rng.random() < 0.15:   # a low-resolution photo blown up to page size
        f = rng.uniform(0.35, 0.7)
        out = cv2.resize(cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:   # phone sharpening
        b = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = out + (out - b) * rng.uniform(0.3, 1.2)
    out = out + g.normal(0, rng.uniform(0, 0.03), out.shape).astype(np.float32)
    return out


def sample(rng, size=448):
    """img uint8 size x size x 3 (BGR, half stage-1 scale), mask float32 2 x size x size"""
    H = W = 2 * size + 64
    pg = layout(rng, H, W)
    mx, my = warp_field(rng, H, W)
    img = cv2.remap(pg.ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    mh = cv2.remap(pg.mh, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mv = cv2.remap(pg.mv, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    surround(rng, img, mh, mv)
    img = degrade(rng, img)
    img = img[32:32 + 2 * size, 32:32 + 2 * size]
    mh, mv = mh[32:32 + 2 * size, 32:32 + 2 * size], mv[32:32 + 2 * size, 32:32 + 2 * size]
    small = cv2.resize(np.clip(img, 0, 1), (size, size), interpolation=cv2.INTER_AREA)
    q = rng.randint(45, 95)
    ok, enc = cv2.imencode('.jpg', (small * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # a rule pixel at half scale: any rule ink in its 2x2 block, rules widened by 1 px across first
    k = np.ones((3, 1), np.uint8)
    mh = cv2.dilate(mh, k); mv = cv2.dilate(mv, k.T)
    m = np.stack([cv2.resize(mh, (size, size), interpolation=cv2.INTER_AREA), cv2.resize(mv, (size, size), interpolation=cv2.INTER_AREA)]).astype(np.float32) / 255
    m = np.clip(m * 2, 0, 1)
    return small, m


if __name__ == '__main__':
    n, out = int(sys.argv[1]), sys.argv[2]
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    rows = []
    for i in range(n):
        im, m = sample(rng, 384)
        v = im.copy()
        v[m[0] > 0.5] = (0, 0, 255); v[m[1] > 0.5] = (255, 0, 0)
        rows.append(np.hstack([im, v]))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
