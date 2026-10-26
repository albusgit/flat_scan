"""Round 11 generator for the rule net: synth.py's pages plus what the real pages showed the Round 10
net getting wrong (DocUNet 13_1 / 5_2, the owner's photo 1), as hard negatives, and more realistic
paper and camera.

Added on top of synth.layout (never labelled unless said):
- show-through: the back side's own page (mirrored, blurred by the paper, darkness ~4 - 24) and,
  beyond the page edge, a sheet behind with its own rules;
- faint printed rules on the front (darkness ~25 - 70): labelled, the boundary to show-through;
- display headings in heavy fonts (Impact, Arial Black, bold caps: straight letter stems), barcodes
  of every size (also turned 90 deg), 2-D codes, handwriting (script fonts, hand-drawn near-straight
  strokes, check marks, circles), text-only pages (justified columns, academic layout);
- creases and crumples (thin light / dark shading lines, also near an axis), stains, specks, paper fibre;
- camera: motion and defocus blur, colour cast, gamma, washed-out contrast, chroma noise.

  sample(rng, size=400) -> (img uint8 HxWx3 BGR, mask float32 2xHxW)   (as synth.sample)
  python synth2.py N OUT.jpg [SEED]    contact sheet (page | rules: red H, blue V)
"""
import math, random, sys, os
import numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
from synth import font, FONTS

WF = os.path.join(os.environ.get('RN_FONT_DIR', 'C:/Windows/Fonts'), '')
def _have(fs):
    return [WF + f for f in fs if os.path.exists(WF + f)] or FONTS
# the Windows faces first; on Linux (RN_FONT_DIR: tools/train/setup's font folder) the open Google fonts stand in
HEAVY_LAT = _have(['impact.ttf', 'ariblk.ttf', 'seguibl.ttf', 'bahnschrift.ttf', 'GOTHICB.TTF', 'framd.ttf', 'ARIALNB.TTF',
                   'verdanab.ttf', 'georgiab.ttf', 'trebucbd.ttf', 'Candarab.ttf', 'corbelb.ttf', 'consolab.ttf', 'BOOKOSB.TTF',
                   'SuezOne-Regular.ttf', 'Karantina-Bold.ttf', 'PT_Sans-Web-Bold.ttf', 'Tinos-Bold.ttf', 'Cousine-Bold.ttf', 'PT_Serif-Web-Bold.ttf'])
HEAVY_HEB = _have(['ahronbd.ttf', 'arialbd.ttf', 'gishabd.ttf', 'davidbd.ttf', 'tahomabd.ttf', 'timesbd.ttf', 'courbd.ttf',
                   'SecularOne-Regular.ttf', 'SuezOne-Regular.ttf', 'Alef-Bold.ttf', 'DavidLibre-Bold.ttf', 'Karantina-Bold.ttf'])
LAT_FONTS = _have(['GARA.TTF', 'CENTURY.TTF', 'BKANT.TTF', 'pala.ttf', 'georgia.ttf', 'verdana.ttf', 'trebuc.ttf', 'cambriai.ttf',
                   'constan.ttf', 'corbel.ttf', 'consola.ttf', 'lucon.ttf', 'SitkaVF.ttf', 'ARIALN.TTF', 'GOTHIC.TTF',
                   'Tinos-Regular.ttf', 'PT_Serif-Web-Regular.ttf', 'PT_Sans-Web-Regular.ttf', 'Cousine-Regular.ttf', 'Bellefair-Regular.ttf',
                   'VarelaRound-Regular.ttf', 'Amiri-Regular.ttf'])
HAND = _have(['Inkfree.ttf', 'segoesc.ttf', 'segoepr.ttf', 'LHANDW.TTF', 'BRADHITC.TTF', 'comic.ttf', 'mvboli.ttf',
              'FREESCPT.TTF', 'ITCKRIST.TTF', 'PRISTINA.TTF', 'MISTRAL.TTF', 'TEMPSITC.TTF',
              'IndieFlower-Regular.ttf', 'Kalam-Regular.ttf', 'Kalam-Light.ttf', 'PatrickHand-Regular.ttf', 'AmaticSC-Regular.ttf'])
LAT = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def lat_word(rng, caps=False):
    n = rng.randint(2, 10)
    w = ''.join(rng.choice(LAT) for _ in range(n))
    return w if caps else (w.lower().capitalize() if rng.random() < 0.4 else w.lower())


def lat_phrase(rng, maxw, fnt, caps=False):
    s = lat_word(rng, caps)
    for _ in range(rng.randint(0, 6)):
        t = s + ' ' + lat_word(rng, caps)
        if fnt.getlength(t) > maxw:
            break
        s = t
    return s


class Page2(synth.Page):
    """synth.Page whose rule colours are sometimes faint (darkness 25 - 70 on white paper), and which
    records the straight sides of pictures, dark bars and dark colour blocks (eh / ev: 2-px bands
    inside the side) - straight printed edges the line stage can use, kept apart from the rules"""
    faint_p = 0.0

    def __init__(self, rng, h, w):
        super().__init__(rng, h, w)
        self.eh = np.zeros((h, w), np.uint8)
        self.ev = np.zeros((h, w), np.uint8)

    def edge_rect(self, x0, y0, x1, y1, t=2):
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return
        cx0, cx1 = max(0, x0), min(self.w, x1); cy0, cy1 = max(0, y0), min(self.h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        for y in (y0, y1 - t):
            if 0 <= y and y + t <= self.h: self.eh[y:y + t, cx0:cx1] = 255
        for x in (x0, x1 - t):
            if 0 <= x and x + t <= self.w: self.ev[cy0:cy1, x:x + t] = 255

    def fill(self, x0, y0, x1, y1, col):
        super().fill(x0, y0, x1, y1, col)
        if 1 - min(col) >= 0.2:
            self.edge_rect(x0, y0, x1, y1)

    def colour(self, kind='ink'):
        if kind == 'rule' and self.rng.random() < self.faint_p:
            v = self.rng.uniform(0.72, 0.9)
            return (v, v, v) if self.rng.random() < 0.8 else (min(1, v + 0.08), v, min(1, v + 0.05))
        return super().colour(kind)


def text_page(pg):
    """mostly text: paragraphs in one or two columns, headings, a few short rules (fraction bars,
    separators) - DocUNet-like academic / book pages"""
    rng = pg.rng
    h, w = pg.h, pg.w
    y = rng.uniform(-0.2, 0.05) * h
    two = rng.random() < 0.35
    while y < h:
        bh = rng.uniform(0.1, 0.6) * h
        mx = rng.uniform(-0.1, 0.12) * w
        if two:
            g = rng.uniform(15, 50)
            synth.paragraph(pg, mx, y, w / 2 - g, y + bh); synth.paragraph(pg, w / 2 + g, y, w - mx, y + bh)
        else:
            synth.paragraph(pg, mx, y, w - mx, y + bh)
        y += bh + rng.uniform(10, 60)
        if rng.random() < 0.35:   # a heading
            f = font(rng.choice(FONTS[1::2] + HEAVY_LAT[:4]), rng.randint(18, 40))
            pg.text(mx + 5, y, lat_phrase(rng, w * 0.7, f), f, pg.colour(), 'ls')
            y += rng.uniform(20, 60)
        if rng.random() < 0.3:   # a picture (sometimes framed by a thin rule) or a dark title bar
            x0 = rng.uniform(0, w * 0.6); pw = rng.uniform(120, w * 0.6); ph = rng.uniform(80, h * 0.35)
            if rng.random() < 0.7:
                picture(pg, x0, y, x0 + pw, y + ph)
                y += ph + rng.uniform(10, 40)
            else:
                dark_bar(pg, x0, y, x0 + pw, y + rng.uniform(25, 70))
                y += rng.uniform(40, 90)
        if rng.random() < 0.15:   # fraction bars / a short separator: rules
            col = pg.colour('rule')
            for _ in range(rng.randint(1, 3)):
                x0 = rng.uniform(0, w * 0.8)
                pg.hline(y + rng.uniform(0, 40), x0, x0 + rng.uniform(40, 260), rng.choice([1, 1.5, 2]), col)


def picture(pg, x0, y0, x1, y1):
    """a photo: smooth texture with detail, its sides recorded as straight edges; a thin frame rule
    around it is a rule"""
    rng = pg.rng
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(pg.w, x1)), int(min(pg.h, y1))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return
    g = np.random.default_rng(rng.randrange(1 << 30))
    tex = cv2.resize(g.uniform(0, 1, (rng.randint(2, 10), rng.randint(2, 10), 3)).astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    det = cv2.resize(g.uniform(-1, 1, (rng.randint(10, 60), rng.randint(10, 60))).astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    tex = tex * rng.uniform(0.3, 1) + cv2.GaussianBlur(det, (0, 0), rng.uniform(0.5, 3))[..., None] * rng.uniform(0, 0.25)
    if rng.random() < 0.4:
        tex = np.repeat(tex.mean(2, keepdims=True), 3, 2)
    tex = np.clip(tex * rng.uniform(0.4, 0.9) + rng.uniform(0, 0.3), 0.02, 1)
    pg.ink[y0:y1, x0:x1] = tex
    pg.mh[y0:y1, x0:x1] = 0; pg.mv[y0:y1, x0:x1] = 0
    pg.edge_rect(x0, y0, x1, y1)
    if rng.random() < 0.25:
        col = pg.colour('rule'); t = rng.choice([1, 1, 2]); d = rng.choice([0, 0, 3, 6])
        pg.hline(y0 - d, x0 - d, x1 + d, t, col); pg.hline(y1 + d, x0 - d, x1 + d, t, col)
        pg.vline(x0 - d, y0 - d, y1 + d, t, col); pg.vline(x1 + d, y0 - d, y1 + d, t, col)


def dark_bar(pg, x0, y0, x1, y1):
    """a dark title bar with its text knocked out in paper colour"""
    rng = pg.rng
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(pg.w, x1)), int(min(pg.h, y1))
    if x1 - x0 < 30 or y1 - y0 < 12:
        return
    v = rng.uniform(0.05, 0.5)
    col = (v, v, v) if rng.random() < 0.6 else tuple(float(q) for q in cv2.cvtColor(np.uint8([[[rng.randrange(180), rng.randint(150, 255), rng.randint(60, 200)]]]), cv2.COLOR_HSV2BGR)[0, 0] / 255.0)
    pg.ink[y0:y1, x0:x1] = np.float32(col)
    im = Image.new('L', (x1 - x0, y1 - y0), 0)
    f = font(rng.choice(FONTS[1::2]), max(10, int((y1 - y0) * 0.6)))
    ImageDraw.Draw(im).text((rng.uniform(4, 30), (y1 - y0) / 2), lat_phrase(rng, (x1 - x0) * 0.8, f), font=f, fill=255, anchor='lm')
    a = np.asarray(im).astype(np.float32)[..., None] / 255
    pg.ink[y0:y1, x0:x1] = pg.ink[y0:y1, x0:x1] * (1 - a) + a
    pg.mh[y0:y1, x0:x1] = 0; pg.mv[y0:y1, x0:x1] = 0
    pg.edge_rect(x0, y0, x1, y1)


def headings(pg):
    rng = pg.rng
    h, w = pg.h, pg.w
    for _ in range(rng.randint(1, 3)):
        sz = rng.randint(40, 170)
        if rng.random() < 0.6:
            f = font(rng.choice(HEAVY_LAT), sz); s = lat_phrase(rng, w * 0.9, f, caps=rng.random() < 0.7)
        else:
            f = font(rng.choice(HEAVY_HEB), sz); s = synth.phrase(rng, w * 0.9, f)
        col = pg.colour() if rng.random() < 0.6 else synth.pastel(rng) if rng.random() < 0.3 else \
            tuple(float(q) for q in cv2.cvtColor(np.uint8([[[rng.randrange(180), rng.randint(120, 255), rng.randint(60, 230)]]]), cv2.COLOR_HSV2BGR)[0, 0] / 255.0)
        if rng.random() < 0.25:   # sideways
            im = Image.new('L', (h, w), 0)
            ImageDraw.Draw(im).text((rng.uniform(0, h * 0.5), rng.uniform(0, w)), s, font=f, fill=255, anchor='lm')
            pg.paint(np.ascontiguousarray(np.rot90(np.asarray(im), rng.choice([1, 3]))), col)
        else:
            pg.text(rng.uniform(-0.1, 0.6) * w, rng.uniform(0, h), s, f, col, 'lm')


def barcodes(pg):
    rng = pg.rng
    h, w = pg.h, pg.w
    for _ in range(rng.randint(1, 3)):
        m = np.zeros((h, w), np.uint8)
        x, cy = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.9)
        bh = rng.uniform(25, 220); unit = rng.choice([1, 1, 2, 2, 3])
        x_end = x + rng.uniform(120, 600)
        while x < x_end:
            bw = unit * rng.choice([1, 1, 2, 3, 4]); cv2.rectangle(m, (int(x), int(cy)), (int(x + bw - 1), int(cy + bh)), 255, -1)
            x += bw + unit * rng.choice([1, 1, 2, 3, 4])
        if rng.random() < 0.2:
            m = cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE); m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        pg.paint(m, pg.colour() if rng.random() < 0.8 else (0.1, 0.1, 0.1))
    if rng.random() < 0.3:   # 2-D code
        n = rng.choice([21, 25, 29, 33]); cell = rng.randint(3, 8)
        g = np.random.default_rng(rng.randrange(1 << 30))
        q = (g.random((n, n)) < 0.5).astype(np.uint8) * 255
        for (a, b) in ((0, 0), (0, n - 7), (n - 7, 0)):
            q[a:a + 7, b:b + 7] = 255; q[a + 1:a + 6, b + 1:b + 6] = 0; q[a + 2:a + 5, b + 2:b + 5] = 255
        q = cv2.resize(q, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST)
        x0, y0 = int(rng.uniform(0, w - n * cell)), int(rng.uniform(0, h - n * cell))
        m = np.zeros((h, w), np.uint8); m[y0:y0 + n * cell, x0:x0 + n * cell] = q
        pg.paint(m, (0.1, 0.1, 0.1))


def handwriting(pg):
    rng = pg.rng
    h, w = pg.h, pg.w
    ink = rng.choice([(0.55, 0.25, 0.1), (0.6, 0.3, 0.15), (0.15, 0.15, 0.15), (0.2, 0.2, 0.7), (0.5, 0.2, 0.1)])
    for _ in range(rng.randint(1, 5)):
        f = font(rng.choice(HAND), rng.randint(18, 60))
        pg.text(rng.uniform(0, w * 0.7), rng.uniform(0, h), lat_phrase(rng, w * 0.6, f), f, ink, 'ls')
    m = np.zeros((h, w), np.uint8)
    t = rng.choice([1, 2, 2, 3, 4])
    for _ in range(rng.randint(1, 5)):
        k = rng.random()
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        if k < 0.5:   # a hand-drawn near-straight line (underline, strike, arrow shaft): wobbles, ends hook
            L = rng.uniform(80, 700); hz = rng.random() < 0.7
            n = int(L / 4)
            tt = np.linspace(0, L, n)
            wob = np.cumsum(np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.15, 0.6), n))
            wob = wob - np.linspace(wob[0], wob[-1], n) * rng.uniform(0, 1) + tt * rng.uniform(-0.04, 0.04) \
                + rng.uniform(1, 6) * np.sin(tt / L * math.pi * rng.uniform(0.5, 2))
            pts = np.stack([cx + tt, cy + wob], 1) if hz else np.stack([cx + wob, cy + tt], 1)
            cv2.polylines(m, [pts.astype(np.int32)], False, 255, t, cv2.LINE_AA)
        elif k < 0.75:   # circle around something / check mark
            if rng.random() < 0.5:
                cv2.ellipse(m, (int(cx), int(cy)), (int(rng.uniform(30, 250)), int(rng.uniform(15, 80))), rng.uniform(-10, 10), rng.uniform(0, 60), rng.uniform(300, 400), 255, t, cv2.LINE_AA)
            else:
                s = rng.uniform(10, 40)
                cv2.polylines(m, [np.int32([[cx, cy], [cx + s * 0.4, cy + s * 0.5], [cx + s * 1.2, cy - s * 0.6]])], False, 255, t, cv2.LINE_AA)
        else:   # signature
            n = rng.randint(8, 25)
            xs = cx + np.cumsum(np.random.default_rng(rng.randrange(1 << 30)).uniform(2, 25, n))
            ys = cy + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 18, n)
            tt = np.linspace(0, 1, 12 * n)
            xs = np.interp(tt, np.linspace(0, 1, n), xs); ys = np.interp(tt, np.linspace(0, 1, n), ys)
            cv2.polylines(m, [np.stack([xs, cv2.GaussianBlur(ys[None], (1, 0), 3).ravel()], 1).astype(np.int32)], False, 255, t, cv2.LINE_AA)
    pg.paint(m, ink)


def creases(pg):
    """folds and crumples: a thin dark line with a light one beside it (the lit side), any angle,
    often near an axis; low contrast"""
    rng = pg.rng
    h, w = pg.h, pg.w
    dark = np.zeros((h, w), np.float32); light = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(1, 8)):
        if rng.random() < 0.5:
            ang = rng.choice([0, 90]) + rng.uniform(-6, 6)
        else:
            ang = rng.uniform(0, 180)
        a = math.radians(ang)
        cx, cy, L = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(100, 1.5 * max(h, w))
        n = 30
        tt = np.linspace(-L / 2, L / 2, n)
        bend = np.cumsum(np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 1.5), n))
        xs = cx + tt * math.cos(a) - bend * math.sin(a); ys = cy + tt * math.sin(a) + bend * math.cos(a)
        p = np.stack([xs, ys], 1)
        o = np.float32([-math.sin(a), math.cos(a)]) * rng.uniform(1, 3)
        cv2.polylines(dark, [p.astype(np.int32)], False, rng.uniform(0.04, 0.16), rng.choice([1, 2]), cv2.LINE_AA)
        cv2.polylines(light, [(p + o).astype(np.int32)], False, rng.uniform(0.0, 0.06), rng.choice([2, 3]), cv2.LINE_AA)
    s = rng.uniform(0.5, 1.5)
    dark = cv2.GaussianBlur(dark, (0, 0), s); light = cv2.GaussianBlur(light, (0, 0), s)
    if rng.random() < 0.5:   # crumpled facets: shading steps between creases
        c = rng.randint(4, 12)
        fac = cv2.resize(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.03, (c, c)).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        fac = cv2.GaussianBlur(fac, (0, 0), rng.uniform(2, 8))
        dark += np.maximum(0, -fac); light += np.maximum(0, fac)
    pg.ink *= np.clip(1 - dark + light, 0.6, 1.15)[..., None]


def show_through(rng, h, w):
    """the back side (another page, mirrored), as a multiplicative factor <= 1"""
    back = synth.layout(rng, h, w).ink
    back = back[:, ::-1] if rng.random() < 0.7 else back[::-1]
    back = cv2.GaussianBlur(back.mean(2), (0, 0), rng.uniform(0.8, 3.5))
    a = rng.uniform(0.02, 0.1)   # darkness of a black rule on the back: ~ 5 - 25
    return 1 - (1 - back) * a


def sheet_behind(rng, img, mh, mv):
    """beyond one page side: another sheet (its own rules and text, never labelled), a shadow at the edge"""
    h, w = img.shape[:2]
    side = rng.randrange(4)
    d = int(rng.uniform(0.04, 0.3) * (w if side % 2 else h))
    other = synth.layout(rng, h, w).ink
    paper = np.float32([rng.uniform(0.75, 1), rng.uniform(0.75, 1), rng.uniform(0.75, 1)])
    other = other * paper * rng.uniform(0.6, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tilt = rng.uniform(-0.05, 0.05)
    dist = {0: yy - d - tilt * xx, 1: (w - d) - xx - tilt * yy, 2: (h - d) - yy - tilt * xx, 3: xx - d - tilt * yy}[side]
    # dist < 0: beyond the edge
    beyond = dist < 0
    shadow = 1 - rng.uniform(0.1, 0.5) * np.exp(-np.maximum(0, -dist) / rng.uniform(3, 25))
    img[beyond] = (other * shadow[..., None])[beyond]
    mh[beyond] = 0; mv[beyond] = 0


def paper_texture(rng, img):
    h, w = img.shape[:2]
    g = np.random.default_rng(rng.randrange(1 << 30))
    f = g.normal(0, 1, (h, w)).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.6, 2.0)) * rng.uniform(0.005, 0.035)
    lo = cv2.resize(g.normal(0, 1, (12, 12)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(0, 0.03)
    t = 1 + f + lo
    if rng.random() < 0.2:   # stains
        for _ in range(rng.randint(1, 4)):
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), (int(rng.uniform(5, 80)), int(rng.uniform(5, 60))), rng.uniform(0, 180), 0, 360, 1, -1)
            t *= 1 - cv2.GaussianBlur(m, (0, 0), rng.uniform(2, 15)) * rng.uniform(0.05, 0.35)
    if rng.random() < 0.3:   # specks
        n = rng.randint(5, 200)
        m = np.zeros((h, w), np.float32)
        for _ in range(n):
            cv2.circle(m, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), rng.choice([1, 1, 2, 3]), rng.uniform(0.2, 0.8), -1)
        t *= 1 - cv2.GaussianBlur(m, (0, 0), 0.7)
    return img * t[..., None]


def camera(rng, img):
    h, w = img.shape[:2]
    g = np.random.default_rng(rng.randrange(1 << 30))
    if rng.random() < 0.15:   # motion blur
        L = rng.randint(3, 9); k = np.zeros((L, L), np.float32); k[L // 2] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1), (L, L))
        img = cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))
    elif rng.random() < 0.12:   # defocus disk
        r = rng.uniform(1.5, 3.5); n = int(2 * math.ceil(r) + 1)
        k = np.zeros((n, n), np.float32); cv2.circle(k, (n // 2, n // 2), int(round(r)), 1, -1, cv2.LINE_AA)
        img = cv2.filter2D(img, -1, k / k.sum())
    if rng.random() < 0.3:   # colour cast
        img = img * np.float32([rng.uniform(0.85, 1.1), rng.uniform(0.9, 1.05), rng.uniform(0.85, 1.15)])
    if rng.random() < 0.25:   # washed out: ink lifted toward the paper
        p = np.percentile(img, 95, axis=(0, 1)).astype(np.float32)
        img = p - (p - img) * rng.uniform(0.35, 0.8)
    if rng.random() < 0.3:
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.2:   # chroma noise
        n = cv2.GaussianBlur(g.normal(0, rng.uniform(0.01, 0.04), (h, w, 3)).astype(np.float32), (0, 0), 1.2)
        img = img + n
    return img


def layout2(rng, h, w):
    Page2.faint_p = 0.15 if rng.random() < 0.4 else 0.0
    if rng.random() < 0.25:
        pg = Page2(rng, h, w)
        text_page(pg)
        if rng.random() < 0.3:
            synth.standalone_rules(pg)
    else:
        pg = synth.layout(rng, h, w, page_cls=Page2)
    if rng.random() < 0.35: headings(pg)
    if rng.random() < 0.3: barcodes(pg)
    if rng.random() < 0.3: handwriting(pg)
    if rng.random() < 0.25: creases(pg)
    if rng.random() < 0.35:
        pg.ink *= show_through(random.Random(rng.randrange(1 << 30)), h, w)[..., None]
    return pg


def sample(rng, size=400):
    """img uint8 size x size x 3, mask float32 2 x size x size (rules), edges uint8 size x size
    (bit 1: a horizontal straight edge, bit 2: a vertical one; pictures, dark bars and blocks)"""
    H = W = 2 * size + 64
    pg = layout2(rng, H, W)
    mx, my = synth.warp_field(rng, H, W)
    img = cv2.remap(pg.ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    mh = cv2.remap(np.dstack([pg.mh, pg.eh]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mv = cv2.remap(np.dstack([pg.mv, pg.ev]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = paper_texture(rng, img)
    if rng.random() < 0.15:
        sheet_behind(random.Random(rng.randrange(1 << 30)), img, mh, mv)
    else:
        synth.surround(rng, img, mh, mv)
    img = synth.degrade(rng, img)
    img = camera(rng, img)
    img = img[32:32 + 2 * size, 32:32 + 2 * size]
    mh, mv = mh[32:32 + 2 * size, 32:32 + 2 * size], mv[32:32 + 2 * size, 32:32 + 2 * size]
    small = cv2.resize(np.clip(img, 0, 1), (size, size), interpolation=cv2.INTER_AREA)
    q = rng.randint(40, 95)
    ok, enc = cv2.imencode('.jpg', (small * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    k = np.ones((3, 1), np.uint8)
    mh = cv2.dilate(np.ascontiguousarray(mh), k); mv = cv2.dilate(np.ascontiguousarray(mv), k.T)
    rh = cv2.resize(mh, (size, size), interpolation=cv2.INTER_AREA); rv = cv2.resize(mv, (size, size), interpolation=cv2.INTER_AREA)
    m = np.stack([rh[..., 0], rv[..., 0]]).astype(np.float32) / 255
    e = ((rh[..., 1] > 60) * 1 + (rv[..., 1] > 60) * 2).astype(np.uint8)
    return small, np.clip(m * 2, 0, 1), e


if __name__ == '__main__':
    n, out = int(sys.argv[1]), sys.argv[2]
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    rows = []
    for i in range(n):
        im, m, e = sample(rng, 400)
        v = im.copy()
        v[m[0] > 0.5] = (0, 0, 255); v[m[1] > 0.5] = (255, 0, 0); v[e > 0] = (0, 200, 0)
        rows.append(np.hstack([im, v]))
    cv2.imwrite(out, np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
