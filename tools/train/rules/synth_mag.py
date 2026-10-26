"""Magazine and advert pages for the rule + text-line net: the content it got wrong on real photos.

  - line-art illustrations (penitentes, mountains, cartoons): long, nearly straight dark outlines that
    are NOT rules. The net marked a spike's outline as a vertical rule, and the line stage made it
    vertical, waving the page beside it.                                                   -> no label
  - curved, arched and slanted display titles ("FAST, POWERFUL RELIEF" printed on an arc): text that
    is meant to be curved. The net marked it as a text row, and the line stage straightened it.
                                                                                           -> no label
  - big display type (60 - 170 px at stage-1 scale): the stems were marked as vertical rules.
                                                                     -> a text row, never a rule
  - what stays labelled: an illustration's frame (a picture edge), thin frame rules, body text rows,
    tables and forms (synth_text's own blocks), knocked-out title bars.

  mag_layout(rng, h, w) -> (TPage, script)   for synth_text.sample(rng, size, layout=mag_layout)
  python synth_mag.py CORPUS.json N OUT.jpg [SEED]   contact sheet (labels drawn: red / blue rules, green / yellow text)"""
import math, os, random, sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'text_net'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rule_net'))
import synth, synth2, synth_text  # noqa: E402
from synth_text import TPage, heading, line_words, paragraph, pick_lang, qfont, qt, render, words_of  # noqa: E402


def hsv(rng, s=(0.3, 0.9), v=(0.3, 1.0)):
    c = cv2.cvtColor(np.uint8([[[rng.randrange(180), int(rng.uniform(*s) * 255), int(rng.uniform(*v) * 255)]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(float(q) / 255 for q in c)


def line_art(pg, x0, y0, x1, y1):
    """a drawn illustration: sky gradient, a ridge, tall jagged spikes / blobs with dark outlines and
    inner crease strokes, a ground band. Its outer sides are a picture edge; nothing inside is labelled."""
    rng = pg.rng
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(pg.w, x1)), int(min(pg.h, y1))
    w, h = x1 - x0, y1 - y0
    if w < 80 or h < 80:
        return
    cv = np.ones((h, w, 3), np.float32)
    top, bot = np.float32(hsv(rng, (0.2, 0.7), (0.6, 1))), np.float32(hsv(rng, (0, 0.3), (0.8, 1)))
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] ** rng.uniform(0.6, 1.6)
    cv[:] = top * (1 - t) + bot * t
    ink = rng.uniform(0.02, 0.2)
    lw = max(2, int(round(w * rng.uniform(0.003, 0.008))))
    aa = cv2.LINE_AA
    # a ridge across the picture
    if rng.random() < 0.7:
        n = rng.randint(4, 9)
        xs = np.linspace(-0.05 * w, 1.05 * w, n)
        ys = h * rng.uniform(0.2, 0.55) + np.array([rng.uniform(-0.15, 0.15) * h for _ in range(n)])
        poly = np.array(list(zip(xs, ys)) + [(1.05 * w, h), (-0.05 * w, h)], np.int32)
        cv2.fillPoly(cv, [poly], hsv(rng, (0.1, 0.5), (0.25, 0.6)), aa)
        cv2.polylines(cv, [poly[:n]], False, (ink,) * 3, lw, aa)
    # tall jagged shapes: long, nearly straight outlines (the confusing part)
    for _ in range(rng.randint(3, 14)):
        cx, base = rng.uniform(-0.05, 1.05) * w, h * rng.uniform(0.85, 1.1)
        top_y, half = h * rng.uniform(0.05, 0.6), w * rng.uniform(0.03, 0.12)
        k = rng.randint(3, 7)
        left = [(cx - half * (1 - i / k) + rng.uniform(-0.15, 0.15) * half, base - (base - top_y) * i / k) for i in range(k)]
        right = [(cx + half * (1 - i / k) + rng.uniform(-0.15, 0.15) * half, base - (base - top_y) * i / k) for i in range(k)][::-1]
        tip = (cx + rng.uniform(-0.2, 0.2) * half, top_y)
        poly = np.array(left + [tip] + right, np.int32)
        v = rng.uniform(0.75, 0.97)
        cv2.fillPoly(cv, [poly], (v * rng.uniform(0.95, 1), v, v * rng.uniform(0.95, 1.05)), aa)
        cv2.polylines(cv, [poly], True, (ink,) * 3, lw, aa)
        for _ in range(rng.randint(0, 5)):   # crease strokes inside
            yy = rng.uniform(top_y, base)
            cv2.line(cv, (int(cx - half * 0.8), int(yy)), (int(cx + half * rng.uniform(-0.3, 0.8)), int(yy + rng.uniform(-0.05, 0.05) * h)), (ink,) * 3, max(1, lw - 1), aa)
    if rng.random() < 0.6:   # ground with stones
        gy = int(h * rng.uniform(0.82, 0.95))
        cv[gy:] = np.float32(hsv(rng, (0.3, 0.7), (0.25, 0.5)))
        for _ in range(rng.randint(3, 15)):
            c = (int(rng.uniform(0, w)), int(rng.uniform(gy, h)))
            cv2.ellipse(cv, c, (int(w * rng.uniform(0.01, 0.04)), int(h * rng.uniform(0.005, 0.02))), rng.uniform(0, 180), 0, 360, (ink,) * 3, max(1, lw - 1), aa)
    if rng.random() < 0.5:   # printed grain
        cv += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.005, 0.03), cv.shape).astype(np.float32)
    pg.ink[y0:y1, x0:x1] = np.clip(cv, 0, 1)
    for m in (pg.mh, pg.mv, pg.th, pg.tv):
        m[y0:y1, x0:x1] = 0
    pg.edge_rect(x0, y0, x1, y1)
    if rng.random() < 0.3:
        col = pg.colour('rule'); t = rng.choice([1, 2]); d = rng.choice([0, 4, 8])
        pg.hline(y0 - d, x0 - d, x1 + d, t, col); pg.hline(y1 + d, x0 - d, x1 + d, t, col)
        pg.vline(x0 - d, y0 - d, y1 + d, t, col); pg.vline(x1 + d, y0 - d, y1 + d, t, col)


def bend(a, rng):
    """an alpha bent on an arc / wave / slope and turned a little: printed that way"""
    h, w = a.shape
    pad = int(0.6 * w * 0.25) + h
    a = cv2.copyMakeBorder(a, pad, pad, 8, 8, cv2.BORDER_CONSTANT, value=0)
    H, W = a.shape
    xs = np.arange(W, dtype=np.float32)
    u = (xs - 8) / max(1, w)
    kind = rng.choice(['arc', 'arc', 'wave', 'slope'])
    amp = w * rng.uniform(0.04, 0.18) * rng.choice([-1, 1])
    if kind == 'arc':
        dy = amp * (1 - (2 * u - 1) ** 2)
    elif kind == 'wave':
        dy = amp * 0.5 * np.sin(2 * math.pi * u * rng.uniform(0.6, 1.4) + rng.uniform(0, 6))
    else:
        dy = w * rng.uniform(0.06, 0.35) * rng.choice([-1, 1]) * u
    mx = np.repeat(xs[None], H, 0)
    my = np.arange(H, dtype=np.float32)[:, None] + dy[None].astype(np.float32)
    out = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        R = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-12, 12), 1)
        out = cv2.warpAffine(out, R, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs2 = np.nonzero(out > 10)
    if len(ys) == 0:
        return out
    return out[ys.min():ys.max() + 1, xs2.min():xs2.max() + 1]


def display_title(pg, lang, x0, y0, x1, px, curved):
    """a heavy title, curved (unlabelled) or straight (a text row)"""
    rng = pg.rng
    q = qfont(rng, lang, px, heavy=True)
    parts, _ = line_words(rng, lang, q, (x1 - x0) * rng.uniform(0.6, 1.0), words_of(rng, lang, rng.randint(1, 4)))
    if not parts:
        return y0
    if lang['script'] in synth_text.NO_SPACE:
        parts = [''.join(parts)]
    a, lab = render(parts, q, lang['rtl'])
    if a is None:
        return y0
    col = hsv(rng, (0.6, 1), (0.7, 1)) if rng.random() < 0.5 else pg.colour()
    if curved:
        b = bend(a, rng)
        pg.put(b, np.zeros_like(b), x0 + rng.uniform(0, 0.15) * (x1 - x0), y0, col)
        return y0 + b.shape[0]
    pg.put(a, lab, x0, y0, col)
    return y0 + a.shape[0]


def slanted_line(pg, lang, x, y, px):
    """a short text line printed at an angle (a sticker, a stamp-like tag): not a text row"""
    rng = pg.rng
    q = qfont(rng, lang, px)
    parts, _ = line_words(rng, lang, q, pg.w * 0.4, words_of(rng, lang, rng.randint(1, 4)))
    if not parts:
        return
    a, _ = render(parts, q, lang['rtl'])
    if a is None:
        return
    h, w = a.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(6, 30) * rng.choice([-1, 1]), 1)
    c, s = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    R[0, 2] += nw / 2 - w / 2; R[1, 2] += nh / 2 - h / 2
    b = cv2.warpAffine(a, R, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    pg.put(b, np.zeros_like(b), x, y, pg.colour())


def mag_layout(rng, h, w):
    pg = TPage(rng, h, w)
    TPage.faint_p = 0.0
    lang = pick_lang(rng, 'LATIN') if rng.random() < 0.6 else pick_lang(rng)
    body = rng.choice([12, 14, 16, 18, 20, 24])
    if rng.random() < 0.4:   # coloured advert ground
        pg.fill(0, 0, w, h * rng.uniform(0.3, 1), hsv(rng, (0.2, 0.9), (0.5, 1)))
    y = rng.uniform(0.02, 0.1) * h
    while y < h:
        x0, x1 = rng.uniform(0.02, 0.12) * w, w - rng.uniform(0.02, 0.12) * w
        r = rng.random()
        if r < 0.3:   # illustration, often with a title on it
            bh = rng.uniform(0.25, 0.7) * h
            line_art(pg, x0, y, x1, y + bh)
            if rng.random() < 0.6:
                display_title(pg, lang, x0 + 0.05 * (x1 - x0), y + 0.05 * bh, x1, body * rng.uniform(3, 8), curved=rng.random() < 0.7)
            if rng.random() < 0.5:   # body text over the picture's sky
                paragraph(pg, lang, x0 + 0.05 * (x1 - x0), y + 0.3 * bh, x0 + rng.uniform(0.4, 0.6) * (x1 - x0), y + 0.6 * bh, body, pg.colour())
            y += bh
        elif r < 0.5:
            y = display_title(pg, lang, x0, y, x1, body * rng.uniform(3, 9), curved=rng.random() < 0.6)
        elif r < 0.75:
            bh = rng.uniform(0.1, 0.4) * h
            if rng.random() < 0.4:
                mid = (x0 + x1) / 2
                paragraph(pg, lang, x0, y, mid - body, y + bh, body, pg.colour())
                y = paragraph(pg, lang, mid + body, y, x1, y + bh, body, pg.colour())
            else:
                y = paragraph(pg, lang, x0, y, x1, y + bh, body, pg.colour())
        elif r < 0.85:
            bh = rng.uniform(0.1, 0.3) * h
            synth2.picture(pg, x0 + rng.uniform(0, 0.4) * w, y, x1 - rng.uniform(0, 0.3) * w, y + bh)
            y += bh
        elif r < 0.92:
            y = heading(pg, lang, x0, y, x1, body * rng.uniform(1.5, 3))
        else:
            bh = body * rng.uniform(1.8, 3)
            synth2.dark_bar(pg, x0, y, x1, y + bh)
            y += bh
        y += body * rng.uniform(0.5, 3)
    for _ in range(rng.choice([0, 0, 1, 2])):
        slanted_line(pg, lang, rng.uniform(0, 0.7) * w, rng.uniform(0, 0.9) * h, body * rng.uniform(1, 2.5))
    if rng.random() < 0.15: synth2.barcodes(pg)
    if rng.random() < 0.2: synth2.creases(pg)
    return pg, lang['script']


if __name__ == '__main__':
    os.environ['TN_CORPUS'] = sys.argv[1]
    synth_text.CORPUS = sys.argv[1]
    n, out = int(sys.argv[2]), sys.argv[3]
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    import time
    t0, rows = time.time(), []
    for i in range(n):
        im, m, e, s = synth_text.sample(rng, 400, layout=mag_layout)
        v = im.copy()
        v[m[2] > 0.5] = (0, 200, 0); v[m[3] > 0.5] = (0, 220, 220)
        v[m[0] > 0.5] = (0, 0, 255); v[m[1] > 0.5] = (255, 0, 0)
        rows.append(np.hstack([im, v]))
    print(f'{(time.time() - t0) / n:.2f} s per sample', flush=True)
    cv2.imwrite(out, np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
