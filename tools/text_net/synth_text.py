"""Synthetic training pages for the text-line net (rules + text lines, every script): a stage-1 page as the rule
net's generator draws it (tools/rule_net/synth.py, synth2.py: residual warp, paper, camera, show-through,
pictures, barcodes, creases), with text in ~380 languages / 28 scripts (corpus.py: the UDHR, real words) plus
pseudo-text in scripts the corpus lacks, shaped by Qt (HarfBuzz: Arabic joining, Indic clusters, bidi).

Labels (4 channels, the net's outputs): horizontal rule, vertical rule (as synth2: printed straight rules, picture
edges in the edge map), horizontal text line, vertical text line. A text line's label is a band on its centre
line, whatever the script: the rows where the line's ink is densest (>= 35 % of the densest row: the x-height
of Latin, the body of Hebrew and Arabic, the full height of CJK, below the headline of Devanagari), the middle
40 % of them, from the line's first to its last ink (word gaps included; columns apart). Vertical lines: CJK
columns, Latin turned 90 deg. Show-through, handwriting-free.

  sample(rng, size=400) -> (img uint8 HxWx3 BGR, mask float32 4xHxW, edge uint8 HxW, script str)
  python synth_text.py CORPUS.json N OUT.jpg [SEED]     contact sheet (red / blue: rules, green / yellow: text lines)
"""
import math, os, random, sys, unicodedata, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rule_net'))
import synth, synth2   # noqa: E402

FONT_DIR = 'C:/Windows/Fonts/'
CORPUS = os.environ.get('TN_CORPUS', 'E:/fs_datasets/textnet/corpus.json')
# relative chance of a page's script (the rest 1 each): the owner's documents first, then the most used scripts
SCRIPT_W = {'LATIN': 6, 'HEBREW': 4, 'ARABIC': 2.5, 'CYRILLIC': 2, 'HAN': 2, 'JAPANESE': 1.5, 'HANGUL': 1.2, 'DEVANAGARI': 1.5,
            'GREEK': 1, 'BENGALI': 1, 'THAI': 1, 'TAMIL': 1}
NO_SPACE = {'HAN', 'JAPANESE', 'THAI', 'LAO', 'KHMER', 'MYANMAR', 'TIBETAN', 'YI'}
VERTICAL = {'HAN', 'JAPANESE', 'HANGUL'}
# scripts with fonts here but no UDHR text: pseudo-words from their letters (first code point, last, font file)
PSEUDO = {'TELUGU': (0x0C05, 0x0C39, 'Nirmala.ttc'), 'ORIYA': (0x0B05, 0x0B39, 'Nirmala.ttc'), 'SINHALA': (0x0D85, 0x0DC6, 'Nirmala.ttc'),
          'CHEROKEE': (0x13A0, 0x13F4, 'gadugi.ttf'), 'TIFINAGH': (0x2D30, 0x2D65, 'ebrima.ttf'), 'NKO': (0x07CA, 0x07EA, 'ebrima.ttf'),
          'OL_CHIKI': (0x1C5A, 0x1C77, 'Nirmala.ttc')}

_qt = {}


def qt():
    """the Qt objects of this process (QGuiApplication on the offscreen platform)"""
    if not _qt:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6 import QtGui, QtCore
        app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(['synth_text'])
        _qt.update(app=app, G=QtGui, C=QtCore, fams={})
    return _qt


def family(file):
    """the Qt family names of a font file (registered once per process)"""
    q = qt()
    if file not in q['fams']:
        fid = q['G'].QFontDatabase.addApplicationFont(FONT_DIR + file)
        q['fams'][file] = list(q['G'].QFontDatabase.applicationFontFamilies(fid)) if fid >= 0 else []
    return q['fams'][file]


_corpus = None


def corpus():
    global _corpus
    if _corpus is None:
        c = json.load(open(CORPUS, encoding='utf-8'))
        by = {}
        for l in c['langs']:
            by.setdefault(l['script'], []).append(l)
        for s, (a, b, f) in PSEUDO.items():
            if os.path.exists(FONT_DIR + f):
                by[s] = [{'code': s.lower(), 'script': s, 'rtl': s == 'NKO', 'paras': None, 'fonts': [[f, 0, None, False, False]], 'range': [a, b]}]
        _corpus = by
    return _corpus


def pick_lang(rng, script=None):
    by = corpus()
    if script is None:
        ss = sorted(by)
        script = rng.choices(ss, weights=[SCRIPT_W.get(s, 0.4 if s in PSEUDO else 1.0) for s in ss])[0]
    return rng.choice(by[script])


def words_of(rng, lang, n):
    """about n words (or, for scripts without spaces, n chunks) of the language's text"""
    if lang.get('range'):
        a, b = lang['range']
        return [''.join(chr(rng.randint(a, b)) for _ in range(rng.randint(2, 8))) for _ in range(n)]
    p = rng.choice(lang['paras'])
    if lang['script'] in NO_SPACE:
        # chunks of 1 - 4 characters, never starting with a combining mark
        out, i = [], rng.randrange(max(1, len(p) // 2))
        while len(out) < n:
            if i >= len(p):
                p, i = rng.choice(lang['paras']), 0
            j = min(len(p), i + rng.randint(1, 4))
            while j < len(p) and unicodedata.category(p[j])[0] == 'M':
                j += 1
            out.append(p[i:j].replace(' ', ''))
            i = j
        return [w for w in out if w]
    ws = p.split()
    k = rng.randrange(max(1, len(ws) - 1))
    out = ws[k:]
    while len(out) < n:
        out += rng.choice(lang['paras']).split()
    return out[:n]


def qfont(rng, lang, px, heavy=False):
    G = qt()['G']
    f, k, fam, bold, ital = rng.choice(lang['fonts'])
    fams = family(f)
    name = fam if fam in fams else (fams[0] if fams else 'Arial')
    q = G.QFont(name)
    q.setPixelSize(max(6, int(px)))
    if bold or heavy or rng.random() < 0.12:
        q.setBold(True)
    if ital:
        q.setItalic(True)
    q.setHintingPreference(G.QFont.PreferNoHinting)
    return q


def render(parts, q, rtl, gaps=None):
    """one line of words (drawn one by one when gaps are given: justified) -> (alpha uint8 H x W, label uint8)"""
    Q = qt(); G, C = Q['G'], Q['C']
    fm = G.QFontMetricsF(q)
    asc, desc = fm.ascent(), fm.descent()
    if gaps is None:
        text = ' '.join(parts) if len(parts) > 1 else parts[0]
        wd = fm.horizontalAdvance(text)
    else:
        ws = [fm.horizontalAdvance(p) for p in parts]
        wd = sum(ws) + sum(gaps)
    pad = int(0.3 * (asc + desc)) + 2
    W, H = int(math.ceil(wd)) + 2 * pad, int(math.ceil(asc + desc)) + 2 * pad
    if W < 4 or W > 6000 or H > 800:
        return None, None
    img = G.QImage(W, H, G.QImage.Format_Grayscale8)
    img.fill(0)
    p = G.QPainter(img)
    p.setRenderHint(G.QPainter.TextAntialiasing, True)
    p.setFont(q)
    p.setPen(G.QColor(255, 255, 255))
    if rtl:
        p.setLayoutDirection(C.Qt.RightToLeft)
    base = pad + asc
    if gaps is None:
        p.drawText(C.QPointF(pad, base), text)
    else:
        x = W - pad if rtl else pad
        for k, (s, w) in enumerate(zip(parts, ws)):
            if rtl:
                x -= w
                p.drawText(C.QPointF(x, base), s)
                x -= gaps[k] if k < len(gaps) else 0
            else:
                p.drawText(C.QPointF(x, base), s)
                x += w + (gaps[k] if k < len(gaps) else 0)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8, img.sizeInBytes()).reshape(H, img.bytesPerLine())[:, :W].copy()
    return a, band(a)


def band(a, horizontal=True):
    """the centre band label of one line's alpha (horizontal: rows are across the line)"""
    m = a > 80 if horizontal else (a > 80).T
    cols = np.nonzero(m.any(0))[0]
    lab = np.zeros(m.shape, np.uint8)
    if len(cols) < 2:
        return lab if horizontal else lab.T
    x0, x1 = cols[0], cols[-1]
    prof = m[:, x0:x1 + 1].mean(1)
    rows = np.nonzero(prof >= 0.35 * prof.max())[0]
    r0, r1 = rows[0], rows[-1]
    c, half = (r0 + r1) / 2, max(1.0, 0.2 * (r1 - r0 + 1))
    lab[int(round(c - half)):int(round(c + half)) + 1, x0:x1 + 1] = 255
    return lab if horizontal else lab.T


class TPage(synth2.Page2):
    faint_p = 0.0

    def __init__(self, rng, h, w):
        super().__init__(rng, h, w)
        self.th = np.zeros((h, w), np.uint8)
        self.tv = np.zeros((h, w), np.uint8)

    def put(self, a, lab, x, y, col, vertical=False, knock=False):
        """paste a line's alpha at (x, y) (top left), its label into th / tv"""
        h, w = a.shape
        x, y = int(round(x)), int(round(y))
        X0, Y0, X1, Y1 = max(0, x), max(0, y), min(self.w, x + w), min(self.h, y + h)
        if X1 <= X0 or Y1 <= Y0:
            return
        aa = a[Y0 - y:Y1 - y, X0 - x:X1 - x].astype(np.float32)[..., None] / 255
        reg = self.ink[Y0:Y1, X0:X1]
        if knock:   # light text on a dark ground
            reg[:] = reg * (1 - aa) + aa * np.float32(col)
        else:
            reg *= 1 - aa * (1 - np.float32(col))
        t = self.tv if vertical else self.th
        np.maximum(t[Y0:Y1, X0:X1], lab[Y0 - y:Y1 - y, X0 - x:X1 - x], out=t[Y0:Y1, X0:X1])


def line_words(rng, lang, q, maxw, words):
    """greedy line fill: (parts, natural width) consuming words"""
    fm = qt()['G'].QFontMetricsF(q)
    sep = '' if lang['script'] in NO_SPACE else ' '
    parts, wd = [], 0.0
    while words:
        t = sep.join(parts + [words[0]])
        tw = fm.horizontalAdvance(t)
        if parts and tw > maxw:
            break
        parts.append(words.pop(0)); wd = tw
        if tw > maxw:
            break
    return parts, wd


def paragraph(pg, lang, x0, y0, x1, y1, px, col):
    rng = pg.rng
    q = qfont(rng, lang, px)
    lead = px * rng.uniform(1.05, 1.8)
    rtl = lang['rtl']
    nospace = lang['script'] in NO_SPACE
    justify = not nospace and rng.random() < 0.4
    align = 'r' if rtl else ('c' if rng.random() < 0.07 else 'l')
    words = words_of(rng, lang, 400)
    y, first = y0, True
    fm = qt()['G'].QFontMetricsF(q)
    while y + lead < y1 and words:
        ind = px * rng.uniform(1, 3) if first and rng.random() < 0.3 else 0
        parts, wd = line_words(rng, lang, q, x1 - x0 - ind, words)
        if not parts:
            break
        last = not words or rng.random() < 0.08
        gaps = None
        if justify and not last and len(parts) > 1:
            ws = sum(fm.horizontalAdvance(p) for p in parts)
            g = (x1 - x0 - ind - ws) / (len(parts) - 1)
            if g < 3 * fm.horizontalAdvance(' ') + 1:
                gaps = [g] * (len(parts) - 1)
        a, lab = render(parts if not nospace else [''.join(parts)], q, rtl, gaps)
        if a is not None:
            pad = int(0.3 * (fm.ascent() + fm.descent())) + 2
            w = a.shape[1] - 2 * pad
            xs = {'l': x0 + (0 if rtl else ind), 'r': x1 - w - (ind if rtl else 0), 'c': (x0 + x1 - w) / 2}[align]
            pg.put(a, lab, xs - pad, y - pad, col)
        y += lead
        first = last and rng.random() < 0.5
        if last and words:
            y += lead * rng.uniform(0, 1.2)
    return y


def table(pg, lang, x0, y0, x1, y1, px):
    rng = pg.rng
    nc = rng.randint(2, 6)
    xs = sorted([x0, x1] + [rng.uniform(x0, x1) for _ in range(nc - 1)])
    rh = px * rng.uniform(1.6, 3.2)
    ys = [y0 + k * rh for k in range(int((y1 - y0) / rh) + 1)]
    rc = pg.colour('rule'); t = rng.choice([1, 1, 2, 2, 3])
    full = rng.random() < 0.7
    for y in ys:
        pg.hline(y, xs[0], xs[-1], t, rc)
    if full:
        for x in xs:
            pg.vline(x, ys[0], ys[-1], t, rc)
    if rng.random() < 0.3:   # shaded header
        pg.fill(xs[0], ys[0], xs[-1], ys[min(1, len(ys) - 1)], (rng.uniform(0.75, 0.95),) * 3)
    q = qfont(rng, lang, px)
    col = pg.colour()
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            if rng.random() < 0.15:
                continue
            cw = xs[c + 1] - xs[c] - 2 * px * 0.3
            if cw < px:
                continue
            if rng.random() < 0.3:
                s = [str(rng.randint(0, 10 ** rng.randint(1, 6))) + rng.choice(['', '.00', ',50', '%'])]
                qq, rtl = qt()['G'].QFont(q), False
            else:
                s, _ = line_words(rng, lang, q, cw, words_of(rng, lang, rng.randint(1, 4)))
                qq, rtl = q, lang['rtl']
            if not s:
                continue
            if lang['script'] in NO_SPACE:
                s = [''.join(s)]
            a, lab = render(s, qq, rtl)
            if a is None or a.shape[1] > cw + a.shape[0]:
                continue
            fm = qt()['G'].QFontMetricsF(qq)
            pad = int(0.3 * (fm.ascent() + fm.descent())) + 2
            w = a.shape[1] - 2 * pad
            xa = xs[c + 1] - px * 0.3 - w if (rtl or rng.random() < 0.3) else xs[c] + px * 0.3
            ya = (ys[r] + ys[r + 1]) / 2 - (fm.ascent() + fm.descent()) / 2
            pg.put(a, lab, xa - pad, ya - pad, col)
    return ys[-1]


def heading(pg, lang, x0, y0, x1, px):
    rng = pg.rng
    q = qfont(rng, lang, px, heavy=rng.random() < 0.6)
    parts, _ = line_words(rng, lang, q, x1 - x0, words_of(rng, lang, rng.randint(1, 6)))
    if not parts:
        return y0
    if lang['script'] in NO_SPACE:
        parts = [''.join(parts)]
    a, lab = render(parts, q, lang['rtl'])
    if a is None:
        return y0
    knock = rng.random() < 0.2
    fm = qt()['G'].QFontMetricsF(q)
    pad = int(0.3 * (fm.ascent() + fm.descent())) + 2
    if knock:
        dark = rng.uniform(0.05, 0.45)
        pg.fill(x0 - px * 0.3, y0 - px * 0.2, x1 + px * 0.3, y0 + px * 1.5, (dark * rng.uniform(0.6, 1), dark * rng.uniform(0.6, 1), dark))
    w = a.shape[1] - 2 * pad
    xs = x1 - w if lang['rtl'] else (x0 if rng.random() < 0.7 else (x0 + x1 - w) / 2)
    pg.put(a, lab, xs - pad, y0 - pad, (rng.uniform(0.85, 1.0),) * 3 if knock else pg.colour(), knock=knock)
    return y0 + px * 1.6


def form(pg, lang, x0, y0, x1, y1, px):
    """label: ____ lines (labelled rules) beside short text"""
    rng = pg.rng
    q = qfont(rng, lang, px)
    y, rc, col = y0, pg.colour('rule'), pg.colour()
    while y + px * 2.2 < y1:
        parts, _ = line_words(rng, lang, q, (x1 - x0) * 0.4, words_of(rng, lang, rng.randint(1, 3)))
        if not parts:
            break
        if lang['script'] in NO_SPACE:
            parts = [''.join(parts)]
        a, lab = render(parts + ([':'] if rng.random() < 0.5 else []), q, lang['rtl'])
        if a is not None:
            fm = qt()['G'].QFontMetricsF(q)
            pad = int(0.3 * (fm.ascent() + fm.descent())) + 2
            w = a.shape[1] - 2 * pad
            if lang['rtl']:
                pg.put(a, lab, x1 - w - pad, y - pad, col)
                pg.hline(y + fm.ascent() + 2, x0, x1 - w - px * 0.5, rng.choice([1, 1, 2]), rc)
            else:
                pg.put(a, lab, x0 - pad, y - pad, col)
                pg.hline(y + fm.ascent() + 2, x0 + w + px * 0.5, x1, rng.choice([1, 1, 2]), rc)
        y += px * rng.uniform(2.0, 3.2)
    return y


def vertical_block(pg, lang, x0, y0, x1, y1, px):
    """CJK columns, right to left, characters stacked"""
    rng = pg.rng
    q = qfont(rng, lang, px)
    fm = qt()['G'].QFontMetricsF(q)
    col = pg.colour()
    x = x1 - px
    chars = [c for w in words_of(rng, lang, 300) for c in w if not unicodedata.category(c).startswith(('Z', 'P', 'M'))]
    while x > x0 and chars:
        n = int((y1 - y0) / (px * 1.05))
        if n < 2:
            break
        col_a = np.zeros((int(y1 - y0) + 2 * px, px * 2), np.uint8)
        k = 0
        for k in range(min(n, len(chars))):
            a, _ = render([chars[k]], q, False)
            if a is None:
                continue
            pad = int(0.3 * (fm.ascent() + fm.descent())) + 2
            ys, xs = int(k * px * 1.05), int((px * 2 - a.shape[1]) / 2)
            hh, ww = min(a.shape[0], col_a.shape[0] - ys), min(a.shape[1], col_a.shape[1] - max(0, xs))
            if hh > 0 and ww > 0:
                np.maximum(col_a[ys:ys + hh, max(0, xs):max(0, xs) + ww], a[:hh, max(0, -xs):max(0, -xs) + ww], out=col_a[ys:ys + hh, max(0, xs):max(0, xs) + ww])
        del chars[:k + 1]
        pg.put(col_a, band(col_a, horizontal=False), x - px, y0, col, vertical=True)
        x -= px * rng.uniform(1.4, 2.0)


def turned_line(pg, lang, px):
    """a text line turned 90 deg (a table's side label, a spine): a vertical text line"""
    rng = pg.rng
    q = qfont(rng, lang, px)
    parts, _ = line_words(rng, lang, q, pg.h * 0.6, words_of(rng, lang, rng.randint(2, 8)))
    if not parts:
        return
    if lang['script'] in NO_SPACE:
        parts = [''.join(parts)]
    a, lab = render(parts, q, lang['rtl'])
    if a is None:
        return
    k = rng.choice([1, 3])
    pg.put(np.ascontiguousarray(np.rot90(a, k)), np.ascontiguousarray(np.rot90(lab, k)), rng.uniform(0, pg.w * 0.9), rng.uniform(0, pg.h * 0.4), pg.colour(), vertical=True)


def text_layout(rng, h, w):
    pg = TPage(rng, h, w)
    TPage.faint_p = 0.15 if rng.random() < 0.3 else 0.0
    lang = pick_lang(rng)
    other = pick_lang(rng, 'LATIN') if rng.random() < 0.3 else lang
    script = lang['script']
    body = rng.choice([12, 14, 16, 18, 20, 22, 26, 30, 36])   # stage-1 px (1600 px wide page)
    y = rng.uniform(-0.15, 0.05) * h
    while y < h:
        mx = rng.uniform(-0.1, 0.12) * w
        x0, x1 = mx, w - rng.uniform(-0.1, 0.12) * w
        L = lang if rng.random() < 0.8 else other
        px = body * rng.uniform(0.85, 1.15)
        r = rng.random()
        if r < 0.45:
            bh = rng.uniform(0.15, 0.8) * h
            if rng.random() < 0.3:
                mid = (x0 + x1) / 2
                paragraph(pg, L, x0, y, mid - px, y + bh, px, pg.colour())
                y = paragraph(pg, L, mid + px, y, x1, y + bh, px, pg.colour())
            else:
                y = paragraph(pg, L, x0, y, x1, y + bh, px, pg.colour())
        elif r < 0.62:
            y = table(pg, L, x0, y, x1, y + rng.uniform(0.15, 0.6) * h, px)
        elif r < 0.74:
            y = heading(pg, L, x0, y, x1, px * rng.uniform(1.4, 3.5))
        elif r < 0.82:
            y = form(pg, L, x0, y, x1, y + rng.uniform(0.1, 0.4) * h, px)
        elif r < 0.9 and L['script'] in VERTICAL:
            bh = rng.uniform(0.3, 0.8) * h
            vertical_block(pg, L, x0, y, x1, y + bh, int(px))
            y += bh
        elif r < 0.95:
            bh = rng.uniform(0.1, 0.35) * h
            synth2.picture(pg, x0 + rng.uniform(0, 0.4) * w, y, x1 - rng.uniform(0, 0.3) * w, y + bh)
            y += bh
        else:
            y = paragraph(pg, L, x0 + rng.uniform(0.3, 0.6) * w, y, x1, y + rng.uniform(0.1, 0.3) * h, px * 0.8, pg.colour())
        y += px * rng.uniform(0.5, 3) if rng.random() < 0.85 else rng.uniform(40, 200)
    if rng.random() < 0.12: turned_line(pg, lang, body * rng.uniform(0.9, 1.5))
    if rng.random() < 0.2: synth.standalone_rules(pg)
    if rng.random() < 0.2: synth2.barcodes(pg)
    if rng.random() < 0.2: synth2.creases(pg)
    if rng.random() < 0.35:
        pg.ink *= synth2.show_through(random.Random(rng.randrange(1 << 30)), h, w)[..., None]
    return pg, script


def sample(rng, size=400):
    """img uint8 size x size x 3, mask float32 4 x size x size [rule h, rule v, text h, text v], edge uint8, script"""
    H = W = 2 * size + 64
    pg, script = text_layout(rng, H, W)
    mx, my = synth.warp_field(rng, H, W)
    img = cv2.remap(pg.ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    lab = np.dstack([pg.mh, pg.mv, pg.th, pg.tv, pg.eh, pg.ev])
    lab = np.dstack([cv2.remap(lab[..., k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for k in range(6)])
    img = synth2.paper_texture(rng, img)
    # [rule, picture edge, text] per direction: the frame beyond the page (sheet behind, surround) clears all three
    mh, mv = lab[..., [0, 4, 2]].copy(), lab[..., [1, 5, 3]].copy()
    if rng.random() < 0.15:
        synth2.sheet_behind(random.Random(rng.randrange(1 << 30)), img, mh, mv)
    else:
        synth.surround(rng, img, mh, mv)
    th, tv = np.ascontiguousarray(mh[..., 2]), np.ascontiguousarray(mv[..., 2])
    mh, mv = np.ascontiguousarray(mh[..., :2]), np.ascontiguousarray(mv[..., :2])
    img = synth.degrade(rng, img)
    img = synth2.camera(rng, img)
    c = slice(32, 32 + 2 * size)
    img, mh, mv, th, tv = img[c, c], mh[c, c], mv[c, c], th[c, c], tv[c, c]
    small = cv2.resize(np.clip(img, 0, 1), (size, size), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode('.jpg', (small * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
    small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    k = np.ones((3, 1), np.uint8)
    mh = cv2.dilate(np.ascontiguousarray(mh), k); mv = cv2.dilate(np.ascontiguousarray(mv), k.T)
    rh = cv2.resize(mh, (size, size), interpolation=cv2.INTER_AREA); rv = cv2.resize(mv, (size, size), interpolation=cv2.INTER_AREA)
    t2 = [cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in (th, tv)]
    m = np.stack([rh[..., 0], rv[..., 0], t2[0], t2[1]]).astype(np.float32) / 255
    e = ((rh[..., 1] > 60) * 1 + (rv[..., 1] > 60) * 2).astype(np.uint8)
    return small, np.clip(m * 2, 0, 1), e, script


if __name__ == '__main__':
    n, out = int(sys.argv[2]), sys.argv[3]
    os.environ['TN_CORPUS'] = sys.argv[1]
    CORPUS = sys.argv[1]
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    rows = []
    import time
    t0 = time.time()
    for i in range(n):
        im, m, e, s = sample(rng, 400)
        v = im.copy()
        v[m[2] > 0.5] = (0, 200, 0); v[m[3] > 0.5] = (0, 220, 220)
        v[m[0] > 0.5] = (0, 0, 255); v[m[1] > 0.5] = (255, 0, 0)
        cv2.putText(v, s, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        rows.append(np.hstack([im, v]))
    print(f'{(time.time() - t0) / n:.2f} s per sample', flush=True)
    cv2.imwrite(out, np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
