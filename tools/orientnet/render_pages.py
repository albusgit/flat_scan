"""render_pages.py OUTDIR N [--seed S] [--fonts held|train|all]: synthetic upright document pages for the orientation
net (orientnet). Every page is upright; train.py turns its crops.

A page: a random layout (columns of paragraphs, headings, a table with rules, a receipt column of monospace
lines, numbers) in one or two scripts (Hebrew, Latin, Cyrillic, Greek, CJK, Hangul, digits) and fonts found in
C:/Windows/Fonts (or --font_dir). Text is random words with each script's letter frequencies plus real sentences,
so the net learns glyph shapes, not words. Hebrew is drawn right to left (visual order; Hebrew needs no shaping).
Saved as 8-bit grey PNG, long side 1400 px (crops are taken at several scales by train.py).

Fonts are split by name into train / held (held fonts render the validation pages only), see HELD.
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTCollection, TTFont

SCRIPTS = {
    # name: (alphabet with rough frequencies, words per line range, fonts that cover it)
    'latin': ('eeeeeeeeeeeettttttttttaaaaaaaaoooooooiiiiiiinnnnnnnsssssshhhhhhrrrrrrddddllllcccuuummwwffggyyppbbvkjxqz'
              'EEETTTAAOOIINNSSHHRRDDLLCCUUMMWWFFGGYYPPBBVK', None),
    'latin_acc': ('eeeeéèêaaaàâoooôiiiîuuùûçnnnsssrrrtttlllddmmppcc' 'ÉÀÇEEAAOO' 'äöüßäöü', None),
    'cyrillic': ('оооооееееааааииииннннттттссссрррвввлллккммддппууяыьгзбчйхжшюцщэфъё' 'ОЕАИНТСРВЛКМДПУ', None),
    'greek': ('αααεεεοοοιιιννντττσσρρηηκκμμπππλλυυωωδγχφθξψζ' 'ΑΕΟΙΝΤΣΡΗΚΜΠΛ', None),
    'hebrew': ('יייייווווווההההההללללללאאאאאאממממממררררתתתתתבבבבבננננששששעעעדדדקקקכככגגגפפפסססזזטטצצחחחןןםםםךךףץ', None),
    'digits': ('0123456789' * 4 + '.,:-/%$€₪*#()' + 'ABCDEFGHKLMNPRSTXYZ', None),
}
SENTENCES = {
    'latin': ['The quick brown fox jumps over the lazy dog.', 'Please keep this receipt for your records.',
              'Total amount due including tax', 'Thank you for shopping with us!', 'Chapter 3. Results and discussion',
              'All rights reserved. No part of this publication may be reproduced.'],
    'hebrew': ['תודה שקניתם אצלנו', 'נא לשמור את החשבונית', 'סך הכל לתשלום כולל מע"מ', 'פרק שלישי: תוצאות ודיון',
               'כל הזכויות שמורות', 'שעות פתיחה: ימים א-ה 9:00-21:00'],
    'cyrillic': ['Спасибо за покупку!', 'Итого к оплате', 'Глава 3. Результаты и обсуждение'],
    'greek': ['Ευχαριστούμε για την αγορά σας', 'Σύνολο πληρωμής'],
}
# fonts per script (file names in the font dir); a font missing on this machine is skipped
FONTS = {
    'latin': ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'georgia.ttf', 'georgiab.ttf',
              'verdana.ttf', 'verdanab.ttf', 'calibri.ttf', 'calibrib.ttf', 'cambria.ttc', 'candara.ttf', 'constan.ttf',
              'corbel.ttf', 'tahoma.ttf', 'segoeui.ttf', 'trebuc.ttf', 'cour.ttf', 'courbd.ttf', 'consola.ttf', 'lucon.ttf',
              'GARA.TTF', 'BOOKOS.TTF', 'CENTURY.TTF', 'pala.ttf', 'framd.ttf', 'ARIALN.TTF', 'comic.ttf', 'impact.ttf',
              'bahnschrift.ttf', 'sylfaen.ttf', 'micross.ttf'],
    'hebrew': ['arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf', 'david.ttf', 'davidbd.ttf', 'frank.ttf', 'gisha.ttf',
               'gishabd.ttf', 'rod.ttf', 'tahoma.ttf', 'tahomabd.ttf', 'cour.ttf', 'courbd.ttf', 'micross.ttf', 'l_10646.ttf',
               'segoeui.ttf', 'segoeuib.ttf'],
    'cyrillic': ['arial.ttf', 'times.ttf', 'verdana.ttf', 'calibri.ttf', 'georgia.ttf', 'tahoma.ttf', 'cour.ttf', 'segoeui.ttf'],
    'greek': ['arial.ttf', 'times.ttf', 'calibri.ttf', 'tahoma.ttf', 'segoeui.ttf'],
    'cjk': ['msyh.ttc', 'msyhbd.ttc', 'simsun.ttc', 'simhei.ttf', 'YuGothR.ttc', 'YuGothB.ttc', 'mingliub.ttc', 'msgothic.ttc'],
    'hangul': ['malgun.ttf', 'malgunbd.ttf'],
    'digits': ['arial.ttf', 'cour.ttf', 'consola.ttf', 'lucon.ttf', 'times.ttf', 'verdana.ttf', 'micross.ttf'],
}
for k in ('latin_acc',):
    FONTS[k] = FONTS['latin']
# held out for validation pages (never in training pages)
HELD = {'georgia.ttf', 'georgiab.ttf', 'gisha.ttf', 'gishabd.ttf', 'rod.ttf', 'corbel.ttf', 'lucon.ttf', 'simhei.ttf'}
WEIGHTS = {'latin': 0.34, 'latin_acc': 0.06, 'hebrew': 0.30, 'cyrillic': 0.07, 'greek': 0.03, 'cjk': 0.10, 'hangul': 0.03, 'digits': 0.07}


_CMAPS = {}


def cmap_of(path):
    """the code points a font file covers (a .ttc: its first font)"""
    if path not in _CMAPS:
        try:
            f = TTCollection(path).fonts[0] if path.lower().endswith('.ttc') else TTFont(path, lazy=True)
            _CMAPS[path] = set(f.getBestCmap() or {})
        except Exception:
            _CMAPS[path] = set()
    return _CMAPS[path]


def covered(text, path):
    cm = cmap_of(path)
    return all(ord(c) in cm or c == ' ' for c in text)


def rand_word(rng, script):
    if script == 'cjk':
        return ''.join(chr(rng.randint(0x4E00, 0x9FA5)) for _ in range(rng.randint(2, 8)))
    if script == 'hangul':
        return ''.join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(rng.randint(1, 4)))
    alpha = SCRIPTS[script][0]
    n = max(1, int(rng.gauss(5, 2.5)))
    w = ''.join(rng.choice(alpha) for _ in range(n))
    if script == 'hebrew':
        # final forms only at the end of a word
        fin = {'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ'}
        body = ''.join({'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}.get(c, c) for c in w[:-1])
        w = body + fin.get(w[-1], w[-1])
    elif script.startswith('latin') and rng.random() < 0.7:
        w = w.lower() if rng.random() < 0.85 else w.capitalize()
    return w


def rand_line(rng, script, nwords):
    if script in SENTENCES and rng.random() < 0.15:
        return rng.choice(SENTENCES[script])
    ws = [rand_word(rng, script) for _ in range(nwords)]
    if rng.random() < 0.25:
        ws.insert(rng.randint(0, len(ws)), str(rng.randint(1, 99999)) if rng.random() < 0.5 else f'{rng.uniform(0, 999):.2f}')
    sep = '' if script == 'cjk' and rng.random() < 0.7 else ' '
    s = sep.join(ws)
    if script not in ('cjk', 'hangul', 'digits') and rng.random() < 0.5:
        s += rng.choice('.,;:!?')
    return s


def font_for(rng, script, split, font_dir, cache):
    names = [f for f in FONTS[script] if os.path.exists(os.path.join(font_dir, f))]
    if split == 'train':
        names = [f for f in names if f not in HELD]
    elif split == 'held':
        held = [f for f in names if f in HELD]
        names = held or names
    if not names:
        return None
    return rng.choice(names)


def get_font(path, size, cache):
    key = (path, size)
    if key not in cache:
        cache[key] = ImageFont.truetype(path, size)
    return cache[key]


def render_page(rng, split, font_dir):
    W = rng.choice([1100, 1200, 1400])
    H = int(W * rng.choice([1.414, 1.294, 2.2, 3.0, 0.707, 1.0]))
    bg = rng.randint(200, 255)
    im = Image.new('L', (W, H), bg)
    d = ImageDraw.Draw(im)
    cache = {}
    scripts = list(WEIGHTS)
    main = rng.choices(scripts, weights=[WEIGHTS[s] for s in scripts])[0]
    second = rng.choice(['digits', 'latin', main, main])
    y = rng.randint(20, 120)
    ncol = rng.choice([1, 1, 1, 2, 2, 3]) if W >= H * 0.6 else 1
    margin = rng.randint(20, 120)
    colw = (W - 2 * margin - (ncol - 1) * 40) // ncol
    blocks = 0
    while y < H - 60 and blocks < 60:
        blocks += 1
        kind = rng.choices(['para', 'head', 'table', 'receipt'], weights=[6, 1.5, 1, 1])[0]
        script = main if rng.random() < 0.8 else second
        fname = font_for(rng, script, split, font_dir, cache)
        if fname is None:
            continue
        rtl = script == 'hebrew'
        size = rng.choice([14, 16, 18, 20, 22, 24, 28, 32, 40, 52]) if kind != 'head' else rng.choice([30, 36, 44, 56, 72])
        font = get_font(os.path.join(font_dir, fname), size, cache)
        ink = rng.randint(0, max(1, bg - 90))
        lh = int(size * rng.uniform(1.15, 1.8))
        if kind == 'table':
            rows, cols = rng.randint(3, 9), rng.randint(2, 5)
            cw = (W - 2 * margin) // cols
            for r in range(rows):
                if y > H - lh - 20:
                    break
                d.line([(margin, y), (W - margin, y)], fill=ink, width=rng.choice([1, 2]))
                for c in range(cols):
                    t = rand_line(rng, script if rng.random() < 0.6 else 'digits', rng.randint(1, 2))
                    cm = cmap_of(font.path)
                    t = ''.join(c for c in t if c == ' ' or ord(c) in cm)
                    if rtl:
                        t = t[::-1]
                    x = margin + c * cw + 8
                    d.text((x, y + 4), t, font=font, fill=ink)
                y += lh + 8
            d.line([(margin, y), (W - margin, y)], fill=ink, width=1)
            y += lh
            continue
        n_lines = 1 if kind == 'head' else rng.randint(2, 14)
        col = rng.randrange(ncol) if kind != 'head' else 0
        x0 = margin + col * (colw + 40)
        width = colw if kind != 'head' else W - 2 * margin
        if kind == 'receipt':
            width = min(width, rng.randint(300, 600))
            fname = rng.choice([f for f in ['cour.ttf', 'consola.ttf', 'lucon.ttf', fname] if os.path.exists(os.path.join(font_dir, f))])
            font = get_font(os.path.join(font_dir, fname), size, cache)
        for _ in range(n_lines):
            if y > H - lh - 10:
                break
            t = rand_line(rng, script, 30)
            fpath = font.path if hasattr(font, 'path') else ''
            if fpath:
                cm = cmap_of(fpath)
                t = ''.join(c for c in t if c == ' ' or ord(c) in cm)
            # wrap to the column
            words = t.split(' ') if ' ' in t else list(t)
            joiner = ' ' if ' ' in t else ''
            cur = ''
            for w in words:
                nxt = (cur + joiner + w) if cur else w
                if d.textlength(nxt, font=font) > width:
                    break
                cur = nxt
            if not cur:
                continue
            s = cur[::-1] if rtl else cur
            tw = d.textlength(s, font=font)
            align = rng.random()
            if rtl:
                x = x0 + width - tw
            elif kind == 'head' and align < 0.5:
                x = x0 + (width - tw) / 2
            else:
                x = x0
            d.text((x, y), s, font=font, fill=ink)
            if kind == 'receipt' and rng.random() < 0.5:
                p = f'{rng.uniform(0, 500):.2f}'  # digits: every font here has them
                d.text((x0 + width + 20, y), p, font=font, fill=ink)
            y += lh
        y += int(lh * rng.uniform(0.3, 1.5))
        if rng.random() < 0.1:
            d.line([(margin, y), (W - margin, y)], fill=ink, width=rng.choice([1, 2, 3]))
            y += 10
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='train', choices=['train', 'held'])
    ap.add_argument('--font_dir', default='C:/Windows/Fonts')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for i in range(a.n):
        im = render_page(rng, a.split, a.font_dir)
        s = 1400 / max(im.size)
        if s < 1:
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        im.save(os.path.join(a.out, f'{a.split}_{a.seed}_{i:05d}.png'))


if __name__ == '__main__':
    main()
