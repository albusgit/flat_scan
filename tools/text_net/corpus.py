"""The text-line net's multilingual corpus: the Universal Declaration of Human Rights in every language of the
NLTK udhr2 corpus (UTF-8, ~380 languages) and, per language, the installed fonts that cover its letters.
  python corpus.py UDHR2_DIR OUT.json [FONT_DIR]     (FONT_DIR default C:/Windows/Fonts)
OUT.json: {"langs": [{"code", "script", "rtl", "paras": [...], "fonts": [[file, face index, family, bold, italic]]}],
           "scripts": {script: n languages}}
A language's script is the most common Unicode script name of its letters (JAPANESE when kana occur, HAN for
Chinese); a font is kept when it covers >= 99.5 % of the language's distinct letters (symbol fonts left out).
Training data only: neither the text nor the fonts ship with the app."""
import glob, json, os, sys, unicodedata
from collections import Counter

from fontTools.ttLib import TTCollection, TTFont

RTL = {'ARABIC', 'HEBREW', 'SYRIAC', 'THAANA', 'NKO', 'ADLAM', 'SAMARITAN', 'MANDAIC'}
SKIP_FONT = ('symbol', 'wingding', 'webding', 'marlett', 'mdl2', 'icons', 'emoji', 'hololens', 'extra', 'bookshelf', 'segoe fluent')


def script_of(ch):
    try:
        n = unicodedata.name(ch)
    except ValueError:
        return None
    w = n.split()[0]
    if w == 'CJK':
        return 'HAN'
    if w in ('HIRAGANA', 'KATAKANA'):
        return 'KANA'
    if w in ('LATIN', 'CYRILLIC', 'GREEK') or unicodedata.category(ch)[0] in 'LM':
        return w
    return None


def faces(font_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(font_dir, '*'))):
        ext = f.lower().rsplit('.', 1)[-1]
        if ext not in ('ttf', 'otf', 'ttc'):
            continue
        try:
            fs = TTCollection(f, lazy=True).fonts if ext == 'ttc' else [TTFont(f, lazy=True)]
        except Exception:
            continue
        for k, t in enumerate(fs):
            try:
                fam = t['name'].getDebugName(1) or ''
                sub = (t['name'].getDebugName(2) or '').lower()
                if any(s in fam.lower() for s in SKIP_FONT):
                    continue
                cps = set(t.getBestCmap() or {})
            except Exception:
                continue
            if len(cps) < 100:
                continue
            out.append((os.path.basename(f), k, fam, 'bold' in sub or 'black' in sub, 'italic' in sub or 'oblique' in sub, cps))
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    font_dir = sys.argv[3] if len(sys.argv) > 3 else 'C:/Windows/Fonts'
    fl = faces(font_dir)
    print(len(fl), 'font faces', flush=True)
    langs = []
    for p in sorted(glob.glob(os.path.join(src, '*.txt'))):
        code = os.path.basename(p)[:-4]
        lines = [ln.strip() for ln in open(p, encoding='utf-8', errors='ignore').read().splitlines()]
        paras = [ln for ln in lines if len(ln) > 20]
        if len(paras) < 5:
            continue
        letters = Counter(ch for s in paras for ch in s if unicodedata.category(ch)[0] in 'LM')
        if sum(letters.values()) < 500:
            continue
        sc = Counter()
        for ch, n in letters.items():
            s = script_of(ch)
            if s:
                sc[s] += n
        if not sc:
            continue
        script = sc.most_common(1)[0][0]
        if 'KANA' in sc and sc['KANA'] > 0.1 * sum(sc.values()):
            script = 'JAPANESE'
        need = set(ord(c) for c in letters)
        ok = [[f, k, fam, b, i] for f, k, fam, b, i, cps in fl if len(need & cps) >= 0.995 * len(need)]
        if not ok:
            continue
        langs.append({'code': code, 'script': script, 'rtl': script in RTL, 'paras': paras, 'fonts': ok})
    scripts = Counter(l['script'] for l in langs)
    json.dump({'langs': langs, 'scripts': scripts}, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(len(langs), 'languages,', len(scripts), 'scripts:', dict(scripts.most_common()), flush=True)
    for s in scripts:
        fams = Counter(fam for l in langs if l['script'] == s for _, _, fam, _, _ in l['fonts'])
        print(f'  {s}: {len(fams)} font families, e.g. {", ".join(f for f, _ in fams.most_common(4))}')


if __name__ == '__main__':
    main()
