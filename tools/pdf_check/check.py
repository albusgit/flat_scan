"""Checks the PDFs of PdfWriterTest (app/build/pdf_check/) in three PDF engines, as the plan's acceptance asks
("open in at least two viewers", "text search works on a Hebrew and an English page", aligned text layer):

  * poppler (Evince, Okular, most Linux viewers): qpdf --check for structure, pdftotext for search, pdftotext
    -bbox for where each word sits, pdftoppm to render;
  * PDFium (Chrome, Edge, Android's own PDF viewer): pypdfium2 for search and rendering;
  * pdf.js (Firefox): pdfjs-dist through node (tools/pdf_check/pdfjs_text.mjs) for search.

It also decodes the CCITT G4 test image with libtiff (Pillow) and compares it with the raw bits.
Usage: python3 -I tools/pdf_check/check.py app/build/pdf_check PDFJS_NODE_MODULES_DIR > report.txt
Exit code 1 when a check fails."""
import json, os, re, struct, subprocess, sys, unicodedata
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

D, NODE_DIR = os.path.abspath(sys.argv[1]), sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "app", "src", "test", "resources", "output")
failed = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""))
    if not ok:
        failed.append(name)


def norm(s):
    # what a viewer's search compares: NFC, no bidi controls, whitespace collapsed
    s = unicodedata.normalize("NFC", s)
    s = re.sub("[‎‏‪-‮⁦-⁩]", "", s)
    return re.sub(r"\s+", " ", s)


def poppler_text(f):
    return subprocess.run(["pdftotext", "-enc", "UTF-8", f, "-"], capture_output=True, text=True).stdout


def pdfium_text(f):
    pdf = pdfium.PdfDocument(f)
    return "\f".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))


def poppler_png(f, page=1, dpi=72):
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-gray", "-f", str(page), "-l", str(page), "-singlefile", f, os.path.join(t, "p")], check=True)
        return Image.open(os.path.join(t, "p.png")).convert("L").copy()


def pdfjs(f):
    env = dict(os.environ, PDFJS_MODULES=NODE_DIR)
    r = subprocess.run(["node", os.path.join(HERE, "pdfjs_text.mjs"), f], capture_output=True, text=True, cwd=os.path.dirname(NODE_DIR), env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout)


def words_of(tsv):
    out = []
    for row in open(os.path.join(RES, tsv), encoding="utf-8").read().splitlines()[1:]:
        c = row.split("\t")
        if len(c) >= 12 and c[0] == "5" and c[11].strip() and float(c[10]) >= 80 and len(c[11].strip()) >= 4 and c[11].strip().isalpha():
            out.append((c[11].strip(), int(c[6]), int(c[7]), int(c[8]), int(c[9])))
    return out


HEB = [w for w in words_of("hebrew_page.tsv") if re.search("[֐-׿]", w[0])]
ENG = words_of("english_page.tsv")
print(f"search words: {len(HEB)} Hebrew, {len(ENG)} English (Tesseract confidence >= 80, 4+ letters)\n")

# 1. structure
for f in sorted(os.listdir(D)):
    if f.endswith(".pdf"):
        r = subprocess.run(["qpdf", "--check", os.path.join(D, f)], capture_output=True, text=True)
        check(f"qpdf --check {f}", r.returncode == 0 and "No syntax or stream encoding errors" in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

# 2. search in three engines
engines = {"poppler": poppler_text, "pdfium": pdfium_text, "pdf.js": lambda f: "\f".join(pdfjs(f)["pages"])}
print()
for f, words, lang in [("hebrew_visual.pdf", HEB, "Hebrew"), ("hebrew_logical.pdf", HEB, "Hebrew"), ("english.pdf", ENG, "English"),
                       ("both_bw_g4.pdf", HEB + ENG, "both")]:
    for eng, fn in engines.items():
        text = norm(fn(os.path.join(D, f)))
        found = sum(1 for w in words if norm(w[0]) in text)
        share = found / max(1, len(words))
        ok = share >= 0.95
        if f == "hebrew_logical.pdf":
            print(f"INFO {eng:8s} {f}: {found}/{len(words)} {lang} words found ({share:.0%}) [the variant not used]")
        else:
            check(f"{eng} search {f}", ok, f"{found}/{len(words)} {lang} words found ({share:.0%})")

# 3. metadata
meta = pdfjs(os.path.join(D, "hebrew_visual.pdf"))
check("pdf.js reads the Hebrew title", meta["title"] == "חשבון בדיקה", repr(meta["title"]))
info = subprocess.run(["pdfinfo", os.path.join(D, "hebrew_visual.pdf")], capture_output=True, text=True).stdout
check("pdfinfo: Creator, Producer, dates", all(k in info for k in ["Creator:", "Producer:", "CreationDate:", "ModDate:"]), " | ".join(l for l in info.splitlines() if l.split(":")[0] in ("Title", "Creator", "Producer", "CreationDate")))

# 4. alignment: where PDFium's search highlights each word (what a user sees after searching) against the
#    word's box in the page image, both in points
print()
for f, tsv, img in [("english.pdf", "english_page.tsv", "english_page.jpg"), ("hebrew_visual.pdf", "hebrew_page.tsv", "hebrew_page.jpg")]:
    iw, ih = Image.open(os.path.join(RES, img)).size
    k = 595.0 / iw
    doc = pdfium.PdfDocument(os.path.join(D, f))   # kept: the text page needs its document alive
    page = doc[0]
    tp = page.get_textpage()
    errs = []
    for t, l, top, w, h in words_of(tsv):
        want = ((l + w / 2) * k, 842 - (top + h / 2) * k)
        hits = []
        se = tp.search(t, match_case=True, match_whole_word=False)
        while (r := se.get_next()) is not None:
            i, n = r
            rects = [tp.get_rect(j) for j in range(tp.count_rects(i, n))]
            if rects:
                x0 = min(q[0] for q in rects); y0 = min(q[1] for q in rects); x1 = max(q[2] for q in rects); y1 = max(q[3] for q in rects)
                hits.append(((x0 + x1) / 2, (y0 + y1) / 2))
        if hits:
            errs.append(min(abs(x - want[0]) + abs(y - want[1]) for x, y in hits))
    errs.sort()
    med = errs[len(errs) // 2] if errs else 99
    p95 = errs[int(len(errs) * 0.95)] if errs else 99
    check(f"search highlight on the word, {f}", len(errs) > 0.95 * len(words_of(tsv)) and med < 3 and p95 < 6,
          f"centre offset median {med:.2f} pt, p95 {p95:.2f} pt, max {errs[-1]:.2f} pt over {len(errs)} words (1 pt = 0.35 mm)")

# 5. rendering: the page looks like the image (text invisible), poppler and PDFium
print()
for f, img in [("english.pdf", "english_page.jpg"), ("hebrew_visual.pdf", "hebrew_page.jpg")]:
    src = np.asarray(Image.open(os.path.join(RES, img)).convert("L").resize((595, 842)), dtype=float)
    a = np.asarray(poppler_png(os.path.join(D, f), dpi=200).resize((595, 842)), dtype=float)
    b = np.asarray(pdfium.PdfDocument(os.path.join(D, f))[0].render(scale=200 / 72).to_pil().convert("L").resize((595, 842)), dtype=float)
    check(f"poppler renders {f} as its image", np.mean(np.abs(a - src)) < 6, f"mean |diff| {np.mean(np.abs(a - src)):.2f} levels")
    check(f"PDFium renders {f} as its image", np.mean(np.abs(b - src)) < 6, f"mean |diff| {np.mean(np.abs(b - src)):.2f} levels")

# 6. CCITT G4 against libtiff
w, h = 97, 41
g4 = open(os.path.join(D, f"g4_{w}x{h}.g4"), "rb").read()
raw = np.frombuffer(open(os.path.join(D, f"g4_{w}x{h}.raw"), "rb").read(), dtype=np.uint8)
want = np.unpackbits(raw).reshape(h, -1)[:, :w] == 0     # black
tags = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 1), (259, 3, 1, 4), (262, 3, 1, 0), (273, 4, 1, 8 + 2 + 12 * 9 + 4),
        (278, 3, 1, h), (279, 4, 1, len(g4)), (293, 4, 1, 0)]
ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", t, ty, c, v) if ty == 4 else struct.pack("<HHIHH", t, ty, c, v, 0) for t, ty, c, v in tags) + struct.pack("<I", 0)
tif = b"II*\x00" + struct.pack("<I", 8) + ifd + g4
got = np.asarray(Image.open(BytesIO(tif)).convert("L")) < 128
check("CCITT G4 decodes (libtiff) to the same pixels", got.shape == want.shape and np.array_equal(got, want), f"{int((got != want).sum()) if got.shape == want.shape else 'shape'} pixels differ")
ga, fa = (np.asarray(poppler_png(os.path.join(D, x), page=2), dtype=float) for x in ("both_bw_g4.pdf", "both_bw_flate.pdf"))
check("G4 and Flate pages render the same (poppler)", np.mean(np.abs(ga - fa)) < 0.5, f"mean |diff| {np.mean(np.abs(ga - fa)):.3f}")

print("\n" + ("ALL PASSED" if not failed else f"{len(failed)} FAILED: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
