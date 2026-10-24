"""Whiteboard mode on testdata/whiteboard and testdata/projector (Phase 1, feature 4): each photo through
pipeline_cli FS_BOARD=1 FS_RENDER=1 (the board's corners or the whole frame, perspective only, the Whiteboard /
Dark slide filter at 0.7, as the app runs it), then a before / after sheet: the photo with the corners used
(green: found, orange: not found, whole frame) next to the result.
Usage: python3 -I tools/board/sheet.py CLI TESTDATA OUTDIR   -> OUTDIR/sheet_<set>_<n>.jpg, OUTDIR/board.md"""
import json, os, re, subprocess, sys
from PIL import Image, ImageDraw

cli, td, out = sys.argv[1:4]
no_sheet = "--no-sheet" in sys.argv
# hand-labelled corners (percent), for the IoU column
gt_path = os.path.join(td, "board_corners.json")
gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}


def iou(a, b, w, h):
    """IoU of two quads (pixels) on a 400 px raster of the photo."""
    k = 400 / max(w, h)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    ma, mb = Image.new("1", size), Image.new("1", size)
    ImageDraw.Draw(ma).polygon([(x * k, y * k) for x, y in a], fill=1)
    ImageDraw.Draw(mb).polygon([(x * k, y * k) for x, y in b], fill=1)
    pa, pb = ma.getdata(), mb.getdata()
    inter = sum(1 for x, y in zip(pa, pb) if x and y)
    union = sum(1 for x, y in zip(pa, pb) if x or y)
    return inter / union if union else 0.0


os.makedirs(out, exist_ok=True)
work = os.path.join(out, "work")
os.makedirs(work, exist_ok=True)
rows = []
tiles = {}
for cat in ("whiteboard", "projector"):
    d = os.path.join(td, cat)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".jpg"):
            continue
        base = os.path.join(work, os.path.splitext(f)[0])
        env = dict(os.environ, FS_BOARD="1")
        if not no_sheet:
            env["FS_RENDER"] = "1"
        r = subprocess.run([cli, os.path.join(d, f), base + ".jpg"], env=env, capture_output=True, text=True, timeout=600)
        m = re.search(r"board: ok (\d) method (\S*) confidence ([\d.]+) area ([\d.]+) dark (\d) corners (.*)", r.stderr)
        if not m or (not no_sheet and not os.path.exists(base + ".jpg.board.jpg")):
            rows.append((cat, f, "error", "", "", "", "", ""))
            continue
        ok, method, conf, area, dark, cs = m.groups()
        pts = [tuple(map(float, p.split(","))) for p in cs.split()]
        photo = Image.open(os.path.join(d, f)).convert("RGB")
        g = gt.get(f)
        score = "" if g is None else "%.2f" % iou(pts, [(x * photo.width / 100, y * photo.height / 100) for x, y in g], photo.width, photo.height)
        if no_sheet:
            rows.append((cat, f, "found" if ok == "1" else "whole frame", method, conf, area, "yes" if dark == "1" else "no", score))
            continue
        dr = ImageDraw.Draw(photo)
        dr.line(pts + [pts[0]], fill=(40, 200, 60) if ok == "1" else (255, 140, 0), width=max(4, photo.width // 200))
        res = Image.open(base + ".jpg.board.jpg").convert("RGB")
        H = 420
        a = photo.resize((max(1, int(photo.width * H / photo.height)), H))
        b = res.resize((max(1, int(res.width * H / res.height)), H))
        t = Image.new("RGB", (a.width + b.width + 12, H + 30), (255, 255, 255))
        t.paste(a, (0, 30)); t.paste(b, (a.width + 12, 30))
        ImageDraw.Draw(t).text((4, 6), f"{f}: {'found (' + method + ')' if ok == '1' else 'not found: whole frame'}{', dark slide' if dark == '1' else ''}", fill=(0, 0, 0))
        tiles.setdefault(cat, []).append(t)
        rows.append((cat, f, "found" if ok == "1" else "whole frame", method, conf, area, "yes" if dark == "1" else "no", score))
for cat, ts in tiles.items():
    for k in range(0, len(ts), 4):
        chunk = ts[k:k + 4]
        W = max(t.width for t in chunk)
        sheet = Image.new("RGB", (W, sum(t.height for t in chunk) + 10 * len(chunk)), (235, 235, 235))
        y = 0
        for t in chunk:
            sheet.paste(t, (0, y)); y += t.height + 10
        sheet.save(os.path.join(out, f"sheet_{cat}_{k // 4 + 1}.jpg"), quality=82)
with open(os.path.join(out, "board.md"), "w") as fh:
    fh.write("The sheets are made from the photos in testdata/whiteboard and testdata/projector (Wikimedia Commons; "
             "authors and licences in testdata/MANIFEST_extra.json; the CC BY-SA ones are shared under the same licence).\n\n")
    fh.write("| set | photo | corners | method | edge support | area | dark slide | IoU with hand-labelled corners |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write("| " + " | ".join(r) + " |\n")
    fh.write("\n")
    for cat in ("whiteboard", "projector"):
        v = [float(r[7]) for r in rows if r[0] == cat and r[7]]
        if v:
            fh.write(f"{cat}: mean IoU {sum(v) / len(v):.2f} over {len(v)} labelled photos, IoU >= 0.8 on {sum(x >= 0.8 for x in v)}, < 0.5 on {sum(x < 0.5 for x in v)}\n\n")
print(open(os.path.join(out, "board.md")).read())
