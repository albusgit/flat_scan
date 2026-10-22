"""Collect free-licence camera photos from Wikimedia Commons into the FlatScan test library.

  python -I commons_collect.py OUT_DIR SPEC.json [--max N] [--width 2000]

SPEC.json: {"category_name": {"cats": ["Category:Receipts", ...], "depth": 2,
                              "search": ["receipt", ...], "max": 300}, ...}
Only JPEG files with camera EXIF (Make or Model: a photograph, not a scan or render), >= 1000 px on
the long side, under CC0 / public domain / CC BY / CC BY-SA, are kept. Each image is saved as a
Commons thumbnail (long side about --width px) in OUT_DIR/<category>/, and OUT_DIR/<category>/
licences.csv records title, licence, author, source page and the downloaded URL."""
import argparse
import csv
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://commons.wikimedia.org/w/api.php"
UA = "FlatScanTestLibrary/1.0 (offline document-scanner test set; https://commons.wikimedia.org)"
FREE = re.compile(r"^(cc0|cc[- ]?zero|public domain|pd|cc by(-sa)? ?\d|cc-by(-sa)?-\d|cc by(-sa)?$)", re.I)


def get(params, tries=4):
    params = dict(params, format="json", maxlag=5)
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("api retry", i, e, file=sys.stderr)
            time.sleep(3 * (i + 1))
    return {}


def strip(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def file_info(titles, width):
    out = {}
    for i in range(0, len(titles), 40):
        d = get(dict(action="query", titles="|".join(titles[i:i + 40]), prop="imageinfo",
                     iiprop="url|size|mime|extmetadata|commonmetadata", iiurlwidth=width))
        for p in d.get("query", {}).get("pages", {}).values():
            if "imageinfo" in p:
                out[p["title"]] = p["imageinfo"][0]
    return out


def cat_files(cat, depth, seen_cats):
    if cat in seen_cats:
        return []
    seen_cats.add(cat)
    files, cont = [], {}
    while True:
        d = get(dict(action="query", list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit=500, **cont))
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 6:
                files.append(m["title"])
            elif m["ns"] == 14 and depth > 0:
                files += cat_files(m["title"], depth - 1, seen_cats)
        if "continue" not in d:
            break
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    return files


def search_files(q, limit=300):
    files, off = [], 0
    while off < limit:
        d = get(dict(action="query", list="search", srsearch=q + " filemime:image/jpeg", srnamespace=6,
                     srlimit=min(100, limit - off), sroffset=off))
        hits = d.get("query", {}).get("search", [])
        files += [h["title"] for h in hits]
        if "continue" not in d or not hits:
            break
        off = d["continue"]["sroffset"]
    return files


def keep(ii):
    if ii.get("mime") != "image/jpeg" or max(ii.get("width", 0), ii.get("height", 0)) < 1000:
        return None
    cm = {m["name"]: m["value"] for m in ii.get("commonmetadata", []) if isinstance(m.get("value"), (str, int, float))}
    if not (str(cm.get("Make", "")).strip() or str(cm.get("Model", "")).strip()):
        return None
    em = ii.get("extmetadata", {})
    lic = strip(em.get("LicenseShortName", {}).get("value", ""))
    if not FREE.match(lic):
        return None
    return dict(licence=lic, author=strip(em.get("Artist", {}).get("value", ""))[:200],
                camera=(str(cm.get("Make", "")).strip() + " " + str(cm.get("Model", "")).strip()).strip(),
                licence_url=em.get("LicenseUrl", {}).get("value", ""))


def fetch(url, path):
    if os.path.exists(path):
        return True
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return True
        except Exception as e:  # noqa: BLE001
            print("dl retry", i, url, e, file=sys.stderr)
            time.sleep(5 * (i + 1))
    return False


def safe(title):
    t = title.split(":", 1)[1]
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", t)
    return t[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("spec")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    taken_global = set()   # titles already in any category of OUT (earlier runs)
    for lic in __import__("glob").glob(os.path.join(a.out, "*", "licences.csv")):
        taken_global.update(r["title"] for r in csv.DictReader(open(lic, encoding="utf-8")))
    for name, s in spec.items():
        if a.only and name not in a.only.split(","):
            continue
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        titles, seen_cats = [], set()
        for q in s.get("search", []):
            titles += search_files(q, s.get("search_limit", 200))
        for c in s.get("cats", []):
            titles += cat_files(c, s.get("depth", 1), seen_cats)[:s.get("per_cat", 1500)]
        titles = [t for t in dict.fromkeys(titles) if t not in taken_global][:s.get("candidates", 5000)]
        print(name, "candidates", len(titles), flush=True)
        info = file_info(titles, a.width)
        rows = []
        cap = s.get("max", a.max)
        for t in titles:
            ii = info.get(t)
            if not ii:
                continue
            k = keep(ii)
            if not k:
                continue
            rows.append(dict(file=safe(t), title=t, source=ii.get("descriptionurl", ""),
                             url=ii.get("thumburl") or ii["url"], width=ii["width"], height=ii["height"], **k))
            if len(rows) >= cap:
                break
        print(name, "kept", len(rows), flush=True)
        with ThreadPoolExecutor(2) as ex:
            ok = list(ex.map(lambda r: fetch(r["url"], os.path.join(d, r["file"])), rows))
        rows = [r for r, o in zip(rows, ok) if o]
        taken_global.update(r["title"] for r in rows)
        with open(os.path.join(d, "licences.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
            w.writeheader()
            w.writerows(rows)
        print(name, "downloaded", len(rows), flush=True)


if __name__ == "__main__":
    main()
