"""Build the set indexes E:/fs_datasets/sets/<set>.json from the downloaded / extracted data.

  python -I prepare_sets.py [SET ...]      (default: every set whose data is present)

Index format (read by run_library.py):
  {"name", "kind": "flat_gt" | "no_gt" | "clean_gt", "frame_corners": bool, "licence", "source", "use",
   "items": [{"id", "photo", "gt"?, "words"?, "tags": [...]}]}
kind flat_gt: "gt" is the flat scan / digital page (LD / MS-SSIM / AD, OCR vs the scan's OCR).
kind clean_gt: "gt" is the same photo without shadows (shadow removal sets; PSNR / SSIM of filters).
"words" (optional): a JSON list of the true words on the page (order-free word recall).
frame_corners: the photos are cropped to the page (run with FS_FRAME_CORNERS=1 too)."""
import csv
import glob
import json
import os
import re
import sys

ROOT = "E:/fs_datasets"
RAW = ROOT + "/raw"
SETS = ROOT + "/sets"


def p(*a):
    return os.path.join(*a).replace("\\", "/")


def save(name, items, **meta):
    os.makedirs(SETS, exist_ok=True)
    d = dict(name=name, **meta, n=len(items), items=items)
    json.dump(d, open(p(SETS, name + ".json"), "w"), indent=1)
    print(name, len(items))


def num_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def dir300():
    d = p(RAW, "dir300/DIR300")
    items = []
    for f in sorted(os.listdir(p(d, "dist")), key=num_key):
        if not f.endswith(".png") or f.endswith(".tmp"):
            continue
        g = p(d, "gt", f)
        if os.path.exists(g):
            items.append(dict(id=f[:-4], photo=p(d, "dist", f), gt=g, tags=[]))
    save("dir300", items, kind="flat_gt", frame_corners=False, use="validation",
         licence="DocGeoNet repo licence (academic; commercial use: ask the authors)",
         source="https://github.com/fh2019ustc/DocGeoNet")


def inv3dreal():
    d = p(RAW, "inv3d_model/input/inv3d_real/data")
    items = []
    for doc in sorted(os.listdir(d)):
        gt = p(d, doc, "flat_document.png")
        words = p(d, doc, "ground_truth_words.json")
        for f in sorted(glob.glob(p(d, doc, "warped_document_*.jpg"))):
            m = re.match(r"warped_document_(\w+)_(\w+)\.jpg", os.path.basename(f))
            items.append(dict(id=f"{doc}_{m.group(1)}_{m.group(2)}", photo=p(f), gt=gt, words=words,
                              tags=[m.group(1), m.group(2), "doc" + doc]))
    save("inv3dreal", items, kind="flat_gt", frame_corners=False, use="validation",
         licence="MIT (inv3d-model repository)",
         source="https://github.com/FelixHertlein/inv3d-model/tree/main/input/inv3d_real/data")


def generic_pairs(name, photo_dir, gt_dir, kind, frame, licence, source, use, photo_glob="*", gt_name=None, tags=()):
    items = []
    for f in sorted(glob.glob(p(photo_dir, photo_glob)), key=num_key):
        if not re.search(r"\.(png|jpe?g|bmp|tif)$", f, re.I):
            continue
        base = os.path.splitext(os.path.basename(f))[0]
        g = gt_name(base) if gt_name else None
        if g is None:
            cand = glob.glob(p(gt_dir, base + ".*"))
            g = cand[0] if cand else None
        if g and os.path.exists(g):
            items.append(dict(id=base, photo=p(f), gt=p(g), tags=list(tags)))
    save(name, items, kind=kind, frame_corners=frame, licence=licence, source=source, use=use)


def real_folder(name, folder, use, licence, source, tags_from_dir=True):
    items = []
    for f in sorted(glob.glob(p(folder, "**", "*"), recursive=True)):
        if not re.search(r"\.(png|jpe?g)$", f, re.I) or ".lines." in f or ".auto." in f:
            continue
        rel = os.path.relpath(f, folder).replace("\\", "/")
        tags = rel.split("/")[:-1] if tags_from_dir else []
        items.append(dict(id=re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(rel)[0]), photo=p(f), tags=tags))
    save(name, items, kind="no_gt", frame_corners=False, licence=licence, source=source, use=use)


def rdd():
    d = p(RAW, "rdd/test")
    generic_pairs("rdd_shadow", p(d, "img"), p(d, "gt"), "clean_gt", False,
                  "none stated (research dataset, CVPR 2023)", "https://github.com/hyyh1314/RDD", "filter validation (shadow removal)",
                  tags=["shadow"])


def jung():
    d = p(RAW, "jung/x/PSNR")
    generic_pairs("jung_shadow", p(d, "Original"), p(d, "GroundTruth"), "clean_gt", False,
                  "Springer ESM supplementary material of Jung et al., ACCV 2018 (research use)",
                  "https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-20887-5_25/MediaObjects/484511_1_En_25_MOESM1_ESM.zip",
                  "filter validation (shadow removal)", tags=["shadow"])


def wildreceipt():
    import cv2
    d = p(RAW, "wildreceipt/wildreceipt")
    out = p(ROOT, "derived/wildreceipt_words")
    os.makedirs(out, exist_ok=True)
    items = []
    for split in ("test", "train"):
        for line in open(p(d, split + ".txt"), encoding="utf-8"):
            j = json.loads(line)
            f = p(d, j["file_name"])
            iid = split + "_" + re.sub(r"[^A-Za-z0-9]+", "_", j["file_name"].replace("image_files/", ""))[:-5]
            words = [a["text"] for a in j["annotations"] if a.get("text")]
            wp = p(out, iid + ".json")
            json.dump(words, open(wp, "w"))
            long_side = max(j.get("height", 0), j.get("width", 0))
            items.append(dict(id=iid, photo=f, words=wp, tags=[split, "hires" if long_side >= 1000 else "lowres"]))
    save("wildreceipt", items, kind="no_gt", frame_corners=False, use="OCR / filter test (receipt photos)",
         licence="Apache-2.0 (MMOCR WildReceipt)", source="https://download.openmmlab.com/mmocr/data/wildreceipt.tar")


def funsd():
    d = p(RAW, "funsd/x/dataset")
    out = p(ROOT, "derived/funsd_words")
    os.makedirs(out, exist_ok=True)
    items = []
    for split in ("testing_data", "training_data"):
        for f in sorted(glob.glob(p(d, split, "images", "*.png"))):
            base = os.path.splitext(os.path.basename(f))[0]
            ann = json.load(open(p(d, split, "annotations", base + ".json"), encoding="utf-8"))
            words = [w["text"] for blk in ann["form"] for w in blk.get("words", []) if w.get("text")]
            wp = p(out, base + ".json")
            json.dump(words, open(wp, "w"))
            items.append(dict(id=base, photo=p(f), words=wp, tags=[split.split("_")[0], "scan"]))
    save("funsd", items, kind="no_gt", frame_corners=True, use="OCR (scanned forms; FS_SCAN / filters, not geometry)",
         licence="FUNSD: non-commercial research and educational use", source="https://guillaumejaume.github.io/FUNSD/")


def commons():
    base = p(ROOT, "real/commons")
    for cat in sorted(os.listdir(base)):
        lic = p(base, cat, "licences.csv")
        if not os.path.exists(lic):
            continue
        items = []
        for r in csv.DictReader(open(lic, encoding="utf-8")):
            f = p(base, cat, r["file"])
            if os.path.exists(f):
                items.append(dict(id=os.path.splitext(r["file"])[0], photo=f, tags=[cat], licence=r["licence"],
                                  source=r["source"]))
        cur = p(base, cat, "curated.txt")   # optional hand-picked subset (one file name per line)
        if os.path.exists(cur):
            keep = {ln.strip() for ln in open(cur, encoding="utf-8") if ln.strip()}
            for it in items:
                if os.path.basename(it["photo"]) in keep:
                    it["tags"].append("curated")
        save("commons_" + cat, items, kind="no_gt", frame_corners=False, use="visual review / filter / monitor work (no GT)",
             licence="per file (field licence; real/commons/%s/licences.csv)" % cat, source="https://commons.wikimedia.org")


def real35():
    real_folder("real35", r"C:/Users/User/Desktop/round4_robustness/real_photos", "visual review (Round 8-10 35 real photos; some are DocUNet photos)",
                "see round4_robustness/real_photos/SOURCES.md", "round4_robustness", tags_from_dir=False)


def uvdoc_bench():
    d = p(RAW, "uvdoc_bench/x/UVDoc_benchmark")
    generic_pairs("uvdoc_bench", p(d, "img"), p(d, "texture_sample"), "flat_gt", False,
                  "none stated (UVDoc project page; textures from public sources, see UVDoc-Dataset README)",
                  "https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip", "validation",
                  tags=["semi-synthetic"])


def warpdoc():
    d = p(RAW, "warpdoc/x/WarpDoc")
    items = []
    for cat in sorted(os.listdir(p(d, "image"))):
        for f in sorted(os.listdir(p(d, "image", cat))):
            g = p(d, "digital_margin", cat, f)
            if os.path.exists(g):
                items.append(dict(id=f"{cat}_{f[:-4]}", photo=p(d, "image", cat, f), gt=g,
                                  gt_nomargin=p(d, "digital", cat, f), tags=[cat]))
    save("warpdoc", items, kind="flat_gt", frame_corners=False, use="validation (category 'incomplete': page partly outside the photo)",
         licence="none stated (WarpDoc, Xue et al. CVPR 2022 'Fourier document restoration')",
         source="https://sg-vilab.github.io/event/warpdoc/ (Drive 1UHzgERrRR6E08bpL1AyCzD01bFczlAVP)")


def docreal():
    d = p(RAW, "docreal/x/DocReal")
    generic_pairs("docreal", p(d, "distorted"), p(d, "scanned"), "flat_gt", False,
                  "none stated (DocReal repo is MIT; data terms not given)", "https://github.com/irisXcoding/DocReal", "validation",
                  gt_name=lambda b: p(d, "scanned", b.split("_")[0] + ".png"))


def uhdm():
    d = p(RAW, "uhdm/x/test")
    items = []
    for f in sorted(glob.glob(p(d, "*_moire.jpg"))):
        b = os.path.basename(f)[:-len("_moire.jpg")]
        items.append(dict(id=b, photo=p(f), gt=p(d, b + "_gt.jpg"), tags=["screen", "moire"]))
    save("uhdm_screens", items, kind="clean_gt", frame_corners=True, use="MONITOR / moire work: real 4K phone photos of screens + the clean screenshot (aligned)",
         licence="not stated (UHDM, Yu et al. ECCV 2022; repo Apache-2.0)", source="https://github.com/CVMI-Lab/UHDM (Drive file test.tar.gz)")


def subsets():
    """Fixed, cheaper validation subsets (deterministic: every k-th item of the full index)."""
    for src, k, name in (("warpdoc", 6, "warpdoc_s170"), ("inv3dreal", 4, "inv3dreal_s90")):
        S = json.load(open(p(SETS, src + ".json")))
        items = S["items"][::k]
        meta = {kk: v for kk, v in S.items() if kk not in ("name", "items", "n")}
        meta["use"] = S.get("use", "") + f" (subset: every {k}th item of {src})"
        save(name, items, **meta)


def cord():
    d = p(RAW, "cord/x")
    items = []
    for split in ("test", "validation"):
        for f in sorted(glob.glob(p(d, split, "[0-9][0-9][0-9].*"))):
            b = os.path.splitext(os.path.basename(f))[0]
            items.append(dict(id=f"{split}_{b}", photo=p(f), words=p(d, split, b + "_words.json"), tags=[split, "receipt"]))
    save("cord", items, kind="no_gt", frame_corners=False, use="OCR / filter (receipt photos, Indonesian shops; true words)",
         licence="CC BY 4.0 (CORD v2)", source="https://huggingface.co/datasets/naver-clova-ix/cord-v2 (test + validation parquet)")


def diw():
    d = p(RAW, "diw/x/5k/img")
    items = [dict(id=os.path.splitext(f)[0], photo=p(d, f), tags=["512px"]) for f in sorted(os.listdir(d)) if f.endswith(".png")]
    save("diw", items, kind="no_gt", frame_corners=False, use="robustness / page finding only: 5667 real photos but resized to 512x512 (seg masks in raw/diw/x/5k/seg)",
         licence="not stated (PaperEdge repo MIT)", source="https://github.com/cvlab-stonybrook/PaperEdge (Drive 1qAmLurt6bK0ro8PnRz6rBgVs1rfrsdKi)")


BUILDERS = {"cord": cord, "diw": diw, "uhdm": uhdm, "warpdoc": warpdoc, "docreal": docreal, "uvdoc_bench": uvdoc_bench, "dir300": dir300, "inv3dreal": inv3dreal, "rdd": rdd, "jung": jung, "wildreceipt": wildreceipt,
            "funsd": funsd, "commons": commons, "real35": real35, "subsets": subsets}

if __name__ == "__main__":
    want = sys.argv[1:] or list(BUILDERS)
    for w in want:
        try:
            BUILDERS[w]()
        except Exception as e:  # noqa: BLE001
            print(w, "skipped:", e)
