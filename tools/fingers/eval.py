"""Finger detection benchmark (Phase 1, feature 5) on testdata/fingers and testdata/control.

Each photo goes through the app's pipeline (pipeline_cli with the nets, the rule net and the shadow net, as
the app runs them) with FS_FINGER_OUT, which writes, in the photo's frame, the page's footprint and what was
filled as a finger. Metrics:
  * finger set: IoU of the filled area with the hand-drawn ground truth (testdata/fingers/masks/*.png),
    both taken inside the page's footprint (fingers outside the page are not the detector's business);
    reported per photo, as the mean, and pooled (all intersections / all unions);
  * control set (no fingers): false-positive rate = share of photos where anything (> 0.05 % of the page)
    was filled, and the mean filled share.
Usage: python3 -I tools/fingers/eval.py CLI TESTDATA OUTDIR [--env KEY=VALUE ...] [--jobs N] [--label NAME]
Writes OUTDIR/<label>.json and prints a summary line. Needs numpy and Pillow."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("cli"); ap.add_argument("testdata"); ap.add_argument("out")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--label", default="run")
ap.add_argument("--only", default="")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
models = os.path.join(ROOT, "app", "src", "main", "assets", "models")
env = dict(os.environ, FS_MODELS=models, FS_NETS="1", FS_RULE_NET="1", FS_SHADE_NET=os.path.join(models, "shadenet.onnx"))
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
os.makedirs(os.path.join(a.out, a.label), exist_ok=True)


def run(photo):
    base = os.path.join(a.out, a.label, os.path.splitext(os.path.basename(photo))[0])
    if not os.path.exists(base + "_footprint.png"):
        e = dict(env, FS_FINGER_OUT=base)
        r = subprocess.run([a.cli, photo, base + ".jpg", "net"], env=e, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return photo, None, r.stderr[-300:]
    return photo, base, None


def load(p, size=None):
    im = Image.open(p).convert("L")
    if size and im.size != size:
        im = im.resize(size, Image.NEAREST)
    return np.asarray(im) > 127


photos = []
for cat in ("fingers", "control"):
    d = os.path.join(a.testdata, cat)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".jpg") and (not a.only or a.only in f):
            photos.append((cat, os.path.join(d, f)))
with ThreadPoolExecutor(a.jobs) as ex:
    results = list(ex.map(lambda cp: (cp[0],) + run(cp[1]), photos))

rows, inter, union = [], 0, 0
ctrl_fp, ctrl_area = [], []
for cat, photo, base, err in results:
    name = os.path.basename(photo)
    if base is None:
        rows.append({"photo": name, "set": cat, "error": err})
        continue
    size = Image.open(photo).size
    foot = load(base + "_footprint.png", size)
    pred = load(base + "_fingers.png", size) & foot
    fa = max(1, int(foot.sum()))
    if cat == "fingers":
        gt_path = os.path.join(a.testdata, "fingers", "masks", os.path.splitext(name)[0] + ".png")
        if not os.path.exists(gt_path):
            rows.append({"photo": name, "set": cat, "error": "no ground truth"})
            continue
        gt = load(gt_path, size) & foot
        i, u = int((gt & pred).sum()), int((gt | pred).sum())
        row = {"photo": name, "set": cat, "gt_share": gt.sum() / fa, "pred_share": pred.sum() / fa,
               "iou": (i / u) if u else None, "inter": i, "union": u}
        if u:
            inter += i; union += u
        rows.append(row)
    else:
        share = pred.sum() / fa
        ctrl_area.append(share)
        ctrl_fp.append(share > 0.0005)
        rows.append({"photo": name, "set": cat, "pred_share": share, "false_positive": bool(share > 0.0005)})

fin = [r for r in rows if r["set"] == "fingers" and r.get("iou") is not None]
summary = {
    "label": a.label, "env": a.env,
    "finger_photos": len(fin),
    "finger_photos_with_finger_on_page": sum(1 for r in fin if r["gt_share"] > 0),
    "mean_iou": float(np.mean([r["iou"] for r in fin])) if fin else None,
    "pooled_iou": inter / union if union else None,
    "detected": sum(1 for r in fin if r["pred_share"] > 0),
    "control_photos": len(ctrl_fp),
    "control_fp_rate": float(np.mean(ctrl_fp)) if ctrl_fp else None,
    "control_mean_filled": float(np.mean(ctrl_area)) if ctrl_area else None,
    "errors": [r for r in rows if "error" in r],
}
json.dump({"summary": summary, "rows": rows}, open(os.path.join(a.out, a.label + ".json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, default=float), "errors:", len(summary["errors"]))
