"""FlatScan test-library harness: run a desktop pipeline_cli build over library sets, score, make contact sheets.

  python -I run_library.py run   --cli PIPELINE_CLI.exe --run NAME --sets dir300,inv3dreal [options]
  python -I run_library.py score --run NAME --sets dir300 [--metrics geom,straight,ocr]
  python -I run_library.py sheets --run NAME --sets dir300
  python -I run_library.py compare RUN_A RUN_B [--sets ...]
  python -I run_library.py list

Sets are E:/fs_datasets/sets/<set>.json (prepare_sets.py); outputs go to E:/fs_datasets/runs/<NAME>/<set>/:
  pages/<id>.png|jpg      the flattened page (the app's colour page, before filters)
  pages/<id>.*.auto.jpg   with --render: the page as the app shows it (Auto colour 0.7)
  pipeline.json           pipeline_cli's JSON per item (+ "sec", "cmd_env")
  metrics.json            per-item metrics + means; summary in E:/fs_datasets/runs/<NAME>/summary.md
  sheets/<set>_NN.jpg     contact sheets: photo | page | ground truth (+ metrics)

run options:
  --cli PATH          pipeline_cli.exe of the build under test (required for run)
  --mode app|frame    app: page finder + nets as the app runs them (FS_NETS=1, LOGITS "net");
                      frame: the photo frame is the page (FS_FRAME_CORNERS=1 FS_NO_ORIENT=1), for
                      page-cropped sets. Default: the set's frame_corners flag decides.
  --env K=V ...       extra environment for pipeline_cli (e.g. FS_NO_LINES=1, FS_GPU=1, FS_SHADE_NET=...)
  --limit N / --every K / --ids a,b / --tags t1,t2   subsets (deterministic)
  --jobs N            parallel pipeline_cli processes (default 4, FS_ORT_THREADS=2 each)
  --render            also write the Auto colour render (FS_RENDER=1) for filter review
  --no-score / --no-sheets, --force (recompute existing pages)
  --no-lock           do not take the heavy lock (only if the caller holds it)
Scoring (score, or run without --no-score):
  geom      LD / MS-SSIM / AD (docunet_bench/docunet_eval.py, same protocol as DocUNet) on flat_gt sets.
            A photo the pipeline could not flatten is scored as the unflattened photo (counted in n_fail),
            so runs that fail more do not look better.
  straight  text-row straightness (straight.py) on every page; on flat_gt sets the GT's own value too.
  ocr       Tesseract 5 (docunet_bench/tess): CER vs the OCR of the GT scan (DocGeoNet protocol) on flat_gt
            sets, and order-free word recall vs true words where the set has them.
Heavy lock: C:/Users/User/Desktop/docunet_bench/.heavy_lock is taken around the pipeline run and around
scoring (one scorer at a time: two SIFT-flow scorers ran out of RAM before)."""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "E:/fs_datasets"
BENCH = r"C:\Users\User\Desktop\docunet_bench"
LOCK = BENCH + r"\.heavy_lock"
OPENCV_BIN = r"C:\Users\User\Android\opencv\build\x64\vc16\bin"
TESS = os.path.join(BENCH, "tess", "bin", "tesseract.exe")
sys.path.insert(0, HERE)


# ------------------------------------------------------------------ lock
@contextmanager
def heavy_lock(tag, enabled=True):
    if not enabled:
        yield
        return
    waited = False
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            own = os.path.join(LOCK, "owner")
            try:
                age = time.time() - os.path.getmtime(own)
                if age > 2 * 3600:
                    print(f"[lock] stale lock ({age / 3600:.1f} h): {open(own).read().strip()} -> removed", flush=True)
                    shutil.rmtree(LOCK, ignore_errors=True)
                    continue
            except OSError:
                pass
            if not waited:
                print("[lock] waiting for the heavy lock ...", flush=True)
                waited = True
            time.sleep(20)
    with open(os.path.join(LOCK, "owner"), "w") as f:
        f.write(f"library {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}\n")
    try:
        yield
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)


# ------------------------------------------------------------------ sets
def load_set(name):
    return json.load(open(os.path.join(ROOT, "sets", name + ".json")))


def select(items, a):
    if a.ids:
        want = set(a.ids.split(","))
        items = [it for it in items if it["id"] in want]
    if a.tags:
        want = set(a.tags.split(","))
        items = [it for it in items if want & set(it.get("tags", []))]
    if a.every and a.every > 1:
        items = items[::a.every]
    if a.limit:
        items = items[:a.limit]
    return items


def run_dir(run, set_name):
    return os.path.join(ROOT, "runs", run, set_name)


# ------------------------------------------------------------------ pipeline
def one(args):
    cli, item, out, env = args
    t = time.time()
    try:
        r = subprocess.run([cli, item["photo"], out, "net"], env=env, capture_output=True, text=True, timeout=600)
        line = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
        j = json.loads(line[-1].replace("\\", "/")) if line else {"ok": False, "error": (r.stdout + r.stderr)[-400:]}
    except subprocess.TimeoutExpired:
        j = {"ok": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        j = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    j["sec"] = round(time.time() - t, 2)
    j.pop("photo", None)
    return item["id"], j


def page_path(d, item, kind):
    return os.path.join(d, "pages", item["id"] + (".png" if kind == "flat_gt" else ".jpg"))


def cmd_run(a):
    cli = os.path.abspath(a.cli)
    if not os.path.exists(cli):
        sys.exit(f"no pipeline_cli at {cli}")
    for set_name in a.sets.split(","):
        S = load_set(set_name)
        items = select(S["items"], a)
        d = run_dir(a.run, set_name)
        os.makedirs(os.path.join(d, "pages"), exist_ok=True)
        frame = S.get("frame_corners", False) if a.mode == "auto" else a.mode == "frame"
        env = dict(os.environ)
        env["PATH"] = OPENCV_BIN + ";" + os.path.dirname(cli) + ";" + env["PATH"]
        env.update(FS_NETS="1", FS_ORT_THREADS=str(a.threads))
        if frame:
            env.update(FS_FRAME_CORNERS="1", FS_NO_ORIENT="1")
        if a.render:
            env["FS_RENDER"] = "1"
        extra = dict(e.split("=", 1) for e in a.env)
        env.update(extra)
        pj = os.path.join(d, "pipeline.json")
        res = json.load(open(pj)) if os.path.exists(pj) and not a.force else {}
        todo = [it for it in items if a.force or it["id"] not in res or not os.path.exists(page_path(d, it, S["kind"])) and res[it["id"]].get("ok")]
        print(f"[run] {set_name}: {len(items)} items, {len(todo)} to run, mode {'frame' if frame else 'app'}, cli {cli}", flush=True)
        t0 = time.time()
        for c0 in range(0, len(todo), a.chunk):   # the heavy lock is released between chunks (<= 40 min each)
            chunk = todo[c0:c0 + a.chunk]
            with heavy_lock(f"run {a.run}/{set_name} {c0}", not a.no_lock):
                with ThreadPoolExecutor(a.jobs) as ex:
                    for k, (iid, j) in enumerate(ex.map(one, [(cli, it, page_path(d, it, S["kind"]), env) for it in chunk])):
                        res[iid] = j
                        if (c0 + k + 1) % 25 == 0:
                            print(f"  {c0 + k + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
                            json.dump(res, open(pj, "w"), indent=1)
            json.dump(res, open(pj, "w"), indent=1)
        meta = dict(cli=cli, mode="frame" if frame else "app", env=extra, threads=a.threads, render=a.render,
                    when=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(res, open(pj, "w"), indent=1)
        json.dump(meta, open(os.path.join(d, "run_meta.json"), "w"), indent=1)
        nok = sum(1 for it in items if res.get(it["id"], {}).get("ok"))
        secs = [res[it["id"]]["sec"] for it in items if it["id"] in res]
        print(f"[run] {set_name}: ok {nok}/{len(items)}, median {np.median(secs):.1f} s/photo", flush=True)
    if not a.no_score:
        cmd_score(a)
    if not a.no_sheets:
        cmd_sheets(a)


# ------------------------------------------------------------------ scoring
def geom_job(args):
    gt, pred = args
    sys.path.insert(0, BENCH)
    import docunet_eval as DE  # noqa: E402
    try:
        return DE.eval_pair((gt, pred))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def tess(path, cache):
    if cache and os.path.exists(cache):
        return open(cache, encoding="utf8").read()
    r = subprocess.run([TESS, path, "stdout"], capture_output=True)
    t = r.stdout.decode("utf8", errors="replace").replace("\r\n", "\n")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        open(cache, "w", encoding="utf8").write(t)
    return t


def lev(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    bb = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    prev = np.arange(len(b) + 1, dtype=np.int64)
    j = np.arange(len(b) + 1, dtype=np.int64)
    for i, ch in enumerate(a, 1):
        sub = prev[:-1] + (bb != ord(ch))
        cand = np.empty_like(prev)
        cand[0] = i
        cand[1:] = np.minimum(prev[1:] + 1, sub)
        prev = np.minimum.accumulate(cand - j) + j
    return int(prev[-1])


def norm_words(t):
    return [w for w in re.sub(r"[^\w$%.,:/#@-]+", " ", t.lower()).split() if any(c.isalnum() for c in w)]


def true_words(path):
    w = json.load(open(path))
    if isinstance(w, list) and w and isinstance(w[0], dict):
        w = [x.get("text", "") for x in w]
    return norm_words(" ".join(w))


def ocr_job(args):
    set_name, item, pred, cache_dir = args
    out = {}
    pt = tess(pred, os.path.join(cache_dir, item["id"] + ".txt"))
    if item.get("gt"):
        gid = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.relpath(item["gt"], ROOT))
        gt_t = tess(item["gt"], os.path.join(ROOT, "cache", "ocr_gt", gid + ".txt"))
        a_, b_ = " ".join(gt_t.split()), " ".join(pt.split())
        out["ed"] = lev(b_, a_)
        out["gt_chars"] = len(a_)
        if len(a_) >= 50:   # pages whose scan OCR has < 50 characters (pictures, blank) give meaningless CER
            out["cer"] = out["ed"] / len(a_)
    if item.get("words"):
        tw = collections.Counter(true_words(item["words"]))
        pw = collections.Counter(norm_words(pt))
        out["word_recall"] = sum((tw & pw).values()) / max(1, sum(tw.values()))
    return item["id"], out


def straight_job(path):
    import straight
    try:
        return straight.score(path)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def mean_of(per, key, ids=None):
    v = [per[i][key] for i in (ids or per) if i in per and isinstance(per[i].get(key), (int, float))]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def cmd_score(a):
    metrics = a.metrics.split(",")
    for set_name in a.sets.split(","):
        S = load_set(set_name)
        items = select(S["items"], a)
        d = run_dir(a.run, set_name)
        pj = json.load(open(os.path.join(d, "pipeline.json")))
        mpath = os.path.join(d, "metrics.json")
        M = json.load(open(mpath)) if os.path.exists(mpath) else {"per_item": {}}
        per = M["per_item"]
        items = [it for it in items if it["id"] in pj]
        for it in items:
            per.setdefault(it["id"], {})["ok"] = bool(pj[it["id"]].get("ok"))
        flat = S["kind"] == "flat_gt"

        def pred_of(it):
            p = page_path(d, it, S["kind"])
            return p if os.path.exists(p) else None
        if "geom" in metrics and flat:
            todo = [it for it in items if a.force_score or "ld" not in per[it["id"]]]
            print(f"[score] {set_name}: geometry on {len(todo)} pages", flush=True)
            t0 = time.time()
            for c0 in range(0, len(todo), a.chunk):   # the lock is released between chunks
                chunk = todo[c0:c0 + a.chunk]
                jobs = [(it["gt"], pred_of(it) or it["photo"]) for it in chunk]
                with heavy_lock(f"score {a.run}/{set_name} {c0}", not a.no_lock):
                    with ProcessPoolExecutor(a.score_jobs) as ex:
                        for k, (it, r) in enumerate(zip(chunk, ex.map(geom_job, jobs))):
                            per[it["id"]].update({kk: r[kk] for kk in ("ms", "ld", "ad", "error") if kk in r})
                            per[it["id"]]["geom_on_photo"] = pred_of(it) is None
                            if (c0 + k + 1) % 25 == 0:
                                print(f"  {c0 + k + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
                                json.dump(M, open(mpath, "w"), indent=1)
                json.dump(M, open(mpath, "w"), indent=1)
        with heavy_lock(f"score {a.run}/{set_name} text", not a.no_lock):
            if "straight" in metrics:
                todo = [it for it in items if pred_of(it) and (a.force_score or "rows" not in per[it["id"]])]
                print(f"[score] {set_name}: straightness on {len(todo)} pages", flush=True)
                with ProcessPoolExecutor(a.score_jobs) as ex:
                    for it, r in zip(todo, ex.map(straight_job, [pred_of(it) for it in todo])):
                        per[it["id"]].update({k: v for k, v in (r or {}).items()})
                if flat:   # the GT pages' own straightness (the floor), cached per set
                    gpath = os.path.join(ROOT, "cache", "straight_gt", set_name + ".json")
                    G = json.load(open(gpath)) if os.path.exists(gpath) else {}
                    gts = sorted({it["gt"] for it in items if it["gt"] not in G})
                    if gts:
                        with ProcessPoolExecutor(a.score_jobs) as ex:
                            for g, r in zip(gts, ex.map(straight_job, gts)):
                                G[g] = r
                        os.makedirs(os.path.dirname(gpath), exist_ok=True)
                        json.dump(G, open(gpath, "w"), indent=1)
                    for it in items:
                        g = G.get(it["gt"]) or {}
                        if "row_dev_p90" in g:
                            per[it["id"]]["gt_row_dev_p90"] = g["row_dev_p90"]
                            per[it["id"]]["gt_angle_spread"] = g.get("angle_spread")
            if "ocr" in metrics and os.path.exists(TESS):
                todo = [it for it in items if pred_of(it) and (it.get("gt") or it.get("words"))
                        and (a.force_score or ("cer" not in per[it["id"]] and "word_recall" not in per[it["id"]]))]
                print(f"[score] {set_name}: OCR on {len(todo)} pages", flush=True)
                cache = os.path.join(d, "ocr")
                with ProcessPoolExecutor(a.score_jobs * 2) as ex:
                    for iid, r in ex.map(ocr_job, [(set_name, it, pred_of(it), cache) for it in todo]):
                        per[iid].update(r)
        ids = [it["id"] for it in items]
        summ = {"n": len(ids), "n_fail": sum(1 for i in ids if not per[i].get("ok"))}
        for k in ("ms", "ld", "ad", "row_dev_p90", "row_dev_med", "angle_spread", "gt_row_dev_p90", "cer", "word_recall"):
            if k == "cer":
                for i in ids:   # items scored before the 50-character rule
                    if per[i].get("gt_chars", 99) < 50:
                        per[i].pop("cer", None)
            m, n = mean_of(per, k, ids)
            if n:
                summ[k] = m
                summ["n_" + k] = n
        eds = [(per[i]["ed"], per[i]["gt_chars"]) for i in ids if "gt_chars" in per[i] and per[i]["gt_chars"] >= 50]
        if eds:   # micro CER: total edit distance / total GT characters
            summ["cer_micro"] = sum(e for e, _ in eds) / sum(n for _, n in eds)
        sec = [pj[i].get("sec") for i in ids if pj[i].get("sec")]
        if sec:
            summ["sec_median"] = float(np.median(sec))
        M["summary"] = summ
        M["set"] = set_name
        json.dump(M, open(mpath, "w"), indent=1)
        print(f"[score] {set_name}: " + fmt_summary(summ), flush=True)
    write_summary(a.run)


def fmt_summary(s):
    parts = [f"n={s['n']} fail={s['n_fail']}"]
    for k, f in (("ms", "MS-SSIM {:.4f}"), ("ld", "LD {:.3f}"), ("ad", "AD {:.4f}"), ("cer", "CER {:.4f}"), ("cer_micro", "CER micro {:.4f}"),
                 ("word_recall", "word recall {:.4f}"), ("row_dev_p90", "row dev p90 {:.2f} px"),
                 ("angle_spread", "angle spread {:.2f} deg"), ("gt_row_dev_p90", "(GT row dev p90 {:.2f})")):
        if s.get(k) is not None:
            parts.append(f.format(s[k]))
    return ", ".join(parts)


def write_summary(run):
    rd = os.path.join(ROOT, "runs", run)
    lines = [f"# Library run {run}", ""]
    for mp in sorted(glob.glob(os.path.join(rd, "*", "metrics.json"))):
        M = json.load(open(mp))
        meta_p = os.path.join(os.path.dirname(mp), "run_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        lines.append(f"- **{M.get('set')}** ({meta.get('mode', '?')}, env {meta.get('env', {})}): " + fmt_summary(M.get("summary", {"n": 0, "n_fail": 0})))
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ contact sheets
def thumb(path, h, label=None):
    im = cv2.imread(path) if path and os.path.exists(path) else None
    if im is None:
        im = np.full((h, int(h * 0.75), 3), 200, np.uint8)
        cv2.putText(im, "none", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    s = h / im.shape[0]
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
    if im.shape[1] > int(h * 1.6):
        im = cv2.resize(im, (int(h * 1.6), h), interpolation=cv2.INTER_AREA)
    return im


def cmd_sheets(a):
    for set_name in a.sets.split(","):
        S = load_set(set_name)
        items = select(S["items"], a)
        d = run_dir(a.run, set_name)
        pj = json.load(open(os.path.join(d, "pipeline.json")))
        mp = os.path.join(d, "metrics.json")
        per = json.load(open(mp))["per_item"] if os.path.exists(mp) else {}
        items = [it for it in items if it["id"] in pj]
        os.makedirs(os.path.join(d, "sheets"), exist_ok=True)
        H, rows_per, W = 300, 6, 1800
        rows = []
        for it in items:
            pg = page_path(d, it, S["kind"])
            tiles = [thumb(it["photo"], H), thumb(pg, H)]
            if a.render or os.path.exists(pg + ".auto.jpg"):
                tiles.append(thumb(pg + ".auto.jpg", H))
            if it.get("gt"):
                tiles.append(thumb(it["gt"], H))
            row = np.full((H + 34, W, 3), 255, np.uint8)
            x = 0
            for t in tiles:
                w = min(t.shape[1], W - x)
                if w <= 0:
                    break
                row[34:, x:x + w] = t[:, :w]
                x += w + 8
            m = per.get(it["id"], {})
            j = pj[it["id"]]
            txt = f"{it['id']}  {'ok' if j.get('ok') else 'FAIL ' + str(j.get('error', ''))[:40]}  mode {j.get('mode', '')}"
            for k, f in (("ld", " LD {:.2f}"), ("ms", " MS {:.3f}"), ("cer", " CER {:.3f}"), ("row_dev_p90", " rowdev {:.2f}"), ("angle_spread", " ang {:.2f}")):
                if isinstance(m.get(k), (int, float)):
                    txt += f.format(m[k])
            cv2.putText(row, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0) if j.get("ok") else (0, 0, 255), 2)
            rows.append(row)
        for s in range(0, len(rows), rows_per):
            cv2.imwrite(os.path.join(d, "sheets", f"{set_name}_{s // rows_per:03d}.jpg"), np.vstack(rows[s:s + rows_per]),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[sheets] {set_name}: {(len(rows) + rows_per - 1) // rows_per} sheets in {os.path.join(d, 'sheets')}", flush=True)


# ------------------------------------------------------------------ compare
def cmd_compare(a):
    ra, rb = a.run_a, a.run_b
    sets = a.sets.split(",") if a.sets else sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "runs", ra, "*", "metrics.json")))
    for s in sets:
        pa, pb = (os.path.join(ROOT, "runs", r, s, "metrics.json") for r in (ra, rb))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(s, "missing metrics in one run")
            continue
        A, B = json.load(open(pa))["per_item"], json.load(open(pb))["per_item"]
        common = sorted(set(A) & set(B))
        print(f"== {s}: {len(common)} common items")
        for k, better_low in (("ld", True), ("ms", False), ("ad", True), ("cer", True), ("word_recall", False), ("row_dev_p90", True), ("angle_spread", True)):
            ids = [i for i in common if isinstance(A[i].get(k), (int, float)) and isinstance(B[i].get(k), (int, float))]
            if not ids:
                continue
            va, vb = np.array([A[i][k] for i in ids]), np.array([B[i][k] for i in ids])
            diff = vb - va
            tol = {"ld": 0.05, "ms": 0.002, "ad": 0.002, "cer": 0.005, "word_recall": 0.005, "row_dev_p90": 0.1, "angle_spread": 0.05}[k]
            better = int(((diff < -tol) if better_low else (diff > tol)).sum())
            worse = int(((diff > tol) if better_low else (diff < -tol)).sum())
            order = np.argsort(diff if better_low else -diff)
            worst = [f"{ids[i]} {va[i]:.3g}->{vb[i]:.3g}" for i in order[::-1][:3]]
            print(f"  {k:12s} A {va.mean():.4f}  B {vb.mean():.4f}  (n={len(ids)})  B better {better} / worse {worse};  worst for B: {', '.join(worst)}")


def cmd_list(_a):
    for p in sorted(glob.glob(os.path.join(ROOT, "sets", "*.json"))):
        S = json.load(open(p))
        tags = collections.Counter(t for it in S["items"] for t in it.get("tags", [])[:1])
        print(f"{S['name']:22s} {S['kind']:9s} n={len(S['items']):5d} frame={S.get('frame_corners')} use={S.get('use')}  {dict(list(tags.items())[:8])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--run", required=True)
        p.add_argument("--sets", required=True)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--every", type=int, default=0)
        p.add_argument("--ids", default="")
        p.add_argument("--tags", default="")
        p.add_argument("--no-lock", action="store_true")
        p.add_argument("--metrics", default="geom,straight,ocr")
        p.add_argument("--score-jobs", type=int, default=4)
        p.add_argument("--force-score", action="store_true")
        p.add_argument("--render", action="store_true")
        p.add_argument("--chunk", type=int, default=100, help="items per heavy-lock hold")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--cli", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "app", "frame"])
    p.add_argument("--env", nargs="*", default=[])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--no-sheets", action="store_true")
    p = sub.add_parser("score")
    common(p)
    p = sub.add_parser("sheets")
    common(p)
    p = sub.add_parser("compare")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--sets", default="")
    sub.add_parser("list")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "sheets": cmd_sheets, "compare": cmd_compare, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    main()
