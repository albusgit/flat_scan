"""Gate B (docs/accel/PLAN.md): pipeline_cli with the nets in-process (core/src/infer.cpp, ONNX Runtime
C++ 1.22.0) against the Python ONNX Runtime path it replaces, photo by photo.

  python -I gate_b.py ref SET OUT --cli BASE_CLI --pf-cli BASE_PF_CLI [--shade] [--jobs N]
  python -I gate_b.py new SET OUT --cli NEW_CLI [--shade] [--backend B] [--jobs N]
  python -I gate_b.py cmp SET REF_OUT NEW_OUT [--images DIR]     (a Python with numpy + cv2)
  --only A,B: just these photos

SET: s35 (the 34 round-4 photos + the hand-held receipt) or docunet (the 130 DocUNet crops).
ref: the Python path, run with this interpreter's onnxruntime (use the 1.22.0 venv, the app's
     version) and a pipeline_cli built from accel/main: page finder (pagefind_cli input -> ORT),
     UVDoc on the whole photo (FS_UVIN FS_FRAME_CORNERS), UVDoc and SyntheticDoc-HR on its re-crop
     (FS_UVPASS1 FS_UVRECROP), then pipeline_cli with the four outputs, as docunet_bench/run_ours.py
     (NETS=nets_rc_f) and the 35-photo harness made them; --shade adds ShadeNet (FS_SHADE_NET,
     the run_onnx.py subprocess, FS_SHADE_AFTER as the app).
new: pipeline_cli PHOTO OUT net with FS_NETS=1 FS_NET_DUMP (everything in-process).
cmp: the net outputs bit for bit (logits, p1, rc0, rc2), the output pages pixel for pixel and the
     JSON (without the new "nets" / "pipeline_ms" keys); --images DIR: also the pages against an
     earlier run's DIR/<name>.png|.jpg (e.g. docunet_bench/work/new_fold2).
DocUNet runs use FS_NO_SHADOWS=1 and write PNG, as run_ours.py."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

S35_DIR = r"C:\Users\User\Desktop\round4_robustness\real_photos"
SCRATCH = r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-flat-scan\922e3e9d-4193-49a6-b4b4-d3421c7157d5\scratchpad"
RECEIPT = os.path.join(SCRATCH, "rc", "photo.jpg")   # the hand-held receipt (round 5 "user" photo)
DOCUNET = r"C:\Users\User\Desktop\docunet_bench\data_crop\crop"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODELS = os.path.join(REPO, "app", "src", "main", "assets", "models")
OPENCV_BIN = r"C:\Users\User\Android\opencv\build\x64\vc16\bin"


def photo_set(name):
    if name == "s35":
        out = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(os.path.join(S35_DIR, "*.jpg")))]
        return out + [("500c2810-image", RECEIPT)]
    if name == "docunet":
        return [(f"{k}_{m}", os.path.join(DOCUNET, f"{k}_{m} copy.png")) for k in range(1, 66) for m in (1, 2)]
    raise SystemExit("unknown set " + name)


def env_base(a):
    e = dict(os.environ)
    e["PATH"] = OPENCV_BIN + ";" + e["PATH"]
    if a.set == "docunet":
        e["FS_NO_SHADOWS"] = "1"
    return e


_sess = {}


def net(model, x):
    import onnxruntime as ort
    if model not in _sess:
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        _sess[model] = ort.InferenceSession(os.path.join(MODELS, model), o, providers=["CPUExecutionProvider"])
    s = _sess[model]
    return s.run(None, {s.get_inputs()[0].name: x})


def ext(a):
    return ".png" if a.set == "docunet" else ".jpg"


def ref_one(job):
    a, n, photo = job
    d = a.out
    base = os.path.join(d, n)
    lg = base + "_logits.npz"
    env = env_base(a)
    t = {}
    if not os.path.exists(lg):
        xp = base + "_pfx.npz"
        subprocess.run([a.pf_cli, "input", photo, xp], check=True, capture_output=True, env=env)
        t0 = time.perf_counter()
        y = net("pagefinder.onnx", np.load(xp)["x"])
        t["page_finder"] = 1000 * (time.perf_counter() - t0)
        np.savez(lg, logits=y[0][0].astype(np.float32))
        os.remove(xp)
    p1 = dict(FS_UVPASS1=base + "_p1.npz", FS_UVPASS1_FRAME="1", FS_UVRECROP="1")
    for tag, model, onnx, extra in (("p1", 0, "uvdoc.onnx", dict(FS_FRAME_CORNERS="1")), ("rc0", 0, "uvdoc.onnx", p1),
                                    ("rc2", 2, "syntheticdoc_hr.onnx", p1)):
        out = f"{base}_{tag}.npz"
        if os.path.exists(out):
            continue
        xin = f"{base}_{tag}_in.npz"
        e = dict(env, FS_UVIN=xin, FS_UVMODEL=str(model), **extra)
        subprocess.run([a.cli, photo, "-", lg], env=e, capture_output=True)
        if not os.path.exists(xin):
            break
        t0 = time.perf_counter()
        ys = net(onnx, np.load(xin)["x"])
        t[tag] = 1000 * (time.perf_counter() - t0)
        np.savez(out, out=ys[0][0].astype(np.float32))
        os.remove(xin)
    e = dict(env)
    if os.path.exists(base + "_p1.npz"):
        e.update(p1)
    if os.path.exists(base + "_rc2.npz"):
        e.update(FS_UVOUT=base + "_rc2.npz", FS_UVMODEL="2")
        if os.path.exists(base + "_rc0.npz"):
            e.update(FS_UVOUT2=base + "_rc0.npz", FS_UVMODEL2="0")
    elif os.path.exists(base + "_rc0.npz"):
        e.update(FS_UVOUT=base + "_rc0.npz", FS_UVMODEL="0")
    if a.shade:
        e.update(FS_SHADE_NET=os.path.join(MODELS, "shadenet.onnx"), FS_SHADE_AFTER="1", FS_PYTHON=sys.executable)
    t0 = time.perf_counter()
    r = subprocess.run([a.cli, photo, base + ext(a), lg], env=e, capture_output=True, text=True)
    t["pipeline_cli"] = 1000 * (time.perf_counter() - t0)
    with open(base + ".json", "w") as f:
        f.write(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({"ok": False, "error": r.stderr[-300:]}))
    return n, t


def new_one(job):
    a, n, photo = job
    base = os.path.join(a.out, n)
    e = dict(env_base(a), FS_NETS="1", FS_NET_DUMP=base)
    if a.shade:
        e.update(FS_SHADE_NET=os.path.join(MODELS, "shadenet.onnx"), FS_SHADE_AFTER="1")
    t0 = time.perf_counter()
    r = subprocess.run([a.cli, "--backend", a.backend, photo, base + ext(a), "net"], env=e, capture_output=True, text=True)
    wall = 1000 * (time.perf_counter() - t0)
    with open(base + ".json", "w") as f:
        f.write(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({"ok": False, "error": r.stderr[-300:]}))
    return n, {"wall": wall}


def same_npz(p, q, key):
    if not os.path.exists(p) and not os.path.exists(q):
        return "absent"
    if not (os.path.exists(p) and os.path.exists(q)):
        return "missing " + ("new" if os.path.exists(p) else "ref")
    x, y = np.load(p)[key], np.load(q)[key]
    if x.shape != y.shape:
        return f"shape {x.shape} vs {y.shape}"
    if x.view(np.uint32).tobytes() == y.view(np.uint32).tobytes():
        return "same"
    return f"differs (max |d| {np.abs(x.astype(np.float64) - y).max():.3g})"


def same_img(p, q):
    import cv2
    if not os.path.exists(p) and not os.path.exists(q):
        return "absent"
    if not (os.path.exists(p) and os.path.exists(q)):
        return "missing"
    x, y = cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.imread(q, cv2.IMREAD_UNCHANGED)
    if x.shape != y.shape:
        return f"size {x.shape} vs {y.shape}"
    d = np.abs(x.astype(np.int16) - y).max()
    return "same" if d == 0 else f"differs (max {d}, {np.count_nonzero(np.any(x != y, axis=-1))} px)"


def json_core(p):
    if not os.path.exists(p):
        return None
    j = json.loads(open(p).read().replace("\\", "/"))
    for k in ("nets", "pipeline_ms", "photo"):
        j.pop(k, None)
    return j


def cmp(a):
    rows, bad = [], 0
    for n, _ in photo_set(a.set):
        r, w = os.path.join(a.ref, n), os.path.join(a.new, n)
        row = {k: same_npz(f"{r}_{k}.npz", f"{w}_{k}.npz", "logits" if k == "logits" else "out") for k in ("logits", "p1", "rc0", "rc2")}
        row["page"] = same_img(r + ext(a), w + ext(a))
        jr, jw = json_core(r + ".json"), json_core(w + ".json")
        row["json"] = "same" if jr == jw else "differs: " + ", ".join(sorted(k for k in set(jr or {}) | set(jw or {}) if (jr or {}).get(k) != (jw or {}).get(k)))
        if a.images:
            row["vs_images"] = same_img(os.path.join(a.images, n + ext(a)), w + ext(a))
        ok = all(v in ("same", "absent") for v in row.values())
        bad += not ok
        rows.append((n, row))
        print(f"{n:34s} " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    keys = rows[0][1].keys()
    print("\nsummary (same / absent / other):")
    for k in keys:
        vals = [row[k] for _, row in rows]
        print(f"  {k:10s} {sum(v == 'same' for v in vals):4d} / {sum(v == 'absent' for v in vals):3d} / {sum(v not in ('same', 'absent') for v in vals):3d}")
    print(f"photos with any difference: {bad} of {len(rows)}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ref", "new", "cmp"])
    ap.add_argument("set")
    ap.add_argument("out")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--cli")
    ap.add_argument("--pf-cli")
    ap.add_argument("--shade", action="store_true")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--images")
    ap.add_argument("--only", help="comma-separated photo names")
    a = ap.parse_args()
    if a.only:
        keep = set(a.only.split(","))
        full = photo_set
        globals()["photo_set"] = lambda name: [x for x in full(name) if x[0] in keep]
    if a.cmd == "cmp":
        a.ref = a.out
        sys.exit(1 if cmp(a) else 0)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a, n, p) for n, p in photo_set(a.set)]
    fn = ref_one if a.cmd == "ref" else new_one
    times = {}
    t0 = time.perf_counter()
    if a.jobs <= 1:
        res = map(fn, jobs)
    else:
        pool = Pool(a.jobs)
        res = pool.imap_unordered(fn, jobs)
    for n, t in res:
        times[n] = t
    print(f"{a.cmd} {a.set}: {len(jobs)} photos in {time.perf_counter() - t0:.0f} s")
    json.dump(times, open(os.path.join(a.out, f"times_{a.cmd}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
