"""cmp_rt.py A_DIR B_DIR: two emulator batch runs (emu_cmp.sh layouts, A = ORT, B = LiteRT): per page and run, net outputs
(npz) max / p99 |d|, page PNG max levels and % of pixels differing, JSON fields differing (without photo, nets,
pipeline_ms), wall s (times_phone.txt) and per-net ms."""
import json, os, sys
import cv2, numpy as np
A, B = sys.argv[1], sys.argv[2]
def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if path == "" and k in ("photo", "nets", "pipeline_ms"): continue
            walk(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (u, v) in enumerate(zip(a, b)): walk(u, v, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-9: out.append((path, abs(a - b)))
    elif a != b: out.append((path, None))
for run, names in (("st", ["photo1"]), ("du", ["1_1", "4_2", "17_2", "18_1"]), ("du2", ["1_1", "4_2", "17_2", "18_1"])):
    ta = json.load(open(os.path.join(A, run, "times_phone.txt"))); tb = json.load(open(os.path.join(B, run, "times_phone.txt")))
    for n in names:
        a, b = os.path.join(A, run, n), os.path.join(B, run, n)
        s = []
        for t in ("logits", "p1", "rc0", "rc2"):
            if not (os.path.exists(f"{a}_{t}.npz") and os.path.exists(f"{b}_{t}.npz")): s.append(f"{t} n/a"); continue
            za, zb = np.load(f"{a}_{t}.npz"), np.load(f"{b}_{t}.npz")
            d = np.concatenate([np.abs(za[k].astype(np.float64) - zb[k]).ravel() for k in za.files if za[k].shape == zb[k].shape])
            s.append(f"{t} {d.max():.2g}/{np.percentile(d, 99):.2g}")
        x, y = cv2.imread(a + ".png"), cv2.imread(b + ".png")
        if x.shape == y.shape:
            dd = np.abs(x.astype(int) - y); pg = f"page max {dd.max()} lv, {np.mean(np.any(dd > 0, -1)) * 100:.2f}% px, >8 lv {np.mean(np.any(dd > 8, -1)) * 100:.2f}%"
        else: pg = f"page size {x.shape[:2]} vs {y.shape[:2]}"
        ja, jb = json.load(open(a + ".json")), json.load(open(b + ".json"))
        diff = []; walk(ja, jb, "", diff)
        nets = lambda j: " ".join(f"{k}={v['ms'] / 1000:.1f}" for k, v in j["nets"].items() if isinstance(v, dict) and "ms" in v)
        print(f"{run}/{n}: {'; '.join(s)} | {pg} | json {len(diff)} diff {[p for p, _ in diff[:4]]} | wall {ta[n]['wall'] / 1000:.1f} vs {tb[n]['wall'] / 1000:.1f} s")
        print(f"    ORT  {nets(ja)}\n    LRT  {nets(jb)}")
