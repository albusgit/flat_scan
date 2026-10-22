"""cmp_work.py OUTCFGDIR WORKRUN: merged pipeline_cli DocUNet pages (OUTCFGDIR/du/<n>.png, .json) vs docunet_bench/work/WORKRUN
(<n>.png, pipeline.json). Pages pixel for pixel, JSON without photo / nets / pipeline_ms."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import DU_NAMES, DROP, load

d, run = sys.argv[1], sys.argv[2]
W = r"C:\Users\User\Desktop\docunet_bench\work"
ref = json.load(open(os.path.join(W, run, "pipeline.json")))
si = sj = n = 0
for k in DU_NAMES:
    p = os.path.join(d, "du", k)
    if not os.path.exists(p + ".json"):
        continue
    n += 1
    a, b = cv2.imread(p + ".png", cv2.IMREAD_UNCHANGED), cv2.imread(os.path.join(W, run, k + ".png"), cv2.IMREAD_UNCHANGED)
    same = a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)
    ja, jb = load(p + ".json"), ref.get(k, {})
    keys = (set(ja) & set(jb)) - DROP
    jd = [x for x in sorted(keys) if ja[x] != jb[x]]
    si += same; sj += not jd
    if not same or jd:
        md = ""
        if a is not None and b is not None and a.shape == b.shape:
            dd = np.abs(a.astype(int) - b.astype(int)); md = f"max |d| {dd.max()}, {100 * (dd.max(axis=2) > 0).mean():.2f}% px"
        print(f"  {k}: page {'same' if same else 'DIFFERS ' + md}; json differs {jd[:6]}")
print(f"{d} vs work/{run}: pages identical {si}/{n}, JSON identical {sj}/{n}; keys only in ref {sorted(set(jb) - set(ja))}, only new {sorted(set(ja) - set(jb) - DROP)}")
