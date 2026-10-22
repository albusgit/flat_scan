"""Merge verification: pipeline_cli builds (Round 10, r11/main before the merge, merged) on the 34 round-4 photos
and DocUNet pages, cached net outputs, CPU.
  python -I verify.py run OUT CFG[,CFG...] [--jobs 4] [--only s35|du]
  python -I verify.py cmp OUT A B [--only s35|du]
"""
import glob, json, os, subprocess, sys
from multiprocessing import Pool

PY = r"C:\Users\User\Desktop\round4_robustness\.venv\Scripts\python.exe"
SCR = r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-flat-scan\922e3e9d-4193-49a6-b4b4-d3421c7157d5\scratchpad"
R10 = r"C:\Users\User\Desktop\fs_round10\tools\desktop_check\build\pipeline_cli.exe"
PRE = SCR + r"\merge\build_pre\pipeline_cli.exe"
NEW = r"C:\Users\User\Desktop\fs_r11\tools\desktop_check\build\pipeline_cli.exe"
RULE = r"C:\Users\User\Desktop\fs_round10\app\src\main\assets\models\rulenet.onnx"
S35 = r"C:\Users\User\Desktop\round4_robustness\real_photos"
S35N = SCR + r"\accel_b\s35_ref"
DU = r"C:\Users\User\Desktop\docunet_bench\data_crop\crop"
W = r"C:\Users\User\Desktop\docunet_bench\work"
DU_NAMES = [f"{k}_{m}" for k in range(1, 11) for m in (1, 2)] + ["52_1", "53_2", "52_2", "13_1", "51_1"]

CFGS = {
    "r10_rn": (R10, {"FS_RULE_NET": RULE, "FS_PYTHON": PY}),
    "m_rn": (NEW, {"FS_RULE_NET": "1"}),
    "m_rnpy": (NEW, {"FS_RULE_NET": "1", "FS_RULE_NET_PY": "1", "FS_PYTHON": PY}),
    "r10_off": (R10, {}),
    "m_off": (NEW, {}),
    "pre_off": (PRE, {}),
    "pre_nl": (PRE, {"FS_NO_LINES": "1"}),
    "m_nl": (NEW, {"FS_NO_LINES": "1"}),
    "m_nets_rn": (NEW, {"FS_RULE_NET": "1", "FS_NETS": "1"}),
    "pre_nets": (PRE, {"FS_NETS": "1", "FS_NO_LINES": "1"}),
    "m_nets_nl": (NEW, {"FS_NETS": "1", "FS_NO_LINES": "1"}),
}


def jobs(only):
    out = []
    if only in (None, "s35"):
        for p in sorted(glob.glob(os.path.join(S35, "*.jpg"))):
            n = os.path.basename(p)[:-4]
            env = {"FS_UVPASS1": f"{S35N}/{n}_p1.npz", "FS_UVPASS1_FRAME": "1", "FS_UVRECROP": "1",
                   "FS_UVOUT": f"{S35N}/{n}_rc2.npz", "FS_UVMODEL": "2", "FS_UVOUT2": f"{S35N}/{n}_rc0.npz", "FS_UVMODEL2": "0"}
            out.append(("s35", n, p.replace("\\", "/"), f"{S35N}/{n}_logits.npz", env, ".jpg"))
    if only in (None, "du"):
        for n in DU_NAMES:
            env = {"FS_NO_SHADOWS": "1", "FS_UVPASS1": f"{W}/nets_frame_f/{n}_uv0.npz", "FS_UVPASS1_FRAME": "1", "FS_UVRECROP": "1",
                   "FS_UVOUT": f"{W}/nets_rc_f/{n}_uv2.npz", "FS_UVMODEL": "2", "FS_UVOUT2": f"{W}/nets_rc_f/{n}_uv0.npz", "FS_UVMODEL2": "0"}
            out.append(("du", n, f"{DU}/{n} copy.png".replace("\\", "/"), f"{W}/nets_rc_f/{n}_logits.npz", env, ".png"))
    return out


def run_one(a):
    out, cfg, (st, n, photo, lg, env, ext) = a
    cli, extra = CFGS[cfg]
    d = os.path.join(out, cfg, st)
    os.makedirs(d, exist_ok=True)
    e = dict(os.environ)
    e["PATH"] = r"C:\Users\User\Android\opencv\build\x64\vc16\bin;" + e["PATH"]
    for k in list(e):
        if k.startswith("FS_"):
            del e[k]
    e.update(env)
    e.update(extra)
    args = [cli, photo, os.path.join(d, n + ext).replace("\\", "/")]
    if "FS_NETS" in extra:
        for k in ("FS_UVPASS1", "FS_UVPASS1_FRAME", "FS_UVRECROP", "FS_UVOUT", "FS_UVMODEL", "FS_UVOUT2", "FS_UVMODEL2"):
            e.pop(k, None)
        args.append("net")
    else:
        args.append(lg)
    r = subprocess.run(args, env=e, capture_output=True, text=True)
    open(os.path.join(d, n + ".json"), "w").write(r.stdout)
    if r.returncode:
        open(os.path.join(d, n + ".err"), "w").write(r.stderr)
    return cfg, n, r.returncode


DROP = {"photo", "nets", "pipeline_ms"}


def load(p):
    try:
        lines = open(p).read().strip().splitlines()
        return json.loads(lines[-1].replace("\\", "/"))
    except Exception as ex:
        return {"_err": str(ex)}


def cmp(out, a, b, only):
    import cv2, numpy as np
    same_img = same_json = 0
    diffs = []
    for st, n, *_r in jobs(only):
        ext = _r[-1]
        pa, pb = (os.path.join(out, c, st, n) for c in (a, b))
        x, y = cv2.imread(pa + ext, cv2.IMREAD_UNCHANGED), cv2.imread(pb + ext, cv2.IMREAD_UNCHANGED)
        img_ok = x is not None and y is not None and x.shape == y.shape and np.array_equal(x, y)
        ja, jb = load(pa + ".json"), load(pb + ".json")
        keys = (set(ja) & set(jb)) - DROP
        jd = [k for k in sorted(keys) if ja[k] != jb[k]]
        only_a, only_b = sorted(set(ja) - set(jb) - DROP), sorted(set(jb) - set(ja) - DROP)
        same_img += img_ok
        same_json += not jd
        if not img_ok or jd:
            md = ""
            if x is not None and y is not None and x.shape == y.shape:
                dd = np.abs(x.astype(int) - y.astype(int))
                md = f"max |d| {dd.max()}, {100 * (dd.max(axis=2) > 0).mean():.2f}% px"
            elif x is not None and y is not None:
                md = f"shape {x.shape} vs {y.shape}"
            diffs.append(f"  {st} {n}: page {'same' if img_ok else 'DIFFERS ' + md}; json keys differ: {jd[:8]} " +
                         " | ".join(f"{k}: {ja[k]} vs {jb[k]}" for k in jd[:4] if len(str(ja[k])) < 80))
        if only_a or only_b:
            pass
    total = len(jobs(only))
    print(f"{a} vs {b}: pages identical {same_img}/{total}, JSON identical {same_json}/{total} (without {sorted(DROP)})")
    if total:
        st, n, *_r = jobs(only)[0]
        ja, jb = load(os.path.join(out, a, st, n) + ".json"), load(os.path.join(out, b, st, n) + ".json")
        print(f"  keys only in {a}: {sorted(set(ja) - set(jb))}; only in {b}: {sorted(set(jb) - set(ja))}")
    for d in diffs:
        print(d)


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    if cmd == "run":
        nj = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
        work = [(out, c, j) for c in sys.argv[3].split(",") for j in jobs(only)]
        with Pool(nj) as p:
            for cfg, n, rc in p.imap_unordered(run_one, work):
                if rc:
                    print("FAILED", cfg, n, rc, flush=True)
        print("done", sys.argv[3])
    else:
        cmp(out, sys.argv[3], sys.argv[4], only)
