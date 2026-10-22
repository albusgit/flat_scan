"""(tools/flatnet copy of docunet_bench/run_ours.py; FS_REPO = which pipeline_cli build, FS_ONLY = name list,
FS_POOL = pool size.) FlatScan's pipeline (desktop build of core/, tools/desktop_check) on the DocUNet benchmark photos.

  python run_ours.py CROP_DIR WORK_DIR nets           # page finder + UVDoc + SyntheticDoc-HR outputs (cached)
  python run_ours.py CROP_DIR WORK_DIR run NAME [ENV=VAL ...]   # one configuration -> WORK_DIR/NAME/{k}_{m}.png

The nets run with onnxruntime on the CPU, as in the app; pipeline_cli does everything else."""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.environ.get("FS_REPO", r"C:\Users\User\Desktop\fs_r11")  # tools/flatnet copy: the r11/main build by default
CLI = os.path.join(REPO, r"tools\desktop_check\build\pipeline_cli.exe")
PF_CLI = os.path.join(REPO, r"tools\desktop_check\build\pagefind_cli.exe")
MODELS = os.path.join(REPO, r"app\src\main\assets\models")
SHADE = os.path.join(MODELS, "shadenet.onnx")
os.environ["PATH"] = r"C:\Users\User\Android\opencv\build\x64\vc16\bin;" + os.environ["PATH"]

crop_dir, work, cmd = sys.argv[1:4]
NETS = os.environ.get("NETS", "nets")   # NETS=nets_frame with FS_FRAME_CORNERS=1: grids for the frame corners
NAMES = os.environ["FS_ONLY"].split(",") if os.environ.get("FS_ONLY") else [f"{k}_{m}" for k in range(1, 66) for m in (1, 2)]
_sess = {}


def sess(name):
    if name not in _sess:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        _sess[name] = ort.InferenceSession(os.path.join(MODELS, name), o, providers=["CPUExecutionProvider"])
    return _sess[name]


def run_net(model, x):
    s = sess(model)
    return s.run(None, {s.get_inputs()[0].name: x})


def photo(n):
    return os.path.join(crop_dir, f"{n} copy.png")


def nets(n):
    d = os.path.join(work, NETS)
    lg = os.path.join(d, n + "_logits.npz")
    base_lg = os.path.join(work, "nets", n + "_logits.npz")
    if not os.path.exists(lg) and os.path.exists(base_lg):
        import shutil
        shutil.copy(base_lg, lg)
    if not os.path.exists(lg):
        xp = os.path.join(d, n + "_pfx.npz")
        subprocess.run([PF_CLI, "input", photo(n), xp], check=True, capture_output=True)
        y = run_net("pagefinder.onnx", np.load(xp)["x"])
        np.savez(lg, logits=y[0][0].astype(np.float32))
        os.remove(xp)
    for model, onnx in ((0, "uvdoc.onnx"), (2, "syntheticdoc_hr.onnx")):
        out = os.path.join(d, f"{n}_uv{model}.npz")
        if os.path.exists(out):
            continue
        xin = os.path.join(d, f"{n}_uvin{model}.npz")
        env = {k: v.replace("{n}", n).replace("{work}", work) for k, v in os.environ.items()}
        env.update(FS_UVIN=xin, FS_UVMODEL=str(model))
        r = subprocess.run([CLI, photo(n), "-", lg], env=env, capture_output=True, text=True)
        if not os.path.exists(xin):
            return n, "no net input: " + r.stdout[-200:]
        ys = run_net(onnx, np.load(xin)["x"])
        np.savez(out, out=ys[0][0].astype(np.float32))
        os.remove(xin)
    return n, "ok"


def run_cfg(args):
    n, name, extra = args
    d = os.path.join(work, NETS)
    out = os.path.join(work, name, n + ".png")
    env = dict(os.environ, FS_NO_SHADOWS="1", FS_PYTHON=sys.executable,
               FS_UVOUT=os.path.join(d, f"{n}_uv0.npz"), FS_UVMODEL="0",
               FS_UVOUT2=os.path.join(d, f"{n}_uv2.npz"), FS_UVMODEL2="2")
    env.update({k: v.replace("{n}", n).replace("{nets}", d).replace("{work}", work) for k, v in extra.items()})
    for k, v in list(env.items()):
        if v == "":
            del env[k]
    r = subprocess.run([CLI, photo(n), out, os.path.join(d, n + "_logits.npz")], env=env, capture_output=True, text=True)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1].replace("\\", "/"))
    except Exception:
        j = {"ok": False, "error": (r.stdout + r.stderr)[-300:]}
    return n, j


if __name__ == "__main__":
    if cmd == "nets":
        os.makedirs(os.path.join(work, NETS), exist_ok=True)
        with Pool(6) as p:
            for n, s in p.imap_unordered(nets, NAMES):
                if s != "ok":
                    print(n, s, flush=True)
        print("nets done")
    else:
        name = sys.argv[4]
        extra = dict(a.split("=", 1) for a in sys.argv[5:])
        os.makedirs(os.path.join(work, name), exist_ok=True)
        with Pool(int(os.environ.get("FS_POOL", "8"))) as p:
            res = dict(p.map(run_cfg, [(n, name, extra) for n in NAMES]))
        json.dump(res, open(os.path.join(work, name, "pipeline.json"), "w"), indent=1)
        bad = [n for n, j in res.items() if not j.get("ok")]
        modes = {}
        for j in res.values():
            modes[j.get("mode")] = modes.get(j.get("mode"), 0) + 1
        print(name, "modes", modes, "failed", bad)
