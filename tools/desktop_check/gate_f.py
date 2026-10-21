"""Gate F (docs/accel/PLAN.md, Stream F) comparisons; run with a Python that has numpy + cv2 (-I).

  python -I gate_f.py nets REF_DIR NEW_DIR [NAME ...]     net outputs, accelerated vs CPU: net_check.exe's
                                                          folders, or pipeline_cli FS_NET_DUMP / gate_b.py runs
  python -I gate_f.py pages REF_DIR NEW_DIR [NAME ...]    output pages (.jpg / .png) pixel for pixel
  python -I gate_f.py json REF_DIR NEW_DIR [--decisions] [NAME ...]   pipeline_cli JSON, without the timing /
                                                          "gpu" keys; --decisions: only the pipeline's decisions
  python -I gate_f.py table OUT.md BENCH.json ...         page_bench.exe JSON lines -> one markdown table

nets: per photo and net input, the grid move in px of the 1600 x 2263 analysis page, max and p99 over the
grid points (tools/accel/compare_outputs.py grid_px, Stream A's measure), pooled p99 per net against Gate
A's tolerance (grid p99 < 0.25 px); ShadeNet as |shading| difference in 8-bit levels; the page finder
(NAME_logits.npz, pipeline_cli dumps) as max |logit| difference and the share of the 256 x 256 mask whose
page / background decision flips. In pipeline_cli dumps a grid's input depends on the nets before it
(corners, re-crop), so only net_check's fixed inputs isolate one net. Exit 1 when a pooled
grid p99 is at or above the tolerance. pages, json: exit 1 when any page / JSON differs (CPU-path checks);
json --decisions is report only for accelerated nets (exit 0), as HUMAN GATE 3's "decisions unchanged".
"""
import json
import os
import sys

import numpy as np

PAGE_W, PAGE_H = 1600, int(round(1600 * 297.0 / 210.0))
TOL_P99_PX = 0.25   # Gate A (accelerated nets vs CPU)
TAGS = [("logits", "page finder"), ("p1", "UVDoc pass 1"), ("rc0", "UVDoc re-crop"), ("rc2", "SyntheticDoc-HR"), ("shade", "ShadeNet")]


def grid_px(a, b):
    """Per grid point move between grids a, b (2 x gh x gw, normalised) in analysis-page px."""
    ext_u = np.median(a[0, :, -1] - a[0, :, 0]) / 2
    ext_v = np.median(a[1, -1, :] - a[1, 0, :]) / 2
    du = (b[0] - a[0]).astype(np.float64) / 2 / ext_u * PAGE_W
    dv = (b[1] - a[1]).astype(np.float64) / 2 / ext_v * PAGE_H
    return np.hypot(du, dv).ravel()


def names_in(d, suffix):
    return sorted({f[: -len(suffix)] for f in os.listdir(d) if f.endswith(suffix)})


def nets(ref, new, names):
    names = names or sorted({f.rsplit("_", 1)[0] for f in os.listdir(new) if f.endswith(".npz") and not f.endswith("_in.npz")})
    pooled = {t: [] for t, _ in TAGS}
    print(f"{'photo':34s} " + " ".join(f"{t:>17s}" for t, _ in TAGS) + "   (max / p99: px of the page; shade: levels; logits: max |d| / mask flips)")
    for n in names:
        row = []
        for t, _ in TAGS:
            p, q = os.path.join(ref, f"{n}_{t}.npz"), os.path.join(new, f"{n}_{t}.npz")
            if not (os.path.exists(p) and os.path.exists(q)):
                row.append(f"{'-':>17s}")
                continue
            key = "logits" if t == "logits" else "out"
            a, b = np.load(p)[key].astype(np.float64), np.load(q)[key].astype(np.float64)
            if t == "logits":
                flip = np.mean(np.argmax(a, axis=0) != np.argmax(b, axis=0))
                d = np.abs(a - b).ravel()
                pooled[t].append(np.array([flip]))
                row.append(f"{d.max():8.4f}/{100 * flip:7.3f}%")
                continue
            if t == "shade":
                d = np.abs(a - b).ravel() * 255
            else:
                a, b = a.reshape(2, a.shape[-2], a.shape[-1]), b.reshape(2, b.shape[-2], b.shape[-1])
                d = grid_px(a, b)
            pooled[t].append(d)
            row.append(f"{d.max():8.4f}/{np.percentile(d, 99):8.4f}")
        print(f"{n:34s} " + " ".join(row))
    bad = 0
    print(f"\npooled over {len(names)} photos (Gate A tolerance: grid p99 < {TOL_P99_PX} px):")
    for t, label in TAGS:
        if not pooled[t]:
            continue
        d = np.concatenate(pooled[t])
        if t == "logits":
            print(f"  {label:16s} mask flips: mean {100 * d.mean():.3f} %, max {100 * d.max():.3f} % of the 256 x 256 mask")
            continue
        p99 = np.percentile(d, 99)
        unit = "levels" if t == "shade" else "px"
        verdict = "" if t == "shade" else ("  PASS" if p99 < TOL_P99_PX else "  FAIL")
        bad += verdict == "  FAIL"
        print(f"  {label:16s} max {d.max():.4f} {unit}, p99 {p99:.4f} {unit}{verdict}")
    return bad


def pages(ref, new, names):
    import cv2
    exts = (".jpg", ".png")
    names = names or sorted({os.path.splitext(f)[0] for f in os.listdir(new) if f.endswith(exts) and ".lines." not in f})
    bad = 0
    for n in names:
        p = next((os.path.join(ref, n + e) for e in exts if os.path.exists(os.path.join(ref, n + e))), None)
        q = next((os.path.join(new, n + e) for e in exts if os.path.exists(os.path.join(new, n + e))), None)
        if not p or not q:
            print(f"{n:34s} missing")
            bad += 1
            continue
        a, b = cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.imread(q, cv2.IMREAD_UNCHANGED)
        if a.shape != b.shape:
            print(f"{n:34s} size {a.shape} vs {b.shape}")
            bad += 1
            continue
        d = np.abs(a.astype(np.int16) - b)
        dm = d.max(axis=-1) if d.ndim == 3 else d
        print(f"{n:34s} " + ("same" if d.max() == 0 else f"differs: max {d.max()}, {100 * np.mean(dm > 0):.2f} % px, {100 * np.mean(dm > 8):.2f} % px > 8"))
        bad += d.max() != 0
    print(f"pages with any difference: {bad} of {len(names)}")
    return bad


DECISIONS = ["ok", "error", "mode", "uv_model", "uv_rejected", "lines_rejected", "needs_manual_crop", "rot", "shade_net",
             "uv_defines_page", "zoom", "assess"]


def json_cmp(ref, new, names, decisions):
    names = names or sorted(f[:-5] for f in os.listdir(new) if f.endswith(".json") and not f.startswith("times"))
    bad = 0
    for n in names:
        p, q = os.path.join(ref, n + ".json"), os.path.join(new, n + ".json")
        if not (os.path.exists(p) and os.path.exists(q)):
            print(f"{n:34s} missing")
            bad += 1
            continue
        a, b = (json.loads(open(x).read().replace("\\", "/")) for x in (p, q))
        for j in (a, b):
            for k in ("nets", "pipeline_ms", "photo", "gpu"):
                j.pop(k, None)
        keys = DECISIONS if decisions else sorted(set(a) | set(b))
        diff = [k for k in keys if a.get(k) != b.get(k)]
        print(f"{n:34s} " + ("same" if not diff else "differs: " + ", ".join(diff)))
        bad += bool(diff)
    what = "decisions" if decisions else "JSON"
    print(f"photos with different {what}: {bad} of {len(names)}")
    return bad


def table(out, files):
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    cols = [("pf_ms", "page finder"), ("detect_ms", "detect"), ("uv_pass1_ms", "UVDoc p1"), ("uvdoc_ms", "UVDoc"),
            ("synthdoc_hr_ms", "SD-HR"), ("shade_ms", "ShadeNet"), ("run_photo_ms", "run_photo"), ("page_ms", "page")]
    configs = []
    for r in rows:
        k = (r["backend"], r["gpu"], r["dense"])
        if k not in configs:
            configs.append(k)
    lines = ["| backend | GPU stages | " + " | ".join(c for _, c in cols) + " | nets sum | cold first page |", "|" + "---|" * (len(cols) + 4)]
    for k in configs:
        rs = [r for r in rows if (r["backend"], r["gpu"], r["dense"]) == k]
        med = {c: float(np.median([r[c] for r in rs])) for c, _ in cols}
        nets = med["pf_ms"] + med["uv_pass1_ms"] + med["uvdoc_ms"] + med["synthdoc_hr_ms"] + med["shade_ms"]
        gpu = ("on + dense" if k[2] else "on") if k[1] else "off"
        lines.append(f"| {k[0]} | {gpu} | " + " | ".join(f"{med[c]:.0f}" for c, _ in cols) + f" | {nets:.0f} | {rs[0]['first_ms']:.0f} |")
    txt = "\n".join(lines) + f"\n\nms; median over {len(rs)} photos of each photo's median ({rs[0]['reps']} timed passes after one warm-up pass). " \
                             "'nets sum': page finder + the three flattening-net runs + ShadeNet. 'cold first page': the first photo's " \
                             f"warm-up pass ({rs[0]['photo']}): session creation, GPU context and shader compile, CUDA tuning (one process " \
                             "per row; the driver's CUDA JIT cache already warm).\n"
    open(out, "w").write(txt)
    print(txt)
    return 0


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    if a[0] == "nets":
        sys.exit(1 if nets(a[1], a[2], a[3:]) else 0)
    if a[0] == "pages":
        sys.exit(1 if pages(a[1], a[2], a[3:]) else 0)
    if a[0] == "json":
        dec = "--decisions" in a
        rest = [x for x in a[1:] if x != "--decisions"]
        bad = json_cmp(rest[0], rest[1], rest[2:], dec)
        sys.exit(0 if dec else (1 if bad else 0))
    if a[0] == "table":
        sys.exit(table(a[1], a[2:]))
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
