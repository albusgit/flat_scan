#!/bin/bash
# Gate D (docs/accel/PLAN.md): the flattening nets beside the CPU stages (PipelineOptions::flat_nets,
# overlap_nets) against the same build run sequentially (FS_SEQUENTIAL=1), CPU backend, bit for bit, on
# the 35 photos (ShadeNet on, as the app) and on DocUNet (FS_NO_SHADOWS, PNG, + the pages against
# docunet_bench/work/new_fold2); how often the speculative stages had to be redone; and the wall time
# with the nets artificially delayed (FS_NET_DELAY_MS, an accelerator's time without its CPU load).
#
#   tools/desktop_check/gate_d.sh OUT [REF_S35 [REF_DU]]      (Git Bash)
#
# REF_S35 / REF_DU (optional): Gate B outputs to compare against as well (gate_b_full.sh's
# OUT/s35_ref and OUT/du_ref, i.e. today's Python ORT path). Needs this checkout's
# tools/desktop_check/build (docs/accel/NOTES.md) and the round-4 venv (PY, numpy + cv2). Takes the
# machine-wide heavy lock around the DocUNet runs. Estimate: about 20 min on the i5-14400F.
set -e
OUT=$(realpath -m "$1"); REF_S35=${2:-}; REF_DU=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/c/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe}
LOCK=/c/Users/User/Desktop/docunet_bench/.heavy_lock
G=$REPO/tools/desktop_check
CLI=$G/build/pipeline_cli.exe
DELAY=${DELAY:-1500}
TIMING_PHOTOS=${TIMING_PHOTOS:-docres__for_deshadowing,user__page3_sideways,scanner_samples__receipt,dewarpnet__1_1_copy,scanner_samples__tax}
mkdir -p "$OUT"
unset FS_SEQUENTIAL FS_NET_DELAY_MS

# 1. 35 photos: overlapped and sequential
"$PY" -I "$G/gate_b.py" new s35 "$OUT/s35_ovl" --cli "$CLI" --shade --jobs 4
FS_SEQUENTIAL=1 "$PY" -I "$G/gate_b.py" new s35 "$OUT/s35_seq" --cli "$CLI" --shade --jobs 4
"$PY" -I "$G/gate_b.py" cmp s35 "$OUT/s35_seq" "$OUT/s35_ovl" | tee "$OUT/cmp_s35.txt" || true
[ -n "$REF_S35" ] && { "$PY" -I "$G/gate_b.py" cmp s35 "$REF_S35" "$OUT/s35_ovl" | tee "$OUT/cmp_s35_vs_gateb.txt" || true; }

# 2. DocUNet, under the heavy lock
until mkdir "$LOCK" 2>/dev/null; do echo "waiting for heavy lock $(date +%T)"; sleep 30; done
trap 'rmdir "$LOCK"' EXIT
"$PY" -I "$G/gate_b.py" new docunet "$OUT/du_ovl" --cli "$CLI" --jobs 4
FS_SEQUENTIAL=1 "$PY" -I "$G/gate_b.py" new docunet "$OUT/du_seq" --cli "$CLI" --jobs 4
rmdir "$LOCK"; trap - EXIT
"$PY" -I "$G/gate_b.py" cmp docunet "$OUT/du_seq" "$OUT/du_ovl" --images C:/Users/User/Desktop/docunet_bench/work/new_fold2 | tee "$OUT/cmp_docunet.txt" || true
[ -n "$REF_DU" ] && { "$PY" -I "$G/gate_b.py" cmp docunet "$REF_DU" "$OUT/du_ovl" | tee "$OUT/cmp_docunet_vs_gateb.txt" || true; }

# 3. wall time, nets delayed by DELAY ms, one photo at a time
FS_NET_DELAY_MS=$DELAY "$PY" -I "$G/gate_b.py" new s35 "$OUT/t_ovl" --cli "$CLI" --shade --jobs 1 --only "$TIMING_PHOTOS"
FS_NET_DELAY_MS=$DELAY FS_SEQUENTIAL=1 "$PY" -I "$G/gate_b.py" new s35 "$OUT/t_seq" --cli "$CLI" --shade --jobs 1 --only "$TIMING_PHOTOS"
"$PY" -I "$G/gate_b.py" cmp s35 "$OUT/t_seq" "$OUT/t_ovl" --only "$TIMING_PHOTOS" > "$OUT/cmp_timing.txt" || true

# 4. summary: redo counts and the timing table
"$PY" -I - "$OUT" "$DELAY" <<'PYEOF' | tee "$OUT/gate_d_summary.txt"
import glob, json, os, statistics, sys
out, delay = sys.argv[1], sys.argv[2]
def load(d):
    r = {}
    for p in sorted(glob.glob(os.path.join(out, d, "*.json"))):
        j = json.loads(open(p).read().replace("\\", "/"))
        r[os.path.basename(p)[:-5]] = j
    return r
for name in ("s35", "du"):
    js = load(name + "_ovl")
    ov = {n: j["nets"]["overlap"] for n, j in js.items() if "overlap" in j.get("nets", {})}
    print(f"{name}: {len(js)} photos, {len(ov)} with the nets beside the CPU stages")
    print(f"  uv_defines_page (corners moved by a grid): {sum(bool(j.get('uv_defines_page')) for j in js.values())}")
    print(f"  orientation page redone: {sum(not o['orient_page_reused'] for o in ov.values())}"
          f"   orientation decision redone: {sum(not o['orient_reused'] for o in ov.values())}")
    print(f"  redone photos: {', '.join(n for n, o in ov.items() if not o['orient_reused'])}")
for f in ("cmp_s35.txt", "cmp_s35_vs_gateb.txt", "cmp_docunet.txt", "cmp_docunet_vs_gateb.txt", "cmp_timing.txt"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        print(f"{f}: {open(p).read().strip().splitlines()[-1]}")
to, ts = load("t_ovl"), load("t_seq")
print(f"\nwall time, nets delayed by {delay} ms (ms; cpu branch = run_photo's start to its wait for the nets):")
print(f"{'photo':34s} {'seq join':>8s} {'seq run':>8s} | {'ovl nets':>8s} {'cpu br.':>8s} {'ovl join':>8s} {'ovl run':>8s} {'saved':>6s}")
rows = []
for n in to:
    if n not in ts: continue
    a, b = ts[n], to[n]
    oa, ob = a["nets"]["overlap"], b["nets"]["overlap"]
    cpu = ob["join_ms"] - ob["wait_ms"]
    rows.append((oa["join_ms"], a["pipeline_ms"], ob["nets_ms"], cpu, ob["join_ms"], b["pipeline_ms"]))
    print(f"{n:34s} {oa['join_ms']:8.0f} {a['pipeline_ms']:8.0f} | {ob['nets_ms']:8.0f} {cpu:8.0f} {ob['join_ms']:8.0f} {b['pipeline_ms']:8.0f} {a['pipeline_ms'] - b['pipeline_ms']:6.0f}")
if rows:
    med = [statistics.median(c) for c in zip(*rows)]
    print(f"{'median':34s} {med[0]:8.0f} {med[1]:8.0f} | {med[2]:8.0f} {med[3]:8.0f} {med[4]:8.0f} {med[5]:8.0f} {med[1] - med[5]:6.0f}")
    print("overlapped join vs max(cpu branch, nets): " + ", ".join(f"{r[4]:.0f} vs {max(r[2], r[3]):.0f}" for r in rows))
PYEOF
# Pass: cmp_s35.txt and cmp_docunet.txt say "photos with any difference: 0 of 35 / 0 of 130" and
# cmp_docunet.txt's vs_images is 130 same; overlapped join ~ max(cpu branch, nets) in the table.
