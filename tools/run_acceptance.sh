#!/usr/bin/env bash
# Acceptance tests of APP_BUILD_BRIEF.md section 5 for the C++ port (host harness).
#
# usage: tools/run_acceptance.sh FS_CLI APP_REF_DIR REF_OUT FIXTURES OUTDIR
#   APP_REF_DIR : unpacked app_ref/ (code/, test_images/, bench_photo/ from photobench.py)
#   REF_OUT     : reference outputs (photo_pipeline.py surface / warpmap.py) -> photo1/, scan_p1/, bench_P1/ ...
#   FIXTURES    : tools/m1_fixtures.py + tools/export_stages.py output
# env: PY (python with the pinned requirements-round3.txt), EGL_PLATFORM=surfaceless on headless hosts
set -euo pipefail
CLI=$1; REF_DIR=$2; REF=$3; FX=$4; OUT=$5; PY=${PY:-python3}
CODE=$REF_DIR/code; IMG=$REF_DIR/test_images; BENCH=$REF_DIR/bench_photo
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $OUT
for path in gpu cpu; do
  flag=""; [ $path = cpu ] && flag="--cpu"
  O=$OUT/$path; mkdir -p $O
  # --reference-orient: the reference's ungated 0/180 decision, so the parity rows stay like-for-like
  for p in photo1 photo2; do $CLI photo $IMG/$p.jpg $O/$p $flag --reference-orient; done
  for c in P1 P2; do $CLI photo $BENCH/photo$c.png $O/bench_$c $flag --reference-orient; done
  for s in p1 p2 p3; do $CLI scan $IMG/scan_$s.jpg $O/scan_$s $flag; done
done
# the app's default (gated) orientation on the synthetic bench pages: they must stay upright
mkdir -p $OUT/gated
for c in P1 P2; do $CLI photo $BENCH/photo$c.png $OUT/gated/bench_$c; done
echo
echo "=== 1. map parity: port_compare.py reference candidate --limit 0.1"
for path in gpu cpu; do
  for n in photo1 photo2 scan_p1 scan_p2 scan_p3 bench_P1 bench_P2; do
    printf "%-4s %-8s " $path $n
    (cd $CODE && $PY port_compare.py $REF/$n/warpmap.npz $OUT/$path/$n/warpmap.npz --limit 0.1 | tr -d '\n ' ; echo) || true
  done
done
echo
echo "=== 2. image parity (8-bit), GPU vs CPU path and against the reference"
$PY -I - $OUT $REF $FX <<'PY'
import sys, cv2, numpy as np
out, ref, fx = sys.argv[1:4]
def d(a, b):
    a, b = cv2.imread(a).astype(int), cv2.imread(b).astype(int)
    if a.shape != b.shape: return 'shape mismatch'
    x = np.abs(a - b).max(2)
    return f'max {x.max():3d}  pixels>1 {int((x > 1).sum()):7d} ({(x > 1).mean() * 100:.3f}%)  pixels>0 {int((x > 0).sum()):7d}'
for n, f in (('photo1', 'out.png'), ('photo2', 'out.png'), ('scan_p1', 'B_warpmap.png'), ('scan_p2', 'B_warpmap.png'), ('scan_p3', 'B_warpmap.png')):
    print(f'{n:8s} GPU vs CPU path:                    {d(f"{out}/gpu/{n}/{f}", f"{out}/cpu/{n}/{f}")}')
    print(f'{n:8s} GPU vs reference image:             {d(f"{out}/gpu/{n}/{f}", f"{ref}/{n}/{f}")}')
    print(f'{n:8s} reference resampled from its grid:  {d(f"{fx}/{n}/m1_resample_ref.png", f"{ref}/{n}/{f}")}   (reference vs itself)')
PY
echo
echo "=== 3-5. ground truth, straightness, photo-1 title border (GPU path outputs)"
$PY -I $HERE/acceptance.py $CODE $OUT/gpu --flat $IMG/flat_page.png --bench $BENCH | tee $OUT/acceptance_gpu.json
echo
echo "=== 6. orientation: app default (orient_min_conf 0.2) vs the reference rule (GPU path)"
$PY -I - $OUT <<'PY'
import sys, json
out = sys.argv[1]
names = {-1: 'none', 0: '90 CW', 1: '180', 2: '90 CCW'}
for n in ('photo1', 'photo2', 'bench_P1', 'bench_P2'):
    ref = json.load(open(f'{out}/gpu/{n}/info.json'))
    line = f'{n:9s} reference rule: rot {ref["rot"]:2d} ({names[ref["rot"]]:6s}) orient [sh, sv, upright] {ref["orient"]}'
    if n.startswith('bench'):
        g = json.load(open(f'{out}/gated/{n}/info.json'))
        line += f' | default: rot {g["rot"]:2d} ({names[g["rot"]]}) -> ' + ('upright, not rotated: PASS' if g['rot'] == -1 else 'ROTATED: FAIL')
    print(line)
PY
echo
echo "=== timings (host, ms)"
for n in photo1 photo2 bench_P1 bench_P2 scan_p1 scan_p2 scan_p3; do echo "$n: $(cat $OUT/gpu/$n/info.json)"; done
