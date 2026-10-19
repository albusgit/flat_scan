#!/usr/bin/env bash
# Stage-by-stage parity of the port against the reference fixtures (tools/export_stages.py):
# detection, surface fit, warp-map stages fed with the reference's own stage input, solver.
# usage: tools/run_stage_parity.sh FS_CLI APP_REF_DIR FIXTURES OUTDIR
set -euo pipefail
CLI=$1; REF_DIR=$2; FX=$3; OUT=$4; PY=${PY:-python3}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $OUT
echo "=== page detection (corners.py) and surface fit (surface.py)"
for p in photo1 photo2 bench_P1 bench_P2; do
  src=$REF_DIR/test_images/$p.jpg; [ -f $src ] || src=$REF_DIR/bench_photo/photo${p#bench_}.png
  $CLI detect $src $OUT/det_$p.npz
  $CLI surface $src > $OUT/surf_$p.json
  $PY -I - $FX/$p $OUT $p <<'PY'
import sys, json, numpy as np
fx, out, p = sys.argv[1:4]
a, b = np.load(f'{fx}/detect.npz'), np.load(f'{out}/det_{p}.npz')
s, m = json.load(open(f'{fx}/info.json'))['surface'], json.load(open(f'{out}/surf_{p}.json'))
print(f"{p:9s} contour identical: {np.array_equal(a['contour'], b['contour'])}  RANSAC inliers equal: {np.array_equal(a['side_inl'], b['side_inl'])}"
      f"  corner max diff {np.abs(a['corners'] - b['corners']).max():.1e} px")
print(f"{'':9s} surface axis {m['axis']}/{s['axis']}  cost {m['cost']:.6f} / {s['cost']:.6f} (app / reference)"
      f"  max |c diff| {np.abs(np.array(m['c']) - s['c']).max():.1e} rad  fit {m['fit_ms']} ms")
PY
done
echo
echo "=== warp map (warpmap.unwarp_image) on the reference's own input for this stage"
for t in photo1:stage1 photo2:stage1 scan_p1:scan scan_p2:scan scan_p3:scan; do
  n=${t%%:*}; tag=${t##*:}
  src=$FX/$n/stage1.png; [ $tag = scan ] && src=$REF_DIR/test_images/$n.jpg
  echo "--- $n"
  $CLI warp $src $OUT/warp_$n.npz $OUT/warp_${n}_A.png
  $PY -I $HERE/compare_warp.py $FX/$n/${tag}_warp.npz $OUT/warp_$n.npz $FX/$n/${tag}_A_img.png $OUT/warp_${n}_A.png | grep -v '^ctrl'
done
echo
echo "=== sparse solver on the reference's exported pass-2 systems"
for t in scan_p1/scan:2686 photo1/stage1:1600; do
  f=${t%%:*}; w=${t##*:}
  for c in y x; do
    [ -f $FX/${f}_pass2_${c}_system.npz ] || continue
    echo "--- $f pass 2 ($c)"; $CLI solvetest $FX/${f}_pass2_${c}_system.npz $((w / 8 + 2))
  done
done
