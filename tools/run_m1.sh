#!/usr/bin/env bash
# Milestone 1: map reconstruction + final resample of the 5 pages from the reference grids.
# usage: tools/run_m1.sh FS_CLI REF_CODE REF_OUT TEST_IMAGES FIXTURES OUTDIR
set -e
CLI=$1; CODE=$2; REF=$3; IMG=$4; FX=$5; OUT=$6; PY=${PY:-python3}
mkdir -p $OUT
declare -A SRC=([photo1]=photo1.jpg [photo2]=photo2.jpg [scan_p1]=scan_p1.jpg [scan_p2]=scan_p2.jpg [scan_p3]=scan_p3.jpg)
for n in photo1 photo2 scan_p1 scan_p2 scan_p3; do
  for path in cpu gpu; do
    flag=""; [ $path = gpu ] && flag="--gpu"
    $CLI resample $IMG/${SRC[$n]} $REF/$n/warpmap.npz $OUT/${n}_$path.png $flag --dense $OUT/${n}_${path}_dense.npz > $OUT/${n}_$path.json
    echo "== $n $path: map parity (port_compare --limit 0.1)"
    (cd $CODE && $PY port_compare.py $REF/$n/warpmap.npz $OUT/${n}_${path}_dense.npz --limit 0.1 | tr -d '\n ' ; echo)
    $PY -I - $FX/$n/m1_resample_ref.png $OUT/${n}_$path.png <<'PY'
import sys, cv2, numpy as np
a = cv2.imread(sys.argv[1]).astype(int); b = cv2.imread(sys.argv[2]).astype(int)
d = np.abs(a - b)
print(f"   image parity vs reference resample: max {d.max()} levels, pixels>1: {(d.max(2) > 1).sum()}, pixels>0: {(d.max(2) > 0).sum()}  -> {'PASS' if d.max() <= 1 else 'FAIL'}")
PY
  done
done
