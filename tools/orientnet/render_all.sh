#!/bin/bash
# render_all.sh OUT: 12 x 200 training pages + 4 x 75 held-font pages, in parallel
OUT=${1:-/e/fs_orient/pages}
PY=/c/Users/User/envs/flatscan_torch/Scripts/python.exe
D=$(dirname "$0")
for s in $(seq 0 11); do $PY $D/render_pages.py $OUT/train 200 --seed $((100 + s)) --split train & done
for s in $(seq 0 3); do $PY $D/render_pages.py $OUT/held 75 --seed $((900 + s)) --split held & done
wait
ls $OUT/train | wc -l; ls $OUT/held | wc -l
