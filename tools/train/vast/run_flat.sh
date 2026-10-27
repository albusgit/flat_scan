#!/usr/bin/env bash
# UVDoc flattening net: fine-tune on the UVDoc dataset's true grids with the app's crops, early stop on the real
# score; best.pt = best two-pass MS-SSIM on dir300. Fetches only img/, grid2d/, metadata_sample/ of UVDoc_final.zip
# (range requests, ~10 GB).
#   bash run_flat.sh [STEPS=30000] [BS=24]
# FLAT_WORKERS: data-loader processes (default: all cores but 2). STAGE: data | train | all (default).
set -euo pipefail
export FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
T=$(cd "$(dirname "$0")/.." && pwd)
STEPS=${1:-30000}; BS=${2:-24}
W=${FLAT_WORKERS:-$(( $(nproc) - 2 ))}; STAGE=${STAGE:-all}
U=$WORK/uvdoc
if [ "$STAGE" != train ]; then
  python "$T/flat/fetch_zip.py" https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip "$U" split.json,metadata_sample/,img/,grid2d/ --jobs 32
  [ -f "$U/prep/index.json" ] || python "$T/flat/prep_uvdoc.py" "$U" "$U/prep"
  [ -s "$U/baseline.txt" ] || python "$T/flat/realval.py" "$T/../../app/src/main/assets/models/uvdoc.onnx" "$FS_DATA/sets/dir300.json" 3 | tee "$U/baseline.txt"
fi
if [ "$STAGE" != data ]; then
  python "$T/flat/train.py" --data "$U/prep" --out "$U/run" --steps "$STEPS" --bs "$BS" --workers "$W" 2>&1 | tee "$U/train.txt"
  python "$T/flat/export.py" "$U/run/best.pt" "$U/export"
  python "$T/flat/realval.py" "$U/export/uvdoc.onnx" "$FS_DATA/sets/dir300.json" 1 | tee "$U/export/realval_all300.txt"
  echo "bring back: $U/export/*.onnx, $U/run/log.jsonl, $U/baseline.txt, $U/export/realval_all300.txt"
fi
