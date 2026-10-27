#!/usr/bin/env bash
# Shadow net: dark cast shadows. Data once (CPU), then GPU-only training with progressive resizing and early
# stopping; best.pt = best on real shadow photos. 12k steps with the size schedule show ~1.15 M pages, more than
# 30k steps at full size did (0.96 M), in ~40 % of the time.
#   bash run_shade.sh [PAGES=5000] [BANK=20000] [STEPS=12000] [BS=32]
# SHADE_WORKERS: CPU processes for the data (default: all cores). STAGE: data | train | all (default).
set -euo pipefail
export FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
T=$(cd "$(dirname "$0")/.." && pwd)
PAGES=${1:-5000}; BANK=${2:-20000}; STEPS=${3:-12000}; BS=${4:-32}
W=${SHADE_WORKERS:-$(nproc)}; STAGE=${STAGE:-all}
D=$WORK/shade
if [ "$STAGE" != train ]; then
  [ -f "$D/pages.done" ] || { python "$T/shade/make_bank.py" pages "$D" "$PAGES" "$W" && touch "$D/pages.done"; }
  [ -f "$D/bank.done" ] || { python "$T/shade/make_bank.py" bank "$D" "$BANK" "$W" && touch "$D/bank.done"; }
fi
if [ "$STAGE" != data ]; then
  python "$T/shade/train.py" --bank "$D" --out "$D/run" --steps "$STEPS" --bs "$BS" --floor 0.12 2>&1 | tee "$D/train.txt"
  python "$T/shade/export.py" "$D/run/best.pt" "$D/export"
  python "$T/shade/realval.py" "$D/export/shadenet.onnx" 0.12 | tee "$D/export/realval.txt"
  echo "bring back: $D/export/*.onnx, $D/run/log.jsonl, $D/export/realval.txt"
fi
