#!/usr/bin/env bash
# SD-HR (the app's first flattening net): exact synthetic ground truth + UVDoc's captured shapes + an anchor set of
# real photos (learning without forgetting); best.pt chosen on real photos run the app's two-pass way.
#   bash run_sdhr.sh [N_SYNTH=40000] [STEPS=30000] [BS=8]
# Needs in FS_DATA (pack_sdhr.py): flat2/pages (page textures), flat2/anchor (index.jsonl + photos),
# models/highres_121x177.ckpt (SD-HR's released weights, MIT), raw/dir300_small + sets (validation).
# STAGE: data | train | all (default). Logs: $WORK/sdhr/*.txt
set -euo pipefail
export FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
T=$(cd "$(dirname "$0")/.." && pwd)
N=${1:-40000}; STEPS=${2:-30000}; BS=${3:-8}
W=${FLAT_WORKERS:-$(( $(nproc) - 2 ))}; STAGE=${STAGE:-all}
S=$WORK/sdhr; mkdir -p "$S"
export FS_SDHR_CKPT=${FS_SDHR_CKPT:-$FS_DATA/models/highres_121x177.ckpt}
U=$WORK/uvdoc
if [ "$STAGE" != train ]; then
  # 1. UVDoc's real shapes (shared with run_flat.sh; ~10 GB by range requests)
  python "$T/flat/fetch_zip.py" https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip "$U" split.json,metadata_sample/,img/,grid2d/ --jobs 32
  [ -f "$U/prep/index.json" ] || python "$T/flat/prep_uvdoc.py" "$U" "$U/prep"
  # 2. exact synthetic samples, every core (resumable: index.jsonl is appended, seeds continue)
  have=$( [ -f "$S/synth/index.jsonl" ] && wc -l < "$S/synth/index.jsonl" || echo 0 )
  if [ "$have" -lt "$N" ]; then
    python "$T/flat2/make_data.py" "$FS_DATA/flat2/pages" "$S/synth" $(( N - have )) --workers "$W" --seed $(( 1000 + have / 50 )) 2>&1 | tail -3 | tee "$S/data.txt"
  fi
fi
if [ "$STAGE" != data ]; then
  python "$T/flat2/train_sdhr.py" --synth "$S/synth" --uvdoc "$U/prep" --anchor "$FS_DATA/flat2/anchor" --out "$S/run" \
      --steps "$STEPS" --bs "$BS" --workers "$W" --real_sets dir300 --real_every 2 2>&1 | tee "$S/train.txt"
  # the best and the last two checkpoints as the app's ONNX (fp16 weights), for the Windows check
  mkdir -p "$S/export"
  for ck in "$S/run/best.pt" $(ls "$S/run"/step*.pt | tail -2); do
    python "$T/../flatnet/export_onnx.py" "$S/export/$(basename "${ck%.pt}").onnx" "$ck" | tail -2
  done
  echo "bring back: $S/export/*.onnx, $S/run/log.jsonl, $S/train.txt"
fi
