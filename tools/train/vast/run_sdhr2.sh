#!/usr/bin/env bash
# SD-HR fine-tune, run 2 (round 18; tools/train/flat2/train_run2.py). On the box, after unpacking flatscan_code.tar
# into /root and flatscan_sdhr2.tar into $FS_DATA:
#   bash run_sdhr2.sh [N_SYNTH=25000] [STEPS=12000]
# STAGE: setup | data | check | train | export | all (default, in that order). GEN_WORKERS: generator processes
# (default: cores - 4; the generator holds ~1 GB a process: lower it on a box with < 2 GB a core).
# Logs and results in $WORK/sdhr2: data.txt, check.txt, preflight.txt, train.txt, run/ (checkpoints, log.jsonl), export/.
set -euo pipefail
export FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
T=$(cd "$(dirname "$0")/.." && pwd)
F2=$T/flat2
N=${1:-25000}; STEPS=${2:-12000}
STAGE=${STAGE:-all}
GW=${GEN_WORKERS:-$(( $(nproc) - 4 ))}
S=$WORK/sdhr2; U=$WORK/uvdoc; mkdir -p "$S"
export FS_SDHR_CKPT=$FS_DATA/models/highres_121x177.ckpt
UVONNX=$T/../../app/src/main/assets/models/uvdoc.onnx
COUNTS=${COUNTS:-9,3,4}
on() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

if on setup; then
  pip install -q numpy opencv-python-headless onnx onnxruntime h5py scipy
  python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
fi

if on data; then
  # UVDoc's captured shapes (~10 GB by range requests) beside the synthetic data, which takes every other core
  ( python "$T/flat/fetch_zip.py" https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip "$U" split.json,metadata_sample/,img/,grid2d/ --jobs 32 \
      && { [ -f "$U/prep/index.json" ] || python "$T/flat/prep_uvdoc.py" "$U" "$U/prep"; } ) > "$S/uvdoc.txt" 2>&1 &
  UVPID=$!
  python "$F2/make_data.py" "$FS_DATA/flat2/pages2" "$S/synth" "$N" --workers "$GW" --seed 10000 --pass1 "$UVONNX" 2>&1 \
      | tail -4 | tee "$S/data.txt"
  wait $UVPID || echo "UVDoc fetch failed: see $S/uvdoc.txt (train with COUNTS=12,0,4)" | tee -a "$S/data.txt"
fi

if on check; then
  # before any training: the precision of the forward, and that UVDoc's targets line up with SD-HR's grid (the
  # original net's error on them is small: a swapped axis or a turned grid would show as an error of ~0.3 and up)
  python "$F2/preflight.py" "$FS_DATA/flat2/held2" --n 100 2>&1 | grep -v Warn | tee "$S/preflight.txt"
  if [ -f "$U/prep/index.json" ]; then
    F2=$F2 HELD=$FS_DATA/flat2/held2 python - "$U/prep" <<'EOF' 2>&1 | grep -v Warn | tee "$S/check.txt"
import sys, os, numpy as np, torch
sys.path[:0] = [os.environ['F2'], os.path.join(os.environ['F2'], '..'), os.path.join(os.environ['F2'], '..', '..', 'flatnet'), os.path.join(os.environ['F2'], '..', 'flat')]
import common as C
from train_sdhr import Mixed
from run2_core import bm_forward, map_terms
ds = Mixed([os.environ['HELD']], sys.argv[1], '', [0, 1, 0], 64, all_ids=True)
net = C.load_sdhr(C.SD_CKPT, 'cuda')
l1 = []
for i in range(64):
    x, t, kind, _ = ds[('uvdoc', i)]
    with torch.no_grad():
        p = bm_forward(net, (x[None].float().cuda() / 127.5 - 1))
    l1.append(map_terms(p, t[None].cuda())['l1'].item())
m = float(np.median(l1))
print(f'UVDoc targets: the original net median L1 {m:.4f} on 64 samples (synthetic held: ~0.02)')
sys.exit(0 if m < 0.08 else 'UVDoc targets do not line up with SD-HR grids: check prep_uvdoc / app_crop before training')
EOF
  fi
fi

if on train; then
  UVARG=""; [ -f "$U/prep/index.json" ] && UVARG="--uvdoc $U/prep" || COUNTS=${COUNTS/,3,/,0,}
  python "$F2/train_run2.py" --synth "$S/synth" --held "$FS_DATA/flat2/held2" $UVARG --anchor "$FS_DATA/flat2/anchor2" \
      --counts "$COUNTS" --steps "$STEPS" --micro 8 --eval 2000 --save 1000,3000 --real_sets "${REAL_SETS-dir300}" --real_every 2 \
      --workers ${LOAD_WORKERS:-8} --out "$S/run" ${RESUME:+--resume "$RESUME"} 2>&1 | grep --line-buffered -v Warn | tee -a "$S/train.txt"
fi

if on export; then
  mkdir -p "$S/export"
  for ck in "$S/run"/step*.pt "$S/run/best_proxy.pt"; do
    [ -f "$ck" ] || continue
    python "$T/../flatnet/export_onnx.py" "$S/export/$(basename "${ck%.pt}").onnx" "$ck" 2>&1 | tail -1
  done
  echo "bring back: $S/export/*.onnx $S/run/log.jsonl $S/run/run.json $S/*.txt"
fi
