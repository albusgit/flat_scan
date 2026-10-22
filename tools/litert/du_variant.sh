#!/bin/bash
# DocUNet run + scoring of one model-size variant (Round 11, stream MODELSIZE), under the heavy lock.
#   bash tools/litert/du_variant.sh NAME MODELS_DIR BASE_RUN [--no-pf]
#   NAME: output folder under $OUT/e2e (du_NAME); MODELS_DIR: pipeline_cli FS_MODELS (quant_e2e.py);
#   BASE_RUN: the run whose pages and metrics the variant is compared with ("" for none: run only).
# 1) pipeline on the 130 crops (lock), 2) pages vs BASE_RUN, 3) the changed pages scored (lock;
# docunet_bench/docunet_eval.py), 4) metrics composed from BASE_RUN's per-image scores.
set -u
NAME=$1; MODELS=$2; BASE=$3; shift 3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-C:/Users/User/Desktop/flatscan_out/r11/modelsize}
PY=/c/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe
CLI=${CLI:-$OUT/bin/pipeline_cli.exe}
DB=/c/Users/User/Desktop/docunet_bench
LOCK=$DB/.heavy_lock
D=$OUT/e2e/du_$NAME
HAVE=0
release() { if [ $HAVE = 1 ]; then rm -rf "$LOCK"; HAVE=0; echo "lock released $(date +%T)"; fi; }
trap release EXIT
take() {
  until mkdir "$LOCK" 2>/dev/null; do
    if [ -f "$LOCK/owner" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK/owner") )) -gt 7200 ]; then
      echo "stale lock ($(cat "$LOCK/owner")), removed $(date +%T)"; rm -rf "$LOCK"; continue
    fi
    sleep 20
  done
  HAVE=1; echo "modelsize $NAME $(date)" > "$LOCK/owner"; echo "lock taken $(date +%T)"
}
if [ ! -f "$D/run.json" ]; then
  take
  FS_ORT_THREADS=2 "$PY" -I "$REPO/tools/litert/quant_e2e.py" run docunet "$D" --models "$MODELS" --cli "$CLI" --jobs 6 "$@"
  release
fi
[ -z "$BASE" ] && exit 0
"$PY" -I "$REPO/tools/litert/quant_e2e.py" cmp docunet "$BASE" "$D" > "$D.cmp.txt"
tail -2 "$D.cmp.txt" | head -1
CH=$(grep '^CHANGED' "$D.cmp.txt" | cut -c9-)
rm -rf "${D}_sub"; mkdir -p "${D}_sub"
for n in $CH; do cp "$D/$n.png" "${D}_sub/"; done
if [ -n "$CH" ]; then
  take
  (cd $DB && "$PY" -I docunet_eval.py data_scan/scan "${D}_sub" --jobs 4 2>&1 | grep -v '^missing' | tail -3)
  release
fi
"$PY" -I "$REPO/tools/litert/quant_e2e.py" compose "$BASE/docunet_metrics.json" "$D" | tee "$D.metrics.txt"
