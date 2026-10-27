#!/usr/bin/env bash
# Rule + text-line net: fine-tune the shipped net with magazine / advert pages added (curved titles, line art,
# display type: what it got wrong). Pools once (CPU, Qt offscreen), then the GPU fine-tune: a fixed number of
# iterations (a faster GPU finishes sooner instead of training longer), RULE_HOURS only a cap, early stop with
# TN_PATIENCE. A fine-tune needs fewer of the old kinds of page than the from-scratch run (36000 / 16000): the new
# magazine pages are the point.
#   bash run_rules.sh [ITERS=40000] [TEXT=8000] [RULE=4000] [MAG=12000]
# STAGE: data | train | all (default).
set -euo pipefail
export FS_DATA=${FS_DATA:-/data}; WORK=${WORK:-/work}
T=$(cd "$(dirname "$0")/../.." && pwd)    # tools/
ITERS=${1:-40000}; NT=${2:-8000}; NR=${3:-4000}; NM=${4:-12000}
STAGE=${STAGE:-all}
P=$WORK/pools; mkdir -p "$P"
export TN_FONT_DIR=$FS_DATA/fonts RN_FONT_DIR=$FS_DATA/fonts QT_QPA_PLATFORM=offscreen
C=$FS_DATA/textnet/corpus_linux.json
export TN_PROCS=${TN_PROCS:-$(( $(nproc) - 2 ))} RN_PROCS=${RN_PROCS:-$(( $(nproc) - 2 ))}
if [ "$STAGE" != train ]; then
  gen_text() { [ -f "$P/$1.log" ] && grep -q done "$P/$1.log" || python "$T/text_net/gen_pool.py" "$C" "$P/$1" "$2" "$3"; }
  gen_text text1 "$NT" 101
  gen_text vtext 1000 9001
  export TN_LAYOUT=mag; gen_text mag1 "$NM" 303
  gen_text vmag 600 9003; unset TN_LAYOUT
  for x in "rule1 $NR 202" "vrule 800 9002"; do set -- $x
    [ -f "$P/$1.done" ] || { RN_GEN=synth2 python "$T/rule_net/gen_pool.py" "$P/$1" "$2" "$3" && touch "$P/$1.done"; }
  done
fi
if [ "$STAGE" != data ]; then
  R=$WORK/rules/run; mkdir -p "$R"
  TN_INIT=$FS_DATA/textnet/run1/best.pt TN_VAL_MAG=$P/vmag TN_ITERS=$ITERS TN_HOURS=${RULE_HOURS:-3} TN_LR=${TN_LR:-6e-4} TN_CL=1 \
  TN_WORKERS=${TN_WORKERS:-$(( $(nproc) - 2 ))} TN_BATCH=${TN_BATCH:-32} TN_SMALL_BATCH=${TN_SMALL_BATCH:-64} TN_VAL=${TN_VAL:-2000} \
  TN_PATIENCE=${TN_PATIENCE:-4} \
      python "$T/text_net/train_text_v2.py" "$R" "$P/text1,$P/mag1" "$P/rule1" "$P/vtext" "$P/vrule" 2>&1 | tee "$R/train.txt"
  python "$T/rule_net/export.py" "$R/best.pt" "$WORK/rules/rulenet.onnx"
  echo "bring back: $WORK/rules/rulenet.onnx, $R/log.jsonl, $R/best.pt"
fi
