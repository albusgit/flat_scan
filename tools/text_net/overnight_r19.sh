#!/bin/bash
# Round 19: the rule + text-line net fine-tuned from the shipped one (run1/best.pt) with product shots and line drawings
# as hard negatives (synth_products.py; the poster's bottles were bent by lines found on them), unattended on the RTX
# 5060. Each step is skipped when its output exists (a stopped run resumes). Log: $ROOT/overnight_r19.log.
#   bash overnight_r19.sh
set -u
P=C:/Users/User/envs/flatscan_torch/Scripts/python.exe
HERE=C:/Users/User/Desktop/flat_scan/tools/text_net
ROOT=E:/fs_datasets/textnet
OLD=C:/Users/User/fs_textnet/pools          # round 12's pools (text, rules, their validation)
POOL=$ROOT/pools_r19                        # the product pools (E:, C: is full)
RUN=$ROOT/run_r19
LOG=$ROOT/overnight_r19.log
mkdir -p "$POOL" "$RUN"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
"$P" -I -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(8 * 3600)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null' EXIT
log "start"
pool() {  # name, n, seed, products share
  local f=$POOL/$1
  if [ -f "$f.done" ]; then log "pool $1: exists"; return; fi
  log "pool $1: $2 samples, products $4"
  TN_PRODUCTS=$4 TN_PROCS=12 "$P" -I "$HERE/gen_pool.py" "$ROOT/corpus.json" "$f" "$2" "$3" > "$f.gen.txt" 2>&1
  local rc=$?
  log "pool $1: exit $rc"
  [ $rc -eq 0 ] && touch "$f.done"
}
pool vprod 600 19001 0.5
pool prod1 8000 19101 0.35
if [ ! -f "$RUN/finished" ]; then
  log "train: 4 h from run1/best.pt"
  TN_INIT=$ROOT/run1/best.pt TN_HOURS=4 TN_LR=6e-4 TN_VAL_PROD=$POOL/vprod TN_WORKERS=10 \
    "$P" -I "$HERE/train_text.py" "$RUN" "$OLD/text1,$POOL/prod1" "$OLD/rule1" "$OLD/vtext" "$OLD/vrule" > "$RUN/train.txt" 2>&1
  rc=$?
  log "train: exit $rc, $(tail -1 "$RUN/train.txt")"
  [ $rc -eq 0 ] && touch "$RUN/finished"
fi
log "done"
