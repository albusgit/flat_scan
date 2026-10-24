#!/bin/bash
# The text-line net, unattended (a night on the RTX 5060): pools -> training -> evaluation, each step skipped when
# its output exists (a stopped run resumes). Log: $ROOT/overnight.log; $ROOT/DONE at the end; the report in
# $ROOT/eval1/report.md. Pools on the SSD (random reads while training; ~63 GB, delete when done with them).
#   bash overnight.sh
set -u
P=C:/Users/User/envs/flatscan_torch/Scripts/python.exe
HERE=C:/Users/User/Desktop/flat_scan/tools/text_net
ROOT=E:/fs_datasets/textnet
POOL=C:/Users/User/fs_textnet/pools
RUN=$ROOT/run1
OUT=$ROOT/eval1
LOG=$ROOT/overnight.log
mkdir -p "$POOL" "$RUN" "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
# no sleep while this runs (the trainer keeps the system awake too)
"$P" -I -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(16 * 3600)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null' EXIT
log "start"
pool() {  # name, generator (text | rules), n, seed
  local f=$POOL/$1
  if [ -f "$f.done" ]; then log "pool $1: exists"; return; fi
  log "pool $1: $3 samples ($2)"
  if [ "$2" = text ]; then
    "$P" -I "$HERE/gen_pool.py" "$ROOT/corpus.json" "$f" "$3" "$4" > "$f.gen.txt" 2>&1
  else
    RN_GEN=synth2 RN_PROCS=14 "$P" -I "$HERE/../rule_net/gen_pool.py" "$f" "$3" "$4" > "$f.gen.txt" 2>&1
  fi
  local rc=$?
  log "pool $1: exit $rc"
  [ $rc -eq 0 ] && touch "$f.done"
}
pool vtext text 1500 9001
pool vrule rules 1000 9002
pool text1 text 36000 101
pool rule1 rules 16000 202
if [ ! -f "$RUN/finished" ]; then
  log "train: 8.5 h"
  TN_HOURS=8.5 "$P" -I "$HERE/train_text.py" "$RUN" "$POOL/text1" "$POOL/rule1" "$POOL/vtext" "$POOL/vrule" > "$RUN/train.txt" 2>&1
  rc=$?
  log "train: exit $rc, $(tail -1 "$RUN/train.txt")"
  [ $rc -eq 0 ] && touch "$RUN/finished"
fi
if [ -f "$RUN/best.pt" ]; then
  log "eval"
  "$P" -I "$HERE/eval_text.py" "$RUN" "$POOL/vtext" "$POOL/vrule" "$OUT" > "$OUT/eval.txt" 2>&1
  log "eval: exit $?"
fi
log "done"
touch "$ROOT/DONE"
