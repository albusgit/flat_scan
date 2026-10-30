#!/bin/bash
# Round 20: orientation net v2, unattended on the RTX 5060. 12,000 training pages (run1: 2,400) + the 300 held-font
# pages, thermal-print patches (OR_THERMAL), longer training; arm a: run1's width, arm b: wider. Each step is skipped
# when its output exists. Log: $ROOT/overnight_r20.log
set -u
P=C:/Users/User/envs/flatscan_torch/Scripts/python.exe
HERE=C:/Users/User/Desktop/flat_scan/tools/orientnet
ROOT=E:/fs_orient
PG=$ROOT/pages_r20
LOG=$ROOT/overnight_r20.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
"$P" -I -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(10 * 3600)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null' EXIT
log "start"
if [ ! -f "$PG/rendered" ]; then
  mkdir -p "$PG/train" "$PG/held"
  pids=""
  for s in $(seq 0 11); do "$P" "$HERE/render_pages.py" "$PG/train" 1000 --seed $((2000 + s)) --split train > /dev/null 2>&1 & pids="$pids $!"; done
  wait $pids   # not a bare wait: that also waits for the keep-awake process
  cp "$ROOT/pages/held/"*.png "$PG/held/"
  log "rendered $(ls "$PG/train" | wc -l) train, $(ls "$PG/held" | wc -l) held"
  touch "$PG/rendered"
fi
arm() {  # name, epochs, width
  local R=$ROOT/run_r20$1
  if [ -f "$R/finished" ]; then log "arm $1: exists"; return; fi
  mkdir -p "$R"
  log "arm $1: $2 epochs, width $3"
  (cd "$HERE" && OR_THERMAL=0.15 "$P" train.py "$PG" "$R" --epochs $2 --per_page 32 --width $3 --workers 6) > "$R/train.txt" 2>&1
  local rc=$?
  log "arm $1: exit $rc, $(tail -1 "$R/train.txt")"
  [ $rc -eq 0 ] && touch "$R/finished"
}
arm a 30 16,32,64,96,128
arm b 30 24,48,96,128,192
log "done"
