#!/usr/bin/env bash
# stage_dump.sh OUT_DIR MODELS_DIR PHOTO...: the desktop app (pipeline_cli, FS_NETS=1 as the app runs the flattening
# nets) on each photo with every stage dumped: OUT_DIR/NAME.jpg (the page), .jpg.lines.jpg (the line stage's fitted
# lines), .json (the result), .outline.npz (the page finder's outline), .map.npz (the final backward map),
# NAME_logits/_p1/_rc0/_rc2.npz (the nets' raw outputs).
# The rule / text-line net runs in the line stage as in the app (FS_RULE_NET=1) unless FS_RULE_NET=0 is set.
set -u
OUT=$1; MODELS=$2; shift 2
mkdir -p "$OUT"
RN=${FS_RULE_NET:-1}
CLI=${FS_CLI:-$(dirname "$0")/build/pipeline_cli.exe}   # FS_CLI: another build of it
export PATH="/c/Users/User/Android/opencv/build/x64/vc16/bin:$PATH"   # opencv_world4110.dll
for p in "$@"; do
  n=$(basename "${p%.*}")
  ( [ "$RN" = 0 ] && unset FS_RULE_NET || export FS_RULE_NET="$RN"
  FS_MODELS="$MODELS" FS_NETS=1 FS_DUMP_OUTLINE="$OUT/$n.outline.npz" FS_DUMP_MAP="$OUT/$n.map.npz" FS_NET_DUMP="$OUT/$n" \
    "$CLI" "$p" "$OUT/$n.jpg" net > "$OUT/$n.json" 2> "$OUT/$n.err" )
  echo "$n: exit $? $(head -c 300 "$OUT/$n.json")"
done
