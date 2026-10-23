#!/bin/bash
# Gate B (docs/accel/PLAN.md), the full run: the Python ONNX Runtime 1.22.0 path (pipeline_cli built from
# BASE_REF) against pipeline_cli with the nets in-process (this checkout), CPU backend, bit for bit, on the
# 35 photos (+ tools/round4_check.py) and on DocUNet (+ the pages against docunet_bench/work/new_fold2).
#
#   tools/desktop_check/gate_b_full.sh OUT [BASE_REF]      (Git Bash; BASE_REF default accel/main)
#
# Needs: this checkout's tools/desktop_check/build (see docs/accel/NOTES.md), the ORT 1.22.0 venv
# (PY122), the round-4 venv (PY, numpy + cv2), the round-4 reference REF35 (photo_pipeline.py outputs).
# Takes the machine-wide heavy lock around the DocUNet runs. About 6 min (35 photos) + 12 min (DocUNet)
# on the i5-14400F with --jobs 4.
set -e
OUT=$(realpath -m "$1"); BASE_REF=${2:-accel/main}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SCR=/c/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad
PY122=${PY122:-$SCR/ort122/Scripts/python.exe}
PY=${PY:-/c/Users/User/envs/flatscan_torch/Scripts/python.exe}
REF35=${REF35:-$SCR/ref_net}
LOCK=/c/Users/User/Desktop/docunet_bench/.heavy_lock
G=$REPO/tools/desktop_check
NEW_CLI=$G/build/pipeline_cli.exe
mkdir -p "$OUT"

# 1. the baseline pipeline_cli / pagefind_cli from BASE_REF (an export, not a worktree)
if [ ! -x "$OUT/base/tools/desktop_check/build/pipeline_cli.exe" ]; then
  rm -rf "$OUT/base"; mkdir -p "$OUT/base"
  git -C "$REPO" archive "$BASE_REF" core tools/desktop_check | tar -x -C "$OUT/base"
  W=$(cygpath -w "$OUT/base/tools/desktop_check")
  cat > "$OUT/build_base.bat" <<EOF
@call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul
cd /d $W
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/User/Android/opencv/build/x64/vc16/lib
cmake --build build -j 8
EOF
  cmd //c "$(cygpath -w "$OUT/build_base.bat")"
fi
BASE_CLI=$OUT/base/tools/desktop_check/build/pipeline_cli.exe
BASE_PF=$OUT/base/tools/desktop_check/build/pagefind_cli.exe

# 2. 35 photos (ShadeNet on, as the app)
"$PY122" -I "$G/gate_b.py" ref s35 "$OUT/s35_ref" --cli "$BASE_CLI" --pf-cli "$BASE_PF" --shade --jobs 4
"$PY" -I "$G/gate_b.py" new s35 "$OUT/s35_new" --cli "$NEW_CLI" --shade --jobs 4
"$PY" -I "$G/gate_b.py" cmp s35 "$OUT/s35_ref" "$OUT/s35_new" | tee "$OUT/cmp_s35.txt"
for k in ref new; do
  rm -rf "$OUT/r4_$k"; mkdir -p "$OUT/r4_$k"; cp "$OUT/s35_$k"/*.json "$OUT/r4_$k/"; rm -f "$OUT/r4_$k/500c2810-image.json"
  "$PY" -I "$REPO/tools/round4_check.py" "$OUT/r4_$k" "$REF35" > "$OUT/round4_$k.txt"
done
diff "$OUT/round4_ref.txt" "$OUT/round4_new.txt" && echo "round4_check: identical" || echo "round4_check: DIFFERS"

# 3. DocUNet (FS_NO_SHADOWS, PNG, as run_ours.py), under the heavy lock
until mkdir "$LOCK" 2>/dev/null; do echo "waiting for heavy lock $(date +%T)"; sleep 30; done
trap 'rmdir "$LOCK"' EXIT
"$PY122" -I "$G/gate_b.py" ref docunet "$OUT/du_ref" --cli "$BASE_CLI" --pf-cli "$BASE_PF" --jobs 4
"$PY" -I "$G/gate_b.py" new docunet "$OUT/du_new" --cli "$NEW_CLI" --jobs 4
rmdir "$LOCK"; trap - EXIT
"$PY" -I "$G/gate_b.py" cmp docunet "$OUT/du_ref" "$OUT/du_new" --images C:/Users/User/Desktop/docunet_bench/work/new_fold2 | tee "$OUT/cmp_docunet.txt"
# Any photo whose vs_images is not "same" must be explained and scored (docunet_bench/docunet_eval.py,
# under the lock, --jobs 4 at most); all same => DocUNet LD 5.162 / MS-SSIM 0.5898 / AD 0.1893 unchanged.
