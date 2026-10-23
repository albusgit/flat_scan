#!/bin/bash
# Gate E (docs/accel/PLAN.md, Stream E) on the Windows build machine, Git Bash:
#   host/gpu_stage_gate.sh OUT_DIR [smoke|full]
# 1. CPU path unchanged: desktop_check pipeline_cli (gpu_stub, CPU only) with the DocUNet baseline
#    config (docunet_bench/work/new_fold2) must decode to the same pixels as work/new_fold2/{n}.png.
# 2. GPU == CPU, byte-exact, on Mesa llvmpipe (GALLIUM_DRIVER=llvmpipe): fs_cli selftest-resample
#    (fixtures) and fs_cli gpu-stages (the photo pipeline, use_gpu off vs on, every GPU stage on)
#    on the DocUNet photos with the cached nets and on the 35 round-4 photos (no nets).
# 3. Timing on the real GPU (Mesa d3d12, the default driver): gpu-stages + gpu-bench.
# 4. Report only: the warp map's CG tolerance 1e-8 vs 1e-6 (fs_cli cg-tol, CPU).
# smoke (default): 3 / 2 photos, ~2 minutes. full: every photo (~30 min on llvmpipe).
# Needs: tools/desktop_check/build (NOTES.md) and build-host-win (host/CMakeLists.txt, MESA_DIR)
# built from this tree; the round-4 venv python for the pixel compare. Takes the machine-wide
# heavy lock (NOTES.md) for the whole run.
set -u -o pipefail
OUT=${1:?OUT_DIR}; MODE=${2:-smoke}
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=C:/Users/User/Desktop/docunet_bench; W=$B/work; CROP=$B/data_crop/crop
R4=E:/fs_datasets/round4/real_photos
PY=/c/Users/User/envs/flatscan_torch/Scripts/python.exe
PCLI=$REPO/tools/desktop_check/build/pipeline_cli.exe
FCLI=$REPO/build-host-win/fs_cli.exe
export PATH="/c/Users/User/Android/opencv/build/x64/vc16/bin:$PATH"
mkdir -p "$OUT/cpu" "$OUT/photos"; M=$(cygpath -m "$OUT")

L=/c/Users/User/Desktop/docunet_bench/.heavy_lock; HAVE=0
drop() { [ $HAVE = 1 ] && rmdir "$L" && HAVE=0; }
trap drop EXIT
until mkdir "$L" 2>/dev/null; do sleep 30; done; HAVE=1

if [ "$MODE" = full ]; then
    NAMES=$(for k in $(seq 1 65); do for m in 1 2; do [ -f "$CROP/${k}_${m} copy.png" ] && echo ${k}_${m}; done; done)
    R4P=$(ls "$R4"/*.jpg)
else
    NAMES="1_1 17_2 42_1"
    R4P="$R4/dewarpnet__1_1_copy.jpg $R4/scanner_samples__receipt.jpg"
fi
NETS="--logits $W/nets_rc_f/{n}_logits.npz --uvpass1 $W/nets_frame_f/{n}_uv0.npz --uvout $W/nets_rc_f/{n}_uv2.npz --uvout2 $W/nets_rc_f/{n}_uv0.npz"
fail=0

echo "== 1. CPU path vs work/new_fold2"
for n in $NAMES; do
    FS_NO_SHADOWS=1 FS_UVPASS1="$W/nets_frame_f/${n}_uv0.npz" FS_UVPASS1_FRAME=1 FS_UVRECROP=1 \
    FS_UVOUT="$W/nets_rc_f/${n}_uv2.npz" FS_UVMODEL=2 FS_UVOUT2="$W/nets_rc_f/${n}_uv0.npz" FS_UVMODEL2=0 \
        "$PCLI" "$CROP/$n copy.png" "$M/cpu/$n.png" "$W/nets_rc_f/${n}_logits.npz" > /dev/null
done
"$PY" -I - "$M/cpu" "$W/new_fold2" $NAMES <<'EOF' | tee "$OUT/cpu_vs_new_fold2.txt" || fail=1
import os, sys
import cv2, numpy as np
a, b, bad = sys.argv[1], sys.argv[2], 0
for n in sys.argv[3:]:
    x, y = (cv2.imread(os.path.join(d, n + ".png"), cv2.IMREAD_UNCHANGED) for d in (a, b))
    same = x is not None and y is not None and x.shape == y.shape and np.array_equal(x, y)
    bad += not same
    print(n, "identical" if same else "DIFFERENT")
print("photos", len(sys.argv) - 3, "not identical", bad)
sys.exit(1 if bad else 0)
EOF

echo "== 2. GPU vs CPU on llvmpipe"
for n in $NAMES; do cp "$CROP/$n copy.png" "$OUT/photos/$n.png"; done
GALLIUM_DRIVER=llvmpipe "$FCLI" selftest-resample > "$OUT/selftest_llvmpipe.txt" 2>&1 || fail=1
tail -2 "$OUT/selftest_llvmpipe.txt"
GALLIUM_DRIVER=llvmpipe "$FCLI" gpu-stages "$M/gpu_stages_docunet_llvmpipe.json" $(for n in $NAMES; do echo "$M/photos/$n.png"; done) $NETS \
    > "$OUT/gpu_stages_docunet_llvmpipe.txt" 2>&1 || fail=1
tail -1 "$OUT/gpu_stages_docunet_llvmpipe.txt"
GALLIUM_DRIVER=llvmpipe "$FCLI" gpu-stages "$M/gpu_stages_round4_llvmpipe.json" $(for p in $R4P; do cygpath -m "$p"; done) \
    > "$OUT/gpu_stages_round4_llvmpipe.txt" 2>&1 || fail=1
tail -1 "$OUT/gpu_stages_round4_llvmpipe.txt"

echo "== 3. timing on the real GPU (d3d12)"
"$FCLI" gpu-stages "$M/gpu_stages_docunet_d3d12.json" $(for n in $NAMES; do echo "$M/photos/$n.png"; done) $NETS --reps 3 \
    > "$OUT/gpu_stages_docunet_d3d12.txt" 2>&1 || fail=1
"$FCLI" gpu-stages "$M/gpu_stages_docunet_d3d12_default.json" $(for n in $NAMES; do echo "$M/photos/$n.png"; done) $NETS --reps 3 --default-policy \
    > "$OUT/gpu_stages_docunet_d3d12_default.txt" 2>&1 || fail=1
"$FCLI" gpu-bench "$M/photos/1_1.png" --reps 5 > "$OUT/gpu_bench_d3d12.txt" 2>&1 || fail=1
GALLIUM_DRIVER=llvmpipe "$FCLI" gpu-bench "$M/photos/1_1.png" --reps 3 > "$OUT/gpu_bench_llvmpipe.txt" 2>&1 || fail=1

echo "== 4. CG tolerance, report only"
"$FCLI" cg-tol "$M/cg_rtol_1e-6_docunet.txt" 1e-6 $(for n in $NAMES; do echo "$M/photos/$n.png"; done) $NETS > /dev/null 2>&1
tail -1 "$OUT/cg_rtol_1e-6_docunet.txt"

drop
echo "gate E: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
