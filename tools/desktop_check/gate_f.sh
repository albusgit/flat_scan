#!/bin/bash
# Gate F (docs/accel/PLAN.md, Stream F) on the Windows build machine, Git Bash:
#   tools/desktop_check/gate_f.sh OUT_DIR [smoke|full] [GATE_B_REF]
# GATE_B_REF: a Gate B reference folder for the 35 photos (gate_b.py ref s35 ... --shade, Python ORT 1.22.0;
# default: Stream B's scratchpad/accel_b/s35_ref). Builds (tools/desktop_check/build_gpu.bat):
#   build         FS_DESKTOP_GPU=OFF, CPU ORT package (the Gate B build)
#   build/f_dml   WGL (gpu_gl43.cpp, the system OpenGL driver) + ORT 1.22.0 DirectML, FS_ORT_DML
#   build/f_cuda  WGL + ORT 1.22.0 GPU (CUDA 12 / cuDNN 9), FS_ORT_CUDA FS_ORT_TENSORRT
#   build/f_mesa  MESA (Mesa EGL / GLES: d3d12, llvmpipe) + CPU ORT
# 1. Gate B unchanged on the CPU backend: each build's pipeline_cli (--backend cpu, GPU stages off) vs
#    GATE_B_REF: net outputs, pages and JSON bit for bit (gate_b.py cmp: 0 photos with a difference).
# 2. GPU stages == CPU as in Gate E: fs_cli selftest-resample on the native driver (ES and core
#    profiles) and on Mesa d3d12 / llvmpipe (worst |diff| 0); fs_cli gpu-stages (every stage on) on
#    the round-4 photos and, in full mode, the DocUNet crops with the cached nets (not identical 0);
#    pipeline_cli FS_GPU=1 --backend cpu vs GATE_B_REF: pages and JSON identical (gate_f.py pages / json).
# 3. Accelerated nets vs CPU on the same inputs (Gate A's tolerance, grid p99 < 0.25 px): net_check.exe
#    dml / cuda on tools/accel/net_inputs.py's cache, gate_f.py nets; end to end (pipeline_cli
#    --backend dml|cuda FS_GPU=1) the pipeline's decisions vs GATE_B_REF (report: gate_f.py json --decisions).
# 4. Timing table: page_bench.exe on the photos for cpu / dml / cuda x GPU stages off / on (gate_f.py table).
# smoke: 5 photos, ~6 minutes plus builds. full: the 35 photos (+ DocUNet gpu-stages), ~45 minutes (estimate).
# CUDA needs CUDA 12.x / cuDNN 9 runtime DLLs on PATH: CUDA_BIN (default: the round-4 venv's torch/lib, cu128).
# The first CUDA run on a new driver JIT-compiles ORT's kernels for sm_120 (~55 s once; the driver caches it).
# full mode takes the machine-wide heavy lock (NOTES.md) around the DocUNet gpu-stages run.
set -u -o pipefail
OUT=${1:?OUT_DIR}; MODE=${2:-smoke}
SC=C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad
REF=${3:-$SC/accel_b/s35_ref}
D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$D/../.." && pwd)
PY=/c/Users/User/envs/flatscan_torch/Scripts/python.exe
CUDA_BIN=${CUDA_BIN:-/c/Users/User/envs/flatscan_torch/Lib/site-packages/torch/lib}
NET_IN=${NET_IN:-$SC/accel_a/in}
R4=E:/fs_datasets/round4/real_photos
B=C:/Users/User/Desktop/docunet_bench; W=$B/work
export PATH="/c/Users/User/Android/opencv/build/x64/vc16/bin:$PATH"
mkdir -p "$OUT"; M=$(cygpath -m "$OUT"); w() { cygpath -w "$1"; }
fail=0
if [ "$MODE" = full ]; then
    ONLY=$(ls "$REF"/*.json | sed 's#.*/##; s#\.json$##' | grep -v '^times' | paste -sd, -)
else
    ONLY=dewarpnet__1_1_copy,docres__for_dewarping,page_dewarp__boston_cooking_a,scanner_samples__receipt,user__projector_screen
fi
NAMES=${ONLY//,/ }
PHOTOS=""
for n in $NAMES; do
    if [ -f "$R4/$n.jpg" ]; then PHOTOS="$PHOTOS $R4/$n.jpg"; else PHOTOS="$PHOTOS $SC/rc/photo.jpg"; fi
done

echo "== 0. builds"
BAT=$(w "$D/build_gpu.bat")
ORT=C:/Users/User/Android/ort
for spec in "build OFF $ORT/onnxruntime-win-x64-1.22.0" "build/f_dml WGL $ORT/onnxruntime-win-x64-directml-1.22.0 DML" \
            "build/f_cuda WGL $ORT/onnxruntime-win-x64-gpu-1.22.0 CUDA TENSORRT" "build/f_mesa MESA $ORT/onnxruntime-win-x64-1.22.0"; do
    set -- $spec
    cmd //c "$BAT" $spec > "$OUT/build_$(basename $1).log" 2>&1 || { echo "build $1 FAILED"; fail=1; }
done

echo "== 1. CPU backend vs the Gate B reference"
for b in build build/f_dml build/f_cuda build/f_mesa; do
    t=$(basename $b)
    "$PY" -I "$D/gate_b.py" new s35 "$(w "$OUT/cpu_$t")" --cli "$(w "$D/$b/pipeline_cli.exe")" --shade --jobs 4 --only $ONLY > /dev/null
    "$PY" -I "$D/gate_b.py" cmp s35 "$(w "$REF")" "$(w "$OUT/cpu_$t")" --only $ONLY > "$OUT/cmp_cpu_$t.txt" || fail=1
    echo "$t: $(tail -1 "$OUT/cmp_cpu_$t.txt")"
done

echo "== 2. GPU stages vs CPU"
F=$D/build/f_dml/fs_cli.exe; FM=$D/build/f_mesa/fs_cli.exe
"$F" selftest-resample > "$OUT/selftest_wgl_es.txt" 2>&1 || fail=1
FS_GL_PROFILE=core "$F" selftest-resample > "$OUT/selftest_wgl_core.txt" 2>&1 || fail=1
"$FM" selftest-resample > "$OUT/selftest_mesa_d3d12.txt" 2>&1 || fail=1
GALLIUM_DRIVER=llvmpipe "$FM" selftest-resample > "$OUT/selftest_mesa_llvmpipe.txt" 2>&1 || fail=1
for f in wgl_es wgl_core mesa_d3d12 mesa_llvmpipe; do echo "$f: $(head -1 "$OUT/selftest_$f.txt") / $(tail -1 "$OUT/selftest_$f.txt")"; done
if [ "$MODE" = full ]; then R4P=$(ls "$R4"/*.jpg); else R4P="$R4/dewarpnet__1_1_copy.jpg $R4/scanner_samples__receipt.jpg"; fi
"$F" gpu-stages "$M/gpu_stages_round4_wgl.json" $(for p in $R4P; do cygpath -m "$p"; done) > "$OUT/gpu_stages_round4_wgl.txt" 2>&1 || fail=1
echo "round-4 gpu-stages (native GL): $(tail -1 "$OUT/gpu_stages_round4_wgl.txt")"
if [ "$MODE" = full ]; then
    L=$B/.heavy_lock; HAVE=0
    drop() { [ $HAVE = 1 ] && rmdir "$L" && HAVE=0; }
    trap drop EXIT
    until mkdir "$L" 2>/dev/null; do sleep 30; done; HAVE=1
    mkdir -p "$OUT/du"
    DN=$(for k in $(seq 1 65); do for m in 1 2; do [ -f "$B/data_crop/crop/${k}_${m} copy.png" ] && echo ${k}_${m}; done; done)
    for n in $DN; do cp "$B/data_crop/crop/$n copy.png" "$OUT/du/$n.png"; done
    "$F" gpu-stages "$M/gpu_stages_docunet_wgl.json" $(for n in $DN; do echo "$M/du/$n.png"; done) \
        --logits $W/nets_rc_f/{n}_logits.npz --uvpass1 $W/nets_frame_f/{n}_uv0.npz --uvout $W/nets_rc_f/{n}_uv2.npz --uvout2 $W/nets_rc_f/{n}_uv0.npz \
        > "$OUT/gpu_stages_docunet_wgl.txt" 2>&1 || fail=1
    drop
    rm -rf "$OUT/du"
    echo "DocUNet gpu-stages (native GL): $(tail -1 "$OUT/gpu_stages_docunet_wgl.txt")"
fi
FS_GPU=1 "$PY" -I "$D/gate_b.py" new s35 "$(w "$OUT/cpu_gpu")" --cli "$(w "$D/build/f_dml/pipeline_cli.exe")" --shade --jobs 4 --only $ONLY > /dev/null
"$PY" -I "$D/gate_f.py" pages "$(w "$REF")" "$(w "$OUT/cpu_gpu")" $NAMES > "$OUT/pages_cpu_gpu.txt" || fail=1
"$PY" -I "$D/gate_f.py" json "$(w "$REF")" "$(w "$OUT/cpu_gpu")" $NAMES > "$OUT/json_cpu_gpu.txt" || fail=1
echo "pipeline_cli FS_GPU=1 cpu: $(tail -1 "$OUT/pages_cpu_gpu.txt"); $(tail -1 "$OUT/json_cpu_gpu.txt")"

echo "== 3. accelerated nets vs CPU"
mkdir -p "$OUT/nc_cpu" "$OUT/nc_dml" "$OUT/nc_cuda"
"$D/build/f_dml/net_check.exe" --backend cpu "$M/nc_cpu" "$(cygpath -m "$NET_IN")" $NAMES > "$OUT/nc_cpu.txt" 2>&1 || fail=1
"$D/build/f_dml/net_check.exe" --backend dml "$M/nc_dml" "$(cygpath -m "$NET_IN")" $NAMES > "$OUT/nc_dml.txt" 2>&1 || fail=1
PATH="$CUDA_BIN:$PATH" "$D/build/f_cuda/net_check.exe" --backend cuda "$M/nc_cuda" "$(cygpath -m "$NET_IN")" $NAMES > "$OUT/nc_cuda.txt" 2>&1 || fail=1
for e in dml cuda; do
    grep -q "FAILED\| cpu " "$OUT/nc_$e.txt" && { echo "$e: some net did not run on $e (see nc_$e.txt)"; fail=1; }
    "$PY" -I "$D/gate_f.py" nets "$(w "$OUT/nc_cpu")" "$(w "$OUT/nc_$e")" > "$OUT/nets_$e.txt" || fail=1
    echo "$e:"; tail -5 "$OUT/nets_$e.txt"
done
for e in dml cuda; do
    b=build/f_$e
    PATH="$CUDA_BIN:$PATH" FS_GPU=1 "$PY" -I "$D/gate_b.py" new s35 "$(w "$OUT/e2e_${e}_gpu")" --cli "$(w "$D/$b/pipeline_cli.exe")" --shade --jobs 1 \
        --backend $e --only $ONLY > /dev/null
    "$PY" -I "$D/gate_f.py" json "$(w "$REF")" "$(w "$OUT/e2e_${e}_gpu")" --decisions $NAMES > "$OUT/decisions_${e}_gpu.txt"
    "$PY" -I "$D/gate_f.py" nets "$(w "$REF")" "$(w "$OUT/e2e_${e}_gpu")" $NAMES > "$OUT/nets_e2e_${e}_gpu.txt"
    echo "$e + GPU stages end to end: $(tail -1 "$OUT/decisions_${e}_gpu.txt")"
done

echo "== 4. timing"
mkdir -p "$OUT/bench"
bench() { tag=$1; shift; mkdir -p "$OUT/bench/$tag"; "$@" --reps 3 --save "$M/bench/$tag" "$M/bench/$tag.json" $PHOTOS > "$OUT/bench/$tag.txt" 2> "$OUT/bench/$tag.err" || fail=1; }
bench cpu "$D/build/page_bench.exe" --backend cpu
bench cpu_gpu "$D/build/f_dml/page_bench.exe" --backend cpu --gpu
bench dml "$D/build/f_dml/page_bench.exe" --backend dml
bench dml_gpu "$D/build/f_dml/page_bench.exe" --backend dml --gpu
PATH="$CUDA_BIN:$PATH" bench cuda "$D/build/f_cuda/page_bench.exe" --backend cuda
PATH="$CUDA_BIN:$PATH" bench cuda_gpu "$D/build/f_cuda/page_bench.exe" --backend cuda --gpu
grep -l '"stable": false' "$OUT"/bench/*.json && { echo "a page changed between passes"; fail=1; }
"$PY" -I "$D/gate_f.py" pages "$(w "$OUT/bench/cpu")" "$(w "$OUT/bench/cpu_gpu")" > "$OUT/bench/pages_cpu_gpu.txt" || fail=1
(cd "$OUT/bench" && "$PY" -I "$D/gate_f.py" table "$(w "$OUT/timing.md")" cpu.json cpu_gpu.json dml.json dml_gpu.json cuda.json cuda_gpu.json)

[ $fail = 0 ] && echo "gate F: PASS" || echo "gate F: FAIL"
exit $fail
