#!/bin/bash
# LiteRT GPU delegate check of the three .tflite files on the x86_64 emulator (AVD fs35) or on a phone.
#   tools/litert/gpu_check.sh OUT [PHOTO ...]
#     OUT: results folder (run.log, logcat.txt, device.txt, raw/); PHOTO: net-input names of INPUTS (default doctr__25)
#   env: INPUTS  net-input cache (tools/accel/net_inputs.py output), default Stream A's scratch cache
#        SERIAL  adb serial of a connected phone (arm64-v8a): no emulator is started; default: boot fs35 (x86_64)
#        EMU_GPU emulator -gpu mode, default host
#        GPU_FLAG gpu_check's backend flag: default --gl on the emulator (no OpenCL there, see gpu_check.c), none on a phone
#        LITERT  folder with litert-1.4.0/x/jni and litert-gpu-1.4.0/x/jni (the extracted AARs)
# Per model: gpu_check without input (the delegate's partition log: all nodes in 1 partition = no CPU
# fallback), then on each photo's input with the GPU delegate (fp32, MAX_PRECISION) and with the LiteRT CPU
# kernels; the outputs are scored against ONNX Runtime on the desktop (compare.py device). Holds the
# machine-wide heavy lock while the emulator runs and kills it at the end (also on failure).
set -u
OUT=$(cygpath -m "$1"); shift
PHOTOS=${*:-doctr__25}
REPO=$(cygpath -m "$(cd "$(dirname "$0")/../.." && pwd)")
export MSYS_NO_PATHCONV=1
SDK=/c/Users/User/Android/sdk
ADB=$SDK/platform-tools/adb.exe
NDK=$SDK/ndk/27.2.12479018/toolchains/llvm/prebuilt/windows-x86_64/bin
PY=/c/Users/User/Android/litert_venv/Scripts/python.exe
LITERT=${LITERT:-C:/Users/User/Android/litert}
INPUTS=${INPUTS:-C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad/accel_a/in}
LOCK=/c/Users/User/Desktop/docunet_bench/.heavy_lock
D=/data/local/tmp/litert_check
MODELS="uvdoc_static syntheticdoc_hr_static shadenet"
model_of() { case $1 in *_uv_p1|*_uv_rc0) echo uvdoc_static;; *_sdhr_rc2) echo syntheticdoc_hr_static;; *_shade) echo shadenet;; esac; }
mkdir -p "$OUT/raw" "$OUT/bin"
if [ -n "${SERIAL:-}" ]; then ABI=arm64-v8a; CL=aarch64; GF=${GPU_FLAG:-}; else ABI=x86_64; CL=x86_64; SERIAL=emulator-5582; GF=${GPU_FLAG:---gl}; fi
"$NDK/clang.exe" --target=$CL-linux-android26 -O2 -Wall -o "$OUT/bin/gpu_check" "$REPO/tools/litert/gpu_check.c" -ldl || exit 1
"$PY" -I "$REPO/tools/litert/compare.py" raw "$INPUTS" "$OUT/raw" $PHOTOS || exit 1
S="$ADB -s $SERIAL"
EMU_PID=
cleanup() {
  if [ "$SERIAL" = emulator-5582 ]; then
    $S emu kill >/dev/null 2>&1
    sleep 5
    [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null
    rmdir "$LOCK"
    echo "lock released $(date +%T)"
  fi
}
if [ "$SERIAL" = emulator-5582 ]; then
  until mkdir "$LOCK" 2>/dev/null; do echo "waiting for heavy lock $(date +%T)"; sleep 30; done
  trap cleanup EXIT
  echo "lock taken $(date +%T)"
  "$SDK/emulator/emulator.exe" -avd fs35 -port 5582 -read-only -no-window -no-audio -no-boot-anim -no-snapshot \
    -gpu "${EMU_GPU:-host}" > "$OUT/emulator.log" 2>&1 &
  EMU_PID=$!
  $S wait-for-device
  for i in $(seq 1 120); do
    [ "$($S shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ] && break
    sleep 2
  done
fi
{
  echo "device $SERIAL abi $($S shell getprop ro.product.cpu.abi | tr -d '\r') model $($S shell getprop ro.product.model | tr -d '\r')"
  echo "GLES $($S shell dumpsys SurfaceFlinger 2>/dev/null | grep -m1 -i 'GLES:' | tr -d '\r')"
} | tee "$OUT/run.log"
$S shell "rm -rf $D; mkdir -p $D/lib $D/models $D/io"
$S push "$LITERT/litert-1.4.0/x/jni/$ABI/libtensorflowlite_jni.so" "$LITERT/litert-gpu-1.4.0/x/jni/$ABI/libtensorflowlite_gpu_jni.so" $D/lib/ >/dev/null
for m in $MODELS; do $S push "$REPO/app/src/main/litert/models/$m.tflite" $D/models/ >/dev/null; done
$S push "$OUT/bin/gpu_check" $D/ >/dev/null
$S shell chmod 755 $D/gpu_check
for f in "$OUT"/raw/*.in.f32; do $S push "$f" $D/io/ >/dev/null; done
$S logcat -c
for m in $MODELS; do
  echo "== $m: GPU delegate partitions $GF" | tee -a "$OUT/run.log"
  $S shell "cd $D && ./gpu_check $D/lib $D/models/$m.tflite $GF 2>&1" | tr -d '\r' | tee -a "$OUT/run.log"
done
for f in "$OUT"/raw/*.in.f32; do
  b=$(basename "$f" .in.f32)
  m=$(model_of "$b")
  for mode in gpu cpu; do
    echo "== $b $m $mode" | tee -a "$OUT/run.log"
    flag=$GF; [ $mode = cpu ] && flag=--cpu
    $S shell "cd $D && ./gpu_check $D/lib $D/models/$m.tflite $D/io/$b.in.f32 $D/io/$b.$mode.out.f32 $flag 2>&1" | tr -d '\r' | tee -a "$OUT/run.log"
    $S pull $D/io/$b.$mode.out.f32 "$OUT/raw/" >/dev/null 2>&1
  done
done
$S logcat -d -s tflite:* > "$OUT/logcat.txt" 2>&1
$S shell "rm -rf $D"
"$PY" -I "$REPO/tools/litert/compare.py" device "$INPUTS" "$OUT/raw" --log "$OUT/device.txt"
