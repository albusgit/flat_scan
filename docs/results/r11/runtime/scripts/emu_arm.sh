#!/bin/bash
# emu_arm.sh OUT RELEASE_APK DEBUG_APK: arm64 APKs on the x86_64 emulator (AVD fs35) through its ARM translation.
# Release: install, launch, alive after 20 s, crash lines. Debug: batch mode on the self-test photo, accel cpu and gpu
# (no OpenCL: the CPU fallback). The caller holds the heavy lock.
OUT=$(cygpath -m "$1"); REL=$(cygpath -m "$2"); DBG=$(cygpath -m "$3")
export MSYS_NO_PATHCONV=1
SDK=/c/Users/User/Android/sdk; ADB=$SDK/platform-tools/adb.exe; PKG=com.albus.flatscan; DEV=/sdcard/Android/data/$PKG/files
PHOTO=C:/Users/User/Desktop/fs_r11_rt_base/app/src/main/assets/selftest/photo1.png
mkdir -p "$OUT"; EMU_PID=
cleanup() { "$ADB" -s emulator-5580 emu kill >/dev/null 2>&1; sleep 5; [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null; echo "emulator stopped $(date +%T)"; }
trap cleanup EXIT
"$SDK/emulator/emulator.exe" -avd fs35 -port 5580 -read-only -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect > "$OUT/emulator.log" 2>&1 &
EMU_PID=$!; S="$ADB -s emulator-5580"; $S wait-for-device
for i in $(seq 1 120); do [ "$($S shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ] && break; sleep 2; done
echo "booted $(date +%T) abilist $($S shell getprop ro.product.cpu.abilist | tr -d '\r') nativebridge $($S shell getprop ro.dalvik.vm.native.bridge | tr -d '\r')"
crash() { timeout 30 $S logcat -d > "$OUT/logcat_$1.txt"; grep -E "FATAL|AndroidRuntime: |SIGSEGV|SIGILL|Abort message|UnsatisfiedLink" "$OUT/logcat_$1.txt" | head -10 > "$OUT/crashes_$1.txt"; echo "$1 crash lines: $(wc -l < "$OUT/crashes_$1.txt")"; head -5 "$OUT/crashes_$1.txt"; }
$S install "$REL" 2>&1 | tail -1; $S logcat -c
$S shell am start -W -n $PKG/.MainActivity 2>&1 | grep -E "Status|TotalTime"
sleep 20; echo "release pid after 20 s: $($S shell pidof $PKG | tr -d '\r')"; crash rel
$S uninstall $PKG >/dev/null 2>&1; $S install "$DBG" 2>&1 | tail -1; $S logcat -c
$S shell am start -W -n $PKG/.MainActivity 2>&1 | grep -E "Status|TotalTime"; sleep 8
$S exec-in run-as $PKG sh -c "mkdir -p files/batch/selftest && cat > files/batch/selftest/photo1.png" < "$PHOTO"
runset() {  # TAG ACCEL MAXWAIT
  local T0=$(date +%s) gone=0
  $S shell am start -S -n $PKG/.MainActivity --es batch selftest --es accel $2 --ez shade false --es ext .png --es out $1 2>&1 | tail -1; sleep 8
  while [ $(( $(date +%s) - T0 )) -lt $3 ]; do
    $S shell ls $DEV/batch_out/$1/done.txt >/dev/null 2>&1 && break
    if [ -z "$($S shell pidof $PKG | tr -d '\r')" ]; then gone=$((gone + 1)); [ $gone -ge 3 ] && { echo "$1: app died"; break; }; else gone=0; fi
    sleep 5
  done
  echo "$1 wall $(( $(date +%s) - T0 )) s"; timeout 90 $S pull $DEV/batch_out/$1 "$OUT/" >/dev/null; cat "$OUT/$1/done.txt" 2>/dev/null || echo "$1: no done.txt"
}
runset a_cpu cpu 1500
runset a_gpu gpu 1500
echo "debug pid at the end: $($S shell pidof $PKG | tr -d '\r')"; crash dbg
grep -E "flatscan" "$OUT/logcat_dbg.txt" | grep -E "infer:|unavailable" | head -12 | cut -c1-200
