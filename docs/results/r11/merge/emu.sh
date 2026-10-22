#!/bin/bash
# Merge stream: install the merged debug APK on the AVD fs35 (x86_64), debug batch mode on the self-test photo and
# three of the 34 photos with Acceleration=CPU, rule net on (default) and off. Run inside locked.sh.
#   emu.sh APK OUT
APK=$(cygpath -m "$1"); OUT=$(cygpath -m "$2")
export MSYS_NO_PATHCONV=1
SDK=/c/Users/User/Android/sdk
ADB=$SDK/platform-tools/adb.exe
PKG=com.albus.flatscan
DEV=/sdcard/Android/data/$PKG/files
PHOTO=C:/Users/User/Desktop/fs_r11/app/src/main/assets/selftest/photo1.png
mkdir -p "$OUT"
EMU_PID=
cleanup() {
  "$ADB" -s emulator-5580 emu kill >/dev/null 2>&1
  sleep 5
  [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null
}
trap cleanup EXIT
"$SDK/emulator/emulator.exe" -avd fs35 -port 5580 -read-only -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect > "$OUT/emulator.log" 2>&1 &
EMU_PID=$!
S="$ADB -s emulator-5580"
$S wait-for-device
for i in $(seq 1 120); do
  [ "$($S shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ] && break
  sleep 2
done
echo "booted $(date +%T) abi $($S shell getprop ro.product.cpu.abi | tr -d '\r')"
$S install -r "$APK" 2>&1 | tail -1
$S logcat -c
$S shell am start -W -n $PKG/.MainActivity 2>&1 | tail -2
sleep 8
echo "pid after launch: $($S shell pidof $PKG | tr -d '\r')"
$S exec-in run-as $PKG sh -c "mkdir -p files/batch/selftest && cat > files/batch/selftest/photo1.png" < "$PHOTO"
for n in $PHOTOS; do
  $S exec-in run-as $PKG sh -c "mkdir -p files/batch/s3 && cat > files/batch/s3/$n.jpg" < "C:/Users/User/Desktop/round4_robustness/real_photos/$n.jpg"
done
$S shell run-as $PKG ls -l files/batch/selftest files/batch/s3
run() {  # run TAG SET MAXWAIT_S EXTRA...
  local tag=$1 set=$2 maxw=$3; shift 3
  local T0=$(date +%s) gone=0
  $S shell am start -S -n $PKG/.MainActivity --es batch $set --es accel cpu --ez shade false --es out $tag "$@" 2>&1 | tail -1
  sleep 8
  while [ $(( $(date +%s) - T0 )) -lt $maxw ]; do
    $S shell ls $DEV/batch_out/$tag/done.txt >/dev/null 2>&1 && break
    if [ -z "$($S shell pidof $PKG | tr -d '\r')" ]; then gone=$((gone + 1)); [ $gone -ge 3 ] && { echo "$tag: app died"; break; }; else gone=0; fi
    sleep 4
  done
  echo "$tag wall $(( $(date +%s) - T0 )) s"
  timeout 60 $S pull $DEV/batch_out/$tag "$OUT/" >/dev/null
  cat "$OUT/$tag/done.txt" 2>/dev/null || echo "$tag: no done.txt"
}
run st_rn selftest 400 --es ext .png
run st_norn selftest 400 --es ext .png --ez rules false
run s3_rn s3 900 --es ext .jpg
echo "pid at the end: $($S shell pidof $PKG | tr -d '\r')"
timeout 30 $S logcat -d > "$OUT/logcat_all.txt"
grep -E "FATAL|AndroidRuntime|SIGSEGV|Abort|tombstone" "$OUT/logcat_all.txt" | head -20 > "$OUT/crashes.txt"
echo "crash lines: $(wc -l < "$OUT/crashes.txt")"
grep -E "FlatScan" "$OUT/logcat_all.txt" | grep -iE "batch|rule|infer" | head -40
