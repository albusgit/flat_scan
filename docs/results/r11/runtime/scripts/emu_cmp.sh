#!/bin/bash
# emu_cmp.sh OUT TAG1=APK1 [TAG2=APK2 ...]: on the x86_64 emulator (AVD fs35, read-only, -gpu swiftshader_indirect),
# for each APK in turn: install, the debug batch mode on the bundled self-test photo (st_<tag>, accel cpu, no shadow
# net, PNG page), on DocUNet 1_1 4_2 17_2 18_1 (du_<tag>, accel cpu), and with ACCEL2 (default gpu: the fallback without
# OpenCL) on the self-test photo (st_<tag>_gpu); shade on for the self-test photo once (sts_<tag>, accel cpu). Pulls the
# outputs to OUT/<tag>/. The caller holds the heavy lock. REPS=2: the DocUNet set twice (timing).
OUT=$(cygpath -m "$1"); shift
export MSYS_NO_PATHCONV=1
SDK=/c/Users/User/Android/sdk
ADB=$SDK/platform-tools/adb.exe
PKG=com.albus.flatscan
DEV=/sdcard/Android/data/$PKG/files
PHOTO=C:/Users/User/Desktop/fs_r11_rt_base/app/src/main/assets/selftest/photo1.png
ACCEL2=${ACCEL2:-gpu}
mkdir -p "$OUT"
EMU_PID=
cleanup() {
  "$ADB" -s emulator-5580 emu kill >/dev/null 2>&1
  sleep 5
  [ -n "$EMU_PID" ] && kill $EMU_PID 2>/dev/null
  echo "emulator stopped $(date +%T)"
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
runset() {  # runset DIR SET TAG ACCEL SHADE MAXWAIT_S
  local T0=$(date +%s) gone=0
  $S shell am start -S -n $PKG/.MainActivity --es batch $2 --es accel $4 --ez shade $5 --es ext .png --es out $3 2>&1 | tail -1
  sleep 8
  while [ $(( $(date +%s) - T0 )) -lt $6 ]; do
    $S shell ls $DEV/batch_out/$3/done.txt >/dev/null 2>&1 && break
    if [ -z "$($S shell pidof $PKG | tr -d '\r')" ]; then gone=$((gone + 1)); [ $gone -ge 3 ] && { echo "$3: app died"; break; }; else gone=0; fi
    sleep 4
  done
  echo "$3 wall $(( $(date +%s) - T0 )) s"
  timeout 90 $S pull $DEV/batch_out/$3 "$1/" >/dev/null
  cat "$1/$3/done.txt" 2>/dev/null || echo "$3: no done.txt"
}
for pair in "$@"; do
  TAG=${pair%%=*}; APK=$(cygpath -m "${pair#*=}")
  D="$OUT/$TAG"; mkdir -p "$D"
  echo "== $TAG $APK $(date +%T)"
  $S uninstall $PKG >/dev/null 2>&1
  $S install "$APK" 2>&1 | tail -1
  $S logcat -c
  $S shell am start -W -n $PKG/.MainActivity 2>&1 | tail -1
  sleep 8
  $S exec-in run-as $PKG sh -c "mkdir -p files/batch/selftest && cat > files/batch/selftest/photo1.png" < "$PHOTO"
  for n in 1_1 17_2 18_1 4_2; do
    $S exec-in run-as $PKG sh -c "mkdir -p files/batch/du && cat > files/batch/du/$n.png" < "C:/Users/User/Desktop/docunet_bench/data_crop/crop/$n copy.png"
  done
  runset "$D" selftest st cpu false 300
  runset "$D" du du cpu false 900
  [ "${REPS:-1}" = 2 ] && runset "$D" du du2 cpu false 900
  runset "$D" selftest sts cpu true 300
  runset "$D" selftest st_$ACCEL2 $ACCEL2 false 600
  if [ "${TAG#new}" != "$TAG" ]; then   # the self-test (map parity, the nets' timings): new builds only
    $S shell am start -S -n $PKG/.MainActivity --ez selftest true 2>&1 | tail -1
    T0=$(date +%s)
    while [ $(( $(date +%s) - T0 )) -lt 600 ]; do $S shell ls $DEV/batch_out/selftest_report.txt >/dev/null 2>&1 && break; sleep 5; done
    timeout 30 $S pull $DEV/batch_out/selftest_report.txt "$D/" >/dev/null; echo "selftest $(( $(date +%s) - T0 )) s"; grep -E "PASS|FAIL" "$D/selftest_report.txt" | head
  fi
  echo "pid at the end: $($S shell pidof $PKG | tr -d '\r')"
  $S exec-out run-as $PKG sh -c 'cat shared_prefs/*.xml' > "$D/shared_prefs.xml" 2>/dev/null
  timeout 30 $S logcat -d > "$D/logcat_all.txt"
  grep -E "FATAL|AndroidRuntime|SIGSEGV|Abort|tombstone" "$D/logcat_all.txt" | head -20 > "$D/crashes.txt"
  echo "crash lines: $(wc -l < "$D/crashes.txt")"
  grep -E "flatscan" "$D/logcat_all.txt" | grep -E "infer:|LiteRT|unavailable" | head -30
done
