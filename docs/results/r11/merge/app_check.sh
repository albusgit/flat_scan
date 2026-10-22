#!/bin/bash
# build the merged x86_64 debug APK, then the emulator batch checks (inside locked.sh)
S=C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad/merge
cd /c/Users/User/Desktop/fs_r11
T0=$(date +%s)
JAVA_HOME=/c/Users/User/Android/jdk17 ./gradlew assembleDebug -Pabis=x86_64 > $S/gradle.log 2>&1
rc=$?
echo "gradle exit $rc $(( $(date +%s) - T0 )) s"
tail -5 $S/gradle.log
[ $rc = 0 ] || exit 1
APK=app/build/outputs/apk/debug/app-debug.apk
ls -l $APK; cp $APK $S/merged-x86_64-debug.apk
PHOTOS="$PHOTOS" bash $S/emu.sh $S/merged-x86_64-debug.apk $S/emu
