#!/bin/bash
# timing.sh BEFORE_BUILD AFTER_BUILD PAGE...  CPU time per page (filter_cli --repeat 3, best of 3, one process at
# a time, interleaved): page_params, Auto color 0.7 (before), Scan 0.7 (after), screen detection, descreen
R=C:/Users/User/Desktop/fs_r11_filters/tools/desktop_check/build
export PATH="/c/Users/User/Android/opencv/build/x64/vc16/bin:$PATH"
T=$(mktemp -d)
B=$1; A=$2; shift 2
for p in "$@"; do
  b=$($R/$B/filter_cli.exe $p $T --filters 2 --repeat 3)
  a=$($R/$A/filter_cli.exe $p $T --filters 2,6 --repeat 3 --screen 1)
  echo "$(basename $p) before: $(echo $b | grep -o '"w": [0-9]*, "h": [0-9]*, "page_params_ms": [0-9.]*\|"ms": [0-9.]*' | tr '\n' ' ')"
  echo "$(basename $p) after:  $(echo $a | grep -o '"page_params_ms": [0-9.]*\|"detect_ms": [0-9.]*\|"descreen_ms": [0-9.]*\|"ms": [0-9.]*' | tr '\n' ' ')"
done
rm -rf $T
