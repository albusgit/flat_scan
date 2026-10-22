#!/bin/bash
# run_set.sh TAG BUILD SET FILTERS STRENGTH [EXTRA filter_cli args]
#   renders SET (a folder under OUT) with tools/desktop_check/build/BUILD/filter_cli.exe into OUT/runs/TAG/SET, scores it
R=C:/Users/User/Desktop/fs_r11_filters
P=C:/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe
O=${FS_FILTER_OUT:-C:/Users/User/Desktop/flatscan_out/r11/filters}
TAG=$1; BUILD=$2; SET=$3; F=$4; S=$5; shift 5
$P -I $R/tools/filters/eval_filters.py render $R/tools/desktop_check/build/$BUILD/filter_cli.exe $O/$SET $O/runs/$TAG/$SET --filters $F --strength $S --jobs 4 --extra "$*" &&
$P -I $R/tools/filters/eval_filters.py score $O/$SET $O/runs/$TAG/$SET --filters $F
