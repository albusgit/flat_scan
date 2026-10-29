#!/bin/bash
PY=/c/Users/User/envs/flatscan_torch/Scripts/python.exe
A="--synth C:/fs_train/pf2/synth_a.npz --real C:/fs_train/pf2/train_real.npz --bench 100"
run() { echo "== $*"; $PY train.py $A "$@" 2>&1 | grep -E '^\{"variant|OutOfMemory|Error' | tail -1; }
run --bs 16 --amp none --channels_last 0 --compile 0 --fused 1 --tf32 0
run --bs 16 --amp none --channels_last 0 --compile 0 --fused 1 --tf32 1
run --bs 16 --amp bf16 --channels_last 0 --compile 0 --fused 1 --tf32 1
run --bs 16 --amp bf16 --channels_last 1 --compile 0 --fused 1 --tf32 1
run --bs 16 --amp bf16 --channels_last 1 --compile 1 --fused 1 --tf32 1
run --bs 24 --amp bf16 --channels_last 1 --compile 0 --fused 1 --tf32 1
