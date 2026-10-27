#!/usr/bin/env bash
# Everything on the GPU box in one go: unpack, setup, then shade, rules and flat each as its own pipeline - data on
# the CPU (shade pages + bank, the rule / text pools, the UVDoc fetch), then training. The data stages run side by
# side, the cores split between the two CPU-heavy ones; the training stages take turns on the GPU (flock on
# /work/gpu.lock), so each trains with the whole GPU and the RAM holds one training set at a time, and the first job
# with its data ready trains while the others are still making theirs. One results file to download at the end.
#   bash all.sh [JOBS=shade,rules,flat]
# Expects /root/flatscan_data.tar and /root/flatscan_code.tar. Writes /root/flatscan_results.tar (the exported
# .onnx models, logs and scores). Logs: /root/all.log and /work/<job>.log.
set -uo pipefail
exec > >(tee -a /root/all.log) 2>&1
JOBS=${1:-shade,rules,flat}
export FS_DATA=/data WORK=/work
mkdir -p /data /work
[ -d /data/round4 ] || tar -xf /root/flatscan_data.tar -C /data
[ -d /root/flat_scan ] || tar -xf /root/flatscan_code.tar -C /root
V=/root/flat_scan/tools/train/vast
[ -f /data/textnet/corpus_linux.json ] || bash "$V/setup.sh"
N=$(nproc); HALF=$(( N / 2 > 2 ? N / 2 : 2 ))
t0=$(date +%s)
mins() { echo $(( ($(date +%s) - t0) / 60 )); }
# one job: its data stage, then its training stage once the GPU is free
job() {
  local j=$1; shift
  ( export "$@"
    STAGE=data bash "$V/run_$j.sh" || exit 1
    echo "=== $j data done after $(mins) min, waiting for the GPU" >> /root/all.log
    flock /work/gpu.lock env STAGE=train bash -c "echo '=== $j training from $(date +%H:%M)' >> /root/all.log; bash '$V/run_$j.sh'" ) > /work/$j.log 2>&1
  echo "=== $j exit $? after $(mins) min"
}
pids=()
for j in ${JOBS//,/ }; do
  echo "=== $j started $(date)"
  case $j in
    shade) job shade SHADE_WORKERS=$HALF & ;;
    rules) job rules TN_PROCS=$HALF RN_PROCS=$HALF & ;;
    flat)  job flat FLAT_WORKERS=$HALF & ;;
  esac
  pids+=($!)
done
# progress every 10 min while they run
while kill -0 "${pids[@]}" 2>/dev/null; do
  sleep 600
  echo "--- $(mins) min: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | $(for j in ${JOBS//,/ }; do printf '%s: %s; ' "$j" "$(tail -c 300 /work/$j.log | tr '\n' ' ' | tail -c 110)"; done)"
done
wait
cd /work
tar -cf /root/flatscan_results.tar --ignore-failed-read \
    shade/export shade/run/log.jsonl shade/train.txt shade.log \
    rules/rulenet.onnx rules/run/log.jsonl rules/run/best.pt rules/run/train.txt rules.log \
    uvdoc/export uvdoc/run/log.jsonl uvdoc/baseline.txt uvdoc/train.txt flat.log 2>/dev/null
ls -la /root/flatscan_results.tar
echo "ALL DONE in $(mins) min. Download /root/flatscan_results.tar"
