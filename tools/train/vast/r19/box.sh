#!/usr/bin/env bash
# box.sh: the round-19 retrain on a rented GPU box (vast.ai), start to end, everything that can overlap overlapping.
#   STORE=hf:USER/fs-r19 HF_TOKEN=... bash box.sh        (or STORE=r2:BUCKET with RCLONE_CONFIG_R2_* set; see README)
# The PC uploaded the bundle beforehand (stage.py): code, page images, real packs, SAM labels, UVDoc, start weights.
#
# Timeline (one GPU: the stages run in turn; two GPUs: page finder on GPU 0, SD-HR on GPU 1, side by side; four GPUs: each
# training on two, its generator in a process of its own on the second (gpugen/feeder.py), so no step waits for it):
#   0  in parallel: the Python packages (uv), the bundle (parallel download), unpack as each part lands
#   1  smoke (2 min): generator speed on this GPU, 30 trainer steps of each net, VRAM peak -> sizes the batches
#   2  page finder v3 (pagefind2/train.py: GPU generator ring + real photos with SAM / interval labels + negatives)
#   3  SD-HR pass 4 (flat2/train_run2.py: GPU generator ring + real flat photos + UVDoc + anchors)
#   all along: checkpoints, logs and a status line pushed to the store every 10 min (the PC evaluates them through
#   the app while the box trains: tools/train/vast/r19/pull.py); a stage that dies restarts from its last.pt.
set -uo pipefail
# vast.ai's PyTorch images keep torch in /venv/main (no `python` on PATH without it): use it, and install into it
[ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ] && { set +u; source /venv/main/bin/activate; set -u; }
UVSYS=$([ -n "${VIRTUAL_ENV:-}" ] || echo --system)
STORE=${STORE:?STORE=hf:USER/REPO or r2:BUCKET}
ROOT=${ROOT:-/root/fs}; DATA=$ROOT/data; WORK=$ROOT/work; CODE=$ROOT/code
PF_STEPS=${PF_STEPS:-30000}; SD_STEPS=${SD_STEPS:-30000}
# the page finder fine-tunes pf_r19_best: 6e-4 as r19b did (2e-3, the from-scratch rate, undid it: SmartDoc 7 -> 44 failures by step 5000)
PF_LR=${PF_LR:-6e-4}
mkdir -p "$DATA" "$WORK" "$CODE"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$WORK/box.log"; }

# ---------------------------------------------------------------- store: get / put
get() {   # get REMOTE_PATH LOCAL_DIR
  case $STORE in
    hf:*) hf download "${STORE#hf:}" --repo-type dataset --include "$1" --local-dir "$2" > /dev/null ;;
    r2:*) rclone copy "r2:${STORE#r2:}/$1" "$2/$(dirname "$1")" --transfers 32 --multi-thread-streams 16 ;;
  esac
}
put() {   # put LOCAL_PATH REMOTE_DIR
  case $STORE in
    hf:*) hf upload "${STORE#hf:}" "$1" "$2/$(basename "$1")" --repo-type dataset > /dev/null ;;
    r2:*) rclone copy "$1" "r2:${STORE#r2:}/$2" --transfers 8 ;;
  esac
}

# ---------------------------------------------------------------- 0: packages and data, in parallel
log "start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd';'), $(nproc) cores, $(free -g | awk '/Mem/{print $2}') GB RAM"
(
  { command -v uv > /dev/null || pip install -q uv; } && uv pip install $UVSYS -q timm opencv-python-headless onnx onnxruntime huggingface_hub[hf_xet] hf_transfer \
    && { python -c "import torch, sys; caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}; sys.exit(0 if caps and all(f'sm_{a}{b}' in torch.cuda.get_arch_list() for a, b in caps) else 1)"       || uv pip install $UVSYS -q torch torchvision --index-url https://download.pytorch.org/whl/cu128; }     && python -c "import torch, torchvision, timm; print('torch', torch.__version__, torchvision.__version__, torch.cuda.get_arch_list())"
) > "$WORK/pip.log" 2>&1 &
PIP=$!
command -v rclone > /dev/null || [ "${STORE%%:*}" != r2 ] || (curl -fsSL https://rclone.org/install.sh | bash > /dev/null 2>&1)
wait $PIP || { log "pip failed: $(tail -3 "$WORK/pip.log")"; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=1
fetch() {   # fetch NAME: the bundle part NAME.tar (stage.py) -> unpacked into DATA (or CODE for code)
  local dest=$DATA; [ "$1" = code ] && dest=$CODE
  [ "$1" != code ] && [ -f "$WORK/got_$1" ] && { log "have $1"; return 0; }   # a restarted box.sh: data parts once
  get "$1.tar" "$WORK/dl" && tar -xf "$WORK/dl/$1.tar" -C "$dest" && rm -f "$WORK/dl/$1.tar" \
    && touch "$WORK/got_$1" && log "got $1" || log "FAILED $1"
}
fetch code
T=$CODE/tools/train
# UVDoc's 12 GB of photos straight from its source (MIT; parallel range requests), beside everything else
if [ -f "$WORK/uvdoc.pid" ] && kill -0 "$(cat "$WORK/uvdoc.pid")" 2> /dev/null; then   # a fetch started before box.sh
  ( while kill -0 "$(cat "$WORK/uvdoc.pid")" 2> /dev/null; do sleep 10; done; log "UVDoc photos (early fetch): $(tail -1 "$WORK/uvdoc.log")"
    touch "$WORK/uvdoc.done" ) &
else
( python "$T/flat/fetch_zip.py" https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip "$DATA/uvdoc" img/ --jobs 32 > "$WORK/uvdoc.log" 2>&1 \
    && log "got UVDoc photos" || log "UVDoc fetch FAILED (SD-HR then runs without UVDoc: SD_COUNTS=7,0,3)"; touch "$WORK/uvdoc.done" ) &
fi
UV=$!
PARTS=()
for part in weights pages real uvdoc anchors val; do fetch $part & PARTS+=($!); done
wait "${PARTS[@]}"
export FS_ROUND4=$DATA/round4 FS_DATA=$DATA FS_SDHR_CKPT=$DATA/models/highres_121x177.ckpt PYTHONUNBUFFERED=1
# rented boxes show 128-224 cores: one thread per core in every process would fight over the share rented
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# ---------------------------------------------------------------- the push loop (checkpoints and logs, every 10 min)
(
  while sleep 600; do
    for d in "$WORK"/pf3 "$WORK"/sd4; do
      [ -d "$d" ] || continue
      for f in "$d"/log.jsonl "$d"/train.log "$WORK/box.log"; do [ -f "$f" ] && put "$f" "runs/$(basename "$d")"; done
      for f in "$d"/step*.pt; do
        [ -f "$f" ] && [ ! -f "$f.pushed" ] && put "$f" "runs/$(basename "$d")" && touch "$f.pushed"
      done
    done
  done
) &
PUSH=$!

# where each training and its generator run: CUDA_VISIBLE_DEVICES and --gen_dev ('' = the generator inside the trainer)
NG=$(nvidia-smi -L | wc -l)
if [ "$NG" -ge 4 ]; then PF_GPUS=0,1 SD_GPUS=2,3 PF_GEN=cuda:1 SD_GEN_DEV=cuda:1
elif [ "$NG" -ge 2 ]; then PF_GPUS=0 SD_GPUS=1 PF_GEN=${PF_GEN_SAME-${GEN_SAME:-}} SD_GEN_DEV=${SD_GEN_SAME-${GEN_SAME:-}}
else PF_GPUS=0 SD_GPUS=0 PF_GEN=${GEN_SAME:-} SD_GEN_DEV=${GEN_SAME:-}; fi
gen_arg() { [ -n "$1" ] && echo "--gen_dev $1"; }

# ---------------------------------------------------------------- 1: smoke
log "smoke"
FS_PAGES=$DATA/pages2 FS_BG_PACK=$DATA/real/train_neg2.npz python "$T/gpugen/gen.py" "$WORK/gen_pf.jpg" --n 32 --bench 10 2>&1 | grep -v Warn | tee -a "$WORK/box.log"
( cd "$T/gpugen" && FS_PAGES=$DATA/pages2 FS_BG_PACK=$DATA/real/train_neg2.npz python check_sdhr.py "$WORK/gen_sd.jpg" --n 16 --bench 5 2>&1 | grep -v Warn | tee -a "$WORK/box.log" )
put "$WORK/gen_pf.jpg" runs/smoke; put "$WORK/gen_sd.jpg" runs/smoke
# the page finder compiled or not, timed on this GPU (on the RTX 5060 compiling gained nothing, SPEED.md; a faster GPU
# may spend more of its time launching the small net's kernels, which compiling fuses): compiled only if >= 3 % faster
PF_COMPILE=${PF_COMPILE:-auto}
if [ "$PF_COMPILE" = auto ]; then
  PF_COMPILE=0; best=0
  for c in 0 1; do
    r=$(cd "$T/pagefind2" && CUDA_VISIBLE_DEVICES=$PF_GPUS python train.py --bench 100 --compile $c $(gen_arg "$PF_GEN") --gpugen_ring 512 --gpugen_new "${PF_NEW:-8}" --pages "$DATA/pages2" \
          --bg_pack "$DATA/real/train_neg2.npz" --real "$DATA/real/train_real_sam.npz" --neg "$DATA/real/train_neg2.npz" --neg_share 0.06 \
          --real_share 0.45 --bs "${PF_BS:-48}" --sdf 1 --zoom 0.3 --no_teacher 1 2>> "$WORK/bench.log" \
        | python -c "import json, sys; print([json.loads(l)['steps_per_s'] for l in sys.stdin if 'steps_per_s' in l][-1])" 2>> "$WORK/bench.log")
    log "page finder, compile $c: ${r:-failed} steps/s"
    if [ -n "$r" ] && python -c "import sys; sys.exit(0 if $r > $best * 1.03 else 1)"; then PF_COMPILE=$c; best=$r; fi
  done
  log "page finder: compile $PF_COMPILE"
fi

# ---------------------------------------------------------------- 2 and 3: the two trainings (restart on failure)
VAL="$DATA/val/val_smartdoc.npz $DATA/val/val_doccorner_clean.npz $DATA/val/val_synth.npz"
pf() {
  local out=$WORK/pf3; mkdir -p "$out"
  for try in 1 2 3 4 5; do
    local resume=(); [ -f "$out/last.pt" ] && resume=(--resume "$out/last.pt")
    ( cd "$T/pagefind2" && python train.py --gpugen_ring 4096 --gpugen_new "${PF_NEW:-8}" --pages "$DATA/pages2" --bg_pack "$DATA/real/train_neg2.npz" \
        --real "$DATA/real/train_real_sam.npz" --neg "$DATA/real/train_neg2.npz" --neg_share 0.06 --real_share 0.45 \
        --val $VAL --out "$out" --steps "$PF_STEPS" --bs "${PF_BS:-48}" --lr "$PF_LR" --sdf 1 --zoom 0.3 --eval 2500 --no_teacher 1 --compile "$PF_COMPILE" $(gen_arg "$PF_GEN") \
        --init "$DATA/weights/pf_r19_best.pt" "${resume[@]}" >> "$out/train.log" 2>&1 )
    grep -q "\"step\": $PF_STEPS" "$out/log.jsonl" 2>/dev/null && { log "page finder done"; return 0; }
    log "page finder stopped (try $try): $(tail -2 "$out/train.log" | tr '\n' ' ' | cut -c1-300)"; sleep 20
    # twice down with the generator on its own GPU: from then on the generator inside the trainer (tested on the PC)
    [ "$try" -ge 2 ] && [ -n "$PF_GEN" ] && { PF_GEN=''; log "page finder: generator back inside the trainer"; }
  done
}
sd() {
  local out=$WORK/sd4; mkdir -p "$out"
  for try in 1 2 3 4 5; do
    local resume=(); [ -f "$out/last.pt" ] && resume=(--resume "$out/last.pt")
    ( cd "$T/flat2" && python train_run2.py --synth "$DATA/real/real_sdhr" --held "$DATA/val/held2" --held_real "$DATA/real/real_sdhr_val" \
        --uvdoc "$DATA/uvdoc/prep" --anchor "$DATA/anchors/anchor2" --init "$DATA/weights/sdhr_p3s30k.pt" \
        --gpugen "${SD_GEN:-10}" --gpugen_ring 512 --gpugen_new "${SD_NEW:-4}" $(gen_arg "$SD_GEN_DEV") --pages "$DATA/pages2" --bg_pack "$DATA/real/train_neg2.npz" \
        --out "$out" --counts "${SD_COUNTS:-4,3,3}" --micro "${SD_MICRO:-8}" --steps "$SD_STEPS" --lr 1e-5 --enc_lr 3e-6 \
        --warmup 300 --eval 1000 --save 0 --ckpt 500 --real_sets '' --workers "$(( $(nproc) > 32 ? 16 : $(nproc) > 16 ? 12 : $(nproc) - 2 ))" \
        "${resume[@]}" >> "$out/train.log" 2>&1 )
    [ -f "$out/step$(printf %06d "$SD_STEPS").pt" ] && { log "SD-HR done"; return 0; }
    log "SD-HR stopped (try $try): $(tail -2 "$out/train.log" | tr '\n' ' ' | cut -c1-300)"; sleep 20
    [ "$try" -ge 2 ] && [ -n "$SD_GEN_DEV" ] && { SD_GEN_DEV=''; log "SD-HR: generator back inside the trainer"; }
  done
}
if [ "$NG" -ge 2 ]; then
  log "$NG GPUs: page finder on $PF_GPUS (generator ${PF_GEN:-inline}), SD-HR on $SD_GPUS (generator ${SD_GEN_DEV:-inline}), side by side"
  CUDA_VISIBLE_DEVICES=$PF_GPUS pf & P1=$!
  # SD-HR starts at once with the UVDoc photos already here (train_sdhr.Mixed takes the whole ones; a restart takes
  # more): ETH's server can be slow (19 KB/s on the round-19 box)
  ( CUDA_VISIBLE_DEVICES=$SD_GPUS sd ) & P2=$!
  wait $P1 $P2
else
  log "1 GPU: page finder, then SD-HR"
  pf; sd
fi

# ---------------------------------------------------------------- the last push
kill $PUSH 2> /dev/null
for d in "$WORK"/pf3 "$WORK"/sd4; do
  for f in "$d"/log.jsonl "$d"/train.log "$d"/best.pt "$d"/step*.pt; do
    [ -f "$f" ] && [ ! -f "$f.pushed" ] && put "$f" "runs/$(basename "$d")"
  done
done
put "$WORK/box.log" runs
log "all done: results in the store under runs/ (the box can be destroyed)"
