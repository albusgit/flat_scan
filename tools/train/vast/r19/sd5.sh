#!/bin/bash
# SD-HR pass 5 on the round-19 box (GPU 0, after the page finder at lr 6e-4 was stopped for the better 3e-4 run):
# from pass 4's step 5000 (best DocReal, net only: LD 22.72 -> 21.57 vs pass 3), 3 x pass 4's learning rate (it moved
# little), per batch 5 real flat photos + 5 UVDoc real curls (all 19k) + 2 anchors + 8 generated (occluders over
# > 8 % of the page left out), the ink-line curvature term doubled (FS_CURVE_FG 0.5: printed lines as rulers).
# Restarts from last.pt; pushes checkpoints and logs to the store every 10 min and at the end.
set -u
set +u; source /venv/main/bin/activate; set -u
W=/root/fs/work; D=/root/fs/data; T=/root/fs/code/tools/train; out=$W/sd5; mkdir -p $out
export FS_ROUND4=$D/round4 FS_DATA=$D FS_SDHR_CKPT=$D/models/highres_121x177.ckpt PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 FS_CURVE_FG=0.5
STEPS=20000
log() { echo "$(date +%H:%M:%S) $*" >> $W/box.log; }
[ -f $out/init.pt ] || python -c "
import torch
ck = torch.load('$W/sd4/step005000.pt', map_location='cpu', weights_only=False)
torch.save({'state_dict': ck['state_dict'], 'step': ck['step']}, '$out/init.pt')"
push() { for f in $out/log.jsonl $out/train.log; do [ -f $f ] && hf upload SealAItest/fs-r19 $f runs/sd5/$(basename $f) --repo-type dataset > /dev/null 2>&1; done
         for f in $out/step*.pt; do [ -f $f ] && [ ! -f $f.pushed ] && hf upload SealAItest/fs-r19 $f runs/sd5/$(basename $f) --repo-type dataset > /dev/null 2>&1 && touch $f.pushed; done; }
( while sleep 600; do push; done ) & PUSH=$!
for try in 1 2 3 4 5; do
  resume=(); [ -f $out/last.pt ] && resume=(--resume $out/last.pt)
  ( cd $T/flat2 && python train_run2.py --synth $D/real/real_sdhr --held $D/val/held2 --held_real $D/real/real_sdhr_val \
      --uvdoc $D/uvdoc/prep --anchor $D/anchors/anchor2 --init $out/init.pt \
      --gpugen 8 --gpugen_ring 512 --gpugen_new 4 --gen_dev cuda:0 --pages $D/pages2 --bg_pack $D/real/train_neg2.npz \
      --out $out --counts 5,5,2 --micro 8 --steps $STEPS --lr 3e-5 --enc_lr 1e-5 \
      --warmup 300 --eval 1000 --save 0 --ckpt 500 --real_sets '' --workers 16 "${resume[@]}" >> $out/train.log 2>&1 )
  [ -f $out/step$(printf %06d $STEPS).pt ] && { log "SD-HR pass 5 done"; break; }
  log "SD-HR pass 5 stopped (try $try): $(tail -2 $out/train.log | tr "\n" " " | cut -c1-300)"; sleep 20
done
kill $PUSH; push; log "SD-HR pass 5: pushed"
