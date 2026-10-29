# Round 19 retrain on a rented GPU

Two trainings:
- **page finder v3:** pagefind2/train.py;
- **SD-HR pass 4:** flat2/train_run2.py.

Both draw fresh synthetic photos on the GPU while they train (gpugen/gen.py), so the box needs no synthetic data on disk.

## On this PC, before renting
1. `python stage.py D:/fs_bundle` builds the plain .tar parts, licence-clean or our own only:
   - code;
   - weights;
   - page images;
   - real photo packs (SAM outlines checked by pagefind_bench/sam_clean.py);
   - UVDoc grids;
   - anchors;
   - validation sets.

   Research sets (dir300, DocReal, WarpDoc, DocUNet, MIDV, Inv3D) are not in it.
2. `hf auth login`: you type your token, it stays on this PC.
3. `python upload.py D:/fs_bundle USER/fs-r19` uploads to a **private** dataset repo. It refuses a public one, and you can re-run it after a break to resume.

## On the box
Set `HF_TOKEN` (write access to that repo) in the instance's environment yourself, then:

    pip install -q "huggingface_hub[hf_xet]" && hf download USER/fs-r19 box.sh --repo-type dataset --local-dir . && STORE=hf:USER/fs-r19 bash box.sh

`box.sh` runs these in parallel:
- package install;
- bundle download;
- UVDoc photos straight from ETH.

Then:
- a 2-minute smoke test: generator speed, VRAM;
- both trainings:
  - 4 GPUs: each training on two, its synthetic-photo generator in a process of its own on the second (`gpugen/feeder.py`), so no step waits for it;
  - 2 GPUs: side by side, the generator inside each trainer;
  - 1 GPU: one after the other;
- every 10 minutes, checkpoints and logs pushed to the repo under `runs/`;
- a stage that dies restarts from its `last.pt`.

## While it trains (this PC)
    python pull.py USER/fs-r19 C:/fs_train/r19_box --pf_eval "..." --sd_eval "..."

Pulls `runs/`, prints each run's latest log line, and runs your evaluation command on each new checkpoint. App-level checks (dir300 / DocReal / the 121 photos) run here, never on the box.

## Knobs (environment for box.sh)
| Variable | Default | Sets |
|---|---|---|
| `PF_STEPS` | 30000 | page finder steps |
| `PF_LR` | 6e-4 | page finder learning rate (a fine-tune of pf_r19_best: 2e-3 undid it) |
| `PF_GEN_SAME` / `SD_GEN_SAME` | `GEN_SAME` | per training: the generator in its own process on its GPU (`cuda:0`) or inline (empty). On the round-19 Linux box: SD-HR 0.35 -> 0.26 s/step; the page finder (its GPU already ~97 % busy) no faster and fewer fresh samples |
| `SD_STEPS` | 30000 | SD-HR steps |
| `PF_BS` | 48 | page finder batch size |
| `SD_MICRO` | 8 | SD-HR micro-batch |
| `SD_GEN` | 10 | generated SD-HR samples per batch |
| `SD_COUNTS` | 4,3,3 | real / UVDoc / anchor per batch |
| `PF_NEW` | 8 | fresh generated photos per page finder step (the rest re-drawn from a 4096 ring) |
| `SD_NEW` | 4 | fresh generated SD-HR samples per step (ring of 512) |
| `PF_COMPILE` | auto | page finder torch.compile: auto times 100 steps each way on the box's GPU and compiles only if it is >= 3 % faster; 0 or 1 to force |
| `GEN_SAME` | (empty) | 1–2 GPUs: a device (`cuda:0`) puts the generator in its own process on the trainer's GPU. Off: on the RTX 5060 the two processes sharing one GPU ran 4 × slower |
| `FS_CURVE_FG` | 0.25 | SD-HR: weight of the curvature error along the generated pages' text lines and rules (0 turns it off) |
