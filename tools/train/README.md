# FlatScan training kit (rented GPU)

Three fine-tunes, each aimed at a failure seen on real photos against CamScanner / Genius Scan (2026-10-26):

| job | net | what it fixes | starts from | picked by |
|---|---|---|---|---|
| `shade/` | ShadeNet (shadow map) | dark cast shadows left behind (the Hi-TEC refill sheet: a hand shadow at 23 % of the paper) | the app's own weights, read back from `shadenet.onnx` | real shadow photos with ground truth |
| `rules` (`../text_net`, `rules/synth_mag.py`) | RuleNet (rules + text rows) | curved titles straightened ("FAST, POWERFUL"), drawing outlines taken for rules (the waved bar), display type marked as rules | the shipped net (`textnet/run1/best.pt`) | validation pools incl. magazine pages, then the app pipeline on the photos |
| `flat/` | UVDoc (page crop + flattening) | creases and curls left in (the coupon's fold, "SAVE $7"), the app-crop mismatch | UVDoc's released weights | two-pass MS-SSIM on dir300, as the app runs it |

Every trainer uses `fast.py`: bf16 autocast, channels_last, `torch.compile`, fused AdamW, TF32, cuDNN autotune, an
EMA of the weights (what is evaluated and exported), linear warm-up + cosine; data made once and augmented on the
GPU (shadow), or decoded / cropped by many workers with pinned memory (rules, flat). Tested on one GPU; `shade` and
`flat` are wired for `torchrun` (DDP, one process per GPU) but that path has not been run - rent one GPU.

**Lesson applied from Round 11** (the SD-HR fine-tune got better on its own validation and worse in the app): every
job starts from the app's current net, trains on inputs made the way the app makes them, and keeps the checkpoint
that scores best on *real photos run the app's way*, not on its training data.

## Measured here (RTX 5060, smoke runs)

- Shadow: the shipped net scores err **0.0760** on 242 real shadow pairs (jung2018, osr2020, RDD test; no removal:
  0.2206). The same net with the map clamp lowered from 0.3 to 0.12 scores **0.0574** (the clamp, not the net, was
  holding dark shadows back). 300 fine-tune steps on a 96-sample bank: 0.0525; with the size schedule, 0.0502
  at step 300.
- Rules: 600 fine-tune steps with 400 magazine pages: magazine-page rule F1 h 0.693 -> 0.703, v 0.651 -> 0.673;
  the old text / rule pools within 0.002 of where they started.
- Flat: the shipped UVDoc, two passes on 100 dir300 photos: MS-SSIM **0.555**, flow LD 20.1 px (net only, no line
  stage). ~15 steps/s at batch 8 after compiling.
- Export round trips: shade |onnx - torch| <= 0.06 levels; UVDoc grid 2e-4; rule net 0.0024 (fp16 weights).

## On the GPU box

Box: 1x RTX 4090 / L40S / A100 / H100, **>= 32 vCPU** (the pools are CPU-bound), >= 64 GB RAM, >= 150 GB disk,
image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` (or any PyTorch >= 2.4 with Triton).

```bash
# on Windows: the bundles (once): python tools/train/vast/pack.py E:/fs_datasets/vast_pack
scp -P PORT E:/fs_datasets/vast_pack/flatscan_*.tar root@HOST:/root/
# on the box
mkdir -p /data /work && tar -xf flatscan_data.tar -C /data && tar -xf flatscan_code.tar -C /root
cd /root/flat_scan/tools/train/vast && bash setup.sh
nohup bash all.sh > /work/all.log 2>&1 &   # the three jobs at once -> /root/flatscan_results.tar
```
`all.sh` runs `run_shade.sh`, `run_rules.sh` and `run_flat.sh` side by side, each with a third of the cores, so
one job's CPU phase (pages, pools, the UVDoc fetch) overlaps another's GPU phase. Logs: `/work/{shade,rules,flat}.log`.
Each job can also run alone.

What makes them fast:
- Shadow: progressive resizing (`--sizes 256x192:0.6,384x288:0.25,512x384:0.15`, batch scaled per size). The net is
  bound by activation size: measured here 991 samples/s at 256x192 vs 239 at 512x384. The data is made once and
  composed on the GPU; best.pt is chosen on real photos; early stop (`--patience`) in the last size phase.
- Flat: channels_last + torch.compile (measured 227 vs 172 samples/s), bf16, the real-photo score threaded, early stop.
- Rules: smaller old pools for a fine-tune, validation on 2000 pages, early stop (`TN_PATIENCE`).

Times for the box are estimates from the local speeds, not measured. Each script prints what to bring back
(the exported ONNX files and the logs).

## Back on Windows: shipping a model

1. LiteRT: `tools/litert/convert.py` in `C:/Users/User/Android/litert_venv` (`shade`, `rule --rulenet PATH`,
   `uvdoc`) -> `app/src/main/litert/models/*.tflite`; parity with `tools/litert/compare*.py`.
2. Shadow net: lower the map clamp in `core/src/pipeline.cpp` `apply_shade_net` (0.3) to the `--floor` it was
   trained for (0.12).
3. Desktop check before any release: `pipeline_cli` with `FS_MODELS` / `FS_SHADE_NET` / `FS_RULE_NET` on the three
   comparison photos and the 121-photo regression set (`flatscan_out` scripts of 2026-10-26), old vs new side by side.

## Data

Training (shippable): synthetic pages from the generators here; the UVDoc dataset (MIT). Validation only (research
licences or none stated; never trained on): jung2018, osr2020, RDD test (shadows); dir300, DocReal (flattening).
Test only: DocUNet (never used for choosing). WarpDoc has no licence: not used.

## What the nets must keep, and what they may change

Keep exactly as printed: pictures and their colours (a sky is not paper); designed curves (an arched title, a
circular stamp, a logo's swash); display type; drawings; coloured paper and tints; highlighter; handwriting.
Straighten / even out: the paper's own bending (curl, folds, creases, a book's spine), the lighting (shadows,
falloff, glare), the camera's perspective. Ignore as evidence of the paper's shape: anything inside a picture,
arched or slanted text, big display letters, barcodes, signatures, show-through, a second sheet behind.
Trust: table rules, form lines, frames around pictures and boxes, body-text rows, page and column edges.

## Beyond CamScanner: next rounds

- **Re-textured UVDoc geometries.** UVDoc_final ships `uvmap`/`img_geom` for its 4033 captured shapes: render them with
  our magazine / coupon / form pages and new backgrounds - unlimited new training photos with exact ground truth,
  in the content we fail on.
- **A phone capture set.** 100-200 pages photographed on the S24 FE (folded, curled, in shadow) plus the flat scan of
  each: the only data that measures the app as people use it. It becomes the selection set for every net.
- **High-resolution flattening data** for SD-HR (SyntheticDoc's shards: slow host, fetch on the box).
- **Per-region filtering:** the paper / picture split the filter now makes by rules (cleanup.cpp) as a small net
  trained on the generators' layouts (they know which pixel is picture).
- **Curled corners:** a corner-curl generator (3-D, not 2-D) for the rule net and a crop rule that keeps curled
  corners inside the page.
