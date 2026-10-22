# Round 11 FLATNET: fine-tuning SyntheticDoc-HR on real photos

Status: DONE. Decision: NOT shipped (better on DIR300 crops, worse end to end on DocUNet). Shipped syntheticdoc_hr stays.

## Plan (written before training)

**Where the nets come from.** SD-HR (`syntheticdoc_hr.onnx`) is the SyntheticDoc high-res model (Woortmann, Magne,
Sorkine-Hornung, ECCV 2026; code MIT, github.com/tanguymagne/SyntheticDoc; checkpoint
`igl.ethz.ch/projects/SyntheticDoc/highres_121x177.ckpt`, no licence stated). It is UVDoc's net (8.45 M params,
BatchNorm) trained on 1M Blender renders of arXiv pages. UVDoc (MIT) is the same architecture at 488x712.
`tools/flatnet/export_onnx.py` re-exports the original checkpoint **byte-identical** to the shipped
`syntheticdoc_hr.onnx` (cmp), so a fine-tuned checkpoint drops in with the same contract and size.

**Data that could be fetched tonight.** SyntheticDoc shards (ETH research collection) download at ~0.6 MB/s and
the site rate-limits (HTTP 429); Doc3D on Hugging Face is gated and 14 GB per BM shard; the UVDoc dataset is
13 GB on the same slow ETH host. Real photo + flat-original pairs are small: **WarpDoc** (1020 phone photos of
printed pages in 6 distortion types, with the digital originals; HF mirror `34data/WarpDoc_merged`, 4.0 GB,
7 MB/s; source FDRNet, Xue et al. CVPR 2022; licence: none stated, research data).

**Recipe chosen: real-photo fine-tuning with pseudo ground truth (+ optional AADD axis-alignment loss).**
SD-HR never saw a real photo. For each WarpDoc pair the original SD-HR grid (teacher) is corrected by dense
optical flow (OpenCV DIS, illumination-normalised, confidence-weighted smoothing at grid scale, 2 iterations)
between the photo unwarped by the grid and the digital original: the corrected grid maps every point of the
original to the photo pixel that shows it, i.e. the label the net should output (`pseudo_gt.py`). Teacher
NCC (unwarped vs original) median 0.24 -> 0.83 after correction. Pairs with NCC < 0.6 or no gain are dropped
(928 of 1020 kept; most rejects are rotated pages where the flow cannot recover). The net is fine-tuned on
these at its own input size with the app's crop rule (page box + 2-10 % margin, jitter, turned when wider than
tall, INTER_AREA to 1024x1440), BatchNorm frozen, shadow decoder untouched, L1 grid loss as the original
training, and optionally AADD's axis-alignment loss (Wang et al., AAAI 2026: variance of the row coordinate along
each grid row / column coordinate along each column, in the target's page coordinates). AADD reports LD 6.90
-> 6.57 on DocUNet from that loss alone (UVDoc base, full training).

**Validation** (DocUNet never used for training or selection): DIR300 (300 photos + scans, library set),
net-only unwarp on fixed crops from the original model (`valset.py`), scored with the DocUNet scorer
(`docunet_bench/docunet_eval.py`). Checkpoints are selected on a 100-image DIR300 subset (every 3rd image);
the full 300 and DocUNet (app pipeline) are run last.

## Results

### Net only, DIR300 (validation)

Fixed crops from the original SD-HR (`valset.py crops`), net-only unwarp, DocUNet scorer. Full set = 300; d100 =
every 3rd image (100). Proxy LD = DIS-flow stand-in (`proxy_ld.py`, single core, for ranking only).

| model | set | MS-SSIM | LD | AD | proxy LD |
|---|---|---|---|---|---|
| shipped SD-HR | DIR300 300 | 0.6349 | 7.121 | 0.2804 | |
| shipped SD-HR | d100 | 0.6380 | 7.592 | 0.2893 | 23.61 |
| A (L1, lr 2e-5) epoch 1 | d100 | 0.6684 | 5.832 | 0.2241 | 20.93 |
| A epoch 3 | d100 | | | | 18.76 |
| A epoch 6 | d100 | | | | 18.58 |
| A epoch 10 | d100 | 0.6881 | 4.795 | 0.1714 | 18.29 |
| B (L1 + 10 x AADD, lr 2e-5) epoch 10 | d100 | 0.6889 | 4.812 | 0.1707 | 18.31 |
| C (L1, lr 5e-5, 16 epochs) epoch 16 | d100 | **0.6963** | **4.451** | **0.1628** | |
| C epoch 16 | DIR300 300 | 0.6876 | 4.471 | 0.1610 | |

Training curves (held-out 10 % of WarpDoc, pseudo-GT L1 in [-1, 1] units): A 0.0361 (epoch 0 = shipped) ->
0.0313 (1) -> 0.0289 (3) -> 0.0276 (6) -> 0.0276 (10); B the same within 0.0001 (val AADD term 7.2e-4 -> 6.7e-4
in both: the L1 fine-tune already straightens as much as the extra loss does at this weight).

Selection: C epoch 16 is best on DIR300 d100 by LD, MS-SSIM and AD (A epoch 6 not scored). B's AADD term did not help
(4.812 vs 4.795). DIR300 300: shipped 7.121 -> C16 4.471 LD (MS-SSIM 0.6349 -> 0.6876, AD 0.2804 -> 0.1610). Shipped
d100 numbers are from the earlier run (`val_dir300/score_orig.log`), same crops. Commands: `valset.py run|score SET OUT TAG CKPT`
(scorer = `docunet_bench/docunet_eval.py`, `--jobs 4`, under the heavy lock).

### ONNX export

`export_onnx.py` of the original checkpoint is byte-identical to the shipped `syntheticdoc_hr.onnx` (`cmp`, 17,027,370 B).
`export_onnx.py OUT.onnx runs/C_lr5/ep16.pt` gives the same I/O contract (image 1x3x1440x1024 -> grid 1x2x121x177,
shading), fp16 weights, 17.0 MB; fp16 vs torch fp32 grid max diff 0.002. Not copied into app/ (not shipped).

### End to end on DocUNet (app pipeline, r11/main build, rule net on)

Net caches recomputed into new folders `work/nets_C16`, `work/nets_A10x` (`docunet_nets.py nets TAG MODEL`; the same
script with the shipped model reproduces `nets_rc_f` uv2 with max abs diff 0, 130 pages). Config = Round 10's:
`FS_RULE_NET=1 FS_NO_SHADOWS=1 FS_UVPASS1=nets_frame_f/{n}_uv0 FS_UVPASS1_FRAME=1 FS_UVRECROP=1 FS_UVOUT=nets/{n}_uv2
FS_UVMODEL=2 FS_UVOUT2=nets/{n}_uv0 FS_UVMODEL2=0` (`tools/flatnet/run_docunet.py`, `docunet_eval.py` --jobs 4).
Mean over all 130 pages (the 5.122 headline of Round 10 is the 128-photo composition; same pipeline on the shipped
nets gives 5.729 over 130, and r10_rn's own per-page file gives 5.733 over the same 130 pages; r11/main differs from r10_rn on one page, 64_1).

| nets | MS-SSIM | LD | AD |
|---|---|---|---|
| shipped SD-HR (r11/main, rule net on) | 0.5863 | 5.729 | 0.2145 |
| A epoch 10 | 0.5514 | 6.432 | 0.2467 |
| C epoch 16 | 0.5426 | 7.036 | 0.2753 |

C16 vs shipped, per page beyond 0.25 LD: 34 wins (largest 55_2 -5.1, 12_1 -4.7, 65_2 -3.4, 7_2 -3.4), 79 losses
(largest 10_2 +27.8, 3_2 +11.9, 7_1 +9.2, 13_2 +8.3, 9_2 +7.7); median change +0.68.

Reading (not tested further): the fine-tune sharpens the net on crops cut from the original SD-HR (the DIR300
validation protocol), but in the app pipeline the re-crop comes from UVDoc and the selection/guards were tuned on the
shipped net's output statistics; fine-tuned grids lose on DocUNet's strongly folded/curled pages. Pseudo-GT of WarpDoc
(mild distortions) probably does not cover DocUNet's.

### Data licence

WarpDoc (FDRNet, Xue et al. CVPR 2022; HF mirror 34data/WarpDoc_merged): licence not stated, research data. Models
trained on it are research artefacts and would need a licence check before any release; none is shipped here.

### Not verified

No phone run; no .tflite made (not shipped); A epoch 6 and DIR300 300 for A/B not scored; DocUNet scoring uses the
MATLAB-free port (LD within +-0.1 of the official).
