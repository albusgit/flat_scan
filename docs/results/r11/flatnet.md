# Round 11 FLATNET: fine-tuning SyntheticDoc-HR on real photos

Status: IN PROGRESS (notes kept as the work goes).

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

(filled in below as measured)
