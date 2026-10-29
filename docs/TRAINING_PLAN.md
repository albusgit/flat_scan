# FlatScan training plan (round 17, 2026-10-26)

What to train next, in order of expected gain per hour and per dollar. Every number below was measured on this
machine on 2026-10-26 unless marked *estimate*.

## What the measurements say

| Finding | Evidence |
|---|---|
| **Choosing the method per photo is worth more than any single net.** | 670 real photos with flat scans (dir300, DocReal, WarpDoc), each run 3 ways (app default = SD-HR first, UVDoc only, no nets). MS-SSIM: default 0.467, always-UVDoc 0.489, **best per photo 0.532**. |
| A small learned selector gets a third of that gap, at no runtime cost. | Gradient-boosted trees on the pipeline's own JSON signals, 5-fold CV grouped by document: **0.497** (dir300 0.499 → 0.550, DocReal 0.475 → 0.494, WarpDoc 0.403 → 0.406; never worse than the default). |
| No single net is best everywhere. | UVDoc beats SD-HR on dir300 (0.548 vs 0.499) and DocReal (0.496 vs 0.475); SD-HR beats UVDoc on WarpDoc (0.403 vs 0.379). |
| SD-HR's known failures. | It invents curl on flat paper (dollar bill: a flat-paper rule now overrides it). It adds waviness on nearly flat pages. It cuts curled corners off (coupon "$1" read "$7": a two-net border fusion now repairs it). It errs most on receipts and narrow objects (synthetic check: 0.03–0.08 grid error vs 0.004–0.01 on A4). |
| Retraining on the stock dataset does not help. | The 2026-10-26 UVDoc fine-tune on UVDoc's own data: dir300 MS-SSIM 0.556 → 0.561 (noise). Round 11's SD-HR pseudo-GT fine-tune got worse in the app. |
| Orientation was a real failure, now fixed locally. | 22 of 121 regression pages came out upside down or sideways. The new orientation net (trained here in 25 min) fixed all 24 cases it touched across 591 photos, with 0 wrong turns. |

## The plan

### P1. Method selector: CPU only, no rented GPU, about a day
The app already computes everything the selector chooses from on every photo: SD-HR's grid, UVDoc's grid and the
perspective page. So picking among them costs nothing at runtime.

1. **Labels from licence-clean data only.** The validation sets above are research-licensed or unlicensed, so the
   shipped selector must not be trained on them; they stay its test.
   - inv3d real: 360 photos with flat documents. CC BY-NC-SA 4.0 (not MIT, checked 2026-10-27): evaluation only.
   - Synthetic full photos with exact flat pages: `flat2/synth.py` already renders them; add a "write the whole photo
     + the flat page" mode.
   - Run each photo 3 ways with `ft.sh` and score with `gtscore.py`.
2. **Train** `tools/train/selector/select_cv.py` (sklearn trees), validate on dir300 / DocReal / WarpDoc.
   - Ship target: a mean gain on every set, and no set worse than the default.
3. **Port** as a small tree ensemble in C++: a few hundred comparisons.
   - Run it after the net check in `run_photo`, with the features already in `PipelineResult`.
4. **Add candidates later:** UVDoc+fusion, SD-HR without the line stage, and the fine-tuned SD-HR from P2.

### P2. SD-HR fine-tune on exact synthetic ground truth: rented GPU (kit: `tools/train/flat2`, `vast/run_sdhr.sh`)
**Data** (`flat2/synth.py` + `geom.py`):
- A virtual sheet of paper is bent the way paper bends and photographed by a virtual phone.
- The target grid is the exact projection of the page points, as the app crops it: pass-1 corners jittered by 2%, a 2–10% margin, landscape pages turned.
- Shapes: flat, nearly flat, book curl, lifted corner, curl + corner, one fold, two folds, wavy curl.
- Papers: A4, letter, receipts (h/w up to 5), bills, cards, coupon strips, books, landscape.
- Content: multi-script text pages (Hebrew, Latin, Cyrillic, Greek, CJK) with pictures, bands, dashed coupons, barcodes and highlighter (`textures.py`).
- Light: lighting, cast shadows, glare, blur, JPEG.
- Checked: every sample unwarps back to a flat page (`check_synth.py`), and self-hiding paper is rejected.

**Mix:** 65% synthetic, 20% UVDoc's captured real shapes, 15% anchor photos. Anchor photos are real licence-clean photos whose target is the original net's own output, so what already works is kept.

**Losses:**
- L1 on the grid;
- its first derivatives (waviness, local stretch);
- its second derivatives on flat and near-flat samples;
- extra weight on the border (cut-off corners);
- a fold-over penalty;
- L1 to the original net on anchor photos.

**Checkpoint choice:**
- Best real-photo MS-SSIM, from dir300 run with the app's two passes (UVDoc pass 1 → crop → SD-HR).
- Among checkpoints that are no worse than the start on any synthetic shape (within 3%).
- Final word: the full Windows pipeline (`ft.sh` + `review.py`) on the 121 photos + dir300 + WarpDoc, looked at page by page.

**Budget** (*estimate*): 1× RTX 4090 or L40S with ≥ 32 vCPUs.
- Data: 40k samples, about 1 h on 30 cores.
- UVDoc fetch: about 20 min.
- Training: 30k steps at batch 8, a few hours.
- Total: roughly $5–15.

**Smoke-tested here:** 400 samples generated; 60 steps on the RTX 5060 (8 GB, batch 2) with synthetic and real evaluation running.

**Run:**
```
# Windows: page textures, anchor set, bundles
python tools/train/flat2/textures.py E:/fs_orient/pages/train E:/fs_flat2/pages 6000
python tools/train/flat2/make_anchor.py E:/fs_flat2/anchor <ft runs on Commons / WildReceipt / CORD> --copy
python tools/train/vast/pack.py E:/fs_datasets/vast_pack && python tools/train/vast/pack_sdhr.py E:/fs_datasets/vast_pack
# box
tar -xf flatscan_data.tar -C /data && tar -xf flatscan_sdhr.tar -C /data && tar -xf flatscan_code.tar -C /root
cd /root/flat_scan/tools/train/vast && bash setup.sh && nohup bash run_sdhr.sh 40000 30000 8 > /work/sdhr.log 2>&1 &
```

### P3. UVDoc pass 1 on the same data: after P2, same box, about 1 h
Pass 1 decides SD-HR's crop, and it fails the same way on narrow objects and curled corners.
- Add `flat2` samples (whole-photo input, 488×712) to `tools/train/flat/train.py`.
- Choose the checkpoint the same way as P2.

### P4. Page finder: shadows, clutter, several objects
Seen failures:
- A shadow splits the page (the highlighter photo: the app asks for a manual crop);
- a receipt among coffee beans or cash;
- magazine racks.

Data:
- The round-4 page generator (`round4/seg`) with cast shadows across the page, occluding pens and hands, cluttered tables and second documents.
- DIW (5.7k real photos with page masks) for validation only.

This is the next box job after P2.

### P5. Shadow net: trained, waiting for a decision
- Real shadow pairs, error: shipped net at floor 0.3 = 0.076; new weights at floor 0.12 = 0.048.
- To do: the visual check on the shadow photos, then ship with the user's OK (`tools/train/README.md`, "shipping a model").

### P6. Orientation net: done (v1), local
- Results: 0 wrong turns in 1,200 dir300 tests at margin 2.0, 144/144 multi-script synthetic pages.
- Next: real Hebrew photos from the phone as a test set; more fonts; then let it also decide quarter turns where the text-axis rule abstains.

### The data that would help most: a phone capture set
- 100–200 pages photographed on the S24 FE: folded, curled, in shadow, receipts, Hebrew documents.
- Each with a flat scan (a flatbed or the same page held flat under glass).
- It measures the app exactly as people use it, and becomes the selection set for P1–P4.
- Nothing else on this list is as informative per hour spent.

## Rules every job follows
- **Start** from the app's current weights, and train on inputs made the way the app makes them.
- **Choose** checkpoints on real photos run the app's way, never on the training data's own validation.
- **Ship** only after a full-pipeline before/after on the 121 photos + dir300 + WarpDoc, every changed page looked at, with better clearly outnumbering worse.
- **Data:** training data is licence-clean (synthetic, MIT / Apache / CC). dir300, DocReal, WarpDoc and DocUNet are never trained on.
