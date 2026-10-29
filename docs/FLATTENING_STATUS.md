# Page flattening: where FlatScan stands (round 17, 2026-10-26)

A brief for anyone planning the next steps toward CamScanner's quality. All numbers were measured on the desktop
build of the app's pipeline (`tools/desktop_check/pipeline_cli`). A change ships only after its changed pages were
looked at side by side and better clearly outnumbered worse. Round 16's brief is at commit 9bf7dc5.

## Round 19 (2026-10-29): pictures keep their shape under the line stage
The poster (`dewarpnet__13_2`): the line stage straightened the bottles' cap edges, label bands and curved logo text,
pinching them (local area vs the nets' page, p1 / p99, bottle region: 0.872 / 1.108). Leaving their lines out is
not enough: the smooth solve still bends an area with no lines to follow the lines beside it.
- **Shape lock (ships, `PipelineOptions::lock_shapes`):** where the user paints Keep as is, the line stage's map
  becomes one rigid motion fitted to the paper just around the area, blended back over 6 % of the page's shorter
  side; the page outline is left out (moving it let the finger fill take the poster's orange edge for a hand).
  Poster: 0.982 / 1.037 with the paint alone, **0.994 / 1.005** with the lock. No paint: all 34 photos
  byte-identical. `tools/desktop_check/lock_test`.
- **Automatic picture masks (off):** the block mask (`picture_guard`, spread 18) + lock passed on the poster
  (0.989 / 1.027, p5 / p95 0.995 / 1.008) but locked 43 % of it and, on the 34, made 3 pages worse (shaded text
  pages taken for pictures) and 1 better. An outside review's stricter mask (`picture_mode` 2: colour that neither
  shading nor one ink on one paper explains, text rows and thin rules vetoed) fired on 4 pages, never on text, but
  missed the bottles (smooth colour) and the Penitentes drawing. An automatic version needs a trained picture output.
- **Creases (tried, not kept):** the poster's blue / orange panel boundary stays ~5 px off straight (p90; nets alone
  6). Robust (reweighted) thin-plate smoothness broke the map into blobs on mis-traced lines and tripped the
  distortion guard; relaxing smoothness only along folds that >= 3 lines' kinks line up on found folds on 5 of 34
  pages and changed nothing measurable. The poster's boundary bends at several soft places, not one fold.
- The arched "FAST, POWERFUL" headline is printed so (CamScanner keeps it too); the line stage leaves it alone.
- Models: the round-19 overnight page finder v2 (`tools/train/pagefind2`) and SD-HR pass 3 replace round 18's (backup:
  `E:/fs_train/r19_app/models_r18`). The app pipeline with them, 2026-10-29: dir300 MS-SSIM 0.574 / LD 19.84,
  DocReal 0.539 / 21.16. The retrained rule / text-line net (r19, product negatives) missed its gate and is not shipped.

## Round 18 (2026-10-27): the fine-tuned SD-HR ships
Trained on a rented RTX 5090 (vast.ai, about $2.20) with `tools/train/flat2/train_run2.py` and `tools/train/vast/run_sdhr2.sh`:
run 2 (12k steps, 20k new synthetic samples + UVDoc + 718 curated anchor photos), then pass 2 (15k more steps, 20k
more samples). Generator fixes from an outside review: pixel centres, perspective-correct raster, stacked bends
isometric, no dark rim at the paper's edge, pale desks, clutter, shadows on the table, tables / forms / ruled paper,
float32 targets, UVDoc pass-1 crops. Trainer: float32 coordinate head (bf16 had made most of the derivative loss
rounding), fixed source counts per batch, the original net as teacher, resumable checkpoints.

| Full app pipeline, MS-SSIM | dir300 | DocReal |
|---|---|---|
| Shipped before (original SD-HR) | 0.500 | 0.475 |
| Run 1 (36k, round 17) | 0.532 | 0.526 |
| **Pass 2 (shipped now)** | **0.551** | **0.524** |

Held-out synthetic grid error (L1 x 1000): original 41.0, run 2 16.6, pass 2 14.2.
121 photos vs the old app, every changed page viewed: 3 better, 5 worse; with `uv_prefer_margin` 3 -> 1 (now the
default) page3's bent table is fixed (4 worse left: the coupon's curled corner, two background strips, one mild skew).
New: `edge_doubt` (the two nets' borders disagree along a side by > 1.5 % of the page) shows "Check the edges" on
the result, one tap to the corner screen. Not yet calibrated on the photo sets. Still to measure: each app check
alone with pass 2, the coupon fusion rule (`fuse_min_gain`), the full LiteRT parity run (3 photos checked).

## Pipeline (photo -> page)
`core/src/pipeline.cpp` `run_photo`, in order:
1. **Page finder:** a segmentation net plus a contour give 4 corners, a paper mask and outline points per side.
2. **Stage 1 geometry**, one of:
   - **homography** between the corners;
   - **surface model** (`surface.cpp`), a sheet bent along one axis plus the camera pose, fitted to the outline;
   - **flattening-net grid**: SyntheticDoc-HR (1024×1440 in, 177×121 backward map) first, UVDoc as the fallback.
     UVDoc's whole-photo pass 1 decides SD-HR's crop. A grid is accepted when its outline agrees with the page
     finder's mask.
3. **Rules after stage 1:**
   - **Flat paper** (`flat_edge_bow`): four straight sides mean the homography replaces the net.
   - **New, `surface_min_sides`:** a surface page with fewer than 2 real paper edges uses the homography. Sides lying on the photo's own edge used to read as "straight".
   - **New, two-net border fusion** (`uvdoc.cpp` `uv_fuse_border`): where the chosen net's border misses the paper and the other net's border fits it, the other net is blended in locally.
4. **Line stage:** text rows and rules are traced and straightened (`warpmap.cpp`).
5. **Orientation:** the text-axis rule, plus a new small orientation net (`tools/orientnet`, `orient.cpp` `orient_net_turn`). It is trusted for 180° turns by itself, and for quarter turns only when the rule agrees.
6. **Shadow net**, then the "Auto color" filter.

The core renders every page A4-shaped. Its true shape is measured (`map_page_hw`), and the app's Auto shape
(the default) rescales the page to it, so a receipt or a bill is not shown as A4. See open problem 1.

## Round 17 results (121 regression photos; 591 including dir300 and WarpDoc)
| Change | Result |
|---|---|
| True page proportions: the camera correction is always applied, measured through the corners' homography | Synthetic tilted A4, bill and receipt within 1–2.5%; bill 3.8% off → 0.9% |
| `surface_min_sides` | 39 pages changed: 22 better (Hi-TEC empty band gone, tax form), 16 same, 0 worse |
| Two-net border fusion | Coupon reads "SAVE $1" (was "$7"): 3 better, 4 same, 0 worse. Without the "other net fits the mask better" condition, 3 pages got worse |
| Orientation net | 24 pages changed, all corrections, 0 wrong. dir300 turned 4 ways: 0 wrong of 1,200. Synthetic Hebrew / Latin / Cyrillic: 144/144 |
| Pass-2 crop from the page finder's corners | 5 better, 7 worse: not shipped (it helps exactly where the finder is right) |
| Corner-curl geometry from the outline alone (Coons push-out; a two-panel fold model) | Worse in both versions: not shipped |
| Grid validity (folded cells, collapsed cells, points outside) | No accepted grid folds today; it is logged only |

**R16 → now on the 8 complaint photos:** 3 better (coupon, Hi-TEC, tax form), 5 same, 0 worse.

## Which method wins (670 real photos with flat scans: dir300, DocReal, WarpDoc; MS-SSIM)
| | Mean |
|---|---|
| App default (SD-HR first) | 0.467 |
| UVDoc only | 0.489 |
| No nets | 0.424 |
| Best of these per photo (oracle) | 0.532 |
| Gradient-boosted selector on the pipeline's own JSON signals, 5-fold CV grouped by document | 0.497, and never below the default on any set |

UVDoc wins on dir300 and DocReal; SD-HR wins on WarpDoc. These sets are research-licensed, so the shipped selector must be trained on licence-clean labels. Plan: synthetic photos with exact flat pages plus licence-clean real photos (SmartDoc, Mendeley, cleared Roboflow sets; round 19). Inv3D real is CC BY-NC-SA 4.0 (KIT RADAR record), not MIT: evaluation only.

## Training (`docs/TRAINING_PLAN.md`)
- **Exact-ground-truth generator** (`tools/train/flat2`):
  - a virtual sheet is bent (flat, near flat, curl, lifted corner, folds, waves) and photographed by a pinhole camera;
  - papers: A4, letter, receipts, bills, cards, coupon strips, books, landscape; multi-script text pages with pictures, bands and barcodes;
  - targets are cropped and turned exactly as the app feeds SD-HR.
- **SD-HR fine-tune** (`train_sdhr.py`):
  - losses: grid L1, derivatives, curvature on flat kinds, border, fold-over;
  - an anchor set of real licence-clean photos with the original net as teacher (learning without forgetting);
  - the checkpoint is chosen on dir300 run through the app's two passes;
  - first full run: overnight 2026-10-26 on a local RTX 5060.
- **Next jobs:** the selector (CPU only), then UVDoc pass 1 on the same data, then the page finder (shadows, clutter), and the shadow net (trained, awaiting sign-off).

## Open problems
1. **The page's shape outside the app's Auto path.** The core's page is always A4, and only the app's Auto shape rescales it to the measured shape.
   - Measured values for the CamScanner photos: coupon 1.75, magazine 1.47, Hi-TEC 1.26. CamScanner's coupon is about 1.79.
   - Rendering at A4 and then rescaling also resamples twice.
   - In progress: the core renders at the measured shape (snapped to A4 / Letter within 4%, as Auto does).
2. **Receipts in clutter:** the Kassazettel receipt keeps the coffee beans; the page finder takes them in.
3. **SD-HR on near-flat and narrow pages:** wavy text, and receipts get 3–8% grid error on synthetic checks against 0.4–1% on A4. The fine-tune targets this.
4. **The selector** is measured but not built.
5. **Paper whiteness:** CamScanner's paper is whiter on the Hi-TEC photo.

## Questions for a reviewer
- Given the oracle gap (0.467 → 0.532), what is the best way to choose per photo between SD-HR, UVDoc, fusion and the homography? A learned selector, a better acceptance rule, or running both nets and merging their grids?
- Is the SD-HR fine-tune set up right? See the data mix, losses, checkpoint choice and anchor set in `train_sdhr.py`.
- Should the core render at the page's true shape? What breaks downstream (page edits stored as fractions, exports, book spreads)?
- What would make the page finder robust to clutter (receipt among coffee beans) without the crop-from-corners regressions?
