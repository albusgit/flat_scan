# FlatScan: Android document scanner

Native Android app (Kotlin + C++/NDK). It photographs or imports a page, finds it, unwarps it with
the surface model and the warp map of the Python reference (`app_ref/code`), then cleans it up and
saves or shares it as JPG or PDF. The C++ core is a line-by-line port of the reference, with every
experimental option off: `UNWARP_PROP_*`, `UNWARP_LSD` and `UNWARP_R3_*`. The same core also builds
on Linux as a test harness. That harness produced all the parity numbers below.

**Debug APK:** `dist/flatscan-debug.apk` (arm64-v8a, minSdk 26, 54 MB: includes the 12.6 MB page-finder model and ONNX Runtime).
It has **not been run on a phone**. Round 4 ran an x86_64 build on the Android 15 emulator (CPU paths only; see [Not done / not verified](#not-done--not-verified-on-a-real-phone)).

## Result screen features (v2)

- **Filters:** a strip with live thumbnails: Original, Lighten, Auto color (default), Grayscale, High contrast, B&W.
  - Strength slider, default 70%. The last filter and strength are remembered.
  - Solid colors is a separate switch (off for B&W), with "Protect a picture" rectangles that Solid colors leaves alone.
  - Measured on the 5 test pages: 0 text marks lost at 70% and 100%; GPU and CPU differ by at most 1 level.
- **Adjust panel:**
  - Background cleanup pushes grey artefacts near paper colour to white.
  - Levels editor with a histogram and black, midtone and white handles.
  - At default settings the output is byte-identical to the unadjusted filters.
- **Hold to see the original:** press and hold shows the page with perspective correction only. A hint appears on first use.
- **Zoom:** pinch from 1x to 6x, pan, and double tap.
- **Wrinkle map:** a switch in the menu, off by default. It shows a heat map and grid of how far each area was moved, with a legend in px.
- **Re-crop:** goes back to the crop screen and keeps the corners, filter and adjustments.
- **Missing corners and gaps:** filled with clean paper using a soft-edged paper mask. Only thin slivers along the page outline are filled; thicker non-paper areas (pictures, colour bands or shadowed text that reach the page edge) are kept.
- **Pictures and colour are kept:** the inside of grey or dark photos and grey cells is not taken for paper, so it is not flattened to white; speck removal skips light strokes (coloured text) and anything not on white paper; background cleanup only whitens near-neutral pixels; B&W keeps the tones of dark and saturated areas; Solid colors leaves photos and gradients with large darker areas alone.
- **Multiple pages:**
  - Import several images from the gallery.
  - The Pages screen reorders, deletes and adds pages.
  - Export as a multi-page PDF, one JPG per page into a folder, or share them all.
- **Orientation (round 4):** sideways pages are turned upright (90°); there is **no automatic 180° turn**. A **Turn 180°** button sits on the page for upside-down results (the dock's Rotate still turns 90°). The Auto-rotate setting turns the sideways turn off.

## Round 4: robustness on real photos

What changed (brief: `round4_robustness/README_FOR_APP_AGENT.md`; reference: its `code/` and `seg/`):

1. **Crash fixes** (C++ ports of the reference fixes):
   - `paper_mask` (`core/src/corners.cpp`): white neutral, then above Otsu with low chroma, then above Otsu; the first whose largest component covers 10-99% of the frame; else the whole frame inset 2 px.
   - Global stage (`warpmap.cpp`): fewer than 20 control points skip it (identity), as do exceptions in line detection.
   - Local stage: no horizontal or no vertical lines give a zero field (text rows are skipped too, as in the reference).
2. **Orientation:** no automatic 180°; sideways detection kept; one-tap **Turn 180°** on the result screen.
3. **Quality gate** (`quality_gate` in `pipeline.cpp`, the reference rules): surface-fit median residual above 15 px (in full-resolution px), a corner more than 2% outside the photo, page area below 5% or above 120%, or the net found no page.
   - A trusted crop goes straight from import or capture to the result.
   - A doubtful crop opens the crop screen with the detected quad and the reasons ("Check the crop: ...") and uses the homography path, even if the corners are kept.
   - Settings > *Check every crop* restores the old always-show-the-crop-screen flow.
4. **Page finder** (`core/src/pagefind.{h,cpp}`, `PageFinder.kt`): `assets/models/pagefinder.onnx` runs on ONNX Runtime Android 1.22.0. The core makes its input (`segnet.to_input`) and ports `pagefind.py`:
   - bicubic logits and sigmoid at 1000 px;
   - `select_page` (component score, hole fill, gutter PCA split, sliver drop);
   - the whole-frame fallback below 2%;
   - `refine_contour` (Lab Scharr gradient, ±8 px normals, Viterbi);
   - the anti-aliased fill and `net_margin` / `edge_strength` / `edge_at_limit`.
   - Feature flag: `Features.PAGE_FINDER_NET` (default on) and Settings > *Page finder (neural net)*. When it is off, or the model cannot load or run, the colour-threshold `paper_mask` is used.
5. **Book spreads:** two pages from one photo become two document pages, each cropped, unwarped and cleaned on its own (the second page is processed right after the first and has its own quality gate).
   - Order: right page first for right-to-left books, left page first otherwise. The default comes from the app language (RTL locale); Settings > *Book spreads: right page first* changes it.
   - "Book spread: swap order" on the result screen swaps the pair.

### Measured (Windows desktop build of the same core, `tools/desktop_check`, CPU paths; outputs in `docs/results/round4/`)

| Check | Result |
|---|---|
| Page masks, app vs `seg/pagefind.py` + `pagefinder.onnx`, 34 real photos, same OpenCV 4.11 | **bit-identical**: 0 differing pixels on every photo; net input identical; page counts, fallbacks and quality signals equal |
| Same, 8 synthetic book spreads (`seg/synth.py` at 1024 px) | bit-identical; all 8 split into 2 pages in both |
| Same, against the training env (OpenCV 5.0, onnxruntime 1.31) | IoU ≥ 0.99989, at most 0.009% of pixels differ (library versions) |
| Page finder vs hand GT (33 photos) | mean IoU 0.947, 4/33 below 0.9 (colour threshold: 0.712, 18/33) |
| Pipeline completes, 34 photos | app 34/34, reference 34/34, with either finder |
| Quality gate decision, app vs reference | same on 33/34 photos with either finder |
| Orientation correct (GT `up_ccw_turns`), 33 photos | net finder: app 31, reference 30; colour finder: app 29, reference 30; no 180° turns |
| Emulator (Android 15, x86_64, CPU fallback) | import with auto-advance, gate to crop screen to homography, ONNX Runtime finder, spread split / order / swap, Turn 180°: all ran, no crash |

### Deviations from the reference (`code/`, `seg/`)

- **The second page of a spread is used:** the reference drops `info['other']`. The app runs the same outline, edge snapping and fill on it and processes it as a page.
- **Detection and surface fit run on a copy capped at 2400 px** (unchanged app design). The page finder itself runs on the full photo, as the reference does. The gate converts the fit residual to full-resolution px.
  - On the 3000 px photos the fit differs from the reference's, and that is where the gate disagrees once: `math_cheat_sheet` with the colour finder (app 8 px, reference 22 px) and `docres__208_in` with the net finder (app 3.9 px, reference 31.7 px).
- **Orientation** is decided on the homography page (round-3 brief), and CW is chosen over CCW only when the upright cue is below -0.2 (unchanged app gate). The reference decides on the stage-1 page, ungated. Results are in the table above.
- **The existing 3 px homography fallback is kept:** between 3 and 15 px the app silently uses the homography; above 15 px the gate also opens the crop screen.
- **Extra gate reasons:** the app adds "no page found" when detection fails entirely and "surface fit failed" when the C++ fit fails; neither case exists in the reference. As in the reference, the whole-frame fallback of the colour-threshold finder is not a gate reason.
- **Book-page order uses geometry:** left or right is taken from the outline centroids, turned by the first page's sideways rotation.
- **ONNX Runtime Android (full package) 1.22.0 instead of "Mobile"**, model in fp32.

### Filters: colour and pictures kept

The filters lost colour and punched white holes into pictures. The root causes and fixes are in `core/src/cleanup.cpp`, `solid.cpp` and `gpu.cpp`:
- the paper fill painted over coloured content at the page edge;
- the inside of grey or dark photos was taken for paper and whitened;
- speck removal hit coloured strokes;
- background cleanup whitened light colours;
- Solid colors flattened photos;
- B&W turned pictures white.

Contact sheets and per-page measurements were made on the 34 pages and on synthetic pages: holes inside pictures at Auto 100% went from 18% to ≤ 0.5%, colour inside pictures is kept (chroma ratio about 1.0), and text marks lost at Auto 70% went from 44 to 4. The GLSL changes compile (glslc) but the GPU path was not run.

## Round 5: user control over the crop and the unwarp

- **Pages open without re-processing.** After processing, the page is stored in `document/<id>/session/`:
  - the unwarped page (JPEG q95);
  - the paper alpha;
  - the before page;
  - the deformation grid;
  - the fitted lines.

  Opening the page again restores the session (`NativeLib.restoreSession`). It is reused only while the page's geometry key (corners, rotation, ignore areas, unwarp switch, output width, finder) is unchanged. Otherwise the page is processed again from its source.
- **Crop screen:**
  - The quality-gate note sits above the photo, never over a corner. It can be closed, and it goes away when a corner or an edge is moved.
  - Each edge has a handle. Dragging it moves the edge along its normal while both ends slide along the neighbouring edges.
  - **Circle the page** (the loop icon): draw a loop around the page. The page finder runs on that part of the photo only; when it finds no page edge there, the loop's extreme points give the corners. Either way the corners count as set by hand (homography path).
- **Lines** (new result-screen tab) draws the lines the unwarp was fitted to: ruled lines, text rows, and grid lines seen only by the global stage.
  - Tapping a line ignores it; dragging a box ignores every line inside it. Then **Flatten again** re-runs the page with those areas left out of the fit (`PipelineOptions::ignore`). In the warp map, global-stage controls and local trace / text-row samples whose straightened position falls in an area are dropped (`unwarp_map(..., ignore)`; with no areas the reference behaviour is unchanged).
  - **Unwarp on/off** (also in the ⋮ menu) turns the unwarp off for the page: perspective correction from the 4 corners only (`PipelineOptions::perspective_only`). This is for flat labels and stickers, or photos where the unwarp makes things worse.
- **Result screen:**
  - Turn 180° is a small round "180°" button in the top bar.
  - Book-spread swap moved to the ⋮ menu.
  - **Protect pictures** has its own panel (count, Undo, Remove all, Done), and tapping a box removes it.
- **Hand-held pages** (`user` receipt photo): the outline followed the thumb and the unwarp pulled the thumb into the page. The neural flattening below fixes this automatically, and the thumb is erased. Unwarp off and re-cropping are still there.

## Neural flattening, shadow removal, finger erase

Research notes (model survey with licences, verified vs inferred) are in `C:\Users\User\Desktop\model_zoo\`: `dewarping_survey.md`, `enhancement_survey.md` and `occluder_survey.md`.

**Models in the app** (`assets/models/`, ONNX Runtime; fp16 weights with fp32 compute; the outputs match PyTorch to 1e-4 for UVDoc and 4e-4 for the SyntheticDoc grid):

| Model | Paper | Licence | File | Input → output |
|---|---|---|---|---|
| UVDoc | Verhoeven et al., SIGGRAPH Asia 2023 | MIT (code and weights in the repo) | `uvdoc.onnx`, 15.8 MB | 488×712 RGB → 45×31 backward grid |
| SyntheticDoc high-res | Woortmann, Magne, Sorkine-Hornung, ECCV 2026 (same net, retrained on 1M simulated pages, 2× input) | code MIT; **the weights file states no licence**, so confirm with the authors before a release | `syntheticdoc_hr.onnx`, 17 MB | 1024×1440 RGB → 121×177 grid (its shading map is not used) |
| ShadeNet (ours) | trained here, `tools/shadow_net/` | ours | `shadenet.onnx`, 1 MB | flattened page at 512 px → its shading map |

**Pipeline** (`core/src/uvdoc.{h,cpp}`, `run_photo`):
1. The page finder gives the corners. Each net sees the corner box plus a 6% margin, turned upright if the page is landscape (both nets were trained on portrait pages). On a book spread the other page is painted out.
2. A grid is used only if its page outline agrees with the page finder's mask (IoU ≥ 0.90, or ≥ 0.85 against hand-set corners). Otherwise the outline's surface model is used, as before.
3. With both nets, a half-size stage-1 page from each is scored by text-row angle spread. SyntheticDoc high-res wins only if it is straighter by more than 0.3°; UVDoc is the safer default. The high-res grid must agree with the page finder at IoU ≥ 0.93 (its fine outline matches a right page at ≥ 0.957 on the 35 photos; a wrong crop passed 0.90 at 0.914).
4. The line refinement runs on top. Pieces of one ruled line now share one target height (RDGR's "same row" constraint; fixes the steps at table cells). The refinement is dropped when it would stretch the page locally (|det J − 1| p99 > 0.2 or more than 1% of cells above 0.2): dense staves and broken rules.
5. **Shadows:** the flattened page is divided by ShadeNet's shading map (round 6, below). SyntheticDoc's own shading maps are no longer used: they bleach colour (projector slide saturation 0.42, cookbook pages 0.40 with the high-res map).
6. **Fingers:** non-paper areas that reach into the page from its border and are skin-coloured, smooth and redder than this page's paper are filled with the surrounding paper (Telea fill at ¼ size, soft edge). Text under a finger is never invented.

**Measured on the 35 real photos** (34 round-4 photos + the hand-held receipt; desktop build of the same core, `tools/uvdoc_check.py`; text-row angle spread in degrees, lower is straighter):

| Version | median | mean | p90 | ruled-line p90 dev. (px, median) |
|---|---|---|---|---|
| Before (outline surface model + lines) | 2.15 | 2.81 | 6.86 | 11.7 |
| Perspective only | 2.60 | 3.55 | 6.50 | 17.0 |
| UVDoc + lines | 1.31 | 1.73 | 2.76 | 5.6 |
| SyntheticDoc + lines | 1.11 | 1.95 | 5.39 | 8.0 |
| **Both nets, straighter wins, + lines + shadows (default)** | **0.97** | **1.57** | **2.53** | 8.2 |

- I checked the final Auto-color pages of all 35 photos side by side, old against new. New is better or equal on every one: the thumb is gone from the receipt, bent and folded pages are flat, the hand shadow on `docres__208_in` is removed, and the money bill is cropped to the note.
- Fingers were erased on 2 of 35 photos (the receipt and a book held at its edge). Two earlier false positives were rejected by the paper-relative and smoothness tests: a beige page edge and a photo margin.
- Phone emulator (x86_64, CPU): both nets together ≈ 0.6 s; the whole page 3.4 s.
- Settings: Neural flattening on/off; both nets / UVDoc only / SyntheticDoc only; Remove shadows. Processing details show which net won and both scores.

**Evaluated and not shipped:**
- **GCDRNet GC-Net** (MIT, 3 MB fp16), a dedicated shadow / illumination net. Its full-page divide removes hand and leaf shadows completely (shadow/lit ratio 0.62 → 1.00 on `docres__208_in`, no halo). But it also bleaches pale table tints and projector-slide colour blocks, and turns dark print circles into blobs.
  - A gate that applies it only where its map is a darker, neutral cast shadow keeps the colours. At settings that protect `docscanner__42_2` and the projector slides, though, it no longer removes the shadows.
  - It would also add about 0.3–0.5 s per page and ~300 MB of peak memory.
  - Details and contact sheets: `C:\Users\User\Desktop\model_zoo\shadow_eval\`.
- **DocShadow** (MIT, 120 MB): barely removes shadows and leaves the paper beige. **DocTr / DocGeoNet / DocScanner**: non-commercial licences. **DocMamba, RDGR**: no code released; RDGR's "same row" idea is used in the line refinement.
- The small illumination net trained on synthetic hard shadows over coloured content is now shipped: ShadeNet, round 6.

### Faint printed rules (the hand-held receipt's fold)

The receipt's fold stayed sheared ("כ.א צמיגי עמוס" box): of its vertical table rules only 1 was found. They are pale grey print in a blurry 900×1600 photo blown up 2.7× to page size, and reach darkness 40–55 against the local paper. The tracer needs 45 to find a rule and 60 to follow one. The horizontal rules just made it, so the vertical shear from the fold was never measured.
- `faint_lines` (app; `PipelineOptions::faint_lines`, `unwarp_map(..., faint)`): long runs at darkness > 25 join the detection in both stages, if they are thin (median width ≤ 7 px) and isolated (paper on at least one side; text rows, which merge into runs at a low threshold, are thick or inked on both sides). Each line is traced at 0.75 × its own darkness (25..60, so dark rules keep 60).
- If the refinement then fails the distortion guard, it is retried with the reference lines alone.
- Receipt: vertical rules 1 → 6, table-line p90 8.8 → 5.3 px. 35 photos: text-row spread median 0.82 → 0.75°, mean 1.60 → 1.52°; table-line p90 median 9.7 → 5.6 px. The projector screen (5.5 → 3.6°) and `doctr__25` (0.99 → 0.38°) improve most; no page got visibly worse.
- (Corrected in round 6: "כ.א צמיגי עמוס" is printed straight. It looked slanted because the refinement took the ellipse's arcs for ruled lines and bent the area around them; see below.)
- Finger erase measured ink over the whole candidate, so the finger's own outline and the rule beside it counted; on the new geometry the receipt's thumb crossed the 0.03 limit. Ink is now measured inside the candidate, eroded by the border width.

## Round 6: shadow net, high-res flattening, line refinement that does not over-constrain

### ShadeNet: a shadow model trained on synthetic data (`tools/shadow_net/`)

- **Model:** a 0.5M-parameter U-Net (encoder to 1/32 with dilated context and a global-mean branch, decoder to 1/2, light full-resolution head). Input: the flattened page at 512 px (portrait; landscape pages turned), BGR in [0, 1]. Output: the page's shading map; the page is divided by it, brightening only, by at most 1 / 0.3. 1 MB fp16 ONNX; 135–160 ms on the emulator.
- **Data** (`synth.py`, generated on the fly): 4,800 flat pages from the round-4 page generator, plus colour content a shadow model must keep: pastel cells, highlighter, saturated / grey / near-black blocks, circles and blobs (also cut by the page edge), gradients, photos, stamps. Shading: light falloff, curl and gutter bands, and cast shadows (hands, heads, phones, straight edges, foliage) with a penumbra that widens along the edge, shadow factor ≥ 0.25. Then colour cast, exposure, blur and noise. The target is the shading divided by its lit level, so lit paper keeps its colour and the filters set white as before.
- **Training:** RTX 5060, about 2 h; L1 of log-shading + 0.5 × L1 of its gradient. Workers are numpy-only and compositing runs on the GPU (3× faster than CPU workers).
- **Synthetic test** (150 held-out pages; error of the predicted map, per-image scale aligned; content = colour or ink wrongly treated as shading, shadow = shadow left):

| | content error | shadow error |
|---|---|---|
| GCDRNet GC-Net | 0.155 | 0.119 |
| **ShadeNet** | **0.046** | **0.053** |

- **Real photos** (35, metrics from `model_zoo/shadow_eval`): with the map from the old standard SyntheticDoc model, ShadeNet after it gave the best numbers (median shadow/lit ratio 0.99 vs 0.94; edge residual 0.018 vs 0.033). The high-res model's map bleaches colour badly, so the app uses ShadeNet alone: median shadow ratio 0.92, saturation kept on every page. The dark circle on `docscanner__42_2` stays dark (an earlier checkpoint turned it into a blob; content shapes and a darkness cap fixed it).
- Weak spot: dense, very dark leaf shadows (`docres__for_deshadowing`) are only partly removed.

### SyntheticDoc high-res replaces the standard SyntheticDoc model

Its grid is 177×121 instead of 45×31 (one cell ≈ 13 px of the page instead of ≈ 50 px), so creases and tight folds are representable: on the creased flyer (`dewarpnet__14_2`) the shampoo bottle on the crease is symmetric again. Geometry only, 35 photos, the same pages:

| Stage 1 | text-row spread mean / p90 (deg) | table-line p90 dev. p90 (px) |
|---|---|---|
| UVDoc + standard SyntheticDoc (round 5) | 1.78 / 3.91 | 27.6 |
| **UVDoc preferred, high-res when clearly straighter** | **1.56 / 2.69** | **13.3** |
| High-res preferred | 1.75 / 3.96 | 13.3 |

Same model family and size (17 MB). Emulator: the pipeline takes 3.0 s, and the whole page about 13 s including the three nets on the emulator's x86 CPU; not measured on a phone.

### Line refinement: no more bent text

The refinement trusted every traced "line" to be straight on the paper, and bent the page to make it so. Fixes, app path only (`warpmap.cpp`):
- **Curves are not rules.** On a neural page, a trace whose p95 distance from its straight fit exceeds 3 px + 1 % of its length is dropped. This fixed the receipt's ellipse, which had been squashed into a rounded rectangle, along with the bent "כ.א צמיגי עמוס" text.
- **Leaning box sides** (the totals boxes by the thumb): vertical ink is widened by 2 px before the run search; a ≥ 45 px side touching one rule counts; a side whose far edge is too faint is not clipped. Pieces whose traces meet within 5 px across a < 40 px gap share one target, even when a curl moved one piece by up to 20 px.
- **Text-column edges** (`text_margins`): aligned left / right ends of text rows, chained down the page and fitted by a degree 2–4 curve within 2 px, are vertical constraints. This fixed the bowed column on the Fortune page (`docres__190_in`).
- **Robust solve** (IRLS): a whole line whose median residual disagrees with the solved field is down-weighted (Cauchy, 3 px) and the field solved again.
- **No spill-over:** a trace may run at most 40 px past a free end (a chart's dotted gridline had walked into the text beside it), and faint rules must be ≥ 2 px wide.
- **Bump guard** (`soften_bumps`): on neural pages a correction that rescales a spot by more than 35 % is faded out there.
- **Zoomed second net pass** (the page cut into overlapping halves or quarters, each run through the net as its own page): built and evaluated, **off**. The parts' grids disagree by a few pixels and the blend shears the overlap band (the receipt's table).

### Finger erase

The thumb's hole now often takes in the background strip beside it (skin share 0.499 on the receipt), so the skin share needed is 35 % (was 50 %). Every other candidate on the 35 photos that reaches 35 % is rejected by the reach or ink tests, or is background at a page corner.

## Round 7: small form boxes and colour-area boundaries as lines

Both in `warpmap.cpp`, app path, neural pages only.

- **Box corners** (`corner_pairs`): the receipt's discount column is a stack of small open boxes ("┌", ~85 × 65 px). Neither side reaches a found rule, so they were never traced and their tops, tilted by the fold, stayed tilted. A horizontal and a vertical stroke ≥ 45 px each, ≤ 9 px thick, whose ends meet within 10 px form a corner, and both are traced. The horizontal stroke must continue a found rule of its row across a gap < 80 px; without that test, bold poster letters ("B", "4") and product label outlines on the flyer counted as corners. The pieces are found unjoined (a box top 20 px from the next column's rule is its own piece) and stay separate lines; `share_ideals` still gives them their row's height. All 8 tilted box tops are straight now.
- **Colour-area boundaries** (`colour_edges`): the darkness map sees only dark ink, so on a page of colour blocks a cell boundary the net left wavy had no constraint (the flyer `dewarpnet__14_2`: the boundary beside the OGX bottle bowed by 19 px over 400 px, in an S). The page is grey-closed 9 × 9 (text and rules go, colour areas stay), and each pixel's step is 2.5 × the Lab distance between the 8-px means on either side. Steps > 30 make segments as ink does. Each piece is traced with its own threshold, and the pieces chain into one boundary when they continue each other. Rules for a chain:
  - within 2° of the axis (content tilted on the paper is no warp);
  - straight like a rule, except that a chain over 500 px may bow by min(4 % of its length, 25 px). That long a step is a cell or column boundary, and its wave is what the net left.
  - Samples where the darkness map already found the same boundary as a line are left out, because two traces of one edge a few px apart tear the page between them.
  - The outer 25 px are skipped (the paper's own edge).

  The flyer's boundary now varies by 3 px (was 19), and the shampoo can's right side on the crease is straight. A small jog remains on the can's left side at the fold.
- **35 photos** vs round 6 (same binary otherwise, geometry only): text-row spread mean 1.26° / p90 2.57° unchanged; table-line p90 dev. p90 17.6 → 16.0 px. Receipt table lines 7.0 → 1.0 px; `docscanner__27_2` 18.0 → 0.0 px. Edges fired on 14 pages, no page newly rejected by the distortion guard.
- Desktop CLI: `FS_NO_EDGES=1` turns the colour edges off; `FS_MAX_DIST=x` overrides the distortion guard; `FS_DUMP=P` also writes `P_det.png`, the line stage's |det J − 1| per 8-px cell (× 1000).

### Filters: light colours kept

Light colours (highlighter, pastel boxes and backgrounds, pale photos) were turned white. Root causes and fixes (`core/src/cleanup.cpp`, `solid.cpp`, `gpu.cpp`):
- the paper test was "chroma < 24 levels" (absolute), so a bright, smooth light-colour area counted as paper. Its own brightness became the illumination and it was flattened to white. Under warm light the same test rejected the paper itself. Paper is now found by colour relative to the paper's own colour (`paper_mask`): seeds near the most common bright colour, grown over slow colour drift but not across colour edges, and separate pieces are kept only if they match the paper around them. Thresholds scale with the colour noise of the page;
- the white point clipped the paper-noise band (`wp`..1) to white in every channel, which halved the chroma of pale colours. On light-colour areas (`tint` weight) the white point is now 1;
- background cleanup and speck removal also skip light-colour areas;
- white balance is now local (`PageParams::paper`), so mixed lighting leaves no tinted paper. Far from paper samples the global white balance is used;
- Solid colors assumed white paper on pages without light low-chroma cells. Below strength 100% the light colour areas then merged with the slightly darker paper into one greyish rectangle.

Synthetic pages (pastel boxes L 85-95 / C 8-30, highlighter, light-blue and pink text, pale photos; neutral, warm, dim and mixed lighting) at Auto 100%:
- worst chroma ratio: 0.00 before, 1.01 after;
- worst region turned white: 100% before, 0% after;
- paper pixels ≥ 250: 75% before, 99.6% after.

At Auto 70% with background cleanup 100%, the worst chroma ratio went from 0.11 to 0.88. Text darkness is unchanged. On the 34 real pages at Auto 100%, white paper went from 60% to 73.5% of the page. Remaining:
- neutral light-grey boxes are still taken for paper;
- the levels white point is applied as before;
- one real page with glare and a blue shadow (`docres__190_in`) keeps a light grey band along the shadow edge.

The GLSL compiles (glslc), but the GPU path was not run.

## Round 8: the nets find the page (DocUNet benchmark)

Measured on the DocUNet benchmark (130 photos, MS-SSIM / LD / AD against flatbed scans; a Python port of
the MATLAB scorer, checked image by image against PaperEdge's published MATLAB results). On the 20 worst
photos the automatic mode lost 93% of its gap to "photo frame = page": the page finder split a folded
page into a book spread, put a corner diagonally on a curled page, or missed a coloured band; or the
nets traced the page right and were rejected for disagreeing with it. Two receipts were turned sideways.

- **Two-pass nets** (`uv_crop_on_grid`, `PipelineOptions::uv_out_pass1`): with automatic corners, UVDoc
  first runs on the whole photo (the other page of a spread hidden); both nets then run on a crop around
  the page *it* found. A grid whose outline matches the first pass (IoU >= 0.95, the page found from two
  framings) is used even when the page finder disagrees (down to IoU 0.5); the page corners and the
  paper outline then come from the grid (`uv_defines_page`).
- **Agreement**: a net is checked against the page finder's corners as well as its mask (the mask can
  leave out a coloured band or a dark page the corners still frame).
- **Net order**: SyntheticDoc high-res first; UVDoc when the high-res grid is rejected, or when UVDoc's
  text rows are straighter by 3 deg or more (`uv_prefer_margin`). With a 0.3 deg margin UVDoc won 12
  DocUNet photos and was worse on 8 of them (LD +36 on a folded receipt): its straighter rows came with
  uneven stretching the row angles cannot see.
- **Orientation**: decided on the page stretched to its measured shape (`page_hw_ratio`). A page turns
  only if its vertical text scores 2x the horizontal (`orient_min_ratio`; sideways pages 3.8-5.9, upright
  pages up to 1.0: a dense index page turned on 1.01), a long narrow page only at 4x (monospaced receipts).
- **Folds are not book spreads** (`select_page`): a gutter splits the paper into two pages only if the
  paper is at least 0.87x as wide across it as it is long, or the frame cuts the paper. A page folded
  down its middle measures 0.58-0.80; 53 synthetic spreads (`seg/synth.py`) 0.94-1.77 when in the frame,
  all still split. On the three folded DocUNet pages the hidden half had cost 0.019 AD, more than the
  app's whole AD gap to the official SyntheticDoc-HR code; with them fixed the gap is gone.
  A wide page folded down its middle still looks like a spread (DocUNet 10-2, across 1.7).

DocUNet, 128 images (document 64 left out, as the papers); our scorer reads ~0.017 high on MS-SSIM and
~0.023 low on AD against the papers, so both scales are given:

| | MS-SSIM | LD | AD | paper scale |
|---|---|---|---|---|
| Round 7 app, automatic | 0.550 | 8.60 | 0.287 | 0.533 / 8.60 / 0.310 |
| High-res first only | 0.558 | 8.00 | 0.274 | |
| **Round 8, automatic** | **0.590** | **5.16** | **0.189** | **0.573 / 5.16 / 0.212** |
| Round 8 before the fold and orientation fixes | 0.588 | 5.30 | 0.208 | 0.571 / 5.30 / 0.231 |
| Round 8, UVDoc chosen at a 0.3 deg margin | 0.584 | 5.76 | 0.229 | 0.567 / 5.76 / 0.252 |
| Round 8, UVDoc first | 0.578 | 6.12 | 0.241 | |
| Round 8, high-res net only | 0.586 | 5.54 | 0.218 | 0.569 / 5.54 / 0.241 |
| Round 8, UVDoc only | 0.558 | 6.93 | 0.306 | |
| SyntheticDoc high-res, official code (photos already cropped to the page) | 0.589 | 5.43 | 0.192 | 0.570 / 5.51 / 0.219 (paper) |
| UVDoc (published outputs) | 0.560 | 6.90 | 0.294 | 0.544 / 6.83 / 0.315 (paper) |

The app now scores better than the best released model on all three metrics, fully automatic. The high-res net alone scores worse (LD 5.54): its grid is rejected on a folded index page
(DocUNet 18-1, LD 34 vs 4.9 with UVDoc) and on two cookbook pages of our 35 photos, where UVDoc
keeps the text rows straight (spread 5.8 -> 2.6 deg). 35 photos: text-row spread mean 1.27 -> 1.65 deg,
table-line p90 15.9 -> 14.7 px; the spread rises on pages that now keep content round 7 cut off (the tax
form's header), that round 7 turned sideways (the dollar bill), and on the DocUNet receipt 4-2 (1.9 ->
4.0 deg, but LD 41.9 -> 5.5: UVDoc's straight rows sat on a folded page). App: one extra UVDoc pass
(0.1 s on the desktop CPU).

## Acceleration: nets on the NPU / GPU, CPU stages beside the nets, more image stages on the GPU

The nets now run inside the C++ core (`core/src/infer.*`, `core/src/nets.*`) on ONNX Runtime **1.22.0**
(C++ API, the app's version), in the app and in the desktop harness. **The CPU path is unchanged, bit for
bit**: every accelerated path is opt-in and falls back to the CPU on any error. Plan, gates and how to run
them: `docs/accel/PLAN.md`, `docs/accel/NOTES.md`, `docs/results/accel/STATUS.md`. **Nothing here has run on
a phone yet**: every phone number is a HUMAN GATE (STATUS.md).

### Backends

| Where | Backend | Nets | Notes |
|---|---|---|---|
| Android, all ABIs | CPU (default) | all five | ORT C++ from the AAR, linked into `libflatscan` (the Java ORT and the Kotlin net code are gone). Same outputs as before. |
| Android arm64 | NPU: Qualcomm QNN HTP, fp16 | UVDoc (both passes), ShadeNet; SyntheticDoc-HR only where Auto measured it within tolerance | `onnxruntime-android-qnn` 1.22.0 + `qnn-runtime` 2.33.0 (HTP V68-V79). Compiled graphs are cached in `noBackupFilesDir/qnn_cache` (key: model SHA-256, SoC, QNN and ORT versions). Licence: Qualcomm "AI Stack License", shipped in `assets/licenses/`; `-Port=plain` builds without any Qualcomm library. |
| Android | NNAPI, fp16 | same | **Debug batch mode only** (`--es accel nnapi`), not in Settings: on the x86_64 emulator ORT's NNAPI EP aborted the app inside Run (`std::terminate`, which no fallback can catch). Only in `onnxruntime-android` (x86_64, or arm64 with `-Port=plain`); the QNN AAR has no NNAPI EP. |
| Android | XNNPACK | same | Compiled in, but not used: as an Auto candidate it aborted the app on the emulator in the same way. |
| Desktop (`tools/desktop_check`) | CPU, DirectML, CUDA | all five | `--backend` / `FS_BACKEND`. CUDA runs **without TF32** (`use_tf32=0`): with ORT's default TF32, SyntheticDoc-HR's grid moved by p99 0.50 px on the RTX 5060. TensorRT, OpenVINO, CoreML and QNN (Snapdragon X) are compiled in behind CMake options (`FS_ORT_<EP>`) but were not run (no hardware or package here). |

- The **page finder always runs on the CPU** (34 NNAPI partitions; HardSigmoid is unsupported).
- Accelerators get **fixed-shape models** (`tools/accel/prepare_models.py`, `tools/shadow_net/export.py --static`):
  `uvdoc_static.onnx`, `syntheticdoc_hr_static.onnx` (unused `shading` output removed) and
  `shadenet_512x352.onnx` (the A4 page always gives the shadow net 512 x 352). NNAPI coverage: 1 partition /
  100 % for all three (before: 5 partitions for SyntheticDoc-HR, 39 of 113 nodes for ShadeNet). On the CPU
  the static UVDoc / SyntheticDoc-HR are bit-identical to the originals (ORT 1.22.0 on x64, 35 photos), but
  the CPU still loads the original files. The three files add 32.8 MB to the APK.
- **QNN / NNAPI fp16 accuracy is not measured on a device.** The fp16 stand-in (the fp16 model on the desktop
  CPU EP; it rounds weights and activations, not the accumulation) moves the grids by p99 0.08 px for UVDoc
  and **0.95 px for SyntheticDoc-HR** (max 2.64 px, 35 photos). If real HTP fp16 behaves like the stand-in,
  SyntheticDoc-HR fails Auto's tolerance. It therefore has its own backend choice: the NPU setting keeps it
  on the CPU, and Auto moves it to QNN only on a phone where it passes. UVDoc and ShadeNet still go to the NPU.
- **GPU image stages** (GLES 3.1 compute, CPU fallback, `use_gpu`): besides the final resample and cleanup,
  the warp map's global remap now runs on the GPU (`Gpu::remap_offset`). It is on by default where the
  context has `precise` (ES 3.2 or `GL_EXT_gpu_shader5`). The stage-1, uv-spread and orientation pages have a
  GPU path too (`Gpu::remap_dense`). It is off by default, because it was slower on the measured desktop
  GPU. Every GPU stage is byte-identical to the CPU (`fs_cli selftest-resample`, `fs_cli gpu-stages`).
- **Overlap** (`PipelineOptions::flat_nets`): `run_photo` starts the flattening nets on a worker thread. Beside
  them it runs detection, surface fit, quality gate, the orientation page and a provisional orientation
  decision. It redoes the orientation page when a grid moved the corners (`uv_defines_page`), and the decision
  when the page-shape test differs. The output is the same as sequential. `pipeline_cli` uses it
  (`FS_NETS=1`; `FS_SEQUENTIAL=1` for the old order), and so does the app (`process()` and the debug batch
  mode hand `run_photo` the nets through `flat_nets_fn`). Processing details report `nets_overlap` and
  `nets_wait_ms`.
- **Desktop GPU**: `-DFS_DESKTOP_GPU=WGL` runs `core/src/gpu.cpp` on the system OpenGL driver (an ES 3.2
  context through WGL, or GL 4.6 core with `FS_GL_PROFILE=core`); `MESA` uses Mesa (d3d12 or llvmpipe);
  `OFF` (default) keeps `gpu_stub.cpp`. `pipeline_cli` turns the GPU stages on with `FS_GPU=1`.

### Settings > Acceleration, and how Auto chooses

The choices are Auto / CPU / NPU (Qualcomm QNN). **The default is CPU**: Auto stays
opt-in until it has been checked on a phone. Backends are chosen per group: page finder (always CPU),
UVDoc (both passes), SyntheticDoc-HR, shadow net. NPU puts UVDoc and the shadow net on QNN and keeps
SyntheticDoc-HR on the CPU (its fp16 stand-in misses the tolerance, above).

**Auto** runs once per (SoC, app version). The bundled self-test photo's nets run on the CPU and on QNN,
if this ORT build has it (NNAPI and XNNPACK are left out, see above; without QNN, Auto is the CPU). Per group, Auto keeps the fastest backend whose outputs
match the CPU within the tolerances below, and stores the choice. Processing details and the benchmark report
(Settings > *Run the acceleration benchmark*) show it. A net that fails on an accelerator at run time moves
to the CPU for the rest of the session.

**Tolerances** (accelerated backends only; `AUTO_*` in `jni_bridge.cpp`; the CPU stays bit-identical):
- flattening nets: grid difference in px of the 1600 x 2263 analysis page (Gate A's measure), p99 <= 0.25 px
  and max <= 1 px;
- shadow net: |shading difference| x 255, p99 <= 0.5 and max <= 2 levels.

### Measured on the desktop

Machine: i5-14400F, RTX 5060, Windows 11. Other jobs shared the machine, so the times are indicative.
Per page with warm sessions, 5 of the 35 photos (`gate_f.sh smoke`, median of 3 passes after a warm-up; ms):

| Nets | GPU image stages | page finder | UVDoc pass 1 | UVDoc | SD-HR | ShadeNet | nets sum | run_photo | page |
|---|---|---|---|---|---|---|---|---|---|
| CPU | off | 17 | 58 | 58 | 515 | 19 | 668 | 1673 | 2433 |
| CPU | on | 17 | 58 | 58 | 513 | 19 | 665 | 1372 | 2132 |
| DirectML | off | 9 | 16 | 15 | 59 | 7 | 106 | 1681 | 1908 |
| DirectML | on | 2 | 7 | 7 | 24 | 6 | 47 | 1354 | 1511 |
| CUDA (no TF32) | on | 5 | 10 | 10 | 29 | 4 | 58 | 1301 | 1512 |

- The nets take 668 ms on the CPU and 47-58 ms on the GPU (11-14x). The page then waits on `run_photo`'s CPU
  stages (about 1.3 s). The plan estimated 1.2-1.5 s per page on an NVIDIA desktop; measured 1.5 s.
- DirectML / CUDA nets vs the CPU, grid p99: UVDoc 0.0006 / 0.025 px, SyntheticDoc-HR 0.0005 / 0.0006 px. No
  pipeline decision changed on the 5 photos.
- CUDA's first run on a new driver compiles ORT's kernels for sm_120. That took 55 s once; the driver caches it.
- Overlap on the CPU backend gains little, because both branches share the cores. With the nets delayed by
  1.5 s (an accelerator's time without its CPU load), the grids were ready at max(CPU branch, nets), and a page
  took 0.3-0.9 s less (3 photos). The provisional orientation page was kept on 32 of 35 photos and 16 of 20
  DocUNet photos; the provisional orientation decision only on 24 of 35 and 3 of 20.
- APK (debug, arm64-v8a): 121.2 MB before the QNN libraries, 162.4 MB with them (+41.3 MB download, +110 MB
  installed; `libQnnHtpPrepare.so` alone is 23 MB). `-PqnnHtp=V73,V75,V79` saves 6.1 MB.
- Plan estimates, **not measured**: a flagship Snapdragon page takes 6-9 s today, 2.5-3.5 s with QNN, and
  1.5-2.5 s with overlap and GPU stages; nets 8-15x faster on Hexagon fp16.

### Checks at the final merge (2026-10-21; details and commands in STATUS.md)

- CPU path with the nets in-process, overlapped and sequential: the 35 photos are bit-identical to the Python
  ORT 1.22.0 path they replace (net outputs, pages, JSON). DocUNet 1_1..10_2 are identical to that path and
  to the Round 8 pages (`work/new_fold2`). DocUNet was not re-scored: identical outputs keep LD 5.16.
- Android x86_64 emulator, bundled self-test photo, debug batch mode: with Acceleration=CPU and with Auto
  (no candidate on x86_64, so it chose the CPU) the page finder output, the three grids and the page are
  bit-identical to the desktop CPU. Only the JSON's surface-fit residual differs (`fit_median_px` 1.87128 vs
  1.87126: MSVC vs NDK clang). The first Auto run aborted the app inside ORT's NNAPI EP, the second inside its
  XNNPACK EP; both were then removed from Auto and Settings (commits d4016c0, 94f9a0a).
- Still needs a phone: STATUS.md, *Open HUMAN GATES*.

## Build

Requirements: JDK 17+, Android SDK platform 35, NDK `27.2.12479018`, CMake `3.22.1`.
OpenCV `4.11.0` comes from Maven Central (`org.opencv:opencv`, prefab). This is the same OpenCV version as the reference (`requirements-round3.txt`).

```bash
echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew assembleDebug                                  # app/build/outputs/apk/debug/app-debug.apk
./gradlew assembleDebug -Pabis=arm64-v8a,x86_64          # also an emulator ABI
```

Host harness (Linux) for the acceptance tests:

```bash
# OpenCV 4.11.0 built WITHOUT IPP (as on Android): core, imgproc, imgcodecs, calib3d
cmake -S host -B build-host -G Ninja -DOpenCV_DIR=<ocv>/lib/cmake/opencv4 && ninja -C build-host
export EGL_PLATFORM=surfaceless              # headless GLES 3.1 through Mesa (libegl-mesa0, libgles-dev)
build-host/fs_cli selftest-resample          # resample: CPU / GPU / kernel model vs cv::remap
```

Reproducing the measurements (Python env from `app_ref/requirements-round3.txt`, `PY=` points to it):

```bash
python3 code/photobench.py test_images/flat_page.png bench_photo            # in app_ref/
python3 code/photo_pipeline.py <photo> out_ref/<name> surface                # reference outputs
python3 code/warpmap.py test_images/scan_pN.jpg out_ref/scan_pN 0.15
python3 -I tools/m1_fixtures.py ...; python3 -I tools/export_stages.py ...   # fixtures
tools/run_m1.sh        fs_cli app_ref/code out_ref app_ref/test_images fixtures out   # milestone 1
tools/run_stage_parity.sh fs_cli app_ref fixtures out                         # stage by stage
tools/run_acceptance.sh   fs_cli app_ref out_ref fixtures out                 # section 5
```

The raw output of those runs is in `docs/results/`.

## Architecture

```
core/src/    portable C++17, shared by app and host (compiled with -ffp-contract=off)
  corners      page detection (corners.py)            nprng    numpy PCG64 + Generator.choice, bit-exact
  surface,lsq  surface model + scipy TRF port          npz      NPY/NPZ I/O for the harness
  orient       orientation (orient.py)                 nputil   polyfit, interp, percentile, np.sum order
  warpmap      warp map (warpmap.py, dewarp.py, clean.py helpers)
  sparse       13-point stencil normal equations, MG-preconditioned CG
  pipeline     photo_pipeline.py orchestration, composition on the 4-px grid
  resample     map expansion + OpenCV-exact bicubic (CPU), table for the GPU
  gpu          OpenGL ES 3.1 compute: resample, remaps, background, ink count, tone pass
  infer        the nets on ONNX Runtime C++ 1.22.0: one session per net, backend list with CPU fallback
  nets         net orchestration: page finder input, UVDoc passes, re-crops, SyntheticDoc-HR, ShadeNet
  cleanup,solid  Auto color / Grayscale / Solid colors
app/         Kotlin + Compose UI, CameraX, JNI bridge (app/src/main/cpp/jni_bridge.cpp)
host/        fs_cli: the stage and acceptance harness (Linux, or Windows with Mesa)
tools/desktop_check/  pipeline_cli and the gate scripts (Windows; CPU, DirectML, CUDA, desktop GL)
tools/       Python fixture exporters, comparators, acceptance runners
```

| Stage | Where | Notes |
|---|---|---|
| Neural nets: page finder, UVDoc (whole photo, then re-crop), SyntheticDoc-HR, ShadeNet | CPU (ORT C++ 1.22.0, in the core); opt-in NPU (QNN HTP fp16) on Android, DirectML / CUDA on the desktop | Settings > Acceleration, default CPU. The page finder always runs on the CPU. Any accelerator failure falls back to the CPU. See [Acceleration](#acceleration-nets-on-the-npu--gpu-cpu-stages-beside-the-nets-more-image-stages-on-the-gpu). |
| Page detection (`paper_mask`, `curved_corners`) | CPU (C++) | RANSAC draws come from the same numpy `default_rng(0)` stream. Runs on a copy no larger than 2400 px; the reference inputs are 2212–2300 px. |
| Surface fit (10 unknowns) | CPU, multi-threaded | Port of `scipy.optimize.least_squares` (TRF, exact SVD trust region, 2-point Jacobian, Cauchy loss). The 10 coarse starts and the Jacobian columns run in parallel; this is deterministic. EXIF focal length is used when present. |
| Homography fallback | CPU | Used when the user moved a corner (> 0.3% of the image size), when detection failed, or when the outline residual median is above 3 px. |
| Stage-1 page, orientation page | CPU; GPU path off by default | `cv::remap` from the exact per-pixel unroll map, in row bands. `Gpu::remap_dense` gives the same bytes, but was slower on the measured desktop GPU. With `flat_nets` (desktop `pipeline_cli`), the orientation page and a provisional decision are made while the flattening nets run. |
| Orientation | CPU | Decided on the homography-rectified page, as the brief asks. Turned off if the user rotated the photo. |
| Global stage, line detection, tracing (`trace_w2`), chaining, text rows | CPU, lines in parallel; global remap on the **GPU** where the context has `precise` | The global remap (`Gpu::remap_offset`) is byte-identical to the CPU; CPU fallback. |
| Sparse solves (global 32 px, pass 1 16 px, pass 2 8 px warm-started) | CPU | CG preconditioned with a geometric multigrid V-cycle (Galerkin, symmetric Gauss-Seidel, dense Cholesky on the coarsest grid), rtol 1e-8. The x and y solves run in parallel. |
| Composition | CPU, 4-px grid only | `F(p) = U(expand(warp4)(p))`, with cv2.remap's 1/32-px quantisation and `BORDER_REPLICATE`, evaluated in the rotated output frame exactly as `photo_pipeline` stores `warpmap.npz`. No full-resolution float map is allocated on the CPU. |
| Map expansion + bicubic resample of the full photo | **GPU** (GLES 3.1 compute), CPU fallback | float32 maps; OpenCV `INTER_CUBIC` semantics: 1/32-px positions, its 15-bit weight table including its sum-correction quirk, white border. |
| Cleanup: ¼-resolution background (block mean, 3×3 dilate, exact 11×11 median), ink counts, one tone pass (illumination, white point, contrast, specks, grayscale) | **GPU**, CPU fallback | The paper-illumination fill and the black/white points are a small CPU step at ¼ resolution. |
| Solid colors | CPU | Port of the cell-fill part of `clean.py`. The grid lines only separate regions; nothing is redrawn; ink is composited on top. |

The GPU paths are chosen at runtime: EGL + GLES ≥ 3.1, texture size limits, and a fallback whenever a call fails.
Analysis runs at 1600 px page width. The output-resolution setting (1240 / 1600 / 2480 px) only rescales the final grid, so the photo is still sampled once.

**App flow:** Home (camera / gallery) → Camera → Crop → Processing → Result → Save/Share.
- **Crop:** auto-detected corners, 44 dp touch targets, a 3× magnifier loupe while dragging, and rotate 90°.
- **Processing:** progress bar with the stage name.
- **Result:** Before/After toggle; Auto color / Grayscale; Solid colors; Rotate; Save (JPG or one-page A4 PDF through the system file picker) and Share; a details dialog with per-stage times.
- **Settings:** default colour mode, solid colours, output resolution, Acceleration (Auto / CPU / NPU), an on-device self-test and the acceleration benchmark.

Light and dark themes follow the system setting.

## Acceptance results (section 5 of the brief)

All numbers below were **measured**, on the Linux host harness: 4-core Xeon 2.1 GHz. The GPU results ran on Mesa llvmpipe, a software GLES 3.2 driver. **None were measured on a phone.**
The reference values were re-measured in this environment with the pinned Python environment (`docs/results/reference_metrics_measured.json`). Where they differ from the brief, both are shown.

### 1. Map parity (`port_compare.py reference candidate --limit 0.1`, app = full C++ pipeline)

| page | max EPE | p99 | result |
|---|---|---|---|
| photo 1 | 0.0886 px | 0.0151 px | pass |
| photo 2 | 0.0843 px | 0.0165 px | pass |
| scan p1 / p2 / p3 | 0.0054 / 0.0024 / 0.0036 px | ≤ 0.0024 px | pass |
| bench P1 / P2 | 0.0308 / 0.0267 px | ≤ 0.0001 px | pass |
| photo 1, homography (`v8`) path | 0.0425 px | 0.0 px | pass |

The GPU and CPU paths give identical maps. **The photo margins are small.** Stage by stage, the port matches the reference much more closely (`docs/results/stage_parity_host.txt`):
- detection: identical contours and RANSAC inlier sets; corners within 2e-12 px;
- surface-fit cost: equal to about 1e-9 relative;
- warp map, run on the reference's own stage-1 page: the global page is bit-identical, the detected lines are identical, the samples agree to 1e-11, and the final map error is ≤ 0.005 px.

The remaining photo error comes from where the two TRF runs stop. Both satisfy the same `ftol=1e-8` stopping test, but BLAS/LAPACK rounding differs. The resulting fit-parameter difference (≤ 4.6e-4 rad) moves the photo 1 unroll map by only 0.0003 px. It still changes 2,941 stage-1 pixels by up to 5 levels: Python's own stage 1 rendered with the C++ parameters differs from the reference in 2,941 pixels, and from the C++ stage 1 in 15. That shifts a few line centroids by 0.05 px, which becomes the 0.08 px above.

### 2. Image parity (8-bit)

- **Parity path (same map, milestone 1):** GPU and CPU resample are **identical to the reference** on all 5 pages: 0 levels, 0 pixels. They are also identical to `cv::remap` on 30 synthetic fixtures (constant, impulse, checkerboard, sub-pixel shift, out-of-range, random warp; 2×2 up to 1203×1609 sources).
- **GPU vs CPU, end to end:** identical on all pages. Cleanup is also identical, GPU vs CPU, in all four modes.
- **End to end against the reference image:** this cannot meet ≤ 1 level, and I did not relax the gate.
  - **Scans:** max 5–6 levels, in 0.004–0.013% of pixels. A 0.005 px map difference can flip OpenCV's 1/32-px position rounding, which is up to ~6 levels at a hard edge.
  - **Photos against the reference resampled from its own stored grid:** max 9 / 8 levels, in 0.12 / 0.13% of pixels.
  - **Photos against the published `out.png`:** max 118 / 140 levels, in 3.4 / 4.5% of pixels. The reference's own stored map differs from its `out.png` by the same amount (see finding 3).

### 3. Ground truth (`photobench.py` + `bench.score`, mean position error)

| case | app | reference (measured here) | brief | within +0.2 px |
|---|---|---|---|---|
| P1 | 1.082 px | 1.082 px | 0.97 px | yes |
| P2 | 1.199 px | 1.199 px | 1.21 px | yes |

### 4. Real-page straightness (`evaluate.py`)

| page | app: lines < 1 px, worst | reference measured here | brief |
|---|---|---|---|
| photo 1 | 98.6%, 1.41 px | 98.6%, 1.42 px | 97%, 1.42 px |
| photo 2 | 100%, 0.621 px | 100%, 0.572 px | 100%, 0.58 px |
| scans p1/p2/p3 | 100%, 0.463 / 0.773 / 0.572 px | identical | 100%, ≤ 0.8 px |

The photo 2 worst-line difference (0.621 vs 0.572 px) comes from composing on the coarse grid, which the brief requires. The reference's own map, resampled from its stored 4-px grid, scores **0.616 px**.

### 5. Photo 1 title border (`fixed_photo_feature.py` logic, on the pipeline output)

p98 **0.261 px**, end **0.191 px**; the reference measured 0.268 / 0.189 px. The gate is ≤ 0.5 px: **pass**.

### 6. Timing

**Not measured on any phone.** No device or emulator was available; the container has no KVM. The app has an on-device benchmark for this: Settings → *Run self-test*. It runs the bundled photo 1 six times (one warm-up) and reports:
- map parity against the bundled reference grid;
- GPU vs CPU resample and cleanup;
- per-stage median and p95;
- peak memory (VmHWM);
- device, SoC, GPU driver and thermal state.

It has a *Share report* button. Run it on two named devices to fill in this section.

For orientation only, single host runs of the whole C++ pipeline (not medians, not a phone) take 1.5–1.7 s per photo and 2.5–2.9 s per scan; per-stage times are in `docs/results/acceptance_host.txt`. The brief quotes about 15–20 s per photo and 10–13 s per scan for the Python reference (desktop, 2 cores).

## Findings about the reference (with evidence; nothing was changed silently)

1. **The orientation page differs between brief and code.** The brief says to decide orientation on the homography-rectified page. `photo_pipeline.py` decides it on the stage-1 page, which in surface mode is the surface unroll. On photo 1, the upright score is **0.0018** on the stage-1 page and **0.49** on the homography page. The app follows the brief. It reaches the same decision as the reference on all 4 photo inputs (`fixtures/*/info.json`), and the option `orient_on_stage1` reproduces the code's behaviour.
2. **The bench pages are flipped 180° by the reference.** Upright scores are −0.03 / −0.07 on the homography page and −0.06 / −0.03 on the stage-1 page. EPE is unaffected, because `bench.score` fits a similarity that includes rotation. But it reports `scale_x_p95` / `scale_y_p95` of about **200%** for every bench run for that reason. Those two columns are meaningless as long as the flip happens. The app reproduces the flip, for parity.
3. **The stored photo map is not the published image.** `photo_pipeline.py` resamples `out.png` from its dense composed map, but stores only every 4th sample in `warpmap.npz`. Within about 8 px of the border, the dense map has `BORDER_REPLICATE` kinks that the grid cannot hold: up to **0.97 px** (photo 1). Resampling the photo from the stored grid changes 3.4% (photo 1) to 4.5% (photo 2) of the pixels by more than one level. Photo 2's worst line is 0.572 px from the dense map and 0.616 px from the stored map.
4. **The pipeline is sensitive end to end.** A 0.0003 px change of the stage-1 unroll map moves the final photo map by up to 0.084 px, through thresholded morphology and line centroids (section 1 above). The 0.1 px gate passes, but other photos may need a looser gate or exact LAPACK.
5. **The brief's numbers differ from the measured reference.** Bench P1 measures 1.082 px here; the brief says 0.97. Photo 1 has 71 of 72 lines under 1 px, which is 98.6%; the brief says 97%. Everything else matches.
6. **IPP:** the reference's OpenCV wheel uses Intel IPP; Android OpenCV does not. Re-running the reference with IPP off changes its maps by at most **0.029 px** (photo 1) and 0.0005 px (scan p1), and about 3 pixels per 10 million in the images. Remap itself is not affected (`IPP_DISABLE_REMAP 1`).
7. **The AMG solve stops early, harmlessly.** Pass 2 stops at relative residual about 9e-6. Its field is within **0.005 px** of the exact solution (`docs/results/solver_audit.txt`), so the port solves to the exact solution; the C++ field is within 2e-6 px of it.
8. **Photo 2 has no text-row constraints.** Stage 1 is still sideways when the warp map runs (orientation is applied only at the end), so `text_rows` finds **0** rows there. The other pages have 81–174. I left this as it is, because it is reference behaviour.
9. **A possible platform dependence in `chain_trace`:** the seed piece is chosen with `np.argsort`, which is not a stable sort. With ties in the longest piece length, numpy's choice can depend on the CPU's SIMD sort. The port takes the first piece. This was not observed to matter on the test pages.
10. **numba works here.** The round-3 reviewer could not use it; in this environment it works, and the reference ran with it.

## Not done / not verified on a real phone

- **Acceleration (QNN, the Auto benchmark, GPU stages, overlap) has not run on a phone**; the steps are in `docs/results/accel/STATUS.md`.
- **Nothing has run on a phone.** Round 4 ran the app on the Android 15 x86_64 emulator: gallery import, crop screen, processing, result screen, filters, the Pages screen, the page finder on ONNX Runtime and book spreads, all on the CPU path (the emulator offers only GLES 3.0).
  - Not verified: CameraX capture, save/share, the PDF output, the GPU paths on a device, and ONNX Runtime on arm64.
  - GLES 3.1 shader compilation on Adreno or Mali drivers has not been seen; the shaders were compiled and checked only on Mesa llvmpipe.
  - Checked statically: the APK builds, and every Kotlin `external` function has a matching exported JNI symbol.
- **Timings on two named devices are not measured.** Use the in-app self-test.
- **Map parity on ARM is not verified.** Android OpenCV is built with clang, which may fuse multiply-adds inside OpenCV's own float code. The core itself is compiled with `-ffp-contract=off`. The self-test reports map parity on the device.
- **Phone photos above 2400 px are downscaled for detection and the surface fit.** These kernel and threshold choices are untested: the reference was only tuned on 2212–2300 px photos. Stage 1 and the final resample use the full photo, which is capped at 4096 px on decode.
- **Cleanup is my design** (the brief gives the steps, the reference has no implementation). It was checked visually on the two photos and checked GPU = CPU. It was not checked on colour-critical or low-light pages.
  - Solid colors does not port v1's "coloured areas outside the grid → rectangles" step, because that step snaps to the redrawn lines.
- **Device behaviour that is not verified:**
  - GPU memory on large photos: a 4096×3072 RGBA texture is about 50 MB; the app falls back to the CPU if a GPU call fails.
  - Behaviour under thermal throttling.
  - The camera focal length from Camera2 on multi-camera phones: only the first lens focal length is read.
- **Not done:** cancelling a running job, text-only pages, and pinch zoom on the crop screen. Edits are saved to the page cache when you leave the page or the app goes to the background; a process kill before that loses the last edits.
- **Solid colors GPU vs CPU on scan p3:** up to 22 levels on 33 values (the other pages are within 1 level); scan p3 also keeps 2207 px of (255,255,254) slivers that are already in the plain Auto color render.
- **Text-only pages:** Only table pages exist among the test inputs, as in the reference.
