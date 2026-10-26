# Page flattening: where FlatScan stands (2026-10-26)

A brief for anyone reviewing how to close the remaining quality gap with CamScanner. All numbers below were
measured on the desktop build of the app's pipeline (`tools/desktop_check/pipeline_cli`). Every verdict comes
from looking at the output pages side by side, not from a metric.

## Pipeline (photo -> page)
`core/src/pipeline.cpp` `run_photo`, in order:
1. **Page finder:** segmentation net + contour → 4 corners, a paper mask and outline points per side
   (`detect_corners`, `corners.h`).
2. **Stage 1 geometry**, one of:
   - **homography** between the corners;
   - **surface model** (`surface.cpp`): a sheet bent along one axis, plus camera pose, fitted to the outline;
   - **flattening net** grid:
     - SyntheticDoc-HR (UVDoc architecture, 1024×1440 in, 177×121 backward map) runs first;
     - UVDoc is the fallback (`uvdoc.cpp`, crops and checks in `run_photo`).
     - A grid is accepted when its outline agrees with the page finder's mask (IoU ≥ 0.93 for HR).
     - A grid can also define the page itself when it agrees with a first pass on the whole photo (`uv_defines_page`).
3. **Flat-paper rule** (new, `flat_edge_bow`): if every side of the page finder's outline is straight
   (p95 bow ≤ 1.2% of the side's length), the homography replaces the net's grid.
4. **Line stage:** text rows and rules traced and straightened on top (`warpmap.cpp`, `neural_line_stage`).
5. **Shadow net** on the flattened page, then the app's "Auto color" filter at 0.7 (`cleanup.h`).

## What the tests showed
**Attribution, 8 problem photos:**
- Perspective only vs net only vs net + line stage: the line stage made no visible difference on any of them.
- All the warping comes from the net.

**Where each method wins:**

| Better | Photos |
|---|---|
| Net | curled pages: magazine, cookbook, folded coupon, a magazine page whose crop the page finder got wrong |
| Plain perspective | flat small objects: a dollar bill (the net bowed it), a receipt (the net invented a wider page) |

**The net alone cannot tell flat from curled.** Its grid's distance from the best homography (p95, page px) is:

| Page | Net's departure from flat |
|---|---|
| Flat bill | 84 |
| Receipt | 51 |
| Curled magazine | 31 |

**The photo can.** The p95 bow of the page finder's sides, divided by side length:

| Page | Side bow |
|---|---|
| Bill | ≤ 0.010 |
| Magazine | 0.15 |
| Cookbook | 0.06–0.09 |
| Coupon | 0.056 |

Since paper bends but doesn't stretch, a sheet with four straight edges is flat. That is the rule now in use.
On 121 real photos it changed 6 pages: 3 better (including the bill), 3 the same, 0 worse.

## Tried and dropped (each measured on 121 photos, every changed page inspected)

| Attempt | Result | Why it failed |
|---|---|---|
| Rules on whose outline wins (net vs page finder): paper share by mask, then by paper colour | fixed the receipt; 5 of 6 other changed pages worse | the page finder's frame is worse on cluttered photos |
| Snapping the net grid's border onto the page finder's mask edge (Coons-patch spread inwards) | 2 better, 12 worse | the mask edge is rough (shadows, fingers), so every bump bent the text |
| Flat test from Hough lines bent by the net | thresholds unstable; broke other pages | — |
| Surface model as the base | wavier than perspective on most pages | — |

## Compared with CamScanner (3 shared photos, the app's default filter)
- **Magazine:** equal.
- **Coupon:** about equal. Our rows are straighter, but see open problem 1.
- **Hi-TEC page with a leaf shadow:**
  - CamScanner removes the shadow completely.
  - Our shipped shadow net leaves it; a retrained shadow net (not yet shipped) removes most of it.
  - Our crop has an empty band at the top.

## Open problems
1. **Corner curl** (the coupon's "SAVE $1" strip):
   - The strip's top-right corner lifts off the table.
   - The net flattens most of the page but leaves the corner curled, and the bent "1" reads as "7".
   - The page finder's outline sees the curl (sides bow 0.047–0.056).
2. **The Hi-TEC crop:** both nets were rejected there, and the surface model left an empty band at the top.
3. **Net noise on near-flat pages:**
   - SD-HR adds wavy text on gently curved pages.
   - Its training data (SyntheticDoc) is all A4 and all curled, with no flat or narrow pages.
4. **Proportions:**
   - `map_page_hw` measures the page's height/width along the map in photo pixels, so perspective shortens it.
   - The app's Auto shape trusts it over a perspective-correct corner estimate.

## Questions for a reviewer
- How do strong scanners handle a curled corner on an otherwise flat page? A local developable model
  (a flat page plus a cylindrical curl beyond a fold line at any angle, fitted to the outline), a
  second net pass on the corner, or something else?
- Is fine-tuning SD-HR with flat, near-flat-with-corner-curl and narrow pages (receipts, bills, cards), with
  ground truth made from SyntheticDoc's UV maps, the right next investment? Or is a selector worth training
  instead? A selector would need labels for which method is best per photo, which we don't have yet.
- Anything in the acceptance checks (`run_photo` net block) that is structurally wrong?
