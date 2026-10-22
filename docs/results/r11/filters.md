# Round 11 FILTERS stream: a "Scan" filter and a screen mode

Branch `r11/filters` (worktree `C:\Users\User\Desktop\fs_r11_filters`). Outputs, contact sheets and score
files: `C:\Users\User\Desktop\flatscan_out\r11\filters\` (`runs/<run>/<set>/`, `sheets/`, `screen/`).

## What changed

### 1. New filter "Scan" (`Filter::Scan = 6`, `FilterKind.Scan`, the new default of new scans)

The existing filters blend with Original at the strength (default 70%): `out = page + 0.7 (filter - page)`.
So at the default strength the paper keeps 30% of the photo's lighting: on the 34 real pages **no** paper
pixel reached 250 at Auto color 70% (measured below). That is the main reason pages did not look scanned.

Scan (`cleanup.cpp scan_px`, mirrored in `gpu.cpp kToneCS` `uKind == 4`):
- its base is the Lighten curve on the flattened page `v = I / P` (lighting and colour cast removed, paper white,
  ink untouched), so the paper is white at **any** strength; the strength only blends the enhancement in:
  `t = base + s (enh - base)` (s = 0 is Lighten, not Original);
- `enh`: a luminance curve `f(l) = ((l - bk) / (wp - bk))^gamma` (1 above the white point `wp = 1 - 2.5 sigma`),
  black point `bk = min(0.2, 0.5 x the page's ink level)` (`PageParams::ink`, median luma(v) of ink on paper;
  0 inside pictures via the picture weight), applied by scaling the pixel `v x f(l) / l`: **hue and saturation are
  kept exactly**, dark text goes black, coloured text stays its colour;
- near white a near-neutral pixel (chroma `max - min` of v below 0.04 -> 0.10) fades to neutral, except on light
  colours (`tint` weight), so paper noise and casts turn white and pastels / highlighter stay;
- edge-limited unsharp mask of the luminance (`sharpen_luma`: sigma 1 px, amount 1.2 x strength, clamped to the
  3 x 3 min / max of the neighbours, so no halos and no white rims; chroma untouched). It runs on the CPU after the
  curve on both paths (`post_filter`; the GPU path computes the curve in GLSL, then the CPU sharpens). Thumbnails
  sharpen at their own scale (skipped below sigma 0.4 px).
- speck removal as Auto color.
Constants: `ScanTune` (cleanup.h), chosen on the synthetic colour set (with ground truth) and checked on the 34 real
pages (table "Scan tuning").

### 2. Screen mode (`core/src/screen.{h,cpp}`)

- `detect_screen(page)`: on the flattened page, 3 x 3 tiles of 512 px; per tile the spectrum (Hann window) of the
  chroma channels Cr, Cb; score = median over tiles of the strongest peak (log magnitude above its 15 x 15 median)
  at periods below 16 px, off the axes and off the JPEG block lattice (k/16 cycles/px). Pages under 1024 px are
  never detected. A screen if score >= 2.5 (`kScreenThreshold`).
- `descreen(page)`: (1) notch filter: peaks of the luminance spectrum (log magnitude > 2.0 above its 21 x 21
  median, periods < 12 px) are cut from Y, Cr and Cb (soft notches) - pixel grid, subpixel stripes, their aliases;
  (2) chroma: Cr, Cb replaced by a guided filter (radius 8, eps 40^2) with the luminance as guide - colour moire
  without its own luminance edge is averaged away, colours of the content stay. Then the normal filter runs on the
  descreened page with its own `page_params` (bands, glow and glare of the screen are flattened like lighting), and
  Scan uses a deeper black point there (`PageParams::screen`: min(0.35, 0.8 x ink): a screen's black glows).
- Option: `RenderParams::screen` = Off / On / Auto (core `ScreenMode`, JNI float array index 7, Kotlin
  `RenderParams.screen` / `ScreenMode`, persisted in the page JSON as `"screen"`, `Settings.screenMode` for new
  scans, `ScanViewModel.setScreen()`). Default Auto. The JNI session runs `detect_screen` once per page (after
  page_params: process, erase, restore) and builds the descreened page and its params on first use (cached, also
  for thumbnails and the histogram). `info()` reports `"screen": {"detected", "score", "threshold"}`.
- GPU: descreen is CPU only (OpenCV DFT); the filter curve then runs on the GPU or CPU as before.

### 3. Eval tools (`tools/filters/`)
`filter_cli` (desktop_check), `eval_filters.py` (render + metrics), `run_set.sh`, `make_synth.py` (synthetic colour
pages and synthetic LCD photos with ground truth), `sheet.py` (contact sheets), `screen_scores.py` (detection),
`flatten_real.sh` (flattened pages as the app makes them), `glsl_check.py` (glslc on the filter shaders),
`screen_proto.py` / `descreen_proto.py` (Python prototypes).

## Data
All sets under `C:\Users\User\Desktop\flatscan_out\r11\filters\` (sources: `E:\fs_datasets\MANIFEST.md`).
- Paper (negatives for detection): `real35` (the 34 real pages, flattened as the app makes them), `real_colour_docs`
  (14), `real_paper_neg` (28), `synth_colour` (40, with GT), `negproxy` DIR300 (30) and DocReal (25): 171 pages.
- Screens (positives): `synth_screen` (30 synthetic LCD photos with GT), `uhdm` (42), `real_screen` (48 real photos
  of screens, Commons): 120 pages.

## Results
All numbers: `filter_cli` built with `tools/filters/build_dc.bat build\cpu5 OFF` (CPU core), desktop, shared machine
(timings are noisy; best of 3).

### Screen detection (`screen_scores.py`, log `screen/scores4.log`)
Paper scores 1.33-2.47 (max: `user__projector_screen` 2.47 in real35, then `Primera_portada_mocador` 2.35);
screens median 1.9-4.2. Threshold sweep over 171 paper pages / 120 screens:

| thr | false positives | screens found |
|---|---|---|
| 2.0 | 10/171 | 79/120 |
| 2.4 (before) | 1/171 | 46/120 |
| **2.5 (shipped)** | **0/171 (0%)** | **43/120** (synthetic 26/30, UHDM 10/42, real photos 7/48) |

The threshold was raised 2.4 -> 2.5 because one real paper page (the projector-screen photo, 2.47) crossed 2.4.
The margin to the top paper page is only 0.03: re-check on new paper data. Detection is conservative: 41/48 real
screen photos are NOT detected, so Auto rarely fires on them; the manual On switch is the way to descreen them.

### Descreen quality, synthetic LCD photos with GT (`synth_screen`, 30 pages, Scan 70%, `run_set.sh`)

| | dE | dE (colour) | PSNR fit | SSIM fit | bg chroma | marks lost |
|---|---|---|---|---|---|---|
| screen Off | 23.41 | 36.71 | 16.2 | 0.52 | 35.3 | 153 |
| screen On (final, fast descreen) | 16.13 | 29.78 | 17.77 | 0.67 | 1.56 | 37 |
| screen On (c1e77ae, before the speed rewrite) | 16.13 | 29.78 | 17.80 | 0.67 | 1.54 | 38 |

The speed rewrite matches c1e77ae in quality, so it was kept. Sheets: `sheets/synth_screen_final.jpg`,
`sheets/real_screen_final.jpg` (48 real screen photos, Off vs On; rendered, not inspected by eye in this last pass).

### Scan vs Auto color at 70% (`eval_filters.py`, README metrics)

| set (N) | filter | paper px >= 250 (%) | marks lost | chroma ratio median / worst page | regions turned white |
|---|---|---|---|---|---|
| real35 (34) | Auto color | 0.0 | 171 | 1.08 / 0.063 | 0 |
| real35 | Scan | 97.4 | 182 | 0.96 / 0.0 | 9 |
| colour_docs (14) | Auto color | 0.01 | 487 | 1.09 / 0.083 | 0 |
| colour_docs | Scan | 92.9 | 536 | 0.99 / 0.0 | 2 |
| synth_colour (40, GT) | Auto color | 0.0 | 653 | 1.10 / 0.148 | 0 |
| synth_colour | Scan | 83.4 | 341 | 0.98 / 0.068 | 1 |

synth_colour (GT): dE 13.35 -> 6.82, dE colour 13.38 -> 9.77, bg chroma 6.3 -> 1.35, PSNR fit 21.5 -> 21.8.
Scan whitens the paper (the goal) and keeps chroma on average, but on real pages some pale coloured regions turn
white (9 on real35, 2 on colour_docs) and marks lost is +6..10% vs Auto color (fewer on synthetic).

### Timing, desktop CPU, 1600x2263 flattened pages (3 real pages) and a 1100x1500 synthetic one, best of 3

| stage | r11/main (Auto color) | c1e77ae | final |
|---|---|---|---|
| Auto color filter | 274-290 ms | | 266-285 ms |
| Scan filter (total with screen options) | | 394-425 ms | 379-403 ms |
| detect_screen | | 323-332 ms | 278-374 ms |
| descreen | | 787-1061 ms | 678-871 ms (synthetic 1100x1500: 473 vs 698) |

Descreen is ~15-30% faster than c1e77ae, still ~0.7-0.9 s per full page (CPU only); detect ~0.3 s, once per page.
Phone timings: not measured.

### Existing filters unchanged
Filters 0-5 at strength 0.7 on the 34 real pages: 204 output images from r11/main (`build\base`) vs this branch
compared with `cmp`: 0 differences (byte-identical).

### Android plumbing
Gradle assembleDebug succeeded on this branch before the last core-only edits (`gradle.log`: BUILD SUCCESSFUL). Since
then only `screen.cpp/.h`, the CLI and the report changed; JNI does not use the removed `gated` field. Not rebuilt in
Gradle in this pass; glsl_check not re-run (no shader change).

## UI hooks
`RenderParams.screen` (`ScreenMode` Off/On/Auto, default Auto), `Settings.screenMode`, `ScanViewModel.setScreen()`,
page JSON `"screen"`, `info()["screen"] = {detected, score, threshold}` for a "screen detected" hint;
`FilterKind.Scan` is the default of new scans.

## Risks / not verified
- Auto finds 43/120 screens (7/48 real photos) at 0/171 false positives; margin to the top paper page 0.03.
- Scan can whiten pale coloured regions on real pages (9 on the 34).
- Descreen not measured on a phone; the GPU path does not descreen (CPU, ~0.7-0.9 s desktop per full page).
- Descreen benefit judged on synthetic LCD GT and sheets only.
