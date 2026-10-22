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
  never detected. A screen if score >= 2.4 (`kScreenThreshold`).
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
(filled below)

## Results
(filled below)
