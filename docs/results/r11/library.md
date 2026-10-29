# Round 11 — LIBRARY stream: a larger real-photo test library + harness

Branch `r11/library` (from r11/main db9ef56), tools in `tools/library/`. The library itself lives on E: (too big for
git): `E:\fs_datasets\` with `MANIFEST.md` (every set: path, count, ground truth, licence, source URL, use).

## What is in the library (all counts measured from the index files, `run_library.py list`)

Validation sets with flat ground truth (DocUNet stays the TEST set and is not in the library):
| set | count | GT | licence |
|---|---|---|---|
| dir300 (DocGeoNet) | 300 real photos | flat scans | DocGeoNet repo licence (academic) |
| docreal | 200 real photos (50 docs x 4) | flat scans | not stated (repo MIT) |
| warpdoc | 1020 real photos, 6 categories (subset warpdoc_s170 = every 6th) | digital page (with margin) | not stated |
| uvdoc_bench | 50 semi-synthetic (real paper warps) | flat texture | not stated |
| inv3dreal | 360 real photos = 20 invoices x 6 warps x 3 lightings (subset inv3dreal_s90 = every 4th) | flat page + true words | CC BY-NC-SA 4.0 (evaluation only; earlier listed as MIT) |

Real photos without geometric GT (filters, monitor scanning, visual review):
| set | count | what | licence |
|---|---|---|---|
| uhdm_screens | 500 | 4K phone photos of screens (moire) + aligned clean screenshot | not stated (repo Apache-2.0) |
| commons_screens + commons_screens2 | 395 + 300 (90 + 52 tagged `curated`: screen content fills the photo) | monitors, laptops, phones, CRTs, e-readers, departure boards, pixel macro shots | per file CC0/PD/CC BY/CC BY-SA |
| commons_colour_docs + commons_colour_docs2 | 389 + 450 (139 + 166 `curated`) | menus, forms, stamps, certificates, flyers, magazines, maps, sticky notes, coloured paper, postcards, banknotes | per file |
| commons_receipts_tickets | 345 (266 `curated`) | receipts, tickets, parking tickets | per file |
| commons_pages_lighting | 347 (99 `curated`) | open books, letters, newspapers under room light | per file |
| rdd_shadow / jung_shadow | 545 / 87 | photos with shadows + shadow-free GT | not stated / research |
| wildreceipt | 1739 | receipt photos + true words | Apache-2.0 |
| cord | 200 | receipt photos + true words | CC BY 4.0 |
| funsd | 199 | scanned forms + true words | non-commercial research |
| diw | 5667 | real document photos, but only 512x512 | not stated |
| real35 | 34 | the Round 8-10 photos | see SOURCES.md |

The Commons photos were filtered automatically (JPEG with camera EXIF Make/Model, >= 1000 px, free licence) and then
reviewed by eye on numbered mosaics (`mosaic.py`); `curated.txt` per folder lists the photos that really show a
screen / a document. 4 photos were removed (2 nude body-paint photos from a "smartphones in use" category, 2 profanity
stamps) to `real/commons/_excluded/`.

## Harness
`tools/library/run_library.py` (usage in its docstring and in MANIFEST.md): `run` (pipeline_cli over a set or subset,
app mode FS_NETS=1 or frame mode, extra env, `--render` for the Auto colour render), `score` (LD / MS-SSIM / AD with
docunet_bench/docunet_eval.py, straightness = tools/library/straight.py (Round 8/10 metric), Tesseract CER / word
recall), `sheets` (contact sheets photo | page | render | GT), `compare RUN_A RUN_B` (means, per-item better/worse,
worst items). It takes the heavy lock itself in chunks of 100 items and resumes interrupted runs.
A photo the pipeline cannot flatten is scored as the raw photo (counted in n_fail). CER is computed only on pages
whose GT-scan OCR has >= 50 characters (pictures / near-blank pages gave CER 13 on WarpDoc 'fold' before that rule).

## Baseline (measured): r11/main db9ef56 pipeline_cli, app mode, run `r11main_base`
| set | n | fail | MS-SSIM | LD | AD | CER (micro) | row dev p90 px (GT) |
|---|---|---|---|---|---|---|---|
| dir300 | 300 | 0 | 0.6258 | 7.897 | 0.2876 | 0.2065 (0.1687) | 5.74 (4.82) |
| docreal | 200 | 0 | 0.5648 | 11.796 | 0.3099 | n/a (Chinese text) | 8.39 (8.11) |
| uvdoc_bench | 50 | 0 | 0.5581 | 10.073 | 0.2025 | 0.3261 (0.2782) | 7.00 (6.57) |
| warpdoc_s170 | 170 | 0 | 0.5113 | 15.530 | 0.4209 | 0.3518 (0.2741) | 6.10 (4.53) |
| inv3dreal_s90 | 90 | 0 | 0.5939 | 24.051 | 0.1432 | 0.4816 (0.4716), word recall 0.5263 | 5.66 (5.35) |

How measured: `run_library.py run --cli <fs_r11_library>/tools/desktop_check/build/pipeline_cli.exe --run r11main_base
--sets <set> --render` (pipeline_cli built from db9ef56 in this worktree, Release, ORT 1.22.0 CPU, 4 processes x 2
threads, heavy lock held), scoring at 4 jobs. Per-item metrics in `E:\fs_datasets\runs\r11main_base\<set>\metrics.json`.
Scorer offset vs MATLAB (DocUNet check): MS-SSIM ~+0.017, AD ~-0.023, LD within 0.1 — compare runs with each other,
not with published numbers.

Observations from the sheets: DocReal 5_4 (table form, strong curl) came out turned 90 degrees (LD 42.8): an
auto-orientation miss on a table-only page. WarpDoc 'perspective' and 'rotate' are the weakest categories (LD 20.4,
19.7; AD 0.58, 0.56). Every page was flattened (no failures) on all 810 validation photos.

## Not done / limits
- Round 10 pipeline_cli baseline: not run (the Round 10 build has no in-process nets; it needs the Python net caches
  of run_r10.py per photo, not cheap for 810 photos).
- TIP2018 / FHDMi moire sets, SD7K, Kligler: not downloaded (size / Kaggle login); UHDM test (500) covers moire.
- UHDM photos are aligned crops of the screen (no bezel); full photos of monitors with bezels are only the 90 curated
  Commons ones.
- DocReal OCR: Tesseract has only `eng` here, so CER is not meaningful on its Chinese pages.
- DIW is 512x512 only (the published zip), so it is for robustness (page finding), not quality.
- Licences: several academic sets state no licence (DIR300, WarpDoc, DocReal, RDD, UHDM): test use only, do not ship.
