# Round 11, UI stream: review, editing settings, visual items

Branch `r11/ui` (from `r11/main` db9ef56, r11/main ddfd6f0 merged in). Screenshots:
`C:\Users\User\Desktop\flatscan_out\r11\ui\` (`before/`, `after/`, `before_sheet.jpg`, `after_sheet.jpg`).
Everything ran on the Android 15 x86_64 emulator (AVD fs35, 1080 x 2400 px, 420 dpi = 411 x 914 dp, no GPU),
CPU nets. **Nothing has run on a real phone.** During the emulator sessions the host was at 100 % CPU from
other streams and the 2 GB emulator was swapping, so timings and smoothness were **not measured**.

## 1. Review of the UI as it was (db9ef56)

Screens walked: home, gallery import (system photo picker), crop (detecting), processing, result (filters,
adjust), export sheet, pages, home with a document, settings (`before/`, `before_sheet.jpg`). The camera
screen, the line editor and landscape were read in the code only.

Common path, taps from the home screen (unchanged by this work, it was already short):
camera 1, shutter 1 (a trusted crop goes straight to processing), Save 1, PDF 1, the system Save 1 = **5 taps**;
gallery: Import 1, photo 1, Add 1, then the same 3 = 6.

Problems found (most important first):

1. **Every page is A4-shaped.** The core outputs an A4-proportioned page whatever the paper
   (`core/src/pipeline.cpp` comment at `page_hw_ratio`), so a receipt (height/width ~3-4) is squeezed to 1.41:
   its letters come out 2-3x too wide. There was no way to change it.
2. **PDF export stretched every image to the A4 page** (`Exporter.writePdf` drew into `Rect(0, 0, 595, 842)`),
   with no paper, margin or size options. Harmless for the A4-shaped pages of (1), wrong for anything else.
3. **No global undo.** Undo existed only inside the box tools (protect, erase); a filter, slider, rotation or
   level change could not be undone.
4. **Few adjustments.** Adjust had background cleanup and levels only: no brightness, contrast, saturation or
   warmth, which scanner apps have. The panel was tall (the histogram) and pushed the page up.
5. **Multi-page friction.** A filter chosen on one page could not be applied to the others (it only becomes the
   default for new pages; every existing page had to be opened and changed). Reordering pages took 2 taps per
   step through a "..." menu (long press opened the same menu); the grid was fixed at 2 columns on any width.
6. **Share was hidden** in the result screen's overflow menu; Save and Share were separate sheets.
7. **No document name**: exported files are always "Scan yyyy-MM-dd HH.mm".
8. **Touch targets < 48 dp**: the editor chips (Solid colors, Erase, Add page, Undo, Remove all ...) were 36 dp
   high with no larger touch area; the page card's "..." button 36 dp; the crop note's close 40 dp (not changed).
9. **Truncated label**: the filter strip cut "High contrast" to "High" (66 dp column; `before/05`).
10. **Landscape / tablets**: the result screen stacks the page over a ~280 dp panel; on a landscape phone
    (~411 dp tall) the page gets a sliver (code reading; the activity allows rotation).
11. **Camera screen**: only a shutter and Back: no torch, no gallery shortcut, no sign of the pages already in
    the document.
12. **Processing screen**: a bar and the core's raw stage names; no sign of which photo is being processed.
    While the page finder runs on the crop screen, the photo area is an empty box with a spinner (`before/03`).
13. **Settings** mixes user preferences with engineering switches (each net, acceleration, self-test,
    benchmark). Not changed here (other streams edit Settings this round); recommended: an "Advanced" section.
14. The "180°" button is text in a circle that reads as a badge; the "Press and hold" hint sits over page text.

Not found to be problems: contrast of the editor and light-theme texts (Editor.Muted A3A5AB on 17181B,
onSurfaceVariant 55575E on F6F5F2: both well above 4.5:1), content descriptions of icon buttons, the dock's
labelled icons, the empty home state.

## 2. What changed

All new behaviour is off or identical at the defaults; new code is mostly in new files.

| Area | Change | Files |
|---|---|---|
| Page shape | Adjust > Page shape: **A4** (default, as processed), **As photographed** (the page's own height/width, measured from the crop corners with Zhang & He's rectangle-from-perspective method, the camera focal length when known, else estimated from the quad; falls back to the side lengths), **US Letter**, **ID card**. The finished page is resized to the shape at the same pixel count (a receipt gets taller and narrower). Protected / erased / ignored areas are page fractions, so they stay put. The chosen shape is remembered for new pages. | `PageLook.kt`, `AdjustTools.kt` |
| Light & colour | Adjust > **Brightness, Contrast, Saturation, Warmth** (-100..+100, tap the value to reset). One colour matrix: a Compose colour filter on screen (no re-render), one Canvas pass when out.jpg is written. Cleanup and Levels moved into the same tool row, so the panel is one control high. A dot marks tools that changed the page. | `PageLook.kt`, `AdjustTools.kt` |
| Undo / redo | Floating Undo / Redo over the page for every edit that does not re-run the geometry (filter, strength, Solid colors, adjust, colours, shape, rotation, protect and erase boxes). Settled states are recorded after 450 ms (a slider drag is one step); 60 steps per page; a new page starts a new history. | `EditHistory.kt`, `ScanViewModel.kt` |
| Compare | A Compare button on the page: original left, cleaned right, drag anywhere to move the split; "Original" / "Cleaned" labels. Hold-to-see-original is unchanged. | `PageViewer.kt`, `ResultScreen.kt` |
| Apply to all | Filters panel chip "Apply to all N pages": the current filter, strength, Solid colors, adjustments and colours on every page (each keeps its shape, rotation and boxes). Each page is rendered again from its stored session on the native thread, one at a time; out.jpg and meta are rewritten. | `ScanViewModel.kt` (`applyToAllPages`), `Document.kt` (`sessionDir`) |
| Export | One sheet with **Save to device / Share** segments, a **File name** field (the document's name), PDF / JPG, and **PDF options**: paper A4 / Letter / Same as the page, margins None / Small / Normal, images Full / Smaller file (<= 150 dpi). The PDF keeps the image's proportions (centred); an image within 1 % of the box's shape fills it as before. Choices are remembered. | `ExportOptions.kt`, `ExportUi.kt`, `Exporter.kt` |
| Pages | Hold and drag a page to reorder; title = document name, tap (or menu > Rename) to rename; adaptive grid (150 dp columns); 48 dp menu button; the subtitle says "hold and drag to reorder". | `PagesScreen.kt` |
| Camera | Torch toggle (when the camera has a flash unit), gallery import next to the shutter, the document's page count (opens Pages). | `CameraScreen.kt` |
| Processing | The photo being flattened with a scan line at the progress, the stage and a percentage; the bar animates between the core's steps ("Opening the page" when a stored page opens). No endless animation. | `Screens.kt` |
| Result layout | Landscape phones / tablets (width > height and >= 600 dp): the panel goes beside the page (400 dp wide, scrolls). | `ResultScreen.kt` |
| Small fixes | Editor chips keep their 36 dp look with a 48 dp touch target; filter column 80 dp ("High contrast" fits); the home card shows the document name. | `ResultScreen.kt`, `Screens.kt` |

Persisted data: `meta.json` gains an optional `look` object (written only when not default; older metas read as
default); new preference files `look` (remembered shape) and `export` (PDF options); `cache/document/name.txt`
(removed with the document).

## 3. Verification

Build: `gradlew assembleDebug -Pabis=x86_64` of 5dbbb96 (BUILD OK); installed on AVD fs35 (Android 15 x86_64), driven with
adb input / uiautomator dump (`flatscan_out/r11/ui/tools/ui.py`). Screenshots: `flatscan_out/r11/ui/after/` (earlier
numbered ones plus `v2_*`). Two photos imported through the system picker (Add (2)), crop -> result -> next page.

Driven on the emulator, no crash (`logcat` has no FATAL EXCEPTION for com.albus.flatscan):
- Adjust: Brightness +48, Warmth, Contrast sliders move, dot marks changed tools; Undo / Redo step the values back and forth.
- Page shape: A4 -> As photographed (caption and page change); Undo restores A4.
- Compare: split view with Original / Cleaned labels, drag moves the split.
- Apply to all 2 pages (chip only after page 2 is processed): chose Grayscale on one page, the other page then showed Grayscale.
- Pages: drag (draganddrop, 1.5 s) swapped the two pages; Rename dialog changed the title to Invoice_test; Save as PDF
  took that name.
- Export: PDF with defaults -> 2 pages (pulled, 2 x /MediaBox, 3.56 MB). Export sheet > PDF options Letter / Normal margins /
  Smaller file -> summary line updates, PDF has 2 pages, MediaBox 612x792, 1.97 MB. (Structure checked from the PDF
  bytes, pages not rendered.)
- Camera: Torch toggles on/off, gallery button opens the picker, shutter goes to Adjust corners; page-count button shown.

Byte identity at defaults (debug batch mode, `--es accel cpu`, photo1 + the Round 4 receipt, same emulator session):
r11/main ddfd6f0 APK (built earlier in fs_r11) vs this branch's APK. `photo1.jpg`, `receipt.jpg` and all `*.npz` are
byte-identical (`cmp`); the JSONs differ only in the `ms` / `total_ms` timings. This covers the core/native path only:
the Kotlin render of the colour matrix at default (identity) and the export path were not diffed against main.

Not checked: no phone (no arm64 build, no real camera or torch hardware), landscape/tablet layout, Processing scan line
visually (seen only in a mid-run dump), the SAF file picker flow beyond Save, PDF viewed in a reader.

