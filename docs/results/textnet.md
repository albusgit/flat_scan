# Rule + text-line net (tools/text_net)

The rule net's model is replaced by a 4-output RuleNet: horizontal rule, vertical rule, horizontal text line,
vertical text line. Same architecture as the rule net (`tools/rule_net/model.py`), widths 12,24,32,48,64,96:
0.52 M parameters, 1.0 MB with fp16 weights (`assets/models/rulenet.onnx`, `litert/models/rulenet*.tflite`).
The file names, the loader and the GPU copies stay those of the rule net.

## Training data and run
- Text: the Universal Declaration of Human Rights in 380 languages / 28 scripts (NLTK udhr2), each language
  checked against the fonts that cover it (fontTools, the Windows fonts), shaped with Qt / HarfBuzz (bidi,
  joined scripts), on synthetic pages (`synth_text.py`: paragraphs, tables, headings, forms, vertical blocks,
  turned lines, over `tools/rule_net/synth2.py`'s pages, warps and lighting). 36000 text pages + 16000 rule-only
  pages (synth2; text outputs weight 0 there), held out: 1500 + 1000.
- Text-line label: the band where a line's ink is densest (rows >= 35 % of the densest row), its middle 40 %,
  from the line's first to its last ink: one definition for every script.
- One run on the RTX 5060, 8.5 h, 264311 iterations (`train_text.py`: bf16, torch.compile, 256 px crops at batch
  32 for 75 % of the time, then 384 px at batch 16). Best checkpoint at iteration 216000 (end of the 256 px
  phase; the 384 px phase ended at the same score, 0.8930 vs 0.8931).

## Measured
Held-out synthetic pages, F1 at 0.5 on centre 384 px crops (`eval_text.py`, report in
E:/fs_datasets/textnet/eval1b/report.md):

| | rule h | rule v | text line h | text line v |
|---|---|---|---|---|
| new net, text pages | 0.8962 | 0.8919 | 0.8905 | 0.9149 |
| old rule net, text pages | 0.7037 | 0.5820 | - | - |
| new net, rule pages | 0.9004 | 0.8647 | - | - |
| old rule net, rule pages | 0.8567 | 0.6968 | - | - |

Text-line F1 per script (h and v together): 0.85 - 0.94 for 32 of 35 scripts; lowest Myanmar 0.770, Arabic
0.797, Thaana 0.813.

LiteRT copies vs the ONNX file (3 real pages, 1152 x 800 and 800 x 1152, CPU interpreter): max difference
4.4e-5, no pixel on the other side of 0.5. GPU op check: every op in a form the GPU delegate takes. Speed on a
phone: not measured (about twice the old rule net's multiply-adds).

Line stage (`core/src/warpmap.cpp` text_rows_net): the text-line maps follow the page through the global stage as
the rule maps do; each band gives a fitted centre line as the ink-blob text rows did (`text_rows`, still used
without the net), vertical text lines go into the vertical solve, and pages with text lines but no rules get the
local stage too (they had a zero field). The lines also count in the line stage's straightness check
(`page_traces`). `FS_NO_TEXT_ROWS=1` (desktop) keeps the net's rule maps without its text lines.

Desktop, the 34 real photos (`tools/library/run_library.py`, real35, CPU, text-row straightness, `straight.py`):

| run | row deviation p90 | angle spread |
|---|---|---|
| sp_new: old rule net (commit 99dd095) | 6.7103 px | 1.7502 deg |
| tn_norows: new net, rules only | 6.7001 px | 1.6830 deg |
| tn_rows: new net, rules + text lines (first fit) | 6.6628 px | 1.3252 deg |
| tn_repo: the same with the shipped fp16 ONNX | 6.7245 px | 1.5214 deg |
| tn_fit2: + the fit that follows curled ends, bends to 40 px | 6.4997 px | 1.4977 deg |

The same without scanner_samples__desk (29 photos scored in every run; on desk the scorer finds other rows from
nearly identical pages: tn_rows 0.27 deg, tn_repo 6.24 deg for maps within 0.001 px of straightness), mean / median:

| run | row deviation p90 | angle spread |
|---|---|---|
| sp_new | 6.635 / 6.499 px | 1.588 / 0.825 deg |
| tn_norows | 6.624 / 5.835 px | 1.519 / 0.751 deg |
| tn_repo | 6.701 / 5.799 px | 1.359 / 0.755 deg |
| tn_fit2 | 6.493 / 6.146 px | 1.289 / 0.693 deg |

The first fit took text_rows' rules: a polynomial of degree <= 3, centres > 2x the median residual (> 1.5 px) left
out, lines bending > 15 px left out. On docres 208 that dropped the two most curved lines (bends 22 and 15.3 px) and
12 - 33 of ~130 centres per body line, many at curled line ends, which stayed curled. tn_fit2: degree 1 - 5 by length,
> 3x the median (> 2.5 px), bends to 40 px; on that page all three deck lines are used and 3 centres per body line
are left out. tn_fit2 vs tn_repo: row deviation better on 16 photos, worse on 5.

tn_rows vs tn_norows: row deviation better on 12 photos, worse on 8; angle spread better on 16, worse on 7. The
two photos that lost most on row deviation (dewarpnet 13_2, docres 208) look the same side by side, as does the
receipt that lost most against sp_new (13.1 -> 18.3 px).

Still open: large display type (a magazine headline) is marked as rules and vertical text by the net (no such
fonts in the synthetic pages): docres 208's title gives 12 + 17 false "rules"; bands taller than 40 px and shorter
than 120 px are not used.

Android: the debug batch mode on the x86_64 emulator ran two real photos with the new net (`text_net` true; the
cookbook page, which has no rules, got 33 text rows). Its net time there (16 s, emulated CPU) says nothing about a
phone.
The DocUNet / DIR300 geometry scorer (Desktop/docunet_bench) no longer exists here, so LD / MS-SSIM were not
measured.
