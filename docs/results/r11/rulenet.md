# Round 11, RULENET stream: rule net v2 (WIP notes, numbers filled in as measured)

Branch `r11/rulenet` (worktree `C:\Users\User\Desktop\fs_r11_rulenet`), from Round 10 (1724d9b), r11/main
(ddfd6f0) merged in. Work files outside the repo: `C:\Users\User\Desktop\flatscan_out\r11\rulenet\`.

## 1. Why Round 10's rule net made DocUNet LD worse (diagnosis)

Round 10: LD 5.122 with the rule net, 5.050 without (composed, 128 photos). Largest losses 52_1 11.34 -> 17.14,
53_2 8.62 -> 13.12, 52_2 5.23 -> 8.58, 13_1 8.32 -> 9.73.

Reproduced bit-exactly (pipeline_cli of the Round 10 worktree, app config: cached nets nets_rc_f / nets_frame_f,
SD-HR first then UVDoc, FS_NO_SHADOWS=1; `diag/run1.py`), with the line stage's segments and the net's map dumped
(FS_DUMP) for 12 pages.

**a. The three largest "losses" are the scorer, not the pages.** On 52_1, 52_2 and 53_2 the rule-net page and
the no-net page differ by 0.08, 0.14 and 0.42 px on average (Farneback flow between the two outputs at 1/4 size,
p90 0.17 / 0.33 / 1.32 px). Re-scoring the *no-net* pages after a 0.25-px shift, a JPEG q95 pass or a 0.2 %
scale (DocUNet scorer, `diag/score_pert.sh`, 8 pages):

| page | no net | shifted 0.25 px | JPEG q95 | scaled 1.002 | rule net (R10) |
|---|---|---|---|---|---|
| 52_1 | 11.34 | 19.52 | 13.65 | 11.31 | 17.14 |
| 53_2 | 8.62 | 18.95 | 18.09 | 17.96 | 13.12 |
| 52_2 | 5.23 | 4.78 | 8.35 | 5.04 | 8.58 |
| 27_2 | 4.85 | 4.90 | 5.58 | 6.47 | 5.91 |
| 7_1 | 13.05 | 13.71 | 12.83 | 11.65 | 11.83 |
| 13_1 | 8.32 | 8.39 | 8.33 | 8.86 | 9.73 |
| 5_2 | 3.91 | 3.94 | 3.91 | 4.10 | 4.67 |
| 56_1 | 12.27 | 12.24 | 13.14 | 12.50 | 13.14 |

SIFT flow on dense text pages jumps by several px for sub-pixel changes. Those three pages alone account for
(5.80 + 4.50 + 3.35) / 128 = 0.107 of the 0.072 mean-LD loss. So every DocUNet comparison below is also
reported on a 0.25-px-shifted copy of each run's pages (scored in full), and per-page wins / losses count only
changes beyond that page's noise.

**b. Real losses: what the net marked.** (13_1, 5_2, 56_1, 14_1, ...)
- It drops straight edges that are not thin rules: the sides of pictures, of dark title bars and of coloured
  boxes (13_1, 56_1). The no-net line stage traced them and used them; Round 10's generator taught the net that
  picture / colour-block edges are *not* rules.
- It marks barcodes as vertical rules and the stems of big bold letters ("RELIEF" on 13_1); added as net-only
  rules, they bend the page (13_1, 5_2).
- On the owner's page 1 it marks rules printed on the back of the sheet and on a sheet behind (darkness 7 - 17):
  the generator had no show-through, no sheet behind, and no faint front rules to tell them apart.

**c. The line stage's use of it.** When the net drops every found line (no rules left), the line stage does not
run at all, so the text rows go too (53_2, 7_1, 51_1, 36_1, 33_1, 36_2, 53_1). That is the round-4 "no grid lines,
skip" rule, not the net; the effect on those pages is mixed (3 better, 4 worse, mostly within scorer noise), so
it was left alone. Joining and the darkness >= 25 gate for net-only rules behaved as designed.

Verdict: mostly the net (its training data), plus scorer noise; no line-stage (C++) change was needed for the
measured losses.

## 2. Generator v2 (`tools/rule_net/synth2.py`)

Adds the cases the diagnosis found missing (see the file's docstring): show-through rules, a sheet behind, faint
front rules, picture / colour-block edges as non-rules. Trained with `train2.py` (0.479 M params, same net as v1,
34 000 its, run `a`; checkpoint `flatscan_out/r11/rulenet/runs/a/best.pt`, ONNX `.../models/v2a.onnx`).
Pool F1 (2000 held-out, mean over classes): v2 0.8785 (v2val and v1val sets; v1 not re-scored here).

## 3. Measurements (all with `tools/desktop_check` pipeline_cli, app config; off = no net, v1 = Round 10 net)

| test | off | v1 | v2a |
|---|---|---|---|
| synth2 e2e (40 held-out pages, `e2e_eval.py`): rules >2 px off, in 34.6 % | n/a | 10.3 % (p90 2.09 px, rejected 6/40) | 12.1 % (p90 2.55, rejected 7/40) |
| dir300 every 6th (n=50, run_library): LD / MS-SSIM / row dev p90 | 7.399 / 0.6501 / 5.00 | 7.414 / 0.6510 / 5.28 | 7.474 / 0.6501 / 5.25 |
| inv3dreal_s90 (n=90): LD / MS-SSIM / row dev p90 | 24.540 / 0.5939 / 5.67 | 24.422 / 0.5944 / 5.60 | 24.571 / 0.5941 / 5.54 |
| 35 real photos (n=30 text pages): text-row dev med / p90 | 4.157 / 6.616 | 4.140 / 6.710 | 4.264 / 6.718 |
| 35 photos table lines med (n~19) | 1.697 | 2.266 | 1.940 |
| DocUNet 130 pp, composed as Round 10 (changed pages scored): LD / MS / AD | 5.050 / 0.5899 / 0.1869 | 5.122 (R10 figure) | 5.152 / 0.5904 / 0.1880 |
| DocUNet, every run shifted 0.25 px, all 130 pages scored: LD | 5.804 | 5.761 | 5.797 |

(docreal not run. v1 on dir300/inv3d/35: same commands as v2a with the Round 10 rulenet.onnx.) v2a on DocUNet:
119 of 130 pages differ from no-net; 39 better, 41 worse beyond 0.05 LD; the biggest "losses" are the same
scorer-noise pages (52_1, 53_1, 52_2). The shifted-copy numbers move by 0.04-0.75 on single pages, so the 0.03
mean differences between runs on DocUNet are inside the noise.

## 4. Decision

Rule: ship v2 only if not worse than v1 and off on validation and on DocUNet beyond noise. v2a is not better than
v1 on any held-out measure that separates them: synth2 e2e is slightly worse than v1 (12.1 vs 10.3 % >2 px), dir300
LD is worse than both (7.474 vs 7.414 / 7.399), inv3d LD worse than v1, 35-photo row dev worse than v1, DocUNet
LD 5.152 vs 5.122 (v1) in both the composed and the shifted runs. Its gains (row dev p90 on inv3d/dir300, table
lines on the 35 photos) are small and mixed. **v2a is not shipped; `rulenet.onnx` (v1) stays.** No model file or
`rulenet.tflite` changes. The tflite of v1 built for the LiteRT stream is in `flatscan_out/r11/rulenet/tflite_v1/`.

The real finding of the stream is the diagnosis: most of the Round 10 DocUNet LD loss of the net is scorer noise
(Section 1), so the net should not be judged on DocUNet LD alone; and a retrain on better synthetic data did not beat
v1 on the real pages.

## 5. Not verified
- docreal; a phone run; run `b` (second v2 config, val F1 0.8807 at 24 000 its) was not evaluated end to end.
- Whether v2 would help after more real-photo training data; only synthetic + the pools were used for training.
- Single runs per config; no seeds repeated.

## 6. Reproduce
`tools/rule_net/{synth2,gen_pool,train2,eval_pool,e2e_eval,to_tflite}.py`; library runs `lib_runs.sh`, DocUNet
`docunet/eval_docunet.sh NAME MODEL.onnx` (in flatscan_out/r11/rulenet).

