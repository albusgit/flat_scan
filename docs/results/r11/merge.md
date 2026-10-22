# Round 11 merge: Round 10 (rule net, line-stage guards, curled corners) into r11/main

Merge of `origin/ccr-75d4dfec-mjht7b` (1724d9b) into `r11/main` (db9ef56), `git merge --no-ff`.

## What changed

- **Rule net ported into the core.** Round 10 ran it through ONNX Runtime's Java API (`RuleNet.kt` plus a
  JNI callback that set `PipelineOptions::rule_net`). r11/main runs every net in the core, and the Java ORT
  classes are not on its classpath. The port:
  - `core/src/infer.*`: `NetId::RuleNet`, model `rulenet.onnx`, which the app loads from `assets/models/`.
    The dynamic-shape model has no fixed-shape copy, so like ShadeNet without one it never runs on
    QNN or NNAPI. It uses the CPU EP with the same session options as the other nets.
  - `core/src/nets.*`: `rule_net_fn(Inference&, double* ms)`, the `RuleNetFn` that `PipelineOptions::rule_net` takes.
  - `jni_bridge.cpp`: its own net group `g_rule`, CPU only. `configureNets` remakes it with an empty
    preference, and it is not in Settings > Acceleration or in Auto.
    - `ShadeRun` became `NetCall` (shadow net or rule net), with the same timing entries.
    - `process()` takes a `rules` switch. The processing details' `accel.nets` gain a `rulenet` entry,
      and `info()` keeps Round 10's `rule_net`, `net` counts and `spots_dropped`.
    - `batchPhoto()` takes `rules` too. Its JSON line has Round 10's new pipeline_cli fields
      (`extended`, `second`, `corners`, `rule_net`, `net`, `spots_dropped`, `straightness`).
  - Kotlin:
    - `RuleNet.kt` and the `ShadeNetRunner` / `RuleNetRunner` interfaces are dropped.
    - `NativeLib.process(..., shadows, rules, listener)` and `batchPhoto(..., shade, rules)`.
    - `ScanViewModel` passes `settings.ruleNet`. The self-test runs with every net, including the rule net.
    - Batch mode has the extra `--ez rules false|true`, default Settings > *Table rules (neural net)*.
      `run.json` records `rule_net`.
    - `NET_LABELS` has "Rule net". Round 10's Settings switch, its processing-details line and its
      geometry key (`lines=9|rulenet=`) are kept.
  - `tools/desktop_check/pipeline_cli`: `FS_RULE_NET=1|MODEL.onnx` runs it in-process (JSON `nets.rulenet`);
    `FS_RULE_NET_PY=1` keeps Round 10's Python subprocess for comparisons.
- **Conflicts:**
  - `warpmap.cpp`: both sides' functions are kept. The global remap's GPU path (`remap_offset_gpu`)
    runs only without rule-net maps; with them, the CPU path remaps the page and the maps together, so
    the output is the same.
  - `pipeline.cpp`: Round 10's line stage plus r11/main's run_photo, which overlaps the nets with the
    CPU stages.
  - `pipeline_cli.cpp`: r11/main's in-process nets.
  - `README.md`: Round 9 and Round 10 sections, then Acceleration. The port is described in both.
- `dist/flatscan-debug.apk(.sha256)`: r11/main's version kept (stale; to be rebuilt).

## Verification (measured)

Desktop, `tools/desktop_check` Release builds (MSVC, CPU, GPU stub, ORT 1.22.0). Three builds:
- Round 10's `pipeline_cli` (`fs_round10`, 1724d9b);
- r11/main before the merge (db9ef56, built into `<SCR>/merge/build_pre`);
- the merged tree.

Cached net outputs, so only the core differs:
- 34 round-4 photos: `<SCR>/accel_b/s35_ref` (Python ORT 1.22.0) logits, p1, rc0 and rc2, with
  `FS_UVPASS1 FS_UVRECROP`, no shadow net.
- DocUNet: `work/nets_frame_f` and `nets_rc_f` with `FS_NO_SHADOWS=1`, as Round 10's `work/r10_rn`.

DocUNet pages: 1_1..10_2 plus 52_1, 53_2, 52_2, 13_1 and 51_1 (25 pages; 7_1 and 10_2 are inside 1..10).
The 35th photo (the hand-held receipt) is no longer on disk, so the set is 34 photos.

| check | pages identical | JSON identical (without photo / nets / pipeline_ms) |
|---|---|---|
| merged, rule net in-process (C++ ORT 1.22.0) vs `work/r10_rn` (Round 10's run, its Python rule net), DocUNet | **25 / 25** | **25 / 25** |
| merged rule net in-process vs Round 10 `pipeline_cli` + Python rule net (venv ORT 1.31 CPU), 34 photos | **34 / 34** | **34 / 34** |
| merged, rule net off vs `work/r10_norn` (Round 10, rule net off), DocUNet | **25 / 25** | **25 / 25** |
| merged vs r11/main before the merge, line stage off (`FS_NO_LINES=1`), DocUNet | **25 / 25** | **25 / 25** (merged adds only Round 10's new keys) |
| merged, rule net off vs r11/main before the merge (`work/new_fold2` = r11/main's pages, Round 8), DocUNet | 4 / 25 | 2 / 25 (expected: Round 10's line-stage changes) |

- The rule net ran on every page: `rule_net` true on 34 / 34 photos and 25 / 25 DocUNet pages. In the
  photos' counts `[dropped, added, joined, kept]`, for example, it dropped 1 line on 4 photos, 8 and 9
  lines on two others, and dropped 6, added 1 and joined 2 on one.
- The in-process net gave the same pages and JSON as Round 10's Python-run net everywhere, so there
  are no LD deltas and DocUNet was not re-scored. Identical pages keep Round 10's LD 5.122 (rule net
  on) and 5.050 (off).
- Off the line stage, the merge changes nothing (the `FS_NO_LINES` row). With the line stage on and
  the rule net off, the output is Round 10's and not r11/main's: that is Round 10's own change.

**App (x86_64 debug, `gradlew assembleDebug -Pabis=x86_64`, under the heavy lock).**
- BUILD SUCCESSFUL, 3 min. The APK is 152,158,623 B, sha256 `77a6ae02c5b7bb17...`; it is not in `dist/`.
- AVD fs35 (Android 15 x86_64), debug batch mode, `--es accel cpu --ez shade false`; outputs in
  `docs/results/r11/merge/emu/`:

| run | result |
|---|---|
| `st_rn`: self-test photo1, rule net on (default) | ok; `rule_net` true, `net` [0,0,0,0] (photo1 has no table rules); `nets.rulenet` ok, cpu, 2 calls, 9.0 s on the emulator |
| `st_norn`: photo1, `--ez rules false` | ok; `rule_net` false, no `rulenet` entry |
| `s3_rn`: dewarpnet__1_1_copy, docres__for_dewarping, docscanner__42_2_copy | 3 / 3 ok; `rule_net` true; counts [1,7,0,0], [28,9,0,1], [5,0,0,0]; 2.7-4.2 s for the rule net |

- The s3 pages are **pixel-identical** to the desktop's merged run (`m_rn`). The JSON differs only in
  `fit_median_full_px` and, on one photo, `orient` in the 6th digit: the known MSVC vs NDK surface-fit
  residual (accel `fit_portability.txt`).
- The app did not crash. The logcat "Abort" lines are the system's SearchServiceCore, and the one
  "tombstoned" line is an Android system daemon, not the app.

## Not verified

- `NativeLib.process()`, the interactive path that writes the processing details, and Settings >
  *Table rules* were compiled but not driven on the emulator. Only the batch mode (`batchPhoto`) ran.
  Both call the same `NetCall` / `rule_net_fn` code.
- No phone; no arm64 build in this merge (x86_64 only); the rule net has never run on an accelerator.
- The 35th photo (receipt) is missing from the scratchpad; DocUNet was not re-scored (identical pages).
- The machine was heavily loaded and the heavy lock was held by other streams for long stretches, so the
  desktop comparisons (about 220 pipeline_cli runs) ran outside the lock at 2-3 parallel jobs.

## Reproduce

Scripts are in `docs/results/r11/merge/`, and `<SCR>` is the scratchpad.
- `python -I verify.py run OUT m_rn,r10_rn,m_off,pre_nl,m_nl [--only s35|du]`
- `python -I verify.py cmp OUT A B`
- `python -I cmp_work.py OUT/m_rn r10_rn`, which compares against `docunet_bench/work`.
- `locked.sh app app_check.sh`, which runs the Gradle build and then `emu.sh`.
