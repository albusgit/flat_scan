# Acceleration: gate status

Final Coordinator merge, 2026-10-21. **Speed mode** (owner's decision): no DocUNet scoring and no full DocUNet
run. Instead, one working check on the merged accel/main shows that everything runs and that the CPU path is
unchanged. Bit-identical outputs mean unchanged scores (DocUNet LD 5.162 / MS-SSIM 0.5898 / AD 0.1893,
`work/new_fold2`). The full gates are listed below with their commands; they have not been run.

- `<SCR>` = `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad`.
- `<OUT>` = `C:/Users/User/Desktop/fs_accel_out`. It holds the final APK, plus copies of the check scripts in
  `<OUT>/check/`, because the scratchpad is temporary.
- `PY` = `C:/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe` (always run with `-I`).

## Merge log (accel/main)

| Commit | What | Branch head |
|---|---|---|
| da8e67f | Stream B contract: `core/src/infer.h` + stub `infer.cpp` | 1a04235 |
| 7412c60 | accel/c: Stream C step 1, Phase 0 instrumentation | 92dd993 |
| 2b4ff45 | accel/a: static UVDoc / SynthDoc-HR, ShadeNet 512x352, Gate A logs | c879229 |
| 233fa5a | accel/b: inference on ORT C++ 1.22.0, nets in the core, pipeline_cli in-process | b0af623 |
| 3ba797d | accel/e: GPU image stages, Windows Mesa host harness, CG tolerance option | 1935dbb |
| 7dc7ce0 | Coordinator wave 1: STATUS, `.gitattributes`, ignore `build-host-win/` | |
| 22d7981 | accel/c2: Stream C steps 2-5: ORT C++ in libflatscan, nets over JNI, Settings > Acceleration, Auto, debug batch mode | b36bb2b |
| f906ac4 | accel/d: flattening nets beside the CPU stages (`PipelineOptions::flat_nets`), Gate D script | dfc7e42 |
| 25c0da1 | accel/f: desktop GPU (WGL / Mesa), DirectML / CUDA builds, CUDA without TF32, Gate F tools | 1386f9d |
| d4016c0 | **Fix (Gate C)**: Auto no longer tries NNAPI | |
| 94f9a0a | **Fix (Gate C)**: Auto tries QNN only; NNAPI is no longer offered in Settings | |
| (this commit) | README Acceleration section and Architecture table; this file | |

- Merge order: B contract -> C1 -> A -> B -> E -> C -> D -> F. PLAN.md puts E after C. E went in early so that
  Stream F could start on B + E.
- All merges were real merges, with no conflicts. accel/f and accel/d both changed
  `tools/desktop_check/pipeline_cli.cpp`; git merged them automatically, and the checks below cover the result.
- Build checks after each merge:
  - after C2: `./gradlew assembleDebug -Pabis=arm64-v8a,x86_64` passed;
  - after D: desktop_check and the same Gradle build passed;
  - after F: desktop_check, the host harness and the four Gate F builds passed.
- Gradle builds always held the heavy lock. The only compiler warnings are ones that were already there
  (warpmap.cpp:1095 unused `y`; MSVC C4068 for `#pragma STDC FP_CONTRACT` in pipeline.cpp).

### Failures found, and what was done about them

1. **Gate C: Auto crashed the app on the x86_64 emulator.** The cause is C2 (22d7981: Auto's candidate list and
   the NNAPI setting).
   - First run: `benchBackends` tried NNAPI (nnapi-reference), and `libonnxruntime.so` called `std::terminate`
     inside `Run` on `uvdoc_static.onnx` (SIGABRT, "libc++abi: terminating"). Logcat: `<SCR>/coordF/emu/logcat_all.txt`.
   - Fix 1 (d4016c0) removed NNAPI from Auto. The second run then aborted the same way on the XNNPACK EP.
     Logcat: `<SCR>/coordF/emu2/logcat_all.txt`.
   - Choosing NNAPI by hand aborts as well (`--es accel nnapi`, same log).
   - An abort inside ORT skips the CPU fallback that PLAN hard rule 1 requires. Fix 2 (94f9a0a) therefore makes
     QNN Auto's only candidate (none on x86_64, so Auto picks the CPU there), and Settings offers Auto / CPU /
     NPU only. A stored `nnapi` setting reads as CPU.
   - The NNAPI path is still in the code and reachable from the debug batch mode (`--es accel nnapi`) for a
     device check later (PLAN Stream G). Why ORT aborts was not found: libonnxruntime.so has no symbols.
     Neither EP has been tried on a real phone.
   - Static models on the emulator's CPU EP (`--ez static_models true`) run fine and give the same output, so
     the static models alone are not the cause.
2. Nothing was reverted. Gates A, B (CPU path), D, E and F passed their checks.

## Final working check (accel/main 94f9a0a; the docs commit changes no code)

| Check | Result | Command (Git Bash) | Output |
|---|---|---|---|
| CPU path, 35 photos (34 round-4 + the receipt), in-process nets, ShadeNet on, overlapped (default) and sequential (`FS_SEQUENTIAL=1`) | **pass**: 0 of 35 differ vs the Python ORT 1.22.0 reference (page finder logits, the three grids, page, JSON), both orders; overlapped vs sequential 0 of 35 | `bash <SCR>/coordF/cpu_check.sh <SCR>/coordF/cpu` (240 s; heavy lock for the DocUNet part) | `<SCR>/coordF/cpu/cmp_s35_ref_vs_ovl.txt`, `cmp_s35_ref_vs_seq.txt`, `cmp_s35_seq_vs_ovl.txt` |
| CPU path, DocUNet 1_1..10_2 (20 photos, NOTES.md config via `gate_b.py`), both orders | **pass**: 0 of 20 differ vs the Gate B reference; `vs_images` 20 of 20 identical to `docunet_bench/work/new_fold2`; overlapped vs sequential 0 of 20 | same script | `<SCR>/coordF/cpu/cmp_du_ref_vs_ovl.txt`, `cmp_du_ref_vs_seq.txt`, `cmp_du_seq_vs_ovl.txt` |
| Overlap redo counts (Stream D) | orientation page redone on 3 of 35 photos and 4 of 20 DocUNet photos; orientation decision redone on 11 of 35 and 17 of 20 | `$PY -I <SCR>/coordF/redo.py <SCR>/coordF/cpu` | stdout (numbers here) |
| A: static vs original, fp16 stand-in, 5 photos | **pass**: fp32 static == original 20 of 20 runs; fp16 stand-in grid p99: UVDoc 0.084 / 0.076 px (meets < 0.25); **SynthDoc-HR 1.48 px, max 2.64 px (misses; already known from Gate A)**; ShadeNet p99 0.06 levels | `C:/Users/User/Android/accel_venv/Scripts/python.exe -I tools/accel/compare_outputs.py <SCR>/accel_a/in --log <SCR>/coordF/gateA/compare_outputs_5.txt dewarpnet__1_1_copy docres__for_dewarping page_dewarp__boston_cooking_a scanner_samples__receipt user__projector_screen` (32 s) | `<SCR>/coordF/gateA/compare_outputs_5.txt` |
| A: NNAPI usability, models reproduce | **pass**: 1 partition / 100 % for uvdoc_static (103/103), syntheticdoc_hr_static (107/107), shadenet_512x352 (64/64); a re-run of `prepare_models.py` leaves `git status` clean | Gate A commands (below) | `<SCR>/coordF/gateA/usability_*.txt` |
| E: GPU vs CPU self-tests | **pass**: `selftest-resample` 30 cases worst \|diff\| 0, dense / offset GPU stages worst \|diff\| 0, on llvmpipe and on Mesa d3d12 (RTX 5060) | `cmd //c <SCR>\coordF\build_host.bat`; `GALLIUM_DRIVER=llvmpipe build-host-win/fs_cli.exe selftest-resample`; `build-host-win/fs_cli.exe selftest-resample` | `<SCR>/coordF/gateE/selftest_{llvmpipe,d3d12}.txt` |
| F: desktop GPU, 5 photos (`gate_f.sh smoke`) | **pass** (`gate F: PASS`, 416 s): CPU backend 0 of 5 differ in all four builds; selftest worst 0 on native ES 3.2, GL 4.6 core, d3d12, llvmpipe; gpu-stages 0 not identical; FS_GPU=1 pages / JSON 0 of 5 differ; DirectML and CUDA nets PASS (grid p99 <= 0.025 px); decisions 0 of 5 changed | `bash tools/desktop_check/gate_f.sh <SCR>/coordF/gateF smoke` | `<SCR>/coordF/gateF/*` (`timing.md`, `nets_{dml,cuda}.txt`, `cmp_cpu_*.txt`, `selftest_*.txt`) |
| C: Gradle | **pass**: `assembleDebug -Pabis=arm64-v8a,x86_64` and `assembleDebug` (arm64, QNN) | `bash <SCR>/coordF/gradle_locked.sh LOG assembleDebug [-Pabis=arm64-v8a,x86_64]` | `<SCR>/coordF/gradle_*.log` |
| C: emulator (AVD fs35, x86_64, API 35), debug batch mode on the bundled self-test photo (`assets/selftest/photo1.png`, no shadow net, PNG), Acceleration=CPU | **pass**: page finder logits, p1, rc0, rc2 bit-identical, page pixel-identical to the desktop CPU (pipeline_cli, in-process nets); JSON differs only in `fit_median_px` 1.87128 vs 1.87126 (known: MSVC vs NDK clang) | `NNAPI_RUN=0 bash <SCR>/coordF/emu_check.sh <SCR>/coordF/final_both3.apk <SCR>/coordF/emu4`, then `$PY -I <SCR>/coordF/cmp_st.py <SCR>/coordF/st_desk <SCR>/coordF/emu4/st_cpu` | `<SCR>/coordF/emu4/cmp_st_cpu.txt`, `emu4/st_cpu/` |
| C: emulator, Acceleration=Auto | **pass after the fixes**: the Auto benchmark ran with no candidates and chose the CPU; output identical to the CPU run; no crash (the app was still running at the end) | same run, `st_auto` | `<SCR>/coordF/emu4/cmp_st_auto.txt`, `emu4/logcat_all.txt` (`auto acceleration {...}`) |
| Final APK | `assembleDebug` (arm64-v8a, QNN): 162,438,548 B, sha256 `3d58dd16...ef206` | | `<OUT>/flatscan-accel-debug.apk`; emulator build (arm64 + x86_64): `<OUT>/check/flatscan-accel-debug-arm64+x86_64.apk` |

The desktop reference for the self-test photo was made with
`FS_NETS=1 FS_SEQUENTIAL=1 FS_NET_DUMP=<SCR>/coordF/st_desk/photo1 tools/desktop_check/build/pipeline_cli.exe --backend cpu app/src/main/assets/selftest/photo1.png <SCR>/coordF/st_desk/photo1.png net`.

## Gates

| Gate | Status | Notes |
|---|---|---|
| A Models | partial (pass except the fp16 target) | NNAPI 1 partition / 100 % and fp32 bit-identity pass; the fp16 stand-in misses for SynthDoc-HR (35 photos: p99 0.954 px, max 2.64 px). Real HTP fp16 is HUMAN GATE A-2 (in HUMAN GATE 3). |
| B Core inference | pass on the reduced set; full gate not run | 35/35 + DocUNet 20/130 bit-identical at the final merge; 35/35, round4_check and DocUNet 34/130 at stream time. |
| C Android (agent-checkable) | pass (after fixes d4016c0, 94f9a0a) | Builds; emulator CPU bit-identical on the self-test photo; Auto falls back to CPU without crashing. The full 35-photo emulator run is not done. |
| D Overlap | pass on the reduced set; full gate not run | Overlapped == sequential on 35 + 20 photos. Timing with delayed nets: stream smoke only (3 photos). |
| E GPU image stages | pass on self-tests; full gate not run | |
| F Desktop GPU | pass (smoke, 5 photos); full gate not run | |
| HUMAN GATES 0, A-1, 3, D-1, E (phone GPU), licence, desktop EPs without hardware | open | Steps below. |

### Full gates (not run; commands for whoever runs them)

- **A** (cheap, about 81 s; no heavy lock), in the tree being gated, with `PYA=C:/Users/User/Android/accel_venv/Scripts/python.exe`
  and `S=<SCR>/accel_a`:

      $PYA -I tools/accel/prepare_models.py
      $PY -I tools/shadow_net/export.py --static C:/Users/User/Desktop/shadow_train/final_11k.pt app/src/main/assets/models 352
      $PYA -I tools/accel/net_inputs.py tools/desktop_check/build $S/in        # only if $S/in is gone
      $PYA -I tools/accel/compare_outputs.py $S/in --log docs/results/accel/compare_outputs.txt --dump $S/fp16
      for f in uvdoc_static syntheticdoc_hr_static shadenet_512x352; do $PYA -I -m onnxruntime.tools.check_onnx_model_mobile_usability \
          --log_level debug app/src/main/assets/models/$f.onnx > docs/results/accel/usability_$f.txt; done

  Pass: compare_outputs exits 0; the usability logs say 1 partition, 100 %; `git status` is clean afterwards.
- **B** (about 20 min, estimate; the script takes the heavy lock): build desktop_check
  (`cmd //c <SCR>\coordF\build_dc.bat`), then `bash tools/desktop_check/gate_b_full.sh <SCR>/gateB_final cf1b05d`.
  **BASE_REF must be cf1b05d** (before Stream B). Pass: `cmp_s35.txt` 0 of 35; `round4_check: identical`;
  `cmp_docunet.txt` 0 of 130 with vs_images 130 same. Any vs_images difference must be explained and scored with
  `docunet_bench/docunet_eval.py` (round4 venv, `-I`, `--jobs 4`, under the lock).
- **D** (about 20 min, estimate): `bash tools/desktop_check/gate_d.sh <SCR>/gateD_final <SCR>/accel_b/s35_ref <SCR>/accel_b/du_ref`.
  Pass: `cmp_s35.txt` 0 of 35, `cmp_docunet.txt` 0 of 130 with vs_images 130 same, and in `gate_d_summary.txt`
  the overlapped join at about max(CPU branch, nets).
- **E** (about 30 min, estimate): build the host harness (`cmd //c <SCR>\coordF\build_host.bat`), then
  `bash host/gpu_stage_gate.sh C:/Users/User/Desktop/docunet_bench/work/e_gate full`. Pass: `gate E: PASS`.
- **F** (about 45 min, estimate): `bash tools/desktop_check/gate_f.sh C:/Users/User/Desktop/docunet_bench/work/f_gate full`.
  It needs Stream A's input cache `<SCR>/accel_a/in` and the Gate B reference `<SCR>/accel_b/s35_ref`. Pass: `gate F: PASS`.
- **C, full emulator run** (about 15-25 min, estimate; heavy lock):
  1. `./gradlew assembleDebug -Pabis=arm64-v8a,x86_64`.
  2. Boot fs35 as in `emu_check.sh`.
  3. Push the 35 photos to `files/batch/s35` (HUMAN GATE 3, step 3, has the commands).
  4. `am start -S ... --es batch s35 --es accel cpu`; pull `batch_out/s35_cpu`.
  5. `$PY -I tools/desktop_check/gate_b.py cmp s35 <SCR>/accel_b/s35_ref <pulled dir>`. Expect logits / p1 / rc0 / rc2
     same on 35/35. The JPEG pages are encoded on Android, and `fit_median_px` can differ (open issue).

## Open issues

- **App does not use the overlap yet.** `jni_bridge.cpp` `process()` and `batchPhoto` run `run_flat_nets` and then
  `run_photo`. To use it, set `o.flat_nets = flat_nets_fn(bgr, rq, *g->inf, &keep)` (nets.h). The callback runs on a
  worker thread: no JNI env, no GL context. HUMAN GATE D-1 needs this first.
- **Core JSON platform difference**: the surface-fit residual `fit_median_px` differs between the MSVC desktop and
  NDK clang (emulator) builds: self-test photo 1.87128 vs 1.87126; DocUNet 17_2 11.9626 vs 12.7768 (Stream C).
  Nets, grids and pages are identical, but a value near the 3 px surface threshold could flip the quality gate on a
  phone. This is in the core's C++ (surface fit), not acceleration code. It needs investigation (libm, or an
  iteration difference).
- **NNAPI / XNNPACK abort inside ORT 1.22.0** on the emulator (above). They are not reachable from Settings or Auto.
  Re-test them on a real phone, or with a newer ORT, before re-enabling them (PLAN Stream G).
- SynthDoc-HR's fp16 stand-in misses the 0.25 px target. Auto (per group) would then keep UVDoc on the CPU too,
  because both flattening nets share one choice. Splitting the choice per net needs an `InferConfig` / nets.h
  contract change.
- The QNN AAR's CPU EP has not been measured on arm64 (HUMAN GATE 3, step 6).
- `ADSP_LIBRARY_PATH` for the HTP skel has not been checked on a device.
- APK +41.3 MB (arm64) for the QNN libraries, and +32.8 MB for the static models. HUMAN GATE A-1 could drop the
  original UVDoc / SynthDoc-HR files.
- On the x86_64 emulator, the static models on the CPU EP gave grids identical to the desktop originals
  (`<SCR>/coordF/emu3/st_cpu_static`); this is x86_64 only, so it does not settle A-1 for arm64.
- CUDA on Blackwell: the first run compiles kernels (55 s, cached by the driver). TF32 is off in infer.cpp.
- `gpu.resample` was false on 2 of 5 photos in Stream F's `FS_GPU=1` runs (the pages are identical); not investigated.
- No in-app or adb hook yet runs `selftest-resample` / `gpu-stages` on a phone.
- Gate scripts hardcode this machine's paths (scratchpad references, venvs, ORT packages).

## Open HUMAN GATES (step by step)

Common setup: Git Bash, `export MSYS_NO_PATHCONV=1`, `PKG=com.albus.flatscan`, the Android SDK's `adb` on PATH.

**APK:** `C:/Users/User/Desktop/fs_accel_out/flatscan-accel-debug.apk` (accel/main, debug, arm64-v8a, QNN; built
with `JAVA_HOME=/c/Users/User/Android/jdk17 ./gradlew assembleDebug`).

1. Install it with `adb install -r C:/Users/User/Desktop/fs_accel_out/flatscan-accel-debug.apk`.
2. Run `adb shell svc power stayon true`.
3. Force-stop the app, let the phone cool down and plug it in before every timing run.

### HUMAN GATE licence (owner; no device)

1. Read `qnn-runtime-2.33.0-LICENSE.pdf`. It is inside the APK under `assets/licenses/`, and in
   `C:/Users/User/Android/ort/qnn-runtime-2.33.0.aar`.
2. It is the Qualcomm "AI Stack License": redistribution only in object code inside the app; no reverse
   engineering; keep the notices; export control; indemnification; revocable.
3. Decide whether to accept it. If not, build with `./gradlew assembleDebug -Port=plain`. That build has no
   Qualcomm libraries and no NPU; the CPU path is unchanged.

### HUMAN GATE 0: per-net CPU timings (any arm64 phone; ideally a Snapdragon 8 Gen 2/3 plus one other SoC)

1. Settings > *Self-test and benchmark* > *Run self-test*. Keep the screen on for a few minutes.
2. *Share report* and send back the whole text, including the SoC line, the 'Neural nets' table (median / p95 per
   net and each net's backend) and the final JSON block.
3. Optional: in Settings turn on UVDoc, both flattening nets and Neural shadows. Process 2-3 photos with
   automatic corners, then send the 'Nets' block from menu > *Processing details*, or the output of
   `adb logcat -s FlatScan | grep accel`.

### HUMAN GATE 3: QNN on a Snapdragon 8 Gen 2/3 (includes A-2 and Stream B's QNN checks)

1. **QNN check.**
   - Settings > Acceleration > *NPU (Qualcomm QNN)*.
   - Import one photo with automatic corners, then open menu > *Processing details*.
   - Every net except the page finder must report backend `qnn` (the page finder is CPU by design). Note the ms;
     the first use includes the QNN graph compile.
   - Process the same photo again and note the cached ms.
   - `adb shell run-as $PKG ls -l no_backup/qnn_cache` must list `*_ctx.onnx` files.
2. **Benchmark.**
   - Settings > *Run the acceleration benchmark* > *Share report* (first_ms = compile, median / p95, diff vs CPU,
     PASS / FAIL per net).
   - Force-stop the app, run the benchmark again (first_ms is now loaded from the cache), and send both reports.
3. **Push the inputs** (once):
   - 35 photos:
     `for f in C:/Users/User/Desktop/round4_robustness/real_photos/*.jpg; do n=$(basename "$f" .jpg); adb exec-in run-as $PKG sh -c "mkdir -p files/batch/s35 && cat > files/batch/s35/$n.jpg" < "$f"; done`
   - The receipt:
     `adb exec-in run-as $PKG sh -c 'cat > files/batch/s35/500c2810-image.jpg' < <SCR>/rc/photo.jpg`
   - DocUNet:
     `for k in $(seq 1 65); do for m in 1 2; do adb exec-in run-as $PKG sh -c "mkdir -p files/batch/docunet && cat > files/batch/docunet/${k}_${m}.png" < "C:/Users/User/Desktop/docunet_bench/data_crop/crop/${k}_${m} copy.png"; done; done`
   - Check: `adb shell run-as $PKG ls files/batch/s35 | wc -l` must say 35, and the same for `docunet` must say 130.
4. **Batch runs**, one at a time. Each is finished when `adb shell ls /sdcard/Android/data/$PKG/files/batch_out/<OUT>/done.txt`
   succeeds. Do not start the next one before that: `-S` kills a running batch, and re-running the same command resumes it.
   - (a) `adb shell am start -S -n $PKG/.MainActivity --es batch s35 --es accel cpu` (OUT `s35_cpu`)
   - (b) `... --es batch s35 --es accel npu` (OUT `s35_npu`)
   - (c) `... --es batch docunet --es accel npu` (OUT `docunet_npu`)
   - (d) `... --es batch docunet --es accel cpu` (OUT `docunet_cpu`)
   - (e) for A-1: `... --es batch docunet --es accel cpu --ez static_models true --es out du_cpu_static --es only 1_1,17_2,42_1`
   - Then `adb pull /sdcard/Android/data/$PKG/files/batch_out C:/Users/User/Desktop/phone_out`.
5. **Send back**: the two benchmark reports, the *Processing details* text from step 1, and the whole `phone_out` folder.
6. **Scoring on the desktop** (Coordinator):
   - CPU bit-identity:
     `$PY -I tools/desktop_check/gate_b.py cmp s35 <SCR>/accel_b/s35_ref phone_out/s35_cpu` and
     `$PY -I tools/desktop_check/gate_b.py cmp docunet <SCR>/accel_b/du_ref phone_out/docunet_cpu --images C:/Users/User/Desktop/docunet_bench/work/new_fold2`.
     Expect logits / p1 / rc0 / rc2 the same. `fit_median_px` may differ (open issue).
   - NPU quality, under the heavy lock:
     `$PY -I C:/Users/User/Desktop/docunet_bench/docunet_eval.py C:/Users/User/Desktop/docunet_bench/data_scan/scan phone_out/docunet_npu --jobs 4 --only <every k_m except 64_1,64_2>`.
     Targets: LD <= 5.40, MS-SSIM >= 0.58.
   - Decisions: `uv_model`, `needs_manual_crop` and `mode` in `s35_npu/*.json` vs `s35_cpu/*.json` must be equal on at least 33/35.
   - Per-net grid difference NPU vs CPU: `grid_px` from `tools/accel/compare_outputs.py` on each `NAME_{p1,rc0,rc2}.npz`
     (key `out`); report p99 and max. Target p99 < 0.25 px; this decides whether SynthDoc-HR can stay on the NPU.
   - Net time: the median over photos of the sum of the `nets` ms in `s35_npu/*.json`. Target <= 0.6 s.

### HUMAN GATE A-1 (optional; could save 31.4 MiB of APK)

After steps 4(d) and 4(e) above, run `$PY -I tools/desktop_check/gate_b.py cmp docunet phone_out/docunet_cpu phone_out/du_cpu_static --only 1_1,17_2,42_1`.
If p1, rc0 and rc2 are all the same, the arm64 CPU path can ship only the static models.

### HUMAN GATE D-1 (phone timing of the overlap)

Blocked until the app sets `flat_nets` (open issue). Then:
1. Set Acceleration = NPU and process 3 photos with automatic corners.
2. Copy from *Processing details* the stage times 'orientation, beside the nets', 'flattening nets, beside the CPU
   stages', 'waiting for the flattening nets' and 'total'.
3. Send them back with the phone model.

### Phone GPU-stage parity (Stream E; Snapdragon and Mali phones)

Blocked until a hook runs `fs_cli selftest-resample` / `gpu-stages` on the device. Until then:
1. Process the same 3-5 photos with GPU on (the default) and off.
2. Check that the pages are byte-identical.
3. Send the times 'stage-1 page', 'global_remap+darkness', 'final resample (GPU)' and 'paper mask + before page',
   plus the GL renderer string (it ends in '| precise' or '| no precise').

### Desktop EPs without hardware here

TensorRT (installable here, 2-3 GB), QNN on Snapdragon X, OpenVINO GPU, CoreML. Stream F's report has the steps:
1. Build: `tools/desktop_check/build_gpu.bat BUILD WGL|OFF <ORT 1.22.0 package> <EP>`.
2. Run `net_check.exe --backend <ep>` on `<SCR>/accel_a/in`, then `gate_f.py nets`, then `page_bench.exe --backend <ep> --reps 3`.
3. Send back the outputs.
