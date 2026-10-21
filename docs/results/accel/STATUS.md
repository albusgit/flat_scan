# Acceleration: gate status

Updated by Coordinator merge wave 1 (2026-10-21). Speed mode (owner's decision): streams and this merge ran
builds and short smoke checks only; the full gates run once at the end, by the final Coordinator, with the
commands below. "not run (full)" means the full gate is still owed.

`<SCR>` = `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-flat-scan/922e3e9d-4193-49a6-b4b4-d3421c7157d5/scratchpad`.

## Merge log (accel/main)

| Merge commit | What | Branch head |
|---|---|---|
| da8e67f | Stream B contract: `core/src/infer.h` + stub `infer.cpp` | 1a04235 |
| 7412c60 | accel/c: Stream C step 1, Phase 0 instrumentation | 92dd993 |
| 2b4ff45 | accel/a: static UVDoc / SynthDoc-HR, ShadeNet 512x352, Gate A logs | c879229 |
| 233fa5a | accel/b: inference on ORT C++ 1.22.0, nets in the core, pipeline_cli in-process | b0af623 |
| 3ba797d | accel/e: GPU image stages, Windows Mesa host harness, CG tolerance option | 1935dbb |

Order deviation from PLAN.md (B contract -> C1 -> A -> B -> C -> E -> D -> F): **E was merged now, before
C's integration (steps 2-5)**, because Stream F needs B + E and should not wait for C. All merges are real
merges and had no conflicts (no two wave-1 branches touch the same file). C's integration, D and F branch from
accel/main at or after this commit.

The same commit adds `.gitattributes` (`*.sh text eol=lf`, so the Git Bash gate scripts survive
`core.autocrlf=true`) and ignores `build-host-win/` (the host harness build folder).

## Post-merge smoke (coordinator-1, accel/main 3ba797d)

- `tools/desktop_check` (MSVC, ORT C++ 1.22.0 from `C:/Users/User/Android/ort/onnxruntime-win-x64-1.22.0`):
  builds (`cmd //c <SCR>\coord1\build_dc.bat`, log `<SCR>/coord1/dc_build.log`).
- `host/` harness (`build-host-win`, MSVC + mesa-dist-win): builds (`cmd //c <SCR>\coord1\build_host.bat`,
  log `<SCR>/coord1/host_build.log`).
- `./gradlew assembleDebug`: BUILD SUCCESSFUL (under the heavy lock, log `<SCR>/coord1/gradle.log`).
  APK 121,170,280 B, arm64-v8a only; it carries the three new models from Stream A. The only warnings are the
  existing ones (warpmap.cpp:1095 unused `y`, CameraScreen.kt LocalLifecycleOwner deprecation).
- pipeline_cli, CPU backend, DocUNet 1_1 and 17_2 with `FS_NO_SHADOWS=1`: with the cached nets (new_fold2
  config) and with the nets in-process (`FS_NETS=1 pipeline_cli --backend cpu PHOTO OUT.png net`), all 4 pages
  are pixel-identical to `docunet_bench/work/new_fold2`.
- `fs_cli selftest-resample` on llvmpipe: 30 cases worst |diff| 0; GPU dense / offset stages worst |diff| 0.
- Script and output: `bash <SCR>/coord1/smoke.sh <SCR>/coord1/smoke` (under the heavy lock),
  `<SCR>/coord1/smoke.log`, `<SCR>/coord1/smoke/`.

## Gates

| Gate | Stream | Status | Command (Git Bash unless noted) | Output |
|---|---|---|---|---|
| A | Models | partial: NNAPI 1 partition / 100 % for all three static models (pass); static fp32 == originals on ORT 1.22.0 CPU, 105/105 runs (pass); fp16 stand-in grid p99 UVDoc 0.083 / 0.078 px (meets < 0.25), **SynthDoc-HR 0.954 px, max 2.64 px (misses)** | see A below | `docs/results/accel/compare_outputs.txt`, `usability_*.txt`, `models.txt` |
| B | Core inference | partial: 35/35 photos bit-identical, round4_check identical, DocUNet 34/130 identical (stream runs); full DocUNet comparison not run | see B below | `<OUT>/cmp_s35.txt`, `round4_{ref,new}.txt`, `cmp_docunet.txt`; times `docs/results/accel/desktop_cpu.txt` |
| C | Android (agent-checkable) | step 1 only: assembleDebug passes; emulator check not run (needs C steps 2-5 and an x86_64 build) | `JAVA_HOME=/c/Users/User/Android/jdk17 ./gradlew assembleDebug` (heavy lock) | `app/build/outputs/apk/debug/app-debug.apk` |
| D | CPU/NPU overlap | open (wave 2) | | |
| E | GPU image stages | not run (full); smoke PASS (3 DocUNet + 2 round-4 photos, llvmpipe and d3d12) | see E below | `docs/results/accel/gpu_stages.txt`, `cg_rtol.txt`; full: `<OUT>/*` |
| F | Desktop GPU | open (wave 2) | | |
| HUMAN GATE 0 | Phase-0 timings on devices | open, APK builds | see below | report text from the phone |
| HUMAN GATE 3 | Snapdragon 8 Gen 2/3 | open (needs C integration) | see below | |

### A (cheap, ~81 s; no heavy lock)

From the tree being gated, `PY=C:/Users/User/Android/accel_venv/Scripts/python.exe`, `S=<SCR>/accel_a`:

    $PY -I tools/accel/prepare_models.py                                   # must reproduce the *_static.onnx bytes
    C:/Users/User/Desktop/round4_robustness/.venv/Scripts/python.exe -I tools/shadow_net/export.py --static \
        C:/Users/User/Desktop/shadow_train/final_11k.pt app/src/main/assets/models 352
    $PY -I tools/accel/net_inputs.py tools/desktop_check/build $S/in        # only if $S/in is gone (several min)
    $PY -I tools/accel/compare_outputs.py $S/in --log docs/results/accel/compare_outputs.txt --dump $S/fp16
    for f in uvdoc_static syntheticdoc_hr_static shadenet_512x352; do $PY -I -m onnxruntime.tools.check_onnx_model_mobile_usability \
        --log_level debug app/src/main/assets/models/$f.onnx > docs/results/accel/usability_$f.txt; done

Pass: compare_outputs exit 0 (fp32 bit-identical); the usability logs say 1 partition, 100 % for NNAPI;
`git status` clean afterwards (the models reproduce byte for byte).

### B (about 20 min, estimate; the script takes the heavy lock itself for DocUNet)

1. Build desktop_check in the tree being gated (NOTES.md recipe; `<SCR>/coord1/build_dc.bat` does it for
   `flat_scan_accel_main`).
2. **BASE_REF must be `cf1b05d`** (accel/main before Stream B). The script's default, `accel/main`, now already
   contains B and would compare B with itself:

       bash tools/desktop_check/gate_b_full.sh <SCR>/gateB_final cf1b05d

Pass: `cmp_s35.txt` says `photos with any difference: 0 of 35`; the script prints `round4_check: identical`;
`cmp_docunet.txt` says 0 of 130 and vs_images 130 same. Any vs_images difference must be explained and scored with
`docunet_bench/docunet_eval.py` (round4 venv, `-I`, `--jobs 4`, under the lock).

### E (about 30 min, estimate; the script takes the heavy lock itself)

1. Build desktop_check (as B) and the host harness: `cmd //c <SCR>\coord1\build_host.bat`
   (inside vcvars64: `cmake -S host -B build-host-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=C:/Users/User/Android/opencv/build/x64/vc16/lib`
   then `cmake --build build-host-win -j 6`; MESA_DIR defaults to `C:/Users/User/Android/mesa`).
2. `bash host/gpu_stage_gate.sh C:/Users/User/Desktop/docunet_bench/work/e_gate full`

Pass: exit 0 and `gate E: PASS`. Outputs in that folder: `cpu_vs_new_fold2.txt` (all DocUNet photos),
`selftest_llvmpipe.txt`, `gpu_stages_docunet_llvmpipe.{txt,json}`, `gpu_stages_round4_llvmpipe.{txt,json}`,
`gpu_stages_docunet_d3d12*.txt`, `gpu_bench_*.txt`, `cg_rtol_1e-6_docunet.txt`.

### C (after C integration)

`./gradlew assembleDebug` (heavy lock); then the emulator run in PLAN.md Gate C. That needs an x86_64 build
(abiFilters is arm64-v8a only today) and the CPU ORT AAR for x86_64 (the QNN AAR is arm64 only).

## Open HUMAN GATES

### HUMAN GATE 0: Phase-0 timings (any Android phone; ideally Snapdragon 8 Gen 2/3 plus one or more other SoCs)

1. Build: on accel/main, `JAVA_HOME=/c/Users/User/Android/jdk17 ./gradlew assembleDebug`; the APK is
   `app/build/outputs/apk/debug/app-debug.apk` (arm64 only). `adb install -r app-debug.apk`.
2. Force-stop the app, open it, let the phone cool down, plug it in. Note the phone model.
3. Settings > 'Self-test and benchmark' > 'Run self-test'. Keep the screen on; wait a few minutes for the report.
4. 'Share report' and send back the whole text (SoC line, thermal status, core checks, the 'Neural nets on CPU'
   table with median / p95 ms per net, page totals, peak memory, and the final 'JSON (send this back ...)' block).
5. Optional real-photo timings: in Settings turn on UVDoc, both flattening nets and Neural shadows; scan or import
   2-3 photos with automatic corners; on each result, menu > 'Processing details', copy the
   'Nets on CPU (ONNX Runtime ...)' block, or run `adb logcat -s FlatScan | grep accel` and send the `accel {...}` lines.

### HUMAN GATE A-1 (optional; decides whether the original UVDoc / SynthDoc-HR files can be dropped, -31.4 MiB)

Once C's batch mode / self-test can pick a model file: on an arm64 phone with Acceleration=CPU, run the self-test
photo once with `uvdoc.onnx` + `syntheticdoc_hr.onnx` and once with the `*_static.onnx` files; send both grids'
SHA-256 (or the batch-mode JSON). Equal => ship only the static models.

### HUMAN GATE 3 (Snapdragon 8 Gen 2/3; needs C integration, steps 2-5)

Includes A-2 and Stream B's QNN checks. Debug APK from accel/main after C integration:
1. Acceleration = NPU (QNN): process a photo; processing details must show backend `qnn` for every net. Process
   it again; report first-run (QNN context compile) and cached-run ms per net, and that
   `noBackupFilesDir/qnn_cache` holds compiled models.
2. Batch mode: `adb push` the 35 photos and the DocUNet crops to the folder C's batch mode names; run with NPU and
   with CPU; pull the JSON and pages. The Coordinator then runs `gate_b.py cmp s35 <Gate B reference> <phone CPU dir>`
   (expected 35/35 identical, i.e. the QNN AAR's CPU EP == onnxruntime-android) and `compare_outputs.py`'s
   px-of-page measure NPU vs CPU per net (decides whether SynthDoc-HR stays on the NPU; fp16 stand-in p99 0.95 px).
3. PLAN.md targets: nets <= 0.6 s; DocUNet LD <= 5.40 and MS-SSIM >= 0.58 with NPU; net winner and quality-gate
   decision unchanged on >= 33/35 photos.

### Phone GPU-stage parity (Stream E; needs an in-app or adb hook first)

On a Snapdragon and a Mali phone: process the same 3-5 photos with GPU on (default) and off; compare the page
images byte for byte and send the times for 'stage-1 page', 'global_remap+darkness', 'final resample (GPU)',
'paper mask + before page', the device model and the GL renderer string (ends in '| precise' or '| no precise').
There is no hook yet that runs `fs_cli selftest-resample` / `gpu-stages` on a device; C or D must add one first.

### Desktop GPU EPs (Stream F)

CUDA / DirectML need the matching ORT 1.22.0 GPU packages in `C:/Users/User/Android/ort/`; rebuild desktop_check
with `-DFS_ORT_ROOT=<package> -DFS_ORT_DML=ON` (or `FS_ORT_CUDA`), then
`gate_b.py new s35 OUT --cli build/pipeline_cli.exe --backend dml` and report per-net ms from the JSON.
TensorRT / QNN (Snapdragon X) / OpenVINO-GPU / CoreML: no hardware on this machine (HUMAN GATE).
