# FlatScan: GPU / NPU acceleration — multi-agent build plan

(The plan as given by the project owner, 2026-10-21. Machine-specific facts and resolved inputs are in
`docs/accel/NOTES.md`; where they differ from this plan, NOTES.md wins.)

You are a team of coding agents working in the `flat_scan` repository (Android app: Kotlin + C++/NDK core in `core/src`, shared with a desktop harness in `host/` and `tools/desktop_check/`). Goal: move the neural nets off the CPU onto the Qualcomm NPU (ONNX Runtime QNN EP), run CPU geometry concurrently with the nets, move remaining image stages to the GPU, and give the desktop build GPU inference. Read `README.md` (Architecture, Round 4–8 sections) before starting.

## Context (measured; do not re-derive)

Per automatic page the app runs these ONNX models (`app/src/main/assets/models/`), today on ONNX Runtime Android 1.22.0, CPU, fp32, from Kotlin (`FlatNet.kt`, `PageFinder.kt`, `ShadeNet.kt`, orchestrated in `ScanViewModel.kt` ~lines 895–945):

| Model | Input | Compute | Host CPU (4-core Xeon) | Runs/page |
|---|---|---|---|---|
| syntheticdoc_hr.onnx | 1×3×1440×1024 | ≥75 GMAC | 700–790 ms | 1 |
| uvdoc.onnx | 1×3×712×488 | 18 GMAC | 107 ms | 2 (whole photo, then crop) |
| shadenet.onnx | 1×3×512×W (W = multiple of 32, dynamic) | ~2–3 GMAC | ~66 ms | 1 |
| pagefinder.onnx | 1×3×256×256 | 0.7 GMAC | 20 ms | 1+ |

Non-net pipeline on host ≈ 1.6 s (`docs/results/acceptance_host.txt`): warp map 700 ms (solves 300, global 170, tracing 140, remaps 90), detect 240, surface fit 100–250, paper mask + before page 200, orientation 120. Final resample + cleanup already run on GPU (`core/src/gpu.cpp`, GLES 3.1 compute, CPU fallback). Nothing has been measured on a phone; estimated flagship page 6–9 s, of which nets 3–6 s.

`python -m onnxruntime.tools.check_onnx_model_mobile_usability` results:
- uvdoc: 100 % NNAPI coverage, 1 partition — ready.
- syntheticdoc_hr: 96 %, 5 partitions — 5 nodes with shape-derived Resize sizes. Its `shading` output is unused by the app.
- shadenet: 35 % — dynamic H/W input; Shape/Expand/Gather from the global-mean branch. Exported with `dynamic_axes` in `tools/shadow_net/export.py`.
- pagefinder: 84 %, 34 partitions — HardSigmoid unsupported + shape-derived Resize. Decision: keep on CPU.

Models store fp16 weights with Cast→fp32 compute. The desktop harness runs nets by spawning Python (`tools/shadow_net/run_onnx.py`) per call and uses `tools/desktop_check/gpu_stub.cpp` (no GPU).

## Hard rules (all agents)

1. The CPU path must stay bit-identical to today's outputs. Any accelerated path is opt-in per backend with automatic CPU fallback on any error (existing pattern: `failed = true` → return null).
2. Never skip, disable or loosen an existing test or parity check to get green. New tolerances apply only to accelerated backends, and must be stated in code and README.
3. Core C++ stays C++17, built with `-ffp-contract=off` (MSVC `/fp:precise`). Don't change numerics in existing CPU code paths.
4. Match surrounding code style and comment density. Keep README sections updated for what you change (Architecture table, a new "Acceleration" section with measured numbers only; mark estimates as estimates).
5. No phone is available to agents. Anything needing a device is a **HUMAN GATE**: build it, make it runnable from the in-app self-test / a batch mode, write exact instructions, and stop there. Never report device numbers you did not measure.
6. Work only in the files your workstream owns (below). Changes to another stream's files go through the Coordinator.
7. Commit small, descriptive commits on your stream branch; the Coordinator merges.

## External inputs

See `docs/accel/NOTES.md` (all resolved except the QNN SDK, which is not available).

## Branches

Integration branch: `accel/main` off the current base branch. Each stream works on `accel/<stream>` off `accel/main` and rebases on it before handing off. The Coordinator merges in dependency order.

## Shared contract (Stream B lands this FIRST, before anything else builds on it)

`core/src/infer.h`:

    namespace fs {
    enum class NetId { PageFinder, UVDoc, SynthDocHR, ShadeNet };
    enum class Backend { CPU, XNNPACK, QNN_HTP, NNAPI, CUDA, TensorRT, DirectML, OpenVINO, CoreML };
    struct InferConfig {
        std::function<std::vector<uint8_t>(const std::string& file)> load;  // model bytes (Android: AAssetManager)
        std::vector<Backend> preference;   // tried in order per net; CPU is always appended last
        std::string cache_dir;             // compiled-model cache (QNN context binaries)
        bool fp16 = true;                  // accelerators only; CPU stays fp32
        int threads = 0;                   // 0: ORT default
    };
    struct Tensor { std::vector<int64_t> shape; std::vector<float> data; };
    class Inference {
    public:
        static std::unique_ptr<Inference> create(const InferConfig& cfg);
        // false on any failure (caller falls back exactly as today); thread-safe per NetId
        bool run(NetId net, const Tensor& in, std::vector<Tensor>& out, std::string* err = nullptr);
        Backend backend(NetId net) const;  // the backend that actually ran last
        double last_ms(NetId net) const;
    };
    }

Changes to this contract need Coordinator sign-off and a note to all streams.

## Workstreams

### Stream A — Models (owns `tools/accel/**`, `tools/shadow_net/export.py`, new files in `app/src/main/assets/models/`)

Depends on: nothing. Start immediately.
1. `tools/accel/prepare_models.py`: for uvdoc and syntheticdoc_hr, produce `*_static.onnx`: all shapes static, constants folded (onnxsim or equivalent), syntheticdoc_hr's unused `shading` output removed. Keep originals untouched.
2. ShadeNet: add an export mode in `export.py` without `dynamic_axes` that writes fixed-shape models for H=512, W ∈ {288, 320, …, 512} (the widths `apply_shade_net` in `core/src/pipeline.cpp:253` can produce). Name `shadenet_512x<W>.onnx`.
3. `tools/accel/compare_outputs.py`: runs original vs static (CPU EP, fp32) and static with fp16 compute (ORT CPU with fp16-converted model, as an fp16 proxy) on the 35 photos' real net inputs; reports max/p99 grid difference in px of page and per-photo.
4. Re-run the mobile usability checker on the outputs; save logs to `docs/results/accel/`.
Gate A: uvdoc_static and syntheticdoc_hr_static are 1 partition / 100 % for NNAPI; static fp32 outputs bit-identical to originals on CPU; fp16 proxy grid p99 diff reported (target < 0.25 px). Commit the logs.

### Stream B — Core inference (owns `core/src/infer.*`, new `core/src/nets.*`, `core/CMakeLists.txt`, `tools/desktop_check/**`, `host/**`)

Depends on: nothing for the contract; Stream A outputs for the static models.
1. Day 1: land `core/src/infer.h` exactly as the contract, with a stub `.cpp` that compiles.
2. Implement `infer.cpp` on the ONNX Runtime C++ API (same ORT version as the app). One `Ort::Session` per NetId, created lazily; per-backend session options: QNN (`backend_path=libQnnHtp.so`, `enable_htp_fp16_precision=1`, `htp_performance_mode=burst`, EP-context caching into `cache_dir` keyed by model SHA-256 + SoC + QNN version), NNAPI (`USE_FP16`, `CPU_DISABLED`), XNNPACK, and on desktop CUDA / TensorRT / DirectML / OpenVINO / CoreML when compiled in (CMake options, all default OFF). Any EP creation or run failure → retry on next backend → CPU.
3. `core/src/nets.{h,cpp}`: move the net orchestration from `ScanViewModel.kt` (whole-photo UVDoc pass on automatic corners, `uv_recrop` crops, UVDoc + SynthDoc-HR runs, which grid fills which slot) and ShadeNet (`apply_shade_net` calls through `Inference` instead of the JNI callback) into C++, using the existing `uv_crop` / `uv_recrop` / `uv_input` / `seg_input` code. Output must feed `PipelineOptions` exactly as the Kotlin path does now.
4. Desktop: `tools/desktop_check/pipeline_cli` uses `Inference` in-process (remove the Python subprocess path for ShadeNet and the zoom pass, keep env flags). Add `--backend` / `FS_BACKEND`.
Gate B: on CPU backend, pipeline_cli grids and outputs bit-identical to the current Python-ORT path on the 35 photos (`tools/round4_check.py` unchanged results), DocUNet baseline unchanged. Record per-net and total times in `docs/results/accel/desktop_cpu.txt`.

### Stream C — Android integration (owns `app/**` except model assets)

Depends on: B step 1 for the contract; B steps 2–3 to finish.
1. Phase 0 instrumentation (start now, against today's Kotlin path): per-net timings in the processing-details JSON and in `selfTest` (`jni_bridge.cpp` ~771) report: page finder, UVDoc pass 1, UVDoc, SynthDoc-HR, ShadeNet, plus SoC (`Build.SOC_MODEL`), backend used. Ship this first as its own commit — the human runs it on devices (HUMAN GATE 0).
2. Link ORT C++ from the AAR into `libflatscan` (`app/src/main/cpp/CMakeLists.txt`; prefab if the AAR provides it, else a Gradle task that extracts headers + `libonnxruntime.so`). Switch dependency to `com.microsoft.onnxruntime:onnxruntime-android-qnn` at the version Stream B targets; confirm the QNN libs' redistribution licence and report APK size delta (per ABI).
3. JNI: `InferConfig.load` reads assets through `AAssetManager`; cache_dir = `context.noBackupFilesDir/qnn_cache`. Replace the Kotlin net calls with the native `run_nets`; delete or thin `FlatNet.kt`, `PageFinder.kt`, `ShadeNet.kt`, `ShadeNetRunner`.
4. Settings › Acceleration: Auto / CPU / NPU (QNN) / NNAPI (experimental). Auto: on first use per (SoC, app version), benchmark the bundled self-test photo on CPU and each available accelerator, compare grids to CPU (tolerance from Gate A), pick the fastest passing backend, persist the choice; show it in processing details. Any runtime failure flips that net back to CPU for the session.
5. Hidden batch mode (debug builds): process every image in a folder (adb-pushed) and write the same JSON that `pipeline_cli` writes, so DocUNet and the 35 photos can be scored from a phone.
Gate C (agent-checkable): `./gradlew assembleDebug` passes; emulator x86_64 run with Acceleration=CPU gives grids bit-identical to Gate B; Auto on the emulator falls back to CPU without crashing.
HUMAN GATE 3 (Snapdragon 8 Gen 2/3 phone): nets ≤ 0.6 s; batch-mode DocUNet LD ≤ 5.40 and MS-SSIM ≥ 0.58 with NPU; net winner and quality-gate decision unchanged on ≥ 33/35 photos; first-run compile time and cached-run time reported.

### Stream D — CPU/NPU overlap (owns `core/src/pipeline.*` stage split; coordinate with E for shared files)

Depends on: B merged.
1. Audit which `process()` stages depend on the net grids. Known: with automatic corners the grid can define the page (`uv_defines_page`, Round 8), replacing finder corners.
2. Split `process()` so that after the page finder, the nets run on one thread while detect → surface fit → stage-1 page → orientation run on the CPU. If `uv_defines_page` fires, redo the dependent stages (speculative execution). Count how often that happens on the 35 photos and DocUNet; report it.
Gate D: outputs bit-identical to sequential execution on CPU backend (run both orders on the 35 photos); wall time ≈ max(CPU branch, net branch) on desktop with a GPU EP or with nets artificially delayed.

### Stream E — GPU image stages (owns `core/src/gpu.*`, `core/src/resample.*`, GPU tests in `host/`)

Depends on: nothing (parallel with A–C).
1. Reuse the bicubic resample compute path for the stage-1 page, global remap + darkness, orientation page and before page (CPU fallback kept; selection at runtime as today).
2. Extend `fs_cli selftest-resample` (and add equivalents) so every new GPU stage is compared to its CPU counterpart on the existing fixtures.
3. Separately (CPU, behind an option default OFF): measure the effect of a looser sparse-solve tolerance than rtol 1e-8 on warp-map time and on the 35-photo / DocUNet metrics. Report only; do not change the default.
Gate E: GPU == CPU within the existing resample/cleanup parity standard (identical, or ≤ 1 level where the README already allows it) on Mesa llvmpipe; timing table in `docs/results/accel/gpu_stages.txt`.

### Stream F — Desktop GPU (owns `tools/desktop_check/gpu_gl43.cpp` or ANGLE integration, desktop CMake options)

Depends on: B (inference), E (shader set).
1. Replace `gpu_stub.cpp` on desktop with a desktop OpenGL 4.3 core build of `gpu.cpp` (same compute shaders; `#version 430` instead of `310 es`, context via GLFW/EGL hidden window) or link ANGLE for EGL + GLES 3.1. Keep `gpu_stub.cpp` as the no-GPU option.
2. Wire CUDA/TensorRT (NVIDIA), DirectML/Windows ML (Windows), QNN (Snapdragon X), OpenVINO (Intel) EPs from Stream B into the desktop build options.
Gate F: Gate B results unchanged on CPU backend; GPU-stage outputs match CPU as in Gate E; per-page timing table for each backend available on the machine used (HUMAN GATE where hardware is missing).

### Stream G (later, only if the human asks after HUMAN GATE 3)
NNAPI as a first-class Auto candidate on MediaTek/Exynos/Pixel; LiteRT GPU delegate path; int8 (QAT) models for mid-range Hexagon. Do not start without explicit go-ahead.

## Coordinator (owns `README.md`, `accel/main`, merge order, contract)

1. Create `accel/main`; spawn A, B (contract first), C (step 1 first), E in parallel. D after B merges; F after B + E.
2. Merge order: B contract → C step 1 → A → B → C → E → D → F. Rebase-check each merge: build host harness, desktop_check, `./gradlew assembleDebug`, run Gate B parity before every merge into `accel/main`.
3. Keep `docs/results/accel/STATUS.md`: each gate, pass/fail, the exact command and output file, and open HUMAN GATES with step-by-step instructions (which build, which setting, which folder to push, which numbers to send back).
4. Final deliverable: README "Acceleration" section (backends, how Auto chooses, tolerances, measured numbers only), and a summary of what still needs device measurement.

## Expected outcome (estimates, to be replaced by measurements)

Flagship Snapdragon page 6–9 s → 2.5–3.5 s after QNN, → 1.5–2.5 s with overlap + GPU stages. Desktop page ~2.7 s → ~1.2–1.5 s with an NVIDIA GPU. Nets alone: ~8–15× on Hexagon fp16, ~10–20× on an NVIDIA GPU.
