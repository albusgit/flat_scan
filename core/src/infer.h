// Neural-net inference for the page pipeline (docs/accel/PLAN.md, shared contract): one session per
// net, created lazily, on the first backend in `preference` that accepts the model; CPU (fp32) is
// always the last resort. Any backend creation or run failure moves that net to the next backend;
// run() returns false only when the CPU run fails too, and the caller then falls back exactly as
// without the net. The CPU is the build's CPU runtime: ONNX Runtime's default EP on the .onnx models
// with FS_WITH_ORT (the desktop builds), else LiteRT's CPU kernels on the .tflite models (the Android
// app). Builds with neither compile a stub whose run() always returns false. LiteRtGpu is not an ORT
// provider: the .tflite models on LiteRT's GPU delegate in fp32. LiteRT's libraries are loaded at run
// time from litert_dir (empty: by name from the library path, the APK's lib/ on Android).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fs {
enum class NetId { PageFinder, UVDoc, SynthDocHR, ShadeNet, RuleNet, Inpaint };
enum class Backend { CPU, XNNPACK, QNN_HTP, NNAPI, CUDA, TensorRT, DirectML, OpenVINO, CoreML, LiteRtGpu };
// A model file in place (Android: an uncompressed asset of the APK, AAsset_getBuffer), valid while keep lives.
struct ModelView {
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::shared_ptr<const void> keep;
};
struct InferConfig {
    std::function<std::vector<uint8_t>(const std::string& file)> load;  // model bytes (Android: AAssetManager)
    // optional: the LiteRT backends read the model files in place instead of a copy on the heap (not ShadeNet's
    // and the rule net's, which are rewritten per input shape); no data: load instead
    std::function<ModelView(const std::string& file)> map;
    std::vector<Backend> preference;   // tried in order per net; CPU is always appended last
    std::string cache_dir;             // compiled-model cache (QNN context binaries, LiteRT GPU kernels)
    bool fp16 = true;                  // accelerators only; CPU stays fp32
    int threads = 0;                   // 0: ORT default; LiteRT CPU: min(cores, 4)
    std::string qnn_dir;               // folder of libQnnHtp.so (Android: the qnn_htp asset pack); empty: the library path
    std::string litert_dir;            // folder of libtensorflowlite_jni.so + libtensorflowlite_gpu_jni.so; empty: by name (Android: the APK's lib/)
};
struct Tensor { std::vector<int64_t> shape; std::vector<float> data; };
class Inference {
public:
    static std::unique_ptr<Inference> create(const InferConfig& cfg);
    // false on any failure (caller falls back exactly as today); thread-safe per NetId
    bool run(NetId net, const Tensor& in, std::vector<Tensor>& out, std::string* err = nullptr);
    Backend backend(NetId net) const;  // the backend that actually ran last
    double last_ms(NetId net) const;

    ~Inference();
private:
    struct Impl;
    explicit Inference(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
// The build's runtimes: cpu ("ONNX Runtime 1.22.0" or "LiteRT <version>"), litert (LiteRT's version when
// compiled in), and whether the LiteRT GPU delegate can be tried here (its library loads and the device
// has an OpenCL library; gpu_why otherwise). Loads LiteRT's libraries from litert_dir.
struct RuntimeInfo { std::string cpu, litert; bool gpu = false; std::string gpu_why; };
RuntimeInfo runtime_info(const std::string& litert_dir = "");
}
