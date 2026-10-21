// Neural-net inference for the page pipeline (docs/accel/PLAN.md, shared contract): one ONNX
// Runtime session per net, created lazily, on the first backend in `preference` that accepts the
// model; CPU (ORT's default EP, fp32) is always the last resort. Any EP creation or run failure
// moves that net to the next backend; run() returns false only when the CPU run fails too, and
// the caller then falls back exactly as without the net. Builds without ONNX Runtime (FS_WITH_ORT
// undefined) compile a stub whose run() always returns false.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fs {
enum class NetId { PageFinder, UVDoc, SynthDocHR, ShadeNet };
enum class Backend { CPU, XNNPACK, QNN_HTP, NNAPI, CUDA, TensorRT, DirectML, OpenVINO, CoreML, LiteRtGpu };
struct InferConfig {
    std::function<std::vector<uint8_t>(const std::string& file)> load;  // model bytes (Android: AAssetManager)
    std::vector<Backend> preference;   // tried in order per net; CPU is always appended last
    std::string cache_dir;             // compiled-model cache (QNN context binaries)
    bool fp16 = true;                  // accelerators only; CPU stays fp32
    int threads = 0;                   // 0: ORT default
    std::string qnn_dir;               // folder of libQnnHtp.so (Android: the qnn_htp asset pack); empty: the library path
    std::string litert_dir;          // folder of libtensorflowlite_jni.so + libtensorflowlite_gpu_jni.so (Android: the litert_gpu pack); empty: no LiteRT
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
}
