// Stub inference (contract first): no ONNX Runtime linked yet, so every run fails and the caller
// takes its no-net path. The ORT implementation replaces this file's body (FS_WITH_ORT).
#include "infer.h"

namespace fs {

struct Inference::Impl {
    InferConfig cfg;
};

Inference::Inference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Inference::~Inference() = default;

std::unique_ptr<Inference> Inference::create(const InferConfig& cfg) {
    auto impl = std::make_unique<Impl>();
    impl->cfg = cfg;
    return std::unique_ptr<Inference>(new Inference(std::move(impl)));
}

bool Inference::run(NetId, const Tensor&, std::vector<Tensor>& out, std::string* err) {
    out.clear();
    if (err) *err = "no inference backend in this build";
    return false;
}

Backend Inference::backend(NetId) const { return Backend::CPU; }
double Inference::last_ms(NetId) const { return 0.0; }

}  // namespace fs
