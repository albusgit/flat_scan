#include "tflite_shape.h"

#include <cstring>
#include <set>

namespace fs {

namespace {

// The few parts of the TFLite schema (tensorflow/lite/schema/schema.fbs) read here, as flatbuffer field ids
enum ModelField { kModelOperatorCodes = 1, kModelSubgraphs = 2, kModelBuffers = 4 };
enum OpCodeField { kOpCodeDeprecatedBuiltin = 0, kOpCodeBuiltin = 3 };
enum SubgraphField { kSubgraphTensors = 0, kSubgraphInputs = 1, kSubgraphOperators = 3 };
enum TensorField { kTensorShape = 0, kTensorBuffer = 2, kTensorShapeSignature = 7 };
enum OperatorField { kOperatorOpcode = 0, kOperatorInputs = 1 };
enum BufferField { kBufferData = 0 };
constexpr int32_t kResizeBilinear = 23, kResizeNearest = 97;

// A bounds-checked reader of the flatbuffer in `b` (any failed read sets ok = false and reads 0). Positions are
// byte offsets into the buffer.
struct Flat {
    uint8_t* b;
    size_t n;
    bool ok = true;

    template <class T> T get(size_t p) {
        T v{};
        if (p > n || n - p < sizeof(T)) {
            ok = false;
            return v;
        }
        std::memcpy(&v, b + p, sizeof(T));
        return v;
    }
    template <class T> void put(size_t p, T v) {
        if (p > n || n - p < sizeof(T)) {
            ok = false;
            return;
        }
        std::memcpy(b + p, &v, sizeof(T));
    }
    // position of table t's field `id`, 0 when absent (vtable at t - soffset: size, table size, field offsets)
    size_t field(size_t t, int id) {
        const int64_t vt = (int64_t)t - get<int32_t>(t);
        if (vt < 0) {
            ok = false;
            return 0;
        }
        const uint16_t vsize = get<uint16_t>((size_t)vt);
        const size_t at = 4 + 2 * (size_t)id;
        if (at + 2 > vsize) return 0;
        const uint16_t off = get<uint16_t>((size_t)vt + at);
        return off ? t + off : 0;
    }
    // the table / vector an offset field points to (0: field absent)
    size_t ref(size_t t, int id) {
        const size_t f = field(t, id);
        return f ? f + get<uint32_t>(f) : 0;
    }
    uint32_t len(size_t vec) { return vec ? get<uint32_t>(vec) : 0; }
    // element i of a vector of tables / of 4-byte scalars
    size_t table_at(size_t vec, uint32_t i) {
        const size_t e = vec + 4 + 4 * (size_t)i;
        return e + get<uint32_t>(e);
    }
    size_t scalar_at(size_t vec, uint32_t i) { return vec + 4 + 4 * (size_t)i; }
};

struct Model {
    Flat f;
    size_t codes = 0, tensors = 0, inputs = 0, operators = 0, buffers = 0;
    explicit Model(std::vector<uint8_t>& bytes) : f{bytes.data(), bytes.size()} {
        if (bytes.size() < 8) {
            f.ok = false;
            return;
        }
        const size_t root = f.get<uint32_t>(0);
        codes = f.ref(root, kModelOperatorCodes);
        buffers = f.ref(root, kModelBuffers);
        const size_t subgraphs = f.ref(root, kModelSubgraphs);
        if (f.len(subgraphs) < 1) {
            f.ok = false;
            return;
        }
        const size_t g = f.table_at(subgraphs, 0);
        tensors = f.ref(g, kSubgraphTensors);
        inputs = f.ref(g, kSubgraphInputs);
        operators = f.ref(g, kSubgraphOperators);
    }
    size_t tensor(int32_t i) { return i >= 0 && (uint32_t)i < f.len(tensors) ? f.table_at(tensors, (uint32_t)i) : (f.ok = false, 0); }
    // the data vector of tensor t's buffer, 0 when it has none (an activation)
    size_t data(size_t t) {
        const size_t fb = f.field(t, kTensorBuffer);
        const uint32_t bi = fb ? f.get<uint32_t>(fb) : 0;
        if (bi >= f.len(buffers)) return 0;
        const size_t d = f.ref(f.table_at(buffers, bi), kBufferData);
        return f.len(d) ? d : 0;
    }
    // builtin code of an operator code (the larger of the deprecated int8 field and the int32 one, as TFLite reads it)
    int32_t builtin(uint32_t code) {
        if (code >= f.len(codes)) {
            f.ok = false;
            return -1;
        }
        const size_t c = f.table_at(codes, code);
        const size_t d = f.field(c, kOpCodeDeprecatedBuiltin), b = f.field(c, kOpCodeBuiltin);
        const int32_t dep = d ? f.get<int8_t>(d) : 0, cur = b ? f.get<int32_t>(b) : 0;
        return dep > cur ? dep : cur;
    }
};

}  // namespace

bool tflite_input_hw(const std::vector<uint8_t>& bytes, int& H, int& W) {
    std::vector<uint8_t>& b = const_cast<std::vector<uint8_t>&>(bytes);   // read only
    Model m(b);
    if (!m.f.ok || m.f.len(m.inputs) < 1) return false;
    const size_t t = m.tensor(m.f.get<int32_t>(m.f.scalar_at(m.inputs, 0)));
    const size_t shape = m.f.ref(t, kTensorShape);
    if (!m.f.ok || m.f.len(shape) != 4) return false;
    H = m.f.get<int32_t>(m.f.scalar_at(shape, 1));
    W = m.f.get<int32_t>(m.f.scalar_at(shape, 2));
    return m.f.ok && H > 0 && W > 0;
}

bool tflite_set_input_hw(std::vector<uint8_t>& bytes, int H, int W, std::string* why) {
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return false;
    };
    int H0 = 0, W0 = 0;
    if (H <= 0 || W <= 0) return fail("bad size");
    if (!tflite_input_hw(bytes, H0, W0)) return fail("not a TFLite model with a 4-D input");
    if (H == H0 && W == W0) return true;
    // the edits, collected first (position, value) so that a failure leaves the bytes as they were
    std::vector<std::pair<size_t, int32_t>> edits;
    bool exact = true;
    // a resize's output size: the input's size divided by a power of two (the U-Nets' levels), or 1
    auto scale = [&](int32_t v, int to, int from) -> int32_t {
        if (v == 1) return 1;
        if (v <= 0 || ((int64_t)v * to) % from) {
            exact = false;
            return v;
        }
        return (int32_t)((int64_t)v * to / from);
    };
    Model m(bytes);
    if (!m.f.ok) return fail("unreadable model");
    // the input: [1, H0, W0, C] -> [1, H, W, C] (shape and signature); the activations' shapes in the file are left
    // as they are: AllocateTensors derives them again from the input (every op's Prepare), so a delegate must be
    // applied after an AllocateTensors (infer.cpp litert_open)
    for (uint32_t i = 0; i < m.f.len(m.inputs); ++i) {
        const size_t t = m.tensor(m.f.get<int32_t>(m.f.scalar_at(m.inputs, i)));
        for (int id : {kTensorShape, kTensorShapeSignature}) {
            const size_t s = m.f.ref(t, id);
            if (m.f.len(s) != 4) continue;
            const size_t ph = m.f.scalar_at(s, 1), pw = m.f.scalar_at(s, 2);
            if (id == kTensorShapeSignature && (m.f.get<int32_t>(ph) < 0 || m.f.get<int32_t>(pw) < 0)) continue;   // already dynamic
            edits.emplace_back(ph, H);
            edits.emplace_back(pw, W);
        }
    }
    // the resizes' constant output sizes (each buffer once: two resizes to one size may share it)
    std::set<size_t> seen;
    for (uint32_t i = 0; i < m.f.len(m.operators); ++i) {
        const size_t op = m.f.table_at(m.operators, i);
        const size_t oc = m.f.field(op, kOperatorOpcode);
        const int32_t code = m.builtin(oc ? m.f.get<uint32_t>(oc) : 0);
        if (code != kResizeBilinear && code != kResizeNearest) continue;
        const size_t in = m.f.ref(op, kOperatorInputs);
        if (m.f.len(in) < 2) return fail("resize without a size input");
        const size_t d = m.data(m.tensor(m.f.get<int32_t>(m.f.scalar_at(in, 1))));
        if (!d || m.f.len(d) != 8) return fail("resize whose size is not a constant int32 [2]");
        if (!seen.insert(d).second) continue;
        const int32_t h = m.f.get<int32_t>(d + 4), w = m.f.get<int32_t>(d + 8);
        edits.emplace_back(d + 4, scale(h, H, H0));
        edits.emplace_back(d + 8, scale(w, W, W0));
    }
    if (!m.f.ok) return fail("unreadable model");
    if (!exact) return fail(std::to_string(H0) + " x " + std::to_string(W0) + " does not scale to " + std::to_string(H) + " x " + std::to_string(W));
    for (const auto& e : edits) m.f.put(e.first, e.second);
    return true;
}

}  // namespace fs
