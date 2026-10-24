// One fixed-shape .tflite file for every input size (rulenet.tflite, shadenet.tflite): the app ships each of these
// nets once, made at one input size (tools/litert/convert.py), and the loader (infer.cpp) rewrites that file's
// sizes in place for the page at hand, for LiteRT's CPU kernels and its GPU delegate alike.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fs {

// Rewrites a fixed-shape TFLite model (bytes: the flatbuffer) for the NHWC input [1, H, W, C] instead of the one it
// was made at, [1, H0, W0, C]: the input's shape (and shape signature), and every RESIZE_BILINEAR /
// RESIZE_NEAREST_NEIGHBOR's constant output size [h, w], scaled to [h H / H0, w W / W0] (a size of 1 stays 1:
// global pooling); every scaled size must be a whole number (the U-Nets take multiples of 32). The activations'
// shapes stored in the file are not touched: AllocateTensors derives them from the input, so the model must be
// allocated before a delegate is applied to it (the GPU's: infer.cpp litert_open; XNNPACK in the interpreter
// options re-reads them at AllocateTensors). The graph is otherwise unchanged, so at its own size it is the file as
// made, and at another one the graph tools/litert/convert.py makes there (bit for bit: tools/litert/check_resize.py).
// No size constant other than the resizes' may be in the graph (convert.py any_hw relies on the same). false (why):
// not a model of this form, or a size that does not scale; bytes are then unchanged.
bool tflite_set_input_hw(std::vector<uint8_t>& bytes, int H, int W, std::string* why = nullptr);

// The NHWC input [H, W] the model was made at (subgraph 0, input 0); false: not readable.
bool tflite_input_hw(const std::vector<uint8_t>& bytes, int& H, int& W);

}  // namespace fs
