// core/src/tflite_shape.cpp on the desktop: writes a fixed-shape .tflite rewritten for another input size, as the
// app's loader makes it in memory (tools/litert/check_resize.py runs the result against the converter's files).
//   tflite_shape_cli IN.tflite OUT.tflite H W     exit code 1 when the model does not rewrite (the reason on stderr)
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "tflite_shape.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: tflite_shape_cli IN.tflite OUT.tflite H W\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int h0 = 0, w0 = 0;
    fs::tflite_input_hw(bytes, h0, w0);
    std::string why;
    if (!fs::tflite_set_input_hw(bytes, std::atoi(argv[3]), std::atoi(argv[4]), &why)) {
        std::fprintf(stderr, "%s (%d x %d): %s\n", argv[1], h0, w0, why.c_str());
        return 1;
    }
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
    std::printf("%s: %d x %d -> %s x %s\n", argv[2], h0, w0, argv[3], argv[4]);
    return out ? 0 : 2;
}
