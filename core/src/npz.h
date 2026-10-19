// Minimal .npy/.npz reader and writer (uncompressed zip, as written by numpy.savez).
// Host-side interchange only: used by the parity harness and the fixtures, never on the GPU.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fs {

struct NpyArray {
    std::string dtype;            // numpy descr, e.g. "<f4", "<f8", "<i8", "|u1"
    std::vector<size_t> shape;
    std::vector<uint8_t> data;    // C order, little endian

    size_t count() const;
    size_t itemsize() const;
    template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
    template <class T> T* as() { return reinterpret_cast<T*>(data.data()); }
    // Element i converted to double, whatever the stored dtype.
    double at(size_t i) const;
    std::vector<double> to_f64() const;

    static NpyArray f32(const std::vector<size_t>& shape, const float* p);
    static NpyArray f64(const std::vector<size_t>& shape, const double* p);
    static NpyArray i64(const std::vector<size_t>& shape, const int64_t* p);
    static NpyArray u8(const std::vector<size_t>& shape, const uint8_t* p);
    static NpyArray scalar_i64(int64_t v);
};

using Npz = std::map<std::string, NpyArray>;

Npz read_npz(const std::string& path);
void write_npz(const std::string& path, const Npz& arrays);

}  // namespace fs
