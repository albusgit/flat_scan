#include "npz.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs {

size_t NpyArray::count() const {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
}

size_t NpyArray::itemsize() const { return std::stoul(dtype.substr(2)); }

double NpyArray::at(size_t i) const {
    char kind = dtype[1];
    size_t sz = itemsize();
    const uint8_t* p = data.data() + i * sz;
    if (kind == 'f' && sz == 4) { float v; std::memcpy(&v, p, 4); return v; }
    if (kind == 'f' && sz == 8) { double v; std::memcpy(&v, p, 8); return v; }
    if (kind == 'i' && sz == 8) { int64_t v; std::memcpy(&v, p, 8); return (double)v; }
    if (kind == 'i' && sz == 4) { int32_t v; std::memcpy(&v, p, 4); return v; }
    if (kind == 'u' && sz == 1) return p[0];
    if (kind == 'b' && sz == 1) return p[0];
    throw std::runtime_error("unsupported dtype " + dtype);
}

std::vector<double> NpyArray::to_f64() const {
    std::vector<double> out(count());
    for (size_t i = 0; i < out.size(); ++i) out[i] = at(i);
    return out;
}

static NpyArray make(const char* descr, const std::vector<size_t>& shape, const void* p, size_t isz) {
    NpyArray a;
    a.dtype = descr;
    a.shape = shape;
    a.data.resize(a.count() * isz);
    if (p) std::memcpy(a.data.data(), p, a.data.size());
    return a;
}
NpyArray NpyArray::f32(const std::vector<size_t>& s, const float* p) { return make("<f4", s, p, 4); }
NpyArray NpyArray::f64(const std::vector<size_t>& s, const double* p) { return make("<f8", s, p, 8); }
NpyArray NpyArray::i64(const std::vector<size_t>& s, const int64_t* p) { return make("<i8", s, p, 8); }
NpyArray NpyArray::u8(const std::vector<size_t>& s, const uint8_t* p) { return make("|u1", s, p, 1); }
NpyArray NpyArray::scalar_i64(int64_t v) { return make("<i8", {}, &v, 8); }

// ---- .npy ----
static NpyArray parse_npy(const uint8_t* p, size_t n) {
    if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) throw std::runtime_error("not an npy file");
    int major = p[6];
    size_t hlen, off;
    if (major == 1) { hlen = p[8] | (p[9] << 8); off = 10; }
    else { hlen = p[8] | (p[9] << 8) | (p[10] << 16) | ((size_t)p[11] << 24); off = 12; }
    std::string h(reinterpret_cast<const char*>(p + off), hlen);
    NpyArray a;
    auto d = h.find("'descr'");
    auto q1 = h.find('\'', h.find(':', d) + 1);
    auto q2 = h.find('\'', q1 + 1);
    a.dtype = h.substr(q1 + 1, q2 - q1 - 1);
    if (h.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("fortran order npy");
    auto s0 = h.find('(', h.find("'shape'"));
    auto s1 = h.find(')', s0);
    std::stringstream ss(h.substr(s0 + 1, s1 - s0 - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_not_of(" ") == std::string::npos) continue;
        a.shape.push_back(std::stoul(tok));
    }
    size_t bytes = a.count() * a.itemsize();
    if (off + hlen + bytes > n) throw std::runtime_error("truncated npy");
    a.data.assign(p + off + hlen, p + off + hlen + bytes);
    return a;
}

static std::string npy_bytes(const NpyArray& a) {
    std::string shape = "(";
    for (size_t i = 0; i < a.shape.size(); ++i) shape += std::to_string(a.shape[i]) + (a.shape.size() == 1 ? "," : (i + 1 < a.shape.size() ? ", " : ""));
    shape += ")";
    std::string h = "{'descr': '" + a.dtype + "', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t total = 10 + h.size() + 1;
    h.append((64 - total % 64) % 64, ' ');
    h += '\n';
    std::string out("\x93NUMPY\x01\x00", 8);
    out += (char)(h.size() & 0xff);
    out += (char)(h.size() >> 8);
    out += h;
    out.append(reinterpret_cast<const char*>(a.data.data()), a.data.size());
    return out;
}

// ---- zip (stored entries only) ----
static uint32_t crc32(const std::string& s) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xffffffffu;
    for (unsigned char ch : s) c = table[(c ^ ch) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

static uint32_t rd32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const uint8_t* p) { return p[0] | (p[1] << 8); }

Npz read_npz(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() >= 6 && std::memcmp(buf.data(), "\x93NUMPY", 6) == 0) {
        Npz z;
        z["arr_0"] = parse_npy(buf.data(), buf.size());
        return z;
    }
    // find end of central directory
    size_t eocd = std::string::npos;
    for (size_t i = buf.size() >= 22 ? buf.size() - 22 : 0; i + 1 > 0; --i) {
        if (rd32(&buf[i]) == 0x06054b50) { eocd = i; break; }
        if (i == 0) break;
    }
    if (eocd == std::string::npos) throw std::runtime_error("not a zip: " + path);
    uint64_t n_entries = rd16(&buf[eocd + 10]);
    uint64_t cd_off = rd32(&buf[eocd + 16]);
    if (cd_off == 0xffffffffu || n_entries == 0xffff) {  // zip64
        size_t loc = eocd - 20;
        if (rd32(&buf[loc]) != 0x07064b50) throw std::runtime_error("bad zip64 locator");
        uint64_t z64 = 0;
        std::memcpy(&z64, &buf[loc + 8], 8);
        std::memcpy(&n_entries, &buf[z64 + 32], 8);
        std::memcpy(&cd_off, &buf[z64 + 48], 8);
    }
    Npz out;
    size_t p = cd_off;
    for (uint64_t e = 0; e < n_entries; ++e) {
        if (rd32(&buf[p]) != 0x02014b50) throw std::runtime_error("bad central directory");
        uint16_t method = rd16(&buf[p + 10]);
        uint64_t csize = rd32(&buf[p + 20]);
        uint16_t nlen = rd16(&buf[p + 28]), xlen = rd16(&buf[p + 30]), clen = rd16(&buf[p + 32]);
        uint64_t loff = rd32(&buf[p + 42]);
        std::string name(reinterpret_cast<const char*>(&buf[p + 46]), nlen);
        // zip64 extra field
        size_t xp = p + 46 + nlen;
        while (xp + 4 <= p + 46 + nlen + xlen) {
            uint16_t id = rd16(&buf[xp]), sz = rd16(&buf[xp + 2]);
            if (id == 1) {
                size_t q = xp + 4;
                if (rd32(&buf[p + 24]) == 0xffffffffu) q += 8;  // usize
                if (csize == 0xffffffffu) { std::memcpy(&csize, &buf[q], 8); q += 8; }
                if (loff == 0xffffffffu) std::memcpy(&loff, &buf[q], 8);
            }
            xp += 4 + sz;
        }
        if (method != 0) throw std::runtime_error("compressed npz entries are not supported: " + name);
        size_t data = loff + 30 + rd16(&buf[loff + 26]) + rd16(&buf[loff + 28]);
        std::string key = name.size() > 4 && name.substr(name.size() - 4) == ".npy" ? name.substr(0, name.size() - 4) : name;
        out[key] = parse_npy(&buf[data], csize);
        p += 46 + nlen + xlen + clen;
    }
    return out;
}

static void w16(std::string& s, uint16_t v) { s += (char)(v & 0xff); s += (char)(v >> 8); }
static void w32(std::string& s, uint32_t v) { for (int i = 0; i < 4; ++i) s += (char)((v >> (8 * i)) & 0xff); }

void write_npz(const std::string& path, const Npz& arrays) {
    std::string body, cd;
    uint16_t n = 0;
    for (const auto& kv : arrays) {
        std::string name = kv.first + ".npy";
        std::string data = npy_bytes(kv.second);
        if (body.size() + data.size() > 0xfffffff0u) throw std::runtime_error("npz too large for zip32");
        uint32_t crc = crc32(data), off = (uint32_t)body.size();
        std::string lh;
        w32(lh, 0x04034b50); w16(lh, 20); w16(lh, 0); w16(lh, 0); w16(lh, 0); w16(lh, 0x21);
        w32(lh, crc); w32(lh, (uint32_t)data.size()); w32(lh, (uint32_t)data.size());
        w16(lh, (uint16_t)name.size()); w16(lh, 0);
        body += lh + name + data;
        w32(cd, 0x02014b50); w16(cd, 20); w16(cd, 20); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0x21);
        w32(cd, crc); w32(cd, (uint32_t)data.size()); w32(cd, (uint32_t)data.size());
        w16(cd, (uint16_t)name.size()); w16(cd, 0); w16(cd, 0); w16(cd, 0); w16(cd, 0); w32(cd, 0); w32(cd, off);
        cd += name;
        ++n;
    }
    std::string end;
    w32(end, 0x06054b50); w16(end, 0); w16(end, 0); w16(end, n); w16(end, n);
    w32(end, (uint32_t)cd.size()); w32(end, (uint32_t)body.size()); w16(end, 0);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << body << cd << end;
}

}  // namespace fs
