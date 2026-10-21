// One backend's net outputs on fixed net inputs (Gate F: accelerated nets vs CPU on the same inputs):
//   net_check [--backend B] OUT_DIR IN_DIR NAME...
// IN_DIR: tools/accel/net_inputs.py's cache (NAME_p1_in / _rc0_in / _rc2_in / _shade_in.npz, key "x").
// Writes OUT_DIR/NAME_<tag>.npz (key "out": the net's first output, shape as returned) and prints the
// backend that ran each and its run time. p1, rc0: UVDoc; rc2: SyntheticDoc-HR; shade: ShadeNet.
// tools/desktop_check/gate_f.py nets compares two such folders (or one against the CPU reference).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "infer.h"
#include "nets.h"
#include "npz.h"

using namespace fs;

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static bool exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

int main(int argc, char** argv) {
    std::string backend = "cpu";
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else pos.push_back(a);
    }
    if (pos.size() < 3) { std::fprintf(stderr, "usage: net_check [--backend B] OUT_DIR IN_DIR NAME...\n"); return 2; }
    const std::string models = std::getenv("FS_MODELS") ? std::getenv("FS_MODELS") : std::string(FS_REPO_DIR) + "/app/src/main/assets/models";
    InferConfig cfg;
    if (!parse_backends(backend, cfg.preference)) { std::fprintf(stderr, "unknown backend in \"%s\"\n", backend.c_str()); return 2; }
    cfg.load = [models](const std::string& file) { return read_file(models + "/" + file); };
    std::unique_ptr<Inference> inf = Inference::create(cfg);
    struct Job { const char* tag; NetId net; };
    const Job jobs[] = {{"p1", NetId::UVDoc}, {"rc0", NetId::UVDoc}, {"rc2", NetId::SynthDocHR}, {"shade", NetId::ShadeNet}};
    int fails = 0;
    for (size_t k = 2; k < pos.size(); ++k) {
        for (const Job& j : jobs) {
            const std::string in = pos[1] + "/" + pos[k] + "_" + j.tag + "_in.npz";
            if (!exists(in)) continue;
            Npz z = read_npz(in);
            const NpyArray& x = z.at("x");
            Tensor t;
            for (size_t d : x.shape) t.shape.push_back((int64_t)d);
            t.data.assign(x.as<float>(), x.as<float>() + x.count());
            std::vector<Tensor> out;
            std::string err;
            // twice: the first run on a new session can include tuning; the second is the one timed and kept
            bool ok = inf->run(j.net, t, out, &err) && inf->run(j.net, t, out, &err);
            if (!ok || out.empty()) {
                std::printf("%s %s: FAILED %s\n", pos[k].c_str(), j.tag, err.c_str());
                ++fails;
                continue;
            }
            std::vector<size_t> sh;
            for (int64_t d : out[0].shape) sh.push_back((size_t)d);
            write_npz(pos[0] + "/" + pos[k] + "_" + j.tag + ".npz", {{"out", NpyArray::f32(sh, out[0].data.data())}});
            std::printf("%s %s: %s %.2f ms\n", pos[k].c_str(), j.tag, backend_name(inf->backend(j.net)), inf->last_ms(j.net));
        }
    }
    return fails ? 1 : 0;
}
