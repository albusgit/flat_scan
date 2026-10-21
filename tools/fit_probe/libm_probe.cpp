// Prints a checksum per libm function over fixed pseudo-random inputs (fit_probe's companion):
// equal lines on two builds mean that function rounds the same there.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint64_t bits(double v) { uint64_t u; std::memcpy(&u, &v, 8); return u; }
static uint32_t bitsf(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }

int main() {
    uint64_t s = 88172645463325252ull;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double)(s >> 11) / 9007199254740992.0; };
    const int N = 200000;
    uint64_t h[10] = {0};
    int diff_sq[1] = {0};
    for (int i = 0; i < N; ++i) {
        double a = (rnd() - 0.5) * 4000, b = (rnd() - 0.5) * 4000, t = (rnd() - 0.5) * 8, p = rnd() * 2;
        float q = (float)((rnd() - 0.5) * 40);
        uint64_t v[10] = {bits(std::hypot(a, b)), bits(std::cos(t)), bits(std::sin(t)), bits(std::pow(p, 3.0)),
                          bits(std::atan(t)), bits(std::exp(t)), bits(std::log(p + 1e-3)), bits(std::atan2(a, b)),
                          (uint64_t)bitsf(std::exp(q)), (uint64_t)bitsf(std::hypot((float)a, (float)b))};
        for (int k = 0; k < 10; ++k) h[k] = h[k] * 1099511628211ull ^ v[k];
        diff_sq[0] += std::hypot(a, b) != std::sqrt(a * a + b * b);
    }
    const char* nm[10] = {"hypot", "cos", "sin", "pow3", "atan", "exp", "log", "atan2", "expf", "hypotf"};
    for (int k = 0; k < 10; ++k) std::printf("%s %016llx\n", nm[k], (unsigned long long)h[k]);
    std::printf("hypot!=sqrt %d of %d\n", diff_sq[0], N);
    return 0;
}
