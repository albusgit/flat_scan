#include <cmath>
#include <cstdio>
#include <cstdint>
int main() {
    uint64_t s = 88172645463325252ull;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double)(s >> 11) / 9007199254740992.0; };
    long nd = 0, nf = 0, nf2 = 0, N = 2000000;
    for (long i = 0; i < N; ++i) {
        double sc = std::pow(10.0, rnd() * 8 - 4);
        double a = (rnd() - 0.5) * sc, b = (rnd() - 0.5) * sc * (rnd() < 0.1 ? 1e-3 : 1);
        nd += std::hypot(a, b) != std::sqrt(a * a + b * b);
        float fa = (float)((int)(rnd() * 256) - 128), fb = (float)((int)(rnd() * 256) - 128);
        if (i & 1) { fa = (float)(a); fb = (float)(b); }
        nf += std::hypot(fa, fb) != (float)std::sqrt((double)fa * fa + (double)fb * fb);
        nf2 += std::hypot(fa, fb) != std::sqrt(fa * fa + fb * fb);
    }
    std::printf("double hypot != sqrt: %ld; float hypot != (float)sqrt(double): %ld; != sqrtf: %ld of %ld\n", nd, nf, nf2, N);
}
