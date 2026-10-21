#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
int main() {
    long n1 = 0, N = 0;
    uint64_t h = 0;
    for (float x = -30.f; x < 30.f; x = std::nextafter(x, 100.f) ) {
        // every 37th float in [-30, 30)
        for (int k = 0; k < 37 && x < 30.f; ++k) x = std::nextafter(x, 100.f);
        float a = std::exp(x), b = (float)std::exp((double)x);
        n1 += a != b; ++N;
        uint32_t u; std::memcpy(&u, &b, 4); h = h * 1099511628211ull ^ u;
    }
    std::printf("expf != (float)exp(double): %ld of %ld; hash of (float)exp(double) %016llx\n", n1, N, (unsigned long long)h);
}
