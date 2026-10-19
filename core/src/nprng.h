// Bit-exact port of numpy's PCG64 bit generator and Generator.choice(n, k, replace=False).
// The reference draws RANSAC triples (corners.ransac_quadratic) and decimates control
// points (warpmap.global_grid) with np.random.default_rng(0); matching those draws keeps
// the port on the same inlier sets and control points as Python.
#pragma once
#include <cstdint>
#include <vector>

namespace fs {

class NpRng {
public:
    // State of np.random.default_rng(0) (SeedSequence(0) -> PCG64), read from numpy 2.2.
    static NpRng seed0() {
        return NpRng(((unsigned __int128)0x1aa1b5345996452dULL << 64) | 0x09585eb7a69561e3ULL,
                     ((unsigned __int128)0x418ddadb3af71a82ULL << 64) | 0x588133bc447873a9ULL);
    }
    NpRng(unsigned __int128 state, unsigned __int128 inc) : state_(state), inc_(inc) {}

    uint64_t next_u64() {
        static const unsigned __int128 MULT = ((unsigned __int128)2549297995355413924ULL << 64) | 4865540595714422341ULL;
        state_ = state_ * MULT + inc_;
        uint64_t hi = (uint64_t)(state_ >> 64), lo = (uint64_t)state_;
        unsigned rot = (unsigned)(state_ >> 122);
        uint64_t x = hi ^ lo;
        return (x >> rot) | (x << ((64 - rot) & 63));
    }
    uint32_t next_u32() {
        if (has_u32_) { has_u32_ = false; return buf_u32_; }
        uint64_t v = next_u64();
        has_u32_ = true;
        buf_u32_ = (uint32_t)(v >> 32);
        return (uint32_t)(v & 0xffffffffu);
    }

    // Generator.choice(pop, size, replace=False, shuffle=True)
    std::vector<int64_t> choice_no_replace(int64_t pop, int64_t size) {
        std::vector<int64_t> idx(size);
        if (pop > 10000 && size > pop / 50) {           // tail shuffle
            std::vector<int64_t> all(pop);
            for (int64_t i = 0; i < pop; ++i) all[i] = i;
            shuffle_int(all.data(), pop, std::max<int64_t>(pop - size, 1));
            for (int64_t i = 0; i < size; ++i) idx[i] = all[pop - size + i];
            return idx;
        }
        uint64_t set_size = (uint64_t)(1.2 * (double)size);
        uint64_t mask = gen_mask(set_size);
        set_size = 1 + mask;
        std::vector<uint64_t> hs(set_size, ~0ULL);
        for (int64_t j = pop - size; j < pop; ++j) {    // Floyd's algorithm
            uint64_t val = bounded_u64((uint64_t)j);
            uint64_t loc = val & mask;
            while (hs[loc] != ~0ULL && hs[loc] != val) loc = (loc + 1) & mask;
            if (hs[loc] == ~0ULL) {
                hs[loc] = val;
                idx[j - pop + size] = (int64_t)val;
            } else {
                loc = (uint64_t)j & mask;
                while (hs[loc] != ~0ULL) loc = (loc + 1) & mask;
                hs[loc] = (uint64_t)j;
                idx[j - pop + size] = j;
            }
        }
        shuffle_int(idx.data(), size, 1);
        return idx;
    }

private:
    static uint64_t gen_mask(uint64_t max) {
        uint64_t m = max;
        m |= m >> 1; m |= m >> 2; m |= m >> 4; m |= m >> 8; m |= m >> 16; m |= m >> 32;
        return m;
    }
    void shuffle_int(int64_t* d, int64_t n, int64_t first) {
        for (int64_t i = n - 1; i >= first; --i) {
            int64_t j = (int64_t)bounded_u64((uint64_t)i);
            std::swap(d[i], d[j]);
        }
    }
    // random_bounded_uint64(off=0, rng, use_masked=false): Lemire's method
    uint64_t bounded_u64(uint64_t rng) {
        if (rng == 0) return 0;
        if (rng <= 0xffffffffULL) {
            if (rng == 0xffffffffULL) return next_u32();
            uint32_t r = (uint32_t)rng, rng_excl = r + 1;
            uint64_t m = (uint64_t)next_u32() * rng_excl;
            uint32_t left = (uint32_t)m;
            if (left < rng_excl) {
                uint32_t thr = (UINT32_MAX - r) % rng_excl;
                while (left < thr) { m = (uint64_t)next_u32() * rng_excl; left = (uint32_t)m; }
            }
            return m >> 32;
        }
        // 64-bit Lemire (not reached for the population sizes used here)
        uint64_t rng_excl = rng + 1;
        unsigned __int128 m = (unsigned __int128)next_u64() * rng_excl;
        uint64_t left = (uint64_t)m;
        if (left < rng_excl) {
            uint64_t thr = (UINT64_MAX - rng) % rng_excl;
            while (left < thr) { m = (unsigned __int128)next_u64() * rng_excl; left = (uint64_t)m; }
        }
        return (uint64_t)(m >> 64);
    }

    unsigned __int128 state_, inc_;
    bool has_u32_ = false;
    uint32_t buf_u32_ = 0;
};

}  // namespace fs
