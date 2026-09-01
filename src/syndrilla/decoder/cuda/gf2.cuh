#pragma once

#include <cstdint>

// True iff bit `c` of a packed row is set.
static __device__ __forceinline__ bool gf2_get(const uint64_t* row, int c) {
    return (row[c >> 6] >> (c & 63)) & 1ULL;
}

// Set bit `c` of a packed row.
static __device__ __forceinline__ void gf2_set(uint64_t* row, int c) {
    row[c >> 6] |= (1ULL << (c & 63));
}

// dst ^= src  over all W words (GF(2) row addition).
static __device__ __forceinline__ void gf2_xor_row(uint64_t* dst,
                                                   const uint64_t* src, int W) {
    for (int w = 0; w < W; w++) dst[w] ^= src[w];
}

// Number of 64-bit words needed to hold (N + 1) columns (N variables + syndrome).
static __device__ __host__ __forceinline__ int gf2_words(int N) {
    return ((N + 1) + 63) >> 6;
}
