/*
 * gf2.cuh — shared GF(2) bit-packed device helpers for the osd_0_cuda decoder.
 *
 * Every row of the augmented matrix [H | s] is stored as W = ceil((N+1)/64)
 * 64-bit words. Column c lives in word (c >> 6), bit (c & 63). The syndrome
 * occupies column N (the last column), so it is just "one more variable index".
 *
 * GF(2) arithmetic is dtype-independent: addition is XOR, so the same kernels
 * serve every tensor dtype the rest of syndrilla uses. The float LLR only
 * decides the *column ordering*, which is computed on the host with torch.sort
 * before launch — no floating point ever enters these kernels.
 *
 * These helpers are the main customization surface: a different pivoting rule
 * or a different read-out only needs to recombine the primitives below.
 *
 * All helpers are `static __device__ __forceinline__` so each translation unit
 * that includes this header gets its own internal-linkage copy (no multiple-
 * definition errors if this header is included from more than one .cu file).
 */
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
