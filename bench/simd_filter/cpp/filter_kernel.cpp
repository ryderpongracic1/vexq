#include "filter_kernel.h"
#include <immintrin.h>

// ═══════════════════════════════════════════════════════════════════════════════
// int64 kernels (primary — matches vexq's Int64Vector, 4 lanes per AVX2 op)
// ═══════════════════════════════════════════════════════════════════════════════

// Lookup table: for each 4-bit mask pattern, store the indices of set bits.
// Entry [mask][j] = j-th index to store; count = __builtin_popcount(mask).
static const uint32_t LUT_I64[16][4] = {
    {0,0,0,0}, {0,0,0,0}, {1,0,0,0}, {0,1,0,0},   // 0000,0001,0010,0011
    {2,0,0,0}, {0,2,0,0}, {1,2,0,0}, {0,1,2,0},   // 0100,0101,0110,0111
    {3,0,0,0}, {0,3,0,0}, {1,3,0,0}, {0,1,3,0},   // 1000,1001,1010,1011
    {2,3,0,0}, {0,2,3,0}, {1,2,3,0}, {0,1,2,3},   // 1100,1101,1110,1111
};

__attribute__((target("avx2")))
size_t filter_gt_i64_avx2(const int64_t* data, size_t n, int64_t threshold,
                          uint32_t* out_indices) {
    size_t count = 0;
    const __m256i thresh_vec = _mm256_set1_epi64x(threshold);

    size_t i = 0;
    const size_t vec_end = n - (n % 4);

    for (; i < vec_end; i += 4) {
        __m256i vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i cmp = _mm256_cmpgt_epi64(vals, thresh_vec);
        // movemask_pd interprets 256-bit register as 4 doubles, extracting sign bit = MSB of each 64-bit lane
        int mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp));

        // Use lookup table for branchless index scatter
        int popcnt = __builtin_popcount(mask);
        const uint32_t* indices = LUT_I64[mask];
        for (int k = 0; k < popcnt; k++) {
            out_indices[count + k] = static_cast<uint32_t>(i + indices[k]);
        }
        count += popcnt;
    }

    // Scalar tail for remainder
    for (; i < n; i++) {
        if (data[i] > threshold) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
    return count;
}

size_t filter_gt_i64_scalar(const int64_t* data, size_t n, int64_t threshold,
                            uint32_t* out_indices) {
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] > threshold) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// int32 kernels (secondary — 8 lanes per AVX2 op)
// ═══════════════════════════════════════════════════════════════════════════════

// Lookup table for int32: 8-bit mask → indices of set bits.
// For 256 entries × up to 8 indices, pre-compute at startup.
static uint32_t LUT_I32_IDX[256][8];
static uint8_t  LUT_I32_CNT[256];

static struct LUT_I32_Init {
    LUT_I32_Init() {
        for (int m = 0; m < 256; m++) {
            int cnt = 0;
            for (int bit = 0; bit < 8; bit++) {
                if (m & (1 << bit)) {
                    LUT_I32_IDX[m][cnt] = static_cast<uint32_t>(bit);
                    cnt++;
                }
            }
            LUT_I32_CNT[m] = static_cast<uint8_t>(cnt);
        }
    }
} lut_i32_init_;

__attribute__((target("avx2")))
size_t filter_gt_i32_avx2(const int32_t* data, size_t n, int32_t threshold,
                          uint32_t* out_indices) {
    size_t count = 0;
    const __m256i thresh_vec = _mm256_set1_epi32(threshold);

    size_t i = 0;
    const size_t vec_end = n - (n % 8);

    for (; i < vec_end; i += 8) {
        __m256i vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i cmp = _mm256_cmpgt_epi32(vals, thresh_vec);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

        int cnt = LUT_I32_CNT[mask];
        const uint32_t* idx = LUT_I32_IDX[mask];
        for (int k = 0; k < cnt; k++) {
            out_indices[count + k] = static_cast<uint32_t>(i + idx[k]);
        }
        count += cnt;
    }

    // Scalar tail for remainder
    for (; i < n; i++) {
        if (data[i] > threshold) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
    return count;
}

size_t filter_gt_i32_scalar(const int32_t* data, size_t n, int32_t threshold,
                            uint32_t* out_indices) {
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] > threshold) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
    return count;
}
