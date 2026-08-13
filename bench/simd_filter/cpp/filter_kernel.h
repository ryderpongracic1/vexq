#ifndef FILTER_KERNEL_H
#define FILTER_KERNEL_H

#include <cstddef>
#include <cstdint>

// ─── int64 kernels (primary — matches vexq's Int64Vector) ───────────────────

// AVX2: compare 4 int64 lanes per iteration, produce selection vector of
// indices where data[i] > threshold.  Returns count of selected indices.
size_t filter_gt_i64_avx2(const int64_t* data, size_t n, int64_t threshold,
                          uint32_t* out_indices);

// Scalar reference with identical semantics.
size_t filter_gt_i64_scalar(const int64_t* data, size_t n, int64_t threshold,
                            uint32_t* out_indices);

// ─── int32 kernels (secondary data point — 8 lanes per AVX2 op) ─────────────

// AVX2: compare 8 int32 lanes per iteration.
size_t filter_gt_i32_avx2(const int32_t* data, size_t n, int32_t threshold,
                          uint32_t* out_indices);

// Scalar reference with identical semantics.
size_t filter_gt_i32_scalar(const int32_t* data, size_t n, int32_t threshold,
                            uint32_t* out_indices);

#endif // FILTER_KERNEL_H
