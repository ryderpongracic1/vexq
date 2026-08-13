#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "filter_kernel.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration — mirrors vexq's BlockRows (1024) batch size
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr size_t BATCH_SIZE = 1024;
static constexpr size_t TOTAL_ROWS = 6001215;  // TPC-H SF=1 lineitem row count
static constexpr int64_t THRESHOLD_I64 = 25;   // ~50% selectivity on uniform [1,50]
static constexpr int32_t THRESHOLD_I32 = 25;
static constexpr int WARMUP_ITERS = 5;
static constexpr int BENCH_ITERS = 50;
static const char* DATA_FILE_I64 = "bench_data_i64.bin";
static const char* DATA_FILE_I32 = "bench_data_i32.bin";

// ═══════════════════════════════════════════════════════════════════════════════
// Data loading / generation
// ═══════════════════════════════════════════════════════════════════════════════

static void generate_data_i64(const char* path, size_t n) {
    std::vector<int64_t> data(n);
    // Deterministic seed matching the Go benchmark
    uint64_t state = 42;
    for (size_t i = 0; i < n; i++) {
        // xorshift64* — same algorithm used in Go generator
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint64_t val = state * 0x2545F4914F6CDD1DULL;
        data[i] = static_cast<int64_t>((val % 50) + 1);  // [1, 50]
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(n * sizeof(int64_t)));
    out.close();
    fprintf(stderr, "Generated %s: %zu rows (%zu bytes)\n", path, n, n * 8);
}

static void generate_data_i32(const char* path, size_t n) {
    std::vector<int32_t> data(n);
    uint64_t state = 42;
    for (size_t i = 0; i < n; i++) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint64_t val = state * 0x2545F4914F6CDD1DULL;
        data[i] = static_cast<int32_t>((val % 50) + 1);
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(n * sizeof(int32_t)));
    out.close();
    fprintf(stderr, "Generated %s: %zu rows (%zu bytes)\n", path, n, n * 4);
}

static std::vector<int64_t> load_data_i64(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        generate_data_i64(path, n);
        in.open(path, std::ios::binary);
    }
    std::vector<int64_t> data(n);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(n * sizeof(int64_t)));
    return data;
}

static std::vector<int32_t> load_data_i32(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        generate_data_i32(path, n);
        in.open(path, std::ios::binary);
    }
    std::vector<int32_t> data(n);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(n * sizeof(int32_t)));
    return data;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Timing helpers
// ═══════════════════════════════════════════════════════════════════════════════

using Clock = std::chrono::high_resolution_clock;
using Ns = std::chrono::nanoseconds;

struct BenchResult {
    double ns_per_row;
    double total_ms;
    size_t selected;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark runners
// ═══════════════════════════════════════════════════════════════════════════════

static BenchResult bench_i64(const int64_t* data, size_t n, int64_t threshold,
                             size_t (*fn)(const int64_t*, size_t, int64_t, uint32_t*)) {
    std::vector<uint32_t> indices(n);
    size_t sel = 0;

    // Warmup
    for (int w = 0; w < WARMUP_ITERS; w++) {
        for (size_t off = 0; off < n; off += BATCH_SIZE) {
            size_t batch_n = std::min(BATCH_SIZE, n - off);
            fn(data + off, batch_n, threshold, indices.data());
        }
    }

    // Timed iterations
    auto start = Clock::now();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        for (size_t off = 0; off < n; off += BATCH_SIZE) {
            size_t batch_n = std::min(BATCH_SIZE, n - off);
            sel = fn(data + off, batch_n, threshold, indices.data());
            (void)sel;  // prevent optimizing away
        }
    }
    auto end = Clock::now();

    double total_ns = static_cast<double>(std::chrono::duration_cast<Ns>(end - start).count());
    double total_rows = static_cast<double>(n) * BENCH_ITERS;
    // Get final selection count from last batch for correctness
    sel = fn(data, std::min(BATCH_SIZE, n), threshold, indices.data());

    return {total_ns / total_rows, total_ns / 1e6, sel};
}

static BenchResult bench_i32(const int32_t* data, size_t n, int32_t threshold,
                             size_t (*fn)(const int32_t*, size_t, int32_t, uint32_t*)) {
    std::vector<uint32_t> indices(n);
    size_t sel = 0;

    for (int w = 0; w < WARMUP_ITERS; w++) {
        for (size_t off = 0; off < n; off += BATCH_SIZE) {
            size_t batch_n = std::min(BATCH_SIZE, n - off);
            fn(data + off, batch_n, threshold, indices.data());
        }
    }

    auto start = Clock::now();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        for (size_t off = 0; off < n; off += BATCH_SIZE) {
            size_t batch_n = std::min(BATCH_SIZE, n - off);
            sel = fn(data + off, batch_n, threshold, indices.data());
            (void)sel;
        }
    }
    auto end = Clock::now();

    double total_ns = static_cast<double>(std::chrono::duration_cast<Ns>(end - start).count());
    double total_rows = static_cast<double>(n) * BENCH_ITERS;
    sel = fn(data, std::min(BATCH_SIZE, n), threshold, indices.data());

    return {total_ns / total_rows, total_ns / 1e6, sel};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Correctness verification
// ═══════════════════════════════════════════════════════════════════════════════

static bool verify_i64(const int64_t* data, size_t n, int64_t threshold) {
    std::vector<uint32_t> avx_out(n), scalar_out(n);
    size_t avx_count = filter_gt_i64_avx2(data, n, threshold, avx_out.data());
    size_t scalar_count = filter_gt_i64_scalar(data, n, threshold, scalar_out.data());

    if (avx_count != scalar_count) {
        fprintf(stderr, "FAIL i64: count mismatch: AVX2=%zu scalar=%zu\n", avx_count, scalar_count);
        return false;
    }
    for (size_t i = 0; i < avx_count; i++) {
        if (avx_out[i] != scalar_out[i]) {
            fprintf(stderr, "FAIL i64: index mismatch at pos %zu: AVX2=%u scalar=%u\n",
                    i, avx_out[i], scalar_out[i]);
            return false;
        }
    }
    return true;
}

static bool verify_i32(const int32_t* data, size_t n, int32_t threshold) {
    std::vector<uint32_t> avx_out(n), scalar_out(n);
    size_t avx_count = filter_gt_i32_avx2(data, n, threshold, avx_out.data());
    size_t scalar_count = filter_gt_i32_scalar(data, n, threshold, scalar_out.data());

    if (avx_count != scalar_count) {
        fprintf(stderr, "FAIL i32: count mismatch: AVX2=%zu scalar=%zu\n", avx_count, scalar_count);
        return false;
    }
    for (size_t i = 0; i < avx_count; i++) {
        if (avx_out[i] != scalar_out[i]) {
            fprintf(stderr, "FAIL i32: index mismatch at pos %zu: AVX2=%u scalar=%u\n",
                    i, avx_out[i], scalar_out[i]);
            return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    // Runtime AVX2 detection
    if (!__builtin_cpu_supports("avx2")) {
        fprintf(stderr, "ERROR: This CPU does not support AVX2 instructions.\n");
        fprintf(stderr, "       The SIMD benchmark requires AVX2. Exiting.\n");
        return 1;
    }

    printf("=== SIMD Filter Kernel Benchmark ===\n");
    printf("Batch size: %zu rows (matches vexq BlockRows)\n", BATCH_SIZE);
    printf("Total rows: %zu (TPC-H SF=1 lineitem count)\n", TOTAL_ROWS);
    printf("Threshold:  >%lld (int64), >%d (int32) — ~50%% selectivity\n",
           (long long)THRESHOLD_I64, THRESHOLD_I32);
    printf("Iterations: %d (warmup: %d)\n\n", BENCH_ITERS, WARMUP_ITERS);

    // Load data
    auto data_i64 = load_data_i64(DATA_FILE_I64, TOTAL_ROWS);
    auto data_i32 = load_data_i32(DATA_FILE_I32, TOTAL_ROWS);

    // ─── Correctness check ────────────────────────────────────────────────────
    printf("--- Correctness Verification ---\n");
    bool ok = true;
    // Test multiple batch sizes to exercise tail handling
    for (size_t test_n : {size_t(1024), size_t(1023), size_t(1025), size_t(7), TOTAL_ROWS}) {
        size_t actual_n = std::min(test_n, TOTAL_ROWS);
        if (!verify_i64(data_i64.data(), actual_n, THRESHOLD_I64)) ok = false;
        if (!verify_i32(data_i32.data(), actual_n, THRESHOLD_I32)) ok = false;
    }
    if (ok) {
        printf("PASS: All correctness checks passed (identical selection vectors)\n\n");
    } else {
        printf("FAIL: Correctness check failed — aborting benchmark\n");
        return 1;
    }

    // ─── int64 benchmarks (PRIMARY — maps to vexq engine) ─────────────────────
    printf("--- int64 (PRIMARY — matches vexq Int64Vector) ---\n");
    auto r64_scalar = bench_i64(data_i64.data(), TOTAL_ROWS, THRESHOLD_I64, filter_gt_i64_scalar);
    auto r64_avx2 = bench_i64(data_i64.data(), TOTAL_ROWS, THRESHOLD_I64, filter_gt_i64_avx2);

    printf("  scalar:  %.2f ns/row  (%.1f ms total)\n", r64_scalar.ns_per_row, r64_scalar.total_ms);
    printf("  AVX2:    %.2f ns/row  (%.1f ms total)\n", r64_avx2.ns_per_row, r64_avx2.total_ms);
    printf("  speedup: %.2fx\n\n", r64_scalar.ns_per_row / r64_avx2.ns_per_row);

    // ─── int32 benchmarks (secondary data point) ──────────────────────────────
    printf("--- int32 (secondary — 8 lanes vs 4 for int64) ---\n");
    auto r32_scalar = bench_i32(data_i32.data(), TOTAL_ROWS, THRESHOLD_I32, filter_gt_i32_scalar);
    auto r32_avx2 = bench_i32(data_i32.data(), TOTAL_ROWS, THRESHOLD_I32, filter_gt_i32_avx2);

    printf("  scalar:  %.2f ns/row  (%.1f ms total)\n", r32_scalar.ns_per_row, r32_scalar.total_ms);
    printf("  AVX2:    %.2f ns/row  (%.1f ms total)\n", r32_avx2.ns_per_row, r32_avx2.total_ms);
    printf("  speedup: %.2fx\n\n", r32_scalar.ns_per_row / r32_avx2.ns_per_row);

    // ─── Machine-readable summary ────────────────────────────────────────────
    printf("--- MACHINE READABLE ---\n");
    printf("RESULT i64_scalar_ns_per_row=%.2f\n", r64_scalar.ns_per_row);
    printf("RESULT i64_avx2_ns_per_row=%.2f\n", r64_avx2.ns_per_row);
    printf("RESULT i64_speedup=%.2f\n", r64_scalar.ns_per_row / r64_avx2.ns_per_row);
    printf("RESULT i32_scalar_ns_per_row=%.2f\n", r32_scalar.ns_per_row);
    printf("RESULT i32_avx2_ns_per_row=%.2f\n", r32_avx2.ns_per_row);
    printf("RESULT i32_speedup=%.2f\n", r32_scalar.ns_per_row / r32_avx2.ns_per_row);
    printf("RESULT batch_size=%zu\n", BATCH_SIZE);
    printf("RESULT total_rows=%zu\n", TOTAL_ROWS);
    printf("RESULT iterations=%d\n", BENCH_ITERS);

    return 0;
}
