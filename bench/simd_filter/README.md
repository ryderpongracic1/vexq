# SIMD Filter Kernel Benchmark

Isolated measurement of the filter-predicate comparison + selection-vector-build kernel using AVX2 intrinsics vs the equivalent Go scalar loop in vexq's execution engine.

## Results

| Kernel | ns/row | Speedup vs Go |
|--------|-------:|:-------------:|
| **Go scalar (engine-equivalent)** | 8.17 | 1.0× |
| C++ scalar `-O3 -fno-tree-vectorize` (no auto-vec) | 4.0 | 2.0× |
| C++ scalar `-O3` (auto-vectorized) | 3.85 | 2.1× |
| **C++ AVX2 int64 intrinsics** | 2.44 | **3.3×** |
| C++ AVX2 int32 (secondary, 8 lanes) | 1.46 | 5.6× |

Numbers are medians of 3 runs; roughly ±5% run-to-run variance observed on this host (an independent re-run measured 2.57 ns/row for the AVX2 int64 kernel vs the reported 2.44, i.e. 3.2–3.3× vs Go either way).

**Primary kernel is int64** — this matches vexq's `Int64Vector`, which is the internal integer representation used in `evalCmpInt64` (the hot comparison loop in `exec/expr.go`). The int32 result is a secondary data point showing the benefit of 8-wide vs 4-wide SIMD lanes.

## Measurement Context

- **CPU**: Intel Xeon 6975P-C (AVX2, no AVX-512)
- **Compiler**: g++ 11.5.0 (GCC, Red Hat)
- **Go**: go1.25.12 linux/amd64
- **OS**: Amazon Linux (linux/amd64)
- **Flags**: `-std=c++17 -O3 -mavx2` (AVX2 build), `-std=c++17 -O3 -mavx2 -fno-tree-vectorize -fno-tree-loop-vectorize` (novec build)
- **Batch size**: 1024 rows (matches vexq `BlockRows` from `storage.BlockRows`)
- **Data**: 6,001,215 synthetic int64 values, uniform in [1, 50], deterministic xorshift64* seed=42
- **Threshold**: >25 (~50% selectivity)
- **Iterations**: 50 full passes over all 6M rows (per benchmark variant), 5 warmup

## Relationship to the DuckDB Gap

The top-level README documents a **28× gap on Q6** (predicate-heavy filter query) between vexq and DuckDB. The filter kernel is one component of that gap.

This benchmark measures the **comparison + selection-vector build** step in isolation — the same work done by `evalCmpInt64` + `BoolToSelVec` in vexq's `Filter` operator. The 3.3× speedup from explicit AVX2 intrinsics over the Go engine loop represents a **bounded ceiling** on what SIMD alone could contribute to closing the Q6 gap. It does NOT account for other components of DuckDB's advantage:

- **Late materialisation** — DuckDB decodes only filter columns first, then payload columns for survivors only
- **SIMD predicate mask composition** — DuckDB evaluates conjunctive predicates as mask intersections, not separate passes
- **JIT compilation** — DuckDB eliminates interpreter/dispatch overhead entirely
- **Predicate-aware I/O** — DuckDB skips decoding non-predicate columns at the storage layer

The 3.3× filter-kernel speedup is one slice of the 28×, not a path to closing it entirely. Realistically, SIMD filter kernels could contribute ~3–4× on the comparison hot loop, which combined with late materialisation (~2× on high-selectivity queries) might close roughly 6–8× of the gap — consistent with the top-level README's estimate of "4–8× improvement on filter-heavy queries" from explicit SIMD.

## Why This Is NOT Wired Into the Live Execution Path

This benchmark deliberately lives in `bench/` and is not integrated via cgo:

1. **cgo call overhead** — Each cgo boundary crossing costs ~100–200 ns. At 1024-row batches, the kernel itself runs in ~2.5 µs. A single cgo call would add 4–8% overhead, and the filter pipeline makes one call per batch per predicate column. For Q6's 5 predicates, that's 5 cgo calls per batch — potentially 20–40% overhead from the boundary alone.

2. **Scope control** — The goal is to measure the *theoretical ceiling* of what SIMD buys for this specific kernel shape, not to prematurely optimize a single operator. The full-engine improvement requires architectural changes (late materialisation, predicate fusion) that compose with SIMD, not just a drop-in kernel replacement.

3. **Go's compiler is improving** — Go 1.22+ auto-vectorizes some patterns. A pure-Go SIMD approach (via `go:linkname` or assembly stubs) avoids cgo overhead entirely and is the right integration path when the time comes.

## Reproducing

```bash
cd bench/simd_filter

# Build and run C++ benchmark (generates data on first run)
make build && make run

# Build and run scalar-novec variant (honest non-vectorized baseline)
make scalar-novec && make run-novec

# Run Go benchmark (uses same shared data file)
make bench-go

# Clean up
make clean
```

## Data Generation

Both the C++ and Go harnesses use the same deterministic data generator (xorshift64* with seed 42) producing values in [1, 50]. The C++ harness writes `bench_data_i64.bin` on first run; the Go benchmark reads it or regenerates identically if missing. This ensures both measure literally the same bytes.

The data is synthetic (not real TPC-H) because no pre-generated `.tbl` or `.vxq` files exist in this workspace. The row count (6,001,215) matches TPC-H SF=1 lineitem, and the value distribution (uniform [1, 50]) approximates `l_quantity`'s range.
