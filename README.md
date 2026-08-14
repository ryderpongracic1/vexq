# vexq

[![CI](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml/badge.svg)](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml)

Built a vectorized analytical SQL execution engine in Go using columnar storage, dictionary encoding, zone maps, and morsel-driven parallel execution; implemented vectorized Volcano operators and a custom query parsing pipeline.

## Overview

vexq is a complete analytical query engine — from a custom on-disk columnar file format through a SQL parser, rule-based optimizer, and vectorized execution engine. The design shares structural principles with [distrikv](https://github.com/ryderpongracic1/distrikv) (append-only manifest, block CRCs, atomic writes via temp+rename) while adding a columnar-vectorized execution model suited for OLAP workloads.

On TPC-H Q1 (GROUP BY aggregate over 6M rows), vexq is within 1.5× of DuckDB single-threaded — closing the per-core gap through dictionary-code integer-keyed aggregation and buffer-reuse elimination of GC pressure.

The engine processes data in batches of 1024 rows, designed to saturate L1 cache and amortize operator dispatch overhead. A pushed-down predicate + zone-map pruning layer skips entire row groups before any I/O, and dictionary encoding reduces string comparisons to integer equality in the filter hot loop.

## Architecture

```
cmd/vexq       — CLI: query execution + fsck integrity check
cmd/vexqgen    — TPC-H .tbl → .vxq converter

sql/           — Lexer + recursive-descent parser (Pratt precedence)
planner/       — Logical plan builder, rule-based optimizer, physical planner
  optimizer    — Predicate pushdown, column pruning
  physical     — Zone-map predicates, logical → exec.Operator tree
  parallel     — Morsel-driven parallel planner (planner.Parallel)

exec/          — Vectorized operator pipeline
  TableScan         — Columnar I/O with zone-map pruning and column projection
  Filter            — Selection-vector based (no allocation on hot path)
  Project           — Lazy materialization through selection vectors
  HashAggregate     — Hash-partitioned GROUP BY with float64-correct SUM/AVG
  ParallelHashAggregate — Work-stealing morsel scheduler + partial-aggregate merge
  ExternalSort      — In-memory sort (spill-to-disk planned for v2)
  HashJoin          — Build/probe inner join
  Limit

catalog/       — Table registry with lazy schema loading from .vxq footer
storage/       — .vxq file format: writer, reader, block codec, zone maps
internal/encoding — Little-endian primitives, CRC32-IEEE helpers
internal/goldentest — End-to-end correctness oracle (48-query suite)
bench/tpch     — TPC-H Q1/Q3/Q6/Q12 benchmarks vs SQLite and DuckDB
bench/simd_filter — Isolated AVX2 filter kernel benchmark (ceiling measurement, x86-64)
```

## Hardware-Level Architecture

### Why 1024-row batches

An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB — well within L1 cache on every modern microarchitecture (32–48 KB on x86 Skylake+, **192 KB** on Apple M4 Pro performance cores). This means the innermost decode, filter, and aggregate loops all operate on data that is almost certainly already in L1, avoiding LLC round-trips that cost ~40 cycles each.

The batch size also amortizes the cost of one virtual `Next()` call across 1024 rows. A virtual dispatch on x86 is ~50 ns (indirect branch, possible iTLB miss); spread over 1024 rows that is ~0.05 ns per row — negligible. Processing row-at-a-time would spend more time dispatching than computing. This is the same constant used by Velox, DuckDB, and Photon.

### Cache-line discipline (64 bytes)

The `TableScan` decode loops in [`exec/scan.go`](exec/scan.go) operate on tightly-packed little-endian payloads. For `INT64` columns, each iteration of the hot loop is:

```go
vals[i] = int64(binary.LittleEndian.Uint64(payload[i*8:]))
```

On x86-64, `binary.LittleEndian.Uint64` compiles to a single `MOVQ` (no byte-swap). On ARM64 (M4 Pro), it compiles to a single `LDR`. Eight consecutive values occupy exactly 64 bytes — one cache line — so the hardware prefetcher can issue fetch requests ahead of the loop at full bandwidth. Big-endian would require a `MOVQ+BSWAP` (x86) or `LDR+REV` (ARM64) pair, penalizing every decode by ~10–20%.

### Branch-predictor friendliness

`Filter` ([`exec/filter.go`](exec/filter.go)) evaluates the predicate across an entire 1024-row batch to produce a `BoolVector`, then converts that vector to a `SelectionVector` of surviving row indices in a single pass:

```
predicate.Eval(batch) → BoolVector → BoolToSelVec → []uint16 indices
```

The predicate eval loop has no data-dependent branches — every row pays the same cost regardless of whether it passes or fails. The branch predictor sees a perfectly regular loop and hits ~100%. Downstream operators receive a `SelVec` and index through it; they never branch on per-row pass/fail.

### Why selection vectors, not filtered copies

`Filter` writes a `[]uint16` of surviving row indices rather than allocating new column vectors. Downstream operators index through the selection vector, which means:
- Zero allocation on the filter hot path
- The 1024-row buffer is reused across the pipeline
- Column data is never copied — only indices change

### Morsel-driven parallelism

`planner.Parallel()` ([`planner/parallel.go`](planner/parallel.go)) detects aggregate plan shapes — including `Sort → Aggregate → (Filter →)? Scan` — and builds a `ParallelHashAggregate` ([`exec/parallel.go`](exec/parallel.go)), wrapping it with a serial sort/limit where needed.

Rather than statically assigning equal slices of row groups to each goroutine (which starves fast workers when selectivity varies), the executor uses an **atomic-counter morsel queue**: each of the `runtime.NumCPU()` goroutines claims row groups one at a time from a shared `atomic.Int64` cursor. Fast workers — those whose morsels have high filter selectivity — claim more work; slow workers naturally finish last without blocking others. The counter is padded to its own 64-byte cache line to prevent false-sharing with other struct fields.

Each goroutine runs a fully independent `TableScan → Filter → Project → HashAggregate` pipeline on its morsels, accumulates partial results locally with no shared mutable state, then sends its `HashAggregate` state on a buffered channel. The calling goroutine merges all partial aggregates (correctly handling float64 SUM/MIN/MAX via IEEE-bit re-encoding and AVG via sum+count).

This design follows the principles of [Leis et al., "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age," SIGMOD 2014](https://db.in.tum.de/~leis/papers/morsels.pdf).

## .vxq File Format

Custom columnar format designed for vectorized reads:

- **Layout**: file header → row groups (65,536 rows each) → footer
- **Blocks**: 1,024 rows per block with 128-byte null bitmap + typed payload + CRC32
- **Endianness**: little-endian throughout (single `MOVQ`/`LDR` on x86-64/ARM64 vs `MOVQ+BSWAP`/`LDR+REV` for big-endian)
- **String columns**: always dictionary-encoded per row group — string equality becomes integer comparison in the filter hot loop
- **Bool columns**: run-length encoded with null sentinel
- **Zone maps**: per-row-group min/max/sum/nullcount in footer — entire row groups skipped before any block I/O
- **Atomic writes**: `write → fsync → rename` guarantees no partial files on crash

## SQL Support

```sql
SELECT expr [AS alias], ...
FROM table [alias] [, table2 [alias2], ...]
[WHERE condition]
[GROUP BY col, ...]
[HAVING condition]
[ORDER BY col [ASC|DESC], ...]
[LIMIT n]

-- Joins: implicit inner join via FROM t1, t2 WHERE t1.key = t2.key
-- Column references: unqualified (col), qualified (table.col), aliased (alias.col)
-- Ambiguous columns: error when an unqualified name exists in multiple tables
-- Cross joins: explicitly rejected — every table must connect via an equi-join condition
-- Aggregate functions: COUNT(*), COUNT(col), COUNT(DISTINCT col), SUM, AVG, MIN, MAX
-- HAVING: supports aggregate expressions directly (HAVING COUNT(*) > 5) and output aliases
-- Predicates: =, <>, <, <=, >, >=, AND, OR, NOT, BETWEEN, IN, LIKE, IS NULL
-- Expressions: arithmetic (+, -, *, /), CASE WHEN (string and numeric results), DISTINCT, unary minus
```

## Usage

```bash
# Convert a TPC-H table to .vxq
vexqgen lineitem lineitem.tbl lineitem.vxq

# Run a query
vexq lineitem.vxq "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"

# Run a query with parallel execution (4 workers)
vexq --workers=4 lineitem.vxq "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"

# Multi-table join query
vexq lineitem.vxq orders.vxq "SELECT o_orderkey, l_quantity FROM orders, lineitem WHERE o_orderkey = l_orderkey LIMIT 10"

# Validate file integrity (CRC, footer, zone maps)
vexq fsck lineitem.vxq
```

## Build

```bash
go build ./cmd/vexq/
go build ./cmd/vexqgen/
go test ./... -race -count=1
```

Requires Go 1.22+. No external runtime dependencies (SQLite and DuckDB are benchmark-only).

## Benchmarks

TPC-H scale factor 1 (6M lineitem rows) on Apple M4 Pro (14-core, 192 KB L1D per performance core, 36 MB SLC). Page cache warm. Each benchmark run 10×; numbers are median wall time. The headline result is per-core parity with DuckDB on Q1 (1.5×); see the DuckDB decomposition below.

### Correctness

All four TPC-H query results verified identical to SQLite output (via the in-harness `TestQ*Correctness` assertions). Additionally, an independent 48-query golden test suite ([`internal/goldentest/`](internal/goldentest/)) verifies the full SQL subset against a naive row-at-a-time reference evaluator — 48/48 passing, zero known correctness issues.

### Ceiling — vs DuckDB (SOTA embedded OLAP)

DuckDB is a state-of-the-art embedded analytical engine: SIMD intrinsics, a cost-based optimizer, radix-partitioned hash aggregation, adaptive parallel morsel scheduling, LLVM JIT compilation, and a columnar execution engine written in C++.

The table below decomposes the gap into **per-core execution efficiency** (single-threaded DuckDB vs vexq serial) and **total throughput** (DuckDB all-cores vs vexq serial). This decomposition separates SIMD/JIT advantages from parallelism, which is the more informative comparison for a from-scratch engine.

Note: this comparison runs DuckDB on ARM64, where its strongest x86 SIMD paths (AVX-512) are unavailable; an x86 comparison would likely widen DuckDB's per-core advantage. The comparison is same-machine and therefore fair, but the 1.5× Q1 figure should be read with that context.

| Query | vexq serial | DuckDB (1 thread) | DuckDB (14 threads) | Per-core gap | Total gap |
|-------|-------------|-------------------|---------------------|:------------:|:---------:|
| Q1 | 239 ms | 159 ms | 25 ms | **1.5×** | **10×** |
| Q6 | 111 ms | 37 ms | 8 ms | **3.0×** | **14×** |
| Q3 | 684 ms | 62 ms | 17 ms | **11×** | **40×** |
| Q12 | 1,050 ms | 81 ms | 16 ms | **13×** | **66×** |

**What the decomposition reveals:**

- **Q1 (1.5× per-core):** The smallest gap. vexq's dictionary-code integer-keyed aggregation (packed `uint64` group keys, never a string in the hot loop) puts it within striking distance of DuckDB single-threaded. The remaining 1.5× is DuckDB's radix-partitioned hash aggregate + SIMD horizontal SUM.

- **Q6 (3.0× per-core):** Consistent with the SIMD filter ceiling measured on x86 in [`bench/simd_filter/`](bench/simd_filter/) (3.3×), though measured on a different microarchitecture (Intel Xeon 6975P-C). The per-core gap is likely dominated by vectorized predicate evaluation, but a precise attribution requires profiling the fraction of Q6 serial time spent in the filter kernel on ARM64.

- **Q3/Q12 (11–13× per-core):** Dominated by the hash join. DuckDB's radix-partitioned, SIMD-probed join keeps each partition L2-resident; vexq's `HashJoin` builds a Go `map[int64][]Batch` and probes with random access (likely L3 misses). Closing this requires a radix-partitioned join — a structural change, not a tuning pass.

- **Parallelism gap:** DuckDB achieves 4.5–6.4× scaling from threads=1 to threads=14. vexq achieves 1.2–1.9× (see parallel section below). The remaining GC pressure from per-batch allocations in filter/project operators is the primary limiter.

### Hardware migration and normalization

The benchmarks moved from Apple M1 Pro (10-core) to Apple M4 Pro (14-core). Using SQLite as a platform-neutral control (same workload, no code changes between runs): Q1 3,463→2,197 ms (1.58×), Q6 599→408 ms (1.47×), Q3 3,719→2,525 ms (1.47×), Q12 1,073→724 ms (1.48×). The hardware is worth roughly 1.5×. Normalizing vexq's improvements against this:

This normalization is approximate: SQLite is a pointer-chasing B-tree row-store dominated by memory latency, while vexq is a streaming columnar scanner — the two do not scale identically across microarchitectures, so the ~1.5× hardware factor is a rough correction rather than an exact one.

| Query | Raw improvement | Hardware factor | Software-only improvement |
|-------|----------------|-----------------|---------------------------|
| Q1 | 2.75× (657→239) | ÷1.58 | **1.74×** (aggregate intkey rework + decode-buffer reuse; both landed in this window; credit not separable without an ablation) |
| Q6 | 1.78× (198→111) | ÷1.47 | **1.21×** (decode-buffer reuse + aggregate intkey rework; both landed in this window; credit not separable without an ablation) |
| Q3 | 1.66× (1133→684) | ÷1.47 | **1.13×** (minor — no join changes) |
| Q12 | 1.56× (1643→1050) | ÷1.48 | **1.06×** (no targeted optimization) |

The Q1 improvement is almost entirely software (the dictionary-code integer-key aggregate rework targeted precisely this query). Q3/Q12 improvements are almost entirely hardware — no join optimizations were made.

### Floor — vs SQLite (row-store OLTP)

SQLite is a B-tree row-store engine designed for OLTP: it reads full rows, applies predicates row-at-a-time, and has no columnar I/O or vectorized aggregation. Beating a row-store on full-table OLAP scans is the **expected** outcome of any columnar engine — this baseline confirms the columnar layout and vectorized operators are paying off, not that the engine is production-grade.

| Query | Description | vexq | SQLite | Speedup |
|-------|-------------|------|--------|---------|
| Q1 | Pricing summary — full scan, GROUP BY 2 string cols | 239 ms | 2,197 ms | **9.2×** |
| Q6 | Revenue forecast — scan with 5 range predicates, SUM | 111 ms | 408 ms | **3.7×** |
| Q3 | Shipping priority — 3-table join, complex SUM, LIMIT 10 | 684 ms | 2,525 ms | **3.7×** |
| Q12 | Shipping modes — 2-table join, CASE WHEN agg, date comparisons | 1,050 ms | 724 ms | 0.69× |

Q12 remains close to parity with SQLite: the `HashJoin` build phase materializes the full orders table and SQLite benefits from its B-tree index on `o_orderkey`. Future work: index-nested-loop join and late materialization would close this gap.

### What's left on the table

- **Explicit SIMD**: Use `avo` or Go assembly to generate AVX2/AVX-512 kernels for the hot decode and comparison loops — likely 1.5–2× end-to-end improvement on filter-heavy queries (filter is an estimated 50–60% of Q6 serial time (pprof measurement pending); a 3.3× kernel speedup yields ~1.8× end-to-end via Amdahl). See [`bench/simd_filter/`](bench/simd_filter/) for the kernel measurement: **AVX2 int64 intrinsics achieve 2.44 ns/row vs 8.17 ns/row (3.3× speedup)**, measured on Intel Xeon 6975P-C (x86-64).
- **Parallel hash join**: Extend `planner.Parallel()` to detect join shapes and partition the build side — would bring Q3/Q12 into the parallel path.
- **Late materialization**: Avoid decoding non-predicate columns until after the filter selection vector is built — saves decode work proportional to filter selectivity.
- **Further GC reduction**: Extend buffer reuse from TableScan (already done) to Filter and Project operators — would improve parallel scaling toward the hardware ceiling.
- **Adaptive compression**: Delta encoding for sorted integer columns (timestamps, order keys) could improve decode throughput and reduce I/O.

### Parallel execution (morsel-driven, 14 goroutines)

`planner.Parallel()` detects aggregate plan shapes — including `Sort → Aggregate` and `Limit → Sort → Aggregate` — and partitions the scan across `runtime.NumCPU()` goroutines using a dynamic atomic-counter morsel queue. Each goroutine runs an independent pipeline on dynamically claimed morsels; a merge step combines partial aggregates in the calling goroutine.

| Query | vexq serial | vexq parallel | Speedup | Notes |
|-------|-------------|---------------|---------|-------|
| Q1 | 239 ms | **129 ms** | **1.85×** | Sort-peeling: parallel aggregate, serial 4-row sort |
| Q6 | 111 ms | **96 ms** | **1.16×** | GC-limited — remaining per-batch allocations in filter/project |

Q3/Q12 fall back to `planner.Physical()` because they contain `HashJoin`, which is not yet parallelized. Parallel scaling is currently limited by Go GC pressure: decode-buffer reuse in `TableScan` reduced GC cycles by 30%, but small per-batch allocations in downstream operators remain. With GC on, parallel efficiency (speedup ÷ N) is 8% (Q6) and 13% (Q1). The ablation below shows this is not a fixed serial-fraction limit: GC overhead grows with worker count (more goroutines allocating → more GC cycles and STW time), which is why an Amdahl model cannot fit these numbers — the ablation exceeds the ceiling any fitted serial fraction would imply.

Ablation with `GOGC=off` proves this is GC pressure, not contention: Q6 parallel drops from 96 ms to 29 ms (3.8× scaling), Q1 from 129 ms to 43 ms (5.6× scaling). The morsel scheduler scales correctly once GC is removed — the remaining gap to linear is most plausibly P/E-core asymmetry: the M4 Pro's 14 cores are 10 performance + 4 efficiency cores, and runtime.NumCPU() spawns 14 workers, four of which land on E-cores with a fraction of P-core throughput. Against a realistic 10-P-core ceiling, Q1's scaling is ~56% efficiency. A worker-count sweep (1→14 under GOGC=off) to demonstrate this empirically is a pending measurement. It is explicitly not memory bandwidth: Q6 decodes ~288 MB in 29 ms (~10 GB/s), under 5% of the platform's available bandwidth.

| Query | Parallel (GC on) | Parallel (GOGC=off) | Scaling (GOGC=off) |
|-------|-----------------|--------------------:|-------------------:|
| Q1 | 129 ms | 43 ms | **5.6×** |
| Q6 | 96 ms | 29 ms | **3.8×** |

These scaling figures divide parallel (GOGC=off) by serial (GC on) and therefore bundle the GC-removal speedup into the parallel speedup; the like-for-like control (serial under GOGC=off) is a pending measurement and will lower these figures somewhat. GOGC=off is a diagnostic, not a production configuration — unbounded heap growth is not viable. The realistic mitigations are the buffer-reuse work already applied to TableScan (extending to Filter/Project), and GOMEMLIMIT as a bounded middle ground.

### Running benchmarks

```bash
# Generate TPC-H SF=1 data (requires tpch-dbgen)
cd data && dbgen -s 1 -f

# Convert to .vxq
vexqgen lineitem  data/lineitem.tbl  data/lineitem.vxq
vexqgen orders    data/orders.tbl    data/orders.vxq
vexqgen customer  data/customer.tbl  data/customer.vxq

# Run serial + parallel benchmarks (10 runs each)
go test ./bench/tpch/ -bench=. -benchtime=10x -count=3 -v

# Run without DuckDB (no CGO dependency)
go test ./bench/tpch/ -bench=. -benchtime=10x -v

# DuckDB single-threaded comparison (install duckdb CLI)
duckdb data/tpch.duckdb -c "SET threads=1; ..."
```

## Profiling

```bash
# CPU profile of Q1 — look for payloadToVector and accumulate in the flame graph
go test ./bench/tpch/ -bench=BenchmarkVexqQ1$ -benchtime=10x \
  -cpuprofile=cpu.out -benchmem
go tool pprof -http=:8080 cpu.out

# Allocation profile — look for per-batch allocations in filter/project
go test ./bench/tpch/ -bench=BenchmarkVexqQ1$ -benchtime=10x \
  -memprofile=mem.out
go tool pprof -alloc_objects -http=:8080 mem.out

# Cache-miss analysis (macOS — Instruments)
xctrace record --template "CPU Counters" --launch -- \
  ./vexq data/lineitem.vxq "SELECT l_returnflag, l_linestatus, SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag, l_linestatus"

# Linux equivalent
perf stat -e cache-misses,cache-references,branch-misses \
  ./vexq data/lineitem.vxq "SELECT ..."
```

## Progress

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | `.vxq` storage format — writer, reader, codec, zone maps | ✅ Complete |
| 2 | Vectorized execution engine — all operators | ✅ Complete |
| 3 | SQL parser — lexer, AST, recursive-descent | ✅ Complete |
| 4 | Catalog + planner — logical plan, optimizer, physical plan | ✅ Complete |
| 5 | CLI binary (`vexq`, `vexqgen`, `fsck`) | ✅ Complete |
| 6 | TPC-H benchmark harness vs SQLite | ✅ Complete |
| 7 | Morsel-driven parallelism — `ParallelHashAggregate`, `planner.Parallel()` | ✅ Complete |
| 8 | DuckDB honest baseline + per-core gap decomposition | ✅ Complete |
| 9 | Multi-table joins — qualified columns, alias resolution, cross-join rejection | ✅ Complete |
| 10 | CLI multi-file queries + `--workers` parallel flag | ✅ Complete |
| 11 | SIMD filter kernel benchmark — AVX2 ceiling measurement ([`bench/simd_filter/`](bench/simd_filter/)) | ✅ Complete |
| 12 | Parallel scaling — decode-buffer reuse, sort-peeling, GC diagnosis | ✅ Complete |
| 13 | Aggregate optimization — packed dictionary-code integer keys | ✅ Complete |
| 14 | Correctness oracle — 48-query golden test suite ([`internal/goldentest/`](internal/goldentest/)) | ✅ Complete |
| 15 | Expression eval hardening — NOT precedence, date coercion, CASE WHEN strings, COUNT(DISTINCT) | ✅ Complete |

## Design Notes

**Why pull-based (Volcano model)?** `LIMIT` and short-circuit predicates terminate naturally — when the root stops calling `Next()`, all upstream work stops with no extra machinery. Simpler to debug single-threaded, and composes cleanly with the morsel-driven parallel layer above it.

**How morsel-driven parallelism works.** Two distinct layers handle plan-shape detection and dynamic scheduling:

- `planner.Parallel()` ([`planner/parallel.go`](planner/parallel.go)) detects aggregate plan shapes (including `Sort → Aggregate` and `Limit → Sort → Aggregate`), resolves group-by/aggregate column indices, and builds a `PipelineFactory` closure (one independent `TableScan → Filter → Project` pipeline per call). Sort/limit nodes above the aggregate are peeled off and applied serially to the small merged result.
- `ParallelHashAggregate.setup()` ([`exec/parallel.go`](exec/parallel.go)) handles *dynamic scheduling*: a `morselQueue` wraps an `atomic.Int64` cursor padded to its own 64-byte cache line (preventing false-sharing). Each of the `numWorkers` goroutines loops calling `q.claim(morselSize)` to atomically reserve the next chunk of row groups, runs the pipeline on that morsel, accumulates into a goroutine-local `HashAggregate`, and claims the next morsel — no coordinator, no barriers between morsels. Fast workers (low-selectivity morsels) complete early and immediately claim more work, eliminating stragglers. After all workers drain the queue, the calling goroutine merges all partial aggregates single-threadedly (correct IEEE float64 via bit-reencoding). `TableScan.Reset(rgStart, rgEnd)` allows a single open `storage.Reader` to be repositioned without reopening the file, enabling pipeline reuse across morsels within a single worker.

**Why 1024-row batches?** An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB, fitting comfortably in L1 (32–48 KB on x86, 192 KB on Apple M4 Pro performance cores). Per-batch overhead (one `Next()` call, type assertions) amortizes over 1024 rows. Same constant used by Velox, DuckDB, and Photon.

**Why selection vectors instead of filtered batches?** `Filter` writes a `[]uint16` of surviving row indices rather than allocating new vectors. Downstream operators index through the selection vector, saving allocation on the hot path and preserving the 1024-row invariant across the pipeline.

**Why little-endian?** The inner loop of every `TableScan` is `binary.LittleEndian.Uint64(buf[i*8:])` — on x86-64 this compiles to a single `MOVQ`; on ARM64 (M4 Pro) a single `LDR`. Big-endian forces an extra byte-reverse (`BSWAP`/`REV`), penalizing column reads by ~10–20%.
