# vexq

[![CI](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml/badge.svg)](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml)

Built a vectorized analytical SQL execution engine in Go using columnar storage, dictionary encoding, zone maps, and morsel-driven parallel execution; implemented vectorized Volcano operators and a custom query parsing pipeline.

## Overview

vexq is a complete analytical query engine — from a custom on-disk columnar file format through a SQL parser, rule-based optimizer, and vectorized execution engine. The design shares structural principles with [distrikv](https://github.com/ryderpongracic1/distrikv) (append-only manifest, block CRCs, atomic writes via temp+rename) while adding a columnar-vectorized execution model suited for OLAP workloads.

On TPC-H Q1 (GROUP BY aggregate over 6M rows), vexq end-to-end (all cores) is faster than DuckDB end-to-end on the same machine (22 ms vs 25 ms), and within 1.2× single-threaded — the result of dictionary-code integer-keyed aggregation, coarse-grained I/O, join column pruning, a radix-partitioned parallel join, and a systematic allocation-elimination campaign verified by GOGC ablation. vexq is faster than SQLite on all four benchmarked TPC-H queries (3.0–18.4×), and its morsel-driven scheduler reaches 8.6× scaling on 14 cores — ~86% efficiency against the 10-performance-core ceiling, exceeding DuckDB's own scaling on every benchmarked query.

The engine processes data in batches of 1024 rows, designed to saturate L1 cache and amortize operator dispatch overhead. A pushed-down predicate + zone-map pruning layer skips entire row groups before any I/O, and dictionary encoding reduces string comparisons to integer equality in the filter hot loop.

## Architecture

```
cmd/vexq       — CLI: query execution + fsck integrity check
cmd/vexqgen    — TPC-H .tbl → .vxq converter

sql/           — Lexer + recursive-descent parser (Pratt precedence)
planner/       — Logical plan builder, rule-based optimizer, physical planner
  optimizer    — Predicate pushdown, column pruning (through joins as well as scans)
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
  ParallelHashJoinAggregate — Radix-partitioned parallel build + morsel-parallel probe, feeding a partial aggregate
  Limit

catalog/       — Table registry with lazy schema loading from .vxq footer
storage/       — .vxq file format: writer, reader, block codec, zone maps
internal/encoding — Little-endian primitives, CRC32-IEEE helpers
internal/goldentest — End-to-end correctness oracle (72-query suite, 4 execution paths)
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

TPC-H scale factor 1 (6M lineitem rows) on Apple M4 Pro (14-core, 192 KB L1D per performance core, 36 MB SLC). Page cache warm. Each benchmark run 10×; numbers are median wall time. All ratios are computed from unrounded medians, so a ratio may differ slightly from one recomputed from the rounded milliseconds shown. The headline results: on Q1, vexq parallel (22 ms) is faster end-to-end than DuckDB all-cores (25 ms) on this machine, and within 1.2× of DuckDB single-threaded per core; vexq is faster than SQLite on all four queries (3.0–18.4×); see the DuckDB decomposition below.

**Query provenance:** The harness runs the canonical TPC-H Q6 (`SUM(l_extendedprice * l_discount)`, inclusive `BETWEEN`). Earlier revisions used a simplified variant (`SUM(l_extendedprice)`) because aggregates over expressions crashed the planner; that bug is fixed (plan-time type coercion + a column-pruning fix), and both engines in every table below now run identical query text. vexq's canonical Q6 result (123,141,078.23) matches SQLite (asserted in-harness to 1e-9 relative tolerance) and DuckDB's canonical result (123141078.2283) exactly. Where a table below reports the *simplified* variant for like-for-like comparison across revisions, it is labeled as such.

### Correctness

All four TPC-H query results verified identical to SQLite output (via the in-harness `TestQ*Correctness` assertions; Q6's SUM is asserted to 1e-9 relative tolerance). vexq's canonical Q6 additionally matches DuckDB's result exactly. Additionally, an independent 72-query golden test suite ([`internal/goldentest/`](internal/goldentest/)) verifies the full SQL subset against a naive row-at-a-time reference evaluator — and every query runs through **both** `planner.Physical()` and `planner.Parallel(…, 4)`, so the radix-partitioned parallel build, parallel probe, and per-morsel expression materialization are all verified against the same oracle (under `-race` in CI). 72/72 passing on all four oracle paths (serial, parallel, optimizer-off, and a stacked-filter path that duplicates pushed-down predicates to cover plan shapes SQL alone cannot reach), zero known correctness issues.

### Ceiling — vs DuckDB (SOTA embedded OLAP)

DuckDB is a state-of-the-art embedded analytical engine: SIMD intrinsics, a cost-based optimizer, radix-partitioned hash aggregation, adaptive parallel morsel scheduling, LLVM JIT compilation, and a columnar execution engine written in C++.

The table below decomposes the gap into **per-core execution efficiency** (single-threaded DuckDB vs vexq serial) and **end-to-end throughput** (DuckDB all-cores vs vexq parallel — both engines using all 14 cores). This decomposition separates SIMD/JIT advantages from parallelism, which is the more informative comparison for a from-scratch engine.

Note: this comparison runs DuckDB on ARM64, where its strongest x86 SIMD paths (AVX-512) are unavailable; an x86 comparison would likely widen DuckDB's per-core advantage. The comparison is same-machine and therefore fair, but the Q1 figures should be read with that context.

| Query | vexq serial | vexq parallel | DuckDB (1 thread) | DuckDB (14 threads) | Per-core gap | End-to-end gap |
|-------|------------:|--------------:|------------------:|--------------------:|:------------:|:--------------:|
| Q1 | 193 ms | 22 ms | 159 ms | 25 ms | **1.2×** | **0.9× (vexq faster)** |
| Q6 | 67 ms | 10 ms | 37 ms | 8 ms | **1.8×** | **1.3×** |
| Q3 | 137 ms | 36 ms | 62 ms | 17 ms | **2.2×** | **2.1×** |
| Q12 | 238 ms | 36 ms | 81 ms | 16 ms | **2.9×** | **2.2×** |

**What the decomposition reveals:**

- **Q1 (1.2× per-core, 0.9× end-to-end):** vexq parallel Q1 is now faster than DuckDB all-cores on the same machine and query — the first crossover. The ingredients: dictionary-code integer-keyed aggregation (packed `uint64` group keys, never a string in the hot loop), coarse-grained I/O, allocation-free accumulation, and 8.6× parallel scaling against DuckDB's 6.4×. The remaining per-core gap is DuckDB's radix-partitioned hash aggregate + SIMD horizontal SUM. One caveat repeated from above: DuckDB on ARM64 lacks its strongest x86 SIMD paths.

- **Q6 (1.8× per-core, 1.3× end-to-end):** An earlier pprof profile showed ~59% of serial Q6 CPU in `pread` syscalls (~35K block-granular preads per 6M-row scan). Row-group-buffered reads cut that to ~373 preads (62.9×), and the allocation campaign (expression scratch buffers, rowSet accumulation) took serial Q6 from 132 to 67 ms across two rounds. The remaining per-core gap is genuinely compute: vectorized predicate evaluation (the measured AVX2 kernel ceiling is 3.3× on the filter kernel alone) and decode.

- **Q3 (2.2× per-core) / Q12 (2.9× per-core):** These gaps were 11× and 13× three rounds ago. Join column pruning took serial Q3 684 → 188 ms; the flat open-addressed join table (one 16-byte slot per key, duplicate rows chained through a row-major store, no per-key allocation), the dictionary string memo, and rowStore presizing took Q3 to 137 ms and Q12 from 585 to 238 ms in the following round. vexq parallel Q3 (36 ms) is now well below DuckDB single-threaded (62 ms). The remaining gap is DuckDB's SIMD-probed, cache-resident join: partitioning only the build side does not make probes cache-resident (measured flat from 1 to 256 partitions) — closing it needs the **probe** stream partitioned as well. Q12's larger gap adds the CASE WHEN aggregate that DuckDB JIT-compiles.

- **Parallelism gap (closed):** DuckDB achieves 3.6–6.4× scaling from threads=1 to threads=14 (from the table above: Q1 6.4×, Q6 4.6×, Q12 5.1×, Q3 3.6×). vexq now achieves 8.6× (Q1), 6.7× (Q12), 6.5× (Q6), and 3.8× (Q3) — exceeding DuckDB's scaling on all four queries (Q3: 3.8× vs 3.65×). All four take the parallel path; the GOGC ablation below shows GC — formerly 26–50% of parallel wall time — is now 0–19% after the allocation campaign.

### Hardware migration and normalization

The benchmarks moved from Apple M1 Pro (10-core) to Apple M4 Pro (14-core). Using SQLite as a platform-neutral control (same workload, no code changes between runs): Q1 3,463→2,197 ms (1.58×), Q6 599→408 ms (1.47×), Q3 3,719→2,525 ms (1.47×), Q12 1,073→724 ms (1.48×). The hardware is worth roughly 1.5×. Normalizing vexq's improvements against this:

The Q6 rows in this normalization use the simplified variant on both machines (the canonical query did not run on the M1), so the comparison is like-for-like. This normalization is approximate: SQLite is a pointer-chasing B-tree row-store dominated by memory latency, while vexq is a streaming columnar scanner — the two do not scale identically across microarchitectures, so the ~1.5× hardware factor is a rough correction rather than an exact one.

| Query | Raw improvement | Hardware factor | Software-only improvement |
|-------|----------------|-----------------|---------------------------|
| Q1 | 2.75× (657→239) | ÷1.58 | **1.74×** (aggregate intkey rework + decode-buffer reuse; both landed in this window; credit not separable without an ablation) |
| Q6 | 1.78× (198→111) | ÷1.47 | **1.21×** (decode-buffer reuse + aggregate intkey rework; both landed in this window; credit not separable without an ablation) |
| Q3 | 1.66× (1133→684) | ÷1.47 | **1.13×** (minor — no join changes) |
| Q12 | 1.56× (1643→1050) | ÷1.48 | **1.06×** (no targeted optimization) |

The Q1 improvement is almost entirely software (the dictionary-code integer-key aggregate rework targeted precisely this query). Q3/Q12 improvements are almost entirely hardware — no join optimizations were made.

### Second optimization round (same hardware — no normalization needed)

A later round landed coarse-grained I/O (row-group-buffered reads, 62.9× fewer preads), join column pruning, a radix-partitioned parallel join build, and parallel expression aggregates. Both measurements below are from the same M4 Pro, so the deltas are pure software:

| Query | Serial before | Serial after | Software improvement | Dominant cause |
|-------|--------------:|-------------:|:--------------------:|----------------|
| Q1 | 239 ms | 205 ms | **1.17×** | coarse-grained I/O |
| Q6 (canonical) | 132 ms | 93 ms | **1.42×** | coarse-grained I/O (pread was ~59% of the profile). 132 ms is the canonical query; the 111 ms in the normalization table above is the labeled simplified variant |
| Q3 | 684 ms | 188 ms | **3.6×** | join column pruning (customer 8→2, orders 9→4, lineitem 16→4 columns) + coarse I/O |
| Q12 | 1,050 ms | 585 ms | **1.8×** | join column pruning + coarse I/O |

### Third optimization round — the allocation campaign (same hardware)

The next round attacked allocation systematically, profile-first: per-instance expression scratch buffers, a flat open-addressed join table over a row-major store, a two-tier storage window pool, per-worker pipeline reuse (one Reader per worker instead of ~105 per query), `rowSet` aggregation (the per-batch `[]int` materialization removed outright), a lazy dictionary string memo (190× fewer objects), explicit rowStore growth + presizing from footer row counts. Same M4 Pro, pure software:

| Query | Serial before | Serial after | Improvement | Parallel before | Parallel after | Improvement |
|-------|--------------:|-------------:|:-----------:|----------------:|---------------:|:-----------:|
| Q1 | 205 ms | 193 ms | 1.06× | 35 ms | **22 ms** | **1.56×** |
| Q6 | 93 ms | 67 ms | **1.39×** | 34 ms | **10 ms** | **3.3×** |
| Q3 | 188 ms | 137 ms | **1.37×** | 62 ms | **36 ms** | **1.73×** |
| Q12 | 585 ms | 238 ms | **2.46×** | 113 ms | **36 ms** | **3.2×** |

The parallel column moved 1.6–3.3× in one round because allocation was the parallel path's noise floor: every worker's garbage fed one shared GC. The ablation below is the proof it worked.

### Floor — vs SQLite (row-store OLTP)

SQLite is a B-tree row-store engine designed for OLTP: it reads full rows, applies predicates row-at-a-time, and has no columnar I/O or vectorized aggregation. Beating a row-store on full-table OLAP scans is the **expected** outcome of any columnar engine — this baseline confirms the columnar layout and vectorized operators are paying off, not that the engine is production-grade.

| Query | Description | vexq serial | SQLite | Speedup |
|-------|-------------|------------:|-------:|:-------:|
| Q1 | Pricing summary — full scan, GROUP BY 2 string cols | 193 ms | 2,197 ms | **11.4×** |
| Q6 | Revenue forecast — scan with 5 range predicates, SUM(expr) | 67 ms | 381 ms | **5.7×** |
| Q3 | Shipping priority — 3-table join, complex SUM, LIMIT 10 | 137 ms | 2,525 ms | **18.4×** |
| Q12 | Shipping modes — 2-table join, CASE WHEN agg, date comparisons | 238 ms | 724 ms | **3.0×** |

vexq is now faster than SQLite on all four queries, serially. Q12 was the last holdout (0.69× before join column pruning): the `HashJoin` build phase materialized all nine orders columns while SQLite walked its B-tree index on `o_orderkey`; pruning the build side to the referenced columns flipped it.

### What's left on the table

- **Coarser-grained I/O (landed)**: `ColumnReader` now buffers a window of its column section (up to 4 MiB, admitting a whole row group) and serves blocks from memory: 23,449 → 373 preads per canonical-Q6 scan (62.9×). Measured on the M4 Pro: serial Q6 132 → 93 ms, and it contributes to every other query's second-round improvement. mmap was considered and rejected — it needs `golang.org/x/sys`, breaking the zero-runtime-dependency claim.
- **Explicit SIMD**: Use `avo` or Go assembly to generate AVX2/AVX-512 kernels for the hot decode and comparison loops. With the pread bottleneck now removed by coarse-grained I/O, filter/decode compute is a much larger fraction of serial Q6 — but that fraction has not been re-profiled since the I/O change, so the Amdahl ceiling on this work is currently unknown; measure it before committing. There is also an ISA decision to confront: the benchmark machine is ARM64 (NEON), while the measured kernel and the narrative so far are x86/AVX2 — in-engine SIMD means NEON via Go assembly, or moving engine benchmarking to x86. SIMD remains valuable for Q1-style queries where aggregation compute dominates. See [`bench/simd_filter/`](bench/simd_filter/) for the kernel measurement: **AVX2 int64 intrinsics achieve 2.44 ns/row vs 8.17 ns/row (3.3× speedup)**, measured on Intel Xeon 6975P-C (x86-64).
- **Parallel hash join — radix-partitioned parallel build (landed)**: Three increments, all measured on a synthetic 300K-order ⋈ 1.2M-lineitem dataset (Intel Xeon 6975P-C, 4 cores available, warm page cache, medians of 5 runs). Phase 1: `planner.Parallel()` detects `Aggregate → HashJoin → (Filter →)? Scan` and parallelizes the **probe** side (`exec.ParallelHashJoinAggregate`), bringing Q3's and Q12's shapes into the parallel path for the first time. Phase 2: the optimizer pushes needed-column sets through `LogicalJoin`, so each side decodes only its join key plus the columns the query references — that halved the isolated build phase and cut the build's share of serial Q12-shaped runtime from ~60% to ~47%. Phase 3: the build side is now **radix-partitioned and built in parallel** over its own row-group morsels — workers bucket rows by mixed-hash radix bits into per-(morsel, partition) buckets, then one goroutine per partition assembles that partition's map, so the build needs no lock. Walking morsels in index order during assembly keeps per-key row order identical to a serial drain, so integer aggregates stay exactly equal to serial results, not merely equivalent. Q12-shaped parallel(4) went 191 ms → 113 ms and its speedup over serial 1.15× → **2.01×** (that this synthetic-fixture figure coincides with the M4 Pro TPC-H Q12 parallel time in the table below is exactly that — a coincidence across different machines, datasets, and core counts); Q3-shaped parallel(4) 63 ms → 52 ms (1.22× → 1.51×); a probe-heavy unfiltered join 203 ms → 149 ms (1.45× → 2.03×); the isolated build phase 114 ms → 54 ms, which had actually been a **regression** against its own 99 ms serial build before this change and is now 1.96× faster than it. Under `GOGC=off`, Q12-shaped parallel scaling is 1.32× → **2.74×** on 4 cores, so GC still absorbs a large share of the win. Two honest costs: allocated bytes rise ~34% for the two-pass shuffle's buckets, and build morsels are row groups, so a build side of only a few row groups leaves a straggler (the 300K-row fixture is 5 row groups, and its in-query build scales 1.59× on 4 workers rather than near-linearly).
- **Radix partitioning did not make the probe cache-resident**: this was the stated motivation and the measurement did not support it. `exec.BenchmarkPartitionedProbe` holds the build rows and probe keys fixed and varies only the partition count: 1 → 256 partitions is flat within noise (4.1–4.7 ms). Only the build side is partitioned, so a probe batch's keys still spread across every partition and the probe touches the whole table per unit time either way. What partitioning does buy is the lock-free parallel build, plus 1.58× on a single-threaded build purely from growing 64 small maps instead of one big one (`exec.BenchmarkRadixBuild`). Getting the cache win needs the **probe** stream partitioned too — buffering probe rows per partition at morsel granularity and then probing partition-at-a-time — which conflicts with a streaming operator whose `TableScan` reuses its decode buffers between batches. That is the remaining structural piece of DuckDB's join advantage, and it is not claimed here. Before attempting it, the premise (that the probe is cache-miss-bound) should itself be measured — L2/SLC miss rates on the Q3/Q12 probe via `perf stat`/Instruments — since Q3's filtered customer side (~30K rows) certainly fits in a 36 MB SLC and cannot benefit.
- **Column pruning through joins is name-based, not cost-based**: a column name present in both joined tables is kept on both sides, since needed-column sets carry unqualified names. That over-approximates rather than under-prunes, so it is correct but leaves a little on the table for schemas that reuse column names across tables. A scan also cannot express "read zero columns" — `SELECT COUNT(*) FROM t` over a single table still decodes every column, because an empty projection means "all" in `exec.NewTableScan`. Under a join the join key saves it, but the single-table case would need a row-count-only scan path.
- **Late materialization**: Avoid decoding non-predicate columns until after the filter selection vector is built. Q6's TPC-H filter selectivity is ~2%, so payload decode for surviving rows only is a large reduction in decode work on the best-understood remaining gap — and less decoding means less allocation, compounding with buffer reuse rather than fighting it.
- **Buffer reuse / allocation campaign (landed)**: per-instance expression scratch buffers under a written aliasing contract, `rowSet` aggregation (the per-batch `[]int` gone from the profile entirely), a two-tier storage window pool, per-worker pipeline reuse, dictionary string memoisation, and rowStore growth + presizing. Combined effect certified by ablation: GC's share of parallel wall time fell from 26–50% to 0–19%.
- **Flat open-addressed join hash table (landed)**: 16-byte slots (key, head row, tail row — four per cache line), duplicate keys chained through a row-major `rowStore` in insertion order, exact presizing from radix pass-1 counts, high hash bits for slot index so partition and slot bits never collide. Probe-heavy shapes gained 2.6× serial / 3.2× parallel on the fixtures; build-phase allocations fell from 1.2M to 1.5K per op.
- **Adaptive compression**: Delta encoding for sorted integer columns (timestamps, order keys) could improve decode throughput and reduce I/O.

### Parallel execution (morsel-driven, 14 goroutines)

`planner.Parallel()` detects aggregate plan shapes — including `Sort → Aggregate`, `Limit → Sort → Aggregate`, and aggregates over an inner hash join — and partitions the scan across `runtime.NumCPU()` goroutines using a dynamic atomic-counter morsel queue. Each goroutine runs an independent pipeline on dynamically claimed morsels; a merge step combines partial aggregates in the calling goroutine. For join shapes, the build side is radix-partitioned and built in parallel first, then probe-side morsels are distributed across workers. All four TPC-H queries now take the parallel path.

| Query | vexq serial | vexq parallel | Speedup | Notes |
|-------|------------:|--------------:|:-------:|-------|
| Q1 | 193 ms | **22 ms** | **8.6×** | Sort-peeling: parallel aggregate, serial 4-row sort |
| Q6 | 67 ms | **10 ms** | **6.5×** | Expression aggregate materialized per morsel |
| Q3 | 137 ms | **36 ms** | **3.8×** | Radix-partitioned parallel build + parallel probe (3-table chain) |
| Q12 | 238 ms | **36 ms** | **6.7×** | Parallel join + CASE WHEN expression aggregate |

Q1's 8.6× on 14 cores (10P+4E) is ~86% efficiency against the realistic 10-P-core ceiling — a long way from the 1.16–1.85× this table showed before the buffer-reuse and allocation campaigns, sort-peeling, parallel expression aggregates, and the parallel join landed.

Expression aggregates parallelize by ending each worker pipeline with the same pre-projection the serial planner applies, materializing the expression (`SUM(a*b)`) into a synthetic column per morsel — the expression is row-local, so evaluating it per morsel is equivalent to evaluating it over the whole scan. Float64 `SUM`/`AVG` results agree with serial execution to within IEEE-754 rounding rather than bit-for-bit — partitioning reorders float additions, which is a property of any partitioned float reduction; integer aggregates and `COUNT` are exact (the parallel join's morsel-ordered build assembly keeps per-key row order identical to a serial drain). `COUNT(DISTINCT)` still falls back to serial. Remaining parallel scaling is limited primarily by P/E-core asymmetry, with GC now a minor factor (0–19%, ablation below) after the allocation campaign.

The `GOGC=off` ablation — now with both denominators measured — tells the allocation campaign's before/after story. Before the campaign, disabling GC cut parallel wall time 26–50%. After it, the same ablation moves Q1 by ~0%, Q3 ~8%, Q12 ~15%, and Q6 ~19%: the GC bottleneck was engineered away, and the ablation that diagnosed it certifies the fix. True like-for-like scaling (parallel GOGC=off ÷ serial GOGC=off) is Q1 8.8×, Q6 7.9×, Q12 7.8×, Q3 4.4× — 78–88% efficiency against the 10-P-core ceiling for three of four queries. The remaining gap to linear is P/E-core asymmetry: the M4 Pro's 14 cores are 10 performance + 4 efficiency cores, and runtime.NumCPU() spawns 14 workers, four of which land on E-cores — the worker sweep below demonstrates the bend at 10 workers empirically. It is explicitly not memory bandwidth.

| Query | Serial (GC on) | Serial (GOGC=off) | Parallel (GC on) | Parallel (GOGC=off) | GC share of parallel | True scaling (off/off) |
|-------|---------------:|------------------:|-----------------:|--------------------:|:--------------------:|:----------------------:|
| Q1 | 193 ms | 198 ms | 22 ms | 22 ms | **~0%** (was ~26%) | **8.8×** |
| Q6 | 67 ms | 66 ms | 10 ms | 8 ms | ~19% (was ~44%) | **7.9×** |
| Q3 | 137 ms | 146 ms | 36 ms | 33 ms | ~8% (was ~31%) | **4.4×** |
| Q12 | 238 ms | 234 ms | 36 ms | 30 ms | ~15% (was ~50%) | **7.8×** |

Serial GOGC=off is now flat-to-slightly-slower than GC on (Q1 +3%, Q3 +7%) — the signature of an engine that no longer allocates enough for GC to matter, paying heap-growth overhead with nothing to amortize it against.

GOGC=off is a diagnostic, not a production configuration — unbounded heap growth is not viable. The realistic mitigation was the allocation campaign itself (Phase 20), now landed across scan, filter, project, aggregate, join, and storage; GOMEMLIMIT remains available as a bounded middle ground for whatever residual pressure appears under new workloads.

#### Worker sweep (GOGC=off, Q6-shaped simple-aggregate query)

Measured on an **earlier engine revision** (before coarse-grained I/O and parallel expression aggregates landed), via CLI with warm page cache; times include ~10 ms of process startup + catalog overhead. The absolute times are therefore not comparable to the in-process tables above — only the sweep's internal ratios are meaningful, and what they demonstrate (the scaling bend at the 10-P-core boundary) is a hardware property independent of the engine revision.

| Workers | Time (ms) | Speedup vs 1 |
|--------:|----------:|:------------:|
| 1 | 131 | 1.0× |
| 2 | 90 | 1.5× |
| 4 | 65 | 2.0× |
| 8 | 48 | 2.7× |
| 10 | 47 | 2.8× |
| 12 | 45 | 2.9× |
| 14 | 44 | 3.0× |

Near-linear benefit through 8 workers, then flat — 8→14 buys only 8%. The bend at the P-core boundary (10 workers) empirically confirms the P/E-core asymmetry explanation: the M4 Pro's 10 performance cores contribute meaningful throughput; the 4 efficiency cores add ~5%. A runtime improvement: cap default workers at the performance-core count rather than `runtime.NumCPU()`.

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
| 14 | Correctness oracle — golden test suite (now 72 queries, 4 oracle paths) ([`internal/goldentest/`](internal/goldentest/)) | ✅ Complete |
| 15 | Expression eval hardening — NOT precedence, date coercion, CASE WHEN strings, COUNT(DISTINCT) | ✅ Complete |
| 16 | Coarse-grained I/O — row-group-buffered reads, 62.9× pread reduction | ✅ Complete |
| 17 | Parallel expression aggregates + parallel hash join (probe side) | ✅ Complete |
| 18 | Join column pruning — needed-column sets pushed through `LogicalJoin` | ✅ Complete |
| 19 | Radix-partitioned parallel join build — lock-free two-pass, 64-partition measured optimum | ✅ Complete |
| 20 | Allocation campaign — scratch buffers, flat join table, window pool, pipeline reuse, rowSet, dict memo, presizing | ✅ Complete |
| 21 | Correctness hardening — stacked-filter physical-length convention; string/date/bool aggregates; oracle to 72 queries / 4 paths | ✅ Complete |

## Design Notes

**Why pull-based (Volcano model)?** `LIMIT` and short-circuit predicates terminate naturally — when the root stops calling `Next()`, all upstream work stops with no extra machinery. Simpler to debug single-threaded, and composes cleanly with the morsel-driven parallel layer above it.

**How morsel-driven parallelism works.** Two distinct layers handle plan-shape detection and dynamic scheduling:

- `planner.Parallel()` ([`planner/parallel.go`](planner/parallel.go)) detects aggregate plan shapes (including `Sort → Aggregate` and `Limit → Sort → Aggregate`), resolves group-by/aggregate column indices, and builds a `PipelineFactory` closure (one independent `TableScan → Filter → Project` pipeline per call). Sort/limit nodes above the aggregate are peeled off and applied serially to the small merged result.
- `ParallelHashAggregate.setup()` ([`exec/parallel.go`](exec/parallel.go)) handles *dynamic scheduling*: a `morselQueue` wraps an `atomic.Int64` cursor padded to its own 64-byte cache line (preventing false-sharing). Each of the `numWorkers` goroutines loops calling `q.claim(morselSize)` to atomically reserve the next chunk of row groups, runs the pipeline on that morsel, accumulates into a goroutine-local `HashAggregate`, and claims the next morsel — no coordinator, no barriers between morsels. Fast workers (low-selectivity morsels) complete early and immediately claim more work, eliminating stragglers. After all workers drain the queue, the calling goroutine merges all partial aggregates single-threadedly (correct IEEE float64 via bit-reencoding). Each worker builds its pipeline **once** and repositions it with `TableScan.Reset(rgStart, rgEnd)` for each morsel it claims, so a query opens one `storage.Reader` per worker rather than one per morsel — before that, a 92-row-group scan re-read and re-parsed the footer ~112 times per query, which made `storage.Open` the second-largest allocation site in a parallel aggregate. Reuse is opt-in per pipeline shape: `exec.MorselPipeline` ([`exec/morsel.go`](exec/morsel.go)) names the operators that can be repositioned (scan, filter, projection, and a join probing a shared build table), and any pipeline containing anything else falls back to being rebuilt per morsel rather than being partially reset.

**Why 1024-row batches?** An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB, fitting comfortably in L1 (32–48 KB on x86, 192 KB on Apple M4 Pro performance cores). Per-batch overhead (one `Next()` call, type assertions) amortizes over 1024 rows. Same constant used by Velox, DuckDB, and Photon.

**Why selection vectors instead of filtered batches?** `Filter` writes a `[]uint16` of surviving row indices rather than allocating new vectors. Downstream operators index through the selection vector, saving allocation on the hot path and preserving the 1024-row invariant across the pipeline.

**Why little-endian?** The inner loop of every `TableScan` is `binary.LittleEndian.Uint64(buf[i*8:])` — on x86-64 this compiles to a single `MOVQ`; on ARM64 (M4 Pro) a single `LDR`. Big-endian forces an extra byte-reverse (`BSWAP`/`REV`), penalizing column reads by ~10–20%.
