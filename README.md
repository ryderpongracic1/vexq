# vexq

Built a vectorized analytical SQL execution engine in Go using columnar storage, dictionary encoding, zone maps, and morsel-driven parallel execution; implemented vectorized Volcano operators and a custom query parsing pipeline.

## Overview

vexq is a complete analytical query engine — from a custom on-disk columnar file format through a SQL parser, rule-based optimizer, and vectorized execution engine. The design shares structural principles with [distrikv](https://github.com/ryderpongracic1/distrikv) (append-only manifest, block CRCs, atomic writes via temp+rename) while adding a columnar-vectorized execution model suited for OLAP workloads.

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
bench/tpch     — TPC-H Q1/Q3/Q6/Q12 benchmarks vs SQLite and DuckDB
```

## Hardware-Level Architecture

### Why 1024-row batches

An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB — well within L1 cache on every modern microarchitecture (32–48 KB on x86 Skylake+, **128 KB** on Apple M1 Pro/Max performance cores). This means the innermost decode, filter, and aggregate loops all operate on data that is almost certainly already in L1, avoiding LLC round-trips that cost ~40 cycles each.

The batch size also amortizes the cost of one virtual `Next()` call across 1024 rows. A virtual dispatch on x86 is ~50 ns (indirect branch, possible iTLB miss); spread over 1024 rows that is ~0.05 ns per row — negligible. Processing row-at-a-time would spend more time dispatching than computing. This is the same constant used by Velox, DuckDB, and Photon.

### Cache-line discipline (64 bytes)

The `TableScan` decode loops in [`exec/scan.go`](exec/scan.go) operate on tightly-packed little-endian payloads. For `INT64` columns, each iteration of the hot loop is:

```go
vals[i] = int64(binary.LittleEndian.Uint64(payload[i*8:]))
```

On x86-64, `binary.LittleEndian.Uint64` compiles to a single `MOVQ` (no byte-swap). On ARM64 (M1 Pro), it compiles to a single `LDR`. Eight consecutive values occupy exactly 64 bytes — one cache line — so the hardware prefetcher can issue fetch requests ahead of the loop at full bandwidth. Big-endian would require a `MOVQ+BSWAP` (x86) or `LDR+REV` (ARM64) pair, penalizing every decode by ~10–20%.

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

`planner.Parallel()` ([`planner/parallel.go`](planner/parallel.go)) detects a `LogicalAggregate → (Filter →)? Scan` plan shape and builds a `ParallelHashAggregate` ([`exec/parallel.go`](exec/parallel.go)).

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
FROM table
[WHERE condition]
[GROUP BY col, ...]
[ORDER BY col [ASC|DESC], ...]
[LIMIT n]

-- Aggregate functions: COUNT(*), COUNT(col), SUM, AVG, MIN, MAX
-- Predicates: =, <>, <, <=, >, >=, AND, OR, NOT, BETWEEN, IN, LIKE, IS NULL
-- Expressions: arithmetic (+, -, *, /), CASE WHEN, unary minus
```

## Usage

```bash
# Convert a TPC-H table to .vxq
vexqgen lineitem lineitem.tbl lineitem.vxq

# Run a query
vexq lineitem.vxq "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag"

# Validate file integrity (CRC, footer, zone maps)
vexq fsck lineitem.vxq
```

## Build

```bash
go build ./cmd/vexq/
go build ./cmd/vexqgen/
go test ./... -race -count=1
```

Requires Go 1.21+. No external runtime dependencies (SQLite and DuckDB are benchmark-only).

## Benchmarks

TPC-H scale factor 1 (6M lineitem rows) on Apple M1 Pro (10-core, 128 KB L1D per performance core, 24 MB SLC). Each benchmark run 3×; numbers are median wall time per run.

### Floor — vs SQLite (row-store OLTP)

SQLite is a B-tree row-store engine designed for OLTP: it reads full rows, applies predicates row-at-a-time, and has no columnar I/O or vectorized aggregation. Beating a row-store on full-table OLAP scans is the **expected** outcome of any columnar engine — this baseline confirms the columnar layout and vectorized operators are paying off, not that the engine is production-grade. SQLite is configured with `WAL`, `NORMAL` sync, 256 MB cache, and `ANALYZE`.

| Query | Description | vexq | SQLite | Speedup |
|-------|-------------|------|--------|---------|
| Q1 | Pricing summary — full scan, GROUP BY 2 string cols | 733 ms | 3,320 ms | **4.5×** |
| Q6 | Revenue forecast — scan with 5 range predicates, SUM | 473 ms | 583 ms | **1.2×** |
| Q3 | Shipping priority — 3-table join, complex SUM, LIMIT 10 | 1,218 ms | 3,764 ms | **3.1×** |
| Q12 | Shipping modes — 2-table join, CASE WHEN agg, date comparisons | 1,903 ms | 1,130 ms | 0.6× |

Q12 is currently slower than SQLite: the `HashJoin` build phase materialises the full orders table and SQLite benefits from its B-tree index on `o_orderkey`. Future work: index-nested-loop join and late materialisation would close this gap.

### Ceiling — vs DuckDB (SOTA embedded OLAP)

DuckDB is a state-of-the-art embedded analytical engine backed by multi-decade research: SIMD intrinsics (AVX-512 horizontal aggregation), a cost-based optimizer, radix-partitioned hash aggregation, adaptive parallel morsel scheduling, and a columnar execution engine written in C++. These results represent the honest performance gap between a from-scratch research engine in Go and a production-grade OLAP system.

| Query | vexq | DuckDB | Gap |
|-------|------|--------|-----|
| Q1 | 733 ms | — | — |
| Q6 | 473 ms | — | — |
| Q3 | 1,218 ms | — | — |
| Q12 | 1,903 ms | — | — |

*DuckDB baseline will be populated once Phase 2 benchmarks are complete.*

**Why we lose (and by how much):**
- **Q1 (GROUP BY heavy):** DuckDB uses AVX-512 horizontal SUM over 8 float64 lanes simultaneously; vexq accumulates one value per iteration. DuckDB's radix-partitioned hash aggregate also avoids the random-access hash map misses that dominate our `accumulate()` hot loop.
- **Q6 (predicate heavy):** DuckDB generates SIMD predicate masks that evaluate 8 comparisons per instruction; our `BoolVector` loop evaluates one comparison per iteration. The gap here should be smallest because zone-map pruning is already eliminating most row groups.
- **Q3/Q12 (join heavy):** DuckDB's hash join uses a SIMD-accelerated probe and a smarter build-side partitioning strategy. Our `HashJoin` is a straightforward in-memory build/probe.

### What's left on the table

- **Explicit SIMD**: Use `avo` or Go assembly to generate AVX2/AVX-512 kernels for the hot decode and comparison loops — likely 4–8× improvement on filter-heavy queries.
- **Parallel hash join**: Extend `planner.Parallel()` to detect join shapes and partition the build side.
- **Late materialization**: Avoid decoding non-predicate columns until after the filter selection vector is built — saves decode work proportional to filter selectivity.
- **Adaptive compression**: Delta encoding for sorted integer columns (timestamps, order keys) could improve decode throughput and reduce I/O.
- **Predicate-aware bloom filters**: Per-row-group bloom filters for high-cardinality string columns would improve the zone-map skipping rate beyond min/max.

### Parallel execution (morsel-driven, 10 goroutines)

`planner.Parallel()` partitions the file's row groups across `runtime.NumCPU()` goroutines using a dynamic atomic-counter morsel queue. Each goroutine runs an independent `TableScan → Filter → Project → partial HashAggregate` pipeline on dynamically claimed morsels; a merge step combines partial aggregates in the calling goroutine.

| Query | vexq serial | vexq parallel | SQLite | Speedup (parallel vs SQLite) |
|-------|------------|---------------|--------|------------------------------|
| Q6 | 473 ms | **233 ms** | 583 ms | **2.5×** |
| Q1† | 733 ms | 733 ms | 3,320 ms | 4.5× |

† Q1 has an `ORDER BY` clause, so the root operator is a `Sort`, not an `Aggregate`. `planner.Parallel()` falls back to `planner.Physical()` for plans it cannot partition (joins, sorts at the root). Parallel execution applies to aggregate-only plans today; Q3/Q12 also fall back because they contain `HashJoin`.

### Running benchmarks

```bash
# Generate TPC-H SF=1 data
cd data && ../tools/dbgen -s 1 -f

# Convert to .vxq
vexqgen lineitem  data/lineitem.tbl  data/lineitem.vxq
vexqgen orders    data/orders.tbl    data/orders.vxq
vexqgen customer  data/customer.tbl  data/customer.vxq

# Load SQLite baseline
go test ./bench/tpch/ -run TestSetupSQLite -v

# Load DuckDB baseline
go test ./bench/tpch/ -run TestSetupDuckDB -v

# Run all benchmarks (serial + parallel + DuckDB)
go test ./bench/tpch/ -bench=. -benchtime=3x -v

# Run just parallel benchmarks
go test ./bench/tpch/ \
  -bench="BenchmarkVexqQ1$|BenchmarkVexqQ1Parallel|BenchmarkVexqQ6$|BenchmarkVexqQ6Parallel" \
  -benchtime=3x -v
```

## Profiling

```bash
# CPU profile of Q1 — look for payloadToVector and accumulate in the flame graph
go test ./bench/tpch/ -bench=BenchmarkVexqQ1$ -benchtime=10x \
  -cpuprofile=cpu.out -benchmem
go tool pprof -http=:8080 cpu.out

# Allocation profile — look for hot-loop allocations in payloadToVector
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
| 8 | DuckDB honest baseline + hot-loop unrolling + work-stealing scheduler | 🔄 In Progress |

## Design Notes

**Why pull-based (Volcano model)?** `LIMIT` and short-circuit predicates terminate naturally — when the root stops calling `Next()`, all upstream work stops with no extra machinery. Simpler to debug single-threaded, and composes cleanly with the morsel-driven parallel layer above it.

**How morsel-driven parallelism works.** Two distinct layers handle plan-shape detection and dynamic scheduling:

- `planner.Parallel()` ([`planner/parallel.go`](planner/parallel.go)) detects the `LogicalAggregate → (Filter →)? Scan` plan shape, resolves group-by/aggregate column indices, and builds a `PipelineFactory` closure (one independent `TableScan → Filter → Project` pipeline per call).
- `ParallelHashAggregate.setup()` ([`exec/parallel.go`](exec/parallel.go)) handles *dynamic scheduling*: a `morselQueue` wraps an `atomic.Int64` cursor padded to its own 64-byte cache line (preventing false-sharing). Each of the `numWorkers` goroutines loops calling `q.claim(morselSize)` to atomically reserve the next chunk of row groups, runs the pipeline on that morsel, accumulates into a goroutine-local `HashAggregate`, and claims the next morsel — no coordinator, no barriers between morsels. Fast workers (low-selectivity morsels) complete early and immediately claim more work, eliminating stragglers. After all workers drain the queue, the calling goroutine merges all partial aggregates single-threadedly (correct IEEE float64 via bit-reencoding). `TableScan.Reset(rgStart, rgEnd)` allows a single open `storage.Reader` to be repositioned without reopening the file, enabling pipeline reuse across morsels within a single worker.

**Why 1024-row batches?** An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB, fitting comfortably in L1 (32–48 KB on x86, 128 KB on Apple M1 Pro performance cores). Per-batch overhead (one `Next()` call, type assertions) amortizes over 1024 rows. Same constant used by Velox, DuckDB, and Photon.

**Why selection vectors instead of filtered batches?** `Filter` writes a `[]uint16` of surviving row indices rather than allocating new vectors. Downstream operators index through the selection vector, saving allocation on the hot path and preserving the 1024-row invariant across the pipeline.

**Why little-endian?** The inner loop of every `TableScan` is `binary.LittleEndian.Uint64(buf[i*8:])` — on x86-64 this compiles to a single `MOVQ`; on ARM64 (M1 Pro) a single `LDR`. Big-endian forces an extra byte-reverse (`BSWAP`/`REV`), penalizing column reads by ~10–20%.
