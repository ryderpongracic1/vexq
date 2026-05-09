# vexq

A vectorized columnar SQL query engine written from scratch in Go, built to be demonstrably faster than SQLite on TPC-H analytical queries.

## Overview

vexq implements a complete analytical query pipeline — from a custom on-disk columnar file format through a SQL parser, rule-based optimizer, and vectorized execution engine. The design follows the same engineering principles as [distrikv](https://github.com/ryderpongracic1/distrikv) (append-only manifest, block CRCs, atomic writes via temp+rename) while introducing a columnar-vectorized execution model suited for analytical workloads.

The engine processes data in batches of 1024 rows, keeping each batch in L1 cache and eliminating per-row function call overhead. A pushed-down predicate + zone-map pruning layer skips row groups entirely before any I/O, and dictionary encoding on string columns reduces string comparisons to integer equality checks in the hot loop.

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
  ParallelHashAggregate — Fan-out across goroutines + partial-aggregate merge
  ExternalSort      — In-memory sort (spill-to-disk planned for v2)
  HashJoin          — Build/probe inner join
  Limit

catalog/       — Table registry with lazy schema loading from .vxq footer
storage/       — .vxq file format: writer, reader, block codec, zone maps
internal/encoding — Little-endian primitives, CRC32-IEEE helpers
bench/tpch     — TPC-H Q1/Q3/Q6/Q12 benchmarks vs SQLite
```

## .vxq File Format

Custom columnar format designed for vectorized reads:

- **Layout**: file header → row groups (65,536 rows each) → footer
- **Blocks**: 1,024 rows per block with 128-byte null bitmap + typed payload + CRC32
- **Endianness**: little-endian throughout (single `MOV` on x86/ARM vs `MOV+BSWAP` for big-endian)
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

Requires Go 1.21+. No external runtime dependencies (SQLite is benchmark-only).

## Benchmarks

TPC-H scale factor 1 (6M lineitem rows) on Apple M1 Max (10 cores). SQLite configured with `WAL`, `NORMAL` sync, 256 MB cache, and `ANALYZE`. Each benchmark run 3×; numbers are median wall time per run.

### Single-core

| Query | Description | vexq | SQLite | Speedup |
|-------|-------------|------|--------|---------|
| Q1 | Pricing summary — full scan, GROUP BY 2 string cols | 733 ms | 3,320 ms | **4.5×** |
| Q6 | Revenue forecast — scan with 5 range predicates, SUM | 473 ms | 583 ms | **1.2×** |
| Q3 | Shipping priority — 3-table join, complex SUM, LIMIT 10 | 1,218 ms | 3,764 ms | **3.1×** |
| Q12 | Shipping modes — 2-table join, CASE WHEN agg, date comparisons | 1,903 ms | 1,130 ms | 0.6× |

Q12 is currently slower than SQLite: the HashJoin build phase materialises the full orders table and SQLite benefits from its B-tree index on `o_orderkey`. Future work: index-nested-loop join and late materialisation would close this gap.

### Parallel execution (morsel-driven, 10 goroutines)

`planner.Parallel()` partitions the file's row groups across `runtime.NumCPU()` goroutines. Each goroutine runs an independent `TableScan → Filter → Project → partial HashAggregate` pipeline on its slice; a merge step combines partial aggregates in the calling goroutine.

| Query | vexq serial | vexq parallel | SQLite | Speedup (parallel vs SQLite) |
|-------|------------|---------------|--------|------------------------------|
| Q6 | 473 ms | **233 ms** | 583 ms | **2.5×** |
| Q1† | 733 ms | 733 ms | 3,320 ms | 4.5× |

† Q1 has an `ORDER BY` clause, so the root operator is a `Sort`, not an `Aggregate`. `planner.Parallel()` falls back to `planner.Physical()` for plans it cannot partition (joins, sorts at the root). Parallel execution applies to aggregate-only plans today; Q3/Q12 also fall back because they contain `HashJoin`.

Run benchmarks (after generating data):

```bash
# Generate TPC-H SF=1 data
cd data && ../tools/dbgen -s 1 -f

# Convert to .vxq
vexqgen lineitem  data/lineitem.tbl  data/lineitem.vxq
vexqgen orders    data/orders.tbl    data/orders.vxq
vexqgen customer  data/customer.tbl  data/customer.vxq

# Load SQLite baseline
go test ./bench/tpch/ -run TestSetupSQLite -v

# Run all benchmarks (serial + parallel)
go test ./bench/tpch/ -bench=. -benchtime=3x -v

# Run just parallel benchmarks
go test ./bench/tpch/ \
  -bench="BenchmarkVexqQ1$|BenchmarkVexqQ1Parallel|BenchmarkVexqQ6$|BenchmarkVexqQ6Parallel" \
  -benchtime=3x -v
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

## Design Notes

**Why pull-based (Volcano model)?** `LIMIT` and short-circuit predicates terminate naturally — when the root stops calling `Next()`, all upstream work stops with no extra machinery. Simpler to debug single-threaded, and composes cleanly with the morsel-driven parallel layer above it.

**How morsel-driven parallelism works.** `planner.Parallel()` detects a `LogicalAggregate → (Filter →)? Scan` plan shape and builds a `ParallelHashAggregate` that partitions the file's row groups into equal slices — one per `runtime.NumCPU()` goroutine. Each goroutine runs a fully independent `TableScan → Filter → Project → HashAggregate` pipeline on its slice, accumulates partial results locally, then sends its `HashAggregate` state on a buffered channel. The main goroutine merges all partial aggregates (correctly handling float64 SUM/MIN/MAX via IEEE-bit re-encoding and AVG via sum+count) and delegates output to a single final `HashAggregate`. No shared mutable state; synchronisation is only via the channel.

**Why 1024-row batches?** An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB, fitting in L1 (typically 32 KB on modern x86). Per-batch overhead (one `Next()` call, type assertions) amortizes over 1024 rows. Same constant used by Velox, DuckDB, and Photon.

**Why selection vectors instead of filtered batches?** `Filter` writes a `[]uint16` of surviving row indices rather than allocating new vectors. Downstream operators index through the selection vector, saving allocation on the hot path and preserving the 1024-row invariant across the pipeline.

**Why little-endian?** The inner loop of every `TableScan` is `binary.LittleEndian.Uint64(buf[i*8:])` — on x86/ARM this compiles to a single `MOV`; big-endian forces `MOV+BSWAP`, penalizing column reads by ~10–20%.
