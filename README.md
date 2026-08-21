# vexq

[![CI](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml/badge.svg)](https://github.com/ryderpongracic1/vexq/actions/workflows/ci.yml)

Built a vectorized analytical SQL execution engine in Go using columnar storage, dictionary encoding, zone maps, and morsel-driven parallel execution; implemented vectorized Volcano operators and a custom query parsing pipeline.

## Overview

vexq is a complete analytical query engine — from a custom on-disk columnar file format through a SQL parser, rule-based optimizer, and vectorized execution engine. The design shares structural principles with [distrikv](https://github.com/ryderpongracic1/distrikv) (append-only manifest, block CRCs, atomic writes via temp+rename) while adding a columnar-vectorized execution model suited for OLAP workloads.

On TPC-H Q1 (GROUP BY aggregate over 6M rows), vexq end-to-end (all cores, 22 ms median) is within 1.2× of DuckDB's steady-state (19 ms warm page cache) and matches its cold-start latency (24 ms first invocation) on the same machine — within 1.2× single-threaded — the result of dictionary-code integer-keyed aggregation, coarse-grained I/O, join column pruning, a radix-partitioned parallel join, and a systematic allocation-elimination campaign verified by GOGC ablation. vexq is faster than SQLite on all four benchmarked TPC-H queries (3.0–18.4×), and its morsel-driven scheduler reaches 8.6× scaling on 14 cores — ~86% efficiency against the 10-performance-core ceiling, exceeding DuckDB's own scaling on every benchmarked query.

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

## Measured results

TPC-H scale factor 1 (6M lineitem rows) on Apple M4 Pro (14-core), warm page cache; each benchmark run 10x, numbers are median wall time, and the harness reports per-iteration min/median/max in the same output line so the spread behind every headline is visible where it was produced. Full methodology, query provenance, the optimization-round history, the GOGC ablation, and the worker sweep are in [docs/benchmarks.md](docs/benchmarks.md).

### Ceiling — vs DuckDB

The comparison decomposes the gap into per-core execution efficiency (single-threaded vs single-threaded) and end-to-end throughput (both engines on all 14 cores). Note: this comparison runs DuckDB on ARM64, where its strongest x86 SIMD paths (AVX-512) are unavailable; an x86 comparison would likely widen DuckDB's per-core advantage. The comparison is same-machine and therefore fair, but the Q1 figures should be read with that context.

| Query | vexq serial | vexq parallel | DuckDB (1 thread) | DuckDB (14 threads) | Per-core gap | End-to-end gap |
|-------|------------:|--------------:|------------------:|--------------------:|:------------:|:--------------:|
| Q1 | 193 ms | 22 ms | 159 ms | 19 ms (steady) / 24 ms (cold) | **1.2×** | **1.2× steady / ~1.0× cold** |
| Q6 | 67 ms | 10 ms | 37 ms | 8 ms | **1.8×** | **1.3×** |
| Q3 | 137 ms | 36 ms | 62 ms | 17 ms | **2.2×** | **2.1×** |
| Q12 | 238 ms | 36 ms | 81 ms | 16 ms | **2.9×** | **2.2×** |

vexq parallel Q1 (22 ms median, 21–24 ms spread) matches DuckDB's cold-start latency (24 ms first invocation) and sits within 1.2× of its warm steady-state (19 ms). The per-query decomposition — what each remaining gap is attributable to, with the measurements behind it — is in [docs/benchmarks.md](docs/benchmarks.md).

### Floor — vs SQLite

Beating a row-store on full-table OLAP scans is the expected outcome of any columnar engine; this baseline confirms the columnar layout and vectorized operators are paying off.

| Query | Description | vexq serial | SQLite | Speedup |
|-------|-------------|------------:|-------:|:-------:|
| Q1 | Pricing summary — full scan, GROUP BY 2 string cols | 193 ms | 2,197 ms | **11.4×** |
| Q6 | Revenue forecast — scan with 5 range predicates, SUM(expr) | 67 ms | 381 ms | **5.7×** |
| Q3 | Shipping priority — 3-table join, complex SUM, LIMIT 10 | 137 ms | 2,525 ms | **18.4×** |
| Q12 | Shipping modes — 2-table join, CASE WHEN agg, date comparisons | 238 ms | 724 ms | **3.0×** |

### Parallel scaling (morsel-driven, 14 goroutines)

| Query | vexq serial | vexq parallel | Speedup | Notes |
|-------|------------:|--------------:|:-------:|-------|
| Q1 | 193 ms | **22 ms** | **8.6×** | Sort-peeling: parallel aggregate, serial 4-row sort |
| Q6 | 67 ms | **10 ms** | **6.5×** | Expression aggregate materialized per morsel |
| Q3 | 137 ms | **36 ms** | **3.8×** | Radix-partitioned parallel build + parallel probe (3-table chain) |
| Q12 | 238 ms | **36 ms** | **6.7×** | Parallel join + CASE WHEN expression aggregate |

Q1's 8.6× on 14 cores (10P+4E) is ~86% efficiency against the realistic 10-P-core ceiling; true like-for-like scaling (GOGC=off both sides) is 4.4–8.8× across the four queries. The GOGC ablation and worker sweep behind those figures are in [docs/benchmarks.md](docs/benchmarks.md).

### Correctness

All four TPC-H query results are verified identical to SQLite output in-harness (Q6's SUM to 1e-9 relative tolerance), and vexq's canonical Q6 matches DuckDB's result exactly. An independent 72-query golden suite ([internal/goldentest/](internal/goldentest/)) verifies the full SQL subset against a naive row-at-a-time reference evaluator across four oracle paths — serial, parallel, optimizer-off, and stacked-filter — under `-race` in CI. 72/72 passing on all four paths, zero known correctness issues.

## Quickstart

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

Build:

```bash
go build ./cmd/vexq/
go build ./cmd/vexqgen/
go test ./... -race -count=1
```

Requires Go 1.22+. No external runtime dependencies (SQLite and DuckDB are benchmark-only).

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Hardware-level design decisions — batch sizing, cache-line discipline, selection vectors, morsel scheduling — and the design notes behind them |
| [docs/file-format.md](docs/file-format.md) | The `.vxq` columnar format: layout, encodings, zone maps, integrity |
| [docs/sql.md](docs/sql.md) | The supported SQL subset |
| [docs/benchmarks.md](docs/benchmarks.md) | Full benchmark methodology and provenance, DuckDB/SQLite results with per-query decomposition, the three optimization rounds, GOGC ablation, worker sweep, profiling recipes, and the what's-left backlog including the measured-and-declined items |
| [docs/progress.md](docs/progress.md) | Phase-by-phase build history |

## Scope and limitations

- **Benchmarks are single-machine, ARM64.** DuckDB's strongest x86 SIMD paths (AVX-512) are unavailable on the benchmark machine; an x86 comparison would likely widen DuckDB's per-core advantage. The comparison is same-machine and therefore fair, but the Q1 figures should be read with that context.
- **`ExternalSort` is in-memory** (spill-to-disk planned); `COUNT(DISTINCT)` falls back to serial execution.
- **In-engine SIMD and probe-stream partitioning were profiled and declined as measured** (end-to-end ceilings ~1.03× and ~1.005× respectively): serial wall time is dominated by pread syscalls and page management, not compute, so the honest remaining optimizations are I/O-side — late materialization and adaptive compression, both open. The profiles are in [docs/benchmarks.md](docs/benchmarks.md).
- **Column pruning through joins is name-based, not cost-based**, and a single-table `SELECT COUNT(*)` still decodes every column.
- **Default worker count is `runtime.NumCPU()`**; capping at the performance-core count (the measured bend in the worker sweep) remains open.
