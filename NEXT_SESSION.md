# Next Session: Where to Resume

## Status
- Phases 1–6: ✅ Complete and pushed to https://github.com/ryderpongracic1/vexq

## Benchmark Results (Phase 6 Final)

TPC-H SF=1 (6M lineitem rows) on Apple M1 Max, 3× runs:

| Query | vexq | SQLite | Speedup |
|-------|------|--------|---------|
| Q1 (full scan + GROUP BY 2 strings) | 693 ms | 3,320 ms | 4.8× |
| Q6 (5 range predicates + SUM) | 457 ms | 583 ms | 1.3× |

Both Q1 and Q6 correctness verified against SQLite output.

## What Was Done This Session

1. Fixed all correctness bugs from simplify review:
   - `planner/physical.go`: LIKE case was discarding computed `child` expression → fixed to pass it directly to `LikeExpr.Child`
   - `planner/physical.go`: BETWEEN zone predicate returned `!x.Not` (backwards) → fixed to `return x.Not`
   - `planner/physical.go`: redundant `literalInt64(lit)` call in float64 zone coercion → simplified to use already-computed `v`

2. Applied all quality/efficiency improvements:
   - `exec/aggregate.go`: removed redundant `outRows` field and no-op `groupCnt[key] = 0`
   - `exec/batch.go`: added shared `newStringVector` helper
   - `exec/sort.go`, `exec/join.go`: inlined `math.Float64frombits`, removed dead wrapper
   - `cmd/vexq/main.go`: hoisted `vals` allocation outside hot loop; removed `max()` shadowing Go 1.21 builtin
   - `cmd/vexqgen/main.go`: added `scanBufSize` constant; use `storage.RowGroupRows` directly; `clear()` in `reset()`

3. All tests pass: `go test ./internal/... ./storage/... ./exec/... ./sql/... -race -count=1`

## Possible Next Steps

- **Q3/Q12 benchmarks**: add TPC-H Q3 (join + GROUP BY + ORDER BY) and Q12 (join + CASE WHEN) to bench_test.go
- **Morsel-driven parallelism**: partition row groups across goroutines; each goroutine runs an independent operator pipeline on a subset; merge results at HashAggregate
- **Spill-to-disk for ExternalSort**: current sort is in-memory only; add two-phase merge-sort with temp files for large ORDER BY results
- **EXPLAIN output**: `vexq explain "SELECT ..."` prints the physical operator tree with zone-map statistics
- **AVG denominator fix**: AVG currently divides by `groupCnt` which counts all rows per group, not just non-null ones — fix to use a per-aggregate non-null counter

## Data Files (local only, not in git)
- `data/lineitem.vxq` (733 MB, 6M rows, 92 row groups)
- `data/orders.vxq` (152 MB, 1.5M rows)
- `data/customer.vxq` (27 MB, 150K rows)
- `data/tpch.db` (SQLite baseline)
- `data/*.tbl` (raw TPC-H pipe-delimited source)
