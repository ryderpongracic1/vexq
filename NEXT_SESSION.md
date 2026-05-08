# Next Session: Where to Resume

## Status
- Phases 1–5: ✅ Complete and pushed to https://github.com/ryderpongracic1/vexq
- Phase 6 (TPC-H benchmark): 🔧 In progress — Q1 passing, Q6 failing

## Immediate Bug to Fix First

**`l_quantity < 24` returns 0 rows** (Q6 broken)

Run to reproduce:
```
go build -o /tmp/vexq ./cmd/vexq/
/tmp/vexq data/lineitem.vxq "SELECT COUNT(*) FROM lineitem WHERE l_quantity < 24"
# returns 0 rows — should return ~2.4M
```

Root cause: `l_discount >= 0.05` (float literal) works fine (~3.2M rows), but
`l_quantity < 24` (int literal `24` being compared to `float64` column) returns 0.

Trace: `coercePair` in `planner/physical.go` converts int literal `24` →
`float64(24.0)` for the `l_quantity` (TypeFloat64) column. This looks correct.
The bug is somewhere in the expression evaluation path — likely in `exec/expr.go`
`BinOp.Eval()` for `Float64Vector < Float64Literal`, or in the zone map predicate.

Narrowing steps:
1. Test `l_quantity < 1000` (should match all 6M rows) — if 0 rows, it's a
   general float64 comparison bug, not a boundary issue.
2. Check `evalCmp` in `exec/expr.go` for `Float64Vector` — the null bitmap
   check uses `storage.IsNullBit(out.NullBitmap, i)` which returns true=null.
   Make sure the null bitmap is being SET correctly by the `for i := 0; i < n`
   loop that calls `storage.SetValidBit`.
3. Check zone map pruning: `literalInt64(FloatLiteral{24.0})` returns
   `int64(math.Float64bits(24.0))`. Compare against `int64(zm.Max)` where
   `zm.Max` stores float64 bits. For `< 24.0` (OpLT), zone keeps if
   `rgMin < v`. If `l_quantity` min in any row group is 1.0 (TypeFloat64 bits
   = `0x3FF0000000000000`), and v = `math.Float64bits(24.0)` = `0x4038000000000000`,
   then `0x3FF0... < 0x4038...` as int64 = true → row group kept. Looks correct.
4. Most likely: the null bitmap issue. In `evalCmp`, the null propagation loop
   calls `lv.IsNull(i)` and `rv.IsNull(i)`. For the Float64Literal vector
   (created by `Literal.Eval()`), `NullBitmap = storage.FullBitmap(n)` (all valid).
   For the column vector from scan, check that null bitmap has 1s for valid rows.

## After Fixing Q6

1. Run Q6 correctness test:
   ```
   go test ./bench/tpch/ -run TestQ6Correctness -v
   ```

2. Run full benchmark suite:
   ```
   go test ./bench/tpch/ -bench=. -benchtime=3x -v -timeout 600s
   ```
   Expected: vexq 2–8× faster than SQLite on Q1 and Q6.

3. Run all tests:
   ```
   go test ./... -race -count=1
   ```

4. Commit benchmark results and push to GitHub.

## Data Files (local only, not in git)
- `data/lineitem.vxq` (733 MB, 6M rows, 92 row groups)
- `data/orders.vxq` (152 MB, 1.5M rows)
- `data/customer.vxq` (27 MB, 150K rows)
- `data/tpch.db` (SQLite, loaded with TestSetupSQLite)
- `data/*.tbl` (raw TPC-H pipe-delimited source)

To regenerate if lost:
```
cd data && ../tools/dbgen -s 1 -f    # requires dists.dss in cwd
go build -o /tmp/vexqgen ./cmd/vexqgen/
/tmp/vexqgen lineitem data/lineitem.tbl data/lineitem.vxq
/tmp/vexqgen orders   data/orders.tbl   data/orders.vxq
/tmp/vexqgen customer data/customer.tbl data/customer.vxq
go test ./bench/tpch/ -run TestSetupSQLite -v -timeout 120s
```

## Key Files Changed This Session (Phase 6 + bug fixes)
- `exec/aggregate.go` — fixed float64 SUM/AVG accumulation; GROUP BY string support
- `exec/sort.go` — fixed ExternalSort string column handling
- `planner/physical.go` — date/float literal type coercion (`coercePair`)
- `planner/logical.go` — `exprName` for `COUNT(*)` alias
- `cmd/vexq/main.go` — CLI binary
- `cmd/vexqgen/main.go` — TPC-H .tbl → .vxq converter
- `bench/tpch/bench_test.go` — Q1/Q6 correctness tests + benchmarks
