package planner_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
)

// q12Shaped is TPC-H Q12: a two-table join with CASE WHEN aggregates over the
// build side, grouped by a probe-side column. This is the primary shape the
// probe-side parallel join targets.
const q12Shaped = `SELECT l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
  FROM orders, lineitem
  WHERE o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= '1994-01-01'
    AND l_receiptdate < '1995-01-01'
  GROUP BY l_shipmode
  ORDER BY l_shipmode`

// benchCatalog opens orders.vxq and lineitem.vxq from $VEXQ_BENCH_DIR, skipping
// when that data is absent — the files are too large to keep in the repository.
// Generate them with cmd/vexqgen from TPC-H (or TPC-H-shaped) .tbl input:
//
//	vexqgen orders   orders.tbl   orders.vxq
//	vexqgen lineitem lineitem.tbl lineitem.vxq
//	VEXQ_BENCH_DIR=<dir> go test ./planner/ -run '^$' \
//	    -bench 'BenchmarkJoinQ12' -benchtime=5x
func benchCatalog(b *testing.B) *catalog.Catalog {
	b.Helper()
	dir := os.Getenv("VEXQ_BENCH_DIR")
	if dir == "" {
		b.Skip("VEXQ_BENCH_DIR not set — see benchCatalog for data setup")
	}
	tables := map[string]string{
		"orders":   filepath.Join(dir, "orders.vxq"),
		"lineitem": filepath.Join(dir, "lineitem.vxq"),
	}
	for _, p := range tables {
		if _, err := os.Stat(p); err != nil {
			b.Skipf("missing benchmark data %s: %v", p, err)
		}
	}
	cat, err := catalog.OpenMulti(context.Background(), tables)
	if err != nil {
		b.Fatalf("OpenMulti: %v", err)
	}
	return cat
}

// runBenchQuery plans and fully drains query once. workers <= 0 selects the
// serial physical planner. The plan is rebuilt per iteration because aggregate
// operators cache their merged state and are single-use.
func runBenchQuery(b *testing.B, cat *catalog.Catalog, query string, workers int) {
	b.Helper()
	ctx := context.Background()

	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		b.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	for range b.N {
		logical, err := planner.Build(ctx, stmt, cat)
		if err != nil {
			b.Fatalf("Build: %v", err)
		}
		logical = planner.Optimize(logical)

		var op exec.Operator
		if workers > 0 {
			op, err = planner.Parallel(ctx, logical, workers)
		} else {
			op, err = planner.Physical(ctx, logical)
		}
		if err != nil {
			b.Fatalf("plan (workers=%d): %v", workers, err)
		}
		rows := 0
		for {
			batch, err := op.Next(ctx)
			if err != nil {
				b.Fatalf("Next: %v", err)
			}
			if batch == nil {
				break
			}
			rows += batch.Length
		}
		_ = op.Close()
		if rows == 0 {
			b.Fatal("query returned no rows — benchmark data does not match the query")
		}
	}
}

func BenchmarkJoinQ12Serial(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q12Shaped, 0)
}

func BenchmarkJoinQ12Parallel1(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q12Shaped, 1)
}

func BenchmarkJoinQ12Parallel2(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q12Shaped, 2)
}

func BenchmarkJoinQ12Parallel4(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q12Shaped, 4)
}

func BenchmarkJoinQ12Parallel8(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q12Shaped, 8)
}

// buildOnlyJoin isolates the serial build phase: the l_receiptdate bound is
// outside every probe row group's zone map, so the probe side is pruned to
// nothing and what remains is the build-side scan plus hash table
// materialisation. The ratio of this to BenchmarkJoinQ12Serial is the Amdahl
// bound on probe-side-only parallelism.
const buildOnlyJoin = `SELECT COUNT(*) AS c
  FROM orders, lineitem
  WHERE o_orderkey = l_orderkey AND l_receiptdate >= '2100-01-01'`

func BenchmarkJoinBuildOnlySerial(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, buildOnlyJoin, 0)
}

func BenchmarkJoinBuildOnlyParallel4(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, buildOnlyJoin, 4)
}

// probeHeavyJoin removes the probe-side predicates so the probe phase dominates,
// showing what probe-side parallelism buys once the build side is not the
// majority of the work.
const probeHeavyJoin = `SELECT l_shipmode, COUNT(*) AS c, SUM(l_extendedprice) AS rev
  FROM orders, lineitem
  WHERE o_orderkey = l_orderkey
  GROUP BY l_shipmode`

func BenchmarkJoinProbeHeavySerial(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, probeHeavyJoin, 0)
}

func BenchmarkJoinProbeHeavyParallel4(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, probeHeavyJoin, 4)
}
