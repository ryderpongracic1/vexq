package planner_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

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

// q3Shaped is TPC-H Q3: a three-table left-deep chain. The whole
// customer ⋈ orders subtree is the build side, lineitem the probe side, so it
// exercises needed-column propagation through a nested join.
const q3Shaped = `SELECT l_orderkey, SUM(l_extendedprice) AS revenue, o_orderdate, o_shippriority
  FROM customer, orders, lineitem
  WHERE c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < '1995-03-15'
    AND l_shipdate > '1995-03-15'
  GROUP BY l_orderkey, o_orderdate, o_shippriority
  ORDER BY revenue DESC
  LIMIT 10`

// benchCatalog opens orders.vxq, lineitem.vxq and customer.vxq from
// $VEXQ_BENCH_DIR, skipping when that data is absent — the files are too large
// to keep in the repository. Generate them with cmd/vexqgen from TPC-H (or
// TPC-H-shaped) .tbl input:
//
//	vexqgen orders   orders.tbl   orders.vxq
//	vexqgen lineitem lineitem.tbl lineitem.vxq
//	vexqgen customer customer.tbl customer.vxq
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
		"customer": filepath.Join(dir, "customer.vxq"),
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

func BenchmarkJoinQ3Serial(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q3Shaped, 0)
}

func BenchmarkJoinQ3Parallel4(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, q3Shaped, 4)
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

func BenchmarkJoinBuildOnlyParallel1(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, buildOnlyJoin, 1)
}

func BenchmarkJoinBuildOnlyParallel2(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, buildOnlyJoin, 2)
}

func BenchmarkJoinBuildOnlyParallel4(b *testing.B) {
	cat := benchCatalog(b)
	runBenchQuery(b, cat, buildOnlyJoin, 4)
}

// ---- Build-phase attribution -------------------------------------------------

// q12NoOrder is q12Shaped without ORDER BY, so the plan root is the parallel join
// operator itself rather than the serial sort peeled above it. That is what lets
// benchBuildPhase read exec.JoinBuildStats off the operator.
const q12NoOrder = `SELECT l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
  FROM orders, lineitem
  WHERE o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= '1994-01-01'
    AND l_receiptdate < '1995-01-01'
  GROUP BY l_shipmode`

// benchBuildPhase splits a parallel join's wall time into its build and probe
// phases using the build stats the operator records, and reports the partition
// count it chose. Unlike BenchmarkJoinBuildOnly*, which isolates the build with a
// query whose probe side prunes to nothing, this measures the build inside the
// real query — so build-ns/op and ns/op come from the same run and probe-ns/op is
// their exact difference.
func benchBuildPhase(b *testing.B, cat *catalog.Catalog, query string, workers int) {
	b.Helper()
	ctx := context.Background()

	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		b.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	var buildTotal time.Duration
	var stats exec.JoinBuildStats
	for range b.N {
		logical, err := planner.Build(ctx, stmt, cat)
		if err != nil {
			b.Fatalf("Build: %v", err)
		}
		logical = planner.Optimize(logical)
		op, err := planner.Parallel(ctx, logical, workers)
		if err != nil {
			b.Fatalf("Parallel: %v", err)
		}
		pj, ok := op.(*exec.ParallelHashJoinAggregate)
		if !ok {
			_ = op.Close()
			b.Fatalf("root operator = %T, want *exec.ParallelHashJoinAggregate — the query must not have ORDER BY or LIMIT", op)
		}
		rows := 0
		for {
			batch, err := pj.Next(ctx)
			if err != nil {
				b.Fatalf("Next: %v", err)
			}
			if batch == nil {
				break
			}
			rows += batch.Length
		}
		stats = pj.BuildStats()
		buildTotal += stats.Elapsed
		_ = pj.Close()
		if rows == 0 {
			b.Fatal("query returned no rows — benchmark data does not match the query")
		}
	}
	b.ReportMetric(float64(buildTotal.Nanoseconds())/float64(b.N), "build-ns/op")
	b.ReportMetric(float64(stats.Partitions), "partitions")
	b.ReportMetric(float64(stats.Rows), "build-rows")
}

func BenchmarkJoinBuildPhaseQ12Parallel1(b *testing.B) {
	benchBuildPhase(b, benchCatalog(b), q12NoOrder, 1)
}

func BenchmarkJoinBuildPhaseQ12Parallel2(b *testing.B) {
	benchBuildPhase(b, benchCatalog(b), q12NoOrder, 2)
}

func BenchmarkJoinBuildPhaseQ12Parallel4(b *testing.B) {
	benchBuildPhase(b, benchCatalog(b), q12NoOrder, 4)
}

func BenchmarkJoinBuildPhaseProbeHeavyParallel4(b *testing.B) {
	benchBuildPhase(b, benchCatalog(b), probeHeavyJoin, 4)
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
