package parallel

// Q1-shaped benchmarks: a high-selectivity filter feeding a GROUP BY with
// several aggregates, over the same synthetic dataset the Q6-shaped benchmarks
// use. Q6-shaped is filter-dominated at ~2% selectivity; Q1-shaped keeps nearly
// every row and pays projection/aggregation cost on all of them, so the two
// together bracket the allocation behaviour of the operator pipeline.

import (
	"context"
	"runtime"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/planner"
	vsql "github.com/ryderpongracic1/vexq/sql"
)

// q1Query groups by a dictionary-encoded string column with four aggregates,
// mirroring the shape of TPC-H Q1 against the synthetic schema.
const q1Query = `SELECT l_returnflag,
       SUM(l_quantity) AS sum_qty,
       SUM(l_extendedprice) AS sum_price,
       AVG(l_discount) AS avg_disc,
       COUNT(*) AS cnt
FROM lineitem
WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag`

// drainRows executes query and returns the number of result rows, discarding
// values. Used by the Q1-shaped benchmarks, whose first output column is a
// string and therefore cannot reuse runSerial/runParallel's float extraction.
func drainRows(tb testing.TB, path, query string, numWorkers int) int {
	tb.Helper()
	ctx := context.Background()

	cat, err := catalog.OpenSingle(ctx, "lineitem", path)
	if err != nil {
		tb.Fatalf("catalog: %v", err)
	}

	p := vsql.NewParser(query)
	stmt, err := p.ParseStatement()
	if err != nil {
		tb.Fatalf("parse: %v", err)
	}

	logical, err := planner.Build(ctx, stmt.(*vsql.SelectStmt), cat)
	if err != nil {
		tb.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	rows := 0
	if numWorkers <= 0 {
		physical, err := planner.Physical(ctx, logical)
		if err != nil {
			tb.Fatalf("physical: %v", err)
		}
		defer physical.Close()
		for {
			batch, err := physical.Next(ctx)
			if err != nil {
				tb.Fatalf("next: %v", err)
			}
			if batch == nil {
				break
			}
			rows += batch.Length
		}
		return rows
	}

	parallel, err := planner.Parallel(ctx, logical, numWorkers)
	if err != nil {
		tb.Fatalf("parallel: %v", err)
	}
	defer parallel.Close()
	for {
		batch, err := parallel.Next(ctx)
		if err != nil {
			tb.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		rows += batch.Length
	}
	return rows
}

// BenchmarkQ1Serial benchmarks the Q1-shaped GROUP BY serially.
func BenchmarkQ1Serial(b *testing.B) {
	path := ensureData(b)
	b.ResetTimer()
	for range b.N {
		drainRows(b, path, q1Query, 0)
	}
}

// BenchmarkQ1Parallel benchmarks the Q1-shaped GROUP BY with NumCPU workers.
func BenchmarkQ1Parallel(b *testing.B) {
	path := ensureData(b)
	numWorkers := runtime.NumCPU()
	b.ResetTimer()
	for range b.N {
		drainRows(b, path, q1Query, numWorkers)
	}
}

// TestQ1SerialParallelEquivalence asserts the Q1-shaped query returns the same
// group count on both execution paths, so the benchmarks compare equal work.
func TestQ1SerialParallelEquivalence(t *testing.T) {
	path := ensureData(t)
	serialRows := drainRows(t, path, q1Query, 0)
	parallelRows := drainRows(t, path, q1Query, runtime.NumCPU())
	if serialRows != parallelRows {
		t.Fatalf("serial produced %d rows, parallel produced %d", serialRows, parallelRows)
	}
	if serialRows == 0 {
		t.Fatal("Q1-shaped query produced no rows")
	}
	t.Logf("Q1-shaped groups: %d", serialRows)
}
