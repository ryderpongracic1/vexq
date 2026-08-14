// Package parallel benchmarks parallel vs serial execution on deterministic
// synthetic data. No TPC-H data or network access required.
//
// The generator produces a ~6M row .vxq file with lineitem-like columns:
//   - l_orderkey      INT64   (random, range 1..6_000_000)
//   - l_quantity      FLOAT64 (random, 1.0..50.0)
//   - l_extendedprice FLOAT64 (random, 1.0..100_000.0)
//   - l_discount      FLOAT64 (random, 0.0..0.10)
//   - l_shipdate      DATE    (random, 1992-01-01 to 1998-12-01 as days-since-epoch)
//   - l_returnflag    STRING  (dict-encoded: "A","N","R")
//
// The benchmark query is Q6-shaped:
//
//	SELECT SUM(l_extendedprice * l_discount) AS revenue
//	FROM lineitem
//	WHERE l_shipdate >= 8766
//	  AND l_shipdate < 9131
//	  AND l_discount >= 0.05
//	  AND l_discount <= 0.07
//	  AND l_quantity < 24
package parallel

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	vsql "github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

const (
	totalRows = 6_000_000
	seed      = 42 // deterministic
)

// Schema for the synthetic lineitem-like table.
var syntheticSchema = storage.Schema{
	Fields: []storage.Field{
		{Name: "l_orderkey", Type: storage.TypeInt64},
		{Name: "l_quantity", Type: storage.TypeFloat64},
		{Name: "l_extendedprice", Type: storage.TypeFloat64},
		{Name: "l_discount", Type: storage.TypeFloat64},
		{Name: "l_shipdate", Type: storage.TypeDate},
		{Name: "l_returnflag", Type: storage.TypeString},
	},
}

// Q6-shaped query: multi-predicate filter + SUM aggregate (no GROUP BY).
// Uses SUM(l_extendedprice) to match the plan shape that planner.Parallel supports
// (the pre-projection for compound expressions is orthogonal to the scaling issue).
// Date literals are strings: the planner coerces them to date comparisons.
const q6Query = `SELECT SUM(l_extendedprice) AS revenue
FROM lineitem
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount >= 0.05
  AND l_discount <= 0.07
  AND l_quantity < 24`

var (
	dataOnce sync.Once
	dataPath string
)

// ensureData generates the synthetic .vxq file once per test run.
func ensureData(tb testing.TB) string {
	tb.Helper()
	dataOnce.Do(func() {
		dir := filepath.Join(os.TempDir(), "vexq-parallel-bench")
		os.MkdirAll(dir, 0o755)
		dataPath = filepath.Join(dir, "lineitem_synth.vxq")

		// Skip generation if file already exists and has expected size.
		if fi, err := os.Stat(dataPath); err == nil && fi.Size() > 100_000_000 {
			return
		}

		generateData(dataPath)
	})
	if dataPath == "" {
		tb.Fatal("data generation failed")
	}
	return dataPath
}

// generateData writes a deterministic ~6M row .vxq file.
func generateData(path string) {
	ctx := context.Background()
	w, err := storage.NewWriter(path, syntheticSchema)
	if err != nil {
		panic(fmt.Sprintf("new writer: %v", err))
	}

	rng := rand.New(rand.NewSource(seed))

	// Date range: 1992-01-01 (8035) to 1998-12-01 (10562) as days since epoch.
	const dateMin, dateMax = 8035, 10562
	flags := []string{"A", "N", "R"}

	rowsRemaining := totalRows
	for rowsRemaining > 0 {
		rgRows := storage.RowGroupRows
		if rgRows > rowsRemaining {
			rgRows = rowsRemaining
		}

		if err := w.BeginRowGroup(rgRows); err != nil {
			panic(fmt.Sprintf("begin row group: %v", err))
		}

		// Generate column data for this row group.
		orderkeys := make([]int64, rgRows)
		quantities := make([]float64, rgRows)
		prices := make([]float64, rgRows)
		discounts := make([]float64, rgRows)
		dates := make([]int32, rgRows)
		flagCodes := make([]string, rgRows)

		for i := range rgRows {
			orderkeys[i] = rng.Int63n(6_000_000) + 1
			quantities[i] = 1.0 + rng.Float64()*49.0
			prices[i] = 1.0 + rng.Float64()*99_999.0
			discounts[i] = rng.Float64() * 0.10
			dates[i] = int32(dateMin + rng.Intn(dateMax-dateMin))
			flagCodes[i] = flags[rng.Intn(3)]
		}

		// Write columns in schema order.
		if err := w.AppendColumn(ctx, 0, nil, orderkeys); err != nil {
			panic(fmt.Sprintf("append orderkey: %v", err))
		}
		if err := w.AppendColumn(ctx, 1, nil, quantities); err != nil {
			panic(fmt.Sprintf("append quantity: %v", err))
		}
		if err := w.AppendColumn(ctx, 2, nil, prices); err != nil {
			panic(fmt.Sprintf("append price: %v", err))
		}
		if err := w.AppendColumn(ctx, 3, nil, discounts); err != nil {
			panic(fmt.Sprintf("append discount: %v", err))
		}
		if err := w.AppendColumn(ctx, 4, nil, dates); err != nil {
			panic(fmt.Sprintf("append date: %v", err))
		}
		if err := w.AppendColumn(ctx, 5, nil, flagCodes); err != nil {
			panic(fmt.Sprintf("append flag: %v", err))
		}

		if err := w.EndRowGroup(); err != nil {
			panic(fmt.Sprintf("end row group: %v", err))
		}
		rowsRemaining -= rgRows
	}

	if err := w.Finish(ctx); err != nil {
		panic(fmt.Sprintf("finish: %v", err))
	}
}

// runSerial executes the query using planner.Physical (serial).
func runSerial(tb testing.TB, path string) float64 {
	tb.Helper()
	ctx := context.Background()

	cat, err := catalog.OpenSingle(ctx, "lineitem", path)
	if err != nil {
		tb.Fatalf("catalog: %v", err)
	}

	p := vsql.NewParser(q6Query)
	stmt, err := p.ParseStatement()
	if err != nil {
		tb.Fatalf("parse: %v", err)
	}
	sel := stmt.(*vsql.SelectStmt)

	logical, err := planner.Build(ctx, sel, cat)
	if err != nil {
		tb.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	op, err := planner.Physical(ctx, logical)
	if err != nil {
		tb.Fatalf("physical: %v", err)
	}
	defer op.Close()

	var revenue float64
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			tb.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		// Q6 returns a single row with SUM(l_extendedprice * l_discount).
		if batch.Length > 0 {
			if fv, ok := batch.Vectors[0].(*exec.Float64Vector); ok {
				revenue = fv.Values[0]
			}
		}
	}
	return revenue
}

// runParallel executes the query using planner.Parallel with numWorkers.
func runParallel(tb testing.TB, path string, numWorkers int) float64 {
	tb.Helper()
	ctx := context.Background()

	cat, err := catalog.OpenSingle(ctx, "lineitem", path)
	if err != nil {
		tb.Fatalf("catalog: %v", err)
	}

	p := vsql.NewParser(q6Query)
	stmt, err := p.ParseStatement()
	if err != nil {
		tb.Fatalf("parse: %v", err)
	}
	sel := stmt.(*vsql.SelectStmt)

	logical, err := planner.Build(ctx, sel, cat)
	if err != nil {
		tb.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	op, err := planner.Parallel(ctx, logical, numWorkers)
	if err != nil {
		tb.Fatalf("parallel: %v", err)
	}
	defer op.Close()

	var revenue float64
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			tb.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		if batch.Length > 0 {
			if fv, ok := batch.Vectors[0].(*exec.Float64Vector); ok {
				revenue = fv.Values[0]
			}
		}
	}
	return revenue
}

// BenchmarkQ6Serial benchmarks serial Q6 execution.
func BenchmarkQ6Serial(b *testing.B) {
	path := ensureData(b)
	b.ResetTimer()
	for range b.N {
		runSerial(b, path)
	}
}

// BenchmarkQ6Parallel benchmarks parallel Q6 execution with runtime.NumCPU() workers.
func BenchmarkQ6Parallel(b *testing.B) {
	path := ensureData(b)
	numWorkers := runtime.NumCPU()
	b.ResetTimer()
	for range b.N {
		runParallel(b, path, numWorkers)
	}
}

// TestSerialParallelEquivalence verifies that serial and parallel produce
// identical results on the synthetic dataset.
func TestSerialParallelEquivalence(t *testing.T) {
	path := ensureData(t)
	serialResult := runSerial(t, path)
	parallelResult := runParallel(t, path, runtime.NumCPU())

	// Float64 results should be very close (within floating point tolerance).
	// Parallel accumulates partial sums in different order → small FP drift.
	diff := math.Abs(serialResult - parallelResult)
	relDiff := diff / math.Max(math.Abs(serialResult), 1.0)
	if relDiff > 1e-6 {
		t.Fatalf("serial (%g) != parallel (%g), relative diff %g", serialResult, parallelResult, relDiff)
	}
	t.Logf("serial=%.6f, parallel=%.6f, relDiff=%.2e", serialResult, parallelResult, relDiff)
}
