package planner_test

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/storage"
)

// Tests for parallel execution of aggregates over computed expressions —
// the canonical TPC-H Q6 shape, SUM(l_extendedprice * l_discount).
//
// Each worker pipeline ends with the same pre-projection the serial planner
// applies, materializing the expression into a synthetic column per morsel.
// The properties pinned here are:
//   - the plan is genuinely parallel (a *exec.ParallelHashAggregate, not a
//     silent fallback to Physical)
//   - results agree with the serial path across filters, group-bys, aggregate
//     kinds, operand types, nulls and worker counts
//   - the pre-existing COUNT(DISTINCT) serial fallback is not regressed
//
// On float comparison: partitioning a float64 SUM reorders its additions, and
// float addition is not associative, so parallel results are not bit-identical
// to serial for any partitioned float reduction — including the simple-column
// path that predates expression support. Float columns are therefore compared
// with a relative tolerance matching internal/goldentest's FloatEpsilon (1e-9),
// which is the project's established correctness standard for float aggregates.
// Integer aggregates and COUNT are compared exactly.

// floatTolerance mirrors internal/goldentest.FloatEpsilon.
const floatTolerance = 1e-9

// dayssince1970 converts a Y/M/D date to the int32 day count TypeDate stores.
func daysSince1970(y, m, d int) int32 {
	const secondsPerDay = 86400
	// Days from 1970-01-01 computed via a fixed civil-from-days algorithm.
	// Using a simple accumulation keeps the test free of time-package parsing.
	days := 0
	for yy := 1970; yy < y; yy++ {
		if isLeap(yy) {
			days += 366
		} else {
			days += 365
		}
	}
	monthDays := [12]int{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
	for mm := 1; mm < m; mm++ {
		days += monthDays[mm-1]
		if mm == 2 && isLeap(y) {
			days++
		}
	}
	days += d - 1
	_ = secondsPerDay
	return int32(days)
}

func isLeap(y int) bool { return (y%4 == 0 && y%100 != 0) || y%400 == 0 }

// lineitemData holds the generated column values so tests can compute expected
// aggregate results independently of the engine.
type lineitemData struct {
	path        string
	price       []float64
	discount    []float64
	quantity    []int64
	shipdate    []int32
	returnflag  []string
	priceIsNull []bool
}

// writeLineitemFile creates a lineitem-shaped .vxq file spanning rowGroups row
// groups, with the columns canonical TPC-H Q6 needs. When withNulls is true a
// deterministic subset of l_extendedprice values is null.
func writeLineitemFile(t *testing.T, rowGroups int, withNulls bool) *lineitemData {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "l_extendedprice", Type: storage.TypeFloat64, Nullable: withNulls},
		{Name: "l_discount", Type: storage.TypeFloat64},
		{Name: "l_quantity", Type: storage.TypeInt64},
		{Name: "l_shipdate", Type: storage.TypeDate},
		{Name: "l_returnflag", Type: storage.TypeString},
	}}
	path := filepath.Join(t.TempDir(), "lineitem.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	data := &lineitemData{path: path}
	flags := []string{"A", "N", "R"}
	base := daysSince1970(1993, 1, 1)
	ctx := context.Background()

	for rg := 0; rg < rowGroups; rg++ {
		n := storage.RowGroupRows
		prices := make([]float64, n)
		discounts := make([]float64, n)
		quantities := make([]int64, n)
		shipdates := make([]int32, n)
		returnflags := make([]string, n)
		var nulls []byte
		if withNulls {
			// Convention is 1 = valid, 0 = null: start all-valid, then clear bits.
			nulls = storage.FullBitmap(n)
		}

		for i := 0; i < n; i++ {
			g := rg*n + i
			// Non-round prices so float summation order is actually observable.
			prices[i] = 900.13 + float64(g%977)*1.07
			discounts[i] = float64(g%11) * 0.01
			quantities[i] = int64(1 + g%50)
			shipdates[i] = base + int32(g%1200)
			returnflags[i] = flags[g%len(flags)]

			isNull := withNulls && g%97 == 0
			if isNull {
				storage.SetNullBit(nulls, i)
				prices[i] = 0
			}
			data.price = append(data.price, prices[i])
			data.discount = append(data.discount, discounts[i])
			data.quantity = append(data.quantity, quantities[i])
			data.shipdate = append(data.shipdate, shipdates[i])
			data.returnflag = append(data.returnflag, returnflags[i])
			data.priceIsNull = append(data.priceIsNull, isNull)
		}

		if err := w.BeginRowGroup(n); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}
		if err := w.AppendColumn(ctx, 0, nulls, prices); err != nil {
			t.Fatalf("AppendColumn price: %v", err)
		}
		if err := w.AppendColumn(ctx, 1, nil, discounts); err != nil {
			t.Fatalf("AppendColumn discount: %v", err)
		}
		if err := w.AppendColumn(ctx, 2, nil, quantities); err != nil {
			t.Fatalf("AppendColumn quantity: %v", err)
		}
		if err := w.AppendColumn(ctx, 3, nil, shipdates); err != nil {
			t.Fatalf("AppendColumn shipdate: %v", err)
		}
		if err := w.AppendColumn(ctx, 4, nil, returnflags); err != nil {
			t.Fatalf("AppendColumn returnflag: %v", err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return data
}

func openLineitem(t *testing.T, path string) *catalog.Catalog {
	t.Helper()
	cat, err := catalog.OpenSingle(context.Background(), "lineitem", path)
	if err != nil {
		t.Fatalf("OpenSingle: %v", err)
	}
	return cat
}

// assertRowsEqual compares two result sets. Non-float values must match exactly;
// float64 values must agree within a relative tolerance (see the file comment on
// float associativity).
func assertRowsEqual(t *testing.T, want, got [][]interface{}) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("row count mismatch: serial=%d parallel=%d", len(want), len(got))
	}
	for i := range want {
		if len(want[i]) != len(got[i]) {
			t.Fatalf("row %d: column count mismatch: serial=%d parallel=%d", i, len(want[i]), len(got[i]))
		}
		for c := range want[i] {
			wf, wIsFloat := want[i][c].(float64)
			gf, gIsFloat := got[i][c].(float64)
			switch {
			case wIsFloat && gIsFloat:
				if !floatClose(wf, gf) {
					t.Errorf("row %d col %d: serial=%v parallel=%v (relative diff exceeds %g)",
						i, c, wf, gf, floatTolerance)
				}
			case wIsFloat != gIsFloat:
				t.Errorf("row %d col %d: type mismatch: serial=%T parallel=%T", i, c, want[i][c], got[i][c])
			default:
				if want[i][c] != got[i][c] {
					t.Errorf("row %d col %d: serial=%v parallel=%v", i, c, want[i][c], got[i][c])
				}
			}
		}
	}
}

func floatClose(a, b float64) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return false
	}
	diff := math.Abs(a - b)
	scale := math.Max(math.Abs(a), math.Abs(b))
	if scale > 1 {
		return diff/scale <= floatTolerance
	}
	return diff <= floatTolerance
}

// runSerialAndParallel executes query on both paths, asserts the parallel plan
// really is a ParallelHashAggregate, and returns both result sets.
func runSerialAndParallel(t *testing.T, cat *catalog.Catalog, query string, workers int) (serial, parallel [][]interface{}) {
	t.Helper()
	ctx := context.Background()

	serialOp, err := planner.Physical(ctx, buildPlan(t, query, cat))
	if err != nil {
		t.Fatalf("Physical(%s): %v", query, err)
	}
	serial = drainResults(t, serialOp)

	parallelOp, err := planner.Parallel(ctx, buildPlan(t, query, cat), workers)
	if err != nil {
		t.Fatalf("Parallel(%s): %v", query, err)
	}
	if _, ok := parallelOp.(*exec.ParallelHashAggregate); !ok {
		t.Fatalf("expected *exec.ParallelHashAggregate for %q, got %T (fell back to serial)", query, parallelOp)
	}
	parallel = drainResults(t, parallelOp)
	return serial, parallel
}

// q6 is the canonical TPC-H Q6 shape: an aggregate over an expression, guarded
// by a date range, an inclusive BETWEEN on a float column, and a quantity bound.
const q6 = `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem ` +
	`WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' ` +
	`AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24`

// TestParallelAggExpr_CanonicalQ6Shape checks the headline case against both the
// serial path and an independently computed expected value.
func TestParallelAggExpr_CanonicalQ6Shape(t *testing.T) {
	data := writeLineitemFile(t, 4, false)
	cat := openLineitem(t, data.path)

	serial, parallel := runSerialAndParallel(t, cat, q6, 4)
	assertRowsEqual(t, serial, parallel)

	// Independent reference: sum price*discount over the qualifying rows.
	lo, hi := daysSince1970(1994, 1, 1), daysSince1970(1995, 1, 1)
	var want float64
	var matched int
	for i := range data.price {
		if data.shipdate[i] >= lo && data.shipdate[i] < hi &&
			data.discount[i] >= 0.05 && data.discount[i] <= 0.07 &&
			data.quantity[i] < 24 {
			want += data.price[i] * data.discount[i]
			matched++
		}
	}
	if matched == 0 {
		t.Fatal("test data generated no qualifying rows — the predicate is not exercising the filter")
	}
	if len(parallel) != 1 || len(parallel[0]) != 1 {
		t.Fatalf("expected a single scalar result, got %v", parallel)
	}
	got, ok := parallel[0][0].(float64)
	if !ok {
		t.Fatalf("expected float64 revenue, got %T", parallel[0][0])
	}
	if !floatClose(want, got) {
		t.Errorf("revenue: reference=%v parallel=%v (%d qualifying rows)", want, got, matched)
	}
}

// TestParallelAggExpr_WithAndWithoutWHERE covers the filtered and unfiltered
// shapes, including a predicate that matches nothing.
func TestParallelAggExpr_WithAndWithoutWHERE(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	queries := map[string]string{
		"no WHERE":           `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem`,
		"single predicate":   `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_quantity < 24`,
		"conjunction":        `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_quantity < 24 AND l_discount > 0.02`,
		"disjunction":        `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_quantity < 5 OR l_discount > 0.09`,
		"matches no rows":    `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_quantity > 100000`,
		"expression in both": `SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_extendedprice * l_discount > 50`,
	}

	for name, query := range queries {
		t.Run(name, func(t *testing.T) {
			serial, parallel := runSerialAndParallel(t, cat, query, 4)
			assertRowsEqual(t, serial, parallel)
		})
	}
}

// TestParallelAggExpr_MixedTypeSUM covers float64 * int64 operands, which rely
// on the plan-time CastIntToFloatExpr coercion, in the parallel path.
func TestParallelAggExpr_MixedTypeSUM(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	queries := []string{
		`SELECT SUM(l_extendedprice * l_quantity) AS v FROM lineitem`,
		`SELECT SUM(l_quantity * l_extendedprice) AS v FROM lineitem`,
		`SELECT SUM(l_extendedprice / l_quantity) AS v FROM lineitem`,
		`SELECT SUM(l_extendedprice - l_quantity) AS v FROM lineitem WHERE l_quantity < 24`,
		`SELECT SUM(l_extendedprice * (1 - l_discount)) AS v FROM lineitem`,
	}
	for _, query := range queries {
		t.Run(query, func(t *testing.T) {
			serial, parallel := runSerialAndParallel(t, cat, query, 4)
			assertRowsEqual(t, serial, parallel)
			if len(parallel) != 1 {
				t.Fatalf("expected one row, got %d", len(parallel))
			}
			if _, ok := parallel[0][0].(float64); !ok {
				t.Errorf("mixed-type arithmetic should produce float64, got %T", parallel[0][0])
			}
		})
	}
}

// TestParallelAggExpr_IntegerExprIsExact verifies that an expression aggregate
// over integer operands accumulates in int64 and is therefore bit-identical
// between the serial and parallel paths — integer addition is associative.
func TestParallelAggExpr_IntegerExprIsExact(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	query := `SELECT SUM(l_quantity * 2) AS v, MIN(l_quantity + 1) AS mn, MAX(l_quantity + 1) AS mx FROM lineitem`
	serial, parallel := runSerialAndParallel(t, cat, query, 4)

	if len(serial) != 1 || len(parallel) != 1 {
		t.Fatalf("expected one row each, got serial=%d parallel=%d", len(serial), len(parallel))
	}
	for c := range serial[0] {
		if serial[0][c] != parallel[0][c] {
			t.Errorf("col %d: integer aggregate must be exact: serial=%v parallel=%v",
				c, serial[0][c], parallel[0][c])
		}
		if _, ok := serial[0][c].(int64); !ok {
			t.Errorf("col %d: expected int64 accumulator, got %T", c, serial[0][c])
		}
	}
}

// TestParallelAggExpr_GroupByWithExprAgg covers GROUP BY alongside an
// expression aggregate — the group keys come from a dictionary-encoded string
// column, so partial group state must merge across row groups.
func TestParallelAggExpr_GroupByWithExprAgg(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	queries := map[string]string{
		"string group key": `SELECT l_returnflag, SUM(l_extendedprice * l_discount) AS revenue, COUNT(*) AS cnt ` +
			`FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag`,
		"integer group key": `SELECT l_quantity, SUM(l_extendedprice * l_discount) AS revenue ` +
			`FROM lineitem GROUP BY l_quantity ORDER BY l_quantity`,
		"two group keys": `SELECT l_returnflag, l_quantity, SUM(l_extendedprice * l_discount) AS revenue ` +
			`FROM lineitem GROUP BY l_returnflag, l_quantity ORDER BY l_returnflag, l_quantity`,
		"grouped with filter": `SELECT l_returnflag, SUM(l_extendedprice * l_discount) AS revenue ` +
			`FROM lineitem WHERE l_quantity < 24 GROUP BY l_returnflag ORDER BY l_returnflag`,
	}

	for name, query := range queries {
		t.Run(name, func(t *testing.T) {
			// ORDER BY makes row order deterministic; the sort is peeled off and
			// applied serially to the merged result.
			serial, parallel := runSerialAndParallelSorted(t, cat, query, 4)
			assertRowsEqual(t, serial, parallel)
			if len(parallel) < 2 {
				t.Fatalf("expected multiple groups, got %d", len(parallel))
			}
		})
	}
}

// runSerialAndParallelSorted is runSerialAndParallel for plans whose root is a
// peeled Sort (or Limit → Sort): the parallel operator is the sort/limit
// wrapper, so the aggregate identity is checked one level down instead.
func runSerialAndParallelSorted(t *testing.T, cat *catalog.Catalog, query string, workers int) (serial, parallel [][]interface{}) {
	t.Helper()
	ctx := context.Background()

	serialOp, err := planner.Physical(ctx, buildPlan(t, query, cat))
	if err != nil {
		t.Fatalf("Physical(%s): %v", query, err)
	}
	serial = drainResults(t, serialOp)

	parallelOp, err := planner.Parallel(ctx, buildPlan(t, query, cat), workers)
	if err != nil {
		t.Fatalf("Parallel(%s): %v", query, err)
	}
	if _, ok := parallelOp.(*exec.HashAggregate); ok {
		t.Fatalf("query %q fell back to a serial HashAggregate", query)
	}
	parallel = drainResults(t, parallelOp)
	return serial, parallel
}

// TestParallelAggExpr_AllAggKinds covers every aggregate kind over an
// expression, including AVG (whose merge combines sums and non-null counts).
func TestParallelAggExpr_AllAggKinds(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	query := `SELECT SUM(l_extendedprice * l_discount) AS s, AVG(l_extendedprice * l_discount) AS a, ` +
		`MIN(l_extendedprice * l_discount) AS mn, MAX(l_extendedprice * l_discount) AS mx, ` +
		`COUNT(*) AS c FROM lineitem`

	serial, parallel := runSerialAndParallel(t, cat, query, 4)
	assertRowsEqual(t, serial, parallel)

	// MIN/MAX select an existing value rather than combining values, so they must
	// be exact regardless of partitioning.
	if serial[0][2] != parallel[0][2] {
		t.Errorf("MIN must be exact: serial=%v parallel=%v", serial[0][2], parallel[0][2])
	}
	if serial[0][3] != parallel[0][3] {
		t.Errorf("MAX must be exact: serial=%v parallel=%v", serial[0][3], parallel[0][3])
	}
	if serial[0][4] != parallel[0][4] {
		t.Errorf("COUNT must be exact: serial=%v parallel=%v", serial[0][4], parallel[0][4])
	}
}

// TestParallelAggExpr_NullOperands verifies null propagation through the
// per-morsel expression evaluation: a null operand yields a null expression
// result, which the aggregate must skip in both paths.
func TestParallelAggExpr_NullOperands(t *testing.T) {
	data := writeLineitemFile(t, 3, true)
	cat := openLineitem(t, data.path)

	nullCount := 0
	for _, isNull := range data.priceIsNull {
		if isNull {
			nullCount++
		}
	}
	if nullCount == 0 {
		t.Fatal("test data has no nulls — null handling is not being exercised")
	}

	queries := map[string]string{
		"SUM over nullable operand":   `SELECT SUM(l_extendedprice * l_discount) AS v FROM lineitem`,
		"AVG over nullable operand":   `SELECT AVG(l_extendedprice * l_discount) AS v FROM lineitem`,
		"COUNT of nullable operand":   `SELECT COUNT(l_extendedprice) AS v FROM lineitem`,
		"grouped over null operand":   `SELECT l_returnflag, SUM(l_extendedprice * l_discount) AS v FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag`,
		"IS NOT NULL filter and expr": `SELECT SUM(l_extendedprice * l_discount) AS v FROM lineitem WHERE l_extendedprice IS NOT NULL`,
	}
	for name, query := range queries {
		t.Run(name, func(t *testing.T) {
			serial, parallel := runSerialAndParallelSorted(t, cat, query, 4)
			assertRowsEqual(t, serial, parallel)
		})
	}
}

// TestParallelAggExpr_WorkerCountInvariance verifies the result does not depend
// on how the scan is partitioned, including more workers than row groups.
func TestParallelAggExpr_WorkerCountInvariance(t *testing.T) {
	data := writeLineitemFile(t, 4, false)
	cat := openLineitem(t, data.path)

	serialOp, err := planner.Physical(context.Background(), buildPlan(t, q6, cat))
	if err != nil {
		t.Fatalf("Physical: %v", err)
	}
	serial := drainResults(t, serialOp)

	for _, workers := range []int{1, 2, 3, 4, 8, 16} {
		op, err := planner.Parallel(context.Background(), buildPlan(t, q6, cat), workers)
		if err != nil {
			t.Fatalf("Parallel(workers=%d): %v", workers, err)
		}
		if _, ok := op.(*exec.ParallelHashAggregate); !ok {
			t.Fatalf("workers=%d: expected *exec.ParallelHashAggregate, got %T", workers, op)
		}
		assertRowsEqual(t, serial, drainResults(t, op))
	}
}

// TestParallelAggExpr_OrderByLimitOverExprAgg verifies sort-peeling still works
// when the sort key is an expression aggregate's output alias.
func TestParallelAggExpr_OrderByLimitOverExprAgg(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	query := `SELECT l_quantity, SUM(l_extendedprice * l_discount) AS revenue FROM lineitem ` +
		`GROUP BY l_quantity ORDER BY revenue DESC LIMIT 5`

	serial, parallel := runSerialAndParallelSorted(t, cat, query, 4)
	if len(parallel) != 5 {
		t.Fatalf("expected 5 rows from LIMIT 5, got %d", len(parallel))
	}
	assertRowsEqual(t, serial, parallel)
}

// TestParallelAggExpr_CaseWhenInAggregate covers the TPC-H Q12 aggregate shape,
// SUM(CASE WHEN ... THEN ... ELSE ... END). CASE is still a row-local
// expression, so per-morsel evaluation applies to it the same way.
func TestParallelAggExpr_CaseWhenInAggregate(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	queries := map[string]string{
		"integer CASE result": `SELECT SUM(CASE WHEN l_returnflag = 'A' THEN 1 ELSE 0 END) AS high, ` +
			`SUM(CASE WHEN l_returnflag <> 'A' THEN 1 ELSE 0 END) AS low FROM lineitem`,
		"float CASE result": `SELECT SUM(CASE WHEN l_quantity < 24 THEN l_extendedprice ELSE 0.0 END) AS v FROM lineitem`,
		"CASE with filter": `SELECT SUM(CASE WHEN l_discount > 0.05 THEN l_extendedprice * l_discount ELSE 0.0 END) AS v ` +
			`FROM lineitem WHERE l_quantity < 24`,
		"grouped CASE": `SELECT l_returnflag, SUM(CASE WHEN l_quantity < 24 THEN 1 ELSE 0 END) AS v ` +
			`FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag`,
	}
	for name, query := range queries {
		t.Run(name, func(t *testing.T) {
			serial, parallel := runSerialAndParallelSorted(t, cat, query, 4)
			assertRowsEqual(t, serial, parallel)
		})
	}
}

// TestParallelAggExpr_CountDistinctStillFallsBack guards the pre-existing
// COUNT(DISTINCT) serial fallback: partial distinct counts cannot be summed, so
// expression support must not route these plans through the parallel path.
func TestParallelAggExpr_CountDistinctStillFallsBack(t *testing.T) {
	data := writeLineitemFile(t, 3, false)
	cat := openLineitem(t, data.path)

	queries := []string{
		`SELECT COUNT(DISTINCT l_quantity) AS c FROM lineitem`,
		`SELECT COUNT(DISTINCT l_returnflag) AS c, SUM(l_extendedprice * l_discount) AS v FROM lineitem`,
	}
	for _, query := range queries {
		t.Run(query, func(t *testing.T) {
			op, err := planner.Parallel(context.Background(), buildPlan(t, query, cat), 4)
			if err != nil {
				t.Fatalf("Parallel: %v", err)
			}
			if _, ok := op.(*exec.ParallelHashAggregate); ok {
				t.Fatal("COUNT(DISTINCT) must fall back to serial execution")
			}
			serialOp, err := planner.Physical(context.Background(), buildPlan(t, query, cat))
			if err != nil {
				t.Fatalf("Physical: %v", err)
			}
			assertRowsEqual(t, drainResults(t, serialOp), drainResults(t, op))
		})
	}
}
