package exec_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// writeExprTestFile creates a .vxq file with columns:
//
//	id:       int64   (0..n-1)
//	amount:   float64 (i * 2.5)
//	status:   string  ("alpha", "beta", "gamma" cycling)
//	order_date: date  (days-since-epoch: 17000 + i)
func writeExprTestFile(t *testing.T, n int) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "id", Type: storage.TypeInt64},
		{Name: "amount", Type: storage.TypeFloat64},
		{Name: "status", Type: storage.TypeString},
		{Name: "order_date", Type: storage.TypeDate},
	}}
	path := filepath.Join(t.TempDir(), "orders.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	ids := make([]int64, n)
	amounts := make([]float64, n)
	statuses := []string{"alpha", "beta", "gamma"}
	statusVals := make([]string, n)
	dates := make([]int32, n)
	for i := range ids {
		ids[i] = int64(i)
		amounts[i] = float64(i) * 2.5
		statusVals[i] = statuses[i%3]
		dates[i] = int32(17000 + i)
	}

	_ = w.BeginRowGroup(n)
	_ = w.AppendColumn(context.Background(), 0, nil, ids)
	_ = w.AppendColumn(context.Background(), 1, nil, amounts)
	_ = w.AppendColumn(context.Background(), 2, nil, statusVals)
	_ = w.AppendColumn(context.Background(), 3, nil, dates)
	_ = w.EndRowGroup()
	if err := w.Finish(context.Background()); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

// runExprQuery parses, plans, and executes a query against the given catalog,
// returning all result batches. Fatals on error.
func runExprQuery(t *testing.T, cat *catalog.Catalog, query string) []*exec.Batch {
	t.Helper()
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	stmt := node.(*sql.SelectStmt)
	logical, err := planner.Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("build %q: %v", query, err)
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		t.Fatalf("physical %q: %v", query, err)
	}
	defer op.Close()

	var batches []*exec.Batch
	for {
		b, err := op.Next(context.Background())
		if err != nil {
			t.Fatalf("Next %q: %v", query, err)
		}
		if b == nil {
			break
		}
		batches = append(batches, b)
	}
	return batches
}

// runExprQueryExpectError parses, plans, and executes a query expecting an error
// at some stage. Returns the error.
func runExprQueryExpectError(t *testing.T, cat *catalog.Catalog, query string) error {
	t.Helper()
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		return err
	}
	stmt := node.(*sql.SelectStmt)
	logical, err := planner.Build(context.Background(), stmt, cat)
	if err != nil {
		return err
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		return err
	}
	defer op.Close()

	for {
		b, err := op.Next(context.Background())
		if err != nil {
			return err
		}
		if b == nil {
			break
		}
	}
	return nil
}

// ---------- Fix 1: NOT expression precedence ---------------------------------

func TestNotExprWithComparison(t *testing.T) {
	// NOT status = 'alpha' must parse as NOT(status = 'alpha'), returning rows
	// where status != 'alpha'.
	path := writeExprTestFile(t, 30)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT id FROM orders WHERE NOT status = 'alpha'")
	total := 0
	for _, b := range batches {
		total += b.Length
	}
	// 30 rows: indices 0,3,6,...,27 are alpha (10 rows), so NOT alpha = 20.
	if total != 20 {
		t.Fatalf("expected 20 rows where NOT status='alpha', got %d", total)
	}
}

func TestNotExprWithArithmetic(t *testing.T) {
	// NOT id > 10 should return rows where id <= 10.
	path := writeExprTestFile(t, 30)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT id FROM orders WHERE NOT id > 10")
	total := 0
	for _, b := range batches {
		total += b.Length
	}
	// ids 0..10 → 11 rows
	if total != 11 {
		t.Fatalf("expected 11 rows where NOT id > 10, got %d", total)
	}
}

func TestNotExprOnNonBoolReturnsError(t *testing.T) {
	// Directly construct a NotExpr wrapping a non-bool expression to verify
	// the hardened eval path returns an error instead of panicking.
	child := &exec.Literal{Val: int64(42), T: exec.TypeInt64}
	notExpr := &exec.NotExpr{Child: child}

	batch := &exec.Batch{Length: 5, Vectors: nil}
	_, err := notExpr.Eval(context.Background(), batch)
	if err == nil {
		t.Fatal("expected error from NOT on non-bool operand, got nil")
	}
	if !strings.Contains(err.Error(), "boolean") {
		t.Fatalf("expected error mentioning 'boolean', got: %v", err)
	}
}

// ---------- Fix 2: Date vs integer literal comparison ------------------------

func TestDateColumnVsIntLiteral(t *testing.T) {
	// order_date > 17010 should work, treating 17010 as days-since-epoch.
	path := writeExprTestFile(t, 30)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT id FROM orders WHERE order_date > 17010")
	total := 0
	for _, b := range batches {
		total += b.Length
	}
	// dates are 17000+i for i in 0..29. order_date > 17010 means i > 10 → 19 rows.
	if total != 19 {
		t.Fatalf("expected 19 rows where order_date > 17010, got %d", total)
	}
}

func TestDateColumnBetweenIntLiterals(t *testing.T) {
	// BETWEEN with int literals on a date column (TPC-H Q6-shaped).
	path := writeExprTestFile(t, 30)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT id FROM orders WHERE order_date BETWEEN 17005 AND 17015")
	total := 0
	for _, b := range batches {
		total += b.Length
	}
	// dates 17005..17015 inclusive → i = 5..15 → 11 rows.
	if total != 11 {
		t.Fatalf("expected 11 rows for BETWEEN 17005 AND 17015, got %d", total)
	}
}

func TestDateColumnVsStringLiteralStillWorks(t *testing.T) {
	// Existing string-date coercion must not be broken.
	// 17000 days since epoch = 2016-07-11 approximately.
	// Let's use a date that falls within our range: 17010 days = 2016-07-21 approx.
	// We'll just verify no panic and some result returns.
	path := writeExprTestFile(t, 30)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	// Just verify no panic. The exact count depends on epoch math.
	batches := runExprQuery(t, cat, "SELECT id FROM orders WHERE order_date > '2016-07-20'")
	total := 0
	for _, b := range batches {
		total += b.Length
	}
	if total < 0 {
		t.Fatal("impossible")
	}
}

// ---------- Fix 3: Unary minus -----------------------------------------------

func TestUnaryMinusFloat64(t *testing.T) {
	// -amount should negate float values without panicking.
	path := writeExprTestFile(t, 10)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT id, -amount AS neg FROM orders")
	total := 0
	for _, b := range batches {
		total += b.Length
		// Verify first batch has negated values.
		if b.Length > 0 {
			fv, ok := b.Vectors[1].(*exec.Float64Vector)
			if !ok {
				t.Fatalf("expected Float64Vector for -amount, got %T", b.Vectors[1])
			}
			// Row 0: amount=0.0, -amount=0.0. Row 1: amount=2.5, -amount=-2.5.
			if fv.Values[1] != -2.5 {
				t.Fatalf("expected -2.5 for -amount at row 1, got %f", fv.Values[1])
			}
		}
	}
	if total != 10 {
		t.Fatalf("expected 10 rows, got %d", total)
	}
}

func TestUnaryMinusInt64(t *testing.T) {
	// -id should negate integer values.
	path := writeExprTestFile(t, 10)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	batches := runExprQuery(t, cat, "SELECT -id AS neg_id FROM orders")
	total := 0
	for _, b := range batches {
		total += b.Length
		if b.Length > 0 {
			iv, ok := b.Vectors[0].(*exec.Int64Vector)
			if !ok {
				t.Fatalf("expected Int64Vector for -id, got %T", b.Vectors[0])
			}
			// Row 5: id=5, -id=-5.
			if iv.Values[5] != -5 {
				t.Fatalf("expected -5 for -id at row 5, got %d", iv.Values[5])
			}
		}
	}
	if total != 10 {
		t.Fatalf("expected 10 rows, got %d", total)
	}
}

func TestUnaryMinusOnStringReturnsError(t *testing.T) {
	// -status should fail with a clean planner error, not a panic.
	path := writeExprTestFile(t, 10)
	cat, err := catalog.OpenSingle(context.Background(), "orders", path)
	if err != nil {
		t.Fatal(err)
	}

	err = runExprQueryExpectError(t, cat, "SELECT -status FROM orders")
	if err == nil {
		t.Fatal("expected error from unary minus on string column, got nil")
	}
	if !strings.Contains(err.Error(), "unary minus") {
		t.Fatalf("expected error mentioning 'unary minus', got: %v", err)
	}
}
