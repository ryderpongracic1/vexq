package planner_test

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

var ctx = context.Background()

// writeTestFile creates a 2-column (id int64, val float64) .vxq file with n rows.
func writeTestFile(t *testing.T, n int) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "id", Type: storage.TypeInt64},
		{Name: "val", Type: storage.TypeFloat64},
	}}
	path := filepath.Join(t.TempDir(), "test.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ids := make([]int64, n)
	vals := make([]float64, n)
	for i := range ids {
		ids[i] = int64(i)
		vals[i] = float64(i) * 2.5
	}
	_ = w.BeginRowGroup(n)
	_ = w.AppendColumn(ctx, 0, nil, ids)
	_ = w.AppendColumn(ctx, 1, nil, vals)
	_ = w.EndRowGroup()
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

func TestAggregateInvalidColumn(t *testing.T) {
	// A query referencing a nonexistent column in SUM should produce a planner
	// error, not a panic in the execution layer.
	path := writeTestFile(t, 10)
	cat, err := catalog.OpenSingle(ctx, "test", path)
	if err != nil {
		t.Fatal(err)
	}

	query := `SELECT SUM(nonexistent_column) FROM test`
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	logical = planner.Optimize(logical)

	_, err = planner.Physical(ctx, logical)
	if err == nil {
		t.Fatal("expected planner error for nonexistent aggregate column, got nil")
	}
	if !strings.Contains(err.Error(), "not found in schema") {
		t.Fatalf("unexpected error message: %v", err)
	}
}
