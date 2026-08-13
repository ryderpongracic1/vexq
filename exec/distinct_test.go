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

// writeDistinctTestFile creates a .vxq file with duplicate rows:
// id: [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
// val: [10, 20, 20, 30, 30, 30, 40, 40, 40, 40]
func writeDistinctTestFile(t *testing.T) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "id", Type: storage.TypeInt64},
		{Name: "val", Type: storage.TypeInt64},
	}}
	path := filepath.Join(t.TempDir(), "distinct_test.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ctx := context.Background()
	ids := []int64{1, 2, 2, 3, 3, 3, 4, 4, 4, 4}
	vals := []int64{10, 20, 20, 30, 30, 30, 40, 40, 40, 40}
	_ = w.BeginRowGroup(len(ids))
	_ = w.AppendColumn(ctx, 0, nil, ids)
	_ = w.AppendColumn(ctx, 1, nil, vals)
	_ = w.EndRowGroup()
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

func runQuery(t *testing.T, path, query string) [][]int64 {
	t.Helper()
	ctx := context.Background()

	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	cat, err := catalog.OpenSingle(ctx, "distinct_test", path)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(ctx, logical)
	if err != nil {
		t.Fatalf("physical: %v", err)
	}
	defer op.Close()

	var results [][]int64
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		if batch.SelVec != nil {
			for _, ri := range batch.SelVec {
				row := make([]int64, len(batch.Vectors))
				for j, vec := range batch.Vectors {
					row[j] = vec.(*exec.Int64Vector).Values[ri]
				}
				results = append(results, row)
			}
		} else {
			for i := 0; i < batch.Length; i++ {
				row := make([]int64, len(batch.Vectors))
				for j, vec := range batch.Vectors {
					row[j] = vec.(*exec.Int64Vector).Values[i]
				}
				results = append(results, row)
			}
		}
	}
	return results
}

func TestSelectDistinct(t *testing.T) {
	path := writeDistinctTestFile(t)

	// Without DISTINCT: should return all 10 rows.
	allRows := runQuery(t, path, "SELECT id, val FROM distinct_test")
	if len(allRows) != 10 {
		t.Fatalf("expected 10 rows without DISTINCT, got %d", len(allRows))
	}

	// With DISTINCT: should return 4 unique rows: (1,10), (2,20), (3,30), (4,40).
	distinctRows := runQuery(t, path, "SELECT DISTINCT id, val FROM distinct_test")
	if len(distinctRows) != 4 {
		t.Fatalf("expected 4 distinct rows, got %d", len(distinctRows))
	}

	// Verify expected values.
	expected := map[[2]int64]bool{
		{1, 10}: true,
		{2, 20}: true,
		{3, 30}: true,
		{4, 40}: true,
	}
	for _, row := range distinctRows {
		key := [2]int64{row[0], row[1]}
		if !expected[key] {
			t.Errorf("unexpected distinct row: %v", row)
		}
		delete(expected, key)
	}
	if len(expected) > 0 {
		t.Errorf("missing distinct rows: %v", expected)
	}
}

func TestDistinctWithLimit(t *testing.T) {
	path := writeDistinctTestFile(t)

	// DISTINCT + LIMIT 2: should return exactly 2 unique rows.
	rows := runQuery(t, path, "SELECT DISTINCT id, val FROM distinct_test LIMIT 2")
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows with DISTINCT LIMIT 2, got %d", len(rows))
	}

	// Verify all returned rows are from the expected set.
	valid := map[[2]int64]bool{
		{1, 10}: true,
		{2, 20}: true,
		{3, 30}: true,
		{4, 40}: true,
	}
	for _, row := range rows {
		key := [2]int64{row[0], row[1]}
		if !valid[key] {
			t.Errorf("unexpected row from DISTINCT LIMIT: %v", row)
		}
	}
}

func TestCountDistinctError(t *testing.T) {
	path := writeDistinctTestFile(t)
	ctx := context.Background()

	query := "SELECT COUNT(DISTINCT id) FROM distinct_test"
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	cat, err := catalog.OpenSingle(ctx, "distinct_test", path)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	_, err = planner.Build(ctx, stmt, cat)
	if err == nil {
		t.Fatal("expected error for COUNT(DISTINCT ...), got nil")
	}
	if !strings.Contains(err.Error(), "DISTINCT aggregates not yet supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}
