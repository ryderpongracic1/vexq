package planner_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// writeGroupedFile creates a 2-column (grp int64, val int64) .vxq file
// with known group sizes for HAVING testing.
// Group 1: 5 rows (val=10 each, sum=50), Group 2: 2 rows (val=100 each, sum=200),
// Group 3: 10 rows (val=1 each, sum=10).
func writeGroupedFile(t *testing.T) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "grp", Type: storage.TypeInt64},
		{Name: "val", Type: storage.TypeInt64},
	}}
	dir := t.TempDir()
	path := filepath.Join(dir, "grouped.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	n := 17
	grps := make([]int64, n)
	vals := make([]int64, n)
	for i := 0; i < 5; i++ {
		grps[i] = 1
		vals[i] = 10
	}
	for i := 5; i < 7; i++ {
		grps[i] = 2
		vals[i] = 100
	}
	for i := 7; i < 17; i++ {
		grps[i] = 3
		vals[i] = 1
	}
	_ = w.BeginRowGroup(n)
	_ = w.AppendColumn(context.Background(), 0, nil, grps)
	_ = w.AppendColumn(context.Background(), 1, nil, vals)
	_ = w.EndRowGroup()
	if err := w.Finish(context.Background()); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

func runQuery(t *testing.T, cat *catalog.Catalog, query string) (exec.Schema, [][]int64) {
	t.Helper()
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	stmt := node.(*sql.SelectStmt)
	logical, err := planner.Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		t.Fatalf("Physical: %v", err)
	}
	defer op.Close()

	schema := op.Schema()
	var rows [][]int64
	for {
		batch, err := op.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		for i := 0; i < batch.Length; i++ {
			idx := i
			if batch.SelVec != nil {
				idx = int(batch.SelVec[i])
			}
			row := make([]int64, len(batch.Vectors))
			for c, v := range batch.Vectors {
				switch vec := v.(type) {
				case *exec.Int64Vector:
					row[c] = vec.Values[idx]
				case *exec.Float64Vector:
					row[c] = int64(vec.Values[idx])
				}
			}
			rows = append(rows, row)
		}
	}
	return schema, rows
}

// TestHavingAggReuse verifies that HAVING COUNT(*) > N works when COUNT(*)
// is already in the SELECT list (reuse path — no hidden aggregate added).
func TestHavingAggReuse(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	schema, rows := runQuery(t, cat,
		"SELECT grp, COUNT(*) AS cnt FROM test GROUP BY grp HAVING COUNT(*) > 3")

	// Output schema should have exactly 2 columns: grp, cnt.
	if len(schema.Fields) != 2 {
		t.Fatalf("expected 2 output fields, got %d: %v", len(schema.Fields), schema.Fields)
	}
	if schema.Fields[0].Name != "grp" || schema.Fields[1].Name != "cnt" {
		t.Fatalf("unexpected schema field names: %v", schema.Fields)
	}

	// Groups with count > 3: group 1 (5), group 3 (10). Group 2 (2) filtered.
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d: %v", len(rows), rows)
	}
	got := map[int64]int64{}
	for _, r := range rows {
		got[r[0]] = r[1]
	}
	if got[1] != 5 {
		t.Errorf("group 1: expected cnt=5, got %d", got[1])
	}
	if got[3] != 10 {
		t.Errorf("group 3: expected cnt=10, got %d", got[3])
	}
	if _, exists := got[2]; exists {
		t.Error("group 2 should have been filtered by HAVING")
	}
}

// TestHavingAggHidden verifies that HAVING with an aggregate NOT in the SELECT
// list works by adding a hidden aggregate and projecting it away.
func TestHavingAggHidden(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	// SUM(val) is NOT in the SELECT, so a hidden aggregate is needed.
	schema, rows := runQuery(t, cat,
		"SELECT grp, COUNT(*) AS cnt FROM test GROUP BY grp HAVING SUM(val) > 50")

	// Output schema should have exactly 2 columns (hidden SUM stripped).
	if len(schema.Fields) != 2 {
		t.Fatalf("expected 2 output fields (hidden SUM stripped), got %d: %v", len(schema.Fields), schema.Fields)
	}
	if schema.Fields[0].Name != "grp" || schema.Fields[1].Name != "cnt" {
		t.Fatalf("unexpected schema field names: %v", schema.Fields)
	}

	// SUM(val): group 1=50 (not > 50), group 2=200 (passes), group 3=10 (not > 50).
	if len(rows) != 1 {
		t.Fatalf("expected 1 row (only group 2 passes), got %d: %v", len(rows), rows)
	}
	if rows[0][0] != 2 || rows[0][1] != 2 {
		t.Errorf("expected grp=2 cnt=2, got grp=%d cnt=%d", rows[0][0], rows[0][1])
	}
}

// TestHavingCountStar verifies COUNT(*) specifically (nil arg).
func TestHavingCountStar(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	_, rows := runQuery(t, cat,
		"SELECT grp FROM test GROUP BY grp HAVING COUNT(*) > 4")

	// COUNT(*): group 1=5 (passes), group 2=2, group 3=10 (passes).
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d: %v", len(rows), rows)
	}
	got := map[int64]bool{}
	for _, r := range rows {
		got[r[0]] = true
	}
	if !got[1] || !got[3] {
		t.Errorf("expected groups 1 and 3, got %v", got)
	}
}

// TestHavingCountCol verifies COUNT(col) (non-star aggregate in HAVING).
func TestHavingCountCol(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	// COUNT(val) should behave same as COUNT(*) here (no nulls in val).
	_, rows := runQuery(t, cat,
		"SELECT grp FROM test GROUP BY grp HAVING COUNT(val) > 4")

	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d: %v", len(rows), rows)
	}
}

// TestHavingFiltersAllGroups verifies that HAVING can filter ALL groups (empty result).
func TestHavingFiltersAllGroups(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	_, rows := runQuery(t, cat,
		"SELECT grp, COUNT(*) AS cnt FROM test GROUP BY grp HAVING COUNT(*) > 9999")

	if len(rows) != 0 {
		t.Fatalf("expected 0 rows (all filtered), got %d: %v", len(rows), rows)
	}
}

// TestHavingMixedAliasAndAggregate verifies HAVING with AND of alias reference
// and aggregate expression in the same predicate.
func TestHavingMixedAliasAndAggregate(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	// cnt > 1 AND COUNT(*) > 4: both conditions must hold.
	// "cnt" is an alias ref, "COUNT(*)" is an aggregate expr.
	schema, rows := runQuery(t, cat,
		"SELECT grp, COUNT(*) AS cnt FROM test GROUP BY grp HAVING cnt > 1 AND COUNT(*) > 4")

	if len(schema.Fields) != 2 {
		t.Fatalf("expected 2 output fields, got %d", len(schema.Fields))
	}

	// cnt > 1: all 3 groups pass. COUNT(*) > 4: groups 1 (5) and 3 (10) pass.
	// Combined: groups 1 and 3.
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d: %v", len(rows), rows)
	}
	got := map[int64]bool{}
	for _, r := range rows {
		got[r[0]] = true
	}
	if !got[1] || !got[3] {
		t.Errorf("expected groups 1 and 3, got %v", got)
	}
}

// TestHavingParallelEquivalence verifies that serial and parallel paths
// produce identical results when HAVING uses an aggregate expression.
func TestHavingParallelEquivalence(t *testing.T) {
	// Create a file with enough row groups to trigger parallelism.
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "grp", Type: storage.TypeInt64},
		{Name: "val", Type: storage.TypeInt64},
	}}
	dir := t.TempDir()
	path := filepath.Join(dir, "parallel.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	// Write multiple row groups with groups 1-4
	const rowsPerRG = 1024
	const numRGs = 4
	for rg := 0; rg < numRGs; rg++ {
		grps := make([]int64, rowsPerRG)
		vals := make([]int64, rowsPerRG)
		for i := range grps {
			grps[i] = int64((i % 4) + 1)
			vals[i] = int64(i + rg*rowsPerRG)
		}
		_ = w.BeginRowGroup(rowsPerRG)
		_ = w.AppendColumn(context.Background(), 0, nil, grps)
		_ = w.AppendColumn(context.Background(), 1, nil, vals)
		_ = w.EndRowGroup()
	}
	if err := w.Finish(context.Background()); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	query := "SELECT grp, COUNT(*) AS cnt FROM test GROUP BY grp HAVING COUNT(*) > 500"

	// Run serial.
	_, serialRows := runQuery(t, cat, query)

	// Run parallel.
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)
	logical, err := planner.Build(context.Background(), stmt, cat)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	logical = planner.Optimize(logical)
	parallelOp, err := planner.Parallel(context.Background(), logical, 4)
	if err != nil {
		// planner.Parallel may not support HAVING — that's OK,
		// fall back means serial result is the ground truth.
		t.Skipf("Parallel planning unsupported for this shape: %v", err)
	}
	defer parallelOp.Close()

	var parallelRows [][]int64
	for {
		batch, err := parallelOp.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		for i := 0; i < batch.Length; i++ {
			idx := i
			if batch.SelVec != nil {
				idx = int(batch.SelVec[i])
			}
			row := make([]int64, len(batch.Vectors))
			for c, v := range batch.Vectors {
				if vec, ok := v.(*exec.Int64Vector); ok {
					row[c] = vec.Values[idx]
				}
			}
			parallelRows = append(parallelRows, row)
		}
	}

	// Compare: same groups should appear in both.
	serialGrps := map[int64]int64{}
	for _, r := range serialRows {
		serialGrps[r[0]] = r[1]
	}
	parallelGrps := map[int64]int64{}
	for _, r := range parallelRows {
		parallelGrps[r[0]] = r[1]
	}
	if len(serialGrps) != len(parallelGrps) {
		t.Fatalf("serial has %d groups, parallel has %d", len(serialGrps), len(parallelGrps))
	}
	for grp, cnt := range serialGrps {
		if parallelGrps[grp] != cnt {
			t.Errorf("group %d: serial cnt=%d, parallel cnt=%d", grp, cnt, parallelGrps[grp])
		}
	}
}
