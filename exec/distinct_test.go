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

	tableName := strings.TrimSuffix(filepath.Base(path), ".vxq")
	cat, err := catalog.OpenSingle(ctx, tableName, path)
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

func TestCountDistinct(t *testing.T) {
	path := writeDistinctTestFile(t)

	// COUNT(DISTINCT id): ids are [1,2,2,3,3,3,4,4,4,4] → 4 distinct values.
	rows := runQuery(t, path, "SELECT COUNT(DISTINCT id) FROM distinct_test")
	if len(rows) != 1 || rows[0][0] != 4 {
		t.Fatalf("COUNT(DISTINCT id): got %v, want [[4]]", rows)
	}

	// COUNT(DISTINCT val): vals are [10,20,20,30,30,30,40,40,40,40] → 4 distinct values.
	rows = runQuery(t, path, "SELECT COUNT(DISTINCT val) FROM distinct_test")
	if len(rows) != 1 || rows[0][0] != 4 {
		t.Fatalf("COUNT(DISTINCT val): got %v, want [[4]]", rows)
	}
}

func TestSumDistinctError(t *testing.T) {
	// SUM(DISTINCT ...) should still be rejected.
	query := "SELECT SUM(DISTINCT id) FROM distinct_test"
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	path := writeDistinctTestFile(t)
	ctx := context.Background()
	cat, err := catalog.OpenSingle(ctx, "distinct_test", path)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	_, err = planner.Build(ctx, stmt, cat)
	if err == nil {
		t.Fatal("expected error for SUM(DISTINCT ...), got nil")
	}
	if !strings.Contains(err.Error(), "is not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- COUNT(DISTINCT) focused tests ---

// writeMultiRowGroupFile creates a .vxq file spanning multiple row groups.
// Each row group has the same string values with potentially different local
// dictionary codes (testing cross-rowgroup dict-code stability).
// Schema: category (string), amount (int64)
func writeMultiRowGroupFile(t *testing.T) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "category", Type: storage.TypeString},
		{Name: "amount", Type: storage.TypeInt64},
	}}
	path := filepath.Join(t.TempDir(), "multi_rg.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ctx := context.Background()

	// Row group 1: categories in order [alpha, beta, alpha, gamma]
	cats1 := []string{"alpha", "beta", "alpha", "gamma"}
	amts1 := []int64{10, 20, 30, 40}
	_ = w.BeginRowGroup(len(cats1))
	_ = w.AppendColumn(ctx, 0, nil, cats1)
	_ = w.AppendColumn(ctx, 1, nil, amts1)
	_ = w.EndRowGroup()

	// Row group 2: categories in different order [gamma, beta, beta, alpha]
	// Different insertion order means different local dict codes.
	cats2 := []string{"gamma", "beta", "beta", "alpha"}
	amts2 := []int64{50, 60, 70, 80}
	_ = w.BeginRowGroup(len(cats2))
	_ = w.AppendColumn(ctx, 0, nil, cats2)
	_ = w.AppendColumn(ctx, 1, nil, amts2)
	_ = w.EndRowGroup()

	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

// writeNullableFile creates a .vxq file with NULLs in both the grouping and
// aggregate columns.
// Schema: grp (string), val (int64)
func writeNullableFile(t *testing.T) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "grp", Type: storage.TypeString},
		{Name: "val", Type: storage.TypeInt64},
	}}
	path := filepath.Join(t.TempDir(), "nullable.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ctx := context.Background()

	// 8 rows: group "a" has vals [1, 2, NULL, 1], group "b" has vals [NULL, NULL, NULL, 3]
	grps := []string{"a", "a", "a", "a", "b", "b", "b", "b"}
	vals := []int64{1, 2, 0, 1, 0, 0, 0, 3}
	// Null bitmap convention: bit=1 means valid, bit=0 means null (LSB-first).
	// Start with all-valid, then mark specific rows null.
	valNulls := storage.FullBitmap(8) // all valid
	storage.SetNullBit(valNulls, 2)   // row 2 null (group "a")
	storage.SetNullBit(valNulls, 4)   // row 4 null (group "b")
	storage.SetNullBit(valNulls, 5)   // row 5 null (group "b")
	storage.SetNullBit(valNulls, 6)   // row 6 null (group "b")

	_ = w.BeginRowGroup(len(grps))
	_ = w.AppendColumn(ctx, 0, nil, grps)
	_ = w.AppendColumn(ctx, 1, valNulls, vals)
	_ = w.EndRowGroup()

	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

func TestCountDistinctMultiRowGroup(t *testing.T) {
	path := writeMultiRowGroupFile(t)

	// categories across both row groups: [alpha, beta, gamma] → 3 distinct
	rows := runQuery(t, path, "SELECT COUNT(DISTINCT category) FROM multi_rg")
	if len(rows) != 1 || rows[0][0] != 3 {
		t.Fatalf("COUNT(DISTINCT category): got %v, want [[3]]", rows)
	}

	// amounts across both row groups: [10,20,30,40,50,60,70,80] → 8 distinct
	rows = runQuery(t, path, "SELECT COUNT(DISTINCT amount) FROM multi_rg")
	if len(rows) != 1 || rows[0][0] != 8 {
		t.Fatalf("COUNT(DISTINCT amount): got %v, want [[8]]", rows)
	}
}

func TestCountDistinctDictCodeStability(t *testing.T) {
	// The key test: same string "alpha" appears in both row groups with
	// potentially different local dictionary codes. COUNT(DISTINCT) must
	// resolve to the actual string value, not the dict code.
	path := writeMultiRowGroupFile(t)
	rows := runQuery(t, path, "SELECT COUNT(DISTINCT category) FROM multi_rg")
	if len(rows) != 1 || rows[0][0] != 3 {
		t.Fatalf("dict-code stability: got %v, want [[3]] (alpha, beta, gamma)", rows)
	}
}

func TestCountDistinctNullsExcluded(t *testing.T) {
	path := writeNullableFile(t)

	// Global COUNT(DISTINCT val): non-null values are [1, 2, 1, 3] → 3 distinct
	rows := runQuery(t, path, "SELECT COUNT(DISTINCT val) FROM nullable")
	if len(rows) != 1 || rows[0][0] != 3 {
		t.Fatalf("COUNT(DISTINCT val) with nulls: got %v, want [[3]]", rows)
	}
}

func TestCountDistinctAllNullGroup(t *testing.T) {
	path := writeNullableFile(t)

	// GROUP BY grp + COUNT(DISTINCT val):
	// group "a": non-null vals [1, 2, 1] → 2 distinct
	// group "b": non-null vals [3] → 1 distinct (3 nulls excluded)
	rows := runQueryStrings(t, path, "SELECT grp, COUNT(DISTINCT val) FROM nullable GROUP BY grp")
	got := make(map[string]int64)
	for _, row := range rows {
		got[row.str] = row.val
	}
	if got["a"] != 2 {
		t.Errorf("group 'a': got %d, want 2", got["a"])
	}
	if got["b"] != 1 {
		t.Errorf("group 'b': got %d, want 1", got["b"])
	}
}

// writeAllNullFile creates a file where a column is entirely NULL.
func writeAllNullFile(t *testing.T) string {
	t.Helper()
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "id", Type: storage.TypeInt64},
	}}
	path := filepath.Join(t.TempDir(), "allnull.vxq")
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ctx := context.Background()

	ids := []int64{0, 0, 0, 0}
	nulls := storage.FullBitmap(4) // start all-valid
	storage.SetNullBit(nulls, 0)   // mark all as null
	storage.SetNullBit(nulls, 1)
	storage.SetNullBit(nulls, 2)
	storage.SetNullBit(nulls, 3)

	_ = w.BeginRowGroup(4)
	_ = w.AppendColumn(ctx, 0, nulls, ids)
	_ = w.EndRowGroup()

	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

func TestCountDistinctAllNull(t *testing.T) {
	path := writeAllNullFile(t)
	// COUNT(DISTINCT) over all-null values → 0
	rows := runQuery(t, path, "SELECT COUNT(DISTINCT id) FROM allnull")
	if len(rows) != 1 || rows[0][0] != 0 {
		t.Fatalf("COUNT(DISTINCT) all-null: got %v, want [[0]]", rows)
	}
}

func TestCountDistinctParallelFallback(t *testing.T) {
	// Verify that planner.Parallel falls back to serial for DISTINCT aggregates,
	// producing correct results (not double-counted).
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
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	// Use Parallel planner — it should fall back to serial internally.
	op, err := planner.Parallel(ctx, logical, 4)
	if err != nil {
		t.Fatalf("parallel: %v", err)
	}
	defer op.Close()

	batch, err := op.Next(ctx)
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if batch == nil {
		t.Fatal("expected result batch, got nil")
	}
	iv := batch.Vectors[0].(*exec.Int64Vector)
	if iv.Values[0] != 4 {
		t.Fatalf("parallel COUNT(DISTINCT id): got %d, want 4", iv.Values[0])
	}
}

// strIntRow holds a (string, int64) result row for GROUP BY queries.
type strIntRow struct {
	str string
	val int64
}

// runQueryStrings runs a query expected to produce (string, int64) rows.
func runQueryStrings(t *testing.T, path, query string) []strIntRow {
	t.Helper()
	ctx := context.Background()
	tableName := filepath.Base(path)
	tableName = strings.TrimSuffix(tableName, ".vxq")

	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	cat, err := catalog.OpenSingle(ctx, tableName, path)
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

	var results []strIntRow
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		sv := batch.Vectors[0].(*exec.StringVector)
		iv := batch.Vectors[1].(*exec.Int64Vector)
		for i := range batch.Length {
			s := sv.Get(i)
			results = append(results, strIntRow{str: s, val: iv.Values[i]})
		}
	}
	return results
}
