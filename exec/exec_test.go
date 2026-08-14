package exec_test

import (
	"context"
	"io"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

var ctx = context.Background()

type testingTB interface {
	Helper()
	TempDir() string
	Fatalf(format string, args ...any)
}

// writeTestFile writes a simple 2-column (id int64, val float64) .vxq file
// and returns the path.
func writeTestFile(t testingTB, n int) string {
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
	for written := 0; written < n; {
		chunk := storage.RowGroupRows
		if written+chunk > n {
			chunk = n - written
		}
		_ = w.BeginRowGroup(chunk)
		_ = w.AppendColumn(ctx, 0, nil, ids[written:written+chunk])
		_ = w.AppendColumn(ctx, 1, nil, vals[written:written+chunk])
		_ = w.EndRowGroup()
		written += chunk
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return path
}

// ---- TableScan basic --------------------------------------------------------

func TestTableScanAllRows(t *testing.T) {
	const N = 4096
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := exec.NewTableScan(r, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	total := 0
	for {
		batch, err := scan.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		total += batch.Length
	}
	if total != N {
		t.Fatalf("expected %d rows, got %d", N, total)
	}
}

func TestTableScanColumnPruning(t *testing.T) {
	const N = 2048
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := exec.NewTableScan(r, []string{"id"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	if len(scan.Schema().Fields) != 1 || scan.Schema().Fields[0].Name != "id" {
		t.Fatalf("unexpected schema: %+v", scan.Schema())
	}
	total := 0
	for {
		batch, err := scan.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		if len(batch.Vectors) != 1 {
			t.Fatalf("expected 1 vector, got %d", len(batch.Vectors))
		}
		total += batch.Length
	}
	if total != N {
		t.Fatalf("expected %d rows, got %d", N, total)
	}
}

// ---- Filter -----------------------------------------------------------------

func TestFilterOperator(t *testing.T) {
	const N = 2048
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, _ := exec.NewTableScan(r, nil, nil)

	// Filter: id >= 1024
	pred := &exec.BinOp{
		Op:    exec.BinGE,
		Left:  &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
		Right: &exec.Literal{Val: int64(1024), T: exec.TypeInt64},
		T:     exec.TypeBool,
	}
	filter, err := exec.NewFilter(scan, pred)
	if err != nil {
		t.Fatal(err)
	}
	defer filter.Close()

	total := 0
	for {
		batch, err := filter.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		total += batch.Length
	}
	if total != 1024 {
		t.Fatalf("expected 1024 rows (id>=1024), got %d", total)
	}
}

// ---- Project ----------------------------------------------------------------

func TestProjectOperator(t *testing.T) {
	const N = 512
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, _ := exec.NewTableScan(r, nil, nil)

	// Project: id * 2 AS doubled
	proj, err := exec.NewProject(scan, []exec.ProjectExpr{
		{
			Name: "doubled",
			Expr: &exec.BinOp{
				Op:    exec.BinMul,
				Left:  &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
				Right: &exec.Literal{Val: int64(2), T: exec.TypeInt64},
				T:     exec.TypeInt64,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer proj.Close()

	total := 0
	for {
		batch, err := proj.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		idVec := batch.Vectors[0].(*exec.Int64Vector)
		for i := 0; i < batch.Length; i++ {
			want := int64((total + i) * 2)
			if idVec.Values[i] != want {
				t.Fatalf("row %d: got %d want %d", total+i, idVec.Values[i], want)
			}
		}
		total += batch.Length
	}
	if total != N {
		t.Fatalf("expected %d rows, got %d", N, total)
	}
}

// ---- Filter + Project pipeline ---------------------------------------------

func TestFilterProjectPipeline(t *testing.T) {
	const N = storage.RowGroupRows * 2
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, _ := exec.NewTableScan(r, nil, nil)

	// Filter id < 100.
	filter, _ := exec.NewFilter(scan, &exec.BinOp{
		Op:    exec.BinLT,
		Left:  &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
		Right: &exec.Literal{Val: int64(100), T: exec.TypeInt64},
		T:     exec.TypeBool,
	})

	// Project id only.
	proj, _ := exec.NewProject(filter, []exec.ProjectExpr{
		{Name: "id", Expr: &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64}},
	})
	defer proj.Close()

	total := 0
	maxID := int64(-1)
	for {
		batch, err := proj.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		idVec := batch.Vectors[0].(*exec.Int64Vector)
		for i := 0; i < batch.Length; i++ {
			if idVec.Values[i] > maxID {
				maxID = idVec.Values[i]
			}
		}
		total += batch.Length
	}
	if total != 100 {
		t.Fatalf("expected 100 rows, got %d", total)
	}
	if maxID != 99 {
		t.Fatalf("expected maxID=99, got %d", maxID)
	}
}

// ---- Zone-map pruning via TableScan ----------------------------------------

func TestZoneMapPruning(t *testing.T) {
	// Write 4 row groups: RG0 has ids 0..65535, RG1 ids 65536..131071, etc.
	const numRGs = 4
	const N = storage.RowGroupRows * numRGs
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}

	// Zone predicate: only pass row groups where max(id) >= 131072
	// That means only RG2 and RG3 (ids 131072..262143).
	zonePred := func(rg *storage.RowGroupMeta) bool {
		return int64(rg.Columns[0].Stats.Max) >= 131072
	}
	scan, _ := exec.NewTableScan(r, []string{"id"}, zonePred)
	defer scan.Close()

	total := 0
	for {
		batch, err := scan.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		total += batch.Length
	}
	// Should read only RG2 + RG3 = 2 * 65536 rows.
	if total != storage.RowGroupRows*2 {
		t.Fatalf("expected %d rows after zone pruning, got %d", storage.RowGroupRows*2, total)
	}
}

// ---- Expr: LIKE -------------------------------------------------------------

func TestLikeExpr(t *testing.T) {
	cases := []struct {
		pattern, s string
		want       bool
	}{
		{"%foo%", "barfoobaz", true},
		{"%foo%", "barbazbaz", false},
		{"foo%", "foobar", true},
		{"foo%", "barfoo", false},
		{"_oo", "foo", true},
		{"_oo", "fo", false},
		{"%", "", true},
		{"%", "anything", true},
	}
	for _, tc := range cases {
		batch := &exec.Batch{
			Schema: storage.Schema{Fields: []storage.Field{{Name: "s", Type: storage.TypeString}}},
			Vectors: []exec.Vector{
				&exec.StringVector{
					Codes:      []uint32{0},
					Dict:       nil,
					NullBitmap: storage.FullBitmap(1),
				},
			},
			Length: 1,
		}
		// Override Get via a stub dict.
		batch.Vectors[0].(*exec.StringVector).Dict = &storage.Dictionary{
			Offsets: []uint32{0},
			Data:    []byte(tc.s),
		}
		likeExpr := &exec.LikeExpr{
			Child:   &exec.ColumnRef{Name: "s", Idx: 0, T: exec.TypeString},
			Pattern: tc.pattern,
		}
		v, err := likeExpr.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("LIKE %q %q: %v", tc.pattern, tc.s, err)
		}
		bv := v.(*exec.BoolVector)
		got := !bv.IsNull(0) && bv.Get(0)
		if got != tc.want {
			t.Errorf("LIKE %q %q: got %v want %v", tc.pattern, tc.s, got, tc.want)
		}
	}
}

// ---- Limit with selection vector -------------------------------------------

func TestLimitWithSelectionVector(t *testing.T) {
	// Write a small file with 10 rows: id=[0..9], val=[0.0, 2.5, 5.0, ...]
	const N = 10
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}

	// Scan all rows, apply filter id >= 5 → selects physical positions [5,6,7,8,9]
	scan, err := exec.NewTableScan(r, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	pred := &exec.BinOp{
		Op:    exec.BinGE,
		Left:  &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
		Right: &exec.Literal{Val: int64(5), T: exec.TypeInt64},
		T:     exec.TypeBool,
	}
	filter, err := exec.NewFilter(scan, pred)
	if err != nil {
		t.Fatal(err)
	}

	// Apply LIMIT 2 — should yield values at positions 5 and 6, NOT 0 and 1.
	limit := exec.NewLimit(filter, 2)
	defer limit.Close()

	batch, err := limit.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if batch == nil {
		t.Fatal("expected a batch, got nil")
	}
	if batch.Length != 2 {
		t.Fatalf("expected Length=2, got %d", batch.Length)
	}

	// Read the id values through the selection vector.
	idVec := batch.Vectors[0].(*exec.Int64Vector)
	for i := 0; i < batch.Length; i++ {
		idx := i
		if batch.SelVec != nil {
			idx = int(batch.SelVec[i])
		}
		got := idVec.Values[idx]
		want := int64(5 + i) // expect id=5, id=6
		if got != want {
			t.Errorf("row %d: got id=%d, want id=%d", i, got, want)
		}
	}

	// Should be exhausted after LIMIT 2.
	batch2, err := limit.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if batch2 != nil {
		t.Fatal("expected nil after limit exhausted")
	}
}

func TestLimitWithoutFilter(t *testing.T) {
	// Verify LIMIT without upstream filter still works correctly (regression).
	const N = 100
	path := writeTestFile(t, N)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}

	scan, err := exec.NewTableScan(r, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	limit := exec.NewLimit(scan, 3)
	defer limit.Close()

	total := 0
	for {
		batch, err := limit.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		idVec := batch.Vectors[0].(*exec.Int64Vector)
		for i := 0; i < batch.Length; i++ {
			idx := i
			if batch.SelVec != nil {
				idx = int(batch.SelVec[i])
			}
			got := idVec.Values[idx]
			want := int64(total + i)
			if got != want {
				t.Errorf("row %d: got id=%d, want id=%d", total+i, got, want)
			}
		}
		total += batch.Length
	}
	if total != 3 {
		t.Fatalf("expected 3 total rows, got %d", total)
	}
}

// ---- BenchmarkScanFilterProject --------------------------------------------

func BenchmarkScanFilterProject(b *testing.B) {
	const N = storage.RowGroupRows * 16
	path := writeTestFile(b, N)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r, err := storage.Open(ctx, path)
		if err != nil {
			b.Fatal(err)
		}
		scan, _ := exec.NewTableScan(r, nil, nil)
		filter, _ := exec.NewFilter(scan, &exec.BinOp{
			Op:    exec.BinLT,
			Left:  &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
			Right: &exec.Literal{Val: int64(N / 2), T: exec.TypeInt64},
			T:     exec.TypeBool,
		})
		proj, _ := exec.NewProject(filter, []exec.ProjectExpr{
			{Name: "id", Expr: &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64}},
		})
		sum := int64(0)
		for {
			batch, err := proj.Next(ctx)
			if err != nil {
				b.Fatal(err)
			}
			if batch == nil {
				break
			}
			idVec := batch.Vectors[0].(*exec.Int64Vector)
			for i := 0; i < batch.Length; i++ {
				sum += idVec.Values[i]
			}
		}
		_ = sum
		_ = proj.Close()

		// Drain remaining (scan is closed by proj.Close chain).
		_, _ = io.Discard.Write(nil)
	}
}

// ---- Aggregate null semantics tests ----------------------------------------

// makeFloat64VecWithNulls creates a Float64Vector with specified values and null positions.
func makeFloat64VecWithNulls(vals []float64, nulls []int) *exec.Float64Vector {
	n := len(vals)
	bmp := storage.FullBitmap(n)
	for _, idx := range nulls {
		storage.SetNullBit(bmp, idx)
	}
	return &exec.Float64Vector{Values: vals, NullBitmap: bmp}
}

// makeInt64VecWithNulls creates an Int64Vector with specified values and null positions.
func makeInt64VecWithNulls(vals []int64, nulls []int) *exec.Int64Vector {
	n := len(vals)
	bmp := storage.FullBitmap(n)
	for _, idx := range nulls {
		storage.SetNullBit(bmp, idx)
	}
	return &exec.Int64Vector{Values: vals, NullBitmap: bmp}
}

// singleBatchOp is a trivial operator that yields one batch then EOF.
type singleBatchOp struct {
	batch *exec.Batch
	done  bool
}

func (s *singleBatchOp) Next(ctx context.Context) (*exec.Batch, error) {
	if s.done {
		return nil, nil
	}
	s.done = true
	return s.batch, nil
}
func (s *singleBatchOp) Schema() exec.Schema { return s.batch.Schema }
func (s *singleBatchOp) Close() error        { return nil }

// emptyOp is an operator that yields no batches (empty input).
type emptyOp struct {
	schema exec.Schema
}

func (e *emptyOp) Next(ctx context.Context) (*exec.Batch, error) { return nil, nil }
func (e *emptyOp) Schema() exec.Schema                           { return e.schema }
func (e *emptyOp) Close() error                                  { return nil }

func TestAvgWithNulls(t *testing.T) {
	// AVG over [1.0, NULL, 3.0] should return 2.0 (not 1.33)
	vec := makeFloat64VecWithNulls([]float64{1.0, 0.0, 3.0}, []int{1})
	batch := &exec.Batch{
		Schema:  exec.Schema{Fields: []exec.Field{{Name: "val", Type: exec.TypeFloat64, Nullable: true}}},
		Vectors: []exec.Vector{vec},
		Length:  3,
	}
	child := &singleBatchOp{batch: batch}
	agg, err := exec.NewHashAggregate(child, nil, []exec.AggExpr{
		{Kind: exec.AggAvg, ColIdx: 0, OutName: "avg_val", AccumType: exec.TypeFloat64},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agg.Close()

	result, err := agg.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("expected one output row, got nil")
	}
	if result.Length != 1 {
		t.Fatalf("expected 1 row, got %d", result.Length)
	}

	fv := result.Vectors[0].(*exec.Float64Vector)
	if fv.IsNull(0) {
		t.Fatal("AVG should not be null when there are non-null inputs")
	}
	got := fv.Values[0]
	want := 2.0 // (1.0 + 3.0) / 2 non-null values
	if got != want {
		t.Errorf("AVG([1, NULL, 3]) = %v, want %v", got, want)
	}
}

func TestSumAllNull(t *testing.T) {
	// SUM over [NULL, NULL, NULL] should return NULL
	vec := makeFloat64VecWithNulls([]float64{0, 0, 0}, []int{0, 1, 2})
	batch := &exec.Batch{
		Schema:  exec.Schema{Fields: []exec.Field{{Name: "val", Type: exec.TypeFloat64, Nullable: true}}},
		Vectors: []exec.Vector{vec},
		Length:  3,
	}
	child := &singleBatchOp{batch: batch}
	agg, err := exec.NewHashAggregate(child, nil, []exec.AggExpr{
		{Kind: exec.AggSum, ColIdx: 0, OutName: "sum_val", AccumType: exec.TypeFloat64},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agg.Close()

	result, err := agg.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("expected one output row, got nil")
	}

	fv := result.Vectors[0].(*exec.Float64Vector)
	if !fv.IsNull(0) {
		t.Errorf("SUM over all-null should be NULL, got %v", fv.Values[0])
	}
}

func TestMinMaxAllNull(t *testing.T) {
	// MIN/MAX over [NULL, NULL] should return NULL
	vec := makeInt64VecWithNulls([]int64{0, 0}, []int{0, 1})
	batch := &exec.Batch{
		Schema:  exec.Schema{Fields: []exec.Field{{Name: "val", Type: exec.TypeInt64, Nullable: true}}},
		Vectors: []exec.Vector{vec},
		Length:  2,
	}

	for _, kind := range []exec.AggKind{exec.AggMin, exec.AggMax} {
		name := "MIN"
		if kind == exec.AggMax {
			name = "MAX"
		}
		t.Run(name, func(t *testing.T) {
			child := &singleBatchOp{batch: batch}
			// Reset done flag for reuse
			child.done = false
			agg, err := exec.NewHashAggregate(child, nil, []exec.AggExpr{
				{Kind: kind, ColIdx: 0, OutName: "agg_val", AccumType: exec.TypeInt64},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer agg.Close()

			result, err := agg.Next(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if result == nil {
				t.Fatal("expected one output row")
			}

			iv := result.Vectors[0].(*exec.Int64Vector)
			if !iv.IsNull(0) {
				t.Errorf("%s over all-null should be NULL, got %d", name, iv.Values[0])
			}
		})
	}
}

func TestEmptyTableAggregate(t *testing.T) {
	// Global aggregate over empty input: COUNT(*)=0, SUM=NULL
	schema := exec.Schema{Fields: []exec.Field{{Name: "val", Type: exec.TypeFloat64, Nullable: true}}}
	child := &emptyOp{schema: schema}
	agg, err := exec.NewHashAggregate(child, nil, []exec.AggExpr{
		{Kind: exec.AggCount, ColIdx: -1, OutName: "cnt", AccumType: exec.TypeInt64},
		{Kind: exec.AggSum, ColIdx: 0, OutName: "sum_val", AccumType: exec.TypeFloat64},
		{Kind: exec.AggAvg, ColIdx: 0, OutName: "avg_val", AccumType: exec.TypeFloat64},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agg.Close()

	result, err := agg.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("global aggregate over empty input should emit one row")
	}
	if result.Length != 1 {
		t.Fatalf("expected 1 row, got %d", result.Length)
	}

	// COUNT(*) should be 0, not NULL
	cntVec := result.Vectors[0].(*exec.Int64Vector)
	if cntVec.IsNull(0) {
		t.Error("COUNT(*) over empty input should not be NULL")
	}
	if cntVec.Values[0] != 0 {
		t.Errorf("COUNT(*) over empty input = %d, want 0", cntVec.Values[0])
	}

	// SUM should be NULL
	sumVec := result.Vectors[1].(*exec.Float64Vector)
	if !sumVec.IsNull(0) {
		t.Errorf("SUM over empty input should be NULL, got %v", sumVec.Values[0])
	}

	// AVG should be NULL
	avgVec := result.Vectors[2].(*exec.Float64Vector)
	if !avgVec.IsNull(0) {
		t.Errorf("AVG over empty input should be NULL, got %v", avgVec.Values[0])
	}

	// Second call should return nil (EOF)
	result2, err := agg.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result2 != nil {
		t.Error("second Next() should return nil after emitting the single global row")
	}
}

func TestCountColumnWithNulls(t *testing.T) {
	// COUNT(col) skips nulls: [1, NULL, 3, NULL, 5] → COUNT=3
	vec := makeInt64VecWithNulls([]int64{1, 0, 3, 0, 5}, []int{1, 3})
	batch := &exec.Batch{
		Schema:  exec.Schema{Fields: []exec.Field{{Name: "val", Type: exec.TypeInt64, Nullable: true}}},
		Vectors: []exec.Vector{vec},
		Length:  5,
	}
	child := &singleBatchOp{batch: batch}
	agg, err := exec.NewHashAggregate(child, nil, []exec.AggExpr{
		{Kind: exec.AggCount, ColIdx: 0, OutName: "cnt_val", AccumType: exec.TypeInt64},
		{Kind: exec.AggCount, ColIdx: -1, OutName: "cnt_star", AccumType: exec.TypeInt64},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agg.Close()

	result, err := agg.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("expected one output row")
	}

	// COUNT(col) should be 3 (skips 2 nulls)
	cntCol := result.Vectors[0].(*exec.Int64Vector)
	if cntCol.Values[0] != 3 {
		t.Errorf("COUNT(col) with 2 nulls in 5 rows = %d, want 3", cntCol.Values[0])
	}

	// COUNT(*) should be 5 (counts all rows regardless of nulls)
	cntStar := result.Vectors[1].(*exec.Int64Vector)
	if cntStar.Values[0] != 5 {
		t.Errorf("COUNT(*) = %d, want 5", cntStar.Values[0])
	}
}

// ---- CASE WHEN string literals ----------------------------------------------

func TestCaseWhenStringBranches(t *testing.T) {
	// CASE WHEN val > 500 THEN 'high' WHEN val > 200 THEN 'medium' ELSE 'low' END
	// Over a batch where val = [100, 300, 600, 50]
	n := 4
	vals := &exec.Float64Vector{
		Values:     []float64{100, 300, 600, 50},
		NullBitmap: storage.FullBitmap(n),
	}
	batch := &exec.Batch{
		Schema:  storage.Schema{Fields: []storage.Field{{Name: "val", Type: storage.TypeFloat64}}},
		Vectors: []exec.Vector{vals},
		Length:  n,
	}

	caseExpr := &exec.CaseExpr{
		Whens: []exec.When{
			{
				Cond:   &exec.BinOp{Op: exec.BinGT, Left: &exec.ColumnRef{Name: "val", Idx: 0, T: exec.TypeFloat64}, Right: &exec.Literal{Val: float64(500), T: exec.TypeFloat64}, T: exec.TypeBool},
				Result: &exec.Literal{Val: "high", T: exec.TypeString},
			},
			{
				Cond:   &exec.BinOp{Op: exec.BinGT, Left: &exec.ColumnRef{Name: "val", Idx: 0, T: exec.TypeFloat64}, Right: &exec.Literal{Val: float64(200), T: exec.TypeFloat64}, T: exec.TypeBool},
				Result: &exec.Literal{Val: "medium", T: exec.TypeString},
			},
		},
		Else: &exec.Literal{Val: "low", T: exec.TypeString},
		T:    exec.TypeString,
	}

	result, err := caseExpr.Eval(ctx, batch)
	if err != nil {
		t.Fatalf("CaseExpr.Eval: %v", err)
	}
	sv := result.(*exec.StringVector)
	// val=100 → low, val=300 → medium, val=600 → high, val=50 → low
	expected := []string{"low", "medium", "high", "low"}
	for i, want := range expected {
		got := sv.Get(i)
		if got != want {
			t.Errorf("row %d: got %q, want %q", i, got, want)
		}
		if sv.IsNull(i) {
			t.Errorf("row %d: unexpected null", i)
		}
	}
}

func TestCaseWhenStringNoElse(t *testing.T) {
	// CASE WHEN val > 500 THEN 'high' END — no ELSE means NULL for unmatched rows
	n := 4
	vals := &exec.Float64Vector{
		Values:     []float64{100, 300, 600, 800},
		NullBitmap: storage.FullBitmap(n),
	}
	batch := &exec.Batch{
		Schema:  storage.Schema{Fields: []storage.Field{{Name: "val", Type: storage.TypeFloat64}}},
		Vectors: []exec.Vector{vals},
		Length:  n,
	}

	caseExpr := &exec.CaseExpr{
		Whens: []exec.When{
			{
				Cond:   &exec.BinOp{Op: exec.BinGT, Left: &exec.ColumnRef{Name: "val", Idx: 0, T: exec.TypeFloat64}, Right: &exec.Literal{Val: float64(500), T: exec.TypeFloat64}, T: exec.TypeBool},
				Result: &exec.Literal{Val: "high", T: exec.TypeString},
			},
		},
		Else: nil,
		T:    exec.TypeString,
	}

	result, err := caseExpr.Eval(ctx, batch)
	if err != nil {
		t.Fatalf("CaseExpr.Eval: %v", err)
	}
	sv := result.(*exec.StringVector)
	// val=100 → NULL, val=300 → NULL, val=600 → high, val=800 → high
	for i := 0; i < n; i++ {
		if i < 2 {
			if !sv.IsNull(i) {
				t.Errorf("row %d: expected null, got %q", i, sv.Get(i))
			}
		} else {
			if sv.IsNull(i) {
				t.Errorf("row %d: unexpected null", i)
			}
			if got := sv.Get(i); got != "high" {
				t.Errorf("row %d: got %q, want %q", i, got, "high")
			}
		}
	}
}

func TestCaseWhenStringWithSelVec(t *testing.T) {
	// Simulate a filtered batch where batch.Length is physical length (Project contract).
	// CASE WHEN val > 200 THEN 'yes' ELSE 'no' END
	n := 4 // physical length
	vals := &exec.Float64Vector{
		Values:     []float64{100, 300, 50, 600},
		NullBitmap: storage.FullBitmap(n),
	}
	batch := &exec.Batch{
		Schema:  storage.Schema{Fields: []storage.Field{{Name: "val", Type: storage.TypeFloat64}}},
		Vectors: []exec.Vector{vals},
		Length:  n, // Project sets this to physical length before Eval
	}

	caseExpr := &exec.CaseExpr{
		Whens: []exec.When{
			{
				Cond:   &exec.BinOp{Op: exec.BinGT, Left: &exec.ColumnRef{Name: "val", Idx: 0, T: exec.TypeFloat64}, Right: &exec.Literal{Val: float64(200), T: exec.TypeFloat64}, T: exec.TypeBool},
				Result: &exec.Literal{Val: "yes", T: exec.TypeString},
			},
		},
		Else: &exec.Literal{Val: "no", T: exec.TypeString},
		T:    exec.TypeString,
	}

	result, err := caseExpr.Eval(ctx, batch)
	if err != nil {
		t.Fatalf("CaseExpr.Eval: %v", err)
	}
	sv := result.(*exec.StringVector)
	// Verify full physical vector: val[0]=100→no, val[1]=300→yes, val[2]=50→no, val[3]=600→yes
	expected := []string{"no", "yes", "no", "yes"}
	if sv.Len() != n {
		t.Fatalf("result length = %d, want %d", sv.Len(), n)
	}
	for i, want := range expected {
		got := sv.Get(i)
		if got != want {
			t.Errorf("row %d: got %q, want %q", i, got, want)
		}
	}
}
