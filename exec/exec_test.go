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
