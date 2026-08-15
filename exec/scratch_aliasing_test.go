package exec

// Aliasing regression tests for the operator scratch buffers introduced with
// per-instance buffer reuse. Each test pins one clause of the contract stated at
// the top of scratch.go; together they are the argument that reuse is safe, with
// the -race goldentest suite as the backstop rather than the proof.

import (
	"context"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// reusingSourceOp emulates TableScan: it hands out the same *Batch every call and
// overwrites its column data in place, so anything that survives a Next() call
// intact is genuinely not aliasing the source.
//
// Columns: a (the input values), b (a×10), t (a constant threshold, so a test can
// write a column-to-column predicate).
type reusingSourceOp struct {
	schema Schema
	rows   [][]int64
	pos    int
	batch  *Batch
	aVec   *Int64Vector
	bVec   *Int64Vector
	tVec   *Int64Vector
}

const reusingSourceThreshold = 250

func newReusingSourceOp(batches [][]int64) *reusingSourceOp {
	schema := Schema{Fields: []Field{
		{Name: "a", Type: TypeInt64, Nullable: true},
		{Name: "b", Type: TypeInt64, Nullable: true},
		{Name: "t", Type: TypeInt64, Nullable: true},
	}}
	return &reusingSourceOp{schema: schema, rows: batches}
}

func (s *reusingSourceOp) Next(_ context.Context) (*Batch, error) {
	if s.pos >= len(s.rows) {
		return nil, nil
	}
	vals := s.rows[s.pos]
	s.pos++
	n := len(vals)
	if s.batch == nil {
		s.aVec = &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		s.bVec = &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		s.tVec = &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		s.batch = &Batch{Schema: s.schema, Vectors: []Vector{s.aVec, s.bVec, s.tVec}}
	}
	for _, v := range []*Int64Vector{s.aVec, s.bVec, s.tVec} {
		v.Values = v.Values[:n]
		v.NullBitmap = v.NullBitmap[:(n+7)/8]
	}
	for i, v := range vals {
		s.aVec.Values[i] = v
		s.bVec.Values[i] = v * 10
		s.tVec.Values[i] = reusingSourceThreshold
		storage.SetValidBit(s.aVec.NullBitmap, i)
		storage.SetValidBit(s.bVec.NullBitmap, i)
		storage.SetValidBit(s.tVec.NullBitmap, i)
	}
	s.batch.Length = n
	s.batch.SelVec = nil
	return s.batch, nil
}

func (s *reusingSourceOp) Schema() Schema { return s.schema }
func (s *reusingSourceOp) Close() error   { return nil }

func colRef(name string, idx int, t DataType) *ColumnRef {
	return &ColumnRef{Name: name, Idx: idx, T: t}
}

// TestExprResultIsReadOnlyToParents pins contract clause 3: only the producing
// node writes its own buffer. NotExpr, AndExpr and OrExpr used to fold their
// result into the child's vector in place, which under buffer reuse would
// corrupt a buffer another consumer still holds.
func TestExprResultIsReadOnlyToParents(t *testing.T) {
	ctx := context.Background()
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12}})
	batch, err := src.Next(ctx)
	if err != nil {
		t.Fatalf("source: %v", err)
	}

	t.Run("NOT does not mutate its child", func(t *testing.T) {
		child := &BinOp{Op: BinGT, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(5), T: TypeInt64}, T: TypeBool}
		childRes, err := child.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("child eval: %v", err)
		}
		childBV := childRes.(*BoolVector)

		not := &NotExpr{Child: child}
		notRes, err := not.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("not eval: %v", err)
		}
		notBV := notRes.(*BoolVector)

		// a > 5 over {1,5,9,12}
		wantChild := []bool{false, false, true, true}
		for i, w := range wantChild {
			if got := childBV.Get(i); got != w {
				t.Errorf("child result row %d = %v after parent Eval, want %v (parent mutated the child's buffer)", i, got, w)
			}
			if got := notBV.Get(i); got != !w {
				t.Errorf("NOT result row %d = %v, want %v", i, got, !w)
			}
		}
		if childBV == notBV {
			t.Error("NOT returned its child's vector; it must own its own scratch")
		}
	})

	t.Run("AND does not mutate its first child", func(t *testing.T) {
		left := &BinOp{Op: BinGT, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(2), T: TypeInt64}, T: TypeBool}
		right := &BinOp{Op: BinLT, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(10), T: TypeInt64}, T: TypeBool}
		leftRes, err := left.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("left eval: %v", err)
		}
		leftBV := leftRes.(*BoolVector)

		and := &AndExpr{Children: []Expr{left, right}}
		andRes, err := and.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("and eval: %v", err)
		}
		andBV := andRes.(*BoolVector)

		wantLeft := []bool{false, true, true, true} // a > 2
		wantAnd := []bool{false, true, true, false} // 2 < a < 10
		for i := range wantLeft {
			if got := leftBV.Get(i); got != wantLeft[i] {
				t.Errorf("left child row %d = %v after AND, want %v (AND folded into the child's buffer)", i, got, wantLeft[i])
			}
			if got := andBV.Get(i); got != wantAnd[i] {
				t.Errorf("AND row %d = %v, want %v", i, got, wantAnd[i])
			}
		}
	})

	t.Run("OR does not mutate its first child", func(t *testing.T) {
		left := &BinOp{Op: BinLT, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(2), T: TypeInt64}, T: TypeBool}
		right := &BinOp{Op: BinGT, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(10), T: TypeInt64}, T: TypeBool}
		leftRes, err := left.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("left eval: %v", err)
		}
		leftBV := leftRes.(*BoolVector)

		or := &OrExpr{Children: []Expr{left, right}}
		orRes, err := or.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("or eval: %v", err)
		}
		orBV := orRes.(*BoolVector)

		wantLeft := []bool{true, false, false, false} // a < 2
		wantOr := []bool{true, false, false, true}    // a < 2 OR a > 10
		for i := range wantLeft {
			if got := leftBV.Get(i); got != wantLeft[i] {
				t.Errorf("left child row %d = %v after OR, want %v (OR folded into the child's buffer)", i, got, wantLeft[i])
			}
			if got := orBV.Get(i); got != wantOr[i] {
				t.Errorf("OR row %d = %v, want %v", i, got, wantOr[i])
			}
		}
	})
}

// TestSharedChildExprEvaluatedTwice covers the one shape where a single Expr
// instance is reached twice in the same tree: BETWEEN rewrites to two
// comparisons over the *same* child. Re-evaluating the child must recompute
// identical values rather than disturb the first comparison's result.
func TestSharedChildExprEvaluatedTwice(t *testing.T) {
	ctx := context.Background()
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12}})
	batch, err := src.Next(ctx)
	if err != nil {
		t.Fatalf("source: %v", err)
	}

	// (a + 0) BETWEEN 5 AND 9 — the shared child is a computed expression with
	// its own scratch, not a bare column reference.
	shared := &BinOp{Op: BinAdd, Left: colRef("a", 0, TypeInt64), Right: &Literal{Val: int64(0), T: TypeInt64}, T: TypeInt64}
	between := &BetweenExpr{
		Child: shared,
		Lo:    &Literal{Val: int64(5), T: TypeInt64},
		Hi:    &Literal{Val: int64(9), T: TypeInt64},
	}

	want := []bool{false, true, true, false}
	// Evaluate twice: the second pass also proves the cached rewrite is reusable.
	for pass := 0; pass < 2; pass++ {
		res, err := between.Eval(ctx, batch)
		if err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
		bv := res.(*BoolVector)
		for i, w := range want {
			if got := bv.Get(i); got != w {
				t.Errorf("pass %d row %d = %v, want %v", pass, i, got, w)
			}
		}
	}
}

// TestFilterSelVecStackedFilters checks that two stacked Filters, each owning its
// own reused selection-vector buffer, compose correctly: the upper Filter reads
// the lower one's vector while writing its own.
//
// The upper predicate compares two columns rather than a column against a
// literal, deliberately. A literal is sized from Batch.Length while a comparison
// is sized from the physical vector length, so a literal above a selection vector
// is short and null-masks the batch's trailing rows. That mismatch predates this
// change and behaves identically before and after it (verified by running the
// same stacked-filter shape on df58edb), but it would obscure what this test is
// here to check.
func TestFilterSelVecStackedFilters(t *testing.T) {
	ctx := context.Background()
	src := newReusingSourceOp([][]int64{
		{1, 5, 9, 12, 20},
		{2, 6, 30, 40, 50},
	})

	lower, err := NewFilter(src, &BinOp{
		Op: BinGT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("lower filter: %v", err)
	}
	// b < t, i.e. a×10 < 250 → a < 25.
	upper, err := NewFilter(lower, &BinOp{
		Op: BinLT, Left: colRef("b", 1, TypeInt64),
		Right: colRef("t", 2, TypeInt64), T: TypeBool,
	})
	if err != nil {
		t.Fatalf("upper filter: %v", err)
	}

	// 4 < a < 25 → batch 1: {5,9,12,20}; batch 2: {6}
	want := [][]int64{{5, 9, 12, 20}, {6}}
	for bi := 0; ; bi++ {
		batch, err := upper.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			if bi != len(want) {
				t.Fatalf("got %d batches, want %d", bi, len(want))
			}
			break
		}
		if bi >= len(want) {
			t.Fatalf("more batches than expected (%d)", len(want))
		}
		if batch.Length != len(want[bi]) {
			t.Fatalf("batch %d length = %d, want %d", bi, batch.Length, len(want[bi]))
		}
		av := batch.Vectors[0].(*Int64Vector)
		for i, idx := range batch.SelVec {
			if got := av.Values[idx]; got != want[bi][i] {
				t.Errorf("batch %d row %d = %d, want %d", bi, i, got, want[bi][i])
			}
		}
	}
}

// TestProjectOutputSurvivesLaterNext is the aliasing regression test for the
// consumer the prompt worries about: one that holds Project's batches across
// Next() and reads them after EOF (exec/expr_eval_test.go's runExprQuery does
// exactly this). Project must keep that working, which is why expression scratch
// is copied out rather than passed downstream.
func TestProjectOutputSurvivesLaterNext(t *testing.T) {
	ctx := context.Background()

	for _, tc := range []struct {
		name     string
		filtered bool
	}{
		{"unfiltered", false},
		{"through a selection vector", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := newReusingSourceOp([][]int64{
				{1, 2, 3, 4},
				{10, 20, 30, 40},
				{100, 200, 300, 400},
			})
			var child Operator = src
			if tc.filtered {
				f, err := NewFilter(src, &BinOp{
					Op: BinGT, Left: colRef("a", 0, TypeInt64),
					Right: &Literal{Val: int64(1), T: TypeInt64}, T: TypeBool,
				})
				if err != nil {
					t.Fatalf("filter: %v", err)
				}
				child = f
			}

			proj, err := NewProject(child, []ProjectExpr{
				// A bare column reference (passthrough) and a computed
				// expression (scratch-backed) in the same projection.
				{Name: "a", Expr: colRef("a", 0, TypeInt64)},
				{Name: "a2", Expr: &BinOp{
					Op: BinMul, Left: colRef("a", 0, TypeInt64),
					Right: &Literal{Val: int64(2), T: TypeInt64}, T: TypeInt64,
				}},
			})
			if err != nil {
				t.Fatalf("project: %v", err)
			}

			// Collect every batch, then assert after EOF — the retaining
			// consumer pattern.
			var batches []*Batch
			var want [][]int64
			for {
				b, err := proj.Next(ctx)
				if err != nil {
					t.Fatalf("next: %v", err)
				}
				if b == nil {
					break
				}
				snapshot := make([]int64, b.Length)
				computed := b.Vectors[1].(*Int64Vector)
				copy(snapshot, computed.Values[:b.Length])
				want = append(want, snapshot)
				batches = append(batches, b)
			}
			if len(batches) != 3 {
				t.Fatalf("got %d batches, want 3", len(batches))
			}

			for bi, b := range batches {
				computed := b.Vectors[1].(*Int64Vector)
				for i := 0; i < b.Length; i++ {
					if got := computed.Values[i]; got != want[bi][i] {
						t.Errorf("batch %d row %d computed column = %d after later Next calls, want %d "+
							"(Project leaked expression scratch downstream)", bi, i, got, want[bi][i])
					}
				}
				// The computed column must also equal 2×a as recorded when the
				// batch was produced, which pins the values themselves.
				for i := 0; i < b.Length; i++ {
					if want[bi][i]%2 != 0 {
						t.Errorf("batch %d row %d = %d is not 2×a", bi, i, want[bi][i])
					}
				}
			}
		})
	}
}

// TestLiteralCacheMatchesFreshLiteral pins the Literal broadcast cache: a reused
// vector must be indistinguishable from a freshly built one at every length,
// including the trailing-bit masking storage.FullBitmap applies.
func TestLiteralCacheMatchesFreshLiteral(t *testing.T) {
	ctx := context.Background()
	lengths := []int{1024, 7, 13, 1024, 1, 0, 100}

	check := func(t *testing.T, cached, fresh Vector, n int) {
		t.Helper()
		if cached.Len() != fresh.Len() {
			t.Fatalf("n=%d: cached len %d, fresh len %d", n, cached.Len(), fresh.Len())
		}
		cn, fn := cached.Nulls(), fresh.Nulls()
		if len(cn) != len(fn) {
			t.Fatalf("n=%d: cached bitmap %d bytes, fresh %d bytes", n, len(cn), len(fn))
		}
		for i := range fn {
			if cn[i] != fn[i] {
				t.Errorf("n=%d: null bitmap byte %d = %#x, want %#x", n, i, cn[i], fn[i])
			}
		}
		switch c := cached.(type) {
		case *Int64Vector:
			f := fresh.(*Int64Vector)
			for i := range f.Values {
				if c.Values[i] != f.Values[i] {
					t.Errorf("n=%d: int64 row %d = %d, want %d", n, i, c.Values[i], f.Values[i])
				}
			}
		case *Float64Vector:
			f := fresh.(*Float64Vector)
			for i := range f.Values {
				if c.Values[i] != f.Values[i] {
					t.Errorf("n=%d: float64 row %d = %v, want %v", n, i, c.Values[i], f.Values[i])
				}
			}
		case *DateVector:
			f := fresh.(*DateVector)
			for i := range f.Values {
				if c.Values[i] != f.Values[i] {
					t.Errorf("n=%d: date row %d = %d, want %d", n, i, c.Values[i], f.Values[i])
				}
			}
		case *BoolVector:
			f := fresh.(*BoolVector)
			for i := 0; i < f.Length; i++ {
				if c.Get(i) != f.Get(i) {
					t.Errorf("n=%d: bool row %d = %v, want %v", n, i, c.Get(i), f.Get(i))
				}
			}
			// Compare the raw bits too: a shrunk cache must not leave bits set
			// for rows past n in the final byte.
			if len(c.Bits) != len(f.Bits) {
				t.Fatalf("n=%d: cached bits %d bytes, fresh %d bytes", n, len(c.Bits), len(f.Bits))
			}
			for i := range f.Bits {
				if c.Bits[i] != f.Bits[i] {
					t.Errorf("n=%d: bits byte %d = %#x, want %#x", n, i, c.Bits[i], f.Bits[i])
				}
			}
		case *StringVector:
			f := fresh.(*StringVector)
			for i := 0; i < f.Len(); i++ {
				if c.Get(i) != f.Get(i) {
					t.Errorf("n=%d: string row %d = %q, want %q", n, i, c.Get(i), f.Get(i))
				}
			}
		default:
			t.Fatalf("unhandled vector type %T", cached)
		}
	}

	cases := []struct {
		name string
		val  any
		typ  DataType
	}{
		{"int64", int64(42), TypeInt64},
		{"float64", 0.07, TypeFloat64},
		{"date", int32(8766), TypeDate},
		{"bool", true, TypeBool},
		{"string", "MAIL", TypeString},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reused := &Literal{Val: tc.val, T: tc.typ}
			for _, n := range lengths {
				b := &Batch{Length: n}
				cached, err := reused.Eval(ctx, b)
				if err != nil {
					t.Fatalf("cached eval n=%d: %v", n, err)
				}
				fresh, err := (&Literal{Val: tc.val, T: tc.typ}).Eval(ctx, b)
				if err != nil {
					t.Fatalf("fresh eval n=%d: %v", n, err)
				}
				check(t, cached, fresh, n)
			}
		})
	}
}

// TestScratchVectorsAreZeroedOnReuse pins the acquire helpers: a reused vector
// must present the same zeroed state a freshly allocated one would, so rows a
// caller leaves unwritten (null rows in arithmetic, for instance) never read back
// as a previous batch's values.
func TestScratchVectorsAreZeroedOnReuse(t *testing.T) {
	var f64 *Float64Vector
	v := acquireFloat64Vector(&f64, 8)
	for i := range v.Values {
		v.Values[i] = float64(i + 1)
	}
	setAllValid(v.NullBitmap, 8)

	v2 := acquireFloat64Vector(&f64, 8)
	if v2 != v {
		t.Fatal("expected the same vector to be reused")
	}
	for i, got := range v2.Values {
		if got != 0 {
			t.Errorf("reused float64 row %d = %v, want 0", i, got)
		}
	}
	for i, got := range v2.NullBitmap {
		if got != 0 {
			t.Errorf("reused null bitmap byte %d = %#x, want 0", i, got)
		}
	}

	var i64 *Int64Vector
	iv := acquireInt64Vector(&i64, 4)
	iv.Values[0] = 99
	storage.SetValidBit(iv.NullBitmap, 0)
	iv2 := acquireInt64Vector(&i64, 4)
	if iv2.Values[0] != 0 || iv2.NullBitmap[0] != 0 {
		t.Errorf("reused int64 vector not zeroed: values[0]=%d bitmap[0]=%#x", iv2.Values[0], iv2.NullBitmap[0])
	}

	var bv *BoolVector
	b := acquireBoolVector(&bv, 10)
	b.Set(3, true)
	setAllValid(b.NullBitmap, 10)
	b2 := acquireBoolVector(&bv, 10)
	if b2.Get(3) {
		t.Error("reused BoolVector kept a set bit")
	}
	for i, got := range b2.NullBitmap {
		if got != 0 {
			t.Errorf("reused BoolVector null bitmap byte %d = %#x, want 0", i, got)
		}
	}
	if b2.Length != 10 {
		t.Errorf("reused BoolVector Length = %d, want 10", b2.Length)
	}
}

// TestSetAllValidMatchesFullBitmap checks the in-place bitmap fill against
// storage.FullBitmap, which is the semantics it replaces.
func TestSetAllValidMatchesFullBitmap(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 63, 64, 1023, 1024} {
		buf := make([]byte, (n+7)/8)
		for i := range buf {
			buf[i] = 0xAA // dirty, as a reused buffer would be
		}
		setAllValid(buf, n)
		want := storage.FullBitmap(n)
		if len(buf) != len(want) {
			t.Fatalf("n=%d: got %d bytes, want %d", n, len(buf), len(want))
		}
		for i := range want {
			if buf[i] != want[i] {
				t.Errorf("n=%d: byte %d = %#x, want %#x", n, i, buf[i], want[i])
			}
		}
	}
}
