package exec

// Regression tests for expression sizing under an active selection vector.
//
// The convention these pin is stated on the Expr interface (expr.go): an Expr
// evaluates over the batch's PHYSICAL rows, so every vector in a tree has the
// same length and is indexed by physical row index. A leaf that sized itself
// from Batch.Length instead produced a short vector once a Filter had installed
// a selection vector, which silently null-masked the batch's trailing physical
// rows — wrong results — and read past the short vector's end on batches large
// enough to reach the 8-row-at-a-time comparison loop.

import (
	"context"
	"testing"
)

// selectedValues drains op and returns the logically selected values of column
// col, batch by batch.
func selectedValues(t *testing.T, op Operator, col int) [][]int64 {
	t.Helper()
	ctx := context.Background()
	var out [][]int64
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			return out
		}
		vec, ok := batch.Vectors[col].(*Int64Vector)
		if !ok {
			t.Fatalf("column %d is %T, want *Int64Vector", col, batch.Vectors[col])
		}
		rows := make([]int64, 0, batch.Length)
		if batch.SelVec == nil {
			for i := 0; i < batch.Length; i++ {
				rows = append(rows, vec.Values[i])
			}
		} else {
			if len(batch.SelVec) != batch.Length {
				t.Fatalf("batch.Length = %d but SelVec has %d entries", batch.Length, len(batch.SelVec))
			}
			for _, idx := range batch.SelVec {
				rows = append(rows, vec.Values[idx])
			}
		}
		out = append(out, rows)
	}
}

func equalRows(got, want [][]int64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			return false
		}
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				return false
			}
		}
	}
	return true
}

// TestStackedFiltersLiteralPredicate is the reported repro: filter a > 4, then
// stack a second filter a < 31 whose right operand is a literal. Every row that
// survived the first filter also satisfies the second, so nothing may be
// dropped. Before the fix the literal was sized from the post-filter logical
// length (4) while the comparison was sized from the physical vector length (5),
// so physical row 4 — the value 20 — was null-masked and lost.
func TestStackedFiltersLiteralPredicate(t *testing.T) {
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12, 20}})

	lower, err := NewFilter(src, &BinOp{
		Op: BinGT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("lower filter: %v", err)
	}
	upper, err := NewFilter(lower, &BinOp{
		Op: BinLT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(31), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("upper filter: %v", err)
	}

	want := [][]int64{{5, 9, 12, 20}}
	got := selectedValues(t, upper, 0)
	if !equalRows(got, want) {
		t.Errorf("stacked filters returned %v, want %v", got, want)
	}
}

// TestStackedFiltersLiteralPredicateWideBatch is the same shape at the batch
// size the engine actually runs (BlockRows). Here the short literal vector is
// read past its end by the 8-rows-at-a-time comparison loop, which does not
// consult the null mask, so the bug surfaced as an index-out-of-range panic
// rather than as quietly dropped rows.
func TestStackedFiltersLiteralPredicateWideBatch(t *testing.T) {
	vals := make([]int64, BlockRows)
	for i := range vals {
		vals[i] = int64(i)
	}
	src := newReusingSourceOp([][]int64{vals})

	// a > 0 drops exactly one row, leaving logical BlockRows-1 over BlockRows
	// physical rows.
	lower, err := NewFilter(src, &BinOp{
		Op: BinGT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(0), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("lower filter: %v", err)
	}
	// Every surviving row passes this, so the second filter must drop nothing.
	upper, err := NewFilter(lower, &BinOp{
		Op: BinLT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(1 << 30), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("upper filter: %v", err)
	}

	got := selectedValues(t, upper, 0)
	if len(got) != 1 {
		t.Fatalf("got %d batches, want 1", len(got))
	}
	if len(got[0]) != BlockRows-1 {
		t.Fatalf("got %d rows, want %d", len(got[0]), BlockRows-1)
	}
	for i, v := range got[0] {
		if v != int64(i+1) {
			t.Fatalf("row %d = %d, want %d", i, v, i+1)
		}
	}
}

// TestStackedFiltersBetweenPredicate covers BETWEEN above a selection vector.
// BETWEEN rewrites to two literal comparisons under an AND, so it inherits the
// literal's sizing.
func TestStackedFiltersBetweenPredicate(t *testing.T) {
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12, 20}})

	lower, err := NewFilter(src, &BinOp{
		Op: BinGT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("lower filter: %v", err)
	}
	upper, err := NewFilter(lower, &BetweenExpr{
		Child: colRef("a", 0, TypeInt64),
		Lo:    &Literal{Val: int64(5), T: TypeInt64},
		Hi:    &Literal{Val: int64(20), T: TypeInt64},
	})
	if err != nil {
		t.Fatalf("upper filter: %v", err)
	}

	want := [][]int64{{5, 9, 12, 20}}
	got := selectedValues(t, upper, 0)
	if !equalRows(got, want) {
		t.Errorf("stacked BETWEEN returned %v, want %v", got, want)
	}
}

// TestStackedFiltersMixedTypeArithmetic covers CastIntToFloatExpr and a float
// literal above a selection vector: b/10.0 < 3.0, i.e. a < 3, over rows that
// already passed a > 4 — so the second filter selects nothing, and must do so
// without reading past a short vector.
func TestStackedFiltersMixedTypeArithmetic(t *testing.T) {
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12, 20}})

	lower, err := NewFilter(src, &BinOp{
		Op: BinGT, Left: colRef("a", 0, TypeInt64),
		Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool,
	})
	if err != nil {
		t.Fatalf("lower filter: %v", err)
	}
	// (float64)b / 10.0 <= 1.2 → a <= 1.2 → nothing above a > 4.
	upper, err := NewFilter(lower, &BinOp{
		Op: BinLE,
		Left: &BinOp{
			Op:    BinDiv,
			Left:  &CastIntToFloatExpr{Inner: colRef("b", 1, TypeInt64)},
			Right: &Literal{Val: 10.0, T: TypeFloat64},
			T:     TypeFloat64,
		},
		Right: &Literal{Val: 1.2, T: TypeFloat64},
		T:     TypeBool,
	})
	if err != nil {
		t.Fatalf("upper filter: %v", err)
	}

	if got := selectedValues(t, upper, 0); len(got) != 0 {
		t.Errorf("expected no surviving batches, got %v", got)
	}
}

// TestExprLengthUnderSelectionVector pins the convention directly on the leaves
// that used to size themselves from Batch.Length: with a selection vector
// installed, each must still produce one element per physical row.
func TestExprLengthUnderSelectionVector(t *testing.T) {
	ctx := context.Background()
	src := newReusingSourceOp([][]int64{{1, 5, 9, 12, 20}})
	batch, err := src.Next(ctx)
	if err != nil {
		t.Fatalf("source: %v", err)
	}
	physical := batch.Vectors[0].Len()

	// Simulate what a Filter below leaves behind: three of five rows selected.
	batch.SelVec = SelectionVector{1, 2, 3}
	batch.Length = len(batch.SelVec)

	cases := []struct {
		name string
		expr Expr
	}{
		{"literal int64", &Literal{Val: int64(7), T: TypeInt64}},
		{"literal float64", &Literal{Val: 7.5, T: TypeFloat64}},
		{"literal bool", &Literal{Val: true, T: TypeBool}},
		{"literal date", &Literal{Val: int32(9000), T: TypeDate}},
		{"literal string", &Literal{Val: "x", T: TypeString}},
		{"empty AND", &AndExpr{}},
		{"empty OR", &OrExpr{}},
		{"case without else", &CaseExpr{
			Whens: []When{{
				Cond: &BinOp{Op: BinGT, Left: colRef("a", 0, TypeInt64),
					Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool},
				Result: &Literal{Val: int64(1), T: TypeInt64},
			}},
			T: TypeInt64,
		}},
		{"case with else", &CaseExpr{
			Whens: []When{{
				Cond: &BinOp{Op: BinGT, Left: colRef("a", 0, TypeInt64),
					Right: &Literal{Val: int64(4), T: TypeInt64}, T: TypeBool},
				Result: &Literal{Val: "hi", T: TypeString},
			}},
			Else: &Literal{Val: "lo", T: TypeString},
			T:    TypeString,
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := tc.expr.Eval(ctx, batch)
			if err != nil {
				t.Fatalf("eval: %v", err)
			}
			if got := v.Len(); got != physical {
				t.Errorf("vector length = %d, want physical length %d", got, physical)
			}
		})
	}
}

// TestExprLengthWithoutSelectionVector is the other half of the convention:
// with no selection vector, Batch.Length remains the authority, so a batch that
// carries no vectors at all (as several expression unit tests construct) still
// gets a vector of Batch.Length rows.
func TestExprLengthWithoutSelectionVector(t *testing.T) {
	ctx := context.Background()
	b := &Batch{Length: 5}
	v, err := (&Literal{Val: int64(3), T: TypeInt64}).Eval(ctx, b)
	if err != nil {
		t.Fatalf("eval: %v", err)
	}
	if got := v.Len(); got != 5 {
		t.Errorf("vector length = %d, want 5", got)
	}
}
