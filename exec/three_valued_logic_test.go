package exec_test

import (
	"context"
	"testing"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// tri is a SQL boolean: false, true or NULL.
type tri int8

const (
	triFalse tri = iota
	triTrue
	triNull
)

func (v tri) String() string { return [...]string{"FALSE", "TRUE", "NULL"}[v] }

// fixedBools is an expression that evaluates to a fixed BoolVector. Its Bits
// are deliberately set for NULL rows, so a consumer that reads a NULL row's
// value bit without checking validity is caught.
type fixedBools struct{ vals []tri }

func (f fixedBools) Type() exec.DataType { return exec.TypeBool }
func (f fixedBools) Eval(_ context.Context, _ *exec.Batch) (exec.Vector, error) {
	n := len(f.vals)
	bv := &exec.BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: make([]byte, (n+7)/8), Length: n}
	for i, v := range f.vals {
		switch v {
		case triTrue:
			storage.SetValidBit(bv.NullBitmap, i)
			bv.Set(i, true)
		case triFalse:
			storage.SetValidBit(bv.NullBitmap, i)
		case triNull:
			bv.Set(i, true)
		}
	}
	return bv, nil
}

func readTri(t *testing.T, e exec.Expr, n int) []tri {
	t.Helper()
	v, err := e.Eval(context.Background(), &exec.Batch{Length: n})
	if err != nil {
		t.Fatal(err)
	}
	bv := v.(*exec.BoolVector)
	out := make([]tri, n)
	for i := range out {
		switch {
		case bv.IsNull(i):
			out[i] = triNull
		case bv.Get(i):
			out[i] = triTrue
		}
	}
	return out
}

func TestAndOrNotThreeValuedLogic(t *testing.T) {
	// Every (left, right) pair.
	var left, right []tri
	for _, l := range []tri{triFalse, triTrue, triNull} {
		for _, r := range []tri{triFalse, triTrue, triNull} {
			left = append(left, l)
			right = append(right, r)
		}
	}
	and := func(l, r tri) tri {
		switch {
		case l == triFalse || r == triFalse:
			return triFalse
		case l == triNull || r == triNull:
			return triNull
		}
		return triTrue
	}
	or := func(l, r tri) tri {
		switch {
		case l == triTrue || r == triTrue:
			return triTrue
		case l == triNull || r == triNull:
			return triNull
		}
		return triFalse
	}
	not := func(v tri) tri {
		switch v {
		case triTrue:
			return triFalse
		case triFalse:
			return triTrue
		}
		return triNull
	}

	n := len(left)
	gotAnd := readTri(t, &exec.AndExpr{Children: []exec.Expr{fixedBools{left}, fixedBools{right}}}, n)
	gotOr := readTri(t, &exec.OrExpr{Children: []exec.Expr{fixedBools{left}, fixedBools{right}}}, n)
	gotNotAnd := readTri(t, &exec.NotExpr{Child: &exec.AndExpr{Children: []exec.Expr{fixedBools{left}, fixedBools{right}}}}, n)
	gotNot := readTri(t, &exec.NotExpr{Child: fixedBools{left}}, n)
	for i := 0; i < n; i++ {
		l, r := left[i], right[i]
		if gotAnd[i] != and(l, r) {
			t.Errorf("%v AND %v = %v, want %v", l, r, gotAnd[i], and(l, r))
		}
		if gotOr[i] != or(l, r) {
			t.Errorf("%v OR %v = %v, want %v", l, r, gotOr[i], or(l, r))
		}
		if gotNotAnd[i] != not(and(l, r)) {
			t.Errorf("NOT (%v AND %v) = %v, want %v", l, r, gotNotAnd[i], not(and(l, r)))
		}
		if gotNot[i] != not(l) {
			t.Errorf("NOT %v = %v, want %v", l, gotNot[i], not(l))
		}
	}
}

func TestInExprNullSemantics(t *testing.T) {
	n := 3
	nulls := storage.FullBitmap(n)
	storage.SetNullBit(nulls, 2)
	col := &exec.Int64Vector{Values: []int64{2, 5, 0}, NullBitmap: nulls}
	batch := &exec.Batch{Vectors: []exec.Vector{col}, Length: n}
	ref := &exec.ColumnRef{Name: "x", Idx: 0, T: exec.TypeInt64}

	eval := func(e exec.Expr) []tri {
		v, err := e.Eval(context.Background(), batch)
		if err != nil {
			t.Fatal(err)
		}
		bv := v.(*exec.BoolVector)
		out := make([]tri, n)
		for i := range out {
			switch {
			case bv.IsNull(i):
				out[i] = triNull
			case bv.Get(i):
				out[i] = triTrue
			}
		}
		return out
	}
	check := func(name string, got []tri, want ...tri) {
		t.Helper()
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("%s row %d = %v, want %v", name, i, got[i], want[i])
			}
		}
	}

	in := &exec.InExpr{Child: ref, Set: []any{int64(2)}, HasNull: true}
	check("x IN (2, NULL)", eval(in), triTrue, triNull, triNull)
	check("x NOT IN (2, NULL)", eval(&exec.NotExpr{Child: &exec.InExpr{Child: ref, Set: []any{int64(2)}, HasNull: true}}), triFalse, triNull, triNull)
	check("x NOT IN (2)", eval(&exec.NotExpr{Child: &exec.InExpr{Child: ref, Set: []any{int64(2)}}}), triFalse, triTrue, triNull)
	// A set value of the wrong Go type never matches and must not panic.
	check("x IN (2.0 as float64)", eval(&exec.InExpr{Child: ref, Set: []any{float64(2)}}), triFalse, triFalse, triNull)
}
