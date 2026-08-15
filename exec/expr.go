package exec

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/ryderpongracic1/vexq/storage"
)

// Expr is the column-at-a-time expression interface.
//
// SIZING CONVENTION. An Expr evaluates over the batch's PHYSICAL rows, not its
// logical ones. Once a Filter has installed a selection vector, Batch.Length
// drops to the number of selected rows while the column vectors keep their full
// physical length; every vector an Expr produces must still hold one element per
// physical row, addressed by physical row index.
//
// The convention is forced by both ends of the tree. ColumnRef hands back the
// batch's own vector, so every node that combines a column with anything else —
// each comparison, each arithmetic op — is already sized physically and reads
// its operands at physical indices. At the top, BoolToSelVecInto indexes a
// predicate's result by the physical indices held in the incoming selection
// vector. A leaf sized from Batch.Length is therefore short, and a short operand
// either null-masks the batch's trailing physical rows (wrong results) or is
// read past its end by the 8-rows-at-a-time comparison kernels (panic). Leaves
// with no child to inherit a length from — Literal, an empty AndExpr or OrExpr,
// CaseExpr's implicit NULL else — MUST call evalLen(b) instead of b.Length.
//
// Evaluating unselected rows does work that is then discarded. That is the
// deliberate trade: the alternative is teaching every comparison and arithmetic
// kernel to walk a selection vector, giving up the branch-free 8-rows-per-byte
// inner loops. Compaction to the selected rows happens once, at the Project
// boundary (project.go).
type Expr interface {
	// Eval evaluates the expression against b, returning a Vector of
	// evalLen(b) rows.
	Eval(ctx context.Context, b *Batch) (Vector, error)
	Type() DataType
}

// ---- ColumnRef ---------------------------------------------------------------

// ColumnRef returns a column from the batch directly (zero copy).
type ColumnRef struct {
	Name string
	Idx  int
	T    DataType
}

func (c *ColumnRef) Type() DataType { return c.T }

func (c *ColumnRef) Eval(_ context.Context, b *Batch) (Vector, error) {
	if c.Idx < 0 || c.Idx >= len(b.Vectors) {
		return nil, fmt.Errorf("expr: column %q index %d out of range", c.Name, c.Idx)
	}
	return b.Vectors[c.Idx], nil
}

// ---- Literal ----------------------------------------------------------------

// Literal is a constant value broadcast to all rows.
//
// The broadcast vector is built once per Literal instance and reused across
// batches: every element already holds Val, so a later batch of the same or
// smaller length needs no refill. See scratch.go for the aliasing contract —
// consumers treat the returned vector as read-only, and Project copies it
// rather than passing it downstream.
type Literal struct {
	Val any
	T   DataType

	// Cached broadcast vector, grown on demand. filled records how many
	// leading elements already hold Val, so growing only fills the new tail.
	cacheInt64   *Int64Vector
	cacheFloat64 *Float64Vector
	cacheDate    *DateVector
	cacheBool    *BoolVector
	cacheString  *StringVector
	litCode      uint32
	filled       int
	validBmp     []byte
}

func (l *Literal) Type() DataType { return l.T }

func (l *Literal) Eval(_ context.Context, b *Batch) (Vector, error) {
	n := evalLen(b)
	switch l.T {
	case TypeInt64:
		v := l.Val.(int64)
		if l.cacheInt64 == nil || cap(l.cacheInt64.Values) < n {
			l.cacheInt64 = &Int64Vector{Values: make([]int64, n)}
			l.filled = 0
		}
		vals := l.cacheInt64.Values[:n]
		for i := l.filled; i < n; i++ {
			vals[i] = v
		}
		l.cacheInt64.Values = vals
		l.cacheInt64.NullBitmap = acquireValidBitmap(&l.validBmp, n)
		if n > l.filled {
			l.filled = n
		}
		return l.cacheInt64, nil
	case TypeFloat64:
		v := l.Val.(float64)
		if l.cacheFloat64 == nil || cap(l.cacheFloat64.Values) < n {
			l.cacheFloat64 = &Float64Vector{Values: make([]float64, n)}
			l.filled = 0
		}
		vals := l.cacheFloat64.Values[:n]
		for i := l.filled; i < n; i++ {
			vals[i] = v
		}
		l.cacheFloat64.Values = vals
		l.cacheFloat64.NullBitmap = acquireValidBitmap(&l.validBmp, n)
		if n > l.filled {
			l.filled = n
		}
		return l.cacheFloat64, nil
	case TypeBool:
		v := l.Val.(bool)
		if l.cacheBool == nil || cap(l.cacheBool.Bits) < (n+7)/8 {
			l.cacheBool = &BoolVector{Bits: make([]byte, (n+7)/8)}
			l.filled = 0
		}
		l.cacheBool.Bits = l.cacheBool.Bits[:(n+7)/8]
		l.cacheBool.Length = n
		for i := l.filled; i < n; i++ {
			l.cacheBool.Set(i, v)
		}
		// When n shrinks, the final byte still carries bits set for rows that no
		// longer exist. Mask them so the vector is byte-identical to a freshly
		// built one — no consumer reads past Length, but parity keeps the
		// contract in scratch.go literally true. Masking discards bits the fill
		// loop had already written, so record filled as exactly n: a later,
		// larger batch refills from here rather than trusting the masked tail.
		if n%8 != 0 {
			l.cacheBool.Bits[(n+7)/8-1] &= 1<<uint(n%8) - 1
		}
		l.cacheBool.NullBitmap = acquireValidBitmap(&l.validBmp, n)
		l.filled = n
		return l.cacheBool, nil
	case TypeDate:
		v := l.Val.(int32)
		if l.cacheDate == nil || cap(l.cacheDate.Values) < n {
			l.cacheDate = &DateVector{Values: make([]int32, n)}
			l.filled = 0
		}
		vals := l.cacheDate.Values[:n]
		for i := l.filled; i < n; i++ {
			vals[i] = v
		}
		l.cacheDate.Values = vals
		l.cacheDate.NullBitmap = acquireValidBitmap(&l.validBmp, n)
		if n > l.filled {
			l.filled = n
		}
		return l.cacheDate, nil
	case TypeString:
		s := l.Val.(string)
		if l.cacheString == nil || cap(l.cacheString.Codes) < n {
			db := storage.NewDictBuilder()
			l.litCode = db.Add(s)
			l.cacheString = newStringVector(db, make([]uint32, n), nil)
			l.filled = 0
		}
		codes := l.cacheString.Codes[:n]
		for i := l.filled; i < n; i++ {
			codes[i] = l.litCode
		}
		l.cacheString.Codes = codes
		l.cacheString.NullBitmap = acquireValidBitmap(&l.validBmp, n)
		if n > l.filled {
			l.filled = n
		}
		return l.cacheString, nil
	default:
		return nil, fmt.Errorf("expr: unknown literal type %v", l.T)
	}
}

// ---- CastIntToFloatExpr -----------------------------------------------------

// CastIntToFloatExpr wraps an int64-returning expression and converts its
// output to a Float64Vector.  Used by the planner to promote int64 operands
// in mixed-type arithmetic so evalArith always receives matching types.
//
// The output vector is per-instance scratch (see scratch.go). Its null bitmap
// aliases the input's, as it always has: the cast never changes validity, and
// the input bitmap is read-only to this node.
type CastIntToFloatExpr struct {
	Inner Expr

	out *Float64Vector
}

func (c *CastIntToFloatExpr) Type() DataType { return TypeFloat64 }

func (c *CastIntToFloatExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	v, err := c.Inner.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	iv, ok := v.(*Int64Vector)
	if !ok {
		return nil, fmt.Errorf("expr: CastIntToFloat: expected *Int64Vector, got %T", v)
	}
	n := iv.Len()
	if c.out == nil || cap(c.out.Values) < n {
		c.out = &Float64Vector{Values: make([]float64, n)}
	}
	out := c.out
	out.Values = out.Values[:n]
	out.NullBitmap = iv.NullBitmap
	for i := 0; i < n; i++ {
		out.Values[i] = float64(iv.Values[i])
	}
	return out, nil
}

// ---- BinOpKind --------------------------------------------------------------

type BinOpKind uint8

const (
	BinEQ BinOpKind = iota
	BinNE
	BinLT
	BinLE
	BinGT
	BinGE
	BinAdd
	BinSub
	BinMul
	BinDiv
)

// ---- BinOp ------------------------------------------------------------------

// BinOp evaluates a binary operation over two column expressions.
// For comparison operators (EQ..GE), the result is a BoolVector.
// For arithmetic, the result has the same type as the inputs.
//
// Both output vectors are per-instance scratch reused across batches (see
// scratch.go): one BoolVector for comparisons, one typed vector for arithmetic.
type BinOp struct {
	Op    BinOpKind
	Left  Expr
	Right Expr
	T     DataType // result type

	cmpOut   *BoolVector
	arithI64 *Int64Vector
	arithF64 *Float64Vector
}

func (b *BinOp) Type() DataType { return b.T }

func (b *BinOp) Eval(ctx context.Context, batch *Batch) (Vector, error) {
	lv, err := b.Left.Eval(ctx, batch)
	if err != nil {
		return nil, err
	}
	rv, err := b.Right.Eval(ctx, batch)
	if err != nil {
		return nil, err
	}
	n := lv.Len()

	switch b.Op {
	case BinEQ, BinNE, BinLT, BinLE, BinGT, BinGE:
		return evalCmp(b.Op, lv, rv, n, &b.cmpOut)
	case BinAdd, BinSub, BinMul, BinDiv:
		return b.evalArith(b.Op, lv, rv, n)
	default:
		return nil, fmt.Errorf("expr: unknown BinOpKind %d", b.Op)
	}
}

// evalCmp writes the comparison of lv and rv into the caller's scratch slot,
// which it grows on first use. out is zeroed before use, so the result is
// identical to what a freshly allocated BoolVector would hold.
func evalCmp(op BinOpKind, lv, rv Vector, n int, slot **BoolVector) (*BoolVector, error) {
	out := acquireBoolVector(slot, n)
	// Byte-level null propagation: a row is valid only when both inputs are
	// non-null. Processing 8 rows per iteration avoids per-row bit extraction.
	la, ra := lv.Nulls(), rv.Nulls()
	for i := range out.NullBitmap {
		lb, rb := byte(0xFF), byte(0xFF)
		if i < len(la) {
			lb = la[i]
		}
		if i < len(ra) {
			rb = ra[i]
		}
		out.NullBitmap[i] = lb & rb
	}

	switch l := lv.(type) {
	case *Int64Vector:
		evalCmpInt64(op, l.Values, rv.(*Int64Vector).Values, out, n)
	case *Float64Vector:
		evalCmpFloat64(op, l.Values, rv.(*Float64Vector).Values, out, n)
	case *DateVector:
		evalCmpDate(op, l.Values, rv.(*DateVector).Values, out, n)
	default:
		return nil, fmt.Errorf("expr: cmp not supported for type %T", lv)
	}
	return out, nil
}

// evalCmpInt64 writes comparison results into out.Bits 8 rows at a time.
// The switch on op is hoisted outside the inner loop — the branch predictor
// learns it on the first iteration and incurs zero misprediction cost thereafter.
// Each outer iteration packs 8 boolean results into one byte and applies the
// null mask in a single AND, replacing 8 separate bit-scatter operations.
func evalCmpInt64(op BinOpKind, lv, rv []int64, out *BoolVector, n int) {
	bits, mask := out.Bits, out.NullBitmap
	i := 0
	switch op {
	case BinEQ:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] == rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] == rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] == rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] == rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] == rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] == rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] == rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] == rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinNE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] != rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] != rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] != rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] != rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] != rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] != rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] != rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] != rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] < rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] < rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] < rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] < rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] < rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] < rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] < rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] < rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] <= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] <= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] <= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] <= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] <= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] <= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] <= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] <= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] > rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] > rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] > rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] > rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] > rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] > rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] > rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] > rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] >= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] >= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] >= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] >= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] >= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] >= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] >= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] >= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	}
	for ; i < n; i++ {
		if !storage.IsNullBit(mask, i) {
			out.Set(i, cmpInt64(op, lv[i], rv[i]))
		}
	}
}

// evalCmpFloat64 mirrors evalCmpInt64 for float64 columns (Q6 discount/quantity).
func evalCmpFloat64(op BinOpKind, lv, rv []float64, out *BoolVector, n int) {
	bits, mask := out.Bits, out.NullBitmap
	i := 0
	switch op {
	case BinEQ:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] == rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] == rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] == rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] == rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] == rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] == rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] == rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] == rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinNE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] != rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] != rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] != rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] != rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] != rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] != rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] != rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] != rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] < rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] < rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] < rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] < rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] < rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] < rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] < rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] < rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] <= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] <= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] <= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] <= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] <= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] <= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] <= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] <= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] > rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] > rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] > rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] > rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] > rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] > rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] > rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] > rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] >= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] >= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] >= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] >= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] >= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] >= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] >= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] >= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	}
	for ; i < n; i++ {
		if !storage.IsNullBit(mask, i) {
			out.Set(i, cmpFloat64(op, lv[i], rv[i]))
		}
	}
}

// evalCmpDate mirrors evalCmpInt64 for int32 date columns (Q1/Q6/Q3/Q12 date predicates).
func evalCmpDate(op BinOpKind, lv, rv []int32, out *BoolVector, n int) {
	bits, mask := out.Bits, out.NullBitmap
	i := 0
	switch op {
	case BinEQ:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] == rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] == rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] == rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] == rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] == rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] == rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] == rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] == rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinNE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] != rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] != rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] != rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] != rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] != rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] != rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] != rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] != rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] < rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] < rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] < rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] < rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] < rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] < rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] < rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] < rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinLE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] <= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] <= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] <= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] <= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] <= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] <= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] <= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] <= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGT:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] > rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] > rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] > rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] > rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] > rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] > rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] > rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] > rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	case BinGE:
		for ; i+8 <= n; i += 8 {
			b := i >> 3
			var byte_ uint8
			if lv[i+0] >= rv[i+0] {
				byte_ |= 0x01
			}
			if lv[i+1] >= rv[i+1] {
				byte_ |= 0x02
			}
			if lv[i+2] >= rv[i+2] {
				byte_ |= 0x04
			}
			if lv[i+3] >= rv[i+3] {
				byte_ |= 0x08
			}
			if lv[i+4] >= rv[i+4] {
				byte_ |= 0x10
			}
			if lv[i+5] >= rv[i+5] {
				byte_ |= 0x20
			}
			if lv[i+6] >= rv[i+6] {
				byte_ |= 0x40
			}
			if lv[i+7] >= rv[i+7] {
				byte_ |= 0x80
			}
			bits[b] = byte_ & mask[b]
		}
	}
	for ; i < n; i++ {
		if !storage.IsNullBit(mask, i) {
			out.Set(i, cmpInt32(op, lv[i], rv[i]))
		}
	}
}

func cmpInt64(op BinOpKind, a, b int64) bool {
	switch op {
	case BinEQ:
		return a == b
	case BinNE:
		return a != b
	case BinLT:
		return a < b
	case BinLE:
		return a <= b
	case BinGT:
		return a > b
	case BinGE:
		return a >= b
	}
	return false
}

func cmpFloat64(op BinOpKind, a, b float64) bool {
	switch op {
	case BinEQ:
		return a == b
	case BinNE:
		return a != b
	case BinLT:
		return a < b
	case BinLE:
		return a <= b
	case BinGT:
		return a > b
	case BinGE:
		return a >= b
	}
	return false
}

func cmpInt32(op BinOpKind, a, b int32) bool {
	switch op {
	case BinEQ:
		return a == b
	case BinNE:
		return a != b
	case BinLT:
		return a < b
	case BinLE:
		return a <= b
	case BinGT:
		return a > b
	case BinGE:
		return a >= b
	}
	return false
}

// evalArith writes the arithmetic result into this BinOp's per-instance scratch
// vector (see scratch.go). The scratch is zeroed on acquisition, so rows left
// unwritten because either input was null read back as zero, exactly as they did
// when every batch allocated a fresh vector.
func (b *BinOp) evalArith(op BinOpKind, lv, rv Vector, n int) (Vector, error) {
	switch l := lv.(type) {
	case *Int64Vector:
		r, ok := rv.(*Int64Vector)
		if !ok {
			return nil, fmt.Errorf("expr: arithmetic type mismatch: left is *Int64Vector but right is %T (missing plan-time coercion?)", rv)
		}
		out := acquireInt64Vector(&b.arithI64, n)
		mergeNullBitmapsInto(out.NullBitmap, lv.Nulls(), rv.Nulls(), n)
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Values[i] = applyArithInt64(op, l.Values[i], r.Values[i])
		}
		return out, nil
	case *Float64Vector:
		r, ok := rv.(*Float64Vector)
		if !ok {
			return nil, fmt.Errorf("expr: arithmetic type mismatch: left is *Float64Vector but right is %T (missing plan-time coercion?)", rv)
		}
		out := acquireFloat64Vector(&b.arithF64, n)
		mergeNullBitmapsInto(out.NullBitmap, lv.Nulls(), rv.Nulls(), n)
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Values[i] = applyArithFloat64(op, l.Values[i], r.Values[i])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expr: arithmetic not supported for type %T", lv)
	}
}

func applyArithInt64(op BinOpKind, a, b int64) int64 {
	switch op {
	case BinAdd:
		return a + b
	case BinSub:
		return a - b
	case BinMul:
		return a * b
	case BinDiv:
		if b == 0 {
			return 0
		}
		return a / b
	}
	return 0
}

func applyArithFloat64(op BinOpKind, a, b float64) float64 {
	switch op {
	case BinAdd:
		return a + b
	case BinSub:
		return a - b
	case BinMul:
		return a * b
	case BinDiv:
		if b == 0 {
			return math.NaN()
		}
		return a / b
	}
	return 0
}

// ---- AndExpr ----------------------------------------------------------------

// AndExpr conjoins its children. The result goes into this node's own scratch
// vector: an Expr's result is read-only to its consumers (see scratch.go), so a
// parent must not fold into a child's buffer.
type AndExpr struct {
	Children []Expr

	out *BoolVector
}

func (a *AndExpr) Type() DataType { return TypeBool }

func (a *AndExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	if len(a.Children) == 0 {
		n := evalLen(b)
		out := acquireBoolVector(&a.out, n)
		setAllValid(out.Bits, n)
		setAllValid(out.NullBitmap, n)
		return out, nil
	}
	first, err := a.Children[0].Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	fv := first.(*BoolVector)
	n := fv.Length
	out := acquireBoolVector(&a.out, n)
	copy(out.Bits, fv.Bits)
	copy(out.NullBitmap, fv.NullBitmap)
	for _, child := range a.Children[1:] {
		cv, err := child.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		cv2 := cv.(*BoolVector)
		for i := 0; i < (n+7)/8; i++ {
			out.Bits[i] &= cv2.Bits[i]
			out.NullBitmap[i] &= cv2.NullBitmap[i]
		}
	}
	return out, nil
}

// ---- OrExpr -----------------------------------------------------------------

// OrExpr disjoins its children into this node's own scratch vector, for the same
// reason AndExpr does.
type OrExpr struct {
	Children []Expr

	out *BoolVector
}

func (o *OrExpr) Type() DataType { return TypeBool }

func (o *OrExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	if len(o.Children) == 0 {
		n := evalLen(b)
		out := acquireBoolVector(&o.out, n)
		setAllValid(out.NullBitmap, n)
		return out, nil
	}
	first, err := o.Children[0].Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	fv := first.(*BoolVector)
	n := fv.Length
	out := acquireBoolVector(&o.out, n)
	copy(out.Bits, fv.Bits)
	copy(out.NullBitmap, fv.NullBitmap)
	for _, child := range o.Children[1:] {
		cv, err := child.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		cv2 := cv.(*BoolVector)
		for i := 0; i < (n+7)/8; i++ {
			out.Bits[i] |= cv2.Bits[i]
			// A row is non-null if either side is non-null AND true,
			// or both sides are non-null. Simple: keep null if both are null.
			out.NullBitmap[i] |= cv2.NullBitmap[i]
		}
	}
	return out, nil
}

// ---- NotExpr ----------------------------------------------------------------

// NotExpr negates its child into this node's own scratch vector.
type NotExpr struct {
	Child Expr

	out *BoolVector
}

func (n *NotExpr) Type() DataType { return TypeBool }

func (n *NotExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := n.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	child, ok := cv.(*BoolVector)
	if !ok {
		return nil, fmt.Errorf("expr: NOT requires a boolean operand, got %T", cv)
	}
	out := acquireBoolVector(&n.out, child.Length)
	for i := 0; i < (child.Length+7)/8; i++ {
		out.NullBitmap[i] = child.NullBitmap[i]
		out.Bits[i] = child.Bits[i] ^ child.NullBitmap[i] // only flip bits that are valid (not null)
	}
	return out, nil
}

// ---- IsNullExpr / IsNotNullExpr --------------------------------------------

type IsNullExpr struct {
	Child Expr

	out *BoolVector
}

func (e *IsNullExpr) Type() DataType { return TypeBool }

func (e *IsNullExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := acquireBoolVector(&e.out, n)
	setAllValid(out.NullBitmap, n)
	for i := 0; i < n; i++ {
		// IS NULL is true when the source bit is 0 (null).
		out.Set(i, cv.IsNull(i))
	}
	return out, nil
}

type IsNotNullExpr struct {
	Child Expr

	out *BoolVector
}

func (e *IsNotNullExpr) Type() DataType { return TypeBool }

func (e *IsNotNullExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := acquireBoolVector(&e.out, n)
	setAllValid(out.NullBitmap, n)
	for i := 0; i < n; i++ {
		out.Set(i, !cv.IsNull(i))
	}
	return out, nil
}

// ---- InExpr -----------------------------------------------------------------

// InExpr checks whether a column value is in a fixed set of literals.
type InExpr struct {
	Child Expr
	// Set holds typed values matching Child's type.
	Set []any

	out *BoolVector
}

func (e *InExpr) Type() DataType { return TypeBool }

func (e *InExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := acquireBoolVector(&e.out, n)
	for i := 0; i < n; i++ {
		if cv.IsNull(i) {
			continue
		}
		storage.SetValidBit(out.NullBitmap, i)
		found := false
		switch col := cv.(type) {
		case *Int64Vector:
			for _, sv := range e.Set {
				if col.Values[i] == sv.(int64) {
					found = true
					break
				}
			}
		case *Float64Vector:
			for _, sv := range e.Set {
				if col.Values[i] == sv.(float64) {
					found = true
					break
				}
			}
		case *StringVector:
			for _, sv := range e.Set {
				if col.Get(i) == sv.(string) {
					found = true
					break
				}
			}
		}
		out.Set(i, found)
	}
	return out, nil
}

// ---- LikeExpr ---------------------------------------------------------------

// LikeExpr implements SQL LIKE with % and _ wildcards (no regex).
type LikeExpr struct {
	Child   Expr
	Pattern string // SQL LIKE pattern

	out *BoolVector
}

func (e *LikeExpr) Type() DataType { return TypeBool }

func (e *LikeExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	col, ok := cv.(*StringVector)
	if !ok {
		return nil, fmt.Errorf("expr: LIKE requires STRING column")
	}
	n := col.Len()
	out := acquireBoolVector(&e.out, n)
	for i := 0; i < n; i++ {
		if col.IsNull(i) {
			continue
		}
		storage.SetValidBit(out.NullBitmap, i)
		out.Set(i, likeMatch(e.Pattern, col.Get(i)))
	}
	return out, nil
}

// likeMatch implements SQL LIKE: % matches any sequence, _ matches one char.
func likeMatch(pattern, s string) bool {
	return likeMatchRec(pattern, s)
}

func likeMatchRec(p, s string) bool {
	for len(p) > 0 {
		switch p[0] {
		case '%':
			p = p[1:]
			if len(p) == 0 {
				return true
			}
			for i := 0; i <= len(s); i++ {
				if likeMatchRec(p, s[i:]) {
					return true
				}
			}
			return false
		case '_':
			if len(s) == 0 {
				return false
			}
			p = p[1:]
			s = s[1:]
		default:
			if len(s) == 0 || p[0] != s[0] {
				return false
			}
			p = p[1:]
			s = s[1:]
		}
	}
	return len(s) == 0
}

// ---- BetweenExpr ------------------------------------------------------------

// BetweenExpr implements BETWEEN lo AND hi (inclusive).
//
// It rewrites to (child >= lo) AND (child <= hi). The rewritten tree is built
// once and cached on the instance: rebuilding it per batch allocated three
// expression nodes each time and, worse, discarded their scratch buffers before
// they could be reused.
type BetweenExpr struct {
	Child  Expr
	Lo, Hi Expr

	rewritten *AndExpr
}

func (e *BetweenExpr) Type() DataType { return TypeBool }

func (e *BetweenExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	if e.rewritten == nil {
		e.rewritten = &AndExpr{Children: []Expr{
			&BinOp{Op: BinGE, Left: e.Child, Right: e.Lo, T: TypeBool},
			&BinOp{Op: BinLE, Left: e.Child, Right: e.Hi, T: TypeBool},
		}}
	}
	return e.rewritten.Eval(ctx, b)
}

// ---- StringEqExpr (fast path for string equality) --------------------------

// StringEqExpr evaluates col = literal for STRING columns, resolving the
// literal to a dictionary code to avoid string comparisons in the hot loop.
type StringEqExpr struct {
	ColIdx  int
	Literal string
	Negate  bool // true for col != literal

	out *BoolVector
}

func (e *StringEqExpr) Type() DataType { return TypeBool }

func (e *StringEqExpr) Eval(_ context.Context, b *Batch) (Vector, error) {
	col, ok := b.Vectors[e.ColIdx].(*StringVector)
	if !ok {
		return nil, fmt.Errorf("expr: StringEqExpr: column %d is not STRING", e.ColIdx)
	}
	n := col.Len()
	out := acquireBoolVector(&e.out, n)
	if col.Dict == nil {
		return out, nil
	}
	code, found := col.Dict.Lookup(e.Literal)
	if !found {
		// Literal not in this row group's dict: no rows match (or all match for !=).
		if e.Negate {
			setAllValid(out.NullBitmap, n)
			for i := 0; i < (n+7)/8; i++ {
				out.Bits[i] = out.NullBitmap[i]
			}
		}
		return out, nil
	}
	for i := 0; i < n; i++ {
		if col.IsNull(i) {
			continue
		}
		storage.SetValidBit(out.NullBitmap, i)
		match := col.Codes[i] == code
		if e.Negate {
			match = !match
		}
		out.Set(i, match)
	}
	return out, nil
}

// ---- CaseExpr ---------------------------------------------------------------

// When is one branch of a CASE expression.
type When struct {
	Cond   Expr // must produce BoolVector
	Result Expr
}

// CaseExpr evaluates CASE WHEN cond THEN result ... ELSE else END.
type CaseExpr struct {
	Whens []When
	Else  Expr
	T     DataType
}

func (e *CaseExpr) Type() DataType { return e.T }

func (e *CaseExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	n := evalLen(b)
	// Start with the ELSE result and then overwrite with WHEN results in
	// reverse priority order (last WHEN overwrites earlier ones).
	// This is simpler than tracking which rows have been matched.
	var result Vector
	var err error
	if e.Else != nil {
		result, err = e.Else.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
	} else {
		result = nullVector(e.T, n)
	}

	// Apply WHENs in reverse so the first matching WHEN wins.
	for i := len(e.Whens) - 1; i >= 0; i-- {
		w := e.Whens[i]
		condV, err := w.Cond.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		cond := condV.(*BoolVector)
		valV, err := w.Result.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		result = mergeVectors(cond, valV, result, n)
	}
	return result, nil
}

// mergeVectors selects valV[i] where cond is true/valid, otherwise keeps base[i].
func mergeVectors(cond *BoolVector, valV, base Vector, n int) Vector {
	switch bv := base.(type) {
	case *Int64Vector:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		vv := valV.(*Int64Vector)
		copy(out.Values, bv.Values)
		copy(out.NullBitmap, bv.NullBitmap)
		for i := 0; i < n; i++ {
			if !cond.IsNull(i) && cond.Get(i) {
				out.Values[i] = vv.Values[i]
				if vv.IsNull(i) {
					storage.SetNullBit(out.NullBitmap, i)
				} else {
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
		}
		return out
	case *Float64Vector:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		vv := valV.(*Float64Vector)
		copy(out.Values, bv.Values)
		copy(out.NullBitmap, bv.NullBitmap)
		for i := 0; i < n; i++ {
			if !cond.IsNull(i) && cond.Get(i) {
				out.Values[i] = vv.Values[i]
				if vv.IsNull(i) {
					storage.SetNullBit(out.NullBitmap, i)
				} else {
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
		}
		return out
	case *StringVector:
		vv := valV.(*StringVector)
		// Build a unified dictionary from both base and value vectors.
		db := storage.NewDictBuilder()
		// Remap base dict entries.
		var baseRemap []uint32
		if bv.Dict != nil {
			baseRemap = make([]uint32, bv.Dict.Len())
			for i := 0; i < bv.Dict.Len(); i++ {
				baseRemap[i] = db.Add(bv.Dict.Get(uint32(i)))
			}
		}
		// Remap val dict entries.
		var valRemap []uint32
		if vv.Dict != nil {
			valRemap = make([]uint32, vv.Dict.Len())
			for i := 0; i < vv.Dict.Len(); i++ {
				valRemap[i] = db.Add(vv.Dict.Get(uint32(i)))
			}
		}
		codes := make([]uint32, n)
		nullBmp := make([]byte, (n+7)/8)
		for i := 0; i < n; i++ {
			if !cond.IsNull(i) && cond.Get(i) {
				if vv.IsNull(i) {
					// null from val: leave code 0, bit stays 0 (null)
				} else {
					codes[i] = valRemap[vv.Codes[i]]
					storage.SetValidBit(nullBmp, i)
				}
			} else {
				if bv.IsNull(i) {
					// null from base: leave code 0, bit stays 0 (null)
				} else {
					codes[i] = baseRemap[bv.Codes[i]]
					storage.SetValidBit(nullBmp, i)
				}
			}
		}
		return newStringVector(db, codes, nullBmp)
	default:
		return base
	}
}

// nullVector returns a fully-null vector of the given type and length.
func nullVector(t DataType, n int) Vector {
	switch t {
	case TypeInt64:
		return &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
	case TypeFloat64:
		return &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
	case TypeString:
		return &StringVector{Codes: make([]uint32, n), Dict: nil, NullBitmap: make([]byte, (n+7)/8)}
	default:
		return &BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: make([]byte, (n+7)/8), Length: n}
	}
}

// ---- BoolToSelVec ----------------------------------------------------------

// BoolToSelVec converts a BoolVector to a SelectionVector, respecting SelVec
// if the batch already has one. It allocates; hot callers should use
// BoolToSelVecInto with a buffer they own.
func BoolToSelVec(b *Batch, bv *BoolVector) SelectionVector {
	return BoolToSelVecInto(b, bv, nil)
}

// BoolToSelVecInto is BoolToSelVec writing into dst's array when it is large
// enough, so a caller that owns dst pays no allocation per batch. dst must not
// alias b.SelVec: the two are read and written in the same pass. Filter
// guarantees this by giving every Filter instance its own buffer, so a stacked
// Filter reads its child's vector and writes its own.
func BoolToSelVecInto(b *Batch, bv *BoolVector, dst SelectionVector) SelectionVector {
	out := growSelVec(dst, bv.Length)
	if b.SelVec == nil {
		for i := 0; i < bv.Length; i++ {
			if !bv.IsNull(i) && bv.Get(i) {
				out = append(out, uint16(i))
			}
		}
	} else {
		for _, idx := range b.SelVec {
			i := int(idx)
			if !bv.IsNull(i) && bv.Get(i) {
				out = append(out, idx)
			}
		}
	}
	return out
}

// ---- StringContains (helper for LIKE '%foo%') -------------------------------

func StringContains(s, sub string) bool { return strings.Contains(s, sub) }
