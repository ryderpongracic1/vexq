package exec

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/ryderpongracic1/vexq/storage"
)

// Expr is the column-at-a-time expression interface.
type Expr interface {
	// Eval evaluates the expression against b, returning a new Vector.
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
type Literal struct {
	Val any
	T   DataType
}

func (l *Literal) Type() DataType { return l.T }

func (l *Literal) Eval(_ context.Context, b *Batch) (Vector, error) {
	n := b.Length
	switch l.T {
	case TypeInt64:
		v := l.Val.(int64)
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = v
		}
		return &Int64Vector{Values: vals, NullBitmap: storage.FullBitmap(n)}, nil
	case TypeFloat64:
		v := l.Val.(float64)
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = v
		}
		return &Float64Vector{Values: vals, NullBitmap: storage.FullBitmap(n)}, nil
	case TypeBool:
		v := l.Val.(bool)
		bv := &BoolVector{
			Bits:       make([]byte, (n+7)/8),
			NullBitmap: storage.FullBitmap(n),
			Length:     n,
		}
		for i := 0; i < n; i++ {
			bv.Set(i, v)
		}
		return bv, nil
	case TypeDate:
		v := l.Val.(int32)
		vals := make([]int32, n)
		for i := range vals {
			vals[i] = v
		}
		return &DateVector{Values: vals, NullBitmap: storage.FullBitmap(n)}, nil
	case TypeString:
		// String literals are not materialized into a StringVector;
		// they are only used via BinOp/InExpr comparisons at the operator level.
		return nil, fmt.Errorf("expr: string literal Eval not supported (compare via BinOp)")
	default:
		return nil, fmt.Errorf("expr: unknown literal type %v", l.T)
	}
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
type BinOp struct {
	Op    BinOpKind
	Left  Expr
	Right Expr
	T     DataType // result type
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
		return evalCmp(b.Op, lv, rv, n)
	case BinAdd, BinSub, BinMul, BinDiv:
		return evalArith(b.Op, lv, rv, n)
	default:
		return nil, fmt.Errorf("expr: unknown BinOpKind %d", b.Op)
	}
}

func evalCmp(op BinOpKind, lv, rv Vector, n int) (*BoolVector, error) {
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: make([]byte, (n+7)/8),
		Length:     n,
	}
	// Propagate nulls: output row is null if either input is null.
	for i := 0; i < n; i++ {
		if !lv.IsNull(i) && !rv.IsNull(i) {
			storage.SetValidBit(out.NullBitmap, i)
		}
	}

	switch l := lv.(type) {
	case *Int64Vector:
		r := rv.(*Int64Vector)
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Set(i, cmpInt64(op, l.Values[i], r.Values[i]))
		}
	case *Float64Vector:
		r := rv.(*Float64Vector)
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Set(i, cmpFloat64(op, l.Values[i], r.Values[i]))
		}
	case *DateVector:
		r := rv.(*DateVector)
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Set(i, cmpInt32(op, l.Values[i], r.Values[i]))
		}
	default:
		return nil, fmt.Errorf("expr: cmp not supported for type %T", lv)
	}
	return out, nil
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

func evalArith(op BinOpKind, lv, rv Vector, n int) (Vector, error) {
	switch l := lv.(type) {
	case *Int64Vector:
		r := rv.(*Int64Vector)
		out := &Int64Vector{
			Values:     make([]int64, n),
			NullBitmap: make([]byte, (n+7)/8),
		}
		copy(out.NullBitmap, mergeNullBitmaps(lv.Nulls(), rv.Nulls(), n))
		for i := 0; i < n; i++ {
			if storage.IsNullBit(out.NullBitmap, i) {
				continue
			}
			out.Values[i] = applyArithInt64(op, l.Values[i], r.Values[i])
		}
		return out, nil
	case *Float64Vector:
		r := rv.(*Float64Vector)
		out := &Float64Vector{
			Values:     make([]float64, n),
			NullBitmap: make([]byte, (n+7)/8),
		}
		copy(out.NullBitmap, mergeNullBitmaps(lv.Nulls(), rv.Nulls(), n))
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

// mergeNullBitmaps returns a bitmap where a bit is valid (1) only if both
// input bitmaps have that bit valid.
func mergeNullBitmaps(a, b []byte, n int) []byte {
	size := (n + 7) / 8
	out := make([]byte, size)
	for i := 0; i < size; i++ {
		ai, bi := byte(0xFF), byte(0xFF)
		if i < len(a) {
			ai = a[i]
		}
		if i < len(b) {
			bi = b[i]
		}
		out[i] = ai & bi
	}
	return out
}

// ---- AndExpr ----------------------------------------------------------------

type AndExpr struct{ Children []Expr }

func (a *AndExpr) Type() DataType { return TypeBool }

func (a *AndExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	if len(a.Children) == 0 {
		return trueVector(b.Length), nil
	}
	result, err := a.Children[0].Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	rv := result.(*BoolVector)
	for _, child := range a.Children[1:] {
		cv, err := child.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		cv2 := cv.(*BoolVector)
		n := rv.Length
		for i := 0; i < (n+7)/8; i++ {
			rv.Bits[i] &= cv2.Bits[i]
			rv.NullBitmap[i] &= cv2.NullBitmap[i]
		}
	}
	return rv, nil
}

// ---- OrExpr -----------------------------------------------------------------

type OrExpr struct{ Children []Expr }

func (o *OrExpr) Type() DataType { return TypeBool }

func (o *OrExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	if len(o.Children) == 0 {
		return falseVector(b.Length), nil
	}
	result, err := o.Children[0].Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	rv := result.(*BoolVector)
	for _, child := range o.Children[1:] {
		cv, err := child.Eval(ctx, b)
		if err != nil {
			return nil, err
		}
		cv2 := cv.(*BoolVector)
		n := rv.Length
		for i := 0; i < (n+7)/8; i++ {
			rv.Bits[i] |= cv2.Bits[i]
			// A row is non-null if either side is non-null AND true,
			// or both sides are non-null. Simple: keep null if both are null.
			rv.NullBitmap[i] |= cv2.NullBitmap[i]
		}
	}
	return rv, nil
}

// ---- NotExpr ----------------------------------------------------------------

type NotExpr struct{ Child Expr }

func (n *NotExpr) Type() DataType { return TypeBool }

func (n *NotExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := n.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	rv := cv.(*BoolVector)
	for i := 0; i < (rv.Length+7)/8; i++ {
		rv.Bits[i] ^= rv.NullBitmap[i] // only flip bits that are valid (not null)
	}
	return rv, nil
}

// ---- IsNullExpr / IsNotNullExpr --------------------------------------------

type IsNullExpr struct{ Child Expr }

func (e *IsNullExpr) Type() DataType { return TypeBool }

func (e *IsNullExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: storage.FullBitmap(n),
		Length:     n,
	}
	for i := 0; i < n; i++ {
		// IS NULL is true when the source bit is 0 (null).
		out.Set(i, cv.IsNull(i))
	}
	return out, nil
}

type IsNotNullExpr struct{ Child Expr }

func (e *IsNotNullExpr) Type() DataType { return TypeBool }

func (e *IsNotNullExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: storage.FullBitmap(n),
		Length:     n,
	}
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
}

func (e *InExpr) Type() DataType { return TypeBool }

func (e *InExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	cv, err := e.Child.Eval(ctx, b)
	if err != nil {
		return nil, err
	}
	n := cv.Len()
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: make([]byte, (n+7)/8),
		Length:     n,
	}
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
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: make([]byte, (n+7)/8),
		Length:     n,
	}
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
type BetweenExpr struct {
	Child    Expr
	Lo, Hi   Expr
}

func (e *BetweenExpr) Type() DataType { return TypeBool }

func (e *BetweenExpr) Eval(ctx context.Context, b *Batch) (Vector, error) {
	loExpr := &BinOp{Op: BinGE, Left: e.Child, Right: e.Lo, T: TypeBool}
	hiExpr := &BinOp{Op: BinLE, Left: e.Child, Right: e.Hi, T: TypeBool}
	and := &AndExpr{Children: []Expr{loExpr, hiExpr}}
	return and.Eval(ctx, b)
}

// ---- StringEqExpr (fast path for string equality) --------------------------

// StringEqExpr evaluates col = literal for STRING columns, resolving the
// literal to a dictionary code to avoid string comparisons in the hot loop.
type StringEqExpr struct {
	ColIdx  int
	Literal string
	Negate  bool // true for col != literal
}

func (e *StringEqExpr) Type() DataType { return TypeBool }

func (e *StringEqExpr) Eval(_ context.Context, b *Batch) (Vector, error) {
	col, ok := b.Vectors[e.ColIdx].(*StringVector)
	if !ok {
		return nil, fmt.Errorf("expr: StringEqExpr: column %d is not STRING", e.ColIdx)
	}
	n := col.Len()
	out := &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: make([]byte, (n+7)/8),
		Length:     n,
	}
	if col.Dict == nil {
		return out, nil
	}
	code, found := col.Dict.Lookup(e.Literal)
	if !found {
		// Literal not in this row group's dict: no rows match (or all match for !=).
		if e.Negate {
			copy(out.NullBitmap, storage.FullBitmap(n))
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
		result = nullVector(e.T, b.Length)
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
		result = mergeVectors(cond, valV, result, b.Length)
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
	default:
		return &BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: make([]byte, (n+7)/8), Length: n}
	}
}

func trueVector(n int) *BoolVector {
	bv := &BoolVector{
		Bits:       storage.FullBitmap(n),
		NullBitmap: storage.FullBitmap(n),
		Length:     n,
	}
	return bv
}

func falseVector(n int) *BoolVector {
	return &BoolVector{
		Bits:       make([]byte, (n+7)/8),
		NullBitmap: storage.FullBitmap(n),
		Length:     n,
	}
}

// ---- BoolToSelVec ----------------------------------------------------------

// BoolToSelVec converts a BoolVector to a SelectionVector, respecting SelVec
// if the batch already has one.
func BoolToSelVec(b *Batch, bv *BoolVector) SelectionVector {
	var out SelectionVector
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
