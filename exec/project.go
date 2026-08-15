package exec

import (
	"context"
	"fmt"

	"github.com/ryderpongracic1/vexq/storage"
)

// ProjectExpr pairs a named output column with the expression that produces it.
type ProjectExpr struct {
	Name string
	Expr Expr
}

// Project evaluates a list of expressions against each input Batch and
// produces a new Batch with the results.  It respects SelVec from upstream
// Filter operators by materialising only the selected rows.
//
// Buffer ownership contract: Project is the boundary between reused expression
// scratch (see scratch.go) and the batches it hands downstream.
//   - With a selection vector, materialize() copies the selected rows into fresh
//     vectors, as it always has.
//   - Without one, a bare column reference still passes the input vector through
//     untouched — so Project's output aliases TableScan's decode buffers exactly
//     as it did before this change — while any computed expression is copied,
//     because its vector is scratch owned by an Expr node that will overwrite it
//     on the next batch. That copy replaces the fresh vector the evaluator used
//     to allocate, so it costs no extra allocation.
//
// The upshot is that Project's output never aliases operator scratch, which is
// what lets a consumer hold a Project batch across Next() as long as it did
// before.
type Project struct {
	child  Operator
	exprs  []ProjectExpr
	schema Schema

	// passthrough[i] is true when exprs[i] is a bare column reference, whose
	// Eval returns the input vector itself rather than a scratch buffer.
	passthrough []bool
}

func NewProject(child Operator, exprs []ProjectExpr) (*Project, error) {
	if len(exprs) == 0 {
		return nil, fmt.Errorf("exec: project: no expressions")
	}
	fields := make([]Field, len(exprs))
	passthrough := make([]bool, len(exprs))
	for i, pe := range exprs {
		fields[i] = Field{Name: pe.Name, Type: pe.Expr.Type(), Nullable: true}
		_, passthrough[i] = pe.Expr.(*ColumnRef)
	}
	return &Project{
		child:       child,
		exprs:       exprs,
		schema:      Schema{Fields: fields},
		passthrough: passthrough,
	}, nil
}

func (p *Project) Schema() Schema { return p.schema }

func (p *Project) Next(ctx context.Context) (*Batch, error) {
	batch, err := p.child.Next(ctx)
	if err != nil {
		return nil, fmt.Errorf("exec: project: %w", err)
	}
	if batch == nil {
		return nil, nil
	}

	// Save the original logical length and selection vector.
	origLen := batch.Length
	origSel := batch.SelVec

	// When a SelVec is active, expressions like Literal.Eval use batch.Length
	// to size broadcast vectors.  But ColumnRef.Eval returns the raw physical
	// vector (full block size).  The mismatch causes index-out-of-range in
	// evalArith when the literal vector is shorter than the column vector.
	//
	// Fix: temporarily set batch.Length to the physical row count so all
	// expression evaluators produce consistently-sized vectors.  Then
	// materialize the selected rows afterward.
	if origSel != nil {
		batch.Length = physicalLen(batch)
	}

	outVecs := make([]Vector, len(p.exprs))
	for i, pe := range p.exprs {
		raw, err := pe.Expr.Eval(ctx, batch)
		if err != nil {
			// Restore batch before returning.
			batch.Length = origLen
			batch.SelVec = origSel
			return nil, fmt.Errorf("exec: project: eval %q: %w", pe.Name, err)
		}
		// If there's a selection vector, materialise only selected rows into
		// freshly allocated vectors (important: output must not alias reused
		// scan buffers from TableScan).  Without one, a computed expression's
		// vector is scratch owned by the Expr node, so copy it; a bare column
		// reference is passed through as before.
		if origSel != nil {
			raw = materialize(raw, origSel)
		} else if !p.passthrough[i] {
			raw = copyVector(raw, raw.Len())
		}
		outVecs[i] = raw
	}

	// Restore the input batch (defensive; callers should not reuse it, but
	// this keeps the contract clean).
	batch.Length = origLen
	batch.SelVec = origSel

	outLen := origLen
	if origSel != nil {
		outLen = len(origSel)
	}
	return &Batch{
		Schema:  p.schema,
		Vectors: outVecs,
		Length:  outLen,
		// No SelVec on output: rows are already materialized.
	}, nil
}

// physicalLen returns the number of rows in the raw underlying vectors.
// This is the true allocation size, independent of any selection vector.
func physicalLen(b *Batch) int {
	if len(b.Vectors) > 0 {
		return b.Vectors[0].Len()
	}
	return b.Length
}

func (p *Project) Close() error { return p.child.Close() }

// materialize compacts a vector down to the rows indicated by sel.
func materialize(v Vector, sel SelectionVector) Vector {
	n := len(sel)
	switch src := v.(type) {
	case *Int64Vector:
		out := &Int64Vector{
			Values:     make([]int64, n),
			NullBitmap: make([]byte, (n+7)/8),
		}
		for i, idx := range sel {
			out.Values[i] = src.Values[idx]
			if !src.IsNull(int(idx)) {
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case *Float64Vector:
		out := &Float64Vector{
			Values:     make([]float64, n),
			NullBitmap: make([]byte, (n+7)/8),
		}
		for i, idx := range sel {
			out.Values[i] = src.Values[idx]
			if !src.IsNull(int(idx)) {
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case *BoolVector:
		out := &BoolVector{
			Bits:       make([]byte, (n+7)/8),
			NullBitmap: make([]byte, (n+7)/8),
			Length:     n,
		}
		for i, idx := range sel {
			out.Set(i, src.Get(int(idx)))
			if !src.IsNull(int(idx)) {
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case *StringVector:
		out := &StringVector{
			Codes:      make([]uint32, n),
			Dict:       src.Dict,
			NullBitmap: make([]byte, (n+7)/8),
		}
		for i, idx := range sel {
			out.Codes[i] = src.Codes[idx]
			if !src.IsNull(int(idx)) {
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case *DateVector:
		out := &DateVector{
			Values:     make([]int32, n),
			NullBitmap: make([]byte, (n+7)/8),
		}
		for i, idx := range sel {
			out.Values[i] = src.Values[idx]
			if !src.IsNull(int(idx)) {
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	default:
		return v
	}
}
