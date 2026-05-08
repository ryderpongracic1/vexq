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
type Project struct {
	child   Operator
	exprs   []ProjectExpr
	schema  Schema
}

func NewProject(child Operator, exprs []ProjectExpr) (*Project, error) {
	if len(exprs) == 0 {
		return nil, fmt.Errorf("exec: project: no expressions")
	}
	fields := make([]Field, len(exprs))
	for i, pe := range exprs {
		fields[i] = Field{Name: pe.Name, Type: pe.Expr.Type(), Nullable: true}
	}
	return &Project{
		child:  child,
		exprs:  exprs,
		schema: Schema{Fields: fields},
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

	outVecs := make([]Vector, len(p.exprs))
	for i, pe := range p.exprs {
		raw, err := pe.Expr.Eval(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("exec: project: eval %q: %w", pe.Name, err)
		}
		// If there's a selection vector, materialise only selected rows.
		if batch.SelVec != nil {
			raw = materialize(raw, batch.SelVec)
		}
		outVecs[i] = raw
	}

	outLen := batch.Length
	return &Batch{
		Schema:  p.schema,
		Vectors: outVecs,
		Length:  outLen,
	}, nil
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
