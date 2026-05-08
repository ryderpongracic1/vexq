package exec

import (
	"context"
	"fmt"
)

// Filter applies a predicate to each Batch from its child, producing a
// SelectionVector of surviving row indices.  It does NOT materialise a new
// batch — downstream operators must honor Batch.SelVec.
type Filter struct {
	child     Operator
	predicate Expr
}

func NewFilter(child Operator, predicate Expr) (*Filter, error) {
	if predicate.Type() != TypeBool {
		return nil, fmt.Errorf("exec: filter predicate must return BOOL, got %v", predicate.Type())
	}
	return &Filter{child: child, predicate: predicate}, nil
}

func (f *Filter) Schema() Schema { return f.child.Schema() }

func (f *Filter) Next(ctx context.Context) (*Batch, error) {
	for {
		batch, err := f.child.Next(ctx)
		if err != nil {
			return nil, fmt.Errorf("exec: filter: %w", err)
		}
		if batch == nil {
			return nil, nil
		}

		bv, err := f.predicate.Eval(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("exec: filter eval: %w", err)
		}
		boolVec, ok := bv.(*BoolVector)
		if !ok {
			return nil, fmt.Errorf("exec: filter: predicate returned %T, expected *BoolVector", bv)
		}

		sel := BoolToSelVec(batch, boolVec)
		if len(sel) == 0 {
			// No rows survive; try the next batch.
			continue
		}
		batch.SelVec = sel
		batch.Length = len(sel)
		return batch, nil
	}
}

func (f *Filter) Close() error { return f.child.Close() }
