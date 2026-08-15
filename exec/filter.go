package exec

import (
	"context"
	"fmt"
)

// Filter applies a predicate to each Batch from its child, producing a
// SelectionVector of surviving row indices.  It does NOT materialise a new
// batch — downstream operators must honor Batch.SelVec.
//
// Buffer ownership contract: the selection vector is a per-Filter buffer reused
// across Next() calls, so it is valid only until the next Next() on this Filter.
// That is exactly the lifetime of the batch it is attached to: Filter returns
// its child's batch unchanged, and TableScan already overwrites that batch's
// vectors on the next call (see scan.go). Every consumer therefore already had
// to read or copy the batch before pulling again — HashAggregate accumulates
// into its maps, sort copies into sortRow, the join build copies into buildRow,
// Distinct replaces the vector with its own — so attaching a reused index buffer
// to that same batch weakens nothing.
//
// Stacked filters are safe because each Filter owns its own buffer: the upper
// Filter reads the lower one's selection vector and writes its own, never both
// at once.
//
// A stacked Filter also hands its child's batch to Eval untouched — selection
// vector installed, Length already down to the selected count. That is the batch
// the expression sizing convention is written for (see the Expr interface in
// expr.go): the predicate is evaluated over every physical row, and
// BoolToSelVecInto then reads the result only at the physical indices the
// incoming selection vector names.
type Filter struct {
	child     Operator
	predicate Expr

	// sel is the reused selection-vector buffer; see the contract above.
	sel SelectionVector
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

		sel := BoolToSelVecInto(batch, boolVec, f.sel)
		f.sel = sel
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
