package exec

import (
	"context"
	"fmt"
)

// Limit passes at most N rows from its child.
type Limit struct {
	child    Operator
	remaining int
}

func NewLimit(child Operator, n int) *Limit {
	return &Limit{child: child, remaining: n}
}

func (l *Limit) Schema() Schema { return l.child.Schema() }

func (l *Limit) Next(ctx context.Context) (*Batch, error) {
	if l.remaining <= 0 {
		return nil, nil
	}
	batch, err := l.child.Next(ctx)
	if err != nil {
		return nil, fmt.Errorf("exec: limit: %w", err)
	}
	if batch == nil {
		return nil, nil
	}
	if batch.Length <= l.remaining {
		l.remaining -= batch.Length
		return batch, nil
	}
	// Truncate the batch.
	sel := make(SelectionVector, l.remaining)
	for i := range sel {
		sel[i] = uint16(i)
	}
	batch.SelVec = sel
	batch.Length = l.remaining
	l.remaining = 0
	return batch, nil
}

func (l *Limit) Close() error { return l.child.Close() }
