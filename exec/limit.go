package exec

import (
	"context"
	"fmt"
)

// Limit skips the first Offset rows from its child and then passes at most N
// rows. A negative N means no row limit, which is how OFFSET without LIMIT is
// expressed.
type Limit struct {
	child     Operator
	remaining int // rows still to emit; negative = unlimited
	skip      int // rows still to discard
}

// NewLimit returns an operator that passes at most n rows.
func NewLimit(child Operator, n int) *Limit {
	return &Limit{child: child, remaining: n}
}

// NewLimitOffset returns an operator that discards the first offset rows and
// then passes at most n of the rest; n < 0 passes all of them.
func NewLimitOffset(child Operator, n, offset int) *Limit {
	return &Limit{child: child, remaining: n, skip: offset}
}

func (l *Limit) Schema() Schema { return l.child.Schema() }

func (l *Limit) Next(ctx context.Context) (*Batch, error) {
	for {
		if l.remaining == 0 {
			return nil, nil
		}
		batch, err := l.child.Next(ctx)
		if err != nil {
			return nil, fmt.Errorf("exec: limit: %w", err)
		}
		if batch == nil {
			return nil, nil
		}
		if l.skip > 0 {
			if batch.Length <= l.skip {
				l.skip -= batch.Length
				continue
			}
			l.keepRows(batch, l.skip, batch.Length-l.skip)
			l.skip = 0
		}
		if l.remaining < 0 || batch.Length <= l.remaining {
			if l.remaining > 0 {
				l.remaining -= batch.Length
			}
			return batch, nil
		}
		l.keepRows(batch, 0, l.remaining)
		l.remaining = 0
		return batch, nil
	}
}

// keepRows narrows batch to n of its logical rows starting at logical row from.
func (l *Limit) keepRows(batch *Batch, from, n int) {
	if batch.SelVec != nil {
		// Preserve the upstream filter's selection vector — just slice it.
		batch.SelVec = batch.SelVec[from : from+n]
	} else {
		// No upstream filter; create sequential indices.
		sel := make(SelectionVector, n)
		for i := range sel {
			sel[i] = uint16(from + i)
		}
		batch.SelVec = sel
	}
	batch.Length = n
}

func (l *Limit) Close() error { return l.child.Close() }
