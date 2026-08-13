package exec

import (
	"context"
	"encoding/binary"
)

// Distinct deduplicates rows using a hash set of serialized row keys.
// It uses the same key serialization approach as HashAggregate.buildKey.
type Distinct struct {
	child  Operator
	seen   map[string]struct{}
	schema Schema
}

// NewDistinct wraps the given operator to emit only unique rows.
func NewDistinct(child Operator) *Distinct {
	return &Distinct{
		child:  child,
		seen:   make(map[string]struct{}),
		schema: child.Schema(),
	}
}

func (d *Distinct) Schema() Schema { return d.schema }

func (d *Distinct) Next(ctx context.Context) (*Batch, error) {
	for {
		batch, err := d.child.Next(ctx)
		if err != nil {
			return nil, err
		}
		if batch == nil {
			return nil, nil
		}

		// Build selection vector of unique rows.
		sel := make([]uint16, 0, batch.Length)

		if batch.SelVec != nil {
			for _, ri := range batch.SelVec {
				key := d.buildKey(batch, int(ri))
				if _, exists := d.seen[key]; !exists {
					d.seen[key] = struct{}{}
					sel = append(sel, ri)
				}
			}
		} else {
			for i := 0; i < batch.Length; i++ {
				key := d.buildKey(batch, i)
				if _, exists := d.seen[key]; !exists {
					d.seen[key] = struct{}{}
					sel = append(sel, uint16(i))
				}
			}
		}

		if len(sel) == 0 {
			// All rows in this batch are duplicates; get next batch.
			continue
		}

		batch.SelVec = sel
		batch.Length = len(sel)
		return batch, nil
	}
}

func (d *Distinct) Close() error {
	return d.child.Close()
}

// buildKey serializes all columns for a given row into a comparable string.
// Format matches HashAggregate.buildKey:
//
//	null:   [0x00, 0xFF]
//	string: [0x02, <4-byte-LE length>, <utf8 bytes>, 0xFF]
//	other:  [0x01, <8-byte-LE uint64>, 0xFF]
func (d *Distinct) buildKey(batch *Batch, rowIdx int) string {
	buf := make([]byte, 0, len(batch.Vectors)*10)
	for _, v := range batch.Vectors {
		if v.IsNull(rowIdx) {
			buf = append(buf, 0x00, 0xFF)
		} else if sv, ok := v.(*StringVector); ok {
			var s string
			if sv.Dict != nil {
				s = sv.Dict.Get(sv.Codes[rowIdx])
			}
			buf = append(buf, 0x02)
			buf = binary.LittleEndian.AppendUint32(buf, uint32(len(s)))
			buf = append(buf, s...)
			buf = append(buf, 0xFF)
		} else {
			buf = append(buf, 0x01)
			buf = binary.LittleEndian.AppendUint64(buf, uint64(extractInt64(v, rowIdx)))
			buf = append(buf, 0xFF)
		}
	}
	return string(buf)
}
