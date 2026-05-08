package exec

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/ryderpongracic1/vexq/storage"
)

// SortKey describes one sort column.
type SortKey struct {
	ColIdx    int
	Descending bool
}

// ExternalSort collects all input, sorts it in memory (v1 — no spill), and
// emits sorted Batches.  The MaxMemRows limit is a no-op in v1; it exists as a
// knob for future spill-to-disk.
type ExternalSort struct {
	child       Operator
	keys        []SortKey
	schema      Schema
	MaxMemRows  int // 0 = unlimited

	// accumulated rows (materialised after first Next call)
	rows    []sortRow
	done    bool
	emitPos int
}

type sortRow struct {
	values    []int64  // raw int64 bits per column (unused for TypeString)
	nulls     []bool
	strValues []string // actual string for TypeString columns; "" otherwise
}

func NewExternalSort(child Operator, keys []SortKey) (*ExternalSort, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("exec: sort: no sort keys")
	}
	return &ExternalSort{
		child:  child,
		keys:   keys,
		schema: child.Schema(),
	}, nil
}

func (s *ExternalSort) Schema() Schema { return s.schema }

func (s *ExternalSort) Next(ctx context.Context) (*Batch, error) {
	if !s.done {
		if err := s.consumeAndSort(ctx); err != nil {
			return nil, err
		}
		s.done = true
	}

	if s.emitPos >= len(s.rows) {
		return nil, nil
	}

	end := s.emitPos + BlockRows
	if end > len(s.rows) {
		end = len(s.rows)
	}
	batch := s.emitBatch(s.rows[s.emitPos:end])
	s.emitPos = end
	return batch, nil
}

func (s *ExternalSort) consumeAndSort(ctx context.Context) error {
	schema := s.child.Schema()
	numCols := len(schema.Fields)

	for {
		batch, err := s.child.Next(ctx)
		if err != nil {
			return fmt.Errorf("exec: sort: %w", err)
		}
		if batch == nil {
			break
		}

		n := batch.Length
		var indices []int
		if batch.SelVec != nil {
			indices = make([]int, len(batch.SelVec))
			for i, v := range batch.SelVec {
				indices[i] = int(v)
			}
		} else {
			indices = make([]int, n)
			for i := range indices {
				indices[i] = i
			}
		}

		for _, rowIdx := range indices {
			row := sortRow{
				values:    make([]int64, numCols),
				nulls:     make([]bool, numCols),
				strValues: make([]string, numCols),
			}
			for c := 0; c < numCols; c++ {
				v := batch.Vectors[c]
				row.nulls[c] = v.IsNull(rowIdx)
				if !row.nulls[c] {
					if sv, ok := v.(*StringVector); ok {
						if sv.Dict != nil {
							row.strValues[c] = sv.Dict.Get(sv.Codes[rowIdx])
						}
					} else {
						row.values[c] = extractInt64(v, rowIdx)
					}
				}
			}
			s.rows = append(s.rows, row)
		}
	}

	// Sort using sort.SliceStable for stability.
	sort.SliceStable(s.rows, func(i, j int) bool {
		return s.less(s.rows[i], s.rows[j])
	})
	return nil
}

func (s *ExternalSort) less(a, b sortRow) bool {
	for _, key := range s.keys {
		ci := key.ColIdx
		aN := a.nulls[ci]
		bN := b.nulls[ci]
		if aN && bN {
			continue
		}
		if aN {
			return true // nulls first
		}
		if bN {
			return false
		}
		if s.schema.Fields[ci].Type == TypeString {
			as, bs := a.strValues[ci], b.strValues[ci]
			if as == bs {
				continue
			}
			if key.Descending {
				return as > bs
			}
			return as < bs
		}
		if a.values[ci] == b.values[ci] {
			continue
		}
		if key.Descending {
			return a.values[ci] > b.values[ci]
		}
		return a.values[ci] < b.values[ci]
	}
	return false
}

func (s *ExternalSort) emitBatch(rows []sortRow) *Batch {
	schema := s.child.Schema()
	n := len(rows)
	vecs := make([]Vector, len(schema.Fields))
	for c, field := range schema.Fields {
		switch field.Type {
		case TypeInt64:
			out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
			for i, r := range rows {
				if !r.nulls[c] {
					out.Values[i] = r.values[c]
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
			vecs[c] = out
		case TypeFloat64:
			out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
			for i, r := range rows {
				if !r.nulls[c] {
					out.Values[i] = float64FromBits(uint64(r.values[c]))
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
			vecs[c] = out
		case TypeDate:
			out := &DateVector{Values: make([]int32, n), NullBitmap: make([]byte, (n+7)/8)}
			for i, r := range rows {
				if !r.nulls[c] {
					out.Values[i] = int32(r.values[c])
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
			vecs[c] = out
		case TypeString:
			db := storage.NewDictBuilder()
			codes := make([]uint32, n)
			nullBmp := make([]byte, (n+7)/8)
			for i, r := range rows {
				if !r.nulls[c] {
					codes[i] = db.Add(r.strValues[c])
					storage.SetValidBit(nullBmp, i)
				}
			}
			rawDict := db.Marshal()
			dict, _ := storage.UnmarshalDictionary(rawDict)
			vecs[c] = &StringVector{Codes: codes, Dict: dict, NullBitmap: nullBmp}

		default:
			out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
			for i, r := range rows {
				if !r.nulls[c] {
					out.Values[i] = r.values[c]
					storage.SetValidBit(out.NullBitmap, i)
				}
			}
			vecs[c] = out
		}
	}
	return &Batch{Schema: s.schema, Vectors: vecs, Length: n}
}

func float64FromBits(b uint64) float64 { return math.Float64frombits(b) }

func (s *ExternalSort) Close() error { return s.child.Close() }
