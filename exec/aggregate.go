package exec

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ryderpongracic1/vexq/storage"
)

// AggKind is the aggregate function kind.
type AggKind uint8

const (
	AggCount AggKind = iota // COUNT(*) or COUNT(col)
	AggSum
	AggAvg
	AggMin
	AggMax
)

// AggExpr describes one aggregate function in the output.
type AggExpr struct {
	Kind      AggKind
	ColIdx    int      // source column index (-1 for COUNT(*))
	OutName   string   // output column name
	AccumType DataType // type of bits stored in the groups accumulator:
	// TypeInt64   for COUNT, SUM/MIN/MAX over integer/date columns
	// TypeFloat64 for SUM/MIN/MAX over float64 columns, and always for AVG
	// Set by the planner; used by mergePartialAgg in parallel execution.
}

// groupByVal stores one group-by column value for a representative row.
type groupByVal struct {
	isNull bool
	strVal string // populated for TypeString
	bits   uint64 // raw bits for all other types
}

// HashAggregate groups input rows by key columns and computes aggregates.
// It accumulates all input before emitting any output (unbounded memory in v1).
type HashAggregate struct {
	child    Operator
	groupBy  []int // column indices in the child schema
	aggExprs []AggExpr
	schema   Schema

	// internal state
	keys       []string                // serialised group-by keys in insertion order
	groups     map[string][]int64      // key → per-aggregate accumulators
	groupCnt   map[string]int64        // key → count of rows in group (legacy)
	aggNonNull map[string][]int64      // key → per-aggregate non-null input count
	samples    map[string][]groupByVal // key → representative group-by values
	done       bool
	outPos     int

	// Integer-key fast path: eliminates string allocation in the hot loop
	// when all GROUP BY columns are dict-encoded strings.
	intKey        intKeyState
	intKeyDecided bool // true once we've checked the first batch
}

func NewHashAggregate(child Operator, groupBy []int, aggExprs []AggExpr) (*HashAggregate, error) {
	if len(aggExprs) == 0 && len(groupBy) == 0 {
		return nil, fmt.Errorf("exec: hash aggregate: no group-by columns or aggregate expressions")
	}
	childSchema := child.Schema()
	var outFields []Field

	for _, idx := range groupBy {
		if idx < 0 || idx >= len(childSchema.Fields) {
			return nil, fmt.Errorf("exec: hash aggregate: group-by column %d out of range", idx)
		}
		outFields = append(outFields, childSchema.Fields[idx])
	}

	// Copy aggExprs so we can fill in AccumType without mutating the caller's slice.
	resolved := make([]AggExpr, len(aggExprs))
	copy(resolved, aggExprs)
	for i := range resolved {
		ae := &resolved[i]
		var t DataType
		switch ae.Kind {
		case AggCount:
			t = TypeInt64
			if ae.AccumType == 0 {
				ae.AccumType = TypeInt64
			}
		case AggSum, AggMin, AggMax:
			if ae.ColIdx < 0 {
				t = TypeInt64
			} else {
				t = childSchema.Fields[ae.ColIdx].Type
			}
			if ae.AccumType == 0 {
				if ae.ColIdx >= 0 && childSchema.Fields[ae.ColIdx].Type == TypeFloat64 {
					ae.AccumType = TypeFloat64
				} else {
					ae.AccumType = TypeInt64
				}
			}
		case AggAvg:
			t = TypeFloat64
			if ae.AccumType == 0 {
				ae.AccumType = TypeFloat64
			}
		}
		outFields = append(outFields, Field{Name: ae.OutName, Type: t, Nullable: true})
	}

	return &HashAggregate{
		child:      child,
		groupBy:    groupBy,
		aggExprs:   resolved,
		schema:     Schema{Fields: outFields},
		groups:     make(map[string][]int64),
		groupCnt:   make(map[string]int64),
		aggNonNull: make(map[string][]int64),
		samples:    make(map[string][]groupByVal),
	}, nil
}

func (h *HashAggregate) Schema() Schema { return h.schema }

func (h *HashAggregate) Next(ctx context.Context) (*Batch, error) {
	if !h.done {
		if err := h.consumeAll(ctx); err != nil {
			return nil, err
		}
		h.done = true

		// Empty-input global aggregate: no GROUP BY, no rows consumed.
		// SQL requires exactly one output row: COUNT=0, NULL for SUM/AVG/MIN/MAX.
		if len(h.keys) == 0 && len(h.groupBy) == 0 {
			return h.buildEmptyGlobalResult(), nil
		}
	}

	if h.outPos >= len(h.keys) {
		return nil, nil // EOF
	}

	// Emit up to BlockRows output rows per call.
	end := h.outPos + BlockRows
	if end > len(h.keys) {
		end = len(h.keys)
	}
	batch := h.buildOutputBatch(h.keys[h.outPos:end])
	h.outPos = end
	return batch, nil
}

// initMaps resets the internal accumulator maps. Called at the start of
// consumeAll and by parallel workers before their first accumulate call.
func (h *HashAggregate) initMaps() {
	h.keys = nil
	h.groups = make(map[string][]int64)
	h.groupCnt = make(map[string]int64)
	h.aggNonNull = make(map[string][]int64)
	h.samples = make(map[string][]groupByVal)
	h.intKeyDecided = false
	h.intKey = intKeyState{}
}

func (h *HashAggregate) consumeAll(ctx context.Context) error {
	h.initMaps()
	for {
		batch, err := h.child.Next(ctx)
		if err != nil {
			return fmt.Errorf("exec: hash agg: %w", err)
		}
		if batch == nil {
			break
		}
		if err := h.accumulate(batch); err != nil {
			return fmt.Errorf("exec: hash agg: %w", err)
		}
	}
	// Materialize integer-key results into string-key maps for output.
	if h.intKey.enabled && len(h.intKey.intKeys) > 0 {
		h.intKey.materializeToStringMaps(h)
	}
	return nil
}

// accumulate processes one batch into the hash aggregate maps.
// Uses AggExpr.AccumType to determine numeric encoding; no child schema needed.
func (h *HashAggregate) accumulate(batch *Batch) error {
	n := batch.Length

	// Resolve each aggregate's source vector once per batch, not once per row.
	// batch.Vectors[ae.ColIdx] is a slice index + interface load — cheap, but
	// multiplied by len(aggExprs) × len(indices) it becomes a non-trivial cost.
	aggVecs := make([]Vector, len(h.aggExprs))
	for j, ae := range h.aggExprs {
		if ae.ColIdx >= 0 {
			aggVecs[j] = batch.Vectors[ae.ColIdx]
		}
	}

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

	// Fast path: no GROUP BY eliminates per-row map lookup (e.g. Q6 SUM with no groups).
	if len(h.groupBy) == 0 {
		return h.accumulateDirect(indices, aggVecs)
	}

	// Integer-key fast path: when all GROUP BY columns are dict-encoded strings,
	// key on packed global dictionary codes instead of building composite string keys.
	if !h.intKeyDecided {
		h.intKeyDecided = true
		if canUseIntKey(batch, h.groupBy) {
			h.intKey.enabled = true
			h.intKey.init(len(h.groupBy))
		}
	}

	if h.intKey.enabled {
		return h.accumulateIntKey(batch, indices, aggVecs)
	}

	for _, rowIdx := range indices {
		key := h.buildKey(batch, rowIdx)
		accs, exists := h.groups[key]
		if !exists {
			accs = make([]int64, len(h.aggExprs))
			for j, ae := range h.aggExprs {
				switch ae.Kind {
				case AggMin:
					if ae.AccumType == TypeFloat64 {
						accs[j] = int64(math.Float64bits(math.MaxFloat64))
					} else {
						accs[j] = math.MaxInt64
					}
				case AggMax:
					if ae.AccumType == TypeFloat64 {
						accs[j] = int64(math.Float64bits(-math.MaxFloat64))
					} else {
						accs[j] = math.MinInt64
					}
				}
			}
			h.groups[key] = accs
			h.aggNonNull[key] = make([]int64, len(h.aggExprs))
			h.keys = append(h.keys, key)
			sample := make([]groupByVal, len(h.groupBy))
			for si, colIdx := range h.groupBy {
				v := batch.Vectors[colIdx]
				if v.IsNull(rowIdx) {
					sample[si] = groupByVal{isNull: true}
				} else if sv, ok := v.(*StringVector); ok {
					var s string
					if sv.Dict != nil {
						s = sv.Dict.Get(sv.Codes[rowIdx])
					}
					sample[si] = groupByVal{strVal: s}
				} else {
					sample[si] = groupByVal{bits: uint64(extractInt64(v, rowIdx))}
				}
			}
			h.samples[key] = sample
		}

		nonNull := h.aggNonNull[key]
		for j, ae := range h.aggExprs {
			v := aggVecs[j] // hoisted: resolved once per batch above
			switch ae.Kind {
			case AggCount:
				if ae.ColIdx < 0 {
					accs[j]++
					nonNull[j]++
				} else if !v.IsNull(rowIdx) {
					accs[j]++
					nonNull[j]++
				}
			case AggSum:
				if v.IsNull(rowIdx) {
					continue
				}
				nonNull[j]++
				if ae.AccumType == TypeFloat64 {
					fv := v.(*Float64Vector)
					cur := math.Float64frombits(uint64(accs[j]))
					accs[j] = int64(math.Float64bits(cur + fv.Values[rowIdx]))
				} else {
					accs[j] += extractInt64(v, rowIdx)
				}
			case AggAvg:
				if v.IsNull(rowIdx) {
					continue
				}
				nonNull[j]++
				if fv, ok := v.(*Float64Vector); ok {
					cur := math.Float64frombits(uint64(accs[j]))
					accs[j] = int64(math.Float64bits(cur + fv.Values[rowIdx]))
				} else {
					cur := math.Float64frombits(uint64(accs[j]))
					accs[j] = int64(math.Float64bits(cur + float64(extractInt64(v, rowIdx))))
				}
			case AggMin:
				if v.IsNull(rowIdx) {
					continue
				}
				nonNull[j]++
				val := extractInt64(v, rowIdx)
				if ae.AccumType == TypeFloat64 {
					if math.Float64frombits(uint64(val)) < math.Float64frombits(uint64(accs[j])) {
						accs[j] = val
					}
				} else if val < accs[j] {
					accs[j] = val
				}
			case AggMax:
				if v.IsNull(rowIdx) {
					continue
				}
				nonNull[j]++
				val := extractInt64(v, rowIdx)
				if ae.AccumType == TypeFloat64 {
					if math.Float64frombits(uint64(val)) > math.Float64frombits(uint64(accs[j])) {
						accs[j] = val
					}
				} else if val > accs[j] {
					accs[j] = val
				}
			}
		}
	}
	return nil
}

// accumulateDirect handles the no-GROUP-BY case (single implicit group "").
// Skipping per-row map lookup and key serialisation is the dominant win for
// queries like Q6 (SUM with no GROUP BY) — the inner loop reduces to a tight
// float64 addition or integer increment with no hash overhead.
func (h *HashAggregate) accumulateDirect(indices []int, aggVecs []Vector) error {
	const key = ""
	accs, exists := h.groups[key]
	if !exists {
		accs = make([]int64, len(h.aggExprs))
		for j, ae := range h.aggExprs {
			switch ae.Kind {
			case AggMin:
				if ae.AccumType == TypeFloat64 {
					accs[j] = int64(math.Float64bits(math.MaxFloat64))
				} else {
					accs[j] = math.MaxInt64
				}
			case AggMax:
				if ae.AccumType == TypeFloat64 {
					accs[j] = int64(math.Float64bits(-math.MaxFloat64))
				} else {
					accs[j] = math.MinInt64
				}
			}
		}
		h.groups[key] = accs
		h.aggNonNull[key] = make([]int64, len(h.aggExprs))
		h.keys = append(h.keys, key)
	}

	nonNull := h.aggNonNull[key]

	for j, ae := range h.aggExprs {
		v := aggVecs[j]
		switch ae.Kind {
		case AggCount:
			if ae.ColIdx < 0 {
				accs[j] += int64(len(indices)) // COUNT(*): no per-row null check needed
				nonNull[j] += int64(len(indices))
			} else {
				for _, rowIdx := range indices {
					if !v.IsNull(rowIdx) {
						accs[j]++
						nonNull[j]++
					}
				}
			}
		case AggSum:
			if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for _, rowIdx := range indices {
					if !fv.IsNull(rowIdx) {
						cur += fv.Values[rowIdx]
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for _, rowIdx := range indices {
					if !v.IsNull(rowIdx) {
						accs[j] += extractInt64(v, rowIdx)
						nonNull[j]++
					}
				}
			}
		case AggAvg:
			if fv, ok := v.(*Float64Vector); ok {
				cur := math.Float64frombits(uint64(accs[j]))
				for _, rowIdx := range indices {
					if !fv.IsNull(rowIdx) {
						cur += fv.Values[rowIdx]
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				cur := math.Float64frombits(uint64(accs[j]))
				for _, rowIdx := range indices {
					if !v.IsNull(rowIdx) {
						cur += float64(extractInt64(v, rowIdx))
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			}
		case AggMin:
			if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for _, rowIdx := range indices {
					if !fv.IsNull(rowIdx) {
						if fv.Values[rowIdx] < cur {
							cur = fv.Values[rowIdx]
						}
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for _, rowIdx := range indices {
					if !v.IsNull(rowIdx) {
						if val := extractInt64(v, rowIdx); val < accs[j] {
							accs[j] = val
						}
						nonNull[j]++
					}
				}
			}
		case AggMax:
			if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for _, rowIdx := range indices {
					if !fv.IsNull(rowIdx) {
						if fv.Values[rowIdx] > cur {
							cur = fv.Values[rowIdx]
						}
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for _, rowIdx := range indices {
					if !v.IsNull(rowIdx) {
						if val := extractInt64(v, rowIdx); val > accs[j] {
							accs[j] = val
						}
						nonNull[j]++
					}
				}
			}
		}
	}
	return nil
}

// buildKey serialises the group-by column values for a row into a string key.
// Format per column:
//
//	null:       [0x00, 0xFF]
//	string:     [0x02, <4-byte-LE length>, <utf8 bytes>, 0xFF]
//	other:      [0x01, <8-byte-LE uint64>, 0xFF]
func (h *HashAggregate) buildKey(batch *Batch, rowIdx int) string {
	if len(h.groupBy) == 0 {
		return ""
	}
	buf := make([]byte, 0, len(h.groupBy)*10)
	for _, colIdx := range h.groupBy {
		v := batch.Vectors[colIdx]
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

func (h *HashAggregate) buildOutputBatch(keys []string) *Batch {
	n := len(keys)
	vecs := make([]Vector, len(h.schema.Fields))
	outIdx := 0

	// Group-by columns: source type == output type (schema copied from child by NewHashAggregate).
	for gbPos := range h.groupBy {
		srcType := h.schema.Fields[gbPos].Type
		vecs[outIdx] = buildGroupByVector(h, keys, gbPos, srcType, n)
		outIdx++
	}

	// Aggregate columns.
	for j, ae := range h.aggExprs {
		switch ae.Kind {
		case AggCount:
			// COUNT never returns NULL.
			if ae.AccumType == TypeFloat64 {
				fOut := &Float64Vector{Values: make([]float64, n), NullBitmap: storage.FullBitmap(n)}
				for i, key := range keys {
					fOut.Values[i] = math.Float64frombits(uint64(h.groups[key][j]))
				}
				vecs[outIdx] = fOut
			} else {
				out := &Int64Vector{Values: make([]int64, n), NullBitmap: storage.FullBitmap(n)}
				for i, key := range keys {
					out.Values[i] = h.groups[key][j]
				}
				vecs[outIdx] = out
			}
		case AggSum, AggMin, AggMax:
			// SUM/MIN/MAX return NULL when all inputs were null.
			if ae.AccumType == TypeFloat64 {
				fOut := &Float64Vector{Values: make([]float64, n), NullBitmap: storage.FullBitmap(n)}
				for i, key := range keys {
					if h.aggNonNull[key][j] == 0 {
						fOut.Values[i] = 0
						storage.SetNullBit(fOut.NullBitmap, i)
					} else {
						fOut.Values[i] = math.Float64frombits(uint64(h.groups[key][j]))
					}
				}
				vecs[outIdx] = fOut
			} else {
				out := &Int64Vector{Values: make([]int64, n), NullBitmap: storage.FullBitmap(n)}
				for i, key := range keys {
					if h.aggNonNull[key][j] == 0 {
						out.Values[i] = 0
						storage.SetNullBit(out.NullBitmap, i)
					} else {
						out.Values[i] = h.groups[key][j]
					}
				}
				vecs[outIdx] = out
			}
		case AggAvg:
			// AVG divides by non-null count; returns NULL when no non-null inputs.
			fOut := &Float64Vector{Values: make([]float64, n), NullBitmap: storage.FullBitmap(n)}
			for i, key := range keys {
				cnt := h.aggNonNull[key][j]
				if cnt > 0 {
					fOut.Values[i] = math.Float64frombits(uint64(h.groups[key][j])) / float64(cnt)
				} else {
					fOut.Values[i] = 0
					storage.SetNullBit(fOut.NullBitmap, i)
				}
			}
			vecs[outIdx] = fOut
		}
		outIdx++
	}

	return &Batch{Schema: h.schema, Vectors: vecs, Length: n}
}

// buildEmptyGlobalResult creates a single-row batch for a global aggregate
// (no GROUP BY) that consumed zero input rows. SQL requires:
//   - COUNT -> 0 (never NULL)
//   - SUM/AVG/MIN/MAX -> NULL
func (h *HashAggregate) buildEmptyGlobalResult() *Batch {
	vecs := make([]Vector, len(h.aggExprs))
	for j, ae := range h.aggExprs {
		switch ae.Kind {
		case AggCount:
			vecs[j] = &Int64Vector{Values: []int64{0}, NullBitmap: storage.FullBitmap(1)}
		case AggSum, AggMin, AggMax:
			if ae.AccumType == TypeFloat64 {
				vecs[j] = &Float64Vector{Values: []float64{0}, NullBitmap: make([]byte, 1)}
			} else {
				vecs[j] = &Int64Vector{Values: []int64{0}, NullBitmap: make([]byte, 1)}
			}
		case AggAvg:
			vecs[j] = &Float64Vector{Values: []float64{0}, NullBitmap: make([]byte, 1)}
		}
	}
	h.outPos = 1 // mark as emitted so subsequent Next returns nil
	return &Batch{Schema: h.schema, Vectors: vecs, Length: 1}
}

// buildGroupByVector reconstructs the group-by column values from stored samples.
func buildGroupByVector(h *HashAggregate, keys []string, gbPos int, srcType DataType, n int) Vector {
	switch srcType {
	case TypeString:
		// Build a flat per-output dictionary from the distinct string values.
		db := storage.NewDictBuilder()
		codes := make([]uint32, n)
		nullBmp := make([]byte, (n+7)/8)
		for i, key := range keys {
			sample := h.samples[key]
			if sample == nil || sample[gbPos].isNull {
				// leave null
				continue
			}
			codes[i] = db.Add(sample[gbPos].strVal)
			storage.SetValidBit(nullBmp, i)
		}
		return newStringVector(db, codes, nullBmp)

	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, key := range keys {
			sample := h.samples[key]
			if sample == nil || sample[gbPos].isNull {
				continue
			}
			out.Values[i] = math.Float64frombits(sample[gbPos].bits)
			storage.SetValidBit(out.NullBitmap, i)
		}
		return out

	case TypeDate:
		out := &DateVector{Values: make([]int32, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, key := range keys {
			sample := h.samples[key]
			if sample == nil || sample[gbPos].isNull {
				continue
			}
			out.Values[i] = int32(sample[gbPos].bits)
			storage.SetValidBit(out.NullBitmap, i)
		}
		return out

	default: // TypeInt64, TypeBool
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, key := range keys {
			sample := h.samples[key]
			if sample == nil || sample[gbPos].isNull {
				continue
			}
			out.Values[i] = int64(sample[gbPos].bits)
			storage.SetValidBit(out.NullBitmap, i)
		}
		return out
	}
}

// extractInt64 returns the raw int64 bits of a value at row index i.
// For float64 columns, returns the IEEE bits.
func extractInt64(v Vector, i int) int64 {
	switch col := v.(type) {
	case *Int64Vector:
		return col.Values[i]
	case *Float64Vector:
		return int64(math.Float64bits(col.Values[i]))
	case *DateVector:
		return int64(col.Values[i])
	case *BoolVector:
		if col.Get(i) {
			return 1
		}
		return 0
	default:
		return 0
	}
}

func (h *HashAggregate) Close() error {
	if h.child == nil {
		return nil
	}
	return h.child.Close()
}
