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
	AggCountDistinct // COUNT(DISTINCT col)
)

// AggExpr describes one aggregate function in the output.
type AggExpr struct {
	Kind      AggKind
	ColIdx    int      // source column index (-1 for COUNT(*))
	OutName   string   // output column name
	Distinct  bool     // true for COUNT(DISTINCT col)
	AccumType DataType // encoding of the accumulator this aggregate runs in:
	// TypeInt64   for COUNT, SUM/MIN/MAX over integer/date/bool columns
	// TypeFloat64 for SUM/MIN/MAX over float64 columns, and always for AVG
	// TypeString  for MIN/MAX over string columns — the running value lives in
	//             HashAggregate.strAccs, not in the int64 groups accumulator
	// Set by the planner (or by NewHashAggregate when left zero) via
	// AccumTypeFor; used by mergePartialAgg in parallel execution.
}

// AccumTypeFor returns the accumulator encoding an aggregate of kind must use
// over a source column of type srcType. Pass TypeInt64 as srcType for kinds that
// read no column (COUNT(*)).
//
// This is the single definition of that mapping, and it is shared rather than
// duplicated on purpose: NewHashAggregate applies it when the caller left
// AccumType zero, and the planner applies it when resolving AggExprs, because
// the parallel path builds its per-worker partial aggregates directly
// (newPartialAggregate) rather than through NewHashAggregate. If the two ever
// disagreed, a partial accumulated under one encoding would be merged and
// decoded under another — the failure mode is a silently wrong answer, not a
// crash, which is exactly how MIN/MAX over string columns came to return 0.
func AccumTypeFor(kind AggKind, srcType DataType) DataType {
	switch kind {
	case AggAvg:
		return TypeFloat64
	case AggSum:
		if srcType == TypeFloat64 {
			return TypeFloat64
		}
		return TypeInt64
	case AggMin, AggMax:
		switch srcType {
		case TypeFloat64:
			return TypeFloat64
		case TypeString:
			return TypeString
		default:
			return TypeInt64
		}
	default: // AggCount, AggCountDistinct
		return TypeInt64
	}
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

	// strAccs holds the running values for MIN/MAX over STRING columns, which
	// have no int64 encoding to live in: strAccs[key][j] is aggregate j's
	// running min/max for group key. Only slots whose AggExpr.AccumType is
	// TypeString are ever written; the rest stay "".
	//
	// There is no "empty" sentinel, because "" is a legitimate string value.
	// aggNonNull[key][j] == 0 is the emptiness test instead: the first non-null
	// row of a group takes its value unconditionally, and a group that saw no
	// non-null row outputs NULL, so the zero value is never read as data.
	//
	// Allocated only when hasStrAgg — a numeric-only aggregate pays nothing.
	strAccs   map[string][]string
	hasStrAgg bool

	// Integer-key fast path: eliminates string allocation in the hot loop
	// when all GROUP BY columns are dict-encoded strings.
	intKey        intKeyState
	intKeyDecided bool // true once we've checked the first batch

	// COUNT(DISTINCT) state: per-group, per-aggregate distinct value sets.
	// Only allocated for aggregates with Kind == AggCountDistinct.
	// Key: group key (string); Value: set of distinct values seen.
	// For int/date/float columns: map[int64]struct{} (raw bits).
	// For string columns: map[string]struct{} (resolved string values).
	distinctInts map[string][]map[int64]struct{}
	distinctStrs map[string][]map[string]struct{}
	hasDistinct  bool // true if any aggExpr has Distinct=true

	// aggVecs is the reused per-batch buffer holding each aggregate's source
	// vector; see acquireAggVecs and the scratch contract in scratch.go.
	aggVecs []Vector
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
	hasDistinct := false
	for i := range resolved {
		ae := &resolved[i]
		srcType := TypeInt64
		if ae.ColIdx >= 0 {
			if ae.ColIdx >= len(childSchema.Fields) {
				return nil, fmt.Errorf("exec: hash aggregate: aggregate column %d out of range", ae.ColIdx)
			}
			srcType = childSchema.Fields[ae.ColIdx].Type
		}
		if ae.AccumType == 0 {
			ae.AccumType = AccumTypeFor(ae.Kind, srcType)
		}
		var t DataType
		switch ae.Kind {
		case AggCount, AggCountDistinct:
			t = TypeInt64
			if ae.Kind == AggCountDistinct {
				hasDistinct = true
			}
		case AggSum, AggMin, AggMax:
			// MIN/MAX return a value of the input type; buildOutputBatch emits a
			// vector matching this declared type for every column type.
			t = srcType
		case AggAvg:
			t = TypeFloat64
		}
		outFields = append(outFields, Field{Name: ae.OutName, Type: t, Nullable: true})
	}

	h := &HashAggregate{
		child:       child,
		groupBy:     groupBy,
		aggExprs:    resolved,
		schema:      Schema{Fields: outFields},
		hasDistinct: hasDistinct,
	}
	h.initMaps()
	return h, nil
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
// consumeAll, by NewHashAggregate, and by parallel workers before their first
// accumulate call.
func (h *HashAggregate) initMaps() {
	h.keys = nil
	h.groups = make(map[string][]int64)
	h.groupCnt = make(map[string]int64)
	h.aggNonNull = make(map[string][]int64)
	h.samples = make(map[string][]groupByVal)
	h.intKeyDecided = false
	h.intKey = intKeyState{}
	// Derived here rather than in NewHashAggregate because newPartialAggregate
	// (parallel.go) builds a HashAggregate by struct literal and reaches this
	// function as its only initialisation step.
	h.hasStrAgg = false
	for _, ae := range h.aggExprs {
		if ae.AccumType == TypeString {
			h.hasStrAgg = true
			break
		}
	}
	if h.hasStrAgg {
		h.strAccs = make(map[string][]string)
	} else {
		h.strAccs = nil
	}
	if h.hasDistinct {
		h.distinctInts = make(map[string][]map[int64]struct{})
		h.distinctStrs = make(map[string][]map[string]struct{})
	}
}

// newAccums allocates and seeds the int64 accumulator slice for a new group.
// MIN/MAX start at the identity for their comparison so the first value always
// wins; SUM/COUNT/AVG start at zero. Aggregates running in a TypeString
// accumulator keep their slot at zero and are tracked in strAccs instead.
func (h *HashAggregate) newAccums() []int64 {
	accs := make([]int64, len(h.aggExprs))
	for j, ae := range h.aggExprs {
		switch ae.Kind {
		case AggMin:
			switch ae.AccumType {
			case TypeFloat64:
				accs[j] = int64(math.Float64bits(math.MaxFloat64))
			case TypeString:
				// no sentinel: aggNonNull is the emptiness test (see strAccs)
			default:
				accs[j] = math.MaxInt64
			}
		case AggMax:
			switch ae.AccumType {
			case TypeFloat64:
				accs[j] = int64(math.Float64bits(-math.MaxFloat64))
			case TypeString:
			default:
				accs[j] = math.MinInt64
			}
		}
	}
	return accs
}

// newStrAccums allocates the string accumulator slice for a new group, or nil
// when this operator has no string-valued aggregate.
func (h *HashAggregate) newStrAccums() []string {
	if !h.hasStrAgg {
		return nil
	}
	return make([]string, len(h.aggExprs))
}

// checkStrAggVecs verifies that every aggregate declared to accumulate strings
// was handed a string vector. AccumType comes from the plan and the vector comes
// from the batch; if they disagree the row values cannot be read, so this reports
// it once per batch instead of letting the per-row assertion panic or — worse —
// letting a fallback path record a zero value.
func (h *HashAggregate) checkStrAggVecs(aggVecs []Vector) error {
	for j, ae := range h.aggExprs {
		if ae.AccumType != TypeString {
			continue
		}
		if _, ok := aggVecs[j].(*StringVector); !ok {
			return fmt.Errorf("exec: aggregate %q accumulates strings but column %d is %T",
				ae.OutName, ae.ColIdx, aggVecs[j])
		}
	}
	return nil
}

// minMaxString folds one non-null string value into aggregate j's running
// min (less) or max. seen reports whether this group has already taken a value;
// the first value always wins, since "" is data rather than an empty marker.
func minMaxString(strs []string, j int, s string, seen bool, less bool) {
	if !seen {
		strs[j] = s
		return
	}
	if less {
		if s < strs[j] {
			strs[j] = s
		}
	} else if s > strs[j] {
		strs[j] = s
	}
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

// acquireAggVecs returns the per-batch buffer holding one source vector per
// aggregate. len(h.aggExprs) is fixed at construction, so this allocates on the
// first batch and reuses that buffer for every batch after it — including across
// morsels, since a worker keeps one pipeline and one HashAggregate for its whole
// run (morsel.go). The caller must write every slot before reading any, which is
// what makes reuse indistinguishable from a fresh make.
func (h *HashAggregate) acquireAggVecs() []Vector {
	if cap(h.aggVecs) < len(h.aggExprs) {
		h.aggVecs = make([]Vector, len(h.aggExprs))
	}
	return h.aggVecs[:len(h.aggExprs)]
}

// accumulate processes one batch into the hash aggregate maps.
// Uses AggExpr.AccumType to determine numeric encoding; no child schema needed.
func (h *HashAggregate) accumulate(batch *Batch) error {
	// Resolve each aggregate's source vector once per batch, not once per row.
	// batch.Vectors[ae.ColIdx] is a slice index + interface load — cheap, but
	// multiplied by len(aggExprs) × the row count it becomes a non-trivial cost.
	//
	// aggVecs is per-instance scratch: one HashAggregate accumulates on one
	// goroutine (newPartialAggregate in parallel.go gives every worker its own),
	// and every slot is written below — including the nil for COUNT(*), which a
	// fresh make used to supply and a reused buffer must not inherit from the
	// previous batch.
	aggVecs := h.acquireAggVecs()
	for j, ae := range h.aggExprs {
		if ae.ColIdx >= 0 {
			aggVecs[j] = batch.Vectors[ae.ColIdx]
		} else {
			aggVecs[j] = nil
		}
	}

	if h.hasStrAgg {
		if err := h.checkStrAggVecs(aggVecs); err != nil {
			return err
		}
	}

	rows := batchRows(batch)

	// Fast path: no GROUP BY eliminates per-row map lookup (e.g. Q6 SUM with no groups).
	if len(h.groupBy) == 0 {
		return h.accumulateDirect(rows, aggVecs)
	}

	// Integer-key fast path: when all GROUP BY columns are dict-encoded strings,
	// key on packed global dictionary codes instead of building composite string keys.
	// Disabled when DISTINCT aggregates are present because the intkey path's
	// []int64 accumulators cannot hold per-group distinct value sets.
	if !h.intKeyDecided {
		h.intKeyDecided = true
		if !h.hasDistinct && canUseIntKey(batch, h.groupBy) {
			h.intKey.enabled = true
			h.intKey.init(len(h.groupBy))
		}
	}

	if h.intKey.enabled {
		return h.accumulateIntKey(batch, rows, aggVecs)
	}

	for ri := 0; ri < rows.n; ri++ {
		rowIdx := rows.at(ri)
		key := h.buildKey(batch, rowIdx)
		accs, exists := h.groups[key]
		if !exists {
			accs = h.newAccums()
			h.groups[key] = accs
			h.aggNonNull[key] = make([]int64, len(h.aggExprs))
			if h.hasStrAgg {
				h.strAccs[key] = h.newStrAccums()
			}
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
			// Initialize distinct sets for this new group.
			if h.hasDistinct {
				intSets := make([]map[int64]struct{}, len(h.aggExprs))
				strSets := make([]map[string]struct{}, len(h.aggExprs))
				for j, ae := range h.aggExprs {
					if ae.Kind == AggCountDistinct {
						if ae.ColIdx >= 0 {
							vec := aggVecs[j]
							if _, isStr := vec.(*StringVector); isStr {
								strSets[j] = make(map[string]struct{})
							} else {
								intSets[j] = make(map[int64]struct{})
							}
						}
					}
				}
				h.distinctInts[key] = intSets
				h.distinctStrs[key] = strSets
			}
		}

		nonNull := h.aggNonNull[key]
		var strs []string
		if h.hasStrAgg {
			strs = h.strAccs[key]
		}
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
			case AggCountDistinct:
				if v.IsNull(rowIdx) {
					continue // NULL values excluded from DISTINCT set (SQL standard)
				}
				if sv, ok := v.(*StringVector); ok {
					s := sv.Get(rowIdx)
					strSets := h.distinctStrs[key]
					if strSets[j] == nil {
						strSets[j] = make(map[string]struct{})
						h.distinctStrs[key] = strSets
					}
					strSets[j][s] = struct{}{}
				} else {
					val := extractInt64(v, rowIdx)
					intSets := h.distinctInts[key]
					if intSets[j] == nil {
						intSets[j] = make(map[int64]struct{})
						h.distinctInts[key] = intSets
					}
					intSets[j][val] = struct{}{}
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
				if ae.AccumType == TypeString {
					minMaxString(strs, j, v.(*StringVector).Get(rowIdx), nonNull[j] > 0, true)
					nonNull[j]++
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
				if ae.AccumType == TypeString {
					minMaxString(strs, j, v.(*StringVector).Get(rowIdx), nonNull[j] > 0, false)
					nonNull[j]++
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
//
// rows names the physical rows of the batch to visit; see rowSet in batch.go.
func (h *HashAggregate) accumulateDirect(rows rowSet, aggVecs []Vector) error {
	const key = ""
	accs, exists := h.groups[key]
	if !exists {
		accs = h.newAccums()
		h.groups[key] = accs
		h.aggNonNull[key] = make([]int64, len(h.aggExprs))
		if h.hasStrAgg {
			h.strAccs[key] = h.newStrAccums()
		}
		h.keys = append(h.keys, key)
		// Initialize distinct sets for the single implicit group.
		if h.hasDistinct {
			h.distinctInts[key] = make([]map[int64]struct{}, len(h.aggExprs))
			h.distinctStrs[key] = make([]map[string]struct{}, len(h.aggExprs))
		}
	}

	nonNull := h.aggNonNull[key]
	var strs []string
	if h.hasStrAgg {
		strs = h.strAccs[key]
	}

	for j, ae := range h.aggExprs {
		v := aggVecs[j]
		switch ae.Kind {
		case AggCount:
			if ae.ColIdx < 0 {
				accs[j] += int64(rows.n) // COUNT(*): no per-row null check needed
				nonNull[j] += int64(rows.n)
			} else {
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !v.IsNull(rowIdx) {
						accs[j]++
						nonNull[j]++
					}
				}
			}
		case AggCountDistinct:
			for ri := 0; ri < rows.n; ri++ {
				rowIdx := rows.at(ri)
				if v.IsNull(rowIdx) {
					continue // NULL excluded from DISTINCT set
				}
				if sv, ok := v.(*StringVector); ok {
					s := sv.Get(rowIdx)
					strSets := h.distinctStrs[key]
					if strSets[j] == nil {
						strSets[j] = make(map[string]struct{})
						h.distinctStrs[key] = strSets
					}
					strSets[j][s] = struct{}{}
				} else {
					val := extractInt64(v, rowIdx)
					intSets := h.distinctInts[key]
					if intSets[j] == nil {
						intSets[j] = make(map[int64]struct{})
						h.distinctInts[key] = intSets
					}
					intSets[j][val] = struct{}{}
				}
			}
		case AggSum:
			if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !fv.IsNull(rowIdx) {
						cur += fv.Values[rowIdx]
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !v.IsNull(rowIdx) {
						accs[j] += extractInt64(v, rowIdx)
						nonNull[j]++
					}
				}
			}
		case AggAvg:
			if fv, ok := v.(*Float64Vector); ok {
				cur := math.Float64frombits(uint64(accs[j]))
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !fv.IsNull(rowIdx) {
						cur += fv.Values[rowIdx]
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				cur := math.Float64frombits(uint64(accs[j]))
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !v.IsNull(rowIdx) {
						cur += float64(extractInt64(v, rowIdx))
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			}
		case AggMin:
			if ae.AccumType == TypeString {
				sv := v.(*StringVector)
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !sv.IsNull(rowIdx) {
						minMaxString(strs, j, sv.Get(rowIdx), nonNull[j] > 0, true)
						nonNull[j]++
					}
				}
			} else if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !fv.IsNull(rowIdx) {
						if fv.Values[rowIdx] < cur {
							cur = fv.Values[rowIdx]
						}
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !v.IsNull(rowIdx) {
						if val := extractInt64(v, rowIdx); val < accs[j] {
							accs[j] = val
						}
						nonNull[j]++
					}
				}
			}
		case AggMax:
			if ae.AccumType == TypeString {
				sv := v.(*StringVector)
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !sv.IsNull(rowIdx) {
						minMaxString(strs, j, sv.Get(rowIdx), nonNull[j] > 0, false)
						nonNull[j]++
					}
				}
			} else if ae.AccumType == TypeFloat64 {
				fv := v.(*Float64Vector)
				cur := math.Float64frombits(uint64(accs[j]))
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
					if !fv.IsNull(rowIdx) {
						if fv.Values[rowIdx] > cur {
							cur = fv.Values[rowIdx]
						}
						nonNull[j]++
					}
				}
				accs[j] = int64(math.Float64bits(cur))
			} else {
				for ri := 0; ri < rows.n; ri++ {
					rowIdx := rows.at(ri)
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
		case AggCountDistinct:
			// COUNT(DISTINCT) resolves to the size of the distinct value set.
			// Never returns NULL (returns 0 for empty/all-null groups).
			out := &Int64Vector{Values: make([]int64, n), NullBitmap: storage.FullBitmap(n)}
			for i, key := range keys {
				var count int64
				if intSets := h.distinctInts[key]; intSets != nil && intSets[j] != nil {
					count = int64(len(intSets[j]))
				}
				if strSets := h.distinctStrs[key]; strSets != nil && strSets[j] != nil {
					count = int64(len(strSets[j]))
				}
				out.Values[i] = count
			}
			vecs[outIdx] = out
		case AggSum:
			// SUM returns NULL when all inputs were null.
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
		case AggMin, AggMax:
			// MIN/MAX return a value of the input type, so the vector kind is
			// chosen by the declared output type rather than by the accumulator
			// encoding. Returning an Int64Vector for every non-float column was
			// what made MIN over a STRING column emit 0, and over a DATE column
			// emit raw day numbers, under a schema that claimed otherwise.
			// SUM/MIN/MAX return NULL when all inputs were null.
			vecs[outIdx] = h.buildMinMaxVector(keys, j, h.schema.Fields[outIdx].Type, n)
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

// buildMinMaxVector materialises aggregate j's MIN/MAX result for the given
// group keys as a vector of outType — the type the operator's schema declares for
// that column, which for MIN/MAX is the input column's type.
//
// A group whose non-null input count is zero produces NULL, which is also what
// keeps the accumulators' zero values from ever being read as data: an
// untouched int64 slot still holds its MaxInt64/MinInt64 sentinel and an
// untouched strAccs slot still holds "".
func (h *HashAggregate) buildMinMaxVector(keys []string, j int, outType DataType, n int) Vector {
	nullBmp := storage.FullBitmap(n)
	isNull := func(key string) bool { return h.aggNonNull[key][j] == 0 }

	switch outType {
	case TypeString:
		// Codes index a fresh output dictionary; the input dictionaries are per
		// row group and carry codes in first-occurrence order, so they cannot be
		// reused or compared here.
		db := storage.NewDictBuilder()
		codes := make([]uint32, n)
		bmp := make([]byte, (n+7)/8)
		for i, key := range keys {
			if isNull(key) {
				continue // leave invalid: NULL
			}
			var s string
			if strs := h.strAccs[key]; strs != nil {
				s = strs[j]
			}
			codes[i] = db.Add(s)
			storage.SetValidBit(bmp, i)
		}
		return newStringVector(db, codes, bmp)

	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: nullBmp}
		for i, key := range keys {
			if isNull(key) {
				storage.SetNullBit(out.NullBitmap, i)
				continue
			}
			out.Values[i] = math.Float64frombits(uint64(h.groups[key][j]))
		}
		return out

	case TypeDate:
		out := &DateVector{Values: make([]int32, n), NullBitmap: nullBmp}
		for i, key := range keys {
			if isNull(key) {
				storage.SetNullBit(out.NullBitmap, i)
				continue
			}
			out.Values[i] = int32(h.groups[key][j])
		}
		return out

	case TypeBool:
		out := &BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: nullBmp, Length: n}
		for i, key := range keys {
			if isNull(key) {
				storage.SetNullBit(out.NullBitmap, i)
				continue
			}
			out.Set(i, h.groups[key][j] != 0)
		}
		return out

	default: // TypeInt64
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: nullBmp}
		for i, key := range keys {
			if isNull(key) {
				storage.SetNullBit(out.NullBitmap, i)
				continue
			}
			out.Values[i] = h.groups[key][j]
		}
		return out
	}
}

// buildEmptyGlobalResult creates a single-row batch for a global aggregate
// (no GROUP BY) that consumed zero input rows. SQL requires:
//   - COUNT -> 0 (never NULL)
//   - SUM/AVG/MIN/MAX -> NULL
func (h *HashAggregate) buildEmptyGlobalResult() *Batch {
	vecs := make([]Vector, len(h.aggExprs))
	for j, ae := range h.aggExprs {
		switch ae.Kind {
		case AggCount, AggCountDistinct:
			vecs[j] = &Int64Vector{Values: []int64{0}, NullBitmap: storage.FullBitmap(1)}
		case AggMin, AggMax:
			// One all-NULL row, in the type the schema declares for this column.
			vecs[j] = nullVectorOfType(h.schema.Fields[j].Type)
		case AggSum:
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

// nullVectorOfType returns a single-row vector of type t holding one NULL.
// The null bitmap is all-zero, which is "invalid" in this codebase's LSB-first
// convention, so no value byte is ever read.
func nullVectorOfType(t DataType) Vector {
	switch t {
	case TypeString:
		return &StringVector{Codes: []uint32{0}, Dict: nil, NullBitmap: make([]byte, 1)}
	case TypeFloat64:
		return &Float64Vector{Values: []float64{0}, NullBitmap: make([]byte, 1)}
	case TypeDate:
		return &DateVector{Values: []int32{0}, NullBitmap: make([]byte, 1)}
	case TypeBool:
		return &BoolVector{Bits: make([]byte, 1), NullBitmap: make([]byte, 1), Length: 1}
	default: // TypeInt64
		return &Int64Vector{Values: []int64{0}, NullBitmap: make([]byte, 1)}
	}
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

	case TypeBool:
		out := &BoolVector{Bits: make([]byte, (n+7)/8), NullBitmap: make([]byte, (n+7)/8), Length: n}
		for i, key := range keys {
			sample := h.samples[key]
			if sample == nil || sample[gbPos].isNull {
				continue
			}
			out.Set(i, sample[gbPos].bits != 0)
			storage.SetValidBit(out.NullBitmap, i)
		}
		return out

	default: // TypeInt64
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
//
// The default arm returns 0 for any vector kind with no int64 encoding — today
// that is only *StringVector. Callers must not route a string column here
// expecting a comparable value: this silent 0 is what made MIN/MAX over a STRING
// column return 0 for every input, since 0 beats both MaxInt64 and MinInt64.
// MIN/MAX now run strings through a TypeString accumulator (see strAccs) and
// never reach this function. SUM over a STRING column still does, and still
// yields 0 — a pre-existing defect left untouched here, since a numeric SUM of
// text has no correct answer to converge on and rejecting it is a separate
// behavioural change.
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
