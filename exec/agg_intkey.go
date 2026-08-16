package exec

import (
	"math"

	"github.com/ryderpongracic1/vexq/storage"
)

// intKeyState holds the state for the integer-key fast path in HashAggregate.
// When all GROUP BY columns are dict-encoded strings with small dictionaries,
// the aggregate can key on packed uint64 dictionary codes instead of building
// a composite string key per row. This eliminates ~6M allocations per Q1 scan.
//
// Cross-rowgroup code stability: dictionaries are per row group, so the same
// string may have different local codes across row groups. A per-column
// globalDict assigns stable global codes; a per-rowgroup remap table (built
// once per new dictionary, O(dict_size)) maps local codes to global codes.
// The hot loop becomes: globalCode = remap[localCode]; key |= globalCode << shift.
type intKeyState struct {
	enabled bool // true if fast path is active

	// Per GROUP BY column: global dictionary for stable code assignment.
	globalDicts []*storage.DictBuilder

	// Per GROUP BY column: bit shift for packing into uint64.
	// Column i occupies bits [shifts[i], shifts[i]+widths[i]).
	shifts []uint8
	widths []uint8

	// Per GROUP BY column: remap table keyed by dict pointer identity.
	// Maps *storage.Dictionary → []uint32 (localCode → globalCode).
	// Built lazily on first batch from each new rowgroup dictionary.
	remaps []map[*storage.Dictionary][]uint32

	// Integer-keyed maps (parallel to the string-keyed maps in HashAggregate).
	intKeys    []uint64                // insertion-order keys
	intGroups  map[uint64][]int64      // key → accumulators
	intNonNull map[uint64][]int64      // key → per-aggregate non-null count
	intSamples map[uint64][]groupByVal // key → representative values
	intStrAccs map[uint64][]string     // key → string MIN/MAX values (see HashAggregate.strAccs)

	// Reused per-batch buffers: the GROUP BY columns' string vectors and their
	// local→global remap tables for the batch being accumulated. See
	// acquireBatchBufs.
	svecBuf  []*StringVector
	remapBuf [][]uint32
}

// acquireBatchBufs returns the per-batch svec and remap buffers sized to n
// GROUP BY columns, allocating on first use and reusing them afterwards.
// accumulateIntKey writes every entry of both before reading either, so a reused
// buffer is indistinguishable from a fresh one — a batch can never observe the
// previous batch's dictionaries. n is len(HashAggregate.groupBy), fixed for the
// operator's lifetime.
func (ik *intKeyState) acquireBatchBufs(n int) ([]*StringVector, [][]uint32) {
	if cap(ik.svecBuf) < n {
		ik.svecBuf = make([]*StringVector, n)
	}
	if cap(ik.remapBuf) < n {
		ik.remapBuf = make([][]uint32, n)
	}
	return ik.svecBuf[:n], ik.remapBuf[:n]
}

// initIntKeyState initializes the integer-key state for the fast path.
// Called from HashAggregate.initMaps when intKeyMode is detected.
func (ik *intKeyState) init(numGroupByCols int) {
	ik.intKeys = nil
	ik.intGroups = make(map[uint64][]int64)
	ik.intNonNull = make(map[uint64][]int64)
	ik.intSamples = make(map[uint64][]groupByVal)
	ik.intStrAccs = make(map[uint64][]string)
	ik.globalDicts = make([]*storage.DictBuilder, numGroupByCols)
	ik.shifts = make([]uint8, numGroupByCols)
	ik.widths = make([]uint8, numGroupByCols)
	ik.remaps = make([]map[*storage.Dictionary][]uint32, numGroupByCols)
	for i := range numGroupByCols {
		ik.globalDicts[i] = storage.NewDictBuilder()
		ik.remaps[i] = make(map[*storage.Dictionary][]uint32)
	}
	// Default: 32 bits per column for up to 2 columns. This handles Q1
	// (l_returnflag, l_linestatus: 3×2 = 6 groups, max code < 4).
	// For >2 columns or if a dict grows beyond width, fall back to string path.
	if numGroupByCols <= 2 {
		for i := range numGroupByCols {
			ik.shifts[i] = uint8(i) * 32
			ik.widths[i] = 32
		}
	} else if numGroupByCols <= 4 {
		for i := range numGroupByCols {
			ik.shifts[i] = uint8(i) * 16
			ik.widths[i] = 16
		}
	} else if numGroupByCols <= 8 {
		for i := range numGroupByCols {
			ik.shifts[i] = uint8(i) * 8
			ik.widths[i] = 8
		}
	}
}

// canUseIntKey checks whether all GROUP BY columns in the batch are dict-encoded
// strings (StringVector with non-nil Dict). Called on the first batch to decide.
func canUseIntKey(batch *Batch, groupBy []int) bool {
	if len(groupBy) == 0 || len(groupBy) > 8 {
		return false
	}
	for _, colIdx := range groupBy {
		v := batch.Vectors[colIdx]
		sv, ok := v.(*StringVector)
		if !ok || sv.Dict == nil {
			return false
		}
	}
	return true
}

// getOrBuildRemap returns the remap table for the given column's dictionary.
// If the dict hasn't been seen before, builds the remap by assigning global codes.
func (ik *intKeyState) getOrBuildRemap(colPos int, dict *storage.Dictionary) []uint32 {
	if remap, ok := ik.remaps[colPos][dict]; ok {
		return remap
	}
	// Build remap: for each local code, look up the string and assign a global code.
	numEntries := len(dict.Offsets)
	remap := make([]uint32, numEntries)
	for localCode := uint32(0); int(localCode) < numEntries; localCode++ {
		s := dict.Get(localCode)
		globalCode := ik.globalDicts[colPos].Add(s)
		remap[localCode] = globalCode
	}
	ik.remaps[colPos][dict] = remap
	return remap
}

// buildIntKey packs the global dictionary codes for one row into a uint64.
// Caller must have already obtained remap tables for all columns in this batch.
func (ik *intKeyState) buildIntKey(remaps [][]uint32, vecs []*StringVector, rowIdx int) uint64 {
	var key uint64
	for i, sv := range vecs {
		globalCode := remaps[i][sv.Codes[rowIdx]]
		key |= uint64(globalCode) << ik.shifts[i]
	}
	return key
}

// codeOverflows returns true if any global dict has grown beyond the bit width
// allocated for that column. In that case the caller should fall back to string keys.
func (ik *intKeyState) codeOverflows() bool {
	for i, gd := range ik.globalDicts {
		maxCode := uint64(gd.Len())
		if maxCode >= (1 << ik.widths[i]) {
			return true
		}
	}
	return false
}

// materializeToStringMaps converts the uint64-keyed state into the string-keyed
// maps used by buildOutputBatch and mergePartialAgg. Called once after accumulation
// is complete — O(num_groups), not O(num_rows).
func (ik *intKeyState) materializeToStringMaps(h *HashAggregate) {
	for _, intKey := range ik.intKeys {
		strKey := ik.intKeyToStringKey(h, intKey)
		h.keys = append(h.keys, strKey)
		h.groups[strKey] = ik.intGroups[intKey]
		h.aggNonNull[strKey] = ik.intNonNull[intKey]
		h.samples[strKey] = ik.intSamples[intKey]
		if h.hasStrAgg {
			h.strAccs[strKey] = ik.intStrAccs[intKey]
		}
	}
}

// intKeyToStringKey converts a uint64 key back to the canonical string key format
// by looking up global codes in the global dictionaries and serializing.
func (ik *intKeyState) intKeyToStringKey(h *HashAggregate, intKey uint64) string {
	buf := make([]byte, 0, len(h.groupBy)*10)
	for i := range h.groupBy {
		globalCode := uint32((intKey >> ik.shifts[i]) & ((1 << ik.widths[i]) - 1))
		// Check if this was a null sentinel (we use maxCode+1 as null marker)
		sample := ik.intSamples[intKey]
		if sample != nil && sample[i].isNull {
			buf = append(buf, 0x00, 0xFF)
		} else {
			// Look up string from global dict
			s := ik.globalDicts[i].GetByCode(globalCode)
			buf = append(buf, 0x02)
			buf = appendUint32LE(buf, uint32(len(s)))
			buf = append(buf, s...)
			buf = append(buf, 0xFF)
		}
	}
	return string(buf)
}

// appendUint32LE appends a little-endian uint32 to buf.
func appendUint32LE(buf []byte, v uint32) []byte {
	return append(buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

// accumulateIntKey is the integer-key fast path for accumulate.
// It replaces the per-row string key construction with a packed uint64 lookup.
//
// rows names the physical rows of the batch to visit; see rowSet in batch.go.
func (h *HashAggregate) accumulateIntKey(batch *Batch, rows rowSet, aggVecs []Vector) error {
	ik := &h.intKey

	// Resolve string vectors and build/retrieve remap tables for this batch.
	// Both buffers are per-instance scratch, fully rewritten here before either
	// is read: one entry per GROUP BY column, and len(h.groupBy) never changes.
	svecs, remaps := ik.acquireBatchBufs(len(h.groupBy))
	for i, colIdx := range h.groupBy {
		sv := batch.Vectors[colIdx].(*StringVector)
		svecs[i] = sv
		remaps[i] = ik.getOrBuildRemap(i, sv.Dict)
	}

	// Check if global dicts have overflowed their bit widths.
	if ik.codeOverflows() {
		// Fall back to string path: materialize what we have so far, disable intkey.
		ik.enabled = false
		if len(ik.intKeys) > 0 {
			ik.materializeToStringMaps(h)
			ik.intKeys = nil
			ik.intGroups = nil
			ik.intNonNull = nil
			ik.intSamples = nil
			ik.intStrAccs = nil
		}
		return h.accumulateStringKey(batch, rows, aggVecs)
	}

	// Hot loop: build uint64 key and accumulate.
	for ri := 0; ri < rows.n; ri++ {
		rowIdx := rows.at(ri)
		// Check for null group-by columns — use a special null-aware key.
		hasNull := false
		for i, sv := range svecs {
			if sv.IsNull(rowIdx) {
				hasNull = true
				_ = i
				break
			}
		}

		var key uint64
		if hasNull {
			// Null handling: fall through to string-key path for this row.
			// Nulls in GROUP BY are rare; correctness > perf here.
			strKey := h.buildKey(batch, rowIdx)
			h.accumulateOneRow(strKey, batch, rowIdx, aggVecs)
			continue
		}

		// Pack global codes into uint64.
		for i, sv := range svecs {
			localCode := sv.Codes[rowIdx]
			globalCode := remaps[i][localCode]
			key |= uint64(globalCode) << ik.shifts[i]
		}

		accs, exists := ik.intGroups[key]
		if !exists {
			accs = h.newAccums()
			ik.intGroups[key] = accs
			ik.intNonNull[key] = make([]int64, len(h.aggExprs))
			if h.hasStrAgg {
				ik.intStrAccs[key] = h.newStrAccums()
			}
			ik.intKeys = append(ik.intKeys, key)

			// Store sample for output reconstruction.
			sample := make([]groupByVal, len(h.groupBy))
			for si, sv := range svecs {
				s := sv.Dict.Get(sv.Codes[rowIdx])
				sample[si] = groupByVal{strVal: s}
			}
			ik.intSamples[key] = sample
		}

		nonNull := ik.intNonNull[key]
		var strs []string
		if h.hasStrAgg {
			strs = ik.intStrAccs[key]
		}
		h.foldRow(accs, nonNull, strs, aggVecs, rowIdx)
	}
	return nil
}

// foldRow folds one input row into a single group's accumulators.
//
// It is shared by accumulateIntKey and accumulateOneRow, which previously each
// carried their own copy of this switch alongside the two in aggregate.go. Four
// copies of one fold is how MIN/MAX over strings had to be fixed in four places
// at once; two of them now route here.
//
// accs, nonNull and strs are the group's int64 accumulators, per-aggregate
// non-null counts, and string accumulators (nil when the operator has no
// string-valued aggregate). COUNT(DISTINCT) is not handled: both callers run
// only when h.hasDistinct is false.
func (h *HashAggregate) foldRow(accs, nonNull []int64, strs []string, aggVecs []Vector, rowIdx int) {
	for j, ae := range h.aggExprs {
		v := aggVecs[j]
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
			cur := math.Float64frombits(uint64(accs[j]))
			if fv, ok := v.(*Float64Vector); ok {
				accs[j] = int64(math.Float64bits(cur + fv.Values[rowIdx]))
			} else {
				accs[j] = int64(math.Float64bits(cur + float64(extractInt64(v, rowIdx))))
			}
		case AggMin, AggMax:
			if v.IsNull(rowIdx) {
				continue
			}
			less := ae.Kind == AggMin
			if ae.AccumType == TypeString {
				minMaxString(strs, j, v.(*StringVector).Get(rowIdx), nonNull[j] > 0, less)
				nonNull[j]++
				continue
			}
			nonNull[j]++
			val := extractInt64(v, rowIdx)
			if ae.AccumType == TypeFloat64 {
				cur := math.Float64frombits(uint64(accs[j]))
				fv := math.Float64frombits(uint64(val))
				if (less && fv < cur) || (!less && fv > cur) {
					accs[j] = val
				}
			} else if (less && val < accs[j]) || (!less && val > accs[j]) {
				accs[j] = val
			}
		}
	}
}

// accumulateStringKey is the original string-key path, extracted so it can be
// called as a fallback when intkey overflows or encounters non-dict columns.
//
// rows names the physical rows of the batch to visit; see rowSet in batch.go.
func (h *HashAggregate) accumulateStringKey(batch *Batch, rows rowSet, aggVecs []Vector) error {
	for ri := 0; ri < rows.n; ri++ {
		rowIdx := rows.at(ri)
		key := h.buildKey(batch, rowIdx)
		h.accumulateOneRow(key, batch, rowIdx, aggVecs)
	}
	return nil
}

// accumulateOneRow inserts or updates a single row into the string-keyed maps.
func (h *HashAggregate) accumulateOneRow(key string, batch *Batch, rowIdx int, aggVecs []Vector) {
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
	}

	var strs []string
	if h.hasStrAgg {
		strs = h.strAccs[key]
	}
	h.foldRow(accs, h.aggNonNull[key], strs, aggVecs, rowIdx)
}
