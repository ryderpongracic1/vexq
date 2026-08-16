package exec

import (
	"context"
	"fmt"
	"math"

	"github.com/ryderpongracic1/vexq/storage"
)

// HashJoin implements an inner hash join.  The build side (smaller table) is
// fully materialised into a hash table; the probe side streams through.
//
// Only equality joins on a single column pair are supported in v1.
type HashJoin struct {
	build    Operator
	probe    Operator
	buildKey int // column index in build schema
	probeKey int // column index in probe schema
	schema   Schema

	// buildSchema is the build side's schema. Held separately from build so a
	// join can probe a prebuilt SharedHashTable with no build operator at all
	// (see NewHashJoinShared in parallel_join.go).
	buildSchema Schema

	// Build-phase hash table: parts holds one flat open-addressed table per
	// radix partition, and a key lives in parts[radixPart(key, partMask)]. A
	// self-building join has exactly one partition and a zero partMask, which
	// radixPart short-circuits — so the serial and shared paths run the same
	// probe code, and the serial path pays nothing for partitioning it does not
	// use. store holds the build rows the tables index into; it is shared across
	// partitions, so a row index identifies a row table-wide.
	parts    []*joinHashTable
	partMask uint64
	store    *rowStore

	buildDone bool

	// probe-phase state
	probeBatch *Batch
	probePos   int
	matchBuf   []joinRow // pending output rows
	matchPos   int
}

// joinRow is one output row: a build row index into HashJoin.store paired with a
// row index into the current probe batch. Eight bytes, so a match buffer of a
// full output batch stays small and emitMatches walks it sequentially — the
// previous representation embedded three slice headers per match.
type joinRow struct {
	build int32
	probe int32
}

func NewHashJoin(build, probe Operator, buildKeyIdx, probeKeyIdx int) (*HashJoin, error) {
	bSchema := build.Schema()
	pSchema := probe.Schema()

	if buildKeyIdx < 0 || buildKeyIdx >= len(bSchema.Fields) {
		return nil, fmt.Errorf("exec: hash join: build key %d out of range", buildKeyIdx)
	}
	if probeKeyIdx < 0 || probeKeyIdx >= len(pSchema.Fields) {
		return nil, fmt.Errorf("exec: hash join: probe key %d out of range", probeKeyIdx)
	}

	// Output schema: all build columns then all probe columns.
	var outFields []Field
	outFields = append(outFields, bSchema.Fields...)
	outFields = append(outFields, pSchema.Fields...)

	return &HashJoin{
		build:       build,
		probe:       probe,
		buildKey:    buildKeyIdx,
		probeKey:    probeKeyIdx,
		buildSchema: bSchema,
		schema:      Schema{Fields: outFields},
	}, nil
}

func (j *HashJoin) Schema() Schema { return j.schema }

func (j *HashJoin) Next(ctx context.Context) (*Batch, error) {
	if !j.buildDone {
		if err := j.buildHashTable(ctx); err != nil {
			return nil, err
		}
		j.buildDone = true
	}

	for {
		// Emit any buffered join rows first.
		if j.matchPos < len(j.matchBuf) {
			batch := j.emitMatches()
			if batch != nil {
				return batch, nil
			}
		}

		// Get next probe batch.
		if j.probeBatch == nil || j.probePos >= j.probeBatch.Length {
			pb, err := j.probe.Next(ctx)
			if err != nil {
				return nil, fmt.Errorf("exec: hash join: probe: %w", err)
			}
			if pb == nil {
				return nil, nil // EOF
			}
			j.probeBatch = pb
			j.probePos = 0
		}

		// Probe hash table.
		j.matchBuf = j.matchBuf[:0]
		j.matchPos = 0
		pBatch := j.probeBatch
		n := pBatch.Length

		// Walk the probe rows without materialising their indices. The batch's
		// selection vector is []uint16 and the unfiltered case is 0..n-1, so the
		// only reason to build a []int was the `range` loop that used to follow —
		// 8 KB of garbage per batch for a value both branches already have. sel
		// is loop-invariant, so the branch inside the loop is perfectly
		// predicted, and the visit order is unchanged: selection-vector order
		// where one is installed, ascending row order where none is.
		sel := pBatch.SelVec
		rows := n
		if sel != nil {
			rows = len(sel)
		}
		pv := pBatch.Vectors[j.probeKey]

		for i := 0; i < rows; i++ {
			rowIdx := i
			if sel != nil {
				rowIdx = int(sel[i])
			}
			if pv.IsNull(rowIdx) {
				continue
			}
			key := extractInt64(pv, rowIdx)
			// One mix serves both indexes: the low bits pick the partition (a
			// zero partMask selects the only one), the high bits pick the slot.
			h := radixHash(key)
			tbl := j.parts[h&j.partMask]
			// One lookup lands on the key's first build row; the rest of that
			// key's rows follow the chain, in build order.
			for ri := tbl.lookupHashed(key, h); ri != noRow; ri = j.store.next[ri] {
				j.matchBuf = append(j.matchBuf, joinRow{build: ri, probe: int32(rowIdx)})
			}
		}
		j.probePos = n
	}
}

func (j *HashJoin) buildHashTable(ctx context.Context) error {
	// What the build side can say about its own row count before producing a
	// row: the scan's footer count where the build side is a scan, and nothing
	// where a filter or a nested join sits in between (see RowCountBound). A
	// presized store never grows, which is the whole of the win — growth was
	// 31.6% of every byte the join benchmarks allocated, and 59.5% of that came
	// from this one call site.
	rows := buildRowsPresize(j.build)
	j.store = newRowStore(j.buildSchema, rows)
	// One unpartitioned table: a self-building serial join has no morsels to
	// split, so radix partitioning would buy it nothing.
	//
	// The same count sizes the slot array. It bounds the distinct key count
	// rather than equalling it, so a duplicate-heavy build side over-allocates
	// slots — the trade newJoinHashTable documents, and the one the parallel
	// build's pass 2 has always made with its per-partition row counts. It is
	// the cheap direction: 16 bytes per unused slot against a rehash of the
	// whole array, which was 12.6% of the same profile.
	tbl := newJoinHashTable(j.store, rows)
	j.parts = []*joinHashTable{tbl}
	j.partMask = 0
	return forEachBuildRow(ctx, j.build, j.buildKey, func(key int64, batch *Batch, rowIdx int) error {
		row, err := j.store.appendFrom(batch, rowIdx)
		if err != nil {
			return err
		}
		tbl.insert(key, row)
		return nil
	})
}

// forEachBuildRow drains src and calls visit once per row that has a non-NULL
// join key, in the order src produces them, passing the row's int64 key and the
// batch coordinates of the row. Rows whose key is NULL are dropped — an inner
// equi-join can never match them.
//
// visit must not retain batch: TableScan reuses its decode buffers between
// batches, so a row that outlives the call has to be copied out, which is what
// rowStore.appendFrom does.
//
// Every build-side strategy drains through here — the serial join's own table,
// the serial radix build, and the morsel-parallel build in parallel_join.go —
// which is what guarantees they see the same rows in the same order.
func forEachBuildRow(ctx context.Context, src Operator, keyIdx int, visit func(key int64, batch *Batch, rowIdx int) error) error {
	for {
		batch, err := src.Next(ctx)
		if err != nil {
			return fmt.Errorf("exec: hash join: build: %w", err)
		}
		if batch == nil {
			return nil
		}

		// Walk the batch's rows without materialising their indices — the same
		// widening HashJoin.Next used to do, and the reason it is eliminated here
		// rather than buffered on a receiver is that this is a free function
		// called concurrently by the parallel build's workers, so there is no
		// single-goroutine owner to hang scratch off.
		//
		// The visit order is exactly what it was: selection-vector order where
		// one is installed, ascending row order where none is. That is load
		// bearing — BuildSharedHashTableParallel's determinism guarantee (per-key
		// build-row order identical to a serial drain) is stated in terms of the
		// order this loop hands rows to visit.
		sel := batch.SelVec
		rows := batch.Length
		if sel != nil {
			rows = len(sel)
		}

		kv := batch.Vectors[keyIdx]
		for i := 0; i < rows; i++ {
			rowIdx := i
			if sel != nil {
				rowIdx = int(sel[i])
			}
			if kv.IsNull(rowIdx) {
				continue
			}
			if err := visit(extractInt64(kv, rowIdx), batch, rowIdx); err != nil {
				return err
			}
		}
	}
}

func (j *HashJoin) emitMatches() *Batch {
	end := j.matchPos + BlockRows
	if end > len(j.matchBuf) {
		end = len(j.matchBuf)
	}
	if j.matchPos >= end {
		return nil
	}
	rows := j.matchBuf[j.matchPos:end]
	j.matchPos = end

	bSchema := j.buildSchema
	pSchema := j.probe.Schema()
	n := len(rows)

	vecs := make([]Vector, len(bSchema.Fields)+len(pSchema.Fields))
	// Build columns.
	for c, field := range bSchema.Fields {
		vecs[c] = j.buildColumnFromRows(rows, c, field.Type, n)
	}
	// Probe columns.
	pOff := len(bSchema.Fields)
	for c, field := range pSchema.Fields {
		vecs[pOff+c] = j.probeColumnFromRows(rows, c, field.Type, n)
	}
	return &Batch{Schema: j.schema, Vectors: vecs, Length: n}
}

func (j *HashJoin) buildColumnFromRows(rows []joinRow, colIdx int, t DataType, n int) Vector {
	s := j.store
	switch t {
	case TypeInt64:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !s.isNull(r.build, colIdx) {
				out.Values[i] = s.value(r.build, colIdx)
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !s.isNull(r.build, colIdx) {
				out.Values[i] = math.Float64frombits(uint64(s.value(r.build, colIdx)))
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeDate:
		out := &DateVector{Values: make([]int32, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !s.isNull(r.build, colIdx) {
				out.Values[i] = int32(s.value(r.build, colIdx))
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeString:
		db := storage.NewDictBuilder()
		codes := make([]uint32, n)
		nullBmp := make([]byte, (n+7)/8)
		for i, r := range rows {
			if !s.isNull(r.build, colIdx) {
				codes[i] = db.Add(s.str(r.build, colIdx))
				storage.SetValidBit(nullBmp, i)
			}
		}
		return newStringVector(db, codes, nullBmp)
	default: // TypeBool
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !s.isNull(r.build, colIdx) {
				out.Values[i] = s.value(r.build, colIdx)
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	}
}

func (j *HashJoin) probeColumnFromRows(rows []joinRow, colIdx int, t DataType, n int) Vector {
	pBatch := j.probeBatch
	src := pBatch.Vectors[colIdx]
	switch t {
	case TypeInt64:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		sv := src.(*Int64Vector)
		for i, r := range rows {
			pRow := int(r.probe)
			if !sv.IsNull(pRow) {
				out.Values[i] = sv.Values[pRow]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		sv := src.(*Float64Vector)
		for i, r := range rows {
			pRow := int(r.probe)
			if !sv.IsNull(pRow) {
				out.Values[i] = sv.Values[pRow]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeDate:
		out := &DateVector{Values: make([]int32, n), NullBitmap: make([]byte, (n+7)/8)}
		sv := src.(*DateVector)
		for i, r := range rows {
			pRow := int(r.probe)
			if !sv.IsNull(pRow) {
				out.Values[i] = sv.Values[pRow]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeString:
		sv := src.(*StringVector)
		// Carry the probe batch's dictionary through and copy the codes, rather
		// than decoding every row to a string and rebuilding a dictionary per
		// emitted batch. Project.materialize and copyVector already reproject a
		// StringVector this way; the reason it is safe is that a Dictionary is
		// immutable once UnmarshalDictionary returns it and nothing rewrites one
		// in place — ColumnReader.Dictionary builds exactly one per row group
		// under a sync.Once and replaces the pointer when the scan moves on, so a
		// batch that outlives its row group keeps its own dictionary alive rather
		// than seeing it change underneath.
		//
		// Codes are still copied into a fresh slice: sv.Codes is one of
		// TableScan's reused decode buffers (scan.go), so aliasing it would leak
		// the next block's codes into an already-emitted batch.
		//
		// The dictionary may describe more strings than this batch uses, which is
		// the same over-approximation materialize produces and costs nothing to
		// read. It also gives the int-key aggregate a stable dictionary pointer
		// per row group instead of a fresh one per batch, so its remap cache
		// (agg_intkey.go getOrBuildRemap) hits instead of rebuilding.
		codes := make([]uint32, n)
		nullBmp := make([]byte, (n+7)/8)
		for i, r := range rows {
			pRow := int(r.probe)
			if !sv.IsNull(pRow) {
				codes[i] = sv.Codes[pRow]
				storage.SetValidBit(nullBmp, i)
			}
		}
		return &StringVector{Codes: codes, Dict: sv.Dict, NullBitmap: nullBmp}
	default: // TypeBool
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			pRow := int(r.probe)
			if !src.IsNull(pRow) {
				out.Values[i] = extractInt64(src, pRow)
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	}
}

// Reset repositions the probe side onto row groups [rgStart, rgEnd) and
// discards this join's probe-phase state, so one worker pipeline can probe every
// morsel it claims (see MorselPipeline).
//
// The build table is deliberately kept: reusableMorselPipeline only accepts a
// join probing a SharedHashTable, which is materialised once by the operator
// above and read-only for the whole probe phase. Every field cleared here holds
// rows or offsets from the previous morsel's probe batch — probeBatch above all,
// which probeColumnFromRows reads through — so keeping any of them would emit
// the previous morsel's rows again. matchBuf keeps its capacity, since
// Next truncates and refills it before reading it.
//
// See Filter.Reset for why an unresettable probe child panics rather than being
// skipped.
func (j *HashJoin) Reset(rgStart, rgEnd int) {
	j.probeBatch = nil
	j.probePos = 0
	j.matchBuf = j.matchBuf[:0]
	j.matchPos = 0
	j.probe.(MorselPipeline).Reset(rgStart, rgEnd)
}

func (j *HashJoin) Close() error {
	// build is nil when probing a SharedHashTable — the table is owned by the
	// ParallelHashJoinAggregate that built it, not by this join.
	if j.build != nil {
		_ = j.build.Close()
	}
	return j.probe.Close()
}
