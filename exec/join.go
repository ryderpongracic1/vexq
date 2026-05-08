package exec

import (
	"context"
	"fmt"

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

	// build-phase hash table: serialised key → list of build row indices
	hashTable map[int64][]buildRow
	buildDone bool

	// probe-phase state
	probeBatch *Batch
	probePos   int
	matchBuf   []joinRow // pending output rows
	matchPos   int
}

type buildRow struct {
	values []int64 // raw bits per column
	nulls  []bool
}

type joinRow struct {
	build buildRow
	probe int // row index in probe batch
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
		build:     build,
		probe:     probe,
		buildKey:  buildKeyIdx,
		probeKey:  probeKeyIdx,
		schema:    Schema{Fields: outFields},
		hashTable: make(map[int64][]buildRow),
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

		var probeIndices []int
		if pBatch.SelVec != nil {
			probeIndices = make([]int, len(pBatch.SelVec))
			for i, v := range pBatch.SelVec {
				probeIndices[i] = int(v)
			}
		} else {
			probeIndices = make([]int, n)
			for i := range probeIndices {
				probeIndices[i] = i
			}
		}

		for _, rowIdx := range probeIndices {
			pv := pBatch.Vectors[j.probeKey]
			if pv.IsNull(rowIdx) {
				continue
			}
			key := extractInt64(pv, rowIdx)
			rows, ok := j.hashTable[key]
			if !ok {
				continue
			}
			for _, br := range rows {
				j.matchBuf = append(j.matchBuf, joinRow{build: br, probe: rowIdx})
			}
		}
		j.probePos = n
	}
}

func (j *HashJoin) buildHashTable(ctx context.Context) error {
	bSchema := j.build.Schema()
	numCols := len(bSchema.Fields)
	for {
		batch, err := j.build.Next(ctx)
		if err != nil {
			return fmt.Errorf("exec: hash join: build: %w", err)
		}
		if batch == nil {
			return nil
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
			kv := batch.Vectors[j.buildKey]
			if kv.IsNull(rowIdx) {
				continue
			}
			key := extractInt64(kv, rowIdx)
			row := buildRow{values: make([]int64, numCols), nulls: make([]bool, numCols)}
			for c := 0; c < numCols; c++ {
				v := batch.Vectors[c]
				row.nulls[c] = v.IsNull(rowIdx)
				if !row.nulls[c] {
					row.values[c] = extractInt64(v, rowIdx)
				}
			}
			j.hashTable[key] = append(j.hashTable[key], row)
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

	bSchema := j.build.Schema()
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
	switch t {
	case TypeInt64:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !r.build.nulls[colIdx] {
				out.Values[i] = r.build.values[colIdx]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !r.build.nulls[colIdx] {
				out.Values[i] = float64FromBits(uint64(r.build.values[colIdx]))
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	default:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			if !r.build.nulls[colIdx] {
				out.Values[i] = r.build.values[colIdx]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	}
}

func (j *HashJoin) probeColumnFromRows(rows []joinRow, colIdx int, t DataType, n int) Vector {
	pBatch := j.probeBatch
	switch t {
	case TypeInt64:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		src := pBatch.Vectors[colIdx].(*Int64Vector)
		for i, r := range rows {
			if !src.IsNull(r.probe) {
				out.Values[i] = src.Values[r.probe]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	case TypeFloat64:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		src := pBatch.Vectors[colIdx].(*Float64Vector)
		for i, r := range rows {
			if !src.IsNull(r.probe) {
				out.Values[i] = src.Values[r.probe]
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	default:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		for i, r := range rows {
			v := pBatch.Vectors[colIdx]
			if !v.IsNull(r.probe) {
				out.Values[i] = extractInt64(v, r.probe)
				storage.SetValidBit(out.NullBitmap, i)
			}
		}
		return out
	}
}

func (j *HashJoin) Close() error {
	_ = j.build.Close()
	return j.probe.Close()
}
