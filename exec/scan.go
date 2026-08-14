package exec

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/ryderpongracic1/vexq/storage"
)

// TableScan reads a .vxq file and yields Batches.  It supports:
//   - Column pruning: only reads columns listed in projectedCols.
//   - Zone map pruning: skips entire row groups via ZonePred.
//   - Row-group range: only scans [rgStart, rgEnd) for morsel-driven parallelism.
//
// Buffer ownership contract: TableScan reuses internal decode buffers across
// blocks. Each call to Next() overwrites the previous batch's vector data.
// Callers MUST fully consume or copy batch data before calling Next() again.
// This contract is satisfied by:
//   - Filter: creates a SelVec referencing the same batch (no retained refs).
//   - Project: materializes new vectors from Eval results (copies selected data).
//   - HashAggregate: extracts scalar values into accumulator maps (no retained refs).
//
// This design eliminates ~35K allocations per full scan (6 cols × 64 blocks/rg
// × 92 rg), reducing GC pressure from ~238 cycles to near-zero for parallel
// morsel-driven execution.
//
// I/O granularity: storage.ColumnReader buffers a whole column section per row
// group, so opening a row group costs about one read per projected column
// rather than one per 1024-row block. Pruned row groups are skipped before any
// ColumnReader is opened, so they still cost zero reads.
type TableScan struct {
	reader   *storage.Reader
	schema   Schema // output schema (projected columns only)
	colMap   []int  // colMap[i] = source column index for output column i
	zonePred ZonePredicate
	rgStart  int // inclusive
	rgEnd    int // exclusive
	rgIdx    int
	crList   []*storage.ColumnReader
	rgDone   bool

	// Reusable decode buffers — allocated once per column, reused across blocks.
	// Each entry corresponds to a projected column (same index as colMap/schema).
	bufs *scanBuffers
}

// scanBuffers holds reusable decode buffers for a TableScan. One buffer set
// per projected column, allocated on first use and reused across all subsequent
// blocks. This eliminates per-block allocations in the decode hot path.
type scanBuffers struct {
	int64Bufs   [][]int64   // one per projected INT64 column
	float64Bufs [][]float64 // one per projected FLOAT64 column
	int32Bufs   [][]int32   // one per projected DATE column
	uint32Bufs  [][]uint32  // one per projected STRING column (dict codes)
	nullBufs    [][]byte    // one per projected column (null bitmap)

	// colBufIdx maps each projected column to its type-specific buffer index.
	// e.g., if col 0 is INT64, col 1 is FLOAT64, col 2 is INT64:
	//   colBufIdx = [0, 0, 1] (col0 → int64Bufs[0], col1 → float64Bufs[0], col2 → int64Bufs[1])
	colBufIdx []int
}

// initScanBuffers allocates the buffer structure for the given schema.
// The actual slice capacity is allocated lazily on first use (when we know rows).
func initScanBuffers(schema Schema) *scanBuffers {
	sb := &scanBuffers{
		colBufIdx: make([]int, len(schema.Fields)),
	}
	var nInt64, nFloat64, nInt32, nUint32 int
	for i, f := range schema.Fields {
		switch f.Type {
		case TypeInt64:
			sb.colBufIdx[i] = nInt64
			nInt64++
		case TypeFloat64:
			sb.colBufIdx[i] = nFloat64
			nFloat64++
		case TypeDate:
			sb.colBufIdx[i] = nInt32
			nInt32++
		case TypeString:
			sb.colBufIdx[i] = nUint32
			nUint32++
		default:
			sb.colBufIdx[i] = -1 // bool and others: not pooled
		}
	}
	sb.int64Bufs = make([][]int64, nInt64)
	sb.float64Bufs = make([][]float64, nFloat64)
	sb.int32Bufs = make([][]int32, nInt32)
	sb.uint32Bufs = make([][]uint32, nUint32)
	sb.nullBufs = make([][]byte, len(schema.Fields))
	return sb
}

// getInt64Buf returns a reusable int64 buffer of exactly `rows` elements.
func (sb *scanBuffers) getInt64Buf(colIdx, rows int) []int64 {
	idx := sb.colBufIdx[colIdx]
	if cap(sb.int64Bufs[idx]) >= rows {
		return sb.int64Bufs[idx][:rows]
	}
	sb.int64Bufs[idx] = make([]int64, rows)
	return sb.int64Bufs[idx]
}

// getFloat64Buf returns a reusable float64 buffer of exactly `rows` elements.
func (sb *scanBuffers) getFloat64Buf(colIdx, rows int) []float64 {
	idx := sb.colBufIdx[colIdx]
	if cap(sb.float64Bufs[idx]) >= rows {
		return sb.float64Bufs[idx][:rows]
	}
	sb.float64Bufs[idx] = make([]float64, rows)
	return sb.float64Bufs[idx]
}

// getInt32Buf returns a reusable int32 buffer of exactly `rows` elements.
func (sb *scanBuffers) getInt32Buf(colIdx, rows int) []int32 {
	idx := sb.colBufIdx[colIdx]
	if cap(sb.int32Bufs[idx]) >= rows {
		return sb.int32Bufs[idx][:rows]
	}
	sb.int32Bufs[idx] = make([]int32, rows)
	return sb.int32Bufs[idx]
}

// getUint32Buf returns a reusable uint32 buffer of exactly `rows` elements.
func (sb *scanBuffers) getUint32Buf(colIdx, rows int) []uint32 {
	idx := sb.colBufIdx[colIdx]
	if cap(sb.uint32Bufs[idx]) >= rows {
		return sb.uint32Bufs[idx][:rows]
	}
	sb.uint32Bufs[idx] = make([]uint32, rows)
	return sb.uint32Bufs[idx]
}

// getNullBuf returns a reusable null bitmap buffer for the given column.
func (sb *scanBuffers) getNullBuf(colIdx, rows int) []byte {
	need := (rows + 7) / 8
	if cap(sb.nullBufs[colIdx]) >= need {
		return sb.nullBufs[colIdx][:need]
	}
	sb.nullBufs[colIdx] = make([]byte, need)
	return sb.nullBufs[colIdx]
}

// ZonePredicate is called with a row group's column stats before reading it.
// Return false to skip the row group entirely.
type ZonePredicate func(rg *storage.RowGroupMeta) bool

// NewTableScan creates a TableScan that covers all row groups.
//   - reader: open VXQReader (TableScan takes ownership; Close closes it)
//   - projectedCols: column names to project (nil = all columns)
//   - zonePred: optional zone-map predicate (nil = scan all row groups)
func NewTableScan(reader *storage.Reader, projectedCols []string, zonePred ZonePredicate) (*TableScan, error) {
	return NewTableScanRange(reader, projectedCols, zonePred, 0, len(reader.Meta().RowGroups))
}

// NewTableScanRange creates a TableScan limited to row groups [rgStart, rgEnd).
// Used by morsel-driven parallel execution to assign disjoint row-group slices
// to independent goroutines.
func NewTableScanRange(reader *storage.Reader, projectedCols []string, zonePred ZonePredicate, rgStart, rgEnd int) (*TableScan, error) {
	srcSchema := reader.Meta().Schema
	var outFields []Field
	var colMap []int

	if len(projectedCols) == 0 {
		// All columns.
		for i, f := range srcSchema.Fields {
			outFields = append(outFields, f)
			colMap = append(colMap, i)
		}
	} else {
		for _, name := range projectedCols {
			idx := srcSchema.IndexOf(name)
			if idx < 0 {
				return nil, fmt.Errorf("exec: scan: column %q not found", name)
			}
			outFields = append(outFields, srcSchema.Fields[idx])
			colMap = append(colMap, idx)
		}
	}

	return &TableScan{
		reader:   reader,
		schema:   Schema{Fields: outFields},
		colMap:   colMap,
		zonePred: zonePred,
		rgStart:  rgStart,
		rgEnd:    rgEnd,
		rgIdx:    rgStart,
	}, nil
}

func (s *TableScan) Schema() Schema { return s.schema }

func (s *TableScan) Next(ctx context.Context) (*Batch, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("exec: scan: %w", err)
		}

		// Open the current row group if needed.
		if s.crList == nil {
			if s.rgIdx >= s.rgEnd {
				return nil, nil // EOF
			}
			rg := &s.reader.Meta().RowGroups[s.rgIdx]
			// Zone-map pruning.
			if s.zonePred != nil && !s.zonePred(rg) {
				s.rgIdx++
				continue
			}
			var err error
			s.crList, err = s.openRowGroup(ctx, s.rgIdx)
			if err != nil {
				return nil, err
			}
			s.rgDone = false
		}

		// Read one block from each column reader.
		batch, done, err := s.readBlock(ctx)
		if err != nil {
			return nil, err
		}
		if done {
			s.closeRowGroup()
			s.rgIdx++
			continue
		}
		return batch, nil
	}
}

func (s *TableScan) openRowGroup(ctx context.Context, rgIdx int) ([]*storage.ColumnReader, error) {
	crs := make([]*storage.ColumnReader, len(s.colMap))
	for i, srcCol := range s.colMap {
		cr, err := s.reader.OpenColumn(ctx, rgIdx, srcCol)
		if err != nil {
			// Close already-opened readers.
			for j := 0; j < i; j++ {
				_ = crs[j].Close()
			}
			return nil, fmt.Errorf("exec: scan: open column %d: %w", srcCol, err)
		}
		crs[i] = cr
	}
	return crs, nil
}

func (s *TableScan) readBlock(ctx context.Context) (*Batch, bool, error) {
	if len(s.crList) == 0 {
		return nil, true, nil
	}

	// Lazily initialize reusable decode buffers on first block read.
	if s.bufs == nil {
		s.bufs = initScanBuffers(s.schema)
	}

	vectors := make([]Vector, len(s.crList))
	var batchLen int

	for i, cr := range s.crList {
		nullBitmap, payload, rows, err := cr.NextBlock(ctx)
		if err == io.EOF {
			return nil, true, nil
		}
		if err != nil {
			return nil, false, fmt.Errorf("exec: scan: read block: %w", err)
		}
		if i == 0 {
			batchLen = rows
		}

		field := s.schema.Fields[i]
		vec, err := s.payloadToVector(field, nullBitmap, payload, rows, cr, i)
		if err != nil {
			return nil, false, err
		}
		vectors[i] = vec
	}

	return &Batch{
		Schema:  s.schema,
		Vectors: vectors,
		Length:  batchLen,
	}, false, nil
}

func (s *TableScan) payloadToVector(
	field Field, nullBitmap, payload []byte, rows int,
	cr *storage.ColumnReader, colIdx int,
) (Vector, error) {
	switch field.Type {
	case TypeInt64:
		vals := s.bufs.getInt64Buf(colIdx, rows)
		// Reslice upfront: the compiler proves all reads are in-bounds and
		// eliminates per-iteration bounds checks. Each 8-element iteration
		// loads exactly 64 bytes — one L1 cache line — with 8 independent
		// MOVs the CPU can pipeline (no carried dependency chain).
		p8 := payload[:rows*8]
		i := 0
		for ; i+8 <= rows; i += 8 {
			vals[i+0] = int64(binary.LittleEndian.Uint64(p8[(i+0)*8:]))
			vals[i+1] = int64(binary.LittleEndian.Uint64(p8[(i+1)*8:]))
			vals[i+2] = int64(binary.LittleEndian.Uint64(p8[(i+2)*8:]))
			vals[i+3] = int64(binary.LittleEndian.Uint64(p8[(i+3)*8:]))
			vals[i+4] = int64(binary.LittleEndian.Uint64(p8[(i+4)*8:]))
			vals[i+5] = int64(binary.LittleEndian.Uint64(p8[(i+5)*8:]))
			vals[i+6] = int64(binary.LittleEndian.Uint64(p8[(i+6)*8:]))
			vals[i+7] = int64(binary.LittleEndian.Uint64(p8[(i+7)*8:]))
		}
		for ; i < rows; i++ {
			vals[i] = int64(binary.LittleEndian.Uint64(p8[i*8:]))
		}
		nb := s.bufs.getNullBuf(colIdx, rows)
		copy(nb, nullBitmap)
		return &Int64Vector{Values: vals, NullBitmap: nb}, nil

	case TypeFloat64:
		vals := s.bufs.getFloat64Buf(colIdx, rows)
		// Same bounds-check elimination and cache-line stride as TypeInt64.
		p8 := payload[:rows*8]
		i := 0
		for ; i+8 <= rows; i += 8 {
			vals[i+0] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+0)*8:]))
			vals[i+1] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+1)*8:]))
			vals[i+2] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+2)*8:]))
			vals[i+3] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+3)*8:]))
			vals[i+4] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+4)*8:]))
			vals[i+5] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+5)*8:]))
			vals[i+6] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+6)*8:]))
			vals[i+7] = math.Float64frombits(binary.LittleEndian.Uint64(p8[(i+7)*8:]))
		}
		for ; i < rows; i++ {
			vals[i] = math.Float64frombits(binary.LittleEndian.Uint64(p8[i*8:]))
		}
		nb := s.bufs.getNullBuf(colIdx, rows)
		copy(nb, nullBitmap)
		return &Float64Vector{Values: vals, NullBitmap: nb}, nil

	case TypeDate:
		vals := s.bufs.getInt32Buf(colIdx, rows)
		// 4-byte values: unroll 16 per iteration to fill one 64-byte cache line.
		p4 := payload[:rows*4]
		i := 0
		for ; i+16 <= rows; i += 16 {
			vals[i+0] = int32(binary.LittleEndian.Uint32(p4[(i+0)*4:]))
			vals[i+1] = int32(binary.LittleEndian.Uint32(p4[(i+1)*4:]))
			vals[i+2] = int32(binary.LittleEndian.Uint32(p4[(i+2)*4:]))
			vals[i+3] = int32(binary.LittleEndian.Uint32(p4[(i+3)*4:]))
			vals[i+4] = int32(binary.LittleEndian.Uint32(p4[(i+4)*4:]))
			vals[i+5] = int32(binary.LittleEndian.Uint32(p4[(i+5)*4:]))
			vals[i+6] = int32(binary.LittleEndian.Uint32(p4[(i+6)*4:]))
			vals[i+7] = int32(binary.LittleEndian.Uint32(p4[(i+7)*4:]))
			vals[i+8] = int32(binary.LittleEndian.Uint32(p4[(i+8)*4:]))
			vals[i+9] = int32(binary.LittleEndian.Uint32(p4[(i+9)*4:]))
			vals[i+10] = int32(binary.LittleEndian.Uint32(p4[(i+10)*4:]))
			vals[i+11] = int32(binary.LittleEndian.Uint32(p4[(i+11)*4:]))
			vals[i+12] = int32(binary.LittleEndian.Uint32(p4[(i+12)*4:]))
			vals[i+13] = int32(binary.LittleEndian.Uint32(p4[(i+13)*4:]))
			vals[i+14] = int32(binary.LittleEndian.Uint32(p4[(i+14)*4:]))
			vals[i+15] = int32(binary.LittleEndian.Uint32(p4[(i+15)*4:]))
		}
		for ; i < rows; i++ {
			vals[i] = int32(binary.LittleEndian.Uint32(p4[i*4:]))
		}
		nb := s.bufs.getNullBuf(colIdx, rows)
		copy(nb, nullBitmap)
		return &DateVector{Values: vals, NullBitmap: nb}, nil

	case TypeBool:
		// Bool uses RLE decoding — not pooled (variable-length output).
		vals, nulls, _, err := storage.DecodeRLEBool(payload)
		if err != nil {
			return nil, fmt.Errorf("exec: scan: decode bool: %w", err)
		}
		bv := &BoolVector{
			Bits:       make([]byte, (rows+7)/8),
			NullBitmap: make([]byte, (rows+7)/8),
			Length:     rows,
		}
		copy(bv.NullBitmap, nulls)
		for i, v := range vals {
			bv.Set(i, v)
		}
		return bv, nil

	case TypeString:
		codes := s.bufs.getUint32Buf(colIdx, rows)
		for i := range codes {
			codes[i] = binary.LittleEndian.Uint32(payload[i*4:])
		}
		nb := s.bufs.getNullBuf(colIdx, rows)
		copy(nb, nullBitmap)
		dict, err := cr.Dictionary()
		if err != nil {
			return nil, fmt.Errorf("exec: scan: load dict: %w", err)
		}
		return &StringVector{Codes: codes, Dict: dict, NullBitmap: nb}, nil

	default:
		return nil, fmt.Errorf("exec: scan: unsupported type %v", field.Type)
	}
}

func (s *TableScan) closeRowGroup() {
	for _, cr := range s.crList {
		if cr != nil {
			_ = cr.Close()
		}
	}
	s.crList = nil
}

// Reset repositions the scan to a new row-group range without reopening the
// underlying storage file. Used by the work-stealing morsel scheduler so a
// single open Reader can serve multiple morsel iterations.
func (s *TableScan) Reset(rgStart, rgEnd int) {
	s.closeRowGroup()
	s.rgStart = rgStart
	s.rgEnd = rgEnd
	s.rgIdx = rgStart
}

func (s *TableScan) Close() error {
	s.closeRowGroup()
	return s.reader.Close()
}
