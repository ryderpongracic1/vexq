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
		vec, err := s.payloadToVector(field, nullBitmap, payload, rows, cr)
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
	cr *storage.ColumnReader,
) (Vector, error) {
	switch field.Type {
	case TypeInt64:
		vals := make([]int64, rows)
		for i := range vals {
			vals[i] = int64(binary.LittleEndian.Uint64(payload[i*8:]))
		}
		nb := make([]byte, (rows+7)/8)
		copy(nb, nullBitmap)
		return &Int64Vector{Values: vals, NullBitmap: nb}, nil

	case TypeFloat64:
		vals := make([]float64, rows)
		for i := range vals {
			bits := binary.LittleEndian.Uint64(payload[i*8:])
			vals[i] = math.Float64frombits(bits)
		}
		nb := make([]byte, (rows+7)/8)
		copy(nb, nullBitmap)
		return &Float64Vector{Values: vals, NullBitmap: nb}, nil

	case TypeDate:
		vals := make([]int32, rows)
		for i := range vals {
			vals[i] = int32(binary.LittleEndian.Uint32(payload[i*4:]))
		}
		nb := make([]byte, (rows+7)/8)
		copy(nb, nullBitmap)
		return &DateVector{Values: vals, NullBitmap: nb}, nil

	case TypeBool:
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
		codes := make([]uint32, rows)
		for i := range codes {
			codes[i] = binary.LittleEndian.Uint32(payload[i*4:])
		}
		nb := make([]byte, (rows+7)/8)
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

func (s *TableScan) Close() error {
	s.closeRowGroup()
	return s.reader.Close()
}
