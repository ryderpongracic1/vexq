package storage

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// Writer builds a .vxq file.  Usage:
//
//	w, _ := NewWriter(path, schema)
//	w.BeginRowGroup(n)
//	w.AppendColumn(ctx, 0, nulls, []int64{...})
//	...
//	w.EndRowGroup()
//	w.Finish(ctx)
type Writer struct {
	path     string
	schema   Schema
	tmpPath  string
	f        *os.File
	bw       *bufio.Writer
	offset   int64 // bytes written so far
	meta     FileMeta
	// per-row-group state
	rgNumRows    int
	rgColWritten []bool
	rgColMeta    []ColumnSectionMeta
	rgStartOff   int64
}

// NewWriter creates a new Writer for the given path.  The file is created at
// path+".tmp" and renamed atomically on Finish.
func NewWriter(path string, schema Schema) (*Writer, error) {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return nil, wrap("new writer", err)
	}
	w := &Writer{
		path:    path,
		schema:  schema,
		tmpPath: tmp,
		f:       f,
		bw:      bufio.NewWriterSize(f, 1<<20), // 1 MB write buffer
		meta:    FileMeta{Schema: schema},
	}
	// Write file header magic.
	if _, err := w.bw.WriteString(Magic); err != nil {
		_ = f.Close()
		return nil, wrap("new writer: write magic", err)
	}
	w.offset = int64(len(Magic))
	return w, nil
}

// BeginRowGroup opens a new row group for numRows rows.
func (w *Writer) BeginRowGroup(numRows int) error {
	if w.rgNumRows != 0 {
		return wrap("begin row group", ErrRowGroupOpen)
	}
	w.rgNumRows = numRows
	w.rgColWritten = make([]bool, len(w.schema.Fields))
	w.rgColMeta = make([]ColumnSectionMeta, len(w.schema.Fields))
	w.rgStartOff = w.offset
	return nil
}

// AppendColumn writes all data for column idx in the current row group.
// values must be one of []int64, []float64, []bool, []string, []int32.
// nulls is a bitmap (1=valid, 0=null, LSB-first) of length ceil(numRows/8);
// pass nil to mark all rows as valid.
func (w *Writer) AppendColumn(ctx context.Context, idx int, nulls []byte, values any) error {
	if err := ctx.Err(); err != nil {
		return wrap("append column", err)
	}
	if w.rgNumRows == 0 {
		return wrap("append column", ErrRowGroupOpen)
	}
	if idx < 0 || idx >= len(w.schema.Fields) {
		return wrap("append column", fmt.Errorf("index %d out of range", idx))
	}
	if w.rgColWritten[idx] {
		return wrap("append column", ErrColumnComplete)
	}
	field := w.schema.Fields[idx]

	// Validate type.
	if err := checkValueType(field.Type, values); err != nil {
		return wrap("append column", err)
	}

	// Build a default all-valid bitmap if nil.
	if nulls == nil {
		nulls = FullBitmap(w.rgNumRows)
	}

	sectionStart := w.offset
	var stats ZoneMap
	var dictOff, dictLen uint32

	switch field.Type {
	case TypeInt64:
		vs := values.([]int64)
		stats = w.writeInt64Blocks(nulls, vs)
	case TypeFloat64:
		vs := values.([]float64)
		stats = w.writeFloat64Blocks(nulls, vs)
	case TypeDate:
		vs := values.([]int32)
		stats = w.writeDateBlocks(nulls, vs)
	case TypeBool:
		vs := values.([]bool)
		stats = w.writeBoolBlocks(nulls, vs)
	case TypeString:
		vs := values.([]string)
		var dictBytes []byte
		stats, dictBytes = w.writeStringBlocks(nulls, vs)
		dictOff = uint32(w.offset - sectionStart)
		w.mustWrite(dictBytes)
		dictLen = uint32(len(dictBytes))
	default:
		return wrap("append column", fmt.Errorf("unsupported type %v", field.Type))
	}

	sectionLen := w.offset - sectionStart
	w.rgColMeta[idx] = ColumnSectionMeta{
		SectionOffset: sectionStart,
		SectionLength: sectionLen,
		Stats:         stats,
		DictOffset:    dictOff,
		DictLength:    dictLen,
	}
	w.rgColWritten[idx] = true
	return nil
}

// EndRowGroup closes the current row group.
func (w *Writer) EndRowGroup() error {
	if w.rgNumRows == 0 {
		return wrap("end row group", ErrRowGroupOpen)
	}
	for i, done := range w.rgColWritten {
		if !done {
			return wrap("end row group", fmt.Errorf("%w: column %d (%s) not written",
				ErrMissingColumns, i, w.schema.Fields[i].Name))
		}
	}
	w.meta.RowGroups = append(w.meta.RowGroups, RowGroupMeta{
		FileOffset: w.rgStartOff,
		NumRows:    w.rgNumRows,
		Columns:    w.rgColMeta,
	})
	w.rgNumRows = 0
	w.rgColWritten = nil
	w.rgColMeta = nil
	return nil
}

// Finish flushes the footer, syncs, closes, and renames to the final path.
func (w *Writer) Finish(_ context.Context) error {
	if w.rgNumRows != 0 {
		return wrap("finish", ErrRowGroupOpen)
	}
	footer := w.marshalFooter()
	w.mustWrite(footer)
	if err := w.bw.Flush(); err != nil {
		return wrap("finish: flush", err)
	}
	if err := w.f.Sync(); err != nil {
		return wrap("finish: sync", err)
	}
	if err := w.f.Close(); err != nil {
		return wrap("finish: close", err)
	}
	if err := os.Rename(w.tmpPath, w.path); err != nil {
		return wrap("finish: rename", err)
	}
	return nil
}

// Abort discards the partially-written file and cleans up the temp file.
func (w *Writer) Abort() error {
	_ = w.f.Close()
	return os.Remove(w.tmpPath)
}

// BytesWritten returns the number of bytes written to the underlying file so far.
func (w *Writer) BytesWritten() int64 { return w.offset }

// ---- internal helpers -------------------------------------------------------

func (w *Writer) mustWrite(b []byte) {
	n, err := w.bw.Write(b)
	if err != nil {
		panic(fmt.Sprintf("storage writer: write error: %v", err))
	}
	w.offset += int64(n)
}

func (w *Writer) writeInt64Blocks(nullBitmap []byte, values []int64) ZoneMap {
	n := len(values)
	stats := ZoneMap{HasMinMax: false}
	for blockStart := 0; blockStart < n; blockStart += BlockRows {
		end := blockStart + BlockRows
		if end > n {
			end = n
		}
		rows := end - blockStart
		blockNulls := subBitmap(nullBitmap, blockStart, rows)
		blockVals := values[blockStart:end]

		// Build payload: null bitmap + values.
		buf := make([]byte, 0, NullBitmapBytes+rows*8+4)
		bitmapFull := make([]byte, NullBitmapBytes)
		copy(bitmapFull, blockNulls)
		buf = append(buf, bitmapFull...)
		for _, v := range blockVals {
			buf = enc.AppendInt64(buf, v)
		}
		buf = enc.AppendChecksum(buf)
		w.mustWrite(buf)

		// Update zone map stats.
		for i, v := range blockVals {
			if IsNullBit(blockNulls, i) {
				stats.NullCount++
				continue
			}
			stats.Sum += v
			uv := uint64(v)
			if !stats.HasMinMax {
				stats.Min = uv
				stats.Max = uv
				stats.HasMinMax = true
			} else {
				if v < int64(stats.Min) {
					stats.Min = uv
				}
				if v > int64(stats.Max) {
					stats.Max = uv
				}
			}
		}
	}
	return stats
}

func (w *Writer) writeFloat64Blocks(nullBitmap []byte, values []float64) ZoneMap {
	n := len(values)
	stats := ZoneMap{HasMinMax: false}
	var sumF float64
	var minF, maxF float64
	for blockStart := 0; blockStart < n; blockStart += BlockRows {
		end := blockStart + BlockRows
		if end > n {
			end = n
		}
		rows := end - blockStart
		blockNulls := subBitmap(nullBitmap, blockStart, rows)
		blockVals := values[blockStart:end]

		buf := make([]byte, 0, NullBitmapBytes+rows*8+4)
		bitmapFull := make([]byte, NullBitmapBytes)
		copy(bitmapFull, blockNulls)
		buf = append(buf, bitmapFull...)
		for _, v := range blockVals {
			buf = enc.AppendFloat64(buf, v)
		}
		buf = enc.AppendChecksum(buf)
		w.mustWrite(buf)

		for i, v := range blockVals {
			if IsNullBit(blockNulls, i) {
				stats.NullCount++
				continue
			}
			sumF += v
			if !stats.HasMinMax {
				minF = v
				maxF = v
				stats.HasMinMax = true
			} else {
				if v < minF {
					minF = v
				}
				if v > maxF {
					maxF = v
				}
			}
		}
	}
	stats.Sum = int64(math.Float64bits(sumF))
	if stats.HasMinMax {
		stats.Min = math.Float64bits(minF)
		stats.Max = math.Float64bits(maxF)
	}
	return stats
}

func (w *Writer) writeDateBlocks(nullBitmap []byte, values []int32) ZoneMap {
	n := len(values)
	stats := ZoneMap{HasMinMax: false}
	for blockStart := 0; blockStart < n; blockStart += BlockRows {
		end := blockStart + BlockRows
		if end > n {
			end = n
		}
		rows := end - blockStart
		blockNulls := subBitmap(nullBitmap, blockStart, rows)
		blockVals := values[blockStart:end]

		buf := make([]byte, 0, NullBitmapBytes+rows*4+4)
		bitmapFull := make([]byte, NullBitmapBytes)
		copy(bitmapFull, blockNulls)
		buf = append(buf, bitmapFull...)
		for _, v := range blockVals {
			buf = enc.AppendInt32(buf, v)
		}
		buf = enc.AppendChecksum(buf)
		w.mustWrite(buf)

		for i, v := range blockVals {
			if IsNullBit(blockNulls, i) {
				stats.NullCount++
				continue
			}
			uv := uint64(uint32(v))
			if !stats.HasMinMax {
				stats.Min = uv
				stats.Max = uv
				stats.HasMinMax = true
			} else {
				if int32(stats.Min) > v {
					stats.Min = uv
				}
				if int32(stats.Max) < v {
					stats.Max = uv
				}
			}
		}
	}
	return stats
}

func (w *Writer) writeBoolBlocks(nullBitmap []byte, values []bool) ZoneMap {
	n := len(values)
	stats := ZoneMap{}
	for blockStart := 0; blockStart < n; blockStart += BlockRows {
		end := blockStart + BlockRows
		if end > n {
			end = n
		}
		rows := end - blockStart
		blockNulls := subBitmap(nullBitmap, blockStart, rows)
		blockVals := values[blockStart:end]

		block := EncodeRLEBool(blockVals, blockNulls)
		w.mustWrite(block)

		for i := range blockVals {
			if IsNullBit(blockNulls, i) {
				stats.NullCount++
			}
		}
	}
	return stats
}

func (w *Writer) writeStringBlocks(nullBitmap []byte, values []string) (ZoneMap, []byte) {
	n := len(values)
	stats := ZoneMap{HasMinMax: false}
	db := NewDictBuilder()

	// Build the full dictionary first (scan all values once).
	codes := make([]uint32, n)
	for i, s := range values {
		if !IsNullBit(nullBitmap, i) {
			codes[i] = db.Add(s)
		} else {
			codes[i] = 0xFFFFFFFF
			stats.NullCount++
		}
	}

	// Update zone map using dict codes (lex-order approximation via insertion order).
	if db.Len() > 0 {
		stats.HasMinMax = true
		stats.Min = 0
		stats.Max = uint64(db.Len() - 1)
	}

	// Write code blocks.
	for blockStart := 0; blockStart < n; blockStart += BlockRows {
		end := blockStart + BlockRows
		if end > n {
			end = n
		}
		rows := end - blockStart
		blockNulls := subBitmap(nullBitmap, blockStart, rows)
		blockCodes := codes[blockStart:end]

		buf := make([]byte, 0, NullBitmapBytes+rows*4+4)
		bitmapFull := make([]byte, NullBitmapBytes)
		copy(bitmapFull, blockNulls)
		buf = append(buf, bitmapFull...)
		for _, c := range blockCodes {
			buf = enc.AppendUint32(buf, c)
		}
		buf = enc.AppendChecksum(buf)
		w.mustWrite(buf)
	}

	return stats, db.Marshal()
}

// marshalFooter serialises the FileMeta as a footer blob with trailing CRC,
// length, and magic.
func (w *Writer) marshalFooter() []byte {
	var body []byte

	// Schema directory.
	body = enc.AppendUint32(body, uint32(len(w.schema.Fields)))
	for _, f := range w.schema.Fields {
		name := []byte(f.Name)
		body = enc.AppendUint16(body, uint16(len(name)))
		body = append(body, name...)
		body = append(body, byte(f.Type))
		body = append(body, byte(f.Encoding))
		flags := byte(0)
		if f.Nullable {
			flags |= 1
		}
		body = append(body, flags)
	}

	// Row group directory.
	body = enc.AppendUint32(body, uint32(len(w.meta.RowGroups)))
	for _, rg := range w.meta.RowGroups {
		body = enc.AppendInt64(body, rg.FileOffset)
		body = enc.AppendUint32(body, uint32(rg.NumRows))
		for _, col := range rg.Columns {
			body = enc.AppendInt64(body, col.SectionOffset)
			body = enc.AppendInt64(body, col.SectionLength)
			body = enc.AppendInt64(body, col.Stats.NullCount)
			body = enc.AppendInt64(body, col.Stats.Sum)
			body = enc.AppendUint64(body, col.Stats.Min)
			body = enc.AppendUint64(body, col.Stats.Max)
			hasMinMax := byte(0)
			if col.Stats.HasMinMax {
				hasMinMax = 1
			}
			body = append(body, hasMinMax)
			body = enc.AppendUint32(body, col.DictOffset)
			body = enc.AppendUint32(body, col.DictLength)
		}
	}

	// Trailer: [4B CRC][8B body length][4B magic]
	crc := enc.ChecksumIEEE(body)
	var trailer []byte
	trailer = enc.AppendUint32(trailer, crc)
	trailer = enc.AppendInt64(trailer, int64(len(body)))
	trailer = append(trailer, Magic...)

	return append(body, trailer...)
}

// subBitmap extracts a sub-bitmap for rows [start, start+rows).
func subBitmap(bitmap []byte, start, rows int) []byte {
	size := (rows + 7) / 8
	sub := make([]byte, size)
	for i := 0; i < rows; i++ {
		if !IsNullBit(bitmap, start+i) {
			SetValidBit(sub, i) // valid in source → set valid in sub
		}
		// null stays null (bit=0, default)
	}
	return sub
}

// checkValueType validates that values matches the expected DataType.
func checkValueType(t DataType, values any) error {
	switch t {
	case TypeInt64:
		if _, ok := values.([]int64); !ok {
			return fmt.Errorf("%w: expected []int64 for INT64", ErrTypeMismatch)
		}
	case TypeFloat64:
		if _, ok := values.([]float64); !ok {
			return fmt.Errorf("%w: expected []float64 for FLOAT64", ErrTypeMismatch)
		}
	case TypeDate:
		if _, ok := values.([]int32); !ok {
			return fmt.Errorf("%w: expected []int32 for DATE", ErrTypeMismatch)
		}
	case TypeBool:
		if _, ok := values.([]bool); !ok {
			return fmt.Errorf("%w: expected []bool for BOOL", ErrTypeMismatch)
		}
	case TypeString:
		if _, ok := values.([]string); !ok {
			return fmt.Errorf("%w: expected []string for STRING", ErrTypeMismatch)
		}
	}
	return nil
}
