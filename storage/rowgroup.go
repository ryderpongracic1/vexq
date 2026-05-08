package storage

import (
	"context"
	"fmt"
	"io"
	"sync"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// ColumnReader streams blocks from a single column section within one row group.
type ColumnReader struct {
	r         *Reader
	rgMeta    *RowGroupMeta
	colMeta   *ColumnSectionMeta
	field     Field
	pos       int64 // current read position (file-absolute)
	end       int64 // end of block data (excludes dict blob for STRING)
	rowsDone  int
	totalRows int

	// dict is lazily loaded for STRING columns.
	dictOnce sync.Once
	dict     *Dictionary
	dictErr  error
}

// NextBlock reads the next block from this column section.
//
// Returns (nullBitmap, payload, rowCount, nil) on success.
// Returns (nil, nil, 0, io.EOF) when all rows have been read.
// Returns a wrapped error for CRC failures.
//
// For Bool columns, payload is the raw RLE bytes (pass to DecodeRLEBool).
// For all other types, payload contains exactly rowCount × valueSize raw bytes.
// nullBitmap is ceil(rowCount/8) bytes, LSB-first (1=valid, 0=null).
// Bool nulls are inlined in the RLE; nullBitmap is returned as nil for Bool.
func (cr *ColumnReader) NextBlock(_ context.Context) (nullBitmap []byte, payload []byte, rows int, err error) {
	if cr.rowsDone >= cr.totalRows {
		return nil, nil, 0, io.EOF
	}
	rows = cr.totalRows - cr.rowsDone
	if rows > BlockRows {
		rows = BlockRows
	}

	switch cr.field.Type {
	case TypeBool:
		return cr.nextBoolBlock(rows)
	default:
		return cr.nextFixedBlock(rows)
	}
}

func (cr *ColumnReader) nextFixedBlock(rows int) ([]byte, []byte, int, error) {
	vs := cr.field.Type.ValueSize()
	blockBytes := NullBitmapBytes + rows*vs + 4

	buf := make([]byte, blockBytes)
	if err := cr.r.readAt(cr.pos, buf); err != nil {
		return nil, nil, 0, wrap(fmt.Sprintf("read block @%d", cr.pos), err)
	}
	cr.pos += int64(blockBytes)

	payload, verr := enc.VerifyTrailing(buf)
	if verr != nil {
		return nil, nil, 0, wrap(
			fmt.Sprintf("crc block rg=%d col=%s off=%d",
				cr.rgIdx(), cr.field.Name, cr.pos-int64(blockBytes)),
			ErrChecksum)
	}

	nullBitmap := make([]byte, (rows+7)/8)
	copy(nullBitmap, payload[:NullBitmapBytes])
	valueBytes := payload[NullBitmapBytes:]

	cr.rowsDone += rows
	return nullBitmap, valueBytes, rows, nil
}

func (cr *ColumnReader) nextBoolBlock(rows int) ([]byte, []byte, int, error) {
	// Bool blocks are variable-size RLE. We must read the run_count first,
	// then compute total block size.
	// Format: [4B run_count][run_count × 5B][4B CRC]
	// Read the header first to discover run_count.
	header := make([]byte, 4)
	if err := cr.r.readAt(cr.pos, header); err != nil {
		return nil, nil, 0, wrap("read bool block header", err)
	}
	runCount := enc.GetUint32(header)
	totalSize := 4 + int(runCount)*5 + 4
	buf := make([]byte, totalSize)
	if err := cr.r.readAt(cr.pos, buf); err != nil {
		return nil, nil, 0, wrap("read bool block", err)
	}
	cr.pos += int64(totalSize)
	cr.rowsDone += rows
	// Return raw RLE payload (caller uses DecodeRLEBool).
	return nil, buf, rows, nil
}

// Dictionary returns the parsed dictionary for a STRING column.
// It is loaded lazily on first call and cached.
func (cr *ColumnReader) Dictionary() (*Dictionary, error) {
	if cr.field.Type != TypeString {
		return nil, fmt.Errorf("storage: column %s is not STRING", cr.field.Name)
	}
	cr.dictOnce.Do(func() {
		off := cr.colMeta.SectionOffset + int64(cr.colMeta.DictOffset)
		buf := make([]byte, cr.colMeta.DictLength)
		if err := cr.r.readAt(off, buf); err != nil {
			cr.dictErr = wrap("read dictionary", err)
			return
		}
		cr.dict, cr.dictErr = UnmarshalDictionary(buf)
	})
	return cr.dict, cr.dictErr
}

// Close is a no-op; the underlying file is owned by Reader.
func (cr *ColumnReader) Close() error { return nil }

func (cr *ColumnReader) rgIdx() int {
	for i := range cr.r.meta.RowGroups {
		if &cr.r.meta.RowGroups[i] == cr.rgMeta {
			return i
		}
	}
	return -1
}
