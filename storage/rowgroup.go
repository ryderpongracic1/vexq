package storage

import (
	"context"
	"fmt"
	"io"
	"sync"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// ColumnReader streams blocks from a single column section within one row group.
//
// I/O is coarse-grained. Rather than one pread per 1024-row block, the reader
// pulls a large window of the column section into a reusable buffer (up to
// maxPrefetchBytes, which admits an entire row group for the current format)
// and then serves blocks from memory. A 6M-row, 6-column scan therefore issues
// roughly one read per (row group, column) instead of one per block — about
// 550 reads instead of ~35,000 — which matters even on a warm page cache
// because each block read was a syscall.
//
// Buffer ownership: the null bitmap and payload returned by NextBlock alias the
// reader's internal window. They are valid only until the next NextBlock or
// Close call on this ColumnReader; callers that need to retain the bytes must
// copy them. This mirrors the buffer-reuse contract exec.TableScan already
// documents for the batches it yields.
type ColumnReader struct {
	r         *Reader
	rgMeta    *RowGroupMeta
	colMeta   *ColumnSectionMeta
	field     Field
	pos       int64 // current read position (file-absolute)
	end       int64 // end of block data (excludes dict blob for STRING)
	rowsDone  int
	totalRows int

	// win buffers the file range [winStart, winStart+len(win)), always a
	// sub-range of [colMeta.SectionOffset, end). Borrowed from r.bufPool and
	// returned by Close.
	win      []byte
	winStart int64

	// dict is lazily loaded for STRING columns.
	dictOnce sync.Once
	dict     *Dictionary
	dictErr  error
}

// window returns a slice aliasing n bytes of this column's section at absolute
// file offset off, issuing a read only when the range is not already buffered.
//
// A refill always starts exactly at off, so a block is never split across two
// windows and never needs stitching.
func (cr *ColumnReader) window(off int64, n int) ([]byte, error) {
	if off < cr.colMeta.SectionOffset || off+int64(n) > cr.end {
		// The block extends past this column's data; the footer's block layout
		// disagrees with the section length.
		return nil, io.ErrUnexpectedEOF
	}
	if cr.win != nil && off >= cr.winStart {
		if lo := off - cr.winStart; lo+int64(n) <= int64(len(cr.win)) {
			return cr.win[lo : lo+int64(n)], nil
		}
	}

	want := cr.end - off
	if want > maxPrefetchBytes {
		want = maxPrefetchBytes
	}
	if want < int64(n) {
		want = int64(n)
	}

	// Release before acquiring so the outgoing window is the buffer we get back.
	if cr.win != nil {
		cr.r.releaseBuf(cr.win)
		cr.win, cr.winStart = nil, 0
	}
	buf := cr.r.acquireBuf(int(want))
	if err := cr.r.readAt(off, buf); err != nil {
		cr.r.releaseBuf(buf)
		return nil, err
	}
	cr.win, cr.winStart = buf, off
	return buf[:n], nil
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
//
// Both returned slices alias an internal buffer — see the type comment.
func (cr *ColumnReader) NextBlock(_ context.Context) (nullBitmap []byte, payload []byte, rows int, err error) {
	if cr.rowsDone >= cr.totalRows {
		return nil, nil, 0, io.EOF
	}
	rows = cr.totalRows - cr.rowsDone
	if rows > BlockRows {
		rows = BlockRows
	}
	// Account for the block up front so every error path still makes progress.
	// A caller that reports a bad block and keeps iterating (cmd/vexq fsck)
	// then reaches io.EOF instead of retrying the same block forever.
	cr.rowsDone += rows

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

	off := cr.pos
	buf, err := cr.window(off, blockBytes)
	if err != nil {
		return nil, nil, 0, wrap(fmt.Sprintf("read block @%d", off), err)
	}
	cr.pos += int64(blockBytes)

	// CRC is still verified per block, not per window: coarser reads must not
	// coarsen corruption detection.
	payload, verr := enc.VerifyTrailing(buf)
	if verr != nil {
		return nil, nil, 0, wrap(
			fmt.Sprintf("crc block rg=%d col=%s off=%d",
				cr.rgIdx(), cr.field.Name, off),
			ErrChecksum)
	}

	return payload[:(rows+7)/8], payload[NullBitmapBytes:], rows, nil
}

func (cr *ColumnReader) nextBoolBlock(rows int) ([]byte, []byte, int, error) {
	// Bool blocks are variable-size RLE, so the run count must be read before
	// the block size is known:
	//   [4B run_count][run_count × 5B][4B CRC]
	// Both reads below come out of the same buffered window, so discovering the
	// size costs no extra syscall.
	off := cr.pos
	header, err := cr.window(off, 4)
	if err != nil {
		return nil, nil, 0, wrap("read bool block header", err)
	}
	runCount := enc.GetUint32(header)
	if runCount > uint32(rows) {
		// Runs partition the block's rows, so run_count can never exceed rows.
		// Reject rather than sizing an allocation from a corrupt length.
		return nil, nil, 0, wrap("read bool block", ErrCorruptFooter)
	}
	totalSize := 4 + int(runCount)*5 + 4
	buf, err := cr.window(off, totalSize)
	if err != nil {
		return nil, nil, 0, wrap("read bool block", err)
	}
	cr.pos += int64(totalSize)
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

// Close releases this reader's section buffer back to the owning Reader's pool
// so the next row group can reuse it. The underlying file is owned by Reader.
//
// After Close, slices previously returned by NextBlock must not be read.
func (cr *ColumnReader) Close() error {
	if cr.win != nil {
		cr.r.releaseBuf(cr.win)
		cr.win, cr.winStart = nil, 0
	}
	return nil
}

func (cr *ColumnReader) rgIdx() int {
	for i := range cr.r.meta.RowGroups {
		if &cr.r.meta.RowGroups[i] == cr.rgMeta {
			return i
		}
	}
	return -1
}
