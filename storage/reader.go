package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync/atomic"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// Reader opens and reads a .vxq file.
type Reader struct {
	path      string
	f         *os.File
	meta      FileMeta
	bytesRead atomic.Int64
}

// Open opens a .vxq file and parses its footer.
func Open(_ context.Context, path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, wrap("open", err)
	}
	r := &Reader{path: path, f: f}
	if err := r.readFooter(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return r, nil
}

// Meta returns the parsed file metadata.
func (r *Reader) Meta() *FileMeta { return &r.meta }

// OpenColumn returns a ColumnReader for the given (rowGroup, column) pair.
func (r *Reader) OpenColumn(_ context.Context, rg, col int) (*ColumnReader, error) {
	if rg < 0 || rg >= len(r.meta.RowGroups) {
		return nil, wrap("open column", fmt.Errorf("row group %d out of range", rg))
	}
	if col < 0 || col >= len(r.meta.Schema.Fields) {
		return nil, wrap("open column", fmt.Errorf("column %d out of range", col))
	}
	rgMeta := &r.meta.RowGroups[rg]
	colMeta := &rgMeta.Columns[col]
	field := r.meta.Schema.Fields[col]
	return &ColumnReader{
		r:       r,
		rgMeta:  rgMeta,
		colMeta: colMeta,
		field:   field,
		pos:     colMeta.SectionOffset,
		end:     colMeta.SectionOffset + colMeta.SectionLength - int64(colMeta.DictLength),
		rowsDone: 0,
		totalRows: rgMeta.NumRows,
	}, nil
}

// Close closes the underlying file.
func (r *Reader) Close() error {
	return r.f.Close()
}

// BytesRead returns the total number of bytes read from the file.
func (r *Reader) BytesRead() int64 { return r.bytesRead.Load() }

// readAt reads exactly len(buf) bytes at offset off, updating BytesRead.
func (r *Reader) readAt(off int64, buf []byte) error {
	n, err := r.f.ReadAt(buf, off)
	r.bytesRead.Add(int64(n))
	if err != nil && err != io.EOF {
		return err
	}
	if n != len(buf) {
		return io.ErrUnexpectedEOF
	}
	return nil
}

// readFooter reads and parses the footer from the end of the file.
func (r *Reader) readFooter() error {
	fi, err := r.f.Stat()
	if err != nil {
		return wrap("open: footer: stat", err)
	}
	size := fi.Size()
	if size < int64(FooterTrailerSize)+int64(len(Magic)) {
		return wrap("open: footer", ErrBadMagic)
	}

	// Read trailer: [4B CRC][8B footer_len][4B magic] = 16 bytes.
	trailer := make([]byte, FooterTrailerSize)
	if err := r.readAt(size-FooterTrailerSize, trailer); err != nil {
		return wrap("open: footer: read trailer", err)
	}
	// Validate trailing magic.
	if string(trailer[12:16]) != Magic {
		return wrap("open: footer", ErrBadMagic)
	}
	footerCRC := enc.GetUint32(trailer[0:4])
	footerLen := enc.GetInt64(trailer[4:12])
	if footerLen < 0 || footerLen > size-int64(FooterTrailerSize)-int64(len(Magic)) {
		return wrap("open: footer", ErrCorruptFooter)
	}

	// Read footer body.
	footerStart := size - int64(FooterTrailerSize) - footerLen
	body := make([]byte, footerLen)
	if err := r.readAt(footerStart, body); err != nil {
		return wrap("open: footer: read body", err)
	}
	// Validate footer CRC.
	if enc.ChecksumIEEE(body) != footerCRC {
		return wrap("open: footer", ErrChecksum)
	}

	return r.parseFooter(body)
}

// parseFooter parses the schema + row group directory from the footer body.
func (r *Reader) parseFooter(body []byte) error {
	b := body
	if len(b) < 4 {
		return wrap("open: footer: parse schema", ErrCorruptFooter)
	}
	numCols, b := enc.ReadUint32(b)
	fields := make([]Field, numCols)
	for i := uint32(0); i < numCols; i++ {
		if len(b) < 2 {
			return wrap("open: footer: parse schema", ErrCorruptFooter)
		}
		nameLen, b2 := enc.ReadUint16(b)
		b = b2
		if len(b) < int(nameLen)+3 {
			return wrap("open: footer: parse schema", ErrCorruptFooter)
		}
		name := string(b[:nameLen])
		b = b[nameLen:]
		t := DataType(b[0])
		e := Encoding(b[1])
		flags := b[2]
		b = b[3:]
		fields[i] = Field{
			Name:     name,
			Type:     t,
			Encoding: e,
			Nullable: flags&1 != 0,
		}
	}
	r.meta.Schema = Schema{Fields: fields}

	if len(b) < 4 {
		return wrap("open: footer: parse row groups", ErrCorruptFooter)
	}
	numRGs, b := enc.ReadUint32(b)
	// Per column in row group: 8+8+8+8+8+8+1+4+4 = 57 bytes
	const colMetaSize = 8 + 8 + 8 + 8 + 8 + 8 + 1 + 4 + 4 // 57 bytes
	rgs := make([]RowGroupMeta, numRGs)
	for i := uint32(0); i < numRGs; i++ {
		if len(b) < 12 {
			return wrap("open: footer: parse row group", ErrCorruptFooter)
		}
		fileOff, b2 := enc.ReadInt64(b)
		b = b2
		numRows, b2 := enc.ReadUint32(b)
		b = b2
		cols := make([]ColumnSectionMeta, numCols)
		for j := uint32(0); j < numCols; j++ {
			if len(b) < colMetaSize {
				return wrap("open: footer: parse column meta", ErrCorruptFooter)
			}
			var cm ColumnSectionMeta
			cm.SectionOffset, b = enc.ReadInt64(b)
			cm.SectionLength, b = enc.ReadInt64(b)
			cm.Stats.NullCount, b = enc.ReadInt64(b)
			cm.Stats.Sum, b = enc.ReadInt64(b)
			cm.Stats.Min, b = enc.ReadUint64(b)
			cm.Stats.Max, b = enc.ReadUint64(b)
			cm.Stats.HasMinMax = b[0] != 0
			b = b[1:]
			cm.DictOffset, b = enc.ReadUint32(b)
			cm.DictLength, b = enc.ReadUint32(b)
			cols[j] = cm
		}
		rgs[i] = RowGroupMeta{
			FileOffset: fileOff,
			NumRows:    int(numRows),
			Columns:    cols,
		}
	}
	r.meta.RowGroups = rgs
	return nil
}
