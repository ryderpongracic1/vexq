package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// maxPrefetchBytes caps a single coarse-grained section read.
//
// A column section for a full row group of 8-byte values is 64 blocks ×
// (128 B null bitmap + 8 KiB values + 4 B CRC) ≈ 520 KiB, so this comfortably
// admits one read per (row group, column) for the current format while
// bounding the buffer a corrupt or future wide-row-group file could force us
// to allocate. Sections larger than this are read in successive windows.
const maxPrefetchBytes = 4 << 20 // 4 MiB

// maxPooledBufs bounds the per-Reader section-buffer free list. A scan holds one
// buffer per concurrently open column, so this admits the widest table the
// format supports being projected in full while keeping retained memory
// bounded. Windows beyond the bound go to the shared pool rather than being
// dropped.
const maxPooledBufs = 32

// bufClassBytes is the size-class granularity of the shared section-buffer pool
// (see sharedBufPool). Window capacities are always an exact multiple of it, so
// the class a buffer is filed under is the same class a later request of that
// size looks in — without this, a 520 KiB buffer and a 520 KiB request would
// round to different classes and the pool would miss every time.
//
// 64 KiB keeps the round-up overhead small: the largest section the current
// format produces is a full row group of 8-byte values, 64 × (128 B bitmap +
// 8 KiB values + 4 B CRC) = 520.25 KiB, which rounds to 576 KiB (+10.7%). A
// 4-byte-value section is 264.25 KiB and rounds to 320 KiB, so the two land in
// different classes and neither is served from an oversized buffer.
const bufClassBytes = 64 << 10

// numBufClasses covers window sizes in (0, maxPrefetchBytes].
const numBufClasses = maxPrefetchBytes / bufClassBytes // 64

// sharedBufPool holds free section windows by size class so a window outlives
// the Reader that allocated it.
//
// Why process-global rather than per-Reader: the morsel scheduler creates one
// Reader per morsel — exec.PipelineFactory opens the file for each claimed
// row-group range and closes it when that morsel finishes — so a per-Reader
// free list is only ever warm within one morsel. In the parallel path a morsel
// is a single row group, so every column's first (and only) window acquire
// missed a cold list and allocated: ~180 MB per Q6-shaped parallel op and
// ~200 MB per Q1-shaped op. Pooling at process scope instead of Reader scope is
// what lets those windows survive Reader churn.
//
// sync.Pool rather than a mutex-guarded slice: Readers on different workers
// acquire concurrently, and sync.Pool's per-P fast path keeps that off a shared
// lock. It also bounds retention for free — entries are dropped by the GC, so an
// idle process does not pin megabytes of read buffers. Entries are *[]byte so a
// Put does not box a slice header into an interface.
var sharedBufPool [numBufClasses + 1]sync.Pool

// sectionBufAllocs and sectionBufReuses count section windows freshly allocated
// versus served from a pool, process-wide. They exist so tests can pin the
// pooling invariant directly — the per-Reader pool silently stopped delivering
// reuse under the parallel path once pipelines became per-morsel, and only an
// allocation profile revealed it. See SectionBufferStats.
var (
	sectionBufAllocs atomic.Int64
	sectionBufReuses atomic.Int64
)

// SectionBufferStats reports how many coarse-grained section windows have been
// allocated versus served from a pool since process start.
//
// It plays the same diagnostic role for buffer reuse that (*Reader).ReadOps
// plays for read syscalls: a steady-state scan should reuse windows and allocate
// only while the pool warms up. Counters are process-wide because the pool is.
func SectionBufferStats() (allocs, reuses int64) {
	return sectionBufAllocs.Load(), sectionBufReuses.Load()
}

// bufClassOf returns the size class that can serve a request of n bytes, or 0
// when n does not fit any class (n <= 0, or a block larger than
// maxPrefetchBytes, which the current format cannot produce).
func bufClassOf(n int) int {
	if n <= 0 || n > maxPrefetchBytes {
		return 0
	}
	return (n + bufClassBytes - 1) / bufClassBytes
}

// poolClassOf returns the class a buffer of the given capacity may be filed
// under, or 0 when it must not be pooled.
//
// Only exact class multiples are poolable. That is what makes bufClassOf's
// round-up safe: every buffer in class c has capacity exactly c*bufClassBytes,
// so a Get for class c never has to reject an entry as too small, and no entry
// is larger than the class it is filed under. Buffers from the unpoolable path
// in getSharedBuf are sized to their request, so they fail this test and are
// correctly dropped rather than pinning more than maxPrefetchBytes.
func poolClassOf(capacity int) int {
	if capacity <= 0 || capacity > maxPrefetchBytes || capacity%bufClassBytes != 0 {
		return 0
	}
	return capacity / bufClassBytes
}

// getSharedBuf returns a buffer of length n from the shared pool, allocating one
// sized to its whole size class on a miss.
func getSharedBuf(n int) []byte {
	c := bufClassOf(n)
	if c == 0 {
		// Unpoolable size: allocate exactly and never file it, so an outlier
		// cannot pin a buffer larger than the largest class.
		sectionBufAllocs.Add(1)
		return make([]byte, n)
	}
	if v := sharedBufPool[c].Get(); v != nil {
		if b := *(v.(*[]byte)); cap(b) >= n {
			sectionBufReuses.Add(1)
			return b[:n]
		}
	}
	sectionBufAllocs.Add(1)
	return make([]byte, n, c*bufClassBytes)
}

// putSharedBuf files a buffer for reuse, dropping it if it is not exactly one
// size class wide.
func putSharedBuf(b []byte) {
	c := poolClassOf(cap(b))
	if c == 0 {
		return
	}
	b = b[:cap(b)]
	sharedBufPool[c].Put(&b)
}

// Reader opens and reads a .vxq file.
//
// A Reader and the ColumnReaders derived from it are single-goroutine objects:
// their free list of section windows (see acquireBuf) is unsynchronised.
// Parallel execution gives each worker its own Reader over the same path, so
// workers never share that list. The shared pool those windows are drawn from
// and returned to is synchronised — see sharedBufPool.
type Reader struct {
	path      string
	f         *os.File
	meta      FileMeta
	bytesRead atomic.Int64
	readOps   atomic.Int64

	// bufPool is this Reader's free list of coarse-grained section read buffers.
	// Buffers are borrowed by ColumnReaders and returned on Close, so a scan
	// allocates on the order of one buffer per concurrently open column rather
	// than one per block — and those buffers survive across row groups and
	// across TableScan.Reset repositioning.
	//
	// A miss here falls through to sharedBufPool, and Reader.Close hands whatever
	// is left back to it, so the buffers also survive this Reader's lifetime.
	bufPool [][]byte
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
		r:         r,
		rgMeta:    rgMeta,
		colMeta:   colMeta,
		field:     field,
		pos:       colMeta.SectionOffset,
		end:       colMeta.SectionOffset + colMeta.SectionLength - int64(colMeta.DictLength),
		rowsDone:  0,
		totalRows: rgMeta.NumRows,
	}, nil
}

// Close closes the underlying file and hands this Reader's free section windows
// to the shared pool.
//
// The handback is what makes pooling effective under morsel-driven parallelism:
// a Reader's lifetime there is one morsel, so windows that stopped at Reader
// scope were garbage on every morsel boundary.
func (r *Reader) Close() error {
	for i, b := range r.bufPool {
		putSharedBuf(b)
		r.bufPool[i] = nil
	}
	r.bufPool = nil
	return r.f.Close()
}

// BytesRead returns the total number of bytes read from the file.
func (r *Reader) BytesRead() int64 { return r.bytesRead.Load() }

// ReadOps returns the number of read syscalls (preads) issued against the file.
// Coarse-grained section buffering keeps this proportional to the number of
// (row group, column) sections touched rather than the number of blocks.
func (r *Reader) ReadOps() int64 { return r.readOps.Load() }

// acquireBuf returns a buffer of exactly n bytes, preferring this Reader's free
// list and falling back to the shared size-classed pool.
func (r *Reader) acquireBuf(n int) []byte {
	if k := len(r.bufPool); k > 0 {
		b := r.bufPool[k-1]
		r.bufPool[k-1] = nil
		r.bufPool = r.bufPool[:k-1]
		if cap(b) >= n {
			sectionBufReuses.Add(1)
			return b[:n]
		}
		// Too small for this section but the right size for another: file it in
		// the shared pool rather than dropping it.
		putSharedBuf(b)
	}
	return getSharedBuf(n)
}

// releaseBuf returns a buffer for reuse by a later section read: to this
// Reader's free list while it has room, otherwise straight to the shared pool.
func (r *Reader) releaseBuf(b []byte) {
	if b == nil {
		return
	}
	if len(r.bufPool) >= maxPooledBufs {
		putSharedBuf(b)
		return
	}
	r.bufPool = append(r.bufPool, b[:cap(b)])
}

// readAt reads exactly len(buf) bytes at offset off, updating BytesRead.
func (r *Reader) readAt(off int64, buf []byte) error {
	n, err := r.f.ReadAt(buf, off)
	r.bytesRead.Add(int64(n))
	r.readOps.Add(1)
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
