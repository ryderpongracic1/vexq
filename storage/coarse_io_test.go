package storage

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"testing"
)

// blocksPerFullRowGroup is how many block reads the old block-granular reader
// issued per full row group and column. The coarse reader must beat it by this
// factor, so tests assert against it rather than a bare literal.
const blocksPerFullRowGroup = RowGroupRows / BlockRows // 64

// writeInt64File writes a single-INT64-column file holding values 0..n-1,
// split into RowGroupRows-sized row groups (the last one possibly partial).
func writeInt64File(t *testing.T, n int) string {
	t.Helper()
	schema := makeSchema(Field{Name: "v", Type: TypeInt64})
	w, path := newTestWriter(t, schema)
	vals := make([]int64, n)
	for i := range vals {
		vals[i] = int64(i)
	}
	for written := 0; written < n; {
		chunk := RowGroupRows
		if written+chunk > n {
			chunk = n - written
		}
		if err := w.BeginRowGroup(chunk); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}
		if err := w.AppendColumn(ctx, 0, nil, vals[written:written+chunk]); err != nil {
			t.Fatalf("AppendColumn: %v", err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
		written += chunk
	}
	finishWriter(t, w)
	return path
}

// drainColumn reads every block of one (row group, column) section, returning
// the number of rows and blocks seen.
func drainColumn(t *testing.T, cr *ColumnReader) (rows, blocks int) {
	t.Helper()
	for {
		_, _, n, err := cr.NextBlock(ctx)
		if err == io.EOF {
			return rows, blocks
		}
		if err != nil {
			t.Fatalf("NextBlock: %v", err)
		}
		rows += n
		blocks++
	}
}

// ---- One read per (row group, column) --------------------------------------

func TestSectionReadIsCoarseGrained(t *testing.T) {
	const rowGroups = 4
	const n = RowGroupRows * rowGroups
	path := writeInt64File(t, n)

	r := openReader(t, path)
	afterOpen := r.ReadOps()

	totalRows, totalBlocks := 0, 0
	for rg := 0; rg < len(r.Meta().RowGroups); rg++ {
		cr, err := r.OpenColumn(ctx, rg, 0)
		if err != nil {
			t.Fatalf("OpenColumn rg=%d: %v", rg, err)
		}
		rows, blocks := drainColumn(t, cr)
		if err := cr.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		totalRows += rows
		totalBlocks += blocks
	}

	if totalRows != n {
		t.Fatalf("read %d rows, want %d", totalRows, n)
	}
	wantBlocks := rowGroups * blocksPerFullRowGroup
	if totalBlocks != wantBlocks {
		t.Fatalf("read %d blocks, want %d", totalBlocks, wantBlocks)
	}

	// The whole point: block count is unchanged, read syscall count collapses to
	// one per section.
	reads := r.ReadOps() - afterOpen
	if reads != rowGroups {
		t.Fatalf("issued %d reads for %d sections (%d blocks), want %d",
			reads, rowGroups, totalBlocks, rowGroups)
	}
}

// ---- Section buffers are reused across row groups --------------------------

func TestSectionBufferReusedAcrossRowGroups(t *testing.T) {
	const rowGroups = 5
	path := writeInt64File(t, RowGroupRows*rowGroups)
	r := openReader(t, path)

	var firstBuf *byte
	for rg := 0; rg < rowGroups; rg++ {
		cr, err := r.OpenColumn(ctx, rg, 0)
		if err != nil {
			t.Fatalf("OpenColumn rg=%d: %v", rg, err)
		}
		if _, _, _, err := cr.NextBlock(ctx); err != nil {
			t.Fatalf("NextBlock rg=%d: %v", rg, err)
		}
		if cr.win == nil {
			t.Fatalf("rg=%d: expected a buffered window", rg)
		}
		buf := &cr.win[0]
		if rg == 0 {
			firstBuf = buf
		} else if buf != firstBuf {
			t.Fatalf("rg=%d: section buffer was reallocated, not reused from the pool", rg)
		}
		drainColumn(t, cr)
		if err := cr.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		// Close must hand the buffer back so the next row group can take it.
		if len(r.bufPool) != 1 {
			t.Fatalf("rg=%d: pool holds %d buffers, want 1", rg, len(r.bufPool))
		}
	}
}

// TestSectionReadDoesNotAllocatePerBlock pins the property the earlier GC fix
// established for decode buffers: draining a whole row group must not allocate
// once per block.
func TestSectionReadDoesNotAllocatePerBlock(t *testing.T) {
	const rowGroups = 3
	path := writeInt64File(t, RowGroupRows*rowGroups)
	r := openReader(t, path)

	rg := 0
	allocs := testing.AllocsPerRun(6, func() {
		cr, err := r.OpenColumn(ctx, rg%rowGroups, 0)
		if err != nil {
			t.Fatalf("OpenColumn: %v", err)
		}
		drainColumn(t, cr)
		_ = cr.Close()
		rg++
	})

	// One ColumnReader struct plus a little slack for the pool slice; the point
	// is that it is nowhere near one allocation per block.
	const maxAllocs = 8
	if allocs > maxAllocs {
		t.Fatalf("%.0f allocations per row group (%d blocks); want <= %d",
			allocs, blocksPerFullRowGroup, maxAllocs)
	}
}

// ---- CRC is still verified per block, not per window -----------------------

func TestCRCDetectedInLaterBlockOfSection(t *testing.T) {
	const badBlock = 5
	const rows = BlockRows * 8
	path := writeInt64File(t, rows)

	// Locate block `badBlock` from the footer rather than assuming a header size.
	r := openReader(t, path)
	sectionOff := r.Meta().RowGroups[0].Columns[0].SectionOffset
	blockBytes := int64(NullBitmapBytes + BlockRows*8 + 4)
	corruptAt := sectionOff + badBlock*blockBytes + int64(NullBitmapBytes) + 16

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[corruptAt] ^= 0xFF
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	r2 := openReader(t, path)
	cr, err := r2.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer cr.Close()

	for i := 0; i < badBlock; i++ {
		if _, _, _, err := cr.NextBlock(ctx); err != nil {
			t.Fatalf("block %d: unexpected error before the corrupt block: %v", i, err)
		}
	}
	if _, _, _, err := cr.NextBlock(ctx); !errors.Is(err, ErrChecksum) {
		t.Fatalf("block %d: got %v, want ErrChecksum", badBlock, err)
	}
	// A caller that logs the error and keeps going must still reach EOF rather
	// than retrying the bad block forever.
	for {
		_, _, _, err := cr.NextBlock(ctx)
		if err == io.EOF {
			break
		}
	}
}

// ---- Last (partial) row group ---------------------------------------------

func TestLastPartialRowGroupCoarseRead(t *testing.T) {
	// Two full row groups plus a partial one whose row count is neither a
	// multiple of BlockRows nor smaller than it: 1500 rows = 1024 + 476.
	const tailRows = 1500
	const n = RowGroupRows*2 + tailRows
	path := writeInt64File(t, n)

	r := openReader(t, path)
	rgs := r.Meta().RowGroups
	if len(rgs) != 3 {
		t.Fatalf("expected 3 row groups, got %d", len(rgs))
	}
	if rgs[2].NumRows != tailRows {
		t.Fatalf("last row group has %d rows, want %d", rgs[2].NumRows, tailRows)
	}

	afterOpen := r.ReadOps()
	next := int64(0)
	var lastBlockRows []int
	for rg := 0; rg < len(rgs); rg++ {
		cr, err := r.OpenColumn(ctx, rg, 0)
		if err != nil {
			t.Fatalf("OpenColumn rg=%d: %v", rg, err)
		}
		for {
			_, payload, rows, err := cr.NextBlock(ctx)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("rg=%d NextBlock: %v", rg, err)
			}
			if len(payload) != rows*8 {
				t.Fatalf("rg=%d: payload is %d bytes for %d rows", rg, len(payload), rows)
			}
			for i := 0; i < rows; i++ {
				got := int64(binary.LittleEndian.Uint64(payload[i*8:]))
				if got != next {
					t.Fatalf("rg=%d row %d: got %d want %d", rg, next, got, next)
				}
				next++
			}
			if rg == 2 {
				lastBlockRows = append(lastBlockRows, rows)
			}
		}
		if err := cr.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}
	if next != n {
		t.Fatalf("read %d rows, want %d", next, n)
	}
	if len(lastBlockRows) != 2 || lastBlockRows[0] != BlockRows || lastBlockRows[1] != tailRows-BlockRows {
		t.Fatalf("partial row group blocks = %v, want [%d %d]",
			lastBlockRows, BlockRows, tailRows-BlockRows)
	}
	if reads := r.ReadOps() - afterOpen; reads != int64(len(rgs)) {
		t.Fatalf("issued %d reads for %d sections, want %d", reads, len(rgs), len(rgs))
	}
}

// ---- Variable-width sections: BOOL (RLE) and STRING (dict) ----------------

func TestBoolAndStringSectionsCoarseRead(t *testing.T) {
	const n = BlockRows*3 + 200 // 3 full blocks + a partial one
	schema := makeSchema(
		Field{Name: "flag", Type: TypeBool, Encoding: EncRLE, Nullable: true},
		Field{Name: "tag", Type: TypeString, Encoding: EncDict, Nullable: true},
	)
	w, path := newTestWriter(t, schema)

	flags := make([]bool, n)
	tags := make([]string, n)
	nulls := FullBitmap(n)
	for i := range flags {
		flags[i] = i%7 < 3 // creates multi-row runs
		tags[i] = string(rune('a' + i%13))
		if i%11 == 0 {
			SetNullBit(nulls, i)
		}
	}
	if err := w.BeginRowGroup(n); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendColumn(ctx, 0, nulls, flags); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendColumn(ctx, 1, nulls, tags); err != nil {
		t.Fatal(err)
	}
	if err := w.EndRowGroup(); err != nil {
		t.Fatal(err)
	}
	finishWriter(t, w)

	r := openReader(t, path)

	// BOOL: variable-length RLE blocks. Discovering each block's size must not
	// cost a second syscall, so the whole section is still one read.
	afterOpen := r.ReadOps()
	crBool, err := r.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	boolRows := 0
	for {
		_, payload, rows, err := crBool.NextBlock(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("bool NextBlock: %v", err)
		}
		vals, bits, decoded, err := DecodeRLEBool(payload)
		if err != nil {
			t.Fatalf("DecodeRLEBool: %v", err)
		}
		if decoded != rows {
			t.Fatalf("bool block: decoded %d rows, block reports %d", decoded, rows)
		}
		for i := 0; i < rows; i++ {
			row := boolRows + i
			wantNull := row%11 == 0
			if IsNullBit(bits, i) != wantNull {
				t.Fatalf("bool row %d: null=%v want %v", row, !wantNull, wantNull)
			}
			if !wantNull && vals[i] != flags[row] {
				t.Fatalf("bool row %d: got %v want %v", row, vals[i], flags[row])
			}
		}
		boolRows += rows
	}
	if err := crBool.Close(); err != nil {
		t.Fatal(err)
	}
	if boolRows != n {
		t.Fatalf("bool: read %d rows, want %d", boolRows, n)
	}
	if reads := r.ReadOps() - afterOpen; reads != 1 {
		t.Fatalf("bool section took %d reads, want 1", reads)
	}

	// STRING: one read for the code blocks plus one for the dictionary blob,
	// which lives outside the block region.
	beforeStr := r.ReadOps()
	crStr, err := r.OpenColumn(ctx, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	dict, err := crStr.Dictionary()
	if err != nil {
		t.Fatalf("Dictionary: %v", err)
	}
	strRows := 0
	for {
		nb, payload, rows, err := crStr.NextBlock(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("string NextBlock: %v", err)
		}
		if len(nb) != (rows+7)/8 {
			t.Fatalf("string block: null bitmap is %d bytes for %d rows", len(nb), rows)
		}
		for i := 0; i < rows; i++ {
			row := strRows + i
			if row%11 == 0 {
				continue // null; the stored code is unspecified
			}
			code := binary.LittleEndian.Uint32(payload[i*4:])
			if got := dict.Get(code); got != tags[row] {
				t.Fatalf("string row %d: got %q want %q", row, got, tags[row])
			}
		}
		strRows += rows
	}
	if err := crStr.Close(); err != nil {
		t.Fatal(err)
	}
	if strRows != n {
		t.Fatalf("string: read %d rows, want %d", strRows, n)
	}
	if reads := r.ReadOps() - beforeStr; reads != 2 {
		t.Fatalf("string section took %d reads, want 2 (blocks + dictionary)", reads)
	}
}

// ---- Duplicate projection of the same column ------------------------------

// TestConcurrentColumnReadersSameColumn guards the pooling scheme: two open
// ColumnReaders over the same column (which a query projecting a column twice
// produces) must get independent buffers.
func TestConcurrentColumnReadersSameColumn(t *testing.T) {
	path := writeInt64File(t, BlockRows*4)
	r := openReader(t, path)

	a, err := r.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := r.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	for i := 0; i < 4; i++ {
		_, pa, rowsA, err := a.NextBlock(ctx)
		if err != nil {
			t.Fatalf("a block %d: %v", i, err)
		}
		_, pb, rowsB, err := b.NextBlock(ctx)
		if err != nil {
			t.Fatalf("b block %d: %v", i, err)
		}
		if rowsA != rowsB {
			t.Fatalf("block %d: row counts diverged: %d vs %d", i, rowsA, rowsB)
		}
		// Reading through b must not have clobbered a's still-live payload.
		for j := 0; j < rowsA; j++ {
			want := int64(i*BlockRows + j)
			if got := int64(binary.LittleEndian.Uint64(pa[j*8:])); got != want {
				t.Fatalf("reader a block %d row %d: got %d want %d", i, j, got, want)
			}
			if got := int64(binary.LittleEndian.Uint64(pb[j*8:])); got != want {
				t.Fatalf("reader b block %d row %d: got %d want %d", i, j, got, want)
			}
		}
	}
	if &a.win[0] == &b.win[0] {
		t.Fatal("two ColumnReaders over the same column share one buffer")
	}
}
