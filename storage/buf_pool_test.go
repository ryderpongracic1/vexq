package storage

import (
	"encoding/binary"
	"io"
	"sync"
	"testing"
)

// writeTwoColumnFile writes a file with an INT64 and a DATE column, split into
// RowGroupRows-sized row groups. The two column types give sections of
// noticeably different sizes (8-byte vs 4-byte values), so tests exercise more
// than one buffer size class.
func writeTwoColumnFile(t *testing.T, rowGroups int) string {
	t.Helper()
	schema := makeSchema(
		Field{Name: "id", Type: TypeInt64},
		Field{Name: "d", Type: TypeDate},
	)
	w, path := newTestWriter(t, schema)

	ids := make([]int64, RowGroupRows)
	dates := make([]int32, RowGroupRows)
	for rg := 0; rg < rowGroups; rg++ {
		for i := range ids {
			ids[i] = int64(rg*RowGroupRows + i)
			dates[i] = int32(8035 + (rg*RowGroupRows+i)%2000)
		}
		if err := w.BeginRowGroup(RowGroupRows); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}
		if err := w.AppendColumn(ctx, 0, nil, ids); err != nil {
			t.Fatalf("AppendColumn id: %v", err)
		}
		if err := w.AppendColumn(ctx, 1, nil, dates); err != nil {
			t.Fatalf("AppendColumn d: %v", err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
	}
	finishWriter(t, w)
	return path
}

// scanMorsel does what one morsel of the parallel scheduler does: open a Reader,
// read every block of the given row group's columns, close the ColumnReaders,
// then close the Reader. Returns the reads it issued.
func scanMorsel(t *testing.T, path string, rg int, cols []int) int64 {
	t.Helper()
	r := openReader(t, path)
	afterOpen := r.ReadOps()

	crs := make([]*ColumnReader, len(cols))
	for i, col := range cols {
		cr, err := r.OpenColumn(ctx, rg, col)
		if err != nil {
			t.Fatalf("OpenColumn rg=%d col=%d: %v", rg, col, err)
		}
		crs[i] = cr
	}
	for _, cr := range crs {
		for {
			if _, _, _, err := cr.NextBlock(ctx); err == io.EOF {
				break
			} else if err != nil {
				t.Fatalf("NextBlock: %v", err)
			}
		}
	}
	for _, cr := range crs {
		if err := cr.Close(); err != nil {
			t.Fatalf("ColumnReader.Close: %v", err)
		}
	}
	reads := r.ReadOps() - afterOpen
	if err := r.Close(); err != nil {
		t.Fatalf("Reader.Close: %v", err)
	}
	return reads
}

// ---- Size classes ----------------------------------------------------------

// TestBufClassInvariants pins the relationship the two classification functions
// must satisfy: a buffer allocated for a request of n bytes is filed under
// exactly the class a later request of n bytes looks in. Rounding the request up
// but the capacity down would miss on every acquire — which is precisely how the
// per-Reader pool failed before — so it is asserted directly rather than inferred
// from a benchmark.
func TestBufClassInvariants(t *testing.T) {
	sizes := []int{
		1,
		bufClassBytes - 1,
		bufClassBytes,
		bufClassBytes + 1,
		NullBitmapBytes + BlockRows*8 + 4, // one INT64 block
		blocksPerFullRowGroup * (NullBitmapBytes + BlockRows*8 + 4), // a full INT64 section
		blocksPerFullRowGroup * (NullBitmapBytes + BlockRows*4 + 4), // a full DATE section
		maxPrefetchBytes - 1,
		maxPrefetchBytes,
	}
	for _, n := range sizes {
		c := bufClassOf(n)
		if c < 1 || c > numBufClasses {
			t.Fatalf("bufClassOf(%d) = %d, want a class in [1,%d]", n, c, numBufClasses)
		}
		if got := c * bufClassBytes; got < n {
			t.Fatalf("class %d holds %d bytes, too small for a %d-byte request", c, got, n)
		}
		if got := poolClassOf(c * bufClassBytes); got != c {
			t.Fatalf("a buffer allocated for %d bytes files under class %d but is fetched from class %d",
				n, got, c)
		}
	}

	// Sizes outside every class are neither served from nor filed in the pool.
	for _, n := range []int{0, -1, maxPrefetchBytes + 1} {
		if c := bufClassOf(n); c != 0 {
			t.Fatalf("bufClassOf(%d) = %d, want 0 (unpoolable)", n, c)
		}
	}
	for _, capacity := range []int{0, -1, bufClassBytes + 1, maxPrefetchBytes + bufClassBytes} {
		if c := poolClassOf(capacity); c != 0 {
			t.Fatalf("poolClassOf(%d) = %d, want 0 (not exactly one class wide)", capacity, c)
		}
	}
}

// TestSharedPoolSizeClassRoundTrip is the behavioural counterpart to
// TestBufClassInvariants: a released window really does come back out.
func TestSharedPoolSizeClassRoundTrip(t *testing.T) {
	sizes := []int{
		1,
		bufClassBytes,
		bufClassBytes + 1,
		blocksPerFullRowGroup * (NullBitmapBytes + BlockRows*8 + 4),
		blocksPerFullRowGroup * (NullBitmapBytes + BlockRows*4 + 4),
		maxPrefetchBytes,
	}
	for _, n := range sizes {
		// A normal build returns the released buffer on the very next acquire.
		// Under -race sync.Pool discards a random quarter of Put calls, so retry
		// enough times that a genuine failure to pool is still caught while a
		// run of unlucky drops is not: 12 attempts leaves a 0.25^12 (~6e-8)
		// chance of a spurious failure.
		attempts := 1
		if raceDetectorEnabled {
			attempts = 12
		}
		reused := false
		for i := 0; i < attempts && !reused; i++ {
			first := getSharedBuf(n)
			if len(first) != n {
				t.Fatalf("n=%d: got length %d", n, len(first))
			}
			if cap(first)%bufClassBytes != 0 {
				t.Fatalf("n=%d: capacity %d is not a whole number of size classes", n, cap(first))
			}
			want := &first[0]
			putSharedBuf(first)

			second := getSharedBuf(n)
			if len(second) != n {
				t.Fatalf("n=%d: got length %d on reuse", n, len(second))
			}
			reused = &second[0] == want
			putSharedBuf(second)
		}
		if !reused {
			t.Fatalf("n=%d (class %d): a released buffer was never returned by a later acquire",
				n, bufClassOf(n))
		}
	}
}

// TestOversizedWindowIsServedButNotPooled guards the retention bound: a request
// larger than any class is served exactly and never filed, so one outlier cannot
// pin a buffer bigger than maxPrefetchBytes.
func TestOversizedWindowIsServedButNotPooled(t *testing.T) {
	const n = maxPrefetchBytes + 1
	b := getSharedBuf(n)
	if len(b) != n {
		t.Fatalf("got length %d, want %d", len(b), n)
	}
	if cap(b) != n {
		t.Fatalf("oversized request got capacity %d, want exactly %d", cap(b), n)
	}
	if c := poolClassOf(cap(b)); c != 0 {
		t.Fatalf("an oversized buffer would be filed under class %d", c)
	}
}

// ---- Reuse across Reader lifetimes (the morsel pattern) --------------------

// TestWindowsReusedAcrossReaders is the regression test for the bug this pool
// exists to fix. The morsel scheduler created a Reader per morsel at the time,
// so before the shared pool every morsel started with a cold free list and
// allocated a fresh window per projected column — 100% miss rate, invisible to
// the pointer-identity test that only covers row groups within one Reader.
// Per-worker pipeline reuse (exec.MorselPipeline) has since made a Reader last a
// worker's whole run, but windows still have to survive Reader boundaries — this
// test drives them directly, one short-lived Reader per morsel, so it keeps
// pinning that property regardless of how long the executor's Readers live.
//
// The assertion is a ratio rather than zero because sync.Pool is cleared by the
// garbage collector: a GC landing mid-test legitimately costs a few allocations.
// The pre-fix behaviour was one allocation per acquire, so any ratio well below 1
// distinguishes the two decisively.
func TestWindowsReusedAcrossReaders(t *testing.T) {
	const rowGroups = 4
	const morsels = 12
	path := writeTwoColumnFile(t, rowGroups)
	cols := []int{0, 1}

	// Warm the pool so steady-state reuse is what gets measured.
	scanMorsel(t, path, 0, cols)

	allocsBefore, reusesBefore := SectionBufferStats()
	for m := 0; m < morsels; m++ {
		scanMorsel(t, path, m%rowGroups, cols)
	}
	allocs, reuses := SectionBufferStats()

	acquires := (allocs - allocsBefore) + (reuses - reusesBefore)
	wantAcquires := int64(morsels * len(cols))
	if acquires != wantAcquires {
		t.Fatalf("%d window acquires over %d morsels, want %d", acquires, morsels, wantAcquires)
	}
	// Not zero: sync.Pool is cleared by the garbage collector, so a GC landing
	// mid-test legitimately costs a few allocations. Under -race sync.Pool also
	// drops a random quarter of Put calls, which raises the floor further. The
	// pre-fix behaviour was one allocation per acquire, so either bound
	// distinguishes the two decisively.
	limit := acquires / 4
	if raceDetectorEnabled {
		limit = acquires / 2
	}
	if fresh := allocs - allocsBefore; fresh > limit {
		t.Fatalf("%d of %d window acquires allocated (limit %d); pooling is not surviving Reader lifetimes",
			fresh, acquires, limit)
	}
}

// TestReaderCloseHandsWindowsToSharedPool pins the mechanism directly: after
// Close the Reader must hold nothing, and the window it was holding must be
// available to the next Reader.
func TestReaderCloseHandsWindowsToSharedPool(t *testing.T) {
	path := writeInt64File(t, RowGroupRows)

	// closeThenReopen runs one Reader lifetime and reports whether the window it
	// released came back out in the next Reader.
	closeThenReopen := func() bool {
		r := openReader(t, path)
		cr, err := r.OpenColumn(ctx, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := cr.NextBlock(ctx); err != nil {
			t.Fatalf("NextBlock: %v", err)
		}
		win := &cr.win[0]
		drainColumn(t, cr)
		if err := cr.Close(); err != nil {
			t.Fatal(err)
		}
		if len(r.bufPool) != 1 {
			t.Fatalf("before Reader.Close the free list holds %d buffers, want 1", len(r.bufPool))
		}
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
		if r.bufPool != nil {
			t.Fatalf("after Close the free list still holds %d buffers", len(r.bufPool))
		}

		r2 := openReader(t, path)
		defer r2.Close()
		cr2, err := r2.OpenColumn(ctx, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer cr2.Close()
		if _, _, _, err := cr2.NextBlock(ctx); err != nil {
			t.Fatalf("NextBlock: %v", err)
		}
		return &cr2.win[0] == win
	}

	// See TestSharedPoolSizeClassRoundTrip for why -race needs retries.
	attempts := 1
	if raceDetectorEnabled {
		attempts = 12
	}
	for i := 0; i < attempts; i++ {
		if closeThenReopen() {
			return
		}
	}
	t.Fatal("the window released by a closing Reader was never reused by the next one")
}

// TestOverflowingFreeListGoesToSharedPool covers the maxPooledBufs bound: past
// it, buffers must be filed in the shared pool rather than dropped.
func TestOverflowingFreeListGoesToSharedPool(t *testing.T) {
	path := writeInt64File(t, BlockRows)
	r := openReader(t, path)
	defer r.Close()

	const n = bufClassBytes

	overflowRecovered := func() bool {
		r.bufPool = nil
		for i := 0; i < maxPooledBufs; i++ {
			r.releaseBuf(make([]byte, n, n))
		}
		if len(r.bufPool) != maxPooledBufs {
			t.Fatalf("free list holds %d, want %d", len(r.bufPool), maxPooledBufs)
		}

		overflow := make([]byte, n, n)
		marker := &overflow[0]
		r.releaseBuf(overflow)
		if len(r.bufPool) != maxPooledBufs {
			t.Fatalf("free list grew past its bound to %d", len(r.bufPool))
		}

		// Drain the free list so the next acquires have to reach the shared pool.
		r.bufPool = nil
		for i := 0; i < maxPooledBufs+2; i++ {
			b := r.acquireBuf(n)
			hit := &b[0] == marker
			putSharedBuf(b)
			if hit {
				return true
			}
		}
		return false
	}

	// See TestSharedPoolSizeClassRoundTrip for why -race needs retries.
	attempts := 1
	if raceDetectorEnabled {
		attempts = 12
	}
	for i := 0; i < attempts; i++ {
		if overflowRecovered() {
			return
		}
	}
	t.Fatal("a buffer released past the free-list bound was dropped instead of pooled")
}

// ---- No read-count regression ---------------------------------------------

// TestMorselScanKeepsOneReadPerSection is the no-regression guard for the
// coarse-I/O win: pooling changes where a window comes from, never how much of
// the section a single read covers.
func TestMorselScanKeepsOneReadPerSection(t *testing.T) {
	const rowGroups = 3
	path := writeTwoColumnFile(t, rowGroups)
	cols := []int{0, 1}

	for m := 0; m < 2*rowGroups; m++ {
		reads := scanMorsel(t, path, m%rowGroups, cols)
		if want := int64(len(cols)); reads != want {
			t.Fatalf("morsel %d issued %d reads for %d sections, want %d",
				m, reads, len(cols), want)
		}
	}
}

// ---- Concurrency ----------------------------------------------------------

// TestConcurrentReadersShareThePool runs the morsel pattern from several
// goroutines at once, each owning its own Reader, which is exactly how the
// parallel scan uses this package. It is a -race target: the per-Reader free
// list stays goroutine-private and only the shared sync.Pool crosses goroutines.
func TestConcurrentReadersShareThePool(t *testing.T) {
	const rowGroups = 4
	const workers = 4
	const morselsPerWorker = 8
	path := writeTwoColumnFile(t, rowGroups)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for m := 0; m < morselsPerWorker; m++ {
				r, err := Open(ctx, path)
				if err != nil {
					t.Errorf("worker %d: open: %v", w, err)
					return
				}
				rg := (w + m) % rowGroups
				for _, col := range []int{0, 1} {
					cr, err := r.OpenColumn(ctx, rg, col)
					if err != nil {
						t.Errorf("worker %d: open column: %v", w, err)
						r.Close()
						return
					}
					rows := 0
					for {
						_, _, n, err := cr.NextBlock(ctx)
						if err == io.EOF {
							break
						}
						if err != nil {
							t.Errorf("worker %d: next block: %v", w, err)
							cr.Close()
							r.Close()
							return
						}
						rows += n
					}
					if rows != RowGroupRows {
						t.Errorf("worker %d rg=%d col=%d: read %d rows, want %d",
							w, rg, col, rows, RowGroupRows)
					}
					if err := cr.Close(); err != nil {
						t.Errorf("worker %d: close column: %v", w, err)
					}
				}
				if err := r.Close(); err != nil {
					t.Errorf("worker %d: close reader: %v", w, err)
				}
			}
		}(w)
	}
	wg.Wait()
}

// TestConcurrentReadersSeeIndependentData is the correctness half of the
// concurrency story: pooled buffers are handed to one holder at a time and fully
// overwritten by the read that fills them, so no Reader can ever observe another
// Reader's bytes.
func TestConcurrentReadersSeeIndependentData(t *testing.T) {
	const rowGroups = 3
	path := writeInt64File(t, RowGroupRows*rowGroups)

	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for pass := 0; pass < 6; pass++ {
				rg := (w + pass) % rowGroups
				r, err := Open(ctx, path)
				if err != nil {
					t.Errorf("open: %v", err)
					return
				}
				cr, err := r.OpenColumn(ctx, rg, 0)
				if err != nil {
					t.Errorf("open column: %v", err)
					r.Close()
					return
				}
				next := int64(rg * RowGroupRows)
				for {
					_, payload, rows, err := cr.NextBlock(ctx)
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Errorf("next block: %v", err)
						break
					}
					for i := 0; i < rows; i++ {
						got := int64(binary.LittleEndian.Uint64(payload[i*8:]))
						if got != next {
							t.Errorf("worker %d rg=%d: got %d want %d", w, rg, got, next)
							cr.Close()
							r.Close()
							return
						}
						next++
					}
				}
				cr.Close()
				r.Close()
			}
		}(w)
	}
	wg.Wait()
}
