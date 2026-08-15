package exec

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// ---- Fixtures ---------------------------------------------------------------

// radixBuildFixture generates a build side split across rowGroups synthetic row
// groups of rowsPerRG rows each, and hands out a PipelineFactory over row-group
// ranges so BuildSharedHashTableParallel can be driven without touching a file.
//
// keyFor maps a global row index to its join key; a nil return means the key is
// NULL for that row.
type radixBuildFixture struct {
	rowGroups int
	rowsPerRG int
	keyFor    func(row int) *int64
}

func (f radixBuildFixture) schema() Schema {
	return Schema{Fields: []Field{
		{Name: "b_key", Type: TypeInt64, Nullable: true},
		{Name: "b_payload", Type: TypeInt64, Nullable: true},
	}}
}

// batchFor materialises row group rg as a single batch. Payload is the global row
// index, which makes every build row distinguishable in assertions.
func (f radixBuildFixture) batchFor(rg int) *Batch {
	schema := f.schema()
	keys := make([]*int64, f.rowsPerRG)
	payload := make([]*int64, f.rowsPerRG)
	for i := range f.rowsPerRG {
		row := rg*f.rowsPerRG + i
		keys[i] = f.keyFor(row)
		payload[i] = ptr(int64(row))
	}
	return &Batch{
		Schema:  schema,
		Vectors: []Vector{int64Vec(keys), int64Vec(payload)},
		Length:  f.rowsPerRG,
	}
}

// factory returns a PipelineFactory over row-group ranges of the fixture.
func (f radixBuildFixture) factory() PipelineFactory {
	return func(_ context.Context, rgStart, rgEnd int) (Operator, error) {
		batches := make([]*Batch, 0, rgEnd-rgStart)
		for rg := rgStart; rg < rgEnd; rg++ {
			batches = append(batches, f.batchFor(rg))
		}
		return &sliceBatchOp{schema: f.schema(), batches: batches}, nil
	}
}

// serialOp returns one operator over every row group, in order — the serial
// drain the parallel build must reproduce exactly.
func (f radixBuildFixture) serialOp() Operator {
	op, _ := f.factory()(context.Background(), 0, f.rowGroups)
	return op
}

// tableContents renders a table as key → ordered payload list, which captures
// both membership and per-key row order.
func tableContents(t *testing.T, sht *SharedHashTable) map[int64][]int64 {
	t.Helper()
	out := make(map[int64][]int64)
	for _, tbl := range sht.parts {
		tbl.forEachKey(func(key int64, head int32) {
			if _, dup := out[key]; dup {
				t.Fatalf("key %d appears in more than one partition", key)
			}
			var vals []int64
			for r := head; r != noRow; r = sht.store.next[r] {
				vals = append(vals, sht.store.value(r, 1))
			}
			out[key] = vals
		})
	}
	return out
}

// ---- RadixBitsFor -----------------------------------------------------------

func TestRadixBitsFor(t *testing.T) {
	cases := []struct {
		estRows  int
		wantBits int
		why      string
	}{
		{0, 0, "no estimate means no partitioning"},
		{1, 0, "tiny build side stays in one map"},
		{radixMinBuildRows - 1, 0, "just below the threshold"},
		{radixMinBuildRows, radixMinBits, "at the threshold, minimum partitions"},
		{16 * radixTargetRows, radixMinBits, "16 partitions still hit the row target"},
		{16*radixTargetRows + 1, radixMinBits + 1, "one row over needs another bit"},
		{300_000, radixMaxBits, "TPC-H SF-scaled orders saturates the clamp"},
		{1_500_000, radixMaxBits, "SF=1 orders saturates the clamp"},
		{1 << 40, radixMaxBits, "absurd estimate still clamps"},
	}
	for _, c := range cases {
		if got := RadixBitsFor(c.estRows); got != c.wantBits {
			t.Errorf("RadixBitsFor(%d) = %d, want %d (%s)", c.estRows, got, c.wantBits, c.why)
		}
	}
}

// TestRadixHashSpreadsSequentialKeys is the reason radixHash mixes at all: TPC-H
// order keys are sequential with gaps, so masking their low bits directly would
// pile rows into a few partitions. The mixed hash must spread them close to
// evenly.
func TestRadixHashSpreadsSequentialKeys(t *testing.T) {
	const (
		numKeys  = 1 << 16
		numParts = 64
		// dbgen leaves gaps in the order key space; step 8 reproduces a sparse
		// key set whose low 6 bits take only 8 distinct values.
		step = 8
	)
	counts := make([]int, numParts)
	for i := range numKeys {
		key := int64(i * step)
		counts[radixPart(key, numParts-1)]++
	}
	want := numKeys / numParts
	for p, got := range counts {
		if got < want/2 || got > want*2 {
			t.Fatalf("partition %d holds %d keys, want within 2x of %d — hash is not spreading", p, got, want)
		}
	}

	// Sanity check on the premise: the unmixed low bits really are that skewed.
	raw := make([]int, numParts)
	for i := range numKeys {
		raw[int(int64(i*step))&(numParts-1)]++
	}
	empties := 0
	for _, got := range raw {
		if got == 0 {
			empties++
		}
	}
	if empties == 0 {
		t.Fatal("expected raw low-bit masking to leave partitions empty; the fixture no longer demonstrates skew")
	}
}

func TestRadixPartUnpartitionedIsAlwaysZero(t *testing.T) {
	for _, key := range []int64{math.MinInt64, -1, 0, 1, 1 << 40, math.MaxInt64} {
		if got := radixPart(key, 0); got != 0 {
			t.Errorf("radixPart(%d, 0) = %d, want 0", key, got)
		}
	}
}

// ---- Build equivalence ------------------------------------------------------

// TestParallelBuildMatchesSerialBuild is the central correctness property: for
// every worker count and partition count, the morsel-parallel build produces the
// same keys, the same rows per key, and the same order within each key as a
// serial drain. Per-key order matters because it is what makes joined output
// order-identical to the serial join, which in turn makes integer aggregates
// exactly equal rather than merely equivalent.
func TestParallelBuildMatchesSerialBuild(t *testing.T) {
	ctx := context.Background()
	fixture := radixBuildFixture{
		rowGroups: 9, // deliberately not a multiple of any worker count below
		rowsPerRG: 500,
		keyFor: func(row int) *int64 {
			switch {
			case row%37 == 0:
				return nil // NULL key: dropped by an inner join
			case row%5 == 0:
				return ptr(int64(row % 11)) // heavy duplicate keys
			default:
				return ptr(int64(row) * 8) // sparse sequential keys
			}
		},
	}

	want, err := BuildSharedHashTable(ctx, fixture.serialOp(), 0)
	if err != nil {
		t.Fatalf("serial build: %v", err)
	}
	wantContents := tableContents(t, want)

	for _, workers := range []int{1, 2, 4, 8} {
		for _, bits := range []int{0, 1, 4, 6, 8} {
			label := fmt.Sprintf("workers=%d/bits=%d", workers, bits)
			got, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
				fixture.rowGroups, workers, 1, bits)
			if err != nil {
				t.Fatalf("%s: parallel build: %v", label, err)
			}
			if got.NumPartitions() != 1<<bits {
				t.Errorf("%s: partitions = %d, want %d", label, got.NumPartitions(), 1<<bits)
			}
			if got.NumRows() != want.NumRows() {
				t.Errorf("%s: rows = %d, want %d", label, got.NumRows(), want.NumRows())
			}
			if got.NumKeys() != want.NumKeys() {
				t.Errorf("%s: keys = %d, want %d", label, got.NumKeys(), want.NumKeys())
			}
			if !reflect.DeepEqual(wantContents, tableContents(t, got)) {
				t.Errorf("%s: table contents differ from the serial build", label)
			}
		}
	}
}

// TestSerialRadixBuildMatchesUnpartitioned covers the middle strategy: a build
// side that cannot be split into morsels is still drained serially into
// partitioned maps, and must agree with the unpartitioned build.
func TestSerialRadixBuildMatchesUnpartitioned(t *testing.T) {
	ctx := context.Background()
	fixture := radixBuildFixture{
		rowGroups: 3,
		rowsPerRG: 400,
		keyFor: func(row int) *int64 {
			if row%13 == 0 {
				return nil
			}
			return ptr(int64(row / 2)) // every key appears twice
		},
	}
	want, err := BuildSharedHashTable(ctx, fixture.serialOp(), 0)
	if err != nil {
		t.Fatalf("serial build: %v", err)
	}
	wantContents := tableContents(t, want)

	for _, bits := range []int{1, 3, 5, 8} {
		got, err := BuildSharedHashTableRadix(ctx, fixture.serialOp(), 0, bits)
		if err != nil {
			t.Fatalf("bits=%d: %v", bits, err)
		}
		if got.NumPartitions() != 1<<bits {
			t.Errorf("bits=%d: partitions = %d, want %d", bits, got.NumPartitions(), 1<<bits)
		}
		if !reflect.DeepEqual(wantContents, tableContents(t, got)) {
			t.Errorf("bits=%d: table contents differ from the unpartitioned build", bits)
		}
	}
}

// TestParallelBuildSkewedKeys drives every row into one partition. Correctness
// must not depend on the spread — only performance does — so the table still has
// to hold every row, and the empty partitions must be usable (probing them
// returns no match rather than panicking on a nil map).
func TestParallelBuildSkewedKeys(t *testing.T) {
	ctx := context.Background()
	const oneKey = int64(42)
	fixture := radixBuildFixture{
		rowGroups: 6,
		rowsPerRG: 300,
		keyFor:    func(int) *int64 { return ptr(oneKey) },
	}
	sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
		fixture.rowGroups, 4, 1, 6)
	if err != nil {
		t.Fatalf("parallel build: %v", err)
	}
	if got, want := sht.NumRows(), fixture.rowGroups*fixture.rowsPerRG; got != want {
		t.Fatalf("rows = %d, want %d", got, want)
	}
	if got := sht.NumKeys(); got != 1 {
		t.Fatalf("keys = %d, want 1", got)
	}

	rows := sht.PartitionRows()
	nonEmpty := 0
	for _, n := range rows {
		if n > 0 {
			nonEmpty++
		}
	}
	if nonEmpty != 1 {
		t.Fatalf("non-empty partitions = %d, want 1 (all rows share one key)", nonEmpty)
	}

	// Per-key order must still be row-group order: payload is the global row index.
	contents := tableContents(t, sht)
	got := contents[oneKey]
	for i, v := range got {
		if v != int64(i) {
			t.Fatalf("row %d of key %d has payload %d, want %d — order was not preserved", i, oneKey, v, i)
		}
	}

	// Probing an empty partition must miss cleanly. Find a key that lands in one.
	empty := -1
	for p, n := range rows {
		if n == 0 {
			empty = p
			break
		}
	}
	if empty < 0 {
		t.Fatal("expected at least one empty partition")
	}
	missKey := int64(0)
	for k := int64(1); k < 1_000_000; k++ {
		if radixPart(k, sht.partMask) == empty {
			missKey = k
			break
		}
	}
	if missKey == 0 {
		t.Fatal("could not find a key mapping to the empty partition")
	}
	probe := probeSideOp([]*int64{ptr(missKey)})
	join, err := NewHashJoinShared(sht, probe, 0)
	if err != nil {
		t.Fatalf("NewHashJoinShared: %v", err)
	}
	defer join.Close()
	if rows := drainJoin(t, join); len(rows) != 0 {
		t.Fatalf("probing an empty partition returned %v, want no rows", rows)
	}
}

// TestParallelBuildEmptyInputs covers the degenerate build sides: no row groups
// at all, row groups that yield no rows, and rows whose keys are all NULL. Each
// must produce a usable, empty table rather than an error or a nil map.
func TestParallelBuildEmptyInputs(t *testing.T) {
	ctx := context.Background()
	schema := Schema{Fields: []Field{
		{Name: "b_key", Type: TypeInt64, Nullable: true},
		{Name: "b_payload", Type: TypeInt64, Nullable: true},
	}}

	cases := []struct {
		name     string
		totalRGs int
		factory  PipelineFactory
	}{
		{
			name:     "no row groups",
			totalRGs: 0,
			factory: func(context.Context, int, int) (Operator, error) {
				return nil, fmt.Errorf("factory must not be called when there are no row groups")
			},
		},
		{
			name:     "row groups yield no batches",
			totalRGs: 4,
			factory: func(context.Context, int, int) (Operator, error) {
				return &sliceBatchOp{schema: schema}, nil
			},
		},
		{
			name:     "every key is NULL",
			totalRGs: 4,
			factory: radixBuildFixture{
				rowGroups: 4, rowsPerRG: 50,
				keyFor: func(int) *int64 { return nil },
			}.factory(),
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sht, err := BuildSharedHashTableParallel(ctx, c.factory, schema, 0, c.totalRGs, 4, 1, 6)
			if err != nil {
				t.Fatalf("parallel build: %v", err)
			}
			if got := sht.NumRows(); got != 0 {
				t.Errorf("rows = %d, want 0", got)
			}
			if got := sht.NumKeys(); got != 0 {
				t.Errorf("keys = %d, want 0", got)
			}
			if got := sht.NumPartitions(); got != 64 {
				t.Errorf("partitions = %d, want 64", got)
			}
			// Every partition must be a usable map, not nil.
			join, err := NewHashJoinShared(sht, probeSideOp([]*int64{ptr(1), ptr(2), ptr(3)}), 0)
			if err != nil {
				t.Fatalf("NewHashJoinShared: %v", err)
			}
			defer join.Close()
			if rows := drainJoin(t, join); len(rows) != 0 {
				t.Fatalf("probing an empty table returned %v, want no rows", rows)
			}
		})
	}
}

func TestParallelBuildKeyOutOfRange(t *testing.T) {
	f := radixBuildFixture{rowGroups: 1, rowsPerRG: 4, keyFor: func(int) *int64 { return ptr(1) }}
	if _, err := BuildSharedHashTableParallel(context.Background(), f.factory(), f.schema(), 9,
		1, 2, 1, 4); err == nil {
		t.Fatal("expected an error for an out-of-range build key")
	}
}

// TestParallelBuildPropagatesPipelineErrors checks that a failing morsel is
// reported rather than silently producing a short table, and that the remaining
// workers still finish so the builder does not deadlock waiting on them.
func TestParallelBuildPropagatesPipelineErrors(t *testing.T) {
	f := radixBuildFixture{rowGroups: 8, rowsPerRG: 100, keyFor: func(row int) *int64 { return ptr(int64(row)) }}
	good := f.factory()
	var mu sync.Mutex
	calls := 0
	failing := func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		if n == 1 {
			return nil, fmt.Errorf("synthetic morsel failure")
		}
		return good(ctx, rgStart, rgEnd)
	}
	_, err := BuildSharedHashTableParallel(context.Background(), failing, f.schema(), 0, 8, 4, 1, 4)
	if err == nil {
		t.Fatal("expected the morsel failure to surface")
	}
}

// TestPartitionedProbeFindsEveryBuildRow guards the invariant that would break
// silently rather than loudly if build and probe ever disagreed on the hash: a
// probe of every build key must return every build row, and probes of absent keys
// must return none.
func TestPartitionedProbeFindsEveryBuildRow(t *testing.T) {
	ctx := context.Background()
	const numKeys = 2000
	fixture := radixBuildFixture{
		rowGroups: 4,
		rowsPerRG: numKeys / 4,
		keyFor:    func(row int) *int64 { return ptr(int64(row) * 3) },
	}
	sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
		fixture.rowGroups, 4, 1, 6)
	if err != nil {
		t.Fatalf("parallel build: %v", err)
	}

	// Interleave present and absent keys; absent keys use *3+1 so they can never
	// collide with a build key.
	var probeKeys []*int64
	wantHits := 0
	for row := range numKeys {
		probeKeys = append(probeKeys, ptr(int64(row)*3))
		wantHits++
		probeKeys = append(probeKeys, ptr(int64(row)*3+1))
	}
	join, err := NewHashJoinShared(sht, probeSideOp(probeKeys), 0)
	if err != nil {
		t.Fatalf("NewHashJoinShared: %v", err)
	}
	defer join.Close()
	if got := len(drainJoin(t, join)); got != wantHits {
		t.Fatalf("join rows = %d, want %d — build and probe disagree on partitioning", got, wantHits)
	}
}

// TestPartitionedTableConcurrentProbe is the race-detector gate on the
// concurrency contract for a partitioned table: many goroutines probe one table
// built before any of them started, and every worker must see the complete table.
func TestPartitionedTableConcurrentProbe(t *testing.T) {
	ctx := context.Background()
	const numKeys = 4096
	fixture := radixBuildFixture{
		rowGroups: 8,
		rowsPerRG: numKeys / 8,
		keyFor:    func(row int) *int64 { return ptr(int64(row) + 1) },
	}
	sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
		fixture.rowGroups, 4, 1, 6)
	if err != nil {
		t.Fatalf("parallel build: %v", err)
	}

	const workers = 8
	found := make([]int, workers)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			var pk []*int64
			for k := w*numKeys/workers + 1; k <= (w+1)*numKeys/workers; k++ {
				pk = append(pk, ptr(int64(k)))
			}
			join, err := NewHashJoinShared(sht, probeSideOp(pk), 0)
			if err != nil {
				t.Errorf("worker %d: %v", w, err)
				return
			}
			defer join.Close()
			n := 0
			for {
				b, err := join.Next(ctx)
				if err != nil {
					t.Errorf("worker %d: Next: %v", w, err)
					return
				}
				if b == nil {
					break
				}
				// Payload is the global row index and key is index+1, so this
				// pairing check catches a row landing under the wrong key.
				for i := range b.Length {
					payload := b.Vectors[1].(*Int64Vector).Values[i]
					key := b.Vectors[2].(*Int64Vector).Values[i]
					if payload+1 != key {
						t.Errorf("worker %d: key %d paired with payload %d", w, key, payload)
						return
					}
				}
				n += b.Length
			}
			found[w] = n
		}(w)
	}
	wg.Wait()

	total := 0
	for w, n := range found {
		if n == 0 {
			t.Fatalf("worker %d found no rows", w)
		}
		total += n
	}
	if total != numKeys {
		t.Fatalf("total joined rows = %d, want %d", total, numKeys)
	}
}

// ---- Probe-side microbenchmark ----------------------------------------------

// BenchmarkPartitionedProbe isolates the probe from the build: the same build
// rows and the same probe keys, looked up through an unpartitioned table and
// through partitioned ones. It is the only measurement that attributes a
// difference to probe-side cache residency rather than to the parallel build,
// since the build is excluded from the timed region.
func BenchmarkPartitionedProbe(b *testing.B) {
	ctx := context.Background()
	const buildRows = 1 << 20
	fixture := radixBuildFixture{
		rowGroups: 16,
		rowsPerRG: buildRows / 16,
		// Sparse sequential keys, as TPC-H order keys are.
		keyFor: func(row int) *int64 { return ptr(int64(row) * 4) },
	}
	// Probe keys stride the key space pseudo-randomly so the access pattern is
	// not a sequential walk the prefetcher can hide.
	probeKeys := make([]*int64, BlockRows*16)
	for i := range probeKeys {
		probeKeys[i] = ptr(int64((i*2654435761)%buildRows) * 4)
	}

	for _, bits := range []int{0, 4, 6, 8} {
		sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
			fixture.rowGroups, 4, 1, bits)
		if err != nil {
			b.Fatalf("build: %v", err)
		}
		b.Run(fmt.Sprintf("partitions=%d", sht.NumPartitions()), func(b *testing.B) {
			for range b.N {
				join, err := NewHashJoinShared(sht, probeSideOp(probeKeys), 0)
				if err != nil {
					b.Fatalf("NewHashJoinShared: %v", err)
				}
				for {
					batch, err := join.Next(ctx)
					if err != nil {
						b.Fatalf("Next: %v", err)
					}
					if batch == nil {
						break
					}
				}
				_ = join.Close()
			}
		})
	}
}

// BenchmarkRadixBuild measures the build phase alone across worker counts, with
// and without partitioning, over a build side large enough to clear the
// partitioning threshold. partitions=1/workers=1 is the phase-1 baseline.
func BenchmarkRadixBuild(b *testing.B) {
	ctx := context.Background()
	fixture := radixBuildFixture{
		rowGroups: 16,
		rowsPerRG: storage.RowGroupRows / 4,
		keyFor:    func(row int) *int64 { return ptr(int64(row) * 4) },
	}
	for _, bits := range []int{0, 4, 6, 8} {
		for _, workers := range []int{1, 2, 4} {
			b.Run(fmt.Sprintf("partitions=%d/workers=%d", 1<<bits, workers), func(b *testing.B) {
				for range b.N {
					if _, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
						fixture.rowGroups, workers, 1, bits); err != nil {
						b.Fatalf("build: %v", err)
					}
				}
			})
		}
	}
}
