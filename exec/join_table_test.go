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

// ---- Capacity and load factor -----------------------------------------------

// TestJoinTableCapacity pins the sizing rule: a power of two, never below the
// floor, and always large enough that the requested key count sits at or under
// the load factor.
func TestJoinTableCapacity(t *testing.T) {
	for _, keys := range []int{-1, 0, 1, 5, 6, 7, 8, 100, 1000, 65536, 700_000} {
		got := joinTableCapacity(keys)
		if got < joinTableMinSlots {
			t.Errorf("joinTableCapacity(%d) = %d, want at least %d", keys, got, joinTableMinSlots)
		}
		if got&(got-1) != 0 {
			t.Errorf("joinTableCapacity(%d) = %d, want a power of two", keys, got)
		}
		if limit := got * joinTableLoadNum / joinTableLoadDen; keys > limit {
			t.Errorf("joinTableCapacity(%d) = %d, whose load-factor limit is %d — too small", keys, got, limit)
		}
		// Not wastefully large either: halving it must be too small (above the
		// floor, where the floor is not what decided the answer).
		if half := got / 2; half >= joinTableMinSlots {
			if limit := half * joinTableLoadNum / joinTableLoadDen; keys <= limit {
				t.Errorf("joinTableCapacity(%d) = %d, but %d would have held it", keys, got, half)
			}
		}
	}
}

// TestJoinTableAlwaysHasAnEmptySlot is the invariant lookup's termination rests
// on: growAt must stay strictly below capacity, so a table can never fill and a
// miss always runs into an empty slot.
func TestJoinTableAlwaysHasAnEmptySlot(t *testing.T) {
	for _, keys := range []int{0, 1, 8, 100, 10_000} {
		tbl := newJoinHashTable(newRowStore(oneColSchema(), 0), keys)
		if tbl.growAt >= len(tbl.slots) {
			t.Errorf("keys=%d: growAt = %d, capacity = %d — a full table would spin", keys, tbl.growAt, len(tbl.slots))
		}
	}
}

// TestJoinTablePresizeAvoidsRehash covers the parallel build's contract: the slot
// array is sized from pass 1's exact row count, so inserting that many keys must
// not resize it once.
func TestJoinTablePresizeAvoidsRehash(t *testing.T) {
	const keys = 5000
	store, rows := storeWithRows(t, keys)
	tbl := newJoinHashTable(store, keys)
	capBefore := len(tbl.slots)
	for i, row := range rows {
		tbl.insert(int64(i)*7, row)
	}
	if got := len(tbl.slots); got != capBefore {
		t.Errorf("capacity = %d after %d inserts, want %d — the presize did not hold", got, keys, capBefore)
	}
	if tbl.keys != keys {
		t.Errorf("keys = %d, want %d", tbl.keys, keys)
	}
}

// TestJoinTableGrowsAndKeepsChains drives the growth path — a table presized for
// nothing, filled with duplicates — and checks every chain survives the rehashes
// intact and in insertion order.
func TestJoinTableGrowsAndKeepsChains(t *testing.T) {
	const (
		distinct = 2000
		perKey   = 3
	)
	store, rows := storeWithRows(t, distinct*perKey)
	tbl := newJoinHashTable(store, 0) // no presize: forces repeated growth
	capBefore := len(tbl.slots)
	for i, row := range rows {
		tbl.insert(int64(i%distinct), row) // round-robin, so chains interleave
	}
	if len(tbl.slots) <= capBefore {
		t.Fatalf("capacity = %d, want growth beyond %d", len(tbl.slots), capBefore)
	}
	if tbl.keys != distinct {
		t.Errorf("keys = %d, want %d", tbl.keys, distinct)
	}
	if tbl.numRow != distinct*perKey {
		t.Errorf("rows = %d, want %d", tbl.numRow, distinct*perKey)
	}
	for k := range distinct {
		var got []int64
		for r := tbl.lookup(int64(k)); r != noRow; r = store.next[r] {
			got = append(got, store.value(r, 0))
		}
		want := make([]int64, perKey)
		for i := range perKey {
			want[i] = int64(i*distinct + k) // payload is the insertion index
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("key %d chain = %v, want %v", k, got, want)
		}
	}
}

// TestJoinTableDuplicateMultiplicities covers the duplicate-key design at the
// multiplicities that matter: one row, two rows, and many. Chain order must be
// insertion order in every case, because that is what keeps parallel integer
// aggregates bit-exact against serial.
func TestJoinTableDuplicateMultiplicities(t *testing.T) {
	mult := map[int64]int{100: 1, 200: 2, 300: 17}
	total := 0
	for _, n := range mult {
		total += n
	}
	store, rows := storeWithRows(t, total)
	tbl := newJoinHashTable(store, total)

	// Interleave the keys so each chain is built across the whole insert stream
	// rather than in one contiguous run.
	want := map[int64][]int64{}
	next := 0
	for i := range 17 {
		for _, key := range []int64{100, 200, 300} {
			if i >= mult[key] {
				continue
			}
			row := rows[next]
			next++
			tbl.insert(key, row)
			want[key] = append(want[key], store.value(row, 0))
		}
	}

	for _, key := range []int64{100, 200, 300} {
		var got []int64
		for r := tbl.lookup(key); r != noRow; r = store.next[r] {
			got = append(got, store.value(r, 0))
		}
		if !reflect.DeepEqual(got, want[key]) {
			t.Errorf("key %d chain = %v, want %v", key, got, want[key])
		}
		if len(got) != mult[key] {
			t.Errorf("key %d has %d rows, want %d", key, len(got), mult[key])
		}
	}
	if tbl.keys != len(mult) {
		t.Errorf("keys = %d, want %d", tbl.keys, len(mult))
	}
}

// TestJoinTableSentinelAndExtremeKeys is the reason emptiness is marked by a row
// index and not by a reserved key: 0 must be an ordinary key, and so must the
// int64 extremes.
func TestJoinTableSentinelAndExtremeKeys(t *testing.T) {
	keys := []int64{0, -1, 1, math.MinInt64, math.MaxInt64}
	store, rows := storeWithRows(t, len(keys))
	tbl := newJoinHashTable(store, len(keys))
	for i, k := range keys {
		tbl.insert(k, rows[i])
	}
	for i, k := range keys {
		r := tbl.lookup(k)
		if r == noRow {
			t.Fatalf("key %d not found", k)
		}
		if got, want := store.value(r, 0), int64(i); got != want {
			t.Errorf("key %d resolved to payload %d, want %d", k, got, want)
		}
		if store.next[r] != noRow {
			t.Errorf("key %d has a chain longer than its one row", k)
		}
	}
	for _, absent := range []int64{2, -2, 999, math.MaxInt64 - 1} {
		if r := tbl.lookup(absent); r != noRow {
			t.Errorf("lookup(%d) = %d, want noRow", absent, r)
		}
	}
}

// TestEmptyJoinTableLookupMisses covers the empty radix partition: probing it
// must miss cleanly rather than panic on an unallocated slot array.
func TestEmptyJoinTableLookupMisses(t *testing.T) {
	tbl := newJoinHashTable(newRowStore(oneColSchema(), 0), 0)
	for _, k := range []int64{0, 1, -7, math.MaxInt64} {
		if r := tbl.lookup(k); r != noRow {
			t.Errorf("lookup(%d) on an empty table = %d, want noRow", k, r)
		}
	}
}

// ---- Row store --------------------------------------------------------------

// TestRowStoreRoundTripsEveryType materialises a row of each supported column
// type — including NULLs — and reads it back, which is the contract
// buildColumnFromRows relies on.
func TestRowStoreRoundTripsEveryType(t *testing.T) {
	schema, batch := mixedTypeBuildBatch()
	store := newRowStore(schema, 0)
	rows := make([]int32, batch.Length)
	for i := range batch.Length {
		row, err := store.appendFrom(batch, i)
		if err != nil {
			t.Fatalf("appendFrom(%d): %v", i, err)
		}
		rows[i] = row
	}

	// Row 0: every column populated.
	if got, want := store.value(rows[0], 0), int64(7); got != want {
		t.Errorf("row0 key = %d, want %d", got, want)
	}
	if got, want := store.str(rows[0], 1), "alpha"; got != want {
		t.Errorf("row0 string = %q, want %q", got, want)
	}
	if got, want := math.Float64frombits(uint64(store.value(rows[0], 2))), 1.5; got != want {
		t.Errorf("row0 float = %v, want %v", got, want)
	}
	if got, want := int32(store.value(rows[0], 3)), int32(9000); got != want {
		t.Errorf("row0 date = %d, want %d", got, want)
	}

	// Row 1: every column NULL except the key.
	for col := 1; col < len(schema.Fields); col++ {
		if !store.isNull(rows[1], col) {
			t.Errorf("row1 col %d reports non-NULL", col)
		}
	}
	if store.isNull(rows[1], 0) {
		t.Error("row1 key reports NULL")
	}

	// Row 2: a different string, to prove the string window is per row.
	if got, want := store.str(rows[2], 1), "beta"; got != want {
		t.Errorf("row2 string = %q, want %q", got, want)
	}

	// The string window holds one entry per row, not one per column: the schema
	// has four columns of which one is a string.
	if got, want := len(store.strs), batch.Length*1; got < want {
		t.Errorf("string window = %d entries, want at least %d", got, want)
	}
	if got := len(store.strs); got > batch.Length*len(schema.Fields)/2 {
		t.Errorf("string window = %d entries for %d rows — not packed to string columns", got, batch.Length)
	}
}

// TestRowStoreGrowthPreservesRows appends past every internal growth step and
// checks nothing shifts: a row index handed out early must still resolve to the
// same payload at the end.
func TestRowStoreGrowthPreservesRows(t *testing.T) {
	const n = 5000
	store, rows := storeWithRows(t, n)
	if store.rows() != n {
		t.Fatalf("rows = %d, want %d", store.rows(), n)
	}
	for i, row := range rows {
		if got, want := store.value(row, 0), int64(i); got != want {
			t.Fatalf("row %d = %d, want %d", i, got, want)
		}
	}
}

// TestRowStoreSetFromCopiesWholeRow covers the parallel build's pass 2, which
// moves rows from a morsel store into the final store by index.
func TestRowStoreSetFromCopiesWholeRow(t *testing.T) {
	schema, batch := mixedTypeBuildBatch()
	src := newRowStore(schema, 0)
	for i := range batch.Length {
		if _, err := src.appendFrom(batch, i); err != nil {
			t.Fatalf("appendFrom: %v", err)
		}
	}
	dst := newSizedRowStore(schema, batch.Length)
	// Copy in reverse, so a positional bug cannot pass by accident.
	for i := range batch.Length {
		dst.setFrom(int32(batch.Length-1-i), src, int32(i))
	}
	for i := range batch.Length {
		s, d := int32(i), int32(batch.Length-1-i)
		for col := range schema.Fields {
			if src.isNull(s, col) != dst.isNull(d, col) {
				t.Errorf("row %d col %d: NULL flag not copied", i, col)
			}
			if src.value(s, col) != dst.value(d, col) {
				t.Errorf("row %d col %d: value not copied", i, col)
			}
			if src.str(s, col) != dst.str(d, col) {
				t.Errorf("row %d col %d: string not copied", i, col)
			}
		}
	}
}

// ---- Serial HashJoin semantics ----------------------------------------------

// TestSerialHashJoinSemantics is the end-to-end statement of the semantics bar
// for the converted serial path: duplicate build keys yield every match in build
// order, key 0 joins like any other key, NULL keys on either side never match,
// and a probe key with no build row produces nothing.
func TestSerialHashJoinSemantics(t *testing.T) {
	// Build keys: 0 twice, 5 three times, 7 once, one NULL-keyed row.
	buildBatch, buildSchema := buildSideBatch(
		[]*int64{ptr(0), ptr(5), nil, ptr(5), ptr(0), ptr(7), ptr(5)},
		[]int64{10, 11, 12, 13, 14, 15, 16},
	)
	build := &sliceBatchOp{schema: buildSchema, batches: []*Batch{buildBatch}}
	probe := probeSideOp([]*int64{ptr(5), ptr(0), nil, ptr(99), ptr(7)})

	join, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	defer join.Close()

	got := drainJoin(t, join)
	want := []string{
		"11:5", "13:5", "16:5", // duplicate key, build order preserved
		"10:0", "14:0", // key 0 is an ordinary key
		"15:7",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("join output = %v, want %v", got, want)
	}
}

// TestSerialHashJoinEmptySides covers the degenerate shapes: no build rows, no
// probe rows, and a build side whose every key is NULL.
func TestSerialHashJoinEmptySides(t *testing.T) {
	schemaFor := func(keys []*int64, payload []int64) (*Batch, Schema) {
		return buildSideBatch(keys, payload)
	}
	cases := []struct {
		name    string
		bKeys   []*int64
		bPay    []int64
		pKeys   []*int64
		wantLen int
	}{
		{"empty build", nil, nil, []*int64{ptr(1), ptr(2)}, 0},
		{"empty probe", []*int64{ptr(1)}, []int64{10}, nil, 0},
		{"all build keys NULL", []*int64{nil, nil}, []int64{10, 11}, []*int64{ptr(1)}, 0},
		{"all probe keys NULL", []*int64{ptr(1)}, []int64{10}, []*int64{nil, nil}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bBatch, bSchema := schemaFor(c.bKeys, c.bPay)
			build := &sliceBatchOp{schema: bSchema, batches: []*Batch{bBatch}}
			join, err := NewHashJoin(build, probeSideOp(c.pKeys), 0, 0)
			if err != nil {
				t.Fatalf("NewHashJoin: %v", err)
			}
			defer join.Close()
			if got := drainJoin(t, join); len(got) != c.wantLen {
				t.Errorf("output = %v, want %d rows", got, c.wantLen)
			}
		})
	}
}

// TestSerialHashJoinEmitsEveryColumnType joins a build side of every column type
// and reads the emitted vectors back, which exercises buildColumnFromRows over
// the flat store — the path that turns stored rows back into output columns.
func TestSerialHashJoinEmitsEveryColumnType(t *testing.T) {
	schema, batch := mixedTypeBuildBatch()
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}
	// Probe every build key once, in build order.
	join, err := NewHashJoin(build, probeSideOp([]*int64{ptr(7), ptr(8), ptr(9)}), 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	defer join.Close()

	out, err := join.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if out == nil || out.Length != 3 {
		t.Fatalf("output batch = %v, want 3 rows", out)
	}
	strs := out.Vectors[1].(*StringVector)
	floats := out.Vectors[2].(*Float64Vector)
	dates := out.Vectors[3].(*DateVector)

	if got, want := strs.Get(0), "alpha"; got != want {
		t.Errorf("row0 string = %q, want %q", got, want)
	}
	if got, want := strs.Get(2), "beta"; got != want {
		t.Errorf("row2 string = %q, want %q", got, want)
	}
	if got, want := floats.Values[0], 1.5; got != want {
		t.Errorf("row0 float = %v, want %v", got, want)
	}
	if got, want := dates.Values[0], int32(9000); got != want {
		t.Errorf("row0 date = %d, want %d", got, want)
	}
	// Row 1 is NULL in every non-key column.
	for col := 1; col < len(schema.Fields); col++ {
		if !out.Vectors[col].IsNull(1) {
			t.Errorf("row1 col %d should be NULL", col)
		}
	}
}

// TestSharedJoinOutputMatchesSerialJoin is the parity statement across build
// strategies: the same build side, materialised serially, by the serial radix
// builder, and by the morsel-parallel builder at 1/2/4/8 workers, must produce
// the identical joined row sequence — not merely the same rows.
func TestSharedJoinOutputMatchesSerialJoin(t *testing.T) {
	ctx := context.Background()
	fixture := radixBuildFixture{
		rowGroups: 5,
		rowsPerRG: 200,
		keyFor: func(row int) *int64 {
			switch {
			case row%23 == 0:
				return nil // NULL key
			case row%3 == 0:
				return ptr(int64(row % 7)) // heavy duplicates, includes key 0
			default:
				return ptr(int64(row) * 4)
			}
		},
	}
	// Probe keys: some present (duplicated and unique), some absent, one NULL.
	probeKeys := []*int64{ptr(0), ptr(1), ptr(6), nil, ptr(4), ptr(400), ptr(999_999), ptr(0)}

	serialJoin, err := NewHashJoin(fixture.serialOp(), probeSideOp(probeKeys), 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	want := drainJoin(t, serialJoin)
	_ = serialJoin.Close()
	if len(want) == 0 {
		t.Fatal("fixture produced no matches — it cannot detect a regression")
	}

	for _, bits := range []int{0, 4, 6} {
		sht, err := BuildSharedHashTableRadix(ctx, fixture.serialOp(), 0, bits)
		if err != nil {
			t.Fatalf("serial radix build (bits=%d): %v", bits, err)
		}
		assertSharedJoinOutput(t, sht, probeKeys, want, fmt.Sprintf("serial-radix/bits=%d", bits))

		for _, workers := range []int{1, 2, 4, 8} {
			sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
				fixture.rowGroups, workers, 1, bits)
			if err != nil {
				t.Fatalf("parallel build (bits=%d workers=%d): %v", bits, workers, err)
			}
			assertSharedJoinOutput(t, sht, probeKeys, want,
				fmt.Sprintf("parallel/bits=%d/workers=%d", bits, workers))
		}
	}
}

func assertSharedJoinOutput(t *testing.T, sht *SharedHashTable, probeKeys []*int64, want []string, label string) {
	t.Helper()
	join, err := NewHashJoinShared(sht, probeSideOp(probeKeys), 0)
	if err != nil {
		t.Fatalf("%s: NewHashJoinShared: %v", label, err)
	}
	defer join.Close()
	if got := drainJoin(t, join); !reflect.DeepEqual(got, want) {
		t.Errorf("%s: joined output differs from the serial join\n got %v\nwant %v", label, got, want)
	}
}

// TestSharedTableRowsAreOneAllocation asserts the layout claim the whole change
// rests on: every build row of a partitioned table lives in one store, so a row
// index is table-wide and no partition owns rows of its own.
func TestSharedTableRowsAreOneAllocation(t *testing.T) {
	ctx := context.Background()
	fixture := radixBuildFixture{
		rowGroups: 4,
		rowsPerRG: 128,
		keyFor:    func(row int) *int64 { return ptr(int64(row) * 3) },
	}
	sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
		fixture.rowGroups, 4, 1, 4)
	if err != nil {
		t.Fatalf("parallel build: %v", err)
	}
	if got, want := sht.store.rows(), fixture.rowGroups*fixture.rowsPerRG; got != want {
		t.Errorf("store rows = %d, want %d", got, want)
	}
	// Every partition indexes the same store, and the partitions together cover
	// every row exactly once.
	seen := make([]bool, sht.store.rows())
	for _, tbl := range sht.parts {
		if tbl.store != sht.store {
			t.Fatal("a partition holds its own store — rows are not contiguous")
		}
		tbl.forEachKey(func(_ int64, head int32) {
			for r := head; r != noRow; r = sht.store.next[r] {
				if seen[r] {
					t.Fatalf("row %d reachable from two chains", r)
				}
				seen[r] = true
			}
		})
	}
	for r, ok := range seen {
		if !ok {
			t.Fatalf("row %d is not reachable from any key", r)
		}
	}
}

// TestPartitionedTableDoesNotClusterSlots is the regression test for the bit
// split between radixPart and slotFor. Every key inside a radix partition shares
// its low hash bits by construction, so indexing slots from those same low bits
// would crowd a 64-partition table's keys onto one slot in 64 and turn each probe
// into a long linear scan. Correctness would survive it; throughput would not, so
// the guard is on probe distance rather than on results.
func TestPartitionedTableDoesNotClusterSlots(t *testing.T) {
	ctx := context.Background()
	const (
		rowGroups = 8
		rowsPerRG = 4096
		bits      = 6 // 64 partitions
	)
	fixture := radixBuildFixture{
		rowGroups: rowGroups,
		rowsPerRG: rowsPerRG,
		// Sparse sequential keys: TPC-H order keys with dbgen's gaps.
		keyFor: func(row int) *int64 { return ptr(int64(row) * 8) },
	}
	sht, err := BuildSharedHashTableParallel(ctx, fixture.factory(), fixture.schema(), 0,
		rowGroups, 4, 1, bits)
	if err != nil {
		t.Fatalf("parallel build: %v", err)
	}

	totalSteps, worst, n := 0, 0, 0
	for row := range rowGroups * rowsPerRG {
		key := int64(row) * 8
		tbl := sht.parts[radixPart(key, sht.partMask)]
		steps := probeSteps(tbl, key)
		if steps < 0 {
			t.Fatalf("key %d not found", key)
		}
		totalSteps += steps
		if steps > worst {
			worst = steps
		}
		n++
	}
	mean := float64(totalSteps) / float64(n)
	// Linear probing at a load factor of 0.7 examines ~2.2 slots per successful
	// lookup, so ~1.2 steps past the first. 3 is generous headroom; the clustered
	// failure mode measures in the tens.
	if mean > 3 {
		t.Errorf("mean probe distance = %.2f over %d keys, want <= 3 — slots are clustering", mean, n)
	}
	if worst > 200 {
		t.Errorf("worst probe distance = %d, want <= 200", worst)
	}
}

// probeSteps counts how many slots past the first a lookup of key examines, or -1
// when the key is absent.
func probeSteps(t *joinHashTable, key int64) int {
	i := t.slotFor(key)
	for steps := 0; ; steps++ {
		s := &t.slots[i]
		if s.head == noRow {
			return -1
		}
		if s.key == key {
			return steps
		}
		i = (i + 1) & t.mask
	}
}

// ---- Row store growth policy ------------------------------------------------

// TestNextRowCap pins the growth policy: double, floored, never short of what was
// asked for, and never past the largest row an int32 index can address.
func TestNextRowCap(t *testing.T) {
	cases := []struct {
		cur, need, want int
	}{
		{0, 1, rowStoreMinRows},                                     // first append jumps to the floor
		{0, rowStoreMinRows + 1, rowStoreMinRows + 1},               // ... unless more is needed at once
		{rowStoreMinRows, rowStoreMinRows + 1, 2 * rowStoreMinRows}, // then doubling takes over
		{1000, 1001, 2000},
		{1 << 20, (1 << 20) + 1, 1 << 21},
		// Near the ceiling: doubling is clamped, and the clamp still satisfies
		// need, because reserve refuses to ask for more than maxBuildRows rows.
		{int(maxBuildRows) - 10, int(maxBuildRows) - 9, int(maxBuildRows)},
		{int(maxBuildRows), int(maxBuildRows), int(maxBuildRows)},
	}
	for _, c := range cases {
		got := nextRowCap(c.cur, c.need)
		if got != c.want {
			t.Errorf("nextRowCap(%d, %d) = %d, want %d", c.cur, c.need, got, c.want)
		}
		if got < c.need {
			t.Errorf("nextRowCap(%d, %d) = %d, short of need", c.cur, c.need, got)
		}
		if int64(got) > maxBuildRows {
			t.Errorf("nextRowCap(%d, %d) = %d, past maxBuildRows", c.cur, c.need, got)
		}
	}
}

// TestRowStoreGrowthDoubles pins the mechanism the allocation win rests on: the
// row capacity doubles rather than creeping, so filling N rows costs O(log N)
// reallocations, and a store presized to N never reallocates at all.
func TestRowStoreGrowthDoubles(t *testing.T) {
	schema, batch := mixedTypeBuildBatch()
	store := newRowStore(schema, 0)
	var caps []int
	for range 200 {
		before := store.capRows
		if _, err := store.appendFrom(batch, 0); err != nil {
			t.Fatalf("appendFrom: %v", err)
		}
		if store.capRows != before {
			caps = append(caps, store.capRows)
		}
	}
	want := []int{8, 16, 32, 64, 128, 256}
	if !reflect.DeepEqual(caps, want) {
		t.Errorf("capacity steps = %v, want %v", caps, want)
	}

	presized := newRowStore(schema, 200)
	for range 200 {
		if _, err := presized.appendFrom(batch, 0); err != nil {
			t.Fatalf("appendFrom: %v", err)
		}
	}
	if presized.capRows != 200 {
		t.Errorf("presized store reallocated: capRows = %d, want 200", presized.capRows)
	}
}

// TestRowStoreGrowthMatchesAppendGrowth is the equivalence statement for the
// growth rewrite: the doubling policy must leave the arrays exactly what the
// previous append-per-row policy left them — same lengths, same payloads at the
// same indices, and the same zero fill where nothing has been written.
//
// Both stores are filled with the same real rows through the same writeRow, so
// the only difference between them is the growth policy. That is what makes this
// an equivalence test rather than a length check.
func TestRowStoreGrowthMatchesAppendGrowth(t *testing.T) {
	schema, batch := mixedTypeBuildBatch()
	// Row counts either side of every doubling boundary, so a policy that grew
	// by too little or reslice arithmetic that was off by a row would show up.
	for _, rows := range []int{0, 1, 2, 7, 8, 9, 15, 16, 17, 100, 255, 256, 257, 1000} {
		got := newRowStore(schema, 0)  // doubling: the policy under test
		want := newRowStore(schema, 0) // append-per-row: the previous policy
		for i := range rows {
			src := i % batch.Length
			row, err := got.appendFrom(batch, src)
			if err != nil {
				t.Fatalf("rows=%d: appendFrom: %v", rows, err)
			}
			appendGrowOneRow(want)
			want.writeRow(row, batch, src)
		}
		if !reflect.DeepEqual(got.values, want.values) {
			t.Errorf("rows=%d: values differ (len %d vs %d)", rows, len(got.values), len(want.values))
		}
		if !reflect.DeepEqual(got.nulls, want.nulls) {
			t.Errorf("rows=%d: nulls differ (len %d vs %d)", rows, len(got.nulls), len(want.nulls))
		}
		if !reflect.DeepEqual(got.strs, want.strs) {
			t.Errorf("rows=%d: strs differ (len %d vs %d)", rows, len(got.strs), len(want.strs))
		}
		if !reflect.DeepEqual(got.next, want.next) {
			t.Errorf("rows=%d: next differs (len %d vs %d)", rows, len(got.next), len(want.next))
		}
		if got.rows() != want.rows() || got.rows() != rows {
			t.Errorf("rows=%d: row counts %d and %d", rows, got.rows(), want.rows())
		}
	}
}

// TestRowStoreReservedRowIsZero covers the invariant appendFrom depends on: a row
// the growth path has just exposed reads as an all-NULL-free row of zeros, with
// "" in its string slot, before anything writes to it.
func TestRowStoreReservedRowIsZero(t *testing.T) {
	schema, _ := mixedTypeBuildBatch()
	store := newRowStore(schema, 0)
	// Reserve across several doublings; check every row, because a growth path
	// that failed to zero would most likely do so only for the copied region.
	for i := range 300 {
		row, err := store.reserve()
		if err != nil {
			t.Fatalf("reserve: %v", err)
		}
		if int(row) != i {
			t.Fatalf("reserve returned row %d, want %d", row, i)
		}
		for col := range schema.Fields {
			if store.isNull(row, col) {
				t.Errorf("row %d col %d: fresh row reports NULL, want NULL-free", row, col)
			}
			if v := store.value(row, col); v != 0 {
				t.Errorf("row %d col %d: fresh row value = %d, want 0", row, col, v)
			}
			if s := store.str(row, col); s != "" {
				t.Errorf("row %d col %d: fresh row string = %q, want empty", row, col, s)
			}
		}
		if store.next[row] != 0 {
			t.Errorf("row %d: fresh next = %d, want 0", row, store.next[row])
		}
	}
}

// TestRowStoreGrowthKeepsEarlierRows crosses many growth steps with a multi-column
// schema that includes a string column — the string window is the array whose
// per-row width differs from numCols, so it is where growth arithmetic is most
// likely to go wrong — and checks every earlier row still resolves to its own
// payload at its original index.
func TestRowStoreGrowthKeepsEarlierRows(t *testing.T) {
	const n = 3000
	schema, batch := mixedTypeBuildBatch()
	store := newRowStore(schema, 0)
	rows := make([]int32, n)
	for i := range n {
		row, err := store.appendFrom(batch, i%batch.Length)
		if err != nil {
			t.Fatalf("appendFrom(%d): %v", i, err)
		}
		rows[i] = row
	}
	for i, row := range rows {
		src := i % batch.Length
		if got, want := store.value(row, 0), int64(7+src); got != want {
			t.Fatalf("row %d key = %d, want %d", i, got, want)
		}
		wantStr := []string{"alpha", "", "beta"}[src]
		if got := store.str(row, 1); got != wantStr {
			t.Fatalf("row %d string = %q, want %q", i, got, wantStr)
		}
		// Row 1 of the fixture is NULL in every non-key column.
		for col := 1; col < len(schema.Fields); col++ {
			if got, want := store.isNull(row, col), src == 1; got != want {
				t.Fatalf("row %d col %d NULL = %v, want %v", i, col, got, want)
			}
		}
	}
}

// TestRowStoreOverflowFiresBeforeAllocating pins maxBuildRows: the error fires at
// the same threshold as before, and it fires before the growth path allocates —
// so a store at the ceiling cannot be made to attempt a doubled allocation.
//
// There is deliberately no positive case one row below the ceiling: accepting
// that row means growing to maxBuildRows rows, which is tens of gigabytes. The
// threshold is pinned by the constant below and the refusal above; that the
// growth policy respects the same ceiling is TestNextRowCap's last two cases.
func TestRowStoreOverflowFiresBeforeAllocating(t *testing.T) {
	if maxBuildRows != int64(math.MaxInt32) {
		t.Errorf("maxBuildRows = %d, want %d — the overflow threshold moved", maxBuildRows, math.MaxInt32)
	}

	store := newRowStore(oneColSchema(), 0)
	// Fake a full store rather than materialising 2^31 rows: reserve reads only
	// s.n to decide, which is exactly the behaviour under test.
	store.n = int32(maxBuildRows)
	capBefore, lenBefore := store.capRows, len(store.values)

	row, err := store.reserve()
	if err == nil {
		t.Fatal("reserve at maxBuildRows returned no error")
	}
	if row != noRow {
		t.Errorf("reserve returned row %d, want noRow", row)
	}
	if store.n != int32(maxBuildRows) {
		t.Errorf("n = %d after a refused reserve, want %d", store.n, maxBuildRows)
	}
	if store.capRows != capBefore || len(store.values) != lenBefore {
		t.Errorf("refused reserve allocated: capRows %d->%d, len %d->%d",
			capBefore, store.capRows, lenBefore, len(store.values))
	}
}

// TestSizedRowStoreConcurrentSetFromTouchesNoSharedState is the parallel build's
// pass-2 contract. A sized store must need no growth and hold no cursor, so
// several goroutines can fill disjoint index ranges at once. Run under -race this
// fails if the growth rewrite introduced any shared mutable state.
func TestSizedRowStoreConcurrentSetFromTouchesNoSharedState(t *testing.T) {
	const (
		workers = 8
		perWork = 250
		total   = workers * perWork
	)
	schema, batch := mixedTypeBuildBatch()
	src := newRowStore(schema, 0)
	for i := range total {
		if _, err := src.appendFrom(batch, i%batch.Length); err != nil {
			t.Fatalf("appendFrom: %v", err)
		}
	}

	dst := newSizedRowStore(schema, total)
	capBefore := dst.capRows
	lens := [4]int{len(dst.values), len(dst.nulls), len(dst.strs), len(dst.next)}

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := w * perWork; i < (w+1)*perWork; i++ {
				dst.setFrom(int32(i), src, int32(i))
			}
		}()
	}
	wg.Wait()

	if dst.capRows != capBefore {
		t.Errorf("capRows moved during concurrent setFrom: %d -> %d", capBefore, dst.capRows)
	}
	if got := [4]int{len(dst.values), len(dst.nulls), len(dst.strs), len(dst.next)}; got != lens {
		t.Errorf("array lengths moved during concurrent setFrom: %v -> %v", lens, got)
	}
	for i := range total {
		r := int32(i)
		for col := range schema.Fields {
			if src.value(r, col) != dst.value(r, col) || src.str(r, col) != dst.str(r, col) ||
				src.isNull(r, col) != dst.isNull(r, col) {
				t.Fatalf("row %d col %d not copied", i, col)
			}
		}
	}
}

// appendGrowOneRow grows a store by one row the way reserve did before this
// change: extend each array with append(sl, make([]T, k)...), letting append pick
// the growth factor. Kept as a test fixture because the growth rewrite's whole
// claim is that it is equivalent to this but allocates less, and both halves of
// that claim need this shape to compare against — the equivalence test above and
// BenchmarkRowStoreGrowth below.
//
// FROZEN: this is a replica of the pre-rewrite growth path, not a second
// implementation of the current one. Do not "fix" it to match reserve — that
// would delete the comparison. It does need updating if rowStore gains another
// per-row array, so that it keeps growing everything reserve grows.
func appendGrowOneRow(s *rowStore) {
	s.n++
	if need := int(s.n) * s.numCols; len(s.values) < need {
		s.values = append(s.values, make([]int64, need-len(s.values))...)
		s.nulls = append(s.nulls, make([]bool, need-len(s.nulls))...)
	}
	if s.strWidth > 0 {
		if need := int(s.n) * s.strWidth; len(s.strs) < need {
			s.strs = append(s.strs, make([]string, need-len(s.strs))...)
		}
	}
	if len(s.next) < int(s.n) {
		s.next = append(s.next, make([]int32, int(s.n)-len(s.next))...)
	}
}

// BenchmarkRowStoreGrowth is the measurement the growth design rests on. It
// separates the two candidate causes of reserve's 41% share of join-path bytes:
//
//	append-from-zero   the previous policy — one append(sl, make([]T, k)...) per
//	                   row per array, growing from no capacity at all
//	doubling-from-zero this policy — explicit make+copy at twice the row capacity
//	presized           the ceiling — a caller-supplied row count, no growth at all
//
// Read the allocs/op of append-from-zero against the row count: a per-row make
// temporary would put it at ~4 allocations per row. It does not, which refutes
// the temporary as a cause (the compiler's extendslice optimisation turns the
// pattern into growslice+memclr — confirmed in the generated assembly, no
// runtime.makeslice call in reserve). What is left is the growth factor: append
// grows a large slice by ~1.25x, so it churns ~5x the final payload, where
// doubling churns ~2x and presizing churns 1x.
func BenchmarkRowStoreGrowth(b *testing.B) {
	const rows = 200_000
	schema, batch := mixedTypeBuildBatch()

	b.Run("append-from-zero", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			s := newRowStore(schema, 0)
			for range rows {
				appendGrowOneRow(s)
			}
		}
	})
	b.Run("doubling-from-zero", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			s := newRowStore(schema, 0)
			for range rows {
				if _, err := s.reserve(); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("presized", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			s := newRowStore(schema, rows)
			for range rows {
				if _, err := s.reserve(); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	// The same three policies with real row payloads, so the growth cost is
	// measured against the work it carries rather than in isolation.
	b.Run("appendFrom-doubling", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			s := newRowStore(schema, 0)
			for i := range rows {
				if _, err := s.appendFrom(batch, i%batch.Length); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("appendFrom-presized", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			s := newRowStore(schema, rows)
			for i := range rows {
				if _, err := s.appendFrom(batch, i%batch.Length); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

// ---- Fixtures ---------------------------------------------------------------

func oneColSchema() Schema {
	return Schema{Fields: []Field{{Name: "payload", Type: TypeInt64, Nullable: true}}}
}

// storeWithRows returns a one-column store holding n rows whose only value is
// the row's ordinal, plus the row indices in append order.
func storeWithRows(t *testing.T, n int) (*rowStore, []int32) {
	t.Helper()
	schema := oneColSchema()
	vals := make([]*int64, n)
	for i := range vals {
		vals[i] = ptr(int64(i))
	}
	batch := &Batch{Schema: schema, Vectors: []Vector{int64Vec(vals)}, Length: n}
	store := newRowStore(schema, 0)
	rows := make([]int32, n)
	for i := range n {
		row, err := store.appendFrom(batch, i)
		if err != nil {
			t.Fatalf("appendFrom(%d): %v", i, err)
		}
		rows[i] = row
	}
	return store, rows
}

// mixedTypeBuildBatch returns a three-row build side covering every column type
// the store handles, with row 1 NULL in every non-key column.
func mixedTypeBuildBatch() (Schema, *Batch) {
	schema := Schema{Fields: []Field{
		{Name: "b_key", Type: TypeInt64, Nullable: true},
		{Name: "b_str", Type: TypeString, Nullable: true},
		{Name: "b_float", Type: TypeFloat64, Nullable: true},
		{Name: "b_date", Type: TypeDate, Nullable: true},
	}}
	const n = 3

	keys := int64Vec([]*int64{ptr(7), ptr(8), ptr(9)})

	db := storage.NewDictBuilder()
	codes := make([]uint32, n)
	strNulls := make([]byte, (n+7)/8)
	codes[0] = db.Add("alpha")
	storage.SetValidBit(strNulls, 0)
	codes[2] = db.Add("beta")
	storage.SetValidBit(strNulls, 2)
	strs := newStringVector(db, codes, strNulls)

	fNulls := make([]byte, (n+7)/8)
	storage.SetValidBit(fNulls, 0)
	storage.SetValidBit(fNulls, 2)
	floats := &Float64Vector{Values: []float64{1.5, 0, -2.25}, NullBitmap: fNulls}

	dNulls := make([]byte, (n+7)/8)
	storage.SetValidBit(dNulls, 0)
	storage.SetValidBit(dNulls, 2)
	dates := &DateVector{Values: []int32{9000, 0, 9001}, NullBitmap: dNulls}

	return schema, &Batch{
		Schema:  schema,
		Vectors: []Vector{keys, strs, floats, dates},
		Length:  n,
	}
}
