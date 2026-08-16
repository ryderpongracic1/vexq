package exec

import (
	"context"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// These tests cover presizing the join's build-side rowStore from a row count the
// build side can predict (row_bound.go). The property under test is the one
// presizing exists for — a store told its row count up front never grows — plus
// the three ways the count can be unavailable, each of which must fall back to
// growth rather than to a bad allocation:
//
//	filter above the build scan   loose bound, deliberately ignored
//	no bound at all               operator does not implement RowCountBound
//	bound above maxBuildRows      refused, so the overflow error still fires
//
// rowStore.grows is what makes "never grows" directly assertable; a capRows value
// would only imply it.

// ---- Scan and filter bounds -------------------------------------------------

func TestTableScanRowCountBoundIsExactPerRange(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)

	cases := []struct {
		name           string
		rgStart, rgEnd int
		want           int
	}{
		{name: "whole file", rgStart: 0, rgEnd: 6, want: 6 * 128},
		{name: "one row group", rgStart: 2, rgEnd: 3, want: 128},
		{name: "two row groups", rgStart: 1, rgEnd: 3, want: 256},
		{name: "empty range", rgStart: 3, rgEnd: 3, want: 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			op, err := fx.scanFactory(t, nil)(context.Background(), c.rgStart, c.rgEnd)
			if err != nil {
				t.Fatalf("scan: %v", err)
			}
			defer func() { _ = op.Close() }()

			scan, ok := op.(*TableScan)
			if !ok {
				t.Fatalf("scan factory returned %T, want *TableScan", op)
			}
			rows, tight := scan.RowCountBound()
			if rows != c.want || !tight {
				t.Errorf("RowCountBound() = (%d, %v), want (%d, true)", rows, tight, c.want)
			}
			// The bound has to be the rows the scan actually yields, not just a
			// number derived from the footer, so drain and compare.
			if got := drainRowCount(t, scan); got != c.want {
				t.Errorf("scan produced %d rows, bound said %d", got, c.want)
			}
		})
	}
}

// TestTableScanRowCountBoundFollowsReset pins the property the parallel build
// depends on: a pipeline reused across morsels reports the morsel it is on. A
// bound read from the range the pipeline was *built* with would presize every
// later morsel from the first one's row count.
func TestTableScanRowCountBoundFollowsReset(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)
	op, err := fx.scanFactory(t, nil)(context.Background(), 0, 1)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	defer func() { _ = op.Close() }()
	scan, ok := op.(*TableScan)
	if !ok {
		t.Fatalf("scan factory returned %T, want *TableScan", op)
	}

	if rows, _ := scan.RowCountBound(); rows != 128 {
		t.Fatalf("initial RowCountBound() = %d, want 128", rows)
	}
	scan.Reset(2, 5)
	if rows, tight := scan.RowCountBound(); rows != 3*128 || !tight {
		t.Errorf("after Reset(2,5) RowCountBound() = (%d, %v), want (384, true)", rows, tight)
	}
	if got := drainRowCount(t, scan); got != 3*128 {
		t.Errorf("after Reset(2,5) scan produced %d rows, want 384", got)
	}
}

// TestTableScanRowCountBoundExcludesPrunedRowGroups checks that the bound counts
// what the scan will produce rather than what the file holds. A build side whose
// predicate prunes most of its row groups would otherwise be presized for the
// whole table — the largest over-allocation this mechanism could make.
func TestTableScanRowCountBoundExcludesPrunedRowGroups(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)
	r, err := storage.Open(context.Background(), fx.path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Keep only row groups whose id range starts below 256 — the first two.
	keep := func(rg *storage.RowGroupMeta) bool {
		return int64(rg.Columns[0].Stats.Min) < 256
	}
	scan, err := NewTableScanRange(r, nil, keep, 0, 6)
	if err != nil {
		_ = r.Close()
		t.Fatalf("NewTableScanRange: %v", err)
	}
	defer func() { _ = scan.Close() }()

	rows, tight := scan.RowCountBound()
	if rows != 2*128 || !tight {
		t.Errorf("RowCountBound() = (%d, %v), want (256, true)", rows, tight)
	}
	if got := drainRowCount(t, scan); got != rows {
		t.Errorf("scan produced %d rows, bound said %d", got, rows)
	}
}

func TestFilterRowCountBoundIsNotTight(t *testing.T) {
	fx := newMorselFixture(t, 4, 128)
	op, err := fx.filterProjectFactory(t, 100, nil)(context.Background(), 0, 4)
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer func() { _ = op.Close() }()

	// Project does not bound its output, so ask the Filter underneath it.
	proj, ok := op.(*Project)
	if !ok {
		t.Fatalf("pipeline top = %T, want *Project", op)
	}
	filter, ok := proj.child.(*Filter)
	if !ok {
		t.Fatalf("under the Project = %T, want *Filter", proj.child)
	}
	rows, tight := filter.RowCountBound()
	if rows != 4*128 {
		t.Errorf("Filter.RowCountBound() rows = %d, want the child's 512", rows)
	}
	if tight {
		t.Error("Filter.RowCountBound() reported a tight bound; a predicate's selectivity is unknown here")
	}
	if got := buildRowsPresize(filter); got != 0 {
		t.Errorf("buildRowsPresize(Filter) = %d, want 0 — a loose bound must not presize", got)
	}
	// The filter really does produce far fewer rows than its bound, which is why
	// presizing from it would over-allocate rather than fit.
	if got := drainRowCount(t, op); got != 100 {
		t.Errorf("filtered pipeline produced %d rows, want 100", got)
	}
}

func TestFilterOverUnboundedChildHasNoBound(t *testing.T) {
	fx := newMorselFixture(t, 2, 64)
	op, err := fx.scanFactory(t, nil)(context.Background(), 0, 2)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	defer func() { _ = op.Close() }()

	filter, err := NewFilter(unbounded{op}, &Literal{Val: true, T: TypeBool})
	if err != nil {
		t.Fatalf("NewFilter: %v", err)
	}
	if rows, tight := filter.RowCountBound(); rows != 0 || tight {
		t.Errorf("Filter over an unbounded child = (%d, %v), want (0, false)", rows, tight)
	}
}

// ---- presizeRows clamping ---------------------------------------------------

func TestPresizeRowsRefusesUnusableCounts(t *testing.T) {
	cases := []struct{ rows, want int }{
		{rows: -1, want: 0},
		{rows: 0, want: 0},
		{rows: 1, want: 1},
		{rows: 1024, want: 1024},
		{rows: int(maxBuildRows), want: int(maxBuildRows)},
	}
	for _, c := range cases {
		if got := presizeRows(c.rows); got != c.want {
			t.Errorf("presizeRows(%d) = %d, want %d", c.rows, got, c.want)
		}
	}
}

// TestHugeBoundIsRefusedRatherThanAllocated is the maxBuildRows half of the
// overflow contract. reserve refuses a build side at maxBuildRows rows before
// allocating anything (TestRowStoreOverflowFiresBeforeAllocating pins that), and
// that refusal is only reachable if presizing declines to attempt the tens of
// gigabytes a bound above the limit asks for — the allocator would kill the
// process first, turning a clean error into an OOM.
func TestHugeBoundIsRefusedRatherThanAllocated(t *testing.T) {
	if ^uint(0)>>32 == 0 {
		t.Skip("32-bit int: no row count above maxBuildRows is representable")
	}
	huge := int(maxBuildRows) + 1
	if got := presizeRows(huge); got != 0 {
		t.Errorf("presizeRows(maxBuildRows+1) = %d, want 0", got)
	}
	if got := buildRowsPresize(fixedBound{rows: huge, tight: true}); got != 0 {
		t.Errorf("buildRowsPresize(bound %d) = %d, want 0", huge, got)
	}
	// And the store that gets built from it allocates nothing up front, so it is
	// on the growth path where reserve's guard lives.
	store := newRowStore(oneColSchema(), buildRowsPresize(fixedBound{rows: huge, tight: true}))
	if store.capRows != 0 || len(store.values) != 0 {
		t.Errorf("store presized from a refused bound: capRows = %d, len = %d, want 0, 0",
			store.capRows, len(store.values))
	}
}

// ---- Call site 1: the serial join's build side -------------------------------

func TestSerialJoinPresizedStoreNeverGrows(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)
	const rows = 6 * 128

	build, err := fx.scanFactory(t, nil)(context.Background(), 0, 6)
	if err != nil {
		t.Fatalf("build scan: %v", err)
	}
	probe, err := fx.scanFactory(t, nil)(context.Background(), 0, 6)
	if err != nil {
		t.Fatalf("probe scan: %v", err)
	}
	join, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	defer func() { _ = join.Close() }()

	// ids are unique across the fixture, so every build row matches exactly one
	// probe row: the join must emit as many rows as the build side holds.
	if got := drainRowCount(t, join); got != rows {
		t.Fatalf("join emitted %d rows, want %d", got, rows)
	}

	if join.store.grows != 0 {
		t.Errorf("presized build store grew %d times, want 0", join.store.grows)
	}
	if join.store.capRows != rows {
		t.Errorf("build store capRows = %d, want exactly the presized %d", join.store.capRows, rows)
	}
	if join.store.rows() != rows {
		t.Errorf("build store holds %d rows, want %d", join.store.rows(), rows)
	}
	// The same count sizes the slot array, and the keys are unique here, so the
	// table must not have rehashed either.
	tbl := join.parts[0]
	if want := joinTableCapacity(rows); len(tbl.slots) != want {
		t.Errorf("hash table has %d slots, want %d — it grew after being presized",
			len(tbl.slots), want)
	}
}

// TestSerialJoinFallsBackToGrowthWithoutATightBound covers the two shapes that
// still grow from zero, and checks they still produce the same join. Growth is
// the correct behaviour here, not a gap: presizing from a filter's loose bound
// over-allocates by the predicate's selectivity, which on this fixture is 6x.
func TestSerialJoinFallsBackToGrowthWithoutATightBound(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)

	cases := []struct {
		name      string
		buildSide func(t *testing.T) Operator
		wantRows  int
	}{
		{
			name: "filter above the build scan",
			buildSide: func(t *testing.T) Operator {
				op, err := fx.filterProjectFactory(t, 128, nil)(context.Background(), 0, 6)
				if err != nil {
					t.Fatalf("build pipeline: %v", err)
				}
				return op
			},
			wantRows: 128,
		},
		{
			name: "build side reports no bound at all",
			buildSide: func(t *testing.T) Operator {
				op, err := fx.scanFactory(t, nil)(context.Background(), 0, 1)
				if err != nil {
					t.Fatalf("build scan: %v", err)
				}
				return unbounded{op}
			},
			wantRows: 128,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			probe, err := fx.scanFactory(t, nil)(context.Background(), 0, 6)
			if err != nil {
				t.Fatalf("probe scan: %v", err)
			}
			join, err := NewHashJoin(c.buildSide(t), probe, 0, 0)
			if err != nil {
				t.Fatalf("NewHashJoin: %v", err)
			}
			defer func() { _ = join.Close() }()

			if got := drainRowCount(t, join); got != c.wantRows {
				t.Fatalf("join emitted %d rows, want %d", got, c.wantRows)
			}
			if join.store.grows == 0 {
				t.Error("build store did not grow, so it was presized from a bound that is not tight")
			}
			if join.store.rows() != c.wantRows {
				t.Errorf("build store holds %d rows, want %d", join.store.rows(), c.wantRows)
			}
		})
	}
}

// ---- Call site 2: the parallel build's pass 1 --------------------------------

func TestPartitionMorselPresizedStoreNeverGrows(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)
	schema := scanSchema(t, fx)

	// One morsel at a time, over ranges of different widths, because the bound is
	// per morsel and a bug that reused the first morsel's count would only show
	// up on a morsel of a different size.
	for _, rg := range [][2]int{{0, 1}, {1, 3}, {3, 6}} {
		pipeline, err := fx.scanFactory(t, nil)(context.Background(), rg[0], rg[1])
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		want := (rg[1] - rg[0]) * fx.rowsPerRG
		bucket, err := partitionMorsel(context.Background(), pipeline, schema, 0, 4, 3)
		_ = pipeline.Close()
		if err != nil {
			t.Fatalf("partitionMorsel: %v", err)
		}
		if bucket.store.grows != 0 {
			t.Errorf("morsel [%d,%d): presized store grew %d times, want 0",
				rg[0], rg[1], bucket.store.grows)
		}
		if bucket.store.capRows != want || bucket.store.rows() != want {
			t.Errorf("morsel [%d,%d): store capRows = %d, rows = %d, want %d for both",
				rg[0], rg[1], bucket.store.capRows, bucket.store.rows(), want)
		}
		// Every row landed in exactly one partition, so the lists account for all
		// of them — a presized store that dropped or duplicated rows would show
		// up here rather than only in the join's output.
		got := 0
		for _, list := range bucket.rows {
			got += len(list)
		}
		if got != want {
			t.Errorf("morsel [%d,%d): partition lists hold %d rows, want %d",
				rg[0], rg[1], got, want)
		}
	}
}

func TestPartitionMorselFallsBackToGrowthWithoutATightBound(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)

	// A filtered build pipeline: the shape a build-side predicate the planner
	// could not fold into the scan produces, and the shape a nested join's
	// morsel produces as well — neither reports a tight bound.
	pipeline, err := fx.filterProjectFactory(t, 400, nil)(context.Background(), 0, 6)
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer func() { _ = pipeline.Close() }()
	schema := pipeline.Schema()

	bucket, err := partitionMorsel(context.Background(), pipeline, schema, 0, 4, 3)
	if err != nil {
		t.Fatalf("partitionMorsel: %v", err)
	}
	if bucket.store.grows == 0 {
		t.Error("morsel store did not grow, so it was presized from a bound that is not tight")
	}
	if bucket.store.rows() != 400 {
		t.Errorf("morsel store holds %d rows, want 400", bucket.store.rows())
	}
}

// TestParallelBuildPresizingKeepsTheTableIdentical is the end-to-end guard on
// both call sites at once: the parallel build over presized morsel stores must
// produce the same table a serial drain does, per key and in per-key row order.
// Presizing changes where rows live in memory, and this is what pins that it does
// not change which rows exist or the order they were inserted in.
func TestParallelBuildPresizingKeepsTheTableIdentical(t *testing.T) {
	fx := newMorselFixture(t, 6, 128)
	ctx := context.Background()
	schema := scanSchema(t, fx)

	serialOp, err := fx.scanFactory(t, nil)(ctx, 0, 6)
	if err != nil {
		t.Fatalf("serial scan: %v", err)
	}
	serial, err := BuildSharedHashTableRadix(ctx, serialOp, 0, 2)
	_ = serialOp.Close()
	if err != nil {
		t.Fatalf("BuildSharedHashTableRadix: %v", err)
	}

	parallel, err := BuildSharedHashTableParallel(ctx, fx.scanFactory(t, nil), schema, 0,
		fx.rowGroups, 4, 1, 2)
	if err != nil {
		t.Fatalf("BuildSharedHashTableParallel: %v", err)
	}

	if serial.NumRows() != parallel.NumRows() || serial.NumKeys() != parallel.NumKeys() {
		t.Fatalf("serial (%d rows, %d keys) != parallel (%d rows, %d keys)",
			serial.NumRows(), serial.NumKeys(), parallel.NumRows(), parallel.NumKeys())
	}
	for key := int64(0); key < int64(fx.rowGroups*fx.rowsPerRG); key++ {
		sRows := keyValues(serial, key)
		pRows := keyValues(parallel, key)
		if len(sRows) != len(pRows) {
			t.Fatalf("key %d: serial has %d rows, parallel %d", key, len(sRows), len(pRows))
		}
		for i := range sRows {
			if sRows[i] != pRows[i] {
				t.Fatalf("key %d row %d: serial %v != parallel %v", key, i, sRows[i], pRows[i])
			}
		}
	}
}

// ---- helpers ----------------------------------------------------------------

// unbounded hides an operator's RowCountBound. The field is explicit rather than
// embedded so no method is promoted; embedding would carry the wrapped
// operator's bound straight through and make this wrapper a no-op.
type unbounded struct{ op Operator }

func (u unbounded) Schema() Schema                           { return u.op.Schema() }
func (u unbounded) Next(ctx context.Context) (*Batch, error) { return u.op.Next(ctx) }
func (u unbounded) Close() error                             { return u.op.Close() }

// fixedBound reports a bound without producing any rows, for the cases where the
// bound itself is what is under test.
type fixedBound struct {
	rows  int
	tight bool
}

func (f fixedBound) Schema() Schema                       { return Schema{} }
func (f fixedBound) Next(context.Context) (*Batch, error) { return nil, nil }
func (f fixedBound) Close() error                         { return nil }
func (f fixedBound) RowCountBound() (int, bool)           { return f.rows, f.tight }

// drainRowCount consumes op and returns the number of rows it produced, counting
// through a selection vector where one is installed.
func drainRowCount(t *testing.T, op Operator) int {
	t.Helper()
	n := 0
	for {
		batch, err := op.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			return n
		}
		if batch.SelVec != nil {
			n += len(batch.SelVec)
			continue
		}
		n += batch.Length
	}
}

// scanSchema returns the schema a bare scan of the fixture exposes.
func scanSchema(t *testing.T, fx morselFixture) Schema {
	t.Helper()
	op, err := fx.scanFactory(t, nil)(context.Background(), 0, 1)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	defer func() { _ = op.Close() }()
	return op.Schema()
}

// keyValues returns the stored payload of every row carrying key, in chain order.
func keyValues(sht *SharedHashTable, key int64) [][2]int64 {
	tbl := sht.parts[radixPart(key, sht.partMask)]
	var out [][2]int64
	for ri := tbl.lookup(key); ri != noRow; ri = sht.store.next[ri] {
		out = append(out, [2]int64{sht.store.value(ri, 0), sht.store.value(ri, 1)})
	}
	return out
}
