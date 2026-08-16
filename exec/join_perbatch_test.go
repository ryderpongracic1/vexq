package exec

import (
	"context"
	"fmt"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// These tests pin the behaviour the per-batch allocation removal in join.go had
// to preserve: the order HashJoin.Next and forEachBuildRow visit rows in, that
// NULL-keyed rows are dropped on both sides, that duplicate keys chain in build
// order, and that a probe batch's string column is reprojected by carrying its
// dictionary rather than rebuilding one.

// ---- helpers ---------------------------------------------------------------

func intVec(vals []int64, nulls ...int) *Int64Vector {
	v := &Int64Vector{Values: vals, NullBitmap: storage.FullBitmap(len(vals))}
	for _, i := range nulls {
		v.NullBitmap[i/8] &^= 1 << uint(i%8)
	}
	return v
}

func strVecFrom(values []string) *StringVector {
	db := storage.NewDictBuilder()
	codes := make([]uint32, len(values))
	for i, s := range values {
		codes[i] = db.Add(s)
	}
	dict, err := storage.UnmarshalDictionary(db.Marshal())
	if err != nil {
		panic(err)
	}
	return &StringVector{Codes: codes, Dict: dict, NullBitmap: storage.FullBitmap(len(values))}
}

// keyBatch builds a one-int64-column batch, optionally with a selection vector.
func keyBatch(name string, vals []int64, sel SelectionVector, nulls ...int) *Batch {
	b := &Batch{
		Schema:  Schema{Fields: []Field{{Name: name, Type: TypeInt64}}},
		Vectors: []Vector{intVec(vals, nulls...)},
		Length:  len(vals),
	}
	if sel != nil {
		b.SelVec = sel
		b.Length = len(sel)
	}
	return b
}

// ---- forEachBuildRow visit order ------------------------------------------

// TestForEachBuildRowVisitOrder pins the order and the row set forEachBuildRow
// hands to visit, with and without a selection vector. Row order is what
// BuildSharedHashTableParallel's determinism guarantee is stated in terms of, so
// it is the invariant the []int removal had to preserve exactly.
func TestForEachBuildRowVisitOrder(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name     string
		vals     []int64
		sel      SelectionVector
		nulls    []int
		wantKeys []int64
		wantRows []int
	}{
		{
			name:     "no selection vector visits every row in ascending order",
			vals:     []int64{10, 20, 30, 40},
			wantKeys: []int64{10, 20, 30, 40},
			wantRows: []int{0, 1, 2, 3},
		},
		{
			name:     "selection vector visits selected rows in selection order",
			vals:     []int64{10, 20, 30, 40},
			sel:      SelectionVector{3, 0, 2},
			wantKeys: []int64{40, 10, 30},
			wantRows: []int{3, 0, 2},
		},
		{
			name:     "NULL keys are dropped without a selection vector",
			vals:     []int64{10, 20, 30},
			nulls:    []int{1},
			wantKeys: []int64{10, 30},
			wantRows: []int{0, 2},
		},
		{
			name:     "NULL keys are dropped through a selection vector",
			vals:     []int64{10, 20, 30, 40},
			sel:      SelectionVector{2, 1, 0},
			nulls:    []int{1},
			wantKeys: []int64{30, 10},
			wantRows: []int{2, 0},
		},
		{
			name:     "empty selection vector visits nothing",
			vals:     []int64{10, 20},
			sel:      SelectionVector{},
			wantKeys: nil,
			wantRows: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := keyBatch("k", tc.vals, tc.sel, tc.nulls...)
			src := &sliceBatchOp{schema: b.Schema, batches: []*Batch{b}}

			var gotKeys []int64
			var gotRows []int
			err := forEachBuildRow(ctx, src, 0, func(key int64, batch *Batch, rowIdx int) error {
				if batch != b {
					t.Errorf("visit got a different batch than the source produced")
				}
				gotKeys = append(gotKeys, key)
				gotRows = append(gotRows, rowIdx)
				return nil
			})
			if err != nil {
				t.Fatalf("forEachBuildRow: %v", err)
			}
			if fmt.Sprint(gotKeys) != fmt.Sprint(tc.wantKeys) {
				t.Errorf("keys = %v, want %v", gotKeys, tc.wantKeys)
			}
			if fmt.Sprint(gotRows) != fmt.Sprint(tc.wantRows) {
				t.Errorf("row indices = %v, want %v", gotRows, tc.wantRows)
			}
		})
	}
}

// TestForEachBuildRowOrderAcrossBatches pins that order is preserved across
// batch boundaries too — a selection-vector batch followed by an unfiltered one
// must produce one continuous sequence.
func TestForEachBuildRowOrderAcrossBatches(t *testing.T) {
	ctx := context.Background()
	b1 := keyBatch("k", []int64{1, 2, 3, 4}, SelectionVector{1, 3})
	b2 := keyBatch("k", []int64{5, 6}, nil)
	src := &sliceBatchOp{schema: b1.Schema, batches: []*Batch{b1, b2}}

	var got []int64
	if err := forEachBuildRow(ctx, src, 0, func(key int64, _ *Batch, _ int) error {
		got = append(got, key)
		return nil
	}); err != nil {
		t.Fatalf("forEachBuildRow: %v", err)
	}
	want := []int64{2, 4, 5, 6}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("keys = %v, want %v", got, want)
	}
}

// TestForEachBuildRowStopsOnVisitError pins that a visit error aborts the drain
// immediately rather than continuing through the rest of the batch.
func TestForEachBuildRowStopsOnVisitError(t *testing.T) {
	ctx := context.Background()
	b := keyBatch("k", []int64{1, 2, 3, 4}, nil)
	src := &sliceBatchOp{schema: b.Schema, batches: []*Batch{b}}

	visited := 0
	wantErr := fmt.Errorf("boom")
	err := forEachBuildRow(ctx, src, 0, func(int64, *Batch, int) error {
		visited++
		if visited == 2 {
			return wantErr
		}
		return nil
	})
	if err != wantErr {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	if visited != 2 {
		t.Errorf("visited %d rows, want 2", visited)
	}
}

// ---- probe-side selection vector and NULL keys -----------------------------

// joinFixture builds a HashJoin over one build batch and one probe batch, both
// single int64 key columns plus a payload, and returns the drained output rows as
// (buildPayload, probePayload) pairs in emission order.
func drainJoinPairs(t *testing.T, buildBatch, probeBatch *Batch) [][2]int64 {
	t.Helper()
	ctx := context.Background()
	build := &sliceBatchOp{schema: buildBatch.Schema, batches: []*Batch{buildBatch}}
	probe := &sliceBatchOp{schema: probeBatch.Schema, batches: []*Batch{probeBatch}}
	j, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	var out [][2]int64
	for {
		b, err := j.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if b == nil {
			break
		}
		bp := b.Vectors[1].(*Int64Vector)
		pp := b.Vectors[3].(*Int64Vector)
		for i := 0; i < b.Length; i++ {
			out = append(out, [2]int64{bp.Values[i], pp.Values[i]})
		}
	}
	return out
}

func twoColBatch(keyName, payloadName string, keys, payload []int64, sel SelectionVector, keyNulls ...int) *Batch {
	b := &Batch{
		Schema: Schema{Fields: []Field{
			{Name: keyName, Type: TypeInt64},
			{Name: payloadName, Type: TypeInt64},
		}},
		Vectors: []Vector{intVec(keys, keyNulls...), intVec(payload)},
		Length:  len(keys),
	}
	if sel != nil {
		b.SelVec = sel
		b.Length = len(sel)
	}
	return b
}

// TestHashJoinProbeSelectionVector pins that the probe loop honours a selection
// vector: only selected rows probe, in selection order, and rows outside the
// selection never appear in the output.
func TestHashJoinProbeSelectionVector(t *testing.T) {
	buildBatch := twoColBatch("bk", "bp", []int64{1, 2, 3}, []int64{100, 200, 300}, nil)
	// Probe rows 0..3 have keys 3,1,2,1; the selection picks rows 3, 0 and 1.
	probeBatch := twoColBatch("pk", "pp", []int64{3, 1, 2, 1}, []int64{10, 11, 12, 13},
		SelectionVector{3, 0, 1})

	got := drainJoinPairs(t, buildBatch, probeBatch)
	want := [][2]int64{{100, 13}, {300, 10}, {100, 11}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("output = %v, want %v (probe selection-vector order)", got, want)
	}
}

// TestHashJoinProbeNullKeysDropped pins that a NULL probe key matches nothing,
// through both the unfiltered and the selection-vector path.
func TestHashJoinProbeNullKeysDropped(t *testing.T) {
	buildBatch := twoColBatch("bk", "bp", []int64{1, 2}, []int64{100, 200}, nil)

	t.Run("no selection vector", func(t *testing.T) {
		probeBatch := twoColBatch("pk", "pp", []int64{1, 0, 2}, []int64{10, 11, 12}, nil, 1)
		got := drainJoinPairs(t, buildBatch, probeBatch)
		want := [][2]int64{{100, 10}, {200, 12}}
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("output = %v, want %v", got, want)
		}
	})

	t.Run("through a selection vector", func(t *testing.T) {
		probeBatch := twoColBatch("pk", "pp", []int64{1, 0, 2}, []int64{10, 11, 12},
			SelectionVector{2, 1, 0}, 1)
		got := drainJoinPairs(t, buildBatch, probeBatch)
		want := [][2]int64{{200, 12}, {100, 10}}
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("output = %v, want %v", got, want)
		}
	})
}

// TestHashJoinBuildNullKeysDropped pins that a NULL build key is never inserted,
// so a probe row carrying the zero value it would have stored finds no match.
func TestHashJoinBuildNullKeysDropped(t *testing.T) {
	buildBatch := twoColBatch("bk", "bp", []int64{1, 0, 2}, []int64{100, 999, 200}, nil, 1)
	probeBatch := twoColBatch("pk", "pp", []int64{0, 1, 2}, []int64{10, 11, 12}, nil)

	got := drainJoinPairs(t, buildBatch, probeBatch)
	want := [][2]int64{{100, 11}, {200, 12}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("output = %v, want %v (NULL-keyed build row must not be inserted)", got, want)
	}
}

// TestHashJoinDuplicateKeyBuildOrder pins that the rows of one key are emitted in
// build order. This is the property BuildSharedHashTableParallel's determinism
// guarantee rests on: the chain order comes from forEachBuildRow's visit order,
// so it is asserted here for both the unfiltered and the selection-vector build.
func TestHashJoinDuplicateKeyBuildOrder(t *testing.T) {
	t.Run("no selection vector", func(t *testing.T) {
		buildBatch := twoColBatch("bk", "bp",
			[]int64{7, 7, 7}, []int64{101, 102, 103}, nil)
		probeBatch := twoColBatch("pk", "pp", []int64{7}, []int64{10}, nil)
		got := drainJoinPairs(t, buildBatch, probeBatch)
		want := [][2]int64{{101, 10}, {102, 10}, {103, 10}}
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("output = %v, want %v (build order)", got, want)
		}
	})

	t.Run("selection vector reorders the chain to match", func(t *testing.T) {
		// The selection vector visits rows 2, 0, 1, so the chain must be
		// 103, 101, 102 — build order means the order forEachBuildRow visited.
		buildBatch := twoColBatch("bk", "bp",
			[]int64{7, 7, 7}, []int64{101, 102, 103}, SelectionVector{2, 0, 1})
		probeBatch := twoColBatch("pk", "pp", []int64{7}, []int64{10}, nil)
		got := drainJoinPairs(t, buildBatch, probeBatch)
		want := [][2]int64{{103, 10}, {101, 10}, {102, 10}}
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("output = %v, want %v (selection-vector build order)", got, want)
		}
	})
}

// ---- probe string column: dictionary carried, codes copied -----------------

// TestProbeStringColumnCarriesDictionary pins the reprojection of a probe string
// column: the emitted vector reads back the same strings, shares the source
// dictionary rather than rebuilding one, and copies the codes so it survives the
// source buffer being overwritten (TableScan reuses its code buffers).
func TestProbeStringColumnCarriesDictionary(t *testing.T) {
	ctx := context.Background()

	buildBatch := twoColBatch("bk", "bp", []int64{1, 2}, []int64{100, 200}, nil)

	names := strVecFrom([]string{"AIR", "RAIL", "SHIP", "AIR"})
	probeBatch := &Batch{
		Schema: Schema{Fields: []Field{
			{Name: "pk", Type: TypeInt64},
			{Name: "mode", Type: TypeString},
		}},
		Vectors: []Vector{intVec([]int64{2, 1, 2, 1}), names},
		Length:  4,
	}

	build := &sliceBatchOp{schema: buildBatch.Schema, batches: []*Batch{buildBatch}}
	probe := &sliceBatchOp{schema: probeBatch.Schema, batches: []*Batch{probeBatch}}
	j, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	out, err := j.Next(ctx)
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if out == nil {
		t.Fatal("no output batch")
	}
	// Output columns: bk, bp, pk, mode.
	sv, ok := out.Vectors[3].(*StringVector)
	if !ok {
		t.Fatalf("probe string column is %T, want *StringVector", out.Vectors[3])
	}
	if sv.Dict != names.Dict {
		t.Errorf("emitted vector does not share the probe batch's dictionary")
	}
	// Probe rows are emitted in probe order, so the mode column reads back the
	// probe batch's rows 0..3 unchanged.
	want := []string{"AIR", "RAIL", "SHIP", "AIR"}
	got := make([]string, out.Length)
	for i := 0; i < out.Length; i++ {
		got[i] = sv.Get(i)
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("strings = %v, want %v", got, want)
	}

	// Codes must be a copy: overwriting the source buffer must not change what
	// the already-emitted batch reads back.
	for i := range names.Codes {
		names.Codes[i] = 0
	}
	for i := 0; i < out.Length; i++ {
		if got2 := sv.Get(i); got2 != want[i] {
			t.Errorf("row %d = %q after the source code buffer was overwritten, want %q "+
				"(emitted batch aliases the probe batch's codes)", i, got2, want[i])
		}
	}
}

// TestProbeStringColumnNullRows pins that a NULL probe string stays NULL and that
// non-NULL rows around it are unaffected.
func TestProbeStringColumnNullRows(t *testing.T) {
	ctx := context.Background()

	buildBatch := twoColBatch("bk", "bp", []int64{1}, []int64{100}, nil)
	names := strVecFrom([]string{"AIR", "RAIL", "SHIP"})
	names.NullBitmap[0] &^= 1 << 1 // row 1 is NULL

	probeBatch := &Batch{
		Schema: Schema{Fields: []Field{
			{Name: "pk", Type: TypeInt64},
			{Name: "mode", Type: TypeString},
		}},
		Vectors: []Vector{intVec([]int64{1, 1, 1}), names},
		Length:  3,
	}
	build := &sliceBatchOp{schema: buildBatch.Schema, batches: []*Batch{buildBatch}}
	probe := &sliceBatchOp{schema: probeBatch.Schema, batches: []*Batch{probeBatch}}
	j, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatalf("NewHashJoin: %v", err)
	}
	out, err := j.Next(ctx)
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	sv := out.Vectors[3].(*StringVector)
	if sv.IsNull(0) || !sv.IsNull(1) || sv.IsNull(2) {
		t.Errorf("null bitmap = [%v %v %v], want [false true false]",
			sv.IsNull(0), sv.IsNull(1), sv.IsNull(2))
	}
	if sv.Get(0) != "AIR" || sv.Get(2) != "SHIP" {
		t.Errorf("non-NULL rows = %q, %q; want AIR, SHIP", sv.Get(0), sv.Get(2))
	}
}

// ---- allocation behaviour ---------------------------------------------------

// TestProbeLoopDoesNotAllocatePerBatch pins the change itself: probing a batch
// that produces no matches must not allocate, which is only true once the probe
// row indices are no longer materialised into a []int. A no-match probe isolates
// the loop from emitMatches, which legitimately allocates the output vectors.
func TestProbeLoopDoesNotAllocatePerBatch(t *testing.T) {
	ctx := context.Background()

	buildBatch := twoColBatch("bk", "bp", []int64{1}, []int64{100}, nil)

	// Every probe key misses the single build key, so Next drains the whole probe
	// side and returns EOF without ever emitting a batch — which isolates the
	// probe loop from emitMatches, whose output vectors legitimately allocate.
	keys := make([]int64, BlockRows)
	payload := make([]int64, BlockRows)
	for i := range keys {
		keys[i] = int64(1000 + i)
	}
	probeSchema := twoColBatch("pk", "pp", keys, payload, nil).Schema

	mkSel := func(withSel bool) SelectionVector {
		if !withSel {
			return nil
		}
		sel := make(SelectionVector, BlockRows/2)
		for k := range sel {
			sel[k] = uint16(k * 2)
		}
		return sel
	}

	measure := func(withSel bool) float64 {
		return testingAllocs(func() {
			pbs := make([]*Batch, 16)
			for i := range pbs {
				pbs[i] = twoColBatch("pk", "pp", keys, payload, mkSel(withSel))
			}
			p := &sliceBatchOp{schema: probeSchema, batches: pbs}
			b := &sliceBatchOp{schema: buildBatch.Schema, batches: []*Batch{buildBatch}}
			j, err := NewHashJoin(b, p, 0, 0)
			if err != nil {
				t.Fatalf("NewHashJoin: %v", err)
			}
			for {
				out, err := j.Next(ctx)
				if err != nil {
					t.Fatalf("Next: %v", err)
				}
				if out == nil {
					return
				}
			}
		})
	}

	// The fixture itself allocates (16 batches, their vectors and selection
	// vectors), so the assertion is on the delta over a run that builds the same
	// fixture without probing — anything scaling with batch count would show up
	// as tens of extra allocations per batch.
	for _, withSel := range []bool{false, true} {
		allocs := measure(withSel)
		fixture := testingAllocs(func() {
			for i := 0; i < 16; i++ {
				_ = twoColBatch("pk", "pp", keys, payload, mkSel(withSel))
			}
		})
		// Allow a small constant for the join, the build table and the store.
		const slack = 40
		if allocs > fixture+slack {
			t.Errorf("selVec=%v: probing 16 batches allocated %.0f objects vs %.0f for the "+
				"fixture alone (+%.0f, want <= +%d) — the probe loop is allocating per batch",
				withSel, allocs, fixture, allocs-fixture, slack)
		}
	}
}

// TestBuildDrainDoesNotAllocatePerBatch pins the same property for
// forEachBuildRow: draining N batches whose every key is NULL visits no rows and
// must not allocate per batch.
func TestBuildDrainDoesNotAllocatePerBatch(t *testing.T) {
	ctx := context.Background()

	allNull := func(n int) []int {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	vals := make([]int64, BlockRows)
	nulls := allNull(BlockRows)

	build := func(count int, withSel bool) []*Batch {
		out := make([]*Batch, count)
		for i := range out {
			var sel SelectionVector
			if withSel {
				sel = make(SelectionVector, BlockRows/2)
				for k := range sel {
					sel[k] = uint16(k * 2)
				}
			}
			out[i] = keyBatch("k", vals, sel, nulls...)
		}
		return out
	}

	for _, withSel := range []bool{false, true} {
		drained := testingAllocs(func() {
			bs := build(16, withSel)
			src := &sliceBatchOp{schema: bs[0].Schema, batches: bs}
			if err := forEachBuildRow(ctx, src, 0, func(int64, *Batch, int) error {
				t.Fatal("visit called for an all-NULL key column")
				return nil
			}); err != nil {
				t.Fatalf("forEachBuildRow: %v", err)
			}
		})
		fixture := testingAllocs(func() { _ = build(16, withSel) })
		const slack = 8
		if drained > fixture+slack {
			t.Errorf("selVec=%v: draining 16 batches allocated %.0f objects vs %.0f for the "+
				"fixture alone (+%.0f, want <= +%d) — forEachBuildRow is allocating per batch",
				withSel, drained, fixture, drained-fixture, slack)
		}
	}
}

// testingAllocs reports the number of heap objects f allocates, averaged over a
// few runs so a single GC-timing artefact does not decide a test.
func testingAllocs(f func()) float64 {
	return testing.AllocsPerRun(3, f)
}
