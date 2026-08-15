package exec

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// These tests cover per-worker pipeline reuse (see morsel.go): a worker builds
// one pipeline and repositions it with Reset for each morsel it claims instead
// of rebuilding it. The correctness risk that buys is state leaking from one
// morsel into the next, so every reuse assertion here is made against a
// rebuild-per-morsel oracle over the same row groups: the two must agree row for
// row, not merely in aggregate.

// ---- Fixture ----------------------------------------------------------------

// morselFixture is a .vxq file of rowGroups row groups, rowsPerRG rows each,
// with columns:
//
//	id  INT64   — the global row index, so every row is identifiable
//	amt FLOAT64 — id * 2, so a projected expression has something to compute
//
// Row groups are small so a test can write several of them cheaply while still
// exercising real footers, real ColumnReaders and real TableScan.Reset.
type morselFixture struct {
	path      string
	rowGroups int
	rowsPerRG int
}

func newMorselFixture(t *testing.T, rowGroups, rowsPerRG int) morselFixture {
	t.Helper()
	path := filepath.Join(t.TempDir(), "morsel.vxq")
	schema := storage.Schema{Fields: []storage.Field{
		{Name: "id", Type: storage.TypeInt64},
		{Name: "amt", Type: storage.TypeFloat64},
	}}
	w, err := storage.NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	ctx := context.Background()
	for rg := range rowGroups {
		if err := w.BeginRowGroup(rowsPerRG); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}
		ids := make([]int64, rowsPerRG)
		amts := make([]float64, rowsPerRG)
		for i := range rowsPerRG {
			ids[i] = int64(rg*rowsPerRG + i)
			amts[i] = float64(ids[i]) * 2
		}
		if err := w.AppendColumn(ctx, 0, nil, ids); err != nil {
			t.Fatalf("AppendColumn id: %v", err)
		}
		if err := w.AppendColumn(ctx, 1, nil, amts); err != nil {
			t.Fatalf("AppendColumn amt: %v", err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if fi, err := os.Stat(path); err != nil || fi.Size() == 0 {
		t.Fatalf("fixture not written: %v", err)
	}
	return morselFixture{path: path, rowGroups: rowGroups, rowsPerRG: rowsPerRG}
}

// scanFactory returns a PipelineFactory producing a bare TableScan over a
// row-group range. calls, when non-nil, counts factory invocations.
func (f morselFixture) scanFactory(t *testing.T, calls *atomic.Int64) PipelineFactory {
	return func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		if calls != nil {
			calls.Add(1) // workers build concurrently
		}
		r, err := storage.Open(ctx, f.path)
		if err != nil {
			return nil, err
		}
		scan, err := NewTableScanRange(r, nil, nil, rgStart, rgEnd)
		if err != nil {
			_ = r.Close()
			return nil, err
		}
		return scan, nil
	}
}

// filterProjectFactory returns a PipelineFactory producing the full shape a
// parallel aggregate worker runs: Scan → Filter(id < cutoff) → Project(amt+1).
// The filter makes the selection vector morsel-dependent and the projection puts
// reused expression scratch in the pipeline, which are the two pieces of state
// reuse could leak.
func (f morselFixture) filterProjectFactory(t *testing.T, cutoff int64, calls *atomic.Int64) PipelineFactory {
	base := f.scanFactory(t, calls)
	return func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		scan, err := base(ctx, rgStart, rgEnd)
		if err != nil {
			return nil, err
		}
		pred := &BinOp{
			Op:    BinLT,
			Left:  &ColumnRef{Name: "id", Idx: 0, T: TypeInt64},
			Right: &Literal{Val: cutoff, T: TypeInt64},
			T:     TypeBool,
		}
		filter, err := NewFilter(scan, pred)
		if err != nil {
			_ = scan.Close()
			return nil, err
		}
		proj, err := NewProject(filter, []ProjectExpr{
			{Name: "id", Expr: &ColumnRef{Name: "id", Idx: 0, T: TypeInt64}},
			{Name: "amt1", Expr: &BinOp{
				Op:    BinAdd,
				Left:  &ColumnRef{Name: "amt", Idx: 1, T: TypeFloat64},
				Right: &Literal{Val: 1.0, T: TypeFloat64},
				T:     TypeFloat64,
			}},
		})
		if err != nil {
			_ = filter.Close()
			return nil, err
		}
		return proj, nil
	}
}

// opaqueOp wraps an Operator without implementing Reset, so a pipeline
// containing it is not reusable. It stands in for any operator morsel.go's
// closed switch does not name.
type opaqueOp struct{ Operator }

// morselRows is what one morsel produced: its (id, amt1) rows in order.
type morselRows struct {
	ids  []int64
	amts []float64
}

// drainMorsels runs [0, rowGroups) one morsel at a time through a runner over
// factory, returning each morsel's rows separately so a comparison catches a
// morsel that emitted another morsel's rows even when the totals happen to match.
func drainMorsels(t *testing.T, factory PipelineFactory, rowGroups int) []morselRows {
	t.Helper()
	ctx := context.Background()
	runner := morselRunner{factory: factory}
	defer runner.close()

	out := make([]morselRows, 0, rowGroups)
	for rg := range rowGroups {
		op, err := runner.open(ctx, rg, rg+1)
		if err != nil {
			t.Fatalf("open morsel %d: %v", rg, err)
		}
		var got morselRows
		for {
			batch, err := op.Next(ctx)
			if err != nil {
				t.Fatalf("morsel %d: Next: %v", rg, err)
			}
			if batch == nil {
				break
			}
			ids := batch.Vectors[0].(*Int64Vector)
			for i := range batch.Length {
				row := i
				if batch.SelVec != nil {
					row = int(batch.SelVec[i])
				}
				got.ids = append(got.ids, ids.Values[row])
				if len(batch.Vectors) > 1 {
					if amts, ok := batch.Vectors[1].(*Float64Vector); ok {
						got.amts = append(got.amts, amts.Values[row])
					}
				}
			}
		}
		runner.release(op)
		out = append(out, got)
	}
	return out
}

func sameMorselRows(a, b []morselRows) error {
	if len(a) != len(b) {
		return fmt.Errorf("morsel count %d != %d", len(a), len(b))
	}
	for i := range a {
		if len(a[i].ids) != len(b[i].ids) {
			return fmt.Errorf("morsel %d: %d rows != %d rows", i, len(a[i].ids), len(b[i].ids))
		}
		for j := range a[i].ids {
			if a[i].ids[j] != b[i].ids[j] {
				return fmt.Errorf("morsel %d row %d: id %d != %d", i, j, a[i].ids[j], b[i].ids[j])
			}
		}
		if len(a[i].amts) != len(b[i].amts) {
			return fmt.Errorf("morsel %d: %d amts != %d amts", i, len(a[i].amts), len(b[i].amts))
		}
		for j := range a[i].amts {
			if a[i].amts[j] != b[i].amts[j] {
				return fmt.Errorf("morsel %d row %d: amt %g != %g", i, j, a[i].amts[j], b[i].amts[j])
			}
		}
	}
	return nil
}

// ---- Reusability detection --------------------------------------------------

func TestReusableMorselPipelineAcceptsScanPipelines(t *testing.T) {
	fx := newMorselFixture(t, 2, 64)
	ctx := context.Background()

	scanOnly, err := fx.scanFactory(t, nil)(ctx, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer scanOnly.Close()
	if reusableMorselPipeline(scanOnly) == nil {
		t.Error("a bare TableScan must be reusable")
	}

	full, err := fx.filterProjectFactory(t, 1_000_000, nil)(ctx, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer full.Close()
	if reusableMorselPipeline(full) == nil {
		t.Error("Scan → Filter → Project must be reusable")
	}
}

func TestReusableMorselPipelineRefusesUnknownOperator(t *testing.T) {
	fx := newMorselFixture(t, 2, 64)
	ctx := context.Background()

	scan, err := fx.scanFactory(t, nil)(ctx, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	// Directly, and nested under operators that are themselves resettable: an
	// unresettable operator anywhere in the pipeline must make the whole
	// pipeline unreusable, since Reset would otherwise skip it silently.
	if got := reusableMorselPipeline(&opaqueOp{scan}); got != nil {
		t.Errorf("opaque operator must not be reusable, got %T", got)
	}
	wrapped, err := NewProject(&opaqueOp{scan}, []ProjectExpr{
		{Name: "id", Expr: &ColumnRef{Name: "id", Idx: 0, T: TypeInt64}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := reusableMorselPipeline(wrapped); got != nil {
		t.Errorf("Project over an opaque operator must not be reusable, got %T", got)
	}
	if got := reusableMorselPipeline(NewSchemaOnly(scan.Schema())); got != nil {
		t.Errorf("SchemaOnly must not be reusable, got %T", got)
	}
}

func TestReusableMorselPipelineRefusesSelfBuildingJoin(t *testing.T) {
	fx := newMorselFixture(t, 2, 64)
	ctx := context.Background()

	build, err := fx.scanFactory(t, nil)(ctx, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	probe, err := fx.scanFactory(t, nil)(ctx, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	join, err := NewHashJoin(build, probe, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer join.Close()
	// A self-building join's table comes from its own build child, so
	// repositioning the probe alone is not a well-defined reset of it.
	if got := reusableMorselPipeline(join); got != nil {
		t.Errorf("self-building HashJoin must not be reusable, got %T", got)
	}
}

// ---- Reuse produces the same rows as rebuilding -----------------------------

func TestMorselRunnerReusesOnePipeline(t *testing.T) {
	fx := newMorselFixture(t, 5, 128)

	var reuseCalls, rebuildCalls atomic.Int64
	reused := drainMorsels(t, fx.scanFactory(t, &reuseCalls), fx.rowGroups)

	base := fx.scanFactory(t, &rebuildCalls)
	rebuilt := drainMorsels(t, func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		op, err := base(ctx, rgStart, rgEnd)
		if err != nil {
			return nil, err
		}
		return &opaqueOp{op}, nil
	}, fx.rowGroups)

	if err := sameMorselRows(reused, rebuilt); err != nil {
		t.Errorf("reused pipeline disagrees with rebuild-per-morsel oracle: %v", err)
	}
	if got := reuseCalls.Load(); got != 1 {
		t.Errorf("reusable pipeline built %d times, want 1", got)
	}
	if got := rebuildCalls.Load(); got != int64(fx.rowGroups) {
		t.Errorf("unresettable pipeline built %d times, want %d", got, fx.rowGroups)
	}
	// Guard the oracle itself: every row group must contribute rows, or the
	// comparison above would pass on two empty results.
	for rg, m := range reused {
		if len(m.ids) != fx.rowsPerRG {
			t.Fatalf("morsel %d yielded %d rows, want %d", rg, len(m.ids), fx.rowsPerRG)
		}
	}
}

func TestMorselReuseDoesNotLeakFilterOrScratchState(t *testing.T) {
	// 5 row groups of 128 rows: ids 0..639. cutoff 300 makes row groups 0 and 1
	// pass in full, row group 2 pass partially (ids 256..299), and row groups 3
	// and 4 produce nothing at all. A leaked selection vector or a stale scratch
	// vector shows up as an empty morsel re-emitting the previous morsel's rows,
	// or as a partial morsel keeping the previous morsel's longer selection.
	fx := newMorselFixture(t, 5, 128)
	const cutoff = 300

	reused := drainMorsels(t, fx.filterProjectFactory(t, cutoff, nil), fx.rowGroups)

	base := fx.filterProjectFactory(t, cutoff, nil)
	rebuilt := drainMorsels(t, func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		op, err := base(ctx, rgStart, rgEnd)
		if err != nil {
			return nil, err
		}
		return &opaqueOp{op}, nil
	}, fx.rowGroups)

	if err := sameMorselRows(reused, rebuilt); err != nil {
		t.Errorf("reused pipeline disagrees with rebuild-per-morsel oracle: %v", err)
	}

	// Independent of the oracle: assert the row counts the fixture implies, so a
	// bug that corrupted both paths identically still fails.
	want := []int{128, 128, 44, 0, 0}
	for rg, n := range want {
		if got := len(reused[rg].ids); got != n {
			t.Errorf("morsel %d: %d rows, want %d (ids %v)", rg, got, n, reused[rg].ids)
		}
	}
	// And the projected expression: amt1 = id*2 + 1 for every surviving row.
	for rg := range reused {
		for j, id := range reused[rg].ids {
			if want := float64(id)*2 + 1; reused[rg].amts[j] != want {
				t.Fatalf("morsel %d row %d: amt1 %g, want %g", rg, j, reused[rg].amts[j], want)
			}
		}
	}
}

// ---- Reuse through the parallel operators -----------------------------------

// sumAggConfig is SUM(amt1) over the pipeline filterProjectFactory produces.
func sumAggConfig() ([]int, []AggExpr, Schema) {
	aggExprs := []AggExpr{{Kind: AggSum, ColIdx: 1, OutName: "sum_amt1", AccumType: TypeFloat64}}
	out := Schema{Fields: []Field{{Name: "sum_amt1", Type: TypeFloat64, Nullable: true}}}
	return nil, aggExprs, out
}

// drainSingleFloat drains op, which must produce exactly one row of one float.
func drainSingleFloat(t *testing.T, op Operator) float64 {
	t.Helper()
	ctx := context.Background()
	got := math.NaN()
	rows := 0
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		for i := range batch.Length {
			got = batch.Vectors[0].(*Float64Vector).Values[i]
			rows++
		}
	}
	if rows != 1 {
		t.Fatalf("expected one output row, got %d", rows)
	}
	return got
}

func TestParallelAggregateReuseMatchesRebuildAcrossWorkerCounts(t *testing.T) {
	fx := newMorselFixture(t, 8, 256)
	const cutoff = 1500
	groupBy, aggExprs, outSchema := sumAggConfig()

	// Exact expected value: ids 0..cutoff-1 survive, each contributing 2*id+1.
	want := 0.0
	for id := int64(0); id < cutoff; id++ {
		want += float64(id)*2 + 1
	}

	for _, workers := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			var reuseCalls atomic.Int64
			pha := NewParallelHashAggregate(
				fx.filterProjectFactory(t, cutoff, &reuseCalls),
				fx.rowGroups, workers, 1, groupBy, aggExprs, outSchema)
			got := drainSingleFloat(t, pha)
			if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
				t.Errorf("parallel SUM %g, want %g", got, want)
			}
			// One pipeline per *active* worker rather than one per morsel. The
			// bound is an inequality, not an equality: a worker that claims
			// several morsels before a slower sibling starts can drain the queue
			// first, leaving that sibling to build nothing. numWorkers is also
			// capped to the row-group count by the constructor. Before pipeline
			// reuse this count was fx.rowGroups for every worker count.
			effective := int64(min(workers, fx.rowGroups))
			built := reuseCalls.Load()
			if built < 1 || built > effective {
				t.Errorf("built %d pipelines, want 1..%d (at most one per worker)", built, effective)
			}
			if int64(fx.rowGroups) > effective && built >= int64(fx.rowGroups) {
				t.Errorf("built %d pipelines for %d morsels — pipelines still scale with morsels",
					built, fx.rowGroups)
			}
		})
	}
}

func TestParallelJoinProbeReuseMatchesSerial(t *testing.T) {
	// Build side: row groups 0..1 (ids 0..255). Probe side: all 6 row groups.
	// The inner join therefore matches exactly the build side's ids, and every
	// probe morsel past row group 1 must contribute nothing — the case where a
	// leaked probe batch would show up as extra rows.
	fx := newMorselFixture(t, 6, 128)
	ctx := context.Background()

	buildFactory := func(bCtx context.Context) (Operator, error) {
		return fx.scanFactory(t, nil)(bCtx, 0, 2)
	}
	var probeCalls atomic.Int64
	probeFactory := fx.scanFactory(t, &probeCalls)

	// COUNT(*) over the join output.
	aggExprs := []AggExpr{{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64}}
	outSchema := Schema{Fields: []Field{{Name: "cnt", Type: TypeInt64, Nullable: true}}}

	const workers = 4
	pj := NewParallelHashJoinAggregate(
		buildFactory, 0,
		probeFactory, 0,
		nil,
		fx.rowGroups, workers, 1,
		nil, aggExprs, outSchema,
	)

	var got int64
	rows := 0
	for {
		batch, err := pj.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		for i := range batch.Length {
			got = batch.Vectors[0].(*Int64Vector).Values[i]
			rows++
		}
	}
	if rows != 1 {
		t.Fatalf("expected one output row, got %d", rows)
	}
	// ids are unique, so each build row matches exactly one probe row.
	if want := int64(2 * fx.rowsPerRG); got != want {
		t.Errorf("join COUNT(*) = %d, want %d", got, want)
	}
	// At most one probe pipeline per worker, and in particular not one per
	// morsel — see TestParallelAggregateParsesFooterPerWorker for why this is a
	// bound rather than an equality.
	if built := probeCalls.Load(); built < 1 || built > workers {
		t.Errorf("built %d probe pipelines over %d morsels, want 1..%d",
			built, fx.rowGroups, workers)
	}
}

// ---- Footer parses no longer scale with row groups --------------------------

func TestParallelAggregateParsesFooterPerWorker(t *testing.T) {
	fx := newMorselFixture(t, 16, 64)
	groupBy, aggExprs, outSchema := sumAggConfig()

	// Single worker: fully deterministic. It claims all 16 morsels, and reuse
	// means it opens the file — and so parses the footer — exactly once. Before
	// pipeline reuse this was 16.
	before := storage.FooterParses()
	pha := NewParallelHashAggregate(
		fx.filterProjectFactory(t, math.MaxInt32, nil),
		fx.rowGroups, 1, 1, groupBy, aggExprs, outSchema)
	_ = drainSingleFloat(t, pha)
	if parses := storage.FooterParses() - before; parses != 1 {
		t.Errorf("one worker parsed the footer %d times over %d row groups, want 1",
			parses, fx.rowGroups)
	}

	// Four workers: at most one parse per worker, and in particular not one per
	// morsel. An equality would be flaky — a worker that drains the queue before
	// a sibling starts leaves that sibling with nothing to open.
	const workers = 4
	before = storage.FooterParses()
	pha = NewParallelHashAggregate(
		fx.filterProjectFactory(t, math.MaxInt32, nil),
		fx.rowGroups, workers, 1, groupBy, aggExprs, outSchema)
	_ = drainSingleFloat(t, pha)
	parses := storage.FooterParses() - before
	if parses < 1 || parses > workers {
		t.Errorf("parsed the footer %d times for %d row groups on %d workers, want 1..%d",
			parses, fx.rowGroups, workers, workers)
	}
}

// ---- Reuse leaves the aggregate merge unchanged -----------------------------

// TestReuseKeepsGroupedAggregateExact pins the grouped case: integer aggregates
// must stay exactly equal to a serial drain, which they only do if each morsel
// contributes its own rows exactly once.
func TestReuseKeepsGroupedAggregateExact(t *testing.T) {
	fx := newMorselFixture(t, 6, 100)

	// GROUP BY (id / 100) — one group per row group — SUM(id), COUNT(*).
	factory := func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		op, err := fx.scanFactory(t, nil)(ctx, rgStart, rgEnd)
		if err != nil {
			return nil, err
		}
		return NewProject(op, []ProjectExpr{
			{Name: "grp", Expr: &BinOp{
				Op:    BinDiv,
				Left:  &ColumnRef{Name: "id", Idx: 0, T: TypeInt64},
				Right: &Literal{Val: int64(100), T: TypeInt64},
				T:     TypeInt64,
			}},
			{Name: "id", Expr: &ColumnRef{Name: "id", Idx: 0, T: TypeInt64}},
		})
	}
	aggExprs := []AggExpr{
		{Kind: AggSum, ColIdx: 1, OutName: "sum_id", AccumType: TypeInt64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	}
	outSchema := Schema{Fields: []Field{
		{Name: "grp", Type: TypeInt64},
		{Name: "sum_id", Type: TypeInt64, Nullable: true},
		{Name: "cnt", Type: TypeInt64, Nullable: true},
	}}

	serialOp, err := factory(context.Background(), 0, fx.rowGroups)
	if err != nil {
		t.Fatal(err)
	}
	serialHA, err := NewHashAggregate(serialOp, []int{0}, aggExprs)
	if err != nil {
		t.Fatal(err)
	}
	serial := collectIntGroups(t, serialHA)

	pha := NewParallelHashAggregate(factory, fx.rowGroups, 4, 1, []int{0}, aggExprs, outSchema)
	parallel := collectIntGroups(t, pha)

	if len(serial) != len(parallel) {
		t.Fatalf("serial produced %d groups, parallel %d", len(serial), len(parallel))
	}
	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("group %d: serial %q != parallel %q", i, serial[i], parallel[i])
		}
	}
	if len(serial) != fx.rowGroups {
		t.Fatalf("expected %d groups, got %d", fx.rowGroups, len(serial))
	}
}

// collectIntGroups drains an aggregate of (int64 group, int64 sum, int64 count)
// into sorted "grp:sum:cnt" strings.
func collectIntGroups(t *testing.T, op Operator) []string {
	t.Helper()
	ctx := context.Background()
	var out []string
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			break
		}
		grp := batch.Vectors[0].(*Int64Vector)
		sum := batch.Vectors[1].(*Int64Vector)
		cnt := batch.Vectors[2].(*Int64Vector)
		for i := range batch.Length {
			out = append(out, fmt.Sprintf("%d:%d:%d", grp.Values[i], sum.Values[i], cnt.Values[i]))
		}
	}
	sort.Strings(out)
	return out
}
