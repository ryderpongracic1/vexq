package exec

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// Benchmarks and tests for the row-index iteration HashAggregate.accumulate
// performs over a batch. Every batch below carries a selection vector, which is
// the shape a filtered scan produces and the shape that dominated the parallel
// path's allocation profile: accumulate used to widen the batch's []uint16
// selection vector into a freshly allocated []int on every batch.

// selVecBatches builds n batches of BlockRows physical rows each, every one
// carrying a selection vector that keeps `keep` of every 2 rows, so the
// benchmarks exercise the filtered path rather than the identity path.
func selVecBatches(numBatches int) ([]*Batch, Schema) {
	rng := rand.New(rand.NewSource(7))
	flagValues := []string{"A", "N", "R"}
	statusValues := []string{"F", "O"}

	fields := []Field{
		{Name: "flag", Type: TypeString},
		{Name: "status", Type: TypeString},
		{Name: "price", Type: TypeFloat64},
	}
	schema := Schema{Fields: fields}

	batches := make([]*Batch, numBatches)
	for bi := range batches {
		flagDict := storage.NewDictBuilder()
		statusDict := storage.NewDictBuilder()
		for _, v := range flagValues {
			flagDict.Add(v)
		}
		for _, v := range statusValues {
			statusDict.Add(v)
		}
		flagCodes := make([]uint32, BlockRows)
		statusCodes := make([]uint32, BlockRows)
		prices := make([]float64, BlockRows)
		for i := range BlockRows {
			flagCodes[i] = uint32(rng.Intn(len(flagValues)))
			statusCodes[i] = uint32(rng.Intn(len(statusValues)))
			prices[i] = rng.Float64() * 1000
		}
		flagDictR, _ := storage.UnmarshalDictionary(flagDict.Marshal())
		statusDictR, _ := storage.UnmarshalDictionary(statusDict.Marshal())

		// Every other row survives: 512 of 1024.
		sel := make(SelectionVector, 0, BlockRows)
		for i := 0; i < BlockRows; i += 2 {
			sel = append(sel, uint16(i))
		}

		batches[bi] = &Batch{
			Schema: schema,
			Vectors: []Vector{
				&StringVector{Codes: flagCodes, Dict: flagDictR, NullBitmap: storage.FullBitmap(BlockRows)},
				&StringVector{Codes: statusCodes, Dict: statusDictR, NullBitmap: storage.FullBitmap(BlockRows)},
				&Float64Vector{Values: prices, NullBitmap: storage.FullBitmap(BlockRows)},
			},
			Length: len(sel),
			SelVec: sel,
		}
	}
	return batches, schema
}

// runAggBench drains a HashAggregate over the prepared batches b.N times.
func runAggBench(b *testing.B, batches []*Batch, schema Schema, groupBy []int, aggs []AggExpr) {
	b.ResetTimer()
	b.ReportAllocs()
	ctx := context.Background()
	for range b.N {
		child := &sliceBatchOp{batches: batches, schema: schema}
		ha, err := NewHashAggregate(child, groupBy, aggs)
		if err != nil {
			b.Fatal(err)
		}
		for {
			batch, err := ha.Next(ctx)
			if err != nil {
				b.Fatal(err)
			}
			if batch == nil {
				break
			}
		}
	}
}

// BenchmarkAccumulateGroupedSelVec drives the dict-code integer-key path over
// selection-vector batches (the TPC-H Q1 shape after a filter).
func BenchmarkAccumulateGroupedSelVec(b *testing.B) {
	batches, schema := selVecBatches(1000)
	runAggBench(b, batches, schema, []int{0, 1}, []AggExpr{
		{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
		{Kind: AggCount, ColIdx: -1, OutName: "count_star", AccumType: TypeInt64},
	})
}

// BenchmarkAccumulateDirectSelVec drives the no-GROUP-BY path over
// selection-vector batches (the TPC-H Q6 shape after a filter).
func BenchmarkAccumulateDirectSelVec(b *testing.B) {
	batches, schema := selVecBatches(1000)
	runAggBench(b, batches, schema, nil, []AggExpr{
		{Kind: AggSum, ColIdx: 2, OutName: "revenue", AccumType: TypeFloat64},
	})
}

// ---- Tests: row iteration and per-instance batch buffers --------------------

// aggRowsSchema is the schema the tests below share. flag is dict-encoded so it
// can drive the integer-key path; region is an INT64 group-by column, which
// forces the composite string-key path because canUseIntKey rejects it.
var aggRowsSchema = Schema{Fields: []Field{
	{Name: "flag", Type: TypeString},
	{Name: "region", Type: TypeInt64},
	{Name: "price", Type: TypeFloat64},
	{Name: "qty", Type: TypeInt64},
}}

// aggRowsBatch builds one batch of len(flags) physical rows. dictOrder gives the
// order the batch's dictionary assigns codes in, so consecutive batches can
// disagree about local codes for the same string — the cross-rowgroup case the
// integer-key remap tables exist for. sel, when non-nil, becomes the batch's
// selection vector and its length becomes Batch.Length.
func aggRowsBatch(flags []string, dictOrder []string, regions []int64, prices []float64, qtys []int64, sel SelectionVector) *Batch {
	n := len(flags)
	db := storage.NewDictBuilder()
	for _, s := range dictOrder {
		db.Add(s)
	}
	dict, err := storage.UnmarshalDictionary(db.Marshal())
	if err != nil {
		panic(err)
	}
	codes := make([]uint32, n)
	for i, f := range flags {
		c, ok := db.Lookup(f)
		if !ok {
			panic("flag not in dictOrder: " + f)
		}
		codes[i] = c
	}
	length := n
	if sel != nil {
		length = len(sel)
	}
	return &Batch{
		Schema: aggRowsSchema,
		Vectors: []Vector{
			&StringVector{Codes: codes, Dict: dict, NullBitmap: storage.FullBitmap(n)},
			&Int64Vector{Values: regions, NullBitmap: storage.FullBitmap(n)},
			&Float64Vector{Values: prices, NullBitmap: storage.FullBitmap(n)},
			&Int64Vector{Values: qtys, NullBitmap: storage.FullBitmap(n)},
		},
		Length: length,
		SelVec: sel,
	}
}

// drainAgg runs a HashAggregate over batches and returns its output rows keyed
// by a printable rendering of the group-by columns ("" for a global aggregate).
func drainAgg(t *testing.T, batches []*Batch, groupBy []int, aggs []AggExpr) map[string][]float64 {
	t.Helper()
	child := &sliceBatchOp{batches: batches, schema: aggRowsSchema}
	ha, err := NewHashAggregate(child, groupBy, aggs)
	if err != nil {
		t.Fatalf("new hash aggregate: %v", err)
	}
	out := make(map[string][]float64)
	ctx := context.Background()
	for {
		batch, err := ha.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		for r := 0; r < batch.Length; r++ {
			key := ""
			for g := range groupBy {
				switch v := batch.Vectors[g].(type) {
				case *StringVector:
					key += v.Get(r) + "|"
				case *Int64Vector:
					key += fmt.Sprintf("%d|", v.Values[r])
				default:
					t.Fatalf("unexpected group-by vector %T", v)
				}
			}
			vals := make([]float64, len(aggs))
			for a := range aggs {
				switch v := batch.Vectors[len(groupBy)+a].(type) {
				case *Int64Vector:
					vals[a] = float64(v.Values[r])
				case *Float64Vector:
					vals[a] = v.Values[r]
				default:
					t.Fatalf("unexpected aggregate vector %T", v)
				}
			}
			out[key] = vals
		}
	}
	return out
}

// twoSelVecBatches returns two batches sharing a schema, each with a selection
// vector, where the second selects strictly fewer rows than the first. That is
// the shape that catches an index buffer whose length is carried over from the
// previous batch: a stale length would make the shorter batch re-process rows
// the longer one left behind.
func twoSelVecBatches() []*Batch {
	// Batch 1: 8 physical rows, 4 selected (rows 0, 2, 4, 6).
	b1 := aggRowsBatch(
		[]string{"A", "B", "A", "B", "A", "B", "A", "B"},
		[]string{"A", "B"},
		[]int64{1, 1, 2, 2, 1, 1, 2, 2},
		[]float64{1, 100, 2, 200, 4, 400, 8, 800},
		[]int64{1, 1, 1, 1, 1, 1, 1, 1},
		SelectionVector{0, 2, 4, 6},
	)
	// Batch 2: 8 physical rows, 1 selected — and the dictionary assigns codes in
	// the opposite order, so a reused remap table must be rebuilt.
	b2 := aggRowsBatch(
		[]string{"B", "B", "B", "B", "B", "B", "B", "A"},
		[]string{"B", "A"},
		[]int64{9, 9, 9, 9, 9, 9, 9, 1},
		[]float64{-1, -1, -1, -1, -1, -1, -1, 16},
		[]int64{1, 1, 1, 1, 1, 1, 1, 1},
		SelectionVector{7},
	)
	return []*Batch{b1, b2}
}

// TestAccumulateSelVecFourPaths covers every accumulate path over batches that
// carry a selection vector: no-GROUP-BY (accumulateDirect), the dict-code
// integer-key path (accumulateIntKey), the composite string-key path (buildKey,
// reached by grouping on an INT64 column), and COUNT(DISTINCT), which disables
// the integer-key path outright.
func TestAccumulateSelVecFourPaths(t *testing.T) {
	// Selected rows across both batches are all flag A: batch 1 contributes
	// (region 1, price 1), (region 2, price 2), (region 1, price 4) and
	// (region 2, price 8); batch 2 contributes (region 1, price 16). So five
	// rows, prices summing to 31, all distinct, every qty 1.
	tests := []struct {
		name    string
		groupBy []int
		aggs    []AggExpr
		want    map[string][]float64
	}{
		{
			name:    "no_group_by_direct",
			groupBy: nil,
			aggs: []AggExpr{
				{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
				{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
				{Kind: AggMin, ColIdx: 2, OutName: "min_price", AccumType: TypeFloat64},
				{Kind: AggMax, ColIdx: 2, OutName: "max_price", AccumType: TypeFloat64},
			},
			want: map[string][]float64{"": {31, 5, 1, 16}},
		},
		{
			name:    "int_key_dict_string_group",
			groupBy: []int{0},
			aggs: []AggExpr{
				{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
				{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
			},
			want: map[string][]float64{"A|": {31, 5}},
		},
		{
			name:    "composite_string_key_group",
			groupBy: []int{1, 0},
			aggs: []AggExpr{
				{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
				{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
			},
			// region 1 holds prices 1, 4 and 16; region 2 holds prices 2 and 8.
			want: map[string][]float64{"1|A|": {21, 3}, "2|A|": {10, 2}},
		},
		{
			name:    "count_distinct_disables_int_key",
			groupBy: []int{0},
			aggs: []AggExpr{
				{Kind: AggCountDistinct, ColIdx: 2, OutName: "d_price", Distinct: true, AccumType: TypeInt64},
				{Kind: AggCountDistinct, ColIdx: 3, OutName: "d_qty", Distinct: true, AccumType: TypeInt64},
				{Kind: AggCountDistinct, ColIdx: 0, OutName: "d_flag", Distinct: true, AccumType: TypeInt64},
			},
			// 5 distinct prices, 1 distinct qty, 1 distinct flag.
			want: map[string][]float64{"A|": {5, 1, 1}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := drainAgg(t, twoSelVecBatches(), tc.groupBy, tc.aggs)
			if len(got) != len(tc.want) {
				t.Fatalf("group count: got %d (%v), want %d (%v)", len(got), got, len(tc.want), tc.want)
			}
			for key, want := range tc.want {
				gotVals, ok := got[key]
				if !ok {
					t.Fatalf("missing group %q in %v", key, got)
				}
				for i := range want {
					if gotVals[i] != want[i] {
						t.Errorf("group %q agg %d: got %v, want %v", key, i, gotVals[i], want[i])
					}
				}
			}
		})
	}
}

// TestAccumulateShrinkingSelVecMatchesPerBatchOracle feeds the same two batches
// (the second selecting fewer rows than the first) to one aggregate and to a
// fresh aggregate per batch, and requires the same totals. Any row-index state
// carried from the first batch into the second — a stale length, a stale
// dictionary remap, a stale source vector — shows up here as a mismatch.
func TestAccumulateShrinkingSelVecMatchesPerBatchOracle(t *testing.T) {
	aggs := []AggExpr{
		{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	}

	both := twoSelVecBatches()
	combined := drainAgg(t, both, []int{0}, aggs)

	// Oracle: accumulate each batch in its own aggregate and add the results.
	oracle := map[string][]float64{}
	for _, b := range twoSelVecBatches() {
		part := drainAgg(t, []*Batch{b}, []int{0}, aggs)
		for key, vals := range part {
			if _, ok := oracle[key]; !ok {
				oracle[key] = make([]float64, len(vals))
			}
			for i, v := range vals {
				oracle[key][i] += v
			}
		}
	}

	if len(combined) != len(oracle) {
		t.Fatalf("group count: combined %v, oracle %v", combined, oracle)
	}
	for key, want := range oracle {
		got, ok := combined[key]
		if !ok {
			t.Fatalf("missing group %q in %v", key, combined)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("group %q agg %d: combined %v, oracle %v", key, i, got[i], want[i])
			}
		}
	}
}

// TestAccumulateAggVecsRewrittenPerBatch pins the one state-leak hazard the
// reused aggVecs buffer introduces: a COUNT(*) slot holds nil, which a fresh
// make used to supply for free. Poisoning the buffer and accumulating must leave
// the COUNT(*) slot nil and every other slot pointing at this batch's vector.
func TestAccumulateAggVecsRewrittenPerBatch(t *testing.T) {
	aggs := []AggExpr{
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
		{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
	}
	batches := twoSelVecBatches()
	child := &sliceBatchOp{batches: batches, schema: aggRowsSchema}
	ha, err := NewHashAggregate(child, nil, aggs)
	if err != nil {
		t.Fatalf("new hash aggregate: %v", err)
	}
	ha.initMaps()

	poison := &Int64Vector{Values: []int64{-1}, NullBitmap: storage.FullBitmap(1)}
	for bi, b := range batches {
		// Fill every slot with a vector that belongs to no batch.
		buf := ha.acquireAggVecs()
		for i := range buf {
			buf[i] = poison
		}
		if err := ha.accumulate(b); err != nil {
			t.Fatalf("batch %d: accumulate: %v", bi, err)
		}
		if ha.aggVecs[0] != nil {
			t.Errorf("batch %d: COUNT(*) slot holds %v, want nil", bi, ha.aggVecs[0])
		}
		if ha.aggVecs[1] != b.Vectors[2] {
			t.Errorf("batch %d: SUM slot does not point at this batch's price vector", bi)
		}
	}
}

// TestAccumulateIntKeyRemapRewrittenPerBatch gives consecutive batches
// dictionaries that assign opposite codes to the same strings. A remap table or
// string vector retained from the previous batch would attribute rows to the
// wrong group, so equality with a per-batch oracle is what proves the reused
// buffers are rewritten.
func TestAccumulateIntKeyRemapRewrittenPerBatch(t *testing.T) {
	ones := []int64{1, 1}
	// Batch 1 dictionary: A=0, B=1. Batch 2 dictionary: B=0, A=1.
	b1 := aggRowsBatch([]string{"A", "B"}, []string{"A", "B"}, ones, []float64{1, 10}, ones, nil)
	b2 := aggRowsBatch([]string{"A", "B"}, []string{"B", "A"}, ones, []float64{2, 20}, ones, nil)

	aggs := []AggExpr{
		{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	}
	got := drainAgg(t, []*Batch{b1, b2}, []int{0}, aggs)
	want := map[string][]float64{"A|": {3, 2}, "B|": {30, 2}}
	if len(got) != len(want) {
		t.Fatalf("group count: got %v, want %v", got, want)
	}
	for key, w := range want {
		g, ok := got[key]
		if !ok {
			t.Fatalf("missing group %q in %v", key, got)
		}
		for i := range w {
			if g[i] != w[i] {
				t.Errorf("group %q agg %d: got %v, want %v", key, i, g[i], w[i])
			}
		}
	}
}

// TestAccumulateNoPerBatchAllocation is the regression guard for the change
// itself: once the reused buffers exist, accumulating a further batch must not
// allocate. It runs the two paths whose profiles motivated this — the
// no-GROUP-BY path and the integer-key path — over a batch carrying a selection
// vector, which is where the []int widening used to cost 8 KB per batch.
//
// The composite string-key path is deliberately not asserted here: buildKey
// allocates a byte buffer and a string per row, so a batch's total would be
// dominated by ~2 allocations per row and a single reintroduced per-batch
// allocation would be invisible against it. That path's correctness under reuse
// is covered by TestAccumulateSelVecFourPaths and the shrinking-selvec oracle
// instead.
func TestAccumulateNoPerBatchAllocation(t *testing.T) {
	cases := []struct {
		name    string
		groupBy []int
	}{
		{"no_group_by_direct", nil},
		{"int_key_dict_string_group", []int{0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			flags := make([]string, BlockRows)
			regions := make([]int64, BlockRows)
			prices := make([]float64, BlockRows)
			qtys := make([]int64, BlockRows)
			sel := make(SelectionVector, 0, BlockRows/2)
			for i := range BlockRows {
				flags[i] = "A"
				regions[i] = 1
				prices[i] = float64(i)
				qtys[i] = 1
				if i%2 == 0 {
					sel = append(sel, uint16(i))
				}
			}
			batch := aggRowsBatch(flags, []string{"A"}, regions, prices, qtys, sel)

			child := &sliceBatchOp{schema: aggRowsSchema}
			ha, err := NewHashAggregate(child, tc.groupBy, []AggExpr{
				{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
				{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
			})
			if err != nil {
				t.Fatalf("new hash aggregate: %v", err)
			}
			ha.initMaps()
			// Warm up: the first batch creates the group and the buffers.
			if err := ha.accumulate(batch); err != nil {
				t.Fatalf("warmup accumulate: %v", err)
			}

			allocs := testing.AllocsPerRun(50, func() {
				if err := ha.accumulate(batch); err != nil {
					t.Fatalf("accumulate: %v", err)
				}
			})
			if allocs != 0 {
				t.Errorf("accumulate allocated %.1f times per batch, want 0", allocs)
			}
		})
	}
}
