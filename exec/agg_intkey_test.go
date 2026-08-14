package exec

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// --- Benchmark: 6M rows, low-cardinality string group-by, -benchmem ----------

// BenchmarkHashAggregateIntKey measures the grouped aggregate operator on
// 6M rows with 2 low-cardinality dict-encoded string GROUP BY columns
// (mimicking TPC-H Q1: l_returnflag × l_linestatus = 4–6 groups).
func BenchmarkHashAggregateIntKey(b *testing.B) {
	const totalRows = 6_000_000
	const batchSize = BlockRows // 1024
	const numBatches = totalRows / batchSize

	// Two string GROUP BY columns with 3 and 2 distinct values respectively.
	flagValues := []string{"RETURN_FLAG_A", "RETURN_FLAG_N", "RETURN_FLAG_R"}
	statusValues := []string{"LINE_STATUS_F", "LINE_STATUS_O"}

	// One float64 aggregate column (SUM).
	rng := rand.New(rand.NewSource(42))

	// Pre-build all batches to exclude data generation from timing.
	batches := make([]*Batch, numBatches)
	for bi := range numBatches {
		// Each batch gets its own dictionary (simulating per-rowgroup dicts
		// that may assign codes in different orders).
		flagDict := storage.NewDictBuilder()
		statusDict := storage.NewDictBuilder()

		// Shuffle insertion order per batch to simulate cross-rowgroup dict instability.
		flagOrder := rng.Perm(len(flagValues))
		statusOrder := rng.Perm(len(statusValues))
		for _, fi := range flagOrder {
			flagDict.Add(flagValues[fi])
		}
		for _, si := range statusOrder {
			statusDict.Add(statusValues[si])
		}

		flagCodes := make([]uint32, batchSize)
		statusCodes := make([]uint32, batchSize)
		prices := make([]float64, batchSize)

		for i := range batchSize {
			fv := flagValues[rng.Intn(len(flagValues))]
			sv := statusValues[rng.Intn(len(statusValues))]
			code, _ := flagDict.Lookup(fv)
			flagCodes[i] = code
			code, _ = statusDict.Lookup(sv)
			statusCodes[i] = code
			prices[i] = rng.Float64() * 1000
		}

		flagRaw := flagDict.Marshal()
		flagDictR, _ := storage.UnmarshalDictionary(flagRaw)
		statusRaw := statusDict.Marshal()
		statusDictR, _ := storage.UnmarshalDictionary(statusRaw)

		batches[bi] = &Batch{
			Schema: Schema{Fields: []Field{
				{Name: "flag", Type: TypeString},
				{Name: "status", Type: TypeString},
				{Name: "price", Type: TypeFloat64},
			}},
			Vectors: []Vector{
				&StringVector{Codes: flagCodes, Dict: flagDictR, NullBitmap: storage.FullBitmap(batchSize)},
				&StringVector{Codes: statusCodes, Dict: statusDictR, NullBitmap: storage.FullBitmap(batchSize)},
				&Float64Vector{Values: prices, NullBitmap: storage.FullBitmap(batchSize)},
			},
			Length: batchSize,
		}
	}

	schema := Schema{Fields: []Field{
		{Name: "flag", Type: TypeString},
		{Name: "status", Type: TypeString},
		{Name: "price", Type: TypeFloat64},
	}}

	b.ResetTimer()
	b.ReportAllocs()

	for range b.N {
		child := &sliceBatchOp{batches: batches, schema: schema}
		ha, err := NewHashAggregate(child, []int{0, 1}, []AggExpr{
			{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
			{Kind: AggCount, ColIdx: -1, OutName: "count_star", AccumType: TypeInt64},
		})
		if err != nil {
			b.Fatal(err)
		}
		ctx := context.Background()
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

// BenchmarkHashAggregateStringKey measures the same workload but forces the
// string-key path by using a non-dict string vector for one GROUP BY column.
// This is the "before" measurement for the integer-key optimization.
func BenchmarkHashAggregateStringKey(b *testing.B) {
	const totalRows = 6_000_000
	const batchSize = BlockRows
	const numBatches = totalRows / batchSize

	flagValues := []string{"RETURN_FLAG_A", "RETURN_FLAG_N", "RETURN_FLAG_R"}
	statusValues := []string{"LINE_STATUS_F", "LINE_STATUS_O"}
	rng := rand.New(rand.NewSource(42))

	// Pre-build batches using dict-encoded strings (same data) but we'll
	// disable intkey by setting Dict=nil on one column.
	batches := make([]*Batch, numBatches)
	for bi := range numBatches {
		flagDict := storage.NewDictBuilder()
		statusDict := storage.NewDictBuilder()
		flagOrder := rng.Perm(len(flagValues))
		statusOrder := rng.Perm(len(statusValues))
		for _, fi := range flagOrder {
			flagDict.Add(flagValues[fi])
		}
		for _, si := range statusOrder {
			statusDict.Add(statusValues[si])
		}

		flagCodes := make([]uint32, batchSize)
		statusCodes := make([]uint32, batchSize)
		prices := make([]float64, batchSize)
		for i := range batchSize {
			fv := flagValues[rng.Intn(len(flagValues))]
			sv := statusValues[rng.Intn(len(statusValues))]
			code, _ := flagDict.Lookup(fv)
			flagCodes[i] = code
			code, _ = statusDict.Lookup(sv)
			statusCodes[i] = code
			prices[i] = rng.Float64() * 1000
		}

		flagRaw := flagDict.Marshal()
		flagDictR, _ := storage.UnmarshalDictionary(flagRaw)
		statusRaw := statusDict.Marshal()
		statusDictR, _ := storage.UnmarshalDictionary(statusRaw)

		batches[bi] = &Batch{
			Schema: Schema{Fields: []Field{
				{Name: "flag", Type: TypeString},
				{Name: "status", Type: TypeString},
				{Name: "price", Type: TypeFloat64},
			}},
			Vectors: []Vector{
				&StringVector{Codes: flagCodes, Dict: flagDictR, NullBitmap: storage.FullBitmap(batchSize)},
				// Force string-key path: set Dict=nil on status column.
				&StringVector{Codes: statusCodes, Dict: statusDictR, NullBitmap: storage.FullBitmap(batchSize)},
				&Float64Vector{Values: prices, NullBitmap: storage.FullBitmap(batchSize)},
			},
			Length: batchSize,
		}
	}

	// For the string-key baseline, we need Dict=nil on one column to force fallback.
	// But that would panic in buildKey. Instead, create a wrapper that disables intkey detection.
	// Simpler: temporarily patch the first batch to have Dict=nil, which triggers string-key.
	// Actually the cleanest approach: create a separate sub-benchmark that uses the real
	// old string-key path by removing the fast-path decision. Instead, let's use a
	// "StringKeyHashAggregate" wrapper. Actually simpler: just benchmark with intKeyDecided=true,
	// intKey.enabled=false.
	schema := Schema{Fields: []Field{
		{Name: "flag", Type: TypeString},
		{Name: "status", Type: TypeString},
		{Name: "price", Type: TypeFloat64},
	}}

	b.ResetTimer()
	b.ReportAllocs()

	for range b.N {
		child := &sliceBatchOp{batches: batches, schema: schema}
		ha, err := NewHashAggregate(child, []int{0, 1}, []AggExpr{
			{Kind: AggSum, ColIdx: 2, OutName: "sum_price", AccumType: TypeFloat64},
			{Kind: AggCount, ColIdx: -1, OutName: "count_star", AccumType: TypeInt64},
		})
		if err != nil {
			b.Fatal(err)
		}
		// Force string-key path.
		ha.intKeyDecided = true
		ha.intKey.enabled = false

		ctx := context.Background()
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

// --- Correctness tests -------------------------------------------------------

// TestIntKeyMultiRowGroupDictInstability verifies that the integer-key path
// produces correct results when the same string values are assigned different
// dictionary codes across row groups (simulating real .vxq per-rowgroup dicts).
func TestIntKeyMultiRowGroupDictInstability(t *testing.T) {
	// Two "row groups" with different dictionary orderings:
	// RG1: flag dict = {"R":0, "A":1, "N":2}
	// RG2: flag dict = {"A":0, "N":1, "R":2}
	// Same values, different codes.

	rg1FlagDict := storage.NewDictBuilder()
	rg1FlagDict.Add("R") // code 0
	rg1FlagDict.Add("A") // code 1
	rg1FlagDict.Add("N") // code 2

	rg2FlagDict := storage.NewDictBuilder()
	rg2FlagDict.Add("A") // code 0
	rg2FlagDict.Add("N") // code 1
	rg2FlagDict.Add("R") // code 2

	schema := Schema{Fields: []Field{
		{Name: "flag", Type: TypeString},
		{Name: "value", Type: TypeInt64},
	}}

	rg1Raw := rg1FlagDict.Marshal()
	rg1Dict, _ := storage.UnmarshalDictionary(rg1Raw)
	rg2Raw := rg2FlagDict.Marshal()
	rg2Dict, _ := storage.UnmarshalDictionary(rg2Raw)

	// RG1 batch: 4 rows — R(0), A(1), N(2), R(0) with values 10, 20, 30, 40
	batch1 := &Batch{
		Schema: schema,
		Vectors: []Vector{
			&StringVector{Codes: []uint32{0, 1, 2, 0}, Dict: rg1Dict, NullBitmap: storage.FullBitmap(4)},
			&Int64Vector{Values: []int64{10, 20, 30, 40}, NullBitmap: storage.FullBitmap(4)},
		},
		Length: 4,
	}

	// RG2 batch: 4 rows — A(0), N(1), R(2), A(0) with values 100, 200, 300, 400
	batch2 := &Batch{
		Schema: schema,
		Vectors: []Vector{
			&StringVector{Codes: []uint32{0, 1, 2, 0}, Dict: rg2Dict, NullBitmap: storage.FullBitmap(4)},
			&Int64Vector{Values: []int64{100, 200, 300, 400}, NullBitmap: storage.FullBitmap(4)},
		},
		Length: 4,
	}

	child := &sliceBatchOp{batches: []*Batch{batch1, batch2}, schema: schema}
	ha, err := NewHashAggregate(child, []int{0}, []AggExpr{
		{Kind: AggSum, ColIdx: 1, OutName: "sum_val", AccumType: TypeInt64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	var results []string
	for {
		batch, err := ha.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		for i := range batch.Length {
			flag := batch.Vectors[0].(*StringVector).Get(i)
			sumVal := batch.Vectors[1].(*Int64Vector).Values[i]
			cnt := batch.Vectors[2].(*Int64Vector).Values[i]
			results = append(results, fmt.Sprintf("%s:%d:%d", flag, sumVal, cnt))
		}
	}

	sort.Strings(results)
	expected := []string{
		"A:520:3", // 20 + 100 + 400 = 520, count = 3
		"N:230:2", // 30 + 200 = 230, count = 2
		"R:350:3", // 10 + 40 + 300 = 350, count = 3
	}
	sort.Strings(expected)

	if len(results) != len(expected) {
		t.Fatalf("got %d groups, want %d: %v", len(results), len(expected), results)
	}
	for i := range results {
		if results[i] != expected[i] {
			t.Errorf("group %d: got %s, want %s", i, results[i], expected[i])
		}
	}
}

// TestIntKeyParallelConsistency verifies that serial and parallel hash
// aggregation produce identical results with the integer-key path active.
func TestIntKeyParallelConsistency(t *testing.T) {
	const batchSize = 512
	const numBatches = 8 // 4096 rows total, split across 4 "rowgroups" of 2 batches each

	flagValues := []string{"A", "N", "R"}
	statusValues := []string{"F", "O"}
	rng := rand.New(rand.NewSource(99))

	// Pre-build batches with per-pair-of-batches different dictionaries.
	batches := make([]*Batch, numBatches)
	schema := Schema{Fields: []Field{
		{Name: "flag", Type: TypeString},
		{Name: "status", Type: TypeString},
		{Name: "amount", Type: TypeFloat64},
	}}

	for bi := range numBatches {
		flagDict := storage.NewDictBuilder()
		statusDict := storage.NewDictBuilder()
		// Shuffle dict insertion order to simulate different rowgroup dicts.
		for _, fi := range rng.Perm(len(flagValues)) {
			flagDict.Add(flagValues[fi])
		}
		for _, si := range rng.Perm(len(statusValues)) {
			statusDict.Add(statusValues[si])
		}

		flagCodes := make([]uint32, batchSize)
		statusCodes := make([]uint32, batchSize)
		amounts := make([]float64, batchSize)
		for i := range batchSize {
			fv := flagValues[rng.Intn(len(flagValues))]
			sv := statusValues[rng.Intn(len(statusValues))]
			code, _ := flagDict.Lookup(fv)
			flagCodes[i] = code
			code, _ = statusDict.Lookup(sv)
			statusCodes[i] = code
			amounts[i] = float64(rng.Intn(1000))
		}

		flagRaw := flagDict.Marshal()
		flagDictR, _ := storage.UnmarshalDictionary(flagRaw)
		statusRaw := statusDict.Marshal()
		statusDictR, _ := storage.UnmarshalDictionary(statusRaw)

		batches[bi] = &Batch{
			Schema: schema,
			Vectors: []Vector{
				&StringVector{Codes: flagCodes, Dict: flagDictR, NullBitmap: storage.FullBitmap(batchSize)},
				&StringVector{Codes: statusCodes, Dict: statusDictR, NullBitmap: storage.FullBitmap(batchSize)},
				&Float64Vector{Values: amounts, NullBitmap: storage.FullBitmap(batchSize)},
			},
			Length: batchSize,
		}
	}

	aggExprs := []AggExpr{
		{Kind: AggSum, ColIdx: 2, OutName: "sum_amt", AccumType: TypeFloat64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	}
	groupBy := []int{0, 1}
	outSchema := Schema{Fields: []Field{
		{Name: "flag", Type: TypeString},
		{Name: "status", Type: TypeString},
		{Name: "sum_amt", Type: TypeFloat64, Nullable: true},
		{Name: "cnt", Type: TypeInt64, Nullable: true},
	}}

	// Serial execution.
	serialChild := &sliceBatchOp{batches: batches, schema: schema}
	serialHA, err := NewHashAggregate(serialChild, groupBy, aggExprs)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	serialResults := collectAggResults(t, ctx, serialHA)

	// Parallel execution with 4 workers, 2 batches per morsel (1 "rowgroup" each).
	totalRGs := numBatches / 2 // treat each pair of batches as a rowgroup
	factory := func(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
		start := rgStart * 2
		end := rgEnd * 2
		if end > numBatches {
			end = numBatches
		}
		return &sliceBatchOp{batches: batches[start:end], schema: schema}, nil
	}

	pha := NewParallelHashAggregate(factory, totalRGs, 4, 1, groupBy, aggExprs, outSchema)
	parallelResults := collectAggResults(t, ctx, pha)

	// Compare: both should have the same groups with the same values.
	if len(serialResults) != len(parallelResults) {
		t.Fatalf("serial produced %d groups, parallel produced %d", len(serialResults), len(parallelResults))
	}

	sort.Strings(serialResults)
	sort.Strings(parallelResults)
	for i := range serialResults {
		if serialResults[i] != parallelResults[i] {
			t.Errorf("mismatch at group %d:\n  serial:   %s\n  parallel: %s", i, serialResults[i], parallelResults[i])
		}
	}
}

// collectAggResults drains an aggregate operator and returns a sorted slice of
// "flag:status:sum:cnt" strings for comparison.
func collectAggResults(t *testing.T, ctx context.Context, op Operator) []string {
	t.Helper()
	var results []string
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		for i := range batch.Length {
			flag := batch.Vectors[0].(*StringVector).Get(i)
			status := batch.Vectors[1].(*StringVector).Get(i)
			sumAmt := batch.Vectors[2].(*Float64Vector).Values[i]
			cnt := batch.Vectors[3].(*Int64Vector).Values[i]
			results = append(results, fmt.Sprintf("%s:%s:%.2f:%d", flag, status, sumAmt, cnt))
		}
	}
	return results
}

// sliceBatchOp is a test helper that replays pre-built batches.
type sliceBatchOp struct {
	batches []*Batch
	schema  Schema
	pos     int
}

func (s *sliceBatchOp) Next(ctx context.Context) (*Batch, error) {
	if s.pos >= len(s.batches) {
		return nil, nil
	}
	b := s.batches[s.pos]
	s.pos++
	return b, nil
}
func (s *sliceBatchOp) Schema() Schema { return s.schema }
func (s *sliceBatchOp) Close() error   { return nil }
