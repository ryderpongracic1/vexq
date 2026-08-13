package exec

import (
	"math"
	"testing"
)

func TestMergePartialAggNonNullCounts(t *testing.T) {
	aggExprs := []AggExpr{{
		Kind:      AggAvg,
		ColIdx:    0,
		OutName:   "avg_value",
		AccumType: TypeFloat64,
	}}
	schema := Schema{Fields: []Field{{Name: "avg_value", Type: TypeFloat64, Nullable: true}}}
	dst := newPartialAggregate(nil, aggExprs, schema)

	first := newPartialAggregate(nil, aggExprs, schema)
	first.keys = []string{""}
	first.groups[""] = []int64{int64(math.Float64bits(4))}
	first.aggNonNull[""] = []int64{2}
	mergePartialAgg(dst, first)

	if got := dst.aggNonNull[""][0]; got != 2 {
		t.Fatalf("new group non-null count = %d, want 2", got)
	}

	second := newPartialAggregate(nil, aggExprs, schema)
	second.keys = []string{""}
	second.groups[""] = []int64{int64(math.Float64bits(6))}
	second.aggNonNull[""] = []int64{2}
	mergePartialAgg(dst, second)

	if got := dst.aggNonNull[""][0]; got != 4 {
		t.Fatalf("merged non-null count = %d, want 4", got)
	}

	batch := dst.buildOutputBatch([]string{""})
	got := batch.Vectors[0].(*Float64Vector).Values[0]
	if want := 2.5; got != want {
		t.Fatalf("merged AVG = %v, want %v", got, want)
	}
}
