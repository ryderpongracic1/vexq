package exec

import (
	"math"
	"testing"
)

// f64 decodes a float64 accumulator from its int64 IEEE-bit encoding.
func f64(bits int64) float64 { return math.Float64frombits(uint64(bits)) }

// Tests for merging string-valued MIN/MAX partial aggregates.
//
// The parallel path accumulates one HashAggregate per worker and folds them
// together with mergePartialAgg. Numeric MIN/MAX survive that fold because a
// float64 partial can travel inside the int64 accumulator as IEEE bits; a string
// cannot, so string partials merge from strAccs instead. Two properties have to
// hold and neither is free:
//
//   - A partial whose rows were all NULL must not win a comparison. Its stored
//     value is "", which sorts before every other string, so a MIN that merged
//     it blindly would report "" for the whole query. The pre-merge non-null
//     count is what distinguishes "saw nothing" from "saw the empty string".
//   - The result must not depend on the order partials are merged in, since
//     worker completion order is nondeterministic.

// strPartial builds one worker's partial aggregate over the given rows.
func strPartial(t *testing.T, aggs []AggExpr, groupBy []int, batches ...*Batch) *HashAggregate {
	t.Helper()
	ha := newPartialAggregate(groupBy, aggs, Schema{})
	for _, b := range batches {
		if err := ha.accumulate(b); err != nil {
			t.Fatalf("accumulate: %v", err)
		}
	}
	if ha.intKey.enabled && len(ha.intKey.intKeys) > 0 {
		ha.intKey.materializeToStringMaps(ha)
	}
	return ha
}

// mergeAll folds partials into a fresh destination in the given order and returns
// the merged (min, max, nonNull) for the single implicit group of an ungrouped
// aggregate.
func mergeAll(t *testing.T, aggs []AggExpr, partials []*HashAggregate) (min, max string, minNonNull, maxNonNull int64) {
	t.Helper()
	dst := newPartialAggregate(nil, aggs, Schema{})
	for _, p := range partials {
		mergePartialAgg(dst, p)
	}
	if len(dst.keys) == 0 {
		t.Fatalf("merge produced no groups")
	}
	strs := dst.strAccs[""]
	nn := dst.aggNonNull[""]
	return strs[0], strs[1], nn[0], nn[1]
}

var strMinMaxAggs = []AggExpr{
	{Kind: AggMin, ColIdx: 1, OutName: "min_mode", AccumType: TypeString},
	{Kind: AggMax, ColIdx: 1, OutName: "max_mode", AccumType: TypeString},
}

func TestMergePartialAggStringMinMax(t *testing.T) {
	p1 := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{mode: "MAIL"}, {mode: "SHIP"},
	}, []string{"MAIL", "SHIP"}, nil))
	p2 := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{mode: "TRUCK"}, {mode: "AIR"},
	}, []string{"TRUCK", "AIR"}, nil))

	min, max, minNN, maxNN := mergeAll(t, strMinMaxAggs, []*HashAggregate{p1, p2})
	if min != "AIR" || max != "TRUCK" {
		t.Errorf("merged MIN/MAX = %q/%q, want AIR/TRUCK", min, max)
	}
	if minNN != 4 || maxNN != 4 {
		t.Errorf("merged non-null counts = %d/%d, want 4/4", minNN, maxNN)
	}
}

// TestMergePartialAggStringAllNullPartial is the property the non-null guard
// exists for: a worker that saw only NULLs carries "" in its accumulator, and ""
// beats every real value under MIN. Merging it must be a no-op.
func TestMergePartialAggStringAllNullPartial(t *testing.T) {
	real := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{mode: "MAIL"}, {mode: "SHIP"},
	}, []string{"MAIL", "SHIP"}, nil))
	allNull := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{modeNull: true}, {modeNull: true},
	}, []string{"MAIL"}, nil))

	// Both merge orders: the empty partial must lose whether it lands first or last.
	for _, tc := range []struct {
		name     string
		partials []*HashAggregate
	}{
		{"real-then-empty", []*HashAggregate{real, allNull}},
		{"empty-then-real", []*HashAggregate{allNull, real}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			min, max, minNN, _ := mergeAll(t, strMinMaxAggs, tc.partials)
			if min != "MAIL" {
				t.Errorf("merged MIN = %q, want MAIL — an all-NULL partial won the comparison with \"\"", min)
			}
			if max != "SHIP" {
				t.Errorf("merged MAX = %q, want SHIP", max)
			}
			if minNN != 2 {
				t.Errorf("merged non-null count = %d, want 2 (NULL rows do not count)", minNN)
			}
		})
	}
}

// TestMergePartialAggStringEmptyStringPartial is the converse: a partial whose
// only value really is "" must win MIN, so the guard cannot simply skip empty
// strings.
func TestMergePartialAggStringEmptyStringPartial(t *testing.T) {
	real := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{mode: "MAIL"},
	}, []string{"MAIL"}, nil))
	empty := strPartial(t, strMinMaxAggs, nil, strAggBatch([]strRow{
		{mode: ""},
	}, []string{""}, nil))

	min, max, _, _ := mergeAll(t, strMinMaxAggs, []*HashAggregate{real, empty})
	if min != "" {
		t.Errorf("merged MIN = %q, want \"\" — a real empty-string value must win MIN", min)
	}
	if max != "MAIL" {
		t.Errorf("merged MAX = %q, want MAIL", max)
	}
}

// TestMergePartialAggStringOrderInvariant asserts the fold is commutative and
// associative over every permutation of four partials, since worker completion
// order is nondeterministic.
func TestMergePartialAggStringOrderInvariant(t *testing.T) {
	mk := func(vals ...string) *HashAggregate {
		rows := make([]strRow, len(vals))
		for i, v := range vals {
			rows[i] = strRow{mode: v}
		}
		return strPartial(t, strMinMaxAggs, nil, strAggBatch(rows, vals, nil))
	}
	base := []*HashAggregate{
		mk("MAIL", "RAIL"),
		mk("TRUCK"),
		mk("AIR", "FOB"),
		mk("SHIP"),
	}

	for _, perm := range permute4() {
		ordered := []*HashAggregate{base[perm[0]], base[perm[1]], base[perm[2]], base[perm[3]]}
		min, max, minNN, maxNN := mergeAll(t, strMinMaxAggs, ordered)
		if min != "AIR" || max != "TRUCK" {
			t.Fatalf("merge order %v: MIN/MAX = %q/%q, want AIR/TRUCK", perm, min, max)
		}
		if minNN != 6 || maxNN != 6 {
			t.Fatalf("merge order %v: non-null counts = %d/%d, want 6/6", perm, minNN, maxNN)
		}
	}
}

// TestMergePartialAggStringNewGroupDeepCopy checks that a group appearing for
// the first time in dst gets its own string accumulator slice rather than an
// alias of src's, which a later merge into dst would otherwise corrupt.
func TestMergePartialAggStringNewGroupDeepCopy(t *testing.T) {
	src := strPartial(t, strMinMaxAggs, []int{0}, strAggBatch([]strRow{
		{grp: "east", mode: "MAIL"},
	}, []string{"MAIL"}, nil))
	other := strPartial(t, strMinMaxAggs, []int{0}, strAggBatch([]strRow{
		{grp: "east", mode: "AIR"},
	}, []string{"AIR"}, nil))

	dst := newPartialAggregate([]int{0}, strMinMaxAggs, Schema{})
	mergePartialAgg(dst, src)
	mergePartialAgg(dst, other)

	key := dst.keys[0]
	if got := dst.strAccs[key][0]; got != "AIR" {
		t.Errorf("merged MIN = %q, want AIR", got)
	}
	// src must be unchanged by the second merge into dst.
	srcKey := src.keys[0]
	if got := src.strAccs[srcKey][0]; got != "MAIL" {
		t.Errorf("src partial was mutated by merging into dst: MIN = %q, want MAIL", got)
	}
}

// TestMergePartialAggNumericMinMaxUnchanged guards the refactor of
// mergePartialAgg's MIN/MAX arms into one shared case: numeric partials must
// merge exactly as before, for both accumulator encodings.
func TestMergePartialAggNumericMinMaxUnchanged(t *testing.T) {
	aggs := []AggExpr{
		{Kind: AggMin, ColIdx: 3, OutName: "min_price", AccumType: TypeFloat64},
		{Kind: AggMax, ColIdx: 3, OutName: "max_price", AccumType: TypeFloat64},
		{Kind: AggMin, ColIdx: 2, OutName: "min_region", AccumType: TypeInt64},
		{Kind: AggMax, ColIdx: 2, OutName: "max_region", AccumType: TypeInt64},
		{Kind: AggSum, ColIdx: 3, OutName: "sum_price", AccumType: TypeFloat64},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt", AccumType: TypeInt64},
	}
	p1 := strPartial(t, aggs, nil, strAggBatch([]strRow{
		{mode: "X", price: 10, region: 5},
		{mode: "X", price: 40, region: 9},
	}, []string{"X"}, nil))
	p2 := strPartial(t, aggs, nil, strAggBatch([]strRow{
		{mode: "X", price: 5, region: 2},
		{mode: "X", price: 20, region: 7},
	}, []string{"X"}, nil))

	dst := newPartialAggregate(nil, aggs, Schema{})
	mergePartialAgg(dst, p1)
	mergePartialAgg(dst, p2)

	accs := dst.groups[""]
	if got := f64(accs[0]); got != 5 {
		t.Errorf("merged MIN(price) = %v, want 5", got)
	}
	if got := f64(accs[1]); got != 40 {
		t.Errorf("merged MAX(price) = %v, want 40", got)
	}
	if accs[2] != 2 {
		t.Errorf("merged MIN(region) = %d, want 2", accs[2])
	}
	if accs[3] != 9 {
		t.Errorf("merged MAX(region) = %d, want 9", accs[3])
	}
	if got := f64(accs[4]); got != 75 {
		t.Errorf("merged SUM(price) = %v, want 75", got)
	}
	if accs[5] != 4 {
		t.Errorf("merged COUNT(*) = %d, want 4", accs[5])
	}
}

// permute4 returns all 24 permutations of {0,1,2,3}.
func permute4() [][4]int {
	var out [][4]int
	idx := []int{0, 1, 2, 3}
	var rec func(k int)
	rec = func(k int) {
		if k == len(idx) {
			out = append(out, [4]int{idx[0], idx[1], idx[2], idx[3]})
			return
		}
		for i := k; i < len(idx); i++ {
			idx[k], idx[i] = idx[i], idx[k]
			rec(k + 1)
			idx[k], idx[i] = idx[i], idx[k]
		}
	}
	rec(0)
	return out
}
