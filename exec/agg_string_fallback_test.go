package exec

import (
	"fmt"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// Tests for the two ways accumulation leaves the integer-key fast path while a
// string MIN/MAX is in flight. Both move a group's running string value between
// the uint64-keyed maps in intKeyState and the string-keyed maps on
// HashAggregate, and a transfer that dropped or aliased intStrAccs would lose or
// corrupt the answer rather than crash.
//
//  1. A NULL in a GROUP BY column: accumulateIntKey cannot pack a null into its
//     key, so that single row is routed to accumulateOneRow on the string-keyed
//     maps. The operator then holds groups in both maps at once and merges them
//     in materializeToStringMaps.
//  2. Dictionary code overflow: once a global dictionary outgrows the bit width
//     its column was allotted, the fast path is abandoned mid-stream —
//     everything accumulated so far is materialized to string keys and the rest
//     of the input accumulates there.

// TestStringMinMaxIntKeyNullGroupByRow covers case 1. The NULL-keyed group and
// the packed-code groups must both come out right, and neither may absorb the
// other's values.
func TestStringMinMaxIntKeyNullGroupByRow(t *testing.T) {
	n := 6
	// grp: dict-encoded, with row 2 and row 5 NULL so they form the NULL group.
	grpDict := storage.NewDictBuilder()
	for _, s := range []string{"east", "west"} {
		grpDict.Add(s)
	}
	grpCodes := []uint32{0, 1, 0, 0, 1, 0}
	grpNulls := make([]byte, (n+7)/8)
	for _, i := range []int{0, 1, 3, 4} {
		storage.SetValidBit(grpNulls, i)
	}
	grpRead, err := storage.UnmarshalDictionary(grpDict.Marshal())
	if err != nil {
		t.Fatal(err)
	}

	// mode values, with code order reversed relative to lexicographic order.
	modeVals := []string{"MAIL", "TRUCK", "ZULU", "AIR", "SHIP", "ALPHA"}
	modeDict := storage.NewDictBuilder()
	for _, s := range []string{"ZULU", "TRUCK", "SHIP", "MAIL", "ALPHA", "AIR"} {
		modeDict.Add(s)
	}
	modeCodes := make([]uint32, n)
	for i, v := range modeVals {
		modeCodes[i], _ = modeDict.Lookup(v)
	}
	modeRead, err := storage.UnmarshalDictionary(modeDict.Marshal())
	if err != nil {
		t.Fatal(err)
	}

	schema := Schema{Fields: []Field{
		{Name: "grp", Type: TypeString},
		{Name: "mode", Type: TypeString},
	}}
	batch := &Batch{
		Schema: schema,
		Vectors: []Vector{
			&StringVector{Codes: grpCodes, Dict: grpRead, NullBitmap: grpNulls},
			&StringVector{Codes: modeCodes, Dict: modeRead, NullBitmap: storage.FullBitmap(n)},
		},
		Length: n,
	}

	child := &sliceBatchOp{batches: []*Batch{batch}, schema: schema}
	ha, err := NewHashAggregate(child, []int{0}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "lo"},
		{Kind: AggMax, ColIdx: 1, OutName: "hi"},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt"},
	})
	if err != nil {
		t.Fatalf("new hash aggregate: %v", err)
	}
	rows, vecTypes, outSchema := drainToCells(t, ha)
	assertVectorTypesMatchSchema(t, outSchema, vecTypes)

	// east: rows 0 ("MAIL") and 3 ("AIR")  → MIN AIR,   MAX MAIL,  cnt 2
	// west: rows 1 ("TRUCK") and 4 ("SHIP") → MIN SHIP,  MAX TRUCK, cnt 2
	// NULL: rows 2 ("ZULU") and 5 ("ALPHA") → MIN ALPHA, MAX ZULU,  cnt 2
	type want struct{ lo, hi string }
	wants := map[string]want{
		"east":   {"AIR", "MAIL"},
		"west":   {"SHIP", "TRUCK"},
		"<NULL>": {"ALPHA", "ZULU"},
	}
	if len(rows) != 3 {
		t.Fatalf("group count = %d, want 3 (east, west, NULL): %v", len(rows), rows)
	}
	for _, r := range rows {
		grp := "<NULL>"
		if !r[0].isNull {
			grp = r[0].str
		}
		w, ok := wants[grp]
		if !ok {
			t.Errorf("unexpected group %q", grp)
			continue
		}
		if r[1].str != w.lo || r[2].str != w.hi {
			t.Errorf("group %q: MIN/MAX = %v/%v, want %s/%s", grp, r[1], r[2], w.lo, w.hi)
		}
		if r[3].num != 2 {
			t.Errorf("group %q: COUNT(*) = %v, want 2", grp, r[3].num)
		}
	}
}

// TestStringMinMaxIntKeyOverflowFallback covers case 2: the fast path is
// abandoned partway through the input, so a group's running string min/max has
// to survive materializeToStringMaps and then keep accumulating on the
// string-keyed maps.
//
// Overflow needs the packed key to run out of bits. intKeyState.init gives each
// column 8 bits once there are 5–8 GROUP BY columns, so a global dictionary
// reaching 256 entries triggers it. Batch 1 stays small (fast path active and
// populated); batch 2 pushes one column past 256 distinct values.
func TestStringMinMaxIntKeyOverflowFallback(t *testing.T) {
	const numGroupCols = 5
	fields := make([]Field, numGroupCols+1)
	for i := range numGroupCols {
		fields[i] = Field{Name: fmt.Sprintf("g%d", i), Type: TypeString}
	}
	fields[numGroupCols] = Field{Name: "mode", Type: TypeString}
	schema := Schema{Fields: fields}

	// build makes one batch: group column 0 takes the given keys, the other four
	// group columns are constant, and mode takes the given values.
	build := func(keys, modes []string) *Batch {
		n := len(keys)
		vecs := make([]Vector, numGroupCols+1)

		d0 := storage.NewDictBuilder()
		c0 := make([]uint32, n)
		for i, k := range keys {
			c0[i] = d0.Add(k)
		}
		r0, err := storage.UnmarshalDictionary(d0.Marshal())
		if err != nil {
			t.Fatal(err)
		}
		vecs[0] = &StringVector{Codes: c0, Dict: r0, NullBitmap: storage.FullBitmap(n)}

		for c := 1; c < numGroupCols; c++ {
			dc := storage.NewDictBuilder()
			dc.Add("const")
			rc, err := storage.UnmarshalDictionary(dc.Marshal())
			if err != nil {
				t.Fatal(err)
			}
			vecs[c] = &StringVector{Codes: make([]uint32, n), Dict: rc, NullBitmap: storage.FullBitmap(n)}
		}

		dm := storage.NewDictBuilder()
		cm := make([]uint32, n)
		for i, m := range modes {
			cm[i] = dm.Add(m)
		}
		rm, err := storage.UnmarshalDictionary(dm.Marshal())
		if err != nil {
			t.Fatal(err)
		}
		vecs[numGroupCols] = &StringVector{Codes: cm, Dict: rm, NullBitmap: storage.FullBitmap(n)}

		return &Batch{Schema: schema, Vectors: vecs, Length: n}
	}

	// Batch 1: two groups, comfortably inside 8 bits.
	b1 := build(
		[]string{"a", "b", "a", "b"},
		[]string{"MAIL", "TRUCK", "SHIP", "AIR"},
	)
	// Batch 2: 300 distinct keys in column 0 pushes its global dictionary past
	// 256, so the packed key can no longer represent it. Group "a" appears again
	// with a value that must beat what batch 1 recorded for it — proving the
	// carried-over accumulator is still live after the transfer.
	keys2 := make([]string, 0, 301)
	modes2 := make([]string, 0, 301)
	keys2 = append(keys2, "a")
	modes2 = append(modes2, "AAA") // new MIN for group "a"
	for i := range 300 {
		keys2 = append(keys2, fmt.Sprintf("k%03d", i))
		modes2 = append(modes2, "ZZZ")
	}
	b2 := build(keys2, modes2)

	groupBy := []int{0, 1, 2, 3, 4}
	child := &sliceBatchOp{batches: []*Batch{b1, b2}, schema: schema}
	ha, err := NewHashAggregate(child, groupBy, []AggExpr{
		{Kind: AggMin, ColIdx: numGroupCols, OutName: "lo"},
		{Kind: AggMax, ColIdx: numGroupCols, OutName: "hi"},
		{Kind: AggCount, ColIdx: -1, OutName: "cnt"},
	})
	if err != nil {
		t.Fatalf("new hash aggregate: %v", err)
	}
	rows, vecTypes, outSchema := drainToCells(t, ha)
	assertVectorTypesMatchSchema(t, outSchema, vecTypes)

	// The fast path must actually have been abandoned, or this test proves nothing.
	if ha.intKey.enabled {
		t.Fatalf("integer-key path was never abandoned; this test does not exercise the fallback")
	}

	// 2 groups from batch 1 (a, b) + 300 new keys from batch 2 = 302.
	if len(rows) != 302 {
		t.Fatalf("group count = %d, want 302", len(rows))
	}
	byGroup := make(map[string][]aggCell, len(rows))
	for _, r := range rows {
		byGroup[r[0].str] = r
	}
	// Output layout: 5 group columns, then lo, hi, cnt.
	const lo, hi, cnt = numGroupCols, numGroupCols + 1, numGroupCols + 2
	// Group "a": MAIL and SHIP from batch 1, AAA from batch 2 (after the transfer).
	if got := byGroup["a"]; got == nil {
		t.Fatal("group \"a\" missing from output")
	} else {
		if got[lo].str != "AAA" {
			t.Errorf("group a: MIN = %v, want AAA — a value accumulated after the "+
				"intkey→string-key transfer did not reach the carried-over accumulator", got[lo])
		}
		if got[hi].str != "SHIP" {
			t.Errorf("group a: MAX = %v, want SHIP — the value accumulated before the "+
				"transfer was lost", got[hi])
		}
		if got[cnt].num != 3 {
			t.Errorf("group a: COUNT(*) = %v, want 3", got[cnt].num)
		}
	}
	// Group "b" saw only batch 1 and must be unchanged by the transfer.
	if got := byGroup["b"]; got == nil {
		t.Fatal("group \"b\" missing from output")
	} else if got[lo].str != "AIR" || got[hi].str != "TRUCK" {
		t.Errorf("group b: MIN/MAX = %v/%v, want AIR/TRUCK", got[lo], got[hi])
	}
	// A group created entirely after the fallback.
	if got := byGroup["k000"]; got == nil {
		t.Fatal("group \"k000\" missing from output")
	} else if got[lo].str != "ZZZ" || got[hi].str != "ZZZ" {
		t.Errorf("group k000: MIN/MAX = %v/%v, want ZZZ/ZZZ", got[lo], got[hi])
	}
}

// TestStringMinMaxWithCountDistinct puts COUNT(DISTINCT) beside a string
// MIN/MAX. COUNT(DISTINCT) disables the integer-key fast path outright, so this
// runs the composite string-key path in accumulate with both a distinct value set
// and a string accumulator live for the same group — two side tables indexed by
// the same group key, neither of which may disturb the other.
func TestStringMinMaxWithCountDistinct(t *testing.T) {
	batch := strAggBatch([]strRow{
		{grp: "east", mode: "MAIL"},
		{grp: "east", mode: "AIR"},
		{grp: "east", mode: "MAIL"}, // repeat: distinct count stays 2
		{grp: "east", modeNull: true},
		{grp: "west", mode: "SHIP"},
	}, []string{"MAIL", "AIR", "SHIP"}, nil)

	rows, vecTypes, schema := drainStrAgg(t, []*Batch{batch}, []int{0}, []AggExpr{
		{Kind: AggMin, ColIdx: 1, OutName: "lo"},
		{Kind: AggMax, ColIdx: 1, OutName: "hi"},
		{Kind: AggCountDistinct, ColIdx: 1, OutName: "ndistinct", Distinct: true},
		{Kind: AggCount, ColIdx: 1, OutName: "n"},
	})
	assertVectorTypesMatchSchema(t, schema, vecTypes)

	byGroup := make(map[string][]aggCell, len(rows))
	for _, r := range rows {
		byGroup[r[0].str] = r
	}
	if got := byGroup["east"]; got == nil {
		t.Fatal("group \"east\" missing")
	} else {
		if got[1].str != "AIR" || got[2].str != "MAIL" {
			t.Errorf("east: MIN/MAX = %v/%v, want AIR/MAIL", got[1], got[2])
		}
		if got[3].num != 2 {
			t.Errorf("east: COUNT(DISTINCT mode) = %v, want 2", got[3].num)
		}
		if got[4].num != 3 {
			t.Errorf("east: COUNT(mode) = %v, want 3", got[4].num)
		}
	}
	if got := byGroup["west"]; got == nil {
		t.Fatal("group \"west\" missing")
	} else if got[1].str != "SHIP" || got[2].str != "SHIP" || got[3].num != 1 {
		t.Errorf("west: MIN/MAX/NDISTINCT = %v/%v/%v, want SHIP/SHIP/1", got[1], got[2], got[3])
	}
}
