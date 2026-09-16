package storage

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// writeAllTypes writes a file with one column of every type across two row
// groups whose row counts span several blocks, with NULLs, negative values and
// NaN-free floats, and returns its path.
func writeAllTypes(t *testing.T) string {
	t.Helper()
	schema := makeSchema(
		Field{Name: "i", Type: TypeInt64, Nullable: true},
		Field{Name: "f", Type: TypeFloat64, Nullable: true},
		Field{Name: "d", Type: TypeDate, Nullable: true},
		Field{Name: "s", Type: TypeString, Encoding: EncDict, Nullable: true},
		Field{Name: "b", Type: TypeBool, Encoding: EncRLE, Nullable: true},
		Field{Name: "empty", Type: TypeInt64, Nullable: true},
	)
	w, path := newTestWriter(t, schema)
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{BlockRows*3 + 17, 5} {
		nulls := FullBitmap(n)
		allNull := make([]byte, (n+7)/8)
		ints := make([]int64, n)
		floats := make([]float64, n)
		dates := make([]int32, n)
		strs := make([]string, n)
		bools := make([]bool, n)
		for i := 0; i < n; i++ {
			if rng.Intn(5) == 0 {
				SetNullBit(nulls, i)
			}
			ints[i] = rng.Int63n(2000) - 1000
			floats[i] = (rng.Float64() - 0.5) * 1e6
			dates[i] = int32(rng.Intn(40000) - 20000)
			strs[i] = []string{"a", "b", "c", "d"}[rng.Intn(4)]
			bools[i] = rng.Intn(2) == 0
		}
		if err := w.BeginRowGroup(n); err != nil {
			t.Fatal(err)
		}
		for col, vals := range []any{ints, floats, dates, strs, bools, ints} {
			bm := nulls
			if col == 5 {
				bm = allNull
			}
			if err := w.AppendColumn(ctx, col, bm, vals); err != nil {
				t.Fatalf("AppendColumn %d: %v", col, err)
			}
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatal(err)
		}
	}
	finishWriter(t, w)
	return path
}

func TestComputeColumnStatsMatchesWriter(t *testing.T) {
	r := openReader(t, writeAllTypes(t))
	for rg := range r.Meta().RowGroups {
		for col, f := range r.Meta().Schema.Fields {
			if err := r.VerifyColumnStats(ctx, rg, col); err != nil {
				t.Errorf("row group %d column %s: %v", rg, f.Name, err)
			}
		}
	}
}

func TestVerifyColumnStatsDetectsMismatch(t *testing.T) {
	cases := []struct {
		name   string
		col    int
		tamper func(z *ZoneMap)
	}{
		{"int_max_too_small", 0, func(z *ZoneMap) { neg := int64(-2000); z.Max = uint64(neg) }},
		{"int_min_too_large", 0, func(z *ZoneMap) { z.Min = 5000 }},
		{"float_min_too_large", 1, func(z *ZoneMap) { z.Min = math.Float64bits(1e9) }},
		{"date_max_too_small", 2, func(z *ZoneMap) { neg := int32(-30000); z.Max = uint64(uint32(neg)) }},
		{"null_count", 3, func(z *ZoneMap) { z.NullCount++ }},
		{"bool_null_count", 4, func(z *ZoneMap) { z.NullCount = 0 }},
		{"all_null_claims_range", 5, func(z *ZoneMap) { z.HasMinMax = true }},
		{"int_sum", 0, func(z *ZoneMap) { z.Sum++ }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := openReader(t, writeAllTypes(t))
			tc.tamper(&r.Meta().RowGroups[0].Columns[tc.col].Stats)
			err := r.VerifyColumnStats(ctx, 0, tc.col)
			if err == nil {
				t.Fatal("tampered zone map verified as consistent")
			}
			if !strings.Contains(err.Error(), "zone map does not match data") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
