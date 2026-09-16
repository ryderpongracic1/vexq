package exec_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// int64Batches returns batches of sizes holding consecutive values from 0. When
// filtered is set, every batch carries a selection vector choosing its even
// physical rows, so the operator under test sees logical rows that are not the
// batch's physical prefix.
func int64Batches(sizes []int, filtered bool) []*exec.Batch {
	var out []*exec.Batch
	next := int64(0)
	for _, n := range sizes {
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = next
			next++
		}
		b := &exec.Batch{
			Vectors: []exec.Vector{&exec.Int64Vector{Values: vals, NullBitmap: storage.FullBitmap(n)}},
			Length:  n,
		}
		if filtered {
			var sel exec.SelectionVector
			for i := 0; i < n; i += 2 {
				sel = append(sel, uint16(i))
			}
			b.SelVec = sel
			b.Length = len(sel)
		}
		out = append(out, b)
	}
	return out
}

func drainInt64(t *testing.T, op exec.Operator) []int64 {
	t.Helper()
	var got []int64
	for {
		b, err := op.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if b == nil {
			return got
		}
		vals := b.Vectors[0].(*exec.Int64Vector).Values
		for i := 0; i < b.Length; i++ {
			idx := i
			if b.SelVec != nil {
				idx = int(b.SelVec[i])
			}
			got = append(got, vals[idx])
		}
	}
}

func TestLimitOffsetAcrossBatches(t *testing.T) {
	schema := exec.Schema{Fields: []exec.Field{{Name: "v", Type: exec.TypeInt64}}}
	sizes := []int{4, 4, 4} // values 0..11

	cases := []struct {
		limit, offset int
		filtered      bool
		want          string
	}{
		{limit: 3, offset: 0, want: "[0 1 2]"},
		{limit: 3, offset: 2, want: "[2 3 4]"},      // offset inside the first batch, limit spans a boundary
		{limit: 2, offset: 4, want: "[4 5]"},        // offset consumes a whole batch exactly
		{limit: 5, offset: 6, want: "[6 7 8 9 10]"}, // offset into the second batch
		{limit: -1, offset: 9, want: "[9 10 11]"},   // OFFSET without LIMIT
		{limit: 4, offset: 12, want: "[]"},          // offset past the end
		{limit: 0, offset: 1, want: "[]"},           // LIMIT 0
		{limit: -1, offset: 0, want: "[0 1 2 3 4 5 6 7 8 9 10 11]"},
		{limit: 2, offset: 3, filtered: true, want: "[6 8]"}, // logical rows 0 2 | 4 6 | 8 10
		{limit: -1, offset: 1, filtered: true, want: "[2 4 6 8 10]"},
	}
	for _, tc := range cases {
		name := fmt.Sprintf("limit=%d/offset=%d/filtered=%v", tc.limit, tc.offset, tc.filtered)
		t.Run(name, func(t *testing.T) {
			child := &mockOperator{schema: schema, batches: int64Batches(sizes, tc.filtered)}
			got := drainInt64(t, exec.NewLimitOffset(child, tc.limit, tc.offset))
			if s := fmt.Sprint(got); s != tc.want && !(len(got) == 0 && tc.want == "[]") {
				t.Fatalf("got %v, want %s", got, tc.want)
			}
		})
	}
}
