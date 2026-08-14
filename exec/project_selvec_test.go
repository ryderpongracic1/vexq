package exec_test

import (
	"context"
	"testing"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// TestProjectWithSelVec verifies that Project correctly handles batches with
// an active selection vector (from upstream Filter or Limit operators).
// This is a regression test for a panic: index out of range in evalArith
// when Literal.Eval uses batch.Length (logical/selected count) to size its
// broadcast vector, but ColumnRef.Eval returns the full physical vector.
func TestProjectWithSelVec(t *testing.T) {
	ctx := context.Background()

	// Build a batch with 16 physical rows.
	const physicalRows = 16
	vals := make([]float64, physicalRows)
	ids := make([]int64, physicalRows)
	for i := range vals {
		vals[i] = float64(i+1) * 100.0
		ids[i] = int64(i + 1)
	}

	makeBatch := func(sel exec.SelectionVector) *exec.Batch {
		idVec := &exec.Int64Vector{
			Values:     ids,
			NullBitmap: storage.FullBitmap(physicalRows),
		}
		valVec := &exec.Float64Vector{
			Values:     vals,
			NullBitmap: storage.FullBitmap(physicalRows),
		}
		b := &exec.Batch{
			Schema: exec.Schema{Fields: []exec.Field{
				{Name: "id", Type: exec.TypeInt64, Nullable: false},
				{Name: "amount", Type: exec.TypeFloat64, Nullable: true},
			}},
			Vectors: []exec.Vector{idVec, valVec},
			Length:  physicalRows,
		}
		if sel != nil {
			b.SelVec = sel
			b.Length = len(sel)
		}
		return b
	}

	// A trivial operator that yields a single batch then nil.
	type oneShotOp struct {
		batch  *exec.Batch
		schema exec.Schema
		done   bool
	}
	nextFn := func(o *oneShotOp) func(context.Context) (*exec.Batch, error) {
		return func(_ context.Context) (*exec.Batch, error) {
			if o.done {
				return nil, nil
			}
			o.done = true
			return o.batch, nil
		}
	}
	_ = nextFn // suppress unused warning — we use the mock below

	tests := []struct {
		name string
		sel  exec.SelectionVector
		want int // expected output row count
	}{
		{
			name: "filtered_batch_sparse",
			sel:  exec.SelectionVector{2, 7, 11, 14}, // 4 survivors
			want: 4,
		},
		{
			name: "all_rows_selected",
			sel:  nil, // no SelVec = all rows
			want: physicalRows,
		},
		{
			name: "last_row_only",
			sel:  exec.SelectionVector{uint16(physicalRows - 1)},
			want: 1,
		},
		{
			name: "zero_survivors",
			sel:  exec.SelectionVector{}, // empty SelVec
			want: 0,
		},
		{
			name: "all_rows_via_selvec",
			sel: func() exec.SelectionVector {
				s := make(exec.SelectionVector, physicalRows)
				for i := range s {
					s[i] = uint16(i)
				}
				return s
			}(),
			want: physicalRows,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			batch := makeBatch(tc.sel)

			// Build a mock operator that yields this batch once.
			child := &mockOperator{
				schema:  batch.Schema,
				batches: []*exec.Batch{batch},
			}

			// Project: amount * 1.1 (Float64 column * Float64 literal)
			proj, err := exec.NewProject(child, []exec.ProjectExpr{
				{
					Name: "id",
					Expr: &exec.ColumnRef{Name: "id", Idx: 0, T: exec.TypeInt64},
				},
				{
					Name: "with_tax",
					Expr: &exec.BinOp{
						Op:    exec.BinMul,
						Left:  &exec.ColumnRef{Name: "amount", Idx: 1, T: exec.TypeFloat64},
						Right: &exec.Literal{Val: float64(1.1), T: exec.TypeFloat64},
						T:     exec.TypeFloat64,
					},
				},
			})
			if err != nil {
				t.Fatalf("NewProject: %v", err)
			}
			defer proj.Close()

			out, err := proj.Next(ctx)
			if err != nil {
				t.Fatalf("Next: %v", err)
			}

			if tc.want == 0 {
				// With zero survivors, output should still be a valid batch
				// (possibly with Length 0).
				if out != nil && out.Length != 0 {
					t.Fatalf("expected 0-length batch or nil, got Length=%d", out.Length)
				}
				return
			}

			if out == nil {
				t.Fatal("expected non-nil batch")
			}
			if out.Length != tc.want {
				t.Fatalf("output Length=%d, want %d", out.Length, tc.want)
			}

			// Verify output has no SelVec (materialized).
			if out.SelVec != nil {
				t.Fatal("output batch should have nil SelVec after materialization")
			}

			// Verify the arithmetic is correct for selected rows.
			idVec := out.Vectors[0].(*exec.Int64Vector)
			taxVec := out.Vectors[1].(*exec.Float64Vector)
			if len(taxVec.Values) != tc.want {
				t.Fatalf("taxVec has %d values, want %d", len(taxVec.Values), tc.want)
			}
			for i := 0; i < tc.want; i++ {
				var srcIdx int
				if tc.sel != nil {
					srcIdx = int(tc.sel[i])
				} else {
					srcIdx = i
				}
				expectedID := int64(srcIdx + 1)
				expectedTax := float64(srcIdx+1) * 100.0 * 1.1
				if idVec.Values[i] != expectedID {
					t.Errorf("row %d: id=%d, want %d", i, idVec.Values[i], expectedID)
				}
				if diff := taxVec.Values[i] - expectedTax; diff > 0.001 || diff < -0.001 {
					t.Errorf("row %d: with_tax=%f, want %f", i, taxVec.Values[i], expectedTax)
				}
			}
		})
	}
}

// mockOperator yields pre-built batches in sequence, then nil.
type mockOperator struct {
	schema  exec.Schema
	batches []*exec.Batch
	pos     int
}

func (m *mockOperator) Schema() exec.Schema { return m.schema }
func (m *mockOperator) Next(_ context.Context) (*exec.Batch, error) {
	if m.pos >= len(m.batches) {
		return nil, nil
	}
	b := m.batches[m.pos]
	m.pos++
	return b, nil
}
func (m *mockOperator) Close() error { return nil }
