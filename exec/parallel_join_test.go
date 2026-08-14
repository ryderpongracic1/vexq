package exec

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// ---- Fixtures ---------------------------------------------------------------

// int64Vec builds an Int64Vector where a nil entry in vals means NULL.
func int64Vec(vals []*int64) *Int64Vector {
	n := len(vals)
	v := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
	for i, p := range vals {
		if p != nil {
			v.Values[i] = *p
			storage.SetValidBit(v.NullBitmap, i)
		}
	}
	return v
}

func ptr(v int64) *int64 { return &v }

// buildSideBatch returns a (key INT64, payload INT64) batch.
func buildSideBatch(keys []*int64, payload []int64) (*Batch, Schema) {
	schema := Schema{Fields: []Field{
		{Name: "b_key", Type: TypeInt64, Nullable: true},
		{Name: "b_payload", Type: TypeInt64, Nullable: true},
	}}
	pv := make([]*int64, len(payload))
	for i := range payload {
		pv[i] = ptr(payload[i])
	}
	return &Batch{
		Schema:  schema,
		Vectors: []Vector{int64Vec(keys), int64Vec(pv)},
		Length:  len(keys),
	}, schema
}

// probeSideOp returns an operator yielding one (key INT64) batch.
func probeSideOp(keys []*int64) *sliceBatchOp {
	schema := Schema{Fields: []Field{
		{Name: "p_key", Type: TypeInt64, Nullable: true},
	}}
	return &sliceBatchOp{
		schema: schema,
		batches: []*Batch{{
			Schema:  schema,
			Vectors: []Vector{int64Vec(keys)},
			Length:  len(keys),
		}},
	}
}

// drainJoin collects "buildPayload:probeKey" strings from a join operator.
func drainJoin(t *testing.T, op Operator) []string {
	t.Helper()
	ctx := context.Background()
	var out []string
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("join Next: %v", err)
		}
		if batch == nil {
			return out
		}
		for i := 0; i < batch.Length; i++ {
			payload := batch.Vectors[1].(*Int64Vector).Values[i]
			pKey := batch.Vectors[2].(*Int64Vector).Values[i]
			out = append(out, fmt.Sprintf("%d:%d", payload, pKey))
		}
	}
}

// ---- Tests ------------------------------------------------------------------

// TestBuildSharedHashTable_DropsNullKeys verifies the build phase excludes rows
// with a NULL join key — an inner equi-join can never match them — and reports
// accurate row and key counts.
func TestBuildSharedHashTable_DropsNullKeys(t *testing.T) {
	batch, schema := buildSideBatch(
		[]*int64{ptr(1), ptr(2), nil, ptr(2), ptr(3)},
		[]int64{10, 20, 30, 40, 50},
	)
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}

	sht, err := BuildSharedHashTable(context.Background(), build, 0)
	if err != nil {
		t.Fatalf("BuildSharedHashTable: %v", err)
	}
	if got, want := sht.NumRows(), 4; got != want {
		t.Errorf("NumRows = %d, want %d (NULL-keyed row dropped)", got, want)
	}
	if got, want := sht.NumKeys(), 3; got != want {
		t.Errorf("NumKeys = %d, want %d", got, want)
	}
	if got, want := len(sht.Schema().Fields), 2; got != want {
		t.Errorf("Schema fields = %d, want %d", got, want)
	}
}

func TestBuildSharedHashTable_KeyOutOfRange(t *testing.T) {
	batch, schema := buildSideBatch([]*int64{ptr(1)}, []int64{10})
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}
	if _, err := BuildSharedHashTable(context.Background(), build, 7); err == nil {
		t.Fatal("expected an error for an out-of-range build key")
	}
}

// TestNewHashJoinShared_ProbeSemantics checks that probing a shared table
// produces the same rows a self-building HashJoin would: one output row per
// (build row, probe row) key match, NULL probe keys dropped, unmatched keys
// dropped, duplicate build keys fanned out.
func TestNewHashJoinShared_ProbeSemantics(t *testing.T) {
	batch, schema := buildSideBatch(
		[]*int64{ptr(1), ptr(2), ptr(2)},
		[]int64{10, 20, 21},
	)
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}
	sht, err := BuildSharedHashTable(context.Background(), build, 0)
	if err != nil {
		t.Fatalf("BuildSharedHashTable: %v", err)
	}

	probe := probeSideOp([]*int64{ptr(1), ptr(2), nil, ptr(99)})
	join, err := NewHashJoinShared(sht, probe, 0)
	if err != nil {
		t.Fatalf("NewHashJoinShared: %v", err)
	}
	defer join.Close()

	// Output schema is build columns then probe columns.
	wantCols := []string{"b_key", "b_payload", "p_key"}
	for i, name := range wantCols {
		if got := join.Schema().Fields[i].Name; got != name {
			t.Fatalf("output column %d = %q, want %q", i, got, name)
		}
	}

	got := drainJoin(t, join)
	want := []string{"10:1", "20:2", "21:2"}
	if len(got) != len(want) {
		t.Fatalf("join rows = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("join row %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestNewHashJoinShared_Validation(t *testing.T) {
	batch, schema := buildSideBatch([]*int64{ptr(1)}, []int64{10})
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}
	sht, err := BuildSharedHashTable(context.Background(), build, 0)
	if err != nil {
		t.Fatalf("BuildSharedHashTable: %v", err)
	}
	if _, err := NewHashJoinShared(nil, probeSideOp(nil), 0); err == nil {
		t.Error("expected an error for a nil shared hash table")
	}
	if _, err := NewHashJoinShared(sht, probeSideOp(nil), 5); err == nil {
		t.Error("expected an error for an out-of-range probe key")
	}
}

// TestSharedHashTable_ConcurrentProbe is the race-detector gate on the
// concurrency contract: many goroutines probe one table built before any of them
// started. Every worker must see the complete table and produce identical
// results. Run with -race to check the read-only sharing claim.
func TestSharedHashTable_ConcurrentProbe(t *testing.T) {
	const numKeys = 500
	keys := make([]*int64, numKeys)
	payload := make([]int64, numKeys)
	for i := range numKeys {
		keys[i] = ptr(int64(i + 1))
		payload[i] = int64((i + 1) * 10)
	}
	batch, schema := buildSideBatch(keys, payload)
	build := &sliceBatchOp{schema: schema, batches: []*Batch{batch}}
	sht, err := BuildSharedHashTable(context.Background(), build, 0)
	if err != nil {
		t.Fatalf("BuildSharedHashTable: %v", err)
	}

	const workers = 8
	results := make([][]string, workers)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			// Each worker probes a disjoint slice of the key space, mimicking
			// the morsel partitioning of the probe scan.
			lo := w*numKeys/workers + 1
			hi := (w + 1) * numKeys / workers
			var pk []*int64
			for k := lo; k <= hi; k++ {
				pk = append(pk, ptr(int64(k)))
			}
			join, err := NewHashJoinShared(sht, probeSideOp(pk), 0)
			if err != nil {
				t.Errorf("worker %d: NewHashJoinShared: %v", w, err)
				return
			}
			defer join.Close()
			ctx := context.Background()
			var rows []string
			for {
				b, err := join.Next(ctx)
				if err != nil {
					t.Errorf("worker %d: Next: %v", w, err)
					return
				}
				if b == nil {
					break
				}
				for i := 0; i < b.Length; i++ {
					rows = append(rows, fmt.Sprintf("%d:%d",
						b.Vectors[1].(*Int64Vector).Values[i],
						b.Vectors[2].(*Int64Vector).Values[i]))
				}
			}
			results[w] = rows
		}(w)
	}
	wg.Wait()

	total := 0
	for w, rows := range results {
		if len(rows) == 0 {
			t.Fatalf("worker %d produced no rows", w)
		}
		for _, r := range rows {
			var payload, key int64
			if _, err := fmt.Sscanf(r, "%d:%d", &payload, &key); err != nil {
				t.Fatalf("worker %d: unparseable row %q", w, r)
			}
			if payload != key*10 {
				t.Fatalf("worker %d: row %q pairs the wrong build row with key %d", w, r, key)
			}
		}
		total += len(rows)
	}
	if total != numKeys {
		t.Fatalf("total joined rows = %d, want %d", total, numKeys)
	}
}

func TestNewSchemaOnly(t *testing.T) {
	schema := Schema{Fields: []Field{{Name: "a", Type: TypeInt64}}}
	op := NewSchemaOnly(schema)
	if got := len(op.Schema().Fields); got != 1 {
		t.Fatalf("schema fields = %d, want 1", got)
	}
	batch, err := op.Next(context.Background())
	if err != nil || batch != nil {
		t.Fatalf("Next = (%v, %v), want (nil, nil)", batch, err)
	}
	if err := op.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
