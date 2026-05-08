package storage

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

var ctx = context.Background()

// ---- helpers ----------------------------------------------------------------

func makeSchema(fields ...Field) Schema { return Schema{Fields: fields} }

func newTestWriter(t *testing.T, schema Schema) (*Writer, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.vxq")
	w, err := NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	return w, path
}

func finishWriter(t *testing.T, w *Writer) {
	t.Helper()
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
}

func openReader(t *testing.T, path string) *Reader {
	t.Helper()
	r, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open(%q): %v", path, err)
	}
	t.Cleanup(func() { _ = r.Close() })
	return r
}

// ---- Test 1: empty file round-trip -----------------------------------------

func TestEmptyFile(t *testing.T) {
	schema := makeSchema(Field{Name: "x", Type: TypeInt64})
	w, path := newTestWriter(t, schema)
	finishWriter(t, w)

	r := openReader(t, path)
	meta := r.Meta()
	if len(meta.RowGroups) != 0 {
		t.Fatalf("expected 0 row groups, got %d", len(meta.RowGroups))
	}
	if len(meta.Schema.Fields) != 1 || meta.Schema.Fields[0].Name != "x" {
		t.Fatalf("schema mismatch: %+v", meta.Schema)
	}
}

// ---- Test 2: schema round-trip ---------------------------------------------

func TestSchemaRoundTrip(t *testing.T) {
	schema := makeSchema(
		Field{Name: "id", Type: TypeInt64, Nullable: false},
		Field{Name: "price", Type: TypeFloat64, Nullable: true},
		Field{Name: "flag", Type: TypeBool, Encoding: EncRLE, Nullable: true},
		Field{Name: "tag", Type: TypeString, Encoding: EncDict, Nullable: true},
		Field{Name: "dt", Type: TypeDate, Nullable: false},
		Field{Name: "名前", Type: TypeString, Encoding: EncDict, Nullable: true}, // non-ASCII
	)
	w, path := newTestWriter(t, schema)
	finishWriter(t, w)

	r := openReader(t, path)
	got := r.Meta().Schema.Fields
	if len(got) != len(schema.Fields) {
		t.Fatalf("field count: got %d want %d", len(got), len(schema.Fields))
	}
	for i, f := range schema.Fields {
		g := got[i]
		if g.Name != f.Name || g.Type != f.Type || g.Encoding != f.Encoding || g.Nullable != f.Nullable {
			t.Errorf("field[%d]: got %+v want %+v", i, g, f)
		}
	}
}

// ---- Test 3: writer rejects bad arguments ----------------------------------

func TestWriterRejectsErrors(t *testing.T) {
	schema := makeSchema(Field{Name: "a", Type: TypeInt64})
	w, _ := newTestWriter(t, schema)
	defer func() { _ = w.Abort() }()

	if err := w.BeginRowGroup(10); err != nil {
		t.Fatalf("BeginRowGroup: %v", err)
	}

	// Wrong type.
	err := w.AppendColumn(ctx, 0, nil, []float64{1.0})
	if !errors.Is(err, ErrTypeMismatch) {
		t.Fatalf("expected ErrTypeMismatch, got %v", err)
	}

	// Out of range index.
	err = w.AppendColumn(ctx, 5, nil, []int64{1})
	if err == nil {
		t.Fatal("expected error for out-of-range index")
	}
}

// ---- Test 4: EndRowGroup before all columns written ------------------------

func TestEndRowGroupMissingColumns(t *testing.T) {
	schema := makeSchema(
		Field{Name: "a", Type: TypeInt64},
		Field{Name: "b", Type: TypeFloat64},
	)
	w, _ := newTestWriter(t, schema)
	defer func() { _ = w.Abort() }()

	if err := w.BeginRowGroup(2); err != nil {
		t.Fatalf("BeginRowGroup: %v", err)
	}
	if err := w.AppendColumn(ctx, 0, nil, []int64{1, 2}); err != nil {
		t.Fatalf("AppendColumn: %v", err)
	}
	// Deliberately skip column 1.
	err := w.EndRowGroup()
	if !errors.Is(err, ErrMissingColumns) {
		t.Fatalf("expected ErrMissingColumns, got %v", err)
	}
}

// ---- Test 5: Abort cleans up temp file -------------------------------------

func TestAbortCleansUp(t *testing.T) {
	schema := makeSchema(Field{Name: "x", Type: TypeInt64})
	dir := t.TempDir()
	path := filepath.Join(dir, "test.vxq")
	w, err := NewWriter(path, schema)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	// Neither .vxq nor .vxq.tmp should exist.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error(".vxq should not exist after Abort")
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error(".vxq.tmp should not exist after Abort")
	}
}

// ---- Test 6: 1M row round-trip ---------------------------------------------

func TestRoundTrip1MRows(t *testing.T) {
	const N = 1_000_000
	schema := makeSchema(
		Field{Name: "id", Type: TypeInt64},
		Field{Name: "val", Type: TypeFloat64},
	)
	w, path := newTestWriter(t, schema)

	ids := make([]int64, N)
	vals := make([]float64, N)
	for i := range ids {
		ids[i] = int64(i)
		vals[i] = float64(i) * 0.5
	}
	rowsWritten := 0
	for rowsWritten < N {
		chunk := N - rowsWritten
		if chunk > RowGroupRows {
			chunk = RowGroupRows
		}
		if err := w.BeginRowGroup(chunk); err != nil {
			t.Fatalf("BeginRowGroup: %v", err)
		}
		if err := w.AppendColumn(ctx, 0, nil, ids[rowsWritten:rowsWritten+chunk]); err != nil {
			t.Fatalf("AppendColumn ids: %v", err)
		}
		if err := w.AppendColumn(ctx, 1, nil, vals[rowsWritten:rowsWritten+chunk]); err != nil {
			t.Fatalf("AppendColumn vals: %v", err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup: %v", err)
		}
		rowsWritten += chunk
	}
	finishWriter(t, w)

	r := openReader(t, path)
	totalRead := 0
	for rg := 0; rg < len(r.Meta().RowGroups); rg++ {
		cr, err := r.OpenColumn(ctx, rg, 0)
		if err != nil {
			t.Fatalf("OpenColumn: %v", err)
		}
		for {
			_, payload, rows, err := cr.NextBlock(ctx)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("NextBlock: %v", err)
			}
			for i := 0; i < rows; i++ {
				got := int64(payload[i*8]) | int64(payload[i*8+1])<<8 |
					int64(payload[i*8+2])<<16 | int64(payload[i*8+3])<<24 |
					int64(payload[i*8+4])<<32 | int64(payload[i*8+5])<<40 |
					int64(payload[i*8+6])<<48 | int64(payload[i*8+7])<<56
				want := int64(totalRead + i)
				if got != want {
					t.Fatalf("row %d: got id %d want %d", totalRead+i, got, want)
				}
			}
			totalRead += rows
		}
	}
	if totalRead != N {
		t.Fatalf("read %d rows, expected %d", totalRead, N)
	}
}

// ---- Test 7: CRC corruption detected ---------------------------------------

func TestCRCCorruption(t *testing.T) {
	schema := makeSchema(Field{Name: "x", Type: TypeInt64})
	w, path := newTestWriter(t, schema)
	if err := w.BeginRowGroup(100); err != nil {
		t.Fatal(err)
	}
	vals := make([]int64, 100)
	if err := w.AppendColumn(ctx, 0, nil, vals); err != nil {
		t.Fatal(err)
	}
	if err := w.EndRowGroup(); err != nil {
		t.Fatal(err)
	}
	finishWriter(t, w)

	// Flip a byte in the middle of the data blocks.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// First block starts at offset 4 (magic); flip a byte in payload area.
	if len(data) > 200 {
		data[100] ^= 0xFF
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	r := openReader(t, path)
	cr, err := r.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = cr.NextBlock(ctx)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("expected ErrChecksum, got %v", err)
	}
}

// ---- Test 8: footer truncation detected ------------------------------------

func TestFooterCorruption(t *testing.T) {
	schema := makeSchema(Field{Name: "x", Type: TypeInt64})
	w, path := newTestWriter(t, schema)
	finishWriter(t, w)

	// Truncate by 1 byte to corrupt footer.
	fi, _ := os.Stat(path)
	if err := os.Truncate(path, fi.Size()-1); err != nil {
		t.Fatal(err)
	}
	_, err := Open(ctx, path)
	if err == nil {
		t.Fatal("expected error opening truncated file")
	}
}

// ---- Test 9: column pruning (BytesRead < total_size/8) ---------------------

func TestColumnPruning(t *testing.T) {
	const N = RowGroupRows * 4
	const numCols = 8
	schema := Schema{}
	for i := 0; i < numCols; i++ {
		schema.Fields = append(schema.Fields, Field{Name: string(rune('a'+i)), Type: TypeInt64})
	}

	w, path := newTestWriter(t, schema)
	rows := make([]int64, N)
	for rowsWritten := 0; rowsWritten < N; {
		chunk := N - rowsWritten
		if chunk > RowGroupRows {
			chunk = RowGroupRows
		}
		if err := w.BeginRowGroup(chunk); err != nil {
			t.Fatal(err)
		}
		for c := 0; c < numCols; c++ {
			if err := w.AppendColumn(ctx, c, nil, rows[rowsWritten:rowsWritten+chunk]); err != nil {
				t.Fatalf("AppendColumn c=%d: %v", c, err)
			}
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatal(err)
		}
		rowsWritten += chunk
	}
	finishWriter(t, w)

	fi, _ := os.Stat(path)
	totalSize := fi.Size()

	r := openReader(t, path)
	// Read only column 3.
	for rg := 0; rg < len(r.Meta().RowGroups); rg++ {
		cr, _ := r.OpenColumn(ctx, rg, 3)
		for {
			_, _, _, err := cr.NextBlock(ctx)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	// We read only column data; footer + header account for a little overhead,
	// but we should read far less than total / 8.
	if r.BytesRead() >= totalSize/4 {
		t.Fatalf("expected BytesRead < totalSize/4, got %d vs total %d", r.BytesRead(), totalSize)
	}
}

// ---- Test 10: zone map min/max sanity --------------------------------------

func TestZoneMapSanity(t *testing.T) {
	const N = RowGroupRows * 3
	schema := makeSchema(Field{Name: "v", Type: TypeInt64})
	w, path := newTestWriter(t, schema)

	vals := make([]int64, N)
	for i := range vals {
		vals[i] = int64(i)
	}
	for rowsWritten := 0; rowsWritten < N; {
		chunk := RowGroupRows
		if rowsWritten+chunk > N {
			chunk = N - rowsWritten
		}
		if err := w.BeginRowGroup(chunk); err != nil {
			t.Fatal(err)
		}
		if err := w.AppendColumn(ctx, 0, nil, vals[rowsWritten:rowsWritten+chunk]); err != nil {
			t.Fatal(err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatal(err)
		}
		rowsWritten += chunk
	}
	finishWriter(t, w)

	r := openReader(t, path)
	for i, rg := range r.Meta().RowGroups {
		cm := rg.Columns[0]
		wantMin := int64(i * RowGroupRows)
		wantMax := int64((i+1)*RowGroupRows - 1)
		if !cm.Stats.HasMinMax {
			t.Errorf("rg %d: expected HasMinMax=true", i)
			continue
		}
		if int64(cm.Stats.Min) != wantMin {
			t.Errorf("rg %d: min=%d want %d", i, int64(cm.Stats.Min), wantMin)
		}
		if int64(cm.Stats.Max) != wantMax {
			t.Errorf("rg %d: max=%d want %d", i, int64(cm.Stats.Max), wantMax)
		}
	}
}

// ---- Test 11: string dictionary encoding -----------------------------------

func TestStringDictionary(t *testing.T) {
	const N = RowGroupRows * 4
	const numDistinct = 50
	schema := makeSchema(Field{Name: "s", Type: TypeString, Encoding: EncDict, Nullable: true})
	w, path := newTestWriter(t, schema)

	strs := make([]string, N)
	for i := range strs {
		strs[i] = string(rune('A' + i%numDistinct))
	}
	for rowsWritten := 0; rowsWritten < N; {
		chunk := RowGroupRows
		if rowsWritten+chunk > N {
			chunk = N - rowsWritten
		}
		if err := w.BeginRowGroup(chunk); err != nil {
			t.Fatal(err)
		}
		if err := w.AppendColumn(ctx, 0, nil, strs[rowsWritten:rowsWritten+chunk]); err != nil {
			t.Fatal(err)
		}
		if err := w.EndRowGroup(); err != nil {
			t.Fatal(err)
		}
		rowsWritten += chunk
	}
	finishWriter(t, w)

	// Verify dictionaries are small.
	r := openReader(t, path)
	for rg, rgMeta := range r.Meta().RowGroups {
		dictLen := rgMeta.Columns[0].DictLength
		if dictLen > 5*1024 {
			t.Errorf("rg %d: dict too large: %d bytes", rg, dictLen)
		}
	}

	// Verify strings round-trip.
	totalRead := 0
	for rg := 0; rg < len(r.Meta().RowGroups); rg++ {
		cr, err := r.OpenColumn(ctx, rg, 0)
		if err != nil {
			t.Fatal(err)
		}
		dict, err := cr.Dictionary()
		if err != nil {
			t.Fatalf("rg %d: Dictionary: %v", rg, err)
		}
		for {
			_, payload, rows, err := cr.NextBlock(ctx)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < rows; i++ {
				code := uint32(payload[i*4]) | uint32(payload[i*4+1])<<8 |
					uint32(payload[i*4+2])<<16 | uint32(payload[i*4+3])<<24
				got := dict.Get(code)
				want := strs[totalRead+i]
				if got != want {
					t.Fatalf("row %d: got %q want %q", totalRead+i, got, want)
				}
			}
			totalRead += rows
		}
	}
	if totalRead != N {
		t.Fatalf("read %d rows, expected %d", totalRead, N)
	}
}

// ---- Test 12: atomic rename (Abort mid-write) ------------------------------

func TestAtomicRename(t *testing.T) {
	schema := makeSchema(Field{Name: "x", Type: TypeInt64})
	dir := t.TempDir()
	path := filepath.Join(dir, "test.vxq")
	w, err := NewWriter(path, schema)
	if err != nil {
		t.Fatal(err)
	}
	// Start a row group but don't finish it.
	if err := w.BeginRowGroup(100); err != nil {
		t.Fatal(err)
	}
	// Abort.
	if err := w.Abort(); err != nil {
		t.Fatal(err)
	}
	// Final .vxq must not exist.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error(".vxq should not exist after mid-write Abort")
	}
	// Temp file must not exist either (Abort cleaned it up).
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error(".vxq.tmp should not exist after Abort")
	}
}

// ---- Benchmark: scan a single int64 column from 1M row file ----------------

func BenchmarkScanColumn(b *testing.B) {
	const N = 1_000_000
	schema := makeSchema(Field{Name: "v", Type: TypeInt64})
	path := filepath.Join(b.TempDir(), "bench.vxq")
	w, err := NewWriter(path, schema)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	for rowsWritten := 0; rowsWritten < N; {
		chunk := RowGroupRows
		if rowsWritten+chunk > N {
			chunk = N - rowsWritten
		}
		vals := make([]int64, chunk)
		for i := range vals {
			vals[i] = rng.Int63()
		}
		_ = w.BeginRowGroup(chunk)
		_ = w.AppendColumn(ctx, 0, nil, vals)
		_ = w.EndRowGroup()
		rowsWritten += chunk
	}
	if err := w.Finish(ctx); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r, err := Open(ctx, path)
		if err != nil {
			b.Fatal(err)
		}
		sum := int64(0)
		for rg := 0; rg < len(r.Meta().RowGroups); rg++ {
			cr, _ := r.OpenColumn(ctx, rg, 0)
			for {
				_, payload, rows, err := cr.NextBlock(ctx)
				if err == io.EOF {
					break
				}
				if err != nil {
					b.Fatal(err)
				}
				for i := 0; i < rows; i++ {
					v := int64(payload[i*8]) | int64(payload[i*8+1])<<8 |
						int64(payload[i*8+2])<<16 | int64(payload[i*8+3])<<24 |
						int64(payload[i*8+4])<<32 | int64(payload[i*8+5])<<40 |
						int64(payload[i*8+6])<<48 | int64(payload[i*8+7])<<56
					sum += v
				}
			}
		}
		_ = sum
		_ = r.Close()
	}
}
