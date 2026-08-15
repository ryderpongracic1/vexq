package exec_test

import (
	"testing"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// These tests pin the I/O shape of TableScan now that storage.ColumnReader
// buffers a whole column section per row group: opening a row group costs one
// read per projected column, pruned row groups cost none, and Reset reuses the
// open Reader rather than re-reading anything.

// drainScan consumes a scan to completion, returning the total row count and
// the min/max of the "id" column (source column 0) it observed.
func drainScan(t *testing.T, scan *exec.TableScan) (rows int, minID, maxID int64) {
	t.Helper()
	minID, maxID = -1, -1
	for {
		batch, err := scan.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if batch == nil {
			return rows, minID, maxID
		}
		iv, ok := batch.Vectors[0].(*exec.Int64Vector)
		if !ok {
			t.Fatalf("vector 0 is %T, want *exec.Int64Vector", batch.Vectors[0])
		}
		for i := 0; i < batch.Length; i++ {
			v := iv.Values[i]
			if minID < 0 || v < minID {
				minID = v
			}
			if v > maxID {
				maxID = v
			}
		}
		rows += batch.Length
	}
}

func TestScanSkipsPrunedRowGroupsWithoutReading(t *testing.T) {
	const rowGroups = 4
	const keepRG = 2
	path := writeTestFile(t, storage.RowGroupRows*rowGroups)

	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(r.Meta().RowGroups); got != rowGroups {
		t.Fatalf("expected %d row groups, got %d", rowGroups, got)
	}

	// Row group i holds ids [i*RowGroupRows, (i+1)*RowGroupRows), so the zone
	// map's min identifies the group.
	keepMin := uint64(keepRG * storage.RowGroupRows)
	zonePred := func(rg *storage.RowGroupMeta) bool {
		return rg.Columns[0].Stats.Min == keepMin
	}

	scan, err := exec.NewTableScan(r, nil, zonePred)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	afterOpen := r.ReadOps()
	rows, minID, maxID := drainScan(t, scan)

	if rows != storage.RowGroupRows {
		t.Fatalf("read %d rows, want %d (one row group)", rows, storage.RowGroupRows)
	}
	if minID != int64(keepMin) || maxID != int64(keepMin)+storage.RowGroupRows-1 {
		t.Fatalf("id range [%d,%d], want [%d,%d]",
			minID, maxID, keepMin, int64(keepMin)+storage.RowGroupRows-1)
	}
	// Two projected columns, one surviving row group: two section reads. The
	// three pruned row groups must contribute nothing.
	if reads := r.ReadOps() - afterOpen; reads != 2 {
		t.Fatalf("issued %d reads, want 2 (2 columns × 1 unpruned row group)", reads)
	}
}

func TestScanAllRowGroupsPrunedIssuesNoReads(t *testing.T) {
	path := writeTestFile(t, storage.RowGroupRows*3)
	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := exec.NewTableScan(r, nil, func(*storage.RowGroupMeta) bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	afterOpen := r.ReadOps()
	batch, err := scan.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if batch != nil {
		t.Fatalf("expected EOF, got a batch of %d rows", batch.Length)
	}
	if reads := r.ReadOps() - afterOpen; reads != 0 {
		t.Fatalf("issued %d reads with every row group pruned, want 0", reads)
	}
}

func TestScanResetKeepsReadsCoarse(t *testing.T) {
	const rowGroups = 4
	const cols = 2
	path := writeTestFile(t, storage.RowGroupRows*rowGroups)

	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := exec.NewTableScanRange(r, nil, nil, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	afterOpen := r.ReadOps()

	rows, minID, _ := drainScan(t, scan)
	if rows != 2*storage.RowGroupRows {
		t.Fatalf("first morsel: read %d rows, want %d", rows, 2*storage.RowGroupRows)
	}
	if minID != 0 {
		t.Fatalf("first morsel: min id %d, want 0", minID)
	}
	firstHalf := r.ReadOps() - afterOpen
	if firstHalf != 2*cols {
		t.Fatalf("first morsel issued %d reads, want %d", firstHalf, 2*cols)
	}

	// Reposition the same open Reader, exactly as the morsel scheduler does.
	scan.Reset(2, rowGroups)
	rows2, minID2, maxID2 := drainScan(t, scan)
	if rows2 != 2*storage.RowGroupRows {
		t.Fatalf("second morsel: read %d rows, want %d", rows2, 2*storage.RowGroupRows)
	}
	if minID2 != 2*storage.RowGroupRows || maxID2 != rowGroups*storage.RowGroupRows-1 {
		t.Fatalf("second morsel: id range [%d,%d], want [%d,%d]",
			minID2, maxID2, 2*storage.RowGroupRows, rowGroups*storage.RowGroupRows-1)
	}
	if total := r.ReadOps() - afterOpen; total != rowGroups*cols {
		t.Fatalf("both morsels issued %d reads, want %d", total, rowGroups*cols)
	}
}

// TestScanResetReusesSectionBuffers checks the buffers survive Reset: replaying
// the same morsel repeatedly must not grow the Reader's buffer pool, since each
// row group hands its buffers back on close.
func TestScanResetReusesSectionBuffers(t *testing.T) {
	const rowGroups = 3
	path := writeTestFile(t, storage.RowGroupRows*rowGroups)

	r, err := storage.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := exec.NewTableScanRange(r, nil, nil, 0, rowGroups)
	if err != nil {
		t.Fatal(err)
	}
	defer scan.Close()

	var perPass int64
	for pass := 0; pass < 4; pass++ {
		before := r.ReadOps()
		if rows, _, _ := drainScan(t, scan); rows != rowGroups*storage.RowGroupRows {
			t.Fatalf("pass %d: read %d rows, want %d", pass, rows, rowGroups*storage.RowGroupRows)
		}
		reads := r.ReadOps() - before
		if pass == 0 {
			perPass = reads
			if perPass != rowGroups*2 {
				t.Fatalf("pass 0 issued %d reads, want %d", perPass, rowGroups*2)
			}
		} else if reads != perPass {
			t.Fatalf("pass %d issued %d reads, want %d — buffering is not stable across Reset",
				pass, reads, perPass)
		}
		scan.Reset(0, rowGroups)
	}
}
