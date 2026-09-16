package main

import (
	"context"
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/storage"
)

// writeInt64File writes a one-column INT64 file with a single row group.
func writeInt64File(t *testing.T, values []int64) string {
	t.Helper()
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "t.vxq")
	w, err := storage.NewWriter(path, storage.Schema{Fields: []storage.Field{{Name: "v", Type: storage.TypeInt64}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.BeginRowGroup(len(values)); err != nil {
		t.Fatal(err)
	}
	if err := w.AppendColumn(ctx, 0, nil, values); err != nil {
		t.Fatal(err)
	}
	if err := w.EndRowGroup(); err != nil {
		t.Fatal(err)
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatal(err)
	}
	return path
}

// setFirstColumnMax overwrites the zone-map max of row group 0, column 0 in the
// file's footer and re-seals the footer CRC, producing a file whose checksums
// are all valid but whose zone map lies about the data.
func setFirstColumnMax(t *testing.T, path string, max int64) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const trailerSize = 16 // [4B CRC][8B body length][4B magic]
	trailer := data[len(data)-trailerSize:]
	bodyLen := int(binary.LittleEndian.Uint64(trailer[4:12]))
	body := data[len(data)-trailerSize-bodyLen : len(data)-trailerSize]

	// Schema directory: [4B count] then per field [2B name len][name][type][enc][flags].
	off := 4
	nameLen := int(binary.LittleEndian.Uint16(body[off:]))
	off += 2 + nameLen + 3
	// Row group directory: [4B count], then [8B offset][4B rows], then the first
	// column: [8B section off][8B section len][8B nulls][8B sum][8B min][8B max].
	off += 4 + 8 + 4
	off += 8 + 8 + 8 + 8 + 8
	binary.LittleEndian.PutUint64(body[off:], uint64(max))
	binary.LittleEndian.PutUint32(trailer[0:4], crc32.ChecksumIEEE(body))

	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFsckAcceptsConsistentFile(t *testing.T) {
	path := writeInt64File(t, []int64{-7, 3, 42, 9})
	if err := runFsck(path); err != nil {
		t.Fatalf("fsck on a freshly written file: %v", err)
	}
}

func TestFsckRejectsZoneMapThatDisagreesWithData(t *testing.T) {
	path := writeInt64File(t, []int64{-7, 3, 42, 9})
	setFirstColumnMax(t, path, 10) // the column really holds 42

	// The file still opens: every checksum is valid.
	r, err := storage.Open(context.Background(), path)
	if err != nil {
		t.Fatalf("tampered file should still open: %v", err)
	}
	if got := int64(r.Meta().RowGroups[0].Columns[0].Stats.Max); got != 10 {
		t.Fatalf("footer max = %d, want the tampered 10", got)
	}
	_ = r.Close()

	err = runFsck(path)
	if err == nil {
		t.Fatal("fsck passed a file whose zone map max (10) excludes a stored value (42)")
	}
	if !strings.Contains(err.Error(), "integrity check failed") {
		t.Fatalf("unexpected error: %v", err)
	}
}
