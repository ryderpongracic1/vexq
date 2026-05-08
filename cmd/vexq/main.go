// vexq — columnar SQL query engine CLI.
//
// Usage:
//
//	vexq <file.vxq> "SELECT ..."   – execute a SQL query against a .vxq file
//	vexq fsck <file.vxq>           – validate file integrity (CRC, footer, zone maps)
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "vexq: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vexq <file.vxq> \"SELECT ...\" | vexq fsck <file.vxq>")
	}
	if args[0] == "fsck" {
		if len(args) < 2 {
			return fmt.Errorf("fsck requires a .vxq file argument")
		}
		return runFsck(args[1])
	}
	if len(args) < 2 {
		return fmt.Errorf("usage: vexq <file.vxq> \"SELECT ...\"")
	}
	return runQuery(args[0], args[1])
}

// ---- query -----------------------------------------------------------------

func runQuery(path, query string) error {
	ctx := context.Background()

	// Derive table name from filename (strip extension and directory).
	base := filepath.Base(path)
	tableName := strings.TrimSuffix(base, filepath.Ext(base))

	cat, err := catalog.OpenSingle(ctx, tableName, path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}

	p := sql.NewParser(query)
	stmt, err := p.ParseStatement()
	if err != nil {
		return fmt.Errorf("parse: %w", err)
	}
	sel, ok := stmt.(*sql.SelectStmt)
	if !ok {
		return fmt.Errorf("only SELECT statements are supported")
	}

	logical, err := planner.Build(ctx, sel, cat)
	if err != nil {
		return fmt.Errorf("plan: %w", err)
	}
	logical = planner.Optimize(logical)

	op, err := planner.Physical(ctx, logical)
	if err != nil {
		return fmt.Errorf("physical: %w", err)
	}
	defer op.Close()

	schema := op.Schema()

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	// Header
	var headers []string
	for _, f := range schema.Fields {
		headers = append(headers, f.Name)
	}
	fmt.Fprintln(tw, strings.Join(headers, "\t"))

	// Separator
	var seps []string
	for _, f := range schema.Fields {
		n := len(f.Name)
		if n < 4 {
			n = 4
		}
		seps = append(seps, strings.Repeat("-", n))
	}
	fmt.Fprintln(tw, strings.Join(seps, "\t"))

	// Rows
	rowCount := 0
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			return fmt.Errorf("exec: %w", err)
		}
		if batch == nil {
			break
		}
		rowCount += batch.Length
		printBatch(tw, batch)
	}

	tw.Flush()
	fmt.Fprintf(os.Stderr, "(%d rows)\n", rowCount)
	return nil
}

func printBatch(w io.Writer, batch *exec.Batch) {
	rows := batch.Length
	vals := make([]string, len(batch.Vectors))
	if batch.SelVec != nil {
		for _, ri := range batch.SelVec {
			for j, vec := range batch.Vectors {
				vals[j] = vecVal(vec, int(ri))
			}
			fmt.Fprintln(w, strings.Join(vals, "\t"))
		}
		return
	}
	for i := 0; i < rows; i++ {
		for j, vec := range batch.Vectors {
			vals[j] = vecVal(vec, i)
		}
		fmt.Fprintln(w, strings.Join(vals, "\t"))
	}
}

func vecVal(vec exec.Vector, i int) string {
	if vec.IsNull(i) {
		return "NULL"
	}
	switch v := vec.(type) {
	case *exec.Int64Vector:
		return fmt.Sprintf("%d", v.Values[i])
	case *exec.Float64Vector:
		return fmt.Sprintf("%g", v.Values[i])
	case *exec.BoolVector:
		byteIdx, bitIdx := i/8, uint(i%8)
		if byteIdx < len(v.Bits) && (v.Bits[byteIdx]>>bitIdx)&1 == 1 {
			return "true"
		}
		return "false"
	case *exec.StringVector:
		if v.Dict != nil && int(v.Codes[i]) < len(v.Dict.Offsets) {
			return v.Dict.Get(v.Codes[i])
		}
		return fmt.Sprintf("code:%d", v.Codes[i])
	case *exec.DateVector:
		d := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, int(v.Values[i]))
		return d.Format("2006-01-02")
	}
	return "?"
}

// ---- fsck ------------------------------------------------------------------

func runFsck(path string) error {
	ctx := context.Background()

	r, err := storage.Open(ctx, path)
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	defer r.Close()

	meta := r.Meta()
	schema := meta.Schema

	fmt.Printf("File: %s\n", path)
	fmt.Printf("Row groups: %d\n", len(meta.RowGroups))
	fmt.Println()

	fmt.Println("Schema:")
	for i, f := range schema.Fields {
		enc := encName(f.Encoding)
		nullable := ""
		if f.Nullable {
			nullable = " NULLABLE"
		}
		fmt.Printf("  [%d] %s %s(%s)%s\n", i, f.Name, typeName(f.Type), enc, nullable)
	}
	fmt.Println()

	var totalErrors int
	var totalBytes int64

	for rg, rgMeta := range meta.RowGroups {
		fmt.Printf("Row group %d: %d rows @ offset %d\n", rg, rgMeta.NumRows, rgMeta.FileOffset)
		for col, colMeta := range rgMeta.Columns {
			f := schema.Fields[col]
			zm := colMeta.Stats
			fmt.Printf("  col %-20s  bytes=%-8d  nulls=%d", f.Name, colMeta.SectionLength, zm.NullCount)
			if zm.HasMinMax {
				fmt.Printf("  min=%s  max=%s", fmtZoneVal(zm.Min, f.Type), fmtZoneVal(zm.Max, f.Type))
			}
			fmt.Println()
			totalBytes += colMeta.SectionLength
		}

		// Validate blocks by reading them.
		colErrors := 0
		for col := range schema.Fields {
			cr, err := r.OpenColumn(ctx, rg, col)
			if err != nil {
				fmt.Printf("  ERROR: open column %d: %v\n", col, err)
				colErrors++
				continue
			}
			blockNum := 0
			for {
				_, _, _, err := cr.NextBlock(ctx)
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					fmt.Printf("  ERROR: row group %d col %d block %d: %v\n", rg, col, blockNum, err)
					colErrors++
				}
				blockNum++
			}
			cr.Close()
		}
		totalErrors += colErrors
	}

	fmt.Println()
	fmt.Printf("Total column bytes: %d\n", totalBytes)
	if totalErrors == 0 {
		fmt.Println("fsck: OK")
	} else {
		fmt.Printf("fsck: FAILED (%d errors)\n", totalErrors)
		return fmt.Errorf("integrity check failed")
	}
	return nil
}

func typeName(t storage.DataType) string {
	switch t {
	case storage.TypeInt64:
		return "INT64"
	case storage.TypeFloat64:
		return "FLOAT64"
	case storage.TypeBool:
		return "BOOL"
	case storage.TypeString:
		return "STRING"
	case storage.TypeDate:
		return "DATE"
	}
	return fmt.Sprintf("TYPE(%d)", t)
}

func encName(e storage.Encoding) string {
	switch e {
	case storage.EncPlain:
		return "plain"
	case storage.EncRLE:
		return "rle"
	case storage.EncDict:
		return "dict"
	}
	return "?"
}

func fmtZoneVal(raw uint64, t storage.DataType) string {
	switch t {
	case storage.TypeInt64:
		return fmt.Sprintf("%d", int64(raw))
	case storage.TypeFloat64:
		return fmt.Sprintf("%g", math.Float64frombits(raw))
	case storage.TypeDate:
		d := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, int(int32(raw)))
		return d.Format("2006-01-02")
	case storage.TypeString:
		return fmt.Sprintf("code:%d", raw)
	}
	return fmt.Sprintf("%d", raw)
}
