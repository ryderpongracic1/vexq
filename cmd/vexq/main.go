// vexq — columnar SQL query engine CLI.
//
// Usage:
//
//	vexq [--workers=N] <file.vxq> [file2.vxq ...] "SELECT ..."  – execute a SQL query
//	vexq fsck <file.vxq>                                         – validate file integrity
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
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
		return fmt.Errorf("usage: vexq [--workers=N] <file.vxq> [file2.vxq ...] \"SELECT ...\" | vexq fsck <file.vxq>")
	}

	// Parse optional flags before positional args.
	workers := 0 // 0 = serial (default)
	var positional []string
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == "--workers" && i+1 < len(args):
			n, err := strconv.Atoi(args[i+1])
			if err != nil {
				return fmt.Errorf("invalid --workers value: %s", args[i+1])
			}
			workers = n
			i++ // skip next arg
		case strings.HasPrefix(args[i], "--workers="):
			n, err := strconv.Atoi(strings.TrimPrefix(args[i], "--workers="))
			if err != nil {
				return fmt.Errorf("invalid --workers value: %s", args[i])
			}
			workers = n
		case args[i] == "-w" && i+1 < len(args):
			n, err := strconv.Atoi(args[i+1])
			if err != nil {
				return fmt.Errorf("invalid -w value: %s", args[i+1])
			}
			workers = n
			i++ // skip next arg
		default:
			positional = append(positional, args[i])
		}
	}

	if len(positional) == 0 {
		return fmt.Errorf("usage: vexq [--workers=N] <file.vxq> [file2.vxq ...] \"SELECT ...\" | vexq fsck <file.vxq>")
	}

	// Handle fsck subcommand.
	if positional[0] == "fsck" {
		if len(positional) < 2 {
			return fmt.Errorf("fsck requires a .vxq file argument")
		}
		return runFsck(positional[1])
	}

	// Need at least one file and a query string.
	if len(positional) < 2 {
		return fmt.Errorf("usage: vexq [--workers=N] <file.vxq> [file2.vxq ...] \"SELECT ...\"")
	}

	// Last arg is the query, everything before is a file path.
	query := positional[len(positional)-1]
	files := positional[:len(positional)-1]

	return runQuery(files, query, workers)
}

// ---- query -----------------------------------------------------------------

func runQuery(paths []string, query string, workers int) error {
	ctx := context.Background()

	// Build table name → file path mapping from positional file args.
	tables := make(map[string]string, len(paths))
	for _, p := range paths {
		base := filepath.Base(p)
		tableName := strings.TrimSuffix(base, filepath.Ext(base))
		if existing, dup := tables[tableName]; dup {
			return fmt.Errorf("duplicate table name %q: %s and %s", tableName, existing, p)
		}
		tables[tableName] = p
	}

	var cat *catalog.Catalog
	var err error
	if len(tables) == 1 {
		// Single table: use OpenSingle for backward compatibility.
		for name, p := range tables {
			cat, err = catalog.OpenSingle(ctx, name, p)
		}
	} else {
		cat, err = catalog.OpenMulti(ctx, tables)
	}
	if err != nil {
		return fmt.Errorf("open catalog: %w", err)
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

	var op exec.Operator
	if workers > 0 {
		op, err = planner.Parallel(ctx, logical, workers)
	} else {
		op, err = planner.Physical(ctx, logical)
	}
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

			f := schema.Fields[col]

			// Validate dictionary for string columns (once per row group).
			if f.Type == storage.TypeString {
				if _, dictErr := cr.Dictionary(); dictErr != nil {
					fmt.Printf("  ERROR: row group %d col %d: dictionary: %v\n", rg, col, dictErr)
					colErrors++
				}
			}

			blockNum := 0
			for {
				_, payload, _, err := cr.NextBlock(ctx)
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					fmt.Printf("  ERROR: row group %d col %d block %d: %v\n", rg, col, blockNum, err)
					colErrors++
					blockNum++
					continue
				}

				// Validate bool RLE decoding.
				if f.Type == storage.TypeBool {
					if _, _, _, decErr := storage.DecodeRLEBool(payload); decErr != nil {
						fmt.Printf("  ERROR: row group %d col %d block %d: bool decode: %v\n", rg, col, blockNum, decErr)
						colErrors++
					}
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
