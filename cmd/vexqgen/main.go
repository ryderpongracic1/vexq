// vexqgen converts TPC-H .tbl files (pipe-delimited) to .vxq columnar files.
//
// Usage:
//
//	vexqgen <table> <input.tbl> <output.vxq>
//
// Supported tables: lineitem, orders, customer, part, supplier, partsupp, nation, region
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ryderpongracic1/vexq/storage"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "vexqgen: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) != 3 {
		return fmt.Errorf("usage: vexqgen <table> <input.tbl> <output.vxq>")
	}
	table, inPath, outPath := args[0], args[1], args[2]

	schema, parsers, err := tableSchema(table)
	if err != nil {
		return err
	}

	f, err := os.Open(inPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer f.Close()

	ctx := context.Background()
	w, err := storage.NewWriter(outPath, schema)
	if err != nil {
		return fmt.Errorf("new writer: %w", err)
	}

	const scanBufSize = 1 << 20 // 1 MiB — enough for the widest TPC-H .tbl line

	ncols := len(schema.Fields)

	// Pre-allocate column buffers.
	colBufs := make([]columnBuffer, ncols)
	for i, f := range schema.Fields {
		colBufs[i] = newColumnBuffer(f.Type, storage.RowGroupRows)
	}

	var totalRows int
	flushRowGroup := func(nRows int) error {
		if nRows == 0 {
			return nil
		}
		if err := w.BeginRowGroup(nRows); err != nil {
			return err
		}
		for i := range colBufs {
			nulls, vals := colBufs[i].slice(nRows)
			if err := w.AppendColumn(ctx, i, nulls, vals); err != nil {
				return fmt.Errorf("col %d: %w", i, err)
			}
			colBufs[i].reset()
		}
		return w.EndRowGroup()
	}

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, scanBufSize), scanBufSize)
	rgRow := 0

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		fields := strings.Split(line, "|")
		// TPC-H .tbl lines end with a trailing |, so len(fields) = ncols+1.
		if len(fields) < ncols {
			return fmt.Errorf("line %d: expected >= %d fields, got %d", totalRows+1, ncols, len(fields))
		}
		for i, parse := range parsers {
			if err := parse(fields[i], &colBufs[i], rgRow); err != nil {
				return fmt.Errorf("row %d col %d (%s): %w", totalRows+1, i, schema.Fields[i].Name, err)
			}
		}
		rgRow++
		totalRows++
		if rgRow == storage.RowGroupRows {
			if err := flushRowGroup(rgRow); err != nil {
				_ = w.Abort()
				return fmt.Errorf("flush row group: %w", err)
			}
			rgRow = 0
		}
	}
	if err := scanner.Err(); err != nil {
		_ = w.Abort()
		return fmt.Errorf("scan: %w", err)
	}
	if rgRow > 0 {
		if err := flushRowGroup(rgRow); err != nil {
			_ = w.Abort()
			return fmt.Errorf("flush final row group: %w", err)
		}
	}
	if err := w.Finish(ctx); err != nil {
		return fmt.Errorf("finish: %w", err)
	}
	fmt.Fprintf(os.Stderr, "wrote %d rows → %s\n", totalRows, outPath)
	return nil
}

// ---- column buffer ---------------------------------------------------------

type columnBuffer struct {
	t       storage.DataType
	nulls   []byte
	int64s  []int64
	float64s []float64
	int32s  []int32 // dates
	strings []string
}

func newColumnBuffer(t storage.DataType, cap int) columnBuffer {
	cb := columnBuffer{t: t, nulls: make([]byte, (cap+7)/8)}
	switch t {
	case storage.TypeInt64:
		cb.int64s = make([]int64, 0, cap)
	case storage.TypeFloat64:
		cb.float64s = make([]float64, 0, cap)
	case storage.TypeDate:
		cb.int32s = make([]int32, 0, cap)
	case storage.TypeString:
		cb.strings = make([]string, 0, cap)
	}
	return cb
}

func (cb *columnBuffer) reset() {
	clear(cb.nulls)
	switch cb.t {
	case storage.TypeInt64:
		cb.int64s = cb.int64s[:0]
	case storage.TypeFloat64:
		cb.float64s = cb.float64s[:0]
	case storage.TypeDate:
		cb.int32s = cb.int32s[:0]
	case storage.TypeString:
		cb.strings = cb.strings[:0]
	}
}

func (cb *columnBuffer) slice(n int) ([]byte, interface{}) {
	nullBytes := make([]byte, (n+7)/8)
	copy(nullBytes, cb.nulls)
	switch cb.t {
	case storage.TypeInt64:
		return nullBytes, cb.int64s[:n]
	case storage.TypeFloat64:
		return nullBytes, cb.float64s[:n]
	case storage.TypeDate:
		return nullBytes, cb.int32s[:n]
	case storage.TypeString:
		return nullBytes, cb.strings[:n]
	}
	return nullBytes, nil
}

// ---- field parsers ---------------------------------------------------------

type fieldParser func(raw string, cb *columnBuffer, row int) error

func parseInt64Parser(nullable bool) fieldParser {
	return func(raw string, cb *columnBuffer, row int) error {
		if nullable && raw == "" {
			cb.int64s = append(cb.int64s, 0)
			return nil // null bit stays 0
		}
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return err
		}
		cb.int64s = append(cb.int64s, v)
		cb.nulls[row/8] |= 1 << uint(row%8)
		return nil
	}
}

func parseFloat64Parser(nullable bool) fieldParser {
	return func(raw string, cb *columnBuffer, row int) error {
		if nullable && raw == "" {
			cb.float64s = append(cb.float64s, 0)
			return nil
		}
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return err
		}
		cb.float64s = append(cb.float64s, v)
		cb.nulls[row/8] |= 1 << uint(row%8)
		return nil
	}
}

var epoch = time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)

func parseDateParser(nullable bool) fieldParser {
	return func(raw string, cb *columnBuffer, row int) error {
		if nullable && raw == "" {
			cb.int32s = append(cb.int32s, 0)
			return nil
		}
		t, err := time.ParseInLocation("2006-01-02", raw, time.UTC)
		if err != nil {
			return err
		}
		days := int32(t.Sub(epoch).Hours() / 24)
		cb.int32s = append(cb.int32s, days)
		cb.nulls[row/8] |= 1 << uint(row%8)
		return nil
	}
}

func parseStringParser(nullable bool) fieldParser {
	return func(raw string, cb *columnBuffer, row int) error {
		if nullable && raw == "" {
			cb.strings = append(cb.strings, "")
			return nil
		}
		cb.strings = append(cb.strings, raw)
		cb.nulls[row/8] |= 1 << uint(row%8)
		return nil
	}
}

// ---- table schemas ---------------------------------------------------------

func tableSchema(name string) (storage.Schema, []fieldParser, error) {
	type col struct {
		name     string
		t        storage.DataType
		enc      storage.Encoding
		nullable bool
	}
	cols := func(cs ...col) (storage.Schema, []fieldParser, error) {
		schema := storage.Schema{Fields: make([]storage.Field, len(cs))}
		parsers := make([]fieldParser, len(cs))
		for i, c := range cs {
			schema.Fields[i] = storage.Field{Name: c.name, Type: c.t, Encoding: c.enc, Nullable: c.nullable}
			switch c.t {
			case storage.TypeInt64:
				parsers[i] = parseInt64Parser(c.nullable)
			case storage.TypeFloat64:
				parsers[i] = parseFloat64Parser(c.nullable)
			case storage.TypeDate:
				parsers[i] = parseDateParser(c.nullable)
			case storage.TypeString:
				parsers[i] = parseStringParser(c.nullable)
			}
		}
		return schema, parsers, nil
	}

	str := func(name string) col { return col{name, storage.TypeString, storage.EncDict, false} }
	i64 := func(name string) col { return col{name, storage.TypeInt64, storage.EncPlain, false} }
	f64 := func(name string) col { return col{name, storage.TypeFloat64, storage.EncPlain, false} }
	dat := func(name string) col { return col{name, storage.TypeDate, storage.EncPlain, false} }

	switch name {
	case "lineitem":
		return cols(
			i64("l_orderkey"), i64("l_partkey"), i64("l_suppkey"), i64("l_linenumber"),
			f64("l_quantity"), f64("l_extendedprice"), f64("l_discount"), f64("l_tax"),
			str("l_returnflag"), str("l_linestatus"),
			dat("l_shipdate"), dat("l_commitdate"), dat("l_receiptdate"),
			str("l_shipinstruct"), str("l_shipmode"), str("l_comment"),
		)
	case "orders":
		return cols(
			i64("o_orderkey"), i64("o_custkey"), str("o_orderstatus"),
			f64("o_totalprice"), dat("o_orderdate"), str("o_orderpriority"),
			str("o_clerk"), i64("o_shippriority"), str("o_comment"),
		)
	case "customer":
		return cols(
			i64("c_custkey"), str("c_name"), str("c_address"), i64("c_nationkey"),
			str("c_phone"), f64("c_acctbal"), str("c_mktsegment"), str("c_comment"),
		)
	case "part":
		return cols(
			i64("p_partkey"), str("p_name"), str("p_mfgr"), str("p_brand"),
			str("p_type"), i64("p_size"), str("p_container"),
			f64("p_retailprice"), str("p_comment"),
		)
	case "supplier":
		return cols(
			i64("s_suppkey"), str("s_name"), str("s_address"), i64("s_nationkey"),
			str("s_phone"), f64("s_acctbal"), str("s_comment"),
		)
	case "partsupp":
		return cols(
			i64("ps_partkey"), i64("ps_suppkey"), i64("ps_availqty"),
			f64("ps_supplycost"), str("ps_comment"),
		)
	case "nation":
		return cols(i64("n_nationkey"), str("n_name"), i64("n_regionkey"), str("n_comment"))
	case "region":
		return cols(i64("r_regionkey"), str("r_name"), str("r_comment"))
	}
	return storage.Schema{}, nil, fmt.Errorf("unknown table %q; supported: lineitem orders customer part supplier partsupp nation region", name)
}
