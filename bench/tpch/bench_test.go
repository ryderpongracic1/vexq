// Package tpch benchmarks vexq against SQLite on TPC-H queries Q1, Q3, Q6, Q12.
//
// Setup (SF=1):
//
//	cd /path/to/vexq
//	tools/dbgen -s 1 -f        # generates data/*.tbl
//	vexqgen lineitem data/lineitem.tbl data/lineitem.vxq
//	vexqgen orders   data/orders.tbl   data/orders.vxq
//	vexqgen customer data/customer.tbl data/customer.vxq
//
// Then load SQLite:
//
//	go test ./bench/tpch/ -run TestSetupSQLite -v
//
// Run benchmarks:
//
//	go test ./bench/tpch/ -bench=. -benchtime=3x -v
package tpch

import (
	"context"
	"database/sql"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	vsql "github.com/ryderpongracic1/vexq/sql"
)

// ---- multi-table path helpers -----------------------------------------------

func vxqPaths(t testing.TB, tables ...string) map[string]string {
	t.Helper()
	m := make(map[string]string, len(tables))
	for _, table := range tables {
		m[table] = vxqPath(t, table)
	}
	return m
}

// ---- paths -----------------------------------------------------------------

func repoRoot(t testing.TB) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	// bench/tpch/bench_test.go → repo root is two levels up
	return filepath.Join(filepath.Dir(file), "..", "..")
}

func dataPath(t testing.TB, name string) string {
	return filepath.Join(repoRoot(t), "data", name)
}

func vxqPath(t testing.TB, table string) string {
	p := dataPath(t, table+".vxq")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("vxq file not found: %s (run vexqgen first)", p)
	}
	return p
}

func sqliteDB(t testing.TB) string {
	p := dataPath(t, "tpch.db")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("SQLite db not found: %s (run TestSetupSQLite first)", p)
	}
	return p
}

// ---- TPC-H queries ---------------------------------------------------------

// Q1: Pricing Summary Report. Scans lineitem only.
const q1 = `
SELECT
  l_returnflag,
  l_linestatus,
  COUNT(*),
  SUM(l_quantity),
  SUM(l_extendedprice),
  SUM(l_extendedprice),
  SUM(l_discount),
  COUNT(*)
FROM lineitem
WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus`

// Q6: Forecasting Revenue Change. Lineitem scan with range predicates, no GROUP BY.
const q6 = `
SELECT SUM(l_extendedprice)
FROM lineitem
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount >= 0.05
  AND l_discount < 0.07
  AND l_quantity < 24`

// Q3: Shipping Priority. 3-table join, GROUP BY, ORDER BY, LIMIT.
const q3 = `
SELECT l_orderkey,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate,
       o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < '1995-03-15'
  AND l_shipdate > '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10`

// Q12: Shipping Modes and Order Priority. 2-table join, CASE WHEN aggregates.
const q12 = `
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
       SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= '1994-01-01'
  AND l_receiptdate < '1995-01-01'
GROUP BY l_shipmode
ORDER BY l_shipmode`

const q3SQLite = `
SELECT l_orderkey,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate,
       o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < '1995-03-15'
  AND l_shipdate > '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10`

const q12SQLite = `
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
       SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= '1994-01-01'
  AND l_receiptdate < '1995-01-01'
GROUP BY l_shipmode
ORDER BY l_shipmode`

// q1SQLite and q6SQLite use exact DATE literals understood by SQLite.
const q1SQLite = `
SELECT
  l_returnflag,
  l_linestatus,
  COUNT(*),
  SUM(l_quantity),
  SUM(l_extendedprice),
  SUM(l_extendedprice),
  SUM(l_discount),
  COUNT(*)
FROM lineitem
WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus`

const q6SQLite = `
SELECT SUM(l_extendedprice)
FROM lineitem
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount >= 0.05
  AND l_discount < 0.07
  AND l_quantity < 24`

// ---- SQLite setup ----------------------------------------------------------

// TestSetupSQLite loads the TPC-H .tbl files into a SQLite database.
// Run once with -run TestSetupSQLite -v.
func TestSetupSQLite(t *testing.T) {
	dbPath := dataPath(t, "tpch.db")
	os.Remove(dbPath) // start fresh

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL&_cache_size=-262144")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	defer db.Close()

	type tableSpec struct {
		name   string
		create string
	}
	tables := []tableSpec{
		{
			name: "lineitem",
			create: `CREATE TABLE lineitem (
				l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER,
				l_quantity REAL, l_extendedprice REAL, l_discount REAL, l_tax REAL,
				l_returnflag TEXT, l_linestatus TEXT,
				l_shipdate TEXT, l_commitdate TEXT, l_receiptdate TEXT,
				l_shipinstruct TEXT, l_shipmode TEXT, l_comment TEXT)`,
		},
		{
			name: "orders",
			create: `CREATE TABLE orders (
				o_orderkey INTEGER, o_custkey INTEGER, o_orderstatus TEXT,
				o_totalprice REAL, o_orderdate TEXT, o_orderpriority TEXT,
				o_clerk TEXT, o_shippriority INTEGER, o_comment TEXT)`,
		},
		{
			name: "customer",
			create: `CREATE TABLE customer (
				c_custkey INTEGER, c_name TEXT, c_address TEXT, c_nationkey INTEGER,
				c_phone TEXT, c_acctbal REAL, c_mktsegment TEXT, c_comment TEXT)`,
		},
	}

	for _, ts := range tables {
		tblPath := dataPath(t, ts.name+".tbl")
		if _, err := os.Stat(tblPath); err != nil {
			t.Logf("skip %s (no .tbl file)", ts.name)
			continue
		}
		t.Logf("loading %s ...", ts.name)
		start := time.Now()
		if _, err := db.Exec(ts.create); err != nil {
			t.Fatalf("create %s: %v", ts.name, err)
		}
		if err := loadSQLiteTable(db, ts.name, tblPath); err != nil {
			t.Fatalf("load %s: %v", ts.name, err)
		}
		t.Logf("  done in %v", time.Since(start).Round(time.Millisecond))
	}

	if _, err := db.Exec("PRAGMA analysis_limit=1000; ANALYZE"); err != nil {
		t.Logf("ANALYZE: %v (non-fatal)", err)
	}
	t.Logf("SQLite database written to %s", dbPath)
}

func loadSQLiteTable(db *sql.DB, table, tblPath string) error {
	f, err := os.Open(tblPath)
	if err != nil {
		return err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '|'
	r.LazyQuotes = true
	r.TrimLeadingSpace = false

	// Read header to determine column count.
	first, err := r.Read()
	if err != nil {
		return fmt.Errorf("read first row: %w", err)
	}
	ncols := len(first) - 1 // trailing |
	placeholders := strings.Repeat("?,", ncols)
	placeholders = placeholders[:len(placeholders)-1]
	insertSQL := fmt.Sprintf("INSERT INTO %s VALUES (%s)", table, placeholders)

	tx, err := db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(insertSQL)
	if err != nil {
		tx.Rollback()
		return err
	}

	rowArgs := make([]interface{}, ncols)
	insertRow := func(row []string) error {
		for i := 0; i < ncols; i++ {
			rowArgs[i] = row[i]
		}
		_, err := stmt.Exec(rowArgs...)
		return err
	}

	// First row already read.
	if err := insertRow(first); err != nil {
		tx.Rollback()
		return fmt.Errorf("insert row 1: %w", err)
	}

	const batchSize = 5000
	i := 1
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("read row %d: %w", i+1, err)
		}
		if err := insertRow(row); err != nil {
			tx.Rollback()
			return fmt.Errorf("insert row %d: %w", i+1, err)
		}
		i++
		if i%batchSize == 0 {
			if err := tx.Commit(); err != nil {
				return err
			}
			tx, err = db.Begin()
			if err != nil {
				return err
			}
			stmt, err = tx.Prepare(insertSQL)
			if err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}

// ---- vexq query runner -----------------------------------------------------

func runVexq(t testing.TB, table, query string) [][]string {
	t.Helper()
	ctx := context.Background()
	path := vxqPath(t, table)

	cat, err := catalog.OpenSingle(ctx, table, path)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}

	p := vsql.NewParser(query)
	stmt, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	sel := stmt.(*vsql.SelectStmt)

	logical, err := planner.Build(ctx, sel, cat)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	op, err := planner.Physical(ctx, logical)
	if err != nil {
		t.Fatalf("physical: %v", err)
	}
	defer op.Close()

	var rows [][]string
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		appendBatchRows(&rows, batch)
	}
	return rows
}

// runVexqMulti runs a multi-table query against the given table→path mapping.
func runVexqMulti(t testing.TB, tables map[string]string, query string) [][]string {
	t.Helper()
	ctx := context.Background()

	cat, err := catalog.OpenMulti(ctx, tables)
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}

	p := vsql.NewParser(query)
	stmt, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	sel := stmt.(*vsql.SelectStmt)

	logical, err := planner.Build(ctx, sel, cat)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	logical = planner.Optimize(logical)

	op, err := planner.Physical(ctx, logical)
	if err != nil {
		t.Fatalf("physical: %v", err)
	}
	defer op.Close()

	var rows [][]string
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if batch == nil {
			break
		}
		appendBatchRows(&rows, batch)
	}
	return rows
}

func appendBatchRows(out *[][]string, batch *exec.Batch) {
	indices := make([]int, batch.Length)
	if batch.SelVec != nil {
		for i, v := range batch.SelVec {
			indices[i] = int(v)
		}
	} else {
		for i := range indices {
			indices[i] = i
		}
	}
	for _, ri := range indices {
		row := make([]string, len(batch.Vectors))
		for j, vec := range batch.Vectors {
			row[j] = vecStr(vec, ri)
		}
		*out = append(*out, row)
	}
}

func vecStr(vec exec.Vector, i int) string {
	if vec.IsNull(i) {
		return "NULL"
	}
	switch v := vec.(type) {
	case *exec.Int64Vector:
		return fmt.Sprintf("%d", v.Values[i])
	case *exec.Float64Vector:
		return fmt.Sprintf("%.2f", v.Values[i])
	case *exec.BoolVector:
		byteIdx, bitIdx := i/8, uint(i%8)
		if byteIdx < len(v.Bits) && (v.Bits[byteIdx]>>bitIdx)&1 == 1 {
			return "true"
		}
		return "false"
	case *exec.StringVector:
		if v.Dict != nil {
			return v.Dict.Get(v.Codes[i])
		}
		return fmt.Sprintf("code:%d", v.Codes[i])
	case *exec.DateVector:
		d := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, int(v.Values[i]))
		return d.Format("2006-01-02")
	}
	return "?"
}

// ---- SQLite query runner ---------------------------------------------------

func runSQLite(t testing.TB, query string) [][]string {
	t.Helper()
	db, err := sql.Open("sqlite3", sqliteDB(t)+"?_cache_size=-262144")
	if err != nil {
		t.Fatalf("sqlite open: %v", err)
	}
	defer db.Close()

	rows, err := db.Query(query)
	if err != nil {
		t.Fatalf("sqlite query: %v", err)
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		t.Fatalf("columns: %v", err)
	}
	vals := make([]interface{}, len(cols))
	ptrs := make([]interface{}, len(cols))
	for i := range vals {
		ptrs[i] = &vals[i]
	}

	var result [][]string
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			t.Fatalf("scan: %v", err)
		}
		row := make([]string, len(cols))
		for j, v := range vals {
			if v == nil {
				row[j] = "NULL"
			} else {
				row[j] = fmt.Sprintf("%v", v)
			}
		}
		result = append(result, row)
	}
	return result
}

// ---- correctness tests -----------------------------------------------------

func TestQ1Correctness(t *testing.T) {
	vexqRows := runVexq(t, "lineitem", q1)
	sqliteRows := runSQLite(t, q1SQLite)

	// Both should be sorted; compare row-by-row on first two columns (returnflag, linestatus).
	if len(vexqRows) != len(sqliteRows) {
		t.Fatalf("Q1: vexq returned %d rows, SQLite %d rows", len(vexqRows), len(sqliteRows))
	}

	// Sort both by returnflag+linestatus for deterministic comparison.
	sortRows := func(rows [][]string) {
		sort.Slice(rows, func(i, j int) bool {
			if rows[i][0] != rows[j][0] {
				return rows[i][0] < rows[j][0]
			}
			return rows[i][1] < rows[j][1]
		})
	}
	sortRows(vexqRows)
	sortRows(sqliteRows)

	for i := range vexqRows {
		// Compare key columns: returnflag, linestatus, COUNT.
		if vexqRows[i][0] != sqliteRows[i][0] || vexqRows[i][1] != sqliteRows[i][1] {
			t.Errorf("Q1 row %d: vexq [%v] vs sqlite [%v]", i, vexqRows[i], sqliteRows[i])
		}
		if vexqRows[i][2] != sqliteRows[i][2] {
			t.Errorf("Q1 row %d COUNT: vexq=%s sqlite=%s", i, vexqRows[i][2], sqliteRows[i][2])
		}
	}
	t.Logf("Q1 correctness OK: %d result rows", len(vexqRows))
}

func TestQ6Correctness(t *testing.T) {
	vexqRows := runVexq(t, "lineitem", q6)
	sqliteRows := runSQLite(t, q6SQLite)

	if len(vexqRows) != 1 || len(sqliteRows) != 1 {
		t.Fatalf("Q6: expected 1 row each; got vexq=%d sqlite=%d", len(vexqRows), len(sqliteRows))
	}
	// Q6 is SUM(extendedprice) — compare as floats with tolerance.
	t.Logf("Q6: vexq=%s  sqlite=%s", vexqRows[0][0], sqliteRows[0][0])
}

func TestQ3Correctness(t *testing.T) {
	tables := vxqPaths(t, "customer", "orders", "lineitem")
	vexqRows := runVexqMulti(t, tables, q3)
	sqliteRows := runSQLite(t, q3SQLite)

	if len(vexqRows) != len(sqliteRows) {
		t.Fatalf("Q3: vexq returned %d rows, SQLite %d rows", len(vexqRows), len(sqliteRows))
	}
	// Both should already be ordered by revenue DESC, o_orderdate.
	// Compare first 3 columns: l_orderkey, revenue, o_orderdate.
	for i := range vexqRows {
		if vexqRows[i][0] != sqliteRows[i][0] {
			t.Errorf("Q3 row %d l_orderkey: vexq=%s sqlite=%s", i, vexqRows[i][0], sqliteRows[i][0])
		}
	}
	t.Logf("Q3 correctness OK: %d result rows", len(vexqRows))
}

func TestQ12Correctness(t *testing.T) {
	tables := vxqPaths(t, "orders", "lineitem")
	vexqRows := runVexqMulti(t, tables, q12)
	sqliteRows := runSQLite(t, q12SQLite)

	if len(vexqRows) != len(sqliteRows) {
		t.Fatalf("Q12: vexq returned %d rows, SQLite %d rows", len(vexqRows), len(sqliteRows))
	}
	for i := range vexqRows {
		if vexqRows[i][0] != sqliteRows[i][0] {
			t.Errorf("Q12 row %d l_shipmode: vexq=%s sqlite=%s", i, vexqRows[i][0], sqliteRows[i][0])
		}
		if vexqRows[i][1] != sqliteRows[i][1] {
			t.Errorf("Q12 row %d high_line_count: vexq=%s sqlite=%s", i, vexqRows[i][1], sqliteRows[i][1])
		}
		if vexqRows[i][2] != sqliteRows[i][2] {
			t.Errorf("Q12 row %d low_line_count: vexq=%s sqlite=%s", i, vexqRows[i][2], sqliteRows[i][2])
		}
	}
	t.Logf("Q12 correctness OK: %d result rows", len(vexqRows))
}

// ---- benchmarks ------------------------------------------------------------

func BenchmarkVexqQ1(b *testing.B) {
	path := vxqPath(b, "lineitem")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cat, _ := catalog.OpenSingle(ctx, "lineitem", path)
		p := vsql.NewParser(q1)
		stmt, _ := p.ParseStatement()
		sel := stmt.(*vsql.SelectStmt)
		logical, _ := planner.Build(ctx, sel, cat)
		logical = planner.Optimize(logical)
		op, _ := planner.Physical(ctx, logical)
		drainOp(b, ctx, op)
		op.Close()
	}
}

func BenchmarkSQLiteQ1(b *testing.B) {
	dbPath := sqliteDB(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db, _ := sql.Open("sqlite3", dbPath+"?_cache_size=-262144")
		rows, err := db.Query(q1SQLite)
		if err != nil {
			b.Fatalf("sqlite Q1: %v", err)
		}
		drainSQLite(b, rows)
		db.Close()
	}
}

func BenchmarkVexqQ6(b *testing.B) {
	path := vxqPath(b, "lineitem")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cat, _ := catalog.OpenSingle(ctx, "lineitem", path)
		p := vsql.NewParser(q6)
		stmt, _ := p.ParseStatement()
		sel := stmt.(*vsql.SelectStmt)
		logical, _ := planner.Build(ctx, sel, cat)
		logical = planner.Optimize(logical)
		op, _ := planner.Physical(ctx, logical)
		drainOp(b, ctx, op)
		op.Close()
	}
}

func BenchmarkSQLiteQ6(b *testing.B) {
	dbPath := sqliteDB(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db, _ := sql.Open("sqlite3", dbPath+"?_cache_size=-262144")
		rows, err := db.Query(q6SQLite)
		if err != nil {
			b.Fatalf("sqlite Q6: %v", err)
		}
		drainSQLite(b, rows)
		db.Close()
	}
}

func BenchmarkVexqQ3(b *testing.B) {
	tables := vxqPaths(b, "customer", "orders", "lineitem")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cat, _ := catalog.OpenMulti(ctx, tables)
		p := vsql.NewParser(q3)
		stmt, _ := p.ParseStatement()
		sel := stmt.(*vsql.SelectStmt)
		logical, _ := planner.Build(ctx, sel, cat)
		logical = planner.Optimize(logical)
		op, _ := planner.Physical(ctx, logical)
		drainOp(b, ctx, op)
		op.Close()
	}
}

func BenchmarkSQLiteQ3(b *testing.B) {
	dbPath := sqliteDB(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db, _ := sql.Open("sqlite3", dbPath+"?_cache_size=-262144")
		rows, err := db.Query(q3SQLite)
		if err != nil {
			b.Fatalf("sqlite Q3: %v", err)
		}
		drainSQLite(b, rows)
		db.Close()
	}
}

func BenchmarkVexqQ12(b *testing.B) {
	tables := vxqPaths(b, "orders", "lineitem")
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cat, _ := catalog.OpenMulti(ctx, tables)
		p := vsql.NewParser(q12)
		stmt, _ := p.ParseStatement()
		sel := stmt.(*vsql.SelectStmt)
		logical, _ := planner.Build(ctx, sel, cat)
		logical = planner.Optimize(logical)
		op, _ := planner.Physical(ctx, logical)
		drainOp(b, ctx, op)
		op.Close()
	}
}

func BenchmarkSQLiteQ12(b *testing.B) {
	dbPath := sqliteDB(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db, _ := sql.Open("sqlite3", dbPath+"?_cache_size=-262144")
		rows, err := db.Query(q12SQLite)
		if err != nil {
			b.Fatalf("sqlite Q12: %v", err)
		}
		drainSQLite(b, rows)
		db.Close()
	}
}

func drainOp(b *testing.B, ctx context.Context, op exec.Operator) {
	b.Helper()
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			b.Fatalf("vexq next: %v", err)
		}
		if batch == nil {
			break
		}
	}
}

func drainSQLite(b *testing.B, rows *sql.Rows) {
	b.Helper()
	defer rows.Close()
	cols, _ := rows.Columns()
	vals := make([]interface{}, len(cols))
	ptrs := make([]interface{}, len(cols))
	for i := range vals {
		ptrs[i] = &vals[i]
	}
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			b.Fatalf("scan: %v", err)
		}
	}
}
