//go:build duckdb

// DuckDB SOTA baseline benchmarks.
// Build and run with: go test ./bench/tpch/ -tags duckdb -bench=BenchmarkDuckDB -benchtime=3x -v
// Setup first:        go test ./bench/tpch/ -tags duckdb -run TestSetupDuckDB -v
package tpch

import (
	"database/sql"
	"fmt"
	"os"
	"testing"

	_ "github.com/marcboeker/go-duckdb"
)

// ---- DuckDB paths & helpers ------------------------------------------------

func duckDBPath(t testing.TB) string {
	t.Helper()
	p := dataPath(t, "tpch.duckdb")
	// duckDBPath is also used by TestSetupDuckDB before the file exists,
	// so callers that need the file to exist must check themselves.
	return p
}

func openDuckDB(t testing.TB) *sql.DB {
	t.Helper()
	p := duckDBPath(t)
	db, err := sql.Open("duckdb", p)
	if err != nil {
		t.Fatalf("duckdb open: %v", err)
	}
	return db
}

// ---- DuckDB setup ----------------------------------------------------------

// TestSetupDuckDB loads the TPC-H .tbl files into a DuckDB database.
// Run once with: go test ./bench/tpch/ -tags duckdb -run TestSetupDuckDB -v
func TestSetupDuckDB(t *testing.T) {
	dbPath := duckDBPath(t)

	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		t.Fatalf("duckdb create: %v", err)
	}
	defer db.Close()

	// lineitem has a trailing '|' per row (TPC-H dbgen artifact). We load it via
	// read_csv with an explicit column list plus a dummy _extra column that absorbs
	// the empty field after the trailing delimiter; the dummy column is excluded
	// when inserting into the real table.
	type tableSpec struct {
		name       string
		createDDL  string
		loadSQL    string
	}

	tables := []tableSpec{
		{
			name: "lineitem",
			createDDL: `CREATE OR REPLACE TABLE lineitem (
				l_orderkey     INTEGER,
				l_partkey      INTEGER,
				l_suppkey      INTEGER,
				l_linenumber   INTEGER,
				l_quantity     DOUBLE,
				l_extendedprice DOUBLE,
				l_discount     DOUBLE,
				l_tax          DOUBLE,
				l_returnflag   VARCHAR,
				l_linestatus   VARCHAR,
				l_shipdate     DATE,
				l_commitdate   DATE,
				l_receiptdate  DATE,
				l_shipinstruct VARCHAR,
				l_shipmode     VARCHAR,
				l_comment      VARCHAR
			)`,
			// _extra absorbs the trailing '|' that TPC-H dbgen appends to every row.
			// We name it in the columns spec so DuckDB parses the file correctly,
			// then simply omit it from the SELECT list — no EXCLUDE clause needed.
			loadSQL: `INSERT INTO lineitem
				SELECT l_orderkey,l_partkey,l_suppkey,l_linenumber,
				       l_quantity,l_extendedprice,l_discount,l_tax,
				       l_returnflag,l_linestatus,
				       l_shipdate::DATE,l_commitdate::DATE,l_receiptdate::DATE,
				       l_shipinstruct,l_shipmode,l_comment
				FROM read_csv('%s', delim='|', header=false, columns={
					'l_orderkey':'INTEGER','l_partkey':'INTEGER',
					'l_suppkey':'INTEGER','l_linenumber':'INTEGER',
					'l_quantity':'DOUBLE','l_extendedprice':'DOUBLE',
					'l_discount':'DOUBLE','l_tax':'DOUBLE',
					'l_returnflag':'VARCHAR','l_linestatus':'VARCHAR',
					'l_shipdate':'VARCHAR','l_commitdate':'VARCHAR',
					'l_receiptdate':'VARCHAR','l_shipinstruct':'VARCHAR',
					'l_shipmode':'VARCHAR','l_comment':'VARCHAR',
					'_extra':'VARCHAR'
				})`,
		},
		{
			name: "orders",
			createDDL: `CREATE OR REPLACE TABLE orders (
				o_orderkey     INTEGER,
				o_custkey      INTEGER,
				o_orderstatus  VARCHAR,
				o_totalprice   DOUBLE,
				o_orderdate    DATE,
				o_orderpriority VARCHAR,
				o_clerk        VARCHAR,
				o_shippriority INTEGER,
				o_comment      VARCHAR
			)`,
			loadSQL: `INSERT INTO orders
				SELECT o_orderkey,o_custkey,o_orderstatus,o_totalprice,
				       o_orderdate::DATE,o_orderpriority,o_clerk,o_shippriority,o_comment
				FROM read_csv('%s', delim='|', header=false, columns={
					'o_orderkey':'INTEGER','o_custkey':'INTEGER',
					'o_orderstatus':'VARCHAR','o_totalprice':'DOUBLE',
					'o_orderdate':'VARCHAR','o_orderpriority':'VARCHAR',
					'o_clerk':'VARCHAR','o_shippriority':'INTEGER',
					'o_comment':'VARCHAR','_extra':'VARCHAR'
				})`,
		},
		{
			name: "customer",
			createDDL: `CREATE OR REPLACE TABLE customer (
				c_custkey    INTEGER,
				c_name       VARCHAR,
				c_address    VARCHAR,
				c_nationkey  INTEGER,
				c_phone      VARCHAR,
				c_acctbal    DOUBLE,
				c_mktsegment VARCHAR,
				c_comment    VARCHAR
			)`,
			loadSQL: `INSERT INTO customer
				SELECT c_custkey,c_name,c_address,c_nationkey,
				       c_phone,c_acctbal,c_mktsegment,c_comment
				FROM read_csv('%s', delim='|', header=false, columns={
					'c_custkey':'INTEGER','c_name':'VARCHAR',
					'c_address':'VARCHAR','c_nationkey':'INTEGER',
					'c_phone':'VARCHAR','c_acctbal':'DOUBLE',
					'c_mktsegment':'VARCHAR','c_comment':'VARCHAR',
					'_extra':'VARCHAR'
				})`,
		},
	}

	for _, ts := range tables {
		tblPath := dataPath(t, ts.name+".tbl")
		if _, err := os.Stat(tblPath); err != nil {
			t.Logf("skip %s (no .tbl file)", ts.name)
			continue
		}
		t.Logf("loading %s ...", ts.name)

		if _, err := db.Exec(ts.createDDL); err != nil {
			t.Fatalf("create %s: %v", ts.name, err)
		}
		sql := fmt.Sprintf(ts.loadSQL, tblPath)
		if _, err := db.Exec(sql); err != nil {
			t.Fatalf("load %s: %v", ts.name, err)
		}
		var n int64
		db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", ts.name)).Scan(&n)
		t.Logf("  loaded %d rows", n)
	}
	t.Logf("DuckDB database written to %s", dbPath)
}

// ---- DuckDB query runner ---------------------------------------------------

func runDuckDB(t testing.TB, query string) [][]string {
	t.Helper()
	db := openDuckDB(t)
	defer db.Close()

	rows, err := db.Query(query)
	if err != nil {
		t.Fatalf("duckdb query: %v", err)
	}
	defer rows.Close()

	cols, _ := rows.Columns()
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

// ---- DuckDB benchmarks -----------------------------------------------------
//
// These benchmarks give an honest picture of where vexq stands relative to a
// production-grade OLAP engine. DuckDB uses AVX-512 vectorised aggregation,
// radix-partitioned hash joins, and an adaptive morsel scheduler — we expect
// to lose by 3–10×; the goal is to quantify the gap and explain it.

func BenchmarkDuckDBQ1(b *testing.B) {
	db := openDuckDB(b)
	defer db.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := db.Query(q1SQLite) // DuckDB accepts the same SQL
		if err != nil {
			b.Fatalf("duckdb Q1: %v", err)
		}
		drainSQLite(b, rows)
	}
}

func BenchmarkDuckDBQ6(b *testing.B) {
	db := openDuckDB(b)
	defer db.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := db.Query(q6SQLite)
		if err != nil {
			b.Fatalf("duckdb Q6: %v", err)
		}
		drainSQLite(b, rows)
	}
}

func BenchmarkDuckDBQ3(b *testing.B) {
	db := openDuckDB(b)
	defer db.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := db.Query(q3SQLite)
		if err != nil {
			b.Fatalf("duckdb Q3: %v", err)
		}
		drainSQLite(b, rows)
	}
}

func BenchmarkDuckDBQ12(b *testing.B) {
	db := openDuckDB(b)
	defer db.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := db.Query(q12SQLite)
		if err != nil {
			b.Fatalf("duckdb Q12: %v", err)
		}
		drainSQLite(b, rows)
	}
}
