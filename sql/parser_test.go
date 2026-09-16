package sql

import (
	"strings"
	"testing"
)

func mustParse(t *testing.T, query string) *SelectStmt {
	t.Helper()
	p := NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	stmt, ok := node.(*SelectStmt)
	if !ok {
		t.Fatalf("expected *SelectStmt, got %T", node)
	}
	return stmt
}

func TestParseSimpleSelect(t *testing.T) {
	stmt := mustParse(t, "SELECT a, b FROM t")
	if len(stmt.Columns) != 2 {
		t.Fatalf("expected 2 columns, got %d", len(stmt.Columns))
	}
	if stmt.From[0].Name != "t" {
		t.Fatalf("expected FROM t, got %q", stmt.From[0].Name)
	}
}

func TestParseWhere(t *testing.T) {
	stmt := mustParse(t, "SELECT x FROM t WHERE x > 10 AND x < 100")
	if stmt.Where == nil {
		t.Fatal("expected WHERE clause")
	}
	bin, ok := stmt.Where.(*BinaryExpr)
	if !ok || bin.Op != OpAnd {
		t.Fatalf("expected AND expression, got %T", stmt.Where)
	}
}

func TestParseGroupBy(t *testing.T) {
	stmt := mustParse(t, "SELECT a, COUNT(*) FROM t GROUP BY a")
	if len(stmt.GroupBy) != 1 {
		t.Fatalf("expected 1 GROUP BY expr, got %d", len(stmt.GroupBy))
	}
	if len(stmt.Columns) != 2 {
		t.Fatalf("expected 2 SELECT cols, got %d", len(stmt.Columns))
	}
	// Second column should be an aggregate.
	_, ok := stmt.Columns[1].Expr.(*AggFuncExpr)
	if !ok {
		t.Fatalf("expected AggFuncExpr, got %T", stmt.Columns[1].Expr)
	}
}

func TestParseOrderByLimit(t *testing.T) {
	stmt := mustParse(t, "SELECT id FROM t ORDER BY id DESC LIMIT 10")
	if len(stmt.OrderBy) != 1 || !stmt.OrderBy[0].Descending {
		t.Fatalf("expected DESC ORDER BY, got %+v", stmt.OrderBy)
	}
	if stmt.Limit == nil || *stmt.Limit != 10 {
		t.Fatalf("expected LIMIT 10")
	}
}

func TestParseBetween(t *testing.T) {
	stmt := mustParse(t, "SELECT x FROM t WHERE x BETWEEN 1 AND 100")
	be, ok := stmt.Where.(*BetweenExpr)
	if !ok {
		t.Fatalf("expected BetweenExpr, got %T", stmt.Where)
	}
	if be.Not {
		t.Fatal("expected NOT=false")
	}
}

func TestParseIn(t *testing.T) {
	stmt := mustParse(t, "SELECT x FROM t WHERE x IN (1, 2, 3)")
	ie, ok := stmt.Where.(*InExpr)
	if !ok {
		t.Fatalf("expected InExpr, got %T", stmt.Where)
	}
	if len(ie.List) != 3 {
		t.Fatalf("expected 3 IN values, got %d", len(ie.List))
	}
}

func TestParseLike(t *testing.T) {
	stmt := mustParse(t, "SELECT s FROM t WHERE s LIKE '%foo%'")
	le, ok := stmt.Where.(*LikeExpr)
	if !ok {
		t.Fatalf("expected LikeExpr, got %T", stmt.Where)
	}
	sl, ok := le.Pattern.(*StringLiteral)
	if !ok || sl.Value != "%foo%" {
		t.Fatalf("expected pattern '%%foo%%', got %T %v", le.Pattern, le.Pattern)
	}
}

func TestParseIsNull(t *testing.T) {
	stmt := mustParse(t, "SELECT x FROM t WHERE x IS NOT NULL")
	ine, ok := stmt.Where.(*IsNullExpr)
	if !ok || !ine.IsNot {
		t.Fatalf("expected IS NOT NULL, got %T %+v", stmt.Where, stmt.Where)
	}
}

func TestParseCase(t *testing.T) {
	stmt := mustParse(t, `SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END FROM t`)
	col := stmt.Columns[0].Expr
	ce, ok := col.(*CaseExpr)
	if !ok {
		t.Fatalf("expected CaseExpr, got %T", col)
	}
	if len(ce.Whens) != 1 {
		t.Fatalf("expected 1 WHEN, got %d", len(ce.Whens))
	}
}

func TestParseAggFunctions(t *testing.T) {
	for _, q := range []string{
		"SELECT COUNT(*) FROM t",
		"SELECT COUNT(x) FROM t",
		"SELECT SUM(x) FROM t",
		"SELECT AVG(x) FROM t",
		"SELECT MIN(x) FROM t",
		"SELECT MAX(x) FROM t",
	} {
		stmt := mustParse(t, q)
		_, ok := stmt.Columns[0].Expr.(*AggFuncExpr)
		if !ok {
			t.Errorf("query %q: expected AggFuncExpr, got %T", q, stmt.Columns[0].Expr)
		}
	}
}

func TestParseTPCHQ6(t *testing.T) {
	q := `SELECT
	sum(l_extendedprice * l_discount) as revenue
FROM
	lineitem
WHERE
	l_shipdate >= '1994-01-01'
	AND l_shipdate < '1995-01-01'
	AND l_discount BETWEEN 0.05 AND 0.07
	AND l_quantity < 24`
	stmt := mustParse(t, q)
	if stmt.From[0].Name != "lineitem" {
		t.Fatalf("expected FROM lineitem, got %q", stmt.From[0].Name)
	}
	if stmt.Where == nil {
		t.Fatal("expected WHERE clause")
	}
}

func TestParseTPCHQ1(t *testing.T) {
	q := `SELECT
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
FROM
	lineitem
WHERE
	l_shipdate <= '1998-09-02'
GROUP BY
	l_returnflag,
	l_linestatus
ORDER BY
	l_returnflag,
	l_linestatus`
	stmt := mustParse(t, q)
	if len(stmt.GroupBy) != 2 {
		t.Fatalf("expected 2 GROUP BY exprs, got %d", len(stmt.GroupBy))
	}
	if len(stmt.OrderBy) != 2 {
		t.Fatalf("expected 2 ORDER BY items, got %d", len(stmt.OrderBy))
	}
}

func TestParseHaving(t *testing.T) {
	stmt := mustParse(t, "SELECT a, COUNT(*) AS cnt FROM t GROUP BY a HAVING cnt > 5")

	// HAVING should be stored in stmt.Having, not stmt.Where.
	if stmt.Having == nil {
		t.Fatal("expected HAVING clause to be set")
	}
	if stmt.Where != nil {
		t.Fatal("HAVING should not pollute stmt.Where")
	}

	// The HAVING expression should be a BinaryExpr (cnt > 5).
	bin, ok := stmt.Having.(*BinaryExpr)
	if !ok {
		t.Fatalf("expected BinaryExpr for HAVING, got %T", stmt.Having)
	}
	if bin.Op != OpGT {
		t.Fatalf("expected > operator, got %s", bin.Op)
	}

	// Left side should be a column reference "cnt".
	col, ok := bin.Left.(*ColumnRefExpr)
	if !ok {
		t.Fatalf("expected ColumnRefExpr on left, got %T", bin.Left)
	}
	if col.Name != "cnt" {
		t.Fatalf("expected column name 'cnt', got %q", col.Name)
	}

	// Right side should be integer literal 5.
	lit, ok := bin.Right.(*IntLiteral)
	if !ok {
		t.Fatalf("expected IntLiteral on right, got %T", bin.Right)
	}
	if lit.Value != 5 {
		t.Fatalf("expected value 5, got %d", lit.Value)
	}
}

func TestParseHavingWithWhere(t *testing.T) {
	stmt := mustParse(t, "SELECT a, SUM(b) AS total FROM t WHERE a > 0 GROUP BY a HAVING total > 100")

	// Both WHERE and HAVING should be set independently.
	if stmt.Where == nil {
		t.Fatal("expected WHERE clause")
	}
	if stmt.Having == nil {
		t.Fatal("expected HAVING clause")
	}

	// WHERE should be a > 0.
	wBin, ok := stmt.Where.(*BinaryExpr)
	if !ok {
		t.Fatalf("expected BinaryExpr for WHERE, got %T", stmt.Where)
	}
	if wBin.Op != OpGT {
		t.Fatalf("expected > in WHERE, got %s", wBin.Op)
	}

	// HAVING should be total > 100.
	hBin, ok := stmt.Having.(*BinaryExpr)
	if !ok {
		t.Fatalf("expected BinaryExpr for HAVING, got %T", stmt.Having)
	}
	if hBin.Op != OpGT {
		t.Fatalf("expected > in HAVING, got %s", hBin.Op)
	}
	hCol, ok := hBin.Left.(*ColumnRefExpr)
	if !ok {
		t.Fatalf("expected ColumnRefExpr on HAVING left, got %T", hBin.Left)
	}
	if hCol.Name != "total" {
		t.Fatalf("expected 'total' in HAVING, got %q", hCol.Name)
	}
}

func TestParseLimitOffset(t *testing.T) {
	cases := []struct {
		query         string
		limit, offset int64 // -1 = absent
	}{
		{"SELECT a FROM t LIMIT 5 OFFSET 10", 5, 10},
		{"SELECT a FROM t OFFSET 10 LIMIT 5", 5, 10},
		{"SELECT a FROM t OFFSET 3", -1, 3},
		{"SELECT a FROM t LIMIT 0", 0, -1},
	}
	for _, tc := range cases {
		stmt := mustParse(t, tc.query)
		gotLimit, gotOffset := int64(-1), int64(-1)
		if stmt.Limit != nil {
			gotLimit = *stmt.Limit
		}
		if stmt.Offset != nil {
			gotOffset = *stmt.Offset
		}
		if gotLimit != tc.limit || gotOffset != tc.offset {
			t.Errorf("%s: limit=%d offset=%d, want limit=%d offset=%d", tc.query, gotLimit, gotOffset, tc.limit, tc.offset)
		}
	}
}

func TestParseRejectsUnconsumedInput(t *testing.T) {
	cases := []struct {
		query, want string
	}{
		{"SELECT a FROM t UNION SELECT b FROM u", "UNION is not supported"},
		{"SELECT a FROM t INTERSECT SELECT b FROM u", "INTERSECT is not supported"},
		{"SELECT a FROM t EXCEPT SELECT b FROM u", "EXCEPT is not supported"},
		{"SELECT a FROM t LIMIT 1 UNION SELECT b FROM u", "UNION is not supported"},
		{"SELECT a FROM t LIMIT 1 2", "unexpected"},
		{"SELECT a FROM t ORDER BY a banana", "unexpected"},
		{"SELECT a FROM t; SELECT b FROM u", "unexpected"},
		{"SELECT a FROM t LIMIT 1 LIMIT 2", "duplicate LIMIT"},
		{"SELECT a FROM t OFFSET -1", "OFFSET requires a non-negative integer"},
		{"SELECT a FROM t LIMIT 99999999999999999999", "out of range"},
		{"SELECT a FROM t WHERE a = 99999999999999999999", "out of range"},
	}
	for _, tc := range cases {
		_, err := NewParser(tc.query).ParseStatement()
		if err == nil {
			t.Errorf("%s: parsed without error", tc.query)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %q does not contain %q", tc.query, err, tc.want)
		}
	}
	// A single trailing semicolon is accepted.
	mustParse(t, "SELECT a FROM t LIMIT 1;")
	// OFFSET is a keyword but still usable as a column name.
	if stmt := mustParse(t, "SELECT offset FROM t WHERE offset > 1 LIMIT 2 OFFSET 1"); *stmt.Offset != 1 {
		t.Errorf("offset = %d, want 1", *stmt.Offset)
	}
}

func TestParseNegativeNumericLiterals(t *testing.T) {
	stmt := mustParse(t, "SELECT a FROM t WHERE a IN (-2, -2.5, -9223372036854775808) AND b > -1")
	and := stmt.Where.(*BinaryExpr)
	in := and.Left.(*InExpr)
	if v, ok := in.List[0].(*IntLiteral); !ok || v.Value != -2 {
		t.Errorf("IN[0] = %#v, want IntLiteral(-2)", in.List[0])
	}
	if v, ok := in.List[1].(*FloatLiteral); !ok || v.Value != -2.5 {
		t.Errorf("IN[1] = %#v, want FloatLiteral(-2.5)", in.List[1])
	}
	if v, ok := in.List[2].(*IntLiteral); !ok || v.Value != -9223372036854775808 {
		t.Errorf("IN[2] = %#v, want the minimum int64", in.List[2])
	}
	if v, ok := and.Right.(*BinaryExpr).Right.(*IntLiteral); !ok || v.Value != -1 {
		t.Errorf("comparison literal = %#v, want IntLiteral(-1)", and.Right.(*BinaryExpr).Right)
	}
	// Minus on a non-literal stays a unary expression.
	if _, ok := mustParse(t, "SELECT -a FROM t").Columns[0].Expr.(*UnaryExpr); !ok {
		t.Error("-a should parse as UnaryExpr")
	}
	// Binary minus is unaffected.
	if b, ok := mustParse(t, "SELECT a - 1 FROM t").Columns[0].Expr.(*BinaryExpr); !ok || b.Op != OpSub {
		t.Error("a - 1 should parse as subtraction")
	}
}
