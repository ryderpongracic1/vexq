package sql

import (
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
	if stmt.From.Name != "t" {
		t.Fatalf("expected FROM t, got %q", stmt.From.Name)
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
	if stmt.From.Name != "lineitem" {
		t.Fatalf("expected FROM lineitem, got %q", stmt.From.Name)
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
