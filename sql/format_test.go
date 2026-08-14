package sql

import "testing"

// parseExprOf parses query and returns the expression of its first projection.
func parseExprOf(t *testing.T, query string) Expr {
	t.Helper()
	node, err := NewParser(query).ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	stmt, ok := node.(*SelectStmt)
	if !ok {
		t.Fatalf("parse %q: expected *SelectStmt, got %T", query, node)
	}
	if len(stmt.Columns) == 0 {
		t.Fatalf("parse %q: no projections", query)
	}
	return stmt.Columns[0].Expr
}

// TestFormatExprRoundTrip renders parsed expressions back to SQL-like text.
// Whitespace and redundant parentheses are normalized, so the expected strings
// are the canonical rendering rather than the input text.
func TestFormatExprRoundTrip(t *testing.T) {
	cases := []struct{ query, want string }{
		// Column references and literals.
		{"SELECT a FROM t", "a"},
		{"SELECT t.a FROM t", "t.a"},
		{"SELECT 42 FROM t", "42"},
		{"SELECT 0.05 FROM t", "0.05"},
		{"SELECT 'hi' FROM t", "'hi'"},
		{"SELECT NULL FROM t", "NULL"},

		// Arithmetic, including precedence-driven parentheses.
		{"SELECT a * b FROM t", "a * b"},
		{"SELECT a + b * c FROM t", "a + b * c"},
		{"SELECT (a + b) * c FROM t", "(a + b) * c"},
		{"SELECT a * (1 - b) FROM t", "a * (1 - b)"},
		{"SELECT a - (b - c) FROM t", "a - (b - c)"},
		{"SELECT a - b - c FROM t", "a - b - c"},
		{"SELECT a * (1 - b) * (1 + c) FROM t", "a * (1 - b) * (1 + c)"},
		{"SELECT -a FROM t", "-a"},
		{"SELECT -(a + b) FROM t", "-(a + b)"},

		// Aggregates.
		{"SELECT COUNT(*) FROM t", "COUNT(*)"},
		{"SELECT SUM(a) FROM t", "SUM(a)"},
		{"SELECT SUM(a * b) FROM t", "SUM(a * b)"},
		{"SELECT COUNT(DISTINCT a) FROM t", "COUNT(DISTINCT a)"},

		// Predicates as expressions.
		{"SELECT a = 1 FROM t", "a = 1"},
		{"SELECT a IS NULL FROM t", "a IS NULL"},
		{"SELECT a IS NOT NULL FROM t", "a IS NOT NULL"},
		{"SELECT a BETWEEN 1 AND 2 FROM t", "a BETWEEN 1 AND 2"},
		{"SELECT a NOT BETWEEN 1 AND 2 FROM t", "a NOT BETWEEN 1 AND 2"},
		{"SELECT a IN (1, 2, 3) FROM t", "a IN (1, 2, 3)"},
		{"SELECT a NOT IN (1, 2) FROM t", "a NOT IN (1, 2)"},
		{"SELECT a LIKE '%x%' FROM t", "a LIKE '%x%'"},
		{"SELECT NOT a = 1 FROM t", "NOT (a = 1)"},
		{"SELECT a = 1 AND b = 2 FROM t", "a = 1 AND b = 2"},
		{"SELECT a = 1 OR b = 2 AND c = 3 FROM t", "a = 1 OR b = 2 AND c = 3"},

		// CASE WHEN.
		{
			"SELECT CASE WHEN a = 1 THEN 'one' ELSE 'other' END FROM t",
			"CASE WHEN a = 1 THEN 'one' ELSE 'other' END",
		},
		{
			"SELECT CASE WHEN a = 1 THEN 1 WHEN a = 2 THEN 2 END FROM t",
			"CASE WHEN a = 1 THEN 1 WHEN a = 2 THEN 2 END",
		},
	}

	for _, tc := range cases {
		got := FormatExpr(parseExprOf(t, tc.query))
		if got != tc.want {
			t.Errorf("%s\n  expected: %s\n  got:      %s", tc.query, tc.want, got)
		}
	}
}

// TestFormatExprLiterals covers renderings the parser cannot produce directly
// from a projection, plus quote escaping.
func TestFormatExprLiterals(t *testing.T) {
	cases := []struct {
		expr Expr
		want string
	}{
		{&StringLiteral{Value: "it's"}, "'it''s'"},
		{&BoolLiteral{Value: true}, "TRUE"},
		{&BoolLiteral{Value: false}, "FALSE"},
		{&StarExpr{}, "*"},
		{&FloatLiteral{Value: 1}, "1"},
		{&IntLiteral{Value: -7}, "-7"},
		{&AggFuncExpr{Func: "count"}, "COUNT(*)"},
		{&AggFuncExpr{Func: "sum", Arg: &ColumnRefExpr{Name: "a"}}, "SUM(a)"},
		{&FuncExpr{Func: "UPPER", Args: []Expr{&ColumnRefExpr{Name: "a"}}}, "UPPER(a)"},
		{&SubqueryExpr{}, "(subquery)"},
	}
	for _, tc := range cases {
		if got := FormatExpr(tc.expr); got != tc.want {
			t.Errorf("%T: expected %q, got %q", tc.expr, tc.want, got)
		}
	}
}

func TestFormatExprNil(t *testing.T) {
	if got := FormatExpr(nil); got != "" {
		t.Errorf("expected empty string for nil expression, got %q", got)
	}
}
