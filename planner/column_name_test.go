package planner_test

import (
	"context"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
)

// headers plans query and returns the output column names the user would see.
func headers(t *testing.T, cat *catalog.Catalog, query string) []string {
	t.Helper()
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	logical, err := planner.Build(context.Background(), node.(*sql.SelectStmt), cat)
	if err != nil {
		t.Fatalf("Build %q: %v", query, err)
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		t.Fatalf("Physical %q: %v", query, err)
	}
	defer op.Close()

	names := make([]string, 0, len(op.Schema().Fields))
	for _, f := range op.Schema().Fields {
		names = append(names, f.Name)
	}
	return names
}

// TestOutputColumnHeaders covers how unaliased projections are named. The
// regression this guards: an aggregate over an expression used to render its
// Go type name (SUM_*sql.BinaryExpr) as the user-visible column header.
func TestOutputColumnHeaders(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name  string
		query string
		want  []string
	}{
		// --- The bug: aggregates over computed expressions ---
		{
			name:  "unaliased_expression_aggregate",
			query: "SELECT SUM(val * 2) FROM test",
			want:  []string{"SUM(val * 2)"},
		},
		{
			name:  "nested_arithmetic",
			query: "SELECT SUM(val * (1 - val)) FROM test",
			want:  []string{"SUM(val * (1 - val))"},
		},
		{
			name:  "unary_minus_argument",
			query: "SELECT SUM(-val) FROM test",
			want:  []string{"SUM(-val)"},
		},
		{
			name:  "case_when_argument",
			query: "SELECT SUM(CASE WHEN grp = 1 THEN 1 ELSE 0 END) FROM test",
			want:  []string{"SUM(CASE WHEN grp = 1 THEN 1 ELSE 0 END)"},
		},
		{
			name:  "unaliased_expression_projection",
			query: "SELECT val + 1 FROM test",
			want:  []string{"val + 1"},
		},

		// --- An explicit alias always wins ---
		{
			name:  "aliased_expression_aggregate",
			query: "SELECT SUM(val * 2) AS total FROM test",
			want:  []string{"total"},
		},
		{
			name:  "aliased_simple_aggregate",
			query: "SELECT SUM(val) AS total FROM test",
			want:  []string{"total"},
		},

		// --- Unchanged: simple column and star arguments keep FUNC_col ---
		{
			name:  "unaliased_simple_column_aggregate",
			query: "SELECT SUM(val) FROM test",
			want:  []string{"SUM_val"},
		},
		{
			name:  "count_star",
			query: "SELECT COUNT(*) FROM test",
			want:  []string{"COUNT_*"},
		},
		{
			name:  "count_distinct_column",
			query: "SELECT COUNT(DISTINCT val) FROM test",
			want:  []string{"COUNT_val"},
		},
		{
			name:  "avg_min_max_simple_columns",
			query: "SELECT AVG(val), MIN(val), MAX(val) FROM test",
			want:  []string{"AVG_val", "MIN_val", "MAX_val"},
		},

		// --- Grouped query mixing all of the above ---
		{
			name:  "group_by_mixed",
			query: "SELECT grp, SUM(val * 2), COUNT(*), SUM(val) AS plain FROM test GROUP BY grp",
			want:  []string{"grp", "SUM(val * 2)", "COUNT_*", "plain"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := headers(t, cat, tc.query)
			if len(got) != len(tc.want) {
				t.Fatalf("query %q: expected %d columns %v, got %d %v",
					tc.query, len(tc.want), tc.want, len(got), got)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("query %q: column %d: expected %q, got %q", tc.query, i, tc.want[i], got[i])
				}
			}
			// No header may leak a Go type name, whatever the expression shape.
			for _, name := range got {
				if strings.Contains(name, "sql.") || strings.Contains(name, "*sql") {
					t.Errorf("query %q: header %q leaks a Go type name", tc.query, name)
				}
			}
		})
	}
}

// TestExpressionAggregateStillExecutes confirms the readable header is only a
// rename: the aggregate over an expression still produces the right value.
// Groups: 1 → 5 rows of val=10, 2 → 2 rows of val=100, 3 → 10 rows of val=1.
// SUM(val * 2) over all 17 rows = 2 * (50 + 200 + 10) = 520.
func TestExpressionAggregateStillExecutes(t *testing.T) {
	path := writeGroupedFile(t)
	cat, err := catalog.OpenSingle(context.Background(), "test", path)
	if err != nil {
		t.Fatal(err)
	}

	schema, rows := runQuery(t, cat, "SELECT SUM(val * 2) FROM test")
	if len(schema.Fields) != 1 || schema.Fields[0].Name != "SUM(val * 2)" {
		t.Fatalf("unexpected schema: %v", schema.Fields)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d: %v", len(rows), rows)
	}
	if rows[0][0] != 520 {
		t.Errorf("expected SUM(val * 2) = 520, got %d", rows[0][0])
	}
}
