package planner

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// writePostOpsTables writes f(k, v) and g(k, w), which share the column name k.
func writePostOpsTables(t *testing.T) *catalog.Catalog {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	write := func(name string, cols ...string) string {
		path := filepath.Join(dir, name+".vxq")
		fields := make([]storage.Field, len(cols))
		for i, c := range cols {
			fields[i] = storage.Field{Name: c, Type: storage.TypeInt64}
		}
		w, err := storage.NewWriter(path, storage.Schema{Fields: fields})
		if err != nil {
			t.Fatal(err)
		}
		for rg := 0; rg < 3; rg++ {
			if err := w.BeginRowGroup(10); err != nil {
				t.Fatal(err)
			}
			for c := range cols {
				vals := make([]int64, 10)
				for i := range vals {
					vals[i] = int64(rg*10 + i + c)
				}
				if err := w.AppendColumn(ctx, c, nil, vals); err != nil {
					t.Fatal(err)
				}
			}
			if err := w.EndRowGroup(); err != nil {
				t.Fatal(err)
			}
		}
		if err := w.Finish(ctx); err != nil {
			t.Fatal(err)
		}
		return path
	}
	cat, err := catalog.OpenMulti(ctx, map[string]string{
		"f": write("f", "k", "v"),
		"g": write("g", "k", "w"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return cat
}

// TestParallelKeepsAggregateParallelUnderPostOps asserts that the operators
// Build places above an aggregate — HAVING, the SELECT-list projection, ORDER BY,
// LIMIT/OFFSET, the final renaming projection — do not push planner.Parallel
// onto its serial fallback. Parallel falls back silently, so without this a
// query shape that stopped being parallel would still pass every result
// comparison.
func TestParallelKeepsAggregateParallelUnderPostOps(t *testing.T) {
	ctx := context.Background()
	cat := writePostOpsTables(t)

	parallel := []string{
		"SELECT v, COUNT(*) FROM f GROUP BY v",
		"SELECT COUNT(*) AS n, v FROM f GROUP BY v",
		"SELECT v AS key, SUM(k) + 1 AS s FROM f GROUP BY v",
		"SELECT v FROM f GROUP BY v HAVING SUM(k) > 3",
		"SELECT v, SUM(k) AS s FROM f GROUP BY v HAVING COUNT(*) > 0 ORDER BY s DESC LIMIT 2 OFFSET 1",
		"SELECT v FROM f GROUP BY v ORDER BY SUM(k)",
		"SELECT COUNT(*), COUNT(*) FROM f",
		"SELECT f.k, SUM(g.w) AS s FROM f, g WHERE f.v = g.w GROUP BY f.k ORDER BY f.k LIMIT 3",
		"SELECT SUM(f.k), SUM(g.k) FROM f, g WHERE f.v = g.w AND f.k < g.k",
	}
	serial := []string{
		// LIMIT without ORDER BY: which groups survive depends on emission order.
		"SELECT v, COUNT(*) FROM f GROUP BY v LIMIT 2",
		// Partial COUNT(DISTINCT) results cannot be merged.
		"SELECT COUNT(DISTINCT v) FROM f",
		// No aggregate.
		"SELECT v FROM f ORDER BY v",
	}

	plan := func(q string) LogicalNode {
		t.Helper()
		node, err := sql.NewParser(q).ParseStatement()
		if err != nil {
			t.Fatalf("parse %q: %v", q, err)
		}
		logical, err := Build(ctx, node.(*sql.SelectStmt), cat)
		if err != nil {
			t.Fatalf("build %q: %v", q, err)
		}
		return Optimize(logical)
	}
	parallelizes := func(q string) bool {
		t.Helper()
		agg := aggregateUnderPostOps(plan(q))
		if agg == nil {
			return false
		}
		op, matched, err := tryParallelJoin(ctx, agg, 4)
		if err == nil && !matched {
			op, matched, err = tryParallelScanAggregate(ctx, agg, 4)
		}
		if err != nil {
			t.Fatalf("%q: %v", q, err)
		}
		if matched {
			_ = op.Close()
		}
		return matched
	}

	for _, q := range parallel {
		if !parallelizes(q) {
			t.Errorf("expected a parallel aggregate for %q", q)
		}
	}
	for _, q := range serial {
		if parallelizes(q) {
			t.Errorf("expected serial fallback for %q", q)
		}
	}
}
