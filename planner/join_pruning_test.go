package planner_test

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// ---- Plan inspection --------------------------------------------------------

// scanNeededCols walks an optimized logical plan and returns each scanned
// table's pruned column list. A nil entry means the scan was left unpruned,
// i.e. it reads every column.
func scanNeededCols(t *testing.T, root planner.LogicalNode) map[string][]string {
	t.Helper()
	out := make(map[string][]string)
	var walk func(planner.LogicalNode)
	walk = func(n planner.LogicalNode) {
		switch x := n.(type) {
		case *planner.LogicalScan:
			out[x.TableName] = x.NeededCols
		case *planner.LogicalJoin:
			walk(x.Left)
			walk(x.Right)
		case *planner.LogicalFilter:
			walk(x.Child)
		case *planner.LogicalProject:
			walk(x.Child)
		case *planner.LogicalAggregate:
			walk(x.Child)
		case *planner.LogicalSort:
			walk(x.Child)
		case *planner.LogicalLimit:
			walk(x.Child)
		case *planner.LogicalDistinct:
			walk(x.Child)
		default:
			t.Fatalf("scanNeededCols: unhandled node %T", n)
		}
	}
	walk(root)
	return out
}

// formatNeeded renders a needed-column list for comparison: sorted for
// order-insensitivity, with nil rendered as "*" (every column).
func formatNeeded(cols []string) string {
	if cols == nil {
		return "*"
	}
	c := make([]string, len(cols))
	copy(c, cols)
	sort.Strings(c)
	return strings.Join(c, ",")
}

// ---- Result oracles ---------------------------------------------------------

// runUnpruned executes a query through the serial planner with optimization
// skipped entirely, so every scan reads every column. Predicate placement still
// happens in Build (buildMultiTablePlan pushes single-table WHERE terms into
// each scan), so this is the same query over unpruned inputs — the oracle for
// "did pruning drop data the query needed?".
func runUnpruned(t *testing.T, cat *catalog.Catalog, query string) [][]any {
	t.Helper()
	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		t.Fatalf("parse %q: %v", query, err)
	}
	logical, err := planner.Build(context.Background(), node.(*sql.SelectStmt), cat)
	if err != nil {
		t.Fatalf("Build(%q): %v", query, err)
	}
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		t.Fatalf("Physical(unoptimized, %q): %v", query, err)
	}
	return drainResults(t, op)
}

// ---- Three-table fixture ----------------------------------------------------

// writeThreeTableDataset extends writeJoinDataset with a customers table keyed
// on c_orderkey, producing the TPC-H Q3 shape: customers ⋈ orders ⋈ items,
// where the build side of the top join is itself a join.
//
// customers: c_orderkey INT64, c_segment STRING, c_nation INT64, c_comment STRING
func writeThreeTableDataset(t *testing.T, ds joinDataset) *catalog.Catalog {
	t.Helper()
	ctx := context.Background()
	cat := writeJoinDataset(t, ds)

	custSchema := storage.Schema{Fields: []storage.Field{
		{Name: "c_orderkey", Type: storage.TypeInt64, Nullable: true},
		{Name: "c_segment", Type: storage.TypeString, Nullable: true},
		{Name: "c_nation", Type: storage.TypeInt64, Nullable: true},
		{Name: "c_comment", Type: storage.TypeString, Nullable: true},
	}}
	custPath := t.TempDir() + "/customers.vxq"
	w, err := storage.NewWriter(custPath, custSchema)
	if err != nil {
		t.Fatalf("NewWriter(customers): %v", err)
	}
	n := ds.buildRows
	keys := make([]int64, n)
	segs := make([]string, n)
	nations := make([]int64, n)
	comments := make([]string, n)
	for i := range n {
		keys[i] = int64(i + 1)
		segs[i] = []string{"BUILDING", "MACHINERY"}[i%2]
		nations[i] = int64(i % 25)
		comments[i] = fmt.Sprintf("comment-%d", i%7)
	}
	if err := w.BeginRowGroup(n); err != nil {
		t.Fatalf("BeginRowGroup(customers): %v", err)
	}
	mustAppend(t, w, 0, nil, keys)
	mustAppend(t, w, 1, nil, segs)
	mustAppend(t, w, 2, nil, nations)
	mustAppend(t, w, 3, nil, comments)
	if err := w.EndRowGroup(); err != nil {
		t.Fatalf("EndRowGroup(customers): %v", err)
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish(customers): %v", err)
	}
	cat.Register("customers", custPath, custSchema)
	return cat
}

// ---- Tests ------------------------------------------------------------------

// TestJoinColumnPruning_NeededCols asserts exactly which columns each side of a
// join is asked to decode. Under-pruning is a lost optimization; over-pruning is
// a correctness bug, so both directions are pinned.
//
// orders (build): o_orderkey, o_priority, o_status, o_total
// items  (probe): l_orderkey, l_shipmode, l_quantity, l_price
func TestJoinColumnPruning_NeededCols(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 2, buildRows: 500})

	tests := []struct {
		name   string
		query  string
		orders string // expected orders NeededCols, "*" = unpruned
		items  string // expected items NeededCols
	}{
		{
			name:   "join key in SELECT",
			query:  `SELECT o_orderkey, l_price FROM orders, items WHERE o_orderkey = l_orderkey`,
			orders: "o_orderkey",
			items:  "l_orderkey,l_price",
		},
		{
			name:   "join key not in SELECT is still needed",
			query:  `SELECT o_status, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey",
		},
		{
			name:   "qualified references resolve to their own side",
			query:  `SELECT orders.o_status, items.l_shipmode FROM orders, items WHERE orders.o_orderkey = items.l_orderkey`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey,l_shipmode",
		},
		{
			name:   "aliased references resolve to their own side",
			query:  `SELECT o.o_total, i.l_quantity FROM orders o, items i WHERE o.o_orderkey = i.l_orderkey`,
			orders: "o_orderkey,o_total",
			items:  "l_orderkey,l_quantity",
		},
		{
			name:   "filter on build side only",
			query:  `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_status = 'F'`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey",
		},
		{
			name:   "filter on probe side only",
			query:  `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_shipmode = 'MAIL'`,
			orders: "o_orderkey",
			items:  "l_orderkey,l_shipmode",
		},
		{
			name:   "filters on both sides",
			query:  `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_status = 'F' AND l_quantity > 10`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey,l_quantity",
		},
		{
			name: "expression aggregate spanning both tables contributes source columns",
			query: `SELECT o_status, SUM(o_total * l_price) AS x FROM orders, items
			        WHERE o_orderkey = l_orderkey GROUP BY o_status`,
			orders: "o_orderkey,o_status,o_total",
			items:  "l_orderkey,l_price",
		},
		{
			name: "CASE WHEN aggregate over build side (Q12 shape)",
			query: `SELECT l_shipmode,
			          SUM(CASE WHEN o_priority = '1-URGENT' THEN 1 ELSE 0 END) AS hi
			        FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
			orders: "o_orderkey,o_priority",
			items:  "l_orderkey,l_shipmode",
		},
		{
			name: "GROUP BY, HAVING and ORDER BY columns all reach their side",
			query: `SELECT o_status, SUM(l_price) AS rev FROM orders, items
			        WHERE o_orderkey = l_orderkey GROUP BY o_status
			        HAVING COUNT(*) > 1 ORDER BY rev DESC`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey,l_price",
		},
		{
			name:   "SELECT star over a join prunes nothing",
			query:  `SELECT * FROM orders, items WHERE o_orderkey = l_orderkey`,
			orders: "*",
			items:  "*",
		},
		{
			name:   "SELECT star with ORDER BY over a join prunes nothing",
			query:  `SELECT * FROM orders, items WHERE o_orderkey = l_orderkey ORDER BY o_orderkey`,
			orders: "*",
			items:  "*",
		},
		{
			name:   "DISTINCT over a join prunes to the projected columns",
			query:  `SELECT DISTINCT o_status, l_shipmode FROM orders, items WHERE o_orderkey = l_orderkey`,
			orders: "o_orderkey,o_status",
			items:  "l_orderkey,l_shipmode",
		},
		{
			name:   "COUNT(*) needs only the join keys",
			query:  `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`,
			orders: "o_orderkey",
			items:  "l_orderkey",
		},
		{
			name: "MIN/MAX/AVG source columns reach their side",
			query: `SELECT l_shipmode, MIN(o_total) AS mn, AVG(l_quantity) AS aq FROM orders, items
			        WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
			orders: "o_orderkey,o_total",
			items:  "l_orderkey,l_quantity,l_shipmode",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			needed := scanNeededCols(t, buildPlan(t, tc.query, cat))
			if got := formatNeeded(needed["orders"]); got != tc.orders {
				t.Errorf("orders NeededCols = %q, want %q", got, tc.orders)
			}
			if got := formatNeeded(needed["items"]); got != tc.items {
				t.Errorf("items NeededCols = %q, want %q", got, tc.items)
			}
		})
	}
}

// TestJoinColumnPruning_ThreeTableNeededCols pins pruning through a nested join:
// the top join's build side is customers ⋈ orders, so the needed set has to be
// split twice on the way down.
func TestJoinColumnPruning_ThreeTableNeededCols(t *testing.T) {
	cat := writeThreeTableDataset(t, joinDataset{probeRowGroups: 2, buildRows: 500})

	// Q3 shape: a filter on the outermost build table, aggregates on the probe
	// table, group-by columns drawn from the middle table.
	query := `SELECT o_status, SUM(l_price) AS revenue
	          FROM customers, orders, items
	          WHERE c_orderkey = o_orderkey AND o_orderkey = l_orderkey
	            AND c_segment = 'BUILDING'
	          GROUP BY o_status
	          ORDER BY revenue DESC`

	needed := scanNeededCols(t, buildPlan(t, query, cat))
	for table, want := range map[string]string{
		"customers": "c_orderkey,c_segment",
		"orders":    "o_orderkey,o_status",
		"items":     "l_orderkey,l_price",
	} {
		if got := formatNeeded(needed[table]); got != want {
			t.Errorf("%s NeededCols = %q, want %q", table, got, want)
		}
	}
}

// TestJoinColumnPruning_ResultsMatchUnpruned is the correctness gate: every
// pruning-sensitive shape must produce the same rows as the same query planned
// with pruning disabled, through the serial planner and through the parallel
// planner at 1, 2 and 4 workers.
func TestJoinColumnPruning_ResultsMatchUnpruned(t *testing.T) {
	cat := writeThreeTableDataset(t, joinDataset{
		probeRowGroups: 4,
		buildRows:      2000,
		nullKeyEvery:   37,
		unmatchedEvery: 53,
	})

	tests := []struct {
		name    string
		query   string
		ordered bool
	}{
		{
			name:    "join key in SELECT",
			query:   `SELECT o_orderkey, l_price FROM orders, items WHERE o_orderkey = l_orderkey ORDER BY o_orderkey, l_price LIMIT 50`,
			ordered: true,
		},
		{
			name:  "join key not in SELECT",
			query: `SELECT o_status, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status`,
		},
		{
			name:  "qualified references",
			query: `SELECT orders.o_status, COUNT(*) AS c FROM orders, items WHERE orders.o_orderkey = items.l_orderkey GROUP BY orders.o_status`,
		},
		{
			name:  "aliased references",
			query: `SELECT o.o_priority, SUM(i.l_price) AS rev FROM orders o, items i WHERE o.o_orderkey = i.l_orderkey GROUP BY o.o_priority`,
		},
		{
			name:  "filter on build side",
			query: `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_status = 'F' GROUP BY l_shipmode`,
		},
		{
			name:  "filter on probe side",
			query: `SELECT o_status, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_shipmode = 'MAIL' GROUP BY o_status`,
		},
		{
			name:  "filters on both sides",
			query: `SELECT o_status, l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_status = 'F' AND l_quantity > 500 GROUP BY o_status, l_shipmode`,
		},
		{
			name:  "expression aggregate spanning both tables",
			query: `SELECT o_status, SUM(o_total * l_price) AS x FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status`,
		},
		{
			name: "CASE WHEN aggregate over build side (Q12 shape)",
			query: `SELECT l_shipmode,
			          SUM(CASE WHEN o_priority = '1-URGENT' OR o_priority = '2-HIGH' THEN 1 ELSE 0 END) AS hi,
			          SUM(CASE WHEN o_priority <> '1-URGENT' AND o_priority <> '2-HIGH' THEN 1 ELSE 0 END) AS lo
			        FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity < 2000
			        GROUP BY l_shipmode`,
		},
		{
			name:    "three-table chain (Q3 shape)",
			query:   `SELECT o_status, SUM(l_price) AS revenue FROM customers, orders, items WHERE c_orderkey = o_orderkey AND o_orderkey = l_orderkey AND c_segment = 'BUILDING' GROUP BY o_status ORDER BY o_status`,
			ordered: true,
		},
		{
			name:  "three-table chain, middle table used only for the join",
			query: `SELECT c_segment, COUNT(*) AS c FROM customers, orders, items WHERE c_orderkey = o_orderkey AND o_orderkey = l_orderkey AND l_shipmode = 'AIR' GROUP BY c_segment`,
		},
		{
			name:    "SELECT star over a join",
			query:   `SELECT * FROM orders, items WHERE o_orderkey = l_orderkey ORDER BY o_orderkey, l_quantity LIMIT 20`,
			ordered: true,
		},
		{
			name:  "HAVING with a hidden aggregate",
			query: `SELECT o_status FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status HAVING SUM(l_price) > 100`,
		},
		{
			name:  "DISTINCT over a join",
			query: `SELECT DISTINCT o_status, l_shipmode FROM orders, items WHERE o_orderkey = l_orderkey`,
		},
		{
			name:  "COUNT(DISTINCT) over a join",
			query: `SELECT o_status, COUNT(DISTINCT l_shipmode) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status`,
		},
		{
			name:    "no aggregate, projection over a join",
			query:   `SELECT o_priority, l_shipmode, l_price FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity = 7 ORDER BY o_priority, l_shipmode, l_price LIMIT 30`,
			ordered: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			want := runUnpruned(t, cat, tc.query)
			if len(want) == 0 {
				t.Fatal("oracle returned no rows — the test query does not exercise the join")
			}
			assertSameRows(t, "serial pruned", want, runSerial(t, cat, tc.query), tc.ordered)
			for _, workers := range []int{1, 2, 4} {
				assertSameRows(t, fmt.Sprintf("parallel workers=%d", workers),
					want, runParallel(t, cat, tc.query, workers), tc.ordered)
			}
		})
	}
}

// TestColumnPruning_NilNeededMeansAllColumns is a regression test for the
// "nil needed set" convention. Filter, Sort, Limit and Distinct pass their input
// through unchanged, so a nil set (an ancestor needs every column, as SELECT *
// does) must stay nil on the way down. Merging into it instead narrowed the set
// to just the pass-through node's own columns, so `SELECT * FROM t ORDER BY c`
// returned only column c.
func TestColumnPruning_NilNeededMeansAllColumns(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 1, buildRows: 100})

	tests := []struct {
		name     string
		query    string
		wantCols []string
	}{
		{
			name:     "select star",
			query:    `SELECT * FROM orders LIMIT 5`,
			wantCols: []string{"o_orderkey", "o_priority", "o_status", "o_total"},
		},
		{
			name:     "select star with ORDER BY",
			query:    `SELECT * FROM orders ORDER BY o_orderkey LIMIT 5`,
			wantCols: []string{"o_orderkey", "o_priority", "o_status", "o_total"},
		},
		{
			name:     "select star with ORDER BY and WHERE",
			query:    `SELECT * FROM orders WHERE o_status = 'F' ORDER BY o_total LIMIT 5`,
			wantCols: []string{"o_orderkey", "o_priority", "o_status", "o_total"},
		},
		{
			name:     "explicit projection still prunes",
			query:    `SELECT o_status FROM orders ORDER BY o_status LIMIT 5`,
			wantCols: []string{"o_status"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			op, err := planner.Physical(context.Background(), buildPlan(t, tc.query, cat))
			if err != nil {
				t.Fatalf("Physical: %v", err)
			}
			defer op.Close()
			var got []string
			for _, f := range op.Schema().Fields {
				got = append(got, f.Name)
			}
			if strings.Join(got, ",") != strings.Join(tc.wantCols, ",") {
				t.Fatalf("output columns = %v, want %v", got, tc.wantCols)
			}
		})
	}
}

// TestJoinColumnPruning_BuildSideMaterializesFewerColumns asserts the pruning
// actually reaches the hash table: the build side's shared hash table carries
// only the columns the query reads, which is what shrinks the per-row slices
// buildHashTableFrom allocates.
func TestJoinColumnPruning_BuildSideMaterializesFewerColumns(t *testing.T) {
	ctx := context.Background()
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 1, buildRows: 200})

	const query = `SELECT l_shipmode, COUNT(*) AS c FROM orders, items
	               WHERE o_orderkey = l_orderkey AND o_status = 'F' GROUP BY l_shipmode`

	logical := buildPlan(t, query, cat)
	join := findJoin(t, logical)

	buildOp, err := planner.Physical(ctx, join.Left)
	if err != nil {
		t.Fatalf("Physical(build side): %v", err)
	}
	defer buildOp.Close()

	var got []string
	for _, f := range buildOp.Schema().Fields {
		got = append(got, f.Name)
	}
	// o_priority and o_total are never referenced, so neither is decoded or
	// materialized into a build row.
	if want := "o_orderkey,o_status"; strings.Join(got, ",") != want {
		t.Fatalf("build side schema = %v, want %s", got, want)
	}

	sht, err := exec.BuildSharedHashTable(ctx, buildOp, 0)
	if err != nil {
		t.Fatalf("BuildSharedHashTable: %v", err)
	}
	if n := len(sht.Schema().Fields); n != 2 {
		t.Fatalf("shared hash table column count = %d, want 2", n)
	}
	if sht.NumRows() == 0 {
		t.Fatal("shared hash table is empty — the build side produced no rows")
	}
}

// findJoin returns the first LogicalJoin found walking down a plan.
func findJoin(t *testing.T, n planner.LogicalNode) *planner.LogicalJoin {
	t.Helper()
	for {
		switch x := n.(type) {
		case *planner.LogicalJoin:
			return x
		case *planner.LogicalFilter:
			n = x.Child
		case *planner.LogicalProject:
			n = x.Child
		case *planner.LogicalAggregate:
			n = x.Child
		case *planner.LogicalSort:
			n = x.Child
		case *planner.LogicalLimit:
			n = x.Child
		case *planner.LogicalDistinct:
			n = x.Child
		default:
			t.Fatalf("findJoin: no join in plan (reached %T)", n)
			return nil
		}
	}
}
