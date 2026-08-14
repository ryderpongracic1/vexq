package planner_test

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/storage"
)

// ---- Synthetic orders ⋈ items dataset ---------------------------------------

// joinDataset describes the shape of a generated two-table dataset.
type joinDataset struct {
	// probeRowGroups is the number of full row groups written to the probe
	// (items) table. Worker count is capped to the probe row group count, so
	// exercising 8 concurrent workers needs at least 8 row groups.
	probeRowGroups int
	// buildRows is the number of rows in the build (orders) table. Join keys are
	// o_orderkey ∈ [1, buildRows].
	buildRows int
	// nullKeyEvery, when > 0, makes every Nth join key NULL on both sides.
	nullKeyEvery int
	// unmatchedEvery, when > 0, makes every Nth probe row reference a key that
	// does not exist on the build side.
	unmatchedEvery int
}

// writeJoinDataset writes orders.vxq (build side) and items.vxq (probe side) to
// a temp dir and returns a catalog over both. Data is fully deterministic so
// serial and parallel executions are directly comparable.
//
// orders: o_orderkey INT64, o_priority STRING, o_status STRING, o_total FLOAT64
// items:  l_orderkey INT64, l_shipmode STRING, l_quantity INT64, l_price FLOAT64
func writeJoinDataset(t *testing.T, ds joinDataset) *catalog.Catalog {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()

	priorities := []string{"1-URGENT", "2-HIGH", "3-MEDIUM", "4-LOW", "5-LOW"}
	statuses := []string{"O", "F", "P"}
	shipmodes := []string{"MAIL", "SHIP", "AIR", "TRUCK"}

	// ---- Build side: orders -------------------------------------------------
	ordersSchema := storage.Schema{Fields: []storage.Field{
		{Name: "o_orderkey", Type: storage.TypeInt64, Nullable: true},
		{Name: "o_priority", Type: storage.TypeString, Nullable: true},
		{Name: "o_status", Type: storage.TypeString, Nullable: true},
		{Name: "o_total", Type: storage.TypeFloat64, Nullable: true},
	}}
	ordersPath := dir + "/orders.vxq"
	ow, err := storage.NewWriter(ordersPath, ordersSchema)
	if err != nil {
		t.Fatalf("NewWriter(orders): %v", err)
	}
	for start := 0; start < ds.buildRows; start += storage.RowGroupRows {
		n := min(storage.RowGroupRows, ds.buildRows-start)
		keys := make([]int64, n)
		keyNulls := make([]byte, (n+7)/8)
		prio := make([]string, n)
		stat := make([]string, n)
		total := make([]float64, n)
		for i := range n {
			id := start + i + 1
			keys[i] = int64(id)
			if ds.nullKeyEvery == 0 || id%ds.nullKeyEvery != 0 {
				storage.SetValidBit(keyNulls, i)
			}
			prio[i] = priorities[id%len(priorities)]
			stat[i] = statuses[id%len(statuses)]
			total[i] = float64(id%997) * 1.25
		}
		if err := ow.BeginRowGroup(n); err != nil {
			t.Fatalf("BeginRowGroup(orders): %v", err)
		}
		mustAppend(t, ow, 0, keyNulls, keys)
		mustAppend(t, ow, 1, nil, prio)
		mustAppend(t, ow, 2, nil, stat)
		mustAppend(t, ow, 3, nil, total)
		if err := ow.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup(orders): %v", err)
		}
	}
	if err := ow.Finish(ctx); err != nil {
		t.Fatalf("Finish(orders): %v", err)
	}

	// ---- Probe side: items --------------------------------------------------
	itemsSchema := storage.Schema{Fields: []storage.Field{
		{Name: "l_orderkey", Type: storage.TypeInt64, Nullable: true},
		{Name: "l_shipmode", Type: storage.TypeString, Nullable: true},
		{Name: "l_quantity", Type: storage.TypeInt64, Nullable: true},
		{Name: "l_price", Type: storage.TypeFloat64, Nullable: true},
	}}
	itemsPath := dir + "/items.vxq"
	iw, err := storage.NewWriter(itemsPath, itemsSchema)
	if err != nil {
		t.Fatalf("NewWriter(items): %v", err)
	}
	for rg := range ds.probeRowGroups {
		n := storage.RowGroupRows
		keys := make([]int64, n)
		keyNulls := make([]byte, (n+7)/8)
		mode := make([]string, n)
		qty := make([]int64, n)
		price := make([]float64, n)
		for i := range n {
			row := rg*n + i
			// Cycle through build keys so every build row gets matches.
			key := int64(row%ds.buildRows) + 1
			if ds.unmatchedEvery > 0 && row%ds.unmatchedEvery == 0 {
				key = int64(ds.buildRows) + int64(row) + 1 // no such order
			}
			keys[i] = key
			if ds.nullKeyEvery == 0 || row%ds.nullKeyEvery != 0 {
				storage.SetValidBit(keyNulls, i)
			}
			mode[i] = shipmodes[row%len(shipmodes)]
			// l_quantity is monotonic within a row group and disjoint across row
			// groups, so a predicate on it can wipe out whole morsels.
			qty[i] = int64(rg*1000 + i%1000)
			price[i] = float64(row%701) * 0.5
		}
		if err := iw.BeginRowGroup(n); err != nil {
			t.Fatalf("BeginRowGroup(items): %v", err)
		}
		mustAppend(t, iw, 0, keyNulls, keys)
		mustAppend(t, iw, 1, nil, mode)
		mustAppend(t, iw, 2, nil, qty)
		mustAppend(t, iw, 3, nil, price)
		if err := iw.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup(items): %v", err)
		}
	}
	if err := iw.Finish(ctx); err != nil {
		t.Fatalf("Finish(items): %v", err)
	}

	cat, err := catalog.OpenMulti(ctx, map[string]string{
		"orders": ordersPath,
		"items":  itemsPath,
	})
	if err != nil {
		t.Fatalf("OpenMulti: %v", err)
	}
	return cat
}

func mustAppend(t *testing.T, w *storage.Writer, col int, nulls []byte, vals any) {
	t.Helper()
	if err := w.AppendColumn(context.Background(), col, nulls, vals); err != nil {
		t.Fatalf("AppendColumn(%d): %v", col, err)
	}
}

// ---- Helpers ----------------------------------------------------------------

// runSerial executes a query through the serial physical planner.
func runSerial(t *testing.T, cat *catalog.Catalog, query string) [][]any {
	t.Helper()
	logical := buildPlan(t, query, cat)
	op, err := planner.Physical(context.Background(), logical)
	if err != nil {
		t.Fatalf("Physical(%q): %v", query, err)
	}
	return drainResults(t, op)
}

// runParallel executes a query through the parallel planner with n workers.
func runParallel(t *testing.T, cat *catalog.Catalog, query string, workers int) [][]any {
	t.Helper()
	logical := buildPlan(t, query, cat)
	op, err := planner.Parallel(context.Background(), logical, workers)
	if err != nil {
		t.Fatalf("Parallel(%q, workers=%d): %v", query, workers, err)
	}
	return drainResults(t, op)
}

// sortRows returns a copy of rows in a deterministic order, for comparing
// results of queries with no ORDER BY. Group emission order after a partial
// aggregate merge depends on which worker saw a group first, so unordered
// queries are compared as row multisets — the same contract the golden test
// suite applies to unordered queries.
func sortRows(rows [][]any) [][]any {
	out := make([][]any, len(rows))
	copy(out, rows)
	sort.Slice(out, func(i, j int) bool {
		return fmt.Sprint(out[i]) < fmt.Sprint(out[j])
	})
	return out
}

// assertSameRows fails unless want and got hold the same rows. When ordered is
// true the row order must match exactly as well.
func assertSameRows(t *testing.T, label string, want, got [][]any, ordered bool) {
	t.Helper()
	if !ordered {
		want, got = sortRows(want), sortRows(got)
	}
	if len(want) != len(got) {
		t.Fatalf("%s: row count = %d, want %d\n got: %v\nwant: %v", label, len(got), len(want), got, want)
	}
	for i := range want {
		if !reflect.DeepEqual(want[i], got[i]) {
			t.Fatalf("%s: row %d = %v, want %v", label, i, got[i], want[i])
		}
	}
}

// ---- Tests ------------------------------------------------------------------

// TestParallelJoin_PlanShape asserts which plans reach the parallel join
// operator and which fall back to a serial or aggregate-only plan.
func TestParallelJoin_PlanShape(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 2, buildRows: 1000})

	tests := []struct {
		name  string
		query string
		// want is the expected root operator type.
		want any
	}{
		{
			name:  "aggregate over join",
			query: `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
			want:  &exec.ParallelHashJoinAggregate{},
		},
		{
			name:  "sort peeled above join aggregate",
			query: `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode ORDER BY l_shipmode`,
			want:  &exec.ExternalSort{},
		},
		{
			name:  "limit and sort peeled above join aggregate",
			query: `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode ORDER BY l_shipmode LIMIT 2`,
			want:  &exec.Limit{},
		},
		{
			name:  "no aggregate falls back to serial join",
			query: `SELECT l_shipmode, o_total FROM orders, items WHERE o_orderkey = l_orderkey LIMIT 5`,
			want:  &exec.Limit{},
		},
		{
			name:  "aggregate over single scan keeps the existing path",
			query: `SELECT l_shipmode, COUNT(*) AS c FROM items GROUP BY l_shipmode`,
			want:  &exec.ParallelHashAggregate{},
		},
		{
			name:  "count distinct over join falls back to serial",
			query: `SELECT l_shipmode, COUNT(DISTINCT l_orderkey) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
			want:  &exec.HashAggregate{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logical := buildPlan(t, tc.query, cat)
			op, err := planner.Parallel(context.Background(), logical, 4)
			if err != nil {
				t.Fatalf("Parallel: %v", err)
			}
			defer op.Close()
			if got, want := reflect.TypeOf(op), reflect.TypeOf(tc.want); got != want {
				t.Fatalf("root operator = %v, want %v", got, want)
			}
		})
	}
}

// TestParallelJoin_MatchesSerial is the core correctness gate: for every query
// shape, parallel results at 1, 2 and 8 workers must equal the serial result.
// The dataset spans 8 probe row groups so an 8-worker run really does run eight
// concurrent probe pipelines (worker count is capped to the row group count).
func TestParallelJoin_MatchesSerial(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: 8,
		buildRows:      5000,
		unmatchedEvery: 97,
	})

	tests := []struct {
		name    string
		query   string
		ordered bool
	}{
		{
			// TPC-H Q12 shape: two-table join, CASE WHEN aggregates over the
			// build side's columns, GROUP BY a probe-side column, ORDER BY.
			name: "q12 shaped case when aggregate",
			query: `SELECT l_shipmode,
			           SUM(CASE WHEN o_priority = '1-URGENT' OR o_priority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
			           SUM(CASE WHEN o_priority <> '1-URGENT' AND o_priority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			          AND l_shipmode IN ('MAIL', 'SHIP')
			          AND l_quantity < 900
			        GROUP BY l_shipmode
			        ORDER BY l_shipmode`,
			ordered: true,
		},
		{
			name:    "count star no group by",
			query:   `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`,
			ordered: true,
		},
		{
			name: "float and int aggregates grouped by probe column",
			query: `SELECT l_shipmode, SUM(l_price) AS sp, AVG(o_total) AS ao, MIN(l_quantity) AS mnq, MAX(o_total) AS mxo, COUNT(l_price) AS cp
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY l_shipmode`,
			ordered: false,
		},
		{
			name: "grouped by build side column",
			query: `SELECT o_status, SUM(l_price) AS sp, COUNT(*) AS c
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY o_status`,
			ordered: false,
		},
		{
			name: "expression aggregate over both sides",
			query: `SELECT l_shipmode, SUM(l_price * o_total) AS rev
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY l_shipmode`,
			ordered: false,
		},
		{
			name: "selective probe predicate spanning few morsels",
			query: `SELECT l_shipmode, COUNT(*) AS c
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey AND l_quantity BETWEEN 2000 AND 2050
			        GROUP BY l_shipmode`,
			ordered: false,
		},
		{
			name: "build side predicate",
			query: `SELECT l_shipmode, COUNT(*) AS c
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey AND o_status = 'F'
			        GROUP BY l_shipmode`,
			ordered: false,
		},
		{
			name: "order by aggregate desc with limit",
			query: `SELECT l_shipmode, SUM(l_price) AS sp
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY l_shipmode
			        ORDER BY sp DESC
			        LIMIT 2`,
			ordered: true,
		},
		{
			name: "having over join aggregate",
			query: `SELECT l_shipmode, COUNT(*) AS c
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY l_shipmode
			        HAVING COUNT(*) > 1`,
			ordered: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			serial := runSerial(t, cat, tc.query)
			if len(serial) == 0 {
				t.Fatalf("serial result is empty — the query is not exercising the join")
			}
			for _, workers := range []int{1, 2, 8} {
				got := runParallel(t, cat, tc.query, workers)
				assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, tc.ordered)
			}
		})
	}
}

// TestParallelJoin_NullJoinKeys verifies that NULL keys on either side are
// dropped identically by both paths — an inner equi-join never matches NULL.
func TestParallelJoin_NullJoinKeys(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: 4,
		buildRows:      2000,
		nullKeyEvery:   3, // every third key NULL on both sides
		unmatchedEvery: 11,
	})

	queries := []string{
		`SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`,
		`SELECT l_shipmode, COUNT(*) AS c, SUM(l_price) AS sp FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
		`SELECT o_status, SUM(o_total) AS st FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY o_status`,
	}
	for _, q := range queries {
		t.Run(q[:40], func(t *testing.T) {
			serial := runSerial(t, cat, q)
			for _, workers := range []int{1, 2, 8} {
				got := runParallel(t, cat, q, workers)
				assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, false)
			}
		})
	}
}

// TestParallelJoin_EmptyAndZeroSurvivorMorsels covers the degenerate inputs that
// a morsel scheduler is most likely to get wrong: morsels whose filter admits no
// rows at all, a probe side that is entirely filtered away, and an empty build
// side (which short-circuits the probe entirely).
func TestParallelJoin_EmptyAndZeroSurvivorMorsels(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 6, buildRows: 3000})

	tests := []struct {
		name      string
		query     string
		wantEmpty bool
		wantRows  [][]any // optional exact expectation for the serial result
	}{
		{
			// l_quantity is disjoint per row group (rg*1000 + i%1000), so this
			// predicate only survives in row group 0 — five of six morsels
			// produce zero rows.
			name:      "only first morsel survives",
			query:     `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity < 500 GROUP BY l_shipmode`,
			wantEmpty: false,
		},
		{
			name:      "every probe morsel filtered away",
			query:     `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity > 1000000 GROUP BY l_shipmode`,
			wantEmpty: true,
		},
		{
			name:      "empty build side",
			query:     `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_status = 'NOPE' GROUP BY l_shipmode`,
			wantEmpty: true,
		},
		{
			// A global aggregate (no GROUP BY) over zero matched rows must still
			// emit exactly one row — COUNT 0 — on both paths.
			name:      "global aggregate with no matching build key",
			query:     `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND o_orderkey > 100000000`,
			wantEmpty: false,
			wantRows:  [][]any{{int64(0)}},
		},
		{
			name:      "global aggregate with every probe row filtered away",
			query:     `SELECT COUNT(*) AS c, SUM(l_price) AS sp FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity > 1000000`,
			wantEmpty: false,
			wantRows:  [][]any{{int64(0), nil}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			serial := runSerial(t, cat, tc.query)
			if tc.wantEmpty && len(serial) != 0 {
				t.Fatalf("expected serial result to be empty, got %v", serial)
			}
			if !tc.wantEmpty && len(serial) == 0 {
				t.Fatalf("expected serial result to be non-empty")
			}
			if tc.wantRows != nil {
				assertSameRows(t, "serial", tc.wantRows, serial, true)
			}
			for _, workers := range []int{1, 2, 8} {
				got := runParallel(t, cat, tc.query, workers)
				assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, false)
			}
		})
	}
}

// TestParallelJoin_LimitWithoutOrderByFallsBack verifies that LIMIT with no
// ORDER BY does not take the parallel join path. Both paths return an arbitrary
// subset of groups, but they would pick different subsets — serial emits groups
// in insertion order while the merged parallel order depends on which worker saw
// a group first. Falling back keeps the two paths result-identical, which this
// test checks by requiring exact row-for-row equality.
func TestParallelJoin_LimitWithoutOrderByFallsBack(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 4, buildRows: 2000})
	query := `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode LIMIT 2`

	serial := runSerial(t, cat, query)
	if len(serial) != 2 {
		t.Fatalf("serial rows = %d, want 2", len(serial))
	}
	for _, workers := range []int{1, 2, 8} {
		got := runParallel(t, cat, query, workers)
		assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, true)
	}
}

// TestParallelJoin_ThreeTableChain covers the TPC-H Q3 shape: a left-deep
// three-table chain where the build side is itself a join, evaluated once into
// the shared hash table while the last table is probed in parallel.
func TestParallelJoin_ThreeTableChain(t *testing.T) {
	ctx := context.Background()
	cat := writeJoinDataset(t, joinDataset{probeRowGroups: 4, buildRows: 2000})

	// Add a third table: customers keyed by o_status (a low-cardinality string
	// join is unsupported, so key on an int64 column derived from o_orderkey).
	dir := t.TempDir()
	custSchema := storage.Schema{Fields: []storage.Field{
		{Name: "c_orderkey", Type: storage.TypeInt64, Nullable: true},
		{Name: "c_segment", Type: storage.TypeString, Nullable: true},
	}}
	custPath := dir + "/customers.vxq"
	w, err := storage.NewWriter(custPath, custSchema)
	if err != nil {
		t.Fatalf("NewWriter(customers): %v", err)
	}
	const custRows = 2000
	keys := make([]int64, custRows)
	segs := make([]string, custRows)
	for i := range custRows {
		keys[i] = int64(i + 1)
		segs[i] = []string{"BUILDING", "MACHINERY"}[i%2]
	}
	if err := w.BeginRowGroup(custRows); err != nil {
		t.Fatalf("BeginRowGroup(customers): %v", err)
	}
	mustAppend(t, w, 0, nil, keys)
	mustAppend(t, w, 1, nil, segs)
	if err := w.EndRowGroup(); err != nil {
		t.Fatalf("EndRowGroup(customers): %v", err)
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish(customers): %v", err)
	}
	cat.Register("customers", custPath, custSchema)

	query := `SELECT l_shipmode, SUM(l_price) AS rev, COUNT(*) AS c
	          FROM customers, orders, items
	          WHERE c_orderkey = o_orderkey AND o_orderkey = l_orderkey
	            AND c_segment = 'BUILDING'
	          GROUP BY l_shipmode
	          ORDER BY l_shipmode`

	logical := buildPlan(t, query, cat)
	op, err := planner.Parallel(ctx, logical, 4)
	if err != nil {
		t.Fatalf("Parallel: %v", err)
	}
	// Sort is peeled, so the root is an ExternalSort over the parallel join.
	if _, ok := op.(*exec.ExternalSort); !ok {
		t.Fatalf("root operator = %T, want *exec.ExternalSort", op)
	}
	_ = op.Close()

	serial := runSerial(t, cat, query)
	if len(serial) == 0 {
		t.Fatal("serial result is empty — the three-table query is not producing rows")
	}
	for _, workers := range []int{1, 2, 4} {
		got := runParallel(t, cat, query, workers)
		assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, true)
	}
}
