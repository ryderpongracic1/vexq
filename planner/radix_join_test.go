package planner_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/storage"
)

// Tests in this file cover the radix-partitioned build side end to end. They
// need a build side above exec's partitioning threshold (one row group,
// 65,536 rows), which is why the datasets here are larger than the ones in
// parallel_join_test.go — those deliberately stay under it and so exercise the
// unpartitioned path.
//
// radixBuildRows spans four row groups, so the parallel build has four morsels to
// distribute; radixProbeRowGroups caps the worker count, so it is four as well.
const (
	radixBuildRows      = 200_000
	radixProbeRowGroups = 4
)

// planJoinOperator plans query through the parallel planner and asserts the root
// is the parallel join operator, so its BuildStats can be inspected. Use a query
// without ORDER BY or LIMIT: those are peeled and wrapped serially, which puts a
// sort or limit at the root instead.
func planJoinOperator(t *testing.T, cat *catalog.Catalog, query string, workers int) *exec.ParallelHashJoinAggregate {
	t.Helper()
	logical := buildPlan(t, query, cat)
	op, err := planner.Parallel(context.Background(), logical, workers)
	if err != nil {
		t.Fatalf("Parallel(%q, workers=%d): %v", query, workers, err)
	}
	pj, ok := op.(*exec.ParallelHashJoinAggregate)
	if !ok {
		_ = op.Close()
		t.Fatalf("root operator = %T, want *exec.ParallelHashJoinAggregate", op)
	}
	return pj
}

// TestRadixJoin_BuildStrategySelection pins down which build strategy each plan
// gets. The strategy is invisible in query results by design, so this is the only
// place the wiring from planner estimate to exec strategy is checked directly —
// every other test here would still pass if the radix path were never taken.
func TestRadixJoin_BuildStrategySelection(t *testing.T) {
	ctx := context.Background()
	const query = `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`

	t.Run("large build side is partitioned and built in parallel", func(t *testing.T) {
		cat := writeJoinDataset(t, joinDataset{
			probeRowGroups: radixProbeRowGroups,
			buildRows:      radixBuildRows,
		})
		pj := planJoinOperator(t, cat, query, 4)
		defer pj.Close()
		if _, err := pj.Next(ctx); err != nil {
			t.Fatalf("Next: %v", err)
		}
		stats := pj.BuildStats()
		wantParts := 1 << exec.RadixBitsFor(radixBuildRows)
		if stats.Partitions != wantParts {
			t.Errorf("partitions = %d, want %d", stats.Partitions, wantParts)
		}
		if stats.Partitions <= 1 {
			t.Error("a build side above the threshold must be partitioned")
		}
		if !stats.Parallel {
			t.Error("a scan-rooted build side must use the parallel builder")
		}
		if stats.Rows != radixBuildRows {
			t.Errorf("build rows = %d, want %d", stats.Rows, radixBuildRows)
		}
		if stats.Keys != radixBuildRows {
			t.Errorf("build keys = %d, want %d (o_orderkey is unique)", stats.Keys, radixBuildRows)
		}
	})

	t.Run("small build side keeps the single-map path", func(t *testing.T) {
		cat := writeJoinDataset(t, joinDataset{probeRowGroups: 2, buildRows: 5000})
		pj := planJoinOperator(t, cat, query, 4)
		defer pj.Close()
		if _, err := pj.Next(ctx); err != nil {
			t.Fatalf("Next: %v", err)
		}
		stats := pj.BuildStats()
		if stats.Partitions != 1 {
			t.Errorf("partitions = %d, want 1 — a 5,000-row build side is below the threshold", stats.Partitions)
		}
		if stats.Parallel {
			t.Error("an unpartitioned build side must not use the parallel builder: its workers would share one map")
		}
		if stats.Rows != 5000 {
			t.Errorf("build rows = %d, want 5000", stats.Rows)
		}
	})
}

// TestRadixJoin_MatchesSerial is the correctness bar for the radix path: over a
// build side large enough to be partitioned and built in parallel, every query
// shape must return exactly what the serial planner returns, at every worker
// count. Integer aggregates are compared for exact equality, which holds because
// the parallel build preserves per-key row order — see
// BuildSharedHashTableParallel's determinism note.
func TestRadixJoin_MatchesSerial(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: radixProbeRowGroups,
		buildRows:      radixBuildRows,
		unmatchedEvery: 97, // probe keys with no build row
	})

	tests := []struct {
		name    string
		query   string
		ordered bool
	}{
		{
			// TPC-H Q12 shape over a partitioned build side.
			name: "q12 shaped case when aggregate",
			query: `SELECT l_shipmode,
			           SUM(CASE WHEN o_priority = '1-URGENT' OR o_priority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
			           SUM(CASE WHEN o_priority <> '1-URGENT' AND o_priority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			          AND l_shipmode IN ('MAIL', 'SHIP')
			        GROUP BY l_shipmode
			        ORDER BY l_shipmode`,
			ordered: true,
		},
		{
			name:    "integer aggregates must be exactly equal",
			query:   `SELECT COUNT(*) AS c, SUM(l_quantity) AS sq, MIN(o_orderkey) AS mn, MAX(o_orderkey) AS mx FROM orders, items WHERE o_orderkey = l_orderkey`,
			ordered: true,
		},
		{
			name: "float aggregates grouped by probe column",
			query: `SELECT l_shipmode, SUM(l_price) AS sp, AVG(o_total) AS ao
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY l_shipmode`,
			ordered: false,
		},
		{
			name: "grouped by build side column",
			query: `SELECT o_status, COUNT(*) AS c, SUM(l_quantity) AS sq
			        FROM orders, items
			        WHERE o_orderkey = l_orderkey
			        GROUP BY o_status`,
			ordered: false,
		},
		{
			// Every probe key misses, so the partitioned probe must return
			// nothing rather than, say, matching whatever shares a partition.
			name:    "no probe key matches any build row",
			query:   `SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_orderkey > 100000000`,
			ordered: true,
		},
		{
			// A predicate that survives in only a couple of morsels, so most
			// probe workers contribute nothing.
			name:    "selective probe predicate spanning few morsels",
			query:   `SELECT l_shipmode, COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey AND l_quantity BETWEEN 2000 AND 2050 GROUP BY l_shipmode`,
			ordered: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			serial := runSerial(t, cat, tc.query)
			for _, workers := range []int{1, 2, 4, 8} {
				got := runParallel(t, cat, tc.query, workers)
				assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, tc.ordered)
			}
		})
	}
}

// TestRadixJoin_NullJoinKeys checks that NULL keys are dropped on both sides of a
// partitioned join, as they are unpartitioned: an inner equi-join can never match
// them, and a NULL key has no partition.
func TestRadixJoin_NullJoinKeys(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: radixProbeRowGroups,
		buildRows:      radixBuildRows,
		nullKeyEvery:   7,
	})
	queries := []string{
		`SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`,
		`SELECT l_shipmode, COUNT(*) AS c, SUM(l_quantity) AS sq FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
	}
	for _, query := range queries {
		serial := runSerial(t, cat, query)
		for _, workers := range []int{1, 2, 4} {
			got := runParallel(t, cat, query, workers)
			assertSameRows(t, fmt.Sprintf("%s workers=%d", query, workers), serial, got, false)
		}
	}
}

// TestRadixJoin_AbsentProbeKeys stresses the miss path of the partitioned probe:
// half the probe rows carry a key that exists nowhere on the build side but is
// still a real key that hashes into some partition, so the probe genuinely has to
// look in that partition and find nothing. A build-and-probe hash mismatch, or a
// partition lookup that fell back to scanning a neighbour, would show up here as
// extra rows.
func TestRadixJoin_AbsentProbeKeys(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: radixProbeRowGroups,
		buildRows:      radixBuildRows,
		unmatchedEvery: 2, // every other probe row references no order
	})
	queries := []string{
		`SELECT COUNT(*) AS c FROM orders, items WHERE o_orderkey = l_orderkey`,
		`SELECT l_shipmode, COUNT(*) AS c, SUM(l_quantity) AS sq FROM orders, items WHERE o_orderkey = l_orderkey GROUP BY l_shipmode`,
	}
	for _, query := range queries {
		serial := runSerial(t, cat, query)
		for _, workers := range []int{1, 2, 4} {
			got := runParallel(t, cat, query, workers)
			assertSameRows(t, fmt.Sprintf("%s workers=%d", query, workers), serial, got, false)
		}
	}
}

// TestRadixJoin_CountDistinctFallsBack pins the one aggregate the parallel join
// planner refuses: COUNT(DISTINCT) partial counts cannot be summed at merge time.
// The plan must fall back rather than take the partitioned path, and the result
// must still match serial — a guard against a future change wiring DISTINCT
// through the parallel join now that the build side has more moving parts.
func TestRadixJoin_CountDistinctFallsBack(t *testing.T) {
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: radixProbeRowGroups,
		buildRows:      radixBuildRows,
	})
	const query = `SELECT l_shipmode, COUNT(DISTINCT o_status) AS d
	               FROM orders, items
	               WHERE o_orderkey = l_orderkey
	               GROUP BY l_shipmode`

	logical := buildPlan(t, query, cat)
	op, err := planner.Parallel(context.Background(), logical, 4)
	if err != nil {
		t.Fatalf("Parallel: %v", err)
	}
	if pj, ok := op.(*exec.ParallelHashJoinAggregate); ok {
		_ = pj.Close()
		t.Fatal("COUNT(DISTINCT) must not reach the parallel join operator")
	}
	_ = op.Close()

	serial := runSerial(t, cat, query)
	for _, workers := range []int{1, 4} {
		got := runParallel(t, cat, query, workers)
		assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, false)
	}
}

// TestRadixJoin_ThreeTableChainParallelBuild covers the second partitionable
// build shape: the build side is itself a join. Its own build side (customers) is
// materialised serially at prepare time and its probe scan (orders) supplies the
// morsels, so a three-table chain gets a parallel build too.
func TestRadixJoin_ThreeTableChainParallelBuild(t *testing.T) {
	ctx := context.Background()
	cat := writeJoinDataset(t, joinDataset{
		probeRowGroups: radixProbeRowGroups,
		buildRows:      radixBuildRows,
	})

	// customers joins to orders on an int64 key; every order has one customer.
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
	for start := 0; start < radixBuildRows; start += storage.RowGroupRows {
		n := min(storage.RowGroupRows, radixBuildRows-start)
		keys := make([]int64, n)
		segs := make([]string, n)
		for i := range n {
			keys[i] = int64(start + i + 1)
			segs[i] = []string{"BUILDING", "MACHINERY"}[(start+i)%2]
		}
		if err := w.BeginRowGroup(n); err != nil {
			t.Fatalf("BeginRowGroup(customers): %v", err)
		}
		mustAppend(t, w, 0, nil, keys)
		mustAppend(t, w, 1, nil, segs)
		if err := w.EndRowGroup(); err != nil {
			t.Fatalf("EndRowGroup(customers): %v", err)
		}
	}
	if err := w.Finish(ctx); err != nil {
		t.Fatalf("Finish(customers): %v", err)
	}
	cat.Register("customers", custPath, custSchema)

	const query = `SELECT COUNT(*) AS c, SUM(l_quantity) AS sq
	          FROM customers, orders, items
	          WHERE c_orderkey = o_orderkey AND o_orderkey = l_orderkey
	            AND c_segment = 'BUILDING'`

	pj := planJoinOperator(t, cat, query, 4)
	if _, err := pj.Next(ctx); err != nil {
		t.Fatalf("Next: %v", err)
	}
	stats := pj.BuildStats()
	_ = pj.Close()
	if !stats.Parallel {
		t.Error("a join-rooted build side whose probe side is a scan must use the parallel builder")
	}
	if stats.Partitions <= 1 {
		t.Errorf("partitions = %d, want more than 1", stats.Partitions)
	}
	// Half the customers are BUILDING, and the nested join is one-to-one.
	if want := radixBuildRows / 2; stats.Rows != want {
		t.Errorf("build rows = %d, want %d", stats.Rows, want)
	}

	serial := runSerial(t, cat, query)
	if len(serial) == 0 {
		t.Fatal("serial result is empty — the three-table query is not producing rows")
	}
	for _, workers := range []int{1, 2, 4} {
		got := runParallel(t, cat, query, workers)
		assertSameRows(t, fmt.Sprintf("workers=%d", workers), serial, got, true)
	}
}
