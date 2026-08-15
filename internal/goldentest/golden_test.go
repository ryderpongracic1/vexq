package goldentest

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// queryCase defines a single test case in the corpus.
type queryCase struct {
	name      string
	query     string
	wantError string // non-empty if we expect a planner/parse error
	ordered   bool   // true if ORDER BY is present (order-sensitive comparison)

	// knownBug documents an engine bug. If non-empty, the test is skipped.
	knownBug string
}

// TestGoldenCorrectness is the main entry point for the golden correctness suite.
// It generates a deterministic dataset, runs each query through both the engine
// and the reference evaluator, and asserts the results match.
func TestGoldenCorrectness(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	// Generate deterministic dataset.
	tables, paths := GenerateDataset(t, dir, DefaultConfig())

	// Register tables in a catalog.
	cat, err := catalog.OpenMulti(ctx, paths)
	if err != nil {
		t.Fatalf("OpenMulti: %v", err)
	}

	corpus := buildCorpus()
	t.Logf("Running %d golden test cases", len(corpus))

	for _, tc := range corpus {
		t.Run(tc.name, func(t *testing.T) {
			if tc.knownBug != "" {
				t.Skipf("KNOWN BUG: %s", tc.knownBug)
			}
			runGoldenCase(t, ctx, tc, cat, tables)
		})
	}
}

func runGoldenCase(t *testing.T, ctx context.Context, tc queryCase, cat *catalog.Catalog, tables []Table) {
	t.Helper()

	// Parse the query.
	p := sql.NewParser(tc.query)
	node, parseErr := p.ParseStatement()

	if tc.wantError != "" {
		// We expect an error from parse or plan.
		if parseErr != nil {
			if !strings.Contains(parseErr.Error(), tc.wantError) {
				t.Fatalf("expected error containing %q, got parse error: %v", tc.wantError, parseErr)
			}
			return
		}
		stmt := node.(*sql.SelectStmt)
		// Try planning — might error there.
		_, planErr := planner.Build(ctx, stmt, cat)
		if planErr != nil {
			if !strings.Contains(planErr.Error(), tc.wantError) {
				t.Fatalf("expected error containing %q, got plan error: %v", tc.wantError, planErr)
			}
			return
		}
		t.Fatalf("expected error containing %q but query succeeded", tc.wantError)
		return
	}

	if parseErr != nil {
		t.Fatalf("parse error: %v", parseErr)
	}

	stmt := node.(*sql.SelectStmt)

	// Get reference result.
	refResult, refErr := Evaluate(stmt, tables)
	if refErr != nil {
		t.Fatalf("reference evaluator error: %v", refErr)
	}

	// Execute through the engine (serial path).
	engineResult, engineErr := executeEngine(ctx, stmt, cat)
	if engineErr != nil {
		t.Fatalf("engine error: %v", engineErr)
	}

	// Compare results.
	compareResults(t, tc, refResult, engineResult)

	// Also run the plan with the optimizer switched off. Predicate pushdown
	// folds a LogicalFilter over a LogicalScan into the scan's own predicate,
	// so the optimized plan reaches exec.Filter through physicalScan and never
	// through physicalFilter — meaning the standalone LogicalFilter operator
	// path, and any plan shape where a Filter sits above an operator that has
	// already installed a selection vector, would otherwise go unoracled.
	// Optimize is an optional public step (planner.Physical accepts an
	// unoptimized plan), so both plans must produce the same answer.
	unoptResult, unoptErr := executeEngineUnoptimized(ctx, stmt, cat)
	if unoptErr != nil {
		t.Fatalf("engine error (unoptimized plan): %v", unoptErr)
	}
	compareResults(t, queryCase{
		name:    tc.name + "/unoptimized",
		ordered: tc.ordered,
	}, refResult, unoptResult)

	// Also run a plan in which every pushed-down scan predicate is ALSO applied
	// as a standalone filter above the scan. Selection is idempotent — σp(σp(R))
	// = σp(R) — so the answer must not change, while the operator tree becomes
	// Filter over Filter over TableScan: the second filter evaluates its
	// predicate against a batch that already carries a selection vector.
	//
	// No query the builder emits today produces that shape, because predicate
	// pushdown folds a LogicalFilter into the scan below it rather than leaving
	// both. Three places in the planner nonetheless construct it — physicalScan
	// plus physicalFilter, planner.Parallel's factory, and the parallel join's
	// morsel factory, each of which applies the scan predicate and then an
	// optional filter above it — so the shape is one pushdown change away from
	// being live, and the oracle should cover it before then.
	stackedResult, stackedErr := executeEngineStackedFilters(ctx, stmt, cat)
	if stackedErr != nil {
		t.Fatalf("engine error (duplicated predicate): %v", stackedErr)
	}
	if stackedResult != nil {
		compareResults(t, queryCase{
			name:    tc.name + "/stacked-filters",
			ordered: tc.ordered,
		}, refResult, stackedResult)
	}

	// Also try the parallel path if the plan shape allows it.
	parallelResult, parallelErr := executeEngineParallel(ctx, stmt, cat)
	if parallelErr == nil && parallelResult != nil {
		compareResults(t, queryCase{
			name:    tc.name + "/parallel",
			ordered: tc.ordered,
		}, refResult, parallelResult)
	}
}

// executeEngineStackedFilters runs the optimized plan with every scan predicate
// duplicated into a LogicalFilter directly above its scan. Returns nil,nil when
// the plan has no pushed-down predicate to duplicate.
func executeEngineStackedFilters(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (*RefResult, error) {
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		return nil, err
	}
	logical = planner.Optimize(logical)
	stacked, changed := duplicateScanPredicates(logical)
	if !changed {
		return nil, nil
	}
	op, err := planner.Physical(ctx, stacked)
	if err != nil {
		return nil, err
	}
	defer op.Close()

	return drainOperator(ctx, op)
}

// duplicateScanPredicates rewrites every LogicalScan that carries a pushed-down
// predicate into LogicalFilter{Predicate: scan.Predicate} over that same scan,
// reporting whether anything was rewritten. The scan keeps its predicate, so the
// filter is a redundant second application of it — semantically a no-op that
// changes only the operator shape.
func duplicateScanPredicates(node planner.LogicalNode) (planner.LogicalNode, bool) {
	switch n := node.(type) {
	case *planner.LogicalScan:
		if n.Predicate == nil {
			return n, false
		}
		return &planner.LogicalFilter{Child: n, Predicate: n.Predicate}, true
	case *planner.LogicalFilter:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalFilter{Child: child, Predicate: n.Predicate}, changed
	case *planner.LogicalProject:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalProject{Child: child, Exprs: n.Exprs}, changed
	case *planner.LogicalAggregate:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalAggregate{Child: child, GroupBy: n.GroupBy, Aggs: n.Aggs}, changed
	case *planner.LogicalSort:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalSort{Child: child, OrderBy: n.OrderBy}, changed
	case *planner.LogicalLimit:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalLimit{Child: child, Count: n.Count}, changed
	case *planner.LogicalDistinct:
		child, changed := duplicateScanPredicates(n.Child)
		return &planner.LogicalDistinct{Child: child}, changed
	case *planner.LogicalJoin:
		left, lc := duplicateScanPredicates(n.Left)
		right, rc := duplicateScanPredicates(n.Right)
		return &planner.LogicalJoin{Left: left, Right: right, Condition: n.Condition}, lc || rc
	}
	return node, false
}

// executeEngineUnoptimized runs a query through the serial engine with the
// rule-based optimizer skipped, so the logical plan keeps the shape the builder
// produced: predicates stay in LogicalFilter nodes and no column pruning is
// applied.
func executeEngineUnoptimized(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (*RefResult, error) {
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		return nil, err
	}
	op, err := planner.Physical(ctx, logical)
	if err != nil {
		return nil, err
	}
	defer op.Close()

	return drainOperator(ctx, op)
}

// executeEngine runs a query through the full serial engine pipeline.
func executeEngine(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (*RefResult, error) {
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		return nil, err
	}
	logical = planner.Optimize(logical)
	op, err := planner.Physical(ctx, logical)
	if err != nil {
		return nil, err
	}
	defer op.Close()

	return drainOperator(ctx, op)
}

// executeEngineParallel tries the parallel path. Returns nil,nil if not applicable.
func executeEngineParallel(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (*RefResult, error) {
	logical, err := planner.Build(ctx, stmt, cat)
	if err != nil {
		return nil, err
	}
	logical = planner.Optimize(logical)
	op, err := planner.Parallel(ctx, logical, 4)
	if err != nil {
		// Parallel may fall back to Physical or return an error for unsupported shapes.
		return nil, nil
	}
	defer op.Close()
	return drainOperator(ctx, op)
}

// drainOperator collects all output batches from an operator into a RefResult.
func drainOperator(ctx context.Context, op exec.Operator) (*RefResult, error) {
	schema := op.Schema()
	result := &RefResult{
		Columns: make([]string, len(schema.Fields)),
		Types:   make([]storage.DataType, len(schema.Fields)),
	}
	for i, f := range schema.Fields {
		result.Columns[i] = f.Name
		result.Types[i] = f.Type
	}

	for {
		batch, err := op.Next(ctx)
		if err != nil {
			return nil, err
		}
		if batch == nil {
			break
		}

		n := batch.Length
		var indices []int
		if batch.SelVec != nil {
			indices = make([]int, len(batch.SelVec))
			for i, v := range batch.SelVec {
				indices[i] = int(v)
			}
		} else {
			indices = make([]int, n)
			for i := range indices {
				indices[i] = i
			}
		}

		for _, rowIdx := range indices {
			row := make(Row, len(schema.Fields))
			for c, f := range schema.Fields {
				vec := batch.Vectors[c]
				if vec.IsNull(rowIdx) {
					row[c] = Value{IsNull: true}
				} else {
					switch f.Type {
					case storage.TypeInt64:
						row[c] = Value{Int64: vec.(*exec.Int64Vector).Values[rowIdx]}
					case storage.TypeFloat64:
						row[c] = Value{Float: vec.(*exec.Float64Vector).Values[rowIdx]}
					case storage.TypeString:
						sv := vec.(*exec.StringVector)
						row[c] = Value{Str: sv.Get(rowIdx)}
					case storage.TypeBool:
						bv := vec.(*exec.BoolVector)
						row[c] = Value{Bool: bv.Get(rowIdx)}
					case storage.TypeDate:
						dv := vec.(*exec.DateVector)
						row[c] = Value{Date: dv.Values[rowIdx]}
					}
				}
			}
			result.Rows = append(result.Rows, row)
		}
	}
	return result, nil
}

// compareResults checks that the engine result matches the reference result.
func compareResults(t *testing.T, tc queryCase, ref, engine *RefResult) {
	t.Helper()

	if len(ref.Rows) != len(engine.Rows) {
		t.Errorf("[%s] row count mismatch: reference=%d, engine=%d", tc.name, len(ref.Rows), len(engine.Rows))
		if len(ref.Rows) < 20 {
			t.Logf("  Reference rows: %v", formatRows(ref.Rows))
		}
		if len(engine.Rows) < 20 {
			t.Logf("  Engine rows: %v", formatRows(engine.Rows))
		}
		return
	}

	if !tc.ordered {
		// Sort both result sets for order-insensitive comparison.
		sortRows(ref.Rows)
		sortRows(engine.Rows)
	}

	for i := range ref.Rows {
		if !rowsEqual(ref.Rows[i], engine.Rows[i]) {
			t.Errorf("[%s] row %d mismatch:\n  reference: %v\n  engine:    %v",
				tc.name, i, formatRow(ref.Rows[i]), formatRow(engine.Rows[i]))
			if i >= 5 {
				t.Logf("  (showing first 5 mismatches only)")
				break
			}
		}
	}
}

func rowsEqual(a, b Row) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !valuesEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func valuesEqual(a, b Value) bool {
	if a.IsNull != b.IsNull {
		return false
	}
	if a.IsNull {
		return true
	}
	// Check all value types.
	if a.Str != "" || b.Str != "" {
		return a.Str == b.Str
	}
	if a.Bool != b.Bool {
		// Only compare bools if at least one is true (to distinguish from zero-valued).
		if a.Bool || b.Bool {
			return false
		}
	}
	if a.Date != 0 || b.Date != 0 {
		return a.Date == b.Date
	}
	// Numeric comparison: use float comparison with epsilon.
	af := valueToFloat(a)
	bf := valueToFloat(b)
	return FloatClose(af, bf)
}

func valueToFloat(v Value) float64 {
	if v.Float != 0 {
		return v.Float
	}
	return float64(v.Int64)
}

func sortRows(rows []Row) {
	sort.SliceStable(rows, func(i, j int) bool {
		for c := range rows[i] {
			if c >= len(rows[j]) {
				return false
			}
			a, b := rows[i][c], rows[j][c]
			if a.IsNull && b.IsNull {
				continue
			}
			if a.IsNull {
				return true
			}
			if b.IsNull {
				return false
			}
			// String comparison.
			if a.Str != b.Str {
				return a.Str < b.Str
			}
			af := valueToFloat(a)
			bf := valueToFloat(b)
			if af != bf {
				return af < bf
			}
		}
		return false
	})
}

func formatRow(row Row) string {
	var parts []string
	for _, v := range row {
		if v.IsNull {
			parts = append(parts, "NULL")
		} else if v.Str != "" {
			parts = append(parts, fmt.Sprintf("%q", v.Str))
		} else if v.Float != 0 {
			parts = append(parts, fmt.Sprintf("%.6f", v.Float))
		} else if v.Date != 0 {
			parts = append(parts, fmt.Sprintf("date(%d)", v.Date))
		} else {
			parts = append(parts, fmt.Sprintf("%d", v.Int64))
		}
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatRows(rows []Row) string {
	var lines []string
	for i, row := range rows {
		if i >= 10 {
			lines = append(lines, fmt.Sprintf("  ... and %d more", len(rows)-10))
			break
		}
		lines = append(lines, "    "+formatRow(row))
	}
	return "\n" + strings.Join(lines, "\n")
}

// --- Query corpus ---

func buildCorpus() []queryCase {
	return []queryCase{
		// --- Basic projections ---
		{
			name:  "select_all_orders",
			query: "SELECT order_id, customer_id, amount FROM orders LIMIT 10",
		},
		{
			name:  "select_star_limit",
			query: "SELECT order_id, amount FROM orders LIMIT 5",
		},

		// --- Filters: comparison operators ---
		{
			name:  "filter_gt",
			query: "SELECT order_id, amount FROM orders WHERE amount > 5000.0",
		},
		{
			name:  "filter_eq_string",
			query: "SELECT order_id, status FROM orders WHERE status = 'alpha'",
		},
		{
			name:  "filter_between",
			query: "SELECT order_id, amount FROM orders WHERE amount BETWEEN 100.0 AND 200.0",
		},
		{
			name:  "filter_in",
			query: "SELECT order_id, status FROM orders WHERE status IN ('alpha', 'beta', 'gamma')",
		},
		{
			name:  "filter_like",
			query: "SELECT order_id, status FROM orders WHERE status LIKE 'al%'",
		},
		{
			name:  "filter_is_null",
			query: "SELECT order_id FROM orders WHERE customer_id IS NULL",
		},
		{
			name:  "filter_is_not_null",
			query: "SELECT order_id, amount FROM orders WHERE amount IS NOT NULL AND amount > 1000.0",
		},
		{
			name:  "filter_and_or",
			query: "SELECT order_id FROM orders WHERE (status = 'alpha' OR status = 'beta') AND amount > 500.0",
		},
		{
			name:  "filter_not",
			query: "SELECT order_id FROM orders WHERE NOT status = 'alpha'",
		},

		// --- Arithmetic and CASE WHEN ---
		{
			name:  "projection_arithmetic",
			query: "SELECT order_id, amount * 1.1 AS with_tax FROM orders WHERE amount IS NOT NULL LIMIT 10",
		},
		{
			name:  "case_when",
			query: `SELECT order_id, CASE WHEN amount > 5000.0 THEN 'high' WHEN amount > 1000.0 THEN 'medium' ELSE 'low' END AS tier FROM orders WHERE amount IS NOT NULL LIMIT 10`,
		},

		// --- Aggregates: basic ---
		{
			name:  "count_star",
			query: "SELECT COUNT(*) FROM orders",
		},
		{
			name:  "count_column",
			query: "SELECT COUNT(customer_id) FROM orders",
		},
		{
			name:  "sum_amount",
			query: "SELECT SUM(amount) FROM orders",
		},
		{
			name:  "avg_amount",
			query: "SELECT AVG(amount) FROM orders",
		},
		{
			name:  "min_max",
			query: "SELECT MIN(amount), MAX(amount) FROM orders",
		},

		// --- GROUP BY ---
		{
			name:  "group_by_string",
			query: "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status",
		},
		{
			name:  "group_by_multi_agg",
			query: "SELECT status, COUNT(*), SUM(amount), AVG(amount) FROM orders GROUP BY status",
		},

		// --- HAVING ---
		{
			name:  "having_count",
			query: "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status HAVING COUNT(*) > 30",
		},
		{
			name:  "having_filters_all",
			query: "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status HAVING COUNT(*) > 9999",
		},

		// --- DISTINCT ---
		{
			name:  "distinct_status",
			query: "SELECT DISTINCT status FROM orders",
		},
		{
			name:  "distinct_with_null",
			query: "SELECT DISTINCT customer_id FROM orders LIMIT 20",
		},

		// --- ORDER BY ---
		{
			name:    "order_by_asc",
			query:   "SELECT order_id, amount FROM orders WHERE amount IS NOT NULL ORDER BY amount LIMIT 10",
			ordered: true,
		},
		{
			name:    "order_by_desc",
			query:   "SELECT order_id, amount FROM orders WHERE amount IS NOT NULL ORDER BY amount DESC LIMIT 10",
			ordered: true,
		},
		{
			name:    "order_by_nulls_first",
			query:   "SELECT order_id, customer_id FROM orders ORDER BY customer_id LIMIT 15",
			ordered: true,
		},

		// --- LIMIT edge cases ---
		{
			name:  "limit_zero",
			query: "SELECT order_id FROM orders LIMIT 0",
		},
		{
			name:  "limit_larger_than_result",
			query: "SELECT order_id FROM orders WHERE order_id > 9999 LIMIT 100",
		},

		// --- NULL-heavy aggregates ---
		{
			name:  "agg_all_null_column",
			query: "SELECT SUM(customer_id) FROM orders WHERE customer_id IS NULL",
		},
		{
			name:  "avg_with_nulls",
			query: "SELECT AVG(customer_id) FROM orders",
		},

		// --- Joins ---
		{
			name:  "inner_join_basic",
			query: "SELECT orders.order_id, items.item_id, items.price FROM orders, items WHERE orders.order_id = items.order_id LIMIT 20",
		},
		{
			name:  "join_with_filter",
			query: "SELECT orders.order_id, items.quantity FROM orders, items WHERE orders.order_id = items.order_id AND items.quantity > 10 LIMIT 15",
		},
		{
			name:  "join_aggregate",
			query: "SELECT orders.order_id, COUNT(*) AS item_count FROM orders, items WHERE orders.order_id = items.order_id GROUP BY orders.order_id HAVING COUNT(*) > 3",
		},

		// --- Error cases ---
		{
			name:      "error_ambiguous_column",
			query:     "SELECT order_id FROM orders, items",
			wantError: "cross join", // Engine rejects this as cross join (no join condition) before reaching ambiguous column detection
		},
		{
			name:      "error_cross_join",
			query:     "SELECT orders.order_id, items.item_id FROM orders, items WHERE orders.amount > 100",
			wantError: "cross join",
		},
		{
			name:      "error_unknown_table",
			query:     "SELECT * FROM nonexistent",
			wantError: "not found",
		},

		// --- Empty result ---
		{
			name:  "empty_result",
			query: "SELECT order_id FROM orders WHERE order_id > 999999",
		},

		// --- Complex: TPC-H Q1-shaped ---
		{
			name:  "tpch_q1_shaped",
			query: "SELECT status, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM orders WHERE order_date > 18000 GROUP BY status",
		},

		// --- Complex: TPC-H Q6-shaped ---
		{
			name:  "tpch_q6_shaped",
			query: "SELECT SUM(amount) FROM orders WHERE order_date BETWEEN 16000 AND 17000 AND amount > 100.0",
		},

		// --- Duplicate rows + DISTINCT ---
		{
			name:  "distinct_duplicates",
			query: "SELECT DISTINCT category FROM items",
		},

		// --- COUNT(DISTINCT) ---
		{
			name:  "count_distinct",
			query: "SELECT COUNT(DISTINCT status) FROM orders",
		},

		// --- Multi-key ORDER BY ---
		{
			name:    "order_by_multi_key",
			query:   "SELECT status, amount FROM orders WHERE amount IS NOT NULL AND status IS NOT NULL ORDER BY status, amount DESC LIMIT 15",
			ordered: true,
		},

		// --- Unary minus ---
		{
			name:  "unary_minus",
			query: "SELECT order_id, -amount AS neg_amount FROM orders WHERE amount IS NOT NULL LIMIT 5",
		},

		// --- NOT BETWEEN ---
		{
			name:  "not_between",
			query: "SELECT order_id FROM orders WHERE amount NOT BETWEEN 100.0 AND 200.0 AND amount IS NOT NULL",
		},

		// --- NOT IN ---
		{
			name:  "not_in",
			query: "SELECT order_id, status FROM orders WHERE status NOT IN ('alpha', 'beta') AND status IS NOT NULL LIMIT 10",
		},

		// --- NOT LIKE ---
		{
			name:  "not_like",
			query: "SELECT order_id, status FROM orders WHERE status NOT LIKE '%a' AND status IS NOT NULL LIMIT 10",
		},

		// --- Aliased table reference ---
		{
			name:  "alias_table",
			query: "SELECT o.order_id, o.amount FROM orders o WHERE o.amount > 5000.0 LIMIT 10",
		},

		// --- Expressions evaluated above a selection vector -----------------
		//
		// Everything in this group puts a literal-bearing expression above a
		// filter, which is the shape that exposed the sizing mismatch between
		// expression leaves (which used to size from the post-filter logical
		// length) and comparison kernels (which size from the physical vector
		// length). See the sizing convention on exec.Expr.
		//
		// Selectivity matters here: the predicates deliberately leave a
		// partially-filled batch, so the physical length exceeds the logical
		// one and a short vector would null-mask real rows rather than land on
		// a batch boundary where the two lengths coincide.
		{
			name:  "literal_column_over_filter",
			query: "SELECT order_id, 1 AS one, 'tag' AS label FROM orders WHERE amount > 5000.0",
		},
		{
			name:  "arith_literal_over_filter",
			query: "SELECT order_id, amount * 2.0 + 1.0 AS scaled FROM orders WHERE amount > 5000.0",
		},
		{
			name:  "mixed_type_arith_over_filter",
			query: "SELECT orders.order_id, amount * quantity AS total FROM orders, items WHERE orders.order_id = items.order_id AND amount > 5000.0",
		},
		{
			name:  "case_over_filter",
			query: `SELECT order_id, CASE WHEN amount > 8000.0 THEN 'high' ELSE 'low' END AS tier FROM orders WHERE amount > 5000.0`,
		},
		{
			name:  "case_no_else_over_filter",
			query: `SELECT order_id, CASE WHEN amount > 8000.0 THEN 'high' END AS tier FROM orders WHERE amount > 5000.0`,
		},
		{
			name:  "between_and_literal_over_filter",
			query: "SELECT order_id, amount, 0 AS pad FROM orders WHERE amount BETWEEN 1000.0 AND 8000.0 AND status = 'alpha'",
		},

		// --- Conjunctions that a partial pushdown would split across filters -
		//
		// Each of these is one WHERE with several terms over one table. Today
		// the optimizer folds them all into a single scan predicate, so they
		// execute as one Filter with an AndExpr; with the optimizer off they
		// execute as one Filter over a raw scan. Either way the answer must
		// match the reference, and if pushdown ever becomes partial — pushing
		// the zone-mappable terms and leaving the rest above — these become
		// genuine stacked filters without needing new cases.
		{
			name:  "multi_term_conjunction",
			query: "SELECT order_id, amount FROM orders WHERE amount > 1000.0 AND amount < 8000.0 AND status = 'alpha' AND customer_id > 5",
		},
		{
			name:  "multi_term_conjunction_with_expr",
			query: "SELECT order_id, amount * 1.5 AS scaled, 7 AS seven FROM orders WHERE amount > 1000.0 AND amount < 8000.0 AND order_date > 16500",
		},
		{
			name:  "disjunction_over_filter",
			query: "SELECT order_id, amount FROM orders WHERE (amount > 8000.0 OR amount < 200.0) AND amount IS NOT NULL",
		},

		// --- Expressions above a HAVING filter -------------------------------
		//
		// HAVING is the one shape where the builder leaves a standalone
		// LogicalFilter that pushdown cannot fold into a scan, and a hidden
		// aggregate puts a Project directly above that filter's selection
		// vector.
		{
			name:  "having_hidden_agg_projection",
			query: "SELECT status FROM orders GROUP BY status HAVING COUNT(*) > 30",
		},
		{
			name:    "having_with_literal_projection",
			query:   "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status HAVING COUNT(*) > 30 ORDER BY status",
			ordered: true,
		},
	}
}

// Ensure we never accidentally expose math in the interface.
var _ = math.Abs
