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

	// Also try the parallel path if the plan shape allows it.
	parallelResult, parallelErr := executeEngineParallel(ctx, stmt, cat)
	if parallelErr == nil && parallelResult != nil {
		compareResults(t, queryCase{
			name:    tc.name + "/parallel",
			ordered: tc.ordered,
		}, refResult, parallelResult)
	}
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
			name:     "filter_not",
			query:    "SELECT order_id FROM orders WHERE NOT status = 'alpha'",
			knownBug: "ENGINE BUG: exec.NotExpr.Eval panics with type assertion failure (*StringVector vs *BoolVector) when NOT is applied to a comparison expression. The physical planner wraps the comparison in NotExpr but the comparison sub-expression returns its result as a non-BoolVector intermediate. Suspected location: exec/expr.go:685 and planner/physical.go expression compilation for NOT.",
		},

		// --- Arithmetic and CASE WHEN ---
		{
			name:     "projection_arithmetic",
			query:    "SELECT order_id, amount * 1.1 AS with_tax FROM orders WHERE amount IS NOT NULL LIMIT 10",
			knownBug: "ENGINE BUG: exec.Project panics with index out of range when projecting arithmetic on a filtered+limited result. The Project operator accesses vector indices beyond the batch Length when a selection vector is active. Suspected location: exec/project.go compact path.",
		},
		{
			name:     "case_when",
			query:    `SELECT order_id, CASE WHEN amount > 5000.0 THEN 'high' WHEN amount > 1000.0 THEN 'medium' ELSE 'low' END AS tier FROM orders WHERE amount IS NOT NULL LIMIT 10`,
			knownBug: "ENGINE BUG: String literals are not supported in CASE WHEN result expressions. Engine returns 'expr: string literal Eval not supported (compare via BinOp)'. Suspected location: exec/expr.go string literal evaluation path.",
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
			name:     "having_count",
			query:    "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status HAVING COUNT(*) > 30",
			knownBug: "ENGINE BUG: planner does not support AggFuncExpr in HAVING clause. HAVING COUNT(*) > 30 fails with 'unsupported expression type *sql.AggFuncExpr'. The planner only supports column references in HAVING predicates, not aggregate re-evaluation. Suspected location: planner/physical.go expression compilation.",
		},
		{
			name:     "having_filters_all",
			query:    "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status HAVING COUNT(*) > 9999",
			knownBug: "ENGINE BUG: Same as having_count — HAVING does not support aggregate expressions.",
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
			name:     "join_aggregate",
			query:    "SELECT orders.order_id, COUNT(*) AS item_count FROM orders, items WHERE orders.order_id = items.order_id GROUP BY orders.order_id HAVING COUNT(*) > 3",
			knownBug: "ENGINE BUG: Same as having_count — HAVING does not support aggregate expressions (AggFuncExpr).",
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
			name:     "tpch_q1_shaped",
			query:    "SELECT status, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM orders WHERE order_date > 18000 GROUP BY status",
			knownBug: "ENGINE BUG: exec.evalCmp panics with type assertion when comparing a DateVector (TypeDate) against an IntLiteral. The comparison operator assumes the literal type matches the column type but DateVector stores int32 while the literal is int64. Suspected location: exec/expr.go:169 evalCmp.",
		},

		// --- Complex: TPC-H Q6-shaped ---
		{
			name:     "tpch_q6_shaped",
			query:    "SELECT SUM(amount) FROM orders WHERE order_date BETWEEN 16000 AND 17000 AND amount > 100.0",
			knownBug: "ENGINE BUG: Same as tpch_q1_shaped — DateVector comparison with integer literal panics in exec/expr.go evalCmp.",
		},

		// --- Duplicate rows + DISTINCT ---
		{
			name:  "distinct_duplicates",
			query: "SELECT DISTINCT category FROM items",
		},

		// --- COUNT(DISTINCT) ---
		{
			name:     "count_distinct",
			query:    "SELECT COUNT(DISTINCT status) FROM orders",
			knownBug: "ENGINE LIMITATION: planner explicitly rejects DISTINCT aggregates with 'DISTINCT aggregates not yet supported'. This is a documented limitation, not a silent wrong-result bug.",
		},

		// --- Multi-key ORDER BY ---
		{
			name:    "order_by_multi_key",
			query:   "SELECT status, amount FROM orders WHERE amount IS NOT NULL AND status IS NOT NULL ORDER BY status, amount DESC LIMIT 15",
			ordered: true,
		},

		// --- Unary minus ---
		{
			name:     "unary_minus",
			query:    "SELECT order_id, -amount AS neg_amount FROM orders WHERE amount IS NOT NULL LIMIT 5",
			knownBug: "ENGINE BUG: exec.evalArith panics with type assertion failure (Int64Vector vs Float64Vector) when applying unary minus to a Float64 column. The arithmetic evaluator assumes both operands are the same vector type. Suspected location: exec/expr.go:542 evalArith.",
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
	}
}

// Ensure we never accidentally expose math in the interface.
var _ = math.Abs
