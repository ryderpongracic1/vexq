package planner_test

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// semanticsCatalog writes two small tables that share a column name and returns
// a catalog over them.
//
//	a: ax  shared  name       fx    nx    flag       day  other_day
//	    1      10  apple     1.5     1    true         0         10
//	    2      20  banana   -2.5  NULL   false    106751         20
//	    3      30  cherry    3.0     3    true    2932896         30
//	   -2      40  apricot  -0.5  NULL   false        40         40
//
//	b: bx  shared  bname
//	    1     100  x
//	    2     999  y
//	    3     300  z
//	    5     500  w
func semanticsCatalog(t *testing.T) *catalog.Catalog {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()

	aPath := filepath.Join(dir, "a.vxq")
	w, err := storage.NewWriter(aPath, storage.Schema{Fields: []storage.Field{
		{Name: "ax", Type: storage.TypeInt64},
		{Name: "shared", Type: storage.TypeInt64},
		{Name: "name", Type: storage.TypeString},
		{Name: "fx", Type: storage.TypeFloat64},
		{Name: "nx", Type: storage.TypeInt64, Nullable: true},
		{Name: "flag", Type: storage.TypeBool},
		{Name: "day", Type: storage.TypeDate},
		{Name: "other_day", Type: storage.TypeDate},
	}})
	if err != nil {
		t.Fatal(err)
	}
	nxNulls := storage.FullBitmap(4)
	storage.SetNullBit(nxNulls, 1)
	storage.SetNullBit(nxNulls, 3)
	mustOK(t, w.BeginRowGroup(4))
	mustOK(t, w.AppendColumn(ctx, 0, nil, []int64{1, 2, 3, -2}))
	mustOK(t, w.AppendColumn(ctx, 1, nil, []int64{10, 20, 30, 40}))
	mustOK(t, w.AppendColumn(ctx, 2, nil, []string{"apple", "banana", "cherry", "apricot"}))
	mustOK(t, w.AppendColumn(ctx, 3, nil, []float64{1.5, -2.5, 3.0, -0.5}))
	mustOK(t, w.AppendColumn(ctx, 4, nxNulls, []int64{1, 0, 3, 0}))
	mustOK(t, w.AppendColumn(ctx, 5, nil, []bool{true, false, true, false}))
	mustOK(t, w.AppendColumn(ctx, 6, nil, []int32{0, 106751, 2932896, 40}))
	mustOK(t, w.AppendColumn(ctx, 7, nil, []int32{10, 20, 30, 40}))
	mustOK(t, w.EndRowGroup())
	mustOK(t, w.Finish(ctx))

	bPath := filepath.Join(dir, "b.vxq")
	w, err = storage.NewWriter(bPath, storage.Schema{Fields: []storage.Field{
		{Name: "bx", Type: storage.TypeInt64},
		{Name: "shared", Type: storage.TypeInt64},
		{Name: "bname", Type: storage.TypeString},
	}})
	if err != nil {
		t.Fatal(err)
	}
	mustOK(t, w.BeginRowGroup(4))
	mustOK(t, w.AppendColumn(ctx, 0, nil, []int64{1, 2, 3, 5}))
	mustOK(t, w.AppendColumn(ctx, 1, nil, []int64{100, 999, 300, 500}))
	mustOK(t, w.AppendColumn(ctx, 2, nil, []string{"x", "y", "z", "w"}))
	mustOK(t, w.EndRowGroup())
	mustOK(t, w.Finish(ctx))

	cat, err := catalog.OpenMulti(ctx, map[string]string{"a": aPath, "b": bPath})
	if err != nil {
		t.Fatal(err)
	}
	return cat
}

func mustOK(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// queryResult is a query's output rendered as strings: column names, then one
// string slice per row.
type queryResult struct {
	cols []string
	rows [][]string
}

// runSemantics plans query and executes it through the serial planner and, for
// the same optimized plan, through planner.Parallel. It fails the test if the two
// disagree, and returns the serial result or the planning/execution error.
func runSemantics(t *testing.T, cat *catalog.Catalog, query string) (queryResult, error) {
	t.Helper()
	ctx := context.Background()
	node, err := sql.NewParser(query).ParseStatement()
	if err != nil {
		return queryResult{}, err
	}
	logical, err := planner.Build(ctx, node.(*sql.SelectStmt), cat)
	if err != nil {
		return queryResult{}, err
	}
	logical = planner.Optimize(logical)

	serialOp, err := planner.Physical(ctx, logical)
	if err != nil {
		return queryResult{}, err
	}
	serial, err := drainStrings(ctx, serialOp)
	if err != nil {
		return queryResult{}, err
	}

	parallelOp, err := planner.Parallel(ctx, logical, 4)
	if err != nil {
		t.Fatalf("%s: parallel planning failed where serial succeeded: %v", query, err)
	}
	par, err := drainStrings(ctx, parallelOp)
	if err != nil {
		t.Fatalf("%s: parallel execution failed where serial succeeded: %v", query, err)
	}
	if fmt.Sprint(serial.cols) != fmt.Sprint(par.cols) {
		t.Fatalf("%s: parallel columns %v, serial %v", query, par.cols, serial.cols)
	}
	if fmt.Sprint(sortedRows(serial.rows)) != fmt.Sprint(sortedRows(par.rows)) {
		t.Fatalf("%s: parallel rows %v, serial %v", query, par.rows, serial.rows)
	}
	return serial, nil
}

func drainStrings(ctx context.Context, op exec.Operator) (queryResult, error) {
	defer op.Close()
	var res queryResult
	for _, f := range op.Schema().Fields {
		res.cols = append(res.cols, f.Name)
	}
	for {
		batch, err := op.Next(ctx)
		if err != nil {
			return queryResult{}, err
		}
		if batch == nil {
			return res, nil
		}
		for i := 0; i < batch.Length; i++ {
			idx := i
			if batch.SelVec != nil {
				idx = int(batch.SelVec[i])
			}
			row := make([]string, len(batch.Vectors))
			for c, v := range batch.Vectors {
				row[c] = vectorString(v, idx)
			}
			res.rows = append(res.rows, row)
		}
	}
}

func vectorString(v exec.Vector, i int) string {
	if v.IsNull(i) {
		return "NULL"
	}
	switch vec := v.(type) {
	case *exec.Int64Vector:
		return fmt.Sprint(vec.Values[i])
	case *exec.Float64Vector:
		return fmt.Sprint(vec.Values[i])
	case *exec.StringVector:
		return vec.Get(i)
	case *exec.BoolVector:
		return fmt.Sprint(vec.Get(i))
	case *exec.DateVector:
		return time.Unix(int64(vec.Values[i])*86400, 0).UTC().Format("2006-01-02")
	}
	return "?"
}

func sortedRows(rows [][]string) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		out[i] = strings.Join(r, "|")
	}
	sort.Strings(out)
	return out
}

type semanticsCase struct {
	name    string
	query   string
	cols    []string // nil = don't check
	rows    []string // "|"-joined rows; compared as a sorted set unless ordered
	ordered bool
	wantErr string
}

func runSemanticsCases(t *testing.T, cases []semanticsCase) {
	t.Helper()
	cat := semanticsCatalog(t)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := runSemantics(t, cat, tc.query)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("%s: expected error containing %q, got rows %v", tc.query, tc.wantErr, res.rows)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("%s: expected error containing %q, got %v", tc.query, tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("%s: %v", tc.query, err)
			}
			if tc.cols != nil && fmt.Sprint(res.cols) != fmt.Sprint(tc.cols) {
				t.Errorf("%s: columns = %v, want %v", tc.query, res.cols, tc.cols)
			}
			got := make([]string, len(res.rows))
			for i, r := range res.rows {
				got[i] = strings.Join(r, "|")
			}
			want := append([]string{}, tc.rows...)
			if !tc.ordered {
				sort.Strings(got)
				sort.Strings(want)
			}
			if fmt.Sprint(got) != fmt.Sprint(want) {
				t.Errorf("%s:\n  rows = %v\n  want = %v", tc.query, got, want)
			}
		})
	}
}

func TestJoinPredicatesAreNeverDropped(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "constant_false_term",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND 1 = 0",
			rows:  nil,
		},
		{
			name:  "second_column_equality",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND a.nx = b.bx",
			rows:  []string{"1", "3"},
		},
		{
			name:  "equality_over_expression",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND a.shared * 10 = b.shared",
			rows:  []string{"1", "3"},
		},
		{
			name:  "reversed_duplicate_equality",
			query: "SELECT a.ax, b.bx FROM a, b WHERE a.ax = b.bx AND b.shared = a.shared",
			rows:  nil,
		},
		{
			name:  "like_on_one_table",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND a.name LIKE 'b%'",
			rows:  []string{"2"},
		},
		{
			name:  "not_like_on_one_table",
			query: "SELECT b.bname FROM a, b WHERE a.ax = b.bx AND b.bname NOT LIKE 'y'",
			rows:  []string{"x", "z"},
		},
		{
			name:  "cross_table_inequality",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND a.shared * 20 > b.shared",
			rows:  []string{"1", "3"},
		},
		{
			name:  "cross_table_or",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND (a.ax = 1 OR b.bname = 'z')",
			rows:  []string{"1", "3"},
		},
		{
			name:  "aggregate_over_join_with_extra_terms",
			query: "SELECT COUNT(*) FROM a, b WHERE a.ax = b.bx AND a.name LIKE 'b%' AND 1 = 1",
			rows:  []string{"1"},
		},
		{
			name:  "aggregate_over_join_constant_false",
			query: "SELECT COUNT(*) FROM a, b WHERE a.ax = b.bx AND 1 = 0",
			rows:  []string{"0"},
		},
	})
}

func TestQualifiedColumnsResolveToTheirTable(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "same_name_both_tables",
			query: "SELECT a.shared, b.shared FROM a, b WHERE a.ax = b.bx",
			cols:  []string{"shared", "shared"},
			rows:  []string{"10|100", "20|999", "30|300"},
		},
		{
			name:  "same_name_filter_on_second_table",
			query: "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND b.shared > 200",
			rows:  []string{"2", "3"},
		},
		{
			name:  "aliases",
			query: "SELECT x.shared AS s1, y.shared AS s2 FROM a x, b y WHERE x.ax = y.bx AND y.shared < 500",
			cols:  []string{"s1", "s2"},
			rows:  []string{"10|100", "30|300"},
		},
		{
			name:  "aggregate_qualified",
			query: "SELECT SUM(a.shared), SUM(b.shared) FROM a, b WHERE a.ax = b.bx",
			rows:  []string{"60|1399"},
		},
		{
			name:  "group_by_qualified",
			query: "SELECT b.shared, COUNT(*) AS n FROM a, b WHERE a.ax = b.bx GROUP BY b.shared",
			cols:  []string{"shared", "n"},
			rows:  []string{"100|1", "999|1", "300|1"},
		},
		{
			name:    "order_by_qualified_unselected",
			query:   "SELECT a.ax FROM a, b WHERE a.ax = b.bx ORDER BY b.shared DESC",
			rows:    []string{"2", "3", "1"},
			ordered: true,
		},
		{
			name:  "self_join",
			query: "SELECT l.ax, r.shared FROM a l, a r WHERE l.ax = r.ax AND l.ax > 1",
			rows:  []string{"2|20", "3|30"},
		},
		{
			name:    "ambiguous_in_select",
			query:   "SELECT shared FROM a, b WHERE a.ax = b.bx",
			wantErr: "ambiguous",
		},
		{
			name:    "ambiguous_in_group_by",
			query:   "SELECT COUNT(*) FROM a, b WHERE a.ax = b.bx GROUP BY shared",
			wantErr: "ambiguous",
		},
		{
			name:    "ambiguous_in_aggregate",
			query:   "SELECT SUM(shared) FROM a, b WHERE a.ax = b.bx",
			wantErr: "ambiguous",
		},
		{
			name:    "unknown_qualifier_single_table",
			query:   "SELECT zz.ax FROM a",
			wantErr: "not found",
		},
		{
			name:    "string_join_key_is_not_hashed_as_int",
			query:   "SELECT a.ax FROM a, b WHERE a.name = b.bname",
			wantErr: "join",
		},
	})
}

func TestInListSemantics(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "negative_literal",
			query: "SELECT ax FROM a WHERE ax IN (-2, 3)",
			rows:  []string{"-2", "3"},
		},
		{
			name:  "negative_float_literal",
			query: "SELECT ax FROM a WHERE fx IN (-2.5, -0.5)",
			rows:  []string{"2", "-2"},
		},
		{
			name:  "not_in_with_null_matches_nothing",
			query: "SELECT ax FROM a WHERE ax NOT IN (2, NULL)",
			rows:  nil,
		},
		{
			name:  "in_with_null_still_matches",
			query: "SELECT ax FROM a WHERE ax IN (2, NULL)",
			rows:  []string{"2"},
		},
		{
			name:  "not_in_plain",
			query: "SELECT ax FROM a WHERE ax NOT IN (2, -2)",
			rows:  []string{"1", "3"},
		},
		{
			name:  "not_in_null_column",
			query: "SELECT ax FROM a WHERE nx NOT IN (1)",
			rows:  []string{"3"},
		},
		{
			name:  "int_column_float_literal",
			query: "SELECT ax FROM a WHERE ax IN (2.0)",
			rows:  []string{"2"},
		},
		{
			name:  "int_column_fractional_literal",
			query: "SELECT ax FROM a WHERE ax IN (2.5)",
			rows:  nil,
		},
		{
			name:  "float_column_int_literal",
			query: "SELECT ax FROM a WHERE fx IN (3)",
			rows:  []string{"3"},
		},
		{
			name:    "non_literal_item",
			query:   "SELECT ax FROM a WHERE ax IN (shared, 1)",
			wantErr: "IN",
		},
		{
			name:    "string_column_int_literal",
			query:   "SELECT ax FROM a WHERE name IN (1)",
			wantErr: "IN",
		},
	})
}

func TestComparisonCoercion(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "int_ge_fractional",
			query: "SELECT ax FROM a WHERE ax >= 2.5",
			rows:  []string{"3"},
		},
		{
			name:  "int_lt_fractional",
			query: "SELECT ax FROM a WHERE ax < 1.5",
			rows:  []string{"1", "-2"},
		},
		{
			name:  "int_eq_fractional",
			query: "SELECT ax FROM a WHERE ax = 2.5",
			rows:  nil,
		},
		{
			name:  "float_negative_range",
			query: "SELECT ax FROM a WHERE fx < -1.0",
			rows:  []string{"2"},
		},
		{
			name:  "float_negative_between",
			query: "SELECT ax FROM a WHERE fx BETWEEN -3.0 AND -1.0",
			rows:  []string{"2"},
		},
		{
			name:  "int_between_fractional",
			query: "SELECT ax FROM a WHERE ax BETWEEN 1.5 AND 3.5",
			rows:  []string{"2", "3"},
		},
		{
			name:  "not_over_null_and_false",
			query: "SELECT ax FROM a WHERE NOT (nx = 1 AND ax = 5)",
			rows:  []string{"1", "2", "3", "-2"},
		},
		{
			name:  "not_over_null_or_false",
			query: "SELECT ax FROM a WHERE NOT (nx = 1 OR ax = 5)",
			rows:  []string{"3"},
		},
	})
}

func TestCaseSupportsAllResultTypesAndRejectsNonBooleanConditions(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:    "boolean_result",
			query:   "SELECT ax, CASE WHEN ax = 1 THEN TRUE ELSE FALSE END AS picked FROM a ORDER BY ax",
			rows:    []string{"-2|false", "1|true", "2|false", "3|false"},
			ordered: true,
		},
		{
			name:    "date_result",
			query:   "SELECT ax, CASE WHEN ax = 1 THEN day ELSE other_day END AS picked FROM a ORDER BY ax",
			rows:    []string{"-2|1970-02-10", "1|1970-01-01", "2|1970-01-21", "3|1970-01-31"},
			ordered: true,
		},
		{
			name:    "date_result_without_else",
			query:   "SELECT ax, CASE WHEN ax = 1 THEN day END AS picked FROM a ORDER BY ax",
			rows:    []string{"-2|NULL", "1|1970-01-01", "2|NULL", "3|NULL"},
			ordered: true,
		},
		{
			name:    "non_boolean_condition",
			query:   "SELECT CASE WHEN ax THEN 1 ELSE 0 END FROM a",
			wantErr: "CASE WHEN condition must be BOOL",
		},
	})
}

func TestSumAndAvgRejectNonNumericArguments(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{name: "sum_string", query: "SELECT SUM(name) FROM a", wantErr: "SUM requires a numeric argument"},
		{name: "avg_string", query: "SELECT AVG(name) FROM a", wantErr: "AVG requires a numeric argument"},
		{name: "sum_bool", query: "SELECT SUM(flag) FROM a", wantErr: "SUM requires a numeric argument"},
		{name: "avg_date", query: "SELECT AVG(day) FROM a", wantErr: "AVG requires a numeric argument"},
	})
}

func TestDateLiteralConversionIsExact(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "distant_date_comparison",
			query: "SELECT ax FROM a WHERE day = '9999-12-31'",
			rows:  []string{"3"},
		},
		{
			name:  "distant_date_in_list",
			query: "SELECT ax FROM a WHERE day IN ('9999-12-31')",
			rows:  []string{"3"},
		},
		{
			name:  "distant_date_between",
			query: "SELECT ax FROM a WHERE day BETWEEN '9999-12-31' AND '9999-12-31'",
			rows:  []string{"3"},
		},
		{
			name:    "integer_day_overflow_comparison",
			query:   "SELECT ax FROM a WHERE day = 4294967296",
			wantErr: "cannot compare",
		},
		{
			name:    "integer_day_overflow_in_list",
			query:   "SELECT ax FROM a WHERE day IN (4294967296)",
			wantErr: "outside the DATE day range",
		},
	})

	t.Run("distant_date_zone_map", func(t *testing.T) {
		ctx := context.Background()
		path := filepath.Join(t.TempDir(), "dates.vxq")
		w, err := storage.NewWriter(path, storage.Schema{Fields: []storage.Field{
			{Name: "id", Type: storage.TypeInt64},
			{Name: "day", Type: storage.TypeDate},
		}})
		if err != nil {
			t.Fatal(err)
		}
		mustOK(t, w.BeginRowGroup(1))
		mustOK(t, w.AppendColumn(ctx, 0, nil, []int64{1}))
		mustOK(t, w.AppendColumn(ctx, 1, nil, []int32{106751}))
		mustOK(t, w.EndRowGroup())
		mustOK(t, w.BeginRowGroup(1))
		mustOK(t, w.AppendColumn(ctx, 0, nil, []int64{2}))
		mustOK(t, w.AppendColumn(ctx, 1, nil, []int32{2932896}))
		mustOK(t, w.EndRowGroup())
		mustOK(t, w.Finish(ctx))

		cat, err := catalog.OpenSingle(ctx, "dates", path)
		if err != nil {
			t.Fatal(err)
		}
		got, err := runSemantics(t, cat, "SELECT id FROM dates WHERE day = '9999-12-31'")
		if err != nil {
			t.Fatal(err)
		}
		if want := [][]string{{"2"}}; fmt.Sprint(got.rows) != fmt.Sprint(want) {
			t.Fatalf("rows = %v, want %v", got.rows, want)
		}
	})
}

func TestDivisionByZeroIsNull(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "int_by_zero",
			query: "SELECT ax, shared / (ax - 1) AS q FROM a",
			rows:  []string{"1|NULL", "2|20", "3|15", "-2|-13"},
		},
		{
			name:  "float_by_zero",
			query: "SELECT ax, fx / (ax - 1) AS q FROM a WHERE ax < 3",
			rows:  []string{"1|NULL", "2|-2.5", "-2|0.16666666666666666"},
		},
		{
			name:  "null_quotient_is_not_filtered_as_zero",
			query: "SELECT ax FROM a WHERE shared / (ax - 1) = 0",
			rows:  nil,
		},
		{
			name:  "sum_skips_null_quotient",
			query: "SELECT SUM(shared / (ax - 1)) AS s, COUNT(shared / (ax - 1)) AS n FROM a",
			rows:  []string{"22|3"},
		},
	})
}

func TestAggregateOutputShape(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "no_extra_group_column",
			query: "SELECT COUNT(*) FROM a GROUP BY ax",
			cols:  []string{"COUNT_*"},
			rows:  []string{"1", "1", "1", "1"},
		},
		{
			name:  "column_order_preserved",
			query: "SELECT COUNT(*) AS c, ax FROM a WHERE ax > 0 GROUP BY ax",
			cols:  []string{"c", "ax"},
			rows:  []string{"1|1", "1|2", "1|3"},
		},
		{
			name:  "group_column_alias",
			query: "SELECT ax AS k, SUM(shared) AS s FROM a GROUP BY ax",
			cols:  []string{"k", "s"},
			rows:  []string{"1|10", "2|20", "3|30", "-2|40"},
		},
		{
			name:  "expression_over_aggregate",
			query: "SELECT SUM(ax) + 1 AS s FROM a",
			cols:  []string{"s"},
			rows:  []string{"5"},
		},
		{
			name:  "expression_mixing_group_and_aggregate",
			query: "SELECT ax * 100 + COUNT(*) AS v FROM a GROUP BY ax",
			rows:  []string{"101", "201", "301", "-199"},
		},
		{
			name:  "duplicate_aggregates",
			query: "SELECT COUNT(*), COUNT(*), SUM(ax) FROM a",
			rows:  []string{"4|4|4"},
		},
		{
			name:    "order_by_alias_of_expression",
			query:   "SELECT ax, SUM(shared) * 2 AS d FROM a GROUP BY ax ORDER BY d DESC LIMIT 2",
			rows:    []string{"-2|80", "3|60"},
			ordered: true,
		},
		{
			name:    "order_by_unselected_aggregate",
			query:   "SELECT ax FROM a GROUP BY ax ORDER BY SUM(shared) DESC",
			rows:    []string{"-2", "3", "2", "1"},
			ordered: true,
		},
		{
			name:    "order_by_unselected_group_column",
			query:   "SELECT COUNT(*) AS n FROM a GROUP BY ax ORDER BY ax",
			cols:    []string{"n"},
			rows:    []string{"1", "1", "1", "1"},
			ordered: true,
		},
		{
			name:    "ungrouped_column",
			query:   "SELECT shared, COUNT(*) FROM a GROUP BY ax",
			wantErr: "GROUP BY",
		},
		{
			name:    "aggregate_in_where",
			query:   "SELECT ax FROM a WHERE COUNT(*) > 1",
			wantErr: "aggregate",
		},
	})
}

func TestHavingIsNeverIgnored(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:  "having_on_group_column",
			query: "SELECT ax, COUNT(*) FROM a GROUP BY ax HAVING ax > 1",
			rows:  []string{"2|1", "3|1"},
		},
		{
			name:  "having_aggregate_inside_case",
			query: "SELECT ax FROM a GROUP BY ax HAVING CASE WHEN SUM(shared) > 25 THEN 1 ELSE 0 END = 1",
			rows:  []string{"3", "-2"},
		},
		{
			name:  "having_aggregate_inside_in",
			query: "SELECT ax FROM a GROUP BY ax HAVING SUM(shared) IN (10, 40)",
			rows:  []string{"1", "-2"},
		},
		{
			name:  "having_aggregate_inside_not",
			query: "SELECT ax FROM a GROUP BY ax HAVING NOT SUM(shared) > 15",
			rows:  []string{"1"},
		},
		{
			name:  "having_alias_of_group_column",
			query: "SELECT ax AS k, COUNT(*) AS n FROM a GROUP BY ax HAVING k < 2",
			rows:  []string{"1|1", "-2|1"},
		},
		{
			name:  "having_without_group_by",
			query: "SELECT COUNT(*) FROM a HAVING COUNT(*) > 10",
			rows:  nil,
		},
		{
			name:    "having_with_order_and_limit",
			query:   "SELECT ax, SUM(shared) AS s FROM a GROUP BY ax HAVING SUM(shared) > 15 ORDER BY s LIMIT 2",
			rows:    []string{"2|20", "3|30"},
			ordered: true,
		},
		{
			name:    "having_without_aggregation",
			query:   "SELECT ax FROM a HAVING ax > 1",
			wantErr: "HAVING",
		},
		{
			name:    "having_ungrouped_column",
			query:   "SELECT ax, COUNT(*) FROM a GROUP BY ax HAVING shared > 1",
			wantErr: "GROUP BY",
		},
	})
}

func TestStatementTailIsNotIgnored(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:    "offset",
			query:   "SELECT ax FROM a ORDER BY ax LIMIT 2 OFFSET 1",
			rows:    []string{"1", "2"},
			ordered: true,
		},
		{
			name:    "offset_past_end",
			query:   "SELECT ax FROM a ORDER BY ax LIMIT 2 OFFSET 10",
			rows:    nil,
			ordered: true,
		},
		{
			name:    "offset_without_limit",
			query:   "SELECT ax FROM a ORDER BY ax OFFSET 3",
			rows:    []string{"3"},
			ordered: true,
		},
		{
			name:    "offset_on_aggregate",
			query:   "SELECT ax, COUNT(*) FROM a GROUP BY ax ORDER BY ax LIMIT 1 OFFSET 1",
			rows:    []string{"1|1"},
			ordered: true,
		},
		{
			name:    "union",
			query:   "SELECT ax FROM a UNION SELECT bx FROM b",
			wantErr: "UNION",
		},
		{
			name:    "trailing_garbage",
			query:   "SELECT ax FROM a LIMIT 1 banana",
			wantErr: "unexpected",
		},
		{
			name:  "trailing_semicolon",
			query: "SELECT ax FROM a WHERE ax = 1;",
			rows:  []string{"1"},
		},
	})
}

// TestDocumentedOrderByAndRejections pins the ORDER BY forms and the rejected
// constructs listed in docs/sql.md, so the document cannot drift from behavior.
func TestDocumentedOrderByAndRejections(t *testing.T) {
	runSemanticsCases(t, []semanticsCase{
		{
			name:    "order_by_position",
			query:   "SELECT name, shared FROM a ORDER BY 2 DESC",
			rows:    []string{"apricot|40", "cherry|30", "banana|20", "apple|10"},
			ordered: true,
		},
		{
			name:    "order_by_unselected_expression",
			query:   "SELECT name FROM a ORDER BY fx * -1",
			rows:    []string{"cherry", "apple", "apricot", "banana"},
			ordered: true,
		},
		{
			name:    "order_by_alias_shadows_column",
			query:   "SELECT ax AS shared FROM a ORDER BY shared DESC",
			rows:    []string{"3", "2", "1", "-2"},
			ordered: true,
		},
		{
			name:    "order_by_position_out_of_range",
			query:   "SELECT ax FROM a ORDER BY 2",
			wantErr: "position",
		},
		{
			name:    "order_by_ambiguous_output_name",
			query:   "SELECT a.shared, b.shared FROM a, b WHERE a.ax = b.bx ORDER BY shared",
			wantErr: "ambiguous",
		},
		{
			name:    "distinct_order_by_unselected",
			query:   "SELECT DISTINCT name FROM a ORDER BY ax",
			wantErr: "DISTINCT",
		},
		{
			name:    "explicit_join_syntax",
			query:   "SELECT ax FROM a JOIN b ON a.ax = b.bx",
			wantErr: "unexpected",
		},
		{
			name:    "group_by_expression",
			query:   "SELECT COUNT(*) FROM a GROUP BY ax + 1",
			wantErr: "GROUP BY only supports column references",
		},
		{
			name:    "group_by_select_alias",
			query:   "SELECT ax AS k, COUNT(*) FROM a GROUP BY k",
			wantErr: "not found",
		},
		{
			name:    "string_ordering_comparison",
			query:   "SELECT ax FROM a WHERE name < 'b'",
			wantErr: "not supported",
		},
		{
			name:    "string_column_equality",
			query:   "SELECT a.ax FROM a, b WHERE a.ax = b.bx AND a.name = b.bname",
			wantErr: "not supported",
		},
		{
			name:    "nested_aggregate",
			query:   "SELECT SUM(COUNT(*)) FROM a",
			wantErr: "nested",
		},
		{
			name:    "sum_distinct",
			query:   "SELECT SUM(DISTINCT ax) FROM a",
			wantErr: "DISTINCT",
		},
		{
			name:    "duplicate_table_name",
			query:   "SELECT ax FROM a, a WHERE ax = ax",
			wantErr: "more than once",
		},
	})
}
