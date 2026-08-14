package goldentest

// This file implements a deliberately naive, row-at-a-time SQL reference evaluator.
// It exists solely as a correctness oracle for vexq's vectorized engine.
//
// DESIGN INTENT: This evaluator is intentionally simple, independent of the engine's
// operator code, and optimized for clarity over speed. It serves as ground truth:
// if this and the engine disagree, we investigate the engine.
//
// NULL SEMANTICS: Standard SQL three-valued logic. NULL propagates through arithmetic
// and comparison. Aggregates (except COUNT(*)) skip NULLs. AVG divides by non-null count.
// SUM/MIN/MAX/AVG over all-NULLs produce NULL. COUNT(*) counts all rows, COUNT(col)
// counts non-NULLs. DISTINCT treats NULLs as equal.
//
// SORT ORDER: NULLs sort first (matching exec/sort.go:less()).

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// RefResult holds the output of the reference evaluator.
type RefResult struct {
	Columns []string // output column names
	Types   []storage.DataType
	Rows    []Row
}

// Evaluate executes a parsed SELECT statement against in-memory tables and
// returns the reference result. It returns an error for unsupported constructs
// or invalid queries (e.g., ambiguous columns, missing tables).
func Evaluate(stmt *sql.SelectStmt, tables []Table) (*RefResult, error) {
	// Step 1: Resolve FROM — build a combined row set (cross join of all tables).
	combined, colMap, err := resolveFrom(stmt.From, tables)
	if err != nil {
		return nil, err
	}

	// Step 2: Apply WHERE filter.
	filtered, err := applyFilter(combined, colMap, stmt.Where)
	if err != nil {
		return nil, err
	}

	// Step 3: GROUP BY + aggregates, or plain projection.
	var result *RefResult
	if hasAggregates(stmt) {
		result, err = evaluateGroupedQuery(stmt, filtered, colMap)
	} else {
		result, err = evaluateSimpleQuery(stmt, filtered, colMap)
	}
	if err != nil {
		return nil, err
	}

	// Step 4: HAVING (post-aggregate filter). Already handled inside evaluateGroupedQuery.

	// Step 5: DISTINCT.
	if stmt.Distinct {
		result = applyDistinct(result)
	}

	// Step 6: ORDER BY.
	if len(stmt.OrderBy) > 0 {
		if err := applyOrderBy(result, stmt.OrderBy, colMap, stmt); err != nil {
			return nil, err
		}
	}

	// Step 7: LIMIT.
	if stmt.Limit != nil {
		limit := int(*stmt.Limit)
		if limit < len(result.Rows) {
			result.Rows = result.Rows[:limit]
		}
	}

	return result, nil
}

// --- FROM resolution (cross join) ---

// colInfo maps a column reference to its position in the combined row.
type colInfo struct {
	index int
	typ   storage.DataType
	table string // source table name or alias
}

type colMap map[string][]colInfo // bare name → list of matching columns

func resolveFrom(refs []sql.TableRef, tables []Table) ([]Row, colMap, error) {
	if len(refs) == 0 {
		return nil, nil, fmt.Errorf("reference: no FROM tables")
	}

	cm := make(colMap)
	var combinedRows []Row
	colOffset := 0

	for i, ref := range refs {
		tbl := findTable(tables, ref.Name)
		if tbl == nil {
			return nil, nil, fmt.Errorf("reference: table %q not found", ref.Name)
		}

		alias := ref.Alias
		if alias == "" {
			alias = ref.Name
		}

		// Register columns in the map.
		for j, field := range tbl.Schema.Fields {
			ci := colInfo{index: colOffset + j, typ: field.Type, table: alias}
			// Register under bare name.
			cm[field.Name] = append(cm[field.Name], ci)
			// Register under qualified name.
			qualName := alias + "." + field.Name
			cm[qualName] = append(cm[qualName], ci)
		}

		// Cross join: combine existing rows with new table rows.
		if i == 0 {
			for _, row := range tbl.Rows {
				combinedRows = append(combinedRows, row)
			}
		} else {
			var newRows []Row
			for _, existing := range combinedRows {
				for _, tblRow := range tbl.Rows {
					combined := make(Row, len(existing)+len(tblRow))
					copy(combined, existing)
					copy(combined[len(existing):], tblRow)
					newRows = append(newRows, combined)
				}
			}
			combinedRows = newRows
		}
		colOffset += len(tbl.Schema.Fields)
	}

	return combinedRows, cm, nil
}

func findTable(tables []Table, name string) *Table {
	for i := range tables {
		if tables[i].Name == name {
			return &tables[i]
		}
	}
	return nil
}

// --- Expression evaluation ---

// evalExpr evaluates an expression against a single row. Returns a Value.
func evalExpr(expr sql.Expr, row Row, cm colMap) (Value, error) {
	switch e := expr.(type) {
	case *sql.ColumnRefExpr:
		return resolveColumn(e, row, cm)
	case *sql.IntLiteral:
		return Value{Int64: e.Value}, nil
	case *sql.FloatLiteral:
		return Value{Float: e.Value}, nil
	case *sql.StringLiteral:
		return Value{Str: e.Value}, nil
	case *sql.BoolLiteral:
		return Value{Bool: e.Value}, nil
	case *sql.NullLiteral:
		return Value{IsNull: true}, nil
	case *sql.BinaryExpr:
		return evalBinary(e, row, cm)
	case *sql.UnaryExpr:
		return evalUnary(e, row, cm)
	case *sql.IsNullExpr:
		return evalIsNull(e, row, cm)
	case *sql.BetweenExpr:
		return evalBetween(e, row, cm)
	case *sql.InExpr:
		return evalIn(e, row, cm)
	case *sql.LikeExpr:
		return evalLike(e, row, cm)
	case *sql.CaseExpr:
		return evalCase(e, row, cm)
	case *sql.StarExpr:
		return Value{}, fmt.Errorf("reference: cannot evaluate * as scalar")
	case *sql.AggFuncExpr:
		// This is handled in the grouping path, not here.
		return Value{}, fmt.Errorf("reference: aggregate %s in non-aggregate context", e.Func)
	default:
		return Value{}, fmt.Errorf("reference: unsupported expression type %T", expr)
	}
}

func resolveColumn(ref *sql.ColumnRefExpr, row Row, cm colMap) (Value, error) {
	// Try qualified first.
	if ref.Table != "" {
		key := ref.Table + "." + ref.Name
		infos, ok := cm[key]
		if !ok || len(infos) == 0 {
			return Value{}, fmt.Errorf("reference: column %s.%s not found", ref.Table, ref.Name)
		}
		return row[infos[0].index], nil
	}
	// Unqualified lookup.
	infos, ok := cm[ref.Name]
	if !ok || len(infos) == 0 {
		return Value{}, fmt.Errorf("reference: column %q not found", ref.Name)
	}
	if len(infos) > 1 {
		return Value{}, fmt.Errorf("reference: ambiguous column %q", ref.Name)
	}
	return row[infos[0].index], nil
}

func evalBinary(e *sql.BinaryExpr, row Row, cm colMap) (Value, error) {
	left, err := evalExpr(e.Left, row, cm)
	if err != nil {
		return Value{}, err
	}
	right, err := evalExpr(e.Right, row, cm)
	if err != nil {
		return Value{}, err
	}

	// Boolean operators: AND/OR have three-valued logic.
	switch e.Op {
	case sql.OpAnd:
		return evalAnd(left, right), nil
	case sql.OpOr:
		return evalOr(left, right), nil
	}

	// NULL propagation for other operators.
	if left.IsNull || right.IsNull {
		return Value{IsNull: true}, nil
	}

	switch e.Op {
	case sql.OpAdd, sql.OpSub, sql.OpMul, sql.OpDiv:
		return evalArithmetic(e.Op, left, right)
	case sql.OpEQ, sql.OpNE, sql.OpLT, sql.OpLE, sql.OpGT, sql.OpGE:
		return evalComparison(e.Op, left, right)
	default:
		return Value{}, fmt.Errorf("reference: unsupported operator %s", e.Op)
	}
}

func evalAnd(l, r Value) Value {
	// SQL three-valued AND: FALSE AND anything = FALSE.
	lFalse := !l.IsNull && !l.Bool
	rFalse := !r.IsNull && !r.Bool
	if lFalse || rFalse {
		return Value{Bool: false}
	}
	if l.IsNull || r.IsNull {
		return Value{IsNull: true}
	}
	return Value{Bool: l.Bool && r.Bool}
}

func evalOr(l, r Value) Value {
	// SQL three-valued OR: TRUE OR anything = TRUE.
	lTrue := !l.IsNull && l.Bool
	rTrue := !r.IsNull && r.Bool
	if lTrue || rTrue {
		return Value{Bool: true}
	}
	if l.IsNull || r.IsNull {
		return Value{IsNull: true}
	}
	return Value{Bool: l.Bool || r.Bool}
}

func evalArithmetic(op sql.BinOp, l, r Value) (Value, error) {
	lf := toFloat(l)
	rf := toFloat(r)
	switch op {
	case sql.OpAdd:
		return Value{Float: lf + rf}, nil
	case sql.OpSub:
		return Value{Float: lf - rf}, nil
	case sql.OpMul:
		return Value{Float: lf * rf}, nil
	case sql.OpDiv:
		if rf == 0 {
			return Value{IsNull: true}, nil // division by zero → NULL
		}
		return Value{Float: lf / rf}, nil
	}
	return Value{}, fmt.Errorf("reference: unknown arithmetic op %s", op)
}

func evalComparison(op sql.BinOp, l, r Value) (Value, error) {
	cmp := compareValues(l, r)
	var result bool
	switch op {
	case sql.OpEQ:
		result = cmp == 0
	case sql.OpNE:
		result = cmp != 0
	case sql.OpLT:
		result = cmp < 0
	case sql.OpLE:
		result = cmp <= 0
	case sql.OpGT:
		result = cmp > 0
	case sql.OpGE:
		result = cmp >= 0
	default:
		return Value{}, fmt.Errorf("reference: unknown comparison op %s", op)
	}
	return Value{Bool: result}, nil
}

// compareValues returns -1, 0, or 1. Both values are guaranteed non-null.
func compareValues(a, b Value) int {
	// String comparison.
	if a.Str != "" || b.Str != "" {
		if a.Str < b.Str {
			return -1
		}
		if a.Str > b.Str {
			return 1
		}
		return 0
	}
	// Numeric comparison (promote to float for mixed types).
	af := toFloat(a)
	bf := toFloat(b)
	if af < bf {
		return -1
	}
	if af > bf {
		return 1
	}
	return 0
}

func toFloat(v Value) float64 {
	if v.Float != 0 {
		return v.Float
	}
	if v.Int64 != 0 {
		return float64(v.Int64)
	}
	if v.Date != 0 {
		return float64(v.Date)
	}
	// Distinguish between actual zero and unset.
	if v.Bool {
		return 1
	}
	return float64(v.Int64) // could be 0
}

func evalUnary(e *sql.UnaryExpr, row Row, cm colMap) (Value, error) {
	val, err := evalExpr(e.Expr, row, cm)
	if err != nil {
		return Value{}, err
	}
	if val.IsNull {
		return Value{IsNull: true}, nil
	}
	switch e.Op {
	case sql.OpNot:
		return Value{Bool: !val.Bool}, nil
	case sql.OpMinus:
		return Value{Float: -toFloat(val), Int64: -val.Int64}, nil
	}
	return Value{}, fmt.Errorf("reference: unknown unary op %s", e.Op)
}

func evalIsNull(e *sql.IsNullExpr, row Row, cm colMap) (Value, error) {
	val, err := evalExpr(e.Expr, row, cm)
	if err != nil {
		return Value{}, err
	}
	isNull := val.IsNull
	if e.IsNot {
		return Value{Bool: !isNull}, nil
	}
	return Value{Bool: isNull}, nil
}

func evalBetween(e *sql.BetweenExpr, row Row, cm colMap) (Value, error) {
	val, err := evalExpr(e.Expr, row, cm)
	if err != nil {
		return Value{}, err
	}
	lo, err := evalExpr(e.Lo, row, cm)
	if err != nil {
		return Value{}, err
	}
	hi, err := evalExpr(e.Hi, row, cm)
	if err != nil {
		return Value{}, err
	}
	if val.IsNull || lo.IsNull || hi.IsNull {
		return Value{IsNull: true}, nil
	}
	inRange := compareValues(val, lo) >= 0 && compareValues(val, hi) <= 0
	if e.Not {
		return Value{Bool: !inRange}, nil
	}
	return Value{Bool: inRange}, nil
}

func evalIn(e *sql.InExpr, row Row, cm colMap) (Value, error) {
	val, err := evalExpr(e.Expr, row, cm)
	if err != nil {
		return Value{}, err
	}
	if val.IsNull {
		return Value{IsNull: true}, nil
	}
	found := false
	hasNull := false
	for _, item := range e.List {
		iv, err := evalExpr(item, row, cm)
		if err != nil {
			return Value{}, err
		}
		if iv.IsNull {
			hasNull = true
			continue
		}
		if compareValues(val, iv) == 0 {
			found = true
			break
		}
	}
	if found {
		if e.Not {
			return Value{Bool: false}, nil
		}
		return Value{Bool: true}, nil
	}
	if hasNull {
		return Value{IsNull: true}, nil
	}
	if e.Not {
		return Value{Bool: true}, nil
	}
	return Value{Bool: false}, nil
}

func evalLike(e *sql.LikeExpr, row Row, cm colMap) (Value, error) {
	val, err := evalExpr(e.Expr, row, cm)
	if err != nil {
		return Value{}, err
	}
	pat, err := evalExpr(e.Pattern, row, cm)
	if err != nil {
		return Value{}, err
	}
	if val.IsNull || pat.IsNull {
		return Value{IsNull: true}, nil
	}
	matched := likeMatch(val.Str, pat.Str)
	if e.Not {
		return Value{Bool: !matched}, nil
	}
	return Value{Bool: matched}, nil
}

// likeMatch implements SQL LIKE pattern matching.
// % matches zero or more characters; _ matches exactly one.
func likeMatch(s, pattern string) bool {
	// Convert to regex.
	var re strings.Builder
	re.WriteString("^")
	for _, ch := range pattern {
		switch ch {
		case '%':
			re.WriteString(".*")
		case '_':
			re.WriteString(".")
		case '.', '^', '$', '+', '?', '{', '}', '[', ']', '|', '(', ')', '\\':
			re.WriteRune('\\')
			re.WriteRune(ch)
		default:
			re.WriteRune(ch)
		}
	}
	re.WriteString("$")
	matched, _ := regexp.MatchString(re.String(), s)
	return matched
}

func evalCase(e *sql.CaseExpr, row Row, cm colMap) (Value, error) {
	for _, when := range e.Whens {
		cond, err := evalExpr(when.Cond, row, cm)
		if err != nil {
			return Value{}, err
		}
		if !cond.IsNull && cond.Bool {
			return evalExpr(when.Result, row, cm)
		}
	}
	if e.Else != nil {
		return evalExpr(e.Else, row, cm)
	}
	return Value{IsNull: true}, nil
}

// --- Filter application ---

func applyFilter(rows []Row, cm colMap, where sql.Expr) ([]Row, error) {
	if where == nil {
		return rows, nil
	}
	var result []Row
	for _, row := range rows {
		val, err := evalExpr(where, row, cm)
		if err != nil {
			return nil, err
		}
		// SQL filter: only TRUE passes (not NULL, not FALSE).
		if !val.IsNull && val.Bool {
			result = append(result, row)
		}
	}
	return result, nil
}

// --- Aggregate detection ---

func hasAggregates(stmt *sql.SelectStmt) bool {
	if len(stmt.GroupBy) > 0 {
		return true
	}
	for _, col := range stmt.Columns {
		if containsAggregate(col.Expr) {
			return true
		}
	}
	return false
}

func containsAggregate(expr sql.Expr) bool {
	switch e := expr.(type) {
	case *sql.AggFuncExpr:
		return true
	case *sql.BinaryExpr:
		return containsAggregate(e.Left) || containsAggregate(e.Right)
	case *sql.UnaryExpr:
		return containsAggregate(e.Expr)
	case *sql.CaseExpr:
		for _, w := range e.Whens {
			if containsAggregate(w.Cond) || containsAggregate(w.Result) {
				return true
			}
		}
		if e.Else != nil && containsAggregate(e.Else) {
			return true
		}
	}
	return false
}

// --- Simple (non-aggregate) query evaluation ---

func evaluateSimpleQuery(stmt *sql.SelectStmt, rows []Row, cm colMap) (*RefResult, error) {
	// Determine output schema by evaluating column expressions.
	result := &RefResult{}

	for _, row := range rows {
		var outRow Row
		for i, col := range stmt.Columns {
			if _, ok := col.Expr.(*sql.StarExpr); ok {
				// Expand * to all columns.
				outRow = append(outRow, row...)
				if len(result.Columns) == 0 {
					for name, infos := range cm {
						// Only register bare names (not qualified).
						if !strings.Contains(name, ".") && len(infos) == 1 {
							_ = i // suppress
						}
					}
				}
			} else {
				val, err := evalExpr(col.Expr, row, cm)
				if err != nil {
					return nil, err
				}
				outRow = append(outRow, val)
			}
		}
		result.Rows = append(result.Rows, outRow)
	}

	// Build column names.
	result.Columns = buildOutputColumns(stmt, cm)
	return result, nil
}

// --- Grouped (aggregate) query evaluation ---

func evaluateGroupedQuery(stmt *sql.SelectStmt, rows []Row, cm colMap) (*RefResult, error) {
	// Build groups.
	type groupKey string
	groups := make(map[groupKey][]Row)
	var keyOrder []groupKey // preserve insertion order for determinism

	if len(stmt.GroupBy) == 0 {
		// Global aggregate — all rows form one group.
		k := groupKey("")
		groups[k] = rows
		keyOrder = append(keyOrder, k)
	} else {
		for _, row := range rows {
			key, err := buildGroupKey(row, stmt.GroupBy, cm)
			if err != nil {
				return nil, err
			}
			k := groupKey(key)
			if _, exists := groups[k]; !exists {
				keyOrder = append(keyOrder, k)
			}
			groups[k] = append(groups[k], row)
		}
	}

	// Evaluate each group.
	result := &RefResult{Columns: buildOutputColumns(stmt, cm)}
	for _, k := range keyOrder {
		groupRows := groups[k]
		outRow, err := evaluateGroupOutput(stmt, groupRows, cm)
		if err != nil {
			return nil, err
		}
		// Apply HAVING.
		if stmt.Having != nil {
			// Evaluate HAVING against the output row using output column aliases.
			pass, err := evalHaving(stmt, outRow, groupRows, cm)
			if err != nil {
				return nil, err
			}
			if !pass {
				continue
			}
		}
		result.Rows = append(result.Rows, outRow)
	}

	// Handle empty-input global aggregate: emit one row with COUNT=0, other aggs=NULL.
	if len(stmt.GroupBy) == 0 && len(rows) == 0 {
		outRow, err := evaluateGroupOutput(stmt, nil, cm)
		if err != nil {
			return nil, err
		}
		result.Rows = append(result.Rows, outRow)
	}

	return result, nil
}

func buildGroupKey(row Row, groupExprs []sql.Expr, cm colMap) (string, error) {
	var parts []string
	for _, expr := range groupExprs {
		val, err := evalExpr(expr, row, cm)
		if err != nil {
			return "", err
		}
		parts = append(parts, valueToKey(val))
	}
	return strings.Join(parts, "|"), nil
}

func valueToKey(v Value) string {
	if v.IsNull {
		return "<NULL>"
	}
	if v.Str != "" {
		return "s:" + v.Str
	}
	if v.Float != 0 || v.Int64 == 0 {
		return fmt.Sprintf("f:%v", v.Float)
	}
	return fmt.Sprintf("i:%d", v.Int64)
}

func evaluateGroupOutput(stmt *sql.SelectStmt, groupRows []Row, cm colMap) (Row, error) {
	var outRow Row
	for _, col := range stmt.Columns {
		val, err := evalAggExpr(col.Expr, groupRows, cm)
		if err != nil {
			return nil, err
		}
		outRow = append(outRow, val)
	}
	return outRow, nil
}

// evalAggExpr evaluates an expression that may contain aggregates against a group of rows.
func evalAggExpr(expr sql.Expr, groupRows []Row, cm colMap) (Value, error) {
	switch e := expr.(type) {
	case *sql.AggFuncExpr:
		return computeAggregate(e, groupRows, cm)
	case *sql.ColumnRefExpr:
		// In a GROUP BY context, column references refer to the first row's value.
		if len(groupRows) == 0 {
			return Value{IsNull: true}, nil
		}
		return resolveColumn(e, groupRows[0], cm)
	case *sql.BinaryExpr:
		left, err := evalAggExpr(e.Left, groupRows, cm)
		if err != nil {
			return Value{}, err
		}
		right, err := evalAggExpr(e.Right, groupRows, cm)
		if err != nil {
			return Value{}, err
		}
		// Handle boolean operators.
		switch e.Op {
		case sql.OpAnd:
			return evalAnd(left, right), nil
		case sql.OpOr:
			return evalOr(left, right), nil
		}
		if left.IsNull || right.IsNull {
			return Value{IsNull: true}, nil
		}
		switch e.Op {
		case sql.OpAdd, sql.OpSub, sql.OpMul, sql.OpDiv:
			return evalArithmetic(e.Op, left, right)
		case sql.OpEQ, sql.OpNE, sql.OpLT, sql.OpLE, sql.OpGT, sql.OpGE:
			return evalComparison(e.Op, left, right)
		default:
			return Value{}, fmt.Errorf("reference: unsupported binary op %s in aggregate context", e.Op)
		}
	case *sql.CaseExpr:
		// Evaluate CASE WHEN with aggregate sub-expressions.
		for _, when := range e.Whens {
			cond, err := evalAggExpr(when.Cond, groupRows, cm)
			if err != nil {
				return Value{}, err
			}
			if !cond.IsNull && cond.Bool {
				return evalAggExpr(when.Result, groupRows, cm)
			}
		}
		if e.Else != nil {
			return evalAggExpr(e.Else, groupRows, cm)
		}
		return Value{IsNull: true}, nil
	case *sql.IntLiteral:
		return Value{Int64: e.Value}, nil
	case *sql.FloatLiteral:
		return Value{Float: e.Value}, nil
	case *sql.StringLiteral:
		return Value{Str: e.Value}, nil
	case *sql.NullLiteral:
		return Value{IsNull: true}, nil
	default:
		return Value{}, fmt.Errorf("reference: unsupported expression in aggregate context: %T", expr)
	}
}

func computeAggregate(agg *sql.AggFuncExpr, groupRows []Row, cm colMap) (Value, error) {
	fn := strings.ToUpper(agg.Func)

	// COUNT(*) counts all rows.
	if fn == "COUNT" && (agg.Arg == nil || isStarExpr(agg.Arg)) {
		return Value{Int64: int64(len(groupRows))}, nil
	}

	// Collect non-null argument values.
	var vals []Value
	seen := make(map[string]bool) // for DISTINCT
	for _, row := range groupRows {
		val, err := evalExpr(agg.Arg, row, cm)
		if err != nil {
			return Value{}, err
		}
		if val.IsNull {
			continue
		}
		if agg.Distinct {
			key := valueToKey(val)
			if seen[key] {
				continue
			}
			seen[key] = true
		}
		vals = append(vals, val)
	}

	// If all values are NULL (empty vals), aggregates return NULL except COUNT.
	if fn == "COUNT" {
		return Value{Int64: int64(len(vals))}, nil
	}
	if len(vals) == 0 {
		return Value{IsNull: true}, nil
	}

	switch fn {
	case "SUM":
		sum := 0.0
		for _, v := range vals {
			sum += toFloat(v)
		}
		return Value{Float: sum}, nil
	case "AVG":
		sum := 0.0
		for _, v := range vals {
			sum += toFloat(v)
		}
		return Value{Float: sum / float64(len(vals))}, nil
	case "MIN":
		min := vals[0]
		for _, v := range vals[1:] {
			if compareValues(v, min) < 0 {
				min = v
			}
		}
		return min, nil
	case "MAX":
		max := vals[0]
		for _, v := range vals[1:] {
			if compareValues(v, max) > 0 {
				max = v
			}
		}
		return max, nil
	default:
		return Value{}, fmt.Errorf("reference: unknown aggregate function %q", fn)
	}
}

func isStarExpr(e sql.Expr) bool {
	_, ok := e.(*sql.StarExpr)
	return ok
}

// --- HAVING evaluation ---

func evalHaving(stmt *sql.SelectStmt, outRow Row, groupRows []Row, cm colMap) (bool, error) {
	// HAVING can reference aggregates — evaluate using the group rows.
	val, err := evalAggExpr(stmt.Having, groupRows, cm)
	if err != nil {
		return false, err
	}
	return !val.IsNull && val.Bool, nil
}

// --- DISTINCT ---

func applyDistinct(result *RefResult) *RefResult {
	seen := make(map[string]bool)
	var unique []Row
	for _, row := range result.Rows {
		key := rowToKey(row)
		if seen[key] {
			continue
		}
		seen[key] = true
		unique = append(unique, row)
	}
	result.Rows = unique
	return result
}

func rowToKey(row Row) string {
	var parts []string
	for _, v := range row {
		parts = append(parts, valueToKey(v))
	}
	return strings.Join(parts, "||")
}

// --- ORDER BY ---

func applyOrderBy(result *RefResult, orderBy []sql.OrderByItem, cm colMap, stmt *sql.SelectStmt) error {
	// Resolve each ORDER BY expression to output column index or evaluate.
	type sortSpec struct {
		colIdx     int // -1 if expression-based
		descending bool
	}
	var specs []sortSpec
	for _, ob := range orderBy {
		idx := resolveOrderByCol(ob.Expr, stmt)
		specs = append(specs, sortSpec{colIdx: idx, descending: ob.Descending})
	}

	sort.SliceStable(result.Rows, func(i, j int) bool {
		for _, spec := range specs {
			ci := spec.colIdx
			if ci < 0 || ci >= len(result.Rows[i]) {
				continue
			}
			a := result.Rows[i][ci]
			b := result.Rows[j][ci]

			// Nulls sort first (matching engine behavior).
			if a.IsNull && b.IsNull {
				continue
			}
			if a.IsNull {
				return true
			}
			if b.IsNull {
				return false
			}

			cmp := compareValues(a, b)
			if cmp == 0 {
				continue
			}
			if spec.descending {
				return cmp > 0
			}
			return cmp < 0
		}
		return false
	})
	return nil
}

// resolveOrderByCol resolves an ORDER BY expression to an output column index.
func resolveOrderByCol(expr sql.Expr, stmt *sql.SelectStmt) int {
	// Check if it's a column reference matching an output column name or alias.
	if ref, ok := expr.(*sql.ColumnRefExpr); ok {
		for i, col := range stmt.Columns {
			if col.Alias == ref.Name {
				return i
			}
			if cr, ok := col.Expr.(*sql.ColumnRefExpr); ok && cr.Name == ref.Name {
				return i
			}
		}
	}
	// Check if it's an integer positional reference.
	if lit, ok := expr.(*sql.IntLiteral); ok {
		return int(lit.Value) - 1 // 1-based
	}
	return -1
}

// --- Output column names ---

func buildOutputColumns(stmt *sql.SelectStmt, cm colMap) []string {
	var names []string
	for _, col := range stmt.Columns {
		if col.Alias != "" {
			names = append(names, col.Alias)
		} else if _, ok := col.Expr.(*sql.StarExpr); ok {
			// Expand star to all column bare names.
			for name, infos := range cm {
				if !strings.Contains(name, ".") && len(infos) == 1 {
					names = append(names, name)
				}
			}
		} else if ref, ok := col.Expr.(*sql.ColumnRefExpr); ok {
			names = append(names, ref.Name)
		} else if agg, ok := col.Expr.(*sql.AggFuncExpr); ok {
			if isStarExpr(agg.Arg) || agg.Arg == nil {
				names = append(names, fmt.Sprintf("%s(*)", strings.ToLower(agg.Func)))
			} else if ref, ok := agg.Arg.(*sql.ColumnRefExpr); ok {
				names = append(names, fmt.Sprintf("%s(%s)", strings.ToLower(agg.Func), ref.Name))
			} else {
				names = append(names, strings.ToLower(agg.Func))
			}
		} else {
			names = append(names, fmt.Sprintf("expr_%d", len(names)))
		}
	}
	return names
}

// --- Utility for comparing float values ---

// FloatEpsilon is the tolerance for float comparison in result verification.
// Justified: floating-point arithmetic differences between the reference
// (which processes values individually) and the engine (which may accumulate
// in different order or use intermediate int64 representations) can produce
// small differences. 1e-9 accommodates IEEE 754 double-precision rounding
// for the value ranges used in our test dataset (values up to ~10000).
const FloatEpsilon = 1e-9

// FloatClose reports whether two float64 values are within epsilon.
func FloatClose(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	diff := math.Abs(a - b)
	// Use relative epsilon for larger values.
	max := math.Max(math.Abs(a), math.Abs(b))
	if max > 1.0 {
		return diff/max < FloatEpsilon
	}
	return diff < FloatEpsilon
}
