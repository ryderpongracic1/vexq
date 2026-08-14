package planner

import (
	"context"
	"fmt"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
)

// Build converts a SQL AST into a logical plan tree.
func Build(ctx context.Context, stmt *sql.SelectStmt, cat *catalog.Catalog) (LogicalNode, error) {
	if len(stmt.From) == 0 {
		return nil, fmt.Errorf("planner: no FROM clause")
	}

	// Build a LogicalScan per table.
	scans := make([]*LogicalScan, len(stmt.From))
	schemas := make([]exec.Schema, len(stmt.From))
	for i, ref := range stmt.From {
		entry, ok := cat.Lookup(ctx, ref.Name)
		if !ok {
			return nil, fmt.Errorf("planner: table %q not found", ref.Name)
		}
		scans[i] = &LogicalScan{
			TableName: entry.Name,
			FilePath:  entry.FilePath,
			Schema:    entry.Schema,
		}
		schemas[i] = entry.Schema
	}

	var root LogicalNode
	if len(scans) == 1 {
		root = scans[0]
		if stmt.Where != nil {
			root = &LogicalFilter{Child: root, Predicate: stmt.Where}
		}
	} else {
		// Multi-table: split WHERE into join conditions and per-table filters.
		var err error
		// Pass stmt.From so symbolTable can resolve qualified column references (table.col).
		root, err = buildMultiTablePlan(scans, schemas, stmt.Where, stmt.From)
		if err != nil {
			return nil, err
		}
	}

	// GROUP BY / aggregates.
	hasAggs := hasAggregates(stmt.Columns)
	if hasAggs || len(stmt.GroupBy) > 0 {
		agg, err := buildAggregate(root, stmt)
		if err != nil {
			return nil, err
		}

		// HAVING — post-aggregate filter applied after aggregation.
		// If the HAVING predicate contains aggregate function expressions
		// (e.g. COUNT(*) > 3), we rewrite them to column references that
		// point at matching output columns of the aggregate. If no match
		// exists in the SELECT list, we add a hidden aggregate and strip
		// it with a projection after the filter.
		if stmt.Having != nil {
			origAggCount := len(agg.Aggs)
			rewritten := rewriteHavingAggs(stmt.Having, agg)
			root = agg
			root = &LogicalFilter{Child: root, Predicate: rewritten}
			// If hidden aggregates were added, project them away so the
			// output schema matches what the user's SELECT requested.
			if len(agg.Aggs) > origAggCount {
				root = buildHavingProjection(root, agg, origAggCount)
			}
		} else {
			root = agg
		}
	} else {
		// Project.
		if !isSelectStar(stmt.Columns) {
			proj, err := buildProject(root, stmt)
			if err != nil {
				return nil, err
			}
			root = proj
		}
	}

	// DISTINCT — deduplicate after projection/aggregation but before ORDER BY/LIMIT.
	if stmt.Distinct {
		root = &LogicalDistinct{Child: root}
	}

	// ORDER BY.
	if len(stmt.OrderBy) > 0 {
		root = &LogicalSort{Child: root, OrderBy: stmt.OrderBy}
	}

	// LIMIT.
	if stmt.Limit != nil {
		root = &LogicalLimit{Child: root, Count: *stmt.Limit}
	}

	return root, nil
}

// qualifiedCol records the position of a column within the multi-table context.
type qualifiedCol struct {
	tableIdx int
	colIdx   int
}

// symbolTable maps qualified "table.col" and unqualified "col" names to their
// positions across all tables in a FROM clause.
type symbolTable struct {
	qualified   map[string]qualifiedCol // "tableName.colName" → location
	unqualified map[string]qualifiedCol // "colName" → location (only if unambiguous)
	ambiguous   map[string]bool         // "colName" → true if appears in multiple tables
	tableNames  []string                // table name (or alias) for each index
}

// newSymbolTable builds a symbol table from the given schemas and table refs.
func newSymbolTable(schemas []exec.Schema, tableRefs []sql.TableRef) *symbolTable {
	st := &symbolTable{
		qualified:   make(map[string]qualifiedCol),
		unqualified: make(map[string]qualifiedCol),
		ambiguous:   make(map[string]bool),
		tableNames:  make([]string, len(schemas)),
	}
	for i, ref := range tableRefs {
		name := ref.Alias
		if name == "" {
			name = ref.Name
		}
		st.tableNames[i] = name
		for j, f := range schemas[i].Fields {
			qc := qualifiedCol{tableIdx: i, colIdx: j}
			st.qualified[name+"."+f.Name] = qc
			if st.ambiguous[f.Name] {
				continue
			}
			if existing, exists := st.unqualified[f.Name]; exists {
				if existing.tableIdx != i {
					st.ambiguous[f.Name] = true
					delete(st.unqualified, f.Name)
				}
			} else {
				st.unqualified[f.Name] = qc
			}
		}
	}
	return st
}

// resolve looks up a column reference in the symbol table.
func (st *symbolTable) resolve(ref *sql.ColumnRefExpr) (qualifiedCol, error) {
	if ref.Table != "" {
		key := ref.Table + "." + ref.Name
		qc, ok := st.qualified[key]
		if !ok {
			return qualifiedCol{}, fmt.Errorf("column %q not found in table %q", ref.Name, ref.Table)
		}
		return qc, nil
	}
	if st.ambiguous[ref.Name] {
		return qualifiedCol{}, fmt.Errorf("column %q is ambiguous; qualify with table name", ref.Name)
	}
	qc, ok := st.unqualified[ref.Name]
	if !ok {
		return qualifiedCol{}, fmt.Errorf("column %q not found in any table", ref.Name)
	}
	return qc, nil
}

// resolveTableIdx returns the table index for a column reference.
func (st *symbolTable) resolveTableIdx(ref *sql.ColumnRefExpr) (int, bool) {
	qc, err := st.resolve(ref)
	if err != nil {
		return 0, false
	}
	return qc.tableIdx, true
}

// predicateColRefs collects all ColumnRefExpr nodes from an expression tree.
func predicateColRefs(e sql.Expr) []*sql.ColumnRefExpr {
	if e == nil {
		return nil
	}
	var refs []*sql.ColumnRefExpr
	collectColRefs(e, &refs)
	return refs
}

func collectColRefs(e sql.Expr, out *[]*sql.ColumnRefExpr) {
	if e == nil {
		return
	}
	switch x := e.(type) {
	case *sql.ColumnRefExpr:
		*out = append(*out, x)
	case *sql.BinaryExpr:
		collectColRefs(x.Left, out)
		collectColRefs(x.Right, out)
	case *sql.UnaryExpr:
		collectColRefs(x.Expr, out)
	case *sql.IsNullExpr:
		collectColRefs(x.Expr, out)
	case *sql.BetweenExpr:
		collectColRefs(x.Expr, out)
		collectColRefs(x.Lo, out)
		collectColRefs(x.Hi, out)
	case *sql.InExpr:
		collectColRefs(x.Expr, out)
		for _, item := range x.List {
			collectColRefs(item, out)
		}
	case *sql.CaseExpr:
		for _, w := range x.Whens {
			collectColRefs(w.Cond, out)
			collectColRefs(w.Result, out)
		}
		collectColRefs(x.Else, out)
	case *sql.AggFuncExpr:
		collectColRefs(x.Arg, out)
	}
}

// buildMultiTablePlan builds a left-deep join tree from multiple table scans,
// pushing single-table predicates into each scan and join conditions into
// LogicalJoin nodes.
func buildMultiTablePlan(scans []*LogicalScan, schemas []exec.Schema, where sql.Expr, tableRefs []sql.TableRef) (LogicalNode, error) {
	st := newSymbolTable(schemas, tableRefs)

	// Validate all column references in WHERE upfront.
	if where != nil {
		for _, ref := range predicateColRefs(where) {
			if _, err := st.resolve(ref); err != nil {
				return nil, fmt.Errorf("planner: %w", err)
			}
		}
	}

	// Partition WHERE terms into per-table filters and join conditions.
	perTableFilters := make([]sql.Expr, len(scans))
	var joinConds []sql.Expr
	if where != nil {
		for _, term := range flattenAnd(where) {
			lt, rt, ok := isEqualityJoinCond(term, st)
			if ok {
				_ = lt
				_ = rt
				joinConds = append(joinConds, term)
			} else {
				refs := predicateColRefs(term)
				tableSet := make(map[int]bool)
				for _, ref := range refs {
					if t, found := st.resolveTableIdx(ref); found {
						tableSet[t] = true
					}
				}
				if len(tableSet) == 1 {
					for t := range tableSet {
						perTableFilters[t] = andExpr(perTableFilters[t], term)
					}
				}
			}
		}
	}

	// Push per-table filters into scan predicates.
	for i, f := range perTableFilters {
		if f != nil {
			scans[i].Predicate = f
		}
	}

	// Build left-deep join tree; returns error for disconnected (cross-join) tables.
	return buildJoinTree(scans, joinConds, st)
}

// flattenAnd flattens a nested AND tree into a list of terms.
func flattenAnd(e sql.Expr) []sql.Expr {
	if e == nil {
		return nil
	}
	bin, ok := e.(*sql.BinaryExpr)
	if !ok || bin.Op != sql.OpAnd {
		return []sql.Expr{e}
	}
	return append(flattenAnd(bin.Left), flattenAnd(bin.Right)...)
}

// andExpr combines two expressions with AND (nil-safe).
func andExpr(a, b sql.Expr) sql.Expr {
	if a == nil {
		return b
	}
	return &sql.BinaryExpr{Op: sql.OpAnd, Left: a, Right: b}
}

// isEqualityJoinCond returns true if e is a col1 = col2 predicate where col1 and
// col2 belong to different tables.
func isEqualityJoinCond(e sql.Expr, st *symbolTable) (leftTable, rightTable int, ok bool) {
	bin, isBin := e.(*sql.BinaryExpr)
	if !isBin || bin.Op != sql.OpEQ {
		return 0, 0, false
	}
	lCR, lok := bin.Left.(*sql.ColumnRefExpr)
	rCR, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		return 0, 0, false
	}
	lt, lfound := st.resolveTableIdx(lCR)
	rt, rfound := st.resolveTableIdx(rCR)
	if !lfound || !rfound || lt == rt {
		return 0, 0, false
	}
	return lt, rt, true
}

// joinCondPair describes an equality join condition between two table indices.
type joinCondPair struct {
	leftTable, rightTable int
	leftCol, rightCol     string
}

// buildJoinTree builds a left-deep join tree using symbolTable for column resolution.
// Returns an error if tables cannot be connected (cross joins unsupported).
func buildJoinTree(scans []*LogicalScan, joinConds []sql.Expr, st *symbolTable) (LogicalNode, error) {
	// Parse join conditions into pairs.
	pairs := make([]joinCondPair, 0, len(joinConds))
	for _, cond := range joinConds {
		bin := cond.(*sql.BinaryExpr)
		lCR := bin.Left.(*sql.ColumnRefExpr)
		rCR := bin.Right.(*sql.ColumnRefExpr)
		lt, _ := st.resolveTableIdx(lCR)
		rt, _ := st.resolveTableIdx(rCR)
		pairs = append(pairs, joinCondPair{
			leftTable:  lt,
			rightTable: rt,
			leftCol:    lCR.Name,
			rightCol:   rCR.Name,
		})
	}

	// Build left-deep tree: start with scan 0, repeatedly find a join condition
	// that connects a new scan to the already-included set.
	included := map[int]bool{0: true}
	var root LogicalNode = scans[0]

	for len(included) < len(scans) {
		joined := false
		for _, pair := range pairs {
			var newTable int
			var joinColInTree, joinColNew string
			switch {
			case included[pair.leftTable] && !included[pair.rightTable]:
				newTable = pair.rightTable
				joinColInTree = pair.leftCol
				joinColNew = pair.rightCol
			case included[pair.rightTable] && !included[pair.leftTable]:
				newTable = pair.leftTable
				joinColInTree = pair.rightCol
				joinColNew = pair.leftCol
			default:
				continue
			}
			cond := &sql.BinaryExpr{
				Op:    sql.OpEQ,
				Left:  &sql.ColumnRefExpr{Name: joinColInTree},
				Right: &sql.ColumnRefExpr{Name: joinColNew},
			}
			root = &LogicalJoin{Left: root, Right: scans[newTable], Condition: cond}
			included[newTable] = true
			joined = true
			break
		}
		if !joined {
			// No join condition found — cross joins not yet supported.
			for i := range scans {
				if !included[i] {
					return nil, fmt.Errorf("planner: no join condition connects table %q to the query; cross joins are not supported", scans[i].TableName)
				}
			}
		}
	}
	return root, nil
}

func buildProject(child LogicalNode, stmt *sql.SelectStmt) (*LogicalProject, error) {
	schema := child.OutputSchema()
	var items []ProjectItem
	for _, col := range stmt.Columns {
		alias := col.Alias
		if alias == "" {
			alias = exprName(col.Expr)
		}
		t := resolveExprType(col.Expr, schema)
		items = append(items, ProjectItem{Alias: alias, Expr: col.Expr, Type: t})
	}
	return &LogicalProject{Child: child, Exprs: items}, nil
}

func buildAggregate(child LogicalNode, stmt *sql.SelectStmt) (*LogicalAggregate, error) {
	schema := child.OutputSchema()
	var aggs []AggItem
	for _, col := range stmt.Columns {
		ae, ok := col.Expr.(*sql.AggFuncExpr)
		if !ok {
			continue // group-by columns handled separately
		}
		if ae.Distinct {
			return nil, fmt.Errorf("planner: DISTINCT aggregates not yet supported (e.g. %s(DISTINCT ...))", ae.Func)
		}
		alias := col.Alias
		if alias == "" {
			alias = exprName(col.Expr)
		}
		colName := ""
		var aggExpr sql.Expr
		if ae.Arg != nil {
			switch arg := ae.Arg.(type) {
			case *sql.StarExpr:
				// COUNT(*) — no source column.
			case *sql.ColumnRefExpr:
				colName = arg.Name
			default:
				// Complex expression (e.g. l_extendedprice * (1 - l_discount)).
				// Generate a synthetic column name; the physical planner will
				// insert a pre-projection to compute it.
				colName = fmt.Sprintf("_agg_%d", len(aggs))
				aggExpr = ae.Arg
			}
		}
		_ = schema
		aggs = append(aggs, AggItem{Func: ae.Func, ColName: colName, AggExpr: aggExpr, Alias: alias})
	}
	return &LogicalAggregate{
		Child:   child,
		GroupBy: stmt.GroupBy,
		Aggs:    aggs,
	}, nil
}

func isSelectStar(cols []sql.SelectColumn) bool {
	return len(cols) == 1 && isStarExpr(cols[0].Expr)
}

func isStarExpr(e sql.Expr) bool {
	_, ok := e.(*sql.StarExpr)
	return ok
}

func hasAggregates(cols []sql.SelectColumn) bool {
	for _, col := range cols {
		if _, ok := col.Expr.(*sql.AggFuncExpr); ok {
			return true
		}
	}
	return false
}

func resolveExprType(expr sql.Expr, schema Schema) DataType {
	switch e := expr.(type) {
	case *sql.ColumnRefExpr:
		for _, f := range schema.Fields {
			if f.Name == e.Name {
				return f.Type
			}
		}
	case *sql.IntLiteral:
		return TypeInt64
	case *sql.FloatLiteral:
		return TypeFloat64
	case *sql.StringLiteral:
		return TypeString
	case *sql.BoolLiteral:
		return TypeBool
	case *sql.BinaryExpr:
		l := resolveExprType(e.Left, schema)
		r := resolveExprType(e.Right, schema)
		if l == TypeFloat64 || r == TypeFloat64 {
			return TypeFloat64
		}
		return l
	case *sql.AggFuncExpr:
		switch e.Func {
		case "COUNT":
			return TypeInt64
		case "AVG":
			return TypeFloat64
		default:
			if e.Arg != nil {
				return resolveExprType(e.Arg, schema)
			}
		}
	case *sql.CaseExpr:
		// Determine result type from the first WHEN result.
		for _, w := range e.Whens {
			t := resolveExprType(w.Result, schema)
			if t != 0 {
				return t
			}
		}
	}
	return TypeInt64
}

// rewriteHavingAggs walks a HAVING expression tree and replaces every
// AggFuncExpr with a ColumnRefExpr pointing to a matching aggregate output
// column. If no matching aggregate exists in the SELECT list, a hidden
// aggregate is appended to the LogicalAggregate node.
func rewriteHavingAggs(expr sql.Expr, agg *LogicalAggregate) sql.Expr {
	if expr == nil {
		return nil
	}
	switch e := expr.(type) {
	case *sql.AggFuncExpr:
		// Find a matching aggregate in the existing list.
		alias := findMatchingAgg(e, agg.Aggs)
		if alias != "" {
			return &sql.ColumnRefExpr{Name: alias}
		}
		// No match — add a hidden aggregate.
		alias = fmt.Sprintf("_having_agg_%d", len(agg.Aggs))
		colName := ""
		var aggExpr sql.Expr
		if e.Arg != nil {
			switch arg := e.Arg.(type) {
			case *sql.StarExpr:
				// COUNT(*) — no source column.
			case *sql.ColumnRefExpr:
				colName = arg.Name
			default:
				colName = alias
				aggExpr = e.Arg
			}
		}
		agg.Aggs = append(agg.Aggs, AggItem{
			Func:    e.Func,
			ColName: colName,
			AggExpr: aggExpr,
			Alias:   alias,
		})
		return &sql.ColumnRefExpr{Name: alias}

	case *sql.BinaryExpr:
		return &sql.BinaryExpr{
			Op:    e.Op,
			Left:  rewriteHavingAggs(e.Left, agg),
			Right: rewriteHavingAggs(e.Right, agg),
		}
	case *sql.UnaryExpr:
		return &sql.UnaryExpr{
			Op:   e.Op,
			Expr: rewriteHavingAggs(e.Expr, agg),
		}
	case *sql.IsNullExpr:
		return &sql.IsNullExpr{
			Expr:  rewriteHavingAggs(e.Expr, agg),
			IsNot: e.IsNot,
		}
	case *sql.BetweenExpr:
		return &sql.BetweenExpr{
			Expr: rewriteHavingAggs(e.Expr, agg),
			Lo:   rewriteHavingAggs(e.Lo, agg),
			Hi:   rewriteHavingAggs(e.Hi, agg),
			Not:  e.Not,
		}
	default:
		// Literals, column references, etc. — return unchanged.
		return expr
	}
}

// findMatchingAgg checks if an AggFuncExpr structurally matches any existing
// AggItem. A match means same function name and same argument structure.
func findMatchingAgg(ae *sql.AggFuncExpr, aggs []AggItem) string {
	for _, a := range aggs {
		if a.Func != ae.Func {
			continue
		}
		// Match argument structure.
		if ae.Arg == nil {
			// COUNT(*) with nil arg — matches COUNT with no source column.
			if a.ColName == "" && a.AggExpr == nil {
				return a.Alias
			}
			continue
		}
		switch arg := ae.Arg.(type) {
		case *sql.StarExpr:
			if a.ColName == "" && a.AggExpr == nil {
				return a.Alias
			}
		case *sql.ColumnRefExpr:
			if a.ColName == arg.Name && a.AggExpr == nil {
				return a.Alias
			}
		default:
			// Complex expression — structural match against AggExpr.
			// For now, only match if both are the exact same pointer
			// (which they will be if the optimizer didn't clone).
			// In practice, HAVING expressions with complex args that aren't
			// in SELECT will create hidden aggregates, which is correct.
			if a.AggExpr == arg {
				return a.Alias
			}
		}
	}
	return ""
}

// buildHavingProjection creates a LogicalProject that strips hidden aggregate
// columns from the output. It projects only the GROUP BY columns plus the
// original aggregates (those before the hidden ones were appended).
func buildHavingProjection(child LogicalNode, agg *LogicalAggregate, origAggCount int) *LogicalProject {
	childSchema := child.OutputSchema()
	var items []ProjectItem
	// The output schema of LogicalAggregate is: [group-by columns...] [aggregate columns...]
	// We want to project everything except the hidden aggregates (indices after origAggCount).
	numGroupBy := len(agg.GroupBy)
	numOrigCols := numGroupBy + origAggCount
	for i := 0; i < numOrigCols; i++ {
		f := childSchema.Fields[i]
		items = append(items, ProjectItem{
			Alias: f.Name,
			Expr:  &sql.ColumnRefExpr{Name: f.Name},
			Type:  f.Type,
		})
	}
	return &LogicalProject{Child: child, Exprs: items}
}
