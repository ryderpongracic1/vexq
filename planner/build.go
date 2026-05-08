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
		root = buildMultiTablePlan(scans, schemas, stmt.Where)
	}

	// GROUP BY / aggregates.
	hasAggs := hasAggregates(stmt.Columns)
	if hasAggs || len(stmt.GroupBy) > 0 {
		agg, err := buildAggregate(root, stmt)
		if err != nil {
			return nil, err
		}
		root = agg
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

// buildMultiTablePlan builds a left-deep join tree from multiple table scans,
// pushing single-table predicates into each scan and join conditions into
// LogicalJoin nodes.
func buildMultiTablePlan(scans []*LogicalScan, schemas []exec.Schema, where sql.Expr) LogicalNode {
	// Map column name → table index for all tables.
	colTable := make(map[string]int)
	for i, s := range schemas {
		for _, f := range s.Fields {
			colTable[f.Name] = i
		}
	}

	// Partition WHERE terms into per-table filters and join conditions.
	perTableFilters := make([]sql.Expr, len(scans))
	var joinConds []sql.Expr
	if where != nil {
		for _, term := range flattenAnd(where) {
			lt, rt, ok := isEqualityJoinCond(term, colTable)
			if ok {
				_ = lt
				_ = rt
				joinConds = append(joinConds, term)
			} else {
				// Find which table(s) this term references.
				cols := predicateCols(term)
				tableSet := make(map[int]bool)
				for _, col := range cols {
					if t, found := colTable[col]; found {
						tableSet[t] = true
					}
				}
				if len(tableSet) == 1 {
					for t := range tableSet {
						perTableFilters[t] = andExpr(perTableFilters[t], term)
					}
				}
				// Predicates crossing multiple tables that are not equality join
				// conditions (e.g. l_commitdate < l_receiptdate on same table) are
				// handled by the single-table branch above; genuine cross-table
				// non-equality conditions are rare in TPC-H and left as future work.
			}
		}
	}

	// Push per-table filters into scan predicates.
	for i, f := range perTableFilters {
		if f != nil {
			scans[i].Predicate = f
		}
	}

	// Build left-deep join tree.
	return buildJoinTree(scans, joinConds, colTable)
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
func isEqualityJoinCond(e sql.Expr, colTable map[string]int) (leftTable, rightTable int, ok bool) {
	bin, isBin := e.(*sql.BinaryExpr)
	if !isBin || bin.Op != sql.OpEQ {
		return 0, 0, false
	}
	lCR, lok := bin.Left.(*sql.ColumnRefExpr)
	rCR, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		return 0, 0, false
	}
	lt, lfound := colTable[lCR.Name]
	rt, rfound := colTable[rCR.Name]
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

// buildJoinTree builds a left-deep join tree from the given scans and join conditions.
func buildJoinTree(scans []*LogicalScan, joinConds []sql.Expr, colTable map[string]int) LogicalNode {
	// Parse join conditions into pairs.
	pairs := make([]joinCondPair, 0, len(joinConds))
	for _, cond := range joinConds {
		bin := cond.(*sql.BinaryExpr)
		lCR := bin.Left.(*sql.ColumnRefExpr)
		rCR := bin.Right.(*sql.ColumnRefExpr)
		pairs = append(pairs, joinCondPair{
			leftTable:  colTable[lCR.Name],
			rightTable: colTable[rCR.Name],
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
			// No join condition found — add the first unincluded table as a cross join.
			for i := range scans {
				if !included[i] {
					root = &LogicalJoin{
						Left:      root,
						Right:     scans[i],
						Condition: &sql.BinaryExpr{Op: sql.OpEQ, Left: &sql.IntLiteral{Value: 1}, Right: &sql.IntLiteral{Value: 1}},
					}
					included[i] = true
					break
				}
			}
		}
	}
	return root
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
