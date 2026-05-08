package planner

import "github.com/ryderpongracic1/vexq/sql"

// Optimize applies rule-based transformations to the logical plan:
//  1. Predicate pushdown: push filters into the scan.
//  2. Column pruning: compute needed columns top-down, restrict scan projection.
//  3. Zone-map pruning is applied during physical planning (at runtime).
func Optimize(root LogicalNode) LogicalNode {
	root = pushPredicates(root)
	root = pruneColumns(root, nil)
	return root
}

// ---- Predicate pushdown ----------------------------------------------------

// pushPredicates walks the tree and merges Filter predicates into the scan
// immediately below them.
func pushPredicates(node LogicalNode) LogicalNode {
	switch n := node.(type) {
	case *LogicalFilter:
		child := pushPredicates(n.Child)
		// If the child is a scan, push the predicate into it.
		if scan, ok := child.(*LogicalScan); ok {
			if scan.Predicate == nil {
				scan.Predicate = n.Predicate
			} else {
				scan.Predicate = &sql.BinaryExpr{
					Op:    sql.OpAnd,
					Left:  scan.Predicate,
					Right: n.Predicate,
				}
			}
			return scan // Filter node eliminated
		}
		return &LogicalFilter{Child: child, Predicate: n.Predicate}

	case *LogicalProject:
		return &LogicalProject{Child: pushPredicates(n.Child), Exprs: n.Exprs}
	case *LogicalAggregate:
		return &LogicalAggregate{Child: pushPredicates(n.Child), GroupBy: n.GroupBy, Aggs: n.Aggs}
	case *LogicalSort:
		return &LogicalSort{Child: pushPredicates(n.Child), OrderBy: n.OrderBy}
	case *LogicalLimit:
		return &LogicalLimit{Child: pushPredicates(n.Child), Count: n.Count}
	case *LogicalJoin:
		return &LogicalJoin{Left: pushPredicates(n.Left), Right: pushPredicates(n.Right), Condition: n.Condition}
	}
	return node
}

// ---- Column pruning --------------------------------------------------------

// pruneColumns propagates the set of needed column names top-down.
// needed == nil means all columns are required.
func pruneColumns(node LogicalNode, needed []string) LogicalNode {
	switch n := node.(type) {
	case *LogicalScan:
		if needed != nil {
			// Merge with predicate columns.
			needed = uniqueStrings(append(needed, predicateCols(n.Predicate)...))
			n.NeededCols = needed
		}
		return n

	case *LogicalFilter:
		// Filter needs its own predicate columns + whatever parent needs.
		predCols := predicateCols(n.Predicate)
		childNeeded := uniqueStrings(append(needed, predCols...))
		return &LogicalFilter{
			Child:     pruneColumns(n.Child, childNeeded),
			Predicate: n.Predicate,
		}

	case *LogicalProject:
		// Project needs its expression columns.
		var exprCols []string
		for _, pe := range n.Exprs {
			exprCols = append(exprCols, predicateCols(pe.Expr)...)
		}
		return &LogicalProject{
			Child: pruneColumns(n.Child, uniqueStrings(exprCols)),
			Exprs: n.Exprs,
		}

	case *LogicalAggregate:
		var cols []string
		for _, gb := range n.GroupBy {
			cols = append(cols, predicateCols(gb)...)
		}
		for _, agg := range n.Aggs {
			if agg.ColName != "" {
				cols = append(cols, agg.ColName)
			}
		}
		return &LogicalAggregate{
			Child:   pruneColumns(n.Child, uniqueStrings(cols)),
			GroupBy: n.GroupBy,
			Aggs:    n.Aggs,
		}

	case *LogicalSort:
		var sortCols []string
		for _, ob := range n.OrderBy {
			sortCols = append(sortCols, predicateCols(ob.Expr)...)
		}
		childNeeded := uniqueStrings(append(needed, sortCols...))
		return &LogicalSort{Child: pruneColumns(n.Child, childNeeded), OrderBy: n.OrderBy}

	case *LogicalLimit:
		return &LogicalLimit{Child: pruneColumns(n.Child, needed), Count: n.Count}

	case *LogicalJoin:
		return &LogicalJoin{
			Left:      pruneColumns(n.Left, nil),
			Right:     pruneColumns(n.Right, nil),
			Condition: n.Condition,
		}
	}
	return node
}

// predicateCols returns column names referenced in a SQL expression.
func predicateCols(e sql.Expr) []string {
	if e == nil {
		return nil
	}
	var cols []string
	collectCols(e, &cols)
	return cols
}

func collectCols(e sql.Expr, out *[]string) {
	if e == nil {
		return
	}
	switch x := e.(type) {
	case *sql.ColumnRefExpr:
		*out = append(*out, x.Name)
	case *sql.BinaryExpr:
		collectCols(x.Left, out)
		collectCols(x.Right, out)
	case *sql.UnaryExpr:
		collectCols(x.Expr, out)
	case *sql.IsNullExpr:
		collectCols(x.Expr, out)
	case *sql.BetweenExpr:
		collectCols(x.Expr, out)
		collectCols(x.Lo, out)
		collectCols(x.Hi, out)
	case *sql.InExpr:
		collectCols(x.Expr, out)
		for _, item := range x.List {
			collectCols(item, out)
		}
	case *sql.LikeExpr:
		collectCols(x.Expr, out)
	case *sql.AggFuncExpr:
		collectCols(x.Arg, out)
	case *sql.CaseExpr:
		for _, w := range x.Whens {
			collectCols(w.Cond, out)
			collectCols(w.Result, out)
		}
		collectCols(x.Else, out)
	}
}

func uniqueStrings(ss []string) []string {
	seen := make(map[string]bool, len(ss))
	out := ss[:0]
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}
