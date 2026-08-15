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
//
// needed == nil means all columns are required; a non-nil but empty set means no
// column from below is required (`SELECT COUNT(*)`). Nodes that pass their input
// through unchanged (Filter, Sort, Limit, Distinct) must preserve the nil
// convention; nodes that fully determine their output from their own expressions
// (Project, Aggregate) compute a fresh non-nil set and ignore needed. Joins
// split the set across their two sides — see pruneJoinColumns.
//
// A scan cannot express "read zero columns" — exec.NewTableScan reads every
// column when handed an empty projection — so an empty set reaching a single
// table still decodes all of it. Under a join it does not: the join adds its key
// columns first, so each side decodes exactly its key.
func pruneColumns(node LogicalNode, needed []string) LogicalNode {
	switch n := node.(type) {
	case *LogicalScan:
		if needed != nil {
			// Merge with predicate columns: the pushed-down predicate is
			// evaluated against the scan's projected schema.
			n.NeededCols = addNeeded(needed, predicateCols(n.Predicate))
		}
		return n

	case *LogicalFilter:
		// Filter needs its own predicate columns + whatever parent needs.
		predCols := predicateCols(n.Predicate)
		childNeeded := addNeeded(needed, predCols)
		return &LogicalFilter{
			Child:     pruneColumns(n.Child, childNeeded),
			Predicate: n.Predicate,
		}

	case *LogicalProject:
		// Project needs its expression columns. The set is non-nil even when
		// empty (`SELECT 1 FROM ...`): empty means "no columns from below", not
		// "every column".
		exprCols := []string{}
		for _, pe := range n.Exprs {
			exprCols = append(exprCols, predicateCols(pe.Expr)...)
		}
		return &LogicalProject{
			Child: pruneColumns(n.Child, uniqueStrings(exprCols)),
			Exprs: n.Exprs,
		}

	case *LogicalAggregate:
		// Non-nil even when empty: `SELECT COUNT(*)` with no GROUP BY needs no
		// column from below, which a nil set would misread as "every column".
		cols := []string{}
		for _, gb := range n.GroupBy {
			cols = append(cols, predicateCols(gb)...)
		}
		for _, agg := range n.Aggs {
			if agg.AggExpr != nil {
				// Complex expression (e.g. SUM(price * discount)): collect the
				// real source columns referenced by the expression, not the
				// synthetic column name (_agg_0) which only exists after
				// buildPreProjection materializes it.
				cols = append(cols, predicateCols(agg.AggExpr)...)
			} else if agg.ColName != "" {
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
		childNeeded := addNeeded(needed, sortCols)
		return &LogicalSort{Child: pruneColumns(n.Child, childNeeded), OrderBy: n.OrderBy}

	case *LogicalLimit:
		return &LogicalLimit{Child: pruneColumns(n.Child, needed), Count: n.Count}

	case *LogicalDistinct:
		// Distinct passes every column of its input through unchanged, so the
		// needed set reaches its child untouched.
		return &LogicalDistinct{Child: pruneColumns(n.Child, needed)}

	case *LogicalJoin:
		return pruneJoinColumns(n, needed)
	}
	return node
}

// pruneJoinColumns splits a needed-column set across the two sides of a join so
// each side's scan decodes only the columns the query actually reads.
//
// A side needs three groups of columns, and all three are already present in the
// set this function works from:
//
//   - its join key, added here from the join condition;
//   - whatever ancestors asked for — SELECT list, aggregate source columns
//     (including the real columns behind an expression aggregate), GROUP BY,
//     HAVING and ORDER BY all reach this node through `needed`;
//   - its own filter predicate columns, which the LogicalScan case merges in
//     after this function has recursed into the scan.
//
// Membership is decided by name against each side's output schema. Names are
// unqualified (buildJoinTree rewrites join conditions to bare names, and
// collectCols drops the table qualifier), so a column name present in both
// tables is kept on both sides. That over-approximates rather than under-prunes,
// which is the safe direction.
//
// No index remapping is needed downstream: every consumer of a join —
// physicalJoin, tryParallelJoin, resolveAggConfig, buildExecExpr and the sort
// key resolver — locates columns with Schema.IndexOf on the operator's runtime
// schema, which is derived from NeededCols. Shrinking a scan's projection
// therefore shifts those indices consistently for planner and executor alike.
func pruneJoinColumns(n *LogicalJoin, needed []string) LogicalNode {
	// A nil needed set means an ancestor requires every column (SELECT *), so
	// neither side may be pruned.
	if needed == nil {
		return &LogicalJoin{
			Left:      pruneColumns(n.Left, nil),
			Right:     pruneColumns(n.Right, nil),
			Condition: n.Condition,
		}
	}

	want := make(map[string]bool, len(needed)+2)
	for _, name := range needed {
		want[name] = true
	}
	// The join keys are consumed by the join itself, so they are required even
	// when the query never selects them.
	for _, name := range predicateCols(n.Condition) {
		want[name] = true
	}

	// Children have not been visited yet, so OutputSchema still reports every
	// column each side can supply. Each side gets its own freshly allocated
	// slice: uniqueStrings writes through its argument's backing array, so the
	// two must not share one.
	leftNeeded := schemaColsIn(n.Left.OutputSchema(), want)
	rightNeeded := schemaColsIn(n.Right.OutputSchema(), want)

	return &LogicalJoin{
		Left:      pruneColumns(n.Left, leftNeeded),
		Right:     pruneColumns(n.Right, rightNeeded),
		Condition: n.Condition,
	}
}

// addNeeded merges extra column names into a needed set, preserving the
// "nil means every column" convention: merging into nil would narrow the set to
// just extra, silently dropping the columns the ancestor still requires. It
// always allocates, so the result never aliases needed's backing array.
func addNeeded(needed, extra []string) []string {
	if needed == nil {
		return nil
	}
	merged := make([]string, 0, len(needed)+len(extra))
	merged = append(merged, needed...)
	merged = append(merged, extra...)
	return uniqueStrings(merged)
}

// schemaColsIn returns the names of schema's fields that appear in want, in
// schema order. It returns nil when nothing matches, which callers and
// LogicalScan both read as "every column" — the safe interpretation.
func schemaColsIn(schema Schema, want map[string]bool) []string {
	var out []string
	for _, f := range schema.Fields {
		if want[f.Name] {
			out = append(out, f.Name)
		}
	}
	return out
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
