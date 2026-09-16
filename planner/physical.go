package planner

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

var epoch = time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)

// Physical converts a logical plan into a physical operator tree.
func Physical(ctx context.Context, node LogicalNode) (exec.Operator, error) {
	switch n := node.(type) {
	case *LogicalScan:
		return physicalScan(ctx, n)
	case *LogicalFilter:
		return physicalFilter(ctx, n)
	case *LogicalProject:
		return physicalProject(ctx, n)
	case *LogicalAggregate:
		return physicalAggregate(ctx, n)
	case *LogicalSort:
		return physicalSort(ctx, n)
	case *LogicalLimit:
		return physicalLimit(ctx, n)
	case *LogicalJoin:
		return physicalJoin(ctx, n)
	case *LogicalDistinct:
		return physicalDistinct(ctx, n)
	case nil:
		return nil, fmt.Errorf("planner: nil logical node")
	}
	return nil, fmt.Errorf("planner: unknown logical node %T", node)
}

func physicalScan(ctx context.Context, n *LogicalScan) (exec.Operator, error) {
	r, err := storage.Open(ctx, n.FilePath)
	if err != nil {
		return nil, fmt.Errorf("planner: open %q: %w", n.FilePath, err)
	}

	// Zone-map predicate from the pushed-down filter.
	var zonePred exec.ZonePredicate
	if n.Predicate != nil {
		zonePred = buildZonePredicate(n.Predicate, n.Schema)
	}

	scan, err := n.openScan(r, zonePred, 0, len(r.Meta().RowGroups))
	if err != nil {
		_ = r.Close()
		return nil, fmt.Errorf("planner: scan %q: %w", n.TableName, err)
	}

	// If there's a pushed-down predicate, wrap the scan in a Filter.
	if n.Predicate != nil {
		exprTree, err := buildExecExpr(n.Predicate, scan.Schema())
		if err != nil {
			_ = scan.Close()
			return nil, err
		}
		return exec.NewFilter(scan, exprTree)
	}
	return scan, nil
}

func physicalFilter(ctx context.Context, n *LogicalFilter) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return buildFilterOp(n, child)
}

// buildFilterOp wraps an already-constructed child operator with a Filter for n.
// Used by both physicalFilter and the parallel planner's factory closure.
func buildFilterOp(n *LogicalFilter, child exec.Operator) (exec.Operator, error) {
	exprTree, err := buildExecExpr(n.Predicate, child.Schema())
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	return exec.NewFilter(child, exprTree)
}

func physicalProject(ctx context.Context, n *LogicalProject) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return buildProjectOp(n, child)
}

// buildProjectOp wraps an already-constructed child operator with a Project for n.
// Used by both physicalProject and the parallel planner's factory closure.
func buildProjectOp(n *LogicalProject, child exec.Operator) (exec.Operator, error) {
	schema := child.Schema()
	var projExprs []exec.ProjectExpr
	for _, pe := range n.Exprs {
		e, err := buildExecExpr(pe.Expr, schema)
		if err != nil {
			_ = child.Close()
			return nil, err
		}
		projExprs = append(projExprs, exec.ProjectExpr{Name: pe.Alias, Expr: e})
	}
	return exec.NewProject(child, projExprs)
}

func physicalAggregate(ctx context.Context, n *LogicalAggregate) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	child, err = buildPreProjection(n, child)
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	groupByIdxs, aggExprs, err := resolveAggConfig(n, child.Schema())
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	return exec.NewHashAggregate(child, groupByIdxs, aggExprs)
}

// buildPreProjection inserts a synthetic projection before the aggregate when any
// AggItem uses a complex expression (not a plain column reference). In that case
// every existing column is passed through unchanged, and each complex expression
// is appended as a synthetic column named by AggItem.ColName.
// If no complex expressions are present, child is returned unchanged.
func buildPreProjection(n *LogicalAggregate, child exec.Operator) (exec.Operator, error) {
	hasComplex := false
	for _, agg := range n.Aggs {
		if agg.AggExpr != nil {
			hasComplex = true
			break
		}
	}
	if !hasComplex {
		return child, nil
	}
	schema := child.Schema()
	projExprs := make([]exec.ProjectExpr, 0, len(schema.Fields)+len(n.Aggs))
	for _, f := range schema.Fields {
		idx := schema.IndexOf(f.Name)
		projExprs = append(projExprs, exec.ProjectExpr{
			Name: f.Name,
			Expr: &exec.ColumnRef{Name: f.Name, Idx: idx, T: f.Type},
		})
	}
	for i := range n.Aggs {
		if n.Aggs[i].AggExpr == nil {
			continue
		}
		synExpr, err := buildExecExpr(n.Aggs[i].AggExpr, schema)
		if err != nil {
			return nil, fmt.Errorf("planner: aggregate expr %q: %w", n.Aggs[i].Alias, err)
		}
		projExprs = append(projExprs, exec.ProjectExpr{Name: n.Aggs[i].ColName, Expr: synExpr})
	}
	return exec.NewProject(child, projExprs)
}

// resolveAggConfig resolves group-by column indices and aggregate descriptors
// (including AccumType for correct parallel merge) from n given the schema that
// will be presented to the aggregate operator (post-pre-projection if any).
// Called by both physicalAggregate and Parallel.
func resolveAggConfig(n *LogicalAggregate, schema exec.Schema) (groupByIdxs []int, aggExprs []exec.AggExpr, err error) {
	// Known limitation — GROUP BY resolves only against the child's schema.
	//
	// Two things are unsupported here, and a GROUP BY on a CASE WHEN output alias
	// needs both, which is why it is not a small fix:
	//
	//  1. Output aliases. buildAggregate (build.go) passes stmt.GroupBy through
	//     verbatim, so nothing maps a SELECT-list alias back to the expression it
	//     names. This is not specific to CASE WHEN — `SELECT c AS x ... GROUP BY x`
	//     fails for a plain column too. The error a user sees comes from further
	//     down: pruneColumns collects predicateCols(gb) into the scan's
	//     needed-column set, so an alias reaches TableScan as a column name and
	//     the failure reads `scan "t": column "x" not found` rather than naming
	//     GROUP BY.
	//  2. Expressions. Resolving the alias in case 1 yields a non-ColumnRefExpr,
	//     which the check below rejects. Supporting it needs the same treatment
	//     aggregate arguments get — a pre-projection materialising the expression
	//     into a synthetic column (buildPreProjection, the _agg_N path), the
	//     expression's real source columns pushed to the scan instead of the
	//     synthetic name, the group-by output column typed and named from the
	//     alias, and the equivalent per-morsel pre-projection on the parallel
	//     path (planner/parallel.go).
	//
	// Fixing only (1) would move a CASE WHEN alias from "column not found" to
	// "GROUP BY only supports column references" without making the query work,
	// so both belong in one change, scoped as a feature rather than a bug fix.
	for _, gbExpr := range n.GroupBy {
		cr, ok := gbExpr.(*sql.ColumnRefExpr)
		if !ok {
			return nil, nil, fmt.Errorf("planner: GROUP BY only supports column references")
		}
		idx := schema.IndexOf(cr.Name)
		if idx < 0 {
			return nil, nil, fmt.Errorf("planner: GROUP BY column %q not found", cr.Name)
		}
		groupByIdxs = append(groupByIdxs, idx)
	}

	for i := range n.Aggs {
		agg := &n.Aggs[i]
		ae := exec.AggExpr{OutName: agg.Alias, Distinct: agg.Distinct}
		switch agg.Func {
		case "COUNT":
			if agg.Distinct {
				ae.Kind = exec.AggCountDistinct
			} else {
				ae.Kind = exec.AggCount
			}
			ae.AccumType = exec.TypeInt64
			if agg.ColName == "" {
				ae.ColIdx = -1
			} else {
				ae.ColIdx = schema.IndexOf(agg.ColName)
			}
		case "SUM":
			ae.Kind = exec.AggSum
			ae.ColIdx = schema.IndexOf(agg.ColName)
		case "AVG":
			ae.Kind = exec.AggAvg
			ae.ColIdx = schema.IndexOf(agg.ColName)
		case "MIN":
			ae.Kind = exec.AggMin
			ae.ColIdx = schema.IndexOf(agg.ColName)
		case "MAX":
			ae.Kind = exec.AggMax
			ae.ColIdx = schema.IndexOf(agg.ColName)
		default:
			return nil, nil, fmt.Errorf("planner: unknown aggregate %q", agg.Func)
		}
		// Validate that the column was found for non-COUNT(*) aggregates.
		if ae.ColIdx == -1 && !(ae.Kind == exec.AggCount && agg.ColName == "") {
			return nil, nil, fmt.Errorf("planner: aggregate column %q not found in schema", agg.ColName)
		}
		// Resolve the accumulator encoding. Shared with exec.NewHashAggregate
		// via exec.AccumTypeFor so the serial and parallel paths cannot
		// disagree: the parallel path builds its per-worker partial aggregates
		// directly from these AggExprs, never through NewHashAggregate.
		srcType := exec.TypeInt64
		if ae.ColIdx >= 0 {
			srcType = schema.Fields[ae.ColIdx].Type
		}
		ae.AccumType = exec.AccumTypeFor(ae.Kind, srcType)
		aggExprs = append(aggExprs, ae)
	}
	return groupByIdxs, aggExprs, nil
}

func physicalSort(ctx context.Context, n *LogicalSort) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return buildSortOp(n, child)
}

// buildSortOp wraps an already-constructed child operator with a sort for n,
// closing child on failure.
func buildSortOp(n *LogicalSort, child exec.Operator) (exec.Operator, error) {
	schema := child.Schema()
	var keys []exec.SortKey
	for _, ob := range n.OrderBy {
		cr, ok := ob.Expr.(*sql.ColumnRefExpr)
		if !ok {
			_ = child.Close()
			return nil, fmt.Errorf("planner: ORDER BY only supports column references")
		}
		idx := schema.IndexOf(cr.Name)
		if idx < 0 {
			_ = child.Close()
			return nil, fmt.Errorf("planner: ORDER BY column %q not found", cr.Name)
		}
		keys = append(keys, exec.SortKey{ColIdx: idx, Descending: ob.Descending})
	}
	op, err := exec.NewExternalSort(child, keys)
	if err != nil {
		_ = child.Close()
		return nil, err
	}
	return op, nil
}

func physicalLimit(ctx context.Context, n *LogicalLimit) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return buildLimitOp(n, child), nil
}

func buildLimitOp(n *LogicalLimit, child exec.Operator) exec.Operator {
	return exec.NewLimitOffset(child, int(n.Count), int(n.Offset))
}

func physicalDistinct(ctx context.Context, n *LogicalDistinct) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return exec.NewDistinct(child), nil
}

func physicalJoin(ctx context.Context, n *LogicalJoin) (exec.Operator, error) {
	build, err := Physical(ctx, n.Left)
	if err != nil {
		return nil, err
	}
	probe, err := Physical(ctx, n.Right)
	if err != nil {
		_ = build.Close()
		return nil, err
	}
	// Extract equality condition: left_col = right_col.
	bin, ok := n.Condition.(*sql.BinaryExpr)
	if !ok || bin.Op != sql.OpEQ {
		_ = build.Close()
		_ = probe.Close()
		return nil, fmt.Errorf("planner: join condition must be a simple equality")
	}
	lCR, lok := bin.Left.(*sql.ColumnRefExpr)
	rCR, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		_ = build.Close()
		_ = probe.Close()
		return nil, fmt.Errorf("planner: join condition must be column = column")
	}
	buildKeyIdx := build.Schema().IndexOf(lCR.Name)
	probeKeyIdx := probe.Schema().IndexOf(rCR.Name)
	if buildKeyIdx < 0 {
		buildKeyIdx = build.Schema().IndexOf(rCR.Name)
		probeKeyIdx = probe.Schema().IndexOf(lCR.Name)
	}
	if buildKeyIdx < 0 || probeKeyIdx < 0 {
		_ = build.Close()
		_ = probe.Close()
		return nil, fmt.Errorf("planner: join key columns not found")
	}
	return exec.NewHashJoin(build, probe, buildKeyIdx, probeKeyIdx)
}

// ---- Expression translation (SQL AST → exec.Expr) --------------------------

func buildExecExpr(e sql.Expr, schema exec.Schema) (exec.Expr, error) {
	switch x := e.(type) {
	case *sql.ColumnRefExpr:
		// Try qualified lookup first (table.col), then bare name.
		idx := -1
		if x.Table != "" {
			idx = schema.IndexOf(x.Table + "." + x.Name)
			if idx < 0 {
				// Fall back to bare name — after join planning, the output
				// schema may carry only unqualified names.
				idx = schema.IndexOf(x.Name)
			}
		} else {
			idx = schema.IndexOf(x.Name)
		}
		if idx < 0 {
			return nil, fmt.Errorf("planner: column %q not found in schema", x.Name)
		}
		return &exec.ColumnRef{Name: x.Name, Idx: idx, T: schema.Fields[idx].Type}, nil

	case *sql.IntLiteral:
		return &exec.Literal{Val: x.Value, T: exec.TypeInt64}, nil

	case *sql.FloatLiteral:
		return &exec.Literal{Val: x.Value, T: exec.TypeFloat64}, nil

	case *sql.StringLiteral:
		return &exec.Literal{Val: x.Value, T: exec.TypeString}, nil

	case *sql.BoolLiteral:
		return &exec.Literal{Val: x.Value, T: exec.TypeBool}, nil

	case *sql.BinaryExpr:
		return buildBinExpr(x, schema)

	case *sql.UnaryExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
		}
		if x.Op == sql.OpNot {
			return &exec.NotExpr{Child: child}, nil
		}
		// Unary minus: multiply by -1, using the correct literal type to
		// match the operand.  evalArith requires both sides to be the same
		// vector type, so the literal must agree with child.Type().
		var negOne exec.Expr
		switch child.Type() {
		case exec.TypeFloat64:
			negOne = &exec.Literal{Val: float64(-1), T: exec.TypeFloat64}
		case exec.TypeInt64:
			negOne = &exec.Literal{Val: int64(-1), T: exec.TypeInt64}
		default:
			return nil, fmt.Errorf("planner: unary minus not supported for type %v", child.Type())
		}
		return &exec.BinOp{
			Op:    exec.BinMul,
			Left:  child,
			Right: negOne,
			T:     child.Type(),
		}, nil

	case *sql.IsNullExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
		}
		if x.IsNot {
			return &exec.IsNotNullExpr{Child: child}, nil
		}
		return &exec.IsNullExpr{Child: child}, nil

	case *sql.BetweenExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
		}
		lo, err := buildExecExpr(x.Lo, schema)
		if err != nil {
			return nil, err
		}
		hi, err := buildExecExpr(x.Hi, schema)
		if err != nil {
			return nil, err
		}
		// Coerce lo/hi literals to match the column type (e.g. int→date). An
		// INT64/FLOAT64 mix that remains (ax BETWEEN 1.5 AND 3.5) is compared
		// as FLOAT64, so all three operands are cast together.
		_, lo = coerceOneSide(child, lo)
		_, hi = coerceOneSide(child, hi)
		if child.Type() == exec.TypeFloat64 || lo.Type() == exec.TypeFloat64 || hi.Type() == exec.TypeFloat64 {
			child, lo, hi = castToFloat(child), castToFloat(lo), castToFloat(hi)
		}
		if _, _, err := comparableOperands(child, lo); err != nil {
			return nil, fmt.Errorf("planner: BETWEEN: %w", err)
		}
		if _, _, err := comparableOperands(child, hi); err != nil {
			return nil, fmt.Errorf("planner: BETWEEN: %w", err)
		}
		between := &exec.BetweenExpr{Child: child, Lo: lo, Hi: hi}
		if x.Not {
			return &exec.NotExpr{Child: between}, nil
		}
		return between, nil

	case *sql.InExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
		}
		inExpr := &exec.InExpr{Child: child}
		for _, item := range x.List {
			v, isNull, ok, err := inListValue(item, child.Type())
			if err != nil {
				return nil, err
			}
			switch {
			case isNull:
				inExpr.HasNull = true
			case ok:
				inExpr.Set = append(inExpr.Set, v)
			}
		}
		if x.Not {
			return &exec.NotExpr{Child: inExpr}, nil
		}
		return inExpr, nil

	case *sql.LikeExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
		}
		if child.Type() != exec.TypeString {
			return nil, fmt.Errorf("planner: LIKE requires a STRING operand, got %v", child.Type())
		}
		pattern, ok := x.Pattern.(*sql.StringLiteral)
		if !ok {
			return nil, fmt.Errorf("planner: LIKE pattern must be a string literal")
		}
		like := &exec.LikeExpr{Child: child, Pattern: pattern.Value}
		if x.Not {
			return &exec.NotExpr{Child: like}, nil
		}
		return like, nil

	case *sql.CaseExpr:
		var whens []exec.When
		for _, w := range x.Whens {
			cond, err := buildExecExpr(w.Cond, schema)
			if err != nil {
				return nil, err
			}
			result, err := buildExecExpr(w.Result, schema)
			if err != nil {
				return nil, err
			}
			whens = append(whens, exec.When{Cond: cond, Result: result})
		}
		var elseExpr exec.Expr
		if x.Else != nil {
			var err error
			elseExpr, err = buildExecExpr(x.Else, schema)
			if err != nil {
				return nil, err
			}
		}
		// Determine result type from the first WHEN branch.
		t := exec.TypeFloat64
		if len(whens) > 0 {
			t = whens[0].Result.Type()
		}
		// Validate all branches produce the same (or coercible) type.
		for i, w := range whens {
			if w.Result.Type() != t {
				return nil, fmt.Errorf(
					"planner: CASE branch %d has type %v, expected %v (all branches must have a common type)",
					i+1, w.Result.Type(), t)
			}
		}
		if elseExpr != nil && elseExpr.Type() != t {
			return nil, fmt.Errorf(
				"planner: CASE ELSE has type %v, expected %v (all branches must have a common type)",
				elseExpr.Type(), t)
		}
		return &exec.CaseExpr{Whens: whens, Else: elseExpr, T: t}, nil
	}
	return nil, fmt.Errorf("planner: unsupported expression type %T", e)
}

func buildBinExpr(x *sql.BinaryExpr, schema exec.Schema) (exec.Expr, error) {
	if x.Op == sql.OpAnd {
		l, err := buildExecExpr(x.Left, schema)
		if err != nil {
			return nil, err
		}
		r, err := buildExecExpr(x.Right, schema)
		if err != nil {
			return nil, err
		}
		return &exec.AndExpr{Children: []exec.Expr{l, r}}, nil
	}
	if x.Op == sql.OpOr {
		l, err := buildExecExpr(x.Left, schema)
		if err != nil {
			return nil, err
		}
		r, err := buildExecExpr(x.Right, schema)
		if err != nil {
			return nil, err
		}
		return &exec.OrExpr{Children: []exec.Expr{l, r}}, nil
	}
	l, err := buildExecExpr(x.Left, schema)
	if err != nil {
		return nil, err
	}
	r, err := buildExecExpr(x.Right, schema)
	if err != nil {
		return nil, err
	}
	// Type coercion: promote literals to match the column type.
	l, r = coercePair(l, r)

	// String equality/inequality: use StringEqExpr (dict-code fast path).
	if (x.Op == sql.OpEQ || x.Op == sql.OpNE) &&
		l.Type() == exec.TypeString && r.Type() == exec.TypeString {
		cr, isCol := l.(*exec.ColumnRef)
		lit, isLit := r.(*exec.Literal)
		if !isCol || !isLit {
			// Try reversed.
			cr, isCol = r.(*exec.ColumnRef)
			lit, isLit = l.(*exec.Literal)
		}
		if isCol && isLit {
			return &exec.StringEqExpr{
				ColIdx:  cr.Idx,
				Literal: lit.Val.(string),
				Negate:  x.Op == sql.OpNE,
			}, nil
		}
	}

	op, isArith, err := sqlOpToExecOp(x.Op)
	if err != nil {
		return nil, err
	}
	if !isArith {
		if l, r, err = comparableOperands(l, r); err != nil {
			return nil, fmt.Errorf("planner: %s: %w", sql.FormatExpr(x), err)
		}
		return &exec.BinOp{Op: op, Left: l, Right: r, T: exec.TypeBool}, nil
	}
	resultType := l.Type()
	if l.Type() == exec.TypeFloat64 || r.Type() == exec.TypeFloat64 {
		resultType = exec.TypeFloat64
		// Mixed-type coercion: wrap the int64 side in a cast so evalArith
		// always receives operands of matching type.
		if l.Type() == exec.TypeInt64 {
			l = &exec.CastIntToFloatExpr{Inner: l}
		}
		if r.Type() == exec.TypeInt64 {
			r = &exec.CastIntToFloatExpr{Inner: r}
		}
	}
	return &exec.BinOp{Op: op, Left: l, Right: r, T: resultType}, nil
}

func sqlOpToExecOp(op sql.BinOp) (exec.BinOpKind, bool, error) {
	switch op {
	case sql.OpEQ:
		return exec.BinEQ, false, nil
	case sql.OpNE:
		return exec.BinNE, false, nil
	case sql.OpLT:
		return exec.BinLT, false, nil
	case sql.OpLE:
		return exec.BinLE, false, nil
	case sql.OpGT:
		return exec.BinGT, false, nil
	case sql.OpGE:
		return exec.BinGE, false, nil
	case sql.OpAdd:
		return exec.BinAdd, true, nil
	case sql.OpSub:
		return exec.BinSub, true, nil
	case sql.OpMul:
		return exec.BinMul, true, nil
	case sql.OpDiv:
		return exec.BinDiv, true, nil
	}
	return 0, false, fmt.Errorf("planner: unknown binary op %q", op)
}

// ---- Zone map predicate builder --------------------------------------------

// buildZonePredicate returns a ZonePredicate function that evaluates the
// pushed-down predicate against a row group's zone map statistics.
// A row group is skipped (returns false) if the predicate provably
// cannot match any row in the row group.
func buildZonePredicate(e sql.Expr, schema exec.Schema) exec.ZonePredicate {
	return func(rg *storage.RowGroupMeta) bool {
		return zonePredEval(e, schema, rg)
	}
}

// zonePredEval returns true if the row group could contain rows matching e.
// Every answer it cannot prove is true: a pruned row group is never read, so a
// wrong false silently loses rows.
func zonePredEval(e sql.Expr, schema exec.Schema, rg *storage.RowGroupMeta) bool {
	switch x := e.(type) {
	case *sql.BinaryExpr:
		switch x.Op {
		case sql.OpAnd:
			return zonePredEval(x.Left, schema, rg) && zonePredEval(x.Right, schema, rg)
		case sql.OpOr:
			return zonePredEval(x.Left, schema, rg) || zonePredEval(x.Right, schema, rg)
		case sql.OpEQ, sql.OpNE, sql.OpLT, sql.OpLE, sql.OpGT, sql.OpGE:
			return zoneRangePred(x, schema, rg)
		}
	case *sql.BetweenExpr:
		if x.Not {
			return true
		}
		// Equivalent to col >= lo AND col <= hi.
		ge := &sql.BinaryExpr{Op: sql.OpGE, Left: x.Expr, Right: x.Lo}
		le := &sql.BinaryExpr{Op: sql.OpLE, Left: x.Expr, Right: x.Hi}
		return zoneRangePred(ge, schema, rg) && zoneRangePred(le, schema, rg)
	}
	return true // conservative: don't prune
}

// zoneRangePred evaluates a column-vs-literal comparison against a row group's
// min/max. The comparison is done in the column's own domain — signed int64 for
// INT64, float64 for FLOAT64, int32 days for DATE — because the zone map stores
// raw bits: comparing float bit patterns as integers, as this once did, orders
// negative floats wrongly and pruned row groups that held matches. STRING
// (dictionary-code) and BOOL zone maps are never used for pruning.
func zoneRangePred(x *sql.BinaryExpr, schema exec.Schema, rg *storage.RowGroupMeta) bool {
	op := x.Op
	cr, ok := x.Left.(*sql.ColumnRefExpr)
	lit := x.Right
	if !ok {
		if cr, ok = x.Right.(*sql.ColumnRefExpr); !ok {
			return true
		}
		lit = x.Left
		switch op { // literal OP col  ≡  col OP' literal
		case sql.OpLT:
			op = sql.OpGT
		case sql.OpLE:
			op = sql.OpGE
		case sql.OpGT:
			op = sql.OpLT
		case sql.OpGE:
			op = sql.OpLE
		}
	}
	colIdx := schema.IndexOf(cr.Name)
	if colIdx < 0 || colIdx >= len(rg.Columns) {
		return true
	}
	zm := rg.Columns[colIdx].Stats
	if !zm.HasMinMax || op == sql.OpNE {
		return true
	}

	// minCmp and maxCmp compare the row group's min and max with the literal:
	// negative when the bound is smaller, zero when equal, positive when larger.
	var minCmp, maxCmp int
	switch schema.Fields[colIdx].Type {
	case exec.TypeInt64:
		v, ok := zoneLiteral(lit)
		if !ok {
			return true
		}
		switch lv := v.(type) {
		case int64:
			minCmp, maxCmp = cmpOrdered(int64(zm.Min), lv), cmpOrdered(int64(zm.Max), lv)
		case float64:
			lo, hi := int64(zm.Min), int64(zm.Max)
			if math.IsNaN(lv) || !exactFloat(lo) || !exactFloat(hi) {
				return true
			}
			minCmp, maxCmp = cmpOrdered(float64(lo), lv), cmpOrdered(float64(hi), lv)
		default:
			return true
		}
	case exec.TypeFloat64:
		v, ok := zoneLiteral(lit)
		if !ok {
			return true
		}
		var lv float64
		switch t := v.(type) {
		case int64:
			lv = float64(t)
		case float64:
			lv = t
		default:
			return true
		}
		lo, hi := math.Float64frombits(zm.Min), math.Float64frombits(zm.Max)
		if math.IsNaN(lv) || math.IsNaN(lo) || math.IsNaN(hi) {
			return true
		}
		minCmp, maxCmp = cmpOrdered(lo, lv), cmpOrdered(hi, lv)
	case exec.TypeDate:
		days, ok := zoneDateLiteral(lit)
		if !ok {
			return true
		}
		minCmp, maxCmp = cmpOrdered(int64(int32(uint32(zm.Min))), days), cmpOrdered(int64(int32(uint32(zm.Max))), days)
	default:
		return true
	}

	switch op {
	case sql.OpEQ:
		return minCmp <= 0 && maxCmp >= 0
	case sql.OpLT:
		return minCmp < 0
	case sql.OpLE:
		return minCmp <= 0
	case sql.OpGT:
		return maxCmp > 0
	case sql.OpGE:
		return maxCmp >= 0
	}
	return true
}

func cmpOrdered[T int64 | float64](a, b T) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// exactFloat reports whether v converts to float64 without rounding, so
// comparing its float64 image with a float literal gives the exact answer.
func exactFloat(v int64) bool {
	return v >= -(1<<53) && v <= 1<<53
}

// zoneLiteral returns a numeric literal's value as int64 or float64.
func zoneLiteral(e sql.Expr) (any, bool) {
	switch x := foldConstant(e).(type) {
	case *sql.IntLiteral:
		return x.Value, true
	case *sql.FloatLiteral:
		return x.Value, true
	}
	return nil, false
}

// zoneDateLiteral returns the days-since-epoch value a literal denotes when
// compared with a DATE column, mirroring coerceOneSide.
func zoneDateLiteral(e sql.Expr) (int64, bool) {
	switch x := foldConstant(e).(type) {
	case *sql.IntLiteral:
		return int64(int32(x.Value)), true
	case *sql.StringLiteral:
		t, err := time.ParseInLocation("2006-01-02", x.Value, time.UTC)
		if err != nil {
			return 0, false
		}
		return int64(int32(t.Sub(epoch).Hours() / 24)), true
	}
	return 0, false
}

// coercePair adjusts literal types so both sides of a BinOp are compatible.
// Rules:
//   - TypeString literal beside TypeDate column → convert string to date int32.
//   - TypeInt64 literal beside TypeFloat64 column → convert int64 to float64.
func coercePair(l, r exec.Expr) (exec.Expr, exec.Expr) {
	l, r = coerceOneSide(l, r)
	r, l = coerceOneSide(r, l)
	return l, r
}

// coerceOneSide coerces b to match a's type when b is a Literal.
func coerceOneSide(a, b exec.Expr) (exec.Expr, exec.Expr) {
	lit, ok := b.(*exec.Literal)
	if !ok {
		return a, b
	}
	switch a.Type() {
	case exec.TypeDate:
		if lit.T == exec.TypeString {
			s, ok := lit.Val.(string)
			if !ok {
				break
			}
			t, err := time.ParseInLocation("2006-01-02", s, time.UTC)
			if err != nil {
				break
			}
			days := int32(t.Sub(epoch).Hours() / 24)
			return a, &exec.Literal{Val: days, T: exec.TypeDate}
		}
		// Integer literal beside a DateVector: interpret as days-since-epoch,
		// matching DateVector's int32 storage format (same epoch as string-date
		// coercion above).  This allows predicates like `order_date > 18000`.
		if lit.T == exec.TypeInt64 {
			v := lit.Val.(int64)
			return a, &exec.Literal{Val: int32(v), T: exec.TypeDate}
		}
	case exec.TypeFloat64:
		if lit.T == exec.TypeInt64 {
			v := float64(lit.Val.(int64))
			return a, &exec.Literal{Val: v, T: exec.TypeFloat64}
		}
	case exec.TypeInt64:
		// Only an integral float literal converts exactly; 2.5 must stay a
		// float so the comparison is done in FLOAT64 (comparableOperands)
		// rather than against a truncated 2.
		if lit.T == exec.TypeFloat64 {
			if v, ok := exactInt64(lit.Val.(float64)); ok {
				return a, &exec.Literal{Val: v, T: exec.TypeInt64}
			}
		}
	}
	return a, b
}

// exactInt64 converts f to int64 when f is an integer that int64 represents.
func exactInt64(f float64) (int64, bool) {
	if f != math.Trunc(f) || f < -(1<<63) || f >= 1<<63 {
		return 0, false
	}
	return int64(f), true
}

// castToFloat wraps an INT64 expression in a cast to FLOAT64; other expressions
// are returned unchanged.
func castToFloat(e exec.Expr) exec.Expr {
	if e.Type() != exec.TypeInt64 {
		return e
	}
	if lit, ok := e.(*exec.Literal); ok {
		return &exec.Literal{Val: float64(lit.Val.(int64)), T: exec.TypeFloat64}
	}
	return &exec.CastIntToFloatExpr{Inner: e}
}

// comparableOperands returns l and r adjusted so the executor can compare them:
// an INT64/FLOAT64 mix is compared as FLOAT64, and any other pair must already
// share a type the comparison kernels support. Rejecting a mismatch here turns
// what used to be an executor panic (a DATE column compared with an INT64
// column) into a planning error.
func comparableOperands(l, r exec.Expr) (exec.Expr, exec.Expr, error) {
	lt, rt := l.Type(), r.Type()
	if lt != rt && (lt == exec.TypeInt64 || lt == exec.TypeFloat64) && (rt == exec.TypeInt64 || rt == exec.TypeFloat64) {
		return castToFloat(l), castToFloat(r), nil
	}
	if lt != rt {
		return nil, nil, fmt.Errorf("cannot compare %v with %v", lt, rt)
	}
	switch lt {
	case exec.TypeInt64, exec.TypeFloat64, exec.TypeDate:
		return l, r, nil
	}
	return nil, nil, fmt.Errorf("comparison of %v values is not supported (only = and <> against a string literal)", lt)
}

// inListValue converts one IN-list entry to the representation exec.InExpr
// compares against a value of type t. ok=false means the entry is a valid
// constant that no value of type t can equal (2.5 against INT64), so it is
// left out of the set. isNull reports a NULL entry.
func inListValue(item sql.Expr, t exec.DataType) (v any, isNull, ok bool, err error) {
	bad := func() (any, bool, bool, error) {
		return nil, false, false, fmt.Errorf("planner: IN list entry %s cannot be compared with a %v value", sql.FormatExpr(item), t)
	}
	switch lit := foldConstant(item).(type) {
	case *sql.NullLiteral:
		return nil, true, false, nil
	case *sql.IntLiteral:
		switch t {
		case exec.TypeInt64:
			return lit.Value, false, true, nil
		case exec.TypeFloat64:
			return float64(lit.Value), false, true, nil
		case exec.TypeDate:
			return int32(lit.Value), false, lit.Value == int64(int32(lit.Value)), nil
		}
		return bad()
	case *sql.FloatLiteral:
		switch t {
		case exec.TypeFloat64:
			return lit.Value, false, true, nil
		case exec.TypeInt64:
			iv, exact := exactInt64(lit.Value)
			return iv, false, exact, nil
		}
		return bad()
	case *sql.StringLiteral:
		switch t {
		case exec.TypeString:
			return lit.Value, false, true, nil
		case exec.TypeDate:
			d, err := time.ParseInLocation("2006-01-02", lit.Value, time.UTC)
			if err != nil {
				return nil, false, false, fmt.Errorf("planner: IN list entry %q is not a date (YYYY-MM-DD)", lit.Value)
			}
			return int32(d.Sub(epoch).Hours() / 24), false, true, nil
		}
		return bad()
	case *sql.BoolLiteral:
		if t == exec.TypeBool {
			return lit.Value, false, true, nil
		}
		return bad()
	}
	return nil, false, false, fmt.Errorf("planner: IN list entries must be literals, got %s", sql.FormatExpr(item))
}

// foldConstant reduces unary minus applied to a numeric literal (-(2), --2.5)
// to a literal, returning any other expression unchanged.
func foldConstant(e sql.Expr) sql.Expr {
	u, ok := e.(*sql.UnaryExpr)
	if !ok || u.Op != sql.OpMinus {
		return e
	}
	switch lit := foldConstant(u.Expr).(type) {
	case *sql.IntLiteral:
		if lit.Value != math.MinInt64 {
			return &sql.IntLiteral{Value: -lit.Value}
		}
	case *sql.FloatLiteral:
		return &sql.FloatLiteral{Value: -lit.Value}
	}
	return e
}
