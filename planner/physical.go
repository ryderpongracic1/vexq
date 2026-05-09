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

	scan, err := exec.NewTableScan(r, n.NeededCols, zonePred)
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
		ae := exec.AggExpr{OutName: agg.Alias}
		switch agg.Func {
		case "COUNT":
			ae.Kind = exec.AggCount
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
			ae.AccumType = exec.TypeFloat64
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
		// Set AccumType for SUM/MIN/MAX based on source column type.
		if ae.Kind == exec.AggSum || ae.Kind == exec.AggMin || ae.Kind == exec.AggMax {
			if ae.ColIdx >= 0 && schema.Fields[ae.ColIdx].Type == exec.TypeFloat64 {
				ae.AccumType = exec.TypeFloat64
			} else {
				ae.AccumType = exec.TypeInt64
			}
		}
		aggExprs = append(aggExprs, ae)
	}
	return groupByIdxs, aggExprs, nil
}

func physicalSort(ctx context.Context, n *LogicalSort) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
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
	return exec.NewExternalSort(child, keys)
}

func physicalLimit(ctx context.Context, n *LogicalLimit) (exec.Operator, error) {
	child, err := Physical(ctx, n.Child)
	if err != nil {
		return nil, err
	}
	return exec.NewLimit(child, int(n.Count)), nil
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
		idx := schema.IndexOf(x.Name)
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
		// Unary minus: multiply by -1.
		return &exec.BinOp{
			Op:    exec.BinMul,
			Left:  child,
			Right: &exec.Literal{Val: int64(-1), T: exec.TypeInt64},
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
		var set []any
		for _, item := range x.List {
			switch v := item.(type) {
			case *sql.IntLiteral:
				set = append(set, v.Value)
			case *sql.FloatLiteral:
				set = append(set, v.Value)
			case *sql.StringLiteral:
				set = append(set, v.Value)
			}
		}
		inExpr := &exec.InExpr{Child: child, Set: set}
		if x.Not {
			return &exec.NotExpr{Child: inExpr}, nil
		}
		return inExpr, nil

	case *sql.LikeExpr:
		child, err := buildExecExpr(x.Expr, schema)
		if err != nil {
			return nil, err
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
		t := exec.TypeFloat64
		if len(whens) > 0 {
			t = whens[0].Result.Type()
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
	resultType := exec.TypeBool
	if isArith {
		resultType = l.Type()
		if l.Type() == exec.TypeFloat64 || r.Type() == exec.TypeFloat64 {
			resultType = exec.TypeFloat64
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
		// Equivalent to lo <= col <= hi; skip if max < lo or min > hi.
		cr, ok := x.Expr.(*sql.ColumnRefExpr)
		if !ok {
			return true
		}
		colIdx := schema.IndexOf(cr.Name)
		if colIdx < 0 || colIdx >= len(rg.Columns) {
			return true
		}
		zm := rg.Columns[colIdx].Stats
		if !zm.HasMinMax {
			return true
		}
		loVal := literalInt64(x.Lo)
		hiVal := literalInt64(x.Hi)
		if loVal == nil || hiVal == nil {
			return true
		}
		rgMin := int64(zm.Min)
		rgMax := int64(zm.Max)
		if rgMax < *loVal || rgMin > *hiVal {
			// Row group entirely outside [lo, hi]: plain BETWEEN has no matches (prune),
			// NOT BETWEEN always matches (don't prune).
			return x.Not
		}
		return true
	}
	return true // conservative: don't prune
}

// zoneRangePred evaluates a simple comparison expression against zone map stats.
func zoneRangePred(x *sql.BinaryExpr, schema exec.Schema, rg *storage.RowGroupMeta) bool {
	cr, ok := x.Left.(*sql.ColumnRefExpr)
	if !ok {
		// Try reversed.
		cr, ok = x.Right.(*sql.ColumnRefExpr)
		if !ok {
			return true
		}
	}
	colIdx := schema.IndexOf(cr.Name)
	if colIdx < 0 || colIdx >= len(rg.Columns) {
		return true
	}
	zm := rg.Columns[colIdx].Stats
	if !zm.HasMinMax {
		return true
	}

	// Get the literal side.
	var lit sql.Expr
	reversed := false
	if _, ok := x.Left.(*sql.ColumnRefExpr); ok {
		lit = x.Right
	} else {
		lit = x.Left
		reversed = true
	}
	litVal := literalInt64(lit)
	if litVal == nil {
		return true
	}
	v := *litVal

	// For float64 columns, zone map stores float64 bit patterns.
	// If the literal was an integer (not yet converted to float bits), coerce it now
	// so the bit-level comparison is valid.
	colType := schema.Fields[colIdx].Type
	if colType == exec.TypeFloat64 {
		if _, isInt := lit.(*sql.IntLiteral); isInt {
			v = int64(math.Float64bits(float64(v)))
		}
	}

	rgMin := int64(zm.Min)
	rgMax := int64(zm.Max)

	op := x.Op
	if reversed {
		// Swap comparison direction.
		switch op {
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

	switch op {
	case sql.OpEQ:
		return rgMin <= v && v <= rgMax
	case sql.OpNE:
		return true // can't easily prune with NE
	case sql.OpLT:
		return rgMin < v
	case sql.OpLE:
		return rgMin <= v
	case sql.OpGT:
		return rgMax > v
	case sql.OpGE:
		return rgMax >= v
	}
	return true
}

func literalInt64(e sql.Expr) *int64 {
	switch x := e.(type) {
	case *sql.IntLiteral:
		return &x.Value
	case *sql.FloatLiteral:
		v := int64(math.Float64bits(x.Value))
		return &v
	case *sql.StringLiteral:
		// Try to parse as a date (YYYY-MM-DD).
		if t, err := time.ParseInLocation("2006-01-02", x.Value, time.UTC); err == nil {
			days := int64(t.Sub(epoch).Hours() / 24)
			return &days
		}
	}
	return nil
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
	case exec.TypeFloat64:
		if lit.T == exec.TypeInt64 {
			v := float64(lit.Val.(int64))
			return a, &exec.Literal{Val: v, T: exec.TypeFloat64}
		}
	case exec.TypeInt64:
		if lit.T == exec.TypeFloat64 {
			v := int64(lit.Val.(float64))
			return a, &exec.Literal{Val: v, T: exec.TypeInt64}
		}
	}
	return a, b
}
