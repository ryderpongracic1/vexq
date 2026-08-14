package planner

import (
	"context"
	"fmt"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// tryParallelJoin detects the probe-side-parallel hash join shape and builds an
// exec.ParallelHashJoinAggregate for it. It is called from Parallel before the
// aggregate-over-scan detection and is purely additive: every shape it does not
// handle returns matched=false, leaving the existing path untouched.
//
// Shapes handled — Limit and Sort above the aggregate are peeled and applied
// serially to the small merged result, exactly as Parallel does for
// aggregate-over-scan plans:
//
//	(LogicalLimit →)? (LogicalSort →)? LogicalAggregate → (LogicalFilter →)?
//	    LogicalJoin{Left: <any subtree>, Right: (LogicalFilter →)? LogicalScan}
//
// The join's Right side becomes the parallel probe side and must bottom out in a
// scan. buildJoinTree always produces that: it attaches each newly joined table
// as the right child, so the last table of a left-deep chain — lineitem in
// TPC-H Q3 and Q12 — is the probe side. The Left subtree is the build side and
// may itself be a join, which is what covers Q3's three-table shape: the whole
// customer ⋈ orders subtree is evaluated once into the shared hash table.
//
// Falls back (matched=false) for:
//   - any root that is not an aggregate under optional Sort/Limit
//   - an aggregate whose child is not a join (Parallel's own path handles scans)
//   - a probe side that is not a scan (e.g. a right-deep join tree)
//   - a join condition serial planning would also reject, so Physical reports
//     the canonical error
//   - COUNT(DISTINCT), whose partial counts cannot be summed across workers
//   - an empty probe table (degenerate; serial is cheaper)
//
// An error return means the shape matched but planning it failed — a failure
// serial planning would hit as well.
func tryParallelJoin(ctx context.Context, root LogicalNode, numWorkers int) (exec.Operator, bool, error) {
	limitNode, sortNode, aggNode := peelToAggregate(root)
	if aggNode == nil {
		return nil, false, nil
	}

	// A residual filter between the aggregate and the join applies to joined
	// rows, so it runs inside each worker pipeline above the join.
	child := aggNode.Child
	var residual *LogicalFilter
	if f, ok := child.(*LogicalFilter); ok {
		residual = f
		child = f.Child
	}
	joinNode, ok := child.(*LogicalJoin)
	if !ok {
		return nil, false, nil
	}

	// Probe side must be a scan, optionally with a filter that predicate
	// pushdown did not fold into the scan.
	probeChild := joinNode.Right
	var probeFilter *LogicalFilter
	if f, ok := probeChild.(*LogicalFilter); ok {
		probeFilter = f
		probeChild = f.Child
	}
	probeScan, ok := probeChild.(*LogicalScan)
	if !ok {
		return nil, false, nil
	}

	// ---- Build-side schema ---------------------------------------------------
	// Constructed once here to resolve the join key index, then discarded; the
	// operator is rebuilt at execution time by buildFactory. Rebuilding costs one
	// file open plus a footer read, which avoids holding an open reader between
	// planning and the first Next call.
	buildOpForSchema, err := Physical(ctx, joinNode.Left)
	if err != nil {
		return nil, true, fmt.Errorf("planner: parallel join: build side: %w", err)
	}
	buildSchema := buildOpForSchema.Schema()
	_ = buildOpForSchema.Close()

	// ---- Probe-side schema and row group count ------------------------------
	var zonePred exec.ZonePredicate
	if probeScan.Predicate != nil {
		zonePred = buildZonePredicate(probeScan.Predicate, probeScan.Schema)
	}

	pr, err := storage.Open(ctx, probeScan.FilePath)
	if err != nil {
		return nil, true, fmt.Errorf("planner: parallel join: open %q: %w", probeScan.FilePath, err)
	}
	totalRGs := len(pr.Meta().RowGroups)
	if totalRGs == 0 {
		_ = pr.Close()
		return nil, false, nil // empty probe table — serial is cheaper
	}
	tempScan, err := exec.NewTableScanRange(pr, probeScan.NeededCols, zonePred, 0, 1)
	if err != nil {
		_ = pr.Close()
		return nil, true, fmt.Errorf("planner: parallel join: probe scan: %w", err)
	}
	// Filter preserves its child's schema, so the scan schema is the probe
	// pipeline schema regardless of any filter above it.
	probeSchema := tempScan.Schema()
	_ = tempScan.Close()

	// ---- Join keys -----------------------------------------------------------
	// Mirrors physicalJoin: resolve one side against each schema, swapping if the
	// condition is written probe-first.
	bin, ok := joinNode.Condition.(*sql.BinaryExpr)
	if !ok || bin.Op != sql.OpEQ {
		return nil, false, nil
	}
	lCR, lok := bin.Left.(*sql.ColumnRefExpr)
	rCR, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		return nil, false, nil
	}
	buildKeyIdx := buildSchema.IndexOf(lCR.Name)
	probeKeyIdx := probeSchema.IndexOf(rCR.Name)
	if buildKeyIdx < 0 {
		buildKeyIdx = buildSchema.IndexOf(rCR.Name)
		probeKeyIdx = probeSchema.IndexOf(lCR.Name)
	}
	if buildKeyIdx < 0 || probeKeyIdx < 0 {
		return nil, false, nil
	}

	// ---- Per-worker operators above the join --------------------------------
	// Neither NewFilter nor NewProject closes its child on failure, so the caller
	// closes the join exactly once on error.
	aboveJoin := func(op exec.Operator) (exec.Operator, error) {
		if residual != nil {
			pred, err := buildExecExpr(residual.Predicate, op.Schema())
			if err != nil {
				return nil, fmt.Errorf("parallel join: residual filter: %w", err)
			}
			op, err = exec.NewFilter(op, pred)
			if err != nil {
				return nil, err
			}
		}
		return buildPreProjection(aggNode, op)
	}

	// The schema the aggregate sees: join output → residual filter →
	// pre-projection. Derived from a schema-only stub, so no file is opened.
	joinFields := make([]exec.Field, 0, len(buildSchema.Fields)+len(probeSchema.Fields))
	joinFields = append(joinFields, buildSchema.Fields...)
	joinFields = append(joinFields, probeSchema.Fields...)
	stub, err := aboveJoin(exec.NewSchemaOnly(exec.Schema{Fields: joinFields}))
	if err != nil {
		return nil, true, fmt.Errorf("planner: parallel join: %w", err)
	}
	pipelineSchema := stub.Schema()
	_ = stub.Close()

	// ---- Aggregate config ----------------------------------------------------
	groupByIdxs, aggExprs, err := resolveAggConfig(aggNode, pipelineSchema)
	if err != nil {
		return nil, true, fmt.Errorf("planner: parallel join: %w", err)
	}
	// Partial COUNT(DISTINCT) counts cannot be summed at merge time — correct
	// parallel support needs per-group value sets shipped to the merge step.
	// Serial execution is correct today.
	for _, ae := range aggExprs {
		if ae.Kind == exec.AggCountDistinct {
			return nil, false, nil
		}
	}
	outSchema := aggOutputSchema(aggNode, pipelineSchema, groupByIdxs, aggExprs)

	// ---- Factories -----------------------------------------------------------
	buildFactory := func(fCtx context.Context) (exec.Operator, error) {
		return Physical(fCtx, joinNode.Left)
	}

	probeFactory := func(fCtx context.Context, rgStart, rgEnd int) (exec.Operator, error) {
		fr, err := storage.Open(fCtx, probeScan.FilePath)
		if err != nil {
			return nil, fmt.Errorf("parallel join probe: open: %w", err)
		}
		scan, err := exec.NewTableScanRange(fr, probeScan.NeededCols, zonePred, rgStart, rgEnd)
		if err != nil {
			_ = fr.Close()
			return nil, fmt.Errorf("parallel join probe: scan: %w", err)
		}
		var op exec.Operator = scan

		// Predicate pushdown moves single-table WHERE terms into
		// LogicalScan.Predicate and drops the LogicalFilter node, so apply the
		// scan predicate as a runtime row filter here — same as physicalScan.
		if probeScan.Predicate != nil {
			pred, err := buildExecExpr(probeScan.Predicate, op.Schema())
			if err != nil {
				_ = op.Close()
				return nil, fmt.Errorf("parallel join probe: scan predicate: %w", err)
			}
			op, err = exec.NewFilter(op, pred)
			if err != nil {
				_ = scan.Close()
				return nil, err
			}
		}
		if probeFilter != nil {
			pred, err := buildExecExpr(probeFilter.Predicate, op.Schema())
			if err != nil {
				_ = op.Close()
				return nil, fmt.Errorf("parallel join probe: filter: %w", err)
			}
			op, err = exec.NewFilter(op, pred)
			if err != nil {
				_ = op.Close()
				return nil, err
			}
		}
		return op, nil
	}

	// morselSize=0 → exec uses defaultMorselSize (one row group).
	var op exec.Operator = exec.NewParallelHashJoinAggregate(
		buildFactory, buildKeyIdx,
		probeFactory, probeKeyIdx,
		aboveJoin,
		totalRGs, numWorkers, 0,
		groupByIdxs, aggExprs, outSchema,
	)

	op, err = wrapSerialSortLimit(op, sortNode, limitNode, outSchema)
	if err != nil {
		return nil, true, err
	}
	return op, true, nil
}

// peelToAggregate strips an optional Limit → Sort prefix and returns the
// aggregate beneath it. aggNode is nil when root is not one of:
//
//	LogicalAggregate | LogicalSort → LogicalAggregate |
//	LogicalLimit → LogicalSort → LogicalAggregate
//
// A Limit is only peeled when a Sort sits under it. LIMIT without ORDER BY takes
// an arbitrary subset of groups, and the merged parallel group order differs from
// the serial insertion order, so parallelizing it would return a different (still
// SQL-valid) subset than serial execution. Matching Parallel's own restriction
// keeps the two paths result-identical.
func peelToAggregate(root LogicalNode) (limitNode *LogicalLimit, sortNode *LogicalSort, aggNode *LogicalAggregate) {
	if l, ok := root.(*LogicalLimit); ok {
		s, ok := l.Child.(*LogicalSort)
		if !ok {
			return nil, nil, nil
		}
		limitNode = l
		sortNode = s
		root = s.Child
	} else if s, ok := root.(*LogicalSort); ok {
		sortNode = s
		root = s.Child
	}
	agg, ok := root.(*LogicalAggregate)
	if !ok {
		return nil, nil, nil
	}
	return limitNode, sortNode, agg
}

// wrapSerialSortLimit applies peeled Sort/Limit nodes serially on top of a
// parallel aggregate. The merged aggregate output is one row per group, so
// sorting and limiting it in the calling goroutine is both correct and cheap.
//
// Parallel currently inlines equivalent logic; it can adopt this helper once the
// parallel-aggregate and parallel-join work streams are integrated.
func wrapSerialSortLimit(op exec.Operator, sortNode *LogicalSort, limitNode *LogicalLimit, outSchema exec.Schema) (exec.Operator, error) {
	if sortNode != nil {
		var keys []exec.SortKey
		for _, ob := range sortNode.OrderBy {
			cr, ok := ob.Expr.(*sql.ColumnRefExpr)
			if !ok {
				_ = op.Close()
				return nil, fmt.Errorf("planner: parallel join: ORDER BY only supports column references")
			}
			idx := outSchema.IndexOf(cr.Name)
			if idx < 0 {
				_ = op.Close()
				return nil, fmt.Errorf("planner: parallel join: ORDER BY column %q not found", cr.Name)
			}
			keys = append(keys, exec.SortKey{ColIdx: idx, Descending: ob.Descending})
		}
		sortOp, err := exec.NewExternalSort(op, keys)
		if err != nil {
			_ = op.Close()
			return nil, fmt.Errorf("planner: parallel join: sort: %w", err)
		}
		op = sortOp
	}
	if limitNode != nil {
		op = exec.NewLimit(op, int(limitNode.Count))
	}
	return op, nil
}
