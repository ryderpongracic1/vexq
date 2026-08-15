package planner

import (
	"context"
	"fmt"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// tryParallelJoin detects the parallel hash join shape and builds an
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
// may itself be a join, which is what covers Q3's three-table shape.
//
// The build side is radix-partitioned, and built in parallel over its own
// row-group morsels when buildSideRadixOptions can decompose it — which covers
// both a scan-rooted build side (Q12) and a build side that is itself a join
// whose probe side is a scan (Q3). A build side smaller than one row group stays
// in a single map, where partitioning would cost more than it saves. None of
// this is visible in results; see exec.ParallelHashJoinAggregate.
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

	// ---- Probe side ----------------------------------------------------------
	// Must be a scan, optionally under a filter predicate pushdown did not fold
	// into the scan. Resolved before the build side because it is the cheaper
	// reject.
	probePipe, err := openScanPipeline(ctx, joinNode.Right)
	if err != nil {
		return nil, true, fmt.Errorf("planner: parallel join: probe side: %w", err)
	}
	if probePipe == nil {
		return nil, false, nil // probe side is not a scan (e.g. a right-deep tree)
	}
	if probePipe.totalRGs == 0 {
		return nil, false, nil // empty probe table — serial is cheaper
	}
	probeSchema := probePipe.schema
	totalRGs := probePipe.totalRGs

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

	// ---- Join keys -----------------------------------------------------------
	// Mirrors physicalJoin: resolve one side against each schema, swapping if the
	// condition is written probe-first.
	buildKeyIdx, probeKeyIdx, ok := resolveJoinKeys(joinNode.Condition, buildSchema, probeSchema)
	if !ok {
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
	probeFactory := probePipe.factory()

	// ---- Build-side radix configuration --------------------------------------
	// The estimate selects the partition count; the partitionable plan, when the
	// build side has one, additionally enables the parallel build. Both are
	// advisory: without them exec falls back to a serial single-partition build,
	// which is what phase 1 did.
	buildOpts := buildSideRadixOptions(ctx, joinNode.Left, buildSchema)

	// morselSize=0 → exec uses defaultMorselSize (one row group).
	var op exec.Operator = exec.NewParallelHashJoinAggregate(
		buildFactory, buildKeyIdx,
		probeFactory, probeKeyIdx,
		aboveJoin,
		totalRGs, numWorkers, 0,
		groupByIdxs, aggExprs, outSchema,
		buildOpts...,
	)

	op, err = wrapSerialSortLimit(op, sortNode, limitNode, outSchema)
	if err != nil {
		return nil, true, err
	}
	return op, true, nil
}

// ---- Scan-rooted morsel pipelines -------------------------------------------

// scanPipeline describes a subtree the parallel planner can split into row-group
// morsels: a scan, optionally under one filter that predicate pushdown did not
// fold into the scan. Both the probe side and a scan-rooted build side are
// described this way, so a morsel of either is constructed identically.
type scanPipeline struct {
	scan     *LogicalScan
	filter   *LogicalFilter // optional filter above the scan; may be nil
	zonePred exec.ZonePredicate
	schema   exec.Schema
	totalRGs int
	numRows  int // rows before filtering, summed from the footer's row groups
}

// openScanPipeline peels an optional filter off node, requires a scan beneath it,
// and reads the scan's row group count, row count and projected schema from the
// .vxq footer. It returns (nil, nil) when node is not scan-rooted, which is a
// fall-back signal rather than an error.
//
// The reader it opens is closed before returning: nothing is held between
// planning and the first Next call.
func openScanPipeline(ctx context.Context, node LogicalNode) (*scanPipeline, error) {
	var filter *LogicalFilter
	if f, ok := node.(*LogicalFilter); ok {
		filter = f
		node = f.Child
	}
	scan, ok := node.(*LogicalScan)
	if !ok {
		return nil, nil
	}

	sp := &scanPipeline{scan: scan, filter: filter}
	if scan.Predicate != nil {
		sp.zonePred = buildZonePredicate(scan.Predicate, scan.Schema)
	}

	r, err := storage.Open(ctx, scan.FilePath)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", scan.FilePath, err)
	}
	defer func() { _ = r.Close() }()

	rgs := r.Meta().RowGroups
	sp.totalRGs = len(rgs)
	for i := range rgs {
		sp.numRows += rgs[i].NumRows
	}
	if sp.totalRGs == 0 {
		return sp, nil
	}
	// Filter preserves its child's schema, so the projected scan schema is the
	// whole pipeline's schema regardless of any filter above it.
	tempScan, err := exec.NewTableScanRange(r, scan.NeededCols, sp.zonePred, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("scan %q: %w", scan.TableName, err)
	}
	sp.schema = tempScan.Schema()
	_ = tempScan.Close()
	return sp, nil
}

// factory returns a PipelineFactory that opens the scan's file and reads row
// groups [rgStart, rgEnd) through it, applying the pushed-down scan predicate and
// then the optional filter above it — the same operator stack physicalScan
// builds, restricted to a row-group range.
//
// Each call opens its own reader, so pipelines from different goroutines share no
// mutable state.
func (sp *scanPipeline) factory() exec.PipelineFactory {
	scan, filter, zonePred := sp.scan, sp.filter, sp.zonePred
	return func(ctx context.Context, rgStart, rgEnd int) (exec.Operator, error) {
		fr, err := storage.Open(ctx, scan.FilePath)
		if err != nil {
			return nil, fmt.Errorf("parallel join morsel: open: %w", err)
		}
		ts, err := exec.NewTableScanRange(fr, scan.NeededCols, zonePred, rgStart, rgEnd)
		if err != nil {
			_ = fr.Close()
			return nil, fmt.Errorf("parallel join morsel: scan: %w", err)
		}
		var op exec.Operator = ts

		// Predicate pushdown moves single-table WHERE terms into
		// LogicalScan.Predicate and drops the LogicalFilter node, so apply the
		// scan predicate as a runtime row filter here — same as physicalScan.
		if scan.Predicate != nil {
			pred, err := buildExecExpr(scan.Predicate, op.Schema())
			if err != nil {
				_ = op.Close()
				return nil, fmt.Errorf("parallel join morsel: scan predicate: %w", err)
			}
			op, err = exec.NewFilter(op, pred)
			if err != nil {
				_ = ts.Close()
				return nil, err
			}
		}
		if filter != nil {
			pred, err := buildExecExpr(filter.Predicate, op.Schema())
			if err != nil {
				_ = op.Close()
				return nil, fmt.Errorf("parallel join morsel: filter: %w", err)
			}
			op, err = exec.NewFilter(op, pred)
			if err != nil {
				_ = op.Close()
				return nil, err
			}
		}
		return op, nil
	}
}

// ---- Build-side radix configuration -----------------------------------------

// buildSideRadixOptions inspects the build side once and returns the options that
// configure its radix partitioning: a row-count estimate, and — when the build
// side decomposes into row-group morsels — a prepare hook that enables the
// parallel build. An empty result leaves exec with phase 1's behaviour: a serial
// drain into a single unpartitioned map.
//
// Two build-side shapes are recognised:
//
//  1. (Filter →)? Scan — morsels are row-group ranges of that scan, and the
//     estimate is the scan's footer row count.
//  2. LogicalJoin whose own probe side is scan-rooted — the nested join's build
//     side is materialised serially at prepare time (for TPC-H Q3 that is the
//     filtered customer table, the small side) and the morsels are row-group
//     ranges of the nested probe scan joined against it. This is what brings a
//     three-table chain's build side into the parallel build. The recursion stops
//     at one level, so a four-table chain's innermost join is still serial.
//
// The estimate deliberately ignores filter selectivity, so it over-estimates for
// a selective build-side predicate. That is the cheap direction to be wrong in:
// over-partitioning costs an empty map per unused partition, while
// under-partitioning costs cache residency. For shape 2 the estimate is the
// nested probe scan's row count rather than the nested join's output row count,
// which bounds the output for the primary-key/foreign-key joins TPC-H uses; a
// many-to-many join would exceed it and end up under-partitioned, which is a lost
// optimisation, never a wrong answer.
//
// wantSchema is the schema the serial build factory produces, which is what the
// join key index was resolved against. A partitionable plan is only offered when
// its schema matches, so a future planner change that made the two diverge fails
// closed into the serial build instead of silently probing the wrong column.
func buildSideRadixOptions(ctx context.Context, build LogicalNode, wantSchema exec.Schema) []exec.JoinBuildOption {
	// Shape 1: the build side is itself a scan.
	if sp, err := openScanPipeline(ctx, build); err == nil && sp != nil {
		opts := []exec.JoinBuildOption{exec.WithBuildRowsEstimate(sp.numRows)}
		if sp.totalRGs > 0 && sameFields(sp.schema, wantSchema) {
			factory, schema, rgs := sp.factory(), sp.schema, sp.totalRGs
			opts = append(opts, exec.WithPartitionableBuild(
				func(context.Context) (*exec.PartitionedBuild, error) {
					return &exec.PartitionedBuild{Factory: factory, Schema: schema, TotalRGs: rgs}, nil
				}))
		}
		return opts
	}

	// Shape 2: the build side is a join whose own probe side is a scan.
	join, ok := build.(*LogicalJoin)
	if !ok {
		return nil
	}
	inner, err := openScanPipeline(ctx, join.Right)
	if err != nil || inner == nil || inner.totalRGs == 0 {
		return nil
	}
	opts := []exec.JoinBuildOption{exec.WithBuildRowsEstimate(inner.numRows)}

	innerBuildOp, err := Physical(ctx, join.Left)
	if err != nil {
		return opts // serial planning will report the canonical error
	}
	innerBuildSchema := innerBuildOp.Schema()
	_ = innerBuildOp.Close()

	innerBuildKey, innerProbeKey, ok := resolveJoinKeys(join.Condition, innerBuildSchema, inner.schema)
	if !ok {
		return opts
	}
	// exec.NewHashJoinShared emits build columns then probe columns, exactly as
	// physicalJoin does, so this is the schema the serial build produces.
	nestedFields := make([]exec.Field, 0, len(innerBuildSchema.Fields)+len(inner.schema.Fields))
	nestedFields = append(nestedFields, innerBuildSchema.Fields...)
	nestedFields = append(nestedFields, inner.schema.Fields...)
	nestedSchema := exec.Schema{Fields: nestedFields}
	if !sameFields(nestedSchema, wantSchema) {
		return opts
	}

	innerProbeFactory, innerRGs := inner.factory(), inner.totalRGs
	return append(opts, exec.WithPartitionableBuild(
		func(pCtx context.Context) (*exec.PartitionedBuild, error) {
			// Serial: the nested join's own build side, once, before any morsel.
			op, err := Physical(pCtx, join.Left)
			if err != nil {
				return nil, fmt.Errorf("nested build side: %w", err)
			}
			innerTable, err := exec.BuildSharedHashTable(pCtx, op, innerBuildKey)
			_ = op.Close()
			if err != nil {
				return nil, fmt.Errorf("nested build side: %w", err)
			}
			factory := func(fCtx context.Context, rgStart, rgEnd int) (exec.Operator, error) {
				probe, err := innerProbeFactory(fCtx, rgStart, rgEnd)
				if err != nil {
					return nil, err
				}
				joinOp, err := exec.NewHashJoinShared(innerTable, probe, innerProbeKey)
				if err != nil {
					_ = probe.Close()
					return nil, err
				}
				return joinOp, nil
			}
			return &exec.PartitionedBuild{Factory: factory, Schema: nestedSchema, TotalRGs: innerRGs}, nil
		}))
}

// resolveJoinKeys resolves an equi-join condition against a build and a probe
// schema, swapping sides when the condition is written probe-first. Same rule
// physicalJoin and tryParallelJoin apply; factored out so the nested join inside
// a partitionable build side resolves its keys identically.
func resolveJoinKeys(cond sql.Expr, buildSchema, probeSchema exec.Schema) (buildKey, probeKey int, ok bool) {
	bin, isBin := cond.(*sql.BinaryExpr)
	if !isBin || bin.Op != sql.OpEQ {
		return 0, 0, false
	}
	lCR, lok := bin.Left.(*sql.ColumnRefExpr)
	rCR, rok := bin.Right.(*sql.ColumnRefExpr)
	if !lok || !rok {
		return 0, 0, false
	}
	buildKey = buildSchema.IndexOf(lCR.Name)
	probeKey = probeSchema.IndexOf(rCR.Name)
	if buildKey < 0 {
		buildKey = buildSchema.IndexOf(rCR.Name)
		probeKey = probeSchema.IndexOf(lCR.Name)
	}
	if buildKey < 0 || probeKey < 0 {
		return 0, 0, false
	}
	return buildKey, probeKey, true
}

// sameFields reports whether two schemas have identical field names and types in
// the same order.
func sameFields(a, b exec.Schema) bool {
	if len(a.Fields) != len(b.Fields) {
		return false
	}
	for i := range a.Fields {
		if a.Fields[i].Name != b.Fields[i].Name || a.Fields[i].Type != b.Fields[i].Type {
			return false
		}
	}
	return true
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
