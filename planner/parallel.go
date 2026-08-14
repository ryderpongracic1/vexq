package planner

import (
	"context"
	"fmt"
	"runtime"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/sql"
	"github.com/ryderpongracic1/vexq/storage"
)

// Parallel returns a ParallelHashAggregate when the plan matches one of these
// patterns:
//
//	LogicalAggregate → (LogicalFilter →)? LogicalScan
//	LogicalSort → LogicalAggregate → (LogicalFilter →)? LogicalScan
//	LogicalLimit → LogicalSort → LogicalAggregate → (LogicalFilter →)? LogicalScan
//
// For Sort/Limit-wrapped shapes, the aggregate is parallelized and the merged
// result (which is small — one row per group) is sorted/limited serially.
//
// It partitions the scan's row groups evenly across numWorkers goroutines,
// each running an independent scan+filter+pre-projection pipeline, then merges
// the partial aggregate results in the calling goroutine.
//
// Aggregates over an inner hash join are handled by tryParallelJoin
// ([planner/parallel_join.go]), which parallelizes the probe side.
//
// Falls back to Physical(ctx, root) when:
//   - root is not a LogicalAggregate (or Sort/Limit above an aggregate)
//   - the aggregate child (after peeling an optional LogicalFilter) is neither a
//     LogicalScan nor a join shape tryParallelJoin recognizes
//
// numWorkers <= 0 defaults to runtime.NumCPU().
func Parallel(ctx context.Context, root LogicalNode, numWorkers int) (exec.Operator, error) {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// Probe-side-parallel hash join (planner/parallel_join.go). Additive: it
	// returns matched=false for every shape it does not handle, so the
	// aggregate-over-scan detection below is unchanged.
	if op, matched, err := tryParallelJoin(ctx, root, numWorkers); err != nil {
		return nil, err
	} else if matched {
		return op, nil
	}

	// ---- Plan shape detection ------------------------------------------------
	// Peel optional Limit → Sort above the aggregate. The aggregate output is
	// small (one row per group), so serial sort/limit is correct and cheap.

	var sortNode *LogicalSort
	var limitNode *LogicalLimit

	aggNode, ok := root.(*LogicalAggregate)
	if !ok {
		// Try Sort → Aggregate or Limit → Sort → Aggregate.
		if s, ok := root.(*LogicalSort); ok {
			if a, ok := s.Child.(*LogicalAggregate); ok {
				sortNode = s
				aggNode = a
			} else {
				return Physical(ctx, root) // sort over non-aggregate — fallback
			}
		} else if l, ok := root.(*LogicalLimit); ok {
			if s, ok := l.Child.(*LogicalSort); ok {
				if a, ok := s.Child.(*LogicalAggregate); ok {
					limitNode = l
					sortNode = s
					aggNode = a
				} else {
					return Physical(ctx, root)
				}
			} else {
				return Physical(ctx, root)
			}
		} else {
			return Physical(ctx, root)
		}
	}

	child := aggNode.Child

	// Peel an optional LogicalFilter.
	var filtNode *LogicalFilter
	if f, ok := child.(*LogicalFilter); ok {
		filtNode = f
		child = f.Child
	}

	// The next node must be a LogicalScan (no join, no subquery).
	scanNode, ok := child.(*LogicalScan)
	if !ok {
		return Physical(ctx, root) // unsupported shape — fallback
	}

	// Fall back to serial execution if any aggregate uses a computed expression
	// (e.g. SUM(price * discount)). The optimizer's pruneColumns places the
	// synthetic column name (_agg_0) into LogicalScan.NeededCols, but that column
	// does not exist in the physical file — it is materialized by
	// buildPreProjection after the scan. The parallel factory would need to
	// resolve the real source columns from AggExpr, scan those, and apply
	// buildPreProjection in each worker pipeline. Until that is implemented,
	// serial execution via Physical handles this correctly.
	for i := range aggNode.Aggs {
		if aggNode.Aggs[i].AggExpr != nil {
			return Physical(ctx, root)
		}
	}

	// ---- Row group count -----------------------------------------------------

	r, err := storage.Open(ctx, scanNode.FilePath)
	if err != nil {
		return nil, fmt.Errorf("planner: parallel: open %q: %w", scanNode.FilePath, err)
	}
	totalRGs := len(r.Meta().RowGroups)
	_ = r.Close()

	if totalRGs == 0 {
		// Empty table: fall back to serial execution (degenerate case).
		return Physical(ctx, root)
	}

	// ---- Pre-projection schema detection ------------------------------------
	// Build one temporary pipeline (covering row group [0,1)) to get the schema
	// that the aggregate sees after scan + filter + optional pre-projection.
	// This is needed to resolve group-by and aggregate column indices correctly.

	zonePred := buildZonePredicate(scanNode.Predicate, scanNode.Schema)

	tempR, err := storage.Open(ctx, scanNode.FilePath)
	if err != nil {
		return nil, fmt.Errorf("planner: parallel: temp open %q: %w", scanNode.FilePath, err)
	}
	endRG := totalRGs
	if endRG > 1 {
		endRG = 1
	}
	tempScan, err := exec.NewTableScanRange(tempR, scanNode.NeededCols, zonePred, 0, endRG)
	if err != nil {
		_ = tempR.Close()
		return nil, fmt.Errorf("planner: parallel: temp scan: %w", err)
	}

	var tempOp exec.Operator = tempScan
	if filtNode != nil {
		tempOp, err = buildFilterOp(filtNode, tempOp)
		if err != nil {
			return nil, fmt.Errorf("planner: parallel: temp filter: %w", err)
		}
	}

	// Apply pre-projection (if any complex aggregate expressions) to get the
	// correct post-projection schema for index resolution below.
	tempOp, err = buildPreProjection(aggNode, tempOp)
	if err != nil {
		_ = tempOp.Close()
		return nil, fmt.Errorf("planner: parallel: temp pre-projection: %w", err)
	}
	pipelineSchema := tempOp.Schema()
	_ = tempOp.Close()

	// ---- Resolve aggregate config -------------------------------------------

	groupByIdxs, aggExprs, err := resolveAggConfig(aggNode, pipelineSchema)
	if err != nil {
		return nil, fmt.Errorf("planner: parallel: %w", err)
	}

	// Fall back to serial execution if any aggregate uses DISTINCT.
	// Partial COUNT(DISTINCT) counts from workers cannot be summed — the correct
	// approach requires shipping per-group value sets and unioning them at merge,
	// which is too invasive for the current mergePartialAgg ([]int64 accumulators).
	// Serial execution is correct; parallel COUNT(DISTINCT) is a future improvement.
	for _, ae := range aggExprs {
		if ae.Kind == exec.AggCountDistinct {
			return Physical(ctx, root)
		}
	}

	// Compute the output schema (mirrors NewHashAggregate's logic).
	outSchema := aggOutputSchema(aggNode, pipelineSchema, groupByIdxs, aggExprs)

	// ---- Factory closure ----------------------------------------------------
	// Each call to factory(ctx, rgStart, rgEnd) builds an independent pipeline:
	//   TableScanRange → ScanPredFilter? → Filter? → PreProjection?
	// This is called once per worker goroutine inside ParallelHashAggregate.setup.

	factory := func(fCtx context.Context, rgStart, rgEnd int) (exec.Operator, error) {
		fr, err := storage.Open(fCtx, scanNode.FilePath)
		if err != nil {
			return nil, fmt.Errorf("parallel factory: open: %w", err)
		}
		scan, err := exec.NewTableScanRange(fr, scanNode.NeededCols, zonePred, rgStart, rgEnd)
		if err != nil {
			_ = fr.Close()
			return nil, fmt.Errorf("parallel factory: scan: %w", err)
		}
		var op exec.Operator = scan

		// The optimizer pushes LogicalFilter predicates into LogicalScan.Predicate
		// (eliminating the LogicalFilter node). Apply the scan predicate as a
		// runtime row filter here, mirroring physicalScan's behaviour.
		if scanNode.Predicate != nil {
			filterExpr, err := buildExecExpr(scanNode.Predicate, op.Schema())
			if err != nil {
				_ = op.Close()
				return nil, fmt.Errorf("parallel factory: scan predicate: %w", err)
			}
			op, err = exec.NewFilter(op, filterExpr)
			if err != nil {
				_ = op.Close()
				return nil, err
			}
		}

		// Apply a LogicalFilter above the scan (rare after pushdown, but possible).
		if filtNode != nil {
			op, err = buildFilterOp(filtNode, op)
			if err != nil {
				return nil, err
			}
		}

		op, err = buildPreProjection(aggNode, op)
		if err != nil {
			_ = op.Close()
			return nil, err
		}
		return op, nil
	}

	// morselSize=0 → exec package uses defaultMorselSize (1 row group).
	// Tune via environment or query hints in future work.
	var op exec.Operator = exec.NewParallelHashAggregate(factory, totalRGs, numWorkers, 0, groupByIdxs, aggExprs, outSchema)

	// ---- Wrap with peeled Sort/Limit ----------------------------------------
	// The merged aggregate output is small (one row per group), so sorting and
	// limiting it serially is correct and cheap.

	if sortNode != nil {
		var keys []exec.SortKey
		for _, ob := range sortNode.OrderBy {
			cr, ok := ob.Expr.(*sql.ColumnRefExpr)
			if !ok {
				_ = op.Close()
				return nil, fmt.Errorf("planner: parallel: ORDER BY only supports column references")
			}
			idx := outSchema.IndexOf(cr.Name)
			if idx < 0 {
				_ = op.Close()
				return nil, fmt.Errorf("planner: parallel: ORDER BY column %q not found", cr.Name)
			}
			keys = append(keys, exec.SortKey{ColIdx: idx, Descending: ob.Descending})
		}
		sortOp, err := exec.NewExternalSort(op, keys)
		if err != nil {
			_ = op.Close()
			return nil, fmt.Errorf("planner: parallel: sort: %w", err)
		}
		op = sortOp
	}

	if limitNode != nil {
		op = exec.NewLimit(op, int(limitNode.Count))
	}

	return op, nil
}

// aggOutputSchema computes the output schema of a HashAggregate without needing
// to construct one. Mirrors the field-building logic in exec.NewHashAggregate.
func aggOutputSchema(n *LogicalAggregate, pipelineSchema exec.Schema, groupByIdxs []int, aggExprs []exec.AggExpr) exec.Schema {
	var fields []exec.Field
	for _, idx := range groupByIdxs {
		fields = append(fields, pipelineSchema.Fields[idx])
	}
	for _, ae := range aggExprs {
		var t exec.DataType
		switch ae.Kind {
		case exec.AggCount:
			t = exec.TypeInt64
		case exec.AggSum, exec.AggMin, exec.AggMax:
			if ae.ColIdx >= 0 {
				t = pipelineSchema.Fields[ae.ColIdx].Type
			} else {
				t = exec.TypeInt64
			}
		case exec.AggAvg:
			t = exec.TypeFloat64
		}
		fields = append(fields, exec.Field{Name: ae.OutName, Type: t, Nullable: true})
	}
	return exec.Schema{Fields: fields}
}
