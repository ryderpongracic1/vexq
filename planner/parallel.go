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
// Aggregates over computed expressions (e.g. SUM(price * discount), the
// canonical TPC-H Q6 shape) are parallelized: each worker pipeline ends with
// the same pre-projection that the serial planner applies, materializing the
// expression into a synthetic column per morsel before local accumulation. The
// expression is row-local, so evaluating it per morsel is equivalent to
// evaluating it over the whole scan.
//
// Aggregates over an inner hash join are handled by tryParallelJoin
// ([planner/parallel_join.go]), which parallelizes the probe side.
//
// Float64 SUM/AVG results agree with serial execution to within IEEE-754
// rounding rather than bit-for-bit: partitioning changes the order of float
// additions, and float addition is not associative. This is a property of any
// partitioned float reduction and already applies to the simple-column parallel
// path; integer SUM/MIN/MAX and COUNT are exact. The project's correctness
// standard for float aggregates is the 1e-9 relative tolerance used by
// internal/goldentest.
//
// Falls back to Physical(ctx, root) when:
//   - root is not a LogicalAggregate (or Sort/Limit above an aggregate)
//   - the aggregate child (after peeling an optional LogicalFilter) is neither a
//     LogicalScan nor a join shape tryParallelJoin recognizes
//   - any aggregate uses DISTINCT (partial distinct counts cannot be summed)
//   - the pipeline schema or aggregate configuration cannot be resolved, in
//     which case Physical is the authoritative implementation
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

	// ---- Factory closure ----------------------------------------------------
	// Each call to factory(ctx, rgStart, rgEnd) builds an independent pipeline:
	//   TableScanRange → ScanPredFilter? → Filter? → PreProjection?
	// This is called once per morsel inside ParallelHashAggregate.setup, and once
	// here to probe the pipeline's output schema.
	//
	// PreProjection is what makes aggregates over expressions parallel-safe: it
	// materializes each AggItem.AggExpr (e.g. price * discount) into the
	// synthetic column that resolveAggConfig resolved against, per morsel. The
	// expression is row-local, so a worker computing it over its own morsels is
	// equivalent to the serial planner computing it over the whole scan.

	zonePred := buildZonePredicate(scanNode.Predicate, scanNode.Schema)

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

	// ---- Pipeline schema detection ------------------------------------------
	// Probe the factory over a single row group to learn the schema the
	// aggregate will see — after scan, filters and any pre-projection. Group-by
	// and aggregate column indices are resolved against that schema, so it must
	// come from the same construction path the workers use.
	//
	// A probe failure means this plan cannot be described to the parallel
	// aggregate, so fall back to Physical: it is the authoritative
	// implementation and will either execute the plan or report the real error.
	// This keeps planner-detection gaps (for example a scan column list naming a
	// column the file does not contain) degrading to serial execution instead of
	// surfacing as a query error.

	// totalRGs >= 1 here, so a single-row-group probe range is always valid.
	probe, err := factory(ctx, 0, 1)
	if err != nil {
		return Physical(ctx, root)
	}
	pipelineSchema := probe.Schema()
	_ = probe.Close()

	// ---- Resolve aggregate config -------------------------------------------

	groupByIdxs, aggExprs, err := resolveAggConfig(aggNode, pipelineSchema)
	if err != nil {
		return Physical(ctx, root)
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
