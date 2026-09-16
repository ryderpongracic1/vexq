package planner

import (
	"context"
	"fmt"
	"runtime"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// Parallel builds a physical plan in which the query's aggregate runs as a
// morsel-parallel operator, when the plan has one it can parallelize:
//
//	(post-aggregate operators →)? LogicalAggregate → (LogicalFilter →)? LogicalScan
//	(post-aggregate operators →)? LogicalAggregate → (LogicalFilter →)? LogicalJoin
//
// The post-aggregate operators are the chain Build places above an aggregate —
// the HAVING filter, the SELECT-list projection, DISTINCT, ORDER BY,
// LIMIT/OFFSET and the final renaming projection — in any combination. The
// aggregate's merged output is small (one row per group), so that chain is built
// serially over the parallel aggregate exactly as Physical would build it over a
// serial one, and the whole query returns the same rows either way.
//
// For an aggregate over a scan, the scan's row groups are partitioned across
// numWorkers goroutines, each running an independent scan+filter+pre-projection
// pipeline, and the partial aggregate results are merged in the calling
// goroutine. Aggregates over computed expressions (e.g. SUM(price * discount),
// the canonical TPC-H Q6 shape) are parallelized: each worker pipeline ends with
// the same pre-projection that the serial planner applies, materializing the
// expression into a synthetic column per morsel before local accumulation. The
// expression is row-local, so evaluating it per morsel is equivalent to
// evaluating it over the whole scan. Aggregates over an inner hash join are
// handled by tryParallelJoin ([planner/parallel_join.go]), which parallelizes the
// probe side.
//
// Float64 SUM/AVG results agree with serial execution to within IEEE-754
// rounding rather than bit-for-bit: partitioning changes the order of float
// additions, and float addition is not associative. This is a property of any
// partitioned float reduction; integer SUM/MIN/MAX and COUNT are exact. The
// project's correctness standard for float aggregates is the 1e-9 relative
// tolerance used by internal/goldentest.
//
// Falls back to Physical(ctx, root) when:
//   - the plan has no aggregate beneath its post-aggregate chain
//   - the chain has a LIMIT or OFFSET with no ORDER BY beneath it: which groups
//     survive would depend on group emission order, which differs between the
//     serial and parallel aggregates
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

	aggNode := aggregateUnderPostOps(root)
	if aggNode == nil {
		return Physical(ctx, root)
	}

	// Probe-side-parallel hash join (planner/parallel_join.go) first; it
	// declines every shape it does not handle.
	aggOp, matched, err := tryParallelJoin(ctx, aggNode, numWorkers)
	if err != nil {
		return nil, err
	}
	if !matched {
		if aggOp, matched, err = tryParallelScanAggregate(ctx, aggNode, numWorkers); err != nil {
			return nil, err
		}
	}
	if !matched {
		return Physical(ctx, root)
	}
	return buildAbove(root, aggNode, aggOp)
}

// aggregateUnderPostOps returns the aggregate beneath root's chain of
// post-aggregate operators, or nil when root has no such aggregate or when the
// chain applies LIMIT/OFFSET without an ORDER BY between it and the aggregate.
func aggregateUnderPostOps(root LogicalNode) *LogicalAggregate {
	unorderedLimit := false
	for node := root; ; {
		switch n := node.(type) {
		case *LogicalAggregate:
			if unorderedLimit {
				return nil
			}
			return n
		case *LogicalLimit:
			unorderedLimit = true
			node = n.Child
		case *LogicalSort:
			unorderedLimit = false
			node = n.Child
		case *LogicalProject:
			node = n.Child
		case *LogicalFilter:
			node = n.Child
		case *LogicalDistinct:
			node = n.Child
		default:
			return nil
		}
	}
}

// buildAbove builds the operators of node's subtree above target, with op
// standing in for target. Each builder closes its child on failure, so op is
// closed exactly once if anything fails.
func buildAbove(node LogicalNode, target *LogicalAggregate, op exec.Operator) (exec.Operator, error) {
	if agg, ok := node.(*LogicalAggregate); ok && agg == target {
		return op, nil
	}
	switch n := node.(type) {
	case *LogicalFilter:
		child, err := buildAbove(n.Child, target, op)
		if err != nil {
			return nil, err
		}
		return buildFilterOp(n, child)
	case *LogicalProject:
		child, err := buildAbove(n.Child, target, op)
		if err != nil {
			return nil, err
		}
		return buildProjectOp(n, child)
	case *LogicalSort:
		child, err := buildAbove(n.Child, target, op)
		if err != nil {
			return nil, err
		}
		return buildSortOp(n, child)
	case *LogicalLimit:
		child, err := buildAbove(n.Child, target, op)
		if err != nil {
			return nil, err
		}
		return buildLimitOp(n, child), nil
	case *LogicalDistinct:
		child, err := buildAbove(n.Child, target, op)
		if err != nil {
			return nil, err
		}
		return exec.NewDistinct(child), nil
	}
	_ = op.Close()
	return nil, fmt.Errorf("planner: parallel: unexpected %T above aggregate", node)
}

// tryParallelScanAggregate builds a ParallelHashAggregate for an aggregate over
// an optionally filtered scan. matched=false means the shape is not handled and
// the caller should fall back to serial planning.
func tryParallelScanAggregate(ctx context.Context, aggNode *LogicalAggregate, numWorkers int) (exec.Operator, bool, error) {
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
		return nil, false, nil
	}

	// ---- Row group count -----------------------------------------------------

	r, err := storage.Open(ctx, scanNode.FilePath)
	if err != nil {
		return nil, false, fmt.Errorf("planner: parallel: open %q: %w", scanNode.FilePath, err)
	}
	totalRGs := len(r.Meta().RowGroups)
	_ = r.Close()

	if totalRGs == 0 {
		// Empty table: serial execution (degenerate case).
		return nil, false, nil
	}

	// ---- Factory closure ----------------------------------------------------
	// Each call to factory(ctx, rgStart, rgEnd) builds an independent pipeline:
	//   TableScanRange → ScanPredFilter? → Filter? → PreProjection?
	// This is called once per worker inside ParallelHashAggregate.setup — a worker
	// repositions the pipeline it gets for each further morsel instead of asking
	// for a new one (exec.MorselPipeline) — and once
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
		scan, err := scanNode.openScan(fr, zonePred, rgStart, rgEnd)
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
				_ = scan.Close()
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

		preOp, err := buildPreProjection(aggNode, op)
		if err != nil {
			_ = op.Close()
			return nil, err
		}
		return preOp, nil
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

	// totalRGs >= 1 here, so a single-row-group probe range is always valid.
	probe, err := factory(ctx, 0, 1)
	if err != nil {
		return nil, false, nil
	}
	pipelineSchema := probe.Schema()
	_ = probe.Close()

	// ---- Resolve aggregate config -------------------------------------------

	groupByIdxs, aggExprs, err := resolveAggConfig(aggNode, pipelineSchema)
	if err != nil {
		return nil, false, nil
	}

	// Fall back to serial execution if any aggregate uses DISTINCT.
	// Partial COUNT(DISTINCT) counts from workers cannot be summed — the correct
	// approach requires shipping per-group value sets and unioning them at merge,
	// which is too invasive for the current mergePartialAgg ([]int64 accumulators).
	// Serial execution is correct; parallel COUNT(DISTINCT) is a future improvement.
	for _, ae := range aggExprs {
		if ae.Kind == exec.AggCountDistinct {
			return nil, false, nil
		}
	}

	// Compute the output schema (mirrors NewHashAggregate's logic).
	outSchema := aggOutputSchema(aggNode, pipelineSchema, groupByIdxs, aggExprs)

	// morselSize=0 → exec package uses defaultMorselSize (1 row group).
	return exec.NewParallelHashAggregate(factory, totalRGs, numWorkers, 0, groupByIdxs, aggExprs, outSchema), true, nil
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
