package planner

import (
	"context"
	"fmt"
	"runtime"

	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/storage"
)

// Parallel returns a ParallelHashAggregate when the plan matches the pattern:
//
//	LogicalAggregate → (LogicalFilter →)? LogicalScan
//
// It partitions the scan's row groups evenly across numWorkers goroutines,
// each running an independent scan+filter+pre-projection pipeline, then merges
// the partial aggregate results in the calling goroutine.
//
// Falls back to Physical(ctx, root) when:
//   - root is not a LogicalAggregate
//   - the aggregate child (after peeling an optional LogicalFilter) is not a LogicalScan
//     (e.g., the plan involves a HashJoin)
//
// numWorkers <= 0 defaults to runtime.NumCPU().
func Parallel(ctx context.Context, root LogicalNode, numWorkers int) (exec.Operator, error) {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// ---- Plan shape detection ------------------------------------------------

	aggNode, ok := root.(*LogicalAggregate)
	if !ok {
		return Physical(ctx, root) // not an aggregate at the top
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

	return exec.NewParallelHashAggregate(factory, totalRGs, numWorkers, groupByIdxs, aggExprs, outSchema), nil
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
