package exec

import (
	"context"
	"fmt"
)

// ---- Schema-only stub -------------------------------------------------------

// schemaOnly is an operator that exposes a schema and immediately reports EOF.
// The planner uses it to derive the schema produced by the operators it will
// stack above the join (residual filter, aggregate pre-projection) without
// opening any files or building a hash table.
type schemaOnly struct{ schema Schema }

// NewSchemaOnly returns an operator with the given schema that yields no rows.
func NewSchemaOnly(schema Schema) Operator { return &schemaOnly{schema: schema} }

func (s *schemaOnly) Schema() Schema                       { return s.schema }
func (s *schemaOnly) Next(context.Context) (*Batch, error) { return nil, nil }
func (s *schemaOnly) Close() error                         { return nil }

// ---- Shared build side -----------------------------------------------------

// SharedHashTable is a build-side hash table that many probe goroutines read
// concurrently. It is immutable once BuildSharedHashTable returns.
//
// Concurrency contract (happens-before):
//
//	BuildSharedHashTable populates the map in the goroutine that calls it and
//	returns before any worker goroutine is created. The Go memory model
//	guarantees that everything sequenced before a `go` statement is observed by
//	the goroutine it starts, so every write performed during the build phase
//	happens-before every read performed by a worker. Nothing mutates the map
//	after the build completes — workers only index into it — and concurrent
//	reads of a Go map with no concurrent writer are race-free. This invariant is
//	what makes probe-side parallelism safe without a lock or a per-worker copy.
//
// Ownership: the SharedHashTable outlives every probe join built on top of it.
// HashJoin.Close therefore must not free it (see HashJoin.Close).
type SharedHashTable struct {
	table   map[int64][]buildRow
	schema  Schema
	keyIdx  int
	numRows int
}

// Schema returns the build side's schema.
func (s *SharedHashTable) Schema() Schema { return s.schema }

// NumRows returns the number of build rows inserted (NULL-keyed rows excluded).
func (s *SharedHashTable) NumRows() int { return s.numRows }

// NumKeys returns the number of distinct join keys in the table.
func (s *SharedHashTable) NumKeys() int { return len(s.table) }

// BuildSharedHashTable drains build and materialises its rows into a hash table
// keyed on buildKeyIdx. It does not close build — the caller owns that operator.
func BuildSharedHashTable(ctx context.Context, build Operator, buildKeyIdx int) (*SharedHashTable, error) {
	schema := build.Schema()
	if buildKeyIdx < 0 || buildKeyIdx >= len(schema.Fields) {
		return nil, fmt.Errorf("exec: shared hash table: build key %d out of range", buildKeyIdx)
	}
	table := make(map[int64][]buildRow)
	if err := buildHashTableFrom(ctx, build, buildKeyIdx, len(schema.Fields), table); err != nil {
		return nil, err
	}
	n := 0
	for _, rows := range table {
		n += len(rows)
	}
	return &SharedHashTable{table: table, schema: schema, keyIdx: buildKeyIdx, numRows: n}, nil
}

// NewHashJoinShared returns a HashJoin that probes an existing SharedHashTable
// instead of materialising its own build side. Each worker goroutine gets its
// own HashJoin (private probe cursor and match buffer) over the same table.
func NewHashJoinShared(sht *SharedHashTable, probe Operator, probeKeyIdx int) (*HashJoin, error) {
	if sht == nil {
		return nil, fmt.Errorf("exec: hash join: nil shared hash table")
	}
	pSchema := probe.Schema()
	if probeKeyIdx < 0 || probeKeyIdx >= len(pSchema.Fields) {
		return nil, fmt.Errorf("exec: hash join: probe key %d out of range", probeKeyIdx)
	}
	outFields := make([]Field, 0, len(sht.schema.Fields)+len(pSchema.Fields))
	outFields = append(outFields, sht.schema.Fields...)
	outFields = append(outFields, pSchema.Fields...)

	return &HashJoin{
		// build intentionally nil: the table is owned by the caller.
		probe:       probe,
		buildKey:    sht.keyIdx,
		probeKey:    probeKeyIdx,
		buildSchema: sht.schema,
		schema:      Schema{Fields: outFields},
		hashTable:   sht.table,
		buildDone:   true,
	}, nil
}

// ---- Parallel probe-side join + aggregate -----------------------------------

// BuildFactory constructs the build-side operator subtree. Called exactly once,
// in the calling goroutine, before any worker starts.
type BuildFactory func(ctx context.Context) (Operator, error)

// AboveJoinFactory wraps the per-worker join output with the operators that must
// run above the join inside each worker pipeline — a residual filter over joined
// rows, and/or the aggregate's pre-projection for expression aggregates. It must
// be deterministic: every worker's wrapped pipeline has to expose the same schema.
type AboveJoinFactory func(Operator) (Operator, error)

// ParallelHashJoinAggregate parallelises the probe side of an inner hash join
// feeding a hash aggregate:
//
//	Aggregate → HashJoin(build = subtree, probe = (Filter →)? Scan)
//
// Execution has two phases:
//
//  1. Build (serial): the build-side subtree is evaluated once in the calling
//     goroutine into a SharedHashTable.
//  2. Probe (parallel): numWorkers goroutines claim row-group morsels of the
//     probe scan from a shared atomic cursor — the same dynamic scheduling
//     ParallelHashAggregate uses — and each runs an independent
//     Scan → Filter? → HashJoinShared → PreProjection? → partial HashAggregate
//     pipeline. Partial aggregates are merged in the calling goroutine by
//     mergePartialAgg, so float64 SUM/MIN/MAX stay IEEE-correct.
//
// The build side is not parallelised in this phase: concurrent inserts would
// need either a lock (serialising the hot loop) or radix-partitioned per-worker
// tables. Amdahl therefore bounds the achievable speedup by the build-side
// fraction of total runtime.
type ParallelHashJoinAggregate struct {
	buildFactory BuildFactory
	buildKeyIdx  int
	probeFactory PipelineFactory
	probeKeyIdx  int
	aboveJoin    AboveJoinFactory // may be nil

	totalRGs   int // probe-side row groups
	numWorkers int
	morselSize int

	groupBy  []int
	aggExprs []AggExpr
	schema   Schema

	delegate *HashAggregate // populated after setup()

	// emptyGlobal holds the single output row that SQL requires from a global
	// aggregate (no GROUP BY) over zero input rows. See finishMerged.
	emptyGlobal        *Batch
	emptyGlobalEmitted bool
}

// NewParallelHashJoinAggregate constructs a ParallelHashJoinAggregate.
// totalRGs is the probe scan's row group count; numWorkers is capped to it.
// morselSize of 0 selects defaultMorselSize.
func NewParallelHashJoinAggregate(
	buildFactory BuildFactory,
	buildKeyIdx int,
	probeFactory PipelineFactory,
	probeKeyIdx int,
	aboveJoin AboveJoinFactory,
	totalRGs int,
	numWorkers int,
	morselSize int,
	groupBy []int,
	aggExprs []AggExpr,
	schema Schema,
) *ParallelHashJoinAggregate {
	if numWorkers > totalRGs && totalRGs > 0 {
		numWorkers = totalRGs
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	if morselSize < 1 {
		morselSize = defaultMorselSize
	}
	return &ParallelHashJoinAggregate{
		buildFactory: buildFactory,
		buildKeyIdx:  buildKeyIdx,
		probeFactory: probeFactory,
		probeKeyIdx:  probeKeyIdx,
		aboveJoin:    aboveJoin,
		totalRGs:     totalRGs,
		numWorkers:   numWorkers,
		morselSize:   morselSize,
		groupBy:      groupBy,
		aggExprs:     aggExprs,
		schema:       schema,
	}
}

func (p *ParallelHashJoinAggregate) Schema() Schema { return p.schema }

func (p *ParallelHashJoinAggregate) Next(ctx context.Context) (*Batch, error) {
	if p.delegate == nil {
		if err := p.setup(ctx); err != nil {
			return nil, err
		}
	}
	if p.emptyGlobal != nil {
		if p.emptyGlobalEmitted {
			return nil, nil
		}
		p.emptyGlobalEmitted = true
		return p.emptyGlobal, nil
	}
	return p.delegate.Next(ctx)
}

func (p *ParallelHashJoinAggregate) Close() error { return nil }

func (p *ParallelHashJoinAggregate) setup(ctx context.Context) error {
	// ---- Phase 1: build, serially, before any worker exists ----------------
	buildOp, err := p.buildFactory(ctx)
	if err != nil {
		return fmt.Errorf("exec: parallel join: build side: %w", err)
	}
	sht, buildErr := BuildSharedHashTable(ctx, buildOp, p.buildKeyIdx)
	_ = buildOp.Close()
	if buildErr != nil {
		return fmt.Errorf("exec: parallel join: %w", buildErr)
	}

	// An empty build side means the inner join emits no rows, so the aggregate
	// sees no input — identical to the serial result, without scanning the probe.
	if p.totalRGs == 0 || sht.NumKeys() == 0 {
		p.finishMerged(newPartialAggregate(p.groupBy, p.aggExprs, p.schema))
		return nil
	}

	// ---- Phase 2: probe, in parallel ---------------------------------------
	partials, err := runMorselAggWorkers(ctx, p.totalRGs, p.numWorkers, p.morselSize,
		p.groupBy, p.aggExprs, p.schema,
		func(wCtx context.Context, rgStart, rgEnd int) (Operator, error) {
			probe, err := p.probeFactory(wCtx, rgStart, rgEnd)
			if err != nil {
				return nil, err
			}
			join, err := NewHashJoinShared(sht, probe, p.probeKeyIdx)
			if err != nil {
				_ = probe.Close()
				return nil, err
			}
			var op Operator = join
			if p.aboveJoin != nil {
				op, err = p.aboveJoin(op)
				if err != nil {
					_ = join.Close()
					return nil, err
				}
			}
			return op, nil
		})
	if err != nil {
		return err
	}

	merged := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
	for _, ha := range partials {
		mergePartialAgg(merged, ha)
	}
	p.finishMerged(merged)
	return nil
}

// finishMerged installs the merged partial aggregate as the output delegate.
//
// A pre-merged aggregate never drains a child, so HashAggregate.Next skips the
// empty-input rule it applies when it does: SQL requires a global aggregate (no
// GROUP BY) over zero rows to emit exactly one row — COUNT 0, NULL for
// SUM/AVG/MIN/MAX. Applying it here keeps the parallel result identical to the
// serial one for queries like `SELECT COUNT(*) FROM a, b WHERE <no matches>`.
func (p *ParallelHashJoinAggregate) finishMerged(merged *HashAggregate) {
	merged.done = true
	p.delegate = merged
	if len(merged.keys) == 0 && len(p.groupBy) == 0 {
		p.emptyGlobal = merged.buildEmptyGlobalResult()
	}
}

// runMorselAggWorkers runs numWorkers goroutines that dynamically claim
// row-group morsels from a shared atomic cursor, build a pipeline per morsel via
// mkPipeline, and accumulate into a goroutine-local partial HashAggregate. It
// returns the partial aggregates for the caller to merge.
//
// This is the same dynamic scheduling ParallelHashAggregate.setup performs
// inline; factored out here so the join operator does not duplicate it.
func runMorselAggWorkers(
	ctx context.Context,
	totalRGs, numWorkers, morselSize int,
	groupBy []int,
	aggExprs []AggExpr,
	schema Schema,
	mkPipeline func(ctx context.Context, rgStart, rgEnd int) (Operator, error),
) ([]*HashAggregate, error) {
	q := &morselQueue{end: int64(totalRGs)}

	type workerResult struct {
		ha  *HashAggregate
		err error
	}
	// Buffered by worker count so no goroutine blocks on send if we return early.
	ch := make(chan workerResult, numWorkers)

	msize := int64(morselSize)
	for range numWorkers {
		go func() {
			ha := newPartialAggregate(groupBy, aggExprs, schema)
			for {
				start, stop, ok := q.claim(msize)
				if !ok {
					break // queue exhausted; this worker is done
				}
				pipeline, err := mkPipeline(ctx, int(start), int(stop))
				if err != nil {
					ch <- workerResult{err: fmt.Errorf("parallel join worker [%d,%d): pipeline: %w", start, stop, err)}
					return
				}
				for {
					batch, err := pipeline.Next(ctx)
					if err != nil {
						pipeline.Close()
						ch <- workerResult{err: fmt.Errorf("parallel join worker [%d,%d): %w", start, stop, err)}
						return
					}
					if batch == nil {
						break
					}
					if err := ha.accumulate(batch); err != nil {
						pipeline.Close()
						ch <- workerResult{err: fmt.Errorf("parallel join worker [%d,%d): accumulate: %w", start, stop, err)}
						return
					}
				}
				pipeline.Close()
			}
			// Materialize integer-key state to string maps before merge.
			if ha.intKey.enabled && len(ha.intKey.intKeys) > 0 {
				ha.intKey.materializeToStringMaps(ha)
			}
			ch <- workerResult{ha: ha}
		}()
	}

	results := make([]*HashAggregate, 0, numWorkers)
	var firstErr error
	for range numWorkers {
		r := <-ch
		if r.err != nil && firstErr == nil {
			firstErr = r.err
		}
		if r.ha != nil {
			results = append(results, r.ha)
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}
