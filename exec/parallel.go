package exec

import (
	"context"
	"fmt"
	"math"
)

// PipelineFactory creates an independent scan→filter→project pipeline covering
// row groups [rgStart, rgEnd). The caller must call Close() on the result.
type PipelineFactory func(ctx context.Context, rgStart, rgEnd int) (Operator, error)

// ParallelHashAggregate runs one independent pipeline per worker goroutine on a
// disjoint slice of row groups, accumulates partial aggregates locally, then
// merges all partial results in the calling goroutine.
//
// Supports the same GROUP BY / aggregate semantics as HashAggregate. The
// delegate field is populated lazily on the first Next() call.
type ParallelHashAggregate struct {
	factory    PipelineFactory
	totalRGs   int
	numWorkers int
	groupBy    []int
	aggExprs   []AggExpr
	schema     Schema

	delegate *HashAggregate // populated after setup()
}

// NewParallelHashAggregate constructs a ParallelHashAggregate.
// totalRGs is the total number of row groups in the file;
// numWorkers is capped to totalRGs if larger.
func NewParallelHashAggregate(
	factory PipelineFactory,
	totalRGs int,
	numWorkers int,
	groupBy []int,
	aggExprs []AggExpr,
	schema Schema,
) *ParallelHashAggregate {
	if numWorkers > totalRGs && totalRGs > 0 {
		numWorkers = totalRGs
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	return &ParallelHashAggregate{
		factory:    factory,
		totalRGs:   totalRGs,
		numWorkers: numWorkers,
		groupBy:    groupBy,
		aggExprs:   aggExprs,
		schema:     schema,
	}
}

func (p *ParallelHashAggregate) Schema() Schema { return p.schema }

func (p *ParallelHashAggregate) Next(ctx context.Context) (*Batch, error) {
	if p.delegate == nil {
		if err := p.setup(ctx); err != nil {
			return nil, err
		}
	}
	return p.delegate.Next(ctx)
}

func (p *ParallelHashAggregate) Close() error { return nil }

func (p *ParallelHashAggregate) setup(ctx context.Context) error {
	chunks := partitionRGs(p.totalRGs, p.numWorkers)
	if len(chunks) == 0 {
		// Empty table: emit a merged aggregate with zero rows.
		merged := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
		merged.done = true
		p.delegate = merged
		return nil
	}

	type workerResult struct {
		ha  *HashAggregate
		err error
	}
	ch := make(chan workerResult, len(chunks))

	for _, c := range chunks {
		go func(rgStart, rgEnd int) {
			pipeline, err := p.factory(ctx, rgStart, rgEnd)
			if err != nil {
				ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): factory: %w", rgStart, rgEnd, err)}
				return
			}
			defer pipeline.Close()

			ha := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
			for {
				batch, err := pipeline.Next(ctx)
				if err != nil {
					ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): %w", rgStart, rgEnd, err)}
					return
				}
				if batch == nil {
					break
				}
				if err := ha.accumulate(batch); err != nil {
					ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): accumulate: %w", rgStart, rgEnd, err)}
					return
				}
			}
			ch <- workerResult{ha: ha}
		}(c[0], c[1])
	}

	// Collect all worker results before merging (channel is buffered to len(chunks)
	// so goroutines never block even if we return early on error).
	results := make([]*HashAggregate, 0, len(chunks))
	var firstErr error
	for range chunks {
		r := <-ch
		if r.err != nil && firstErr == nil {
			firstErr = r.err
		}
		if r.ha != nil {
			results = append(results, r.ha)
		}
	}
	if firstErr != nil {
		return firstErr
	}

	merged := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
	for _, ha := range results {
		mergePartialAgg(merged, ha)
	}
	merged.done = true // skip consumeAll in Next()
	p.delegate = merged
	return nil
}

// newPartialAggregate creates a HashAggregate with no child, ready for direct
// accumulate() calls followed by delegate-based Next() emission.
func newPartialAggregate(groupBy []int, aggExprs []AggExpr, schema Schema) *HashAggregate {
	ha := &HashAggregate{
		groupBy:  groupBy,
		aggExprs: aggExprs,
		schema:   schema,
	}
	ha.initMaps()
	return ha
}

// mergePartialAgg folds src's accumulated state into dst.
// For each group key in src:
//   - If the key is new in dst: deep-copy the accumulators and sample.
//   - If the key exists: combine accumulators per AggKind/AccumType.
//
// groupCnt (used for AVG) is always summed.
func mergePartialAgg(dst, src *HashAggregate) {
	for _, key := range src.keys {
		srcAccs := src.groups[key]
		dstAccs, exists := dst.groups[key]
		if !exists {
			// New group — deep-copy.
			copied := make([]int64, len(srcAccs))
			copy(copied, srcAccs)
			dst.groups[key] = copied
			dst.keys = append(dst.keys, key)
			dst.samples[key] = src.samples[key]
		} else {
			for j, ae := range dst.aggExprs {
				switch ae.Kind {
				case AggCount:
					dstAccs[j] += srcAccs[j]
				case AggSum, AggAvg:
					// Both SUM and AVG store running sums; AVG count is in groupCnt.
					if ae.AccumType == TypeFloat64 {
						df := math.Float64frombits(uint64(dstAccs[j]))
						sf := math.Float64frombits(uint64(srcAccs[j]))
						dstAccs[j] = int64(math.Float64bits(df + sf))
					} else {
						dstAccs[j] += srcAccs[j]
					}
				case AggMin:
					if ae.AccumType == TypeFloat64 {
						df := math.Float64frombits(uint64(dstAccs[j]))
						sf := math.Float64frombits(uint64(srcAccs[j]))
						if sf < df {
							dstAccs[j] = srcAccs[j]
						}
					} else {
						if srcAccs[j] < dstAccs[j] {
							dstAccs[j] = srcAccs[j]
						}
					}
				case AggMax:
					if ae.AccumType == TypeFloat64 {
						df := math.Float64frombits(uint64(dstAccs[j]))
						sf := math.Float64frombits(uint64(srcAccs[j]))
						if sf > df {
							dstAccs[j] = srcAccs[j]
						}
					} else {
						if srcAccs[j] > dstAccs[j] {
							dstAccs[j] = srcAccs[j]
						}
					}
				}
			}
		}
		// Always sum the row counts (used for AVG finalization and correctness).
		dst.groupCnt[key] += src.groupCnt[key]
	}
}

// partitionRGs splits [0, total) into up to workers contiguous chunks.
// The last chunk absorbs the remainder. Returns [][2]int of {start, end} pairs.
func partitionRGs(total, workers int) [][2]int {
	if total == 0 || workers == 0 {
		return nil
	}
	if workers > total {
		workers = total
	}
	size := total / workers
	chunks := make([][2]int, workers)
	for i := range chunks {
		start := i * size
		end := start + size
		if i == workers-1 {
			end = total // last chunk absorbs remainder
		}
		chunks[i] = [2]int{start, end}
	}
	return chunks
}
