package exec

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
)

// PipelineFactory creates an independent scan→filter→project pipeline covering
// row groups [rgStart, rgEnd). The caller must call Close() on the result.
type PipelineFactory func(ctx context.Context, rgStart, rgEnd int) (Operator, error)

// morselQueue is a lock-free work queue for morsel-driven parallelism.
// Workers call claim() to atomically reserve the next chunk of row groups.
// The atomic counter occupies its own 64-byte cache line so contention on
// `next` does not cause false-sharing with any other struct fields.
type morselQueue struct {
	next atomic.Int64 // workers atomically advance this to claim morsels
	_    [56]byte     // padding: atomic.Int64 is 8 B; 8+56 = 64 B (one cache line)
	end  int64        // read-only after construction; no write contention
}

// claim reserves [start, stop) row groups of the given size.
// Returns ok=false when the queue is exhausted.
func (q *morselQueue) claim(size int64) (start, stop int64, ok bool) {
	start = q.next.Add(size) - size
	if start >= q.end {
		return 0, 0, false
	}
	stop = start + size
	if stop > q.end {
		stop = q.end
	}
	return start, stop, true
}

// ParallelHashAggregate runs one goroutine per worker. Each goroutine
// dynamically claims row-group morsels from a shared atomic counter, runs an
// independent pipeline on each morsel, and accumulates partial results locally.
// After all workers drain the queue the calling goroutine merges the partial
// aggregates — no shared mutable state during execution.
//
// Unlike the previous static-partition design, workers self-schedule: a
// goroutine whose morsels have high filter selectivity finishes fast and claims
// more work, eliminating stragglers caused by uneven per-row-group cost.
type ParallelHashAggregate struct {
	factory    PipelineFactory
	totalRGs   int
	numWorkers int
	morselSize int // row groups per morsel; 0 → defaultMorselSize (1)
	groupBy    []int
	aggExprs   []AggExpr
	schema     Schema

	delegate *HashAggregate // populated after setup()
}

// defaultMorselSize is one row group (65,536 rows). At SF=1 a single lineitem
// row group takes ~5–20 ms to scan+filter — contention on the atomic counter
// is at most ~1000 CAS ops/sec across 10 workers, negligible overhead.
const defaultMorselSize = 1

// NewParallelHashAggregate constructs a ParallelHashAggregate.
// morselSize is the number of row groups per morsel (0 = defaultMorselSize).
// numWorkers is capped to totalRGs if larger.
func NewParallelHashAggregate(
	factory PipelineFactory,
	totalRGs int,
	numWorkers int,
	morselSize int,
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
	if morselSize < 1 {
		morselSize = defaultMorselSize
	}
	return &ParallelHashAggregate{
		factory:    factory,
		totalRGs:   totalRGs,
		numWorkers: numWorkers,
		morselSize: morselSize,
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
	if p.totalRGs == 0 {
		merged := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
		merged.done = true
		p.delegate = merged
		return nil
	}

	q := &morselQueue{end: int64(p.totalRGs)}

	type workerResult struct {
		ha  *HashAggregate
		err error
	}
	// Buffer = numWorkers so goroutines never block on send even if we return early.
	ch := make(chan workerResult, p.numWorkers)

	msize := int64(p.morselSize)
	for range p.numWorkers {
		go func() {
			ha := newPartialAggregate(p.groupBy, p.aggExprs, p.schema)
			for {
				start, stop, ok := q.claim(msize)
				if !ok {
					break // queue exhausted; this worker is done
				}
				pipeline, err := p.factory(ctx, int(start), int(stop))
				if err != nil {
					ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): factory: %w", start, stop, err)}
					return
				}
				for {
					batch, err := pipeline.Next(ctx)
					if err != nil {
						pipeline.Close()
						ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): %w", start, stop, err)}
						return
					}
					if batch == nil {
						break
					}
					if err := ha.accumulate(batch); err != nil {
						pipeline.Close()
						ch <- workerResult{err: fmt.Errorf("parallel agg worker [%d,%d): accumulate: %w", start, stop, err)}
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

	results := make([]*HashAggregate, 0, p.numWorkers)
	var firstErr error
	for range p.numWorkers {
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
	merged.done = true
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
			// Deep-copy non-null counts.
			if srcNN := src.aggNonNull[key]; srcNN != nil {
				copiedNN := make([]int64, len(srcNN))
				copy(copiedNN, srcNN)
				dst.aggNonNull[key] = copiedNN
			}
		} else {
			for j, ae := range dst.aggExprs {
				switch ae.Kind {
				case AggCount:
					dstAccs[j] += srcAccs[j]
				case AggSum, AggAvg:
					// Both SUM and AVG store running sums; AVG count is in aggNonNull.
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
			// Existing group: add this partial aggregate's non-null counts.
			if dstNN, srcNN := dst.aggNonNull[key], src.aggNonNull[key]; dstNN != nil && srcNN != nil {
				for j := range dstNN {
					dstNN[j] += srcNN[j]
				}
			}
		}
		// Always sum the row counts (legacy).
		dst.groupCnt[key] += src.groupCnt[key]
	}
}
