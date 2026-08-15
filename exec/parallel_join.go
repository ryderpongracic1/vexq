package exec

import (
	"context"
	"fmt"
	"time"
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

// ---- Radix partitioning ----------------------------------------------------

const (
	// radixMinBuildRows is the build-side row count below which the build side is
	// left unpartitioned: one .vxq row group. Below it there is only one morsel to
	// distribute, so there is no parallel build to win, and the two-pass shuffle
	// would be pure overhead.
	radixMinBuildRows = 65536

	// radixTargetRows is the rows-per-partition RadixBitsFor aims for, and
	// radixMinBits / radixMaxBits clamp the partition count to [16, 256].
	//
	// These are measured, not derived. BenchmarkRadixBuild over a 262,144-row
	// build side on an Intel Xeon 6975P-C (4 cores available) reports, in ms:
	//
	//	partitions |  1 worker | 2 workers | 4 workers
	//	-----------+-----------+-----------+----------
	//	         1 |     113.5 |      91.6 |      88.3
	//	        16 |      86.0 |      57.6 |      50.2
	//	        64 |      71.9 |      52.5 |      47.5
	//	       256 |      72.9 |      54.9 |      49.1
	//
	// Most of the gain arrives by 16 partitions, the rest by 64, and 256 is no
	// better than 64 — so 64 is the ceiling. The 1-worker column is worth reading
	// on its own: partitioning is 1.58x faster single-threaded, before any
	// parallelism, because growing 64 small maps rehashes far less than growing
	// one big one.
	radixTargetRows = 8192
	radixMinBits    = 4
	radixMaxBits    = 6

	// radixHardMaxBits bounds what the table constructors accept, so a caller
	// asking for an absurd bit count gets a large-but-finite partition array
	// rather than an allocation failure. RadixBitsFor never returns more than
	// radixMaxBits; this only exists to keep the constructors total, and lets
	// BenchmarkRadixBuild and BenchmarkPartitionedProbe measure past the policy
	// ceiling — which is how the ceiling was chosen.
	radixHardMaxBits = 12
)

// RadixBitsFor chooses how many radix bits to partition a build side of estRows
// rows on, returning 0 for "do not partition" below radixMinBuildRows.
//
// estRows only sizes partitions — it never affects which rows join — so an
// inaccurate estimate costs at most some efficiency. The planner's estimate is
// the build scan's row count before filtering, which over-estimates for a
// selective build-side predicate; over-partitioning is the cheap direction to be
// wrong in, since an unused partition is an empty map.
func RadixBitsFor(estRows int) int {
	if estRows < radixMinBuildRows {
		return 0
	}
	bits := radixMinBits
	for bits < radixMaxBits && (1<<bits)*radixTargetRows < estRows {
		bits++
	}
	return bits
}

// radixHash mixes a join key so the low bits used for partitioning depend on
// every bit of the key. This is the murmur3 64-bit finalizer; without the mixing
// step, masking the low bits of a TPC-H order key would partition on the sparse
// low-order pattern dbgen leaves in the key space and skew partition sizes.
//
// Build and probe must agree on this function — a mismatch would silently drop
// matches rather than fail, which is why every radix test asserts against serial
// results rather than only against itself.
func radixHash(key int64) uint64 {
	x := uint64(key)
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return x
}

// radixPart returns the partition index for key. A zero mask means the table is
// unpartitioned, and short-circuiting it keeps an unpartitioned probe from
// paying for the hash at all.
func radixPart(key int64, mask uint64) int {
	if mask == 0 {
		return 0
	}
	return int(radixHash(key) & mask)
}

// ---- Shared build side -----------------------------------------------------

// SharedHashTable is a build-side hash table that many probe goroutines read
// concurrently. It is immutable once its constructor returns.
//
// The table is radix-partitioned: parts holds a power-of-two number of maps and
// a key lives in parts[radixHash(key) & partMask]. A single partition (partMask
// 0) is the unpartitioned case and behaves exactly like the one big map that
// preceded partitioning.
//
// What partitioning buys, and what it does not:
//
//   - A lock-free parallel build. Each partition is owned by exactly one
//     goroutine during assembly, so workers never write the same map. This is the
//     main reason partitioning is here.
//   - Cheaper map growth, even single-threaded. Growing one map to N entries
//     rehashes progressively larger bucket arrays; growing 2^k maps to N/2^k
//     entries each does the same number of insertions over far smaller arrays.
//     Measured at 1.58x on a one-worker build (see radixTargetRows).
//   - It does NOT make the probe cache-resident. That was the original
//     hypothesis, and BenchmarkPartitionedProbe — which holds the build rows and
//     probe keys fixed and varies only the partition count — measures no
//     improvement from 1 to 256 partitions. The reason is that only the build
//     side is partitioned: a probe batch's keys are spread across every
//     partition, so the memory the probe touches per unit time is the whole
//     table either way. Getting the cache win needs the probe stream partitioned
//     too, which means buffering probe rows per partition at morsel granularity —
//     incompatible with a streaming operator whose TableScan reuses its decode
//     buffers between batches. That remains open work; nothing here claims it.
//
// Concurrency contract (happens-before):
//
//	Every constructor finishes populating the table in the goroutine that calls
//	it — collecting from its own workers through a channel, whose send/receive
//	pairs are synchronisation points — and returns before any probe goroutine is
//	created. The Go memory model guarantees that everything sequenced before a
//	`go` statement is observed by the goroutine it starts, so every write
//	performed during the build phase happens-before every read performed by a
//	probe worker. Nothing mutates the table afterwards — workers only index into
//	it — and concurrent reads of a Go map with no concurrent writer are
//	race-free. This invariant is what makes probe-side parallelism safe without a
//	lock or a per-worker copy.
//
// Ownership: the SharedHashTable outlives every probe join built on top of it.
// HashJoin.Close therefore must not free it (see HashJoin.Close).
type SharedHashTable struct {
	parts    []map[int64][]buildRow
	partMask uint64
	schema   Schema
	keyIdx   int
	numRows  int
}

// Schema returns the build side's schema.
func (s *SharedHashTable) Schema() Schema { return s.schema }

// NumRows returns the number of build rows inserted (NULL-keyed rows excluded).
func (s *SharedHashTable) NumRows() int { return s.numRows }

// NumKeys returns the number of distinct join keys in the table.
func (s *SharedHashTable) NumKeys() int {
	n := 0
	for _, m := range s.parts {
		n += len(m)
	}
	return n
}

// NumPartitions returns the number of radix partitions (1 when unpartitioned).
func (s *SharedHashTable) NumPartitions() int { return len(s.parts) }

// PartitionRows returns the build-row count per partition. Exposed so tests can
// assert both that partitioning spreads keys and that a deliberately skewed key
// set still produces correct results.
func (s *SharedHashTable) PartitionRows() []int {
	out := make([]int, len(s.parts))
	for p, m := range s.parts {
		for _, rows := range m {
			out[p] += len(rows)
		}
	}
	return out
}

// newSharedHashTable allocates numParts empty partitions.
func newSharedHashTable(schema Schema, keyIdx, radixBits int) *SharedHashTable {
	numParts := 1 << radixBits
	parts := make([]map[int64][]buildRow, numParts)
	for i := range parts {
		parts[i] = make(map[int64][]buildRow)
	}
	return &SharedHashTable{
		parts:    parts,
		partMask: uint64(numParts - 1),
		schema:   schema,
		keyIdx:   keyIdx,
	}
}

// BuildSharedHashTable drains build and materialises its rows into a single
// unpartitioned hash table. It does not close build — the caller owns that
// operator.
func BuildSharedHashTable(ctx context.Context, build Operator, buildKeyIdx int) (*SharedHashTable, error) {
	return BuildSharedHashTableRadix(ctx, build, buildKeyIdx, 0)
}

// BuildSharedHashTableRadix drains build in the calling goroutine into a
// radix-partitioned hash table of 2^radixBits partitions. radixBits of 0 yields
// one partition, which is the unpartitioned table.
//
// This is the build path for a build side that cannot be split into row-group
// morsels — a nested join subtree the planner could not decompose. It still
// benefits from cheaper map growth, and it produces a table with the same probe
// layout as the parallel builder, so the probe path has one shape only.
//
// Row order within a key is drain order, identical to the serial HashJoin.
func BuildSharedHashTableRadix(ctx context.Context, build Operator, buildKeyIdx, radixBits int) (*SharedHashTable, error) {
	schema := build.Schema()
	if buildKeyIdx < 0 || buildKeyIdx >= len(schema.Fields) {
		return nil, fmt.Errorf("exec: shared hash table: build key %d out of range", buildKeyIdx)
	}
	sht := newSharedHashTable(schema, buildKeyIdx, clampRadixBits(radixBits))
	err := forEachBuildRow(ctx, build, buildKeyIdx, len(schema.Fields), func(key int64, row buildRow) {
		m := sht.parts[radixPart(key, sht.partMask)]
		m[key] = append(m[key], row)
		sht.numRows++
	})
	if err != nil {
		return nil, err
	}
	return sht, nil
}

// clampRadixBits keeps a caller-supplied bit count inside [0, radixHardMaxBits].
// 0 is allowed and means unpartitioned. Note this is the constructors' structural
// bound, not the planner's policy ceiling — that is radixMaxBits, applied by
// RadixBitsFor.
func clampRadixBits(bits int) int {
	if bits < 0 {
		return 0
	}
	if bits > radixHardMaxBits {
		return radixHardMaxBits
	}
	return bits
}

// keyedBuildRow is a materialised build row plus its join key, the unit the
// parallel build shuffles between its two passes. The key is carried explicitly
// rather than re-derived from the row because a string-typed key column stores
// its value in strVals, leaving values[keyIdx] zero.
type keyedBuildRow struct {
	key int64
	row buildRow
}

// BuildSharedHashTableParallel materialises the build side with numWorkers
// goroutines and assembles a radix-partitioned hash table. factory produces an
// independent build pipeline over row groups [rgStart, rgEnd) of the build side,
// exactly as PipelineFactory does for the probe side; schema is the schema every
// such pipeline exposes.
//
// Two passes, neither of which takes a lock:
//
//  1. Partition, parallel over build row groups. Each worker claims morsels from
//     a shared atomic cursor, runs factory over the morsel, and appends every
//     materialised row into a per-(morsel, partition) bucket. A morsel is
//     claimed by exactly one worker, so no two goroutines write the same bucket
//     slot.
//  2. Assemble, parallel over partitions. Each partition is claimed by exactly
//     one goroutine, which builds that partition's map by walking morsels in
//     ascending index order. Distinct goroutines write distinct parts[p], so
//     again no two write the same location.
//
// Happens-before: pass 1's bucket writes are published to the calling goroutine
// by each worker's send on its result channel; pass 2's goroutines are started
// only after every pass-1 worker has been received, so they observe all bucket
// writes; pass 2's writes to parts are published to the caller the same way. The
// function returns before any probe worker exists, so SharedHashTable's
// read-only contract holds unchanged.
//
// Determinism: pass 1 preserves within-morsel order, pass 2 walks morsels in
// index order, and morsel index order is row-group order — so the row slice
// stored for each key is identical to what a serial drain of the same build side
// produces, whatever order workers happened to claim morsels in. Join output is
// therefore per-key order-identical to the serial join, and aggregate results
// are exact for integers rather than merely equivalent.
func BuildSharedHashTableParallel(
	ctx context.Context,
	factory PipelineFactory,
	schema Schema,
	buildKeyIdx int,
	totalRGs, numWorkers, morselSize, radixBits int,
) (*SharedHashTable, error) {
	if buildKeyIdx < 0 || buildKeyIdx >= len(schema.Fields) {
		return nil, fmt.Errorf("exec: shared hash table: build key %d out of range", buildKeyIdx)
	}
	if morselSize < 1 {
		morselSize = defaultMorselSize
	}
	sht := newSharedHashTable(schema, buildKeyIdx, clampRadixBits(radixBits))
	numParts := len(sht.parts)
	if totalRGs < 1 {
		return sht, nil
	}

	numMorsels := (totalRGs + morselSize - 1) / morselSize
	// Pass 1's parallelism is bounded by the number of morsels, pass 2's by the
	// number of partitions. They are capped separately so a build side with few
	// row groups but many partitions still assembles in parallel.
	pass1Workers := min(numWorkers, numMorsels)
	if pass1Workers < 1 {
		pass1Workers = 1
	}
	pass2Workers := min(numWorkers, numParts)
	if pass2Workers < 1 {
		pass2Workers = 1
	}

	// ---- Pass 1: partition ---------------------------------------------------
	// buckets[morselIdx][partition]. Pre-allocated so workers only ever write
	// the slot for a morsel they claimed.
	buckets := make([][][]keyedBuildRow, numMorsels)
	q := &morselQueue{end: int64(totalRGs)}
	msize := int64(morselSize)
	errCh := make(chan error, pass1Workers)

	// A failed morsel makes every other worker's output unusable, so cancel them
	// rather than letting them finish work that is about to be discarded:
	// TableScan.Next checks ctx.Err() once per batch, so a cancelled worker stops
	// within one batch. Each worker still sends exactly once, so the collection
	// loop below cannot deadlock, and firstErr is set before cancel() so the
	// genuine failure is reported rather than a cancellation it caused.
	wCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	for range pass1Workers {
		go func() {
			for {
				start, stop, ok := q.claim(msize)
				if !ok {
					break // queue exhausted; this worker is done
				}
				pipeline, err := factory(wCtx, int(start), int(stop))
				if err != nil {
					errCh <- fmt.Errorf("radix build worker [%d,%d): pipeline: %w", start, stop, err)
					return
				}
				local := make([][]keyedBuildRow, numParts)
				err = forEachBuildRow(wCtx, pipeline, buildKeyIdx, len(schema.Fields),
					func(key int64, row buildRow) {
						p := radixPart(key, sht.partMask)
						local[p] = append(local[p], keyedBuildRow{key: key, row: row})
					})
				pipeline.Close()
				if err != nil {
					errCh <- fmt.Errorf("radix build worker [%d,%d): %w", start, stop, err)
					return
				}
				buckets[int(start)/morselSize] = local
			}
			errCh <- nil
		}()
	}
	var firstErr error
	for range pass1Workers {
		if err := <-errCh; err != nil && firstErr == nil {
			firstErr = err
			cancel() // stop the surviving workers early
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}

	// ---- Pass 2: assemble ----------------------------------------------------
	// done carries no value because pass 2 cannot fail: it only reads buckets and
	// writes maps.
	pq := &morselQueue{end: int64(numParts)}
	partRows := make([]int, numParts)
	done := make(chan struct{}, pass2Workers)
	for range pass2Workers {
		go func() {
			for {
				p64, _, ok := pq.claim(1)
				if !ok {
					break
				}
				p := int(p64)
				// Size the map from the exact row count so assembly does not
				// rehash. Duplicate keys make this an over-estimate of the key
				// count, which costs memory but never a rehash.
				n := 0
				for mi := range buckets {
					if buckets[mi] != nil {
						n += len(buckets[mi][p])
					}
				}
				m := make(map[int64][]buildRow, n)
				for mi := range buckets {
					if buckets[mi] == nil {
						continue
					}
					for _, kr := range buckets[mi][p] {
						m[kr.key] = append(m[kr.key], kr.row)
					}
					// Release the bucket so its backing array can be collected
					// while later partitions are still assembling. Only this
					// goroutine reads or writes slot [mi][p].
					buckets[mi][p] = nil
				}
				sht.parts[p] = m
				partRows[p] = n
			}
			done <- struct{}{}
		}()
	}
	for range pass2Workers {
		<-done
	}
	for _, n := range partRows {
		sht.numRows += n
	}
	return sht, nil
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
		parts:       sht.parts,
		partMask:    sht.partMask,
		buildDone:   true,
	}, nil
}

// ---- Parallel probe-side join + aggregate -----------------------------------

// BuildFactory constructs the build-side operator subtree. Called exactly once,
// in the calling goroutine, before any worker starts.
type BuildFactory func(ctx context.Context) (Operator, error)

// PartitionedBuild describes a build side that can be split into row-group
// morsels, which is what lets the build phase run in parallel.
type PartitionedBuild struct {
	// Factory produces an independent build pipeline over row groups
	// [rgStart, rgEnd), the same contract PipelineFactory has for the probe side.
	Factory PipelineFactory
	// Schema is the schema every pipeline Factory returns exposes. It must match
	// the schema the serial BuildFactory produces, because the join key index was
	// resolved against that schema; the planner checks this when it constructs a
	// PartitionedBuild.
	Schema Schema
	// TotalRGs is the number of row groups Factory partitions over.
	TotalRGs int
}

// PreparePartitionedBuild does whatever serial work a partitioned build needs
// before its morsels can run — for a build side that is itself a join, that
// means materialising that join's own build side once — and returns the
// resulting per-morsel plan. It runs in the calling goroutine before any build
// worker is created.
//
// Returning (nil, nil) means "this build side turned out not to be
// partitionable"; the caller falls back to the serial build path.
type PreparePartitionedBuild func(ctx context.Context) (*PartitionedBuild, error)

// JoinBuildOption configures the build side of a ParallelHashJoinAggregate.
type JoinBuildOption func(*ParallelHashJoinAggregate)

// WithBuildRowsEstimate supplies the planner's estimate of the build side's row
// count, which selects the radix partition count (see RadixBitsFor). Without it
// the build side is left unpartitioned, which is the phase-1 behaviour.
func WithBuildRowsEstimate(rows int) JoinBuildOption {
	return func(p *ParallelHashJoinAggregate) { p.buildRowsEst = rows }
}

// WithPartitionableBuild supplies a row-group-morsel plan for the build side,
// enabling the parallel build. It is only used when the row estimate also clears
// the partitioning threshold.
func WithPartitionableBuild(prepare PreparePartitionedBuild) JoinBuildOption {
	return func(p *ParallelHashJoinAggregate) { p.prepareBuild = prepare }
}

// AboveJoinFactory wraps the per-worker join output with the operators that must
// run above the join inside each worker pipeline — a residual filter over joined
// rows, and/or the aggregate's pre-projection for expression aggregates. It must
// be deterministic: every worker's wrapped pipeline has to expose the same schema.
type AboveJoinFactory func(Operator) (Operator, error)

// ParallelHashJoinAggregate parallelises an inner hash join feeding a hash
// aggregate:
//
//	Aggregate → HashJoin(build = subtree, probe = (Filter →)? Scan)
//
// Execution has two phases:
//
//  1. Build. The build side is materialised once into a radix-partitioned
//     SharedHashTable. When the planner could split the build side into row-group
//     morsels (WithPartitionableBuild), this phase runs on numWorkers goroutines
//     with one goroutine owning each partition during assembly, so it needs no
//     lock; otherwise the build side is drained serially into the same
//     partitioned layout. Build sides below radixMinBuildRows are left in a
//     single partition, where partitioning would cost more than it saves.
//  2. Probe (parallel): numWorkers goroutines claim row-group morsels of the
//     probe scan from a shared atomic cursor — the same dynamic scheduling
//     ParallelHashAggregate uses — and each runs an independent
//     Scan → Filter? → HashJoinShared → PreProjection? → partial HashAggregate
//     pipeline. Each probe hashes its key and consults only that key's
//     partition. Partial aggregates are merged in the calling goroutine by
//     mergePartialAgg, so float64 SUM/MIN/MAX stay IEEE-correct.
//
// The parallel build shrinks the Amdahl term that bounded phase 1, where the
// whole build side ran serially. It does not eliminate it: build morsels are row
// groups, so a build side of only a few row groups leaves a straggler, and pass 2
// of the build plus the aggregate merge stay serial — both proportional to
// distinct keys rather than to rows. Partitioning does not speed up the probe;
// see the SharedHashTable comment for the measurement.
type ParallelHashJoinAggregate struct {
	buildFactory BuildFactory
	buildKeyIdx  int
	probeFactory PipelineFactory
	probeKeyIdx  int
	aboveJoin    AboveJoinFactory // may be nil

	// buildRowsEst and prepareBuild are set by JoinBuildOption; zero and nil
	// respectively reproduce the phase-1 single-map serial build.
	buildRowsEst int
	prepareBuild PreparePartitionedBuild

	totalRGs   int // probe-side row groups
	numWorkers int
	morselSize int

	groupBy  []int
	aggExprs []AggExpr
	schema   Schema

	delegate *HashAggregate // populated after setup()

	// buildStats records how the build side was materialised. Written once in
	// setup, before any probe worker starts; read only through BuildStats.
	buildStats JoinBuildStats

	// emptyGlobal holds the single output row that SQL requires from a global
	// aggregate (no GROUP BY) over zero input rows. See finishMerged.
	emptyGlobal        *Batch
	emptyGlobalEmitted bool
}

// JoinBuildStats describes how a ParallelHashJoinAggregate materialised its build
// side. It exists so tests can assert which build strategy a plan actually chose
// and so benchmarks can report build-phase time separately from probe time;
// execution never reads it.
type JoinBuildStats struct {
	Rows       int           // build rows inserted (NULL-keyed rows excluded)
	Keys       int           // distinct join keys
	Partitions int           // radix partitions; 1 means unpartitioned
	Parallel   bool          // true when the morsel-parallel builder ran
	Elapsed    time.Duration // wall time of the whole build phase
}

// BuildStats returns the build-phase statistics. It is only meaningful after the
// first Next call has returned, which is when the build phase has completed and
// published them.
func (p *ParallelHashJoinAggregate) BuildStats() JoinBuildStats { return p.buildStats }

// NewParallelHashJoinAggregate constructs a ParallelHashJoinAggregate.
// totalRGs is the probe scan's row group count; numWorkers is capped to it.
// morselSize of 0 selects defaultMorselSize. Pass WithBuildRowsEstimate and
// WithPartitionableBuild to enable radix partitioning and the parallel build.
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
	opts ...JoinBuildOption,
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
	p := &ParallelHashJoinAggregate{
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
	for _, opt := range opts {
		opt(p)
	}
	return p
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
	// ---- Phase 1: build, before any probe worker exists --------------------
	sht, err := p.materializeBuildSide(ctx)
	if err != nil {
		return fmt.Errorf("exec: parallel join: %w", err)
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

// materializeBuildSide chooses among the three build strategies and returns the
// completed table. The choice never affects results — only how long the build
// takes and how the finished table is laid out for probing:
//
//	radix bits | partitionable | strategy
//	-----------+---------------+-------------------------------------------------
//	0          | either        | serial drain, one partition (phase-1 behaviour)
//	>0         | no            | serial drain, 2^bits partitions
//	>0         | yes           | morsel-parallel build, 2^bits partitions
//
// A build side the planner could not split into row-group morsels still gets
// partitioned, because partitioning also makes map growth cheaper and leaves the
// probe with a single code shape to deal with.
func (p *ParallelHashJoinAggregate) materializeBuildSide(ctx context.Context) (*SharedHashTable, error) {
	started := time.Now()
	sht, parallel, err := p.runBuild(ctx)
	if err != nil {
		return nil, err
	}
	p.buildStats = JoinBuildStats{
		Rows:       sht.NumRows(),
		Keys:       sht.NumKeys(),
		Partitions: sht.NumPartitions(),
		Parallel:   parallel,
		Elapsed:    time.Since(started),
	}
	return sht, nil
}

// runBuild executes the selected build strategy and reports whether the
// morsel-parallel builder ran.
func (p *ParallelHashJoinAggregate) runBuild(ctx context.Context) (*SharedHashTable, bool, error) {
	bits := RadixBitsFor(p.buildRowsEst)

	if bits > 0 && p.prepareBuild != nil {
		pb, err := p.prepareBuild(ctx)
		if err != nil {
			return nil, false, fmt.Errorf("build side: %w", err)
		}
		if pb != nil && pb.TotalRGs > 0 {
			sht, err := BuildSharedHashTableParallel(ctx, pb.Factory, pb.Schema, p.buildKeyIdx,
				pb.TotalRGs, p.numWorkers, p.morselSize, bits)
			return sht, err == nil, err
		}
		// Not partitionable after all — fall through to the serial build.
	}

	buildOp, err := p.buildFactory(ctx)
	if err != nil {
		return nil, false, fmt.Errorf("build side: %w", err)
	}
	sht, buildErr := BuildSharedHashTableRadix(ctx, buildOp, p.buildKeyIdx, bits)
	_ = buildOp.Close()
	if buildErr != nil {
		return nil, false, buildErr
	}
	return sht, false, nil
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
