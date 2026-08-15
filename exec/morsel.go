package exec

import "context"

// MorselPipeline is a morsel pipeline that can be repositioned onto a new
// row-group range instead of being torn down and rebuilt.
//
// Why it exists: a worker in the morsel scheduler used to call its
// PipelineFactory once per claimed morsel, and every planner factory opens the
// .vxq file — so one query over N row groups opened N Readers and read and
// parsed the footer N times. At defaultMorselSize=1 over a 92-row-group scan
// that made storage.Open the second-largest allocation site in a parallel
// aggregate (~7.7 MB/op, ~20% of everything the operator allocated), and cost
// one wasted pread per morsel. A worker that builds its pipeline once and
// repositions it pays that once per worker instead of once per morsel.
//
// Reset contract. Reset(rgStart, rgEnd) must leave the pipeline in the state a
// freshly built one covering [rgStart, rgEnd) would be in, with one deliberate
// exception: a buffer whose contents are fully rewritten before anything reads
// them may be retained, because retaining those buffers is the point —
// TableScan's decode buffers, Filter's selection vector, an Expr's scratch (see
// scratch.go). State that survives a Reset and *is* read before being written
// would leak one morsel's rows into the next, so each Reset below is written
// against that rule and morsel_reuse_test.go checks every reused pipeline
// against a rebuild-per-morsel oracle.
//
// The pipeline retains no context: an Operator takes ctx per Next call, and the
// factories' only ctx use is the storage.Open that reuse removes. A reused
// pipeline is therefore not pinned to the context of the morsel that built it.
//
// Reset must only be called on a pipeline reusableMorselPipeline accepted.
type MorselPipeline interface {
	Operator
	Reset(rgStart, rgEnd int)
}

// reusableMorselPipeline returns op as a MorselPipeline when every operator in
// it can be repositioned, and nil otherwise.
//
// The switch is deliberately closed. An operator not named here makes the whole
// pipeline non-reusable, so a future operator appearing inside a morsel pipeline
// degrades to rebuilding per morsel rather than silently skipping its own reset
// — which, for anything holding per-morsel rows, would mean wrong results
// instead of lost performance. The check runs once per worker before the first
// Reset, so a pipeline is either reused for every morsel or rebuilt for every
// morsel, never half-repositioned.
func reusableMorselPipeline(op Operator) MorselPipeline {
	switch o := op.(type) {
	case *TableScan:
		return o
	case *Filter:
		if reusableMorselPipeline(o.child) == nil {
			return nil
		}
		return o
	case *Project:
		if reusableMorselPipeline(o.child) == nil {
			return nil
		}
		return o
	case *HashJoin:
		// Only a join probing a SharedHashTable. That table is materialised once
		// by the operator above and is read-only, so repositioning the probe
		// leaves it untouched. A self-building join owns a table drained from its
		// own build child — repositioning would either rebuild it per morsel or
		// keep a table the reset was supposed to invalidate — and it never
		// appears in a morsel pipeline, so it is refused here.
		if o.build != nil || reusableMorselPipeline(o.probe) == nil {
			return nil
		}
		return o
	default:
		return nil
	}
}

// morselRunner supplies one worker goroutine with the pipeline for each morsel
// it claims: it retains and repositions a single pipeline when the factory
// produces a resettable one, and rebuilds per morsel when it does not.
//
// A runner belongs to exactly one goroutine, like the pipeline it holds — which
// is what keeps the single-goroutine scratch contract in scratch.go intact:
// reuse narrows an Expr tree's exposure from one goroutine per morsel to one
// goroutine for the worker's whole run, it never widens it.
//
// Usage inside a worker:
//
//	runner := morselRunner{factory: f}
//	defer runner.close()
//	for {
//		start, stop, ok := q.claim(msize)
//		if !ok {
//			break
//		}
//		op, err := runner.open(ctx, int(start), int(stop))
//		...       // drain op
//		runner.release(op)
//	}
//
// release is a no-op while a pipeline is retained and close releases it, so
// every exit path — including an error return part-way through a morsel — closes
// the pipeline exactly once.
type morselRunner struct {
	factory PipelineFactory

	held   Operator       // retained across morsels; nil when rebuilding per morsel
	reuse  MorselPipeline // held, typed for Reset
	probed bool           // reusability already tested
}

// open returns the pipeline covering row groups [rgStart, rgEnd).
func (m *morselRunner) open(ctx context.Context, rgStart, rgEnd int) (Operator, error) {
	if m.reuse != nil {
		m.reuse.Reset(rgStart, rgEnd)
		return m.held, nil
	}
	op, err := m.factory(ctx, rgStart, rgEnd)
	if err != nil {
		return nil, err
	}
	if !m.probed {
		m.probed = true
		if rp := reusableMorselPipeline(op); rp != nil {
			m.held, m.reuse = op, rp
		}
	}
	return op, nil
}

// release ends a morsel, closing the pipeline unless it is being retained for
// the next one.
func (m *morselRunner) release(op Operator) {
	if m.held == nil && op != nil {
		_ = op.Close()
	}
}

// close releases the retained pipeline, if any. Safe to call more than once, so
// a worker can defer it and still release on every path.
func (m *morselRunner) close() {
	if m.held != nil {
		_ = m.held.Close()
		m.held, m.reuse = nil, nil
	}
}
