package exec

// This file carries one question from the operators that can answer it to the
// join's build side, which is the only place that needs it: how many rows is
// this subtree about to produce?
//
// The join materialises its build side into a rowStore (join_table.go). Told
// nothing, the store grows by doubling, which churns ~2x the final payload in
// allocation and holds old and new arrays at once across every copy — measured
// at 31.6% of every byte the join benchmarks allocate, the largest single site
// in the join path. Told the row count up front, it allocates once and copies
// nothing.
//
// A .vxq scan already knows its answer exactly: the footer records NumRows per
// row group, and a scan reads whole row groups. That is where every count used
// here comes from.

// RowCountBound is implemented by an operator that can bound how many rows it
// will produce before producing any of them.
//
// The bound is an upper bound, so a consumer that preallocates for it never
// under-allocates — but a bound is only worth preallocating for when it is
// close to the truth, which is what tight reports:
//
//   - tight: the operator expects to produce this many rows, give or take rows
//     a consumer drops for its own reasons. A scan over unpruned row groups is
//     tight: it yields every row of every row group it opens.
//   - not tight: the count bounds the output but says nothing about how close
//     it is. A Filter's output is between zero rows and its child's count, and
//     which one depends on data the planner has not looked at.
//
// Preallocating for a loose bound is a memory regression, not a saving: on the
// Q3-shaped benchmark the build side is a filtered nested join whose loose bound
// over-states its row count by roughly 5x, so presizing from it would allocate
// more than doubling ever does and hold all of it for the whole probe phase.
// buildRowsPresize therefore ignores a loose bound and lets the store grow, and
// the shapes that still grow are named in the join's commit message rather than
// silently regressing.
//
// An operator that cannot bound its output does not implement this interface at
// all, which is the same answer as a zero bound and needs no sentinel.
type RowCountBound interface {
	Operator

	// RowCountBound returns an upper bound on the rows this operator will
	// produce from its current position, and whether that bound is tight.
	//
	// It must be safe to call before the first Next and after a Reset, and it
	// must not consume rows or perform I/O.
	RowCountBound() (rows int, tight bool)
}

// RowCountBound reports the rows this scan will yield: the row counts of the row
// groups in [rgStart, rgEnd) that its zone-map predicate does not prune. Tight,
// because a scan yields every row of every row group it opens — column pruning
// drops columns, never rows, and there is no other row-dropping step inside a
// scan.
//
// Zone-map pruning is included rather than ignored because it is decided from
// the same footer metadata, by the same predicate, before any row is read: the
// count is what the scan will actually produce, not what the file holds. That
// matters most where it is largest — a build side whose predicate prunes most of
// its row groups would otherwise be presized for the whole table.
//
// Valid after Reset: rgStart and rgEnd are what Reset last set, so a reused
// morsel pipeline (see MorselPipeline) reports its current morsel.
func (s *TableScan) RowCountBound() (int, bool) {
	rgs := s.reader.Meta().RowGroups
	rows := 0
	// rgEnd is caller-supplied, and this function only reads footer metadata —
	// so it stops at the last row group rather than indexing past it. A range
	// past the end is outside TableScan's contract and Next panics on one; a
	// predictive helper has no business being the first place that shows up.
	for i := s.rgStart; i < s.rgEnd && i < len(rgs); i++ {
		rg := &rgs[i]
		if s.zonePred != nil && !s.zonePred(rg) {
			continue
		}
		rows += rg.NumRows
	}
	return rows, true
}

// RowCountBound passes the child's bound through, never tight: a filter's output
// is anything from zero rows to all of them, and nothing here knows which
// without evaluating the predicate over the data.
//
// The bound is still worth propagating — a caller may want the upper bound for
// something other than an allocation — but every consumer in this package
// declines to presize from it. See RowCountBound's comment for the measurement.
func (f *Filter) RowCountBound() (int, bool) {
	child, ok := f.child.(RowCountBound)
	if !ok {
		return 0, false
	}
	rows, _ := child.RowCountBound()
	return rows, false
}

// buildRowsPresize returns the row capacity a build-side rowStore should be
// preallocated for when it is about to materialise src, and 0 for "no idea,
// grow on demand".
//
// Only a tight bound is used, for the reason RowCountBound documents. The bound
// is an upper bound even when tight — the build side drops rows with a NULL join
// key (forEachBuildRow), so a store presized from it can end up holding fewer
// rows than it has capacity for. That direction is free: the store never grows,
// and the unused tail is capacity the allocator would have rounded up to anyway
// at any realistic null rate.
func buildRowsPresize(src Operator) int {
	b, ok := src.(RowCountBound)
	if !ok {
		return 0
	}
	rows, tight := b.RowCountBound()
	if !tight {
		return 0
	}
	return presizeRows(rows)
}
