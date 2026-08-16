package exec

import (
	"fmt"
	"math"
	"math/bits"
)

// This file holds the join's build side: a flat row store and a flat,
// open-addressed hash table over int64 join keys. Together they replace the
// map[int64][]buildRow the join used to probe, and the three per-row slices each
// build row used to own.
//
// What the old representation cost, per probe and per build row:
//
//   - Probing map[int64][]buildRow is hash → bucket → slice header → payload
//     pointer: three dependent loads before the first payload byte, each a
//     likely cache miss at Q3's build cardinality.
//   - Every distinct key allocated a []buildRow (~700K allocations at Q3's
//     filtered orders cardinality), and every build row allocated three slices
//     of its own (values, strVals, nulls).
//
// What replaces it:
//
//   - rowStore keeps every build row in a handful of flat, row-major arrays. A
//     row is addressed by an int32 index, so materialising N rows costs O(1)
//     allocations rather than 3N, and rows of the same table are contiguous.
//   - joinHashTable is an open-addressed table with linear probing and
//     power-of-two capacity. A slot is 16 bytes (key + head + tail), so four
//     slots share a cache line and a probe of a well-sized table touches one
//     line: the key comparison and the row index it yields arrive together.
//   - Duplicate keys chain through the store rather than through a per-key
//     slice: the slot holds the first and last row index for its key, and
//     rowStore.next links a key's rows in insertion order. No per-key
//     allocation, and per-key row order is exactly insertion order, which is
//     what keeps parallel integer aggregates bit-exact against serial (see
//     BuildSharedHashTableParallel's determinism note).

const (
	// noRow marks the absence of a row: an empty table slot, and the end of a
	// key's chain of duplicate rows. It is -1 rather than 0 so that a build key
	// of 0 — and a build row at index 0 — are ordinary values needing no special
	// case. Open addressing usually reserves a sentinel *key* for empty slots,
	// which would make key 0 unrepresentable; reserving a sentinel row index
	// instead costs nothing, because no row can be at index -1.
	noRow int32 = -1

	// maxBuildRows is the largest build side a rowStore can address, bounded by
	// the int32 row index. Reaching it is reported as an error rather than
	// silently wrapping; at any realistic row width the machine runs out of
	// memory long first (2^31 rows of a single int64 column is already 17 GB of
	// payload plus 8.6 GB of chain links).
	maxBuildRows = int64(math.MaxInt32)

	// joinTableMinSlots is the smallest slot array a table allocates. Empty
	// partitions of a radix-partitioned table are common — a skewed key set can
	// leave 63 of 64 empty — so the floor is kept small, but it must stay
	// non-zero: lookup relies on there always being an empty slot to terminate
	// on, and an empty table is all empty slots.
	joinTableMinSlots = 8

	// joinTableLoadNum / joinTableLoadDen express the maximum load factor, 0.7.
	//
	// Linear probing degrades sharply as a table fills: the expected number of
	// slots examined for a successful lookup is (1 + 1/(1-a))/2, which is 2.2
	// at a=0.7 but 4.5 at a=0.9 — and probes past the first slot are what turn
	// a one-cache-line probe into two. 0.7 is the conventional knee, and at 16
	// bytes per slot it costs ~23 bytes of table per key, well under what a Go
	// map spends on buckets, overflow pointers and tophash bytes for the same
	// key count.
	//
	// Integers rather than a float so capacity arithmetic is exact.
	joinTableLoadNum = 7
	joinTableLoadDen = 10

	// rowStoreMinRows is the row capacity a growing store jumps to on its first
	// append. It is deliberately small: the doubling below is what makes growth
	// cheap, not the floor, and an empty or near-empty store is common (a radix
	// build morsel whose rows all filtered out, a probe-only test fixture), so
	// paying a batch's worth of payload up front would be waste rather than
	// headroom.
	rowStoreMinRows = 8
)

// ---- Flat build-row store ---------------------------------------------------

// rowStore is the flat home for build rows. One store backs one join's build
// side: a serial HashJoin's own table, or every partition of a SharedHashTable
// (partitions share a store, so a row index is unique across the whole table and
// a probe needs only that index to reach the payload).
//
// Layout is row-major and struct-of-arrays:
//
//	values[row*numCols + col]      raw int64 bits for non-string columns
//	nulls [row*numCols + col]      per-column NULL flag
//	strs  [row*strWidth + slot]    materialised strings, only for string columns
//	next  [row]                    next row with the same join key, or noRow
//
// String columns get their own narrow window rather than a full numCols-wide
// string array, because a []string entry is a 16-byte header the GC must scan:
// Q12's pruned orders side is four columns of which one is a string, so the
// window is a quarter the width it would otherwise be.
//
// A store is append-only. A row is never renumbered or rewritten after it is
// appended, so a row index stays valid for the store's lifetime — that is what
// joinHashTable.insert, the next chain and the probe all rely on. Growth does
// reallocate and copy, so the *address* of a row is not stable while a store is
// still being appended to; nothing may hold a pointer into the arrays across a
// reserve. Once the build phase has published the store the arrays stop growing
// and may be read concurrently.
type rowStore struct {
	numCols  int
	strWidth int     // number of TypeString columns
	strSlot  []int32 // column index → index in the per-row string window, or -1

	values []int64
	strs   []string
	nulls  []bool
	// next links rows that share a join key, in insertion order. Written only by
	// joinHashTable.insert, which owns every row it links.
	next []int32

	n int32 // rows appended (or, for a sized store, rows reserved up front)

	// capRows is the row count the arrays are currently allocated for: every
	// array's capacity is at least capRows times that array's per-row width.
	// Tracked explicitly rather than read back from cap(), because the allocator
	// rounds a capacity up to a size class and it rounds each of the four arrays
	// up by a different amount — a single row capacity all four agree on is what
	// lets reserve reslice them in lockstep without a bounds check per array.
	//
	// Written only by the one goroutine that appends to a store. A sized store
	// (newSizedRowStore) sets it once at construction and never grows, so the
	// parallel build's assembling goroutines never touch it.
	capRows int
}

// newRowStore returns an empty store for rows of the given schema, growing on
// demand. capRows is a hint for the initial allocation; 0 means "no idea", and
// the store then doubles its way up from rowStoreMinRows. A caller that knows the
// row count should say so: the hint removes the growth entirely, and growth —
// even the geometric growth reserve now does — is the largest single allocator in
// the join path.
func newRowStore(schema Schema, capRows int) *rowStore {
	s := &rowStore{numCols: len(schema.Fields)}
	s.strSlot = make([]int32, s.numCols)
	for c, f := range schema.Fields {
		if f.Type == TypeString {
			s.strSlot[c] = int32(s.strWidth)
			s.strWidth++
		} else {
			s.strSlot[c] = -1
		}
	}
	if capRows > 0 {
		s.values = make([]int64, 0, capRows*s.numCols)
		s.nulls = make([]bool, 0, capRows*s.numCols)
		s.next = make([]int32, 0, capRows)
		if s.strWidth > 0 {
			s.strs = make([]string, 0, capRows*s.strWidth)
		}
		s.capRows = capRows
	}
	return s
}

// newSizedRowStore returns a store whose rows are all allocated up front and
// whose row indices are therefore [0, rows). Rows are written by setFrom rather
// than appended, which is what lets the parallel build fill disjoint index
// ranges from several goroutines without a shared append cursor.
func newSizedRowStore(schema Schema, rows int) *rowStore {
	s := newRowStore(schema, 0)
	s.values = make([]int64, rows*s.numCols)
	s.nulls = make([]bool, rows*s.numCols)
	s.next = make([]int32, rows)
	if s.strWidth > 0 {
		s.strs = make([]string, rows*s.strWidth)
	}
	s.n = int32(rows)
	s.capRows = rows
	return s
}

// rows returns the number of rows the store holds.
func (s *rowStore) rows() int { return int(s.n) }

// reserve adds one row and returns its index, growing the arrays as needed.
//
// The row index is stable for the store's lifetime: growth copies rows to a
// larger allocation, it never renumbers them, so an index handed out here stays
// valid for joinHashTable.insert, for the next chain, and for the probe.
//
// The elements the new row exposes are zero — which is what an all-NULL-free row
// looks like before writeRow fills it, and "" for strs. next is the exception that
// does not matter: its zero is 0 rather than noRow, but no reader can observe it,
// because joinHashTable.insert writes next[row] before that row is reachable from
// any chain.
//
// The zero holds because nothing ever writes past the arrays' length: every write
// is indexed by a row below s.n, so the region between length and capacity is
// untouched allocator-zeroed memory from the moment it is allocated until the
// reslices below expose it.
func (s *rowStore) reserve() (int32, error) {
	if int64(s.n) >= maxBuildRows {
		return noRow, fmt.Errorf("exec: hash join: build side exceeds %d rows", maxBuildRows)
	}
	row := s.n
	s.n++
	rows := int(s.n)
	if rows > s.capRows {
		s.reallocate(rows)
	}
	// Lengths track the row count exactly, so an index past the last appended row
	// panics rather than reading a row that was never appended.
	s.values = s.values[:rows*s.numCols]
	s.nulls = s.nulls[:rows*s.numCols]
	s.next = s.next[:rows]
	if s.strWidth > 0 {
		s.strs = s.strs[:rows*s.strWidth]
	}
	return row, nil
}

// reallocate moves the arrays to allocations big enough for at least rows rows,
// doubling the row capacity. Lengths are unchanged; only capacity grows.
//
// Doubling is the point of this function rather than a detail of it. Go's own
// append growth is what reserve used to rely on, and above ~256 elements
// append grows a slice by only ~1.25x (runtime.nextslicecap), so filling an
// array of N elements one row at a time churns roughly N * 1.25/0.25 = 5N
// elements of total allocation. Doubling churns 2N, and it is the whole win
// here: on the four join benchmarks reserve was 41% of every byte the join path
// allocated.
//
// The remaining 2N is growth that a caller-supplied row count would remove
// outright — see newRowStore's capRows hint.
func (s *rowStore) reallocate(rows int) {
	capRows := nextRowCap(s.capRows, rows)
	// Written out per array rather than through one shared helper: this package
	// is developed against `pprof -top`/`-list`, and a helper — generic most of
	// all, which pprof reports per instantiated shape — moves these bytes off
	// reallocate and onto a symbol that says nothing about which array grew.
	//
	// make zeroes the whole allocation and copy writes only the prefix, so
	// everything past the length is zero. That is the invariant reserve exposes.
	values := make([]int64, len(s.values), capRows*s.numCols)
	copy(values, s.values)
	s.values = values

	nulls := make([]bool, len(s.nulls), capRows*s.numCols)
	copy(nulls, s.nulls)
	s.nulls = nulls

	next := make([]int32, len(s.next), capRows)
	copy(next, s.next)
	s.next = next

	if s.strWidth > 0 {
		strs := make([]string, len(s.strs), capRows*s.strWidth)
		copy(strs, s.strs)
		s.strs = strs
	}
	s.capRows = capRows
}

// nextRowCap returns the row capacity to grow to when a store allocated for cur
// rows needs to hold need rows: double, but never below the floor, never short of
// need, and never past what an int32 row index can address.
func nextRowCap(cur, need int) int {
	capRows := cur * 2
	if capRows < need {
		capRows = need
	}
	if capRows < rowStoreMinRows {
		capRows = rowStoreMinRows
	}
	// reserve refuses to hand out a row at or past maxBuildRows, so there is
	// never a reason to allocate past it — doubling must not turn a store that
	// is about to report overflow into an allocation twice the size of the one
	// that is already the largest the row index can address.
	if int64(capRows) > maxBuildRows {
		capRows = int(maxBuildRows)
	}
	return capRows
}

// appendFrom materialises row rowIdx of batch into a new store row and returns
// its index. Values are copied out, so the row outlives the batch — TableScan
// reuses its decode buffers between batches.
//
// This is the single place a build row is materialised from a batch. Every
// build-side strategy goes through it, which is what guarantees the serial
// join, the serial radix build and the morsel-parallel build produce identical
// build rows from the same input.
func (s *rowStore) appendFrom(batch *Batch, rowIdx int) (int32, error) {
	row, err := s.reserve()
	if err != nil {
		return noRow, err
	}
	s.writeRow(row, batch, rowIdx)
	return row, nil
}

// writeRow copies row rowIdx of batch into the already-reserved store row row.
// Split out of appendFrom so that growth and payload can be exercised
// independently — see BenchmarkRowStoreGrowth and the growth-equivalence test.
func (s *rowStore) writeRow(row int32, batch *Batch, rowIdx int) {
	vOff := int(row) * s.numCols
	sOff := int(row) * s.strWidth
	for c := 0; c < s.numCols; c++ {
		v := batch.Vectors[c]
		if v.IsNull(rowIdx) {
			s.nulls[vOff+c] = true
			continue
		}
		if sv, ok := v.(*StringVector); ok {
			// A string column keeps its materialised value; a StringVector in a
			// column the schema does not call TypeString has nowhere to go, and
			// nothing would read it back (see buildColumnFromRows), so it is
			// dropped exactly as the previous representation dropped it.
			if slot := s.strSlot[c]; slot >= 0 && sv.Dict != nil {
				s.strs[sOff+int(slot)] = sv.Dict.Get(sv.Codes[rowIdx])
			}
			continue
		}
		s.values[vOff+c] = extractInt64(v, rowIdx)
	}
}

// setFrom copies row srcRow of src into this store's row dst, which must already
// exist (see newSizedRowStore). src must have been created from the same schema.
//
// Concurrency: two goroutines writing different dst rows touch disjoint elements
// of the destination arrays and need no synchronisation. That includes nulls,
// where neighbouring rows can share a machine word: a []bool element is written
// with a byte store, and distinct bytes are distinct memory locations under the
// Go memory model, so there is no read-modify-write to interleave. next is
// untouched here — the chain is built by joinHashTable.insert, which each
// partition's assembling goroutine calls only for rows it owns.
func (s *rowStore) setFrom(dst int32, src *rowStore, srcRow int32) {
	dOff := int(dst) * s.numCols
	sOff := int(srcRow) * src.numCols
	copy(s.values[dOff:dOff+s.numCols], src.values[sOff:sOff+src.numCols])
	copy(s.nulls[dOff:dOff+s.numCols], src.nulls[sOff:sOff+src.numCols])
	if s.strWidth > 0 {
		dOff = int(dst) * s.strWidth
		sOff = int(srcRow) * src.strWidth
		copy(s.strs[dOff:dOff+s.strWidth], src.strs[sOff:sOff+src.strWidth])
	}
}

// isNull reports whether column col of row is NULL.
func (s *rowStore) isNull(row int32, col int) bool { return s.nulls[int(row)*s.numCols+col] }

// value returns the raw int64 bits stored for column col of row. Float64 values
// are held as their IEEE-754 bits and dates as their int32 day count, exactly as
// the vectors store them.
func (s *rowStore) value(row int32, col int) int64 { return s.values[int(row)*s.numCols+col] }

// str returns the materialised string for column col of row, or "" for a column
// that holds no string.
func (s *rowStore) str(row int32, col int) string {
	slot := s.strSlot[col]
	if slot < 0 {
		return ""
	}
	return s.strs[int(row)*s.strWidth+int(slot)]
}

// ---- Flat open-addressed table ----------------------------------------------

// joinSlot is one table slot: the key, and the first and last row index of that
// key's chain. head is noRow in an empty slot, which is why key 0 needs no
// special case. tail exists so appending the second and later rows of a
// duplicate key is O(1) and preserves insertion order without walking the chain.
//
// 16 bytes, so four slots share a 64-byte cache line.
type joinSlot struct {
	key  int64
	head int32
	tail int32
}

// joinHashTable maps an int64 join key to the rows of one rowStore that carry
// it. Open addressing with linear probing over a power-of-two slot array; keys
// are distributed by radixHash, the murmur3 finalizer the radix partitioning
// already uses, so a sparse key space (TPC-H order keys) spreads evenly instead
// of clustering.
//
// The slot index comes from the *high* bits of the hash while radixPart takes the
// *low* bits, and that split is load-bearing rather than stylistic: every key in
// a radix partition has, by construction, the same low log2(numParts) hash bits.
// Masking the low bits for the slot index too would leave only
// log2(capacity) - log2(numParts) bits varying inside a partition, so a
// 64-partition table would pile every key onto one slot in 64 and turn linear
// probing into a linear scan. Taking the high bits keeps the two indexes
// independent.
//
// Immutable once its build phase finishes: lookup only reads, so any number of
// probe goroutines may share a table provided the build happens-before them (see
// SharedHashTable's concurrency contract).
type joinHashTable struct {
	slots  []joinSlot
	mask   uint64 // capacity-1, for probe wrap-around
	shift  uint   // 64 - log2(capacity), for the initial slot index
	keys   int    // distinct keys
	numRow int    // rows inserted, including duplicates
	growAt int    // key count at which the slot array doubles

	// store holds the payloads. Shared with sibling partitions of the same
	// SharedHashTable, so a row index means the same thing in every partition.
	store *rowStore
}

// newJoinHashTable returns a table over store, sized so that expectedKeys
// distinct keys fit without rehashing. Passing the partition's *row* count is
// the intended use where the key count is unknown: it over-estimates whenever
// keys repeat, which costs slot memory but guarantees no rehash — the same
// trade-off the map-based build made when it presized from row counts.
func newJoinHashTable(store *rowStore, expectedKeys int) *joinHashTable {
	t := &joinHashTable{store: store}
	t.setCapacity(joinTableCapacity(expectedKeys))
	return t
}

// joinTableCapacity returns the power-of-two slot count that holds keys entries
// at or below the load factor.
func joinTableCapacity(keys int) int {
	slots := joinTableMinSlots
	if keys <= 0 {
		return slots
	}
	// need is the smallest slot count whose load-factor limit is >= keys.
	need := (keys*joinTableLoadDen + joinTableLoadNum - 1) / joinTableLoadNum
	for slots < need {
		next := slots * 2
		if next <= slots { // int overflow; nothing larger is representable
			return slots
		}
		slots = next
	}
	return slots
}

// setCapacity allocates an empty slot array of exactly capacity slots, which must
// be a power of two of at least joinTableMinSlots.
func (t *joinHashTable) setCapacity(capacity int) {
	t.slots = make([]joinSlot, capacity)
	for i := range t.slots {
		t.slots[i].head = noRow
	}
	t.mask = uint64(capacity - 1)
	t.shift = uint(64 - bits.TrailingZeros(uint(capacity)))
	t.growAt = capacity * joinTableLoadNum / joinTableLoadDen
}

// slotFor returns the slot a key hashes to: the top log2(capacity) bits of the
// mixed hash (see the type comment for why the top and not the bottom).
func (t *joinHashTable) slotFor(key int64) uint64 { return radixHash(key) >> t.shift }

// insert links row into key's chain, appending it after any row already stored
// for that key so per-key order is insertion order.
func (t *joinHashTable) insert(key int64, row int32) {
	if t.keys >= t.growAt {
		t.grow()
	}
	t.store.next[row] = noRow
	t.numRow++
	i := t.slotFor(key)
	for {
		s := &t.slots[i]
		switch {
		case s.head == noRow:
			s.key, s.head, s.tail = key, row, row
			t.keys++
			return
		case s.key == key:
			t.store.next[s.tail] = row
			s.tail = row
			return
		}
		i = (i + 1) & t.mask
	}
}

// lookup returns the first row index carrying key, or noRow when the key is
// absent. Walk the rest of the key's rows through store.next.
func (t *joinHashTable) lookup(key int64) int32 { return t.lookupHashed(key, radixHash(key)) }

// lookupHashed is lookup for a caller that has already mixed the key — the probe
// loop needs the same hash to pick the partition, and mixing once instead of
// twice takes a handful of ALU ops off every probed row.
//
// The scan terminates on the first empty slot, which always exists: the load
// factor keeps growAt strictly below capacity, so a full table is impossible.
func (t *joinHashTable) lookupHashed(key int64, h uint64) int32 {
	i := h >> t.shift
	for {
		s := &t.slots[i]
		if s.head == noRow {
			return noRow
		}
		if s.key == key {
			return s.head
		}
		i = (i + 1) & t.mask
	}
}

// grow doubles the slot array and reinserts the existing keys. Only slots move —
// rows and chains are untouched, because a slot carries its whole chain in head
// and tail.
func (t *joinHashTable) grow() {
	old := t.slots
	t.setCapacity(len(old) * 2)
	for _, s := range old {
		if s.head == noRow {
			continue
		}
		i := t.slotFor(s.key)
		for t.slots[i].head != noRow {
			i = (i + 1) & t.mask
		}
		t.slots[i] = s
	}
}

// forEachKey calls fn once per distinct key with the head of its chain, in
// unspecified order. Used by tests and by table statistics, never on a hot path.
func (t *joinHashTable) forEachKey(fn func(key int64, head int32)) {
	for _, s := range t.slots {
		if s.head != noRow {
			fn(s.key, s.head)
		}
	}
}
