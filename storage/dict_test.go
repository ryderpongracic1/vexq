package storage

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

// dictFrom round-trips values through DictBuilder.Marshal + UnmarshalDictionary,
// which is how every production Dictionary comes into being. values must be
// distinct: DictBuilder collapses duplicates, so a caller that passes the same
// value twice would silently get a shorter dictionary than it wrote.
func dictFrom(t testing.TB, values ...string) *Dictionary {
	t.Helper()
	db := NewDictBuilder()
	for _, v := range values {
		db.Add(v)
	}
	if db.Len() != len(values) {
		t.Fatalf("dictFrom: %d values collapsed to %d; values must be distinct", len(values), db.Len())
	}
	d, err := UnmarshalDictionary(db.Marshal())
	if err != nil {
		t.Fatalf("UnmarshalDictionary: %v", err)
	}
	return d
}

// TestDictionaryGetValues covers the cases where the memo could plausibly return
// something different from a fresh copy out of Data: the empty string (which is
// the memo's own miss sentinel), the last code (whose end bound is len(Data)
// rather than the next offset), a single-entry dictionary, and repeated lookups
// of every code in both ascending and descending order.
func TestDictionaryGetValues(t *testing.T) {
	cases := []struct {
		name   string
		values []string
	}{
		{"single entry", []string{"only"}},
		{"single empty entry", []string{""}},
		{"empty first", []string{"", "b", "c"}},
		{"empty last", []string{"a", "b", ""}},
		{"empty middle", []string{"a", "", "c"}},
		{"tpch returnflag", []string{"A", "N", "R"}},
		{"tpch shipmode", []string{"AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"}},
		{"utf8 and nul bytes", []string{"héllo", "\x00\x01", "日本語", ""}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := dictFrom(t, tc.values...)
			if d.Len() != len(tc.values) {
				t.Fatalf("Len() = %d, want %d", d.Len(), len(tc.values))
			}
			// Ascending, then descending, then ascending again: the second and
			// third passes are memo hits, and the descending pass makes sure the
			// last code is not special-cased into the wrong bound once cached.
			for pass, order := range [][]int{ascending(len(tc.values)), descending(len(tc.values)), ascending(len(tc.values))} {
				for _, i := range order {
					if got := d.Get(uint32(i)); got != tc.values[i] {
						t.Fatalf("pass %d: Get(%d) = %q, want %q", pass, i, got, tc.values[i])
					}
				}
			}
			// Lookup walks every code through Get, so it exercises the memo from
			// the other direction and must still find every value.
			for want, v := range tc.values {
				code, ok := d.Lookup(v)
				if !ok {
					t.Fatalf("Lookup(%q) not found", v)
				}
				if int(code) != want {
					t.Fatalf("Lookup(%q) = code %d, want %d", v, code, want)
				}
			}
			if _, ok := d.Lookup("definitely-absent"); ok {
				t.Fatal("Lookup found an absent value")
			}
		})
	}
}

func ascending(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func descending(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = n - 1 - i
	}
	return out
}

// getUncached is Get exactly as it read before memoisation: one copy out of Data
// per call. It exists so the correctness test can compare the memo against the
// definition it replaced, and so the microbenchmark can measure the two side by
// side in one binary rather than across two builds.
func getUncached(d *Dictionary, code uint32) string {
	start := d.Offsets[code]
	end := uint32(len(d.Data))
	if int(code)+1 < len(d.Offsets) {
		end = d.Offsets[code+1]
	}
	return string(d.Data[start:end])
}

// TestDictionaryGetMatchesUncachedCopy pins Get against the pre-memo definition
// directly: for every code, the memoised result must equal a fresh copy taken
// straight out of Offsets and Data.
func TestDictionaryGetMatchesUncachedCopy(t *testing.T) {
	// Hand-built rather than round-tripped so it can hold duplicate and empty
	// entries, which DictBuilder would collapse: Data "abbcccc" with entries
	// "", "a", "bb", "", "cccc", "", "".
	d := &Dictionary{
		Offsets: []uint32{0, 0, 1, 3, 3, 7, 7},
		Data:    []byte("abbcccc"),
	}

	for code := 0; code < d.Len(); code++ {
		want := getUncached(d, uint32(code))
		for rep := 0; rep < 3; rep++ {
			if got := d.Get(uint32(code)); got != want {
				t.Fatalf("Get(%d) rep %d = %q, want %q", code, rep, got, want)
			}
		}
	}
}

// TestDictionaryGetOutOfRangePanics keeps the out-of-range contract: the memo
// must not turn a bad code into a zero value, and the bounds check must run
// before the memo is even allocated.
func TestDictionaryGetOutOfRangePanics(t *testing.T) {
	for _, tc := range []struct {
		name string
		dict *Dictionary
		code uint32
	}{
		{"one past the end", dictFrom(t, "a", "b"), 2},
		{"far past the end", dictFrom(t, "a", "b"), 1 << 20},
		{"empty dictionary", dictFrom(t), 0},
		{"cold dictionary", dictFrom(t, "a"), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected a panic")
				}
				msg, ok := r.(string)
				if !ok {
					t.Fatalf("panic value %#v is not a string", r)
				}
				if !strings.Contains(msg, "out of range") {
					t.Fatalf("panic %q does not mention the range", msg)
				}
			}()
			_ = tc.dict.Get(tc.code)
		})
	}
}

// TestDictionaryGetOnStructLiteral covers a Dictionary built by hand rather than
// by UnmarshalDictionary — exec tests do this — so the memo must size itself
// without a constructor having run. It also covers a duplicate value, which
// DictBuilder would have collapsed but a hand-built dictionary can hold.
func TestDictionaryGetOnStructLiteral(t *testing.T) {
	// Data "xyzw": code 0 = [0,1), code 1 = [1,1) empty, code 2 = [1,3),
	// code 3 = [3,len) — the last code's end bound comes from len(Data).
	d := &Dictionary{Offsets: []uint32{0, 1, 1, 3}, Data: []byte("xyzw")}
	want := []string{"x", "", "yz", "w"}
	for rep := 0; rep < 2; rep++ {
		for code, w := range want {
			if got := d.Get(uint32(code)); got != w {
				t.Fatalf("rep %d Get(%d) = %q, want %q", rep, code, got, w)
			}
		}
	}
}

// TestDictionaryGetDoesNotMutateContent pins the invariant the whole read side
// leans on: Get memoises, but it must never touch the parsed content. A future
// zero-copy or in-place rewrite of Get would fail here rather than silently
// change what an already-returned string says.
func TestDictionaryGetDoesNotMutateContent(t *testing.T) {
	values := []string{"AIR", "", "REG AIR", "TRUCK"}
	d := dictFrom(t, values...)

	offsetsBefore := append([]uint32(nil), d.Offsets...)
	dataBefore := append([]byte(nil), d.Data...)
	dataPtrBefore := &d.Data[0]

	// Hold on to the first round's strings so a later mutation of Data would be
	// observable through them, not just through Data itself.
	held := make([]string, d.Len())
	for code := range held {
		held[code] = d.Get(uint32(code))
	}
	for rep := 0; rep < 3; rep++ {
		for code := range held {
			if got := d.Get(uint32(code)); got != held[code] {
				t.Fatalf("Get(%d) changed across calls: %q then %q", code, held[code], got)
			}
		}
	}

	if string(d.Data) != string(dataBefore) {
		t.Fatalf("Get mutated Data: %q, want %q", d.Data, dataBefore)
	}
	if &d.Data[0] != dataPtrBefore {
		t.Fatal("Get reallocated Data")
	}
	for i := range offsetsBefore {
		if d.Offsets[i] != offsetsBefore[i] {
			t.Fatalf("Get mutated Offsets[%d]: %d, want %d", i, d.Offsets[i], offsetsBefore[i])
		}
	}
	for code, want := range values {
		if held[code] != want {
			t.Fatalf("held string %d = %q, want %q", code, held[code], want)
		}
	}
}

// TestDictionaryPerReaderIsDistinct pins the structural fact the ownership
// contract rests on: two Readers over one file hand out different *Dictionary
// values for the same (row group, column), so two parallel workers can never be
// memoising into the same one. If a future change ever shares Dictionaries across
// Readers — a dictionary cache, say — this test fails and the memo has to become
// eager or atomic before that change can land.
func TestDictionaryPerReaderIsDistinct(t *testing.T) {
	schema := makeSchema(Field{Name: "s", Type: TypeString, Nullable: false})
	w, path := newTestWriter(t, schema)
	const rows = 300
	strs := make([]string, rows)
	for i := range strs {
		strs[i] = fmt.Sprintf("v%d", i%3)
	}
	if err := w.BeginRowGroup(rows); err != nil {
		t.Fatalf("BeginRowGroup: %v", err)
	}
	if err := w.AppendColumn(ctx, 0, FullBitmap(rows), strs); err != nil {
		t.Fatalf("AppendColumn: %v", err)
	}
	if err := w.EndRowGroup(); err != nil {
		t.Fatalf("EndRowGroup: %v", err)
	}
	finishWriter(t, w)

	dictOf := func() (*Dictionary, func()) {
		r, err := Open(ctx, path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		cr, err := r.OpenColumn(ctx, 0, 0)
		if err != nil {
			t.Fatalf("OpenColumn: %v", err)
		}
		d, err := cr.Dictionary()
		if err != nil {
			t.Fatalf("Dictionary: %v", err)
		}
		return d, func() {
			_ = cr.Close()
			_ = r.Close()
		}
	}

	d1, close1 := dictOf()
	defer close1()
	d2, close2 := dictOf()
	defer close2()

	if d1 == d2 {
		t.Fatal("two Readers returned the same *Dictionary; the memo would be shared across workers")
	}
	// Same ColumnReader, though, must keep returning the one it cached — that is
	// what makes the memo pay for itself over a row group.
	r := openReader(t, path)
	cr, err := r.OpenColumn(ctx, 0, 0)
	if err != nil {
		t.Fatalf("OpenColumn: %v", err)
	}
	defer func() { _ = cr.Close() }()
	a, _ := cr.Dictionary()
	b, _ := cr.Dictionary()
	if a != b {
		t.Fatal("one ColumnReader returned two *Dictionary values; the memo would not survive a row group")
	}
	if got := a.Get(0); got != "v0" {
		t.Fatalf("Get(0) = %q, want %q", got, "v0")
	}
}

// TestDictionaryGetPerGoroutineIsRaceFree exercises the pattern the engine
// actually uses — one Dictionary per goroutine, parsed from the same bytes — so
// the race detector has something to say about the memo in its intended use. It
// deliberately does NOT share one Dictionary across goroutines: that is
// unsupported, and asserting it were safe would be asserting the opposite of the
// contract.
func TestDictionaryGetPerGoroutineIsRaceFree(t *testing.T) {
	values := []string{"A", "", "N", "R"}
	db := NewDictBuilder()
	for _, v := range values {
		db.Add(v)
	}
	blob := db.Marshal()

	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d, err := UnmarshalDictionary(blob)
			if err != nil {
				t.Errorf("UnmarshalDictionary: %v", err)
				return
			}
			for rep := 0; rep < 200; rep++ {
				for code, want := range values {
					if got := d.Get(uint32(code)); got != want {
						t.Errorf("Get(%d) = %q, want %q", code, got, want)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// TestDictionaryGetAllocations enforces the two allocation claims the memo rests
// on, rather than leaving them to be eyeballed in benchmark output:
//
//   - A memoised lookup allocates nothing.
//   - A lookup of an EMPTY entry allocates nothing either, even though "" is the
//     memo's not-yet-populated sentinel and therefore never hits. That is what
//     makes the sentinel free and a parallel "is populated" bitmap unnecessary.
func TestDictionaryGetAllocations(t *testing.T) {
	d := dictFrom(t, "", "REG AIR", "TRUCK")
	// Warm every code so the memo, and its one-off backing slice, are in place.
	for code := 0; code < d.Len(); code++ {
		sink = d.Get(uint32(code))
	}

	for _, tc := range []struct {
		name string
		code uint32
	}{
		{"memoised multi-byte value", 1},
		{"empty value, a permanent memo miss", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := testing.AllocsPerRun(1000, func() { sink = d.Get(tc.code) })
			if got != 0 {
				t.Fatalf("Get(%d) allocates %.1f times per call, want 0", tc.code, got)
			}
		})
	}
}

// ---- Microbenchmarks -------------------------------------------------------

// benchDict builds a dictionary of n entries with realistic value widths.
func benchDict(b testing.TB, n int) *Dictionary {
	values := make([]string, n)
	for i := range values {
		values[i] = fmt.Sprintf("value-%06d", i)
	}
	return dictFrom(b, values...)
}

// BenchmarkDictionaryGet isolates Get so the engine-level win is attributable
// rather than inferred.
//
//   - hot: repeated lookups of one already-memoised code — the join probe's
//     shape, where a low-cardinality column repeats across a row group.
//   - cold: a freshly parsed dictionary per lookup, so every lookup misses. This
//     is the memo's worst case and must not be slower than the copy it replaced
//     by more than the one-off memo allocation.
//   - cardN: a full sweep over N codes, repeated — one miss then hits, which is
//     what a real row group does. Low cardinality is the TPC-H case (3 return
//     flags, 7 ship modes); high cardinality is the pathological column where the
//     memo buys nothing but a header slice.
func BenchmarkDictionaryGet(b *testing.B) {
	// Single-byte values were ALREADY allocation-free before the memo: Go's
	// []byte→string conversion returns a pointer into runtime.staticbytes for a
	// one-byte result. TPC-H's l_returnflag and l_linestatus are single-char, so
	// Q1's group-by columns never paid the per-lookup allocation and Q1 is not
	// where this change shows up. Kept as a benchmark so that stays visible.
	b.Run("hot/singleByteValue", func(b *testing.B) {
		d := dictFrom(b, "A", "N", "R")
		_ = d.Get(1)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = d.Get(1)
		}
	})

	b.Run("hot/singleByteValue-uncached", func(b *testing.B) {
		d := dictFrom(b, "A", "N", "R")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = getUncached(d, 1)
		}
	})

	// Multi-byte values are the case that allocated per lookup — l_shipmode,
	// o_orderpriority, and every string column the join materialises.
	b.Run("hot/multiByteValue", func(b *testing.B) {
		d := dictFrom(b, "AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK")
		_ = d.Get(4)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = d.Get(4)
		}
	})

	b.Run("hot/multiByteValue-uncached", func(b *testing.B) {
		d := dictFrom(b, "AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = getUncached(d, 4)
		}
	})

	b.Run("cold/firstTouchPerDict", func(b *testing.B) {
		db := NewDictBuilder()
		for _, v := range []string{"A", "N", "R"} {
			db.Add(v)
		}
		blob := db.Marshal()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			d, err := UnmarshalDictionary(blob)
			if err != nil {
				b.Fatal(err)
			}
			sink = d.Get(uint32(i % 3))
		}
	})

	for _, n := range []int{3, 7, 1024, 65536} {
		b.Run(fmt.Sprintf("sweep/cardinality%d", n), func(b *testing.B) {
			d := benchDict(b, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = d.Get(uint32(i % n))
			}
		})
		b.Run(fmt.Sprintf("sweep/cardinality%d-uncached", n), func(b *testing.B) {
			d := benchDict(b, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = getUncached(d, uint32(i%n))
			}
		})
	}

	// The empty string is the memo's miss sentinel, so it never hits. It must
	// still be allocation-free, which is why the sentinel costs nothing.
	b.Run("hot/emptyValue", func(b *testing.B) {
		d := dictFrom(b, "", "x")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = d.Get(0)
		}
	})
}

// BenchmarkDictionaryLookup covers the linear scan, which called Get once per
// candidate and therefore allocated per candidate per lookup before the memo.
func BenchmarkDictionaryLookup(b *testing.B) {
	d := dictFrom(b, "AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		code, ok := d.Lookup("TRUCK")
		if !ok || code != 6 {
			b.Fatalf("Lookup = %d, %v", code, ok)
		}
	}
}

// sink keeps the compiler from eliminating the benchmarked call.
var sink string
