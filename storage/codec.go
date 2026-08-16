package storage

import (
	"fmt"

	enc "github.com/ryderpongracic1/vexq/internal/encoding"
)

// ---- Null bitmap helpers ---------------------------------------------------

// SetNullBit marks row i as null in bitmap (bit = 0 means null).
// The bitmap convention is 1 = valid, 0 = null, LSB-first.
func SetNullBit(bitmap []byte, i int) {
	bitmap[i/8] &^= 1 << uint(i%8)
}

// SetValidBit marks row i as valid in bitmap.
func SetValidBit(bitmap []byte, i int) {
	bitmap[i/8] |= 1 << uint(i%8)
}

// IsNullBit reports whether row i is null (bit == 0).
func IsNullBit(bitmap []byte, i int) bool {
	return bitmap[i/8]>>uint(i%8)&1 == 0
}

// FullBitmap returns a bitmap of size bytes with all bits set (all valid).
func FullBitmap(rows int) []byte {
	size := (rows + 7) / 8
	b := make([]byte, size)
	for i := range b {
		b[i] = 0xFF
	}
	// Zero out the trailing bits beyond 'rows' to avoid confusion.
	if rows%8 != 0 {
		b[size-1] = (1 << uint(rows%8)) - 1
	}
	return b
}

// ---- Dictionary builder ----------------------------------------------------

// DictBuilder builds a per-row-group dictionary for a STRING column.
// Codes are assigned in insertion order (first occurrence = code 0).
type DictBuilder struct {
	index map[string]uint32
	keys  []string
}

func NewDictBuilder() *DictBuilder {
	return &DictBuilder{index: make(map[string]uint32)}
}

// Add returns the dictionary code for s, inserting it if new.
func (db *DictBuilder) Add(s string) uint32 {
	if code, ok := db.index[s]; ok {
		return code
	}
	code := uint32(len(db.keys))
	db.index[s] = code
	db.keys = append(db.keys, s)
	return code
}

// Lookup returns (code, true) if s is in the dictionary.
func (db *DictBuilder) Lookup(s string) (uint32, bool) {
	code, ok := db.index[s]
	return code, ok
}

// Len returns the number of distinct entries.
func (db *DictBuilder) Len() int { return len(db.keys) }

// GetByCode returns the string for a given code. Panics if code is out of range.
func (db *DictBuilder) GetByCode(code uint32) string {
	return db.keys[code]
}

// Marshal serialises the dictionary:
//
//	[4B num_entries][num_entries × 4B offset][4B total_len][data bytes][4B CRC32]
//
// Offsets are exclusive-prefix sums into data: entry[i] = data[offset[i]:offset[i+1]].
func (db *DictBuilder) Marshal() []byte {
	n := uint32(len(db.keys))
	// Build concatenated string data and offsets.
	offsets := make([]uint32, n)
	var data []byte
	for i, s := range db.keys {
		offsets[i] = uint32(len(data))
		data = append(data, s...)
	}
	totalLen := uint32(len(data))

	var b []byte
	b = enc.AppendUint32(b, n)
	for _, o := range offsets {
		b = enc.AppendUint32(b, o)
	}
	b = enc.AppendUint32(b, totalLen)
	b = append(b, data...)
	b = enc.AppendChecksum(b)
	return b
}

// ---- Dictionary (reader side) ----------------------------------------------

// Dictionary is the read-side representation of a per-row-group string dictionary.
//
// Content immutability: Offsets and Data are written once, by the constructor,
// and never again. Data is always a private allocation — UnmarshalDictionary
// copies out of the buffer it parses — so it is never a view into a Reader's
// pooled section window and never changes under a string that Get returned.
//
// Single-goroutine ownership: Get memoises into cache, so Get is a mutating
// method and a *Dictionary must not be read by two goroutines at once. Every
// production path satisfies this because a Dictionary is reachable only from the
// thing that made it:
//
//   - Read path: ColumnReader.Dictionary() parses one per (row group, column)
//     behind a sync.Once and caches it on that ColumnReader. ColumnReaders belong
//     to one Reader, a Reader is single-goroutine by contract, and each parallel
//     worker opens its own Reader — so two workers never hold the same
//     *Dictionary even when they scan the same file.
//   - Output path: newStringVector marshals a DictBuilder and parses the result,
//     producing a fresh Dictionary per output batch in the goroutine that built
//     it.
//   - Cross-goroutine handoffs materialise strings before they hand anything off:
//     the join's build side copies Get's results into rowStore.strs, and the
//     parallel aggregate flattens its integer-key state to string keys, both
//     inside the owning worker and before the channel send. No *Dictionary and no
//     Batch is ever sent to another goroutine, so a Dictionary is never live in
//     two goroutines at once.
//
// A future operator that hands batches across a goroutine boundary would break
// this — but it would first break TableScan's rule that a Batch is only valid
// until the next Next() call, which is the louder of the two.
type Dictionary struct {
	Offsets []uint32
	Data    []byte

	// cache memoises Get per code. Sized len(Offsets) on first use; see Get.
	cache []string
}

// Get returns the string for code. Panics if code is out of range.
//
// The result is memoised per code. A dictionary-encoded column repeats a small
// number of distinct values across a row group's 65,536 rows — three return
// flags, seven ship modes — so the first lookup of a code copies out of Data and
// every later lookup of it is a slice index. Before memoisation this copied per
// lookup, which at 13.4M lookups was 88% of every object the join path
// allocated.
//
// The cost is one []string of len(Offsets) per Dictionary that is read at all:
// 16 bytes per entry, so 112 bytes for a seven-value column and 1 MB for a
// pathological 65,536-entry one. That is bounded per row group and dies with the
// ColumnReader that parsed it.
//
// Get mutates the memo, so it is not safe to call on one Dictionary from two
// goroutines at once. See the ownership note on Dictionary.
func (d *Dictionary) Get(code uint32) string {
	if int(code) >= len(d.Offsets) {
		panic(fmt.Sprintf("storage: dict code %d out of range (len %d)", code, len(d.Offsets)))
	}
	if int(code) < len(d.cache) {
		// "" doubles as the not-yet-memoised sentinel, and that costs nothing:
		// an empty dictionary entry is a zero-length []byte→string conversion,
		// which Go satisfies with the empty string constant and no allocation.
		// So a miss on an empty entry is exactly as cheap as a hit would have
		// been, and no parallel "is populated" bitmap is needed to tell a
		// cached empty string from an unpopulated slot.
		if s := d.cache[code]; s != "" {
			return s
		}
	} else {
		// Sized on first use rather than in the constructor, so a Dictionary
		// that is parsed but never read costs nothing, and a Dictionary built as
		// a struct literal (which tests do) needs no constructor to be correct.
		d.cache = make([]string, len(d.Offsets))
	}

	start := d.Offsets[code]
	var end uint32
	if int(code)+1 < len(d.Offsets) {
		end = d.Offsets[code+1]
	} else {
		end = uint32(len(d.Data))
	}
	s := string(d.Data[start:end])
	d.cache[code] = s
	return s
}

// Len returns the number of entries in the dictionary.
func (d *Dictionary) Len() int { return len(d.Offsets) }

// Lookup returns (code, true) if s is in the dictionary.
func (d *Dictionary) Lookup(s string) (uint32, bool) {
	// Linear scan — dictionaries are small per row group (v1).
	for code := uint32(0); int(code) < len(d.Offsets); code++ {
		if d.Get(code) == s {
			return code, true
		}
	}
	return 0, false
}

// UnmarshalDictionary parses a dictionary blob (including its trailing CRC).
//
// Data is copied rather than aliased into b on purpose: b is often a Reader's
// pooled section window, which is handed back on ColumnReader.Close and reused by
// the next row group. Owning the bytes is what lets a Dictionary outlive the
// buffer it was parsed from, and what makes the immutability the ownership note
// on Dictionary relies on a property of this package rather than of its callers.
func UnmarshalDictionary(b []byte) (*Dictionary, error) {
	payload, err := enc.VerifyTrailing(b)
	if err != nil {
		return nil, wrap("unmarshal dict", ErrChecksum)
	}
	if len(payload) < 4 {
		return nil, wrap("unmarshal dict", ErrCorruptFooter)
	}
	n, rest := enc.ReadUint32(payload)
	if len(rest) < int(n)*4+4 {
		return nil, wrap("unmarshal dict", ErrCorruptFooter)
	}
	offsets := make([]uint32, n)
	for i := range offsets {
		offsets[i], rest = enc.ReadUint32(rest)
	}
	totalLen, rest := enc.ReadUint32(rest)
	if uint32(len(rest)) < totalLen {
		return nil, wrap("unmarshal dict", ErrCorruptFooter)
	}
	data := make([]byte, totalLen)
	copy(data, rest[:totalLen])
	return &Dictionary{Offsets: offsets, Data: data}, nil
}

// ---- RLE Bool encoder / decoder -------------------------------------------

// EncodeRLEBool encodes a Bool column block as RLE.
// values and nulls must have the same length; nulls uses the standard bitmap
// convention (1 = valid, 0 = null).  Null runs are encoded with value byte 0xFF.
//
// Format: [4B run_count][run_count × {4B length, 1B value}][4B CRC32]
func EncodeRLEBool(values []bool, nullBitmap []byte) []byte {
	type run struct {
		length uint32
		val    byte
	}
	var runs []run
	n := len(values)
	i := 0
	for i < n {
		var cur byte
		if IsNullBit(nullBitmap, i) {
			cur = 0xFF // null sentinel
		} else if values[i] {
			cur = 1
		} else {
			cur = 0
		}
		length := uint32(1)
		for i+int(length) < n {
			var next byte
			if IsNullBit(nullBitmap, i+int(length)) {
				next = 0xFF
			} else if values[i+int(length)] {
				next = 1
			} else {
				next = 0
			}
			if next != cur {
				break
			}
			length++
		}
		runs = append(runs, run{length, cur})
		i += int(length)
	}

	var b []byte
	b = enc.AppendUint32(b, uint32(len(runs)))
	for _, r := range runs {
		b = enc.AppendUint32(b, r.length)
		b = append(b, r.val)
	}
	return enc.AppendChecksum(b)
}

// DecodeRLEBool decodes a RLE-encoded Bool block (with trailing CRC).
// Returns the decoded values, the null bitmap, and the row count.
func DecodeRLEBool(data []byte) (values []bool, nullBitmap []byte, rows int, err error) {
	payload, err := enc.VerifyTrailing(data)
	if err != nil {
		return nil, nil, 0, wrap("decode rle bool", ErrChecksum)
	}
	if len(payload) < 4 {
		return nil, nil, 0, wrap("decode rle bool", ErrCorruptFooter)
	}
	runCount, rest := enc.ReadUint32(payload)
	total := 0
	type run struct {
		length uint32
		val    byte
	}
	runs := make([]run, runCount)
	for i := uint32(0); i < runCount; i++ {
		if len(rest) < 5 {
			return nil, nil, 0, wrap("decode rle bool", ErrCorruptFooter)
		}
		var l uint32
		l, rest = enc.ReadUint32(rest)
		v := rest[0]
		rest = rest[1:]
		runs[i] = run{l, v}
		total += int(l)
	}
	values = make([]bool, total)
	nullBitmap = FullBitmap(total)
	pos := 0
	for _, r := range runs {
		for j := uint32(0); j < r.length; j++ {
			switch r.val {
			case 0xFF:
				SetNullBit(nullBitmap, pos)
			case 1:
				values[pos] = true
			}
			pos++
		}
	}
	return values, nullBitmap, total, nil
}
