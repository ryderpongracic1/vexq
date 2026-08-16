package exec

import "github.com/ryderpongracic1/vexq/storage"

// BlockRows is the canonical vector (batch) size.
const BlockRows = storage.BlockRows // 1024

// SelectionVector contains the row indices within a Batch that survived a
// filter predicate.  nil means "all rows are selected".
type SelectionVector []uint16

// Batch is a collection of typed column vectors plus an optional selection
// vector.  Length is the logical row count:
//   - if SelVec == nil: Length == len(Vectors[0].Values) (or 0 if no columns)
//   - if SelVec != nil: Length == len(SelVec)
type Batch struct {
	Schema  Schema
	Vectors []Vector
	Length  int
	SelVec  SelectionVector
}

// NewBatch allocates an empty batch with the given schema.
func NewBatch(schema Schema) *Batch {
	vecs := make([]Vector, len(schema.Fields))
	for i, f := range schema.Fields {
		vecs[i] = makeVector(f.Type)
	}
	return &Batch{Schema: schema, Vectors: vecs}
}

// physicalLen returns the number of rows in b's underlying vectors, independent
// of any selection vector. This is the true allocation size.
func physicalLen(b *Batch) int {
	if len(b.Vectors) > 0 {
		return b.Vectors[0].Len()
	}
	return b.Length
}

// evalLen returns the number of rows an expression must produce for b, per the
// sizing convention documented on the Expr interface (expr.go): the physical row
// count whenever a selection vector is active, Batch.Length otherwise.
//
// The two answers differ only when a selection vector is installed — Batch's own
// invariant above says Length equals the vectors' length when SelVec is nil — so
// this widens the length in exactly the case that needs it and changes nothing
// elsewhere. Expression leaves that have no child to take their length from must
// call this rather than reading b.Length.
func evalLen(b *Batch) int {
	if b.SelVec == nil {
		return b.Length
	}
	return physicalLen(b)
}

// rowSet names the physical row indices an operator must visit in one batch.
//
// It exists so a per-row consumer can honour a selection vector without first
// widening the batch's []uint16 into a []int of physical indices. That widening
// was HashAggregate.accumulate's single largest cost in the allocation profile —
// 285 MB of 393 MB sampled over three parallel benchmarks, an 8 KB throwaway per
// 1024-row batch per worker for the whole scan — and the []int it produced was a
// pure adapter: every consumer only ever read it back one index at a time.
//
// A rowSet is a value with no backing buffer, so constructing one allocates
// nothing and there is no scratch to own, grow, or reset between batches (see
// the ownership rules in scratch.go — this sidesteps them rather than obeying
// them). sel == nil means the identity range 0..n-1, which is Batch.SelVec's own
// convention, so an unfiltered batch needs no index storage at all.
type rowSet struct {
	sel SelectionVector // nil means the identity range
	n   int
}

// batchRows returns the rows of b a per-row consumer must visit, honouring
// b.SelVec when one is installed.
func batchRows(b *Batch) rowSet {
	if b.SelVec != nil {
		return rowSet{sel: b.SelVec, n: len(b.SelVec)}
	}
	return rowSet{n: b.Length}
}

// at returns the physical row index of the i-th selected row. i must be in
// [0, n). Both this and the nil check inline; the branch is loop-invariant
// across a batch and so predicts perfectly.
func (r rowSet) at(i int) int {
	if r.sel == nil {
		return i
	}
	return int(r.sel[i])
}

func makeVector(t DataType) Vector {
	switch t {
	case TypeInt64:
		return &Int64Vector{Values: make([]int64, 0, BlockRows), NullBitmap: make([]byte, BlockRows/8)}
	case TypeFloat64:
		return &Float64Vector{Values: make([]float64, 0, BlockRows), NullBitmap: make([]byte, BlockRows/8)}
	case TypeBool:
		return &BoolVector{NullBitmap: make([]byte, BlockRows/8)}
	case TypeString:
		return &StringVector{Codes: make([]uint32, 0, BlockRows), NullBitmap: make([]byte, BlockRows/8)}
	case TypeDate:
		return &DateVector{Values: make([]int32, 0, BlockRows), NullBitmap: make([]byte, BlockRows/8)}
	default:
		panic("exec: unknown data type")
	}
}

// ---- Vector interface -------------------------------------------------------

// Vector is a typed column of up to BlockRows values plus a null bitmap.
type Vector interface {
	Len() int
	Type() DataType
	IsNull(i int) bool
	Nulls() []byte // raw bitmap, LSB-first, 1=valid, 0=null
}

// ---- Concrete vector types -------------------------------------------------

type Int64Vector struct {
	Values     []int64
	NullBitmap []byte
}

func (v *Int64Vector) Len() int          { return len(v.Values) }
func (v *Int64Vector) Type() DataType    { return TypeInt64 }
func (v *Int64Vector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *Int64Vector) Nulls() []byte     { return v.NullBitmap }

type Float64Vector struct {
	Values     []float64
	NullBitmap []byte
}

func (v *Float64Vector) Len() int          { return len(v.Values) }
func (v *Float64Vector) Type() DataType    { return TypeFloat64 }
func (v *Float64Vector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *Float64Vector) Nulls() []byte     { return v.NullBitmap }

type BoolVector struct {
	// Bits holds the actual bool values: 1 = true.
	Bits       []byte
	NullBitmap []byte
	Length     int
}

func (v *BoolVector) Len() int          { return v.Length }
func (v *BoolVector) Type() DataType    { return TypeBool }
func (v *BoolVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *BoolVector) Nulls() []byte     { return v.NullBitmap }
func (v *BoolVector) Get(i int) bool    { return v.Bits[i/8]>>(uint(i%8))&1 == 1 }
func (v *BoolVector) Set(i int, val bool) {
	if val {
		v.Bits[i/8] |= 1 << uint(i%8)
	} else {
		v.Bits[i/8] &^= 1 << uint(i%8)
	}
}

type StringVector struct {
	Codes      []uint32
	Dict       *storage.Dictionary
	NullBitmap []byte
}

func (v *StringVector) Len() int          { return len(v.Codes) }
func (v *StringVector) Type() DataType    { return TypeString }
func (v *StringVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *StringVector) Nulls() []byte     { return v.NullBitmap }
func (v *StringVector) Get(i int) string {
	if v.Dict == nil {
		return ""
	}
	return v.Dict.Get(v.Codes[i])
}

type DateVector struct {
	Values     []int32
	NullBitmap []byte
}

func (v *DateVector) Len() int          { return len(v.Values) }
func (v *DateVector) Type() DataType    { return TypeDate }
func (v *DateVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *DateVector) Nulls() []byte     { return v.NullBitmap }

// newStringVector builds a StringVector from a DictBuilder, pre-filled codes,
// and a null bitmap.  Shared by aggregate and sort output paths.
func newStringVector(db *storage.DictBuilder, codes []uint32, nullBmp []byte) *StringVector {
	rawDict := db.Marshal()
	dict, _ := storage.UnmarshalDictionary(rawDict)
	return &StringVector{Codes: codes, Dict: dict, NullBitmap: nullBmp}
}
