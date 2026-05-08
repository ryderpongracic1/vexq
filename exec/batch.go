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

func (v *Int64Vector) Len() int        { return len(v.Values) }
func (v *Int64Vector) Type() DataType  { return TypeInt64 }
func (v *Int64Vector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *Int64Vector) Nulls() []byte   { return v.NullBitmap }

type Float64Vector struct {
	Values     []float64
	NullBitmap []byte
}

func (v *Float64Vector) Len() int        { return len(v.Values) }
func (v *Float64Vector) Type() DataType  { return TypeFloat64 }
func (v *Float64Vector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *Float64Vector) Nulls() []byte   { return v.NullBitmap }

type BoolVector struct {
	// Bits holds the actual bool values: 1 = true.
	Bits       []byte
	NullBitmap []byte
	Length     int
}

func (v *BoolVector) Len() int        { return v.Length }
func (v *BoolVector) Type() DataType  { return TypeBool }
func (v *BoolVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *BoolVector) Nulls() []byte   { return v.NullBitmap }
func (v *BoolVector) Get(i int) bool  { return v.Bits[i/8]>>(uint(i%8))&1 == 1 }
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

func (v *StringVector) Len() int        { return len(v.Codes) }
func (v *StringVector) Type() DataType  { return TypeString }
func (v *StringVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *StringVector) Nulls() []byte   { return v.NullBitmap }
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

func (v *DateVector) Len() int        { return len(v.Values) }
func (v *DateVector) Type() DataType  { return TypeDate }
func (v *DateVector) IsNull(i int) bool { return storage.IsNullBit(v.NullBitmap, i) }
func (v *DateVector) Nulls() []byte   { return v.NullBitmap }
