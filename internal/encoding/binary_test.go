package encoding

import (
	"math"
	"testing"
)

func TestRoundTripInt64(t *testing.T) {
	cases := []int64{0, 1, -1, math.MaxInt64, math.MinInt64, 42}
	for _, v := range cases {
		b := AppendInt64(nil, v)
		got, rest := ReadInt64(b)
		if got != v || len(rest) != 0 {
			t.Fatalf("int64 round-trip %d: got %d, rest %d", v, got, len(rest))
		}
	}
}

func TestRoundTripFloat64(t *testing.T) {
	cases := []float64{0, 1.5, -1.5, math.MaxFloat64, math.SmallestNonzeroFloat64, math.NaN()}
	for _, v := range cases {
		b := AppendFloat64(nil, v)
		got, rest := ReadFloat64(b)
		if math.IsNaN(v) {
			if !math.IsNaN(got) {
				t.Fatalf("float64 round-trip NaN: got %v", got)
			}
		} else if got != v || len(rest) != 0 {
			t.Fatalf("float64 round-trip %v: got %v", v, got)
		}
	}
}

func TestRoundTripUint32(t *testing.T) {
	cases := []uint32{0, 1, math.MaxUint32, 0xDEADBEEF}
	for _, v := range cases {
		b := AppendUint32(nil, v)
		got, rest := ReadUint32(b)
		if got != v || len(rest) != 0 {
			t.Fatalf("uint32 round-trip %d: got %d", v, got)
		}
	}
}

func TestRoundTripInt32(t *testing.T) {
	cases := []int32{0, 1, -1, math.MaxInt32, math.MinInt32}
	for _, v := range cases {
		b := AppendInt32(nil, v)
		got, rest := ReadInt32(b)
		if got != v || len(rest) != 0 {
			t.Fatalf("int32 round-trip %d: got %d", v, got)
		}
	}
}

func TestChainedAppend(t *testing.T) {
	b := AppendUint32(nil, 0xABCD1234)
	b = AppendInt64(b, -999)
	b = AppendUint16(b, 0xFFFF)

	u, b2 := ReadUint32(b)
	if u != 0xABCD1234 {
		t.Fatalf("uint32 %x", u)
	}
	i, b3 := ReadInt64(b2)
	if i != -999 {
		t.Fatalf("int64 %d", i)
	}
	u16, b4 := ReadUint16(b3)
	if u16 != 0xFFFF || len(b4) != 0 {
		t.Fatalf("uint16 %x rest %d", u16, len(b4))
	}
}
