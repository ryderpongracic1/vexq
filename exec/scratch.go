package exec

// Operator scratch buffers — the aliasing contract
// ================================================
//
// TableScan reuses its decode buffers across blocks (see scan.go). The helpers
// in this file extend the same pattern to the operators above it: an expression
// node or operator owns one buffer per output it produces and rewrites that
// buffer on every Eval/Next instead of allocating a new vector per batch.
//
// The contract every scratch buffer in exec/ obeys:
//
//  1. OWNERSHIP. A buffer belongs to exactly one Expr or Operator *instance*.
//     Two nodes never share one, so two different nodes in the same expression
//     tree can hold live results simultaneously.
//
//  2. LIFETIME. The vector an Expr returns stays valid until the next Eval on
//     that same node. A consumer must read it (or copy out of it) before
//     evaluating that node again — the same rule TableScan already imposes on
//     its batches.
//
//  3. READ-ONLY TO CONSUMERS. Only the producing node writes its own buffer.
//     No parent mutates a child's result in place. (AndExpr, OrExpr and NotExpr
//     used to; they now combine into their own scratch. That is what makes it
//     safe for the same node to appear twice in one tree: re-evaluating it
//     recomputes identical values rather than clobbering a mutated result.)
//
//  4. GOROUTINE LOCALITY. Scratch is not synchronised, so an Expr tree must not
//     be shared across goroutines. Every parallel worker builds its own
//     pipeline and its own Expr tree — planner.Parallel's factory calls
//     buildExecExpr for the worker that invoked it (planner/parallel.go), as do
//     the join factories in planner/parallel_join.go and buildPreProjection in
//     planner/physical.go — so worker pipelines share no expression state. A
//     worker that reuses one pipeline across its morsels (exec/morsel.go) keeps
//     that tree on the one goroutine for the whole run, which narrows the
//     exposure rather than widening it.
//
//  5. NO SCRATCH ESCAPES Project. Project is the boundary between reused
//     expression scratch and the batches it hands downstream: it materialises
//     through the selection vector when one is present, and copies non-passthrough
//     expression results when one is not. Bare column references still pass
//     through, exactly as before, so Project's output aliases the same buffers
//     it always did — TableScan's — and nothing more.
//
// The acquire helpers below all zero the parts of a reused buffer that a fresh
// allocation would have zeroed, so a reused vector is byte-identical to the
// freshly made one it replaces.

// acquireBoolVector returns a BoolVector of exactly n rows, reusing *slot when
// its buffers are large enough. Bits and NullBitmap are zeroed, matching what a
// freshly allocated BoolVector would contain.
func acquireBoolVector(slot **BoolVector, n int) *BoolVector {
	size := (n + 7) / 8
	bv := *slot
	if bv == nil || cap(bv.Bits) < size || cap(bv.NullBitmap) < size {
		bv = &BoolVector{
			Bits:       make([]byte, size),
			NullBitmap: make([]byte, size),
		}
		*slot = bv
	} else {
		bv.Bits = bv.Bits[:size]
		bv.NullBitmap = bv.NullBitmap[:size]
		clear(bv.Bits)
		clear(bv.NullBitmap)
	}
	bv.Length = n
	return bv
}

// acquireInt64Vector returns an Int64Vector of exactly n rows with zeroed values
// and a zeroed (all-null) bitmap, reusing *slot when it is large enough.
func acquireInt64Vector(slot **Int64Vector, n int) *Int64Vector {
	size := (n + 7) / 8
	v := *slot
	if v == nil || cap(v.Values) < n || cap(v.NullBitmap) < size {
		v = &Int64Vector{
			Values:     make([]int64, n),
			NullBitmap: make([]byte, size),
		}
		*slot = v
		return v
	}
	v.Values = v.Values[:n]
	v.NullBitmap = v.NullBitmap[:size]
	clear(v.Values)
	clear(v.NullBitmap)
	return v
}

// acquireFloat64Vector returns a Float64Vector of exactly n rows with zeroed
// values and a zeroed (all-null) bitmap, reusing *slot when it is large enough.
func acquireFloat64Vector(slot **Float64Vector, n int) *Float64Vector {
	size := (n + 7) / 8
	v := *slot
	if v == nil || cap(v.Values) < n || cap(v.NullBitmap) < size {
		v = &Float64Vector{
			Values:     make([]float64, n),
			NullBitmap: make([]byte, size),
		}
		*slot = v
		return v
	}
	v.Values = v.Values[:n]
	v.NullBitmap = v.NullBitmap[:size]
	clear(v.Values)
	clear(v.NullBitmap)
	return v
}

// acquireValidBitmap resizes *slot to cover n rows and marks every row valid,
// reproducing storage.FullBitmap exactly — including zeroing the bits past n in
// the final byte.
func acquireValidBitmap(slot *[]byte, n int) []byte {
	size := (n + 7) / 8
	b := *slot
	if cap(b) < size {
		b = make([]byte, size)
		*slot = b
	} else {
		b = b[:size]
	}
	setAllValid(b, n)
	return b
}

// growSelVec resizes sel to hold up to n indices, reusing its array. The
// returned slice has length 0.
func growSelVec(sel SelectionVector, n int) SelectionVector {
	if cap(sel) < n {
		return make(SelectionVector, 0, n)
	}
	return sel[:0]
}

// copyVector returns a freshly allocated copy of v holding rows [0, n). It is
// the barrier Project uses so reused expression scratch never escapes into a
// batch handed downstream.
func copyVector(v Vector, n int) Vector {
	switch src := v.(type) {
	case *Int64Vector:
		out := &Int64Vector{Values: make([]int64, n), NullBitmap: make([]byte, (n+7)/8)}
		copy(out.Values, src.Values[:n])
		copy(out.NullBitmap, src.NullBitmap)
		return out
	case *Float64Vector:
		out := &Float64Vector{Values: make([]float64, n), NullBitmap: make([]byte, (n+7)/8)}
		copy(out.Values, src.Values[:n])
		copy(out.NullBitmap, src.NullBitmap)
		return out
	case *DateVector:
		out := &DateVector{Values: make([]int32, n), NullBitmap: make([]byte, (n+7)/8)}
		copy(out.Values, src.Values[:n])
		copy(out.NullBitmap, src.NullBitmap)
		return out
	case *BoolVector:
		out := &BoolVector{
			Bits:       make([]byte, (n+7)/8),
			NullBitmap: make([]byte, (n+7)/8),
			Length:     n,
		}
		copy(out.Bits, src.Bits)
		copy(out.NullBitmap, src.NullBitmap)
		return out
	case *StringVector:
		out := &StringVector{
			Codes:      make([]uint32, n),
			Dict:       src.Dict,
			NullBitmap: make([]byte, (n+7)/8),
		}
		copy(out.Codes, src.Codes[:n])
		copy(out.NullBitmap, src.NullBitmap)
		return out
	default:
		return v
	}
}

// mergeNullBitmapsInto writes into dst the bitmap where a bit is valid only if
// it is valid in both a and b. dst must already cover (n+7)/8 bytes.
func mergeNullBitmapsInto(dst, a, b []byte, n int) {
	size := (n + 7) / 8
	for i := 0; i < size && i < len(dst); i++ {
		ai, bi := byte(0xFF), byte(0xFF)
		if i < len(a) {
			ai = a[i]
		}
		if i < len(b) {
			bi = b[i]
		}
		dst[i] = ai & bi
	}
}

// setAllValid marks rows [0, n) valid in bmp, zeroing the trailing bits of the
// last byte the way storage.FullBitmap does.
func setAllValid(bmp []byte, n int) {
	size := (n + 7) / 8
	for i := 0; i < size && i < len(bmp); i++ {
		bmp[i] = 0xFF
	}
	if n%8 != 0 && size > 0 && size <= len(bmp) {
		bmp[size-1] = 1<<uint(n%8) - 1
	}
}
