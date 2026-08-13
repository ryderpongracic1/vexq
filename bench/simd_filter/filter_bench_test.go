// Package simd_filter benchmarks the filter-kernel hot loop that is the
// primary predicate evaluation path in vexq's execution engine.
//
// This file replicates the real structure from exec/expr.go (evalCmpInt64 +
// BoolToSelVec) to measure what Go actually achieves on the same workload the
// C++ SIMD kernel benchmarks. It does NOT import exec/ packages to keep this
// benchmark zero-dependency and avoid polluting the module with test-only
// build constraints.
//
// The filter loop operates on int64 slices (matching vexq's Int64Vector) and
// produces a []uint16 selection vector (matching vexq's SelectionVector).
package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"testing"
)

const (
	blockRows   = 1024      // vexq's BlockRows constant
	totalRows   = 6001215   // TPC-H SF=1 lineitem row count
	threshold   = int64(25) // ~50% selectivity on uniform [1,50]
	dataFileI64 = "bench_data_i64.bin"
)

// loadData reads the shared binary data file (same bytes as C++ harness).
func loadData(path string, n int) ([]int64, error) {
	f, err := os.Open(path)
	if err != nil {
		// Generate if missing (same xorshift64* algorithm as C++)
		return generateData(path, n)
	}
	defer f.Close()

	data := make([]int64, n)
	err = binary.Read(f, binary.LittleEndian, data)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return data, nil
}

func generateData(path string, n int) ([]int64, error) {
	data := make([]int64, n)
	state := uint64(42)
	for i := range data {
		state ^= state >> 12
		state ^= state << 25
		state ^= state >> 27
		val := state * 0x2545F4914F6CDD1D
		data[i] = int64((val % 50) + 1) // [1, 50]
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	err = binary.Write(f, binary.LittleEndian, data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// ─── Filter kernel: exact replica of vexq's evalCmpInt64 (BinGT case) ────────
// This is the same unrolled 8-at-a-time comparison loop from exec/expr.go
// that produces a BoolVector (packed bits), followed by the BoolToSelVec
// conversion to produce the selection vector.
//
// Methodology note: this benchmark holds the comparison threshold in a scalar
// register, whereas the real engine's evalCmpInt64 loads rv[i] per element from
// a broadcast vector. The measured Go ns/row is therefore slightly optimistic;
// the true engine gap vs the C++ SIMD kernel may be marginally larger than
// reported.

func filterGtI64(data []int64, n int, thresh int64, bits []byte) {
	// Exact replica of evalCmpInt64 BinGT case from exec/expr.go
	i := 0
	for ; i+8 <= n; i += 8 {
		b := i >> 3
		var byte_ uint8
		if data[i+0] > thresh {
			byte_ |= 0x01
		}
		if data[i+1] > thresh {
			byte_ |= 0x02
		}
		if data[i+2] > thresh {
			byte_ |= 0x04
		}
		if data[i+3] > thresh {
			byte_ |= 0x08
		}
		if data[i+4] > thresh {
			byte_ |= 0x10
		}
		if data[i+5] > thresh {
			byte_ |= 0x20
		}
		if data[i+6] > thresh {
			byte_ |= 0x40
		}
		if data[i+7] > thresh {
			byte_ |= 0x80
		}
		bits[b] = byte_
	}
	// Tail
	if i < n {
		b := i >> 3
		var byte_ uint8
		for ; i < n; i++ {
			if data[i] > thresh {
				byte_ |= 1 << uint(i&7)
			}
		}
		bits[b] = byte_
	}
}

func boolToSelVec(bits []byte, n int, sel []uint16) int {
	// Exact replica of BoolToSelVec from exec/filter.go (nil SelVec case)
	count := 0
	for i := 0; i < n; i++ {
		if bits[i/8]>>(uint(i%8))&1 == 1 {
			sel[count] = uint16(i)
			count++
		}
	}
	return count
}

// ─── Benchmarks ──────────────────────────────────────────────────────────────

var (
	benchData []int64
	sinkCount int
)

func setupData(b *testing.B) {
	b.Helper()
	if benchData == nil {
		var err error
		benchData, err = loadData(dataFileI64, totalRows)
		if err != nil {
			b.Fatalf("load data: %v", err)
		}
	}
}

// BenchmarkFilterGtI64_BatchLoop benchmarks the full filter pipeline:
// evalCmpInt64 → BoolToSelVec, operating on 1024-row batches over the entire
// dataset, exactly as vexq's Filter operator does.
func BenchmarkFilterGtI64_BatchLoop(b *testing.B) {
	setupData(b)

	bits := make([]byte, (blockRows+7)/8)
	sel := make([]uint16, blockRows)

	b.SetBytes(int64(totalRows) * 8) // int64 = 8 bytes
	b.ResetTimer()

	for iter := 0; iter < b.N; iter++ {
		totalSel := 0
		for off := 0; off < totalRows; off += blockRows {
			batchN := blockRows
			if off+batchN > totalRows {
				batchN = totalRows - off
			}
			// Clear bits for this batch
			for j := range bits {
				bits[j] = 0
			}
			filterGtI64(benchData[off:off+batchN], batchN, threshold, bits)
			count := boolToSelVec(bits, batchN, sel)
			totalSel += count
		}
		sinkCount = totalSel
	}
}

// BenchmarkFilterGtI64_SingleBatch benchmarks a single 1024-row batch
// (the innermost hot loop) for latency measurement.
func BenchmarkFilterGtI64_SingleBatch(b *testing.B) {
	setupData(b)

	bits := make([]byte, (blockRows+7)/8)
	sel := make([]uint16, blockRows)

	b.SetBytes(int64(blockRows) * 8)
	b.ResetTimer()

	for iter := 0; iter < b.N; iter++ {
		for j := range bits {
			bits[j] = 0
		}
		filterGtI64(benchData[:blockRows], blockRows, threshold, bits)
		sinkCount = boolToSelVec(bits, blockRows, sel)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
