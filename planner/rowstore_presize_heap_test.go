package planner_test

import (
	"context"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ryderpongracic1/vexq/catalog"
	"github.com/ryderpongracic1/vexq/exec"
	"github.com/ryderpongracic1/vexq/planner"
	"github.com/ryderpongracic1/vexq/sql"
)

// This file measures heap *retention* rather than allocation volume, because
// presizing a build-side rowStore trades one for the other and only one of them
// shows up in -benchmem.
//
// Growing by doubling allocates ~2x the final payload in total (all of which
// -benchmem counts) and, at the moment of the last copy, holds the old arrays
// and the new ones at once — so a store that ends up holding N rows can peak at
// close to 3N rows of live payload and settle at up to 2N, since the final
// doubling overshoots. Presizing from a known row count allocates exactly N and
// holds exactly N. The prediction under test is therefore that presizing reduces
// peak live heap as well as allocation volume, which is not the direction an
// allocation optimisation usually goes.
//
// Two numbers, because the two build paths retain differently:
//
//   - peak-heap-MB samples live heap through the whole query and takes the
//     maximum. It is the only way to see the parallel build, whose per-morsel
//     stores and final store are all live at once during pass 2 and all garbage
//     by the time the query returns. Sampled, so it is approximate and slightly
//     under-reports narrow spikes.
//   - held-heap-MB forces a GC with the build side still reachable and reports
//     what survives. Deterministic, and directly the capacity overshoot the
//     serial store settles at.
//
// Run with:
//
//	VEXQ_BENCH_DIR=<dir> go test ./planner/ -run '^$' \
//	    -bench 'BenchmarkJoinHeap' -benchtime=5x
func BenchmarkJoinHeapQ12Serial(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), q12Shaped, 0)
}

func BenchmarkJoinHeapQ12Parallel4(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), q12Shaped, 4)
}

func BenchmarkJoinHeapQ3Serial(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), q3Shaped, 0)
}

func BenchmarkJoinHeapQ3Parallel4(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), q3Shaped, 4)
}

func BenchmarkJoinHeapProbeHeavySerial(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), probeHeavyJoin, 0)
}

func BenchmarkJoinHeapProbeHeavyParallel4(b *testing.B) {
	benchJoinHeap(b, benchCatalog(b), probeHeavyJoin, 4)
}

// benchJoinHeap runs query once per iteration and reports the maximum live heap
// observed during the run, plus the live heap that survives a forced GC while the
// operator tree — and so the build side it holds — is still reachable.
//
// Both are reported as the maximum over iterations rather than the mean: the
// quantity of interest is a peak, and averaging peaks across iterations reports
// something that is neither a peak nor a typical value.
func benchJoinHeap(b *testing.B, cat *catalog.Catalog, query string, workers int) {
	b.Helper()
	ctx := context.Background()

	p := sql.NewParser(query)
	node, err := p.ParseStatement()
	if err != nil {
		b.Fatalf("parse: %v", err)
	}
	stmt := node.(*sql.SelectStmt)

	var peak, held uint64
	for range b.N {
		logical, err := planner.Build(ctx, stmt, cat)
		if err != nil {
			b.Fatalf("Build: %v", err)
		}
		logical = planner.Optimize(logical)

		var op exec.Operator
		if workers > 0 {
			op, err = planner.Parallel(ctx, logical, workers)
		} else {
			op, err = planner.Physical(ctx, logical)
		}
		if err != nil {
			b.Fatalf("plan (workers=%d): %v", workers, err)
		}

		// Start from a known heap so the peak is this query's, not the previous
		// iteration's garbage waiting to be collected.
		runtime.GC()
		stop, maxHeap := sampleHeap()

		rows := 0
		for {
			batch, err := op.Next(ctx)
			if err != nil {
				b.Fatalf("Next: %v", err)
			}
			if batch == nil {
				break
			}
			rows += batch.Length
		}
		stop()

		// op is still reachable here, so its build side is too: what survives
		// this GC is what the finished build side holds.
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		if ms.HeapAlloc > held {
			held = ms.HeapAlloc
		}
		if h := maxHeap(); h > peak {
			peak = h
		}

		_ = op.Close()
		if rows == 0 {
			b.Fatal("query returned no rows — benchmark data does not match the query")
		}
	}
	const mb = 1 << 20
	b.ReportMetric(float64(peak)/mb, "peak-heap-MB")
	b.ReportMetric(float64(held)/mb, "held-heap-MB")
}

// sampleHeap starts a goroutine that records the maximum HeapAlloc it observes
// until stop is called, and returns stop plus an accessor for the maximum. The
// accessor is only valid after stop returns.
//
// runtime.ReadMemStats stops the world briefly, so the sampling interval is a
// compromise: short enough to catch a build phase that lasts tens of
// milliseconds, long enough not to change what it measures. At 200us over an
// ~80ms query that is ~400 samples and well under 1% of wall time.
func sampleHeap() (stop func(), max func() uint64) {
	var peak atomic.Uint64
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		t := time.NewTicker(200 * time.Microsecond)
		defer t.Stop()
		var ms runtime.MemStats
		for {
			select {
			case <-done:
				return
			case <-t.C:
				runtime.ReadMemStats(&ms)
				for {
					cur := peak.Load()
					if ms.HeapAlloc <= cur || peak.CompareAndSwap(cur, ms.HeapAlloc) {
						break
					}
				}
			}
		}
	}()
	return func() {
			close(done)
			<-stopped
		}, func() uint64 {
			return peak.Load()
		}
}
