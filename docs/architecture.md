# Hardware-level architecture

### Why 1024-row batches

An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB — well within L1 cache on every modern microarchitecture (32–48 KB on x86 Skylake+, **192 KB** on Apple M4 Pro performance cores). This means the innermost decode, filter, and aggregate loops all operate on data that is almost certainly already in L1, avoiding LLC round-trips that cost ~40 cycles each.

The batch size also amortizes the cost of one virtual `Next()` call across 1024 rows. A virtual dispatch on x86 is ~50 ns (indirect branch, possible iTLB miss); spread over 1024 rows that is ~0.05 ns per row — negligible. Processing row-at-a-time would spend more time dispatching than computing. This is the same constant used by Velox, DuckDB, and Photon.

### Cache-line discipline (64 bytes)

The `TableScan` decode loops in [`exec/scan.go`](../exec/scan.go) operate on tightly-packed little-endian payloads. For `INT64` columns, each iteration of the hot loop is:

```go
vals[i] = int64(binary.LittleEndian.Uint64(payload[i*8:]))
```

On x86-64, `binary.LittleEndian.Uint64` compiles to a single `MOVQ` (no byte-swap). On ARM64 (M4 Pro), it compiles to a single `LDR`. Eight consecutive values occupy exactly 64 bytes — one cache line — so the hardware prefetcher can issue fetch requests ahead of the loop at full bandwidth. Big-endian would require a `MOVQ+BSWAP` (x86) or `LDR+REV` (ARM64) pair, penalizing every decode by ~10–20%.

### Branch-predictor friendliness

`Filter` ([`exec/filter.go`](../exec/filter.go)) evaluates the predicate across an entire 1024-row batch to produce a `BoolVector`, then converts that vector to a `SelectionVector` of surviving row indices in a single pass:

```
predicate.Eval(batch) → BoolVector → BoolToSelVec → []uint16 indices
```

The predicate eval loop has no data-dependent branches — every row pays the same cost regardless of whether it passes or fails. The branch predictor sees a perfectly regular loop and hits ~100%. Downstream operators receive a `SelVec` and index through it; they never branch on per-row pass/fail.

### Why selection vectors, not filtered copies

`Filter` writes a `[]uint16` of surviving row indices rather than allocating new column vectors. Downstream operators index through the selection vector, which means:
- Zero allocation on the filter hot path
- The 1024-row buffer is reused across the pipeline
- Column data is never copied — only indices change

### Morsel-driven parallelism

`planner.Parallel()` ([`planner/parallel.go`](../planner/parallel.go)) detects aggregate plan shapes — including `Sort → Aggregate → (Filter →)? Scan` — and builds a `ParallelHashAggregate` ([`exec/parallel.go`](../exec/parallel.go)), wrapping it with a serial sort/limit where needed.

Rather than statically assigning equal slices of row groups to each goroutine (which starves fast workers when selectivity varies), the executor uses an **atomic-counter morsel queue**: each of the `runtime.NumCPU()` goroutines claims row groups one at a time from a shared `atomic.Int64` cursor. Fast workers — those whose morsels have high filter selectivity — claim more work; slow workers naturally finish last without blocking others. The counter is padded to its own 64-byte cache line to prevent false-sharing with other struct fields.

Each goroutine runs a fully independent `TableScan → Filter → Project → HashAggregate` pipeline on its morsels, accumulates partial results locally with no shared mutable state, then sends its `HashAggregate` state on a buffered channel. The calling goroutine merges all partial aggregates (correctly handling float64 SUM/MIN/MAX via IEEE-bit re-encoding and AVG via sum+count).

This design follows the principles of [Leis et al., "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age," SIGMOD 2014](https://db.in.tum.de/~leis/papers/morsels.pdf).

## Design notes

**Why pull-based (Volcano model)?** `LIMIT` and short-circuit predicates terminate naturally — when the root stops calling `Next()`, all upstream work stops with no extra machinery. Simpler to debug single-threaded, and composes cleanly with the morsel-driven parallel layer above it.

**How morsel-driven parallelism works.** Two distinct layers handle plan-shape detection and dynamic scheduling:

- `planner.Parallel()` ([`planner/parallel.go`](../planner/parallel.go)) detects aggregate plan shapes (including `Sort → Aggregate` and `Limit → Sort → Aggregate`), resolves group-by/aggregate column indices, and builds a `PipelineFactory` closure (one independent `TableScan → Filter → Project` pipeline per call). Sort/limit nodes above the aggregate are peeled off and applied serially to the small merged result.
- `ParallelHashAggregate.setup()` ([`exec/parallel.go`](../exec/parallel.go)) handles *dynamic scheduling*: a `morselQueue` wraps an `atomic.Int64` cursor padded to its own 64-byte cache line (preventing false-sharing). Each of the `numWorkers` goroutines loops calling `q.claim(morselSize)` to atomically reserve the next chunk of row groups, runs the pipeline on that morsel, accumulates into a goroutine-local `HashAggregate`, and claims the next morsel — no coordinator, no barriers between morsels. Fast workers (low-selectivity morsels) complete early and immediately claim more work, eliminating stragglers. After all workers drain the queue, the calling goroutine merges all partial aggregates single-threadedly (correct IEEE float64 via bit-reencoding). Each worker builds its pipeline **once** and repositions it with `TableScan.Reset(rgStart, rgEnd)` for each morsel it claims, so a query opens one `storage.Reader` per worker rather than one per morsel — before that, a 92-row-group scan re-read and re-parsed the footer ~112 times per query, which made `storage.Open` the second-largest allocation site in a parallel aggregate. Reuse is opt-in per pipeline shape: `exec.MorselPipeline` ([`exec/morsel.go`](../exec/morsel.go)) names the operators that can be repositioned (scan, filter, projection, and a join probing a shared build table), and any pipeline containing anything else falls back to being rebuilt per morsel rather than being partially reset.

**Why 1024-row batches?** An `Int64Vector` of 1024 rows is 8 KB values + 128 B nulls ≈ 8.2 KB, fitting comfortably in L1 (32–48 KB on x86, 192 KB on Apple M4 Pro performance cores). Per-batch overhead (one `Next()` call, type assertions) amortizes over 1024 rows. Same constant used by Velox, DuckDB, and Photon.

**Why selection vectors instead of filtered batches?** `Filter` writes a `[]uint16` of surviving row indices rather than allocating new vectors. Downstream operators index through the selection vector, saving allocation on the hot path and preserving the 1024-row invariant across the pipeline.

**Why little-endian?** The inner loop of every `TableScan` is `binary.LittleEndian.Uint64(buf[i*8:])` — on x86-64 this compiles to a single `MOVQ`; on ARM64 (M4 Pro) a single `LDR`. Big-endian forces an extra byte-reverse (`BSWAP`/`REV`), penalizing column reads by ~10–20%.
