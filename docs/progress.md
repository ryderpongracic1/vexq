# Build history

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | `.vxq` storage format — writer, reader, codec, zone maps | ✅ Complete |
| 2 | Vectorized execution engine — all operators | ✅ Complete |
| 3 | SQL parser — lexer, AST, recursive-descent | ✅ Complete |
| 4 | Catalog + planner — logical plan, optimizer, physical plan | ✅ Complete |
| 5 | CLI binary (`vexq`, `vexqgen`, `fsck`) | ✅ Complete |
| 6 | TPC-H benchmark harness vs SQLite | ✅ Complete |
| 7 | Morsel-driven parallelism — `ParallelHashAggregate`, `planner.Parallel()` | ✅ Complete |
| 8 | DuckDB honest baseline + per-core gap decomposition | ✅ Complete |
| 9 | Multi-table joins — qualified columns, alias resolution, cross-join rejection | ✅ Complete |
| 10 | CLI multi-file queries + `--workers` parallel flag | ✅ Complete |
| 11 | SIMD filter kernel benchmark — AVX2 ceiling measurement ([`bench/simd_filter/`](../bench/simd_filter/)) | ✅ Complete |
| 12 | Parallel scaling — decode-buffer reuse, sort-peeling, GC diagnosis | ✅ Complete |
| 13 | Aggregate optimization — packed dictionary-code integer keys | ✅ Complete |
| 14 | Correctness oracle — golden test suite (now 72 queries, 4 oracle paths) ([`internal/goldentest/`](../internal/goldentest/)) | ✅ Complete |
| 15 | Expression eval hardening — NOT precedence, date coercion, CASE WHEN strings, COUNT(DISTINCT) | ✅ Complete |
| 16 | Coarse-grained I/O — row-group-buffered reads, 62.9× pread reduction | ✅ Complete |
| 17 | Parallel expression aggregates + parallel hash join (probe side) | ✅ Complete |
| 18 | Join column pruning — needed-column sets pushed through `LogicalJoin` | ✅ Complete |
| 19 | Radix-partitioned parallel join build — lock-free two-pass, 64-partition measured optimum | ✅ Complete |
| 20 | Allocation campaign — scratch buffers, flat join table, window pool, pipeline reuse, rowSet, dict memo, presizing | ✅ Complete |
| 21 | Correctness hardening — stacked-filter physical-length convention; string/date/bool aggregates; oracle to 72 queries / 4 paths | ✅ Complete |
