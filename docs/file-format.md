# The .vxq file format

Custom columnar format designed for vectorized reads:

- **Layout**: file header → row groups (65,536 rows each) → footer
- **Blocks**: 1,024 rows per block with 128-byte null bitmap + typed payload + CRC32
- **Endianness**: little-endian throughout (single `MOVQ`/`LDR` on x86-64/ARM64 vs `MOVQ+BSWAP`/`LDR+REV` for big-endian)
- **String columns**: always dictionary-encoded per row group — string equality becomes integer comparison in the filter hot loop
- **Bool columns**: run-length encoded with null sentinel
- **Zone maps**: per-row-group min/max/sum/nullcount in footer — entire row groups skipped before any block I/O. Pruning compares min/max in the column's own domain (signed integer, float, days) and does not use `STRING` or `BOOL` zone maps
- **Integrity**: `vexq fsck` verifies every block CRC, the dictionaries and RLE blocks, and recomputes each column's zone map from its stored values. A zone map that disagrees with the data would make queries silently skip matching rows, and the footer CRC cannot detect that because it covers the wrong stats as readily as the right ones
- **Atomic writes**: `write → fsync → rename` guarantees no partial files on crash
