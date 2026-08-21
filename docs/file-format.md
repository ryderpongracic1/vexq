# The .vxq file format

Custom columnar format designed for vectorized reads:

- **Layout**: file header → row groups (65,536 rows each) → footer
- **Blocks**: 1,024 rows per block with 128-byte null bitmap + typed payload + CRC32
- **Endianness**: little-endian throughout (single `MOVQ`/`LDR` on x86-64/ARM64 vs `MOVQ+BSWAP`/`LDR+REV` for big-endian)
- **String columns**: always dictionary-encoded per row group — string equality becomes integer comparison in the filter hot loop
- **Bool columns**: run-length encoded with null sentinel
- **Zone maps**: per-row-group min/max/sum/nullcount in footer — entire row groups skipped before any block I/O
- **Atomic writes**: `write → fsync → rename` guarantees no partial files on crash
