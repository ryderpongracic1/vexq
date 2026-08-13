# bench/simd_filter/Makefile — SIMD filter kernel benchmark
#
# Targets:
#   build       — compile AVX2-enabled benchmark binary
#   scalar-novec — compile with auto-vectorization disabled (honest scalar baseline)
#   run         — execute the AVX2 benchmark (generates data if missing)
#   run-novec   — execute the scalar-novec variant
#   bench-go    — run Go benchmark (mirrors vexq's real filter loop)
#   gendata     — regenerate benchmark data files
#   clean       — remove binaries and generated data

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -mavx2 -Wall -Wextra
# -mavx2 is required so explicit AVX2 intrinsics in the same binary compile;
# -fno-tree-vectorize/-fno-tree-loop-vectorize prevent auto-vectorization of the
# scalar functions, so scalar-novec numbers are honestly non-vectorized.
NOVEC_FLAGS := -std=c++17 -O3 -mavx2 -fno-tree-vectorize -fno-tree-loop-vectorize -Wall -Wextra

SRC      := cpp/filter_kernel.cpp cpp/main.cpp
BIN      := filter_bench
BIN_NOVEC:= filter_bench_novec
DATA_I64 := bench_data_i64.bin
DATA_I32 := bench_data_i32.bin

.PHONY: build scalar-novec run run-novec bench-go gendata clean

build: $(BIN)

$(BIN): $(SRC) cpp/filter_kernel.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

scalar-novec: $(BIN_NOVEC)

$(BIN_NOVEC): $(SRC) cpp/filter_kernel.h
	$(CXX) $(NOVEC_FLAGS) -o $@ $(SRC)

run: $(BIN)
	./$(BIN)

run-novec: $(BIN_NOVEC)
	./$(BIN_NOVEC)

bench-go:
	go test -bench=. -benchtime=5x -count=3 -timeout=600s .

gendata: $(BIN)
	rm -f $(DATA_I64) $(DATA_I32)
	./$(BIN)

clean:
	rm -f $(BIN) $(BIN_NOVEC) $(DATA_I64) $(DATA_I32)
