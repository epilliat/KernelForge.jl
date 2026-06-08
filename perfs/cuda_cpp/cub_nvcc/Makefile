# CUB benchmark binaries — used by perfs/julia/benchmarks/*_perf_comparison.jl
# via bench_cub_or_nan().
#
# Usage:
#   make                          # build everything
#   make cub_sort_benchmark       # one target
#   make ARCH=sm_89               # override SM (default: -arch=native)
#   make clean

NVCC   ?= nvcc
ARCH   ?= native
NVCC_FLAGS := -O3 -std=c++17 -arch=$(ARCH)
# `--extended-lambda` is needed by the Thrust binary's `[=] __device__`
# device lambdas in `thrust::transform`; harmless for the CUB targets.
NVCC_FLAGS += --extended-lambda

BINDIR := bin

BENCHMARKS := cub_sort_benchmark thrust_sort_benchmark cub_scan_benchmark cub_sum_benchmark cub_memcpy_benchmark
BINARIES   := $(addprefix $(BINDIR)/, $(BENCHMARKS))

.PHONY: all clean $(BENCHMARKS)

all: $(BINARIES)

$(BENCHMARKS): %: $(BINDIR)/%

$(BINDIR)/%: %.cu | $(BINDIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f $(BINARIES)
