# rocPRIM (hipCUB) benchmark binaries — AMD analogue of perfs/cuda_cpp/cub_nvcc/Makefile.
# Used by perfs/julia/benchmarks/{scan,mapreduce}_perf_comparison.jl via
# bench_rocprim_or_nan(). hipCUB's DeviceReduce/DeviceScan dispatch to rocPRIM.
#
# Usage:
#   make                          # build everything
#   make rocprim_scan_benchmark   # one target
#   make AMDARCH=gfx90a           # override GPU arch (default: gfx942 = MI300X/CDNA3)
#   make clean

HIPCC   ?= hipcc
AMDARCH ?= gfx942
HIPCC_FLAGS := -O3 -std=c++17 --offload-arch=$(AMDARCH)

BINDIR := bin

BENCHMARKS := rocprim_scan_benchmark rocprim_sum_benchmark rocprim_sort_benchmark
BINARIES   := $(addprefix $(BINDIR)/, $(BENCHMARKS))

.PHONY: all clean $(BENCHMARKS)

all: $(BINARIES)

$(BENCHMARKS): %: $(BINDIR)/%

$(BINDIR)/%: %.cpp | $(BINDIR)
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f $(BINARIES)
