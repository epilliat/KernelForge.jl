# Performance

KernelForge.jl achieves performance comparable to optimized CUDA C++ libraries such as CUB. Benchmarks report two metrics:

- **Kernel time**: Execution time of the main kernel, measured using `@profile` from CUDA.jl
- **Overhead**: Total time minus kernel time, including memory allocations and data transfers

## Copy

CUDA.jl leverages the proprietary libcuda library for memory copies, which internally vectorizes loads and stores. In contrast, the cross-platform GPUArrayCore.jl relies on KernelAbstractions.jl, which does not currently perform vectorization. KernelForge's `vcopy!` bridges this gap by using `vload` and `vstore` operations built on unsafe pointer access via LLVMPtrs from KernelIntrinsics.jl.

The graph below compares memory bandwidth for Float32 and UInt8 data types. With vectorized loads and stores, KernelForge achieves bandwidth comparable to CUDA.jl for both types. The slight underperformance below the L2 cache threshold stems from our current vectorization factor (×8 for Float32); increasing this to ×16 would close the remaining gap.

![Copy Bandwidth](assets/copy_bandwidth.png)

## Map-Reduce

KernelForge.jl matches CUDA.jl performance on Float32 and significantly outperforms it on smaller types (UInt8, UnitFloat8), even when converting to Float32 during reduction. These gains result from optimized memory access patterns and vectorized loads/stores.

![Map-Reduce Benchmark](assets/mapreduce_benchmark_comparison.png)

## Scan

KernelForge's scan kernel rivals CUB performance on Float32 and Float64, while additionally supporting non-commutative operations and custom types such as Quaternions. This is achieved through an efficient decoupled lookback algorithm combined with optimized memory access.

![Scan Benchmark](assets/scan_benchmark_comparison.png)

## Matrix-Vector Operations

KernelForge implements matrix-vector and vector-matrix operations for general types and operators. For benchmarking, we compare against CUDA.jl on Float32, which internally calls cuBLAS's `gemv` routine.

Due to column-major memory layout, matrix-vector and vector-matrix multiplications have fundamentally different access patterns. KernelForge therefore provides separate optimized kernels for each operation.

For both benchmarks, we fix the total matrix size (n × p) and vary n from 10 to (n × p) / 10, sweeping from tall-narrow to short-wide matrices. The black line indicates the reduced overhead achieved when the user provides pre-allocated temporary memory.

**Matrix-Vector**
![Matrix-Vector Benchmark](assets/matvec_benchmark_comparison.png)

**Vector-Matrix**

![Vector-Matrix Benchmark](assets/vecmat_benchmark_comparison.png)

## Sort

KernelForge ships a 1D radix `sort!` competitive with CUB's
`DeviceRadixSort` across the standard bitwise types (UInt32, UInt64,
Float32, Float64). At large N (≥ 10⁷) KernelForge typically wins on
narrow types and ties on wide types; at small N (< 10⁵) CUB pulls
ahead because its launch overhead amortizes better. The `keys=...`
keyval variant, `sortperm!`, and the batched `sort_columns!` family
all reuse the same one-sweep radix kernel.

For workloads with custom comparators or non-radix bitstypes,
`sample_sort` provides a general path; it currently matches
CUDA.jl's bitonic sort on Float32 / Float64 at N ~ 10⁷ and remains
correct (where bitonic doesn't) for arbitrary `lt` predicates.

Benchmark scripts: `perfs/julia/benchmarks/sort_perf_comparison.jl`
and `perfs/julia/benchmarks/sort_columns_perf_comparison.jl`.
Raw result CSVs live in the sibling repo
[KernelForge-benchmarks](https://github.com/epilliat/KernelForge-benchmarks);
plot figures will land in a follow-up minor.

## Random

The Philox4x32-10 RNG path matches Random123's CPU implementation
bit-for-bit and outperforms CUDA.jl's cuRAND host-only launcher at
small N because the Philox stream is fully GPU-resident — no
host↔device round-trip. At large N (≥ 10⁷), Float32 Uniform draws
saturate device memory bandwidth on RTX1000.

`randperm!` builds on the keyval sort path (fill Float32 uniform
keys + `sortperm!`); throughput tracks `sort!` directly.

Benchmark scripts: `perfs/julia/benchmarks/random_perf_comparison.jl`
and `perfs/julia/benchmarks/randperm_perf_comparison.jl`. Raw CSVs in
[KernelForge-benchmarks](https://github.com/epilliat/KernelForge-benchmarks).