# Luma.jl

High-performance, portable GPU primitives for Julia. A pure Julia implementation delivering performance competitive with optimized CUDA C++ libraries.

> ⚠️ **Experimental Status**
> 
> This package is in an experimental phase. Although extensive testing has been performed, the current implementation does not support views or strided arrays. No bounds checking is performed, which may lead to unexpected behavior with non-contiguous data. Correctness and performance have been validated only on a small NVIDIA RTX 1000.

> ℹ️ **Architecture & Contributions**
> 
> Luma.jl builds on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for GPU kernel dispatch. However, certain low-level operations—including warp shuffle instructions, vectorized memory access, and memory ordering semantics—are not yet available in KA.jl, so we use [KernelIntrinsics.jl](https://github.com/...) for these primitives. As KernelIntrinsics.jl currently supports only CUDA, Luma.jl is likewise restricted to CUDA.
> 
> **The core contribution of this package lies in the GPU kernel implementations themselves**, designed to be portable once the underlying intrinsics become available on other backends. Extending support to AMD and Intel GPUs would primarily require work in KernelIntrinsics.jl, with minimal adaptations in Luma.jl.

> 📄 A paper describing this work is in preparation. If you use this code, please check back for citation details.

## Acknowledgments

This package builds on the foundation provided by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). The API design draws inspiration from several packages in the Julia ecosystem. Development of the API and documentation was assisted by [Claude](https://claude.ai) (Anthropic).

## Installation
```julia
using Pkg
Pkg.add("Luma")
```

## Features

- **Matrix-vector operations** with customizable element-wise and reduction operations
- **Prefix scan** supporting non-commutative operations
- **Map-reduce** with custom functions and operators, supporting 1D and 2D reductions
- **Vectorized copy** with configurable load/store widths
- Currently only for 1D and 2D arrays
- Currently CUDA-only; cross-platform support via KernelAbstractions.jl planned
- Includes `UnitFloat8`, a custom 8-bit floating-point type with range (-1, 1) for testing

## Examples

### Vectorized Copy

Perform memory copies with vectorized loads and stores for improved bandwidth utilization:
```julia
using Luma
using CUDA

src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar(src)

# Copy with vectorized loads and stores (4 elements per thread)
vcopy!(dst, src, Nitem=4)

isapprox(dst, src)  # true
```

### Map-Reduce

Apply a transformation and reduce the result with a custom operator:
```julia
using Luma
using CUDA

src = CUDA.rand(Float32, 10^6)

# Full reduction
total = Luma.mapreduce(identity, +, src; to_cpu=true)

# With custom map function
sum_of_squares = Luma.mapreduce(abs2, +, src; to_cpu=true)
```

#### Multi-dimensional Reductions

Luma supports reductions along specified dimensions for 2D and higher-dimensional arrays:
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Column sums (reduce along dim=1)
col_sums = Luma.mapreduce(identity, +, A; dims=1)

# Row maximums (reduce along dim=2)
row_maxs = Luma.mapreduce(identity, max, A; dims=2)

# Column means with post-reduction transformation
col_means = Luma.mapreduce(identity, +, A; dims=1, g=x -> x / size(A, 1))

# Sum of squares per row
row_ss = Luma.mapreduce(abs2, +, A; dims=2)
```

For higher-dimensional arrays, the `dims` argument must specify contiguous dimensions from either the beginning (e.g., `(1,)`, `(1,2)`) or the end (e.g., `(n,)`, `(n-1,n)`).

#### Custom Types: UnitFloat8

Luma supports custom numeric types. Here's an example with `UnitFloat8`:
```julia
using Luma: UnitFloat8
using CUDA

n = 1000000
f(x::UnitFloat8) = Float32(x)

src = CuArray{UnitFloat8}([rand(UnitFloat8) for _ in 1:n])
dst = CuArray{UnitFloat8}([0])

Luma.mapreduce!(f, +, dst, src)

# dst is in (-1,1) range due to UnitFloat8 overflow, BUT since we reduce
# Float32 values, the result has the correct sign:
sign(Float32(CUDA.@allowscalar dst[1])) == sign(mapreduce(f, +, Array(Float32.(src))))  # true
```

### Prefix Scan

Compute cumulative operations with support for non-commutative operators:
```julia
using Luma
using CUDA

src = CUDA.rand(Float32, 10^6)
dst = similar(src)

Luma.scan!(+, dst, src)

# Matches Base.accumulate:
isapprox(Array(dst), accumulate(+, Array(src)))  # true
```

#### Cumulative Sum of Squares
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 10_000)
result = Luma.scan(x -> x^2, +, x)
```

#### Non-Commutative Types: Quaternions

Luma correctly handles non-commutative operations without requiring a neutral element or init value:
```julia
using Luma
using CUDA
using Quaternions

n = 1000000
op(x::QuaternionF64...) = *(x...)

# Generate unit quaternions
src_cpu = [QuaternionF64(x ./ sqrt(sum(x .^ 2))...) for x in eachcol(randn(4, n))]
src = CuArray{QuaternionF64}(src_cpu)
dst = similar(src)

Luma.scan!(op, dst, src)

# Works with non-commutative structures!
isapprox(Array(dst), accumulate(op, src_cpu))  # true
```

### Matrix-Vector Operations

Generalized matrix-vector multiplication with customizable element-wise and reduction operations:
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# Standard matrix-vector multiply: y = A * x
y = Luma.matvec(A, x)

# Row-wise sum: y[i] = sum(A[i, :])
y = Luma.matvec(A, nothing)

# Row-wise maximum: y[i] = max_j(A[i, j])
y = Luma.matvec(identity, max, A, nothing)

# Softmax numerator: y[i] = sum_j(exp(A[i,j] - x[j]))
y = Luma.matvec((a, b) -> exp(a - b), +, A, x)

# In-place version
dst = CUDA.zeros(Float32, 1000)
Luma.matvec!(dst, A, x)
```

For tall matrices (many rows, few columns), each row is processed by a single block. For wide matrices (few rows, many columns), multiple blocks collaborate on each row.

### Vector-Matrix Operations

Column-wise reductions and vector-matrix products:
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 1000)
dst = CUDA.zeros(Float32, 500)

# Vector-matrix multiply: dst[j] = sum_i(x[i] * A[i,j])
Luma.vecmat!(dst, x, A)

# Column sums (x = nothing): dst[j] = sum_i(A[i,j])
Luma.vecmat!(dst, nothing, A)
```

### Pre-allocating Temporary Buffers

For repeated operations, pre-allocate temporary buffers to avoid allocation overhead:
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 10_000)
dst = similar(x)

# Pre-allocate for scan
tmp = Luma.get_allocation(Luma.scan!, dst, x)

for i in 1:100
    Luma.scan!(+, dst, x; tmp)
end
```

### Synchronization Flags

By default, Luma uses `UInt8` flags which require zeroing before each call. For higher throughput without zeroing overhead, use `UInt64` flags with random target generation (correctness probability > 1 − 2⁻⁶⁴):
```julia
Luma.scan!(+, dst, src; FlagType=UInt64)
```

## Performance

Luma.jl achieves performance comparable to optimized CUDA C++ libraries such as CUB. Benchmarks report two metrics:

- **Kernel time**: Execution time of the main kernel, measured using `@profile` from CUDA.jl
- **Overhead**: Total time minus kernel time, including memory allocations and data transfers

### Copy Performance

CUDA.jl leverages the proprietary libcuda library for memory copies, which internally vectorizes loads and stores. In contrast, the cross-platform GPUArrayCore.jl relies on KernelAbstractions.jl, which does not currently perform vectorization. Luma's `vcopy!` bridges this gap by using `vload` and `vstore` operations built on unsafe pointer access via LLVMPtrs from KernelIntrinsics.jl.

The graph below compares memory bandwidth for Float32 and UInt8 data types. With vectorized loads and stores, Luma achieves bandwidth comparable to CUDA.jl for both types. The slight underperformance below the L2 cache threshold stems from our current vectorization factor (×8 for Float32); increasing this to ×16 would close the remaining gap.

<p align="center">
  <img src="perfs/cuda/figures/benchmark/copy_bandwidth.png" width="50%">
</p>

### Map-Reduce Performance

Luma.jl matches CUDA.jl performance on Float32 and significantly outperforms it on smaller types (UInt8, UnitFloat8), even when converting to Float32 during reduction. These gains result from optimized memory access patterns and vectorized loads/stores.

<p align="center">
  <img src="perfs/cuda/figures/benchmark/mapreduce_benchmark_comparison.png" width="50%">
</p>

### Scan Performance

Luma's scan kernel rivals CUB performance on Float32 and Float64, while additionally supporting non-commutative operations and custom types such as Quaternions. This is achieved through an efficient decoupled lookback algorithm combined with optimized memory access.

<p align="center">
  <img src="perfs/cuda/figures/benchmark/scan_benchmark_comparison.png" width="50%">
</p>

### Matrix-Vector Operations

Luma implements matrix-vector and vector-matrix operations for general types and operators. For benchmarking, we compare against CUDA.jl on Float32, which internally calls cuBLAS's `gemv` routine.

Due to column-major memory layout, matrix-vector and vector-matrix multiplications have fundamentally different access patterns. Luma therefore provides separate optimized kernels for each operation.

For both benchmarks, we fix the total matrix size (n × p) and vary n from 10 to (n × p) / 10, sweeping from tall-narrow to short-wide matrices. The black line indicates the reduced overhead achieved when the user provides pre-allocated temporary memory.

<p align="center">
  <b>Matrix-Vector</b><br>
  <img src="perfs/cuda/figures/benchmark/matvec_benchmark_comparison.png" width="50%">
</p>

<p align="center">
  <b>Vector-Matrix</b><br>
  <img src="perfs/cuda/figures/benchmark/vecmat_benchmark_comparison.png" width="50%">
</p>


## API Reference

| Function | Description |
|----------|-------------|
| `vcopy!(dst, src; Nitem)` | Vectorized memory copy |
| `mapreduce(f, op, src; dims, g, to_cpu)` | Map-reduce with optional dimension reduction |
| `mapreduce!(f, op, dst, src)` | In-place map-reduce |
| `mapreduce2d(f, op, src, dim; g)` | 2D reduction along specified dimension |
| `mapreduce2d!(f, op, dst, src, dim; g)` | In-place 2D reduction |
| `scan(op, src)` / `scan(f, op, src)` | Allocating prefix scan |
| `scan!(op, dst, src)` / `scan!(f, op, dst, src)` | In-place prefix scan |
| `matvec(A, x)` / `matvec(f, op, A, x; g)` | Generalized matrix-vector product |
| `matvec!(dst, A, x)` / `matvec!(f, op, dst, A, x; g)` | In-place matrix-vector product |
| `vecmat!(dst, x, A)` / `vecmat!(f, op, dst, x, A; g)` | Vector-matrix product / column reduction |
| `get_allocation(func!, dst, src)` | Pre-allocate temporary buffer for repeated calls |

## License

MIT