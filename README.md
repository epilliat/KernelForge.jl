# Luma.jl

High-performance, portable GPU primitives for Julia (currently CUDA-only). Pure Julia implementation with performance competitive against optimized CUDA C++ libraries.

## Installation

```julia
using Pkg
Pkg.add("Luma")
```

## Features

- **Vectorized copy** with configurable load/store widths
- **Map-reduce** with custom functions and operators
- **Prefix scan** supporting non-commutative operations
- Currently only for 1D arrays, with plans for multi-dimensional support
- Currently CUDA-only; cross-platform support via KernelAbstractions.jl planned
- Includes `UnitFloat8`, a custom 8-bit floating-point type with range (-1, 1) for testing


## Performances

Luma.jl achieves performance comparable to optimized CUDA C++ libraries such as CUB and Thrust.

- **Blue**: Main kernel execution time, measured using the `@profile` macro from CUDA.jl
- **Other colors**: Auxiliary kernel execution times
- **Gray**: Overhead (total time minus kernel times), including memory allocations, data transfers, and other operations

### Copy Performance

Cub optimizes copy for large input sizes (byte size of dst + src $\geq$ L2 cache size), while KernelAbstractions.jl performs better on smaller arrays. Our vcopy! function can be tuned via parameters to achieve high performance across all input sizes. Automatic adaptation based on L2 cache size could be planned in future releases.

![](perfs/cuda/figures/combined_plot_copy.png)

### Map-Reduce Performance

For map-reduce, Luma.jl matches CUDA.jl performance on Float32 (slightly better with UInt64 flags, yielding success probability $> 1 - 2^{-64}$), and significantly outperforms it on smaller types (UInt8, UnitFloat8) even with Float32 conversion. This speedup stems from optimized memory access patterns and vectorized loads/stores.


![](perfs/cuda/figures/mapreduce_nof64_1e6_1e8.png)

### Scan Performance

Our scan rivals CUB performance on Float32 and Float64, while also supporting non-commutative operations and custom types like Quaternions—enabled by an efficient decoupled lookback algorithm and optimized memory access.

![](perfs/cuda/figures/combined_plot_scan.png)


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

src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar([0.])

f(x) = x^2
op(x) = +(x...)

Luma.mapreduce!(f, op, dst, src)

isapprox(Array(dst), [mapreduce(f, op, Array(src))])  # true
```

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

src = CuArray{Float64}(rand(Float32, 10^6))
dst = similar(src)

op(x, y) = x + y
op(x...) = op(x[1], op(x[2:end]...))

Luma.scan!(op, dst, src)

# Matches Base.accumulate:
isapprox(Array(dst), accumulate(+, Array(src)))  # true
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
dst = CuArray{QuaternionF64}([0 for _ in 1:n])

Luma.scan!(op, dst, src)

# Works with non-commutative structures!
isapprox(Array(dst), accumulate(op, src_cpu))  # true
```

## API Reference

| Function | Description |
|----------|-------------|
| `vcopy!(dst, src; Nitem)` | Vectorized memory copy |
| `mapreduce!(f, op, dst, src)` | Map-reduce with custom function and operator |
| `scan!(op, dst, src)` | Inclusive prefix scan |

## License

MIT