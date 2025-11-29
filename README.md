# Luma.jl

High-performance GPU computing primitives for Julia, providing portable implementations of common parallel algorithms.

## Installation

```julia
using Pkg
Pkg.add("Luma")
```

## Features

- **Vectorized memory operations** with configurable load/store widths
- **Map-reduce** with custom functions and operators
- **Prefix scan** supporting non-commutative operations
- Cross-platform GPU support via KernelAbstractions.jl

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