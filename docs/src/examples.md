# Examples

This page provides practical examples demonstrating Luma.jl's capabilities.

## Vectorized Copy

Perform memory copies with vectorized loads and stores for improved bandwidth utilization:
```julia
using Luma
using CUDA

src = CUDA.rand(Float32, 10^6)
dst = similar(src)

# Copy with vectorized loads and stores
Luma.vcopy!(dst, src; Nitem=4)

@assert dst ≈ src
```

### Custom Types

Luma supports copying custom struct types:
```julia
using Luma
using CUDA

struct Point3D
    x::Float32
    y::Float32
    z::Float32
end

n = 100_000
src = CuArray([Point3D(rand(), rand(), rand()) for _ in 1:n])
dst = similar(src)

Luma.vcopy!(dst, src; Nitem=2)

@assert all(dst .== src)
```

### Set Values

Fill an array with a constant value:
```julia
using Luma
using CUDA

dst = CUDA.ones(UInt8, 100_000)
Luma.setvalue!(dst, 0x00; Nitem=4)

@assert all(dst .== 0x00)
```

## Map-Reduce

### Basic Reductions
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 10^6)

# Sum
total = Luma.mapreduce(identity, +, x; to_cpu=true)

# Sum of squares
sum_sq = Luma.mapreduce(abs2, +, x; to_cpu=true)

# Maximum
max_val = Luma.mapreduce(identity, max, x; to_cpu=true)

# Minimum
min_val = Luma.mapreduce(identity, min, x; to_cpu=true)
```

### Post-Reduction Transformation, Example with Luma.UnitFloat8

Apply a function after reduction using the `g` parameter:
```julia
using Luma
using CUDA

import Luma: UnitFloat8
x = CuArray{UnitFloat8}([rand(UnitFloat8) for _ in 1:10^6])

# convert to float 32 for reduction to avoid overflow
to_f32(x) = Float32(x)
# then back to unitfloat8
to_uif8(x) = UnitFloat8(x)
sum_uif8 = Luma.mapreduce(to_f32, +, x; g=to_uif8, to_cpu=true)

# compare the sign only :
@assert sign(Float32(sum_uif8)) == sign(sum(Float32.(x)))

# We can store only 8 bit, keep rather good precision (because of intermediate Float32 conversion) and get 
# same performance as for UInt8 addition !!!
```

### Dot Product and Distance

Reduce multiple arrays simultaneously:
```julia
using Luma
using CUDA

n = 100_000
a = CUDA.rand(Float32, n)
b = CUDA.rand(Float32, n)

# Dot product: sum(a .* b)
dot_prod = Luma.mapreduce1d((x, y) -> x * y, +, (a, b); to_cpu=true)

# Euclidean distance: sqrt(sum((a - b)^2))
distance = Luma.mapreduce1d((x, y) -> (x - y)^2, +, (a, b); g=sqrt, to_cpu=true)

# Weighted sum: sum(w .* x)
w = CUDA.rand(Float32, n)
x = CUDA.rand(Float32, n)
weighted_sum = Luma.mapreduce1d((wi, xi) -> wi * xi, +, (w, x); to_cpu=true)
```

### 2D Reductions

Reduce along rows or columns:
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Column sums (reduce along dim=1)
col_sums = Luma.mapreduce(identity, +, A; dims=1)
@assert size(col_sums) == (500,)

# Row sums (reduce along dim=2)
row_sums = Luma.mapreduce(identity, +, A; dims=2)
@assert size(row_sums) == (1000,)

# Column means
let u = size(A, 1) # Note that the let block is necessary for compilation of the kernel
    g(x)::Float32 = x / u
    col_means = Luma.mapreduce(identity, +, A; dims=1, g=g)
end
# Row-wise sum of squares
row_ss = Luma.mapreduce(abs2, +, A; dims=2)

# Row maximums
row_max = Luma.mapreduce(identity, max, A; dims=2)
```

### Higher-Dimensional Reductions
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 20, 30, 40)


# !!! For now, we provide support only for contiguous dim at start or end, so we cannot reduce on second dim here
try
    Luma.mapreduce(identity, +, A; dims=2)
catch e
    println(e)
end

# Reduce first dimension: (20, 30, 40) → (30, 40)
result = Luma.mapreduce(identity, +, A; dims=1)
@assert size(result) == (30, 40)

# Reduce first two dimensions: (20, 30, 40) → (40,)
result = Luma.mapreduce(identity, +, A; dims=(1, 2))
@assert size(result) == (40,)

# Reduce last dimension: (20, 30, 40) → (20, 30)
result = Luma.mapreduce(identity, +, A; dims=3)
@assert size(result) == (20, 30)

# Reduce last two dimensions: (20, 30, 40) → (20,)
result = Luma.mapreduce(identity, +, A; dims=(2, 3))
@assert size(result) == (20,)
```

### Custom Structs
```julia
using Luma
using CUDA

struct Stats
    sum::Float32
    sum_sq::Float32
    count::Float32
end

# Map: convert each value to Stats
f(x) = Stats(x, x^2, 1f0)

# Reduce: combine Stats
op(a::Stats, b::Stats) = Stats(a.sum + b.sum, a.sum_sq + b.sum_sq, a.count + b.count)

x = CUDA.rand(Float32, 100_000)
result = Luma.mapreduce(f, op, x; to_cpu=true)

mean = result.sum / result.count
variance = result.sum_sq / result.count - mean^2
```

## Prefix Scan

### Basic Scans
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 10^6)
dst = similar(x)

# Cumulative sum
Luma.scan!(+, dst, x)
@assert dst ≈ CuArray(accumulate(+, Array(x)))

# Cumulative product
Luma.scan!(*, dst, x)
@assert dst ≈ CuArray(accumulate(*, Array(x)))

# Cumulative maximum
Luma.scan!(max, dst, x)
@assert dst ≈ CuArray(accumulate(max, Array(x)))
```

### Scan with Map Function
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 10_000)

# Cumulative sum of squares
result = Luma.scan(abs2, +, x)
@assert result ≈ CuArray(accumulate(+, Array(x).^2))

# Cumulative sum of absolute values
result = Luma.scan(abs, +, x)
```

### Non-Commutative Operations: Quaternions

Luma scan function (not mapreduce) correctly handles non-commutative operations:
```julia
using Luma
using CUDA
using Quaternions

n = 100_000

# Generate random unit quaternions
src_cpu = [QuaternionF64((x ./ sqrt(sum(x.^2)))...) for x in eachcol(randn(4, n))]
src = CuArray(src_cpu)
dst = similar(src)

# Quaternion multiplication is non-commutative: q1 * q2 ≠ q2 * q1
Luma.scan!(*, dst, src)

@assert dst ≈ CuArray(accumulate(*, src_cpu))
```

## Matrix-Vector Operations

### Standard Matrix-Vector Multiply
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# y = A * x
y = Luma.matvec(A, x)
@assert y ≈ A * x

# In-place version
dst = CUDA.zeros(Float32, 1000)
Luma.matvec!(dst, A, x)
@assert dst ≈ A * x
```

### Row-wise Reductions
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Row sums: y[i] = sum(A[i, :])
row_sums = Luma.matvec(A, nothing)
@assert row_sums ≈ vec(sum(A; dims=2))

# Row maximums: y[i] = max_j(A[i, j])
row_max = Luma.matvec(identity, max, A, nothing)
@assert row_max ≈ vec(maximum(A; dims=2))

# Row minimums
row_min = Luma.matvec(identity, min, A, nothing)
@assert row_min ≈ vec(minimum(A; dims=2))
```

### Custom Operations
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# Softmax numerator: y[i] = sum_j(exp(A[i,j] - x[j]))
y = Luma.matvec((a, b) -> exp(a - b), +, A, x)

# Row-wise dot product with scaling: y[i] = sqrt(sum_j(A[i,j]^2 * x[j]^2))
y = Luma.matvec((a, b) -> a^2 * b^2, +, A, x; g=sqrt)
```

### Custom Struct Output
```julia
using Luma
using CUDA

struct Vec3
    x::Float32
    y::Float32
    z::Float32
end

# Map: combine matrix and vector elements
f(a, b) = Vec3(a * b, a + b, a - b)

# Reduce: component-wise sum
op(v1::Vec3, v2::Vec3) = Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

A = CUDA.rand(Float32, 200, 500)
x = CUDA.rand(Float32, 500)
dst = CuArray{Vec3}(undef, 200)

Luma.matvec!(f, op, dst, A, x)
```

## Vector-Matrix Operations

### Standard Vector-Matrix Multiply
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 1000)

# y = x' * A (column-wise weighted sum)
dst = CUDA.zeros(Float32, 500)
Luma.vecmat!(dst, x, A)
@assert dst ≈ vec(x' * A)
```

### Column-wise Reductions
```julia
using Luma
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Column sums: y[j] = sum(A[:, j])
dst = CUDA.zeros(Float32, 500)
Luma.vecmat!(dst, nothing, A)
@assert dst ≈ vec(sum(A; dims=1))
```

## Pre-allocating Temporary Buffers

For repeated operations, pre-allocate temporary buffers to avoid allocation overhead:
```julia
using Luma
using CUDA

x = CUDA.rand(Float32, 100_000)
dst = similar(x)

# Pre-allocate for scan
tmp = Luma.get_allocation(Luma.scan!, dst, x)

# Reuse in a loop
for i in 1:100
    CUDA.rand!(x)  # new random data
    Luma.scan!(+, dst, x; tmp=tmp)
end
```


## Complete Example: Online Statistics

Compute running mean and variance in a single pass:
```julia
using Luma
using CUDA

struct RunningStats
    n::Float32
    mean::Float32
    m2::Float32  # sum of squared deviations
end

# Welford's online algorithm for combining statistics
function combine(a::RunningStats, b::RunningStats)
    n = a.n + b.n
    delta = b.mean - a.mean
    mean = a.mean + delta * b.n / n
    m2 = a.m2 + b.m2 + delta^2 * a.n * b.n / n
    return RunningStats(n, mean, m2)
end

# Initialize each element as a single observation
init(x) = RunningStats(1f0, x, 0f0)

x = CUDA.rand(Float32, 1_000_000)
dst = CuArray{RunningStats}(undef, length(x))

Luma.scan!(init, combine, dst, x)

# Final statistics
final_stats = Array(dst)[end]
mean = final_stats.mean
variance = final_stats.m2 / final_stats.n
std = sqrt(variance)
```