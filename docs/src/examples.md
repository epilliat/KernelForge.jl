# Examples

This page provides practical examples demonstrating KernelForge.jl's capabilities.

## Vectorized Copy

Perform memory copies with vectorized loads and stores for improved bandwidth utilization:
```julia
using KernelForge
using CUDA

src = CUDA.rand(Float32, 10^6)
dst = similar(src)

# Copy with vectorized loads and stores
KernelForge.vcopy!(dst, src; Nitem=4)

@assert dst ≈ src
```

### Custom Types

KernelForge supports copying custom struct types:
```julia
using KernelForge
using CUDA

struct Point3D
    x::Float32
    y::Float32
    z::Float32
end

n = 100_000
src = CuArray([Point3D(rand(), rand(), rand()) for _ in 1:n])
dst = similar(src)

KernelForge.vcopy!(dst, src; Nitem=2)

@assert all(dst .== src)
```

### Set Values

Fill an array with a constant value:
```julia
using KernelForge
using CUDA

dst = CUDA.ones(UInt8, 100_000)
KernelForge.setvalue!(dst, 0x00; Nitem=4)

@assert all(dst .== 0x00)
```

## Map-Reduce

### Basic Reductions
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 10^6)

# Sum
total = KernelForge.mapreduce(identity, +, x)

# Sum of squares
sum_sq = KernelForge.mapreduce(abs2, +, x)

# Maximum
max_val = KernelForge.mapreduce(identity, max, x)

# Minimum
min_val = KernelForge.mapreduce(identity, min, x)
```

### Post-Reduction Transformation, Example with KernelForge.UnitFloat8

Apply a function after reduction using the `g` parameter:
```julia
using KernelForge
using CUDA

import KernelForge: UnitFloat8
x = CuArray{UnitFloat8}([rand(UnitFloat8) for _ in 1:10^6])

# convert to float 32 for reduction to avoid overflow
to_f32(x) = Float32(x)
# then back to unitfloat8
to_uif8(x) = UnitFloat8(x)
sum_uif8 = KernelForge.mapreduce(to_f32, +, x; g=to_uif8)

# compare the sign only :
@assert sign(Float32(sum_uif8)) == sign(sum(Float32.(x)))

# We can store only 8 bit, keep rather good precision (because of intermediate Float32 conversion) and get 
# same performance as for UInt8 addition !!!
```

### Dot Product and Distance

Reduce multiple arrays simultaneously:
```julia
using KernelForge
using CUDA

n = 100_000
a = CUDA.rand(Float32, n)
b = CUDA.rand(Float32, n)

# Dot product: sum(a .* b)
dot_prod = KernelForge.mapreduce1d((x, y) -> x * y, +, (a, b))

# Euclidean distance: sqrt(sum((a - b)^2))
distance = KernelForge.mapreduce1d((x, y) -> (x - y)^2, +, (a, b); g=sqrt)

# Weighted sum: sum(w .* x)
w = CUDA.rand(Float32, n)
x = CUDA.rand(Float32, n)
weighted_sum = KernelForge.mapreduce1d((wi, xi) -> wi * xi, +, (w, x))
```

### 2D Reductions

Reduce along rows or columns:
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Column sums (reduce along dim=1)
col_sums = KernelForge.mapreduce(identity, +, A; dims=1)
@assert size(col_sums) == (500,)

# Row sums (reduce along dim=2)
row_sums = KernelForge.mapreduce(identity, +, A; dims=2)
@assert size(row_sums) == (1000,)

# Column means
let u = size(A, 1) # Note that the let block is necessary for compilation of the kernel
    g(x)::Float32 = x / u
    col_means = KernelForge.mapreduce(identity, +, A; dims=1, g=g)
end
# Row-wise sum of squares
row_ss = KernelForge.mapreduce(abs2, +, A; dims=2)

# Row maximums
row_max = KernelForge.mapreduce(identity, max, A; dims=2)
```

### Higher-Dimensional Reductions
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 20, 30, 40)

# Reduce first dimension: (20, 30, 40) → (1, 30, 40)
result = KernelForge.mapreduce(identity, +, A; dims=1)
@assert size(result) == (1, 30, 40)

# Reduce first two dimensions: (20, 30, 40) → (1, 1, 40)
result = KernelForge.mapreduce(identity, +, A; dims=(1, 2))
@assert size(result) == (1, 1, 40)

# Reduce last dimension: (20, 30, 40) → (20, 30, 1)
result = KernelForge.mapreduce(identity, +, A; dims=3)
@assert size(result) == (20, 30, 1)

# Reduce last two dimensions: (20, 30, 40) → (20, 1, 1)
result = KernelForge.mapreduce(identity, +, A; dims=(2, 3))
@assert size(result) == (20, 1, 1)

# General reduction on non-contiguous dims: (3, 4, 5, 6, 7) → (3, 1, 5, 1, 1)
B = CUDA.rand(Float32, 3, 4, 5, 6, 7)
result = KernelForge.mapreduce(identity, +, B; dims=(2, 4, 5))
@assert size(result) == (3, 1, 5, 1, 1)
```

### Custom Structs
```julia
using KernelForge
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
result = KernelForge.mapreduce(f, op, x)

mean = result.sum / result.count
variance = result.sum_sq / result.count - mean^2
```

### Operations on Views

All KernelForge functions work transparently on GPU array views, with no copies:
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 100, 200)

# Reduce over a column slice
col_slice = view(A, :, 50:150)
result = KernelForge.mapreduce(identity, +, col_slice; dims=1)
@assert size(result) == (1, 101)

# Full reduction on a row slice
row_slice = view(A, 10:90, :)
total = KernelForge.mapreduce(identity, +, row_slice)
@assert total ≈ sum(Array(A)[10:90, :])
```

## Prefix Scan

### Basic Scans
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 10^6)
dst = similar(x)

# Cumulative sum
KernelForge.scan!(+, dst, x)
@assert dst ≈ CuArray(accumulate(+, Array(x)))

# Cumulative product
KernelForge.scan!(*, dst, x)
@assert dst ≈ CuArray(accumulate(*, Array(x)))

# Cumulative maximum
KernelForge.scan!(max, dst, x)
@assert dst ≈ CuArray(accumulate(max, Array(x)))
```

### Scan with Map Function
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 10_000)

# Cumulative sum of squares
result = KernelForge.scan(abs2, +, x)
@assert result ≈ CuArray(accumulate(+, Array(x).^2))

# Cumulative sum of absolute values
result = KernelForge.scan(abs, +, x)
```

### Scan on Views
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 1_000)
dst = CUDA.zeros(Float32, 1_000)

# Scan over a subrange, writing into a subrange of dst
KernelForge.scan!(+, view(dst, 3:800), view(x, 3:800))

@assert Array(dst)[3:800] ≈ accumulate(+, Array(x)[3:800])
# Elements outside the view are untouched
@assert all(Array(dst)[1:2] .== 0f0)
```

### Non-Commutative Operations: Quaternions

KernelForge scan function (not mapreduce) correctly handles non-commutative operations:
```julia
using KernelForge
using CUDA
using Quaternions

n = 100_000

# Generate random unit quaternions
src_cpu = [QuaternionF64((x ./ sqrt(sum(x.^2)))...) for x in eachcol(randn(4, n))]
src = CuArray(src_cpu)
dst = similar(src)

# Quaternion multiplication is non-commutative: q1 * q2 ≠ q2 * q1
KernelForge.scan!(*, dst, src)

@assert dst ≈ CuArray(accumulate(*, src_cpu))
```

## Matrix-Vector Operations

### Standard Matrix-Vector Multiply
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# y = A * x
y = KernelForge.matvec(A, x)
@assert y ≈ A * x

# In-place version
dst = CUDA.zeros(Float32, 1000)
KernelForge.matvec!(dst, A, x)
@assert dst ≈ A * x
```

### Row-wise Reductions
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Row sums: y[i] = sum(A[i, :])
row_sums = KernelForge.matvec(A, nothing)
@assert row_sums ≈ vec(sum(A; dims=2))

# Row maximums: y[i] = max_j(A[i, j])
row_max = KernelForge.matvec(identity, max, A, nothing)
@assert row_max ≈ vec(maximum(A; dims=2))

# Row minimums
row_min = KernelForge.matvec(identity, min, A, nothing)
@assert row_min ≈ vec(minimum(A; dims=2))
```

### Custom Operations
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# Softmax numerator: y[i] = sum_j(exp(A[i,j] - x[j]))
y = KernelForge.matvec((a, b) -> exp(a - b), +, A, x)

# Row-wise dot product with scaling: y[i] = sqrt(sum_j(A[i,j]^2 * x[j]^2))
y = KernelForge.matvec((a, b) -> a^2 * b^2, +, A, x; g=sqrt)
```

### Custom Struct Output
```julia
using KernelForge
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

KernelForge.matvec!(f, op, dst, A, x)
```

### Matvec on Views
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 100, 200)
x = CUDA.rand(Float32, 200)

# Multiply using only a submatrix and a subvector
result = KernelForge.matvec(view(A, 10:90, 50:150), view(x, 50:150))
@assert Array(result) ≈ Array(A)[10:90, 50:150] * Array(x)[50:150]
```

## Vector-Matrix Operations

### Standard Vector-Matrix Multiply
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 1000)

# y = x' * A (column-wise weighted sum)
dst = CUDA.zeros(Float32, 500)
KernelForge.vecmat!(dst, x, A)
@assert dst ≈ vec(x' * A)
```

### Column-wise Reductions
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 1000, 500)

# Column sums: y[j] = sum(A[:, j])
dst = CUDA.zeros(Float32, 500)
KernelForge.vecmat!(dst, nothing, A)
@assert dst ≈ vec(sum(A; dims=1))
```

## Search

### findfirst

Find the index of the first element satisfying a predicate:
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 100_000)

# First element above threshold
i = KernelForge.findfirst(>(0.999f0), x)
# i is either an Int index or nothing

# First exact match
CUDA.@allowscalar x[42] = -1f0
i = KernelForge.findfirst(==(-1f0), x)
@assert i == 42
```

### findlast
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 100_000)
CUDA.@allowscalar x[99_000] = -1f0
i = KernelForge.findlast(==(-1f0), x)
@assert i == 99_000
```

### findfirst on a 2D array

For multidimensional arrays, a `CartesianIndex` is returned:
```julia
using KernelForge
using CUDA

A = CUDA.rand(Float32, 100, 200)
CUDA.@allowscalar A[37, 82] = -1f0

idx = KernelForge.findfirst(==(-1f0), A)
@assert idx == CartesianIndex(37, 82)
```

### findfirst on custom structs
```julia
using KernelForge
using CUDA

struct Particle
    mass::Float32
    charge::Float32
end

particles = CuArray([Particle(rand(), rand() - 0.5f0) for _ in 1:10_000])

# First negatively charged particle
i = KernelForge.findfirst(p -> p.charge < 0f0, particles)
```

## argmax / argmin

### Basic usage

`argmax` and `argmin` return the **index** of the extremal element:
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 100_000)

i = KernelForge.argmax(x)   # index of maximum
j = KernelForge.argmin(x)   # index of minimum

@assert Array(x)[i] == maximum(Array(x))
@assert Array(x)[j] == minimum(Array(x))
```

### Custom relation

Pass any binary relation to find the extremum under a custom ordering:
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 100_000)

# Equivalent to argmax
i = KernelForge.argmax(>, x)

# Equivalent to argmin
j = KernelForge.argmax(<, x)
```

### argmax with custom structs

Use `argmax1d` directly with a map function `f` and a relation `rel` for full control:
```julia
using KernelForge
using CUDA

struct PriorityItem
    priority::Int32
    cost::Float32
end

# "better" means higher priority; ties broken by lower cost
better(a::PriorityItem, b::PriorityItem) =
    a.priority > b.priority || (a.priority == b.priority && a.cost < b.cost)

items = CuArray([PriorityItem(rand(Int32(1):Int32(5)), rand(Float32)) for _ in 1:10_000])

# Index of the best item
i = KernelForge.argmax(better, items)
```

### Tie-breaking

When multiple elements share the extremal value, the **first** index is always returned:
```julia
using KernelForge
using CUDA

x = CuArray(Float32[5, 5, 5, 5, 5])
@assert KernelForge.argmax(x) == 1
@assert KernelForge.argmin(x) == 1
```

## Pre-allocating Temporary Buffers

For repeated operations, pre-allocate temporary buffers to avoid allocation overhead:
```julia
using KernelForge
using CUDA

x = CUDA.rand(Float32, 100_000)
dst = similar(x)

# Pre-allocate for scan
tmp = KernelForge.get_allocation(KernelForge.scan!, dst, x)

# Reuse in a loop
for i in 1:100
    CUDA.rand!(x)  # new random data
    KernelForge.scan!(+, dst, x; tmp=tmp)
end
```


## Complete Example: Online Statistics

Compute running mean and variance in a single pass:
```julia
using KernelForge
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

KernelForge.scan!(init, combine, dst, x)

# Final statistics
final_stats = Array(dst)[end]
mean = final_stats.mean
variance = final_stats.m2 / final_stats.n
std = sqrt(variance)
```