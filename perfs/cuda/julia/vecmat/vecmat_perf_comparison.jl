#=
VecMat Performance Benchmarking Script
======================================
Compares Luma.vecmat! against cuBLAS (via x' * A) for vector-matrix multiplication.

Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing
- Tests varying aspect ratios with fixed total elements
=#

using Revise
using Pkg
Pkg.activate("$(@__DIR__())/../../")

using Luma
using CUDA


# Helper functions
function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

function bench(name, f; duration=0.5)
    warmup(f; duration)
    println("=== $name ===")
    prof = CUDA.@profile f()
    display(prof)
    prof
end

function run_vecmat_benchmarks(n::Int, p::Int)
    T = Float32
    x = CuArray{T}(1:n)
    A = CUDA.ones(T, n, p)
    dst = CUDA.zeros(T, 1, p)

    println("\n" * "="^60)
    println("n=$n, p=$p  (n×p = $(n*p))")
    println("="^60)

    bench("Luma.vecmat!", () -> Luma.vecmat!(*, +, dst, x, A))
    bench("cuBLAS (x' * A)", () -> x' * A)
end

#=============================================================================
  n × p = 1,000,000 elements
=============================================================================#
#%%
for n in [10, 100, 1000, 10_000, 100_000, 1_000_000]
    p = 1_000_000 ÷ n
    run_vecmat_benchmarks(n, p)
end

#=============================================================================
  n × p = 10,000,000 elements
=============================================================================#
#%%
for n in [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    p = 10_000_000 ÷ n
    run_vecmat_benchmarks(n, p)
end

function sum_kernel_durations_μs(prof)
    df = prof.device
    # Filter out copy operations
    kernels = filter(row -> !startswith(row.name, "[copy"), df)
    # Sum durations: stop - start is in seconds, convert to μs
    total_s = sum(row.stop - row.start for row in eachrow(kernels))
    return total_s * 1e6  # convert to μs
end