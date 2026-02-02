#=
MatVec Performance Benchmarking Script
======================================
Compares Luma.matvec! against cuBLAS (via src * x) for matrix-vector multiplication.
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

function run_matvec_benchmarks(n::Int, p::Int)
    T = Float32
    src = CUDA.ones(T, n, p)
    x = CuArray{T}(1:p)
    dst = CUDA.zeros(T, n, 1)

    println("\n" * "="^60)
    println("n=$n, p=$p  (n×p = $(n*p))")
    println("="^60)

    bench("Luma.matvec!", () -> Luma.matvec!(*, +, dst, src, x))
    bench("cuBLAS (src * x)", () -> src * x)
end

#=============================================================================
  n × p = 1,000,000 elements
=============================================================================#
#%%
for n in [10, 100, 1000, 10_000, 100_000, 1_000_000]
    p = 1_000_000 ÷ n
    run_matvec_benchmarks(n, p)
end

#=============================================================================
  n × p = 10,000,000 elements
=============================================================================#
#%%
for n in [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    p = 10_000_000 ÷ n
    run_matvec_benchmarks(n, p)
end

function sum_kernel_durations_μs(prof)
    df = prof.device
    # Filter out copy operations
    kernels = filter(row -> !startswith(row.name, "[copy"), df)
    # Sum durations: stop - start is in seconds, convert to μs
    total_s = sum(row.stop - row.start for row in eachrow(kernels))
    return total_s * 1e6  # convert to μs
end