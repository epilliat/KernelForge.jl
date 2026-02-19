#=
Scan Performance Benchmarking Script
====================================
Compares KernelForge.scan! against CUDA.accumulate! and AcceleratedKernels for prefix scan.
Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing
- Tests varying sizes and data types
=#
using Revise
using Pkg
Pkg.activate("$(@__DIR__)/../../")
using KernelForge
using CUDA
using AcceleratedKernels
using Quaternions

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

function sum_kernel_durations_μs(prof)
    df = prof.device
    # Filter out copy operations
    kernels = filter(row -> !startswith(row.name, "[copy"), df)
    # Sum durations: stop - start is in seconds, convert to μs
    total_s = sum(row.stop - row.start for row in eachrow(kernels))
    return total_s * 1e6  # convert to μs
end

function run_scan_benchmarks(n::Int, ::Type{T}, op=+; src_init=:range, KernelForge_kw...) where T
    src = src_init == :ones ? CUDA.ones(T, n) : CuArray{T}(1:n)
    dst = CUDA.zeros(T, n)

    println("\n" * "="^60)
    println("Scan: n=$n, T=$T, op=$op")
    println("="^60)

    bench("KernelForge.scan!", () -> KernelForge.scan!(op, dst, src; KernelForge_kw...))
    bench("CUDA.accumulate!", () -> CUDA.accumulate!(op, dst, src))
    bench("AcceleratedKernels", () -> AcceleratedKernels.accumulate!(op, dst, src; init=zero(T)))
end

#=============================================================================
  Float32 scans - varying sizes
=============================================================================#
#%%
for n in [1_000_000, 10_000_000, 100_000_000]
    run_scan_benchmarks(n, Float32, +; src_init=:ones)
end

#=============================================================================
  Float64 scans - varying sizes
=============================================================================#
#%%
for n in [1_000_000, 10_000_000, 100_000_000]
    run_scan_benchmarks(n, Float64, +)
end

#=============================================================================
  UInt8 scans - varying sizes
=============================================================================#
#%%
for n in [1_000_000, 10_000_000, 100_000_000]
    run_scan_benchmarks(n, UInt8, +; src_init=:ones)
end

#=============================================================================
  Float32 with UInt64 flags (pre-allocated tmp)
=============================================================================#
#%%
let n = 1_000_000, T = Float32
    src = CuArray{T}(1:n)
    dst = CUDA.zeros(T, n)
    tmp = KernelForge.get_allocation(KernelForge.scan!, dst, src; FlagType=UInt64)

    println("\n" * "="^60)
    println("Scan with UInt64 flags: n=$n, T=$T")
    println("="^60)

    bench("KernelForge.scan! (UInt64 flags)", () -> KernelForge.scan!(+, dst, src; tmp, FlagType=UInt64))
    bench("CUDA.accumulate!", () -> CUDA.accumulate!(+, dst, src))
    bench("AcceleratedKernels", () -> AcceleratedKernels.accumulate!(+, dst, src; init=zero(T)))
end

#=============================================================================
  Quaternions (non-commutative operation)
=============================================================================#
#%%
for n in [10_000, 100_000, 1_000_000]
    let T = QuaternionF64, op = *
        src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
        src = CuArray(src_cpu)
        dst = CuArray(zeros(T, n))

        println("\n" * "="^60)
        println("Quaternion scan: n=$n, T=$T, op=$op")
        println("="^60)

        bench("KernelForge.scan!", () -> KernelForge.scan!(op, dst, src; Nitem=4))
        bench("CUDA.accumulate!", () -> CUDA.accumulate!(op, dst, src))
        bench("AcceleratedKernels", () -> AcceleratedKernels.accumulate!(op, dst, src; init=one(T)))
    end
end