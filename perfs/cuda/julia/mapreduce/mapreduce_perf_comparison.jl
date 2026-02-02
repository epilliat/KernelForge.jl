#=
MapReduce Performance Benchmarking Script
==========================================
Compares Luma.mapreduce! against CUDA.mapreduce and AcceleratedKernels.mapreduce
across different data types and configurations.

Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing (excludes launch overhead)
=#

using Revise
using Pkg
Pkg.activate("$(@__DIR__())/../../")

using Luma
using Luma: UnitFloat8
using KernelAbstractions, CUDA, BenchmarkTools
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
end

function run_mapreduce_benchmarks(src::CuArray{T}, f=identity; init=zero(T), luma_kw...) where T
  dst = CuArray{T}([init])

  bench("Luma.mapreduce!", () -> Luma.mapreduce!(f, +, dst, src; luma_kw...))
  bench("CUDA.mapreduce", () -> mapreduce(f, +, src))
  bench("AcceleratedKernels", () -> AcceleratedKernels.mapreduce(f, +, src; init))
end

#=============================================================================
  Float32 (n=1M) - Basic mapreduce with default parameters
=============================================================================#
#%%
run_mapreduce_benchmarks(CuArray{Float32}(1:1_000_000))

#=============================================================================
  Float32 with Pre-allocated Temporaries (n=1M)
  - UInt64 flags, Nitem=4 for improved ILP
=============================================================================#
#%%
let src = CuArray{Float32}(1:1_000_000)
  tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,); blocks=100, FlagType=UInt64)
  run_mapreduce_benchmarks(src; tmp, FlagType=UInt64, Nitem=4)
end

#=============================================================================
  Float64 (n=1M)
  - Nitem=1 due to larger register pressure
  - Memory bandwidth bound: 2x bytes per element vs Float32
=============================================================================#
#%%
run_mapreduce_benchmarks(CuArray{Float64}(1:1_000_000))

#=============================================================================
  UInt8 (n=1M)
  - Tests vectorized load capabilities (8 UInt8 = 8 bytes at once)
  - Challenges memory coalescing efficiency
=============================================================================#
#%%
run_mapreduce_benchmarks(CUDA.ones(UInt8, 1_000_000))

#=============================================================================
  UnitFloat8 Custom Type (n=1M)
  - f(x) = Float32(x): promotes to avoid overflow during accumulation
  - Manual tmp allocation with explicit H=Float32 config
=============================================================================#
#%%
let n = 1_000_000, T = UnitFloat8
  src = CuArray([rand(T) for _ in 1:n])
  dst = CuArray{T}([T(0)])
  f = Float32  # Promote to Float32 during reduction

  tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,); blocks=1000, FlagType=UInt64)

  bench("Luma.mapreduce! (UnitFloat8→Float32)", () -> Luma.mapreduce!(f, +, dst, src; tmp))
  bench("CUDA.mapreduce (UnitFloat8)", () -> mapreduce(f, +, src))
  bench("AcceleratedKernels (UnitFloat8)", () -> AcceleratedKernels.mapreduce(f, +, src; init=T(0)))
end