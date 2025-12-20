#=
MapReduce Comprehensive Benchmark Suite
=======================================
Compares Luma.mapreduce! against CUDA.jl, AcceleratedKernels (AK), and CUB
across different data types (Float32, UnitFloat8) and problem sizes (1M, 100M).

Methodology:
- 500ms warm-up to ensure JIT compilation and steady-state GPU behavior
- 100 profiled runs for kernel timing statistics
- ~2s of timed runs for pipeline/overhead measurements
- Results collected in DataFrame for visualization

Output: Stacked bar plots comparing kernel times across implementations.
=#

using Revise
using Pkg

Pkg.activate("$(@__DIR__)/../")

# Development paths (uncomment if needed)
# luma_path = abspath("$(@__DIR__)/../../../")
# Pkg.develop(path=luma_path)
# ma_path = abspath("/home/emmanuel/Packages/MemoryAccess.jl/")
# Pkg.develop(path=ma_path)

using Luma
using Luma: UnitFloat8
using Plots
using KernelAbstractions, Test, CUDA, BenchmarkTools, DataFrames
import AcceleratedKernels as AK
using Quaternions

include("helpers/extract_infos.jl")
include("helpers/illustration_tools.jl")

#=============================================================================
  CONFIGURATION
=============================================================================#

tmax_timed = 1  # Minimum seconds of timed runs for statistical stability

#=============================================================================
  HELPER: Benchmark Runner
  ------------------------
  Standardized benchmarking procedure:
  1. 500ms warm-up (JIT + GPU initialization)
  2. 100 profiled kernel executions
  3. Timed runs until 2× tmax_timed reached
  4. Results appended to DataFrame via benchmark_summary!
=============================================================================#

"""
    run_benchmark!(bench, kernel_fn, T, N, name, algo)

Execute standardized benchmark and append results to `bench` DataFrame.
"""
function run_benchmark!(bench::DataFrame, kernel_fn::Function, T::Type, N::Int,
  name::String, algo::String)
  # Warm-up phase: ensures JIT compilation doesn't pollute measurements
  start_time = time()
  while time() - start_time < 0.5
    CUDA.@sync kernel_fn()
  end

  # Profiling: captures kernel-level GPU timing (100 samples)
  prof = [CUDA.@profile kernel_fn() for _ in 1:100]

  # Timed runs: captures full pipeline including CPU overhead
  dt = 0.0
  dts = Float64[]
  while dt <= 2 * tmax_timed
    timed = CUDA.@timed kernel_fn()
    push!(dts, timed[:time])
    dt += timed[:time]
  end

  timed = CUDA.@timed kernel_fn()
  benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
end

#=============================================================================
  SECTION 1: Small Problem Size (N = 1M)
  --------------------------------------
  At 1M elements, kernel launch overhead is significant relative to compute.
  This tests how well each implementation minimizes fixed costs.
=============================================================================#

bench = DataFrame()
algo = "Sum"
N = 1_000_000

#-----------------------------------------------------------------------------
# 1.1 Luma Default Configuration
#     Uses automatic parameter selection (Nitem, FlagType, etc.)
#-----------------------------------------------------------------------------

for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity  # Promote UnitFloat8 to avoid overflow

  src = CuArray{T}(ones(T, N))
  dst = CuArray{T}([zero(T)])

  run_benchmark!(bench, () -> Luma.mapreduce!(f, op, dst, src), T, N, "Luma Def", algo)
end

#-----------------------------------------------------------------------------
# 1.2 Luma Optimized Configuration
#     - UInt64 flags: skips flag initialization (faster for large problems)
#     - Pre-allocated tmp: excludes allocation from timing
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity
  H = T == UnitFloat8 ? Float32 : T  # Intermediate type after f
  FlagType = UInt64
  blocks = 100  # Default block count for pre-allocation

  src = CuArray{T}(ones(T, N))
  dst = CuArray{T}([zero(T)])

  # Pre-allocate temporary buffers
  tmp = Luma.get_allocation(Luma.mapreduce1d!, (src,); blocks, eltype=H, FlagType)

  run_benchmark!(bench,
    () -> Luma.mapreduce!(f, op, dst, src; tmp, FlagType),
    T, N, "Luma Opt", algo)
end

#-----------------------------------------------------------------------------
# 1.3 CUDA.jl Reference
#     Built-in mapreduce, returns value (allocating).
#     Uses dims=1 for explicit 1D reduction.
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity

  src = CuArray{T}(ones(T, N))

  run_benchmark!(bench,
    () -> mapreduce(f, op, src; dims=1),
    T, N, "CUDA", algo)
end

#-----------------------------------------------------------------------------
# 1.4 AcceleratedKernels Reference
#     Portable GPU library. Note: dims=1 significantly slower (avoid it).
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity

  src = CuArray{T}(ones(T, N))

  # Note: AK.mapreduce with dims=1 is much slower; omit it
  run_benchmark!(bench,
    () -> AK.mapreduce(f, op, src; init=T(0)),
    T, N, "AK", algo)
end

#=============================================================================
  SECTION 2: CUB Reference Times (N = 1M)
  ---------------------------------------
  Pre-measured CUB times from NVCC benchmark (external C++ program).
  CUB = CUDA Unbound, NVIDIA's optimized primitives library.

  Times in microseconds, converted to milliseconds for DataFrame.
  Temporary buffer sizes noted for reference.
=============================================================================#

#%%
# CUB benchmark results (μs → ms conversion)
time_cub_u8_1e6 = 0.0070 * 1000  # tmp: 407.5 KB - surprisingly large for UInt8
time_cub_f32_1e6 = 0.0092 * 1000  # tmp: 4.75 KB

# Helper to create CUB reference DataFrame row
function make_cub_row(T::Type, N::Int, time_ms::Float64)
  DataFrame(
    name="Cub",
    datatype=T,
    algo="Sum",
    datalength=N,
    kernel1=0.0, kernel2=0.0, kernel3=0.0, kernel4=time_ms,
    kernel1_acc=0.0, kernel2_acc=0.0, kernel3_acc=0.0, kernel4_acc=time_ms,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=time_ms,
    median_duration_pipeline=time_ms,
  )
end

cub_f32 = make_cub_row(Float32, N, time_cub_f32_1e6)
cub_u8 = make_cub_row(UnitFloat8, N, time_cub_u8_1e6)  # UInt8 proxy for UnitFloat8

# Prepare DataFrame for plotting
bench.kernel1_name .= missing
bench.kernel2_name .= missing
bench_with_cub = vcat(bench, cub_f32, cub_u8; cols=:union)

#=============================================================================
  SECTION 3: Generate Plots (N = 1M)
=============================================================================#

#%%
plot_names = ["CUDA", "AK", "Luma Def", "Luma Opt", "Cub"]

plot1 = create_kernel_stacked_barplot(
  bench_with_cub;
  algo="Sum",
  kernel_colors=[:blue, :red, :green, :orange],
  overhead_alpha=0.7,
  names=plot_names,
  size_anotation=9
)
savefig(plot1, "$(@__DIR__)/../figures/mapreduce_comparison_1e6.png")

#=============================================================================
  SECTION 4: Large Problem Size (N = 100M)
  ----------------------------------------
  At 100M elements, compute dominates over launch overhead.
  This tests raw throughput and memory bandwidth utilization.

  Memory footprint:
  - Float32: 100M × 4B = 400 MB
  - UnitFloat8: 100M × 1B = 100 MB
=============================================================================#

#%%
bench = DataFrame()
N = 100_000_000

#-----------------------------------------------------------------------------
# 4.1 Luma Default
#-----------------------------------------------------------------------------

for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity

  src = CuArray{T}(ones(T, N))
  dst = CuArray{T}([zero(T)])

  run_benchmark!(bench, () -> Luma.mapreduce!(f, op, dst, src), T, N, "Luma Def", algo)
end

#-----------------------------------------------------------------------------
# 4.2 Luma Optimized (with pre-allocated temporaries)
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity
  H = T == UnitFloat8 ? Float32 : T  # Intermediate type after f
  FlagType = UInt64
  blocks = 100  # Default block count for pre-allocation

  src = CuArray{T}(ones(T, N))
  dst = CuArray{T}([zero(T)])

  tmp = Luma._get_allocation(Luma.mapreduce1d!, (src,); blocks, H, FlagType)

  run_benchmark!(bench,
    () -> Luma.mapreduce!(f, op, dst, src; tmp, FlagType),
    T, N, "Luma Opt", algo)
end

#-----------------------------------------------------------------------------
# 4.3 CUDA.jl Reference
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity

  src = CuArray{T}(ones(T, N))

  run_benchmark!(bench,
    () -> mapreduce(f, op, src; dims=1),
    T, N, "CUDA", algo)
end

#-----------------------------------------------------------------------------
# 4.4 AcceleratedKernels Reference
#-----------------------------------------------------------------------------

#%%
for T in [Float32, UnitFloat8]
  op = +
  f = T == UnitFloat8 ? (x -> Float32(x)) : identity

  src = CuArray{T}(ones(T, N))

  run_benchmark!(bench,
    () -> AK.mapreduce(f, op, src; init=T(0)),
    T, N, "AK", algo)
end

#=============================================================================
  SECTION 5: CUB Reference Times (N = 100M)
  -----------------------------------------
  Larger problem reveals true throughput characteristics.
  Note: CUB doesn't natively support UnitFloat8; we benchmark UInt8 as proxy.
=============================================================================#

#%%
time_cub_u8_1e8 = 0.5554 * 1000  # tmp: 407.5 KB
time_cub_f32_1e8 = 2.1917 * 1000  # tmp: 407.5 KB

cub_f32 = make_cub_row(Float32, N, time_cub_f32_1e8)
cub_u8 = make_cub_row(UnitFloat8, N, time_cub_u8_1e8)

bench.kernel1_name .= missing
bench.kernel2_name .= missing
bench_with_cub = vcat(bench, cub_f32, cub_u8; cols=:union)

#=============================================================================
  SECTION 6: Generate Plots (N = 100M)
=============================================================================#

#%%
plot2 = create_kernel_stacked_barplot(
  bench_with_cub;
  algo="Sum",
  kernel_colors=[:blue, :red, :green, :orange],
  overhead_alpha=0.7,
  names=plot_names,
  size_anotation=9,
  time_unit=:ms  # Milliseconds appropriate for larger times
)
savefig(plot2, "$(@__DIR__)/../figures/mapreduce_comparison_1e8.png")

#=============================================================================
  SECTION 7: Combined Comparison Plot
  -----------------------------------
  Side-by-side comparison showing scaling behavior from 1M to 100M elements.
  Reveals which implementations scale well vs. have high fixed overhead.
=============================================================================#

#%%
combined_plot = Plots.plot(
  plot1, plot2;
  layout=(1, 2),
  size=(1000, 500),
  left_margin=7Plots.mm,
  bottom_margin=10Plots.mm,
  right_margin=0Plots.mm
)
savefig(combined_plot, "$(@__DIR__)/../figures/mapreduce_nof64_1e6_1e8.png")










