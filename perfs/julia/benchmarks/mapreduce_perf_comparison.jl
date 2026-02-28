#=
MapReduce Performance Benchmarking Script
==========================================
Compares KernelForge.mapreduce! against CUDA.mapreduce and AcceleratedKernels.mapreduce
across different data types and configurations.
Methodology:
- 500ms warm-up phase (via helpers.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization is used for full pipeline timing
- Results saved to results/<gpu_short>/mapreduce.csv
- Final DataFrame printed at the end
=#
using Pkg
Pkg.activate("perfs/envs/benchenv")
using Revise
include("../bench_utils.jl")
include("../architectures.jl")

const DEFAULT_CUB_EXE = joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_sum_benchmark")

# ---------------------------------------------------------------------------
# Benchmark runner — returns a vector of NamedTuples
# ---------------------------------------------------------------------------

"""
    run_mapreduce_benchmarks(n, label_T; f, init, cub_exe, cub_T)

Runs bench() for KernelForge, CUDA, AcceleratedKernels, and CUB on `src`.
`cub_T` is the dtype passed to CUB (e.g. UInt8 as proxy for UnitFloat8).
Returns a Vector of row NamedTuples for the results DataFrame.
"""
function run_mapreduce_benchmarks(src::CuArray{T}, label_T::String, n::Int;
  f=identity, init=zero(T),
  cub_exe::String=DEFAULT_CUB_EXE, cub_T::Type=T) where T

  rows = NamedTuple[]

  for (name, method, call) in [
    ("CUDA [$label_T]", "CUDA", () -> mapreduce(f, +, src)),
    ("AcceleratedKernels [$label_T]", "AK", () -> AcceleratedKernels.mapreduce(f, +, src; init)),
    ("KernelForge [$label_T]", "KernelForge", () -> KernelForge.mapreduce(f, +, src)),
  ]
    s = bench(name, call)
    push!(rows, (; n, type=label_T, method,
      s.mean_kernel_μs, s.std_kernel_μs,
      s.mean_total_μs, s.std_total_μs))
  end

  s = bench_cub_or_nan(cub_exe, n, cub_T)
  push!(rows, (; n, type=label_T, method="CUB",
    s.mean_kernel_μs, s.std_kernel_μs,
    mean_total_μs=NaN, std_total_μs=NaN))

  return rows
end


# Simple profiling example (without warmup here which gives slower results)

src = CuArray{Float32}(1:1000000)

CUDA.@profile CUDA.mapreduce(identity, +, src)
CUDA.@profile AcceleratedKernels.mapreduce(identity, +, src; init=0.0f0)
CUDA.@profile KernelForge.mapreduce(identity, +, src)


# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

sizes = [1_000_000, 100_000_000]
types = [Float32, UnitFloat8]

all_rows = NamedTuple[]

for n in sizes, T in types
  if T === UnitFloat8
    src = CuArray([rand(T) for _ in 1:n])
    f(x) = Float32(x)
    label = "UnitFloat8→Float32"
    for (name, method, call) in [
      ("CUDA [$label]", "CUDA", () -> mapreduce(f, +, src)),
      ("AcceleratedKernels [$label]", "AK", () -> AcceleratedKernels.mapreduce(f, +, src; init=T(0))),
      ("KernelForge [$label]", "KernelForge", () -> KernelForge.mapreduce(f, +, src)),
    ]
      s = bench(name, call)
      push!(all_rows, (; n, type=label, method,
        s.mean_kernel_μs, s.std_kernel_μs,
        s.mean_total_μs, s.std_total_μs))
    end
    s = bench_cub_or_nan(DEFAULT_CUB_EXE, n, UInt8)
    push!(all_rows, (; n, type=label, method="CUB",
      s.mean_kernel_μs, s.std_kernel_μs,
      mean_total_μs=NaN, std_total_μs=NaN))
  else
    src = T === UInt8 ? CUDA.ones(T, n) : CuArray{T}(1:n)
    append!(all_rows, run_mapreduce_benchmarks(src, string(T), n))
  end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)

out_dir = RESULT_DIR
mkpath(out_dir)
csv_path = joinpath(out_dir, "mapreduce.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

df_display = transform(df,
  [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
    (x -> round.(x; digits=2)) .=>
      [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs]
)
df_display.n_str = map(n -> "1e$(round(Int, log10(n)))", df_display.n)
select!(df_display, :n_str, :)  # move it first
select!(df_display, Not(:n))    # drop the raw n column
println("=== MapReduce Benchmark Results — GPU: $GPU_TAG ===")

hl = TextHighlighter(
  (data, i, j) -> data[i, :method] == "KernelForge",
  crayon"blue bold"
)
pretty_table(df_display; highlighters=[hl])