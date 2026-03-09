#=
Scan Performance Benchmarking Script
====================================
Compares KernelForge.scan! against CUDA.accumulate! and AcceleratedKernels for prefix scan.
Methodology:
- 500ms warm-up phase (via bench_utils.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization is used for full pipeline timing
- Results saved to results/<gpu_short>/scan.csv
- Final DataFrame printed at the end
=#
include("../meta_helper.jl")
using Pkg
Pkg.activate("perfs/envs/benchenv/$backend_str")
Pkg.instantiate()
using Revise
include("../architecture.jl")
include("../bench_utils.jl")
using DataFrames
using CSV

const DEFAULT_CUB_EXE = joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_scan_benchmark")

# ---------------------------------------------------------------------------
# Benchmark runner — returns a vector of NamedTuples
# ---------------------------------------------------------------------------

"""
    run_scan_benchmarks(src, dst, label_T, n; init, cub_exe, cub_T)

Runs bench() for KernelForge, CUDA/generic accumulate!, AcceleratedKernels, and CUB on `src`.
Returns a Vector of row NamedTuples for the results DataFrame.
"""
function run_scan_benchmarks(src::AT{T}, dst::AT{T}, label_T::String, n::Int;
    init=zero(T),
    cub_exe::String=DEFAULT_CUB_EXE, cub_T::Type=T) where T

    rows = NamedTuple[]

    for (name, method, call) in [
        (has_cuda() ? "CUDA [$label_T]" : "Base [$label_T]",
            has_cuda() ? "CUDA" : "Base",
            () -> accumulate!(+, dst, src)),
        ("AcceleratedKernels [$label_T]", "AK",
            () -> AcceleratedKernels.accumulate!(+, dst, src; init)),
        ("KernelForge [$label_T]", "KernelForge",
            () -> KernelForge.scan!(+, dst, src)),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, type=label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    if has_cuda()
        s = bench_cub_or_nan(cub_exe, n, cub_T; extra_flags="-m inclusive")
        push!(rows, (; n, type=label_T, method="CUB",
            s.mean_kernel_μs, s.std_kernel_μs,
            mean_total_μs=NaN, std_total_μs=NaN))
    end

    return rows
end


n = 10^6
T = Float32
src = fill!(AT{T}(undef, n), one(T))
dst = fill!(AT{T}(undef, n), zero(T))

KernelForge.scan!(identity, +, dst, src)
KA.synchronize(backend)
accumulate!(+, dst, src)
KA.synchronize(backend)
AcceleratedKernels.accumulate!(+, dst, src; init=0.0f0)
KA.synchronize(backend)
# Simple profiling example (without warmup here which gives slower results)

@btime (KernelForge.scan!(identity, +, dst, src); KA.synchronize(backend))
@btime (accumulate!(+, dst, src); KA.synchronize(backend))
@btime (AcceleratedKernels.accumulate!(+, dst, src; init=0.0f0); KA.synchronize(backend))

#%%
# CUDA.@profile accumulate!(+, dst, src)
# CUDA.@profile AcceleratedKernels.accumulate!(+, dst, src; init=0.0f0)
# CUDA.@profile KernelForge.scan!(identity, +, dst, src; Nitem=8)


# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------

sizes = [10^6, 10^7, 10^8]
types = [Float32, Float64]

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

for n in sizes, T in types
    println("\n" * "="^60)
    println("Scan: n=$n, T=$T")
    println("="^60)

    if T === QuaternionF64
        src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
        src = AT(src_cpu)
        dst = fill!(AT{T}(undef, n), zero(T))

        for (name, method, call) in [
            (has_cuda() ? "CUDA [QuaternionF64]" : "Base [QuaternionF64]",
                has_cuda() ? "CUDA" : "Base",
                () -> accumulate!(*, dst, src)),
            ("AcceleratedKernels [QuaternionF64]", "AK",
                () -> AcceleratedKernels.accumulate!(*, dst, src; init=one(T))),
            ("KernelForge [QuaternionF64]", "KernelForge",
                () -> KernelForge.scan!(*, dst, src; Nitem=4)),
        ]
            s = bench(name, call; backend)
            push!(all_rows, (; n, type="QuaternionF64", method,
                s.mean_kernel_μs, s.std_kernel_μs,
                s.mean_total_μs, s.std_total_μs))
        end
        push!(all_rows, (; n, type="QuaternionF64", method="CUB",
            mean_kernel_μs=NaN, std_kernel_μs=NaN,
            mean_total_μs=NaN, std_total_μs=NaN))

    else
        src = fill!(AT{T}(undef, n), one(T))
        dst = fill!(AT{T}(undef, n), zero(T))
        append!(all_rows, run_scan_benchmarks(src, dst, string(T), n))
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "scan.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits=2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs]
)
df_display.n_str = map(n -> "1e$(round(Int, log10(n)))", df_display.n)
select!(df_display, :n_str, :)
select!(df_display, Not(:n))

println("=== Scan Benchmark Results — GPU: $GPU_TAG ===")
if has_cuda()
    hl = TextHighlighter(
        (data, i, j) -> data[i, :method] == "KernelForge",
        crayon"blue bold"
    )
    pretty_table(df_display; highlighters=[hl])
else
    hl = Highlighter(
        (data, i, j) -> data[i, Symbol("method")] == "KernelForge",
        crayon"blue bold"
    )
    pretty_table(df_display; highlighters=(hl,), backend=Val(:text))
end