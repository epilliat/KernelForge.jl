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
using Pkg
Pkg.activate("perfs/envs/benchenv")
using Revise
include("../bench_utils.jl")
include("../architectures.jl")


const DEFAULT_CUB_EXE = joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_scan_benchmark")

# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------

sizes = [1_000_000, 100_000_000]
types = [Float32, Float64]# UInt8, QuaternionF64

# ---------------------------------------------------------------------------
# Benchmark runner — returns a vector of NamedTuples
# ---------------------------------------------------------------------------

"""
    run_scan_benchmarks(src, dst, label_T, n; init, cub_exe, cub_T)

Runs bench() for KernelForge, CUDA, AcceleratedKernels, and CUB on `src`.
Returns a Vector of row NamedTuples for the results DataFrame.
"""
function run_scan_benchmarks(src::CuArray{T}, dst::CuArray{T}, label_T::String, n::Int;
    init=zero(T),
    cub_exe::String=DEFAULT_CUB_EXE, cub_T::Type=T) where T

    rows = NamedTuple[]

    for (name, method, call) in [
        ("CUDA [$label_T]", "CUDA", () -> CUDA.accumulate!(+, dst, src)),
        ("AcceleratedKernels [$label_T]", "AK", () -> AcceleratedKernels.accumulate!(+, dst, src; init)),
        ("KernelForge [$label_T]", "KernelForge", () -> KernelForge.scan!(+, dst, src)),
    ]
        s = bench(name, call)
        push!(rows, (; n, type=label_T, method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    s = bench_cub_or_nan(cub_exe, n, cub_T; extra_flags="-m inclusive")
    push!(rows, (; n, type=label_T, method="CUB",
        s.mean_kernel_μs, s.std_kernel_μs,
        mean_total_μs=NaN, std_total_μs=NaN))

    return rows
end

# Simple profiling example (without warmup here which gives slower results)

src = CuArray{Float32}(1:1000000)
dst = CUDA.zeros(Float32, 1000000)

CUDA.@profile accumulate!(+, dst, src)
CUDA.@profile AcceleratedKernels.accumulate!(+, dst, src; init=0.0f0)
CUDA.@profile KernelForge.scan!(identity, +, dst, src)



# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

for n in sizes, T in types
    println("\n" * "="^60)
    println("Scan: n=$n, T=$T")
    println("="^60)

    if T === QuaternionF64
        # Non-commutative multiply scan — no CUB support
        src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
        src = CuArray(src_cpu)
        dst = CuArray(zeros(T, n))

        for (name, method, call) in [
            ("CUDA [QuaternionF64]", "CUDA", () -> CUDA.accumulate!(*, dst, src)),
            ("AcceleratedKernels [QuaternionF64]", "AK", () -> AcceleratedKernels.accumulate!(*, dst, src; init=one(T))),
            ("KernelForge [QuaternionF64]", "KernelForge", () -> KernelForge.scan!(*, dst, src; Nitem=4)),
        ]
            s = bench(name, call)
            push!(all_rows, (; n, type="QuaternionF64", method,
                s.mean_kernel_μs, s.std_kernel_μs,
                s.mean_total_μs, s.std_total_μs))
        end
        push!(all_rows, (; n, type="QuaternionF64", method="CUB",
            mean_kernel_μs=NaN, std_kernel_μs=NaN,
            mean_total_μs=NaN, std_total_μs=NaN))

    else
        src = T === UInt8 ? CUDA.ones(T, n) : CuArray{T}(1:n)
        dst = CUDA.zeros(T, n)
        append!(all_rows, run_scan_benchmarks(src, dst, string(T), n))
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)

out_dir = RESULT_DIR
mkpath(out_dir)
csv_path = joinpath(out_dir, "scan.csv")
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
hl = TextHighlighter(
    (data, i, j) -> data[i, :method] == "KernelForge",
    crayon"blue bold"
)
pretty_table(df_display; highlighters=[hl])