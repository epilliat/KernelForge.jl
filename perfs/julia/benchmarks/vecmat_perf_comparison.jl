#=
VecMat Performance Benchmarking Script
======================================
Compares KernelForge.vecmat! against cuBLAS (via x' * A) for vector-matrix multiplication.
Methodology:
- 500ms warm-up phase (via bench_utils.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization is used for full pipeline timing
- Results saved to results/<gpu_short>/profiling_benchmarks/vecmat.csv
- Final DataFrame printed at the end
=#
using Pkg
Pkg.activate("perfs/envs/benchenv")
using Revise
include("../bench_utils.jl")
include("../architectures.jl")

using DataFrames
using CSV

# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------

total_elements = [10^6, 10^7]
types = [Float32]

# n ranges from 10 to total÷10, powers of 10
# e.g. total=10^6 → [10, 100, 1_000, 100_000]
# recomputed per total in the loop below

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

function run_vecmat_benchmarks(n::Int, p::Int, ::Type{T}) where T
    x = CuArray{T}(1:n)
    A = CUDA.ones(T, n, p)
    dst = CUDA.zeros(T, 1, p)

    println("\n" * "="^60)
    println("VecMat: n=$n, p=$p, T=$T  (n×p = $(n*p))")
    println("="^60)

    rows = NamedTuple[]

    for (name, method, call) in [
        ("KernelForge [n=$n, p=$p]", "KernelForge", () -> KernelForge.vecmat(*, +, x, A)),
        ("cuBLAS [n=$n, p=$p]", "cuBLAS", () -> x' * A),
    ]
        s = bench(name, call)
        push!(rows, (; n, p, type=string(T), method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end

    return rows
end


# Simple profiling example (without warmup here which gives slower results.)

src = CUDA.ones(Float32, 1000, 1000)
x = CUDA.ones(Float32, 1000)

CUDA.@profile x' * src
CUDA.@profile KernelForge.vecmat(*, +, x, src)


# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]

for total in total_elements, T in types
    n_values = [10^k for k in 1:floor(Int, log10(total / 10))]
    for n in n_values
        p = total ÷ n
        append!(all_rows, run_vecmat_benchmarks(n, p, T))
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------

df = DataFrame(all_rows)

out_dir = RESULT_DIR
mkpath(out_dir)
csv_path = joinpath(out_dir, "vecmat.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

fmt_e(x) = "1e$(round(Int, log10(x)))"

df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits=2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs]
)
df_display.np_str = map((n, p) -> fmt_e(n * p), df_display.n, df_display.p)
df_display.n_str = fmt_e.(df_display.n)
df_display.p_str = fmt_e.(df_display.p)
select!(df_display, :np_str, :n_str, :p_str, :)
select!(df_display, Not([:n, :p]))

println("=== VecMat Benchmark Results — GPU: $GPU_TAG ===")
hl = TextHighlighter(
    (data, i, j) -> data[i, :method] == "KernelForge",
    crayon"blue bold"
)
pretty_table(df_display; highlighters=[hl])