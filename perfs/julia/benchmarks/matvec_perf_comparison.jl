#=
MatVec Performance Benchmarking Script
======================================
Compares KernelForge.matvec against cuBLAS (via A * x) for matrix-vector multiplication.
Methodology:
- 500ms warm-up phase (via bench_utils.jl `bench`)
- 100 profiled trials for accurate kernel timing
- benchmark + synchronization is used for full pipeline timing
- Results saved to results/<gpu_short>/profiling_benchmarks/matvec.csv
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

# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------
total_elements = [10^6, 10^7]
types = [Float32]
# n ranges from 10 to total÷10, powers of 10
# recomputed per total in the loop below

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
function run_matvec_benchmarks(n::Int, p::Int, ::Type{T}) where T
    A = fill!(AT{T}(undef, n, p), one(T))
    x = AT{T}(1:p)
    dst = fill!(AT{T}(undef, n, 1), zero(T))

    println("\n" * "="^60)
    println("MatVec: n=$n, p=$p, T=$T  (n×p = $(n*p))")
    println("="^60)

    rows = NamedTuple[]
    for (name, method, call) in [
        ("KernelForge [n=$n, p=$p]", "KernelForge", () -> KernelForge.matvec(*, +, A, x)),
        (has_cuda() ? "cuBLAS [n=$n, p=$p]" : "LinearAlgebra [n=$n, p=$p]",
            has_cuda() ? "cuBLAS" : "LinearAlgebra",
            () -> A * x),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, p, type=string(T), method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end
    return rows
end


# warup
p = 100
n = 1000000
T = Float32
src = fill!(AT{T}(undef, n, p), one(T))
x = fill!(AT{T}(undef, p), one(T))

src * x
KA.synchronize(backend)
KernelForge.matvec(*, +, src, x)
KA.synchronize(backend)

CUDA.@profile src * x
KA.synchronize(backend)



# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------
all_rows = NamedTuple[]
total_elements = [10^6, 10^7, 10^8]
types = [Float32]
# n ranges from 10 to total÷10, powers of 10
# recomputed per total in the loop below
for total in total_elements, T in types
    n_values = [10^k for k in 0:round(Int, log10(total))]
    for n in n_values
        p = total ÷ n
        append!(all_rows, run_matvec_benchmarks(n, p, T))
        GC.gc(true)
        has_cuda() && CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------
df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "matvec.csv")
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

println("=== MatVec Benchmark Results — GPU: $GPU_TAG ===")
hl = TextHighlighter(
    (data, i, j) -> data[i, :method] == "KernelForge",
    crayon"blue bold"
)
pretty_table(df_display; highlighters=[hl])