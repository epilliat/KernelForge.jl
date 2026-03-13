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
include("../init.jl")


# n ranges from 10 to total÷10, powers of 10
# e.g. total=10^6 → [10, 100, 1_000, 100_000]
# recomputed per total in the loop below

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

function run_vecmat_benchmarks(n::Int, p::Int, ::Type{T}) where T
    x = AT{T}(1:n)
    A = fill!(AT{T}(undef, n, p), one(T))
    dst = fill!(AT{T}(undef, 1, p), zero(T))

    println("\n" * "="^60)
    println("VecMat: n=$n, p=$p, T=$T  (n×p = $(n*p))")
    println("="^60)

    rows = NamedTuple[]
    for (name, method, call) in [
        ("KernelForge [n=$n, p=$p]", "KernelForge", () -> KernelForge.vecmat(*, +, x, A)),
        (has_cuda() ? "cuBLAS [n=$n, p=$p]" : "LinearAlgebra [n=$n, p=$p]",
            has_cuda() ? "cuBLAS" : "LinearAlgebra",
            () -> x' * A),
    ]
        s = bench(name, call; backend)
        push!(rows, (; n, p, type=string(T), method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end
    return rows
end


# Simple profiling example (without warmup here which gives slower results.)

n = 10000
p = 10000
x = AT{Float32}(undef, n)
fill!(x, one(Float32))
src = AT{Float32}(undef, n, p)
fill!(src, one(Float32))

x' * src
KA.synchronize(backend)
KernelForge.vecmat(*, +, x, src)
KA.synchronize(backend)


# CUDA.@profile x' * src
# CUDA.@profile KernelForge.vecmat(*, +, x, src)
# n = 10
# p = 1000000
# x = AT{Float32}(undef, n)
# fill!(x, one(Float32))
# src = AT{Float32}(undef, n, p)
# fill!(src, one(Float32))
# CUDA.@profile x' * src
# CUDA.@profile KernelForge.vecmat(*, +, x, src)


# ---------------------------------------------------------------------------
# Configuration — edit these to control what gets benchmarked
# ---------------------------------------------------------------------------
total_elements = [10^7, 10^8]
types = [Float32]

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------
all_rows = NamedTuple[]
for total in total_elements, T in types
    n_values = total <= 10^8 ? [10^k for k in 0:round(Int, log10(total))] : [10^k for k in 1:(round(Int, log10(total))-1)]
    for n in n_values
        p = total ÷ n
        append!(all_rows, run_vecmat_benchmarks(n, p, T))
    end
end

# ---------------------------------------------------------------------------
# Build DataFrame, save CSV, display
# ---------------------------------------------------------------------------
df = DataFrame(all_rows)
mkpath(RESULT_DIR)
csv_path = joinpath(RESULT_DIR, "vecmat.csv")
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
# hl = TextHighlighter(
#     (data, i, j) -> data[i, :method] == "KernelForge",
#     crayon"blue bold"
# )
# pretty_table(df_display; highlighters=[hl])