#=
VecMat Performance Benchmarking Script
======================================
=#
include("../init.jl")

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
function run_vecmat_benchmarks(n::Int, p::Int, ::Type{T}) where T
    A = fill!(AT{T}(undef, n, p), one(T))
    x = AT{T}(1:n)
    dst = fill!(AT{T}(undef, p), zero(T))
    tmp = KF.get_allocation(KF.VecMat, *, +, x, A)

    println("\n" * "="^60)
    println("VecMat: n=$n, p=$p, T=$T  (n×p = $(n*p))")
    println("="^60)

    rows = NamedTuple[]
    for (name, method, call, g) in [
            ("KernelForge [n=$n, p=$p]", "KernelForge",
             () -> KernelForge.vecmat!(dst, x, A; tmp),
             @isdefined(AMDGPU) ? () -> KernelForge.vecmat!(dst, x, A; tmp) : () -> ()),
            (has_cuda() ? "cuBLAS [n=$n, p=$p]" : "LinearAlgebra [n=$n, p=$p]",
             has_cuda() ? "cuBLAS" : "LinearAlgebra",
             () -> x' * A,
             () -> ()),
        ]
        s = bench(name, call; backend, g)
        push!(rows, (; n, p, type=string(T), method,
            s.mean_kernel_μs, s.std_kernel_μs,
            s.mean_total_μs, s.std_total_μs))
    end
    return rows
end

# warmup
n = 10000000
p = 10
x = fill!(AT{Float32}(undef, n), one(Float32))
src = fill!(AT{Float32}(undef, n, p), one(Float32))
dst_warmup = fill!(AT{Float32}(undef, p), zero(Float32))
tmp_warmup = KF.get_allocation(KF.VecMat, *, +, x, src)
x' * src; KA.synchronize(backend)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
total_elements = [10^7, 10^8, 10^9]
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