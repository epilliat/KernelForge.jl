#=
MatVec Performance Benchmarking Script
======================================
Compares KernelForge.matvec against cuBLAS (A * x) for Float32 matrix-vector multiplication.
Tests varying aspect ratios with fixed total elements (n × p).
=#

include("helper.jl")

function run_matvec_benchmarks(n::Int, p::Int; tmp=nothing, FlagType=UInt8)
    T = Float32
    A = CUDA.ones(T, n, p)
    x = CuArray{T}(1:p)
    dst = CUDA.zeros(T, n)

    println("\n" * "="^60)
    println("n=$n, p=$p  (n×p = $(n*p))")
    println("="^60)

    cublas_stats = bench("cuBLAS (A * x)", () -> A * x)
    forge_stats = bench("KernelForge.matvec", () -> KernelForge.matvec(*, +, A, x))

    return (forge=forge_stats, cublas=cublas_stats)
end

function run_all_benchmarks(sizes::Vector{Int}; tmp=nothing, FlagType=UInt8)
    rows = NamedTuple[]
    for total in sizes
        p_values = filter(p -> total % p == 0 && p <= total,
            [10, 100, 1_000, 10_000, 100_000, 1_000_000])
        for p in p_values
            n = total ÷ p
            s = run_matvec_benchmarks(n, p; tmp, FlagType)
            for (method, st) in [("Forge", s.forge), ("cuBLAS", s.cublas)]
                push!(rows, (; total_elements=total, n, p, method,
                    st.mean_kernel_μs, st.std_kernel_μs, st.mean_total_μs, st.std_total_μs))
            end
        end
    end
    return DataFrame(rows)
end

#=============================================================================
  Main execution
=============================================================================#
#%%
tmp = CUDA.zeros(UInt8, 8 * 100_000)
run_matvec_benchmarks(1, 1_000_000; tmp)

sizes = [1_000_000, 10_000_000]
df = run_all_benchmarks(sizes)

println("\n" * "="^60);
println("BENCHMARK RESULTS");
println("="^60);
show(df, allrows=true, allcols=true);
println();

figures = Dict(total => plot_npbar(df, total; x_col=:p, xlabel="p (vector length)")
               for total in sort(unique(df.total_elements)))
for (total, fig) in figures
    save("perfs/cuda/figures/benchmark/matvec_benchmark_$(total).png", fig)
end

fig_multi = plot_npbar_multi(df, sizes; x_col=:p, xlabel="p (vector length)")
save("perfs/cuda/figures/benchmark/matvec_benchmark_comparison.png", fig_multi)

@info "Benchmarks complete. Results in `df`, figures in `figures`."
