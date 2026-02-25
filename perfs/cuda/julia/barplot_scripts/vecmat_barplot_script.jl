#=
VecMat Performance Benchmarking Script
======================================
Compares KernelForge.vecmat against cuBLAS (x' * A) for Float32 vector-matrix multiplication.
Tests varying aspect ratios with fixed total elements (n × p).
=#

include("helper.jl")

function run_vecmat_benchmarks(n::Int, p::Int; tmp=nothing, FlagType=UInt8)
    T = Float32
    x = CuArray{T}(1:n)
    A = CUDA.ones(T, n, p)
    dst = CUDA.zeros(T, 1, p)

    println("\n" * "="^60)
    println("n=$n, p=$p  (n×p = $(n*p))")
    println("="^60)

    cublas_stats = bench("cuBLAS (x' * A)", () -> x' * A)
    forge_stats = bench("KernelForge.vecmat", () -> KernelForge.vecmat(*, +, x, A))

    return (forge=forge_stats, cublas=cublas_stats)
end

function run_all_benchmarks(sizes::Vector{Int}; tmp=nothing, FlagType=UInt8)
    rows = NamedTuple[]
    for total in sizes
        n_values = filter(n -> total % n == 0 && n <= total,
            [10, 100, 1_000, 10_000, 100_000, 1_000_000])
        for n in n_values
            p = total ÷ n
            s = run_vecmat_benchmarks(n, p; tmp, FlagType)
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
run_vecmat_benchmarks(1_000_000, 1; tmp)

sizes = [1_000_000, 10_000_000]
df = run_all_benchmarks(sizes)

println("\n" * "="^60);
println("BENCHMARK RESULTS");
println("="^60);
show(df, allrows=true, allcols=true);
println();

figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (vector length)")
               for total in sort(unique(df.total_elements)))
for (total, fig) in figures
    save("perfs/cuda/figures/benchmark/vecmat_benchmark_$(total).png", fig)
end

fig_multi = plot_npbar_multi(df, sizes; x_col=:n, xlabel="n (vector length)")
save("perfs/cuda/figures/benchmark/vecmat_benchmark_comparison.png", fig_multi)

@info "Benchmarks complete. Results in `df`, figures in `figures`."
