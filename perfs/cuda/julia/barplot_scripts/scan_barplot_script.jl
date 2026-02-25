#=
Scan Performance Benchmarking Script
=====================================
Compares KernelForge.scan! against CUDA.accumulate!, AcceleratedKernels, and CUB
for prefix scan (Float32, Float64).
=#

include("helper.jl")

const DEFAULT_CUB_EXE = "$(@__DIR__)/../../cuda_cpp/cub_nvcc/bin/cub_scan_benchmark"

function run_scan_benchmarks(n::Int, ::Type{T}, op=+; cub_exe::String=DEFAULT_CUB_EXE) where T
    src = CuArray{T}(1:n)
    dst = CUDA.zeros(T, n)

    println("\n" * "="^60)
    println("Scan: n=$n, T=$T, op=$op")
    println("="^60)

    cuda_stats = bench("CUDA", () -> CUDA.accumulate!(op, dst, src))
    forge_stats = bench("Forge", () -> KernelForge.scan!(op, dst, src))
    ak_stats = bench("AK", () -> AcceleratedKernels.accumulate!(op, dst, src; init=zero(T)))
    # CUB scan accepts an extra -m flag for inclusive/exclusive mode
    cub_stats = bench_cub_or_nan(cub_exe, n, T; extra_flags="-m inclusive")

    return (forge=forge_stats, cuda=cuda_stats, ak=ak_stats, cub=cub_stats)
end

function run_all_benchmarks(sizes::Vector{Int}, types::Vector{DataType}=DataType[Float32, Float64];
    cub_exe::String=DEFAULT_CUB_EXE)
    rows = NamedTuple[]
    for n in sizes, T in types
        s = run_scan_benchmarks(n, T; cub_exe)
        for (method, st) in [("Forge", s.forge), ("CUDA", s.cuda), ("AK", s.ak), ("CUB", s.cub)]
            push_stats!(rows, st; n, T, method)
        end
    end
    return DataFrame(rows)
end

#=============================================================================
  Main execution
=============================================================================#
#%%
run_scan_benchmarks(1000000, Float32)
sizes = [1_000_000, 100_000_000]
df = run_all_benchmarks(sizes, [Float32, Float64])

println("\n" * "="^60);
println("BENCHMARK RESULTS");
println("="^60);
show(df, allrows=true, allcols=true);
println();

figures = Dict{Int,Figure}()
for n in sizes
    figures[n] = plot_grouped_barplot(df, n;
        title="Scan Performance (n = $(format_number(n)))",
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
    )
    save("perfs/cuda/figures/benchmark/scan_benchmark_$(n).png", figures[n])
end

fig_multi = plot_grouped_barplot_multi(df, sizes;
    method_order=REDUCE_METHOD_ORDER,
    highlight_method="Forge",
)
save("perfs/cuda/figures/benchmark/scan_benchmark_comparison.png", fig_multi)

@info "Benchmarks complete. Results in `df`, figures in `figures`."
