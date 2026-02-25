#=
MapReduce Performance Benchmarking Script
==========================================
Compares KernelForge.mapreduce! against CUDA.mapreduce, AcceleratedKernels, and CUB
for Float32 and UnitFloat8.
=#

include("helper.jl")

const DEFAULT_CUB_EXE = "$(@__DIR__)/../../cuda_cpp/cub_nvcc/bin/cub_sum_benchmark"

function run_mapreduce_benchmarks(n::Int, ::Type{T}; cub_exe::String=DEFAULT_CUB_EXE) where T
    src = CuArray{T}([i % 10 for i in (1:n)])
    dst = CuArray{T}([zero(T)])
    f(x)::Float32 = Float32(x)   # only relevant for UnitFloat8; identity for Float32

    println("\n" * "="^60)
    println("MapReduce: n=$n, T=$T")
    println("="^60)
    if T === Float32
        tmp = KernelForge.get_allocation(KF.MapReduce1D, identity, +, src)
        cuda_stats = bench("CUDA", () -> mapreduce(identity, +, src))
        forge_stats = bench("Forge", () -> KernelForge.mapreduce(identity, +, src))
        ak_stats = bench("AK", () -> AcceleratedKernels.mapreduce(identity, +, src; init=zero(T)))
        cub_stats = bench_cub_or_nan(cub_exe, n, T)
    else  # UnitFloat8 – promote to Float32 during reduction
        cuda_stats = bench("CUDA", () -> mapreduce(f, +, src))
        forge_stats = bench("Forge", () -> KernelForge.mapreduce(f, +, src))
        ak_stats = bench("AK", () -> AcceleratedKernels.mapreduce(f, +, src; init=T(0)))
        cub_stats = bench_cub_or_nan(cub_exe, n, UInt8)   # UInt8 as proxy
    end

    return (forge=forge_stats, cuda=cuda_stats, ak=ak_stats, cub=cub_stats)
end

function run_all_benchmarks(sizes::Vector{Int}; cub_exe::String=DEFAULT_CUB_EXE)
    rows = NamedTuple[]
    for n in sizes, T in [Float32, UnitFloat8]
        s = run_mapreduce_benchmarks(n, T; cub_exe)
        for (method, st) in [("Forge", s.forge), ("CUDA", s.cuda), ("AK", s.ak), ("CUB", s.cub)]
            push_stats!(rows, st; n, T, method)
        end
    end
    return DataFrame(rows)
end

# Label suffix: mark CUB bars on UnitFloat8 as "(UInt8)"
cub_suffix(method, T) = (method == "CUB" && T == "UnitFloat8") ? "\n(UInt8)" : ""

#=============================================================================
  Main execution
=============================================================================#
#%%


run_mapreduce_benchmarks(1000000, Float32)



sizes = [1_000_000, 100_000_000]
df = run_all_benchmarks(sizes)

println("\n" * "="^60);
println("BENCHMARK RESULTS");
println("="^60);
show(df, allrows=true, allcols=true);
println();

figures = Dict{Int,Figure}()
for n in sizes
    figures[n] = plot_grouped_barplot(df, n;
        title="MapReduce Performance (n = $(format_number(n)))",
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        label_suffix_fn=cub_suffix,
    )
    save("perfs/cuda/figures/benchmark/mapreduce_benchmark_$(n).png", figures[n])
end

fig_multi = plot_grouped_barplot_multi(df, sizes;
    method_order=REDUCE_METHOD_ORDER,
    highlight_method="Forge",
    label_suffix_fn=cub_suffix,
)
save("perfs/cuda/figures/benchmark/mapreduce_benchmark_comparison.png", fig_multi)

@info "Benchmarks complete. Results in `df`, figures in `figures`."
