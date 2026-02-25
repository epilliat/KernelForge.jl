#=
Copy Performance Benchmarking Script
=====================================
Compares KernelForge.vcopy! against CUDA.copyto! and CUB for Float32 and UInt8.
=#

include("helper.jl")

const DEFAULT_CUB_EXE = "$(@__DIR__)/../../cuda_cpp/cub_nvcc/bin/cub_memcpy_benchmark"

# ---------------------------------------------------------------------------
# Copy-specific bench: vcopy! does not filter [copy*] kernels
# ---------------------------------------------------------------------------
bench_copy(name, f; kw...) = bench(name, f; exclude_copy=false, kw...)

function run_copy_benchmarks(n::Int, ::Type{T}; cub_exe::String=DEFAULT_CUB_EXE) where T
    src = CUDA.ones(T, n);  dst = CuArray{T}(undef, n)

    println("\n" * "="^60); println("Copy: n=$n, T=$T"); println("="^60)

    v1_stats   = bench_copy("Forge v1",     () -> KernelForge.vcopy!(dst, src; Nitem=1))
    v4_stats   = bench_copy("Forge v4",     () -> KernelForge.vcopy!(dst, src; Nitem=4))
    cuda_stats = bench_copy("CUDA.copyto!", () -> copyto!(dst, src))
    cub_stats  = bench_cub_or_nan(cub_exe, n, T)

    return (forge_v1=v1_stats, forge_v4=v4_stats, cuda=cuda_stats, cub=cub_stats)
end

function run_all_benchmarks(sizes::Vector{Int}; cub_exe::String=DEFAULT_CUB_EXE)
    rows = NamedTuple[]
    for n in sizes, (T, stats_fn) in [(Float32, n -> run_copy_benchmarks(n, Float32; cub_exe)),
                                       (UInt8,   n -> run_copy_benchmarks(n, UInt8;   cub_exe))]
        s = stats_fn(n)
        for (method, st) in [("Forge v1", s.forge_v1), ("Forge v4", s.forge_v4),
                              ("CUDA",     s.cuda),     ("CUB",      s.cub)]
            push_stats!(rows, st; n, T, method)
        end
    end
    return DataFrame(rows)
end

# Scaling benchmark (no CUB, many sizes)
function run_scaling_benchmarks(sizes::Vector{Int}; trials::Int=50)
    rows = NamedTuple[]
    for (idx, n) in enumerate(sizes)
        println("\n[$idx/$(length(sizes))] n=$n")
        for T in [Float32, UInt8]
            src = CUDA.ones(T, n); dst = CuArray{T}(undef, n)
            for (method, f) in [
                ("Forge v1", () -> KernelForge.vcopy!(dst, src; Nitem=1)),
                ("Forge v4", () -> KernelForge.vcopy!(dst, src; Nitem=4)),
                ("CUDA",     () -> copyto!(dst, src)),
            ]
                warmup(f)
                times = [sum_kernel_durations_μs(CUDA.@profile f(); exclude_copy=false) for _ in 1:trials]
                push!(rows, (; n, T=string(T), method,
                    mean_kernel_μs=mean(times), std_kernel_μs=std(times)))
            end
        end
    end
    return DataFrame(rows)
end

#=============================================================================
  Main execution
=============================================================================#
#%%
sizes_bar = [1_000_000, 100_000_000]
df_bar = run_all_benchmarks(sizes_bar)

println("\n" * "="^60); println("BENCHMARK RESULTS"); println("="^60)
show(df_bar, allrows=true, allcols=true); println()

for n in sizes_bar
    fig = plot_grouped_barplot(df_bar, n;
        title="Copy Performance (n = $(format_number(n)))",
        method_colors=COPY_METHOD_COLORS, method_order=COPY_METHOD_ORDER,
        highlight_method="Forge v4",
    )
    save("perfs/cuda/figures/benchmark/copy_benchmark_$(n).png", fig)
end

fig_multi = plot_grouped_barplot_multi(df_bar, sizes_bar;
    method_colors=COPY_METHOD_COLORS, method_order=COPY_METHOD_ORDER,
    figsize=(2 * 550, 450), highlight_method="Forge v4",
)
save("perfs/cuda/figures/benchmark/copy_benchmark_comparison.png", fig_multi)

#%%
# Scaling benchmarks
sizes_scaling = [Int(i * 1e5) for i in 1:500]
df_scaling = run_scaling_benchmarks(sizes_scaling; trials=30)

# (add plot_scaling / plot_bandwidth calls here as needed)
