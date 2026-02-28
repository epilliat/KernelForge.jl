#=
Copy Scaling Benchmarking Script
=================================
Benchmarks copy performance as a function of array size.
Compares KernelForge v1/v4/v8 against CUDA.copyto!

Methodology:
- Single warmup pass per dtype (outside size loop)
- 10 profiled trials per configuration
- Results saved to results/<gpu_short>/copy.csv
=#

using Pkg
Pkg.activate("perfs/envs/benchenv")
using Revise
include("../bench_utils.jl")
include("../architectures.jl")

using DataFrames
using CSV


# ---------------------------------------------------------------------------
# Copy scaling plots
# ---------------------------------------------------------------------------

const COPY_SCALING_METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "Forge v8"]
const COPY_ELEMENT_SIZES = Dict("Float32" => 4, "UInt8" => 1)

function get_l2_cache_size()
    return CUDA.attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
end

"""
    plot_copy_scaling(df; ...) -> Figure

Log-log plot of copy kernel time vs array size, one panel per dtype.
Error band shows ± std.
"""
function plot_copy_scaling(
    df::DataFrame;
    figsize::Tuple{Int,Int}=(1000, 450),
    method_colors::Dict{String,<:Any}=BENCH_COLORS,
    method_order::Vector{String}=COPY_SCALING_METHOD_ORDER,
)
    fig = Figure(size=figsize)
    types = ["Float32", "UInt8"]
    l2_bytes = get_l2_cache_size()

    for (col, T) in enumerate(types)
        subset = sort(filter(r -> r.T == T, df), :n)
        isempty(subset) && continue

        ax = Axis(fig[1, col];
            xlabel="Array size (elements)", ylabel="Time (μs)",
            xscale=log10, yscale=log10,
            title="Copy Performance: $T", titlesize=18,
            xlabelsize=14, ylabelsize=14,
        )

        for method in method_order
            d = sort(filter(r -> r.method == method, subset), :n)
            isempty(d) && continue
            c = method_colors[method]
            lw = endswith(method, "v8") ? 3 : 2
            lines!(ax, d.n, d.mean_kernel_μs; color=c, linewidth=lw, label=method)
            band!(ax, d.n,
                d.mean_kernel_μs .- d.std_kernel_μs,
                d.mean_kernel_μs .+ d.std_kernel_μs;
                color=(c, 0.2))
        end

        elem_size = get(COPY_ELEMENT_SIZES, T, 4)
        vlines!(ax, [l2_bytes ÷ (2 * elem_size)];
            color=:black, linestyle=:dashdot, linewidth=1, label="L2 limit")
    end

    Legend(fig[2, :], fig.content[1];
        orientation=:horizontal, tellheight=true, tellwidth=false)
    return fig
end

"""
    plot_copy_bandwidth(df; ...) -> Figure

Achieved memory bandwidth vs array size, one panel per dtype.
Bandwidth computed as 2 × n × sizeof(T) / kernel_time.
"""
function plot_copy_bandwidth(
    df::DataFrame;
    figsize::Tuple{Int,Int}=(1000, 450),
    method_colors::Dict{String,<:Any}=BENCH_COLORS,
    method_order::Vector{String}=COPY_SCALING_METHOD_ORDER,
)
    fig = Figure(size=figsize)
    types = ["Float32", "UInt8"]
    l2_bytes = get_l2_cache_size()

    for (col, T) in enumerate(types)
        subset = sort(filter(r -> r.T == T, df), :n)
        isempty(subset) && continue

        ax = Axis(fig[1, col];
            xlabel="Array size (elements)", ylabel="Bandwidth (GB/s)",
            xscale=log10,
            title="Memory Bandwidth: $T", titlesize=18,
            xlabelsize=14, ylabelsize=14,
        )

        elem_size = get(COPY_ELEMENT_SIZES, T, 4)

        for method in method_order
            d = sort(filter(r -> r.method == method, subset), :n)
            isempty(d) && continue
            bw = (d.n .* elem_size .* 2) ./ (d.mean_kernel_μs .* 1e-6) ./ 1e9
            c = method_colors[method]
            lw = endswith(method, "v8") ? 3 : 2
            lines!(ax, d.n, bw; color=c, linewidth=lw, label=method)
        end

        vlines!(ax, [l2_bytes ÷ (2 * elem_size)];
            color=:black, linestyle=:dashdot, linewidth=1, label="L2 limit")
    end

    Legend(fig[2, :], fig.content[1];
        orientation=:horizontal, tellheight=true, tellwidth=false)
    return fig
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sizes = [Int(i * 1e5) for i in 1:500]   # 100k to 50M
types = [Float32, UInt8]
trials = 10

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

function run_copy_benchmarks(sizes::Vector{Int}, types::Vector{DataType}; trials::Int=10)
    rows = NamedTuple[]

    for T in types
        src = CUDA.ones(T, sizes[end])
        dst = CuArray{T}(undef, sizes[end])

        # Single warmup per dtype on largest size
        warmup(() -> copyto!(dst, src))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=1))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=4))
        warmup(() -> KernelForge.vcopy!(dst, src; Nitem=8))

        for (idx, n) in enumerate(sizes)
            println("[$idx/$(length(sizes))] T=$T, n=$n")

            src = CUDA.ones(T, n)
            dst = CuArray{T}(undef, n)

            for (method, call) in [
                ("CUDA", () -> copyto!(dst, src)),
                ("Forge v1", () -> KernelForge.vcopy!(dst, src; Nitem=1)),
                ("Forge v4", () -> KernelForge.vcopy!(dst, src; Nitem=4)),
                ("Forge v8", () -> KernelForge.vcopy!(dst, src; Nitem=8)),
            ]
                s = bench(method, call; trials, duration=-1, exclude_copy=false)
                push!(rows, (; n, T=string(T), method,
                    s.mean_kernel_μs, s.std_kernel_μs,
                    s.mean_total_μs, s.std_total_μs))
            end
        end
    end

    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Run and save
# ---------------------------------------------------------------------------

df = run_copy_benchmarks(sizes, types; trials)

csv_path = joinpath(RESULT_DIR, "copy.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")