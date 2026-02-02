#=
Copy Scaling Benchmarking Script
=================================
Benchmarks copy performance as a function of array size.
Compares Luma v1/v4/v8 against CUDA.copyto!

Methodology:
- 500ms warm-up phase
- CUDA.@profile for accurate kernel timing
- Tests varying sizes for scaling analysis
=#

using Revise
using Pkg
Pkg.activate("$(@__DIR__)/../../")

using Luma
using CUDA
using KernelAbstractions
using Statistics
using DataFrames
using CairoMakie
using Printf

# Colorblind-safe palette
const METHOD_COLORS = Dict(
    "CUDA" => colorant"#CC79A7",      # pink/mauve
    "Luma v1" => colorant"#0072B2",   # blue
    "Luma v4" => colorant"#E69F00",   # orange
    "Luma v8" => colorant"#009E73"    # bluish green
)

const METHOD_ORDER = ["CUDA", "Luma v1", "Luma v4", "Luma v8"]

# Get L2 cache size in bytes from CUDA device
function get_l2_cache_size()
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
end

function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

function sum_kernel_durations_μs(prof)
    df = prof.device
    if isempty(df)
        return 0.0
    end
    total_s = sum(row.stop - row.start for row in eachrow(df))
    return total_s * 1e6
end

"""
    format_number(n::Int) -> String

Format large numbers with underscores for readability.
"""
function format_number(n::Int)
    s = string(n)
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, "_")
end

"""
    run_scaling_benchmarks(sizes::Vector{Int}; trials=10) -> DataFrame

Run benchmarks across many sizes for scaling analysis.
"""
function run_scaling_benchmarks(sizes::Vector{Int}; trials::Int=10)
    rows = NamedTuple[]

    for (idx, n) in enumerate(sizes)
        println("\n[$idx/$(length(sizes))] Benchmarking n=$n")

        for T in [Float32, UInt8]
            src = CUDA.ones(T, n)
            dst = CuArray{T}(undef, n)

            # Warm-up
            warmup(() -> Luma.vcopy!(dst, src; Nitem=1))
            warmup(() -> Luma.vcopy!(dst, src; Nitem=4))
            warmup(() -> Luma.vcopy!(dst, src; Nitem=8))
            warmup(() -> copyto!(dst, src))

            # Benchmark
            for (method, f) in [
                ("Luma v1", () -> Luma.vcopy!(dst, src; Nitem=1)),
                ("Luma v4", () -> Luma.vcopy!(dst, src; Nitem=4)),
                ("Luma v8", () -> Luma.vcopy!(dst, src; Nitem=8)),
                ("CUDA", () -> copyto!(dst, src))
            ]
                kernel_times = Float64[]
                for _ in 1:trials
                    prof = CUDA.@profile f()
                    push!(kernel_times, sum_kernel_durations_μs(prof))
                end

                push!(rows, (
                    n=n,
                    T=string(T),
                    method=method,
                    mean_kernel_μs=mean(kernel_times),
                    std_kernel_μs=std(kernel_times)
                ))
            end
        end
    end

    return DataFrame(rows)
end

"""
    plot_scaling(df::DataFrame; kwargs...) -> Figure

Create a plot showing copy performance as a function of array size.
Two panels: Float32 and UInt8, with log-log scale.
"""
function plot_scaling(
    df::DataFrame;
    figsize::Tuple{Int,Int}=(1000, 450),
    method_colors::Dict{String,<:Any}=METHOD_COLORS,
    method_order::Vector{String}=METHOD_ORDER
)
    fig = Figure(size=figsize)

    types = ["Float32", "UInt8"]
    l2_size_bytes = get_l2_cache_size()

    for (col, T) in enumerate(types)
        subset = filter(row -> row.T == T, df)

        if isempty(subset)
            @warn "No data found for T = $T, skipping"
            continue
        end

        ax = Axis(
            fig[1, col],
            xlabel="Array size (elements)",
            ylabel="Time (μs)",
            xscale=log10,
            yscale=log10,
            title="Copy Performance: $T",
            titlesize=18,
            xlabelsize=14,
            ylabelsize=14
        )

        for method in method_order
            method_subset = filter(r -> r.method == method, subset)
            if isempty(method_subset)
                continue
            end

            # Sort by n
            sorted_data = sort(method_subset, :n)
            ns = sorted_data.n
            means = sorted_data.mean_kernel_μs
            stds = sorted_data.std_kernel_μs

            color = method_colors[method]
            linewidth = method == "Luma v8" ? 3 : 2

            # Plot line
            lines!(ax, ns, means, color=color, linewidth=linewidth, label=method)

            # Add error band
            band!(ax, ns, means .- stds, means .+ stds, color=(color, 0.2))
        end

        # Add L2 cache line (auto-detected)
        element_size = T == "Float32" ? 4 : 1
        l2_elements = l2_size_bytes ÷ (2 * element_size)  # divide by 2 for read+write

        vlines!(ax, [l2_elements], color=:black, linestyle=:dashdot, linewidth=1,
            label="L2 limit (read+write)")
    end

    # Shared legend
    Legend(
        fig[2, :],
        fig.content[1],
        orientation=:horizontal,
        tellheight=true,
        tellwidth=false
    )

    return fig
end

"""
    plot_bandwidth(df::DataFrame; kwargs...) -> Figure

Create a plot showing achieved memory bandwidth as a function of array size.
"""
function plot_bandwidth(
    df::DataFrame;
    figsize::Tuple{Int,Int}=(1000, 450),
    method_colors::Dict{String,<:Any}=METHOD_COLORS,
    method_order::Vector{String}=METHOD_ORDER,
    peak_bandwidth_gb_s::Float64=1008.0  # RTX 4090 peak bandwidth
)
    fig = Figure(size=figsize)

    types = ["Float32", "UInt8"]
    element_sizes = Dict("Float32" => 4, "UInt8" => 1)
    l2_size_bytes = get_l2_cache_size()

    for (col, T) in enumerate(types)
        subset = filter(row -> row.T == T, df)

        if isempty(subset)
            @warn "No data found for T = $T, skipping"
            continue
        end

        ax = Axis(
            fig[1, col],
            xlabel="Array size (elements)",
            ylabel="Bandwidth (GB/s)",
            xscale=log10,
            title="Memory Bandwidth: $T",
            titlesize=18,
            xlabelsize=14,
            ylabelsize=14
        )

        for method in method_order
            method_subset = filter(r -> r.method == method, subset)
            if isempty(method_subset)
                continue
            end

            # Sort by n
            sorted_data = sort(method_subset, :n)
            ns = sorted_data.n
            times_μs = sorted_data.mean_kernel_μs

            # Calculate bandwidth: bytes = n * element_size * 2 (read + write)
            elem_size = element_sizes[T]
            bytes = ns .* elem_size .* 2
            times_s = times_μs .* 1e-6
            bandwidth_gb_s = bytes ./ times_s ./ 1e9

            color = method_colors[method]
            linewidth = method == "Luma v8" ? 3 : 2

            # Plot line
            lines!(ax, ns, bandwidth_gb_s, color=color, linewidth=linewidth, label=method)
        end

        # Add peak bandwidth line
        #hlines!(ax, [peak_bandwidth_gb_s], color=:red, linestyle=:dash, linewidth=1,
        #    label="Peak bandwidth")

        # Add L2 cache line (auto-detected)
        elem_size = element_sizes[T]
        l2_elements = l2_size_bytes ÷ (2 * elem_size)

        vlines!(ax, [l2_elements], color=:black, linestyle=:dashdot, linewidth=1,
            label="L2 limit")
    end

    # Shared legend
    Legend(
        fig[2, :],
        fig.content[1],
        orientation=:horizontal,
        tellheight=true,
        tellwidth=false
    )

    return fig
end

#%%
#=============================================================================
  Main execution
=============================================================================#

# Run scaling benchmarks
sizes_scaling = [Int(i * 1e5) for i in 1:500]  # 100k to 50M
df_scaling = run_scaling_benchmarks(sizes_scaling; trials=10)

using CSV
CSV.write("perfs/cuda/julia/copy/scaling.csv", df_scaling)

# Display scaling results
println("\n" * "="^60)
println("BENCHMARK RESULTS (Scaling)")
println("="^60)
show(df_scaling[1:min(20, nrow(df_scaling)), :], allrows=true, allcols=true)
println("... ($(nrow(df_scaling)) total rows)")

# Create scaling plot
fig_scaling = plot_scaling(df_scaling)
save("perfs/cuda/figures/benchmark/copy_scaling.png", fig_scaling)

# Create bandwidth plot
fig_bandwidth = plot_bandwidth(df_scaling)
save("perfs/cuda/figures/benchmark/copy_bandwidth.png", fig_bandwidth)

@info "Scaling benchmarks complete. Access results via `df_scaling`"


