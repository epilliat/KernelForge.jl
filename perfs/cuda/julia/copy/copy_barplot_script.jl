#=
Copy Performance Benchmarking Script
=====================================
Compares KernelForge.vcopy! against CUDA.copyto!, KernelAbstractions, and CUB
for Float32 and UInt8 data types.

Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing
- Tests varying sizes and data types
- Results stored in DataFrame for analysis and visualization
=#

using Revise
using Pkg
Pkg.activate("$(@__DIR__)/../../")

using KernelForge
using CUDA
using KernelAbstractions
using BenchmarkTools
using Statistics
using DataFrames
using CairoMakie
using JSON3
using Printf

# Colorblind-safe palette (Wong) + dark blue for CUB
const METHOD_COLORS = Dict(
    "CUDA" => colorant"#CC79A7",   # pink/mauve
    "Forge v1" => colorant"#0072B2",   # blue
    "Forge v4" => colorant"#E69F00",   # orange (vectorized)
    "CUB" => colorant"#00008B"     # dark blue
)

# Method order: CUDA -> KernelForge v1 -> KernelForge v4 -> CUB
const METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "CUB"]

# Helper functions
function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

function sum_kernel_durations_μs(prof)
    df = prof.device
    # Sum all kernel durations (including copy kernels, which are what we're benchmarking)
    # stop - start is in seconds, convert to μs
    if isempty(df)
        return 0.0
    end
    total_s = sum(row.stop - row.start for row in eachrow(df))
    return total_s * 1e6  # convert to μs
end

function bench(name, f; duration=0.5, trials=100, backend=CUDABackend())
    warmup(f; duration)
    println("=== $name ===")

    # Profile multiple times for kernel time statistics
    kernel_times = Vector{Float64}(undef, trials)
    local prof
    for i in 1:trials
        prof = CUDA.@profile f()
        kernel_times[i] = sum_kernel_durations_μs(prof)
    end

    # Display last profile
    display(prof)

    mean_kernel_μs = mean(kernel_times)
    std_kernel_μs = std(kernel_times)

    # Measure total time using BenchmarkTools
    result = @benchmark begin
        $f()
        KernelAbstractions.synchronize($backend)
    end samples = trials evals = 1

    mean_total_μs = mean(result).time / 1000  # ns to μs
    std_total_μs = std(result).time / 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    println("Total time: $(mean_total_μs) ± $(std_total_μs) μs (n=$trials)")

    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs)
end

"""
    run_cub_benchmark(exe::String; N, iterations, warmup_ms, dtype)

Run the CUB copy benchmark and return parsed JSON results.
"""
function run_cub_benchmark(exe::String;
    N::Int=100_000_000,
    iterations::Int=100,
    warmup_ms::Real=500,
    dtype::Type=Float32
)
    dtype_str = if dtype == Float32
        "float"
    elseif dtype == Float64
        "double"
    elseif dtype == Int32
        "int"
    elseif dtype == UInt64
        "uint64"
    elseif dtype == UInt8
        "uint8"
    else
        error("Unsupported dtype: $dtype. Use Float32, Float64, Int32, UInt64, or UInt8.")
    end

    exe_path = abspath(expanduser(exe))
    cmd = `$exe_path -n $N -i $iterations -w $warmup_ms -t $dtype_str -j`
    return JSON3.read(read(cmd, String))
end

"""
    bench_cub(exe::String, n::Int, T::Type; trials=100) -> NamedTuple

Benchmark CUB copy and return stats in the same format as bench().
CUB has no CPU overhead, so kernel time == total time.
"""
function bench_cub(exe::String, n::Int, ::Type{T}; trials::Int=100) where T
    println("=== CUB ===")

    results = run_cub_benchmark(exe; N=n, iterations=trials, dtype=T)

    # Extract mean time from JSON (mean_ms -> convert to μs)
    mean_ms = results[1]["mean_ms"]
    std_ms = results[1]["std_ms"]

    mean_kernel_μs = mean_ms * 1000  # ms to μs
    std_kernel_μs = std_ms * 1000

    # CUB benchmark measures pure kernel time, no CPU overhead
    mean_total_μs = mean_kernel_μs
    std_total_μs = std_kernel_μs

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    println("Total time: $(mean_total_μs) ± $(std_total_μs) μs (n=$trials)")

    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs)
end

"""
    run_copy_benchmarks(n::Int, ::Type{T}; cub_exe::String="") -> NamedTuple

Run copy benchmarks for a given type T.
"""
function run_copy_benchmarks(n::Int, ::Type{T}; cub_exe::String="") where T
    src = CUDA.ones(T, n)
    dst = CuArray{T}(undef, n)

    println("\n" * "="^60)
    println("Copy: n=$n, T=$T")
    println("="^60)

    # KernelForge v1 (scalar)
    KernelForge_v1_stats = bench("Forge v1", () -> KernelForge.vcopy!(dst, src; Nitem=1))

    # KernelForge v4 (vectorized)
    KernelForge_v4_stats = bench("Forge v4", () -> KernelForge.vcopy!(dst, src; Nitem=4))

    # CUDA.copyto!
    cuda_stats = bench("CUDA.copyto!", () -> copyto!(dst, src))

    # CUB benchmark (if executable provided)
    cub_stats = if !isempty(cub_exe) && isfile(cub_exe)
        bench_cub(cub_exe, n, T)
    else
        @warn "CUB executable not found: $cub_exe"
        (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end

    return (KernelForge_v1=KernelForge_v1_stats, KernelForge_v4=KernelForge_v4_stats, cuda=cuda_stats, cub=cub_stats)
end

"""
    run_all_benchmarks(sizes::Vector{Int}; cub_exe) -> DataFrame

Run benchmarks for all (n, T) combinations and return results as a DataFrame.
Types: Float32 and UInt8
"""
function run_all_benchmarks(sizes::Vector{Int};
    cub_exe::String="$(@__DIR__)/../../cuda_cpp/cub_nvcc/bin/cub_memcpy_benchmark")
    rows = NamedTuple[]

    for n in sizes
        # Float32 benchmarks
        stats_f32 = run_copy_benchmarks(n, Float32; cub_exe)

        for (method, method_stats) in [("Forge v1", stats_f32.KernelForge_v1),
            ("Forge v4", stats_f32.KernelForge_v4),
            ("CUDA", stats_f32.cuda),
            ("CUB", stats_f32.cub)]
            push!(rows, (
                n=n,
                T="Float32",
                method=method,
                mean_kernel_μs=method_stats.mean_kernel_μs,
                std_kernel_μs=method_stats.std_kernel_μs,
                mean_total_μs=method_stats.mean_total_μs,
                std_total_μs=method_stats.std_total_μs
            ))
        end

        # UInt8 benchmarks
        stats_u8 = run_copy_benchmarks(n, UInt8; cub_exe)

        for (method, method_stats) in [("Forge v1", stats_u8.KernelForge_v1),
            ("Forge v4", stats_u8.KernelForge_v4),
            ("CUDA", stats_u8.cuda),
            ("CUB", stats_u8.cub)]
            push!(rows, (
                n=n,
                T="UInt8",
                method=method,
                mean_kernel_μs=method_stats.mean_kernel_μs,
                std_kernel_μs=method_stats.std_kernel_μs,
                mean_total_μs=method_stats.mean_total_μs,
                std_total_μs=method_stats.std_total_μs
            ))
        end
    end

    return DataFrame(rows)
end

"""
    run_scaling_benchmarks(sizes::Vector{Int}; cub_exe) -> DataFrame

Run benchmarks across many sizes for scaling analysis.
"""
function run_scaling_benchmarks(sizes::Vector{Int};
    cub_exe::String="$(@__DIR__)/../../cuda_cpp/cub_nvcc/bin/cub_memcpy_benchmark",
    trials::Int=50)
    rows = NamedTuple[]

    for (idx, n) in enumerate(sizes)
        println("\n[$idx/$(length(sizes))] Benchmarking n=$n")

        for T in [Float32, UInt8]
            src = CUDA.ones(T, n)
            dst = CuArray{T}(undef, n)

            # Warm-up
            warmup(() -> KernelForge.vcopy!(dst, src; Nitem=1))
            warmup(() -> KernelForge.vcopy!(dst, src; Nitem=4))
            warmup(() -> copyto!(dst, src))

            # Quick benchmark (fewer trials for scaling)
            for (method, f) in [
                ("Forge v1", () -> KernelForge.vcopy!(dst, src; Nitem=1)),
                ("Forge v4", () -> KernelForge.vcopy!(dst, src; Nitem=4)),
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
    format_3digits(x::Real) -> String

Format a number with exactly 3 significant digits.
"""
function format_3digits(x::Real)
    if x == 0
        return "0"
    end
    # Determine the order of magnitude
    mag = floor(Int, log10(abs(x)))
    # Number of decimal places needed for 3 significant digits
    decimals = max(0, 2 - mag)
    return @sprintf("%.*f", decimals, x)
end

"""
    plot_copy_comparison(df::DataFrame, n::Int; kwargs...) -> Figure

Create a grouped barplot comparing copy implementations.
Groups are data types (Float32, UInt8), bars within groups are methods (CUDA, KA, KernelForge, CUB).
Each bar is stacked: kernel time (solid) + overhead (alpha).
"""
function plot_copy_comparison(
    df::DataFrame,
    n::Int;
    title::String="Copy Performance (n = $(format_number(n)))",
    figsize::Tuple{Int,Int}=(800, 500),
    method_colors::Dict{String,<:Any}=METHOD_COLORS,
    method_order::Vector{String}=METHOD_ORDER,
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02
)
    # Filter data for the specified n
    subset = filter(row -> row.n == n, df)

    if isempty(subset)
        error("No data found for n = $n")
    end

    # Get unique types (groups on x-axis)
    types = sort(unique(subset.T))

    # Create figure
    fig = Figure(size=figsize)
    ax = Axis(
        fig[1, 1],
        ylabel="Time (μs)",
        ylabelsize=16,
        title=title,
        titlesize=20,
        xticks=(1:length(types), types),
        xticklabelsize=16
    )

    # Bar layout parameters
    n_methods = length(method_order)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = range(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, length=n_methods)

    # First pass: collect all data and find max height for consistent label spacing
    method_data = Dict{String,NamedTuple{(:kernel_vals_μs, :overhead_vals_μs, :err_vals_μs, :x),Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}}()
    max_height_μs = 0.0

    for (method_idx, method) in enumerate(method_order)
        kernel_vals_μs = Float64[]
        overhead_vals_μs = Float64[]
        err_vals_μs = Float64[]

        for T in types
            row = filter(r -> r.method == method && r.T == T, subset)
            if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                k = row.mean_kernel_μs[1]
                t = row.mean_total_μs[1]
                e = row.std_kernel_μs[1]
                push!(kernel_vals_μs, k)
                push!(overhead_vals_μs, max(0, t - k))
                push!(err_vals_μs, e)
                max_height_μs = max(max_height_μs, k + max(0, t - k) + e)
            else
                push!(kernel_vals_μs, 0.0)
                push!(overhead_vals_μs, 0.0)
                push!(err_vals_μs, 0.0)
            end
        end

        x = collect(1:length(types)) .+ offsets[method_idx]
        method_data[method] = (; kernel_vals_μs, overhead_vals_μs, err_vals_μs, x)
    end

    # Fixed offset for all labels
    fixed_offset_μs = max_height_μs * label_offset_frac

    # Second pass: plot bars and labels
    for (method_idx, method) in enumerate(method_order)
        color = method_colors[method]
        overhead_color = (color, overhead_alpha)
        data = method_data[method]

        # Kernel time bars
        barplot!(ax, data.x, data.kernel_vals_μs, width=bar_width, color=color, label="$method (kernel)")

        # Overhead bars (stacked on top) - CUB has no overhead
        if any(data.overhead_vals_μs .> 0)
            barplot!(ax, data.x, data.overhead_vals_μs, width=bar_width, color=overhead_color, offset=data.kernel_vals_μs)
        end

        # Error bars
        errorbars!(ax, data.x, data.kernel_vals_μs, data.err_vals_μs, color=:black, whiskerwidth=6)

        # Value labels with fixed spacing above bars
        # Use bold font for KernelForge v4
        label_font = method == "Forge v4" ? :bold : :regular
        for (i, (xi, ki, oi, ei)) in enumerate(zip(data.x, data.kernel_vals_μs, data.overhead_vals_μs, data.err_vals_μs))
            if ki > 0
                bar_top = ki + oi
                label_y = bar_top + fixed_offset_μs
                label_text = format_3digits(ki)
                text!(ax, xi, label_y; text=label_text, align=(:center, :bottom), fontsize=10, font=label_font)
            end
        end
    end

    # Add secondary y-axis on the right (ms)
    ylims_μs = ax.yaxis.attributes.limits[]
    if ylims_μs === nothing || ylims_μs === Makie.automatic
        autolimits!(ax)
    end

    ax_right = Axis(
        fig[1, 1],
        ylabel="Time (ms)",
        ylabelsize=16,
        yaxisposition=:right,
        yticklabelalign=(:left, :center),
        xticksvisible=false,
        xticklabelsvisible=false,
        xlabelvisible=false,
        xgridvisible=false,
        ygridvisible=false,
    )
    hidespines!(ax_right, :t, :b, :l)
    linkaxes!(ax, ax_right)

    ax_right.ytickformat = values -> [format_3digits(v / 1000) for v in values]

    # Legend
    legend_elements = []
    legend_labels = String[]
    for method in method_order
        color = method_colors[method]
        push!(legend_elements, PolyElement(color=color))
        push!(legend_labels, "$method (kernel)")
        if method != "CUB"
            push!(legend_elements, PolyElement(color=(color, overhead_alpha)))
            push!(legend_labels, "$method (overhead)")
        end
    end
    Legend(fig[1, 2], legend_elements, legend_labels, "Method")

    return fig
end

"""
    plot_copy_comparison_multi(df::DataFrame, sizes::Vector{Int}; kwargs...) -> Figure

Create multiple grouped barplots side by side, one for each n value.
Groups are data types (Float32, UInt8), bars are methods (CUDA, KA, KernelForge, CUB).
Left plot uses μs, right plot uses ms.
"""
function plot_copy_comparison_multi(
    df::DataFrame,
    sizes::Vector{Int};
    figsize::Tuple{Int,Int}=(600 * length(sizes), 450),
    method_colors::Dict{String,<:Any}=METHOD_COLORS,
    method_order::Vector{String}=METHOD_ORDER,
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02
)
    fig = Figure(size=figsize)

    types = sort(unique(df.T))
    n_methods = length(method_order)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = range(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, length=n_methods)

    for (col, n) in enumerate(sizes)
        subset = filter(row -> row.n == n, df)

        if isempty(subset)
            @warn "No data found for n = $n, skipping"
            continue
        end

        # Use ms for last column (large N), μs for first column (small N)
        use_ms = (col == length(sizes))
        unit_label = use_ms ? "Time (ms)" : "Time (μs)"
        unit_divisor = use_ms ? 1000.0 : 1.0

        ax = Axis(
            fig[1, col],
            ylabel=unit_label,
            ylabelsize=16,
            title="n = $(format_number(n))",
            titlesize=20,
            xticks=(1:length(types), types),
            xticklabelsize=14
        )

        # First pass: collect all data and find max height
        method_data = Dict{String,NamedTuple{(:kernel_vals, :overhead_vals, :err_vals, :x),Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}}()
        max_height = 0.0

        for (method_idx, method) in enumerate(method_order)
            kernel_vals = Float64[]
            overhead_vals = Float64[]
            err_vals = Float64[]

            for T in types
                row = filter(r -> r.method == method && r.T == T, subset)
                if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                    k = row.mean_kernel_μs[1] / unit_divisor
                    t = row.mean_total_μs[1] / unit_divisor
                    e = row.std_kernel_μs[1] / unit_divisor
                    push!(kernel_vals, k)
                    push!(overhead_vals, max(0, t - k))
                    push!(err_vals, e)
                    max_height = max(max_height, k + max(0, t - k) + e)
                else
                    push!(kernel_vals, 0.0)
                    push!(overhead_vals, 0.0)
                    push!(err_vals, 0.0)
                end
            end

            x = collect(1:length(types)) .+ offsets[method_idx]
            method_data[method] = (; kernel_vals, overhead_vals, err_vals, x)
        end

        # Fixed offset for all labels in this subplot
        fixed_offset = max_height * label_offset_frac

        # Second pass: plot bars and labels
        for (method_idx, method) in enumerate(method_order)
            color = method_colors[method]
            overhead_color = (color, overhead_alpha)
            data = method_data[method]

            barplot!(ax, data.x, data.kernel_vals, width=bar_width, color=color)
            if any(data.overhead_vals .> 0)
                barplot!(ax, data.x, data.overhead_vals, width=bar_width, color=overhead_color, offset=data.kernel_vals)
            end
            errorbars!(ax, data.x, data.kernel_vals, data.err_vals, color=:black, whiskerwidth=6)

            # Value labels with fixed spacing above bars
            # Use bold font for KernelForge
            label_font = method == "Forge" ? :bold : :regular
            for (i, (xi, ki, oi, ei)) in enumerate(zip(data.x, data.kernel_vals, data.overhead_vals, data.err_vals))
                if ki > 0
                    bar_top = ki + oi
                    label_y = bar_top + fixed_offset
                    label_text = format_3digits(ki)
                    text!(ax, xi, label_y; text=label_text, align=(:center, :bottom), fontsize=12, font=label_font)
                end
            end
        end
    end

    # Shared legend at bottom
    legend_elements = []
    legend_labels = String[]
    for method in method_order
        color = method_colors[method]
        push!(legend_elements, PolyElement(color=color))
        push!(legend_labels, "$method (kernel)")
        if method != "CUB"
            push!(legend_elements, PolyElement(color=(color, overhead_alpha)))
            push!(legend_labels, "$method (overhead)")
        end
    end

    Legend(
        fig[2, :],
        legend_elements,
        legend_labels,
        orientation=:horizontal,
        tellheight=true,
        tellwidth=false
    )

    return fig
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
    method_order::Vector{String}=["CUDA", "Forge v1", "Forge v4"]  # CUB excluded for scaling
)
    fig = Figure(size=figsize)

    types = ["Float32", "UInt8"]

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
            linewidth = method == "Forge v4" ? 3 : 2

            # Plot line
            lines!(ax, ns, means, color=color, linewidth=linewidth, label=method)

            # Add error band
            band!(ax, ns, means .- stds, means .+ stds, color=(color, 0.2))
        end

        # Add L2 cache line (24MB = 12M Float32 or 24M UInt8 for read+write)
        l2_size_bytes = 24 * 1024 * 1024  # 24MB typical L2 cache
        element_size = T == "Float32" ? 4 : 1
        l2_elements = l2_size_bytes ÷ (2 * element_size)  # divide by 2 for read+write

        vlines!(ax, [l2_elements], color=:black, linestyle=:dashdot, linewidth=1,
            label="L2 limit (read+write)")
    end

    # Shared legend
    Legend(
        fig[2, :],
        fig.content[1],  # Get legend from first axis
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
    method_order::Vector{String}=["CUDA", "Forge v1", "Forge v4"],
    peak_bandwidth_gb_s::Float64=1008.0  # RTX 4090 peak bandwidth
)
    fig = Figure(size=figsize)

    types = ["Float32", "UInt8"]
    element_sizes = Dict("Float32" => 4, "UInt8" => 1)

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
            # bandwidth_gb_s = bytes / (time_s * 1e9)
            elem_size = element_sizes[T]
            bytes = ns .* elem_size .* 2
            times_s = times_μs .* 1e-6
            bandwidth_gb_s = bytes ./ times_s ./ 1e9

            color = method_colors[method]
            linewidth = method == "Forge v4" ? 3 : 2

            # Plot line
            lines!(ax, ns, bandwidth_gb_s, color=color, linewidth=linewidth, label=method)
        end

        # Add peak bandwidth line
        #hlines!(ax, [peak_bandwidth_gb_s], color=:red, linestyle=:dash, linewidth=1,
        #    label="Peak bandwidth")

        # Add L2 cache line
        l2_size_bytes = 24 * 1024 * 1024
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
run_copy_benchmarks(1000, UInt8)
# run_copy_benchmarks(1000, Float32)
# Run bar chart benchmarks (1e6 and 1e8)
sizes_bar = [1_000_000, 100_000_000]
df_bar = run_all_benchmarks(sizes_bar)

# Display results
println("\n" * "="^60)
println("BENCHMARK RESULTS (Bar Charts)")
println("="^60)
show(df_bar, allrows=true, allcols=true)
println()

# Create individual plots
figures = Dict{Int,Figure}()
for n in sizes_bar
    figures[n] = plot_copy_comparison(df_bar, n)
    save("perfs/cuda/figures/benchmark/copy_benchmark_$(n).png", figures[n])
end

# Create multi-panel comparison
fig_multi = plot_copy_comparison_multi(df_bar, sizes_bar, figsize=(2 * 550, 450))
save("perfs/cuda/figures/benchmark/copy_benchmark_comparison.png", fig_multi)

#%%
# Run scaling benchmarks
sizes_scaling = [Int(i * 1e5) for i in 1:500]  # 100k to 50M
df_scaling = run_scaling_benchmarks(sizes_scaling; trials=30)

# Display scaling results
println("\n" * "="^60)
println("BENCHMARK RESULTS (Scaling)")
println("="^60)

show(df_scaling[1:min(20, nrow(df_scaling)), :], allrows=true, allcols=true)
println("... ($(nrow(df_scaling)) total rows)")

# Create scaling plot
save("perfs/cuda/figures/benchmark/copy_scaling.png", fig_scaling)