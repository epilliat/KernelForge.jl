#=
MatVec Performance Benchmarking Script
======================================
Compares KernelForge.matvec! against cuBLAS (via A * x) for matrix-vector multiplication.

Methodology:
- 500ms warm-up phase to ensure JIT compilation and GPU initialization
- CUDA.@profile for accurate kernel timing
- Tests varying aspect ratios with fixed total elements
- Results stored in DataFrame for analysis and visualization
=#

using Revise
using Pkg
Pkg.activate("perfs/cuda")

using KernelForge
using CUDA
using KernelAbstractions
using BenchmarkTools
using Statistics
using DataFrames
using CairoMakie
using Printf

# Helper functions
function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

function sum_kernel_durations_μs(prof)
    df = prof.device
    # Filter out copy operations
    kernels = filter(row -> !startswith(row.name, "[copy"), df)
    # Sum durations: stop - start is in seconds, convert to μs
    total_s = sum(row.stop - row.start for row in eachrow(kernels))
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

function run_matvec_benchmarks(n::Int, p::Int; tmp=nothing, FlagType=UInt8)
    T = Float32
    A = CUDA.ones(T, n, p)
    x = CuArray{T}(1:p)
    dst = CUDA.zeros(T, n)

    println("\n" * "="^60)
    println("n=$n, p=$p  (n×p = $(n*p))")
    println("="^60)
    cublas_stats = bench("cuBLAS (A * x)", () -> A * x)
    KernelForge_stats = bench("KernelForge.matvec!", () -> KernelForge.matvec(*, +, A, x; FlagType=FlagType, tmp=tmp))

    return (KernelForge=KernelForge_stats, cublas=cublas_stats)
end

"""
    run_all_benchmarks(sizes::Vector{Int}) -> DataFrame

Run benchmarks for all (n, p) combinations and return results as a DataFrame.

# Arguments
- `sizes`: Vector of total element counts (n × p) to test

# Returns
DataFrame with columns:
- `total_elements`: n × p
- `n`: number of rows
- `p`: vector length
- `method`: "Forge" or "cuBLAS"
- `mean_kernel_μs`: mean kernel time in microseconds
- `std_kernel_μs`: standard deviation of kernel time
- `mean_total_μs`: mean total time in microseconds
- `std_total_μs`: standard deviation of total time
"""
function run_all_benchmarks(sizes::Vector{Int}; tmp=nothing, FlagType=UInt8)
    rows = NamedTuple[]

    for total in sizes
        # Generate p values that divide total evenly
        p_values = filter(p -> total % p == 0, [10, 100, 1000, 10_000, 100_000, 1_000_000])
        p_values = filter(p -> p <= total, p_values)

        for p in p_values
            n = total ÷ p
            stats = run_matvec_benchmarks(n, p; tmp=tmp, FlagType=FlagType)

            # Add KernelForge result
            push!(rows, (
                total_elements=total,
                n=n,
                p=p,
                method="Forge",
                mean_kernel_μs=stats.KernelForge.mean_kernel_μs,
                std_kernel_μs=stats.KernelForge.std_kernel_μs,
                mean_total_μs=stats.KernelForge.mean_total_μs,
                std_total_μs=stats.KernelForge.std_total_μs
            ))

            # Add cuBLAS result
            push!(rows, (
                total_elements=total,
                n=n,
                p=p,
                method="cuBLAS",
                mean_kernel_μs=stats.cublas.mean_kernel_μs,
                std_kernel_μs=stats.cublas.std_kernel_μs,
                mean_total_μs=stats.cublas.mean_total_μs,
                std_total_μs=stats.cublas.std_total_μs
            ))
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
    plot_kernel_comparison(df::DataFrame, total_elements::Int; df2=nothing, kwargs...) -> Figure

Create a grouped barplot comparing KernelForge vs cuBLAS kernel times for a given n×p.
Each bar is stacked: kernel time (solid color) + overhead (same color with alpha).

If `df2` is provided, an additional black rectangle is shown above the KernelForge kernel time
representing the overhead from the alternative configuration (tmp=tmp, FlagType=UInt64).

# Arguments
- `df`: DataFrame from `run_all_benchmarks` (default configuration)
- `total_elements`: The n×p value to plot

# Keyword Arguments
- `df2`: Optional DataFrame from `run_all_benchmarks` with alternative config (tmp=tmp, FlagType=UInt64)
- `title`: Custom title (default: auto-generated)
- `figsize`: Figure size tuple (default: (800, 500))
- `colors`: Tuple of colors for (KernelForge, cuBLAS) (default: (:steelblue, :coral))
- `overhead_alpha`: Alpha value for the overhead portion (default: 0.3)
- `label_offset_frac`: Fraction of max height for label offset (default: 0.02)

# Returns
CairoMakie Figure object
"""
function plot_kernel_comparison(
    df::DataFrame,
    total_elements::Int;
    df2::Union{DataFrame,Nothing}=nothing,
    title::String="Kernel Time Comparison (n×p = $(format_number(total_elements)))",
    figsize::Tuple{Int,Int}=(800, 500),
    colors::Tuple=(:steelblue, :coral),
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02
)
    # Filter data for the specified total_elements
    subset = filter(row -> row.total_elements == total_elements, df)

    if isempty(subset)
        error("No data found for total_elements = $total_elements")
    end

    # Get unique p values and sort them
    p_values = sort(unique(subset.p))

    # Prepare data for plotting
    KernelForge_kernel = Float64[]
    KernelForge_overhead = Float64[]
    KernelForge_err = Float64[]
    cublas_kernel = Float64[]
    cublas_overhead = Float64[]
    cublas_err = Float64[]

    for p in p_values
        KernelForge_row = filter(r -> r.p == p && r.method == "Forge", subset)
        cublas_row = filter(r -> r.p == p && r.method == "cuBLAS", subset)

        KernelForge_k = KernelForge_row.mean_kernel_μs[1]
        KernelForge_t = KernelForge_row.mean_total_μs[1]
        cublas_k = cublas_row.mean_kernel_μs[1]
        cublas_t = cublas_row.mean_total_μs[1]

        push!(KernelForge_kernel, KernelForge_k)
        push!(KernelForge_overhead, max(0, KernelForge_t - KernelForge_k))  # Ensure non-negative
        push!(KernelForge_err, KernelForge_row.std_kernel_μs[1])

        push!(cublas_kernel, cublas_k)
        push!(cublas_overhead, max(0, cublas_t - cublas_k))
        push!(cublas_err, cublas_row.std_kernel_μs[1])
    end

    # Prepare df2 data if provided (alternative KernelForge overhead)
    KernelForge_alt_overhead = Float64[]
    if df2 !== nothing
        subset2 = filter(row -> row.total_elements == total_elements, df2)
        for p in p_values
            KernelForge_row2 = filter(r -> r.p == p && r.method == "Forge", subset2)
            if !isempty(KernelForge_row2)
                KernelForge_t2 = KernelForge_row2.mean_total_μs[1]
                KernelForge_k2 = KernelForge_row2.mean_kernel_μs[1]
                push!(KernelForge_alt_overhead, max(0, KernelForge_t2 - KernelForge_k2))
            else
                push!(KernelForge_alt_overhead, 0.0)
            end
        end
    end

    # Calculate max height for consistent label spacing
    max_height = 0.0
    for i in 1:length(p_values)
        KernelForge_total = KernelForge_kernel[i] + KernelForge_overhead[i] + KernelForge_err[i]
        if df2 !== nothing && i <= length(KernelForge_alt_overhead)
            KernelForge_total = max(KernelForge_total, KernelForge_kernel[i] + KernelForge_alt_overhead[i] + KernelForge_err[i])
        end
        max_height = max(max_height, KernelForge_total)
        max_height = max(max_height, cublas_kernel[i] + cublas_overhead[i] + cublas_err[i])
    end
    fixed_offset = max_height * label_offset_frac

    # Create figure
    fig = Figure(size=figsize)
    ax = Axis(
        fig[1, 1],
        xlabel="p (vector length)",
        ylabel="Time (μs)",
        title=title,
        xticks=(1:length(p_values), string.(p_values)),
        xticklabelrotation=π / 6,
        xlabelsize=18,
        ylabelsize=18,
        titlesize=20
    )

    # Bar positions
    x = collect(1:length(p_values))
    width = 0.35
    offset = width / 2

    # Convert colors to RGBA for alpha support
    KernelForge_color = Makie.to_color(colors[1])
    cublas_color = Makie.to_color(colors[2])
    KernelForge_overhead_color = (KernelForge_color, overhead_alpha)
    cublas_overhead_color = (cublas_color, overhead_alpha)

    # Plot stacked bars: kernel time (bottom) + overhead (top)
    # KernelForge bars
    barplot!(ax, x .- offset, KernelForge_kernel, width=width, color=colors[1], label="Forge (kernel)")
    barplot!(ax, x .- offset, KernelForge_overhead, width=width, color=KernelForge_overhead_color,
        offset=KernelForge_kernel, label="Forge (overhead)")

    # cuBLAS bars
    barplot!(ax, x .+ offset, cublas_kernel, width=width, color=colors[2], label="cuBLAS (kernel)")
    barplot!(ax, x .+ offset, cublas_overhead, width=width, color=cublas_overhead_color,
        offset=cublas_kernel, label="cuBLAS (overhead)")

    # Add alternative overhead (horizontal lines) if df2 is provided
    if df2 !== nothing && !isempty(KernelForge_alt_overhead)
        # Draw horizontal lines at kernel + alt_overhead level
        for (i, xi) in enumerate(x)
            alt_top = KernelForge_kernel[i] + KernelForge_alt_overhead[i]
            lines!(ax, [xi - offset - width / 2, xi - offset + width / 2], [alt_top, alt_top],
                color=:black, linewidth=2.5, label=(i == 1 ? "Forge (overhead, Opt)" : nothing))
        end
    end

    # Error bars on kernel time (at the top of the kernel portion)
    errorbars!(ax, x .- offset, KernelForge_kernel, KernelForge_err, color=:black, whiskerwidth=6)
    errorbars!(ax, x .+ offset, cublas_kernel, cublas_err, color=:black, whiskerwidth=6)

    # Value labels above bars
    for (i, xi) in enumerate(x)
        # KernelForge label - position above the taller of the two overheads
        KernelForge_top = KernelForge_kernel[i] + KernelForge_overhead[i]
        if df2 !== nothing && i <= length(KernelForge_alt_overhead)
            KernelForge_top = max(KernelForge_top, KernelForge_kernel[i] + KernelForge_alt_overhead[i])
        end
        text!(ax, xi - offset, KernelForge_top + fixed_offset;
            text=format_3digits(KernelForge_kernel[i]), align=(:center, :bottom), fontsize=12, font=:bold)

        # cuBLAS label
        cublas_top = cublas_kernel[i] + cublas_overhead[i]
        text!(ax, xi + offset, cublas_top + fixed_offset;
            text=format_3digits(cublas_kernel[i]), align=(:center, :bottom), fontsize=12)
    end

    # Add legend
    axislegend(ax, position=:rt, labelsize=14)

    return fig
end

"""
    plot_kernel_comparison_multi(df::DataFrame, total_elements_list::Vector{Int}; df2=nothing, kwargs...) -> Figure

Create multiple grouped barplots side by side, one for each total_elements value.

If `df2` is provided, an additional black rectangle is shown above the KernelForge kernel time
representing the overhead from the alternative configuration (tmp=tmp, FlagType=UInt64).

# Arguments
- `df`: DataFrame from `run_all_benchmarks`
- `total_elements_list`: Vector of n×p values to plot

# Keyword Arguments
- `df2`: Optional DataFrame from `run_all_benchmarks` with alternative config (tmp=tmp, FlagType=UInt64)
- `figsize`: Figure size tuple (default: auto-scaled by number of plots)
- `colors`: Tuple of colors for (KernelForge, cuBLAS) (default: (:steelblue, :coral))
- `overhead_alpha`: Alpha value for the overhead portion (default: 0.3)
- `label_offset_frac`: Fraction of max height for label offset (default: 0.02)

# Returns
CairoMakie Figure object
"""
function plot_kernel_comparison_multi(
    df::DataFrame,
    total_elements_list::Vector{Int};
    df2::Union{DataFrame,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(500 * length(total_elements_list), 450),
    colors::Tuple=(:steelblue, :coral),
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02
)
    fig = Figure(size=figsize)

    # Convert colors to RGBA for alpha support
    KernelForge_color = Makie.to_color(colors[1])
    cublas_color = Makie.to_color(colors[2])
    KernelForge_overhead_color = (KernelForge_color, overhead_alpha)
    cublas_overhead_color = (cublas_color, overhead_alpha)

    local barplots  # For legend reference

    for (col, total_elements) in enumerate(total_elements_list)
        subset = filter(row -> row.total_elements == total_elements, df)

        if isempty(subset)
            @warn "No data found for total_elements = $total_elements, skipping"
            continue
        end

        p_values = sort(unique(subset.p))

        # Prepare data
        KernelForge_kernel = Float64[]
        KernelForge_overhead = Float64[]
        KernelForge_err = Float64[]
        cublas_kernel = Float64[]
        cublas_overhead = Float64[]
        cublas_err = Float64[]

        for p in p_values
            KernelForge_row = filter(r -> r.p == p && r.method == "Forge", subset)
            cublas_row = filter(r -> r.p == p && r.method == "cuBLAS", subset)

            KernelForge_k = KernelForge_row.mean_kernel_μs[1]
            KernelForge_t = KernelForge_row.mean_total_μs[1]
            cublas_k = cublas_row.mean_kernel_μs[1]
            cublas_t = cublas_row.mean_total_μs[1]

            push!(KernelForge_kernel, KernelForge_k)
            push!(KernelForge_overhead, max(0, KernelForge_t - KernelForge_k))
            push!(KernelForge_err, KernelForge_row.std_kernel_μs[1])

            push!(cublas_kernel, cublas_k)
            push!(cublas_overhead, max(0, cublas_t - cublas_k))
            push!(cublas_err, cublas_row.std_kernel_μs[1])
        end

        # Prepare df2 data if provided (alternative KernelForge overhead)
        KernelForge_alt_overhead = Float64[]
        if df2 !== nothing
            subset2 = filter(row -> row.total_elements == total_elements, df2)
            for p in p_values
                KernelForge_row2 = filter(r -> r.p == p && r.method == "Forge", subset2)
                if !isempty(KernelForge_row2)
                    KernelForge_t2 = KernelForge_row2.mean_total_μs[1]
                    KernelForge_k2 = KernelForge_row2.mean_kernel_μs[1]
                    push!(KernelForge_alt_overhead, max(0, KernelForge_t2 - KernelForge_k2))
                else
                    push!(KernelForge_alt_overhead, 0.0)
                end
            end
        end

        # Calculate max height for consistent label spacing
        max_height = 0.0
        for i in 1:length(p_values)
            KernelForge_total = KernelForge_kernel[i] + KernelForge_overhead[i] + KernelForge_err[i]
            if df2 !== nothing && i <= length(KernelForge_alt_overhead)
                KernelForge_total = max(KernelForge_total, KernelForge_kernel[i] + KernelForge_alt_overhead[i] + KernelForge_err[i])
            end
            max_height = max(max_height, KernelForge_total)
            max_height = max(max_height, cublas_kernel[i] + cublas_overhead[i] + cublas_err[i])
        end
        fixed_offset = max_height * label_offset_frac

        # Create axis
        ax = Axis(
            fig[1, col],
            xlabel="p (vector length)",
            ylabel=col == 1 ? "Time (μs)" : "",
            title="n×p = $(format_number(total_elements))",
            xticks=(1:length(p_values), string.(p_values)),
            xticklabelrotation=π / 6,
            xlabelsize=16,
            ylabelsize=16,
            titlesize=18
        )

        # Bar positions
        x = collect(1:length(p_values))
        width = 0.35
        offset = width / 2

        # Plot stacked bars
        b1 = barplot!(ax, x .- offset, KernelForge_kernel, width=width, color=colors[1])
        barplot!(ax, x .- offset, KernelForge_overhead, width=width, color=KernelForge_overhead_color, offset=KernelForge_kernel)

        b2 = barplot!(ax, x .+ offset, cublas_kernel, width=width, color=colors[2])
        barplot!(ax, x .+ offset, cublas_overhead, width=width, color=cublas_overhead_color, offset=cublas_kernel)

        # Add alternative overhead (horizontal lines) if df2 is provided
        if df2 !== nothing && !isempty(KernelForge_alt_overhead)
            for (i, xi) in enumerate(x)
                alt_top = KernelForge_kernel[i] + KernelForge_alt_overhead[i]
                lines!(ax, [xi - offset - width / 2, xi - offset + width / 2], [alt_top, alt_top],
                    color=:black, linewidth=2.5)
            end
        end

        # Error bars on kernel time
        errorbars!(ax, x .- offset, KernelForge_kernel, KernelForge_err, color=:black, whiskerwidth=6)
        errorbars!(ax, x .+ offset, cublas_kernel, cublas_err, color=:black, whiskerwidth=6)

        # Value labels above bars
        for (i, xi) in enumerate(x)
            # KernelForge label - position above the taller of the two overheads
            KernelForge_top = KernelForge_kernel[i] + KernelForge_overhead[i]
            if df2 !== nothing && i <= length(KernelForge_alt_overhead)
                KernelForge_top = max(KernelForge_top, KernelForge_kernel[i] + KernelForge_alt_overhead[i])
            end
            text!(ax, xi - offset, KernelForge_top + fixed_offset;
                text=format_3digits(KernelForge_kernel[i]), align=(:center, :bottom), fontsize=12, font=:bold)

            # cuBLAS label
            cublas_top = cublas_kernel[i] + cublas_overhead[i]
            text!(ax, xi + offset, cublas_top + fixed_offset;
                text=format_3digits(cublas_kernel[i]), align=(:center, :bottom), fontsize=12)
        end

        barplots = (b1, b2)
    end

    # Shared legend at bottom - include Opt overhead if df2 provided
    if df2 !== nothing
        Legend(
            fig[2, :],
            [PolyElement(color=colors[1]), PolyElement(color=(Makie.to_color(colors[1]), overhead_alpha)),
                LineElement(color=:black, linewidth=2.5),
                PolyElement(color=colors[2]), PolyElement(color=(Makie.to_color(colors[2]), overhead_alpha))],
            ["Forge (kernel)", "Forge (overhead)", "Forge (overhead, Opt)",
                "cuBLAS (kernel)", "cuBLAS (overhead)"],
            orientation=:horizontal,
            tellheight=true,
            tellwidth=false,
            labelsize=16
        )
    else
        Legend(
            fig[2, :],
            [PolyElement(color=colors[1]), PolyElement(color=(Makie.to_color(colors[1]), overhead_alpha)),
                PolyElement(color=colors[2]), PolyElement(color=(Makie.to_color(colors[2]), overhead_alpha))],
            ["Forge (kernel)", "Forge (overhead)", "cuBLAS (kernel)", "cuBLAS (overhead)"],
            orientation=:horizontal,
            tellheight=true,
            tellwidth=false,
            labelsize=16
        )
    end

    return fig
end

"""
    plot_all_comparisons(df::DataFrame; df2=nothing, kwargs...) -> Dict{Int, Figure}

Create barplots for all unique total_elements values in the DataFrame.

# Arguments
- `df`: DataFrame from `run_all_benchmarks`
- `df2`: Optional DataFrame from `run_all_benchmarks` with alternative config

# Returns
Dictionary mapping total_elements to Figure objects
"""
function plot_all_comparisons(df::DataFrame; df2::Union{DataFrame,Nothing}=nothing, kwargs...)
    totals = sort(unique(df.total_elements))
    figures = Dict{Int,Figure}()

    for total in totals
        figures[total] = plot_kernel_comparison(df, total; df2=df2, kwargs...)
    end

    return figures
end

"""
    save_plots(figures::Dict{Int, Figure}, prefix::String="matvec_benchmark")

Save all figures to PNG files.
"""
function save_plots(figures::Dict{Int,Figure}, prefix::String="matvec_benchmark")
    for (total, fig) in figures
        filename = "$(prefix)_$(total).png"
        save(filename, fig)
        println("Saved: $filename")
    end
end

#=============================================================================
  Main execution
=============================================================================#
#%%
tmp = CUDA.zeros(UInt8, 8 * 100000)
run_matvec_benchmarks(1, 1000000; tmp=tmp)

# Run benchmarks
sizes = [1_000_000, 10_000_000]
df = run_all_benchmarks(sizes)

# Run alternative benchmarks with tmp and FlagType=UInt64
#df2 = run_all_benchmarks(sizes; tmp=tmp)

# Display results
println("\n" * "="^60)
println("BENCHMARK RESULTS (default)")
println("="^60)
show(df, allrows=true, allcols=true)
println()

println("\n" * "="^60)
println("BENCHMARK RESULTS (tmp=tmp, FlagType=UInt64)")
println("="^60)
#show(df2, allrows=true, allcols=true)
println()

# Create and save plots with comparison
figures = plot_all_comparisons(df)
save_plots(figures)

# Create multi-panel comparison with both configurations
fig_multi = plot_kernel_comparison_multi(df, sizes)
save("perfs/cuda/figures/benchmark/matvec_benchmark_comparison.png", fig_multi)

@info "Benchmarks complete. Access results via `df`, `df2`, and figures via `figures`"