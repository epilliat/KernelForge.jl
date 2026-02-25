#=
Benchmark Helper Utilities
===========================
Shared functions for KernelForge GPU benchmarking scripts.
All benchmark scripts (copy, mapreduce, scan, matvec, vecmat) include this file.
=#

using Pkg
Pkg.activate("$(@__DIR__)/../../")  # adjust if needed

using Revise
using KernelForge
using KernelForge: UnitFloat8
using CUDA
using KernelAbstractions
using BenchmarkTools
using Statistics
using DataFrames
using CairoMakie
using AcceleratedKernels
using JSON3
using Printf

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong) + dark blue for CUB
# ---------------------------------------------------------------------------

const BENCH_COLORS = Dict(
    "CUDA"     => colorant"#CC79A7",   # pink/mauve
    "AK"       => colorant"#009E73",   # bluish green
    "Forge"    => colorant"#0072B2",   # blue
    "Forge v1" => colorant"#0072B2",   # blue  (copy script uses v1/v4 variants)
    "Forge v4" => colorant"#E69F00",   # orange
    "cuBLAS"   => colorant"#56B4E9",   # sky blue
    "CUB"      => colorant"#00008B",   # dark blue
)

# Method orders per operation (can be overridden per script)
const COPY_METHOD_ORDER     = ["CUDA", "Forge v1", "Forge v4", "CUB"]
const REDUCE_METHOD_ORDER   = ["CUDA", "AK", "Forge", "CUB"]  # used by mapreduce & scan
const COPY_METHOD_COLORS    = Dict(k => BENCH_COLORS[k] for k in COPY_METHOD_ORDER)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

"""
    format_number(n::Int) -> String

Format large integers with underscores for readability (e.g. 1_000_000).
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
    x == 0 && return "0"
    mag = floor(Int, log10(abs(x)))
    decimals = max(0, 2 - mag)
    return @sprintf("%.*f", decimals, x)
end

# ---------------------------------------------------------------------------
# GPU benchmarking primitives
# ---------------------------------------------------------------------------

"""
    warmup(f; duration=0.5)

Repeatedly call `f()` for `duration` seconds to warm up JIT and GPU state.
"""
function warmup(f; duration=0.5)
    start = time()
    while time() - start < duration
        CUDA.@sync f()
    end
end

"""
    sum_kernel_durations_μs(prof; exclude_copy=true) -> Float64

Sum all kernel durations from a CUDA profile, returning microseconds.
Pass `exclude_copy=false` to include `[copy*]` entries (needed for copy benchmarks).
"""
function sum_kernel_durations_μs(prof; exclude_copy=true)
    df = prof.device
    isempty(df) && return 0.0
    rows = exclude_copy ? filter(r -> !startswith(r.name, "[copy"), df) : df
    return sum(r.stop - r.start for r in eachrow(rows)) * 1e6
end

"""
    bench(name, f; duration=0.5, trials=100, backend=CUDABackend(), exclude_copy=true)
        -> NamedTuple{mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs}

Benchmark `f()` by:
- Warming up for `duration` seconds.
- Profiling `trials` times for kernel time via `CUDA.@profile`.
- Timing `trials` samples with `BenchmarkTools` for total wall-clock time.
"""
function bench(name, f;
    duration=0.5,
    trials=100,
    backend=CUDABackend(),
    exclude_copy=true
)
    warmup(f; duration)
    println("=== $name ===")

    kernel_times = Vector{Float64}(undef, trials)
    local prof
    for i in 1:trials
        prof = CUDA.@profile f()
        kernel_times[i] = sum_kernel_durations_μs(prof; exclude_copy)
    end
    display(prof)

    mean_kernel_μs = mean(kernel_times)
    std_kernel_μs  = std(kernel_times)

    result = @benchmark begin
        $f()
        KernelAbstractions.synchronize($backend)
    end samples=trials evals=1

    mean_total_μs = mean(result).time / 1000
    std_total_μs  = std(result).time / 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    println("Total time:  $(mean_total_μs) ± $(std_total_μs) μs (n=$trials)")

    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs)
end

# ---------------------------------------------------------------------------
# CUB benchmark runner
# ---------------------------------------------------------------------------

const _DTYPE_STRINGS = Dict(
    Float32 => "float",
    Float64 => "double",
    Int32   => "int",
    UInt64  => "uint64",
    UInt8   => "uint8",
)

"""
    run_cub_benchmark(exe; N, iterations, warmup_ms, dtype, extra_flags="") -> JSON3 result

Run an external CUB benchmark executable and return parsed JSON.
`extra_flags` can pass additional CLI arguments (e.g. `-m inclusive` for scan).
"""
function run_cub_benchmark(exe::String;
    N::Int=100_000_000,
    iterations::Int=100,
    warmup_ms::Real=500,
    dtype::Type=Float32,
    extra_flags::String=""
)
    dtype_str = get(_DTYPE_STRINGS, dtype, nothing)
    dtype_str === nothing && error("Unsupported dtype: $dtype")

    exe_path = abspath(expanduser(exe))
    base_cmd = `$exe_path -n $N -i $iterations -w $warmup_ms -t $dtype_str -j`
    cmd = isempty(extra_flags) ? base_cmd : Cmd([base_cmd.exec..., split(extra_flags)...])
    return JSON3.read(read(cmd, String))
end

"""
    bench_cub(exe, n, T; trials=100, extra_flags="")
        -> NamedTuple{mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs}

Run a CUB benchmark and return stats compatible with `bench()`.
CUB measures pure kernel time, so kernel time == total time.
"""
function bench_cub(exe::String, n::Int, ::Type{T}; trials::Int=100, extra_flags::String="") where T
    println("=== CUB ===")
    results = run_cub_benchmark(exe; N=n, iterations=trials, dtype=T, extra_flags)

    mean_kernel_μs = results[1]["mean_ms"] * 1000
    std_kernel_μs  = results[1]["std_ms"]  * 1000

    println("Kernel time: $(mean_kernel_μs) ± $(std_kernel_μs) μs (n=$trials)")
    return (; mean_kernel_μs, std_kernel_μs, mean_total_μs=mean_kernel_μs, std_total_μs=std_kernel_μs)
end

"""
    bench_cub_or_nan(exe, n, T; kw...) -> NamedTuple

Like `bench_cub` but returns NaN stats if the executable is missing/empty.
"""
function bench_cub_or_nan(exe::String, n::Int, ::Type{T}; kw...) where T
    if !isempty(exe) && isfile(exe)
        return bench_cub(exe, n, T; kw...)
    else
        isempty(exe) || @warn "CUB executable not found: $exe"
        return (; mean_kernel_μs=NaN, std_kernel_μs=NaN, mean_total_μs=NaN, std_total_μs=NaN)
    end
end

# ---------------------------------------------------------------------------
# DataFrame row builder
# ---------------------------------------------------------------------------

"""
    push_stats!(rows, stats; n, T, method)

Append a benchmark result NamedTuple to `rows` with metadata fields.
"""
function push_stats!(rows, stats; n, T, method)
    push!(rows, (;
        n, T=string(T), method,
        stats.mean_kernel_μs, stats.std_kernel_μs,
        stats.mean_total_μs,  stats.std_total_μs,
    ))
end

# ---------------------------------------------------------------------------
# Generic grouped-barplot (for copy / mapreduce / scan)
# ---------------------------------------------------------------------------

"""
    plot_grouped_barplot(df, n; title, figsize, method_colors, method_order,
                         overhead_alpha, label_offset_frac,
                         label_suffix_fn) -> Figure

Create a grouped barplot where groups = data types, bars = methods.
Each bar is stacked: kernel time (solid) + overhead (alpha).

`label_suffix_fn(method, T) -> String` optionally appends a suffix to bar labels
(e.g. "(UInt8)" for CUB on UnitFloat8 in mapreduce).
"""
function plot_grouped_barplot(
    df::DataFrame,
    n::Int;
    title::String="Performance (n = $(format_number(n)))",
    figsize::Tuple{Int,Int}=(800, 500),
    method_colors::Dict{String,<:Any}=BENCH_COLORS,
    method_order::Vector{String}=["CUDA", "AK", "Forge", "CUB"],
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
    label_suffix_fn=nothing,   # (method, T) -> "" or extra suffix string
    highlight_method=nothing,  # method name rendered in bold
)
    subset = filter(r -> r.n == n, df)
    isempty(subset) && error("No data found for n = $n")

    types = sort(unique(subset.T))
    n_methods = length(method_order)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = range(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, length=n_methods)

    # --- first pass: gather data & find max height ---
    method_data = Dict{String,@NamedTuple{kernel::Vector{Float64}, overhead::Vector{Float64}, err::Vector{Float64}, x::Vector{Float64}}}()
    max_height = 0.0

    for (idx, method) in enumerate(method_order)
        kernel = Float64[]; overhead = Float64[]; err = Float64[]
        for T in types
            row = filter(r -> r.method == method && r.T == T, subset)
            if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                k = row.mean_kernel_μs[1]; t = row.mean_total_μs[1]; e = row.std_kernel_μs[1]
                push!(kernel, k); push!(overhead, max(0.0, t - k)); push!(err, e)
                max_height = max(max_height, k + max(0.0, t - k) + e)
            else
                push!(kernel, 0.0); push!(overhead, 0.0); push!(err, 0.0)
            end
        end
        x = collect(1:length(types)) .+ offsets[idx]
        method_data[method] = (; kernel, overhead, err, x)
    end

    fixed_offset = max_height * label_offset_frac

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1],
        ylabel="Time (μs)", ylabelsize=16,
        title=title, titlesize=20,
        xticks=(1:length(types), types), xticklabelsize=16,
    )

    # --- second pass: draw bars ---
    for method in method_order
        color = method_colors[method]
        d = method_data[method]

        barplot!(ax, d.x, d.kernel,  width=bar_width, color=color, label="$method (kernel)")
        if any(d.overhead .> 0)
            barplot!(ax, d.x, d.overhead, width=bar_width,
                color=(color, overhead_alpha), offset=d.kernel)
        end
        errorbars!(ax, d.x, d.kernel, d.err, color=:black, whiskerwidth=6)

        is_bold = !isnothing(highlight_method) && method == highlight_method
        for (i, (xi, ki, oi, ei)) in enumerate(zip(d.x, d.kernel, d.overhead, d.err))
            ki > 0 || continue
            suffix = isnothing(label_suffix_fn) ? "" : label_suffix_fn(method, types[i])
            lbl = format_3digits(ki) * suffix
            text!(ax, xi, ki + oi + fixed_offset;
                text=lbl, align=(:center, :bottom), fontsize=10,
                font=is_bold ? :bold : :regular)
        end
    end

    # secondary ms axis on the right
    autolimits!(ax)
    ax_right = Axis(fig[1, 1],
        ylabel="Time (ms)", ylabelsize=16,
        yaxisposition=:right,
        yticklabelalign=(:left, :center),
        xticksvisible=false, xticklabelsvisible=false,
        xlabelvisible=false, xgridvisible=false, ygridvisible=false,
    )
    hidespines!(ax_right, :t, :b, :l)
    linkaxes!(ax, ax_right)
    ax_right.ytickformat = vs -> [format_3digits(v / 1000) for v in vs]

    # legend
    elems = []; lbls = String[]
    for method in method_order
        c = method_colors[method]
        push!(elems, PolyElement(color=c));        push!(lbls, "$method (kernel)")
        if method != "CUB"
            push!(elems, PolyElement(color=(c, overhead_alpha))); push!(lbls, "$method (overhead)")
        end
    end
    Legend(fig[1, 2], elems, lbls, "Method")

    return fig
end

"""
    plot_grouped_barplot_multi(df, sizes; title_fn, figsize, method_colors, method_order,
                               overhead_alpha, label_offset_frac, label_suffix_fn,
                               highlight_method) -> Figure

Multiple side-by-side grouped barplots, one per element of `sizes`.
`title_fn(n) -> String` builds the subplot title (default: `"n = …"`).
"""
function plot_grouped_barplot_multi(
    df::DataFrame,
    sizes::Vector{Int};
    title_fn=n -> "n = $(format_number(n))",
    figsize::Tuple{Int,Int}=(500 * length(sizes), 450),
    method_colors::Dict{String,<:Any}=BENCH_COLORS,
    method_order::Vector{String}=["CUDA", "AK", "Forge", "CUB"],
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
    label_suffix_fn=nothing,
    highlight_method=nothing,
)
    fig = Figure(size=figsize)
    types = sort(unique(df.T))
    n_methods = length(method_order)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = range(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, length=n_methods)

    for (col, n) in enumerate(sizes)
        subset = filter(r -> r.n == n, df)
        if isempty(subset); @warn "No data for n=$n, skipping"; continue; end

        use_ms = col == length(sizes)
        unit_div = use_ms ? 1000.0 : 1.0
        unit_lbl = use_ms ? "Time (ms)" : "Time (μs)"

        ax = Axis(fig[1, col],
            ylabel=unit_lbl, ylabelsize=16,
            title=title_fn(n), titlesize=20,
            xticks=(1:length(types), types), xticklabelsize=14,
        )

        method_data = Dict{String,@NamedTuple{kernel::Vector{Float64}, overhead::Vector{Float64}, err::Vector{Float64}, x::Vector{Float64}}}()
        max_height = 0.0

        for (idx, method) in enumerate(method_order)
            kernel = Float64[]; overhead = Float64[]; err = Float64[]
            for T in types
                row = filter(r -> r.method == method && r.T == T, subset)
                if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                    k = row.mean_kernel_μs[1] / unit_div
                    t = row.mean_total_μs[1]  / unit_div
                    e = row.std_kernel_μs[1]  / unit_div
                    push!(kernel, k); push!(overhead, max(0.0, t - k)); push!(err, e)
                    max_height = max(max_height, k + max(0.0, t - k) + e)
                else
                    push!(kernel, 0.0); push!(overhead, 0.0); push!(err, 0.0)
                end
            end
            x = collect(1:length(types)) .+ offsets[idx]
            method_data[method] = (; kernel, overhead, err, x)
        end

        fixed_offset = max_height * label_offset_frac

        for method in method_order
            color = method_colors[method]
            d = method_data[method]
            barplot!(ax, d.x, d.kernel,  width=bar_width, color=color)
            if any(d.overhead .> 0)
                barplot!(ax, d.x, d.overhead, width=bar_width,
                    color=(color, overhead_alpha), offset=d.kernel)
            end
            errorbars!(ax, d.x, d.kernel, d.err, color=:black, whiskerwidth=6)

            is_bold = !isnothing(highlight_method) && method == highlight_method
            for (i, (xi, ki, oi)) in enumerate(zip(d.x, d.kernel, d.overhead))
                ki > 0 || continue
                suffix = isnothing(label_suffix_fn) ? "" : label_suffix_fn(method, types[i])
                lbl = format_3digits(ki) * suffix
                text!(ax, xi, ki + oi + fixed_offset;
                    text=lbl, align=(:center, :bottom), fontsize=12,
                    font=is_bold ? :bold : :regular)
            end
        end
    end

    elems = []; lbls = String[]
    for method in method_order
        c = method_colors[method]
        push!(elems, PolyElement(color=c));        push!(lbls, "$method (kernel)")
        if method != "CUB"
            push!(elems, PolyElement(color=(c, overhead_alpha))); push!(lbls, "$method (overhead)")
        end
    end
    Legend(fig[2, :], elems, lbls, orientation=:horizontal, tellheight=true, tellwidth=false)

    return fig
end

# ---------------------------------------------------------------------------
# Two-method (Forge vs cuBLAS) barplot over (n×p) configurations
# Used by matvec and vecmat scripts.
# ---------------------------------------------------------------------------

"""
    plot_npbar(df, total_elements; x_col, xlabel, df2, title, figsize,
               colors, overhead_alpha, label_offset_frac) -> Figure

Grouped barplot with two methods (Forge, cuBLAS) as a function of one dimension
(`x_col` = `:p` for matvec, `:n` for vecmat).
If `df2` is provided a horizontal line marks the alternative-config overhead.
"""
function plot_npbar(
    df::DataFrame,
    total_elements::Int;
    x_col::Symbol=:p,
    xlabel::String=string(x_col),
    df2::Union{DataFrame,Nothing}=nothing,
    title::String="Kernel Time Comparison (n×p = $(format_number(total_elements)))",
    figsize::Tuple{Int,Int}=(800, 500),
    colors::Tuple=(:steelblue, :coral),
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
)
    subset = filter(r -> r.total_elements == total_elements, df)
    isempty(subset) && error("No data for total_elements=$total_elements")

    x_vals = sort(unique(getproperty(subset, x_col)))

    function get_series(d, method)
        k = Float64[]; ov = Float64[]; err = Float64[]
        for v in x_vals
            row = filter(r -> getproperty(r, x_col) == v && r.method == method, d)
            if !isempty(row)
                ki = row.mean_kernel_μs[1]; ti = row.mean_total_μs[1]
                push!(k, ki); push!(ov, max(0.0, ti-ki)); push!(err, row.std_kernel_μs[1])
            else; push!(k,0.0); push!(ov,0.0); push!(err,0.0); end
        end
        return k, ov, err
    end

    fk, fo, fe = get_series(subset, "Forge")
    ck, co, ce = get_series(subset, "cuBLAS")

    fo_alt = Float64[]
    if df2 !== nothing
        sub2 = filter(r -> r.total_elements == total_elements, df2)
        for v in x_vals
            row = filter(r -> getproperty(r, x_col)==v && r.method=="Forge", sub2)
            push!(fo_alt, isempty(row) ? 0.0 : max(0.0, row.mean_total_μs[1]-row.mean_kernel_μs[1]))
        end
    end

    fc = Makie.to_color(colors[1]); cc = Makie.to_color(colors[2])
    max_height = max(
        maximum(fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i]) + fe[i] for i in eachindex(x_vals); init=0.0),
        maximum(ck .+ co .+ ce; init=0.0)
    )
    fixed_offset = max_height * label_offset_frac

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel=xlabel, ylabel="Time (μs)", title=title,
        xticks=(1:length(x_vals), string.(x_vals)), xticklabelrotation=π/6,
        xlabelsize=18, ylabelsize=18, titlesize=20,
    )
    x = collect(1:length(x_vals)); w = 0.35; off = w/2

    barplot!(ax, x.-off, fk, width=w, color=colors[1], label="Forge (kernel)")
    barplot!(ax, x.-off, fo, width=w, color=(fc,overhead_alpha), offset=fk, label="Forge (overhead)")
    barplot!(ax, x.+off, ck, width=w, color=colors[2], label="cuBLAS (kernel)")
    barplot!(ax, x.+off, co, width=w, color=(cc,overhead_alpha), offset=ck, label="cuBLAS (overhead)")

    if !isempty(fo_alt)
        for (i,xi) in enumerate(x)
            alt_top = fk[i]+fo_alt[i]
            lines!(ax, [xi-off-w/2, xi-off+w/2], [alt_top,alt_top],
                color=:black, linewidth=2.5, label=(i==1 ? "Forge (overhead, Opt)" : nothing))
        end
    end

    errorbars!(ax, x.-off, fk, fe, color=:black, whiskerwidth=6)
    errorbars!(ax, x.+off, ck, ce, color=:black, whiskerwidth=6)

    for (i,xi) in enumerate(x)
        f_top = fk[i]+max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i])
        text!(ax, xi-off, f_top+fixed_offset; text=format_3digits(fk[i]),
            align=(:center,:bottom), fontsize=12, font=:bold)
        text!(ax, xi+off, ck[i]+co[i]+fixed_offset; text=format_3digits(ck[i]),
            align=(:center,:bottom), fontsize=12)
    end

    axislegend(ax, position=:rt, labelsize=14)
    return fig
end

"""
    plot_npbar_multi(df, totals; x_col, xlabel, df2, figsize, colors,
                    overhead_alpha, label_offset_frac) -> Figure

Side-by-side version of `plot_npbar` for multiple `total_elements` values.
"""
function plot_npbar_multi(
    df::DataFrame,
    totals::Vector{Int};
    x_col::Symbol=:p,
    xlabel::String=string(x_col),
    df2::Union{DataFrame,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(500*length(totals), 450),
    colors::Tuple=(:steelblue, :coral),
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
)
    fig = Figure(size=figsize)
    fc = Makie.to_color(colors[1]); cc = Makie.to_color(colors[2])

    for (col, total) in enumerate(totals)
        subset = filter(r -> r.total_elements == total, df)
        if isempty(subset); @warn "No data for total_elements=$total"; continue; end

        x_vals = sort(unique(getproperty(subset, x_col)))

        function get_series(d, method)
            k = Float64[]; ov = Float64[]; err = Float64[]
            for v in x_vals
                row = filter(r -> getproperty(r, x_col)==v && r.method==method, d)
                if !isempty(row)
                    ki = row.mean_kernel_μs[1]; ti = row.mean_total_μs[1]
                    push!(k, ki); push!(ov, max(0.0, ti-ki)); push!(err, row.std_kernel_μs[1])
                else; push!(k,0.0); push!(ov,0.0); push!(err,0.0); end
            end
            return k, ov, err
        end

        fk, fo, fe = get_series(subset, "Forge")
        ck, co, ce = get_series(subset, "cuBLAS")

        fo_alt = Float64[]
        if df2 !== nothing
            sub2 = filter(r -> r.total_elements == total, df2)
            for v in x_vals
                row = filter(r -> getproperty(r, x_col)==v && r.method=="Forge", sub2)
                push!(fo_alt, isempty(row) ? 0.0 : max(0.0, row.mean_total_μs[1]-row.mean_kernel_μs[1]))
            end
        end

        max_height = max(
            maximum(fk[i]+max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i])+fe[i] for i in eachindex(x_vals); init=0.0),
            maximum(ck .+ co .+ ce; init=0.0)
        )
        fixed_offset = max_height * label_offset_frac

        ax = Axis(fig[1, col];
            xlabel=xlabel, ylabel=col==1 ? "Time (μs)" : "",
            title="n×p = $(format_number(total))",
            xticks=(1:length(x_vals), string.(x_vals)), xticklabelrotation=π/6,
            xlabelsize=16, ylabelsize=16, titlesize=18,
        )
        x = collect(1:length(x_vals)); w = 0.35; off = w/2

        barplot!(ax, x.-off, fk, width=w, color=colors[1])
        barplot!(ax, x.-off, fo, width=w, color=(fc,overhead_alpha), offset=fk)
        barplot!(ax, x.+off, ck, width=w, color=colors[2])
        barplot!(ax, x.+off, co, width=w, color=(cc,overhead_alpha), offset=ck)

        if !isempty(fo_alt)
            for (i,xi) in enumerate(x)
                alt_top = fk[i]+fo_alt[i]
                lines!(ax, [xi-off-w/2, xi-off+w/2], [alt_top,alt_top], color=:black, linewidth=2.5)
            end
        end

        errorbars!(ax, x.-off, fk, fe, color=:black, whiskerwidth=6)
        errorbars!(ax, x.+off, ck, ce, color=:black, whiskerwidth=6)

        for (i,xi) in enumerate(x)
            f_top = fk[i]+max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i])
            text!(ax, xi-off, f_top+fixed_offset; text=format_3digits(fk[i]),
                align=(:center,:bottom), fontsize=12, font=:bold)
            text!(ax, xi+off, ck[i]+co[i]+fixed_offset; text=format_3digits(ck[i]),
                align=(:center,:bottom), fontsize=12)
        end
    end

    legend_elems = if df2 !== nothing
        [PolyElement(color=colors[1]), PolyElement(color=(fc,overhead_alpha)),
         LineElement(color=:black,linewidth=2.5),
         PolyElement(color=colors[2]), PolyElement(color=(cc,overhead_alpha))]
    else
        [PolyElement(color=colors[1]), PolyElement(color=(fc,overhead_alpha)),
         PolyElement(color=colors[2]), PolyElement(color=(cc,overhead_alpha))]
    end
    legend_lbls = if df2 !== nothing
        ["Forge (kernel)","Forge (overhead)","Forge (overhead, Opt)","cuBLAS (kernel)","cuBLAS (overhead)"]
    else
        ["Forge (kernel)","Forge (overhead)","cuBLAS (kernel)","cuBLAS (overhead)"]
    end
    Legend(fig[2,:], legend_elems, legend_lbls,
        orientation=:horizontal, tellheight=true, tellwidth=false, labelsize=16)

    return fig
end
