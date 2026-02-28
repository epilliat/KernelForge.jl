#=
plot_utils.jl
=============
Plotting utilities for benchmark results.
Requires the full bench environment (CairoMakie, Colors).
Expects bench_utils.jl to have been included first.
=#

using CairoMakie
using DataFrames
using CSV
using Printf


include("number_format.jl")

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong) + dark blue for CUB
# ---------------------------------------------------------------------------

const BENCH_COLORS = Dict(
    "CUDA" => colorant"#CC79A7",
    "AK" => colorant"#009E73",
    "Forge" => colorant"#0072B2",
    "Forge v1" => colorant"#0072B2",
    "Forge v4" => colorant"#E69F00",
    "cuBLAS" => colorant"#56B4E9",
    "CUB" => colorant"#00008B",
)

const COPY_METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "CUB"]
const REDUCE_METHOD_ORDER = ["CUDA", "AK", "Forge", "CUB"]
const COPY_METHOD_COLORS = Dict(k => BENCH_COLORS[k] for k in COPY_METHOD_ORDER)

# ---------------------------------------------------------------------------
# Generic grouped-barplot (for copy / mapreduce / scan)
# ---------------------------------------------------------------------------




"""
    plot_grouped_barplot(df, n; ...) -> Figure

Create a grouped barplot where groups = data types, bars = methods.
Each bar is stacked: kernel time (solid) + overhead (alpha).
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
    label_suffix_fn=nothing,
    highlight_method=nothing,
)
    subset = filter(r -> r.n == n, df)
    isempty(subset) && error("No data found for n = $n")

    types = sort(unique(subset.T))
    n_methods = length(method_order)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = range(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, length=n_methods)

    method_data = Dict{String,@NamedTuple{kernel::Vector{Float64}, overhead::Vector{Float64}, err::Vector{Float64}, x::Vector{Float64}}}()
    max_height = 0.0

    for (idx, method) in enumerate(method_order)
        kernel = Float64[]
        overhead = Float64[]
        err = Float64[]
        for T in types
            row = filter(r -> r.method == method && r.T == T, subset)
            if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                k = row.mean_kernel_μs[1]
                t = row.mean_total_μs[1]
                e = row.std_kernel_μs[1]
                push!(kernel, k)

                ov = isnan(t) ? 0.0 : max(0.0, t - k)
                push!(overhead, ov)
                push!(err, e)
                max_height = max(max_height, k + ov + e)

            else
                push!(kernel, 0.0)
                push!(overhead, 0.0)
                push!(err, 0.0)
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

    for method in method_order
        color = method_colors[method]
        d = method_data[method]

        barplot!(ax, d.x, d.kernel, width=bar_width, color=color, label="$method (kernel)")
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

    elems = []
    lbls = String[]
    for method in method_order
        c = method_colors[method]
        push!(elems, PolyElement(color=c))
        push!(lbls, "$method (kernel)")
        if method != "CUB"
            push!(elems, PolyElement(color=(c, overhead_alpha)))
            push!(lbls, "$method (overhead)")
        end
    end
    Legend(fig[1, 2], elems, lbls, "Method")

    return fig
end

"""
    plot_grouped_barplot_multi(df, sizes; ...) -> Figure

Multiple side-by-side grouped barplots, one per element of `sizes`.
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
    offsets = range(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, length=n_methods)

    for (col, n) in enumerate(sizes)
        subset = filter(r -> r.n == n, df)
        if isempty(subset)
            @warn "No data for n=$n, skipping"
            continue
        end

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
            kernel = Float64[]
            overhead = Float64[]
            err = Float64[]
            for T in types
                row = filter(r -> r.method == method && r.T == T, subset)
                if !isempty(row) && !isnan(row.mean_kernel_μs[1])
                    k = row.mean_kernel_μs[1] / unit_div
                    t = row.mean_total_μs[1] / unit_div
                    e = row.std_kernel_μs[1] / unit_div
                    push!(kernel, k)

                    ov = isnan(t) ? 0.0 : max(0.0, t - k)
                    push!(overhead, ov)
                    push!(err, e)
                    max_height = max(max_height, k + ov + e)
                else
                    push!(kernel, 0.0)
                    push!(overhead, 0.0)
                    push!(err, 0.0)
                end
            end
            x = collect(1:length(types)) .+ offsets[idx]
            method_data[method] = (; kernel, overhead, err, x)
        end

        fixed_offset = max_height * label_offset_frac

        for method in method_order
            color = method_colors[method]
            d = method_data[method]
            barplot!(ax, d.x, d.kernel, width=bar_width, color=color)
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

    elems = []
    lbls = String[]
    for method in method_order
        c = method_colors[method]
        push!(elems, PolyElement(color=c))
        push!(lbls, "$method (kernel)")
        if method != "CUB"
            push!(elems, PolyElement(color=(c, overhead_alpha)))
            push!(lbls, "$method (overhead)")
        end
    end
    Legend(fig[2, :], elems, lbls, orientation=:horizontal, tellheight=true, tellwidth=false)

    return fig
end

# ---------------------------------------------------------------------------
# Two-method (Forge vs cuBLAS) barplot — matvec / vecmat
# ---------------------------------------------------------------------------

"""
    plot_npbar(df, total_elements; ...) -> Figure
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
        k = Float64[]
        ov = Float64[]
        err = Float64[]
        for v in x_vals
            row = filter(r -> getproperty(r, x_col) == v && r.method == method, d)
            if !isempty(row)
                ki = row.mean_kernel_μs[1]
                ti = row.mean_total_μs[1]
                push!(k, ki)
                push!(ov, max(0.0, ti - ki))
                push!(err, row.std_kernel_μs[1])
            else
                push!(k, 0.0)
                push!(ov, 0.0)
                push!(err, 0.0)
            end
        end
        return k, ov, err
    end

    fk, fo, fe = get_series(subset, "Forge")
    ck, co, ce = get_series(subset, "cuBLAS")

    fo_alt = Float64[]
    if df2 !== nothing
        sub2 = filter(r -> r.total_elements == total_elements, df2)
        for v in x_vals
            row = filter(r -> getproperty(r, x_col) == v && r.method == "Forge", sub2)
            push!(fo_alt, isempty(row) ? 0.0 : max(0.0, row.mean_total_μs[1] - row.mean_kernel_μs[1]))
        end
    end

    fc = Makie.to_color(colors[1])
    cc = Makie.to_color(colors[2])
    max_height = max(
        maximum(fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i]) + fe[i] for i in eachindex(x_vals); init=0.0),
        maximum(ck .+ co .+ ce; init=0.0)
    )
    fixed_offset = max_height * label_offset_frac

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel=xlabel, ylabel="Time (μs)", title=title,
        xticks=(1:length(x_vals), string.(x_vals)), xticklabelrotation=π / 6,
        xlabelsize=18, ylabelsize=18, titlesize=20,
    )
    x = collect(1:length(x_vals))
    w = 0.35
    off = w / 2

    barplot!(ax, x .- off, fk, width=w, color=colors[1], label="Forge (kernel)")
    barplot!(ax, x .- off, fo, width=w, color=(fc, overhead_alpha), offset=fk, label="Forge (overhead)")
    barplot!(ax, x .+ off, ck, width=w, color=colors[2], label="cuBLAS (kernel)")
    barplot!(ax, x .+ off, co, width=w, color=(cc, overhead_alpha), offset=ck, label="cuBLAS (overhead)")

    if !isempty(fo_alt)
        for (i, xi) in enumerate(x)
            alt_top = fk[i] + fo_alt[i]
            lines!(ax, [xi - off - w / 2, xi - off + w / 2], [alt_top, alt_top],
                color=:black, linewidth=2.5, label=(i == 1 ? "Forge (overhead, Opt)" : nothing))
        end
    end

    errorbars!(ax, x .- off, fk, fe, color=:black, whiskerwidth=6)
    errorbars!(ax, x .+ off, ck, ce, color=:black, whiskerwidth=6)

    for (i, xi) in enumerate(x)
        f_top = fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i])
        text!(ax, xi - off, f_top + fixed_offset; text=format_3digits(fk[i]),
            align=(:center, :bottom), fontsize=12, font=:bold)
        text!(ax, xi + off, ck[i] + co[i] + fixed_offset; text=format_3digits(ck[i]),
            align=(:center, :bottom), fontsize=12)
    end

    axislegend(ax, position=:rt, labelsize=14)
    return fig
end

"""
    plot_npbar_multi(df, totals; ...) -> Figure

Side-by-side version of `plot_npbar` for multiple `total_elements` values.
"""
function plot_npbar_multi(
    df::DataFrame,
    totals::Vector{Int};
    x_col::Symbol=:p,
    xlabel::String=string(x_col),
    df2::Union{DataFrame,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(500 * length(totals), 450),
    colors::Tuple=(:steelblue, :coral),
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
)
    fig = Figure(size=figsize)
    fc = Makie.to_color(colors[1])
    cc = Makie.to_color(colors[2])

    for (col, total) in enumerate(totals)
        subset = filter(r -> r.total_elements == total, df)
        if isempty(subset)
            @warn "No data for total_elements=$total"
            continue
        end

        x_vals = sort(unique(getproperty(subset, x_col)))

        function get_series(d, method)
            k = Float64[]
            ov = Float64[]
            err = Float64[]
            for v in x_vals
                row = filter(r -> getproperty(r, x_col) == v && r.method == method, d)
                if !isempty(row)
                    ki = row.mean_kernel_μs[1]
                    ti = row.mean_total_μs[1]
                    push!(k, ki)
                    push!(ov, max(0.0, ti - ki))
                    push!(err, row.std_kernel_μs[1])
                else
                    push!(k, 0.0)
                    push!(ov, 0.0)
                    push!(err, 0.0)
                end
            end
            return k, ov, err
        end

        fk, fo, fe = get_series(subset, "Forge")
        ck, co, ce = get_series(subset, "cuBLAS")

        fo_alt = Float64[]
        if df2 !== nothing
            sub2 = filter(r -> r.total_elements == total, df2)
            for v in x_vals
                row = filter(r -> getproperty(r, x_col) == v && r.method == "Forge", sub2)
                push!(fo_alt, isempty(row) ? 0.0 : max(0.0, row.mean_total_μs[1] - row.mean_kernel_μs[1]))
            end
        end

        max_height = max(
            maximum(fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i]) + fe[i] for i in eachindex(x_vals); init=0.0),
            maximum(ck .+ co .+ ce; init=0.0)
        )
        fixed_offset = max_height * label_offset_frac

        ax = Axis(fig[1, col];
            xlabel=xlabel, ylabel=col == 1 ? "Time (μs)" : "",
            title="n×p = $(format_number(total))",
            xticks=(1:length(x_vals), string.(x_vals)), xticklabelrotation=π / 6,
            xlabelsize=16, ylabelsize=16, titlesize=18,
        )
        x = collect(1:length(x_vals))
        w = 0.35
        off = w / 2

        barplot!(ax, x .- off, fk, width=w, color=colors[1])
        barplot!(ax, x .- off, fo, width=w, color=(fc, overhead_alpha), offset=fk)
        barplot!(ax, x .+ off, ck, width=w, color=colors[2])
        barplot!(ax, x .+ off, co, width=w, color=(cc, overhead_alpha), offset=ck)

        if !isempty(fo_alt)
            for (i, xi) in enumerate(x)
                alt_top = fk[i] + fo_alt[i]
                lines!(ax, [xi - off - w / 2, xi - off + w / 2], [alt_top, alt_top], color=:black, linewidth=2.5)
            end
        end

        errorbars!(ax, x .- off, fk, fe, color=:black, whiskerwidth=6)
        errorbars!(ax, x .+ off, ck, ce, color=:black, whiskerwidth=6)

        for (i, xi) in enumerate(x)
            f_top = fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i])
            text!(ax, xi - off, f_top + fixed_offset; text=format_3digits(fk[i]),
                align=(:center, :bottom), fontsize=12, font=:bold)
            text!(ax, xi + off, ck[i] + co[i] + fixed_offset; text=format_3digits(ck[i]),
                align=(:center, :bottom), fontsize=12)
        end
    end

    legend_elems = if df2 !== nothing
        [PolyElement(color=colors[1]), PolyElement(color=(fc, overhead_alpha)),
            LineElement(color=:black, linewidth=2.5),
            PolyElement(color=colors[2]), PolyElement(color=(cc, overhead_alpha))]
    else
        [PolyElement(color=colors[1]), PolyElement(color=(fc, overhead_alpha)),
            PolyElement(color=colors[2]), PolyElement(color=(cc, overhead_alpha))]
    end
    legend_lbls = if df2 !== nothing
        ["Forge (kernel)", "Forge (overhead)", "Forge (overhead, Opt)", "cuBLAS (kernel)", "cuBLAS (overhead)"]
    else
        ["Forge (kernel)", "Forge (overhead)", "cuBLAS (kernel)", "cuBLAS (overhead)"]
    end
    Legend(fig[2, :], legend_elems, legend_lbls,
        orientation=:horizontal, tellheight=true, tellwidth=false, labelsize=16)

    return fig
end