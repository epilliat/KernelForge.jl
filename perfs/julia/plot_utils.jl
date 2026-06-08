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
using JSON


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
    "Forge v8" => colorant"#D55E00",
    "Forge v16" => colorant"#009E73",
    "cuBLAS" => colorant"#56B4E9",
    "CUB" => colorant"#00008B",
)

const COPY_METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "CUB"]
const COPY_SCALING_METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "Forge v8", "Forge v16"]
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
    plot_grouped_barplot_multi(df, sizes; metric=:time, show_overhead=true, ...) -> Figure

Multiple side-by-side grouped barplots, one per element of `sizes`.

`metric`:
- `:time` (default) — bars are kernel time; the rightmost panel auto-
  switches to ms; overhead bars stack on top if `show_overhead=true`.
- `:throughput` — bars are kernel throughput in **Gkeys/s**
  (`n / kernel_μs / 1000`). Compresses the fast/slow dynamic range so a
  slow library doesn't dominate the y-axis. Overhead bars are skipped
  (throughput isn't additive with wall-time overhead).

`show_overhead`: drop overhead bars and their legend entries even in
`:time` mode (useful when overhead is negligible relative to kernel
time).

`label_methods`: restrict above-bar value labels to this method subset
(default: all). Use e.g. `["Forge", "CUB"]` to declutter when only two
bars are the "real" comparison.

`label_fmt_fn`: formatter for above-bar labels (default `format_3digits`;
pass `format_1digit` for very coarse labels).
"""
function plot_grouped_barplot_multi(
    df::DataFrame,
    sizes::Vector{Int};
    title_fn=n -> "n = $(format_number(n))",
    figsize::Tuple{Int,Int}=(600 * length(sizes), 450),
    method_colors::Dict{String,<:Any}=BENCH_COLORS,
    method_order::Vector{String}=["CUDA", "AK", "Forge", "CUB"],
    overhead_alpha::Float64=0.3,
    label_offset_frac::Float64=0.02,
    label_suffix_fn=nothing,
    highlight_method=nothing,
    metric::Symbol=:time,
    show_overhead::Bool=true,
    label_methods::Union{Nothing,Vector{String}}=nothing,
    label_fmt_fn=format_3digits,
    types_per_panel::Bool=false,
    type_order::Union{Nothing,Vector{String}}=nothing,
    bytes_per_key_fn=_ -> 1,
    # Zoom the y-axis to fit only a subset of methods (taller bars from
    # excluded methods overflow the panel). Useful for sort time view
    # where CUDA's bitonic is ~100× slower than the rest.
    ymax_methods::Union{Nothing,Vector{String}}=nothing,
    ymax_headroom::Float64=0.15,
)
    metric in (:time, :throughput) ||
        error("metric must be :time or :throughput, got $metric")
    # Throughput is a rate, not a duration — there's no geometric way
    # to "stack" overhead on top of a throughput bar. The bar height is
    # n / kernel_μs (kernel-only throughput), directly comparable to
    # CUB's reported kernel rate.
    draw_overhead = show_overhead && metric === :time

    fig = Figure(size=figsize)
    # `type_order`: explicit ordering wins over alphabetical, but unknown
    # types still appear at the end (sorted) so accidental additions
    # don't silently disappear from the plot.
    all_types = if type_order === nothing
        sort(unique(df.T))
    else
        present = unique(df.T)
        known   = filter(t -> t in present, type_order)
        extra   = sort(filter(t -> !(t in type_order), present))
        vcat(known, extra)
    end
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

        # `types_per_panel=true` lets each panel show only the groups
        # that have data, instead of forcing a uniform x-axis across
        # all panels. Useful when panels span different (K, M) regimes.
        types = if types_per_panel
            if type_order === nothing
                sort(unique(subset.T))
            else
                present = unique(subset.T)
                known   = filter(t -> t in present, type_order)
                extra   = sort(filter(t -> !(t in type_order), present))
                vcat(known, extra)
            end
        else
            all_types
        end

        if metric === :time
            use_ms = col == length(sizes)
            unit_div = use_ms ? 1000.0 : 1.0
            unit_lbl = use_ms ? "Time (ms)" : "Time (μs)"
            to_value = (μs, _n, _T) -> μs / unit_div
        else
            # Throughput in Gbytes/s when `bytes_per_key_fn` returns the
            # per-key byte width (typically `sizeof(T)`); falls back to
            # Gkeys/s when the fn returns 1.
            bpk_probe = bytes_per_key_fn(first(types))
            unit_lbl = bpk_probe == 1 ? "Throughput (Gkeys/s)" : "Throughput (Gbytes/s)"
            to_value = (μs, nval, Tval) -> nval * bytes_per_key_fn(Tval) / μs / 1000.0
        end
        # In time mode the bar grows upward from kernel to total: solid
        # = kernel, alpha cap = overhead. In throughput mode the bar
        # ceiling is the kernel-only throughput and we draw the *gap*
        # down to the wall-clock throughput as the alpha cap, so a tall
        # light cap means "overhead is eating measurably into the
        # effective rate". Either way the convention is: solid = "what
        # you'd get in the best case", alpha = "what overhead adds/
        # subtracts".
        invert_overhead = (metric === :throughput)

        ax = Axis(fig[1, col],
            ylabel=unit_lbl, ylabelsize=16,
            title=title_fn(n), titlesize=20,
            xticks=(1:length(types), string.(types)), xticklabelsize=14,
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
                    μs_k = row.mean_kernel_μs[1]
                    μs_t = row.mean_total_μs[1]
                    μs_e = row.std_kernel_μs[1]

                    k_kernel = to_value(μs_k, n, T)
                    e = metric === :time ? to_value(μs_e, n, T) :
                        # Δ(throughput) ≈ throughput * (Δμs / μs)
                        (μs_k > 0 ? k_kernel * (μs_e / μs_k) : 0.0)

                    if draw_overhead && !isnan(μs_t)
                        if invert_overhead
                            # Bar = effective throughput (solid) +
                            #       (kernel - effective) cap (alpha).
                            k_total = to_value(μs_t, n, T)
                            base = min(k_kernel, k_total)
                            cap  = max(0.0, k_kernel - k_total)
                            push!(kernel, base)
                            push!(overhead, cap)
                            max_height = max(max_height, base + cap + e)
                        else
                            t = to_value(μs_t, n, T)
                            ov = max(0.0, t - k_kernel)
                            push!(kernel, k_kernel)
                            push!(overhead, ov)
                            max_height = max(max_height, k_kernel + ov + e)
                        end
                    else
                        push!(kernel, k_kernel)
                        push!(overhead, 0.0)
                        max_height = max(max_height, k_kernel + e)
                    end
                    push!(err, e)
                else
                    # NaN tells Makie's barplot!/errorbars! to skip this
                    # x slot — leaves a clean gap instead of a 1-px tall
                    # baseline stub for methods deliberately missing from
                    # a type group (e.g. CUB on UnitFloat8→Float32, which
                    # has no equivalent in CUB).
                    push!(kernel, NaN)
                    push!(overhead, NaN)
                    push!(err, NaN)
                end
            end
            x = collect(1:length(types)) .+ offsets[idx]
            method_data[method] = (; kernel, overhead, err, x)
        end

        # If a zoom subset is provided, re-compute max_height from those
        # methods only (taller-bar methods overflow visually). Label-offset
        # still uses the in-frame max_height — labels for off-frame bars
        # end up clipped along with the bar, which is the desired effect.
        if ymax_methods !== nothing
            zoom_max = 0.0
            for method in ymax_methods
                haskey(method_data, method) || continue
                d = method_data[method]
                for (k, ov, e) in zip(d.kernel, d.overhead, d.err)
                    isnan(k) && continue
                    h = (invert_overhead ? (k + ov + e) : (k + ov + e))
                    zoom_max = max(zoom_max, h)
                end
            end
            if zoom_max > 0
                max_height = zoom_max
            end
        end

        fixed_offset = max_height * label_offset_frac

        for method in method_order
            color = method_colors[method]
            d = method_data[method]
            barplot!(ax, d.x, d.kernel, width=bar_width, color=color)
            if draw_overhead && any(d.overhead .> 0)
                barplot!(ax, d.x, d.overhead, width=bar_width,
                    color=(color, overhead_alpha), offset=d.kernel)
            end
            errorbars!(ax, d.x, d.kernel, d.err, color=:black, whiskerwidth=6)

            is_bold = !isnothing(highlight_method) && method == highlight_method
            labels_on = isnothing(label_methods) || (method in label_methods)
            if labels_on
                for (i, (xi, ki, oi)) in enumerate(zip(d.x, d.kernel, d.overhead))
                    ki > 0 || continue
                    suffix = isnothing(label_suffix_fn) ? "" : label_suffix_fn(method, types[i])
                    # In invert_overhead mode the alpha cap is *above*
                    # the solid bar, so labeling `ki` would point at the
                    # base. Always label the bar TOP — that's the
                    # kernel-only value the reader expects to compare.
                    label_value = invert_overhead ? (ki + oi) : ki
                    lbl = label_fmt_fn(label_value) * suffix
                    text!(ax, xi, ki + oi + fixed_offset;
                        text=lbl, align=(:center, :bottom), fontsize=12,
                        font=is_bold ? :bold : :regular)
                end
            end
        end

        # Apply the zoom limit AFTER bars are plotted, so Makie's auto-
        # ranging doesn't override us. headroom accommodates the label
        # rendered above the tallest in-frame bar.
        if ymax_methods !== nothing && max_height > 0
            ylims!(ax, 0, max_height * (1.0 + ymax_headroom))
        end
    end

    elems = []
    lbls = String[]
    for method in method_order
        c = method_colors[method]
        push!(elems, PolyElement(color=c))
        if draw_overhead
            push!(lbls, "$method (kernel)")
            if method != "CUB"
                push!(elems, PolyElement(color=(c, overhead_alpha)))
                push!(lbls, "$method (overhead)")
            end
        else
            push!(lbls, method)
        end
    end
    Legend(fig[2, :], elems, lbls, orientation=:horizontal, tellheight=true, tellwidth=false)

    return fig
end

# ---------------------------------------------------------------------------
# Sort-columns bar plot — one panel per dtype, x = K, bars per method.
# Supports time (μs) and throughput (Gbytes/s) views. Time view uses log y
# (4-decade dynamic range across {OEM,Radix,Loop}); throughput uses
# linear y (naturally bounded).
# ---------------------------------------------------------------------------

"""
    plot_sort_columns_bars(df; metric=:time, method_specs, ...) -> Figure

Per-dtype grouped bar view of `sort_columns!` results, styled to match
the matvec / vecmat / mapreduce panels: **linear y starting at 0**,
**per-panel autoscale** (each dtype panel uses its own y range), kernel
bars + alpha-shaded overhead caps, horizontal value labels at bar tops.

The 2D schema (K, M, type, method, mean_kernel_μs, mean_total_μs) is
mapped to a 2×2 panel grid (one per dtype). Within each panel, K
values appear on the x-axis and methods become side-by-side bars.

`metric`:
- `:time` — bar = `mean_kernel_μs`; alpha cap = `max(0, total - kernel)`.
- `:throughput` — bar = `K * M * sizeof(T) / kernel_μs / 1000`
  (Gbytes/s); overhead caps are inverted (alpha cap = kernel rate −
  effective rate, mirroring `plot_grouped_barplot_multi`).

`method_specs`: vector of `(label, color)` in plot order (left to
right). Bars for missing/NaN cells leave a clean gap.

`highlight_method`: bold the value labels above this method's bars.

`show_overhead`: include the alpha-cap overhead bars (default true in
time mode, false in throughput mode — `total_μs` is constant per cell
and the overhead caps would just clutter without helping comparison).
"""
function plot_sort_columns_bars(
    df::DataFrame;
    metric::Symbol = :time,
    dtype_order::Vector{String}=["UInt32", "Float32", "Float64", "UInt8"],
    method_specs::Vector=[
        ("Forge",       BENCH_COLORS["Forge"]),
        ("Forge OEM",   colorant"#E69F00"),
        ("Forge Radix", BENCH_COLORS["CUDA"]),
    ],
    title_fn = t -> "T = $t",
    figsize::Tuple{Int,Int} = (1400, 850),
    label_methods::Union{Nothing,Vector{String}} = nothing,
    label_fmt_fn = format_2digits,
    highlight_method = "Forge",
    supertitle::AbstractString = "",
    overhead_alpha::Float64 = 0.3,
    show_overhead::Bool = (metric === :time),
    label_offset_frac::Float64 = 0.025,
    # When set, bars taller than `clip_factor × max(other-method bar
    # heights)` per panel are clipped to that ceiling and drawn with a
    # hatched cap to mark the overflow. The numeric label still shows
    # the real value. Use 2.0 in time view to keep Thrust loop's
    # 100×-slower Float64 outliers from collapsing every other bar to
    # a 1-pixel sliver. Set to `nothing` to disable.
    clip_factor::Union{Real,Nothing} = nothing,
    clip_reference_methods::Union{Nothing,Vector{String}} = nothing,
)
    metric in (:time, :throughput) || error("metric must be :time or :throughput")

    present = unique(df.type)
    dtypes  = vcat(filter(t -> t in present, dtype_order),
                   sort(filter(t -> !(t in dtype_order), present)))

    Ks = sort(unique(df.K))
    xticks_labels = [k >= 1024 ? "$(k ÷ 1024)k" : string(k) for k in Ks]

    methods = [m for (m, _) in method_specs]
    colors  = Dict(method_specs)
    n_methods = length(methods)
    total_width = 0.82
    bar_width = total_width / n_methods
    offsets = range(-total_width / 2 + bar_width / 2,
                    total_width / 2 - bar_width / 2,
                    length=n_methods)

    function cell_value(T_::String, K_::Int, method_::String)
        sub = filter(r -> r.type == T_ && r.K == K_ && r.method == method_, df)
        isempty(sub) && return (NaN, NaN, NaN)
        μs_k = sub.mean_kernel_μs[1]
        μs_t = sub.mean_total_μs[1]
        μs_e = sub.std_kernel_μs[1]
        (isnan(μs_k) || μs_k <= 0) && return (NaN, NaN, NaN)
        M_   = sub.M[1]
        if metric === :time
            base = μs_k
            cap  = isnan(μs_t) ? 0.0 : max(0.0, μs_t - μs_k)
            err  = isnan(μs_e) ? 0.0 : μs_e
            return (base, cap, err)
        else
            bytes_per_key = sizeof_dtype(T_)
            kernel_rate = K_ * M_ * bytes_per_key / μs_k / 1000.0
            err = μs_k > 0 ? kernel_rate * (isnan(μs_e) ? 0.0 : μs_e / μs_k) : 0.0
            if isnan(μs_t) || μs_t <= μs_k
                return (kernel_rate, 0.0, err)
            end
            effective_rate = K_ * M_ * bytes_per_key / μs_t / 1000.0
            base = effective_rate
            cap  = max(0.0, kernel_rate - effective_rate)
            return (base, cap, err)
        end
    end

    fig = Figure(size=figsize)
    if !isempty(supertitle)
        Label(fig[0, 1:2], supertitle, fontsize=18, halign=:center,
              padding=(0, 0, 6, 4))
    end

    rows = cld(length(dtypes), 2)
    ylabel_text = metric === :time ? "Time (μs)" : "Throughput (Gbytes/s)"

    for (i, T_) in enumerate(dtypes)
        row_i = ((i - 1) ÷ 2) + 1
        col_i = ((i - 1) % 2) + 1

        # Per-panel max — each dtype panel autoscales independently,
        # like mapreduce/matvec. Gives every panel its own readable y
        # range instead of being dominated by the largest cell across
        # all dtypes.
        #
        # When `clip_factor` is set, compute the cap from the reference
        # methods only (default: all methods except the last in
        # `method_specs`, i.e. "well-behaved baselines"). Bars from
        # outlier methods that exceed the cap get clipped + labelled.
        reference_methods = clip_reference_methods === nothing ?
            (clip_factor === nothing ? methods :
                                       methods[1:max(1, end-1)]) :
            clip_reference_methods

        panel_ref_max = 0.0
        panel_full_max = 0.0
        for K_ in Ks, m_ in methods
            base, cap, err = cell_value(T_, K_, m_)
            isnan(base) && continue
            top = base + (show_overhead ? cap : 0.0) + err
            panel_full_max = max(panel_full_max, top)
            if m_ in reference_methods
                panel_ref_max = max(panel_ref_max, top)
            end
        end
        # `bar_cap` = where overflow bars are clipped to. `axis_top` =
        # ylim ceiling. `axis_top > bar_cap` so the overflow label sits
        # inside the axis above its clipped bar.
        bar_cap = if clip_factor === nothing
            panel_full_max
        else
            min(panel_full_max, panel_ref_max * clip_factor)
        end
        bar_cap = bar_cap > 0 ? bar_cap : 1.0
        axis_top = bar_cap * (1 + 4 * label_offset_frac)
        panel_max = bar_cap     # used below for clipping
        label_y_off = axis_top * label_offset_frac

        ax = Axis(fig[row_i, col_i];
            xlabel = "K (column size)",
            ylabel = ylabel_text,
            xlabelsize = 14, ylabelsize = 14,
            title = title_fn(T_), titlesize = 16,
            xticks = (1:length(Ks), xticks_labels),
            xticklabelsize = 12, xticklabelrotation = π/6,
            xgridvisible = false, ygridvisible = true,
            ygridstyle = :dash, ygridcolor = (:gray, 0.3),
        )

        for (idx, m_) in enumerate(methods)
            xs    = Float64[]; bases = Float64[]; caps = Float64[]
            errs  = Float64[]; tops  = Float64[]
            real_bases = Float64[]; real_tops = Float64[]
            for (ki, K_) in enumerate(Ks)
                base, cap, err = cell_value(T_, K_, m_)
                isnan(base) && continue
                push!(xs, ki + offsets[idx])
                top_full = base + (show_overhead ? cap : 0.0)
                push!(real_bases, base)
                push!(real_tops,  top_full)
                # Clip both base and cap to fit under panel_max if
                # `clip_factor` was set.
                clipped_base = min(base, panel_max)
                clipped_cap  = show_overhead ?
                               max(0.0, min(top_full, panel_max) - clipped_base) :
                               0.0
                push!(bases, clipped_base)
                push!(caps,  clipped_cap)
                push!(errs,  base > panel_max ? 0.0 : err)
                push!(tops,  clipped_base + clipped_cap)
            end
            isempty(xs) && continue
            color = colors[m_]
            barplot!(ax, xs, bases; width=bar_width, color=color)
            if show_overhead && any(caps .> 0)
                barplot!(ax, xs, caps; width=bar_width,
                         color=(color, overhead_alpha), offset=bases)
            end
            errorbars!(ax, xs, bases, errs; color=:black, whiskerwidth=4)

            # Visual overflow marker: hatched cap on bars that were
            # clipped. Drawn as a contrasting thin band at top.
            for (xi, real_top) in zip(xs, real_tops)
                if real_top > panel_max
                    band_y0 = panel_max * 0.965
                    band_y1 = panel_max * 0.995
                    poly!(ax,
                          Point2f[(xi - bar_width / 2, band_y0),
                                  (xi + bar_width / 2, band_y0),
                                  (xi + bar_width / 2, band_y1),
                                  (xi - bar_width / 2, band_y1)];
                          color = (:white, 0.0),
                          strokecolor = :black, strokewidth = 0.8)
                    # Three short vertical hash marks across the band.
                    for frac in (0.25, 0.5, 0.75)
                        x_hash = xi - bar_width / 2 + bar_width * frac
                        lines!(ax, [x_hash, x_hash], [band_y0, band_y1];
                               color = :black, linewidth = 0.6)
                    end
                end
            end

            labels_on = isnothing(label_methods) || (m_ in label_methods)
            if labels_on
                bold = (highlight_method !== nothing && m_ == highlight_method)
                # In throughput mode the alpha cap is *above* the solid
                # bar — label points to the kernel rate (top) so the
                # reader sees the best-case number directly. For
                # clipped bars always use the real value so the reader
                # sees how big the actual bar would be.
                real_cap_tops = real_bases .+
                    (show_overhead ? (real_tops .- real_bases) : zeros(length(real_bases)))
                for (xi, drawn_top, real_base, real_cap_top) in
                        zip(xs, tops, real_bases, real_cap_tops)
                    real_val = metric === :throughput ? real_cap_top : real_base
                    text!(ax, xi, drawn_top + label_y_off;
                          text = label_fmt_fn(real_val),
                          align = (:center, :bottom),
                          fontsize = 10,
                          font = bold ? :bold : :regular,
                    )
                end
            end
        end

        ylims!(ax, 0, axis_top)
        xlims!(ax, 0.5, length(Ks) + 0.5)
    end

    # Shared bottom legend — kernel + overhead entries when stacking.
    elems  = []
    labels = String[]
    for (m_, c) in method_specs
        push!(elems, PolyElement(color=c))
        push!(labels, show_overhead ? "$m_ (kernel)" : string(m_))
        if show_overhead
            push!(elems, PolyElement(color=(c, overhead_alpha)))
            push!(labels, "$m_ (overhead)")
        end
    end
    Legend(fig[rows + 1, 1:2], elems, labels;
           orientation=:horizontal, tellheight=true, tellwidth=false,
           framevisible=false)

    return fig
end

# ---------------------------------------------------------------------------
# Sort-columns log-log line plot — one panel per dtype
# ---------------------------------------------------------------------------

"""
    plot_sort_columns_lines(df; ...) -> Figure

Build the per-dtype log-log line view for `sort_columns!`. `df` must
carry the canonical 2D schema (K, M, type, method, mean_kernel_μs,
std_kernel_μs).

For each (dtype, method) the helper plots time-per-key on a log y axis
against K on a log x axis. Per-key time normalises away the M
differences across the grid, so all cells of equal K × M show comparable
amplitudes regardless of column count. NaN rows (e.g. OEM for K > 4096)
are dropped per method, leaving clean gaps in the line.

`threshold_fn(T::String) -> Int|Nothing` returns the per-dtype dispatch
threshold to mark with a vertical dashed line (the K at which `Forge`
flips from OEM to radix). Return `nothing` to skip the vline.

`method_specs` is a vector of `(method_label, color, linestyle, marker)`
in plot order — later entries draw on top. Default puts `Forge`
(dispatched) on top so the reader sees it tracking `min(OEM, Radix)`.
"""
function plot_sort_columns_lines(
    df::DataFrame;
    dtype_order::Vector{String}=["UInt32", "Float32", "Float64", "UInt8"],
    title_fn=t -> "T = $t",
    figsize::Tuple{Int,Int}=(1100, 700),
    method_specs=[
        ("Forge Loop",  colorant"#888888", :dot,    :xcross),
        ("Forge Radix", BENCH_COLORS["CUDA"],   :solid,  :utriangle),
        ("Forge OEM",   colorant"#E69F00",      :dash,   :rect),
        ("Forge",       BENCH_COLORS["Forge"],  :solid,  :circle),
    ],
    threshold_fn=t -> begin
        t in ("UInt8", "Int8", "Bool") ? 256 :
        t in ("UInt32", "Int32", "UInt64", "Int64",
              "Float32", "Float64")     ? 1024 : nothing
    end,
    supertitle::AbstractString =
        "Sort-columns: time per key vs column size K   (K × M ≈ 4 M keys per cell)",
    show_loop::Bool = true,
)
    # Available dtypes in the data, in requested order; unknown ones appended.
    present = unique(df.type)
    dtypes  = vcat(filter(t -> t in present, dtype_order),
                   sort(filter(t -> !(t in dtype_order), present)))

    fig = Figure(size=figsize)
    Label(fig[0, 1:2], supertitle, fontsize=18, halign=:center,
          padding=(0, 0, 6, 4))

    # Compute a shared y-limit per row so panels in the same row are
    # comparable; with log scale this also avoids each panel having its
    # own auto-range that hides the cross-panel ordering.
    rows = cld(length(dtypes), 2)

    # Per-dtype y range, gathered before drawing for consistent limits.
    y_lo, y_hi = Inf, 0.0
    for t in dtypes, (label, _c, _ls, _m) in method_specs
        (!show_loop && label == "Forge Loop") && continue
        sub = filter(r -> r.type == t && r.method == label, df)
        for r in eachrow(sub)
            isnan(r.mean_kernel_μs) && continue
            (r.K <= 0 || r.M <= 0) && continue
            v = r.mean_kernel_μs * 1000.0 / (r.K * r.M)  # ns / key
            v > 0 || continue
            y_lo = min(y_lo, v); y_hi = max(y_hi, v)
        end
    end
    # Round y limits to next decade for clean log ticks.
    y_lo = isfinite(y_lo) ? 10.0 ^ floor(log10(y_lo)) : 1e-2
    y_hi = isfinite(y_hi) && y_hi > 0 ? 10.0 ^ ceil(log10(y_hi)) : 1e3

    Ks_all = sort(unique(df.K))
    xticks_vals = collect(Ks_all)
    xticks_labels = [k >= 1024 ? "$(k ÷ 1024)k" : string(k) for k in xticks_vals]

    axes_list = Axis[]
    for (i, t) in enumerate(dtypes)
        row = ((i - 1) ÷ 2) + 1
        col = ((i - 1) % 2) + 1
        ax = Axis(fig[row, col];
            xscale=log10, yscale=log10,
            xlabel="K (column size)", ylabel="Time per key (ns/key)",
            xlabelsize=14, ylabelsize=14,
            title=title_fn(t), titlesize=16,
            xticks=(xticks_vals, xticks_labels),
            xticklabelrotation=π/6, xticklabelsize=11,
            xgridvisible=true, ygridvisible=true,
            xgridstyle=:dash, ygridstyle=:dash,
            xgridcolor=(:gray, 0.25), ygridcolor=(:gray, 0.25),
        )
        push!(axes_list, ax)

        thr = threshold_fn(t)
        if thr !== nothing
            vlines!(ax, [thr]; color=(:black, 0.5), linestyle=:dash,
                    linewidth=1.5)
            text!(ax, thr, y_hi * 0.7;
                  text=" dispatch threshold ($(thr))",
                  align=(:left, :top), fontsize=10, color=(:black, 0.6))
        end

        for (label, color, linestyle, marker) in method_specs
            (!show_loop && label == "Forge Loop") && continue
            sub = filter(r -> r.type == t && r.method == label &&
                              !isnan(r.mean_kernel_μs),
                         df)
            isempty(sub) && continue
            sort!(sub, :K)
            xs = Float64.(sub.K)
            ys = sub.mean_kernel_μs .* 1000.0 ./ (sub.K .* sub.M)
            # Drop non-positive (shouldn't happen but log scale chokes).
            keep = ys .> 0
            isempty(xs[keep]) && continue
            lw = label == "Forge" ? 3.5 : 2.0
            ms = label == "Forge" ? 12  : 9
            scatterlines!(ax, xs[keep], ys[keep];
                color=color, linestyle=linestyle,
                linewidth=lw, marker=marker, markersize=ms,
                strokecolor=color, strokewidth=0,
                label=label,
            )
        end

        ylims!(ax, y_lo, y_hi)
        xlims!(ax, minimum(xticks_vals) * 0.7, maximum(xticks_vals) * 1.3)
    end

    # Shared legend across the bottom row.
    legend_handles = []
    legend_labels  = String[]
    for (label, color, linestyle, marker) in method_specs
        (!show_loop && label == "Forge Loop") && continue
        push!(legend_handles,
              [LineElement(color=color, linestyle=linestyle,
                           linewidth=label == "Forge" ? 3.5 : 2.0),
               MarkerElement(color=color, marker=marker,
                             markersize=label == "Forge" ? 12 : 9)])
        push!(legend_labels, label)
    end
    Legend(fig[rows + 1, 1:2], legend_handles, legend_labels;
           orientation=:horizontal, tellheight=true, tellwidth=false,
           framevisible=false)

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
    label_fmt_fn=format_3digits,
    time_scale::Float64=1.0,
    time_unit::String="μs",
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
                ki = row.mean_kernel_μs[1] / time_scale
                ti = row.mean_total_μs[1] / time_scale
                push!(k, ki)
                push!(ov, max(0.0, ti - ki))
                push!(err, row.std_kernel_μs[1] / time_scale)
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
            push!(fo_alt, isempty(row) ? 0.0 : max(0.0, (row.mean_total_μs[1] - row.mean_kernel_μs[1]) / time_scale))
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
        xlabel=xlabel, ylabel="Time ($time_unit)", title=title,
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
        text!(ax, xi - off, f_top + fixed_offset; text=label_fmt_fn(fk[i]),
            align=(:center, :bottom), fontsize=12, font=:bold)
        text!(ax, xi + off, ck[i] + co[i] + fixed_offset; text=label_fmt_fn(ck[i]),
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
    label_fmt_fn=format_3digits,
    time_scale_fn=total -> 1.0,
    time_unit_fn=total -> "μs",
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
        t_scale = time_scale_fn(total)
        t_unit = time_unit_fn(total)

        function get_series(d, method, scale)
            k = Float64[]
            ov = Float64[]
            err = Float64[]
            for v in x_vals
                row = filter(r -> getproperty(r, x_col) == v && r.method == method, d)
                if !isempty(row)
                    ki = row.mean_kernel_μs[1] / scale
                    ti = row.mean_total_μs[1] / scale
                    push!(k, ki)
                    push!(ov, max(0.0, ti - ki))
                    push!(err, row.std_kernel_μs[1] / scale)
                else
                    push!(k, 0.0)
                    push!(ov, 0.0)
                    push!(err, 0.0)
                end
            end
            return k, ov, err
        end

        fk, fo, fe = get_series(subset, "Forge", t_scale)
        ck, co, ce = get_series(subset, "cuBLAS", t_scale)

        fo_alt = Float64[]
        if df2 !== nothing
            sub2 = filter(r -> r.total_elements == total, df2)
            for v in x_vals
                row = filter(r -> getproperty(r, x_col) == v && r.method == "Forge", sub2)
                push!(fo_alt, isempty(row) ? 0.0 : max(0.0, (row.mean_total_μs[1] - row.mean_kernel_μs[1]) / t_scale))
            end
        end

        max_height = max(
            maximum(fk[i] + max(fo[i], isempty(fo_alt) ? 0.0 : fo_alt[i]) + fe[i] for i in eachindex(x_vals); init=0.0),
            maximum(ck .+ co .+ ce; init=0.0)
        )
        fixed_offset = max_height * label_offset_frac

        ax = Axis(fig[1, col];
            xlabel=xlabel, ylabel="Time ($t_unit)",
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
            text!(ax, xi - off, f_top + fixed_offset; text=label_fmt_fn(fk[i]),
                align=(:center, :bottom), fontsize=12, font=:bold)
            text!(ax, xi + off, ck[i] + co[i] + fixed_offset; text=label_fmt_fn(ck[i]),
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

function get_l2_cache_size()
    return CUDA.attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
end

# ---------------------------------------------------------------------------
# Copy scaling plots
# ---------------------------------------------------------------------------

const COPY_SCALING_METHOD_ORDER = ["CUDA", "Forge v1", "Forge v4", "Forge v8", "Forge v16"]
const COPY_ELEMENT_SIZES = Dict("Float32" => 4, "UInt8" => 1)

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
            lw = endswith(method, "v16") ? 3 : (endswith(method, "v8") ? 3 : 2)
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
    results_dir::String="perfs/julia/results/A40",
)
    fig = Figure(size=figsize)
    types = ["Float32", "UInt8"]
    device_info = JSON.parsefile(joinpath(results_dir, "device_info.json"))
    l2_bytes = device_info["attributes"]["CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"]
    panel_methods = Dict(
        "Float32" => ["CUDA", "Forge v1", "Forge v4"],
        "UInt8" => ["CUDA", "Forge v4", "Forge v16"],
    )
    version_colors = Dict(
        "CUDA" => BENCH_COLORS["CUDA"],
        "Forge v1" => BENCH_COLORS["Forge v1"],
        "Forge v4" => BENCH_COLORS["Forge v4"],
        "Forge v16" => BENCH_COLORS["Forge v16"],
    )

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

        for method in panel_methods[T]
            d = sort(filter(r -> r.method == method, subset), :n)
            isempty(d) && continue
            bw = (d.n .* elem_size .* 2) ./ (d.mean_kernel_μs .* 1e-6) ./ 1e9
            lines!(ax, d.n, bw; color=version_colors[method], linewidth=2, label=method)
        end

        vlines!(ax, [l2_bytes ÷ (2 * elem_size)];
            color=:black, linestyle=:solid, linewidth=1, label="L2 limit")
    end

    all_methods = ["CUDA", "Forge v1", "Forge v4", "Forge v16"]
    elems = [LineElement(color=version_colors[m], linewidth=2) for m in all_methods]
    push!(elems, LineElement(color=:black, linewidth=1))
    lbls = [all_methods..., "L2 limit"]
    Legend(fig[2, :], elems, lbls;
        orientation=:horizontal, tellheight=true, tellwidth=false)

    return fig
end