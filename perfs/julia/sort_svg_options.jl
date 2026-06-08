#=
sort_svg_options.jl
====================
Produces three SVG variants of the single-vector sort benchmark chart
from `perfs/julia/results/<GPU_TAG>/sort.csv`. Each variant exports
both `.svg` (scalable) and `.png` (preview) so you can compare side-
by-side and pick the one you want to keep.

Variants:
  1. classic   — mirror of the current PNG (grouped bars, CUDA/AK/Forge/CUB,
                 throughput in Gbytes/s, two panels n=1e7 & n=1e8).
  2. minimal   — same data, no labels except on Forge/CUB, no overhead
                 bars, thinner bars, looser whitespace.
  3. focus     — Forge vs CUB only, one stacked layout with both n values
                 grouped by dtype, big numbers, side annotations
                 showing % of RTX1000's 192 GB/s peak.

Run:
  julia --project=perfs/envs/plotenv perfs/julia/sort_svg_options.jl
  # or override the GPU tag:
  KF_GPU_TAG=A40 julia --project=perfs/envs/plotenv perfs/julia/sort_svg_options.jl
=#

using Pkg
Pkg.activate("perfs/envs/plotenv")

include("./plot_utils.jl")
using DataFrames, CSV, Printf

# ---------------------------------------------------------------------------
# Setup — match plots_from_csv.jl's GPU-tag detection.
# ---------------------------------------------------------------------------

function _resolve_results_root()
    haskey(ENV, "KF_RESULTS_ROOT") && return ENV["KF_RESULTS_ROOT"]
    sibling = normpath(joinpath(@__DIR__, "..", "..", "..", "KernelForge-benchmarks", "results"))
    isdir(sibling) && return sibling
    legacy = "./perfs/julia/results"
    isdir(legacy) && return legacy
    error("Benchmark results not found. Clone KernelForge-benchmarks at ../KernelForge-benchmarks or set KF_RESULTS_ROOT.")
end
const RESULTS_ROOT = _resolve_results_root()

function _detect_gpu_tag()
    haskey(ENV, "KF_GPU_TAG") && return ENV["KF_GPU_TAG"]
    isdir(RESULTS_ROOT) || return "A40"
    dirs = filter(d -> isdir(joinpath(RESULTS_ROOT, d)), readdir(RESULTS_ROOT))
    isempty(dirs) && return "A40"
    function dir_mtime(d)
        path = joinpath(RESULTS_ROOT, d)
        files = filter(f -> endswith(f, ".csv"), readdir(path))
        isempty(files) ? typemin(Float64) : maximum(mtime(joinpath(path, f)) for f in files)
    end
    return last(sort(dirs, by = dir_mtime))
end

const GPU_TAG = _detect_gpu_tag()
const SORT_CSV = "$RESULTS_ROOT/$GPU_TAG/sort.csv"
const OUT_DIR  = "perfs/figures/$GPU_TAG"
mkpath(OUT_DIR)
@info "sort_svg_options" GPU_TAG OUT_DIR

const _DTYPE_BYTES = Dict("UInt32" => 4, "Int32" => 4, "Float32" => 4,
                          "UInt64" => 8, "Int64" => 8, "Float64" => 8,
                          "UInt8"  => 1, "Int8"  => 1, "Bool" => 1)
sizeof_dtype(s::AbstractString) = get(_DTYPE_BYTES, s, 1)

function load_sort_df(path)
    df = CSV.read(path, DataFrame)
    if :type in propertynames(df); df = rename(df, :type => :T); end
    df.method = replace(df.method, "KernelForge" => "Forge")
    return df
end

const DF = load_sort_df(SORT_CSV)
const PANEL_SIZES = sort(intersect([10^7, 10^8], unique(DF.n)))
@info "sort_svg_options data" sizes=PANEL_SIZES rows=nrow(DF)

# ---------------------------------------------------------------------------
# Variant 1 — Classic: mirror plots_from_csv.jl's `sort_RTX1000_comparison`
# but save as SVG.
# ---------------------------------------------------------------------------

function variant_classic()
    fig = plot_grouped_barplot_multi(DF, PANEL_SIZES;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        metric=:throughput,
        label_methods=["Forge", "CUB"],
        label_fmt_fn=format_2digits,
        bytes_per_key_fn=sizeof_dtype,
    )
    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v1_classic.svg"), fig)
    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v1_classic.png"), fig)
    @info "wrote classic"
end

# ---------------------------------------------------------------------------
# Variant 2 — Minimal: thin bars, no errorbars, no gridlines, white bg,
# only Forge & CUB labelled, generous padding.
# ---------------------------------------------------------------------------

function variant_minimal()
    fig = Figure(size=(1000, 420), backgroundcolor=:white)
    types = sort(unique(DF.T))
    methods = ["CUDA", "AK", "Forge", "CUB"]
    colors  = Dict(m => BENCH_COLORS[m] for m in methods)

    total_w  = 0.7
    bar_w    = total_w / length(methods)
    offsets  = range(-total_w/2 + bar_w/2, total_w/2 - bar_w/2, length=length(methods))

    for (col, n) in enumerate(PANEL_SIZES)
        subset = filter(r -> r.n == n, DF)
        # Compute per-(method, T) Gbytes/s.
        getval(m, T) = begin
            row = filter(r -> r.method == m && r.T == T, subset)
            isempty(row) || isnan(row.mean_kernel_μs[1]) && return NaN
            n * sizeof_dtype(T) / row.mean_kernel_μs[1] / 1000.0
        end
        # Max for axis.
        vals = [getval(m, T) for m in methods, T in types]
        maxh = maximum(filter(!isnan, vals); init=1.0)

        ax = Axis(fig[1, col];
            title="n = $(format_number(n))", titlesize=18,
            ylabel="Throughput (GB/s)", ylabelsize=14,
            xticks=(1:length(types), types), xticklabelsize=13,
            ygridvisible=true, xgridvisible=false,
            spinewidth=0.5, ytickwidth=0.5, xtickwidth=0.5,
            topspinevisible=false, rightspinevisible=false,
        )
        ylims!(ax, 0, maxh * 1.18)

        for (idx, m) in enumerate(methods)
            xs = collect(1:length(types)) .+ offsets[idx]
            ys = [getval(m, T) for T in types]
            ys_safe = [isnan(v) ? 0.0 : v for v in ys]
            barplot!(ax, xs, ys_safe; width=bar_w, color=colors[m], strokewidth=0)

            if m in ("Forge", "CUB")
                for (xi, yi, T) in zip(xs, ys, types)
                    yi > 0 && !isnan(yi) || continue
                    text!(ax, xi, yi + maxh * 0.025;
                        text = format_2digits(yi),
                        align = (:center, :bottom), fontsize = 10,
                        font = m == "Forge" ? :bold : :regular,
                    )
                end
            end
        end
    end

    elems = [PolyElement(color=colors[m]) for m in methods]
    Legend(fig[2, :], elems, methods;
        orientation=:horizontal, tellheight=true, tellwidth=false,
        framevisible=false, labelsize=13)

    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v2_minimal.svg"), fig)
    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v2_minimal.png"), fig)
    @info "wrote minimal"
end

# ---------------------------------------------------------------------------
# Variant 3 — Focus on Forge vs CUB. Two panels (1e7, 1e8), per dtype
# pair of bars + a faint guideline at 192 GB/s (RTX1000 peak BW).
# ---------------------------------------------------------------------------

const RTX1000_PEAK_BW_GB_S = 192.0

function variant_focus()
    fig = Figure(size=(1100, 460), backgroundcolor=:white)
    types = sort(unique(DF.T))
    methods = ["Forge", "CUB"]
    colors  = Dict("Forge" => BENCH_COLORS["Forge"], "CUB" => BENCH_COLORS["CUB"])

    total_w = 0.6
    bar_w   = total_w / length(methods)
    offsets = range(-total_w/2 + bar_w/2, total_w/2 - bar_w/2, length=length(methods))

    for (col, n) in enumerate(PANEL_SIZES)
        subset = filter(r -> r.n == n, DF)
        getval(m, T) = begin
            row = filter(r -> r.method == m && r.T == T, subset)
            (isempty(row) || isnan(row.mean_kernel_μs[1])) && return NaN
            n * sizeof_dtype(T) / row.mean_kernel_μs[1] / 1000.0
        end
        vals = [getval(m, T) for m in methods, T in types]
        maxh = maximum(filter(!isnan, vals); init=1.0)
        # Round max up to a nice number for the axis.
        axis_max = ceil(maxh / 5) * 5

        ax = Axis(fig[1, col];
            title="n = $(format_number(n))", titlesize=20,
            ylabel="Throughput (GB/s)", ylabelsize=15,
            xticks=(1:length(types), types), xticklabelsize=14,
            ygridvisible=true, xgridvisible=false,
            spinewidth=0.6,
            topspinevisible=false, rightspinevisible=false,
        )
        ylims!(ax, 0, axis_max * 1.05)

        # % of peak guideline at 10 %, 20 %, 30 % of RTX1000 peak.
        for (frac, lbl) in ((0.10, "10 % peak"), (0.20, "20 %"))
            y = RTX1000_PEAK_BW_GB_S * frac
            y < axis_max || continue
            hlines!(ax, [y]; color=(:gray, 0.45), linestyle=:dot, linewidth=1)
            text!(ax, length(types) + 0.55, y;
                text=lbl, align=(:left, :center), color=(:gray, 0.8), fontsize=10)
        end

        for (idx, m) in enumerate(methods)
            xs = collect(1:length(types)) .+ offsets[idx]
            ys = [getval(m, T) for T in types]
            ys_safe = [isnan(v) ? 0.0 : v for v in ys]
            barplot!(ax, xs, ys_safe; width=bar_w, color=colors[m], strokewidth=0)

            for (xi, yi, T) in zip(xs, ys, types)
                yi > 0 && !isnan(yi) || continue
                # Above-bar value (GB/s).
                text!(ax, xi, yi + axis_max * 0.015;
                    text=format_2digits(yi), align=(:center, :bottom),
                    fontsize=12, font=(m == "Forge" ? :bold : :regular))
            end
        end
    end

    elems = [PolyElement(color=colors[m]) for m in methods]
    Legend(fig[2, :], elems, methods;
        orientation=:horizontal, tellheight=true, tellwidth=false,
        framevisible=false, labelsize=14)
    Label(fig[0, :], "Sort throughput — $GPU_TAG (peak BW $(Int(RTX1000_PEAK_BW_GB_S)) GB/s)";
        fontsize=16)

    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v3_focus.svg"), fig)
    save(joinpath(OUT_DIR, "sort_$(GPU_TAG)_v3_focus.png"), fig)
    @info "wrote focus"
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

variant_classic()
variant_minimal()
variant_focus()

@info "All three variants written" outdir=OUT_DIR
