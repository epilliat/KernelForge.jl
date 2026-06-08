#=
MapReduce & Scan Plotting Script
=================================
Reads CSV results and produces grouped barplots using plot_utils.jl.
=#
using Pkg
Pkg.activate("perfs/envs/plotenv")
include("./plot_utils.jl")

using DataFrames
using CSV
using Printf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Benchmark CSVs live in the sibling repo `KernelForge-benchmarks`
# (https://github.com/epilliat/KernelForge-benchmarks). Default to the
# canonical sibling-clone location relative to this file; override with
# KF_RESULTS_ROOT=… if your clone lives elsewhere.
function _resolve_results_root()
    haskey(ENV, "KF_RESULTS_ROOT") && return ENV["KF_RESULTS_ROOT"]
    sibling = normpath(joinpath(@__DIR__, "..", "..", "..", "KernelForge-benchmarks", "results"))
    isdir(sibling) && return sibling
    legacy = "./perfs/julia/results"   # pre-v0.2.0 in-tree fallback
    isdir(legacy) && return legacy
    error("""
        Benchmark results not found.
        Clone the sibling repo:
            git clone git@github.com:epilliat/KernelForge-benchmarks.git ../KernelForge-benchmarks
        or set KF_RESULTS_ROOT=/path/to/results.
        """)
end
const RESULTS_ROOT = _resolve_results_root()

# Default to the most-recently-written results directory so a fresh
# run on a new machine plots itself without editing this file. Override
# with `KF_GPU_TAG=…` (e.g. when you have A40 *and* RTX1000 results
# side-by-side and want to (re)plot A40 specifically).
function _detect_gpu_tag()
    haskey(ENV, "KF_GPU_TAG") && return ENV["KF_GPU_TAG"]
    isdir(RESULTS_ROOT) || return "A40"
    dirs = filter(d -> isdir(joinpath(RESULTS_ROOT, d)), readdir(RESULTS_ROOT))
    isempty(dirs) && return "A40"
    # Pick the dir whose newest CSV is most recent.
    function dir_mtime(d)
        path = joinpath(RESULTS_ROOT, d)
        files = filter(f -> endswith(f, ".csv"), readdir(path))
        isempty(files) ? typemin(Float64) : maximum(mtime(joinpath(path, f)) for f in files)
    end
    return last(sort(dirs, by = dir_mtime))
end
const GPU_TAG = _detect_gpu_tag()
@info "plots_from_csv: RESULTS_ROOT=$RESULTS_ROOT  GPU_TAG=$GPU_TAG  (overrides: KF_RESULTS_ROOT=…, KF_GPU_TAG=…)"

const CSV_PATH_MAPREDUCE = joinpath(RESULTS_ROOT, GPU_TAG, "mapreduce.csv")
const CSV_PATH_SCAN = joinpath(RESULTS_ROOT, GPU_TAG, "scan.csv")
const CSV_PATH_SORT = joinpath(RESULTS_ROOT, GPU_TAG, "sort.csv")
const CSV_PATH_SORT_COLUMNS = joinpath(RESULTS_ROOT, GPU_TAG, "sort_columns_min.csv")

mkpath("perfs/figures/$GPU_TAG")

# ---------------------------------------------------------------------------
# Size filters (nothing = use all sizes found in CSV)
# ---------------------------------------------------------------------------

const SIZES_MAPREDUCE = [10^6, 10^7, 10^8]
const SIZES_SCAN = [10^6, 10^7, 10^8]
const SIZES_SORT = [10^6, 10^7, 10^8]
const SIZES_COPY = nothing
const TOTALS_MATVEC = [10^7, 10^8]
const TOTALS_VECMAT = [10^7, 10^8]

function filter_sizes(df_col, override)
    present = sort(unique(df_col))
    isnothing(override) && return present
    # Keep the override's preferred order; drop sizes the CSV doesn't have.
    return [s for s in override if s in present]
end

# ---------------------------------------------------------------------------
# Label formatter: integer (no decimals) for matvec / vecmat
# ---------------------------------------------------------------------------

fmt_int(x) = @sprintf("%d", round(Int, x))

# sizeof for the dtype strings produced by the bench scripts. Used by
# plot_grouped_barplot_multi to convert Gkeys/s → Gbytes/s.
const _DTYPE_BYTES = Dict(
    "UInt8" => 1, "Int8" => 1, "Bool" => 1,
    "UInt16" => 2, "Int16" => 2,
    "UInt32" => 4, "Int32" => 4, "Float32" => 4,
    "UInt64" => 8, "Int64" => 8, "Float64" => 8,
)
sizeof_dtype(T::AbstractString) = get(_DTYPE_BYTES, T, 1)
sizeof_dtype(T) = 1   # fallback (e.g. when T is an Int K-value column)

np_time_scale(total) = total > 10^7 ? 100.0 : 1.0
np_time_unit(total) = total > 10^7 ? "×100 μs" : "μs"

# ---------------------------------------------------------------------------
# Helper: load and normalize a CSV
# ---------------------------------------------------------------------------

function load_df(path)
    df = CSV.read(path, DataFrame)
    if :type in propertynames(df)
        df = rename(df, :type => :T)
    end
    df.method = replace(df.method, "KernelForge" => "Forge")
    return df
end

# ---------------------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_MAPREDUCE)
    # Drop CUB rows for UnitFloat8→Float32: CUB doesn't do the UInt8→Float32
    # conversion, it just sums UInt8s in a UInt32 accumulator. Showing it
    # next to KF's full conversion path is comparing different work.
    df = filter(r -> !(r.method == "CUB" && r.T == "UnitFloat8→Float32"), df)

    sizes = filter_sizes(df.n, SIZES_MAPREDUCE)

    # Float64 first (widest type → tallest bars set the panel scale and
    # don't clip the smaller-type groups).
    mr_type_order = ["Float64", "Float32", "UInt8", "UnitFloat8→Float32"]

    # Time view (per-call μs/ms, with overhead bands).
    fig_time = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        type_order=mr_type_order,
    )
    save("perfs/figures/$GPU_TAG/mapreduce_$(GPU_TAG)_comparison.png", fig_time)

    # Throughput view (Gbytes/s on src read; same scale across types).
    fig_throughput = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        type_order=mr_type_order,
        metric=:throughput,
        label_methods=["Forge", "CUB"],
        label_fmt_fn=format_2digits,
        bytes_per_key_fn=sizeof_dtype,
    )
    save("perfs/figures/$GPU_TAG/mapreduce_$(GPU_TAG)_throughput.png", fig_throughput)
    @info "MapReduce figures saved (time + throughput)."
end

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_SCAN)
    sizes = filter_sizes(df.n, SIZES_SCAN)

    # Float64 first (mirrors mapreduce). QuaternionF64 appended if present.
    scan_type_order = ["Float64", "Float32", "QuaternionF64"]

    fig_time = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        type_order=scan_type_order,
    )
    save("perfs/figures/$GPU_TAG/scan_$(GPU_TAG)_comparison.png", fig_time)

    fig_throughput = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        type_order=scan_type_order,
        metric=:throughput,
        label_methods=["Forge", "CUB"],
        label_fmt_fn=format_2digits,
        bytes_per_key_fn=sizeof_dtype,
    )
    save("perfs/figures/$GPU_TAG/scan_$(GPU_TAG)_throughput.png", fig_throughput)
    @info "Scan figures saved (time + throughput)."
end

# ---------------------------------------------------------------------------
# Helper: load matvec/vecmat CSV (different structure: n, p columns)
# ---------------------------------------------------------------------------

function load_npdf(path)
    df = CSV.read(path, DataFrame)
    df = rename(df, :type => :T)
    df.method = replace(df.method, "KernelForge" => "Forge")
    df.total_elements = df.n .* df.p
    return df
end

# ---------------------------------------------------------------------------
# MatVec
# ---------------------------------------------------------------------------

let df = load_npdf(joinpath(RESULTS_ROOT, GPU_TAG, "matvec.csv"))
    totals = filter_sizes(df.total_elements, TOTALS_MATVEC)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (rows)",
        label_fmt_fn=fmt_int,
        time_scale=np_time_scale(total), time_unit=np_time_unit(total))
                   for total in totals)
    for (total, fig) in figures
        #save("perfs/figures/$GPU_TAG/matvec_np$(total).png", fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (rows)",
        label_fmt_fn=fmt_int,
        time_scale_fn=np_time_scale, time_unit_fn=np_time_unit)
    save("perfs/figures/$GPU_TAG/matvec_$(GPU_TAG)_comparison.png", fig_multi)
    @info "MatVec figures saved."
end

# ---------------------------------------------------------------------------
# VecMat
# ---------------------------------------------------------------------------

let df = load_npdf(joinpath(RESULTS_ROOT, GPU_TAG, "vecmat.csv"))
    totals = filter_sizes(df.total_elements, TOTALS_VECMAT)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (vector length)",
        label_fmt_fn=fmt_int,
        time_scale=np_time_scale(total), time_unit=np_time_unit(total))
                   for total in totals)
    for (total, fig) in figures
        #save("perfs/figures/$GPU_TAG/vecmat_np$(total).png", fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (vector length)",
        label_fmt_fn=fmt_int,
        time_scale_fn=np_time_scale, time_unit_fn=np_time_unit)
    save("perfs/figures/$GPU_TAG/vecmat_$(GPU_TAG)_comparison.png", fig_multi)
    @info "VecMat figures saved."
end

# ---------------------------------------------------------------------------
# Sort (single-vector) — same shape as scan/mapreduce
# ---------------------------------------------------------------------------

if isfile(CSV_PATH_SORT)
    let df = load_df(CSV_PATH_SORT)
        sizes = filter_sizes(df.n, SIZES_SORT)

        # Widest types first → tallest bars set the panel scale and the
        # narrower types stay readable on the same axis. Mirrors the
        # convention from mapreduce/scan plots.
        sort_type_order = ["UInt64", "Float64", "UInt32", "Float32"]

        # Time view. CUDA.jl's bitonic is ~50-100× slower → drop it from
        # the time panel entirely (it stays in the throughput panel for
        # completeness). The remaining three (AK, Forge, CUB) span ~10×
        # of dynamic range, which auto-scale handles cleanly: AK fits
        # at the top and the Forge/CUB delta stays readable below.
        sort_time_methods = ["AK", "Forge", "CUB"]
        fig_time = plot_grouped_barplot_multi(df, sizes;
            method_order=sort_time_methods,
            highlight_method="Forge",
            type_order=sort_type_order,
            label_methods=["AK", "Forge", "CUB"],
            label_fmt_fn=format_1digit,
        )
        save("perfs/figures/$GPU_TAG/sort_$(GPU_TAG)_comparison.png", fig_time)

        # Throughput view. Compresses the dynamic range so CUDA's bitonic
        # doesn't dominate — kernel throughput in Gbytes/s. Annotate only
        # Forge/CUB so the labels don't overcrowd the small bars.
        fig_throughput = plot_grouped_barplot_multi(df, sizes;
            method_order=REDUCE_METHOD_ORDER,
            highlight_method="Forge",
            type_order=sort_type_order,
            metric=:throughput,
            label_methods=["Forge", "CUB"],
            label_fmt_fn=format_2digits,
            bytes_per_key_fn=sizeof_dtype,
        )
        save("perfs/figures/$GPU_TAG/sort_$(GPU_TAG)_throughput.png", fig_throughput)
        @info "Sort figures saved (time + throughput)."
    end
else
    @info "Skip sort plot: $CSV_PATH_SORT not found (run sort_perf_comparison.jl first)."
end

# ---------------------------------------------------------------------------
# Sort-columns — per-dtype grouped bar view, Forge vs CUB vs Thrust.
# Two figures (time in µs, throughput in Gbytes/s) matching the layout
# convention used by sort / mapreduce / scan. The `Forge OEM` and
# `Forge Radix` rows in the CSV are intermediate diagnostics — Forge
# internally dispatches to whichever is faster, so the user-facing
# plot only shows the dispatched `Forge` line against external
# baselines.
# ---------------------------------------------------------------------------

const SORT_COLUMNS_EXTERNAL_SPECS = [
    ("Forge",  BENCH_COLORS["Forge"]),
    ("CUB",    BENCH_COLORS["CUB"]),
    ("Thrust", BENCH_COLORS["AK"]),
]

if isfile(CSV_PATH_SORT_COLUMNS)
    let df = load_df(CSV_PATH_SORT_COLUMNS)
        if :T in propertynames(df) && eltype(df.T) <: AbstractString
            rename!(df, :T => :type)
        end

        df_ext = filter(r -> r.method in ("Forge", "CUB", "Thrust"), df)
        if isempty(df_ext)
            @info "Skip sort-columns plots: no CUB / Thrust rows in $CSV_PATH_SORT_COLUMNS."
        else
            fig_t = plot_sort_columns_bars(df_ext;
                metric = :time,
                method_specs = SORT_COLUMNS_EXTERNAL_SPECS,
                supertitle = "Sort-columns: Forge vs CUB SegSort vs Thrust   " *
                             "(K × M ≈ 4 M keys per cell)",
                # Thrust loop on Float64 at small K (K=64 M=65536 →
                # 65 k serial thrust::sort launches) hits ~6.6 s and
                # would collapse the Forge/CUB bars (~750 µs) to a
                # 1-pixel sliver. Clip Thrust bars at 2× the slowest
                # Forge/CUB result and label them with the actual
                # value (compact-formatted so "6597000" → "6.6M").
                clip_factor = 2.0,
                clip_reference_methods = ["Forge", "CUB"],
                label_fmt_fn = format_compact,
            )
            Label(fig_t[end + 1, 1:2],
                  "CUB = DeviceSegmentedSort::SortKeys     " *
                  "Thrust = packed (sizeof ≤ 4)  /  loop (Float64)",
                  fontsize=11, halign=:center, padding=(0, 0, 4, 6))
            save("perfs/figures/$GPU_TAG/sort_columns_min_$(GPU_TAG)_comparison.png",
                 fig_t)

            fig_g = plot_sort_columns_bars(df_ext;
                metric = :throughput,
                method_specs = SORT_COLUMNS_EXTERNAL_SPECS,
                supertitle = "Sort-columns throughput: Forge vs CUB SegSort vs Thrust",
            )
            Label(fig_g[end + 1, 1:2],
                  "CUB = DeviceSegmentedSort::SortKeys     " *
                  "Thrust = packed (sizeof ≤ 4)  /  loop (Float64)",
                  fontsize=11, halign=:center, padding=(0, 0, 4, 6))
            save("perfs/figures/$GPU_TAG/sort_columns_min_$(GPU_TAG)_throughput.png",
                 fig_g)

            @info "Sort-columns figures saved (time + throughput)."
        end
    end
else
    @info "Skip sort-columns plot: $CSV_PATH_SORT_COLUMNS not found."
end

# ---------------------------------------------------------------------------
# Copy Bandwidth
# ---------------------------------------------------------------------------
const CSV_PATH_COPY = joinpath(RESULTS_ROOT, GPU_TAG, "copy2.csv")

if isfile(CSV_PATH_COPY)
    let df = load_df(CSV_PATH_COPY)
        fig = plot_copy_bandwidth(df; results_dir = joinpath(RESULTS_ROOT, GPU_TAG))
        save("perfs/figures/$GPU_TAG/copy_$(GPU_TAG)_bandwidth.png", fig)
        @info "Copy bandwidth figure saved."
    end
else
    @info "Skip copy plot: $CSV_PATH_COPY not found."
end

@info "All done. Figures in perfs/figures/$GPU_TAG/"