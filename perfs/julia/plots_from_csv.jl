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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

const GPU_TAG = "A40"   # adjust or import from architectures.jl

const CSV_PATH_MAPREDUCE = "./perfs/julia/results/$GPU_TAG/mapreduce.csv"
const CSV_PATH_SCAN = "./perfs/julia/results/$GPU_TAG/scan.csv"

mkpath("perfs/figures/$GPU_TAG")

# ---------------------------------------------------------------------------
# Size filters (nothing = use all sizes found in CSV)
# ---------------------------------------------------------------------------

const SIZES_MAPREDUCE = nothing        # e.g. [10^7, 10^8, 10^9]
const SIZES_SCAN = nothing
const SIZES_COPY = nothing        # copy uses n directly (not total_elements)
const TOTALS_MATVEC = nothing        # filters on total_elements = n*p
const TOTALS_VECMAT = nothing

filter_sizes(df_col, override) =
    isnothing(override) ? sort(unique(df_col)) : override

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
    sizes = filter_sizes(df.n, SIZES_MAPREDUCE)

    cub_suffix(method, T) = (method == "CUB" && T == "UnitFloat8→Float32") ? "\n(UInt8)" : ""

    for n in sizes
        fig = plot_grouped_barplot(df, n;
            title="MapReduce Performance (n = $(format_number(n)))",
            method_order=REDUCE_METHOD_ORDER,
            highlight_method="Forge",
            label_suffix_fn=cub_suffix,
        )
        #save("perfs/figures/$GPU_TAG/mapreduce_n$(n).png", fig)
    end

    fig_multi = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        label_suffix_fn=cub_suffix,
    )
    save("perfs/figures/$GPU_TAG/mapreduce_$(GPU_TAG)_comparison.png", fig_multi)
    @info "MapReduce figures saved."
end

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_SCAN)
    sizes = filter_sizes(df.n, SIZES_SCAN)

    cub_suffix_scan(method, T) = ""

    for n in sizes
        fig = plot_grouped_barplot(df, n;
            title="Scan Performance (n = $(format_number(n)))",
            method_order=REDUCE_METHOD_ORDER,
            highlight_method="Forge",
            label_suffix_fn=cub_suffix_scan,
        )
        #save("perfs/figures/$GPU_TAG/scan_n$(n).png", fig)
    end

    fig_multi = plot_grouped_barplot_multi(df, sizes;
        method_order=REDUCE_METHOD_ORDER,
        highlight_method="Forge",
        label_suffix_fn=cub_suffix_scan,
    )
    save("perfs/figures/$GPU_TAG/scan_$(GPU_TAG)_comparison.png", fig_multi)
    @info "Scan figures saved."
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

let df = load_npdf("./perfs/julia/results/$GPU_TAG/matvec.csv")
    totals = filter_sizes(df.total_elements, TOTALS_MATVEC)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (rows)")
                   for total in totals)
    for (total, fig) in figures
        #save("perfs/figures/$GPU_TAG/matvec_np$(total).png", fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (rows)")
    save("perfs/figures/$GPU_TAG/matvec_$(GPU_TAG)_comparison.png", fig_multi)
    @info "MatVec figures saved."
end

# ---------------------------------------------------------------------------
# VecMat
# ---------------------------------------------------------------------------

let df = load_npdf("./perfs/julia/results/$GPU_TAG/vecmat.csv")
    totals = filter_sizes(df.total_elements, TOTALS_VECMAT)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (vector length)")
                   for total in totals)
    for (total, fig) in figures
        #save("perfs/figures/$GPU_TAG/vecmat_np$(total).png", fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (vector length)")
    save("perfs/figures/$GPU_TAG/vecmat_$(GPU_TAG)_comparison.png", fig_multi)
    @info "VecMat figures saved."
end

# ---------------------------------------------------------------------------
# Copy Bandwidth
# ---------------------------------------------------------------------------
const CSV_PATH_COPY = "./perfs/julia/results/$GPU_TAG/copy2.csv"

let df = load_df(CSV_PATH_COPY)
    fig = plot_copy_bandwidth(df)
    save("perfs/figures/$GPU_TAG/copy_$(GPU_TAG)_bandwidth.png", fig)
    @info "Copy bandwidth figure saved."
end

@info "All done. Figures in perfs/figures/$GPU_TAG/"