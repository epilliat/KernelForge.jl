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

const GPU_TAG = "RTX1000"   # adjust or import from architectures.jl

const CSV_PATH_MAPREDUCE = "./perfs/julia/results/$GPU_TAG/mapreduce.csv"
const CSV_PATH_SCAN = "./perfs/julia/results/$GPU_TAG/scan.csv"

mkpath("perfs/figures/$GPU_TAG")

# ---------------------------------------------------------------------------
# Helper: load and normalize a CSV
# ---------------------------------------------------------------------------

function load_df(path)
    df = CSV.read(path, DataFrame)
    df = rename(df, :type => :T)
    df.method = replace(df.method, "KernelForge" => "Forge")
    return df
end

# ---------------------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_MAPREDUCE)
    sizes = sort(unique(df.n))

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
    sizes = sort(unique(df.n))

    cub_suffix_scan(method, T) = (method == "CUB" && T == "UInt8") ? "\n(UInt8)" : ""

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
    totals = sort(unique(df.total_elements))

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
    totals = sort(unique(df.total_elements))

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (vector length)")
                   for total in totals)
    for (total, fig) in figures
        #save("perfs/figures/$GPU_TAG/vecmat_np$(total).png", fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (vector length)")
    save("perfs/figures/$GPU_TAG/vecmat_$(GPU_TAG)_comparison.png", fig_multi)
    @info "VecMat figures saved."
end

@info "All done. Figures in perfs/figures/$GPU_TAG/"