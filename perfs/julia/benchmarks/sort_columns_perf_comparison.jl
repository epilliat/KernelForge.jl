#=
Batched column-sort benchmark
=============================
Compares the four ways a user can sort each column of a K × M matrix
on the GPU:

    1. KF.sort_columns!                — dispatched (auto-routes per K)
    2. KF.sort_columns! algorithm=:oem — forced bitonic (K ≤ 4096 only)
    3. KF.sort_columns! algorithm=:radix — forced batched LSD radix
    4. for j in 1:M; KF.sort!(@view A[:, j]) — per-column radix loop

The "dispatched" row should match `min(oem, radix)` across the entire
sweep — that's the proof the autotuned threshold is doing its job.

Results saved to `perfs/julia/results/<GPU_TAG>/sort_columns_min.csv` with
the canonical 2D schema:
    K, M, type, method, mean_kernel_μs, std_kernel_μs,
    mean_total_μs, std_total_μs

(`:oem` rows are NaN for K > 4096 — the bitonic path doesn't support
those sizes.)
=#
include("../init.jl")

# ---------------------------------------------------------------------------
# Sweep cells: regular K sweep with K * M ≈ 4 M elements held constant
# per cell. Same grid for every dtype, so the per-key time line plot
# (`plots_from_csv.jl`) is directly comparable across panels and shows
# the OEM/radix crossover at a glance.
# ---------------------------------------------------------------------------

const SORT_COLUMNS_KS = (64, 256, 1_024, 4_096,
                         16_384, 65_536, 262_144, 1_048_576)
const SORT_COLUMNS_TOTAL = 4_194_304  # K * M target (== 4 * 2^20)
_m_for(K::Int) = max(SORT_COLUMNS_TOTAL ÷ K, 4)
# Two representative dtypes: Float32 covers the FP comparator regime
# (where `uint_map` does real work), UInt8 covers the narrow-key regime
# (where every byte transferred dominates). Halving the dtype count
# halves the bench wall-clock — important when porting to AMD/Metal.
# The full 4-dtype run is still in `sort_columns.csv` for reference.
const SORT_COLUMNS_DTYPES = (Float32, UInt8)

const SORT_COLUMNS_CELLS = [
    (T, K, _m_for(K))
    for T in SORT_COLUMNS_DTYPES
    for K in SORT_COLUMNS_KS
]

# External baselines (CUDA only). NaN rows are pushed on unsupported
# dtypes so every cell × method pair has a row — the line plots filter
# NaN per-method.
const DEFAULT_CUB_EXE =
    cub_exe(joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/cub_sort_benchmark"))
const DEFAULT_THRUST_EXE = get(ENV, "KF_THRUST_EXE",
    joinpath(@__DIR__, "../../cuda_cpp/cub_nvcc/bin/thrust_sort_benchmark"))

# Binaries: float, double, int, uint8, uint32, uint64 (uint8 added by
# the local recompile that also expanded both CUB and Thrust dispatch).
const _CUB_DTYPES         = (UInt32, Float32, Float64, UInt8)
# Thrust `packed` requires sizeof(T) ≤ 4 (composite seg<<32|val key
# fits in u64). Wider types fall back to `loop` (M serial thrust::sort
# calls — slower at small K with large M but the only choice).
const _THRUST_PACKED      = (UInt32, Float32, UInt8)
const _THRUST_LOOP_ONLY   = (Float64,)

_nan_stats() = (; mean_kernel_μs = NaN, std_kernel_μs = NaN,
                  mean_total_μs  = NaN, std_total_μs  = NaN)

# ---------------------------------------------------------------------------
# Per-cell runner — one row per (cell × method)
# ---------------------------------------------------------------------------

function _make_data(::Type{T}, K::Int, M::Int) where T
    src_cpu = T <: AbstractFloat ? randn(T, K, M) : rand(T, K, M)
    A = KA.allocate(backend, T, K, M)
    copyto!(A, src_cpu)
    return A
end

function run_sort_columns_cell(::Type{T}, K::Int, M::Int) where T
    label_T = string(T)
    println("\n────────── T=$T  K=$K  M=$M ──────────")
    flush(stdout)

    A_init = _make_data(T, K, M)
    A_disp = copy(A_init)
    tmp_radix = (K > 4096) ?
                KF.get_allocation(KF.SortColumns, A_disp) :
                nothing

    rows = NamedTuple[]

    # 1. Dispatched — the only Forge row the headline plot uses.
    # `Forge Loop` / `Forge OEM` / `Forge Radix` used to live here for
    # the dispatch-correctness line plot, but that view is now retired
    # (one bar plot, Forge vs CUB vs Thrust). Dropping them ~halves
    # wall-clock and removes the per-cell Loop method that launched
    # M serial sort1d kernels (≈5 s/cell at K=64 M=65536).
    s = bench("KF.sort_columns! (dispatched) [$label_T K=$K M=$M]", function ()
        copyto!(A_disp, A_init)
        KF.sort_columns!(A_disp; tmp = tmp_radix)
    end; backend)
    push!(rows, (; K, M, type = label_T, method = "Forge",
                 s.mean_kernel_μs, s.std_kernel_μs,
                 s.mean_total_μs,  s.std_total_μs))

    # 2. CUB DeviceSegmentedSort::SortKeys (adaptive batched primitive)
    if T in _CUB_DTYPES
        s = bench_cub_or_nan(DEFAULT_CUB_EXE, K, T; safety_factor=1.5,
                             extra_flags="-M $M --batched-mode segsort")
    else
        s = _nan_stats()
    end
    push!(rows, (; K, M, type = label_T, method = "CUB",
                 s.mean_kernel_μs, s.std_kernel_μs,
                 s.mean_total_μs,  s.std_total_μs))

    # 3. Thrust: packed mode (sizeof ≤ 4) or loop fallback (Float64).
    if T in _THRUST_PACKED
        s = bench_cub_or_nan(DEFAULT_THRUST_EXE, K, T; safety_factor=1.5,
                             extra_flags="-M $M --mode packed")
    elseif T in _THRUST_LOOP_ONLY
        # M serial thrust::sort calls dominate at small K with large M
        # (K=64 M=65536 → ~5 s/iter). Cap iter count aggressively so the
        # cell finishes in ~10 s even at the worst shape.
        thrust_iters = M >= 16_384 ? 5 :
                       M >= 1_024  ? 20 : 100
        s = bench_cub_or_nan(DEFAULT_THRUST_EXE, K, T; safety_factor=1.5,
                             trials=thrust_iters,
                             extra_flags="-M $M --mode loop")
    else
        s = _nan_stats()
    end
    push!(rows, (; K, M, type = label_T, method = "Thrust",
                 s.mean_kernel_μs, s.std_kernel_μs,
                 s.mean_total_μs,  s.std_total_μs))

    flush(stdout)
    return rows
end

# ---------------------------------------------------------------------------
# Collect, save, display
# ---------------------------------------------------------------------------

all_rows = NamedTuple[]
for (T, K, M) in SORT_COLUMNS_CELLS
    append!(all_rows, run_sort_columns_cell(T, K, M))
end

df = DataFrame(all_rows)
csv_path = joinpath(RESULT_DIR, "sort_columns_min.csv")
CSV.write(csv_path, df)
println("\nResults saved to: $csv_path\n")

# ---------------------------------------------------------------------------
# Display — the 2D schema can't reuse `display_results` directly (no `n`
# column), so format here.
# ---------------------------------------------------------------------------

df_display = transform(df,
    [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs] .=>
        (x -> round.(x; digits = 2)) .=>
            [:mean_kernel_μs, :std_kernel_μs, :mean_total_μs, :std_total_μs],
)

println("=== Sort-columns Benchmark — GPU: $GPU_TAG ===")
hl = TextHighlighter(
    (data, i, j) -> data[i, :method] == "Forge",
    crayon"blue bold",
)
pretty_table(df_display; highlighters = [hl])
