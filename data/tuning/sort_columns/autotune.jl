# =============================================================================
# data/tuning/sort_columns/autotune.jl
# =============================================================================
# Per-type threshold autotuner for `KernelForge.sort_columns!`. For each
# eltype `T`, find the smallest `K` such that the batched LSD radix path
# is consistently faster than `oem_sort_columns!` across a few M values.
# That K becomes the cross-over threshold `default_sort_columns_threshold`.
#
# Why this only needs one threshold per T: the algorithmic cross-over is
# driven by per-column work (bitonic log²K vs radix npasses(T)) and
# launch-overhead amortisation across M — both of which depend almost
# entirely on K and T, not M (validated on RTX1000, K ≤ 4096 cells:
# OEM win/loss ranking is stable across M ∈ {64, 256, 1024, 4096}).
#
# Output:
#   - `data/tuning/<Arch>.json` under key `sort_columns.size_<sizeof(T)>`
#     (default; matches the size-keyed convention in
#     [data/tuning/matvec/autotune.jl:14-21]) and optionally
#     `sort_columns.<Type>` (type-precise override).
#   - `data/tuning/<Arch>_inline.jl`: a `# --- BEGIN SortColumns ---` block
#     with `@inline KernelForge.default_sort_columns_threshold(...)`
#     methods, byte-size-keyed.
#
# Usage:
#   julia --project=test/envs/cuda data/tuning/sort_columns/autotune.jl
#   julia --project=test/envs/cuda data/tuning/sort_columns/autotune.jl --dev
#
# Dev mode writes to `<Arch>_dev.json` and `<Arch>_inline_dev.jl` only.
# =============================================================================

using KernelForge
using KernelAbstractions
using JSON3
using Printf
using Statistics
using Dates

const KF = KernelForge
const KA = KernelAbstractions

# -----------------------------------------------------------------------------
# Backend abstraction (same shape as data/tuning/matvec/autotune.jl).
# -----------------------------------------------------------------------------

const _CUDA_LOADED = try @eval import CUDA; true catch; false end
const _AMD_LOADED  = try @eval import AMDGPU; true catch; false end

const _BACKEND_KIND =
    _CUDA_LOADED ? :cuda :
    _AMD_LOADED  ? :roc  :
    error("autotune: no CUDA.jl or AMDGPU.jl in this project")

if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend() = CUDA.CUDABackend()
        _gpu_rand(::Type{T}, dims...) where T = CUDA.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T = CUDA.zeros(T, dims...)
        function _bench_us(f; trials = 20, warmup_ms = 200)
            t0 = time_ns()
            while (time_ns() - t0) < warmup_ms * 1_000_000
                f(); CUDA.synchronize()
            end
            starts = [CUDA.CuEvent() for _ in 1:trials]
            stops  = [CUDA.CuEvent() for _ in 1:trials]
            for i in 1:trials
                CUDA.record(starts[i]); f(); CUDA.record(stops[i])
            end
            CUDA.synchronize(stops[trials])
            return median(Float64[CUDA.elapsed(starts[i], stops[i]) * 1e6 for i in 1:trials])
        end
    end
else
    @eval begin
        _gpu_backend() = AMDGPU.ROCBackend()
        _gpu_rand(::Type{T}, dims...) where T = AMDGPU.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T = AMDGPU.zeros(T, dims...)
        function _bench_us(f; trials = 20, warmup_ms = 200)
            t0 = time_ns()
            while (time_ns() - t0) < warmup_ms * 1_000_000
                f(); KA.synchronize(_gpu_backend())
            end
            samples = Float64[]
            for _ in 1:trials
                push!(samples, AMDGPU.@elapsed(f()) * 1e6)
            end
            return median(samples)
        end
    end
end

# -----------------------------------------------------------------------------
# Sweep grid + scoring.
# -----------------------------------------------------------------------------

const SWEEP_TYPES = (UInt8, UInt16, UInt32, UInt64,
                     Int8,  Int16,  Int32,  Int64,
                     Float32, Float64)

# K grid covers the (warp → shared → multi-block) transitions.
const SWEEP_KS = (64, 128, 256, 512, 1024, 1500, 2048, 3000, 4096)

# Three M regimes per K. K * M held roughly constant (~1 M items).
@inline _ms_for_k(k::Int) = (max(1, 1 << 20 ÷ k), max(1, 1 << 18 ÷ k), max(1, 1 << 16 ÷ k))

function _gen_data(::Type{T}, K::Int, M::Int) where T
    if T <: AbstractFloat
        # randn produces well-behaved values; KF.uint_map handles signs.
        return _gpu_rand(T, K, M)
    else
        return _gpu_rand(T, K, M)
    end
end

function _bench_one_cell(::Type{T}, K::Int, M::Int) where T
    A_init = _gen_data(T, K, M)
    A_oem  = copy(A_init)
    A_rd   = copy(A_init)

    # Warmup both paths.
    KF.sort_columns!(A_oem; algorithm = :oem)
    KF.sort_columns!(A_rd;  algorithm = :radix)
    KA.synchronize(_gpu_backend())

    us_oem = _bench_us(function()
        copyto!(A_oem, A_init)
        KF.sort_columns!(A_oem; algorithm = :oem)
    end)
    us_rd = _bench_us(function()
        copyto!(A_rd, A_init)
        KF.sort_columns!(A_rd; algorithm = :radix)
    end)
    return us_oem, us_rd
end

"""
For each T, sweep K and (3 M values per K). Return per-T threshold = the
smallest K such that **median over M** of `us_rd ≤ us_oem` holds for that
K AND every K above it in the sweep. Clamped to OEM's hard cap (4096).
"""
function sweep_thresholds()
    arch = KF.detect_arch(_gpu_zeros(UInt32, 1))
    @info "Sweeping sort_columns thresholds" arch=arch backend=_BACKEND_KIND

    raw_results = Dict{Tuple{Type,Int,Int}, NamedTuple{(:oem, :rd), Tuple{Float64, Float64}}}()
    for T in SWEEP_TYPES
        for K in SWEEP_KS
            for M in _ms_for_k(K)
                try
                    us_oem, us_rd = _bench_one_cell(T, K, M)
                    raw_results[(T, K, M)] = (oem = us_oem, rd = us_rd)
                    @printf("  %-9s K=%-5d M=%-6d  OEM %8.1f µs   radix %8.1f µs   %s\n",
                            T, K, M, us_oem, us_rd,
                            us_rd < us_oem ? "radix" : "OEM")
                catch err
                    @warn "skipping cell" T=T K=K M=M err=err
                end
            end
        end
    end

    thresholds = Dict{Type, Int}()
    for T in SWEEP_TYPES
        # For each K in the sweep, score "fraction of M's where radix wins".
        winners = Dict{Int, Bool}()
        for K in SWEEP_KS
            ms = collect(_ms_for_k(K))
            wins = 0
            counted = 0
            for M in ms
                if haskey(raw_results, (T, K, M))
                    counted += 1
                    raw_results[(T, K, M)].rd < raw_results[(T, K, M)].oem && (wins += 1)
                end
            end
            winners[K] = counted > 0 && wins > counted ÷ 2
        end
        # Threshold = smallest K such that THIS K and every K above it has radix-win.
        sorted_ks = sort(collect(SWEEP_KS))
        thr = 4096
        for (i, K) in enumerate(sorted_ks)
            if all(winners[K2] for K2 in sorted_ks[i:end])
                thr = K
                break
            end
        end
        # Above 4096 OEM is unsupported — clamp.
        thr = min(thr, 4096)
        thresholds[T] = thr
        @info "threshold[$T] = $thr"
    end
    return arch, thresholds, raw_results
end

# -----------------------------------------------------------------------------
# Output: JSON + inline emit.
# -----------------------------------------------------------------------------

function _arch_filename(arch)
    s = string(typeof(arch).name.name)
    return s
end

const _SIZE_TO_TYPES = Dict(
    1 => (UInt8,   Int8,   Bool),
    2 => (UInt16,  Int16),
    4 => (UInt32,  Int32,  Float32),
    8 => (UInt64,  Int64,  Float64),
)

# Convert per-T thresholds → per-sizeof(T) representative (median across
# types of equal sizeof). Size-keyed is the default lookup; type-keyed
# overrides are an optional second tier (omitted in v1 unless types of
# the same size disagree by > 2× — then we emit per-type).
function _by_size(thresholds::Dict{Type, Int})
    size_to_thrs = Dict{Int, Vector{Int}}()
    for (T, k) in thresholds
        s = sizeof(T)
        push!(get!(size_to_thrs, s, Int[]), k)
    end
    return Dict(s => round(Int, median(v)) for (s, v) in size_to_thrs)
end

function write_outputs(arch, thresholds; dev::Bool=false)
    arch_name = _arch_filename(arch)
    json_path = joinpath(@__DIR__, "..", dev ? "$(arch_name)_dev.json" : "$(arch_name).json")
    inline_path = joinpath(@__DIR__, "..", dev ? "$(arch_name)_inline_dev.jl" : "$(arch_name)_inline.jl")

    # ---- JSON: merge into existing file under sort_columns.* ----
    existing = isfile(json_path) ? JSON3.read(read(json_path, String), Dict{String, Any}) : Dict{String, Any}()
    sc = get!(existing, "sort_columns", Dict{String, Any}())
    sc isa AbstractDict || error("existing sort_columns entry is not a dict")
    size_thr = _by_size(thresholds)
    for (s, k) in size_thr
        sc["size_$s"] = Dict("threshold" => k)
    end
    # Type-precise overrides ALWAYS emitted (cheap, ~10 entries):
    for (T, k) in thresholds
        sc[string(T)] = Dict("threshold" => k)
    end
    existing["_meta_sort_columns_generated"] = string(now())
    open(json_path, "w") do io
        JSON3.pretty(io, existing)
    end
    @info "wrote JSON" path=json_path

    # ---- Inline emit ----
    lines = String[]
    push!(lines, "")
    push!(lines, "# --- BEGIN SortColumns ---")
    push!(lines, "# Generated by data/tuning/sort_columns/autotune.jl on $(today())")
    push!(lines, "# Cross-over thresholds between oem_sort_columns! (K ≤ thr) and")
    push!(lines, "# batched_radix_sort_columns! (K > thr), per eltype.")

    # Size-keyed defaults
    for (s, k) in sort(collect(size_thr))
        for T in _SIZE_TO_TYPES[s]
            push!(lines,
                "@inline KernelForge.default_sort_columns_threshold(" *
                "::KernelForge.$(arch_name), ::Type{$T}) = $k")
        end
    end
    push!(lines, "# --- END SortColumns ---")

    # Append (or, if a previous SortColumns block exists, replace it).
    if isfile(inline_path)
        existing_text = read(inline_path, String)
        marker_begin = "# --- BEGIN SortColumns ---"
        marker_end   = "# --- END SortColumns ---"
        i0 = findfirst(marker_begin, existing_text)
        i1 = findfirst(marker_end, existing_text)
        if i0 !== nothing && i1 !== nothing
            head = existing_text[1:first(i0) - 1]
            tail = existing_text[(last(i1) + 1):end]
            existing_text = rstrip(head) * "\n\n" * join(lines, "\n") * tail
        else
            existing_text *= "\n" * join(lines, "\n") * "\n"
        end
        open(inline_path, "w") do io
            write(io, existing_text)
        end
    else
        open(inline_path, "w") do io
            write(io, join(lines, "\n") * "\n")
        end
    end
    @info "wrote inline" path=inline_path
end

# -----------------------------------------------------------------------------
# Entry points.
# -----------------------------------------------------------------------------

function autotune(; dev::Bool = false)
    arch, thresholds, _ = sweep_thresholds()
    write_outputs(arch, thresholds; dev)
    return thresholds
end

if abspath(PROGRAM_FILE) == @__FILE__
    dev = "--dev" in ARGS
    autotune(; dev)
end
