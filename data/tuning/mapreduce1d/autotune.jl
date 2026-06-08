# =============================================================================
# data/tuning/mapreduce1d/autotune.jl
# =============================================================================
# Per-N autotuner for `KernelForge.mapreduce1d`. At each N in the tuning
# grid (≥ 1M elements — below that the launch-overhead regime dominates
# and tuning is in bench noise), sweep the full `(Nitem, workgroup, blocks)`
# Cartesian product on hardware-derived grids, pick the per-N winner
# triple, and emit three `@inline default_*` regime functions into
# `data/tuning/<Arch>_inline.jl`.
#
# Codegen output (NO dict, NO runtime lookup — all `?:` chains constant-folded):
#
#     @inline default_nitem(::Arch, ::Type{MapReduce1D}, n, ::Type{T}) where T
#         nb = n * sizeof(T)
#         nitem_f32 = nb < THR0 ? Ni0 : nb < THR1 ? Ni1 : … : NiLAST
#         clamp(nitem_f32 * 4 ÷ sizeof(T), 1, 16)
#     end
#     @inline default_workgroup(::Arch, ::Type{MapReduce1D}, n, ::Type{T}) where T
#         n < N0 ? FALLBACK_WG : n < N1 ? WG1 : … : WG_LAST
#     end
#     @inline default_blocks(::Arch, ::Type{MapReduce1D}, n, ::Type{T}) where T
#         n < N0 ? FALLBACK_BL : n < N1 ? BL1 : … : BL_LAST
#     end
#
# The N thresholds in `default_workgroup` / `default_blocks` are the
# log-midpoints between consecutive tuned-N values, identical across the
# three methods. The `nb` thresholds for `default_nitem` are
# `threshold_n × sizeof(Float32)` (= 4 × n) — keeping the regime boundaries
# byte-aligned for cross-dtype generalisation.
#
# Hardware-derived candidate grids:
#   * `blocks ∈ BLOCKS_FACTORS × num_sms(backend)`  — `{1,2,3,4,6,8}× N_SMs`
#   * `workgroup ∈ WG_WAVES × warpsize(arch)`       — `{8,16,32}× warpsz`
# (4-warp/block workgroups are excluded — they're rarely optimal on NVIDIA
# Ada+ and noise-unstable in min-max scoring.)
#
# Usage (run once per arch; commit `data/tuning/<Arch>_inline.jl`):
#   julia --project=… data/tuning/mapreduce1d/autotune.jl
# =============================================================================

using KernelForge
using KernelAbstractions
using Printf
using Statistics
using Dates

const KF = KernelForge
const KA = KernelAbstractions

# -----------------------------------------------------------------------------
# Backend abstraction.
# -----------------------------------------------------------------------------

const _CUDA_LOADED = try @eval import CUDA; true catch; false end
const _AMD_LOADED  = try @eval import AMDGPU; true catch; false end

const _BACKEND_KIND =
    _CUDA_LOADED ? :cuda :
    _AMD_LOADED  ? :roc  :
    error("autotune: no CUDA.jl or AMDGPU.jl in this project's environment")

if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend() = CUDA.CUDABackend()
        _gpu_rand(::Type{T}, dims...) where T = CUDA.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T = CUDA.zeros(T, dims...)
        function _bench_us(f; trials::Int = 30, reducer::Function = median)
            starts = [CUDA.CuEvent() for _ in 1:trials]
            stops  = [CUDA.CuEvent() for _ in 1:trials]
            for i in 1:trials
                CUDA.record(starts[i]); f(); CUDA.record(stops[i])
            end
            CUDA.synchronize(stops[trials])
            samples = Float64[CUDA.elapsed(starts[i], stops[i]) * 1e6
                              for i in 1:trials]
            return reducer(samples)
        end
    end
else # :roc
    @eval begin
        _gpu_backend() = AMDGPU.ROCBackend()
        _gpu_rand(::Type{T}, dims...) where T = AMDGPU.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T = AMDGPU.zeros(T, dims...)
        function _bench_us(f; trials::Int = 30, reducer::Function = median)
            inner = 20
            samples = Float64[]
            for _ in 1:trials
                t = Float64(AMDGPU.@elapsed begin
                    for _ in 1:inner; f(); end
                end)
                push!(samples, t * 1e6 / inner)
            end
            isempty(samples) && return NaN
            return reducer(samples)
        end
    end
end

# -----------------------------------------------------------------------------
# Tuning N grid + param grids.
# -----------------------------------------------------------------------------

# N grid for per-N tuning. Below 2^20 (1M elements) the kernel is dominated
# by launch overhead and tuning is in the bench noise (±15 %); we hardcode
# the smallest-tuned-N winner as the small-N fallback in the emitted code.
#
# 2^27 (≈ 134 M) added 2026-05-31: the original (20, 22, 24, 26, 28, 30)
# grid put n=1e8 in the [2^26, 2^28] interval where the winner for n=2^26
# (smaller block count) doesn't transfer well — measured ~24 % regression
# on UInt8 at n=1e8 when extrapolated from the n=2^26 winner.
const TUNED_N_LOG2 = (20, 22, 24, 26, 27, 28, 30)   # 1M, 4M, 16M, 64M, 134M, 256M, 1G

const NITEM_GRID     = (1, 2, 4, 8, 16)
# BLOCKS_FACTORS extended 2026-05-31. Original max was 8× nSMs ≈ launch
# overhead minimum. For bandwidth-bound kernels at large N, the optimal
# block count rises so each thread does fewer inner-loop passes (better
# cache locality). At n=1e8 UInt8, bl=200× nSMs ≈ 4000 gives ~24 % less
# time than bl=8× nSMs ≈ 160 because the per-thread pass count drops
# from 38 to 3.
const BLOCKS_FACTORS = (1, 2, 4, 8, 16, 32, 64, 128, 200)
const WG_WAVES       = (8, 16, 32)   # warps/block; floors at 8 (= 256 threads on NVIDIA)

function _blocks_grid(backend)
    nsm = KF.num_sms(backend)
    return Tuple(nsm * k for k in BLOCKS_FACTORS)
end

function _wg_grid(arch)
    wp = KF.get_warpsize(arch)
    return Tuple(wp * m for m in WG_WAVES if wp * m <= 1024)
end

function _warmup(f, ms::Int = 5)
    t0 = time_ns()
    while (time_ns() - t0) < ms * 1_000_000
        f(); KA.synchronize(_gpu_backend())
    end
end

# -----------------------------------------------------------------------------
# Per-N tune: full sweep at one N, return winner triple + bench.
# -----------------------------------------------------------------------------

function _tune_per_n(N::Int, ::Type{T_ref}, wg_grid, bl_grid; trials::Int = 30,
                     verbose::Bool = true) where T_ref
    A = _gpu_rand(T_ref, N)
    runs = NamedTuple[]
    for Ni in NITEM_GRID, wg in wg_grid, bl in bl_grid
        Ni * wg <= N || continue
        f = () -> KF.mapreduce1d(*, +, A; Nitem=Ni, workgroup=wg, blocks=bl,
                                  to_cpu=false)
        try
            _warmup(f)
            us = _bench_us(f; trials)
            push!(runs, (; Nitem=Ni, workgroup=wg, blocks=bl, us))
        catch
        end
    end
    isempty(runs) && return nothing
    sort!(runs, by = r -> r.us)
    best = runs[1]
    verbose && @printf("  N=2^%-2d (%11d) : Ni=%-2d wg=%-4d bl=%-4d  %10.2f µs\n",
                       round(Int, log2(N)), N,
                       best.Nitem, best.workgroup, best.blocks, best.us)
    return (; N, best.Nitem, best.workgroup, best.blocks, best.us)
end

# -----------------------------------------------------------------------------
# Regime construction — collapse consecutive same-value cells into ranges.
#
# Returns Vector{(threshold_n, value)} where rule is
#   `n < threshold_n ? value : ...next...`
# Last entry uses `typemax(Int)` as the threshold (= `else` branch).
# Threshold between cells `i, i+1` = geometric midpoint sqrt(N_i × N_{i+1}).
#
# `extract` picks which field (e.g. `:Nitem`, `:workgroup`, `:blocks`)
# from the winner namedtuple list.
# -----------------------------------------------------------------------------

function _regimes(winners::Vector{<:NamedTuple}, extract::Symbol)
    isempty(winners) && return Tuple{Int,Int}[]
    sorted = sort(winners, by = w -> w.N)
    out = Tuple{Int,Int}[]
    cur = getproperty(sorted[1], extract)
    for i in 2:length(sorted)
        nxt = getproperty(sorted[i], extract)
        if nxt != cur
            thr = isqrt(sorted[i-1].N * sorted[i].N)
            push!(out, (thr, cur))
            cur = nxt
        end
    end
    push!(out, (typemax(Int), cur))
    return out
end

# -----------------------------------------------------------------------------
# Codegen.
# -----------------------------------------------------------------------------

# Fully-qualified name for an arbitrary type, suitable for the loader
# to evaluate inside the KernelForge module (where `MyType` may not be
# in scope). For Base/Core types, `string(T)` suffices; for user types,
# prepend the parent module.
function _qualified_type_name(::Type{T}) where T
    m = parentmodule(T); n = nameof(T)
    m === Core || m === Base ? string(n) : string(m, ".", n)
end

# Emit a piecewise n-based function (for workgroup / blocks). `type_spec`
# is the type-dispatch portion of the signature: either
# `"::Type{T}) where T"` (generic) or `"::Type{MyType}"` (specific).
function _emit_n_func(io, arch_tag, op_sym, func_name,
                      regimes::Vector{Tuple{Int,Int}}, fallback::Int,
                      N_min::Int, type_spec::AbstractString)
    println(io, "@inline function KernelForge.", func_name, "(")
    println(io, "    ::KernelForge.", arch_tag, ", ::Type{KernelForge.",
                op_sym, "}, n, ", type_spec)
    println(io, "    n < ", N_min, " ? ", fallback, " :")
    for (thr, val) in regimes
        if thr == typemax(Int)
            println(io, "        ", val)
        else
            println(io, "        n < ", thr, " ? ", val, " :")
        end
    end
    println(io, "end")
    println(io)
end

# Generic `default_nitem`: cross-dtype scaling via `nb = n × sizeof(T)`.
function _emit_nitem_func_generic(io, arch_tag, op_sym,
                                  regimes::Vector{Tuple{Int,Int}},
                                  fallback_ni::Int, nb_min::Int)
    println(io, "@inline function KernelForge.default_nitem(")
    println(io, "    ::KernelForge.", arch_tag, ", ::Type{KernelForge.",
                op_sym, "}, n::Int, ::Type{T}) where T")
    println(io, "    nb = n * sizeof(T)")
    println(io, "    nitem_f32 = nb < ", nb_min, " ? ", fallback_ni, " :")
    for (thr, val) in regimes
        thr_nb = thr == typemax(Int) ? thr : 4 * thr
        if thr_nb == typemax(Int)
            println(io, "        ", val)
        else
            println(io, "        nb < ", thr_nb, " ? ", val, " :")
        end
    end
    println(io, "    return clamp(nitem_f32 * 4 ÷ sizeof(T), 1, 16)")
    println(io, "end")
    println(io)
end

# Type-specific `default_nitem`: direct N-keyed, no scaling (the method
# only ever sees one concrete T).
function _emit_nitem_func_specific(io, arch_tag, op_sym,
                                   regimes::Vector{Tuple{Int,Int}},
                                   fallback_ni::Int, N_min::Int,
                                   type_spec::AbstractString)
    println(io, "@inline function KernelForge.default_nitem(")
    println(io, "    ::KernelForge.", arch_tag, ", ::Type{KernelForge.",
                op_sym, "}, n::Int, ", type_spec)
    println(io, "    n < ", N_min, " ? ", fallback_ni, " :")
    for (thr, val) in regimes
        if thr == typemax(Int)
            println(io, "        ", val)
        else
            println(io, "        n < ", thr, " ? ", val, " :")
        end
    end
    println(io, "end")
    println(io)
end

function _emit_block(arch_tag::AbstractString, op_sym::Symbol,
                     winners::Vector{<:NamedTuple},
                     ::Type{T_ref}) where T_ref
    buf = IOBuffer()
    is_generic = (T_ref === Float32)   # generic codegen anchored at F32
    type_spec = is_generic ? "::Type{T}) where T" :
                "::Type{$(_qualified_type_name(T_ref))})"
    label = is_generic ? "" : " (type-specific: $(_qualified_type_name(T_ref)))"

    println(buf, "# Generated by data/tuning/mapreduce1d/autotune.jl on ",
                 today(), label)
    println(buf, "# Per-N tuned at N=",
                 join(["2^$(round(Int, log2(w.N)))" for w in winners], "/"))

    sorted = sort(winners, by = w -> w.N)
    N_min = sorted[1].N
    fallback_ni = sorted[1].Nitem
    fallback_wg = sorted[1].workgroup
    fallback_bl = sorted[1].blocks

    regs_wg = _regimes(sorted, :workgroup)
    regs_bl = _regimes(sorted, :blocks)
    regs_ni = _regimes(sorted, :Nitem)

    _emit_n_func(buf, arch_tag, op_sym, "default_workgroup",
                 regs_wg, fallback_wg, N_min, type_spec)
    _emit_n_func(buf, arch_tag, op_sym, "default_blocks",
                 regs_bl, fallback_bl, N_min, type_spec)

    if is_generic
        _emit_nitem_func_generic(buf, arch_tag, op_sym, regs_ni,
                                 fallback_ni, 4 * N_min)
    else
        _emit_nitem_func_specific(buf, arch_tag, op_sym, regs_ni,
                                  fallback_ni, N_min, type_spec)
    end

    return String(take!(buf))
end

# -----------------------------------------------------------------------------
# Merge new block into <Arch>_inline.jl (preserves other ops' blocks).
# -----------------------------------------------------------------------------

function _write_merged(path::AbstractString, op_sym::Symbol,
                       new_block::AbstractString)
    begin_marker = "# --- BEGIN $(op_sym) ---"
    end_marker   = "# --- END $(op_sym) ---"
    existing = isfile(path) ? read(path, String) : ""
    if occursin(begin_marker, existing) && occursin(end_marker, existing)
        i = findfirst(begin_marker, existing)
        j = findfirst(end_marker, existing)
        if i !== nothing && j !== nothing
            jstop = j.stop
            if jstop < length(existing) && existing[jstop + 1] == '\n'
                jstop += 1
            end
            existing = existing[1:i.start - 1] * existing[jstop + 1:end]
        end
    end
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, existing)
        if !isempty(existing) && !endswith(existing, "\n\n")
            endswith(existing, "\n") || println(io)
            println(io)
        end
        println(io, begin_marker)
        write(io, new_block)
        println(io, end_marker)
    end
    return path
end

# -----------------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------------

"""
    autotune(::Type{T_ref} = Float32; trials=30, verbose=true)

Tune `mapreduce1d` for arch detected from the active GPU device.

- `T_ref = Float32` (default): emit **generic** `default_*` methods that
  generalize cross-dtype via byte-count scaling (`nb = n × sizeof(T)`).
  Overrides any prior generic block in `<Arch>_inline.jl`.
- `T_ref = <any concrete type>`: emit **type-specific** methods dispatched
  on `::Type{T_ref}` exactly. Takes priority over the generic block at
  runtime via Julia method specificity. The block is delimited by markers
  named after the type (`# --- BEGIN MapReduce1D_<TypeName> ---`) so
  re-tuning one type doesn't disturb the generic block or other types.

For user-defined types, ensure the type is loadable from the KernelForge
module — the generated `<Arch>_inline.jl` is evaluated inside KernelForge.
"""
function autotune(::Type{T_ref} = Float32;
                  trials::Int = 30, verbose::Bool = true) where T_ref
    arch = KF.detect_arch(_gpu_zeros(Float32, 1))
    backend = _gpu_backend()
    arch_tag = String(nameof(typeof(arch)))
    root = pkgdir(KernelForge)
    out_path = joinpath(root, "data", "tuning", "$(arch_tag)_inline.jl")

    wg_grid = _wg_grid(arch)
    bl_grid = _blocks_grid(backend)
    nsm = KF.num_sms(backend)

    # Marker symbol — generic block keeps :MapReduce1D; per-type blocks
    # get a suffix so re-tuning one type doesn't clobber the generic.
    op_marker = T_ref === Float32 ? :MapReduce1D :
                Symbol("MapReduce1D_", _qualified_type_name(T_ref))

    verbose && @printf("autotune(MapReduce1D, %s) on %s%s\n",
                       string(T_ref), arch_tag,
                       T_ref === Float32 ? "" : "  [type-specific]")
    verbose && @printf("  hardware: nSMs=%d, warpsz=%d\n",
                       nsm, KF.get_warpsize(arch))
    verbose && @printf("  wg grid: %s\n  bl grid: %s\n", wg_grid, bl_grid)
    verbose && @printf("  output: %s\n", out_path)
    verbose && @printf("  block marker: %s\n", op_marker)

    # Session warmup.
    A0 = _gpu_rand(T_ref, 1_000_000)
    _warmup(() -> KF.mapreduce1d(*, +, A0; to_cpu=false), 500)
    A0 = nothing

    # Per-N tune.
    verbose && println("\nPer-N winners (full (Nitem, wg, bl) sweep):")
    winners = NamedTuple[]
    for k in TUNED_N_LOG2
        N = 1 << k
        result = try
            _tune_per_n(N, T_ref, wg_grid, bl_grid; trials, verbose)
        catch e
            msg = sprint(showerror, e)
            if occursin("Out of GPU memory", msg) || occursin("hipErrorOutOfMemory", msg)
                @warn "  N=2^$k skipped — OOM ($(round(N*sizeof(T_ref)/2^30; digits=2)) GiB > GPU mem)"
                nothing
            else
                rethrow(e)
            end
        end
        result === nothing && continue
        push!(winners, result)
    end

    isempty(winners) && error("No tuning succeeded.")

    # Emit codegen. The block is dispatched on `:MapReduce1D` regardless,
    # but the function signature is generic or type-specific based on T_ref.
    block = _emit_block(arch_tag, :MapReduce1D, winners, T_ref)
    _write_merged(out_path, op_marker, block)
    verbose && @printf("\nWrote methods:\n%s", block)
    verbose && @printf("File: %s\n", out_path)
    return winners
end

if abspath(PROGRAM_FILE) == @__FILE__
    autotune()
end
