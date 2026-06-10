# =============================================================================
# data/tuning/scan/autotune.jl
# =============================================================================
# Per-N autotuner for `KernelForge.scan` — same design as the mapreduce1d
# script, but scan has no `blocks` knob so we only emit 2 inline regime
# methods: `default_workgroup(n)` and `default_nitem(n, T)`.
#
# Tune at N ≥ 1M only; small-N is dominated by launch overhead and
# tuning there is in the bench noise. Below 1M, the codegen returns the
# smallest-tuned-N winner as fallback.
#
# Usage:
#   julia --project=… data/tuning/scan/autotune.jl
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
            # Adaptive inner (mirrors matvec/vecmat): fixed 20 made giant-N cells
            # cost 20× per candidate. Probe once, size inner to ~3ms of work.
            probe_us = Float64(AMDGPU.@elapsed f()) * 1e6
            inner = probe_us <= 0 ? 20 : clamp(round(Int, 3000.0 / probe_us), 1, 20)
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

# N ≥ 1M only. Extended to 2^30 for MI300X (192 GB; 1G F32 + dst = 8 GB) — also
# matches the bench's top size (1e9). OOM-guarded per-N, so small-VRAM cards skip.
const TUNED_N_LOG2 = (20, 22, 24, 26, 28, 30)

const NITEM_GRID = (1, 2, 4, 8, 16)
const WG_WAVES   = (8, 16, 32)

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
# Per-N tune: full sweep at one N.
# -----------------------------------------------------------------------------

function _tune_per_n(N::Int, ::Type{T_ref}, wg_grid;
                     arch = nothing, trials::Int = 30, verbose::Bool = true) where T_ref
    A = _gpu_rand(T_ref, N); dst = similar(A)
    runs = NamedTuple[]
    for Ni in NITEM_GRID, wg in wg_grid
        Ni * wg <= N || continue
        f = () -> KF.scan!(*, dst, A; Nitem=Ni, workgroup=wg)
        try
            _warmup(f)
            us = _bench_us(f; trials)
            push!(runs, (; Nitem=Ni, workgroup=wg, us))
        catch
        end
    end
    # Never-worse-than-default guard: bench the handwritten default heuristic as
    # a candidate so the per-N winner can never regress it. The wave64 wg grid
    # ({512,1024}) can't reach the 128/256 workgroups the default may use.
    if arch !== nothing
        try
            dni = KF.default_nitem(arch, KF.Scan1D, N, T_ref)
            dwg = KF.default_workgroup(arch, KF.Scan1D, N, T_ref)
            if dni * dwg <= N
                f = () -> KF.scan!(*, dst, A; Nitem=dni, workgroup=dwg)
                _warmup(f)
                push!(runs, (; Nitem=dni, workgroup=dwg, us=_bench_us(f; trials)))
            end
        catch
        end
    end
    isempty(runs) && return nothing
    sort!(runs, by = r -> r.us)
    best = runs[1]
    verbose && @printf("  N=2^%-2d (%11d) : Ni=%-2d wg=%-4d  %10.2f µs\n",
                       round(Int, log2(N)), N,
                       best.Nitem, best.workgroup, best.us)
    return (; N, best.Nitem, best.workgroup, best.us)
end

# -----------------------------------------------------------------------------
# Regime construction + codegen.
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

function _qualified_type_name(::Type{T}) where T
    m = parentmodule(T); n = nameof(T)
    m === Core || m === Base ? string(n) : string(m, ".", n)
end

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

# Generic `default_nitem`: cross-dtype via byte-count scaling.
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
    println(io, "    return clamp(nitem_f32 * 4 ÷ sizeof(T), 1, 32)")
    println(io, "end")
    println(io)
end

# Type-specific `default_nitem`: direct N-keyed, no scaling.
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
    is_generic = (T_ref === Float32)
    type_spec = is_generic ? "::Type{T}) where T" :
                "::Type{$(_qualified_type_name(T_ref))})"
    label = is_generic ? "" : " (type-specific: $(_qualified_type_name(T_ref)))"

    println(buf, "# Generated by data/tuning/scan/autotune.jl on ",
                 today(), label)
    println(buf, "# Per-N tuned at N=",
                 join(["2^$(round(Int, log2(w.N)))" for w in winners], "/"))

    sorted = sort(winners, by = w -> w.N)
    N_min = sorted[1].N
    fallback_ni = sorted[1].Nitem
    fallback_wg = sorted[1].workgroup

    regs_wg = _regimes(sorted, :workgroup)
    regs_ni = _regimes(sorted, :Nitem)

    _emit_n_func(buf, arch_tag, op_sym, "default_workgroup",
                 regs_wg, fallback_wg, N_min, type_spec)

    if is_generic
        _emit_nitem_func_generic(buf, arch_tag, op_sym, regs_ni,
                                 fallback_ni, 4 * N_min)
    else
        _emit_nitem_func_specific(buf, arch_tag, op_sym, regs_ni,
                                  fallback_ni, N_min, type_spec)
    end

    return String(take!(buf))
end

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

Tune `scan!` for the active GPU's arch.

- `T_ref = Float32` (default): emit generic `default_*` methods that
  generalize cross-dtype via byte-count scaling.
- `T_ref = <concrete type>`: emit type-specific overrides dispatched on
  `::Type{T_ref}`. The block uses a type-suffixed marker
  (`# --- BEGIN Scan1D_<TypeName> ---`) so it doesn't clobber the generic
  block or other type-specific blocks.

User-defined types must be loadable from the KernelForge module (the
generated `<Arch>_inline.jl` is `Base.include`d into that module).
"""
function autotune(::Type{T_ref} = Float32;
                  trials::Int = 30, verbose::Bool = true) where T_ref
    arch = KF.detect_arch(_gpu_zeros(Float32, 1))
    arch_tag = String(nameof(typeof(arch)))
    root = pkgdir(KernelForge)
    out_path = joinpath(root, "data", "tuning", "$(arch_tag)_inline.jl")

    wg_grid = _wg_grid(arch)
    op_marker = T_ref === Float32 ? :Scan1D :
                Symbol("Scan1D_", _qualified_type_name(T_ref))

    verbose && @printf("autotune(Scan1D, %s) on %s%s\n",
                       string(T_ref), arch_tag,
                       T_ref === Float32 ? "" : "  [type-specific]")
    verbose && @printf("  warpsz=%d  wg grid: %s\n", KF.get_warpsize(arch), wg_grid)
    verbose && @printf("  output: %s\n", out_path)
    verbose && @printf("  block marker: %s\n", op_marker)

    A0 = _gpu_rand(T_ref, 1_000_000); dst0 = similar(A0)
    _warmup(() -> KF.scan!(*, dst0, A0), 500)
    A0 = nothing; dst0 = nothing

    verbose && println("\nPer-N winners (full (Nitem, wg) sweep):")
    winners = NamedTuple[]
    for k in TUNED_N_LOG2
        N = 1 << k
        result = try
            _tune_per_n(N, T_ref, wg_grid; arch, trials, verbose)
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

    block = _emit_block(arch_tag, :Scan1D, winners, T_ref)
    _write_merged(out_path, op_marker, block)
    verbose && @printf("\nWrote methods:\n%s", block)
    verbose && @printf("File: %s\n", out_path)
    return winners
end

if abspath(PROGRAM_FILE) == @__FILE__
    autotune()
end
