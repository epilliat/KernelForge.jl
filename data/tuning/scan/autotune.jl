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

# Ni=32 added 2026-06: now that the vload/vstore alignment overflow is fixed
# (KI), 32-item blocks compile for 8-byte aggregates (32xFloat64 = 256 B). F64
# scan is ~22% faster at Ni=32 (halves the tile/block count → less lookback
# protocol). Ni=64 (512 B) over-spills and regresses, so 32 is the ceiling.
#
# Non-pow-2 candidates (12/20/24/28) added 2026-07 for the transposed wide-H
# kernel (scan_kernel_transposed!, sizeof(H)≥8 on CUDA): its shared-memory
# transpose hits bank conflicts on power-of-2 Nitem strides (A100 F64 Ni16 = 50%)
# while the odd/non-pow-2 stride is conflict-free (Ni20 = 73%). The dispatch
# (scan!) auto-routes each fitting candidate to transposed-vs-blocked, so this one
# grid tunes both families; the winner emerges per dtype. Harmless for narrow H
# (blocked/packed path ranks them and pow-2 still wins there).
const NITEM_GRID = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
# Extended down to 2/4 waves (2026-06: 128/256 threads on wave64) so the sweep
# can reach the smaller ~16 KB tiles (e.g. 256 threads × 16 items × 4 B for F32)
# that vendor scans (rocPRIM/CUB) favour on MI300X — the post-packed-descriptor
# kernel may prefer them. The `wp*m <= 1024` filter caps per arch.
const WG_WAVES   = (2, 4, 8, 16, 32)

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
    warpsz = arch === nothing ? 32 : KF.get_warpsize(arch)
    # Which split-path families to bench (2-family knob). For the autotune f=g=identity
    # so H=S=T_ref; `:transposed` is a candidate only where the transposed kernel is
    # applicable (wide isbits H) AND its tile fits shared. Benching BOTH families at
    # every (Ni,wg) — including `:blocked` at small tiles — is what lets the winner be
    # discovered per arch with no fit-gate blind spot (see scan.jl default_scan_family).
    # `:transposed` is only a real candidate where the runtime would actually LAUNCH it. Two
    # gates, and the second one was missing:
    #   * the transposed kernel must apply at all (wide isbits H);
    #   * the packed-128 path must NOT pre-empt it. `_scan_impl!` tests `scan_use_packed128`
    #     FIRST, so on CDNA3 with an 8-byte primitive H (MI300A + Float64) the transposed kernel
    #     is never launched, whatever `family` says. Benching it there measured the SAME kernel
    #     twice and let noise crown a "transposed" winner, which then got emitted as a
    #     `default_scan_family` method the runtime ignores. It also fed non-pow-2 Nitem into a
    #     path that rejects it, so half the grid died in the `catch` below without a word.
    applicable = KF.scan_transposed_applicable(T_ref, T_ref) &&
                 !(arch !== nothing && KF.scan_use_packed128(arch, T_ref))
    runs = NamedTuple[]
    skipped = NamedTuple[]
    for Ni in NITEM_GRID, wg in wg_grid
        Ni * wg <= N || continue
        # Blocked `vload`/`vstore!` compiles ONLY for pow-2 Nitem (non-pow-2 hits
        # `vload_multi`, a dynamic call that fails GPU codegen on some KI versions —
        # env-dependent, so never bench/emit it). Transposed handles any Nitem. This
        # guarantees `:blocked` is only ever paired with a pow-2 Nitem, on any machine.
        fams = Symbol[]
        ispow2(Ni) && push!(fams, :blocked)
        (applicable && KF.scan_transposed_fits(T_ref, Ni, wg, warpsz)) && push!(fams, :transposed)
        isempty(fams) && continue
        for fam in fams
            # Scan with `+`, NOT `*`. Scan is bandwidth-bound and the emitted
            # `default_nitem/workgroup` are op-agnostic, so we must measure the pure
            # memory cost. A cumulative PRODUCT of rand∈[0,1) underflows to denormals
            # within ~40 elements; denormal-multiply latency is Nitem-dependent (more
            # within-thread underflow at larger Nitem) and mis-ranked the tiles —
            # it made the autotune pick Nitem=8 when Nitem=16 is ~3-13% faster in the
            # real (`+`) benchmark on A100. `+` on rand∈[0,1) never degenerates.
            f = () -> KF.scan!(+, dst, A; Nitem=Ni, workgroup=wg, family=fam)
            try
                _warmup(f)
                us = _bench_us(f; trials)
                push!(runs, (; Nitem=Ni, workgroup=wg, family=fam, us))
            catch e
                # A config that will not compile or launch is expected (shared-memory limits,
                # register pressure) — but count it and say so. This `catch` used to be silent,
                # and it quietly ate every non-pow-2 Nitem on the packed-128 path: half the grid
                # vanished without a word, which is exactly how a tuning table ends up wrong.
                push!(skipped, (; Nitem=Ni, workgroup=wg, family=fam,
                                  why=first(sprint(showerror, e), 60)))
            end
        end
    end
    if verbose && !isempty(skipped)
        @printf("  N=%d: %d/%d candidates skipped (e.g. Ni=%d wg=%d %s: %s)\n",
                N, length(skipped), length(skipped) + length(runs),
                skipped[1].Nitem, skipped[1].workgroup, skipped[1].family, skipped[1].why)
    end

    # Never-worse-than-default guard: bench the current default heuristic as a candidate
    # so the per-N winner can never regress it. The wave64 wg grid ({512,1024}) can't
    # reach the 128/256 workgroups the default may use. IMPORTANT: label it with the
    # family it would ACTUALLY run as — a non-pow-2 default Nitem is a transposed config
    # (the runtime forces `!ispow2 -> transposed`); recording it as `:blocked` would
    # mislabel a transposed win as blocked and corrupt the emitted family.
    if arch !== nothing
        try
            dni = KF.default_nitem(arch, KF.Scan1D, N, T_ref)
            dwg = KF.default_workgroup(arch, KF.Scan1D, N, T_ref)
            dfam = ispow2(dni) ? :blocked : :transposed
            fits_ok = dfam === :blocked ||
                      (applicable && KF.scan_transposed_fits(T_ref, dni, dwg, warpsz))
            if dni * dwg <= N && fits_ok
                f = () -> KF.scan!(+, dst, A; Nitem=dni, workgroup=dwg, family=dfam)
                _warmup(f)
                push!(runs, (; Nitem=dni, workgroup=dwg, family=dfam, us=_bench_us(f; trials)))
            end
        catch
        end
    end
    isempty(runs) && return nothing
    sort!(runs, by = r -> r.us)
    best = runs[1]
    verbose && @printf("  N=2^%-2d (%11d) : Ni=%-2d wg=%-4d %-11s %10.2f µs\n",
                       round(Int, log2(N)), N,
                       best.Nitem, best.workgroup, string(best.family), best.us)
    return (; N, best.Nitem, best.workgroup, best.family, best.us)
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

# Symbol-valued regimes for the `:family` knob (`_regimes` is Int-typed).
function _family_regimes(winners::Vector{<:NamedTuple})
    isempty(winners) && return Tuple{Int,Symbol}[]
    sorted = sort(winners, by = w -> w.N)
    out = Tuple{Int,Symbol}[]
    cur = sorted[1].family
    for i in 2:length(sorted)
        nxt = sorted[i].family
        if nxt != cur
            thr = isqrt(sorted[i-1].N * sorted[i].N)
            push!(out, (thr, cur))
            cur = nxt
        end
    end
    push!(out, (typemax(Int), cur))
    return out
end

# COHERENT joint regimes: segment on the WHOLE (Nitem, workgroup, family) triple so all
# three emitted functions share the same N-thresholds. This is REQUIRED for wide types:
# a non-pow-2 Nitem is only valid with `:transposed` (the blocked kernel can't compile
# it), so `default_nitem` and `default_scan_family` must never disagree at a boundary.
# Returns (regs_wg, regs_ni, regs_fam), each Vector of (threshold, value) over the same
# thresholds (adjacent-equal values are left unmerged — harmless, just a redundant ternary).
function _coherent_regimes(winners::Vector{<:NamedTuple})
    sorted = sort(winners, by = w -> w.N)
    key(w) = (w.Nitem, w.workgroup, w.family)
    segs = Tuple{Int,Int,Int,Symbol}[]
    cur = key(sorted[1])
    for i in 2:length(sorted)
        k = key(sorted[i])
        if k != cur
            thr = isqrt(sorted[i-1].N * sorted[i].N)
            push!(segs, (thr, cur[1], cur[2], cur[3]))
            cur = k
        end
    end
    push!(segs, (typemax(Int), cur[1], cur[2], cur[3]))
    regs_wg  = Tuple{Int,Int}[(s[1], s[3]) for s in segs]
    regs_ni  = Tuple{Int,Int}[(s[1], s[2]) for s in segs]
    regs_fam = Tuple{Int,Symbol}[(s[1], s[4]) for s in segs]
    return regs_wg, regs_ni, regs_fam
end

# Emit `default_scan_family` (Symbol-valued: :blocked / :transposed), same regime
# shape as `_emit_n_func`. Only called when some regime is :transposed (else the
# generic `default_scan_family(::AbstractArch,…)=:blocked` fallback already applies).
function _emit_family_func(io, arch_tag, op_sym,
                           regimes::Vector{Tuple{Int,Symbol}}, fallback::Symbol,
                           N_min::Int, type_spec::AbstractString)
    println(io, "@inline function KernelForge.default_scan_family(")
    println(io, "    ::KernelForge.", arch_tag, ", ::Type{KernelForge.",
                op_sym, "}, n, ", type_spec)
    println(io, "    n < ", N_min, " ? ", repr(fallback), " :")
    for (thr, val) in regimes
        if thr == typemax(Int)
            println(io, "        ", repr(val))
        else
            println(io, "        n < ", thr, " ? ", repr(val), " :")
        end
    end
    println(io, "end")
    println(io)
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

    if is_generic
        # F32 generic: byte-scaled default_nitem; family never :transposed (sizeof<8),
        # and blocked candidates at non-pow-2 Nitem fail to compile so winners are always
        # pow-2 — no coherence hazard, keep independent per-knob regimes.
        _emit_n_func(buf, arch_tag, op_sym, "default_workgroup",
                     _regimes(sorted, :workgroup), fallback_wg, N_min, type_spec)
        _emit_nitem_func_generic(buf, arch_tag, op_sym, _regimes(sorted, :Nitem),
                                 fallback_ni, 4 * N_min)
    else
        # Type-specific (may use the transposed family + non-pow-2 Nitem): emit
        # workgroup / nitem / family from COHERENT joint segments so the triple never
        # disagrees (non-pow-2 Nitem always pairs with :transposed — see _coherent_regimes).
        regs_wg, regs_ni, regs_fam = _coherent_regimes(sorted)
        _emit_n_func(buf, arch_tag, op_sym, "default_workgroup",
                     regs_wg, fallback_wg, N_min, type_spec)
        _emit_nitem_func_specific(buf, arch_tag, op_sym, regs_ni,
                                  fallback_ni, N_min, type_spec)
        # Only emit default_scan_family when transposed wins somewhere (else the generic
        # `:blocked` fallback in scan.jl already covers this arch/type).
        if any(r -> r[2] === :transposed, regs_fam)
            _emit_family_func(buf, arch_tag, op_sym, regs_fam, sorted[1].family,
                              N_min, type_spec)
        end
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
    _warmup(() -> KF.scan!(+, dst0, A0), 500)
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
