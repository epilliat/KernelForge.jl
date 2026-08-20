# =============================================================================
# data/tuning/sort/autotune.jl
# =============================================================================
# Sort1D (radix onesweep) autotuner for KernelForge.jl. Searches the launch-shape
# knobs (Nitem, Nchunks, workgroup) per (dtype-size, N) and persists the winners:
#   • data/tuning/<arch>_sort.json         — human-readable per-cell winners
#   • data/tuning/<arch>_inline.jl          — Sort1D block of default_* overrides
#     (regenerated in place; other blocks untouched), loaded at extension __init__.
#
# WHY WHOLE-SORT, NOT PER-KERNEL. A per-onesweep-kernel nsys metric is MISLEADING
# for sort — a kernel-level speedup need not survive end-to-end (proven 2026-07-10:
# a batched-5b kernel that nsys said was −18% moved the whole sort ~0% and even
# +21% at 1e7). So `_bench_sort_us` times the ACTUAL `KF.sort!` whole sort
# (histogram + npasses × onesweep), the only metric that ships.
#
# KNOBS & CONSTRAINTS:
#   Nitem   — vector-load width / items per warp lane. MUST be pow-2 (the bucket
#             histogram kernel's KI `vload_multi` fails GPU codegen on non-pow-2,
#             same invariant as scan's blocked kernel).
#   Nchunks — chunks per warp. block_size_max = workgroup·Nchunks·Nitem.
#   workgroup — threads/block.
#   SHARED FIT: shared_sorted = block_size_max·sizeof(T) plus the histogram
#             (256·(wg/32)·4) + aux must fit the 48 KB static cap → we require
#             block_size_max·sizeof(T) ≤ ~40 KB. 8-byte types therefore get
#             smaller tiles than 4-byte (matches the hand-written defaults).
#
# Backend-agnostic (CUDA / AMDGPU), same five hooks as the matvec autotuner.
#
# Usage:
#   # ~30 s wiring smoke → writes <arch>_sort_dev.json, never the real files:
#   julia --project=test/envs/cuda data/tuning/sort/autotune.jl --dev
#   # full sweep for the 4-byte and 8-byte size classes (~min, arch-dependent):
#   julia --project=test/envs/cuda -e \
#     'include("data/tuning/sort/autotune.jl"); autotune_sort_all()'
#   # single size class (UInt32 ⇒ size_4):
#   julia ... -e 'include("data/tuning/sort/autotune.jl"); autotune_sort(UInt32)'
# =============================================================================

using KernelForge
using KernelAbstractions
using KernelIntrinsics
using JSON3
using Printf
using Statistics
using Dates

const KF = KernelForge
const KA = KernelAbstractions
const KI = KernelIntrinsics

# -----------------------------------------------------------------------------
# Backend abstraction (same convention as data/tuning/matvec/autotune.jl):
#   _gpu_backend / _total_memory / _gpu_rand / _gpu_zeros / _bench_us
# Each backend body is wrapped in `@eval begin … end` so the dead branch's
# backend-specific macros aren't expanded at lowering time.
# -----------------------------------------------------------------------------
const _CUDA_LOADED = try; @eval import CUDA; true; catch; false; end
const _AMD_LOADED  = try; @eval import AMDGPU; true; catch; false; end
const _BACKEND_KIND =
    _CUDA_LOADED ? :cuda :
    _AMD_LOADED  ? :roc  :
    error("sort autotune: no CUDA.jl or AMDGPU.jl in this project's environment")

if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend()                             = CUDA.CUDABackend()
        _total_memory()                            = Int(CUDA.totalmem(CUDA.device()))
        _gpu_rand_prim(::Type{T}, dims...) where T = CUDA.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T     = CUDA.zeros(T, dims...)
        _gpu_sync()                                = CUDA.synchronize()
        function _bench_us(f; trials::Int = 8)
            starts = [CUDA.CuEvent() for _ in 1:trials]
            stops  = [CUDA.CuEvent() for _ in 1:trials]
            for i in 1:trials
                CUDA.record(starts[i]); f(); CUDA.record(stops[i])
            end
            CUDA.synchronize(stops[trials])
            median(Float64[CUDA.elapsed(starts[i], stops[i]) * 1e6 for i in 1:trials])
        end
    end
else
    @eval begin
        _gpu_backend()                             = AMDGPU.ROCBackend()
        _total_memory()                            = Int(AMDGPU.HIP.properties(AMDGPU.device()).totalGlobalMem)
        _gpu_rand_prim(::Type{T}, dims...) where T = AMDGPU.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T     = AMDGPU.zeros(T, dims...)
        _gpu_sync()                                = AMDGPU.synchronize()
        function _bench_us(f; trials::Int = 8)
            probe_us = Float64(AMDGPU.@elapsed f()) * 1e6
            inner = probe_us <= 0 ? 10 : clamp(round(Int, 5000.0 / probe_us), 1, 10)
            samples = Float64[]
            for _ in 1:trials
                t = Float64(AMDGPU.@elapsed begin; for _ in 1:inner; f(); end; end)
                push!(samples, t * 1e6 / inner)
            end
            isempty(samples) ? NaN : median(samples)
        end
    end
end

# Large-array-safe uniform fill (some backends truncate the RNG count to Int32
# at ≥2^31 elements). rand over the FULL type range — sort perf is key-
# distribution sensitive; [0,1) floats are degenerate (clustered exponents →
# high passes scatter to few buckets → under-measures the scatter cost). Integer
# rand(UInt) is full-range, matching CUB's --type uintNN benchmark.
const _RAND_CHUNK = 1 << 30
function _gpu_rand(::Type{T}, n::Integer) where T
    a = KA.allocate(_gpu_backend(), T, Int(n))
    flat = reshape(a, :)
    total = Int(n); off = 0
    while off < total
        m = min(_RAND_CHUNK, total - off)
        copyto!(view(flat, off+1:off+m), _rand_host(T, m))
        off += m
    end
    a
end
# host-side random (cheap enough vs the sort; avoids the Int32 RNG-count issue
# entirely and gives full-range integer keys for every backend).
_rand_host(::Type{T}, m::Int) where {T<:Integer} = rand(T, m)
_rand_host(::Type{T}, m::Int) where {T<:AbstractFloat} = rand(T, m)

# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------
# The shared-memory ceiling is a RUNTIME DEVICE QUERY, never an arch table:
# 99 KB on Ada, 163 KB on A100, 64 KB of LDS on gfx942. The onesweep allocates
# dynamically, so this — not the old 48 KB static cap — is what bounds the tile.
_shared_cap() = KI.max_dynamic_localmem(KI.device(_gpu_backend(), 1))
_warpsz() = Int(KI.get_warpsize(KI.device(_gpu_backend(), 1)))

const NITEM_GRID = (8, 16, 32)          # pow-2 only
const NCHUNKS_GRID = (1, 2, 3, 4, 6, 8)
# The workgroup is no longer pinned to Nbuckets=256. Larger workgroups buy a big
# tile with FEWER items/thread, which is what stops the register spill; wg=1024
# and wg=512 were the A100 winners at 1e7. `workgroup >= 256` is required (the
# first 256 threads carry the buckets).
const WG_GRID_DEFAULT = (256, 512, 1024)
# RR: re-read the item from src in phase 5a instead of holding it in a register.
# Wins on 4-byte keys, loses on 8-byte ones -> genuinely per-(dtype, N).
const RR_GRID = (false, true)

# The :idxstage family (2-byte index staging, CDNA3 8-byte keys) runs its OWN
# config space. Its whole win is the BIG tile, whose MEASURED optimum is Nitem=4
# — BELOW the value family's grid floor of 8. Sweeping idxstage on the value grid
# therefore could not even reach the shipped config, and the never-worse anchor
# read the value-family `default_*` while the runtime launches `default_idxstage_*`
# → the guard was anchored on a config that never runs. Mirrors sort1d.jl.
const NITEM_GRID_IDX = (4, 8, 16, 32)

# Families to sweep, mirroring the runtime gate in `default_sort_family`: where
# the runtime would actually launch :idxstage (CDNA3 AND 8-byte keys), BOTH
# families are legal candidates and the winner is DISCOVERED per (arch, dtype, N)
# — the same 2-family-knob precedent as scan's :blocked/:transposed. Everywhere
# else :value is the only kernel, so the sweep is byte-identical to before.
_sort_families(arch, ::Type{T}, n) where T =
    KF.default_sort_family(arch, KF.Sort1D, n, T) === :idxstage ?
        (:value, :idxstage) : (:value,)

# A candidate must beat the default by MORE than this fraction to be shipped.
# Below it, configs are within run-to-run sort variance (±~20% at large N) and
# flipping between tied configs just encodes noise → keep the default.
const SELECT_MARGIN = 0.06
const CONFIRM_ROUNDS = 6                 # interleaved winner-vs-default re-measures

# All (Nitem, Nchunks, workgroup, rr) with pow-2 Nitem whose shared layout fits
# the DEVICE cap. Checking host-side keeps the sweep cheap; a config that slipped
# through would throw (catchably, non-sticky) rather than hang, so the sweep can
# always carry on in-process either way.
function sort_candidates(::Type{T}; wg_grid = WG_GRID_DEFAULT,
                         families = (:value,)) where T
    cap = _shared_cap(); wsz = _warpsz()
    cands = Tuple{Int,Int,Int,Bool,Symbol}[]
    for fam in families
        if fam === :value
            for wg in wg_grid, ni in NITEM_GRID, nc in NCHUNKS_GRID, rr in RR_GRID
                wg >= 256 || continue
                KF.onesweep_shmem(T, 256, wsz, wg, nc, ni) <= cap || continue
                push!(cands, (ni, nc, wg, rr, :value))
            end
        else
            # idxstage stages a UInt16 index, so its shared layout is sized on
            # UInt16 (NOT T) -> `onesweep_idxstage_shmem`, and it has NO `rr`
            # knob (phase 5b already re-reads the value from src).
            for wg in wg_grid, ni in NITEM_GRID_IDX, nc in NCHUNKS_GRID
                wg >= 256 || continue
                KF.onesweep_idxstage_shmem(256, wsz, wg, nc, ni) <= cap || continue
                push!(cands, (ni, nc, wg, false, :idxstage))
            end
        end
    end
    cands
end

# -----------------------------------------------------------------------------
# Whole-sort benchmark for one config. Pre-allocates tmp; guards compile/OOM
# failures → NaN (a bad candidate must not abort the sweep).
# -----------------------------------------------------------------------------
function _bench_sort_us(::Type{T}, N::Int, ni::Int, nc::Int, wg::Int, rr::Bool,
                        fam::Symbol = :value;
                        trials::Int = 8, warm::Int = 4) where T
    src = dst = tmp = nothing
    try
        src = _gpu_rand(T, N); dst = KA.allocate(_gpu_backend(), T, N)
        tmp = KF.get_allocation(KF.Sort1D, src; Nitem=ni, Nchunks=nc, workgroup=wg)
        # `family=fam` is what makes the sweep honest: without it every candidate
        # ran whatever `default_sort_family` picked, so on CDNA3 8-byte keys the
        # value-family grid was silently driving the idxstage kernel.
        f() = KF.sort!(dst, src; algorithm=:radix, Nitem=ni, Nchunks=nc, workgroup=wg,
                       rr=rr, family=fam, tmp=tmp)
        f(); for _ in 1:warm; f(); end; _gpu_sync()
        sorted = issorted(Array(dst))
        sorted || return (NaN, false)
        (_bench_us(f; trials), true)
    catch e
        (NaN, false)
    finally
        src = dst = tmp = nothing
    end
end

# Interleaved re-measure of a winner config vs the default config, alternating
# rounds so any thermal/clock/allocator drift hits both equally → the RATIO is
# drift-free. Returns (median_winner_us, median_default_us).
function _confirm_pair(::Type{T}, N::Int, win::Tuple{Int,Int,Int,Bool,Symbol},
                       def::Tuple{Int,Int,Int,Bool,Symbol};
                       rounds::Int = CONFIRM_ROUNDS) where T
    ws = Float64[]; ds = Float64[]
    for _ in 1:rounds
        w, okw = _bench_sort_us(T, N, win[1], win[2], win[3], win[4], win[5]; trials = 6, warm = 2)
        d, okd = _bench_sort_us(T, N, def[1], def[2], def[3], def[4], def[5]; trials = 6, warm = 2)
        okw && push!(ws, w); okd && push!(ds, d)
    end
    (isempty(ws) ? NaN : median(ws), isempty(ds) ? NaN : median(ds))
end

# -----------------------------------------------------------------------------
# Tune one size class (represented by dtype T) across N.
# Returns Vector of (N => (best_ni, best_nc, best_wg, best_us, cub_ratio_unknown)).
# -----------------------------------------------------------------------------
function autotune_sort(::Type{T} = UInt32;
                       sizes = (1_000_000, 10_000_000, 100_000_000),
                       wg_grid = WG_GRID_DEFAULT,
                       arch = nothing, dev::Bool = false, verbose::Bool = true) where T
    arch = something(arch, KF.detect_arch(_gpu_zeros(T, 1)))
    arch_tag = String(nameof(typeof(arch)))
    n0 = first(sizes)
    fams = _sort_families(arch, T, n0)
    cands = sort_candidates(T; wg_grid, families = fams)
    verbose && @printf("[sort autotune] arch=%s  T=%s (size_%d)  %d candidates  families=%s  sizes=%s\n",
                       arch_tag, string(T), sizeof(T), length(cands),
                       string(fams), string(sizes))

    # Default (untuned) config for the never-worse guard, READ FROM THE FAMILY THE
    # RUNTIME ACTUALLY LAUNCHES. The value family's `default_*` and the idxstage
    # family's `default_idxstage_*` are SEPARATE config functions (sort1d.jl); on
    # CDNA3 8-byte keys the runtime ships idxstage, so anchoring the guard on the
    # value defaults measured a config that never runs.
    dfam = KF.default_sort_family(arch, KF.Sort1D, n0, T)
    if dfam === :idxstage
        dni = KF.default_idxstage_nitem(arch, T)
        dnc = KF.default_idxstage_nchunks(arch, T)
        dwg = KF.default_idxstage_workgroup(arch, T)
        drr = false          # idxstage has no rr knob
    else
        dni = KF.default_nitem(arch, KF.Sort1D, n0, T)
        dnc = KF.default_nchunks(arch, KF.Sort1D, n0, T)
        dwg = KF.default_workgroup(arch, KF.Sort1D, n0, T)
        drr = KF.default_sort_rr(arch, KF.Sort1D, n0, T)
    end
    dcfg = (dni, dnc, dwg, drr, dfam)

    # N, ni, nc, wg, rr, family, us, default_us
    results = Tuple{Int,Int,Int,Int,Bool,Symbol,Float64,Float64}[]
    for N in sizes
        # measure the current default first (anchor for the never-worse guard)
        default_us, ok0 = _bench_sort_us(T, N, dni, dnc, dwg, drr, dfam)
        rawbest = (dni, dnc, dwg, drr, dfam, ok0 ? default_us : Inf)
        for (ni, nc, wg, rr, fam) in cands
            (ni, nc, wg, rr, fam) == dcfg && continue
            us, ok = _bench_sort_us(T, N, ni, nc, wg, rr, fam)
            ok || continue
            if us < rawbest[6]; rawbest = (ni, nc, wg, rr, fam, us); end
        end
        # NOISE MARGIN + PASS-2: only override the default if a candidate beats it
        # by > MARGIN, AND the gap survives an interleaved re-measure (large-N sort
        # has ±20% run-to-run variance → single-shot wins flap; encoding that noise
        # produces alternating (32,1)/(16,2) defaults with no physical basis).
        chosen = (dni, dnc, dwg, drr, dfam, isfinite(default_us) ? default_us : rawbest[6])
        is_cand_win = rawbest[1:5] != dcfg &&
                      isfinite(default_us) && rawbest[6] < default_us * (1 - SELECT_MARGIN)
        if is_cand_win
            wd, dd = _confirm_pair(T, N, rawbest[1:5], dcfg)
            if isfinite(wd) && isfinite(dd) && wd < dd * (1 - SELECT_MARGIN)
                chosen = (rawbest[1], rawbest[2], rawbest[3], rawbest[4], rawbest[5], wd); default_us = dd
            else
                # revert to default; adopt the confirmed (interleaved) baseline for BOTH
                # the chosen time and default_us so the report isn't skewed by pass-1 variance.
                dd_use = isfinite(dd) ? dd : default_us
                chosen = (dni, dnc, dwg, drr, dfam, dd_use); default_us = dd_use
            end
        end
        push!(results, (N, chosen[1], chosen[2], chosen[3], chosen[4], chosen[5], chosen[6], default_us))
        if verbose
            spd = isfinite(default_us) && chosen[6] > 0 ? default_us / chosen[6] : NaN
            tag = chosen[1:5] == dcfg ? "[default]" : "[WIN vs ($dni,$dnc,$dwg,rr=$drr,$dfam)]"
            @printf("  N=%-11d -> (ni=%d, nc=%d, wg=%d, rr=%s, fam=%s) tile=%-6d %8.1fµs  default %8.1fµs  (%.2f×) %s\n",
                    N, chosen[1], chosen[2], chosen[3], chosen[4], chosen[5],
                    chosen[3] * chosen[2] * chosen[1], chosen[6], default_us, spd, tag)
        end
        flush(stdout)
    end
    any_win = any(r -> (r[2], r[3], r[4], r[5], r[6]) != dcfg, results)
    (; arch_tag, T, size_class = sizeof(T), results, default = dcfg, any_win)
end

autotune_sort_all(; kwargs...) = begin
    r4 = autotune_sort(UInt32; kwargs...)
    r8 = autotune_sort(UInt64; kwargs...)
    persist_sort_tunings([r4, r8]; dev = get(kwargs, :dev, false))
    (r4, r8)
end

# -----------------------------------------------------------------------------
# Persistence: JSON summary + regenerate the Sort1D block of <arch>_inline.jl.
# -----------------------------------------------------------------------------
_tuning_dir() = joinpath(pkgdir(KernelForge), "data", "tuning")

function persist_sort_tunings(tunes; dev::Bool = false)
    isempty(tunes) && return
    arch_tag = first(tunes).arch_tag
    dir = _tuning_dir()

    # 1) JSON summary --------------------------------------------------------
    payload = Dict{String,Any}("arch" => arch_tag, "generated" => string(today()), "sort" => Dict{String,Any}())
    for t in tunes
        cell = Dict{String,Any}()
        for (N, ni, nc, wg, rr, fam, us, dus) in t.results
            cell[string(N)] = Dict("Nitem"=>ni, "Nchunks"=>nc, "workgroup"=>wg, "rr"=>rr,
                                   "family"=>String(fam), "tile"=>wg*nc*ni,
                                   "us"=>round(us; digits=1), "default_us"=>round(dus; digits=1))
        end
        payload["sort"]["size_$(t.size_class)"] = cell
    end
    any_win_all = any(t -> get(t, :any_win, true), tunes)
    payload["sort_override"] = any_win_all
    jpath = joinpath(dir, dev ? "$(arch_tag)_sort_dev.json" : "$(arch_tag)_sort.json")
    open(jpath, "w") do io; JSON3.pretty(io, payload); end
    @printf("[sort autotune] wrote %s\n", jpath)
    dev && return  # dev never touches the inline artifact

    # 2) inline default_* block — ONLY if a confirmed win exists. A tuning file
    # must only OVERRIDE the generic defaults where a real, interleaved-confirmed
    # win beats them; if every cell chose the default, emit NOTHING (strip any
    # stale block) so the generic defaults stand and re-runs can't self-perpetuate
    # a noise-driven override (the loaded block would become the next "default").
    ipath = joinpath(dir, "$(arch_tag)_inline.jl")
    if any_win_all
        block = _emit_sort_inline_block(arch_tag, tunes)
        _splice_inline_block!(ipath, "Sort1D", block)
        # Retire the legacy per-size-class blocks. They dispatch on
        # `T<:Union{UInt32,Int32,Float32}`, which is MORE SPECIFIC than the block
        # we just wrote (`where T` + a sizeof(T) branch), so leaving them in place
        # would let them silently shadow the new tuning for exactly the types they
        # name. The new block covers both size classes, so they are superseded.
        for legacy in ("Sort1D_size4", "Sort1D_size8")
            _strip_inline_block!(ipath, legacy) &&
                @printf("[sort autotune] retired legacy %s block (superseded)\n", legacy)
        end
        @printf("[sort autotune] spliced Sort1D block into %s (confirmed wins)\n", ipath)
    else
        removed = _strip_inline_block!(ipath, "Sort1D")
        @printf("[sort autotune] NO confirmed win — %s (generic defaults optimal for %s sort)\n",
                removed ? "stripped stale Sort1D block from $ipath" : "no override emitted", arch_tag)
    end
end

# Remove a `# --- BEGIN <name> --- … # --- END <name> ---` block if present.
# Returns true if a block was removed.
function _strip_inline_block!(path, name)
    isfile(path) || return false
    marker_begin = "# --- BEGIN $name ---"; marker_end = "# --- END $name ---"
    txt = read(path, String)
    occursin(marker_begin, txt) || return false
    pre = txt[1:prevind(txt, first(findfirst(marker_begin, txt)))]
    post_start = last(findfirst(marker_end, txt)) + 1
    post = post_start <= lastindex(txt) ? txt[post_start:end] : ""
    open(path, "w") do io; write(io, rstrip(pre) * (isempty(strip(post)) ? "\n" : "\n" * lstrip(post))); end
    true
end

# Build `n < c ? v : …` regime branches from tuned (N=>value) points. Cutoffs at
# the geometric midpoint between consecutive tuned N; last value is the tail.
function _regime_expr(pairs::Vector{<:Pair{Int,<:Any}})
    length(pairs) == 1 && return string(pairs[1].second)
    parts = String[]
    for i in 1:length(pairs)-1
        cut = round(Int, sqrt(float(pairs[i].first) * float(pairs[i+1].first)))
        push!(parts, "n < $cut ? $(pairs[i].second)")
    end
    push!(parts, string(pairs[end].second))
    join(parts, " :\n            ")
end

function _emit_sort_inline_block(arch_tag, tunes)
    io = IOBuffer()
    println(io, "# Generated by data/tuning/sort/autotune.jl on $(today())")
    println(io, "# WHOLE-SORT tuned (not per-kernel); per-N winners for Nitem/Nchunks/workgroup/rr.")

    # The winning FAMILY per size class is taken at the LARGEST tuned N: the shipped
    # `default_sort_family` is per-(arch,n,T) but `default_idxstage_*` takes NO `n`,
    # so one idxstage config per size class is all the runtime API can express — and
    # the largest N is where the sort machinery (and the benchmark) actually lives.
    fam_of  = Dict(t.size_class => last(t.results)[6] for t in tunes)
    dfam_of = Dict(t.size_class => t.default[5]       for t in tunes)
    val_tunes = [t for t in tunes if fam_of[t.size_class] === :value]
    idx_tunes = [t for t in tunes if fam_of[t.size_class] === :idxstage]

    # `UnknownArch` (concrete) — NOT `AbstractArch`, which is an ABSTRACT type and
    # therefore not instantiable: the emitted fallback branch would throw
    # `MethodError: no constructors have been defined for AbstractArch`. Latent in
    # the A100/MI300A/RTX1000 artifacts generated before this fix; unreachable in
    # practice only because radix keys need a `uint_map` (so sizeof(T) <= 8).
    fallback_arch = "KernelForge.UnknownArch()"

    # --- value family: per-N regimes for the four value-family knobs ----------
    # Size classes whose winner is :idxstage are DELIBERATELY absent here: that
    # family reads `default_idxstage_*`, so emitting value knobs for them would be
    # dead code that merely looks like tuning.
    if !isempty(val_tunes)
        for knob in ("nitem", "nchunks", "workgroup", "sort_rr")
            field = knob == "nitem" ? 2 : knob == "nchunks" ? 3 : knob == "workgroup" ? 4 : 5
            println(io, "@inline function KernelForge.default_$(knob)(")
            println(io, "        ::KernelForge.$(arch_tag), ::Type{KernelForge.Sort1D}, n, ::Type{T}) where T")
            println(io, "    s = sizeof(T)")
            first_class = true
            for t in val_tunes
                pairs = [Int(r[1]) => (knob == "sort_rr" ? r[field] : Int(r[field])) for r in t.results]
                cond = t.size_class <= 4 ? "s <= 4" : "s <= 8"
                kw = first_class ? "if" : "elseif"
                println(io, "    $kw $cond")
                println(io, "        $(_regime_expr(pairs))")
                first_class = false
            end
            println(io, "    else")
            println(io, "        KernelForge.default_$(knob)($fallback_arch, KernelForge.Sort1D, n, T)")
            println(io, "    end")
            println(io, "end\n")
        end
    end

    # --- family override, emitted only when the tuned winner disagrees with the
    # arch default. Every tuned class is listed, so the emitted method can never
    # shadow one class with another class's answer.
    if any(c -> fam_of[c] !== dfam_of[c], keys(fam_of))
        println(io, "@inline function KernelForge.default_sort_family(")
        println(io, "        ::KernelForge.$(arch_tag), ::Type{KernelForge.Sort1D}, n, ::Type{T}) where T")
        println(io, "    s = sizeof(T)")
        first_class = true
        for t in tunes
            cond = t.size_class <= 4 ? "s <= 4" : "s <= 8"
            kw = first_class ? "if" : "elseif"
            println(io, "    $kw $cond")
            println(io, "        :$(fam_of[t.size_class])")
            first_class = false
        end
        println(io, "    else")
        println(io, "        KernelForge.default_sort_family($fallback_arch, KernelForge.Sort1D, n, T)")
        println(io, "    end")
        println(io, "end\n")
    end

    # --- idxstage config: the shipped signature is (arch, T) with NO `n`, so one
    # config per size class (winner at the largest tuned N; see fam_of above).
    if !isempty(idx_tunes)
        for (knob, field) in (("nitem", 2), ("nchunks", 3), ("workgroup", 4))
            println(io, "@inline function KernelForge.default_idxstage_$(knob)(")
            println(io, "        ::KernelForge.$(arch_tag), ::Type{T}) where T")
            println(io, "    s = sizeof(T)")
            first_class = true
            for t in idx_tunes
                cond = t.size_class <= 4 ? "s <= 4" : "s <= 8"
                kw = first_class ? "if" : "elseif"
                println(io, "    $kw $cond")
                println(io, "        $(Int(last(t.results)[field]))")
                first_class = false
            end
            println(io, "    else")
            println(io, "        KernelForge.default_idxstage_$(knob)($fallback_arch, T)")
            println(io, "    end")
            println(io, "end\n")
        end
    end
    String(take!(io))
end

# Replace (or append) a `# --- BEGIN <name> --- … # --- END <name> ---` block.
function _splice_inline_block!(path, name, block)
    marker_begin = "# --- BEGIN $name ---"
    marker_end   = "# --- END $name ---"
    wrapped = "$marker_begin\n$block$marker_end\n"
    txt = isfile(path) ? read(path, String) : ""
    if occursin(marker_begin, txt)
        pre = txt[1:prevind(txt, first(findfirst(marker_begin, txt)))]
        post_start = last(findfirst(marker_end, txt)) + 1
        post = post_start <= lastindex(txt) ? txt[post_start:end] : ""
        txt = pre * wrapped * post
    else
        txt = (isempty(txt) ? "" : rstrip(txt) * "\n\n") * wrapped
    end
    open(path, "w") do io; write(io, txt); end
end

# -----------------------------------------------------------------------------
# CLI: `--dev` runs a fast wiring smoke (small sizes, wg=256) → *_sort_dev.json.
# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if "--dev" in ARGS
        println("[sort autotune] DEV SMOKE (small sizes, no inline write)")
        r4 = autotune_sort(UInt32; sizes = (1_000_000, 4_000_000), dev = true)
        persist_sort_tunings([r4]; dev = true)
    else
        autotune_sort_all()
    end
    println("DONE")
end
