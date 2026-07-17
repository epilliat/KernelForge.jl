# =============================================================================
# data/tuning/sample_sort/autotune.jl
# =============================================================================
# Sample-sort (general-comparator path) autotuner for KernelForge.jl.
#
# TUNES the leaf cap `default_leaf_max(arch, T)` — the max bucket handed to the
# batched bitonic leaf. A bigger 4-byte leaf lets the partition loop stop one
# level earlier (pure SMALL-N lever: at 1e8 the big class never fires), so the
# search benches only small/mid N.
#
# SEARCH SPACE IS TINY BY CONSTRUCTION. The leaf uses STATIC `@localmem` (L·sizeof
# (T) B, +L B on the tag path) and the bitonic ladder tops at Val(8192), so the
# only valid caps are {4096, 8192} for 4-byte and {4096} for 8-byte (8192·8 B =
# 64 KB blows the 48 KB static cap). `leaf_max_cap(T)` is that ceiling; the tuned
# value is clamped to [LEAF_MAX, cap]. 8-byte therefore has nothing to search —
# only the 4-byte class is a real 1-bit (4096 vs 8192) occupancy-vs-leaf-size
# choice, which this confirms per-arch (replacing the hand-set S1 heuristic).
#
# WHY END-TO-END. Like the radix autotuner, the leaf cap only ships if it moves
# the WHOLE `sample_sort`, so `_bench_sample_us` times the actual public call
# (partition loop + leaf), gated on issorted + multiset correctness.
#
# This is also the SCAFFOLD for the richer sample-sort knobs (partition tile =
# ITEMS_PER_THREAD, adaptive FANOUT): add a `_*_candidates` + a bench and emit
# them alongside the leaf cap under new inline blocks.
#
# Usage (backend auto-detected from the project's CUDA.jl / AMDGPU.jl):
#   julia --project=test/envs/cuda -e \
#     'include("data/tuning/sample_sort/autotune.jl"); autotune_sample_all()'
#   # single size class:
#   julia ... -e 'include("data/tuning/sample_sort/autotune.jl"); autotune_sample_leaf(UInt32)'
#   # ~20 s wiring smoke (writes <arch>_sample.json only, never the inline file):
#   julia ... data/tuning/sample_sort/autotune.jl --dev
# =============================================================================

using KernelForge
using KernelAbstractions
using KernelIntrinsics
using Printf
using Statistics
using Dates

const KF = KernelForge
const KA = KernelAbstractions

# -----------------------------------------------------------------------------
# Backend abstraction — same convention as data/tuning/sort/autotune.jl. Each
# backend body is wrapped in `@eval begin … end` so the dead branch's macros
# aren't expanded at lowering time on the other backend.
# -----------------------------------------------------------------------------
const _CUDA_LOADED = try; @eval import CUDA; true; catch; false; end
const _AMD_LOADED  = try; @eval import AMDGPU; true; catch; false; end
const _BACKEND_KIND =
    _CUDA_LOADED ? :cuda :
    _AMD_LOADED  ? :roc  :
    error("sample_sort autotune: no CUDA.jl or AMDGPU.jl in this project's environment")

if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend() = CUDA.CUDABackend()
        _gpu_sync()    = CUDA.synchronize()
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
        _gpu_backend() = AMDGPU.ROCBackend()
        _gpu_sync()    = AMDGPU.synchronize()
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

# Full-range host-random keys (sort perf is key-distribution sensitive; [0,1)
# floats cluster exponents and under-measure). Sample sort is O(N)-buffered, so
# small/mid N here never stresses RNG-count Int32 truncation.
_rand_host(::Type{T}, m::Int) where {T<:Integer}      = rand(T, m)
_rand_host(::Type{T}, m::Int) where {T<:AbstractFloat} = rand(T, m)
function _gpu_rand(::Type{T}, n::Integer) where T
    a = KA.allocate(_gpu_backend(), T, Int(n))
    copyto!(a, _rand_host(T, Int(n)))
    a
end

# -----------------------------------------------------------------------------
# Candidate leaf caps: powers of two the ladder supports, within [LEAF_MAX, cap].
# -----------------------------------------------------------------------------
function _leaf_candidates(::Type{T}) where {T}
    cap = KF.leaf_max_cap(T)
    filter(L -> KF.LEAF_MAX <= L <= cap, (4096, 8192))
end

# End-to-end sample_sort time (µs) at leaf cap `L`, or Inf on incorrectness.
# Workspace is built ONCE per (T,N) and reused across trials (leaf cap ≥ LEAF_MAX
# ⇒ the LEAF_MAX-sized workspace is never under-provisioned).
function _bench_sample_us(::Type{T}, N::Int, L::Int, ws) where {T}
    src = _gpu_rand(T, N)
    ref = sort(Array(src))
    out = KF.sample_sort(src; leaf_max = L, ws = ws)
    got = Array(out)
    (issorted(got) && got == ref) || return Inf
    _bench_us(() -> (KF.sample_sort(src; leaf_max = L, ws = ws); nothing))
end

const SELECT_MARGIN = 0.02   # require a ≥2% win over 4096 to move off the floor

# Tune the leaf cap for one size class. Returns (winner_L, rows) where rows =
# [(N, L4096_us, L8192_us_or_nothing, chosen_L)]. The cap is N-independent in the
# emit (large N ignores it), so we pick the value with the best geomean time
# across the benched small/mid N, defaulting to 4096 unless 8192 clearly wins.
function autotune_sample_leaf(::Type{T};
                              Ns = (10^5, 10^6, 10^7),
                              dev::Bool = false) where {T}
    backend = _gpu_backend()
    arch    = KF.detect_arch(_gpu_rand(T, 16))
    cands   = _leaf_candidates(T)
    @printf("\n== sample_sort leaf autotune : %s on %s ==\n", T, arch)
    @printf("   candidates: %s\n", collect(cands))

    rows = Vector{NamedTuple}()
    ratios = Dict(L => Float64[] for L in cands)
    for N in Ns
        ws = KF.sample_sort_workspace(T, N; backend)
        base = _bench_sample_us(T, N, 4096, ws)
        times = Dict{Int,Float64}()
        for L in cands
            times[L] = L == 4096 ? base : _bench_sample_us(T, N, L, ws)
            if isfinite(times[L]) && isfinite(base) && times[L] > 0
                push!(ratios[L], base / times[L])
            end
        end
        chosen = 4096
        for L in cands
            L == 4096 && continue
            if isfinite(times[L]) && times[L] < base * (1 - SELECT_MARGIN)
                chosen = L
            end
        end
        push!(rows, (N = N, base_us = base,
                     big_us = get(times, 8192, NaN), chosen = chosen))
        @printf("   N=%-9d 4096=%8.1fµs  8192=%8.1fµs  → %d\n",
                N, base, get(times, 8192, NaN), chosen)
    end

    # N-independent winner: 8192 only if its geomean speedup over 4096 clears the
    # margin (and it is a candidate at all).
    winner = 4096
    if 8192 in cands && !isempty(ratios[8192])
        g = exp(mean(log.(ratios[8192])))
        @printf("   geomean 8192/4096 speedup: %.3f×\n", g)
        g >= 1 + SELECT_MARGIN && (winner = 8192)
    end
    @printf("   → leaf cap for %s (sizeof %d): %d\n", T, sizeof(T), winner)
    (arch = arch, winner = winner, rows = rows)
end

# -----------------------------------------------------------------------------
# Inline-block splice (copied from data/tuning/sort/autotune.jl — small, kept
# local so this script has no include-order dependency on the radix tuner).
# -----------------------------------------------------------------------------
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

# Emit the per-arch `default_leaf_max` override. Only the 4-byte class is tunable
# (8-byte is pinned to 4096 by the static cap), so the method body is a single
# `sizeof(T) <= 4 ? <winner4> : 4096`.
function _emit_sample_leaf_block(arch_tag::String, winner4::Int)
    """
    # Generated by data/tuning/sample_sort/autotune.jl on $(Dates.today())
    # Leaf cap for the general-comparator sample-sort path (small-N lever).
    # 4-byte class tuned end-to-end; 8-byte pinned to 4096 by the 48 KB static cap.
    @inline KernelForge.default_leaf_max(
        ::KernelForge.$(arch_tag), ::Type{T}) where {T} =
        sizeof(T) <= 4 ? $(winner4) : 4096
    """
end

# Drive both size classes and persist. Emits the inline override ONLY if the
# 4-byte winner differs from the generic default (8192) — a tuning file records
# DEVIATIONS from the shipped heuristic; if the default already wins, emit nothing.
function autotune_sample_all(; dev::Bool = false)
    r4 = autotune_sample_leaf(UInt32; dev = dev)   # size_4 (== Float32 cap-wise)
    r8 = autotune_sample_leaf(UInt64; dev = dev)   # size_8 (pinned 4096, sanity)
    arch_tag = string(nameof(typeof(r4.arch)))
    dir  = dirname(@__DIR__)                        # data/tuning/
    generic4 = KF.default_leaf_max(r4.arch, UInt32)

    # Plain-text record (no JSON dep — the roc/cuda test envs don't carry JSON3;
    # the winner is also printed above and spliced into the inline file below).
    tpath = joinpath(dir, "$(arch_tag)_sample.txt")
    open(tpath, "w") do io
        println(io, "# sample_sort leaf autotune — $(arch_tag) — $(Dates.today())")
        println(io, "leaf_max_size_4 = $(r4.winner)   # (4-byte, tunable 4096|8192)")
        println(io, "leaf_max_size_8 = $(r8.winner)   # (8-byte, pinned 4096 by static cap)")
        println(io, "# rows (N, 4096_us, 8192_us, chosen):")
        for r in r4.rows
            println(io, "#   $(r.N)  $(round(r.base_us; digits=1))  $(round(r.big_us; digits=1))  $(r.chosen)")
        end
    end
    @printf("\nwrote %s\n", tpath)
    dev && (@printf("[dev] inline untouched\n"); return)

    ipath = joinpath(dir, "$(arch_tag)_inline.jl")
    if r4.winner != generic4
        _splice_inline_block!(ipath, "SampleLeaf", _emit_sample_leaf_block(arch_tag, r4.winner))
        @printf("spliced SampleLeaf block into %s (leaf4=%d)\n", ipath, r4.winner)
    else
        @printf("4-byte winner == generic default (%d) — no inline override needed\n", generic4)
    end
end

# CLI: `julia ... autotune.jl [--dev]`
if abspath(PROGRAM_FILE) == @__FILE__
    autotune_sample_all(; dev = ("--dev" in ARGS))
end
