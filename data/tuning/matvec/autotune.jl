# =============================================================================
# data/tuning/matvec/autotune.jl
# =============================================================================
# Matvec autotuner for KernelForge.jl. Produces one JSON per arch under
# data/tuning/<arch>.json with per-cell optimal (Nitem, chunksz, Nblocks,
# workgroup).
#
# Two-tier section keying inside `matvec`:
#   "size_4", "size_8", …      ← default, covers all dtypes of that size
#   "Float32", "Int32", …      ← optional type-specific override (precise=true)
# Lookup (future PR) prefers a type-specific section over the size section.
#
# Backend-agnostic: runs on CUDA (NVIDIA) or AMDGPU (AMD) — whichever package
# is installed and functional in the active project. The five backend hooks
# are defined in the "Backend abstraction" block below.
#
# Usage:
#   # ~30 s wiring smoke (writes a separate <arch>_dev.json, never pollutes
#   # the real <arch>.json):
#   julia --project=perfs/envs/benchenv/cuda \
#         data/tuning/matvec/autotune.jl --dev          # NVIDIA
#   julia --project=test/envs/roc \
#         data/tuning/matvec/autotune.jl --dev          # AMD
#
#   # Full size-keyed sweep for Float32 (size_4 section, ~30 min):
#   julia ... -e 'include("data/tuning/matvec/autotune.jl"); autotune(Float32)'
#
#   # Type-specific override for Float32 (Float32 section beats size_4 at
#   # lookup time):
#   julia ... -e 'include("data/tuning/matvec/autotune.jl");
#                 autotune(Float32; precise=true)'
#
#   # All representative sizes (size_4 then size_8):
#   julia ... -e 'include("data/tuning/matvec/autotune.jl"); autotune_all()'
#
# Note on dev mode: dev runs write to <arch>_dev.json, NEVER to <arch>.json,
# so a dev smoke can never accidentally land in the production JSON. The two
# files coexist; commit only <arch>.json.
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
# Backend abstraction — load whichever GPU backend is installed in this
# project, then define five hooks the rest of the script uses:
#   _gpu_backend()       :: KA backend object
#   _total_memory()      :: Int  — device global memory in bytes
#   _gpu_rand(T, dims…)  :: device array of uniform randoms
#   _gpu_zeros(T, dims…) :: zeroed device array (used for arch detection)
#   _bench_us(f; trials) :: Float64 — median kernel time of `f()` in µs
#
# `_bench_us` is backend-specific: CUDA uses `@profile` device timestamps
# (pure kernel time, the proven autotune metric); AMD uses `@elapsed` GPU
# events (kernel time + a near-constant host-dispatch gap — slightly noisier
# but still rank-consistent).
# -----------------------------------------------------------------------------

# Load each backend in its OWN top-level statement. `@eval import X` only
# becomes visible to code in a *newer* world age, so the import and any use of
# `X` must not share an expression. The `@eval import` succeeding is signal
# enough that the backend is usable — no `functional()` call needed here
# (which would reintroduce the world-age problem).
const _CUDA_LOADED = try
    @eval import CUDA
    true
catch
    false
end

const _AMD_LOADED = try
    @eval import AMDGPU
    true
catch
    false
end

const _BACKEND_KIND =
    _CUDA_LOADED ? :cuda :
    _AMD_LOADED  ? :roc  :
    error("autotune: no CUDA.jl or AMDGPU.jl in this project's environment")

# Backend hooks. Each branch is wrapped in `@eval begin … end`: the body is
# only lowered (and its `@elapsed` macro expanded) when that branch actually
# runs. Without the `@eval`, Julia would macro-expand the dead branch's
# `AMDGPU.@elapsed` at lowering time and hit `UndefVarError: AMDGPU`.
if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend()                         = CUDA.CUDABackend()
        _total_memory()                        = Int(CUDA.totalmem(CUDA.device()))
        _gpu_rand(::Type{T}, dims...) where T   = CUDA.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T  = CUDA.zeros(T, dims...)

        # Pure kernel time: sum of device events per trial, then median.
        # CUDA event pairs (no CUPTI, no @profile session). Per-trial wall is
        # the elapsed GPU time between two events bracketing f(); one host
        # sync drains all trials. Equivalent measurement to the @profile
        # approach (per-trial medians agree within 0.1%), but ~2-2.5× lower
        # wall per call on small-kernel candidates and dramatically lower
        # variance — CUPTI session start/stop was adding 40+ ms of jittery
        # overhead per _bench_us call.
        function _bench_us(f; trials::Int = 5)
            starts = [CUDA.CuEvent() for _ in 1:trials]
            stops  = [CUDA.CuEvent() for _ in 1:trials]
            for i in 1:trials
                CUDA.record(starts[i])
                f()
                CUDA.record(stops[i])
            end
            CUDA.synchronize(stops[trials])
            samples = Float64[CUDA.elapsed(starts[i], stops[i]) * 1e6
                              for i in 1:trials]
            return median(samples)
        end
    end
else # :roc
    @eval begin
        _gpu_backend()                         = AMDGPU.ROCBackend()
        # HIP hipDeviceProp_t.totalGlobalMem — same struct the AMDGPU
        # extension reads for multiProcessorCount.
        _total_memory()                        = Int(AMDGPU.HIP.properties(AMDGPU.device()).totalGlobalMem)
        _gpu_rand(::Type{T}, dims...) where T   = AMDGPU.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T  = AMDGPU.zeros(T, dims...)

        # Block-timing: each `@elapsed` brackets `inner` back-to-back calls.
        # Per-call host dispatch overlaps the previous kernel's execution and
        # amortizes to ~`cpu_dispatch / inner`, so the result converges on
        # pure kernel time. Dividing by `inner` also shrinks the CPU-side
        # variance (much noisier than kernel time) by the same factor.
        # Median over `trials` blocks rejects the occasional slow block.
        #
        # `inner` is ADAPTIVE: amortizing host dispatch (~µs) only matters when
        # the kernel itself is µs-scale. The giant-p cells (e.g. n=8, p=2^27)
        # have candidates running 100-500 ms/launch — a fixed inner=20 makes
        # each candidate cost 2-10 s (× hundreds per cell × 109 cells × 2
        # dtypes). A single probe launch sizes inner to ~a few ms of timed work:
        # inner=1 for ms-scale kernels (already pure kernel time, no accuracy
        # loss), up to 20 for µs-scale. Ranking stays self-consistent per cell.
        function _bench_us(f; trials::Int = 5)
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
            return median(samples)
        end
    end
end

# -----------------------------------------------------------------------------
# Shape grid — pow-2 for log2 ≤ 6, pow-8 (step 3) above. n*p memory-capped per
# (dtype, GPU): allocated `n*p*sizeof(T)` bytes for A must fit in
# `totalmem(device) / safety` so the card has room for x, dst, scratch, driver
# state, etc. Default `safety=4` ≈ uses 25 % of GPU RAM for the test matrix.
# -----------------------------------------------------------------------------

const LOG2_LEVELS = (0, 1, 2, 3, 4, 5, 6,
                     9, 12, 15, 18, 21, 24, 27, 30)
const MIN_DIM     = 100   # skip cells where n < MIN_DIM AND p < MIN_DIM

# Intermediate (non-pow2) n-levels for the wide (small-n, large-p) corner.
# The per-shape optimum flips at pow-2 batch-packing boundaries — n=8 packs
# into Nitem=8 with zero waste, but n=10 wants Nitem=1 (Nitem=8 idles ~60% of
# the vector lanes). Tuning ONLY pow-2 n therefore makes the lookup
# systematically OVER-estimate Nitem for the far-more-common non-pow2 n
# (A100: shipped 2.9× vs achievable 1.24× on n=10,p=1e6).
#
# GEOMETRIC log-midpoints round(2^(k+0.5)) between pow-2 rows, restricted to the
# SMALL-n regime {3,6,11,23,45}. Two reasons for the restriction:
#  (1) The pow-2 Nitem-packing bias bites HARDEST at small n — n=8 packs Ni=8
#      with 0 waste but n=10 wants Ni=1 (60% idle lanes); at n≈100 the relative
#      waste is small (cld(100,8)=13 → 4/104), so Ni is a weaker lever there.
#  (2) Empirically (A100): adding the LARGE intermediate levels (91,181,…)
#      REGRESSED n=100,p=1e5 (1.79→2.03×) — the n=91 cell tuned to worse
#      chunksz/workgroup than the pow-2 n=64 cell, and n=100 sits between grid
#      points on BOTH axes so a "closer" n-cell isn't a better cell. Keeping
#      only n≤45 lets n≈100 fall back to its (better) pow-2 n=64 cell — no
#      regression — while n≈10 still lands on the n=11 cell (2.96→1.11× win).
# None here is divisible by 8, so each has genuine Ni=8 packing waste and tunes
# to the lower-Nitem regime the common small-n query shapes need.
const INTERMEDIATE_N = (3, 6, 11, 23, 45)

# Wide/upper-mid corner cells the pow-2 n-grid MISSES. Two grid holes conspire
# here: (1) the n-axis jumps 64→512 (LOG2_LEVELS skips exponents 7,8), and
# (2) the diagonal cap kn+kp ≤ maximum(LOG2_LEVELS)=30 keeps n=128,256 from
# reaching the largest p. Without cells here, common wide queries (n≈100..500,
# p≈1e6..1e7) map to the n=64 cell → generic → 40% peak / 2× cuBLAS (the wide bug).
#
# CRITICAL: these tuning n MUST be non-multiples-of-8. The generic kernel's
# `vload` of Nitem elements needs the column stride (n·sizeof(T)) 32-byte
# aligned; when 8 ∤ n every other column's vload misaligns → ~2× BW loss. So a
# cell tuned at n=96 (=8·12, aligned) picks high-Nitem generic at ~74% peak, but
# the SAME config serving the common query n=100 (8 ∤ 100) collapses to ~35%.
# Tuning at non-8-multiple n (100,150,250,500) exposes that penalty, so the
# autotune correctly prefers the row-thread kernel (1 elem/thread, always
# coalesced, no Nitem-alignment dependence) — ~66% at n=100 vs generic's ~40%
# ceiling. The nearest-cell lookup then serves arbitrary wide-corner n from a
# robust row-thread cell instead of an alignment-fragile generic one.
const WIDE_CORNER_N = (100, 150, 250, 500)
const WIDE_CORNER_P = (1_000_000, 10_000_000)

# The representative wide-corner (n, p) cells that fit this dtype's memory
# budget. Reused by both `pow28_shapes` (full sweep) and the targeted merge run.
function wide_corner_shapes(::Type{T} = Float32; safety::Int = 4) where T
    budget_np = (_total_memory() ÷ safety) ÷ sizeof(T)     # max n*p that fits
    return [(n, p) for n in WIDE_CORNER_N, p in WIDE_CORNER_P if n * p <= budget_np] |> vec
end

"""
    max_log2_np(::Type{T}; safety=4) -> Int

Compute the largest `log2(n*p)` such that `n*p*sizeof(T) ≤ totalmem/safety`.
Queries the active GPU device (CUDA or AMDGPU).
"""
function max_log2_np(::Type{T}; safety::Int = 4) where T
    total_bytes = _total_memory()
    budget_bytes = total_bytes ÷ safety
    max_np = budget_bytes ÷ sizeof(T)
    cap = floor(Int, log2(max(1, max_np)))
    return min(cap, maximum(LOG2_LEVELS))   # never exceed the grid's top level
end

function pow28_shapes(::Type{T} = Float32; safety::Int = 4) where T
    cap = max_log2_np(T; safety)
    out = Tuple{Int,Int}[]
    for kn in LOG2_LEVELS, kp in LOG2_LEVELS
        kn + kp <= cap || continue
        n = 1 << kn; p = 1 << kp
        (n >= MIN_DIM || p >= MIN_DIM) || continue
        push!(out, (n, p))
    end
    # Intermediate non-pow2 n × large-p (the wide corner) — see INTERMEDIATE_N.
    # Only paired with p large enough that n*p is a genuine wide reduction
    # (MIN_DIM guard already forces p ≥ 512 for these small n).
    for n in INTERMEDIATE_N, kp in LOG2_LEVELS
        p = 1 << kp
        log2(n) + kp <= cap || continue
        (n >= MIN_DIM || p >= MIN_DIM) || continue
        push!(out, (n, p))
    end
    # Wide/upper-mid corner (n=96..256 × large p) — fills the 64→512 n-grid hole
    # and breaks the diagonal cap where memory permits. See WIDE_CORNER_N.
    append!(out, wide_corner_shapes(T; safety))
    return unique!(out)
end

# Dev shape grid — one per regime, ~30 s budget.
const DEV_SHAPES = [(10_000, 10_000), (10, 1_000_000), (1_000_000, 10)]

# -----------------------------------------------------------------------------
# Param grid — full Cartesian (validity asserts only). No n-dependent pruning.
# -----------------------------------------------------------------------------

function nitems_for(::Type{T}) where T
    # Two caps stack:
    #  - `cld(32, sizeof(T))`: per-thread unroll count drives kernel compile
    #    time. NTuple{Nitem,H} and `for i in 1:Nitem` are fully unrolled
    #    throughout matvec_kernel.jl. Nitem=16 made LLVM blow up to 15 GB RSS
    #    on a single candidate.
    #  - hard cap of 16: for T with sizeof(T) == 1 (Int8/UInt8/Bool), the
    #    `vload` of Nitem elements maps to ld.global.v* PTX with alignment ==
    #    Nitem * sizeof(T). At Nitem=32, that's a 32-byte load from a
    #    byte-strided source pointer — id_base addresses rarely align,
    #    yielding ERROR_MISALIGNED_ADDRESS at launch. Cap 16 is the largest
    #    Nitem that empirically works for Int8 on RTX1000.
    cap = min(cld(32, sizeof(T)), 16)
    return Tuple(1 << k for k in 0:floor(Int, log2(cap)))
end

const CHUNKSZ_GRID   = (1, 2, 4, 8, 16, 32, 64, 128, 256)
const NBLOCKS_GRID   = (1, 2, 4, 8, 16, 32, 64, 128, 256)
const WORKGROUP_GRID = (128, 256, 512)
# Wave/warp width of the *target* device — drives the candidate pruning and the
# @localmem element-count estimate below (lines using cld(workgroup, WARPSZ)).
# Must match what the launched kernel actually uses: matvec.jl passes
# `Val(get_warpsize(arch))`. Hardcoding 32 (NVIDIA) on a wave64 device (CDNA /
# MI300X) over-estimates `cld(wg, warpsz)` by 2× → SHARED_ELEM/MEM caps prune
# large-chunksz candidates too aggressively and under-explore the wave64 sweet
# spot. Detect it from the device instead: CDNA→64, RDNA→32 (wave32 mode),
# NVIDIA→32. Falls back to 32 if detection fails.
const WARPSZ = try
    KF.get_warpsize(KF.detect_arch(_gpu_zeros(Float32, 1)))
catch
    32
end

# Dev param grid — smaller subset (~10 valid combos per cell after asserts).
const DEV_CHUNKSZ    = (4, 16, 64)
const DEV_NBLOCKS    = (1, 8, 64)
const DEV_WORKGROUP  = (256, 512)
dev_nitems_for(::Type{T}) where T = sizeof(T) <= 4 ? (1, 4, 8) : (1, 2, 4)

# Restrictive caps on per-thread work and the @localmem aggregate array
# (matvec_kernel.jl:21 allocates `NTuple{Nitem,H}` × chunksz·cld(wg,warpsz)).
# Large Nitem×chunksz / Nitem×chunksz×workgroup combos blow up kernel
# compilation (LLVM scalarizes the aggregate → 15+ GB RSS). Kept deliberately
# tight — exotic large combos are never autotune winners anyway.
#  - NITEM_CHUNKSZ_LIMIT — cap on Nitem*chunksz (per-thread work).
#  - SHARED_ELEM_LIMIT   — cap on Nitem*chunksz*cld(workgroup,warpsz)
#                          (the @localmem element count).
#  - SHARED_MEM_LIMIT    — byte count, 48 KB static-shared launch ceiling.
const NITEM_CHUNKSZ_LIMIT = 1024
const SHARED_ELEM_LIMIT   = 4096
const SHARED_MEM_LIMIT    = 48 * 1024

function param_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T
    nitems     = dev ? dev_nitems_for(T) : nitems_for(T)
    chunkszs   = dev ? DEV_CHUNKSZ       : CHUNKSZ_GRID
    nblocks_s  = dev ? DEV_NBLOCKS       : NBLOCKS_GRID
    workgroups = dev ? DEV_WORKGROUP     : WORKGROUP_GRID
    szH        = sizeof(Base.promote_op(*, T, T))   # accumulator element size

    out = NamedTuple[]
    for Nitem in nitems, chunksz in chunkszs, Nblocks in nblocks_s, workgroup in workgroups
        # Validity asserts mirroring resolve_parameters (matvec.jl:269-270).
        chunksz <= workgroup || continue
        cld(workgroup, chunksz) * Nblocks <= p || continue
        valid = ispow2(Nblocks) ||
                chunksz * Nblocks >= workgroup ||
                chunksz * Nblocks <= WARPSZ
        valid || continue
        workgroup <= n * p || continue
        # A workgroup covers Nitem*chunksz output rows — never provision more
        # than the matrix has. Candidates with Nitem*chunksz > n are
        # over-provisioned (idle lanes), redundant with the capped version,
        # and never autotune winners. Only bites small-n cells (n ≤ 512),
        # where it cuts the candidate count drastically (n=1: 855 → ~27).
        Nitem * chunksz <= n || continue
        # Large-n regime: with enough independent output rows each workgroup
        # is already saturated, so cross-workgroup partial reduction (Nblocks>1)
        # is wasted overhead, and small chunksz under-uses per-row reuse.
        # Empirically (RTX1000 sweep): 23/24 cells with n ≥ 10_000 picked
        # chunksz ≥ 32 AND Nblocks == 1. Filter early to cut compile+bench.
        if n >= 10_000
            chunksz >= 32 || continue
            Nblocks == 1  || continue
        end
        # Restrictive caps — per-thread work and the @localmem aggregate
        # (matvec_kernel.jl:21). See the const block above.
        Nitem * chunksz <= NITEM_CHUNKSZ_LIMIT || continue
        sh_elems = Nitem * chunksz * cld(workgroup, WARPSZ)
        sh_elems <= SHARED_ELEM_LIMIT          || continue
        sh_elems * szH <= SHARED_MEM_LIMIT     || continue
        push!(out, (; kernel = :generic, Nitem, chunksz, Nblocks, workgroup))
    end
    return out
end

# -----------------------------------------------------------------------------
# Row-thread candidate family (matvec_rowthread_kernel!): 1 thread = 1 row,
# coalesced column streaming, grid tiled (row_tiles × ncb column-splits). Three
# knobs: U (col-unroll/MLP), workgroup (row-tile height, stored as the effective
# warp-multiple ≤ 1024), ncb (column-split — the key parallelism knob). The
# kernel type is parametrized ONLY by Val(U) (wg/ncb are runtime args), so the
# whole grid compiles to just |U| kernels per dtype — cheap to sweep.
# -----------------------------------------------------------------------------
const RT_U_GRID   = (8, 16)
const RT_WG_GRID   = (128, 256, 512)
const RT_NCB_GRID  = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
const DEV_RT_NCB   = (4, 32, 256)

# Row-thread only competes for wide/square/middle-band shapes with real column
# parallelism; skip tiny/tall-thin cells where the generic kernel always wins
# (few columns → nothing to split, few rows → too few warps per block).
_rowthread_applicable(n::Int, p::Int) = n >= 64 && p >= 4096

function rowthread_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T
    _rowthread_applicable(n, p) || return NamedTuple[]
    Us   = dev ? (16,)      : RT_U_GRID
    wgs  = dev ? (256,)     : RT_WG_GRID
    ncbs = dev ? DEV_RT_NCB : RT_NCB_GRID
    seen = Set{NamedTuple}()
    out  = NamedTuple[]
    for U in Us, wg0 in wgs
        # Effective row-tile height: warp-multiple, ≥ enough to cover a warp,
        # ≤ 1024. wg0 candidates that exceed n collapse to the same effective
        # wg (deduped below) — no redundant compiles.
        wg = clamp(cld(min(wg0, n), WARPSZ) * WARPSZ, WARPSZ, 1024)
        rtiles = cld(n, wg)
        # Cap ncb so total blocks (rtiles·ncb) stay ≲ 4096 and never exceed p
        # (can't split into more column-blocks than there are columns).
        ncb_cap = clamp(min(p, 4096 ÷ max(1, rtiles)), 1, 512)
        for ncb in ncbs
            (ncb <= ncb_cap && ncb <= p) || continue
            cand = (; kernel = :rowthread, U, ncb, workgroup = wg)
            cand in seen && continue
            push!(seen, cand); push!(out, cand)
        end
    end
    return out
end

# Combined candidate list across BOTH kernel families for a cell. The autotune
# ranks all of them together and stores the winning family's params + tag.
function all_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T
    return vcat(param_candidates(T, n, p; dev), rowthread_candidates(T, n, p; dev))
end

# -----------------------------------------------------------------------------
# Bench protocol — warmup + `_bench_us` (backend-specific, defined above).
# `trials` is overridden to 2 in dev mode.
# -----------------------------------------------------------------------------

function _warmup_for(f, ms::Int)
    t0 = time_ns()
    while (time_ns() - t0) < ms * 1_000_000
        f()
        # Sync per iteration: otherwise async kernel launches pile up in the
        # stream queue during the CPU spin, and the trailing synchronize ends
        # up draining all of them — turning a 5 ms warmup budget into
        # N_queued × per-iter on slow kernels (measured ~1.3 s instead of
        # 6 ms on cell (16, 16777216)).
        KA.synchronize(_gpu_backend())
    end
end

function _session_warmup()
    A = _gpu_rand(Float32, 10^4, 10^4)
    x = _gpu_rand(Float32, 10^4)
    _warmup_for(() -> KF.matvec(*, +, A, x), 1000)
end

# -----------------------------------------------------------------------------
# Parallel pre-compilation.
#
# The autotune's wall is ~50% GPUCompiler IR build for the ~500-700 unique
# `(Nitem, chunksz, Nblocks, workgroup)` tuples reused across the grid.
# Each compile is single-threaded LLVM work; running them concurrently on
# Threads.nthreads() workers cuts the compile phase by ~N×.
#
# Strategy:
#   1. Enumerate all unique specs across the cells we're about to tune.
#   2. Throttle to `nthreads()` workers via a channel; each worker compiles
#      one spec on the smallest (n, p) that satisfies validity.
#   3. After this returns, every spec is in the in-process GPUCompiler cache;
#      the main tune loop sees cache hits on every candidate (compile_ms < 1ms).
#
# Falls back to single-threaded sequential when `Threads.nthreads() == 1` —
# the wall is unchanged from the current behavior in that case (compile cost
# is paid up-front instead of inside the bench loop).
# -----------------------------------------------------------------------------

# Build a zero-arg closure that runs one candidate (either family) and leaves
# the result in `dst`. Generic → `matvec!` with the tuned knobs (+ reused tmp);
# row-thread → `_matvec_rowthread_impl!` (self-allocates its small partial each
# call — a host-side pool alloc that doesn't perturb device-time measurement).
function _run_closure(params::NamedTuple, A, x, dst, ::Type{H}, arch) where H
    n, p = size(A)
    if params.kernel === :rowthread
        return () -> (KF._matvec_rowthread_impl!(*, +, identity, dst, A, x, H, n, p, arch;
                        U = params.U, wg = params.workgroup, ncb = params.ncb); dst)
    else
        tmp = params.Nblocks > 1 ?
            KF.get_allocation(KF.MatVec, *, +, A, x,
                              params.chunksz, params.Nblocks, params.workgroup,
                              nothing, params.Nitem, nothing) :
            nothing
        return () -> (KF.matvec!(*, +, dst, A, x;
                        Nitem = params.Nitem, chunksz = params.chunksz,
                        Nblocks = params.Nblocks, workgroup = params.workgroup, tmp); dst)
    end
end

function _precompile_one(params::NamedTuple, ::Type{T}) where T
    # Allocate-launch-sync per spec. Populates the in-process GPUCompiler
    # cache so the bench loop sees `compile_ms < 1ms`. NOTE: lever 1 (compile
    # without GPU launch, via `KF.compile_kernel_only`) was tried and
    # measured no speedup — the dominant cost here is the CPU compile under
    # GPUCompiler's global lock (~190 ms/spec wall on 22 threads, only ~2×
    # speedup vs single-threaded), not the GPU launch+sync (~8 ms/spec on
    # cached specs). Kept this simpler path; `KF.compile_kernel_only` is
    # retained as infrastructure for future autotunes that may benefit.
    if params.kernel === :rowthread
        n = clamp(params.workgroup, 64, 4096)
        p = max(params.ncb * 8, 4096)
    else
        min_n = params.Nitem * params.chunksz
        min_p = cld(params.workgroup, params.chunksz) * params.Nblocks
        n = nextpow(2, max(min_n, 1))
        p = nextpow(2, max(min_p, 1))
    end
    A = _gpu_rand(T, n, p)
    x = _gpu_rand(T, p)
    backend = get_backend(A)
    H = Base.promote_op(*, T, T)
    dst = KA.allocate(backend, H, n)
    arch = KF.detect_arch(A)
    _run_closure(params, A, x, dst, H, arch)()
    KA.synchronize(backend)
    return
end

function _precompile_specs(shapes, ::Type{T};
                           skip_done::Set = Set{Tuple{Int,Int}}(),
                           dev::Bool = false,
                           verbose::Bool = true) where T
    # Collect unique specs across cells that haven't been resumed-skipped.
    specs_set = Set{NamedTuple}()
    for (n, p) in shapes
        (n, p) in skip_done && continue
        for params in all_candidates(T, n, p; dev)
            push!(specs_set, params)
        end
    end
    specs = collect(specs_set)
    n_specs = length(specs)
    n_threads = max(1, Threads.nthreads())
    verbose && @printf("  precompile: %d unique specs (threads=%d)\n",
                       n_specs, n_threads)
    n_specs == 0 && return 0

    t0 = time_ns()
    ch = Channel{NamedTuple}(n_specs)
    for s in specs; put!(ch, s); end
    close(ch)

    done = Threads.Atomic{Int}(0)
    n_failed = Threads.Atomic{Int}(0)
    @sync for _ in 1:n_threads
        Threads.@spawn begin
            for spec in ch
                try
                    _precompile_one(spec, T)
                catch
                    # Invalid combos that survived param_candidates may still
                    # trip the kernel's runtime asserts — count and skip.
                    Threads.atomic_add!(n_failed, 1)
                end
                d = Threads.atomic_add!(done, 1) + 1
                verbose && d % 25 == 0 &&
                    print(stderr, "\r  precompile: ", d, "/", n_specs, "          ")
            end
        end
    end
    wall = (time_ns() - t0) / 1e9
    verbose && @printf("\n  precompile done: %.1f s (%d failed)\n",
                       wall, n_failed[])
    return n_specs
end

# -----------------------------------------------------------------------------
# Per-cell tune + per-candidate diagnostics.
#
# For each candidate we record:
#   compile_ms : wall of the FIRST f() call (includes GPUCompiler IR build + 1
#                bench iteration). For the very first candidate of the very
#                first cell it also absorbs session-startup; flagged is_first
#                in the CSV so analysis can filter.
#   bench_us   : median per-trial pure kernel time (CUDA.@profile / AMDGPU
#                @elapsed device-side).
#   wall_ms    : total wall the candidate consumed (setup + compile + warmup
#                + bench + correctness check).
# -----------------------------------------------------------------------------

_diag_csv_path(json_out::AbstractString) =
    replace(json_out, r"\.json$" => "_diag.csv")

function _diag_init(csv_path::AbstractString)
    isfile(csv_path) && return
    mkpath(dirname(csv_path))
    open(csv_path, "w") do io
        println(io, "cell_idx,n,p,cand_idx,n_cands,Nitem,chunksz,Nblocks,workgroup,compile_ms,bench_us,wall_ms,ok,is_first")
    end
end

# Map either family's params into the fixed (Nitem,chunksz,Nblocks,workgroup)
# diag columns. Row-thread has no such knobs — encode (U, ncb, 0, workgroup)
# so the diag CSV stays parseable (it's diagnostic only, not read at runtime).
_diag_params(p) = p.kernel === :rowthread ?
    (Nitem = p.U, chunksz = p.ncb, Nblocks = 0, workgroup = p.workgroup) :
    (Nitem = p.Nitem, chunksz = p.chunksz, Nblocks = p.Nblocks, workgroup = p.workgroup)

function _diag_row(io, cell_idx, n, p, cand_idx, n_cands, params, compile_ms, bench_us, wall_ms, ok, is_first)
    dp = _diag_params(params)
    println(io, cell_idx, ",", n, ",", p, ",", cand_idx, ",", n_cands, ",",
            dp.Nitem, ",", dp.chunksz, ",", dp.Nblocks, ",", dp.workgroup, ",",
            compile_ms, ",", bench_us, ",", wall_ms, ",", ok ? 1 : 0, ",", is_first ? 1 : 0)
end

function tune_cell(n::Int, p::Int, ::Type{T};
                   trials::Int = 3, dev::Bool = false,
                   correctness::Bool = false,
                   top_k::Int = 10,
                   coarse_k::Int = 20,
                   cell_idx::Int = 0, n_cells::Int = 0,
                   diag_csv::Union{Nothing,AbstractString} = nothing) where T
    cands = all_candidates(T, n, p; dev)
    isempty(cands) && return nothing

    A = try
        _gpu_rand(T, n, p)
    catch e
        @warn "alloc failed; skipping cell" n p T e
        return nothing
    end
    x = _gpu_rand(T, p)
    backend = get_backend(A)
    H = Base.promote_op(*, T, T)
    dst = KA.allocate(backend, H, n)
    arch = KF.detect_arch(A)
    # GPU-side Float64 reference (only built if `correctness=true`).
    ref_gpu = ref_max = nothing
    if correctness
        A_f64 = Float64.(A)
        x_f64 = Float64.(x)
        ref_gpu = A_f64 * x_f64
        ref_max = maximum(abs.(ref_gpu))
        A_f64 = x_f64 = nothing
    end

    n_cands = length(cands)

    # =========================================================================
    # COARSE PHASE — 1 trial per candidate, no warmup. With precompile already
    # done, the first f() is a cache hit and compile_ms ≈ 0.1ms; bench_us is
    # the pure kernel time of one launch. Goal: identify the top `coarse_k`
    # candidates cheaply, eliminate the rest.
    # =========================================================================
    coarse = Vector{NamedTuple}(undef, n_cands)
    for (ci, params) in enumerate(cands)
        t_wall0 = time_ns()
        compile_ms, coarse_us, ok_c = NaN, NaN, false
        try
            f = _run_closure(params, A, x, dst, H, arch)
            t_c0 = time_ns()
            f(); KA.synchronize(backend)
            compile_ms = (time_ns() - t_c0) / 1e6
            coarse_us  = _bench_us(f; trials=1)
            ok_c = isfinite(coarse_us)
        catch
        end
        coarse[ci] = (; ci, params, compile_ms,
                        coarse_us, coarse_wall_ms = (time_ns() - t_wall0) / 1e6,
                        ok = ok_c)

        if cell_idx > 0
            msg = string("cell ", cell_idx, "/", n_cells, "  coarse ",
                         ci, "/", n_cands, "  compile=", round(compile_ms, digits=1), "ms",
                         "  bench=", round(coarse_us, digits=1), "µs",
                         "  (n=", n, " p=", p, ")")
            print(stderr, "\r  ", msg, "          ")
            try
                open("/tmp/autotune_progress.txt", "w") do io
                    println(io, msg)
                end
            catch
            end
        end
    end

    # =========================================================================
    # FINE PHASE — full bench protocol (warmup + `trials`-trial bench +
    # correctness check) on the top `coarse_k` survivors PER FAMILY.
    #
    # Stratify survivors by kernel family. The coarse rank is a single noisy
    # trial; a family with many mediocre candidates (generic: hundreds) can
    # crowd the other family's genuine winner (rowthread: tens) out of a
    # global top-k, so it never gets an accurate fine measurement and loses by
    # default. Taking the top-k of EACH family guarantees both are fine-benched,
    # and the final pick is decided by the accurate multi-trial fine numbers.
    # Measured impact: without this, n=128/256 large-p cells mis-picked generic
    # (74% peak) over the faster rowthread (80%); with it they pick rowthread.
    # =========================================================================
    valid = sort!([c for c in coarse if c.ok], by = c -> c.coarse_us)
    gen_valid = [c for c in valid if c.params.kernel === :generic]
    rt_valid  = [c for c in valid if c.params.kernel === :rowthread]
    survivors = vcat(first(gen_valid, min(coarse_k, length(gen_valid))),
                     first(rt_valid,  min(coarse_k, length(rt_valid))))

    # ci → fine bench/relerr/wall, only populated for survivors that benched OK.
    fine_us     = Dict{Int,Float64}()
    fine_relerr = Dict{Int,Float64}()
    fine_wall   = Dict{Int,Float64}()
    for (k, c) in enumerate(survivors)
        t_wall0 = time_ns()
        params = c.params
        f_us, f_re, f_ok = NaN, NaN, false
        try
            f = _run_closure(params, A, x, dst, H, arch)
            _warmup_for(f, 5)
            f_us = _bench_us(f; trials)
            if correctness
                out = f()
                diff_max = maximum(abs.(Float64.(out) .- ref_gpu))
                f_re = diff_max / max(eps(Float64), ref_max)
                f_ok = f_re < 1e-3
            else
                f_ok = isfinite(f_us)
            end
        catch
        end
        fine_wall[c.ci] = (time_ns() - t_wall0) / 1e6
        if f_ok && isfinite(f_us)
            fine_us[c.ci]     = f_us
            fine_relerr[c.ci] = f_re
        end

        if cell_idx > 0
            msg = string("cell ", cell_idx, "/", n_cells, "  fine ",
                         k, "/", length(survivors),
                         "  bench=", round(f_us, digits=1), "µs",
                         "  (n=", n, " p=", p, ")")
            print(stderr, "\r  ", msg, "          ")
        end
    end

    # =========================================================================
    # Diag CSV — one row per original candidate. bench_us = fine median if
    # the candidate survived to fine phase and passed; else its coarse value.
    # wall_ms aggregates both phases.
    # =========================================================================
    diag_io = diag_csv === nothing ? nothing : open(diag_csv, "a")
    try
        for c in coarse
            bench_us = get(fine_us, c.ci, c.coarse_us)
            wall_ms  = c.coarse_wall_ms + get(fine_wall, c.ci, 0.0)
            row_ok   = haskey(fine_us, c.ci) || c.ok
            diag_io !== nothing &&
                _diag_row(diag_io, cell_idx, n, p, c.ci, n_cands, c.params,
                          c.compile_ms, bench_us, wall_ms, row_ok, c.ci == 1)
        end
    finally
        diag_io !== nothing && close(diag_io)
        cell_idx > 0 && println(stderr)
    end

    isempty(fine_us) && return nothing
    # Build the fine-sorted survivor list (winner first), in cand-id order
    # of the survivors for stable downstream behavior.
    fine_rows = NamedTuple[]
    for c in coarse
        haskey(fine_us, c.ci) || continue
        push!(fine_rows, (; c.params, median_us = fine_us[c.ci],
                            relerr = get(fine_relerr, c.ci, NaN), ok = true))
    end
    sort!(fine_rows, by = r -> r.median_us)
    best = fine_rows[1]
    top  = [(; r.params, r.median_us) for r in first(fine_rows, min(top_k, length(fine_rows)))]
    return (; best.params, best.median_us, best.relerr, n_candidates = n_cands, top = top)
end

# -----------------------------------------------------------------------------
# Pass 2 — robustness refinement.
#
# For each tuned cell, take the top-K candidates (from pass 1) and re-bench
# each on the cell itself + 4 axis-neighbors in the shape grid. Pick the
# candidate with the lowest MEAN bench across those 5 shapes — that's the
# most "robust" winner, less likely to mis-rank the cell because of a noisy
# single-shape median.
#
# Cost: ~94 cells × ~10 top-K × ~5 shapes × ~3 trials ≈ 14k extra benches.
# Most shapes are sub-100MB, so allocation overhead is small; the biggest
# cells dominate.
# -----------------------------------------------------------------------------

# 4 axis-neighbors of (n, p) within `shapes` — the closest shape with the
# next-smaller / next-larger n (p fixed), then likewise for p.
function _axis_neighbors(n::Int, p::Int, shapes::AbstractVector{<:Tuple})
    ns = sort!(unique([nn for (nn, pp) in shapes if pp == p]))
    ps = sort!(unique([pp for (nn, pp) in shapes if nn == n]))
    neighbors = Tuple{Int,Int}[]
    ni = searchsortedfirst(ns, n)
    ni > 1            && push!(neighbors, (ns[ni-1], p))
    ni < length(ns)   && push!(neighbors, (ns[ni+1], p))
    pi = searchsortedfirst(ps, p)
    pi > 1            && push!(neighbors, (n, ps[pi-1]))
    pi < length(ps)   && push!(neighbors, (n, ps[pi+1]))
    return neighbors
end

# Bench one (params, n, p, T) — same protocol as the per-candidate path in
# tune_cell, factored out for reuse in pass 2. Returns bench_us (Float64) or
# NaN if the candidate fails (e.g. validity assertion).
function _bench_shape(params::NamedTuple, n::Int, p::Int, ::Type{T};
                      trials::Int = 3) where T
    A   = _gpu_rand(T, n, p)
    x   = _gpu_rand(T, p)
    backend = get_backend(A)
    H   = Base.promote_op(*, T, T)
    dst = KA.allocate(backend, H, n)
    arch = KF.detect_arch(A)
    f = _run_closure(params, A, x, dst, H, arch)
    try
        # first call warms; per-iter sync inside _warmup_for
        f(); KA.synchronize(backend)
        _warmup_for(f, 5)
        return _bench_us(f; trials)
    catch
        return NaN
    end
end

# For each tuned cell, evaluate its top-K candidates on (cell + neighbors)
# and replace `cells[i]` params with the robustness-winning candidate.
# `cells` is the Vector{Dict{String,Any}} the writer emits. `cell_tops` is
# a parallel Vector aligned with `cells` holding the top-K from pass 1.
function robust_refine_cells!(cells::Vector{Dict{String,Any}},
                              cell_tops::AbstractVector,
                              shapes::AbstractVector{<:Tuple},
                              ::Type{T};
                              trials::Int = 3, verbose::Bool = true) where T
    length(cells) == length(cell_tops) || error("cells / cell_tops length mismatch")
    n_changed = 0
    t0 = time_ns()
    for (i, cell) in enumerate(cells)
        top = cell_tops[i]
        length(top) <= 1 && continue
        n, p = Int(cell["n"]), Int(cell["p"])
        neighbors = _axis_neighbors(n, p, shapes)
        isempty(neighbors) && continue

        # Build per-candidate measurement vector across (cell + neighbors).
        # The cell's own bench is already in `top[k].median_us` from pass 1.
        scores = Vector{Float64}(undef, length(top))
        for (k, c) in enumerate(top)
            cand_us = Float64[c.median_us]
            for (nn, pp) in neighbors
                us = _bench_shape(c.params, nn, pp, T; trials)
                isfinite(us) && push!(cand_us, us)
            end
            # Normalize per-shape to that shape's own min (across the top-K)
            # would be ideal, but the simpler mean already favors candidates
            # that don't blow up on any shape.
            scores[k] = sum(cand_us) / length(cand_us)
        end

        # Find the most robust candidate.
        kbest = argmin(scores)
        if kbest != 1
            winner = top[kbest]
            # The robust winner may switch families vs pass 1, so rebuild the
            # cell's param keys wholesale (clear the old family's fields first).
            empty!(cell)
            merge!(cell, _cell_dict(n, p, winner.params))
            n_changed += 1
            verbose && @printf("  [%3d] (n=%-10d p=%-10d) robust winner #%d %s (mean=%.1fµs vs pass1 %.1fµs)\n",
                               i, n, p, kbest,
                               string(winner.params), scores[kbest], top[1].median_us)
        end
    end
    wall = (time_ns() - t0) / 1e9
    verbose && @printf("\n  pass-2 robustness: %d / %d winners changed (%.1fs)\n",
                       n_changed, length(cells), wall)
    return n_changed
end

# -----------------------------------------------------------------------------
# JSON I/O — merge-on-write so multiple runs accumulate in one arch JSON.
# -----------------------------------------------------------------------------

# Serialize one cell's winning params (either family) to the mutable Dict the
# writers consume. Always stamps `kernel` so lookups are unambiguous.
function _cell_dict(n::Int, p::Int, params::NamedTuple)
    if params.kernel === :rowthread
        return Dict{String,Any}("n" => n, "p" => p, "kernel" => "rowthread",
            "U" => params.U, "ncb" => params.ncb, "workgroup" => params.workgroup)
    else
        return Dict{String,Any}("n" => n, "p" => p, "kernel" => "generic",
            "Nitem" => params.Nitem, "chunksz" => params.chunksz,
            "Nblocks" => params.Nblocks, "workgroup" => params.workgroup)
    end
end

function _default_json_out(arch_tag::AbstractString; dev::Bool = false)
    base = joinpath(pkgdir(KernelForge), "data", "tuning")
    return joinpath(base, dev ? "$(arch_tag)_dev.json" : "$(arch_tag).json")
end

function _section_key(::Type{T}, precise::Bool) where T
    precise ? string(T) : "size_$(sizeof(T))"
end

# Convert a JSON3 tree to mutable Dict/Vector so we can splice in our new
# section without disturbing the rest of the file.
function _json_to_mutable(x)
    if x isa JSON3.Object
        return Dict{String,Any}(string(k) => _json_to_mutable(v) for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return Any[_json_to_mutable(v) for v in x]
    else
        return x
    end
end

function _load_existing(path::AbstractString)
    isfile(path) || return Dict{String,Any}()
    try
        data = JSON3.read(read(path, String))
        return _json_to_mutable(data)
    catch e
        @warn "failed to parse existing tuning JSON; starting fresh" path e
        return Dict{String,Any}()
    end
end

function _write_merged_json(path::AbstractString, arch_tag::AbstractString,
                            section_key::AbstractString,
                            cells::Vector{Dict{String,Any}})
    existing = _load_existing(path)
    payload  = isempty(existing) ? Dict{String,Any}() : existing

    payload["schema_version"] = 1
    payload["gpu_tag"]        = arch_tag
    payload["tuned_at"]       = string(today())

    matvec_sect = get!(() -> Dict{String,Any}(), payload, "matvec")
    matvec_sect = matvec_sect isa Dict ? matvec_sect : Dict{String,Any}()
    matvec_sect[section_key] = cells
    payload["matvec"] = matvec_sect

    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    # Also emit the matching `.jl` companion the runtime loader will read.
    # The JSON stays as the human-readable / portable canonical source; the
    # `.jl` lets KernelForge's loader pick up the tunings with no JSON parser
    # dependency at runtime.
    _write_jl_companion(replace(path, r"\.json$" => ".jl"), payload)
    return path
end

function _write_jl_companion(path::AbstractString, payload::AbstractDict)
    open(path, "w") do io
        println(io, "# Auto-generated by data/tuning/{matvec,vecmat}/autotune.jl — do not edit.")
        println(io, "# Read at runtime by src/tuning/loader.jl via `Base.include`.")
        println(io, "Dict{String,Any}(")
        println(io, "    \"schema_version\" => ", payload["schema_version"], ",")
        println(io, "    \"gpu_tag\"        => ", repr(String(payload["gpu_tag"])), ",")
        println(io, "    \"tuned_at\"       => ", repr(String(payload["tuned_at"])), ",")
        # Serialize BOTH sections if present. `_load_existing` seeds `payload`
        # with the whole file, so a matvec-only re-tune must still re-emit any
        # existing `vecmat` section — else it silently drops vecmat from the
        # `.jl` the loader reads. (Mirrors vecmat/autotune.jl's writer.)
        if haskey(payload, "matvec")
            println(io, "    \"matvec\" => Dict{String,Any}(")
            for (k, cells) in pairs(payload["matvec"])
                println(io, "        ", repr(String(k)), " => [")
                for c in cells
                    # Per-cell kernel family decides the field set. Legacy cells
                    # with no "kernel" key are generic.
                    if get(c, "kernel", "generic") == "rowthread"
                        @printf(io, "            (n=%d, p=%d, kernel=\"rowthread\", U=%d, ncb=%d, workgroup=%d),\n",
                                Int(c["n"]), Int(c["p"]),
                                Int(c["U"]), Int(c["ncb"]), Int(c["workgroup"]))
                    else
                        @printf(io, "            (n=%d, p=%d, kernel=\"generic\", Nitem=%d, chunksz=%d, Nblocks=%d, workgroup=%d),\n",
                                Int(c["n"]), Int(c["p"]),
                                Int(c["Nitem"]), Int(c["chunksz"]),
                                Int(c["Nblocks"]), Int(c["workgroup"]))
                    end
                end
                println(io, "        ],")
            end
            println(io, "    ),")
        end
        if haskey(payload, "vecmat")
            println(io, "    \"vecmat\" => Dict{String,Any}(")
            for (k, cells) in pairs(payload["vecmat"])
                println(io, "        ", repr(String(k)), " => [")
                for c in cells
                    @printf(io, "            (n=%d, p=%d, Nitem=%d, Nthreads=%d, workgroup=%d, blocks=%d),\n",
                            Int(c["n"]), Int(c["p"]),
                            Int(c["Nitem"]), Int(c["Nthreads"]),
                            Int(c["workgroup"]), Int(c["blocks"]))
                end
                println(io, "        ],")
            end
            println(io, "    ),")
        end
        println(io, ")")
    end
    return path
end

# -----------------------------------------------------------------------------
# Public driver.
# -----------------------------------------------------------------------------

"""
    autotune(::Type{T} = Float32; arch, trials, shapes, dev, precise, json_out, verbose) -> Vector{Dict}

Sweep the pow-2/pow-8 shape grid for matvec on this arch and write the winners
to a JSON under `data/tuning/<arch>.json` (or `<arch>_dev.json` if `dev=true`).

Keywords:
- `arch`: auto-detected from the active GPU. Override to pre-tune for another arch.
- `trials`: bench trials per candidate (default 5; forced to 2 when `dev=true`).
- `shapes`: override the shape grid. Defaults to `pow28_shapes(T)` (or
  `DEV_SHAPES` when `dev=true`).
- `dev`: minimal grid for ~30 s wiring smoke. Writes a separate
  `<arch>_dev.json` so it never pollutes the production file.
- `precise`: when `true`, write under `matvec["<T>"]` instead of
  `matvec["size_<sizeof(T)>"]`. Lookup will prefer the type-specific entry.
- `json_out`: override the output path.
"""
function autotune(::Type{T} = Float32;
                  arch              = nothing,
                  trials::Int       = 3,
                  shapes            = nothing,
                  dev::Bool         = false,
                  precise::Bool     = false,
                  correctness::Bool = false,
                  safety::Int       = 4,
                  resume::Bool      = true,
                  json_out          = nothing,
                  verbose::Bool     = true,
                  robust::Bool      = true,
                  top_k::Int        = 10,
                  coarse_k::Int     = 20,
                  precompile::Bool  = true,
                  max_cells_per_run::Int =
                      parse(Int, get(ENV, "AUTOTUNE_MAX_CELLS", "0")),
                  ) where T
    arch       = something(arch, KF.detect_arch(_gpu_zeros(T, 1)))
    arch_tag   = String(nameof(typeof(arch)))
    shapes     = something(shapes, dev ? DEV_SHAPES : pow28_shapes(T; safety))
    trials_eff = dev ? 2 : trials
    section    = _section_key(T, precise)
    json_out   = something(json_out, _default_json_out(arch_tag; dev))

    sort!(shapes, by = ((n, p),) -> (Int(round(log2(n * p))), Int(round(log2(n)))))

    verbose && @printf("autotune(MatVec, %s) on %s — %d cells, %d trials%s%s\n",
                       T, arch_tag, length(shapes), trials_eff,
                       dev ? " [DEV]" : "",
                       precise ? " [precise]" : "")
    verbose && @printf("  section: matvec.%s\n  output:  %s\n",
                       section, json_out)

    # Resume — reload cells already in the JSON for this section so a re-run
    # after a crash continues instead of restarting from scratch.
    cells = Dict{String,Any}[]
    # Parallel vector of top-K candidates per cell, aligned with `cells`.
    # Filled during pass 1; consumed in pass 2 by robust_refine_cells!.
    # Resumed cells have an empty top-K (we don't re-bench them just to
    # collect top-K). Pass 2 skips cells with <2 candidates.
    cell_tops = Vector{Vector{NamedTuple}}()
    done  = Set{Tuple{Int,Int}}()
    if resume && isfile(json_out)
        prev = _load_existing(json_out)
        if haskey(prev, "matvec") && prev["matvec"] isa AbstractDict &&
           haskey(prev["matvec"], section)
            for c in prev["matvec"][section]
                cell = Dict{String,Any}(c)
                push!(cells, cell)
                push!(cell_tops, NamedTuple[])     # resumed: no top-K to refine
                push!(done, (Int(cell["n"]), Int(cell["p"])))
            end
        end
    end
    verbose && !isempty(done) &&
        @printf("  resume: %d cells already tuned — skipping them\n", length(done))

    # Skip-list — cells the orchestrator flagged as stalling (hang-tolerant
    # mode). Skipped here; the runtime lookup falls back to the heuristic for
    # them. Format: one "n p" per line in /tmp/autotune_skip.txt.
    skip = Set{Tuple{Int,Int}}()
    let skipfile = "/tmp/autotune_skip.txt"
        if isfile(skipfile)
            for ln in eachline(skipfile)
                parts = split(strip(ln))
                length(parts) >= 2 || continue
                push!(skip, (parse(Int, parts[1]), parse(Int, parts[2])))
            end
        end
    end
    verbose && !isempty(skip) &&
        @printf("  skip-list: %d cells flagged stalling — skipping them\n", length(skip))

    _session_warmup()

    # Parallel pre-compilation of all unique kernel specs reused across the
    # remaining cells. Cuts ~5 min off the wall on multi-threaded launches;
    # no-op (sequential, no speedup) when nthreads()==1.
    precompile && _precompile_specs(shapes, T; skip_done=done, dev, verbose)

    # Per-candidate diagnostics CSV, alongside the JSON. Append across runs.
    diag_csv = _diag_csv_path(json_out)
    _diag_init(diag_csv)
    verbose && @printf("  diag:    %s\n", diag_csv)

    t_global = time_ns()
    fresh_tuned = 0
    for (i, (n, p)) in enumerate(shapes)
        ((n, p) in done || (n, p) in skip) && continue
        # Marker for the orchestrator's stall detector: cell index, n, p, time.
        try
            open("/tmp/autotune_cell.txt", "w") do io
                println(io, "$i $n $p ", time())
            end
        catch
        end
        t_cell = time_ns()
        result = tune_cell(n, p, T;
                           trials=trials_eff, dev, correctness,
                           top_k=top_k, coarse_k=coarse_k,
                           cell_idx=i, n_cells=length(shapes),
                           diag_csv=diag_csv)
        wall = (time_ns() - t_cell) / 1e9

        if isnothing(result)
            verbose && @printf("  [%3d/%d] (n=%-10d, p=%-10d) SKIP  [%.1fs]\n",
                               i, length(shapes), n, p, wall)
            continue
        end

        push!(cells, _cell_dict(n, p, result.params))
        push!(cell_tops, result.top)
        if result.params.kernel === :rowthread
            verbose && @printf("  [%3d/%d] (n=%-10d, p=%-10d) [RT] U=%-2d ncb=%-3d wg=%-4d  %8.1fµs  (%4d cands, %.1fs)\n",
                               i, length(shapes), n, p,
                               result.params.U, result.params.ncb,
                               result.params.workgroup,
                               result.median_us, result.n_candidates, wall)
        else
            verbose && @printf("  [%3d/%d] (n=%-10d, p=%-10d) Ni=%-2d csz=%-3d Nb=%-3d wg=%-4d  %8.1fµs  (%4d cands, %.1fs)\n",
                               i, length(shapes), n, p,
                               result.params.Nitem, result.params.chunksz,
                               result.params.Nblocks, result.params.workgroup,
                               result.median_us, result.n_candidates, wall)
        end

        # Checkpoint after every cell — a stall-kill / crash then loses zero
        # tuned cells (the JSON is small, the write is sub-millisecond).
        _write_merged_json(json_out, arch_tag, section, cells)

        # Multi-pass budget: exit cleanly after N fresh cells so the next
        # process starts with an empty GPUCompiler in-process cache (which
        # has no public mid-process release API). Counts FRESH cells, not
        # total — resumed runs start their budget from zero.
        fresh_tuned += 1
        if max_cells_per_run > 0 && fresh_tuned >= max_cells_per_run
            verbose && @printf("\nPARTIAL_RUN_EXIT: %d cells this process, %d total in JSON\n",
                               fresh_tuned, length(cells))
            verbose && @printf("JSON: %s\n", json_out)
            exit(0)
        end
    end

    # Persist pass-1 winners before pass-2 (so a pass-2 crash doesn't lose them).
    _write_merged_json(json_out, arch_tag, section, cells)

    # Pass 2 — robustness refinement.
    if robust && length(cells) >= 2
        verbose && @printf("\nPass 2 (robust): testing top-%d on (cell + 4 axis-neighbors)\n", top_k)
        n_changed = robust_refine_cells!(cells, cell_tops, shapes, T;
                                         trials=trials_eff, verbose)
        n_changed > 0 && _write_merged_json(json_out, arch_tag, section, cells)
    end

    wall_total = (time_ns() - t_global) / 1e9
    verbose && @printf("\nDone. %d cells tuned, total wall = %.1fs (%.1f min)\n",
                       length(cells), wall_total, wall_total / 60)
    verbose && @printf("JSON: %s\n", json_out)
    verbose && _diag_summary(diag_csv, wall_total)
    return cells
end

# Diagnostic summary from the per-candidate CSV. Reads the file (since rows
# from previous resumed processes are also there) and prints a breakdown.
function _diag_summary(csv_path::AbstractString, wall_total::Float64)
    isfile(csv_path) || return
    rows = Dict{Tuple{Int,Int},NamedTuple}()  # (n,p) -> per-cell aggregate
    total_compile_ms = 0.0
    total_bench_us  = 0.0
    total_wall_ms   = 0.0
    n_records       = 0
    n_failed        = 0
    slowest_compiles = Tuple{Float64,Int,Int,NamedTuple}[]  # (compile_ms, n, p, params)
    for (li, line) in enumerate(eachline(csv_path))
        li == 1 && continue                   # header
        parts = split(line, ',')
        length(parts) == 14 || continue
        n = parse(Int, parts[2]); p = parse(Int, parts[3])
        compile_ms = tryparse(Float64, parts[10]); compile_ms = compile_ms === nothing ? NaN : compile_ms
        bench_us   = tryparse(Float64, parts[11]); bench_us   = bench_us === nothing   ? NaN : bench_us
        wall_ms    = parse(Float64, parts[12])
        ok         = parts[13] == "1"
        if isfinite(compile_ms); total_compile_ms += compile_ms; end
        if ok && isfinite(bench_us); total_bench_us += bench_us; end
        total_wall_ms += wall_ms
        n_records += 1
        ok || (n_failed += 1)
        # Aggregate per cell.
        agg = get(rows, (n,p), (n_cands=0, compile_ms=0.0, bench_us=0.0, wall_ms=0.0))
        rows[(n,p)] = (n_cands=agg.n_cands + 1,
                       compile_ms=agg.compile_ms + (isfinite(compile_ms) ? compile_ms : 0.0),
                       bench_us =agg.bench_us  + (ok && isfinite(bench_us) ? bench_us : 0.0),
                       wall_ms  =agg.wall_ms   + wall_ms)
        # Track slowest individual compiles.
        if isfinite(compile_ms)
            params = (Nitem=parse(Int,parts[6]), chunksz=parse(Int,parts[7]),
                      Nblocks=parse(Int,parts[8]), workgroup=parse(Int,parts[9]))
            push!(slowest_compiles, (compile_ms, n, p, params))
        end
    end
    isempty(rows) && return

    @printf("\n──── diagnostics ────────────────────────────────────────────────\n")
    @printf("  candidates tried:      %d (%d failed)\n", n_records, n_failed)
    @printf("  Σ compile (1st call):  %.1f s   (%.1f min)\n", total_compile_ms/1e3, total_compile_ms/6e4)
    @printf("  Σ bench (kernel time): %.3f s\n", total_bench_us/1e6)
    @printf("  Σ wall (per-candidate): %.1f s\n", total_wall_ms/1e3)
    @printf("  wall_total (autotune): %.1f s\n", wall_total)
    @printf("  overhead (warmup+setup+sync+wall_total_outside_candidates): %.1f s\n",
            wall_total - total_wall_ms/1e3)

    # Top-10 cells by total wall.
    cells_sorted = sort(collect(rows), by = kv -> -kv[2].wall_ms)
    @printf("\n  top-10 cells by wall_ms:\n")
    @printf("  %-25s %-7s %-12s %-12s %-12s\n", "(n,p)", "cands", "compile_s", "bench_s", "wall_s")
    for (i, ((n,p), a)) in enumerate(cells_sorted)
        i > 10 && break
        @printf("  %-25s %-7d %-12.2f %-12.4f %-12.2f\n",
                "($n, $p)", a.n_cands, a.compile_ms/1e3, a.bench_us/1e6, a.wall_ms/1e3)
    end

    # Top-10 individual compile spikes.
    sort!(slowest_compiles, by = t -> -t[1])
    @printf("\n  top-10 slowest single compiles:\n")
    @printf("  %-12s %-25s %s\n", "compile_ms", "(n,p)", "params")
    for i in 1:min(10, length(slowest_compiles))
        c, n, p, prm = slowest_compiles[i]
        @printf("  %-12.1f %-25s Nitem=%d cs=%d Nb=%d wg=%d\n",
                c, "($n, $p)", prm.Nitem, prm.chunksz, prm.Nblocks, prm.workgroup)
    end
    @printf("──────────────────────────────────────────────────────────────────\n")
end

# -----------------------------------------------------------------------------
# Multi-size driver — one JSON, multiple size sections.
# -----------------------------------------------------------------------------

_representative_dtype(sz::Int) =
    sz == 1 ? Int8 :
    sz == 2 ? Float16 :
    sz == 4 ? Float32 :
    sz == 8 ? Float64 :
    error("no representative dtype for size $sz")

"""
    autotune_all(; arch, dev, sizes=(4, 8))

Run `autotune` for each size in `sizes` using a representative dtype. All
results land in the same `data/tuning/<arch>.json` via the merge writer.
"""
function autotune_all(; arch = nothing, dev::Bool = false, sizes = (4, 8))
    for sz in sizes
        T = _representative_dtype(sz)
        autotune(T; arch, dev)
    end
end

# -----------------------------------------------------------------------------
# CLI entry — `julia autotune.jl [--dev] [--precise] [--all]`.
# -----------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    dev       = "--dev"     in ARGS
    precise   = "--precise" in ARGS
    all_sizes = "--all"     in ARGS
    if all_sizes
        autotune_all(; dev)
    else
        autotune(Float32; dev, precise)
    end
end
