# =============================================================================
# data/tuning/vecmat/autotune.jl
# =============================================================================
# Vecmat autotuner for KernelForge.jl. Sibling of data/tuning/matvec/autotune.jl;
# shares the same backend abstraction, parallel-precompile, hierarchical
# coarse→fine search and pass-2 robustness refinement. Differences vs matvec:
#
#   - Parameter surface: (Nitem, Nthreads, workgroup, blocks). Nblocks_runtime
#     is derived inside get_allocation as `cld(Nthreads, workgroup)`.
#   - Sole validity assert: `Nthreads * Nitem <= n` (vecmat.jl:202).
#   - vecmat_kernel.jl's `@localmem H warpsz` is INDEPENDENT of Nitem/Nthreads
#     (unlike matvec). The LLVM-RSS blowup that bit matvec doesn't apply here,
#     but per-thread register pressure still scales with Nitem — keep the
#     `Nitem <= min(cld(32,sz), 16)` cap to avoid misaligned-address traps for
#     small-int types.
#
# JSON section key: `vecmat.size_<sizeof(T)>` (default) or `vecmat.<DType>`
# (precise override).
#
# Usage mirrors the matvec script:
#   julia ... -e 'include("data/tuning/vecmat/autotune.jl"); autotune(Float32)'
#   julia ... -e 'include("data/tuning/vecmat/autotune.jl"); autotune_all()'
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
# Backend abstraction — same dual CUDA/AMD pattern as matvec autotune.
# -----------------------------------------------------------------------------

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

if _BACKEND_KIND === :cuda
    @eval begin
        _gpu_backend()                         = CUDA.CUDABackend()
        _total_memory()                        = Int(CUDA.totalmem(CUDA.device()))
        _gpu_rand_prim(::Type{T}, dims...) where T  = CUDA.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T  = CUDA.zeros(T, dims...)

        # CuEvent pairs — pure GPU-side per-trial time, no CUPTI session.
        # `reducer` picks the per-call statistic: `median` for coarse/pass-2
        # (default), `minimum` for the fine phase where noise can only add
        # time so the best sample is the most representative.
        function _bench_us(f; trials::Int = 5, reducer::Function = median)
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
            return reducer(samples)
        end
    end
else # :roc
    @eval begin
        _gpu_backend()                         = AMDGPU.ROCBackend()
        _total_memory()                        = Int(AMDGPU.HIP.properties(AMDGPU.device()).totalGlobalMem)
        _gpu_rand_prim(::Type{T}, dims...) where T  = AMDGPU.rand(T, dims...)
        _gpu_zeros(::Type{T}, dims...) where T  = AMDGPU.zeros(T, dims...)

        function _bench_us(f; trials::Int = 5, reducer::Function = median)
            # `inner` is ADAPTIVE (mirrors matvec/autotune.jl): amortizing host
            # dispatch (~µs) only matters for µs-scale kernels. Giant-p cells run
            # 100-500 ms/launch — a fixed inner=20 made each candidate cost
            # 2-10 s. A probe launch sizes inner to ~a few ms: inner=1 for
            # ms-scale kernels (already pure kernel time, no accuracy loss).
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

# Large-array-safe random fill — backend-agnostic wrapper over `_gpu_rand_prim`.
# `AMDGPU.rand(T, n, p)` truncates the RNG kernel's element count to Int32 →
# `InexactError: trunc(Int32, …)` at ≥ 2^31 elements (the autotune's giant cells,
# n*p up to ~2^31, hit this on MI300A). CUDA indexes Int64, no limit. Keep ONE
# arch-general path: fast single-call path for total < 2^30; above that allocate
# uninitialized (`KA.allocate`, no kernel, any size) and fill through a contiguous
# `reshape(A,:)` in ≤2^30-element chunks (loop covers [1,total] → no uninit reads).
# CUDA behavior is byte-identical (only ≥2^30 cells chunk). Mirrors the matvec fix.
const _RAND_CHUNK = 1 << 30
function _gpu_rand(::Type{T}, dims::Integer...) where T
    total = prod(map(Int, dims))
    total < _RAND_CHUNK && return _gpu_rand_prim(T, dims...)
    A = KA.allocate(_gpu_backend(), T, dims...)
    flat = reshape(A, :)
    off = 0
    while off < total
        m = min(_RAND_CHUNK, total - off)
        copyto!(view(flat, off+1:off+m), _gpu_rand_prim(T, m))
        off += m
    end
    return A
end

# -----------------------------------------------------------------------------
# Shape grid — same memory-capped pow-2/pow-8 schedule as matvec.
# -----------------------------------------------------------------------------

const LOG2_LEVELS = (0, 1, 2, 3, 4, 5, 6,
                     9, 12, 15, 18, 21, 24, 27, 30)
const MIN_DIM     = 100

function max_log2_np(::Type{T}; safety::Int = 4) where T
    total_bytes = _total_memory()
    budget_bytes = total_bytes ÷ safety
    max_np = budget_bytes ÷ sizeof(T)
    cap = floor(Int, log2(max(1, max_np)))
    return min(cap, maximum(LOG2_LEVELS))
end

# Tall/small-p corner the pow-2 grid MISSES: LOG2_LEVELS jumps n 2^15→2^18 and
# p 2^9→2^12, so a tall vecmat (n≫p) at p≈1e3 borrows a distant cell whose
# Nthreads/workgroup are wrong for its reduction length. Measured A100 (end-to-end):
# 100000×1000 62→74% peak (1.29→1.09× cuBLAS) once it maps to the tuned (131072,1024)
# cell. p=1024 sits nearer a p≈1000 query than 512, so these cells get picked.
# n capped at 131072 ON PURPOSE: adding n=262144 cells stole the nearest-cell mapping
# of EXTREME-tall queries (n≈1e6) — their reduction is far longer than 262144, so the
# 262144-tuned params regressed them (1e6×1000 77→73%). Keeping n≤131072 leaves those
# queries on their old (262144,512) cell. Residual ~1.1× cuBLAS is the kernel ceiling.
const TALL_CORNER = [(n, p) for n in (32768, 131072) for p in (1024, 2048)]

function pow28_shapes(::Type{T} = Float32; safety::Int = 4) where T
    cap = max_log2_np(T; safety)
    out = Tuple{Int,Int}[]
    for kn in LOG2_LEVELS, kp in LOG2_LEVELS
        kn + kp <= cap || continue
        n = 1 << kn; p = 1 << kp
        (n >= MIN_DIM || p >= MIN_DIM) || continue
        push!(out, (n, p))
    end
    for (n, p) in TALL_CORNER
        n * p <= (1 << cap) && push!(out, (n, p))
    end
    return unique!(out)
end

const DEV_SHAPES = [(10_000, 10_000), (10, 1_000_000), (1_000_000, 10)]

# -----------------------------------------------------------------------------
# Param grid — vecmat-specific tuple (Nitem, Nthreads, workgroup, blocks).
# -----------------------------------------------------------------------------

# Cap derivation mirrors matvec.jl:nitems_for — `cld(32, sizeof(T))` keeps LLVM
# unroll cost bounded, hard cap 16 protects sizeof(T)==1 (Int8) from
# ERROR_MISALIGNED_ADDRESS on byte-strided `vload`.
function nitems_for(::Type{T}) where T
    cap = min(cld(32, sizeof(T)), 16)
    return Tuple(1 << k for k in 0:floor(Int, log2(cap)))
end

# Extended upward for MI300X: large reductions (tall-skinny, p small, n huge)
# want far more parallelism than the original NVIDIA-era 16384 ceiling — the
# default heuristic uses Nthreads=131072 there, and 17/109 winners pressed the
# old ceiling. The validity filters (`Nthreads*Nitem<=n`, per-thread cap
# `cld(n,Nthreads)<=2^18`) self-restrict the high values to large-n cells, so
# small cells are unaffected. (262144 only activates if the Nblocks_runtime<=256
# cap below is also raised; kept here so the search reaches it once that lifts.)
const NTHREADS_GRID  = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
const WORKGROUP_GRID = (128, 256, 512)
# 256 added: multi-block reductions win there on MI300X (default uses blocks=256).
const BLOCKS_GRID    = (1, 2, 4, 8, 16, 32, 64, 128, 256)
const WARPSZ         = 32

# Dev grid — small Cartesian for the ~30 s smoke.
const DEV_NTHREADS   = (128, 1024, 8192)
const DEV_WORKGROUP  = (256, 512)
const DEV_BLOCKS     = (1, 16, 128)
dev_nitems_for(::Type{T}) where T = sizeof(T) <= 4 ? (1, 4, 8) : (1, 2, 4)

function param_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T
    nitems     = dev ? dev_nitems_for(T) : nitems_for(T)
    nthreadss  = dev ? DEV_NTHREADS      : NTHREADS_GRID
    workgroups = dev ? DEV_WORKGROUP     : WORKGROUP_GRID
    blockss    = dev ? DEV_BLOCKS        : BLOCKS_GRID

    out = NamedTuple[]
    for Nitem in nitems, Nthreads in nthreadss, workgroup in workgroups, blocks in blockss
        # Hard validity (vecmat.jl:202 `@assert Nthreads * Nitem <= n`).
        Nthreads * Nitem <= n || continue
        # workgroup <= n*p (mirrors matvec).
        workgroup <= n * p || continue
        # Nthreads<workgroup is meaningful: the workgroup then handles
        # workgroup/Nthreads columns in parallel (single-block path).
        Nblocks_runtime = cld(Nthreads, workgroup)
        # Cap on cross-block partial slots. RTX1000 saturates at ~28 active
        # blocks; 256 leaves ~9× oversubscription and bounds the partial
        # buffer init + flag-wait overhead at small p.
        Nblocks_runtime <= 256 || continue
        # Cap per-thread serial work along the column. The reduction over n
        # is split into Nthreads chunks of length ~n/Nthreads. Letting one
        # thread do >2^18 sequential ops on the largest cells (n~1e8) made
        # individual coarse-bench trials take 8+ seconds each (cell 90 dev
        # autotune hang). The cap only bites huge-n cells; small-n is
        # unaffected.
        cld(n, max(Nthreads, 1)) <= 1 << 18 || continue
        # When Nthreads<=workgroup we hit the single-block path; `blocks` is
        # unused. Collapse to a single canonical `blocks=1` representative to
        # avoid duplicate specs (same kernel compile, identical bench).
        if Nblocks_runtime == 1 && blocks != 1
            continue
        end
        push!(out, (; kernel = :generic, Nitem, Nthreads, workgroup, blocks))
    end
    return out
end

# -----------------------------------------------------------------------------
# MLP candidate family (vecmat_mlp_kernel!): warp-per-column, U independent loads
# in flight. Two knobs: U (unroll/MLP depth) and workgroup. Wins the medium-large-n
# × large-p band where the generic kernel's serial reduction is MLP-starved.
# -----------------------------------------------------------------------------
const MLP_U_GRID   = (4, 8, 16)
const MLP_WG_GRID  = (128, 256, 512)

# MLP requires n ≥ warpsz (every lane seeds an element); wins for large p (fill the
# GPU) and n not so tall that a single warp's serial reduction dominates.
_mlp_applicable(n::Int, p::Int) = 256 <= n <= 131072 && p >= 4096

function mlp_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T
    _mlp_applicable(n, p) || return NamedTuple[]
    Us  = dev ? (8,)   : MLP_U_GRID
    wgs = dev ? (256,) : MLP_WG_GRID
    out = NamedTuple[]
    for U in Us, wg in wgs
        push!(out, (; kernel = :mlp, U, workgroup = wg))
    end
    return out
end

all_candidates(::Type{T}, n::Int, p::Int; dev::Bool = false) where T =
    vcat(param_candidates(T, n, p; dev), mlp_candidates(T, n, p; dev))

# Zero-arg closure running one candidate (either family), result left in `dst`.
function _run_closure(params::NamedTuple, x, A, dst, ::Type{H}, n, p, arch) where H
    if params.kernel === :mlp
        return () -> (KF._vecmat_mlp_impl!(*, +, identity, dst, x, A, H, n, p, arch;
                        U = params.U, workgroup = params.workgroup); dst)
    else
        return () -> (KF.vecmat!(dst, x, A;
                        params.Nitem, params.Nthreads,
                        params.workgroup, params.blocks); dst)
    end
end

# Serialize a cell's winning params (either family) to a mutable Dict.
function _cell_dict(n::Int, p::Int, params::NamedTuple)
    if params.kernel === :mlp
        return Dict{String,Any}("n" => n, "p" => p, "kernel" => "mlp",
            "U" => params.U, "workgroup" => params.workgroup)
    else
        return Dict{String,Any}("n" => n, "p" => p, "kernel" => "generic",
            "Nitem" => params.Nitem, "Nthreads" => params.Nthreads,
            "workgroup" => params.workgroup, "blocks" => params.blocks)
    end
end

# -----------------------------------------------------------------------------
# Bench protocol.
# -----------------------------------------------------------------------------

function _warmup_for(f, ms::Int)
    t0 = time_ns()
    while (time_ns() - t0) < ms * 1_000_000
        f()
        KA.synchronize(_gpu_backend())
    end
end

function _session_warmup()
    A = _gpu_rand(Float32, 10^4, 10^4)
    x = _gpu_rand(Float32, 10^4)
    _warmup_for(() -> KF.vecmat(x, A), 1000)
end

# -----------------------------------------------------------------------------
# Parallel pre-compilation. Same Channel/Threads.@spawn structure as matvec.
# -----------------------------------------------------------------------------

function _precompile_one(params::NamedTuple, ::Type{T}) where T
    # Shape big enough to satisfy validity for this spec.
    if params.kernel === :mlp
        n = max(params.workgroup, 256)
        p = 4096
    else
        min_n = params.Nthreads * params.Nitem
        min_p = max(cld(params.Nthreads, params.workgroup), 1)
        n = nextpow(2, max(min_n, 1))
        p = nextpow(2, max(min_p, 1))
    end
    A = _gpu_rand(T, n, p)
    x = _gpu_rand(T, n)
    backend = get_backend(A)
    H = Base.promote_op(*, T, T)
    arch = KF.detect_arch(A)
    _run_closure(params, x, A, dst_for(T, backend, p), H, n, p, arch)()
    KA.synchronize(backend)
    return
end

dst_for(::Type{T}, backend, p) where T = KA.allocate(backend, Base.promote_op(*, T, T), p)

function _precompile_specs(shapes, ::Type{T};
                           skip_done::Set = Set{Tuple{Int,Int}}(),
                           dev::Bool = false,
                           verbose::Bool = true) where T
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
# Per-cell tune — same hierarchical coarse→fine structure as matvec.
# -----------------------------------------------------------------------------

# Diag CSV is namespaced for vecmat — the matvec autotune writes
# `<arch>_diag.csv` with a different column layout, so the two must not share.
_diag_csv_path(json_out::AbstractString) =
    replace(json_out, r"\.json$" => "_vecmat_diag.csv")

function _diag_init(csv_path::AbstractString)
    isfile(csv_path) && return
    mkpath(dirname(csv_path))
    open(csv_path, "w") do io
        println(io, "cell_idx,n,p,cand_idx,n_cands,Nitem,Nthreads,workgroup,blocks,compile_ms,bench_us,wall_ms,ok,is_first")
    end
end

# Map either family's params into the fixed (Nitem,Nthreads,workgroup,blocks) diag
# columns. mlp has no such knobs — encode (U, 0, workgroup, 0); diagnostic only.
_diag_params(p) = p.kernel === :mlp ?
    (Nitem = p.U, Nthreads = 0, workgroup = p.workgroup, blocks = 0) :
    (Nitem = p.Nitem, Nthreads = p.Nthreads, workgroup = p.workgroup, blocks = p.blocks)

function _diag_row(io, cell_idx, n, p, cand_idx, n_cands, params, compile_ms, bench_us, wall_ms, ok, is_first)
    dp = _diag_params(params)
    println(io, cell_idx, ",", n, ",", p, ",", cand_idx, ",", n_cands, ",",
            dp.Nitem, ",", dp.Nthreads, ",", dp.workgroup, ",", dp.blocks, ",",
            compile_ms, ",", bench_us, ",", wall_ms, ",", ok ? 1 : 0, ",", is_first ? 1 : 0)
end

function tune_cell(n::Int, p::Int, ::Type{T};
                   trials::Int = 10, dev::Bool = false,
                   correctness::Bool = false,
                   top_k::Int = 10,
                   coarse_k::Int = 20,
                   arch = nothing,
                   cell_idx::Int = 0, n_cells::Int = 0,
                   diag_csv::Union{Nothing,AbstractString} = nothing) where T
    cands = all_candidates(T, n, p; dev)
    isempty(cands) && return nothing

    # Never-worse-than-default guard. The autotuner ranks only among its swept
    # candidate grid; on tall-skinny (small-p) cells the runtime default
    # heuristic out-parallelizes that grid, so without this guard the "tuned"
    # winner regressed vs the default (e.g. vecmat (1e8,1): 839µs / 476 GB/s
    # tuned vs ~212µs / ~1900 GB/s default). Force the default params into the
    # candidate set (last index) and protect them into the fine pass below, so
    # the winner is benched against the default with full trials and can never
    # be slower than it. Needs `arch` (passed from autotune()); skipped if absent.
    default_ci = 0
    if arch !== nothing
        wg0 = KF.default_workgroup(arch, KF.VecMat, n, p, T)
        bl0 = KF.default_blocks(arch, KF.VecMat, n, p, T)
        ni0 = KF.default_nitem(arch, KF.VecMat, n, p, T)
        nt0 = KF.default_nthreads(arch, KF.VecMat, n, p, T, ni0, wg0, bl0)
        push!(cands, (; kernel = :generic, Nitem = ni0, Nthreads = nt0, workgroup = wg0, blocks = bl0))
        default_ci = length(cands)
    end

    A = try
        _gpu_rand(T, n, p)
    catch e
        @warn "alloc failed; skipping cell" n p T e
        return nothing
    end
    x = _gpu_rand(T, n)
    backend = get_backend(A)
    H = Base.promote_op(*, T, T)
    dst = KA.allocate(backend, H, p)
    ref_gpu = ref_max = nothing
    if correctness
        A_f64 = Float64.(A)
        x_f64 = Float64.(x)
        ref_gpu = vec(x_f64' * A_f64)
        ref_max = maximum(abs.(ref_gpu))
        A_f64 = x_f64 = nothing
    end

    n_cands = length(cands)

    coarse = Vector{NamedTuple}(undef, n_cands)
    for (ci, params) in enumerate(cands)
        t_wall0 = time_ns()
        compile_ms, coarse_us, ok_c = NaN, NaN, false
        try
            f = _run_closure(params, x, A, dst, H, n, p, arch)
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

    # Family-stratified survivors: top-k of EACH kernel family reaches the fine
    # pass, so a family with many mediocre candidates can't crowd the other's
    # winner out of a global top-k (mirrors the matvec 2-family autotune).
    valid = sort!([c for c in coarse if c.ok], by = c -> c.coarse_us)
    gen_valid = [c for c in valid if c.params.kernel === :generic]
    mlp_valid = [c for c in valid if c.params.kernel === :mlp]
    survivors = vcat(first(gen_valid, min(coarse_k, length(gen_valid))),
                     first(mlp_valid, min(coarse_k, length(mlp_valid))))
    # Protect the default candidate into the fine pass even if its single
    # coarse trial fluked slow (see never-worse-than-default guard above).
    if default_ci > 0 && coarse[default_ci].ok && !any(c -> c.ci == default_ci, survivors)
        survivors = vcat(survivors, [coarse[default_ci]])
    end

    fine_us     = Dict{Int,Float64}()
    fine_relerr = Dict{Int,Float64}()
    fine_wall   = Dict{Int,Float64}()
    for (k, c) in enumerate(survivors)
        t_wall0 = time_ns()
        params = c.params
        f_us, f_re, f_ok = NaN, NaN, false
        try
            f = _run_closure(params, x, A, dst, H, n, p, arch)
            _warmup_for(f, 5)
            # Fine phase trials default to 10 (autotune driver passes
            # `fine_trials`). Median is the right reducer here: we tried
            # min-of-5 — it cut the F64 (64, 262144) bad-winner from 1126µs
            # to 848µs but the min-of-K ranking was unstable. Median-of-10
            # trades a bit of wall (10 events + 2 ms warmup amortised) for
            # stabler rankings on cells where trial-to-trial drift bites.
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
# Pass 2 — robustness refinement on cell + 4 axis-neighbors.
# -----------------------------------------------------------------------------

function _axis_neighbors(n::Int, p::Int, shapes::AbstractVector{<:Tuple})
    ns = sort!(unique([nn for (nn, pp) in shapes if pp == p]))
    ps = sort!(unique([pp for (nn, pp) in shapes if nn == n]))
    neighbors = Tuple{Int,Int}[]
    ni = searchsortedfirst(ns, n)
    ni > 1          && push!(neighbors, (ns[ni-1], p))
    ni < length(ns) && push!(neighbors, (ns[ni+1], p))
    pi = searchsortedfirst(ps, p)
    pi > 1          && push!(neighbors, (n, ps[pi-1]))
    pi < length(ps) && push!(neighbors, (n, ps[pi+1]))
    return neighbors
end

function _bench_shape(params::NamedTuple, n::Int, p::Int, ::Type{T};
                      trials::Int = 5) where T
    # ALL of it (alloc + rand + launch + bench) inside the try: pass 2 re-benches
    # axis-neighbor shapes larger than the tuned cell, and any per-shape failure
    # — OOM, a backend rand/alloc limit, a kernel assert — must degrade to "skip"
    # (NaN), never crash the autotune. (pass 1's `tune_cell` guards its alloc the
    # same way; this path used to alloc OUTSIDE the try.) Mirrors the matvec fix.
    try
        A   = _gpu_rand(T, n, p)
        x   = _gpu_rand(T, n)
        backend = get_backend(A)
        H   = Base.promote_op(*, T, T)
        dst = KA.allocate(backend, H, p)
        arch = KF.detect_arch(A)
        f = _run_closure(params, x, A, dst, H, n, p, arch)
        f(); KA.synchronize(backend)
        _warmup_for(f, 5)
        return _bench_us(f; trials)
    catch
        return NaN
    end
end

function robust_refine_cells!(cells::Vector{Dict{String,Any}},
                              cell_tops::AbstractVector,
                              shapes::AbstractVector{<:Tuple},
                              ::Type{T};
                              trials::Int = 5, verbose::Bool = true) where T
    length(cells) == length(cell_tops) || error("cells / cell_tops length mismatch")
    n_changed = 0
    t0 = time_ns()
    for (i, cell) in enumerate(cells)
        top = cell_tops[i]
        length(top) <= 1 && continue
        n, p = Int(cell["n"]), Int(cell["p"])
        neighbors = _axis_neighbors(n, p, shapes)
        isempty(neighbors) && continue

        scores = Vector{Float64}(undef, length(top))
        for (k, c) in enumerate(top)
            cand_us = Float64[c.median_us]
            for (nn, pp) in neighbors
                us = _bench_shape(c.params, nn, pp, T; trials)
                isfinite(us) && push!(cand_us, us)
            end
            scores[k] = sum(cand_us) / length(cand_us)
        end

        kbest = argmin(scores)
        if kbest != 1
            winner = top[kbest]
            # Robust winner may switch families — rebuild the cell wholesale.
            empty!(cell); merge!(cell, _cell_dict(n, p, winner.params))
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
# JSON I/O — merges with existing data/tuning/<Arch>.json. Section root key is
# "vecmat" (matvec autotune writes "matvec"), so the two coexist cleanly.
# -----------------------------------------------------------------------------

function _default_json_out(arch_tag::AbstractString; dev::Bool = false)
    base = joinpath(pkgdir(KernelForge), "data", "tuning")
    return joinpath(base, dev ? "$(arch_tag)_dev.json" : "$(arch_tag).json")
end

function _section_key(::Type{T}, precise::Bool) where T
    precise ? string(T) : "size_$(sizeof(T))"
end

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

    vecmat_sect = get!(() -> Dict{String,Any}(), payload, "vecmat")
    vecmat_sect = vecmat_sect isa Dict ? vecmat_sect : Dict{String,Any}()
    vecmat_sect[section_key] = cells
    payload["vecmat"] = vecmat_sect

    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
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
        # matvec section — kernel-tag aware (generic | rowthread). Preserve any
        # existing matvec tuning verbatim so a vecmat re-tune never drops/corrupts
        # the matvec rowthread cells (mirrors matvec/autotune.jl's writer).
        if haskey(payload, "matvec")
            println(io, "    \"matvec\" => Dict{String,Any}(")
            for (k, cells) in pairs(payload["matvec"])
                println(io, "        ", repr(String(k)), " => [")
                for c in cells
                    if get(c, "kernel", "generic") == "rowthread"
                        @printf(io, "            (n=%d, p=%d, kernel=\"rowthread\", U=%d, ncb=%d, workgroup=%d),\n",
                                Int(c["n"]), Int(c["p"]), Int(c["U"]), Int(c["ncb"]), Int(c["workgroup"]))
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
        # vecmat section — kernel-tag aware (generic | mlp).
        if haskey(payload, "vecmat")
            println(io, "    \"vecmat\" => Dict{String,Any}(")
            for (k, cells) in pairs(payload["vecmat"])
                println(io, "        ", repr(String(k)), " => [")
                for c in cells
                    if get(c, "kernel", "generic") == "mlp"
                        @printf(io, "            (n=%d, p=%d, kernel=\"mlp\", U=%d, workgroup=%d),\n",
                                Int(c["n"]), Int(c["p"]), Int(c["U"]), Int(c["workgroup"]))
                    else
                        @printf(io, "            (n=%d, p=%d, kernel=\"generic\", Nitem=%d, Nthreads=%d, workgroup=%d, blocks=%d),\n",
                                Int(c["n"]), Int(c["p"]),
                                Int(c["Nitem"]), Int(c["Nthreads"]),
                                Int(c["workgroup"]), Int(c["blocks"]))
                    end
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

function autotune(::Type{T} = Float32;
                  arch              = nothing,
                  fine_trials::Int  = 10,
                  robust_trials::Int = 5,
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
    fine_eff   = dev ? 2 : fine_trials
    robust_eff = dev ? 2 : robust_trials
    section    = _section_key(T, precise)
    json_out   = something(json_out, _default_json_out(arch_tag; dev))

    sort!(shapes, by = ((n, p),) -> (Int(round(log2(n * p))), Int(round(log2(n)))))

    verbose && @printf("autotune(VecMat, %s) on %s — %d cells, fine_trials=%d robust_trials=%d%s%s\n",
                       T, arch_tag, length(shapes), fine_eff, robust_eff,
                       dev ? " [DEV]" : "",
                       precise ? " [precise]" : "")
    verbose && @printf("  section: vecmat.%s\n  output:  %s\n",
                       section, json_out)

    cells = Dict{String,Any}[]
    cell_tops = Vector{Vector{NamedTuple}}()
    done  = Set{Tuple{Int,Int}}()
    if resume && isfile(json_out)
        prev = _load_existing(json_out)
        if haskey(prev, "vecmat") && prev["vecmat"] isa AbstractDict &&
           haskey(prev["vecmat"], section)
            for c in prev["vecmat"][section]
                cell = Dict{String,Any}(c)
                push!(cells, cell)
                push!(cell_tops, NamedTuple[])
                push!(done, (Int(cell["n"]), Int(cell["p"])))
            end
        end
    end
    verbose && !isempty(done) &&
        @printf("  resume: %d cells already tuned — skipping them\n", length(done))

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

    precompile && _precompile_specs(shapes, T; skip_done=done, dev, verbose)

    diag_csv = _diag_csv_path(json_out)
    _diag_init(diag_csv)
    verbose && @printf("  diag:    %s\n", diag_csv)

    t_global = time_ns()
    fresh_tuned = 0
    for (i, (n, p)) in enumerate(shapes)
        ((n, p) in done || (n, p) in skip) && continue
        try
            open("/tmp/autotune_cell.txt", "w") do io
                println(io, "$i $n $p ", time())
            end
        catch
        end
        t_cell = time_ns()
        result = tune_cell(n, p, T;
                           trials=fine_eff, dev, correctness,
                           top_k=top_k, coarse_k=coarse_k, arch,
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
        if result.params.kernel === :mlp
            verbose && @printf("  [%3d/%d] (n=%-10d, p=%-10d) [MLP] U=%-2d wg=%-4d  %8.1fµs  (%4d cands, %.1fs)\n",
                               i, length(shapes), n, p,
                               result.params.U, result.params.workgroup,
                               result.median_us, result.n_candidates, wall)
        else
            verbose && @printf("  [%3d/%d] (n=%-10d, p=%-10d) Ni=%-2d Nt=%-5d wg=%-4d bl=%-4d  %8.1fµs  (%4d cands, %.1fs)\n",
                               i, length(shapes), n, p,
                               result.params.Nitem, result.params.Nthreads,
                               result.params.workgroup, result.params.blocks,
                               result.median_us, result.n_candidates, wall)
        end

        _write_merged_json(json_out, arch_tag, section, cells)

        fresh_tuned += 1
        if max_cells_per_run > 0 && fresh_tuned >= max_cells_per_run
            verbose && @printf("\nPARTIAL_RUN_EXIT: %d cells this process, %d total in JSON\n",
                               fresh_tuned, length(cells))
            verbose && @printf("JSON: %s\n", json_out)
            exit(0)
        end
    end

    _write_merged_json(json_out, arch_tag, section, cells)

    if robust && length(cells) >= 2
        verbose && @printf("\nPass 2 (robust): testing top-%d on (cell + 4 axis-neighbors)\n", top_k)
        n_changed = robust_refine_cells!(cells, cell_tops, shapes, T;
                                         trials=robust_eff, verbose)
        n_changed > 0 && _write_merged_json(json_out, arch_tag, section, cells)
    end

    wall_total = (time_ns() - t_global) / 1e9
    verbose && @printf("\nDone. %d cells tuned, total wall = %.1fs (%.1f min)\n",
                       length(cells), wall_total, wall_total / 60)
    verbose && @printf("JSON: %s\n", json_out)
    verbose && _diag_summary(diag_csv, wall_total)
    return cells
end

function _diag_summary(csv_path::AbstractString, wall_total::Float64)
    isfile(csv_path) || return
    rows = Dict{Tuple{Int,Int},NamedTuple}()
    total_compile_ms = 0.0
    total_bench_us  = 0.0
    total_wall_ms   = 0.0
    n_records       = 0
    n_failed        = 0
    slowest_compiles = Tuple{Float64,Int,Int,NamedTuple}[]
    for (li, line) in enumerate(eachline(csv_path))
        li == 1 && continue
        parts = split(line, ',')
        length(parts) == 14 || continue
        n = parse(Int, parts[2]); p = parse(Int, parts[3])
        compile_ms = tryparse(Float64, parts[10]); compile_ms = compile_ms === nothing ? NaN : compile_ms
        bench_us   = tryparse(Float64, parts[11]); bench_us   = bench_us   === nothing ? NaN : bench_us
        wall_ms    = parse(Float64, parts[12])
        ok         = parts[13] == "1"
        if isfinite(compile_ms); total_compile_ms += compile_ms; end
        if ok && isfinite(bench_us); total_bench_us += bench_us; end
        total_wall_ms += wall_ms
        n_records += 1
        ok || (n_failed += 1)
        agg = get(rows, (n,p), (n_cands=0, compile_ms=0.0, bench_us=0.0, wall_ms=0.0))
        rows[(n,p)] = (n_cands=agg.n_cands + 1,
                       compile_ms=agg.compile_ms + (isfinite(compile_ms) ? compile_ms : 0.0),
                       bench_us =agg.bench_us  + (ok && isfinite(bench_us) ? bench_us : 0.0),
                       wall_ms  =agg.wall_ms   + wall_ms)
        if isfinite(compile_ms)
            params = (Nitem=parse(Int,parts[6]), Nthreads=parse(Int,parts[7]),
                      workgroup=parse(Int,parts[8]), blocks=parse(Int,parts[9]))
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

    cells_sorted = sort(collect(rows), by = kv -> -kv[2].wall_ms)
    @printf("\n  top-10 cells by wall_ms:\n")
    @printf("  %-25s %-7s %-12s %-12s %-12s\n", "(n,p)", "cands", "compile_s", "bench_s", "wall_s")
    for (i, ((n,p), a)) in enumerate(cells_sorted)
        i > 10 && break
        @printf("  %-25s %-7d %-12.2f %-12.4f %-12.2f\n",
                "($n, $p)", a.n_cands, a.compile_ms/1e3, a.bench_us/1e6, a.wall_ms/1e3)
    end

    sort!(slowest_compiles, by = t -> -t[1])
    @printf("\n  top-10 slowest single compiles:\n")
    @printf("  %-12s %-25s %s\n", "compile_ms", "(n,p)", "params")
    for i in 1:min(10, length(slowest_compiles))
        c, n, p, prm = slowest_compiles[i]
        @printf("  %-12.1f %-25s Nitem=%d Nthreads=%d workgroup=%d blocks=%d\n",
                c, "($n, $p)", prm.Nitem, prm.Nthreads, prm.workgroup, prm.blocks)
    end
    @printf("──────────────────────────────────────────────────────────────────\n")
end

_representative_dtype(sz::Int) =
    sz == 1 ? Int8 :
    sz == 2 ? Float16 :
    sz == 4 ? Float32 :
    sz == 8 ? Float64 :
    error("no representative dtype for size $sz")

function autotune_all(; arch = nothing, dev::Bool = false, sizes = (4, 8))
    for sz in sizes
        T = _representative_dtype(sz)
        autotune(T; arch, dev)
    end
end

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
