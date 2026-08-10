# =============================================================================
# data/tuning/gemm/autotune.jl
# =============================================================================
# GEMM (K2 / tensor-core MMA path) autotuner. For each SHAPE REGIME × MMA-capable
# dtype it searches the warp/register tiling knobs (WM, WN, NWM, NWN, WK, PC, DB;
# PA left at default — "noise" per the gemm docstring) and records the fastest
# feasible config. Writes data/tuning/<Arch>_gemm.json.
#
# Shape regimes (fixed FLOPs M·N·K, base L, aspect s — mirrors the benchmark):
#   square, outer, inner, A_thin, B_thin, tall.  Tuned at ONE representative L; the
#   loader (future) maps any (M,N,K) to its regime.
#
# Only tunes types whose `family=:mma` path is actually available on the device
# (probed at runtime — e.g. F16 on Ada; F64 tensor cores are absent on consumer
# Ada so it is skipped there). F32 uses the K1 generic path (different knobs) and
# is NOT handled here.
#
# Usage (RTX1000, local CUDA):
#   julia --project=perfs/envs/benchenv/cuda data/tuning/gemm/autotune.jl
#   julia ... data/tuning/gemm/autotune.jl --dev    # tiny smoke → <Arch>_gemm_dev.json
# =============================================================================
# Local run: `include ../init.jl` sets up backend / AT / KF / KA / GPU_TAG via the
# perfs env. On a cluster where that env is not instantiated, a driver can pre-define
# those globals (against e.g. ~/kf/env) and `include` this file — init.jl is skipped.
if !@isdefined(backend)
    include(joinpath(@__DIR__, "..", "..", "..", "perfs", "julia", "init.jl"))
end
using JSON3, Statistics

const DEV = "--dev" in ARGS
const S = 8                      # aspect factor
const L = DEV ? 512 : 2048       # representative base edge

# Arch handle (for the warp size that bounds the wg filter) + backend kind.
const ARCH   = KF.detect_arch(AT{Float16}(undef, 2, 2))
const WARPSZ = KF.get_warpsize(ARCH)
const _KIND  = nameof(typeof(backend)) === :ROCBackend ? :roc : :cuda

# ── Kernel-time bench, event-based (mirrors data/tuning/matvec/autotune.jl) ──
# `@eval begin … end` so the dead branch's `AMDGPU.@elapsed`/`CUDA` macros are
# only expanded when that branch actually runs (else `UndefVarError` on a
# single-backend env). CUDA: device event pairs, one host sync drains all
# trials → pure kernel time. AMD: block-timing with an adaptive `inner` so the
# host-dispatch gap amortizes (gemm kernels are mostly ms-scale → inner≈1).
if _KIND === :cuda
    @eval begin
        function _bench_us(f; trials::Int = $(DEV ? 2 : 10))
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
        function _bench_us(f; trials::Int = $(DEV ? 2 : 8))
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

# (name, M, N, K) at ≈ L³ FLOPs — same construction as gemm_perf_comparison.jl.
r64(x) = max(64, 64 * round(Int, x / 64))
shape_regimes(L, s) = [
    ("square", r64(L),      r64(L),       r64(L)),
    ("outer",  r64(L * s),  r64(L * s),   r64(L / s^2)),
    ("inner",  r64(L / s),  r64(L / s),   r64(L * s^2)),
    ("A_thin", r64(L * s),  r64(L / s^2), r64(L * s)),
    ("B_thin", r64(L / s^2), r64(L * s),  r64(L * s)),
    ("tall",   r64(L * s^2), r64(L / s),  r64(L / s)),
]

# Candidate knob grid — MUST cover the hand-coded `default_gemm_mma_tile` config so
# an accurately-timed search can never land BELOW it (we deliberately do NOT clamp
# with a never-worse guard — that would bake the heuristic in as a floor and break
# the cross-arch honesty of the tune; instead the default is simply a candidate).
# Reachability the old grid broke → regressions: (a) deep-K panels use `WK = 64÷Kt`
# (=4 for F16 Kt=16) but the grid capped `WK≤2` → `inner` regressed 1.51× on RTX1000;
# (b) the 16-warp tier uses `NWM=8`, absent from `{1,2,4}`; (c) `PC` = `16÷sizeof(CT)`
# (2 for F64, 16 for Int8) — a fixed `PC=8` couldn't reproduce the default pad off
# F16. So `WK∈{1,2,4}`, `NWM∈{1,2,4,8}`, `PC∈{default_pad(CT), 8}`, and the wg cap is
# warpsize-aware (`NWM·NWN·warpsz ≤ 1024`; ws=64 on CDNA halves it — the old fixed 32
# silently over-generated infeasible AMD configs).
padof(::Type{T}) where {T} = 16 ÷ sizeof(T)
function candidate_knobs(::Type{CT}, warpsz) where {CT}
    maxNW = 1024 ÷ warpsz
    pcs = unique((padof(CT), 8))
    cs = NamedTuple[]
    for WM in (2, 4), WN in (2, 4), NWM in (1, 2, 4, 8), NWN in (1, 2, 4),
        WK in (1, 2, 4), PC in pcs, DB in (false, true)
        NWM * NWN <= maxNW || continue
        DB && WK == 1 && continue             # DB is meaningful only for WK≥2 tiles
        push!(cs, (; WM, WN, NWM, NWN, WK, PC, PA = 0, DB))
    end
    DEV ? cs[1:min(6, end)] : cs
end

# K1 (generic, non-tensor) tile grid — for dtypes WITHOUT an MMA path (F32 on every
# arch; F64/Int8 where the device lacks those tensor cores). Knobs (BM,BN,BK,TM,TN);
# the K1 default_gemm_tile is a tiny 64×64×8 (TM/TN=4) that badly under-tiles the
# CUDA-core GEMM vs a vendor SGEMM's 128×128 register-blocked tiles, so there is a lot
# of headroom. Constraints mirror the launcher asserts (resolve_gemm_parameters):
# BM%TM==0, BN%TN==0, wg=(BM÷TM)(BN÷TN) a warp-multiple ≤1024, shared (BM·BK+BK·BN)
# fits the 48 KB @localmem cap. The default (64,64,8,4,4) is itself in the grid, so —
# as with the MMA grid — an accurately-timed search can never land below it.
function candidate_knobs_k1(::Type{T}, warpsz) where {T}
    cap = 48 * 1024
    cs = NamedTuple[]
    for BM in (64, 128, 256), BN in (64, 128, 256), BK in (8, 16, 32),
        TM in (4, 8), TN in (4, 8)
        (BM % TM == 0 && BN % TN == 0) || continue
        wg = (BM ÷ TM) * (BN ÷ TN)
        (wg <= 1024 && wg % warpsz == 0) || continue
        (BM * BK + BK * BN) * sizeof(T) <= cap || continue
        push!(cs, (; BM, BN, BK, TM, TN))
    end
    DEV ? cs[1:min(6, end)] : cs
end

# Kernel time (µs) of a gemm call; NaN on any error (infeasible tile, OOM…).
# Warmup/compile first (catches the infeasible configs), then hand off to the
# event-based `_bench_us` defined above.
function bench_us(f)
    try
        f(); KA.synchronize(backend)          # warmup / compile
    catch
        return NaN
    end
    try
        return _bench_us(f)
    catch
        return NaN
    end
end

# Does the :mma path compile+run for this dtype on this device?
function mma_available(::Type{T}) where {T}
    A = fill!(AT{T}(undef, 64, 64), one(T)); B = fill!(AT{T}(undef, 64, 64), one(T))
    ok = try
        KF.gemm(A, B; family = :mma); KA.synchronize(backend); true
    catch
        false
    end
    return ok
end

function tune_cell(name, M, N, K, ::Type{T}) where {T}
    A = fill!(AT{T}(undef, M, K), one(T)); B = fill!(AT{T}(undef, K, N), one(T))
    best = nothing; best_us = Inf
    for kn in candidate_knobs(T, WARPSZ)
        us = bench_us(() -> KF.gemm(A, B; family = :mma,
                                     kn.WM, kn.WN, kn.NWM, kn.NWN, kn.WK, kn.PC, kn.PA, kn.DB))
        if us < best_us
            best_us = us; best = kn
        end
    end
    println("  $name $T  →  best $(best_us == Inf ? "NONE" : round(best_us; digits=1))µs  $best")
    return best, best_us
end

# Same search over the K1 generic (BM,BN,BK,TM,TN) grid. Any generic tile knob forces
# the generic family, so this exercises the CUDA-core kernel (no MMA).
function tune_cell_k1(name, M, N, K, ::Type{T}) where {T}
    A = fill!(AT{T}(undef, M, K), one(T)); B = fill!(AT{T}(undef, K, N), one(T))
    best = nothing; best_us = Inf
    for kn in candidate_knobs_k1(T, WARPSZ)
        us = bench_us(() -> KF.gemm(A, B; kn.BM, kn.BN, kn.BK, kn.TM, kn.TN))
        if us < best_us
            best_us = us; best = kn
        end
    end
    println("  [K1] $name $T  →  best $(best_us == Inf ? "NONE" : round(best_us; digits=1))µs  $best")
    return best, best_us
end

# -----------------------------------------------------------------------------
tag = DEV ? "$(GPU_TAG)_gemm_dev" : "$(GPU_TAG)_gemm"
# All MMA-capable dtypes; `mma_available` auto-skips those without a tensor-core path
# on this device (RTX1000 Ada → F16 only; A100/MI300A → also F64, BF16, Int8).
types = [Float16, Core.BFloat16, Float32, Float64, Int8]
out = Dict{String,Any}()      # MMA (K2) knobs → section "gemm"
out_k1 = Dict{String,Any}()   # generic (K1) knobs → section "gemm_k1"
for T in types
    # Per-type guard: a type that can't even be constructed/probed on this device
    # (e.g. BFloat16 without BFloat16s loaded) must skip, not abort the whole run.
    try
        tkey = string(T)
        if mma_available(T)
            out[tkey] = Dict{String,Any}()
            for (name, M, N, K) in shape_regimes(L, S)
                best, us = tune_cell(name, M, N, K, T)
                best === nothing && continue
                out[tkey][name] = Dict("M" => M, "N" => N, "K" => K, "us" => us,
                    "WM" => best.WM, "WN" => best.WN, "NWM" => best.NWM, "NWN" => best.NWN,
                    "WK" => best.WK, "PC" => best.PC, "PA" => best.PA, "DB" => best.DB)
            end
        else
            # No tensor-core path → tune the K1 generic (CUDA-core) tile instead of
            # leaving it on the tiny hand-coded default. This is where F32 lives on
            # every arch (and F64/Int8 without those tensor cores).
            @info "MMA path unavailable for $T on $GPU_TAG — tuning the K1 generic tile"
            # skip integer dtypes that can't `one(T)`-fill a float GEMM meaningfully?
            # No — F32/F64 are the target; Int8 K1 is valid too. Probe feasibility via
            # a tiny call; if even the default tile can't run, skip the type.
            probe = try
                A0 = fill!(AT{T}(undef, 64, 64), one(T)); B0 = fill!(AT{T}(undef, 64, 64), one(T))
                KF.gemm(A0, B0); KA.synchronize(backend); true
            catch
                false
            end
            if !probe
                @info "K1 generic path also unavailable for $T on $GPU_TAG — skipped"
                continue
            end
            out_k1[tkey] = Dict{String,Any}()
            for (name, M, N, K) in shape_regimes(L, S)
                best, us = tune_cell_k1(name, M, N, K, T)
                best === nothing && continue
                out_k1[tkey][name] = Dict("M" => M, "N" => N, "K" => K, "us" => us,
                    "BM" => best.BM, "BN" => best.BN, "BK" => best.BK,
                    "TM" => best.TM, "TN" => best.TN)
            end
        end
    catch e
        @warn "tuning $T failed — skipped this type, continuing" exception = (e, catch_backtrace())
    end
end

payload = Dict("schema" => 1, "gpu_tag" => GPU_TAG, "base_L" => L, "aspect_s" => S,
    "gemm" => out, "gemm_k1" => out_k1)
path = joinpath(@__DIR__, "..", "$(tag).json")
open(path, "w") do io
    JSON3.pretty(io, payload)
end
println("\nAutotune JSON written to: $path")

# Runtime artifact for the loader (src/tuning/loader.jl `lookup_gemm`): a Dict literal
# `<tag>.jl` (= <Arch>_gemm.jl) holding schema_version + "gemm" → dtype → regime →
# knobs. Only the winning knobs are kept (M/N/K/us dropped). NOT written in --dev mode.
if !DEV
    knobs = Dict{String,Any}("schema_version" => 1,
        "gemm" => Dict{String,Any}(
            tk => Dict{String,Any}(sk => Dict{String,Any}(
                    "WM" => c["WM"], "WN" => c["WN"], "NWM" => c["NWM"], "NWN" => c["NWN"],
                    "WK" => c["WK"], "PC" => c["PC"], "PA" => c["PA"], "DB" => c["DB"])
                for (sk, c) in tv) for (tk, tv) in out),
        "gemm_k1" => Dict{String,Any}(
            tk => Dict{String,Any}(sk => Dict{String,Any}(
                    "BM" => c["BM"], "BN" => c["BN"], "BK" => c["BK"],
                    "TM" => c["TM"], "TN" => c["TN"])
                for (sk, c) in tv) for (tk, tv) in out_k1))
    jlpath = joinpath(@__DIR__, "..", "$(tag).jl")
    open(jlpath, "w") do io
        println(io, "# Auto-generated by data/tuning/gemm/autotune.jl — do not edit by hand.")
        print(io, repr(knobs))
    end
    println("Runtime artifact written to: $jlpath")
end
