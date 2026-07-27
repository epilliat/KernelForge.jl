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

# Candidate knob grid. Centered on the WM=2/WN=4 plateau; DB only with WK≥2.
function candidate_knobs()
    cs = NamedTuple[]
    for WM in (2, 4), WN in (2, 4), NWM in (1, 2, 4), NWN in (1, 2, 4),
        WK in (1, 2), DB in (false, true)
        NWM * NWN <= 32 || continue          # wg = NWM·NWN·warpsize ≤ 1024 (ws=32)
        DB && WK == 1 && continue             # DB is meaningful only for WK≥2 tiles
        push!(cs, (; WM, WN, NWM, NWN, WK, PC = 8, PA = 0, DB))
    end
    DEV ? cs[1:min(6, end)] : cs
end

# Median kernel time (µs) of a gemm call; NaN on any error (infeasible tile, OOM…).
function time_call(f; trials = DEV ? 5 : 15)
    try
        f(); KA.synchronize(backend)          # warmup / compile
    catch
        return NaN
    end
    ts = Float64[]
    for _ in 1:trials
        push!(ts, @elapsed (f(); KA.synchronize(backend)))
    end
    return median(ts) * 1e6
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
    for kn in candidate_knobs()
        us = time_call(() -> KF.gemm(A, B; family = :mma,
                                     kn.WM, kn.WN, kn.NWM, kn.NWN, kn.WK, kn.PC, kn.PA, kn.DB))
        if us < best_us
            best_us = us; best = kn
        end
    end
    println("  $name $T  →  best $(best_us == Inf ? "NONE" : round(best_us; digits=1))µs  $best")
    return best, best_us
end

# -----------------------------------------------------------------------------
tag = DEV ? "$(GPU_TAG)_gemm_dev" : "$(GPU_TAG)_gemm"
# All MMA-capable dtypes; `mma_available` auto-skips those without a tensor-core path
# on this device (RTX1000 Ada → F16 only; A100/MI300A → also F64, BF16, Int8).
types = [Float16, Core.BFloat16, Float32, Float64, Int8]
out = Dict{String,Any}()
for T in types
    # Per-type guard: a type that can't even be constructed/probed on this device
    # (e.g. BFloat16 without BFloat16s loaded) must skip, not abort the whole run.
    try
        if !mma_available(T)
            @info "MMA path unavailable for $T on $GPU_TAG — skipped (K1 generic path, not tuned here)"
            continue
        end
        tkey = string(T)
        out[tkey] = Dict{String,Any}()
        for (name, M, N, K) in shape_regimes(L, S)
            best, us = tune_cell(name, M, N, K, T)
            best === nothing && continue
            out[tkey][name] = Dict("M" => M, "N" => N, "K" => K, "us" => us,
                "WM" => best.WM, "WN" => best.WN, "NWM" => best.NWM, "NWN" => best.NWN,
                "WK" => best.WK, "PC" => best.PC, "PA" => best.PA, "DB" => best.DB)
        end
    catch e
        @warn "tuning $T failed — skipped this type, continuing" exception = (e, catch_backtrace())
    end
end

payload = Dict("schema" => 1, "gpu_tag" => GPU_TAG, "base_L" => L, "aspect_s" => S, "gemm" => out)
path = joinpath(@__DIR__, "..", "$(tag).json")
open(path, "w") do io
    JSON3.pretty(io, payload)
end
println("\nAutotune JSON written to: $path")

# Runtime artifact for the loader (src/tuning/loader.jl `lookup_gemm`): a Dict literal
# `<tag>.jl` (= <Arch>_gemm.jl) holding schema_version + "gemm" → dtype → regime →
# knobs. Only the winning knobs are kept (M/N/K/us dropped). NOT written in --dev mode.
if !DEV
    knobs = Dict{String,Any}("schema_version" => 1, "gemm" => Dict{String,Any}(
        tk => Dict{String,Any}(sk => Dict{String,Any}(
                "WM" => c["WM"], "WN" => c["WN"], "NWM" => c["NWM"], "NWN" => c["NWN"],
                "WK" => c["WK"], "PC" => c["PC"], "PA" => c["PA"], "DB" => c["DB"])
            for (sk, c) in tv) for (tk, tv) in out))
    jlpath = joinpath(@__DIR__, "..", "$(tag).jl")
    open(jlpath, "w") do io
        println(io, "# Auto-generated by data/tuning/gemm/autotune.jl — do not edit by hand.")
        print(io, repr(knobs))
    end
    println("Runtime artifact written to: $jlpath")
end
