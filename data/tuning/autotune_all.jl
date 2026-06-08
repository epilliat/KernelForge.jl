# =============================================================================
# data/tuning/autotune_all.jl
# =============================================================================
# Master autotune driver — runs all four ops (matvec, vecmat, mapreduce1d,
# scan) for the requested dtypes on the active GPU. Kwargs let the user
# filter ops and dtypes.
#
# Each per-op script lives in its own sub-folder (`data/tuning/<op>/`) and
# defines its own `autotune()` function. To avoid name clashes when
# pulling all four into one Julia session, each is included inside a
# wrapper sub-module here. The dispatch table in `autotune_all()` calls
# the right one with the right kwargs (matvec/vecmat take `safety`,
# mapreduce1d/scan do not).
#
# Where output lands:
# - matvec/vecmat → `data/tuning/<Arch>.json` (+ `.jl` companion). Per
#   dtype, the writer picks `size_<sizeof(T)>` (generic) or `<T>`
#   (precise) sections. Default = generic.
# - mapreduce1d/scan → `data/tuning/<Arch>_inline.jl` (codegen). For
#   `Float32`, emits the generic `where T` block; for other dtypes,
#   emits a type-specific `::Type{T}` block under a separate marker.
#
# Usage:
#
#   # Everything (4 ops × 3 dtypes, ~55-60 min wall):
#   julia --project=… data/tuning/autotune_all.jl
#
#   # Filtered:
#   julia --project=… -e '
#     include("data/tuning/autotune_all.jl")
#     autotune_all(; ops=[:matvec, :vecmat], dtypes=[Float32])
#   '
#
#   # Just one op for one dtype:
#   autotune_all(; ops=[:scan], dtypes=[Float64])
# =============================================================================

# Each sub-module wraps one per-op script. Wrapping is necessary because
# every script defines a top-level `autotune()` — included flat they'd
# collide. The sub-modules also isolate the backend-detection @eval blocks.

module _Matvec
    include(joinpath(@__DIR__, "matvec", "autotune.jl"))
end
module _Vecmat
    include(joinpath(@__DIR__, "vecmat", "autotune.jl"))
end
module _MR1D
    include(joinpath(@__DIR__, "mapreduce1d", "autotune.jl"))
end
module _Scan
    include(joinpath(@__DIR__, "scan", "autotune.jl"))
end

using Printf

"""
    autotune_all(; ops, dtypes, safety, verbose)

Run all selected autotunes on the active GPU.

Kwargs:
- `ops::Vector{Symbol}` — subset of `[:matvec, :vecmat, :mapreduce1d, :scan]`.
  Default: all four.
- `dtypes::Vector{Type}` — Default is `[Float32, Float64, UInt8]`.
  UInt8 is the canonical 1-byte stress test — the generic Float32-anchored
  formula (`clamp(nitem_f32 × 4 ÷ sizeof(T), 1, 16)`) extrapolates badly
  to single-byte types because the coalesced-load minimum (32 bytes per
  warp transaction) requires a higher Nitem than the formula yields.
  Adding a type-specific UInt8 block fixes the gap; the same block
  effectively covers Int8 too at the kernel level (same sizeof, same
  access pattern), but Int8 dispatch is exact so add it explicitly if
  you need it.
  Float16 / other bitstypes also work — pass them explicitly.
- `safety::Int = 2` — GPU-mem safety factor (used only by matvec/vecmat;
  controls the shape grid via `pow28_shapes(T; safety)`).
- `verbose::Bool = true`.

Behaviour per (op, dtype):
- `matvec`, `vecmat`: writes a size-keyed JSON section (`size_<sizeof(T)>`)
  by default. `lookup_*` checks dtype-specific first, then size-keyed,
  then cross-size — so a single dtype tune already benefits all dtypes
  of the same size class.
- `mapreduce1d`, `scan`: for `T = Float32`, refreshes the generic
  `where T` codegen block. For other dtypes, emits a type-specific
  override block alongside the generic one (Julia dispatch will pick the
  specific one at runtime). Re-running for the same dtype overwrites only
  that block's markers; other ops/dtypes are preserved.

Wall-time estimates per (op, dtype) on a mid-range GPU:
- matvec/vecmat: 5-12 min (varies with dtype; Int8 is the longest)
- mapreduce1d / scan: 2-3 min
"""
function autotune_all(; ops::Vector{Symbol} = [:matvec, :vecmat, :mapreduce1d, :scan],
                        dtypes::Vector = Type[Float32, Float64, UInt8],
                        safety::Int = 2,
                        verbose::Bool = true)
    valid_ops = Set([:matvec, :vecmat, :mapreduce1d, :scan])
    for op in ops
        op in valid_ops || error("Unknown op $op (valid: $valid_ops)")
    end

    n_total = length(ops) * length(dtypes)
    verbose && @printf("\n══ autotune_all: %d ops × %d dtypes = %d runs ══\n",
                       length(ops), length(dtypes), n_total)
    verbose && @printf("  ops:    %s\n  dtypes: %s\n  safety: %d\n",
                       ops, dtypes, safety)

    t_total0 = time()
    n_done = 0
    for T in dtypes, op in ops
        n_done += 1
        verbose && @printf("\n── [%d/%d] %s × %s ──\n", n_done, n_total, op, T)
        t0 = time()
        try
            if op === :matvec
                _Matvec.autotune(T; safety, verbose)
            elseif op === :vecmat
                _Vecmat.autotune(T; safety, verbose)
            elseif op === :mapreduce1d
                _MR1D.autotune(T; verbose)
            elseif op === :scan
                _Scan.autotune(T; verbose)
            end
        catch e
            @warn "  autotune failed: $op × $T" exception=(e, catch_backtrace())
        end
        verbose && @printf("  done in %.1f min\n", (time() - t0) / 60)
    end

    verbose && @printf("\n══ autotune_all wall: %.1f min ══\n",
                       (time() - t_total0) / 60)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    autotune_all()
end
