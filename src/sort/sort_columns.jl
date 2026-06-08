# ============================================================================
# Public batched column-sort dispatcher.
#
# `KF.sort_columns!(A)` sorts each column of `A` independently. Picks the
# right algorithm per (K, T) automatically:
#
#   - custom `lt` / `tmax` / `reverse=true` / non-`uint_map` `T`
#                                         → OEM (asserts K ≤ 4096)
#   - `uint_map`-eligible T, K ≤ Kthr(T)  → OEM
#   - `uint_map`-eligible T, K >  Kthr(T) → batched radix
#
# `Kthr(T)` is the per-type cross-over threshold tuned by
# `data/tuning/sort_columns/autotune.jl`; the cross-arch default lives in
# `default_sort_columns_threshold` below.
# ============================================================================


# ── Per-arch threshold defaults ─────────────────────────────────────────

"""
    default_sort_columns_threshold(arch, ::Type{T}) -> Int

K threshold below which `oem_sort_columns!` is used and at/above which
the batched radix path is used, for column eltype `T`. Per-arch overrides
land in `data/tuning/<arch>_inline.jl`; the generic fallback below is the
RTX1000-measured cross-over.
"""
# Semantics: `threshold(T)` is the SMALLEST K at which batched radix is
# preferred over OEM for column eltype T. The dispatcher routes
# `K < threshold(T)` to OEM and `K >= threshold(T)` to batched radix.
# OEM's hard cap is K ≤ 4096, so the threshold is clamped at OEM_K_MAX + 1
# at the call site — even an autotune emitting 5000 still works correctly.
const OEM_K_MAX = 4096

@inline default_sort_columns_threshold(::AbstractArch, ::Type{T}) where T = 1024
@inline default_sort_columns_threshold(::AbstractArch, ::Type{UInt8})  = 256
@inline default_sort_columns_threshold(::AbstractArch, ::Type{Int8})   = 256
@inline default_sort_columns_threshold(::AbstractArch, ::Type{Bool})   = 256


# ── Public API ──────────────────────────────────────────────────────────

"""
    sort_columns(A::AbstractMatrix; ...) -> Matrix
    sort_columns!(A::AbstractMatrix; algorithm=:auto, by=identity,
                  uint_map=KF.uint_map, lt=nothing, tmax=nothing,
                  reverse=false, tmp=nothing, arch=nothing) -> A

Sort each column of `A` independently and in-place (`sort_columns!`) or
into a fresh matrix (`sort_columns`).

# Dispatch

`algorithm=:auto` (default):
- Custom `lt`/`tmax`/`reverse=true` or non-`uint_map`-eligible `T`:
  routed to `oem_sort_columns!`. Requires `K ≤ 4096`.
- Otherwise: picks OEM or batched LSD radix per the per-type threshold
  in `default_sort_columns_threshold`.

Both batched-radix paths bounds-check OOB tile slots and never reference
`typemax(T)` — user-defined bitstypes with `uint_map` (and no `typemax`)
work end-to-end, at any K, without falling through to OEM.

`algorithm=:oem`: forces OEM. `K ≤ 4096`.
`algorithm=:radix`: forces batched radix. Requires `uint_map`-eligible
`T` (no custom `lt`).

# Workspace

Pass `tmp = get_allocation(SortColumns, A)` to reuse the batched-radix
workspace across calls. Ignored when the chosen algorithm is OEM.

# Examples
```julia
A = CuArray(rand(Float32, 10_000, 16))
KF.sort_columns!(A)
@assert Array(A) == hcat(sort.(eachcol(initial))...)

A2 = CuArray(rand(Int64, 1024, 1024))
KF.sort_columns!(A2; lt = >)   # routed to OEM (custom lt)
```
"""
function sort_columns end

function sort_columns!(A::AbstractMatrix{T};
                       algorithm::Symbol = :auto,
                       by::F = identity,
                       uint_map::UM = uint_map,
                       lt = nothing,
                       tmax = nothing,
                       reverse::Bool = false,
                       tmp::TMP = nothing,
                       arch = nothing) where {T, F, UM, TMP<:Union{Nothing,KernelBuffer}}
    algorithm in (:auto, :oem, :radix) ||
        error("`algorithm` must be :auto, :oem, or :radix; got $algorithm")

    arch_ = something(arch, detect_arch(A))

    # Custom comparator / pad sentinel / reverse: only OEM understands these.
    must_oem = (lt !== nothing) || (tmax !== nothing) || reverse

    # Whether T is uint_map-eligible. We probe rather than constrain on a
    # type-set so a user's own uint_map extension Just Works.
    uint_map_eligible = false
    if !must_oem
        K_uint = Base.promote_op(uint_map ∘ by, T)
        uint_map_eligible = K_uint <: Unsigned
    end

    if algorithm === :oem || must_oem || !uint_map_eligible
        size(A, 1) <= 4096 ||
            error("oem_sort_columns! requires K ≤ 4096; got K=$(size(A,1)). " *
                  "Use algorithm=:radix (uint_map types) or per-column sample-sort.")
        return oem_sort_columns!(A; lt)
    end

    K = size(A, 1)
    if algorithm === :radix
        return batched_radix_sort_columns!(A; tmp, by, uint_map)
    end

    # algorithm === :auto, uint_map-eligible, no custom comparator.
    # Clamp threshold at OEM_K_MAX+1 so K > OEM_K_MAX always routes to radix
    # (OEM can't handle K > 4096 regardless of what autotune emitted).
    K_thr = min(default_sort_columns_threshold(arch_, T), OEM_K_MAX + 1)

    if K < K_thr
        return oem_sort_columns!(A)
    else
        return batched_radix_sort_columns!(A; tmp, by, uint_map)
    end
end

function sort_columns(A::AbstractMatrix; kwargs...)
    B = similar(A)
    copyto!(B, A)
    sort_columns!(B; kwargs...)
    return B
end

# Both names share the docstring above sort_columns.
@doc (@doc sort_columns) sort_columns!
