# Sortperm: returns the permutation that would sort `src` ascending under
# `uint_map(by(·))`. Implementation reuses the keyval onesweep kernel —
# initialize `perm` to the identity `[1..n]`, then sort it as the values
# with `src` as the keys. After all passes, `perm[j]` is the input index
# that ends up at position j in sorted order, matching Julia's stdlib
# `sortperm`.
#
# No new kernel: the keyval radix machinery in keyval_onesweep_kernel.jl
# handles the (val, key) ping-pong and we just supply `vals = identity_perm`
# and `keys = src`. The throwaway `dst_keys` buffer (final-pass sorted keys
# we discard) lives inside the Sortperm `KernelBuffer`.

using KernelAbstractions


# ============================================================================
# Tiny identity-fill kernel
# ============================================================================

# Fill `out[i] = IndexType(i)` for i in 1..length(out). Backend-portable;
# the codebase prefers @kernel-based primitives over CuArray-only broadcasts.
@kernel inbounds = true unsafe_indices = true function _fill_iota_kernel!(out)
    i = Int(@index(Global))
    if i <= length(out)
        @inbounds out[i] = oftype(out[1], i)
    end
end

@inline function fill_iota!(out::AbstractVector)
    n = length(out)
    n == 0 && return out
    backend = get_backend(out)
    workgroup = 256
    ndrange = cld(n, workgroup) * workgroup
    _fill_iota_kernel!(backend, workgroup, ndrange)(out)
    return out
end


# ============================================================================
# Defaults
# ============================================================================

# Default index type: Int32 keeps the radix at 4 byte-passes (vs 8 for
# Int64). Auto-promotes to Int64 only when n exceeds typemax(Int32).
@inline _default_index_type(n::Integer) =
    n <= typemax(Int32) ? Int32 : Int64


# ============================================================================
# Public API
# ============================================================================

"""
    sortperm(src::AbstractVector; by=identity, uint_map=KernelForge.uint_map,
             IndexType=nothing, kwargs...) -> AbstractVector{IndexType}

GPU sortperm: returns a permutation vector `p` such that `src[p]` is sorted
ascending under `uint_map(by(·))`. Stable (matches Julia's stdlib
`sortperm`). Reuses the keyval radix kernel — see [`sort`](@ref).

# Keyword Arguments
- `by=identity`: Field extractor applied to elements of `src` (the sort
  keys). For tuples, e.g. `by=first`.
- `uint_map=KernelForge.uint_map`: Order-preserving projection of `by(x)`
  to an unsigned integer.
- `IndexType=nothing`: Element type of the returned permutation. Defaults
  to `Int32` when `length(src) ≤ typemax(Int32)`, else `Int64`. Smaller
  index types reduce the number of byte passes (4 for Int32, 8 for Int64).
- `tmp=nothing`: Pre-allocated `KernelBuffer` from
  `get_allocation(Sortperm, src; ...)` for hot-loop reuse.
- `Nitem`, `Nchunks`, `workgroup`, `arch`: tuning knobs.

# Examples
```julia
src = CuArray(rand(Float32, 1_000_000))
p = KernelForge.sortperm(src)
@assert Array(src)[Array(p)] == sort(Array(src))
```

See also: [`sortperm!`](@ref).
"""
function sortperm end

"""
    sortperm!(perm, src; by=identity, uint_map=KernelForge.uint_map, kwargs...) -> perm

In-place form of [`sortperm`](@ref): writes the permutation into `perm`
(which must have integer eltype and the same length as `src`).
"""
function sortperm! end


# ============================================================================
# Allocation
# ============================================================================

"""
    get_allocation(::Type{Sortperm}, src; IndexType=nothing,
                   Nitem=nothing, Nchunks=nothing, workgroup=nothing,
                   arch=nothing, by=identity,
                   uint_map=KernelForge.uint_map) -> KernelBuffer

Allocate a `KernelBuffer` for `sortperm!`. Strict superset of the keyval
`Sort1D` buffer, with one extra `keys_dst` field for the throwaway
final-pass sorted keys.
"""
function get_allocation(
    ::Type{Sortperm},
    src::AT;
    IndexType=nothing,
    Nitem=nothing,
    Nchunks=nothing,
    workgroup=nothing,
    arch=nothing,
    by::F=identity,
    uint_map::M=uint_map,
) where {AT<:AbstractArray,F,M}
    n = length(src)
    IT = something(IndexType, _default_index_type(n))

    # Build the underlying keyval Sort1D buffer with an IT-typed template
    # for the "values" side and `src` providing the keys side.
    backend = get_backend(src)
    perm_template = KernelAbstractions.allocate(backend, IT, n)
    sub = get_allocation(Sort1D, perm_template;
                         keys=src,
                         Nitem, Nchunks, workgroup, arch, by, uint_map)

    # Throwaway dst_keys: ping-pong target for the final pass's keys.
    # eltype(keys_dst) = eltype(src) since the keyval kernel writes `src`-
    # typed values into it.
    keys_dst = KernelAbstractions.allocate(backend, eltype(src), n)
    return KernelBuffer((; sub.arrays..., keys_dst))
end


# ============================================================================
# Allocating front-end
# ============================================================================

function sortperm(src::AT;
                  by::F=identity,
                  uint_map::M=uint_map,
                  IndexType=nothing,
                  tmp::TMP=nothing,
                  Nitem=nothing, Nchunks=nothing,
                  workgroup=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,Nothing}}
    n = length(src)
    IT = something(IndexType, _default_index_type(n))
    perm = similar(src, IT, n)
    sortperm!(perm, src; by, uint_map, tmp, Nitem, Nchunks, workgroup, arch)
    return perm
end


# ============================================================================
# In-place entry
# ============================================================================

function sortperm!(perm::AbstractVector, src::AT;
                   by::F=identity,
                   uint_map::M=uint_map,
                   tmp::TMP=nothing,
                   Nitem=nothing, Nchunks=nothing,
                   workgroup=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,Nothing}}
    eltype(perm) <: Integer || error("`perm` must have an integer eltype; got $(eltype(perm))")
    n = length(src)
    length(perm) == n || error("`perm` must have the same length as `src`")
    n == 0 && return perm

    IT = eltype(perm)
    T  = eltype(AT)
    KT = T                                # keyval kernel: KT = eltype(src_keys) = eltype(src)
    K = Base.promote_op(uint_map ∘ by, KT)
    K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")

    backend = get_backend(src)
    arch_ = something(arch, detect_arch(src))
    # sortperm runs on the KEY/VALUE onesweep, which still indexes `bucket = lid`
    # and is therefore pinned to workgroup == Nbuckets. It must not pick up the
    # autotuned `default_workgroup`, which is tuned for — and only valid on — the
    # workgroup-decoupled keys-only kernel. See `sort_workgroup` in sort1d.jl.
    workgroup_ = something(workgroup, sort_workgroup(arch_, n, IT, false))
    # Use the keyval defaults: shared_sorted (IT) + shared_keys (KT) must
    # fit alongside hist + aux in 48 KB.
    Nitem_   = something(Nitem,   default_nitem_keyval(arch_, n, IT, KT))
    Nchunks_ = something(Nchunks, default_nchunks_keyval(arch_, n, IT, KT))
    npasses = sizeof(K)

    # Allocate (or reuse) the buffer. We need our own keys_dst — the
    # underlying `_sort1d_keyval_impl!` reads `tmp.arrays.{...}` by name
    # and tolerates extra fields, so a Sortperm buffer is a drop-in.
    tmp_ = something(tmp, get_allocation(Sortperm, src;
                                          IndexType=IT,
                                          Nitem=Nitem_, Nchunks=Nchunks_,
                                          workgroup=workgroup_, arch=arch_,
                                          by, uint_map))
    keys_dst = tmp_.arrays.keys_dst

    # Initialize identity. The keyval impl will copy `perm → scratch` (odd
    # npasses) or `perm → dst` (even, no-op when dst === src === perm).
    fill_iota!(perm)

    _dispatch_keyval!(Val(Nitem_), Val(Nchunks_),
                      by, uint_map,
                      perm, perm,                       # dst, src — aliased
                      keys_dst, src,                    # dst_keys (throwaway), src_keys
                      tmp_,
                      Nitem_, Nchunks_, workgroup_, npasses, K, backend, arch_)
    return perm
end
