# ============================================================================
# Per-architecture defaults
# ============================================================================

@inline default_workgroup(arch::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = 256
@inline default_workgroup(::AMDArch,          ::Type{Sort1D}, n, ::Type{T}) where T = 256

# Sort1D's "Nitem" is the vector load width per warp lane.
# Sweep on RTX 1000 Ada Laptop (sm_89) found (Nitem=16, Nchunks=2) the
# all-N champion. Other architectures haven't been swept yet — same
# defaults until tuning data exists.
@inline default_nitem(::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::RTX1000,      ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::A40,          ::Type{Sort1D}, n, ::Type{T}) where T = 16
@inline default_nitem(::AMDArch,      ::Type{Sort1D}, n, ::Type{T}) where T = 16

# Chunks per warp. We size block_size_max = workgroup * Nchunks * Nitem so
# that shared_sorted (= block_size_max * sizeof(T) bytes) fits alongside
# shared_hist (8 KB) + shared_aux (~1 KB) in the 48 KB static-shared budget.
# For sizeof(T) ≤ 4 bytes, (Nitem=16, Nchunks=2) → 8192 elements → up to 32 KB
# shared_sorted, total ≤ 41 KB.
# For sizeof(T) = 8, halve Nchunks → 4096 elements × 8 B = 32 KB, total 41 KB.
@inline default_nchunks(::AbstractArch, ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::RTX1000,      ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::A40,          ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1
@inline default_nchunks(::AMDArch,      ::Type{Sort1D}, n, ::Type{T}) where T = sizeof(T) <= 4 ? 2 : 1


# ============================================================================
# Public API
# ============================================================================

"""
    sort1d(src::AbstractVector; by=identity, uint_map=KernelForge.uint_map, kwargs...)

Stable LSD radix sort on the GPU. Sorts `src` ascending by
`uint_map(by(x))`. Returns a new array with the sorted values; `src` is
not modified.

# Keyword Arguments
- `by=identity`: Field extractor. For tuples, e.g. `by=first` to sort by
  the first element.
- `uint_map=KernelForge.uint_map`: Order-preserving projection of
  `by(x)` to an unsigned integer (UInt8/16/32/64). Defaults are provided
  for built-in numeric types.
- `tmp=nothing`: Pre-allocated `KernelBuffer` (or `nothing` to allocate
  automatically).
- `Nitem=nothing`, `Nchunks=nothing`, `workgroup=nothing`,
  `arch=nothing`: tuning knobs (auto-selected if nothing).

# Examples
```julia
x = CuArray(rand(Float64, 10_000_000))
y = KernelForge.sort1d(x)
@assert Array(y) == sort(Array(x))

# Sort tuples by the first element.
pairs = CuArray([(rand(Float32), rand(UInt32)) for _ in 1:1_000_000])
sorted_pairs = KernelForge.sort1d(pairs; by=first)
```

See also: [`sort1d!`](@ref) for the in-place form.
"""
function sort1d end

"""
    sort1d!(dst, src; by=identity, uint_map=KernelForge.uint_map, kwargs...)

In-place form of [`sort1d`](@ref): writes the sorted output into `dst`.
`dst` and `src` may alias only if you don't mind `src` being clobbered.

See [`sort1d`](@ref) for keyword arguments.
"""
function sort1d! end

"""
    get_allocation(::Type{Sort1D}, src; Nitem=nothing, Nchunks=nothing,
                   workgroup=nothing, arch=nothing, by=identity,
                   uint_map=KernelForge.uint_map)

Allocate a `KernelBuffer` for `sort1d!`. Holds the global byte-histogram,
the two `partial` buffers used by decoupled lookback, the per-block
flag, and a same-size scratch ping-pong buffer.
"""
function get_allocation(
    ::Type{Sort1D},
    src::AT;
    Nitem=nothing,
    Nchunks=nothing,
    workgroup=nothing,
    arch=nothing,
    by::F=identity,
    uint_map::M=uint_map,
) where {AT<:AbstractArray,F,M}
    T = eltype(AT)
    n = length(src)
    arch = something(arch, detect_arch(src))
    workgroup = something(workgroup, default_workgroup(arch, Sort1D, n, T))
    Nitem = something(Nitem, default_nitem(arch, Sort1D, n, T))
    Nchunks = something(Nchunks, default_nchunks(arch, Sort1D, n, T))

    K = Base.promote_op(uint_map ∘ by, T)
    K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")
    npasses = sizeof(K)

    block_size_max = workgroup * Nchunks * Nitem
    nblocks = cld(n, block_size_max)

    backend = get_backend(src)
    Nbuckets = 256
    hist     = KernelAbstractions.allocate(backend, UInt32, Nbuckets, npasses)
    partial1 = KernelAbstractions.allocate(backend, UInt32, Nbuckets, nblocks)
    partial2 = KernelAbstractions.allocate(backend, UInt32, Nbuckets, nblocks)
    flag     = KernelAbstractions.allocate(backend, UInt8,  nblocks)
    scratch  = KernelAbstractions.allocate(backend, T,      n)
    return KernelBuffer((; hist, partial1, partial2, flag, scratch))
end


# ============================================================================
# Allocating front-end
# ============================================================================

function sort1d(src::AT;
                by::F=identity,
                uint_map::M=uint_map,
                tmp::TMP=nothing,
                Nitem=nothing, Nchunks=nothing,
                workgroup=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,Nothing}}
    dst = similar(src)
    sort1d!(dst, src; by, uint_map, tmp, Nitem, Nchunks, workgroup, arch)
    return dst
end


# ============================================================================
# In-place entry
# ============================================================================

function sort1d!(dst::AbstractArray, src::AT;
                 by::F=identity,
                 uint_map::M=uint_map,
                 tmp::TMP=nothing,
                 Nitem=nothing, Nchunks=nothing,
                 workgroup=nothing, arch=nothing) where
        {AT<:AbstractArray,F,M,TMP<:Union{KernelBuffer,Nothing}}
    T = eltype(AT)
    K = Base.promote_op(uint_map ∘ by, T)
    K <: Unsigned || error("`uint_map ∘ by` must return a UInt8/16/32/64; got $K")

    n = length(src)
    backend = get_backend(src)
    arch = something(arch, detect_arch(src))
    workgroup = something(workgroup, default_workgroup(arch, Sort1D, n, T))
    Nitem_ = something(Nitem, default_nitem(arch, Sort1D, n, T))
    Nchunks_ = something(Nchunks, default_nchunks(arch, Sort1D, n, T))
    npasses = sizeof(K)

    # Dispatch to a Val-typed helper. For pre-compiled specs (Nitem=16 with
    # Nchunks∈{1,2}, byte-sort Nitem=16), Julia statically resolves to a
    # fast method that calls `_sort1d_impl!` directly with a typed kernel
    # function — no Dict lookup, no invokelatest, no dynamic dispatch in
    # the hot path. The generic fallbacks @eval the kernel and use
    # invokelatest just on first call for that spec.
    if npasses == 1
        _dispatch_byte_sort!(Val(Nitem_),
                             by, uint_map, dst, src, tmp,
                             Nitem_, workgroup, K, backend, arch)
    else
        _dispatch_onesweep!(Val(Nitem_), Val(Nchunks_),
                            by, uint_map, dst, src, tmp,
                            Nitem_, Nchunks_, workgroup, npasses, K, backend, arch)
    end
    return dst
end


# ----------------------------------------------------------------------------
# Val-typed dispatchers
# ----------------------------------------------------------------------------

# Fast paths: pre-compiled specs, statically dispatched, no invokelatest.
# We retrieve the kernel via `get_*_kernel(Val(...))` rather than referencing
# `byte_sort_kernel_16!` / `onesweep_kernel_16_{1,2}!` by bare name. The
# typed `Val{...}` methods (installed at package load via Core.eval) embed
# the kernel as a literal value in the method body, so the @inline expands
# to a literal-returning call with no global-binding access — sidestepping
# Julia 1.12's "access to binding in a world prior to its definition world"
# warning that bare-name references would trigger here.
@inline _dispatch_byte_sort!(::Val{16},
        by, uint_map, dst, src, tmp,
        Nitem, workgroup, ::Type{K}, backend, arch) where K =
    _sort1d_impl_byte!(get_byte_sort_kernel(Val(16)),
                       by, uint_map, dst, src, tmp,
                       Nitem, workgroup, K, backend, arch)

@inline _dispatch_onesweep!(::Val{16}, ::Val{1},
        by, uint_map, dst, src, tmp,
        Nitem, Nchunks, workgroup, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_impl!(get_radix_kernel(Val(16), Val(1)),
                  by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, npasses, K, backend, arch)

@inline _dispatch_onesweep!(::Val{16}, ::Val{2},
        by, uint_map, dst, src, tmp,
        Nitem, Nchunks, workgroup, npasses, ::Type{K}, backend, arch) where K =
    _sort1d_impl!(get_radix_kernel(Val(16), Val(2)),
                  by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, npasses, K, backend, arch)

# Fallbacks: exotic specs. Factory may @eval; we then invokelatest the
# impl exactly once per fresh spec. Subsequent calls for the same spec
# hit the typed method installed by `_define_*!` (statically dispatched).
function _dispatch_byte_sort!(::Val{Nitem},
        by, uint_map, dst, src, tmp,
        Nitem_int, workgroup, ::Type{K}, backend, arch) where {Nitem,K}
    ker_fn = get_byte_sort_kernel(Val(Nitem))
    Base.invokelatest(_sort1d_impl_byte!, ker_fn,
                      by, uint_map, dst, src, tmp,
                      Nitem_int, workgroup, K, backend, arch)
end

function _dispatch_onesweep!(::Val{Nitem}, ::Val{Nchunks},
        by, uint_map, dst, src, tmp,
        Nitem_int, Nchunks_int, workgroup, npasses, ::Type{K}, backend, arch) where {Nitem,Nchunks,K}
    ker_fn = get_radix_kernel(Val(Nitem), Val(Nchunks))
    Base.invokelatest(_sort1d_impl!, ker_fn,
                      by, uint_map, dst, src, tmp,
                      Nitem_int, Nchunks_int, workgroup, npasses, K, backend, arch)
end


# ============================================================================
# Core implementation
# ============================================================================

# Stage 1+2 are shared across the byte-sort fast path and the onesweep
# multi-pass path. Mutates `hist` in place.
@inline function _sort_histograms!(
        by, uint_map, src, hist,
        Nitem::Int, Nbuckets::Int, warpsz::Int, wpb::Int, npasses::Int,
        workgroup::Int, backend)
    n = length(src)
    fill!(hist, UInt32(0))
    bucket_histogram_kernel!(backend, workgroup,
                              cld(cld(n, Nitem), wpb * warpsz) * workgroup)(
        hist, src, by, uint_map,
        Val(Nitem), Val(Nbuckets), Val(warpsz), Val(npasses),
    )
    scan_histogram_kernel!(backend, Nbuckets, Nbuckets * npasses)(hist, Val(Nbuckets))
    return nothing
end


# --- Byte-sort path (npasses == 1) -------------------------------------------

# Nothing-dispatch: allocate buffer then forward.
function _sort1d_impl_byte!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    ::Nothing,
    Nitem::Int, workgroup::Int, ::Type{K}, backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    tmp = get_allocation(Sort1D, src;
                         Nitem, Nchunks=1, workgroup, arch, by, uint_map)
    _sort1d_impl_byte!(ker, by, uint_map, dst, src, tmp,
                       Nitem, workgroup, K, backend, arch)
end

function _sort1d_impl_byte!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    tmp::KernelBuffer,
    Nitem::Int, workgroup::Int, ::Type{K}, backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    n = length(src)
    Nbuckets = 256
    warpsz = get_warpsize(arch)
    wpb = workgroup ÷ warpsz

    hist = tmp.arrays.hist

    _sort_histograms!(by, uint_map, src, hist,
                      Nitem, Nbuckets, warpsz, wpb, 1, workgroup, backend)

    # Single-byte path: skip onesweep entirely. `hist[:, 1]` (after scan)
    # is the global exclusive prefix; the kernel mutates it into running
    # offsets via per-block atomic-add.
    byte_block_size = workgroup * Nitem
    byte_nblocks = cld(n, byte_block_size)
    global_counter = @view hist[:, 1]
    inst = ker(backend, workgroup, byte_nblocks * workgroup)
    inst(dst, src, by, uint_map, global_counter, Val(Nbuckets), Val(warpsz))
    return dst
end


# --- Onesweep path (npasses ≥ 2) ---------------------------------------------

# Nothing-dispatch: allocate buffer then forward.
function _sort1d_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    ::Nothing,
    Nitem::Int, Nchunks::Int,
    workgroup::Int,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    tmp = get_allocation(Sort1D, src;
                         Nitem, Nchunks, workgroup, arch, by, uint_map)
    _sort1d_impl!(ker, by, uint_map, dst, src, tmp,
                  Nitem, Nchunks, workgroup, npasses, K, backend, arch)
end

function _sort1d_impl!(
    ker::KER, by::F, uint_map::M,
    dst::DS, src::AT,
    tmp::KernelBuffer,
    Nitem::Int, Nchunks::Int,
    workgroup::Int,
    npasses::Int, ::Type{K},
    backend, arch,
) where {KER,F,M,DS<:AbstractArray,AT<:AbstractArray,K}
    n = length(src)
    Nbuckets = 256
    warpsz = get_warpsize(arch)
    wpb = workgroup ÷ warpsz
    block_size_max = workgroup * Nchunks * Nitem
    nblocks = cld(n, block_size_max)

    hist     = tmp.arrays.hist
    partial1 = tmp.arrays.partial1
    partial2 = tmp.arrays.partial2
    flag     = tmp.arrays.flag
    scratch  = tmp.arrays.scratch

    _sort_histograms!(by, uint_map, src, hist,
                      Nitem, Nbuckets, warpsz, wpb, npasses, workgroup, backend)

    # Ping-pong between dst and scratch so the final write lands in dst
    # regardless of parity.
    inst = ker(backend, workgroup, nblocks * workgroup)

    if iseven(npasses)
        copyto!(dst, src)
        a, b = dst, scratch
    else
        copyto!(scratch, src)
        a, b = scratch, dst
    end

    for pass in 1:npasses
        fill!(partial1, UInt32(0))
        fill!(partial2, UInt32(0))
        fill!(flag, UInt8(0))
        excl_col = @view hist[:, pass]
        inst(b, a, by, uint_map, Int32(8 * (pass - 1)),
             excl_col, partial1, partial2, flag,
             Val(Nbuckets), Val(warpsz), Val(wpb))
        a, b = b, a
    end
    return dst
end
