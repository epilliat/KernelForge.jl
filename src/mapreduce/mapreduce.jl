# ============================================================================
# Public API: mapreduce!, mapreduce, reduce!
# ============================================================================

"""
    mapreduce!(f, op, dst, src; kwargs...)

In-place GPU map-reduce. Applies `f` to each element, then reduces with `op`.

# Keyword Arguments
- `g=identity`: Post-reduction transformation
- `dims=nothing`: Dimensions to reduce over (currently only full reduction supported)
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `config=nothing`: Launch configuration `(workgroup=W, blocks=B)`
- `FlagType=UInt8`: Synchronization flag type
"""
function mapreduce! end

"""
    mapreduce(f, op, src; kwargs...)

GPU map-reduce returning a new array (or scalar if `to_cpu=true`).

# Keyword Arguments
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `config=nothing`: Launch configuration `(workgroup=W, blocks=B)`
- `FlagType=UInt8`: Synchronization flag type
- `to_cpu=false`: If true, return scalar on CPU instead of 1-element GPU array
"""
function mapreduce end

"""
    reduce!(op, dst, src; kwargs...)

In-place GPU reduction (equivalent to `mapreduce!(identity, op, dst, src; ...)`).
"""
function reduce! end

# ----------------------------------------------------------------------------
# mapreduce! — Vector (1D) case
# ----------------------------------------------------------------------------

function mapreduce!(
    f, op,
    dst,
    src::AbstractGPUVector{T};
    g=identity,
    dims=nothing,
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {T,FT}
    # For vectors, only dims=1 or dims=(1,) makes sense
    if !isnothing(dims) && dims ∉ (1, (1,), Colon())
        throw(ArgumentError("For vectors, dims must be 1, (1,), or Colon(); got $dims"))
    end
    _mapreduce_impl!(f, op, g, dst, (src,), dims, tmp, Nitem, config, FT)
end

# ----------------------------------------------------------------------------
# mapreduce! — General N-dimensional case
# ----------------------------------------------------------------------------

function mapreduce!(
    f, op,
    dst,
    src::AbstractGPUArray{T};
    g=identity,
    dims=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {T,FT}
    _mapreduce_impl!(f, op, g, dst, (src,), dims, tmp, Nitem, config, FT)
end

# ----------------------------------------------------------------------------
# mapreduce — Allocating version
# ----------------------------------------------------------------------------

function mapreduce(
    f, op,
    src::AbstractGPUArray{T};
    g=identity,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8,
    to_cpu::Bool=false
) where {T,FT}
    S = Base.promote_op(g ∘ f, T)
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, S, 1)

    _mapreduce_impl!(f, op, g, dst, (src,), nothing, tmp, Nitem, config, FT)

    return to_cpu ? (@allowscalar dst[1]) : dst
end

# ----------------------------------------------------------------------------
# reduce! — Convenience wrapper
# ----------------------------------------------------------------------------

function reduce!(
    op,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,FT}
    _mapreduce_impl!(identity, op, identity, dst, (src,), nothing, tmp, Nitem, config, FT)
end

# ============================================================================
# Core implementation
# ============================================================================

function _mapreduce_impl!(
    f::F, op::O, g::G,
    dst,
    srcs::NTuple{U,AbstractGPUArray{T}},
    dims,
    tmp::Union{AbstractGPUArray{UInt8},Nothing},
    Nitem,
    config,
    ::Type{FT}
) where {U,T,F,O,G,FT}
    src = srcs[1]
    sz = size(src)

    # Determine which axes are trivial (size 1) vs non-trivial
    nonflat_axes = Tuple(i for (i, s) in enumerate(sz) if s > 1)
    flat_axes = Tuple(i for (i, s) in enumerate(sz) if s == 1)

    # Normalize dims
    _dims = _normalize_dims(dims, ndims(src))

    if isnothing(_dims) || nonflat_axes ⊆ _dims
        # Full reduction (or reducing all non-trivial dimensions)
        _mapreduce1d_impl!(
            f, op, g, dst, srcs,
            something(Nitem, default_nitem(mapreduce1d!, T)),
            something(config, DEFAULT_MAPREDUCE_CONFIG),
            tmp,
            Base.promote_op(f, T),
            FT,
            length(src),
            get_backend(src)
        )
        # Reshape dst to match expected output shape (all ones)
        # Note: caller may need to handle this reshape
    elseif flat_axes ⊆ _dims
        # Reducing only trivial dimensions → result is a copy
        copyto!(dst, src)
    else
        # Partial reduction along specific dimensions — not yet implemented
        throw(ArgumentError("Partial reduction along dims=$dims not yet supported"))
    end

    return dst
end

# Helper to normalize dims argument
@inline function _normalize_dims(dims::Nothing, ::Int)
    return nothing
end

@inline function _normalize_dims(dims::Colon, ndim::Int)
    return ntuple(identity, ndim)
end

@inline function _normalize_dims(dims::Int, ::Int)
    return (dims,)
end

@inline function _normalize_dims(dims::Tuple, ::Int)
    return dims
end