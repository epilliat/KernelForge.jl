"""
    mapreduce1d!(f, op, dst, src; kwargs...)
    mapreduce1d!(f, op, dst, srcs::NTuple; kwargs...)

GPU parallel map-reduce operation using decoupled lookback algorithm.

# Keyword Arguments
- `g=identity`: Optional post-reduction transformation
- `tmp=nothing`: Pre-allocated temporary buffer (from `_get_allocation`)
- `Nitem=nothing`: Number of items per thread (auto-selected based on element size if nothing)
- `config=nothing`: Launch configuration as `(workgroup=W, blocks=B)` NamedTuple
- `FlagType=UInt8`: Type for synchronization flags
"""
function mapreduce1d! end

# ============================================================================
# Configuration helpers
# ============================================================================

@inline function default_nitem(::typeof(mapreduce1d!), ::Type{T}) where {T}
    if sizeof(T) == 1
        return 8
    elseif sizeof(T) == 2
        return 4
    else
        return 1
    end
end

const DEFAULT_MAPREDUCE_CONFIG = (workgroup=256, blocks=100)

# Main public entry point
function get_allocation(
    fn::typeof(mapreduce1d!),
    srcs::NTuple{U,AbstractGPUArray{T}};
    blocks::Integer=DEFAULT_MAPREDUCE_CONFIG.blocks,
    eltype::Union{Type,Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {U,T,FT}
    H = something(eltype, T)
    return _get_allocation(fn, srcs, blocks, H, FT)
end

# Core implementation (positional args)
function _get_allocation(
    ::typeof(mapreduce1d!),
    srcs::NTuple{U,AbstractGPUArray{T}},
    blocks::Integer,
    ::Type{H},
    ::Type{FT}
) where {U,T,H,FT}
    backend = get_backend(srcs[1])
    sz = sum(get_partition_sizes(blocks, H, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Public API (kwargs wrappers)
# ============================================================================

# Single array convenience wrapper
function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    g=identity,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,FT}
    return mapreduce1d!(f, op, dst, (src,); g, tmp, Nitem, config, FlagType)
end

# Main public entry point
function mapreduce1d!(
    f::F, op::O,
    dst::AbstractGPUArray{S},
    srcs::NTuple{U,AbstractGPUArray{T}};
    g::G=identity,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {U,S,T,F<:Function,O<:Function,G<:Function,FT}
    n = length(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, T)

    # Resolve defaults
    _Nitem = something(Nitem, default_nitem(mapreduce1d!, T))
    _config = something(config, DEFAULT_MAPREDUCE_CONFIG)

    _mapreduce1d_impl!(f, op, g, dst, srcs, _Nitem, _config, tmp, H, FT, n, backend)
end

# ============================================================================
# Core implementation (positional args for type stability)
# ============================================================================

function _mapreduce1d_impl!(
    f::F, op::O, g::G,
    dst::AbstractGPUArray{S},
    srcs::NTuple{U,AbstractGPUArray{T}},
    Nitem::Int,
    config::NamedTuple{(:workgroup, :blocks)},
    tmp::Union{AbstractGPUArray{UInt8},Nothing},
    ::Type{H},
    ::Type{FT},
    n::Int,
    backend
) where {U,S,T,F,O,G,H,FT}
    workgroup, blocks = config.workgroup, config.blocks

    # Adjust workgroup and ndrange to fit problem size
    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max(fld(n, workgroup) * workgroup, 1))

    # Ensure ndrange * Nitem ≤ n; reduce Nitem if necessary (power of two for alignment)
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    # Allocate temporaries if not provided
    _tmp = something(tmp, _get_allocation(mapreduce1d!, srcs, blocks, H, FT))
    partial, flag = partition(_tmp, blocks, H, FT)

    # Initialize flags and select target value
    # For UInt8: deterministic 0→1 transition
    # For larger types: random target to avoid ABA problem with concurrent blocks
    if FT === UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FT)
    end

    mapreduce1d_kernel!(backend, workgroup, ndrange)(
        f, op, dst, srcs, g, Val(Nitem), partial, flag, targetflag
    )
end