"""
    mapreduce1d(f, op, src; kwargs...) -> GPU array or scalar
    mapreduce1d(f, op, srcs::NTuple; kwargs...) -> GPU array or scalar

GPU parallel map-reduce operation.

Applies `f` to each element, reduces with `op`, and optionally applies `g` to the final result.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to final result
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=256`: Workgroup size
- `blocks=100`: Number of blocks
- `FlagType=UInt8`: Synchronization flag type
- `to_cpu=true`: If true, return scalar; otherwise return 1-element GPU array

# Examples
```julia
# Sum of squares (returns GPU array)
x = CUDA.rand(Float32, 10_000)
result = mapreduce1d(x -> x^2, +, x)

# Sum of squares (returns scalar)
result = mapreduce1d(x -> x^2, +, x; to_cpu=true)

# Dot product of two arrays
x, y = CUDA.rand(Float32, 10_000), CUDA.rand(Float32, 10_000)
result = mapreduce1d((a, b) -> a * b, +, (x, y); to_cpu=true)
```

See also: [`KernelForge.mapreduce1d!`](@ref) for the in-place version.
"""
function mapreduce1d end

"""
    mapreduce1d!(f, op, dst, src; kwargs...)
    mapreduce1d!(f, op, dst, srcs::NTuple; kwargs...)

In-place GPU parallel map-reduce, writing result to `dst[1]`.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator  
- `dst`: Output array (result written to first element)
- `src` or `srcs`: Input GPU array(s)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to final result
- `tmp=nothing`: Pre-allocated temporary buffer
- `Nitem=nothing`: Items per thread (auto-selected if nothing)
- `workgroup=256`: Workgroup size
- `blocks=100`: Number of blocks
- `FlagType=UInt8`: Synchronization flag type

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
dst = CUDA.zeros(Float32, 1)

# Sum
mapreduce1d!(identity, +, dst, x)

# With pre-allocated temporary for repeated calls
tmp = KernelForge.get_allocation(mapreduce1d!, x)
for i in 1:100
    mapreduce1d!(identity, +, dst, x; tmp)
end
```

See also: [`KernelForge.mapreduce1d`](@ref) for the allocating version.
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
        return 1 # even for Float32
    end
end

const DEFAULT_MAPREDUCE_CONFIG = (workgroup=256, blocks=100)

# ============================================================================
# Temporary buffer allocation
# ============================================================================

"""
    get_allocation(::typeof(mapreduce1d!), src; blocks=100, eltype=nothing, FlagType=UInt8)

Allocate temporary buffer for `mapreduce1d!`. Useful for repeated reductions.

# Arguments
- `src` or `srcs`: Input GPU array(s) (used for backend and default element type)

# Keyword Arguments
- `blocks=100`: Number of blocks (must match the `blocks` used in `mapreduce1d!`)
- `eltype=nothing`: Element type for intermediate values. If `nothing`, defaults to 
  the element type of `src`. For proper type inference, pass `promote_op(f, T, ...)`.
- `FlagType=UInt8`: Synchronization flag type

# Examples
```julia
x = CUDA.rand(Float32, 10_000)
tmp = KernelForge.get_allocation(mapreduce1d!, x)
dst = CUDA.zeros(Float32, 1)

for i in 1:100
    mapreduce1d!(identity, +, dst, x; tmp)
end
```
"""
function get_allocation(
    fn::typeof(mapreduce1d!),
    src::AbstractArray{T};
    blocks::Integer=DEFAULT_MAPREDUCE_CONFIG.blocks,
    eltype::Type,
    FlagType::Type{FT}=UInt8
) where {T,FT}
    return get_allocation(fn, (src,); blocks, eltype, FlagType)
end

function get_allocation(
    fn::typeof(mapreduce1d!),
    srcs::NTuple{U,AbstractArray{T}};
    blocks::Integer=DEFAULT_MAPREDUCE_CONFIG.blocks,
    eltype::Type,
    FlagType::Type{FT}=UInt8
) where {U,T,FT}
    H = eltype
    backend = get_backend(srcs[1])
    sz = sum(get_partition_sizes(blocks, H, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Allocating API
# ============================================================================

# Single array
function mapreduce1d(
    f, op,
    src::AbstractArray{T};
    g=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_MAPREDUCE_CONFIG.workgroup,
    blocks::Int=DEFAULT_MAPREDUCE_CONFIG.blocks,
    FlagType::Type{FT}=UInt8,
    to_cpu::Bool=true
) where {T,FT}
    H = Base.promote_op(f, T)   # Intermediate type after f
    S = Base.promote_op(g, H)  # Final output type after g
    backend = get_backend(src)
    dst = KernelAbstractions.allocate(backend, S, 1)
    _Nitem = something(Nitem, default_nitem(mapreduce1d!, T))
    _tmp = something(tmp, get_allocation(mapreduce1d!, (src,); blocks, eltype=H, FlagType=FT))
    _mapreduce1d_impl!(f, op, g, dst, (src,), _Nitem, workgroup, blocks, _tmp, H, FT, length(src), backend)
    return to_cpu ? (@allowscalar dst[1]) : dst
end

# Tuple of arrays
function mapreduce1d(
    f::F, op::O,
    srcs::NTuple{U,AbstractArray{T}};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_MAPREDUCE_CONFIG.workgroup,
    blocks::Int=DEFAULT_MAPREDUCE_CONFIG.blocks,
    FlagType::Type{FT}=UInt8,
    to_cpu::Bool=true
) where {U,T,F<:Function,O<:Function,G<:Function,FT}
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)  # Intermediate type after f
    S = Base.promote_op(g, H)  # Final output type after g
    backend = get_backend(srcs[1])
    dst = KernelAbstractions.allocate(backend, S, 1)
    _Nitem = something(Nitem, default_nitem(mapreduce1d!, T))
    _tmp = something(tmp, get_allocation(mapreduce1d!, srcs; blocks, eltype=H, FlagType=FT))
    _mapreduce1d_impl!(f, op, g, dst, srcs, _Nitem, workgroup, blocks, _tmp, H, FT, length(srcs[1]), backend)
    return to_cpu ? (@allowscalar dst[1]) : dst
end

# ============================================================================
# In-place API
# ============================================================================

# Single array convenience wrapper
function mapreduce1d!(
    f, op,
    dst::AbstractArray{S},
    src::AbstractArray{T};
    g=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_MAPREDUCE_CONFIG.workgroup,
    blocks::Int=DEFAULT_MAPREDUCE_CONFIG.blocks,
    FlagType::Type{FT}=UInt8
) where {S,T,FT}
    return mapreduce1d!(f, op, dst, (src,); g, tmp, Nitem, workgroup, blocks, FlagType)
end

# Main in-place entry point
function mapreduce1d!(
    f::F, op::O,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    workgroup::Int=DEFAULT_MAPREDUCE_CONFIG.workgroup,
    blocks::Int=DEFAULT_MAPREDUCE_CONFIG.blocks,
    FlagType::Type{FT}=UInt8
) where {U,S,T,F<:Function,O<:Function,G<:Function,FT}
    n = length(srcs[1])
    backend = get_backend(srcs[1])

    # Correctly compute H based on the number of input arrays
    H = Base.promote_op(f, ntuple(_ -> T, Val(U))...)

    _Nitem = something(Nitem, default_nitem(mapreduce1d!, T))
    _tmp = something(tmp, get_allocation(mapreduce1d!, srcs; blocks, eltype=H, FlagType=FT))

    _mapreduce1d_impl!(f, op, g, dst, srcs, _Nitem, workgroup, blocks, _tmp, H, FT, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

function _mapreduce1d_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    Nitem::Int,
    workgroup::Int,
    blocks::Int,
    tmp::AbstractArray{UInt8},
    ::Type{H},
    ::Type{FT},
    n::Int,
    backend
) where {U,S,T,F,O,G,H,FT}
    # Adjust workgroup and ndrange to fit problem size
    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max(fld(n, workgroup) * workgroup, 1))

    # Ensure ndrange * Nitem ≤ n
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    # Partition temporary buffer
    partial, flag = partition(tmp, blocks, H, FT)

    # Initialize flags and select target value
    if FT === UInt8
        fill!(flag, 0x00)
        targetflag = 0x01
    else
        targetflag = rand(FT)
    end

    mapreduce1d_kernel!(backend, workgroup, ndrange)(
        f, op, dst, srcs, g, Val(Nitem), partial, flag, targetflag
    )
end