# ============================================================================
# mapreduce.jl - Unified GPU mapreduce with dimension support
# ============================================================================

"""
    mapreduce(f, op, src::AbstractGPUArray; dims=nothing, kwargs...) -> Array or scalar

GPU parallel map-reduce operation with optional dimension reduction.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `src`: Input GPU array

# Keyword Arguments
- `dims=nothing`: Dimensions to reduce over. Options:
  - `nothing` or `:`: Reduce over all dimensions (returns scalar or 1-element array)
  - `Int`: Reduce over single dimension
  - `Tuple{Int...}`: Reduce over multiple dimensions (must be contiguous from start or end)
- `g=identity`: Post-reduction transformation
- `init=nothing`: Initial value (currently unused, for API compatibility)
- `to_cpu=false`: If true and `dims=nothing`, return scalar; otherwise return GPU array
- Additional kwargs passed to underlying implementations

# Dimension Constraints
The `dims` argument must specify contiguous dimensions from either:
- The beginning: `(1,)`, `(1,2)`, `(1,2,3)`, etc.
- The end: `(n-1,n)`, `(n,)`, etc. for an n-dimensional array

# Examples
```julia
A = CUDA.rand(Float32, 100, 50, 20)

# Full reduction (all dimensions)
total = mapreduce(identity, +, A; to_cpu=true)

# Reduce along dim 1: (100, 50, 20) -> (50, 20)
col_sums = mapreduce(identity, +, A; dims=1)

# Reduce along dims (1,2): (100, 50, 20) -> (20,)
plane_sums = mapreduce(identity, +, A; dims=(1,2))

# Reduce along last dim: (100, 50, 20) -> (100, 50)
depth_sums = mapreduce(identity, +, A; dims=3)

# Reduce along last two dims: (100, 50, 20) -> (100,)
slice_sums = mapreduce(identity, +, A; dims=(2,3))
```

See also: [`Luma.mapreduce!`](@ref), [`mapreduce1d`](@ref), [`mapreduce2d`](@ref)
"""
function mapreduce(
    f::F, op::O,
    src::AbstractGPUArray{T};
    dims=nothing,
    g::G=identity,
    init=nothing,
    to_cpu::Bool=false,
    kwargs...
) where {T,F<:Function,O<:Function,G<:Function}

    # Handle dims=: as dims=nothing (full reduction)
    if dims === Colon()
        dims = nothing
    end

    # Full reduction case
    if dims === nothing
        return mapreduce1d(f, op, src; g, to_cpu, kwargs...)
    end

    nd = ndims(src)

    # Fast path for 1D arrays
    if nd == 1
        dims == 1 || dims == (1,) || throw(ArgumentError("dimension $dims out of range for 1-dimensional array"))
        return mapreduce1d(f, op, src; g, to_cpu, kwargs...)
    end

    # Fast path for 2D arrays
    if nd == 2
        if dims == 1 || dims == (1,)
            return mapreduce2d(f, op, src, 1; g, kwargs...)
        elseif dims == 2 || dims == (2,)
            return mapreduce2d(f, op, src, 2; g, kwargs...)
        elseif dims == (1, 2) || dims == (2, 1)
            return mapreduce1d(f, op, src; g, to_cpu, kwargs...)
        else
            throw(ArgumentError("invalid dims=$dims for 2-dimensional array"))
        end
    end

    # General case: normalize dims to sorted tuple
    dims_tuple = _normalize_dims(dims, nd)

    # Validate and determine reduction strategy
    _validate_dims(dims_tuple, nd)

    return _mapreduce_dims(f, op, g, src, dims_tuple; to_cpu, kwargs...)
end

"""
    mapreduce!(f, op, dst, src; dims=nothing, kwargs...)

In-place GPU parallel map-reduce with dimension support.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `dst`: Output array
- `src`: Input GPU array

# Keyword Arguments
- `dims=nothing`: Dimensions to reduce over (see `mapreduce` for details)
- `g=identity`: Post-reduction transformation
- Additional kwargs passed to underlying implementations

# Examples
```julia
A = CUDA.rand(Float32, 100, 50)
col_sums = CUDA.zeros(Float32, 50)
row_sums = CUDA.zeros(Float32, 100)

# Column sums (reduce dim 1)
mapreduce!(identity, +, col_sums, A; dims=1)

# Row sums (reduce dim 2)
mapreduce!(identity, +, row_sums, A; dims=2)
```

See also: [`Luma.mapreduce`](@ref)
"""
function mapreduce!(
    f::F, op::O,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    dims=nothing,
    g::G=identity,
    kwargs...
) where {S,T,F<:Function,O<:Function,G<:Function}

    # Handle dims=: as dims=nothing
    if dims === Colon()
        dims = nothing
    end

    # Full reduction case
    if dims === nothing
        return mapreduce1d!(f, op, dst, src; g, kwargs...)
    end

    nd = ndims(src)

    # Fast path for 1D arrays
    if nd == 1
        dims == 1 || dims == (1,) || throw(ArgumentError("dimension $dims out of range for 1-dimensional array"))
        return mapreduce1d!(f, op, dst, src; g, kwargs...)
    end

    # Fast path for 2D arrays
    if nd == 2
        if dims == 1 || dims == (1,)
            return mapreduce2d!(f, op, dst, src, 1; g, kwargs...)
        elseif dims == 2 || dims == (2,)
            return mapreduce2d!(f, op, dst, src, 2; g, kwargs...)
        elseif dims == (1, 2) || dims == (2, 1)
            return mapreduce1d!(f, op, dst, src; g, kwargs...)
        else
            throw(ArgumentError("invalid dims=$dims for 2-dimensional array"))
        end
    end

    # General case: normalize dims to sorted tuple
    dims_tuple = _normalize_dims(dims, nd)

    # Validate dimensions
    _validate_dims(dims_tuple, nd)

    return _mapreduce_dims!(f, op, g, dst, src, dims_tuple; kwargs...)
end

# ============================================================================
# Dimension normalization and validation
# ============================================================================

"""
    _normalize_dims(dims, ndim) -> NTuple{N,Int}

Convert dims specification to a sorted tuple of positive integers.
"""
function _normalize_dims(dims::Int, ndim::Int)
    d = dims < 0 ? ndim + dims + 1 : dims
    return (d,)
end

function _normalize_dims(dims::NTuple{N,Int}, ndim::Int) where {N}
    normalized = ntuple(N) do i
        d = dims[i]
        d < 0 ? ndim + d + 1 : d
    end
    return Tuple(sort(collect(normalized)))
end

function _normalize_dims(dims::AbstractVector{<:Integer}, ndim::Int)
    return _normalize_dims(Tuple(dims), ndim)
end

"""
    _validate_dims(dims::NTuple, ndim::Int)

Validate that dims are contiguous from either start or end.
Throws ArgumentError if invalid.
"""
function _validate_dims(dims::NTuple{N,Int}, ndim::Int) where {N}
    # assume ndim >= 2
    # Check bounds
    for d in dims
        if d < 1 || d > ndim
            throw(ArgumentError("dimension $d out of range for $ndim-dimensional array"))
        end
    end

    # Check for duplicates
    if length(unique(dims)) != length(dims)
        throw(ArgumentError("duplicate dimensions in dims=$dims"))
    end

    # Single dimension is always valid
    if N == 1
        if 1 < dims[1] < ndim
            throw(ArgumentError(
                "dims must be contiguous, got $dims. " *
                "Valid examples: (1,2,3) or (n-2,n-1,n) for n-dim array"
            ))
        else
            return nothing
        end
    end

    # Check contiguity
    sorted_dims = sort(collect(dims))
    is_contiguous = all(i -> sorted_dims[i+1] == sorted_dims[i] + 1, 1:length(sorted_dims)-1)
    @show sorted_dims
    if !is_contiguous
        throw(ArgumentError(
            "dims must be contiguous, got $dims. " *
            "Valid examples: (1,2,3) or (n-2,n-1,n) for n-dim array"
        ))
    end

    # Check if contiguous from start or end
    starts_at_1 = sorted_dims[1] == 1
    ends_at_n = sorted_dims[end] == ndim

    if !starts_at_1 && !ends_at_n
        throw(ArgumentError(
            "dims must be contiguous from start (1,2,...) or end (...,n-1,n), got $dims"
        ))
    end

    return nothing
end

# ============================================================================
# Core dimension reduction implementation
# ============================================================================

"""
    _mapreduce_dims(f, op, g, src, dims; kwargs...)

Internal implementation that reshapes array and dispatches to appropriate kernel.
"""
function _mapreduce_dims(
    f::F, op::O, g::G,
    src::AbstractGPUArray{T},
    dims::NTuple{N,Int};
    to_cpu::Bool=false,
    kwargs...
) where {T,F,O,G,N}

    ndim = ndims(src)
    src_size = size(src)
    sorted_dims = sort(collect(dims))

    # Determine if reducing from start or end
    reducing_from_start = sorted_dims[1] == 1

    # Calculate sizes for reshape
    if reducing_from_start
        # Reducing first k dimensions: reshape to (prod(first_k), remaining...)
        k = length(dims)
        reduce_size = prod(src_size[1:k])
        keep_size = prod(src_size[k+1:end])
        output_shape = src_size[k+1:end]
    else
        # Reducing last k dimensions: reshape to (remaining..., prod(last_k))
        k = length(dims)
        first_kept = sorted_dims[1] - 1
        keep_size = prod(src_size[1:first_kept])
        reduce_size = prod(src_size[first_kept+1:end])
        output_shape = src_size[1:first_kept]
    end

    # Handle edge case: output is scalar (all dims reduced)
    if isempty(output_shape)
        return mapreduce1d(f, op, src; g, to_cpu, kwargs...)
    end

    # Reshape to 2D matrix
    if reducing_from_start
        # Shape: (reduce_size, keep_size) - reduce along dim 1
        src_2d = reshape(src, reduce_size, keep_size)
        result_flat = mapreduce2d(f, op, src_2d, 1; g, kwargs...)
    else
        # Shape: (keep_size, reduce_size) - reduce along dim 2
        src_2d = reshape(src, keep_size, reduce_size)
        result_flat = mapreduce2d(f, op, src_2d, 2; g, kwargs...)
    end

    # Reshape result to output shape
    if length(output_shape) == 1
        return result_flat
    else
        return reshape(result_flat, output_shape)
    end
end

"""
    _mapreduce_dims!(f, op, g, dst, src, dims; kwargs...)

In-place version of dimension reduction.
"""
function _mapreduce_dims!(
    f::F, op::O, g::G,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T},
    dims::NTuple{N,Int};
    kwargs...
) where {S,T,F,O,G,N}

    ndim = ndims(src)
    src_size = size(src)
    sorted_dims = sort(collect(dims))

    # Determine if reducing from start or end
    reducing_from_start = sorted_dims[1] == 1

    # Calculate sizes for reshape
    if reducing_from_start
        k = length(dims)
        reduce_size = prod(src_size[1:k])
        keep_size = prod(src_size[k+1:end])
    else
        k = length(dims)
        first_kept = sorted_dims[1] - 1
        keep_size = prod(src_size[1:first_kept])
        reduce_size = prod(src_size[first_kept+1:end])
    end

    # Reshape destination to flat vector for the kernel
    dst_flat = reshape(dst, keep_size)

    # Reshape source and perform reduction
    if reducing_from_start
        src_2d = reshape(src, reduce_size, keep_size)
        mapreduce2d!(f, op, dst_flat, src_2d, 1; g, kwargs...)
    else
        src_2d = reshape(src, keep_size, reduce_size)
        mapreduce2d!(f, op, dst_flat, src_2d, 2; g, kwargs...)
    end

    return dst
end

# ============================================================================
# Convenience methods and Base overloads
# ============================================================================

# Support for multiple source arrays (like dot product)
"""
    mapreduce(f, op, srcs::NTuple{N,AbstractGPUArray}; dims=nothing, kwargs...)

Multi-array mapreduce. Only supports full reduction (dims=nothing).
"""
function mapreduce(
    f::F, op::O,
    srcs::NTuple{U,AbstractGPUArray{T}};
    dims=nothing,
    g::G=identity,
    to_cpu::Bool=false,
    kwargs...
) where {U,T,F<:Function,O<:Function,G<:Function}

    if dims !== nothing && dims !== Colon()
        throw(ArgumentError("Multi-array mapreduce only supports full reduction (dims=nothing)"))
    end

    return mapreduce1d(f, op, srcs; g, to_cpu, kwargs...)
end