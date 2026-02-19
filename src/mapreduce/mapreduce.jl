# ============================================================================
# mapreduce.jl - Unified GPU mapreduce with dimension support
# ============================================================================

"""
    mapreduce(f, op, src::AbstractArray; dims=nothing, kwargs...) -> scalar or GPU array

GPU parallel map-reduce operation with optional dimension reduction.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `src`: Input GPU array

# Keyword Arguments
- `dims=nothing`: Dimensions to reduce over. Options:
  - `nothing` or `:`: Reduce over all dimensions → scalar (controlled by `to_cpu`)
  - `Int` or `Tuple{Int...}`: Reduce over those dims → GPU array
- `g=identity`: Post-reduction transformation
- `to_cpu=true`: Only applies when `dims=nothing`. If true, return scalar; otherwise 1-element GPU array.
- Additional kwargs passed to underlying implementations

# Fast paths
- Full reduction (`dims=nothing`) → `mapreduce1d`
- All dims explicit → `mapreduce1d` (returns GPU array)
- Contiguous leading dims `(1,...,k)` → reshape, `mapreduce2d` on dim 1, reshape back
- Contiguous trailing dims `(k,...,n)` → reshape, `mapreduce2d` on dim 2, reshape back
- Both leading and trailing contiguous blocks → two `mapreduce2d` passes
- General dims → `mapreduce_dims` fallback

# Examples
```julia
A = CUDA.rand(Float32, 100, 50, 20)

# Full reduction → scalar
total = mapreduce(identity, +, A)

# Reduce along dim 1: (100, 50, 20) -> (1, 50, 20)
col_sums = mapreduce(identity, +, A; dims=1)

# Reduce along dims (1,2): (100, 50, 20) -> (1, 1, 20)
plane_sums = mapreduce(identity, +, A; dims=(1,2))

# Reduce along last dim: (100, 50, 20) -> (100, 50, 1)
depth_sums = mapreduce(identity, +, A; dims=3)
```

See also: [`KernelForge.mapreduce!`](@ref), [`mapreduce1d`](@ref), [`mapreduce2d`](@ref), [`mapreduce_dims`](@ref)
"""
function mapreduce(
    f::F, op::O,
    src::AbstractArray{T};
    dims=nothing,
    g::G=identity,
    to_cpu::Bool=true,
    kwargs...
) where {T,F<:Function,O<:Function,G<:Function}

    dims === Colon() && (dims = nothing)

    # --- Full reduction: to_cpu controls scalar vs 1-element GPU array ---
    dims === nothing && return mapreduce1d(f, op, src; g, to_cpu, kwargs...)

    nd = ndims(src)
    dims_tuple = _normalize_dims(dims, nd)
    _validate_dims(dims_tuple, nd)

    # --- All dims explicit: always return GPU array (Base-compatible) ---
    length(dims_tuple) == nd && return mapreduce1d(f, op, src; g, to_cpu=false, kwargs...)

    src_size = size(src)
    leading, trailing = _classify_dims(dims_tuple, nd)

    # --- Single contiguous block from start ---
    if leading && !trailing
        k = length(dims_tuple)
        reduce_size = prod(src_size[1:k])
        keep_size = src_size[k+1:nd]
        src_2d = reshape(src, reduce_size, prod(keep_size))
        result_flat = mapreduce2d(f, op, src_2d, 1; g, kwargs...)
        out_shape = ntuple(i -> i <= k ? 1 : src_size[i], nd)
        return reshape(result_flat, out_shape)
    end

    # --- Single contiguous block from end ---
    if trailing && !leading
        k = length(dims_tuple)
        first_kept = dims_tuple[1] - 1
        keep_size = src_size[1:first_kept]
        reduce_size = prod(src_size[first_kept+1:nd])
        src_2d = reshape(src, prod(keep_size), reduce_size)
        result_flat = mapreduce2d(f, op, src_2d, 2; g, kwargs...)
        out_shape = ntuple(i -> i >= dims_tuple[1] ? 1 : src_size[i], nd)
        return reshape(result_flat, out_shape)
    end

    # --- Two contiguous blocks: leading (1..k) and trailing (j..nd) ---
    if leading && trailing
        lead_end = findlast(i -> dims_tuple[i] == i, 1:length(dims_tuple))
        lead_dims = dims_tuple[1:lead_end]
        tail_dims = dims_tuple[lead_end+1:end]

        lead_k = length(lead_dims)
        tail_start = tail_dims[1]
        lead_size = prod(src_size[1:lead_k])
        mid_size = prod(src_size[lead_k+1:tail_start-1])
        tail_size = prod(src_size[tail_start:nd])

        # First pass: reduce trailing block (no g yet)
        src_tmp_2d = reshape(src, lead_size * mid_size, tail_size)
        result_tmp = mapreduce2d(f, op, src_tmp_2d, 2; kwargs...)

        # Second pass: reduce leading block (apply g here)
        result_2d = reshape(result_tmp, lead_size, mid_size)
        result_flat = mapreduce2d(identity, op, result_2d, 1; g, kwargs...)

        out_shape = ntuple(nd) do i
            i in dims_tuple ? 1 : src_size[i]
        end
        return reshape(result_flat, out_shape)
    end

    # --- General fallback ---
    return mapreduce_dims(f, op, src, dims_tuple; g, kwargs...)
end

"""
    mapreduce!(f, op, dst, src; dims=nothing, kwargs...)

In-place GPU parallel map-reduce with dimension support.

# Examples
```julia
A = CUDA.rand(Float32, 100, 50)
dst = CUDA.zeros(Float32, 1, 50)
mapreduce!(identity, +, dst, A; dims=1)
```

See also: [`KernelForge.mapreduce`](@ref)
"""
function mapreduce!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractArray{T};
    dims=nothing,
    g::G=identity,
    kwargs...
) where {S,T,F<:Function,O<:Function,G<:Function}

    dims === Colon() && (dims = nothing)

    dims === nothing && return mapreduce1d!(f, op, dst, src; g, kwargs...)

    nd = ndims(src)
    dims_tuple = _normalize_dims(dims, nd)
    _validate_dims(dims_tuple, nd)

    length(dims_tuple) == nd && return mapreduce1d!(f, op, dst, src; g, kwargs...)

    src_size = size(src)
    leading, trailing = _classify_dims(dims_tuple, nd)

    if leading && !trailing
        k = length(dims_tuple)
        reduce_size = prod(src_size[1:k])
        keep_size = prod(src_size[k+1:nd])
        src_2d = reshape(src, reduce_size, keep_size)
        dst_flat = reshape(dst, keep_size)
        return mapreduce2d!(f, op, dst_flat, src_2d, 1; g, kwargs...)
    end

    if trailing && !leading
        first_kept = dims_tuple[1] - 1
        keep_size = prod(src_size[1:first_kept])
        reduce_size = prod(src_size[first_kept+1:nd])
        src_2d = reshape(src, keep_size, reduce_size)
        dst_flat = reshape(dst, keep_size)
        return mapreduce2d!(f, op, dst_flat, src_2d, 2; g, kwargs...)
    end

    if leading && trailing
        lead_end = findlast(i -> dims_tuple[i] == i, 1:length(dims_tuple))
        lead_dims = dims_tuple[1:lead_end]
        tail_dims = dims_tuple[lead_end+1:end]

        lead_k = length(lead_dims)
        tail_start = tail_dims[1]
        lead_size = prod(src_size[1:lead_k])
        mid_size = prod(src_size[lead_k+1:tail_start-1])
        tail_size = prod(src_size[tail_start:nd])

        backend = get_backend(src)
        H = Base.promote_op(f, T)
        tmp = KernelAbstractions.allocate(backend, H, lead_size * mid_size)

        src_tmp_2d = reshape(src, lead_size * mid_size, tail_size)
        mapreduce2d!(f, op, tmp, src_tmp_2d, 2; kwargs...)

        result_2d = reshape(tmp, lead_size, mid_size)
        dst_flat = reshape(dst, mid_size)
        mapreduce2d!(identity, op, dst_flat, result_2d, 1; g, kwargs...)
        return dst
    end

    return mapreduce_dims!(f, op, dst, src, dims_tuple; g, kwargs...)
end

# ============================================================================
# Multi-array variant (full reduction only)
# ============================================================================

"""
    mapreduce(f, op, srcs::NTuple; kwargs...)

Multi-array mapreduce. Only supports full reduction (`dims=nothing`).
"""
function mapreduce(
    f::F, op::O,
    srcs::NTuple{U,AbstractArray{T}};
    dims=nothing,
    g::G=identity,
    to_cpu::Bool=true,
    kwargs...
) where {U,T,F<:Function,O<:Function,G<:Function}
    if dims !== nothing && dims !== Colon()
        throw(ArgumentError("Multi-array mapreduce only supports full reduction (dims=nothing)"))
    end
    return mapreduce1d(f, op, srcs; g, to_cpu, kwargs...)
end

# ============================================================================
# Helpers
# ============================================================================

"""
    _validate_dims(dims::NTuple, ndim::Int)

Check bounds and no duplicates. Contiguity is handled by dispatch.
"""
function _validate_dims(dims::NTuple{N,Int}, ndim::Int) where {N}
    for d in dims
        1 <= d <= ndim || throw(ArgumentError("dimension $d out of range for $ndim-dimensional array"))
    end
    length(unique(dims)) == length(dims) ||
        throw(ArgumentError("duplicate dimensions in dims=$dims"))
    return nothing
end

"""
    _classify_dims(dims, nd) -> (leading::Bool, trailing::Bool)

Classify whether dims form a contiguous block at the start, end, or both.
Both can be true only when dims split into two separate contiguous blocks
(one at the start, one at the end).
"""
function _classify_dims(dims::NTuple{N,Int}, nd::Int) where {N}
    lead_end = 0
    for k in 1:N
        dims[k] == k ? (lead_end = k) : break
    end
    leading = lead_end > 0 && dims[1] == 1

    tail_start = N + 1
    for k in N:-1:1
        dims[k] == nd - (N - k) ? (tail_start = k) : break
    end
    trailing = tail_start <= N && dims[end] == nd

    # Single contiguous block touching both ends → treat as leading only
    # (full reduction already handled upstream)
    if leading && trailing
        is_one_block = all(i -> dims[i+1] == dims[i] + 1, 1:N-1)
        is_one_block && return (true, false)
    end

    return (leading, trailing)
end