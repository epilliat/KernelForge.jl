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
  - `nothing` or `:`: Reduce over all dimensions → scalar
  - `Int` or `Tuple{Int...}`: Reduce over those dims → GPU array
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated buffer — a `MapReduceBuffer` (full reduction),
  `VecMatBuffer` (dim=1 reduction), or `MatVecBuffer` (dim=2 reduction)
- Additional kwargs passed to underlying implementations

# Fast paths
- Full reduction (`dims=nothing`) → `mapreduce1d`
- All dims explicit → `mapreduce_dims`
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
    kwargs...
) where {T,F<:Function,O<:Function,G<:Function}

    dims === Colon() && (dims = nothing)

    # --- Full reduction → scalar (to_cpu not exposed to user) ---
    dims === nothing && return mapreduce1d(f, op, src; g, to_cpu=true, kwargs...)

    nd = ndims(src)
    dims_tuple = _normalize_dims(dims, nd)
    _validate_dims(dims_tuple, nd)

    # --- All dims explicit: return GPU array with shape (1,...,1), Base-compatible ---
    length(dims_tuple) == nd && return mapreduce_dims(f, op, src, dims_tuple; g, kwargs...)

    src_size = size(src)
    lead_end, tail_start = _classify_dims(dims_tuple, nd)
    leading = lead_end > 0
    trailing = tail_start <= length(dims_tuple)

    # --- Single contiguous block from start ---
    if leading && !trailing
        reduce_size = prod(src_size[1:lead_end])
        keep_size = src_size[lead_end+1:nd]
        src_2d = reshape(src, reduce_size, prod(keep_size))
        result_flat = mapreduce2d(f, op, src_2d, 1; g, kwargs...)
        out_shape = ntuple(i -> i <= lead_end ? 1 : src_size[i], nd)
        return reshape(result_flat, out_shape)
    end

    # --- Single contiguous block from end ---
    if trailing && !leading
        tail_start_dim = dims_tuple[tail_start]
        keep_size = prod(src_size[1:tail_start_dim-1])
        reduce_size = prod(src_size[tail_start_dim:nd])
        src_2d = reshape(src, keep_size, reduce_size)
        result_flat = mapreduce2d(f, op, src_2d, 2; g, kwargs...)
        out_shape = ntuple(i -> i >= tail_start_dim ? 1 : src_size[i], nd)
        return reshape(result_flat, out_shape)
    end

    # --- Two contiguous blocks: leading (1..lead_end) and trailing (tail_start_dim..nd) ---
    if leading && trailing
        tail_start_dim = dims_tuple[tail_start]
        lead_size = prod(src_size[1:lead_end])
        mid_size = prod(src_size[lead_end+1:tail_start_dim-1])
        tail_size = prod(src_size[tail_start_dim:nd])

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

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `dst`: Output array
- `src`: Input GPU array

# Keyword Arguments
- `dims=nothing`: Dimensions to reduce over. Options:
  - `nothing` or `:`: Reduce over all dimensions → writes to `dst[1]`
  - `Int` or `Tuple{Int...}`: Reduce over those dims → writes to `dst`
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated buffer — a `MapReduceBuffer` (full reduction),
  `VecMatBuffer` (dim=1 reduction), or `MatVecBuffer` (dim=2 reduction)
- Additional kwargs passed to underlying implementations

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

    length(dims_tuple) == nd && return mapreduce_dims!(f, op, dst, src, dims_tuple; g, kwargs...)

    src_size = size(src)
    lead_end, tail_start = _classify_dims(dims_tuple, nd)
    leading = lead_end > 0
    trailing = tail_start <= length(dims_tuple)

    if leading && !trailing
        reduce_size = prod(src_size[1:lead_end])
        keep_size = prod(src_size[lead_end+1:nd])
        src_2d = reshape(src, reduce_size, keep_size)
        dst_flat = reshape(dst, keep_size)
        return mapreduce2d!(f, op, dst_flat, src_2d, 1; g, kwargs...)
    end

    if trailing && !leading
        tail_start_dim = dims_tuple[tail_start]
        keep_size = prod(src_size[1:tail_start_dim-1])
        reduce_size = prod(src_size[tail_start_dim:nd])
        src_2d = reshape(src, keep_size, reduce_size)
        dst_flat = reshape(dst, keep_size)
        return mapreduce2d!(f, op, dst_flat, src_2d, 2; g, kwargs...)
    end

    if leading && trailing
        tail_start_dim = dims_tuple[tail_start]
        lead_size = prod(src_size[1:lead_end])
        mid_size = prod(src_size[lead_end+1:tail_start_dim-1])
        tail_size = prod(src_size[tail_start_dim:nd])

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

# Keyword Arguments
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated `MapReduceBuffer`
- Additional kwargs passed to `mapreduce1d`
"""
function mapreduce(
    f::F, op::O,
    srcs::NTuple{U,AbstractArray{T}};
    dims=nothing,
    g::G=identity,
    kwargs...
) where {U,T,F<:Function,O<:Function,G<:Function}
    if dims !== nothing && dims !== Colon()
        throw(ArgumentError("Multi-array mapreduce only supports full reduction (dims=nothing)"))
    end
    return mapreduce1d(f, op, srcs; g, to_cpu=true, kwargs...)
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
    _classify_dims(dims, nd) -> (lead_end::Int, tail_start::Int)

Returns the index (within `dims`) where the leading contiguous block ends,
and the index where the trailing contiguous block starts.

- `lead_end == 0` means no leading block.
- `tail_start == length(dims) + 1` means no trailing block.

Both can be active only when dims split into two separate contiguous blocks
(one at the start, one at the end of the array dimensions).
Assumes `dims` is sorted.
"""
function _classify_dims(dims::NTuple{N,Int}, nd::Int) where {N}
    lead_end = 0
    for k in 1:N
        dims[k] == k ? (lead_end = k) : break
    end

    tail_start = N + 1
    for k in N:-1:1
        dims[k] == nd - (N - k) ? (tail_start = k) : break
    end

    # Single contiguous block touching both ends → treat as leading only
    if lead_end > 0 && tail_start <= N
        is_one_block = all(i -> dims[i+1] == dims[i] + 1, 1:N-1)
        is_one_block && return (lead_end, N + 1)
    end

    # Verify the two blocks together cover exactly all dims
    # If not (e.g. dims=(2,4,5) on 5D), neither block is valid → general fallback
    n_lead = lead_end
    n_tail = tail_start <= N ? (N - tail_start + 1) : 0
    if n_lead + n_tail != N
        return (0, N + 1)  # force general fallback
    end

    return (lead_end, tail_start)
end