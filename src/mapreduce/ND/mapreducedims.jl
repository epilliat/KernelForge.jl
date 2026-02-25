"""
    mapreducedims(f, op, src, dims; kwargs...) -> GPU array

GPU parallel map-reduce over specified dimensions.

Applies `f` to each element, reduces along `dims` with `op`, and optionally
applies `g` to each final element.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `src`: Input GPU array
- `dims`: Dimension(s) to reduce over (Int or tuple of Ints)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to each result element
- `workgroup=256`: Workgroup size

# Examples
```julia
# Sum along rows (reduce dim 1)
x = CUDA.rand(Float32, 128, 64)
result = mapreducedims(identity, +, x, 1)   # shape: (1, 64)

# Sum of squares along columns (reduce dim 2)
result = mapreducedims(x -> x^2, +, x, 2)  # shape: (128, 1)

# Reduce multiple dimensions
x = CUDA.rand(Float32, 4, 8, 16)
result = mapreducedims(identity, +, x, (1, 3))  # shape: (1, 8, 1)
```

See also: [`KernelForge.mapreducedims!`](@ref) for the in-place version.
"""
function mapreducedims end

"""
    mapreducedims!(f, op, dst, src, dims; kwargs...)

In-place GPU parallel map-reduce over specified dimensions, writing result to `dst`.

# Arguments
- `f`: Map function applied to each element
- `op`: Associative binary reduction operator
- `dst`: Output array (must have size 1 along each reduced dimension)
- `src`: Input GPU array
- `dims`: Dimension(s) to reduce over (Int or tuple of Ints)

# Keyword Arguments
- `g=identity`: Post-reduction transformation applied to each result element
- `workgroup=256`: Workgroup size

# Examples
```julia
x = CUDA.rand(Float32, 128, 64)
dst = CUDA.zeros(Float32, 1, 64)

# Sum along dim 1
mapreducedims!(identity, +, dst, x, 1)

# Sum of squares along dim 2 with pre-allocated dst
dst2 = CUDA.zeros(Float32, 128, 1)
mapreducedims!(x -> x^2, +, dst2, x, 2)
```

See also: [`KernelForge.mapreducedims`](@ref) for the allocating version.
"""
function mapreducedims! end

# Compute output size: size 1 along reduced dims, original size elsewhere
function _output_size(src_size::NTuple{N,Int}, reduce_dims) where {N}
    ntuple(Val(N)) do d
        d in reduce_dims ? 1 : src_size[d]
    end
end

# Build iteration ranges for the reduced dimensions
@inline function _reduce_iters(src_size::NTuple{N,Int}, reduce_dims) where {N}
    ntuple(length(reduce_dims)) do i
        1:src_size[reduce_dims[i]]
    end
end

# ============================================================================
# Core implementation
# ============================================================================

function _mapreducedims_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray,
    src::AbstractArray{T,Ndims},
    reduce_dims::NTuple{R,Int},
    workgroup::Int,
    backend
) where {F,O,G,T,Ndims,R}
    iters = _reduce_iters(size(src), reduce_dims)
    keep_size = _output_size(size(src), reduce_dims)
    ndrange = prod(keep_size)

    dim_map = ntuple(Ndims) do d
        pos = Base.findfirst(==(d), reduce_dims)
        pos === nothing ? 0 : pos
    end

    mapreducedims_kernel!(backend, workgroup, ndrange)(
        f, op, dst, src, g,
        Val(reduce_dims), Val(iters), Val(keep_size), Val(dim_map)
    )
end

# ============================================================================
# Allocating API
# ============================================================================

function mapreducedims(
    f::F, op::O,
    src::AbstractArray{T},
    dims;
    g::G=identity,
    workgroup::Int=DEFAULT_WORKGROUP
) where {T,F<:Function,O<:Function,G<:Function}
    nd = ndims(src)
    reduce_dims = _normalize_dims(dims, nd)

    H = Base.promote_op(f, T)
    S = Base.promote_op(g, H)

    backend = get_backend(src)
    out_size = _output_size(size(src), reduce_dims)
    dst = KernelAbstractions.allocate(backend, S, out_size)

    _mapreducedims_impl!(f, op, g, dst, src, reduce_dims, workgroup, backend)
    return dst
end

# ============================================================================
# In-place API
# ============================================================================

function mapreducedims!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractArray{T},
    dims;
    g::G=identity,
    workgroup::Int=DEFAULT_WORKGROUP
) where {S,T,F<:Function,O<:Function,G<:Function}
    nd = ndims(src)
    reduce_dims = _normalize_dims(dims, nd)
    backend = get_backend(src)
    _mapreducedims_impl!(f, op, g, dst, src, reduce_dims, workgroup, backend)
    return dst
end