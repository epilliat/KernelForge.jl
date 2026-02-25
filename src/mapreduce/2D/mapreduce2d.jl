"""
    mapreduce2d(f, op, src, dim; kwargs...) -> Vector

GPU parallel reduction along dimension `dim`.

- `dim=1`: Column-wise reduction (vertical), output length = number of columns
- `dim=2`: Row-wise reduction (horizontal), output length = number of rows

# Arguments
- `f`: Element-wise transformation
- `op`: Reduction operator
- `src`: Input matrix of size `(n, p)`
- `dim`: Dimension to reduce along (1 or 2)

# Keyword Arguments
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated `VecMatBuffer` (dim=1) or `MatVecBuffer` (dim=2)

For `dim=1` (column-wise):
- `Nitem=nothing`: Items per thread
- `Nthreads=nothing`: Threads per column reduction
- `workgroup=nothing`: Workgroup size
- `blocks=nothing`: Number of blocks

For `dim=2` (row-wise):
- `chunksz=nothing`: Chunk size for row processing
- `Nblocks=nothing`: Number of blocks per row
- `workgroup=nothing`: Workgroup size
- `blocks_row=nothing`: Blocks per row

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)

# Column sums (reduce along dim=1)
col_sums = mapreduce2d(identity, +, A, 1)

# Row maximums (reduce along dim=2)
row_maxs = mapreduce2d(identity, max, A, 2)

# Column means
col_means = mapreduce2d(identity, +, A, 1; g=x -> x / size(A, 1))

# Sum of squares per row
row_ss = mapreduce2d(abs2, +, A, 2)
```

See also: [`KernelForge.mapreduce2d!`](@ref) for the in-place version.
"""
function mapreduce2d(
    f::F, op::O,
    src::AbstractMatrix{T},
    dim::Int;
    g::G=identity,
    tmp::Union{VecMatBuffer,MatVecBuffer,Nothing}=nothing,
    kwargs...
) where {T,F<:Function,O<:Function,G<:Function}
    if dim == 1
        return vecmat(f, op, nothing, src; g, tmp, kwargs...)
    elseif dim == 2
        return matvec(f, op, src, nothing; g, tmp, kwargs...)
    else
        throw(ArgumentError("dim must be 1 or 2, got $dim"))
    end
end

"""
    mapreduce2d!(f, op, dst, src, dim; kwargs...)

In-place GPU parallel reduction along dimension `dim`.

- `dim=1`: Column-wise reduction (vertical), `dst` length = number of columns
- `dim=2`: Row-wise reduction (horizontal), `dst` length = number of rows

# Arguments
- `f`: Element-wise transformation
- `op`: Reduction operator
- `dst`: Output vector
- `src`: Input matrix of size `(n, p)`
- `dim`: Dimension to reduce along (1 or 2)

# Keyword Arguments
- `g=identity`: Post-reduction transformation
- `tmp=nothing`: Pre-allocated `VecMatBuffer` (dim=1) or `MatVecBuffer` (dim=2)

For `dim=1` (column-wise):
- `Nitem=nothing`: Items per thread
- `Nthreads=nothing`: Threads per column reduction
- `workgroup=nothing`: Workgroup size
- `blocks=nothing`: Number of blocks

For `dim=2` (row-wise):
- `chunksz=nothing`: Chunk size for row processing
- `Nblocks=nothing`: Number of blocks per row
- `workgroup=nothing`: Workgroup size
- `blocks_row=nothing`: Blocks per row

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
col_sums = CUDA.zeros(Float32, 500)
row_maxs = CUDA.zeros(Float32, 1000)

# Column sums
mapreduce2d!(identity, +, col_sums, A, 1)

# Row maximums
mapreduce2d!(identity, max, row_maxs, A, 2)
```

See also: [`KernelForge.mapreduce2d`](@ref) for the allocating version.
"""
function mapreduce2d!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    dim::Int;
    g::G=identity,
    tmp::Union{VecMatBuffer,MatVecBuffer,Nothing}=nothing,
    kwargs...
) where {S,T,F<:Function,O<:Function,G<:Function}
    if dim == 1
        return vecmat!(f, op, dst, nothing, src; g, tmp, kwargs...)
    elseif dim == 2
        return matvec!(f, op, dst, src, nothing; g, tmp, kwargs...)
    else
        throw(ArgumentError("dim must be 1 or 2, got $dim"))
    end
end