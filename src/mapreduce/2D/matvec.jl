# ============================================================================
# Buffer type
# ============================================================================

"""
    MatVecBuffer{A1,A2}

Pre-allocated buffer for `matvec!`, holding typed intermediate arrays.

- `A1`: array type for partial reduction values (eltype = output of `f`)
- `A2`: array type for synchronization flags (eltype = `UInt8`)
"""
struct MatVecBuffer{A1,A2}
    partial::A1
    flag::A2
end

# ============================================================================
# Public API docstrings
# ============================================================================

"""
    matvec([f, op,] src::AbstractMatrix, x; kwargs...) -> GPU array
    matvec!([f, op,] dst, src, x; kwargs...)

Generalized matrix-vector operation with customizable element-wise and reduction operations.

Computes `dst[i] = g(op_j(f(src[i,j], x[j])))` for each row `i`, where `op_j` denotes
reduction over columns. For standard matrix-vector multiplication, this is
`dst[i] = sum_j(src[i,j] * x[j])`.

The allocating version `matvec` returns a newly allocated result vector.
The in-place version `matvec!` writes to `dst`.

# Arguments
- `f`: Binary operation applied element-wise (default: `*`)
- `op`: Reduction operation across columns (default: `+`)
- `dst`: Output vector (in-place versions only)
- `src`: Input matrix
- `x`: Input vector, or `nothing` for row-wise reduction of `src` alone

# Keyword Arguments
- `g=identity`: Unary transformation applied to each reduced row
- `tmp=nothing`: Pre-allocated `MatVecBuffer` (or `nothing` to allocate automatically)
- `chunksz=nothing`: Elements per thread (auto-tuned if `nothing`)
- `Nblocks=nothing`: Number of thread blocks (auto-tuned if `nothing`)
- `workgroup=nothing`: Threads per block (auto-tuned if `nothing`)
- `blocks_row=nothing`: Number of blocks used to process a single row; relevant only
  for wide matrices (many columns, few rows) where parallelizing across columns is
  beneficial. Auto-tuned if `nothing`.

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)

# Standard matrix-vector multiply: y = A * x
y = matvec(A, x)

# Row-wise sum: y[i] = sum(A[i, :])
y = matvec(A, nothing)

# Row-wise maximum: y[i] = max_j(A[i, j])
y = matvec(identity, max, A, nothing)

# Softmax numerator: y[i] = sum_j(exp(A[i,j] - x[j]))
y = matvec((a, b) -> exp(a - b), +, A, x)

# In-place version
dst = CUDA.zeros(Float32, 1000)
matvec!(dst, A, x)

# With pre-allocated buffer for repeated calls
tmp = KernelForge.get_allocation(MatVec, *, +, A, x)
for i in 1:100
    matvec!(dst, A, x; tmp)
end
```
"""
function matvec end, function matvec! end

# ============================================================================
# Buffer allocation
# ============================================================================

"""
    get_allocation(::Type{MatVec}, f, op, src, x[, Nblocks]) -> MatVecBuffer

Allocate a `MatVecBuffer` for `matvec!`. Useful for repeated calls.

# Arguments
- `f`: Map function (used to infer intermediate eltype)
- `op`: Reduction operator
- `src`: Input GPU matrix (used to determine backend and eltype)
- `x`: Input vector or `nothing`
- `Nblocks`: Number of blocks (auto-computed if omitted)

# Returns
A `MatVecBuffer{A1,A2}` holding typed `partial` and `flag` arrays.

# Examples
```julia
A = CUDA.rand(Float32, 1000, 500)
x = CUDA.rand(Float32, 500)
tmp = KernelForge.get_allocation(MatVec, *, +, A, x)
dst = CUDA.zeros(Float32, 1000)

for i in 1:100
    matvec!(dst, A, x; tmp)
end
```
"""
function get_allocation(
    ::Type{MatVec},
    f::F,
    op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    Nblocks::Integer
) where {T,F<:Function,O<:Function}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    n = size(src, 1)
    backend = get_backend(src)
    partial = KernelAbstractions.allocate(backend, H, n * Nblocks)
    flag = KernelAbstractions.allocate(backend, UInt8, n * Nblocks)
    return MatVecBuffer(partial, flag)
end

function get_allocation(
    ::Type{MatVec},
    f::F,
    op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing}
) where {T,F<:Function,O<:Function}
    n, p = size(src)
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    chunksz, Nblocks, _, _ = _resolve_parameters(MatVec, nothing, nothing, nothing, nothing, n, p)
    return get_allocation(MatVec, f, op, src, x, Nblocks)
end

# ============================================================================
# Parameter resolution
# ============================================================================

function _resolve_parameters(
    ::Type{MatVec},
    chunksz::Union{Int,Nothing},
    Nblocks::Union{Int,Nothing},
    workgroup::Union{Int,Nothing},
    blocks_row::Union{Int,Nothing},
    n::Int,
    p::Int
)
    if isnothing(blocks_row)
        blocks_row = DEFAULT_BLOCKS
    end
    if isnothing(chunksz) && isnothing(Nblocks)
        chunksz = min(cld(nextpow(2, n), 8), 64)
        Nblocks = prevpow(2, cld(blocks_row, cld(n, chunksz)))
    end
    if isnothing(workgroup)
        workgroup = DEFAULT_WORKGROUP
        if Nblocks == 1
            workgroup = 128
        end
    end

    workgroup = max(
        min(workgroup, prevpow(2, n * p)),
        warpsz
    )
    Nblocks = min(Nblocks, prevpow(2, max(fld(p * chunksz, workgroup), 1)))
    chunksz = max(chunksz, nextpow(2, cld(workgroup * Nblocks, p)))
    chunksz = max(chunksz, nextpow(2, cld(workgroup * Nblocks, p)))
    chunksz = min(chunksz, workgroup)
    if workgroup == warpsz
        chunksz = workgroup
    end

    @assert !isnothing(chunksz) && !isnothing(Nblocks) "Must provide both chunksz and Nblocks, or neither"
    @assert cld(workgroup, chunksz) * Nblocks <= p
    @assert ispow2(Nblocks) || chunksz * Nblocks >= workgroup || chunksz * Nblocks <= warpsz
    return chunksz, Nblocks, workgroup, blocks_row
end

# ============================================================================
# Public API
# ============================================================================

# Simplified interface (standard mat-vec multiply) - allocating version
function matvec(
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{MatVecBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp)
    return dst
end

# Full interface with f and op - allocating version
function matvec(
    f::F, op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing
) where {T,F<:Function,O<:Function,G<:Function,TMP<:Union{MatVecBuffer,Nothing}}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp)
    return dst
end

# Simplified interface (standard mat-vec multiply) - in-place version
function matvec!(
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    f::F=*,
    op::O=+,
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{MatVecBuffer,Nothing}}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp)
end

# Full interface with f and op - in-place version
function matvec!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::TMP=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing
) where {S,T,F<:Function,O<:Function,G<:Function,TMP<:Union{MatVecBuffer,Nothing}}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp)
end

# ============================================================================
# Entry point (validation and parameter resolution)
# ============================================================================

function _matvec_entry!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Union{Int,Nothing},
    Nblocks::Union{Int,Nothing},
    workgroup::Union{Int,Nothing},
    blocks_row::Union{Int,Nothing},
    tmp::Union{MatVecBuffer,Nothing}
) where {S,T,F,O,G}
    n, p = size(src)
    if !isnothing(x)
        @assert length(x) == p "Vector length must match matrix columns"
    end
    @assert length(dst) == n "Output length must match matrix rows"

    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, eltype(x))

    chunksz_, Nblocks_, workgroup_, _ = _resolve_parameters(
        MatVec, chunksz, Nblocks, workgroup, blocks_row, n, p
    )

    _matvec_impl!(f, op, g, dst, src, x, chunksz_, Nblocks_, workgroup_, tmp, H, n, p)
end

# ============================================================================
# Core implementation
# ============================================================================

# Nothing dispatch: allocate buffer then forward to MatVecBuffer dispatch
function _matvec_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    ::Nothing,
    ::Type{H},
    n::Int,
    p::Int
) where {S,T,F,O,G,H}
    if Nblocks == 1
        _matvec_impl_single!(f, op, g, dst, src, x, chunksz, workgroup, H)
    else
        tmp = get_allocation(MatVec, f, op, src, x, Nblocks)
        _matvec_impl_multi!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, tmp, H)
    end
end

# MatVecBuffer dispatch
function _matvec_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    tmp::MatVecBuffer,
    ::Type{H},
    n::Int,
    p::Int
) where {S,T,F,O,G,H}
    if Nblocks == 1
        _matvec_impl_single!(f, op, g, dst, src, x, chunksz, workgroup, H)
    else
        _matvec_impl_multi!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, tmp, H)
    end
end

# Single-block case (no synchronization needed)
function _matvec_impl_single!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    workgroup::Int,
    ::Type{H}
) where {S,T,F,O,G,H}
    n, p = size(src)
    backend = get_backend(src)
    ndrange = cld(n, chunksz) * workgroup
    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(1), nothing, nothing, H
    )
end

# Multi-block case (with synchronization)
function _matvec_impl_multi!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    tmp::MatVecBuffer,
    ::Type{H}
) where {S,T,F,O,G,H}
    n, p = size(src)
    backend = get_backend(src)
    ndrange = cld(n, chunksz) * Nblocks * workgroup
    fill!(tmp.flag, 0x00)
    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(Nblocks), tmp.partial, tmp.flag, H
    )
end