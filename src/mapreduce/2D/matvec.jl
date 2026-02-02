"""
    matvec([f, op,] src::AbstractMatrix, x; kwargs...) -> dst
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
- `tmp=nothing`: Pre-allocated temporary buffer for inter-block communication
- `chunksz=nothing`: Elements per thread (auto-tuned if `nothing`)
- `Nblocks=nothing`: Number of thread blocks (auto-tuned if `nothing`)
- `workgroup=nothing`: Threads per block (auto-tuned if `nothing`)
- `blocks_row=nothing`: Number of blocks used to process a single row; relevant only
  for wide matrices (many columns, few rows) where parallelizing across columns is
  beneficial. Auto-tuned if `nothing`.
- `FlagType=UInt8`: Integer type for synchronization flags

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
```

# Extended Help

For tall matrices (many rows, few columns), each row is processed by a single block.
For wide matrices (few rows, many columns), multiple blocks collaborate on each row
via a number of blocks Nblocks computed from `blocks_row`. `blocks_row` is equal to 
Nblocks for a large row matrix.

Pre-allocating `tmp` avoids repeated allocation when calling `matvec!` in a loop.
With `FlagType=UInt8` (default), the flag buffer must be zeroed before each call.
Using `FlagType=UInt64` skips this zeroing by generating a random target flag at each
call; correctness holds with probability `1 - n/2^64`, which is negligible for practical `n`.
Output element type is inferred as `promote_op(g, promote_op(f, eltype(src), eltype(x)))`.
"""
function matvec end, function matvec! end

# ============================================================================
# Configuration helpers
# ============================================================================

const DEFAULT_MATVEC_WORKGROUP = 256
const DEFAULT_MATVEC_BLOCKS_ROW = 100 # Order of magnitude of number of blocks when the matrix is row matrix

# ============================================================================
# Allocation
# ============================================================================

# Main public entry point for allocation
function get_allocation(
    fn::typeof(matvec!),
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    Nblocks::Integer,
    eltype::Union{Type,Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {T,FT}
    H = if !isnothing(eltype)
        eltype
    else
        isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    end
    return _get_allocation(fn, src, x, Nblocks, H, FT)
end

# Core implementation (positional args)
function _get_allocation(
    ::typeof(matvec!),
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    Nblocks::Integer,
    ::Type{H},
    ::Type{FT}
) where {T,H,FT}
    Nblocks > 1 || error("Nblocks must be > 1, otherwise tmp allocation is unnecessary")
    backend = get_backend(src)
    n, p = size(src)
    sz = sum(get_partition_sizes(n * Nblocks, H, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Parameter resolution
# ============================================================================

function _resolve_parameters(
    ::typeof(matvec!),
    chunksz::Union{Int,Nothing},
    Nblocks::Union{Int,Nothing},
    workgroup::Union{Int,Nothing},
    blocks_row::Union{Int,Nothing},
    n::Int,
    p::Int
)
    if isnothing(blocks_row)
        blocks_row = DEFAULT_MATVEC_BLOCKS_ROW
    end
    if isnothing(chunksz) && isnothing(Nblocks)
        # Auto-tune based on problem size
        chunksz = min(cld(nextpow(2, n), 8), 64)
        Nblocks = prevpow(2, cld(blocks_row, cld(n, chunksz)))
    end
    if isnothing(workgroup)
        workgroup = DEFAULT_MATVEC_WORKGROUP
        if Nblocks == 1
            workgroup = 128 # small optimization
        end
    end

    # edge cases handling
    workgroup = max(
        min(workgroup, prevpow(2, n * p)),
        warpsz # edge case, n small
    )
    # (workgroup / chunksz) * Nblocks must be smaller than p
    Nblocks = min(Nblocks, prevpow(2, max(fld(p * chunksz, workgroup), 1))) # edge case, p small
    chunksz = max(chunksz, nextpow(2, cld(workgroup * Nblocks, p)))
    # workgroup / chunksz must also be smaller than Nblocks if Nblocks > 1
    chunksz = max(chunksz, nextpow(2, cld(workgroup * Nblocks, p)))

    chunksz = min(chunksz, workgroup)
    if workgroup == warpsz
        chunksz = workgroup
    end

    #@show workgroup, chunksz, Nblocks
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
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    FlagType::Type{FT}=UInt8
) where {T,F<:Function,O<:Function,G<:Function,FT}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp, FT)
    return dst
end

# Full interface with f and op - allocating version
function matvec(
    f::F, op::O,
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    FlagType::Type{FT}=UInt8
) where {T,F<:Function,O<:Function,G<:Function,FT}
    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))
    S = Base.promote_op(g, H)
    backend = get_backend(src)
    n = size(src, 1)
    dst = KernelAbstractions.allocate(backend, S, n)
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp, FT)
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
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,F<:Function,O<:Function,G<:Function,FT}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp, FT)
end

# Full interface with f and op - in-place version
function matvec!(
    f::F, op::O,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    chunksz=nothing,
    Nblocks=nothing,
    workgroup=nothing,
    blocks_row=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,F<:Function,O<:Function,G<:Function,FT}
    _matvec_entry!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, blocks_row, tmp, FT)
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
    tmp::Union{AbstractArray{UInt8},Nothing},
    ::Type{FT}
) where {S,T,F,O,G,FT}
    n, p = size(src)
    if !isnothing(x)
        @assert length(x) == p "Vector length must match matrix columns"
    end
    @assert length(dst) == n "Output length must match matrix rows"

    H = isnothing(x) ? Base.promote_op(f, T) : Base.promote_op(f, T, Base.eltype(x))

    # Resolve to get suitable chunksz, Nblocks and workgroup size
    chunksz_, Nblocks_, workgroup_, blocks_row_ = _resolve_parameters(
        matvec!, chunksz, Nblocks, workgroup, blocks_row, n, p
    )

    # Function barrier: dispatch based on tmp type
    _matvec_dispatch!(f, op, g, dst, src, x, chunksz_, Nblocks_, workgroup_, tmp, H, FT)
end

# ============================================================================
# Dispatch (function barrier for type stability)
# ============================================================================

# Dispatch when user provides tmp buffer
function _matvec_dispatch!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    tmp::AbstractArray{UInt8},
    ::Type{H},
    ::Type{FT}
) where {S,T,F,O,G,H,FT}
    if Nblocks == 1
        _matvec_impl_single!(f, op, g, dst, src, x, chunksz, workgroup, H, FT)
    else
        _matvec_impl_multi!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, tmp, H, FT)
    end
end

# Dispatch when no tmp buffer provided
function _matvec_dispatch!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    Nblocks::Int,
    workgroup::Int,
    tmp::Nothing,
    ::Type{H},
    ::Type{FT}
) where {S,T,F,O,G,H,FT}
    if Nblocks == 1
        _matvec_impl_single!(f, op, g, dst, src, x, chunksz, workgroup, H, FT)
    else
        tmp_ = _get_allocation(matvec!, src, x, Nblocks, H, FT)
        _matvec_impl_multi!(f, op, g, dst, src, x, chunksz, Nblocks, workgroup, tmp_, H, FT)
    end
end

# ============================================================================
# Core implementations
# ============================================================================

# Single-block case (no synchronization needed)
function _matvec_impl_single!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    src::AbstractMatrix{T},
    x::Union{AbstractArray,Nothing},
    chunksz::Int,
    workgroup::Int,
    ::Type{H},
    ::Type{FT},
) where {S,T,F,O,G,H,FT}
    n, p = size(src)
    backend = get_backend(src)
    ndrange = cld(n, chunksz) * workgroup
    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(1), nothing, nothing, one(FT), H
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
    tmp::AbstractArray{UInt8},
    ::Type{H},
    ::Type{FT},
) where {S,T,F,O,G,H,FT}
    n, p = size(src)
    backend = get_backend(src)
    ndrange = cld(n, chunksz) * Nblocks * workgroup

    partial, flag = partition(tmp, n * Nblocks, H, FT)
    if FT === UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FT)
    end

    matvec_kernel!(backend, workgroup, ndrange)(
        f, op, g, dst, src, x, Val(chunksz), Val(Nblocks), partial, flag, targetflag, H
    )
end